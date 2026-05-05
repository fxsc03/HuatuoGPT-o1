"""
DAPO Stage 2 — Single GPU (RTX A6000 / 40-80 GB VRAM)
======================================================
Mirrors GRPO_stage2_single_gpu.py 的结构，但实现 DAPO（ByteDance, 2025-03）
的核心改进：

  1. Token-level loss     —— 全 batch token 求和 / token 总数（替代 TRL
                              0.15.2 的 sequence-mean → batch-mean）
  2. β = 0                —— 通过 --kl_coef 0.0 关闭 KL 项
  3. Overlong shaping     —— 完成长度接近 max_completion_length 时线性扣分
  4. Clip-Higher 接口      —— --epsilon_low/--epsilon_high CLI 已就位，但
                              MVP 单 epoch on-policy 训练下 ratio ≡ 1，
                              clip 不会触发；保留参数以便 v2 多 epoch PPO

奖励仍然走 DeepSeek-R1 API（与 GRPO 保持一致），数据集 / 系统提示 /
LoRA 配置都复用同一份。
"""

import os
import re
import json
import time
import random
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Compatibility patch: torch 2.5.x 把 FSDPModule 放在私有路径，
# torch 2.6+ 才移到公开 API。TRL 所有新版本都从公开路径 import，
# 所以在 import trl 之前把符号补到公开位置。
# ---------------------------------------------------------------------------
import torch
import torch.distributed.fsdp as _fsdp_mod
if not hasattr(_fsdp_mod, "FSDPModule"):
    try:
        from torch.distributed._composable.fsdp import FSDPModule as _FSDPModule
        _fsdp_mod.FSDPModule = _FSDPModule
    except ImportError:
        pass
# ---------------------------------------------------------------------------

import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from torch.utils.data import Sampler
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.grpo_trainer import RepeatRandomSampler
from trl.trainer.utils import selective_log_softmax


class GRPOTrainerWithEntropy(GRPOTrainer):
    """记录 policy token-level 平均熵到 self._metrics['entropy']。"""

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]
        input_ids_local = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]

        if torch.is_grad_enabled():
            with torch.no_grad():
                log_probs = F.log_softmax(logits.float(), dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean().item()
            self._metrics.setdefault("entropy", []).append(entropy)

        return selective_log_softmax(logits, input_ids_local)


class DAPOTrainer(GRPOTrainerWithEntropy):
    """
    DAPO 训练器（v2，off-policy）。相对于 TRL 0.15.2 GRPOTrainer 改三处：

    1. ``compute_loss`` 替换为 token-level loss + Clip-Higher（DAPO §3.2）
       使用真正的 importance ratio = exp(new_logp − old_logp)，配合非对称裁
       [1−ε_low, 1+ε_high]。

    2. ``_prepare_inputs`` 加缓存：每 K 个连续 batch 才真做一次 rollout
       （生成 + reward API + advantages），中间 K-1 个 batch 直接复用缓存。
       同时在 rollout 时刻把当时的 logp 保存到 ``old_per_token_logps`` 字段，
       供 ``compute_loss`` 算真 ratio。

    3. ``_get_train_sampler`` 把 RepeatRandomSampler 的 ``repeat_count`` 从
       ``num_generations`` 提升到 ``num_generations * num_ppo_epochs``，让
       dataloader 把同一组 prompt 连续吐 K 次。

    K = ``num_ppo_epochs``。K=1 行为退化到原 GRPO（single-epoch on-policy），
    此时 ratio ≡ 1，Clip-Higher 不会触发——可用作正确性回归测试。
    K≥2 时第 2 个 inner-epoch 起 new_logp ≠ old_logp，Clip-Higher 真正生效。

    单卡约束：本实现假设 ``gradient_accumulation_steps == 1``。否则
    "K inner-epoch = 1 rollout 周期" 与 grad accum 边界会错位。
    """

    def __init__(
        self,
        *args,
        eps_low: float = 0.2,
        eps_high: float = 0.28,
        num_ppo_epochs: int = 2,
        dynamic_sampling_threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.args.gradient_accumulation_steps != 1:
            raise ValueError(
                "DAPOTrainer 当前实现要求 gradient_accumulation_steps == 1 "
                f"（当前 = {self.args.gradient_accumulation_steps}）。理由：K-epoch "
                "rollout 缓存的 cycle 计数器假设每 batch 一次 optimizer.step()。"
            )
        if num_ppo_epochs < 1:
            raise ValueError(f"num_ppo_epochs must be >= 1, got {num_ppo_epochs}")
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.num_ppo_epochs = num_ppo_epochs
        self.dynamic_sampling_threshold = dynamic_sampling_threshold
        self._inner_epoch_counter = 0  # 训练时 0..K-1 循环递增
        self._cached_inputs = None      # 当前 rollout 周期的缓存

    # ------------------------------------------------------------------
    # Sampler：repeat_count = G * K（GRPO 是 G）
    # ------------------------------------------------------------------
    def _get_train_sampler(self) -> Sampler:
        return RepeatRandomSampler(
            self.train_dataset,
            self.num_generations * self.num_ppo_epochs,
            seed=self.args.seed,
        )

    # ------------------------------------------------------------------
    # _prepare_inputs：每 K 步真 rollout 一次，其余 K-1 步复用缓存
    # ------------------------------------------------------------------
    def _prepare_inputs(self, inputs):
        # eval 阶段不缓存（eval 走 prediction_step → compute_loss，但 inputs
        # 来自 GRPOTrainer.prediction_step 内部对 _prepare_inputs 的直接调用，
        # 也会进这个分支；用 model.training 标志区分训练/评估）
        if not self.model.training:
            return super()._prepare_inputs(inputs)

        cycle = self._inner_epoch_counter % self.num_ppo_epochs
        if cycle == 0:
            # 第 0 次：真做 rollout（生成 + reward + advantages）
            result = super()._prepare_inputs(inputs)

            # 在当前 model（grad 未更新）上算一次 logp 作为 old_per_token_logps
            input_ids = torch.cat([result["prompt_ids"], result["completion_ids"]], dim=1)
            attention_mask = torch.cat([result["prompt_mask"], result["completion_mask"]], dim=1)
            logits_to_keep = result["completion_ids"].size(1)
            with torch.inference_mode():
                # 走 GRPOTrainer 的底层路径，避免 GRPOTrainerWithEntropy
                # 在 grad-disabled 下还塞 entropy 指标
                old_logps = GRPOTrainer._get_per_token_logps(
                    self, self.model, input_ids, attention_mask, logits_to_keep
                )
            result["old_per_token_logps"] = old_logps.detach()

            # Dynamic Sampling：丢弃组内 advantage std 低于阈值的 group
            adv = result["advantages"]
            G = self.num_generations
            n_groups = adv.numel() // G
            n_filtered = 0
            if n_groups > 0:
                grouped_std = adv.view(n_groups, G).std(dim=1)  # (n_groups,)
                low_std_mask = grouped_std < self.dynamic_sampling_threshold
                n_filtered = int(low_std_mask.sum().item())
                if n_filtered > 0:
                    # 零掉低信号组的 advantage，梯度贡献变 0（等效丢弃）
                    token_mask = low_std_mask.repeat_interleave(G)
                    result["advantages"] = adv.clone()
                    result["advantages"][token_mask] = 0.0
            self._metrics.setdefault("dapo_n_zero_adv_groups", []).append(float(n_filtered))

            self._cached_inputs = result
        else:
            # 后续 K-1 次：直接复用缓存（不重新 generate / 不再调 reward API）
            result = self._cached_inputs

        self._inner_epoch_counter += 1
        return result

    # ------------------------------------------------------------------
    # compute_loss：使用真 ratio + Clip-Higher + token-level loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("DAPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # new_logp（带梯度）
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # old_logp：训练时用缓存的 rollout 时刻 logp；eval 时退化为 detach 的当前 logp
        if "old_per_token_logps" in inputs:
            old_per_token_logps = inputs["old_per_token_logps"]
        else:
            old_per_token_logps = per_token_logps.detach()

        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps) - 1
        )

        advantages = inputs["advantages"]
        # *** 真正的 importance ratio：第 1 个 inner epoch 时 new == old → 1.0；
        # ***                          第 2+ 个 inner epoch 时偏离 1.0
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        ratio_clipped = torch.clamp(ratio, 1.0 - self.eps_low, 1.0 + self.eps_high)
        per_token_loss_unclipped = ratio * advantages.unsqueeze(1)
        per_token_loss_clipped   = ratio_clipped * advantages.unsqueeze(1)
        per_token_loss = torch.min(per_token_loss_unclipped, per_token_loss_clipped)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        # DAPO §3.2 token-level loss
        denom = completion_mask.sum().clamp(min=1)
        loss = (per_token_loss * completion_mask).sum() / denom

        # ── 指标 ────────────────────────────────────────────────────────────
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        with torch.no_grad():
            mask = completion_mask.bool()
            ratio_mean = ratio[mask].mean().item() if mask.any() else 1.0
            clip_high = ((ratio > 1.0 + self.eps_high) & mask).float().sum().item()
            clip_low  = ((ratio < 1.0 - self.eps_low)  & mask).float().sum().item()
            n_tok = mask.sum().item() or 1
        self._metrics.setdefault("dapo_ratio_mean",      []).append(ratio_mean)
        self._metrics.setdefault("dapo_clip_high_ratio", []).append(clip_high / n_tok)
        self._metrics.setdefault("dapo_clip_low_ratio",  []).append(clip_low  / n_tok)

        return loss


from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from torch.utils.tensorboard import SummaryWriter

os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from grpo_utils.tb_logger import GRPOTensorBoardCallback, step_metrics_buf


# ---------------------------------------------------------------------------
# Reward fn — 与 GRPO 共用同一份 LLM-as-judge，但加上 DAPO overlong shaping
# ---------------------------------------------------------------------------

_reward_stats: List[float] = []
_OUTPUT_RE = re.compile(r"## Final Response\n\n(.*)", re.S)

_API_VERIFIER_PROMPT = """You are a strict medical expert. Evaluate the candidate response against the reference answer.

Score the candidate from 0 to 100:
  100  = answer fully correct, matches reference
  60-99 = mostly correct, minor issues
  20-59 = partially correct, or right reasoning but wrong final option
  0-19  = wrong / off-topic / unsupported

<Candidate Response>
{}
</Candidate Response>

<Reference Answer>
{}
</Reference Answer>

Output ONLY a single integer between 0 and 100 on the last line. No other text after the integer."""


class MedicalRewardFnAPI:
    """
    与 GRPO_stage2_single_gpu.py 中的同名类几乎一致，新增两个 DAPO 字段：

      overlong_buffer_len  在 [L-buf, L] 区间内线性扣分；0 = 关闭 shaping
      overlong_factor      扣分上限（reward 最多被减去 overlong_factor）

    长度估计用 `len(c.split())`（word-level）作为 token 的近似（±15% 误差），
    避免依赖 policy tokenizer。
    """

    __name__ = "medical_verifier_reward"

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_id: str,
        max_workers: int = 8,
        max_retries: int = 3,
        timeout: int = 180,
        max_output_tokens: int = 2048,
        max_completion_length: int = 512,
        overlong_buffer_len: int = 0,
        overlong_factor: float = 1.0,
    ):
        if not api_key:
            raise ValueError("reward_api_key is empty — set it in run_dapo.sh")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._score_re = re.compile(r"\b(\d{1,3})\b")
        self.max_completion_length = max_completion_length
        self.overlong_buffer_len = overlong_buffer_len
        self.overlong_factor = overlong_factor
        print(f"[reward-api] using {model_id} via {self.api_base}")
        if overlong_buffer_len > 0:
            print(f"[reward-api] overlong shaping: penalty in "
                  f"[{max_completion_length - overlong_buffer_len}, "
                  f"{max_completion_length}] words, max -{overlong_factor:.2f}")

    def _score_one(self, model_response: str, reference_answer: str) -> Optional[float]:
        prompt = _API_VERIFIER_PROMPT.format(model_response, reference_answer)
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": self.max_output_tokens,
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                matches = list(self._score_re.finditer(content))
                if matches:
                    score = int(matches[-1].group(1))
                    score = max(0, min(100, score))
                    return score / 100.0
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"[reward-api] failed after {self.max_retries} retries: {e}")
                    return None
                time.sleep(2.0 * (2 ** attempt))
        return None

    def __call__(
        self,
        completions: List[str],
        ground_truth: List[str],
        **kwargs,
    ) -> List[float]:
        n = len(completions)

        fmt_ok: List[bool] = []
        answers: List[str] = []
        for resp in completions:
            ok = (
                resp.count("## Final Response\n\n") == 1
                and resp.count("## Thinking") == 1
            )
            fmt_ok.append(ok)
            if ok:
                m = _OUTPUT_RE.search(resp)
                answers.append(m.group(1).strip() if m else "")
            else:
                answers.append("")

        rewards: List[float] = [0.0] * n
        future_to_idx = {}
        for i in range(n):
            if fmt_ok[i] and answers[i]:
                fut = self.executor.submit(self._score_one, answers[i], ground_truth[i])
                future_to_idx[fut] = i

        n_api_fail = 0
        for fut, i in future_to_idx.items():
            try:
                score = fut.result()
            except Exception:
                score = None
            if score is None:
                n_api_fail += 1
                rewards[i] = 0.05
            else:
                rewards[i] = max(0.05, score)

        # ── DAPO §3.3 overlong reward shaping ───────────────────────────────
        # 线性 ramp：长度 ≤ L-buf 时不罚；> L-buf 时按 (len-(L-buf))/buf 比例
        # 扣 overlong_factor，最多扣到 0。
        overlong_penalties: List[float] = [0.0] * n
        if self.overlong_buffer_len > 0:
            L = self.max_completion_length
            buf = self.overlong_buffer_len
            for i, comp in enumerate(completions):
                toks = len(comp.split())
                if toks > L - buf:
                    p = min(1.0, (toks - (L - buf)) / buf)
                    overlong_penalties[i] = p
                    rewards[i] = max(0.0, rewards[i] - p * self.overlong_factor)

        # ── 指标 ────────────────────────────────────────────────────────────
        n_correct = sum(1 for r in rewards if r >= 0.5)
        n_fmt_ok = sum(fmt_ok)
        n_eos = n_fmt_ok
        per_completion_lens = [len(c.split()) for c in completions]
        per_completion_hash = [hash(c.strip()[:200]) for c in completions]
        mean_tok_len = sum(per_completion_lens) / max(n, 1)

        step_metrics_buf.update({
            "reward/accuracy":       n_correct / n,
            "reward/format":         n_fmt_ok / n,
            "reward/total":          sum(rewards) / n,
            "gen/completion_length": mean_tok_len,
            "gen/eos_ratio":         n_eos / n,
            "_raw_rewards":          list(rewards),
            "_completion_lengths":   per_completion_lens,
            "_completion_hashes":    per_completion_hash,
        })
        if self.overlong_buffer_len > 0:
            step_metrics_buf["dapo/overlong_penalty_mean"] = sum(overlong_penalties) / n

        _reward_stats.append(sum(rewards) / n)
        if random.random() < 0.05:
            print(f"[reward] resp[:150]: {completions[0][:150]}…  →  {rewards[0]:.3f}")
            w = _reward_stats[-50:]
            print(f"[reward] avg-50={sum(w)/len(w):.4f}  "
                  f"≥0.5 ratio={n_correct/n*100:.1f}%  "
                  f"fmt_ok={n_fmt_ok/n*100:.1f}%  "
                  f"api_fail={n_api_fail}/{len(future_to_idx)}  "
                  f"overlong_pen={sum(overlong_penalties)/n:.3f}")

        return rewards


# ---------------------------------------------------------------------------
# 数据集 — 与 GRPO 完全一致
# ---------------------------------------------------------------------------

def load_grpo_dataset(
    dataset_name: str,
    tokenizer,
    eval_ratio: float = 0.05,
    debug_size: int = 0,
    max_prompt_length: int = 512,
):
    if os.path.isfile(dataset_name):
        with open(dataset_name) as f:
            raw = json.load(f)
    else:
        print(f"[data] downloading {dataset_name} from HuggingFace Hub …")
        raw = list(load_dataset(dataset_name, split="train"))

    SYSTEM_PROMPT = (
        "You are a medical expert. When answering questions, you must follow this exact format:\n\n"
        "## Thinking\n"
        "[Your detailed reasoning process here]\n\n"
        "## Final Response\n"
        "[Your final answer here]\n\n"
        "Always use exactly these two section headers."
    )

    records = []
    for item in raw:
        q = item.get("Open-ended Verifiable Question", "").strip()
        a = item.get("Ground-True Answer", "").strip()
        if not q or not a:
            continue
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_len > max_prompt_length:
            continue
        records.append({
            "prompt": prompt,
            "ground_truth": a,
        })

    random.shuffle(records)
    if debug_size > 0:
        records = records[:debug_size]

    eval_n = min(int(len(records) * eval_ratio), 200)
    train_ds = Dataset.from_list(records[eval_n:])
    eval_ds  = Dataset.from_list(records[:eval_n])
    print(f"[data] train={len(train_ds)}  eval={len(eval_ds)}")
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DAPO single-GPU training with QLoRA")

    p.add_argument("--model_name_or_path",          default="FreedomIntelligence/HuatuoGPT-o1-8B")
    p.add_argument("--reward_api_base",              default="https://api.siliconflow.cn/v1")
    p.add_argument("--reward_api_key",               default="")
    p.add_argument("--reward_model_id",              default="deepseek-ai/DeepSeek-R1")
    p.add_argument("--dataset_name",                 default="FreedomIntelligence/medical-o1-verifiable-problem")
    p.add_argument("--output_dir",                   default="./grpo_ckpts")
    p.add_argument("--run_name",                     default="medical_dapo_lora")

    p.add_argument("--lora_rank",                    type=int,   default=16)
    p.add_argument("--lora_alpha",                   type=int,   default=32)
    p.add_argument("--lora_dropout",                 type=float, default=0.05)

    p.add_argument("--load_in_4bit",                 action="store_true", default=True)
    p.add_argument("--load_in_8bit",                 action="store_true", default=False)

    p.add_argument("--num_generations",              type=int,   default=4)
    p.add_argument("--max_prompt_length",            type=int,   default=512)
    p.add_argument("--max_completion_length",        type=int,   default=2048)
    p.add_argument("--temperature",                  type=float, default=0.8)
    p.add_argument("--kl_coef",                      type=float, default=0.0,
                   help="DAPO 默认 β=0；GRPO 默认 0.001")

    # DAPO 专属
    p.add_argument("--epsilon_low",                  type=float, default=0.2,
                   help="Clip-Higher 下界 ε_low（MVP 单 epoch 下不生效，保留供 v2 用）")
    p.add_argument("--epsilon_high",                 type=float, default=0.28,
                   help="Clip-Higher 上界 ε_high（同上）")
    p.add_argument("--overlong_buffer_len",          type=int,   default=0,
                   help="overlong shaping 缓冲区长度（word），0 = 关闭")
    p.add_argument("--overlong_factor",              type=float, default=1.0,
                   help="overlong 最大扣分（reward 最多减此值）")
    p.add_argument("--num_ppo_epochs",               type=int,   default=2,
                   help="K = PPO inner-epoch 数；K=1 等价 on-policy（Clip-Higher "
                        "不生效），K>=2 才让 ratio 偏离 1。注意 max_steps 解读为"
                        "内层步数，效果上 rollout 次数 = max_steps / K")
    p.add_argument("--dynamic_sampling_threshold",  type=float, default=0.01,
                   help="Dynamic sampling：组内 advantage std 低于此阈值的 group "
                        "被丢弃（advantages 置 0）。0 = 仅过滤真正 std=0 的组；"
                        "0.1 = 过滤低信号组；DAPO 原文等价于 0（连续 reward 下建议 0.01）")

    p.add_argument("--num_train_epochs",             type=int,   default=3)
    p.add_argument("--max_steps",                    type=int,   default=-1)
    p.add_argument("--per_device_train_batch_size",  type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps",  type=int,   default=4)
    p.add_argument("--lr",                           type=float, default=1e-5)
    p.add_argument("--warmup_steps",                 type=int,   default=10)
    p.add_argument("--save_steps",                   type=int,   default=200)
    p.add_argument("--eval_steps",                   type=int,   default=100)
    p.add_argument("--eval_size",                    type=int,   default=50)
    p.add_argument("--logging_steps",                type=int,   default=10)

    p.add_argument("--debug_size",                   type=int,   default=0)
    p.add_argument("--tb_log_dir",                   type=str,   default="")
    p.add_argument("--resume_from_checkpoint",       type=str,   default=None,
                   help="从指定 checkpoint 路径恢复训练，例如 ./grpo_ckpts/dapo-430/.../checkpoint-500")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.num_ppo_epochs < 1:
        raise ValueError(f"--num_ppo_epochs must be >= 1, got {args.num_ppo_epochs}")
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            f"--gradient_accumulation_steps must be 1 for DAPO v2 "
            f"(got {args.gradient_accumulation_steps})。理由：K-epoch rollout 缓存的"
            "周期与 grad accum 边界不能错位；如需更大 effective batch，请加大 "
            "--per_device_train_batch_size 或 --num_generations。"
        )

    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "<|eot_id|>" in tokenizer.vocab:
        tokenizer.pad_token    = "<|end_of_text|>"
        tokenizer.pad_token_id = tokenizer.encode(
            "<|end_of_text|>", add_special_tokens=False
        )[0]
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 2. Dataset ────────────────────────────────────────────────────────
    train_ds, eval_ds = load_grpo_dataset(
        args.dataset_name, tokenizer,
        debug_size=args.debug_size,
        max_prompt_length=args.max_prompt_length,
    )
    if args.eval_size > 0 and len(eval_ds) > args.eval_size:
        eval_ds = eval_ds.select(range(args.eval_size))
        print(f"[data] eval dataset truncated to {len(eval_ds)}")

    # ── 3. Policy model（4-bit QLoRA）────────────────────────────────────
    if args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        print("[model] loading in 8-bit")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("[model] loading in 4-bit NF4")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── 4. LoRA ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ── 5. GRPOConfig（DAPO 复用）──────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        run_name=args.run_name,
        report_to="none",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,

        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.kl_coef,

        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",

        gradient_checkpointing=True,
        bf16=True,
        remove_unused_columns=False,
    )

    # ── 6. 奖励函数（带 overlong shaping）─────────────────────────────────
    reward_fn = MedicalRewardFnAPI(
        api_base=args.reward_api_base,
        api_key=args.reward_api_key,
        model_id=args.reward_model_id,
        max_workers=args.num_generations,
        max_completion_length=args.max_completion_length,
        overlong_buffer_len=args.overlong_buffer_len,
        overlong_factor=args.overlong_factor,
    )

    # ── 7. TensorBoard callback ───────────────────────────────────────────
    tb_log_dir = args.tb_log_dir or os.path.join(output_dir, "tb_logs")
    tb_callback = GRPOTensorBoardCallback(
        log_dir=tb_log_dir,
        num_generations=args.num_generations,
    )

    # ── 8. 训练 ────────────────────────────────────────────────────────────
    trainer = DAPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[tb_callback],
        eps_low=args.epsilon_low,
        eps_high=args.epsilon_high,
        num_ppo_epochs=args.num_ppo_epochs,
        dynamic_sampling_threshold=args.dynamic_sampling_threshold,
    )

    print(f"[train] starting DAPO  (β={args.kl_coef}, "
          f"ε_low={args.epsilon_low}, ε_high={args.epsilon_high}, "
          f"K={args.num_ppo_epochs}, "
          f"dyn_thresh={args.dynamic_sampling_threshold}, "
          f"overlong_buf={args.overlong_buffer_len}) …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[done] LoRA adapter saved → {output_dir}")


if __name__ == "__main__":
    main()
