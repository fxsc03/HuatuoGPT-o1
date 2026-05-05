"""
GRPO Stage 2 — Single GPU (RTX A6000 / 40-80 GB VRAM)
======================================================
Uses TRL GRPOTrainer + QLoRA (4-bit NF4 quantisation).

verl (原 GRPO_stage2.py) 依赖 Ray 多 worker，单卡无法运行。
本脚本改用 TRL GRPOTrainer，原生支持单卡 + PEFT/bitsandbytes。

算法与奖励逻辑和 GRPO_stage2.py 完全一致：
  格式错误               → 0.0
  格式正确 + P(True)≤0.4 → 0.1
  格式正确 + P(True)>0.4 → 1.0

Usage:
    python GRPO_stage2_single_gpu.py \
        --model_name_or_path FreedomIntelligence/HuatuoGPT-o1-8B \
        --reward_model_path  FreedomIntelligence/medical_o1_verifier_3B \
        --dataset_name       FreedomIntelligence/medical-o1-verifiable-problem
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
        pass  # 实在找不到就跳过，单卡训练不走 FSDP 路径
# ---------------------------------------------------------------------------

import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import selective_log_softmax


class GRPOTrainerWithEntropy(GRPOTrainer):
    """记录 policy 的 token-level 平均熵到 self._metrics['entropy']。

    只在训练 forward（torch.is_grad_enabled() == True）时记录，
    避免 ref-model 的 no_grad 调用污染指标。
    """

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
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from torch.utils.tensorboard import SummaryWriter

os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from grpo_utils.tb_logger import GRPOTensorBoardCallback, step_metrics_buf


# ---------------------------------------------------------------------------
# Reward model — 全局 singleton，只加载一次
# ---------------------------------------------------------------------------

_reward_model = None
_reward_tokenizer = None
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
    通过 OpenAI 兼容 API（如硅基流动 DeepSeek-R1）做 LLM-as-judge 打分，
    返回连续 reward ∈ [0, 1]。

    - 格式校验失败 → 0.0
    - 格式 OK → max(0.05, score/100)，保证 > 格式失败
    - 8 条 completion 并发请求；失败重试，多次失败回退 0.0
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
    ):
        if not api_key:
            raise ValueError("reward_api_key is empty — set it in run_grpo.sh")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._score_re = re.compile(r"\b(\d{1,3})\b")
        print(f"[reward-api] using {model_id} via {self.api_base}")

    def _score_one(self, model_response: str, reference_answer: str) -> Optional[float]:
        """API 调一次。成功返回 [0,1]；耗尽重试或解析失败返回 None。"""
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
                # R1 会在前面写 reasoning，最后才给数字 → 取最后一个 1-3 位整数
                matches = list(self._score_re.finditer(content))
                if matches:
                    score = int(matches[-1].group(1))
                    score = max(0, min(100, score))
                    return score / 100.0
                # 解析不到数字 → 视作一次失败重试
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"[reward-api] failed after {self.max_retries} retries: {e}")
                    return None
                time.sleep(2.0 * (2 ** attempt))   # 2s → 4s → 8s 指数退避
        return None

    def __call__(
        self,
        completions: List[str],
        ground_truth: List[str],
        **kwargs,
    ) -> List[float]:
        n = len(completions)

        # 1. 格式校验 + 抽 final response
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

        # 2. 并发 API 调用（仅对格式 OK 的）
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
                rewards[i] = 0.05  # API 失败但格式 OK → 给最小 floor，不影响其他兄弟样本
            else:
                rewards[i] = max(0.05, score)   # floor = 0.05，保证 > format-fail = 0

        # 3. 指标统计
        n_correct = sum(1 for r in rewards if r >= 0.5)   # accuracy 重新定义为 score >= 50
        n_fmt_ok = sum(fmt_ok)
        n_eos = n_fmt_ok
        mean_tok_len = sum(len(c.split()) for c in completions) / max(n, 1)

        per_completion_lens = [len(c.split()) for c in completions]
        per_completion_hash = [hash(c.strip()[:200]) for c in completions]
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

        _reward_stats.append(sum(rewards) / n)
        if random.random() < 0.05:
            print(f"[reward] resp[:150]: {completions[0][:150]}…  →  {rewards[0]:.3f}")
            w = _reward_stats[-50:]
            print(f"[reward] avg-50={sum(w)/len(w):.4f}  "
                  f"≥0.5 ratio={n_correct/n*100:.1f}%  "
                  f"fmt_ok={n_fmt_ok/n*100:.1f}%  "
                  f"api_fail={n_api_fail}/{len(future_to_idx)}")

        return rewards


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

def load_grpo_dataset(
    dataset_name: str,
    tokenizer,
    eval_ratio: float = 0.05,
    debug_size: int = 0,
    max_prompt_length: int = 512,
):
    """
    支持：
      - HuggingFace Hub id  (e.g. FreedomIntelligence/medical-o1-verifiable-problem)
      - 本地 JSON 文件路径  (与 RL_stage2.py 格式相同)
    """
    if os.path.isfile(dataset_name):
        with open(dataset_name) as f:
            raw = json.load(f)
    else:
        print(f"[data] downloading {dataset_name} from HuggingFace Hub …")
        raw = list(load_dataset(dataset_name, split="train"))

    # 系统提示：强制模型输出 ## Thinking / ## Final Response 格式
    # 这是奖励函数格式校验的前提条件，没有此提示基础模型不会自动使用该格式
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
        # 在数据层截断过长 prompt（GRPOConfig 无此参数，需手动处理）
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
# CLI 参数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GRPO single-GPU training with QLoRA")

    # 模型 / 数据
    p.add_argument("--model_name_or_path",          default="FreedomIntelligence/HuatuoGPT-o1-8B")
    p.add_argument("--reward_model_path",            default="FreedomIntelligence/medical_o1_verifier_3B",
                   help="(legacy) 本地 verifier 路径；改用 API 后忽略")
    p.add_argument("--reward_api_base",              default="https://api.siliconflow.cn/v1",
                   help="OpenAI 兼容 API 根地址（不含 /chat/completions）")
    p.add_argument("--reward_api_key",               default="",
                   help="API Key（必填，从 run_grpo.sh 传入）")
    p.add_argument("--reward_model_id",              default="deepseek-ai/DeepSeek-R1",
                   help="judge 模型名，硅基流动上推荐 deepseek-ai/DeepSeek-R1")
    p.add_argument("--dataset_name",                 default="FreedomIntelligence/medical-o1-verifiable-problem")
    p.add_argument("--output_dir",                   default="./grpo_ckpts")
    p.add_argument("--run_name",                     default="medical_grpo_lora")

    # LoRA
    p.add_argument("--lora_rank",                    type=int,   default=16)
    p.add_argument("--lora_alpha",                   type=int,   default=32)
    p.add_argument("--lora_dropout",                 type=float, default=0.05)

    # 量化
    p.add_argument("--load_in_4bit",                 action="store_true", default=True)
    p.add_argument("--load_in_8bit",                 action="store_true", default=False,
                   help="用 8-bit 代替 4-bit（显存更多但精度更高）")

    # GRPO 核心
    p.add_argument("--num_generations",              type=int,   default=4,
                   help="GRPO 组大小 G：每个 prompt 生成几条回复（建议单卡用 4）")
    p.add_argument("--max_prompt_length",            type=int,   default=512)
    p.add_argument("--max_completion_length",        type=int,   default=2048)
    p.add_argument("--temperature",                  type=float, default=0.8)
    p.add_argument("--kl_coef",                      type=float, default=0.001)

    # 训练超参
    p.add_argument("--num_train_epochs",             type=int,   default=3)
    p.add_argument("--max_steps",                    type=int,   default=-1,
                   help="最大训练步数，设置后覆盖 num_train_epochs（-1 表示不限制）")
    p.add_argument("--per_device_train_batch_size",  type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps",  type=int,   default=4)
    p.add_argument("--lr",                           type=float, default=1e-5)
    p.add_argument("--warmup_steps",                 type=int,   default=10)
    p.add_argument("--save_steps",                   type=int,   default=200)
    p.add_argument("--eval_steps",                   type=int,   default=100)
    p.add_argument("--eval_size",                    type=int,   default=50, help="Cap eval dataset size (0 = use full)")
    p.add_argument("--logging_steps",                type=int,   default=10)
    p.add_argument("--max_reward_length",            type=int,   default=2000)

    # 调试
    p.add_argument("--debug_size",                   type=int,   default=0,
                   help="只取前 N 条数据用于调试（0 = 全量）")
    p.add_argument("--tb_log_dir",                   type=str,   default="",
                   help="TensorBoard 日志目录，默认为 output_dir/tb_logs")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # LLaMA-3 系列需要手动设置 pad token
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
            bnb_4bit_use_double_quant=True,       # 二次量化，再省 ~0.4 GB
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("[model] loading in 4-bit NF4")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        attn_implementation="sdpa",  # PyTorch 内置高效 attention，免编译
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False  # 与 gradient_checkpointing 不兼容
    # 4-bit + LoRA + gradient_checkpointing 必须调，否则梯度无法回流到 LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── 4. LoRA 配置 ──────────────────────────────────────────────────────
    # target_modules 覆盖所有 attention + FFN 投影层
    # 对 LLaMA / Qwen 结构通用
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # FFN
        ],
    )

    # ── 5. GRPO 训练配置 ───────────────────────────────────────────────────
    # 注意：TRL 0.29.x 参数名与旧版有差异
    #   max_prompt_length → 已在数据集层截断，此处无此参数
    #   kl_coef           → 改名为 beta
    grpo_config = GRPOConfig(
        # 路径 & 日志
        output_dir=output_dir,
        run_name=args.run_name,
        report_to="none",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,

        # 训练轮次（max_steps > 0 时覆盖 num_train_epochs）
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # GRPO 核心参数
        num_generations=args.num_generations,       # G = 组大小
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.kl_coef,                          # TRL 0.29.x 中 kl_coef 改名为 beta

        # 优化器
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",   # bitsandbytes 8-bit AdamW，节省优化器显存

        # 显存优化
        gradient_checkpointing=True,
        bf16=True,
        remove_unused_columns=False,
    )

    # ── 6. 奖励函数（DeepSeek-R1 API as judge，连续 reward）────────────────
    reward_fn = MedicalRewardFnAPI(
        api_base=args.reward_api_base,
        api_key=args.reward_api_key,
        model_id=args.reward_model_id,
        max_workers=args.num_generations,
    )

    # ── 7. TensorBoard callback ───────────────────────────────────────────
    tb_log_dir = args.tb_log_dir or os.path.join(output_dir, "tb_logs")
    tb_callback = GRPOTensorBoardCallback(
        log_dir=tb_log_dir,
        num_generations=args.num_generations,
    )

    # ── 8. 启动训练 ────────────────────────────────────────────────────────
    trainer = GRPOTrainerWithEntropy(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[tb_callback],
    )

    print("[train] starting GRPO …")
    trainer.train()

    # 只保存 LoRA adapter（全量权重已量化，adapter 才是可复用的）
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[done] LoRA adapter saved → {output_dir}")


if __name__ == "__main__":
    main()
