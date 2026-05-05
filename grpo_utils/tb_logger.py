"""
TensorBoard logging for GRPO training.

两个公开对象：
  GRPOTensorBoardLogger   —— 核心日志类，接收指标 dict，写入 SummaryWriter
  GRPOTensorBoardCallback —— TRL TrainerCallback，自动从 trainer 日志 + 全局缓冲
                              拼装指标 dict，每 logging_steps 触发一次写入

全局缓冲 step_metrics_buf
  由 MedicalRewardFn.__call__ 在每次 reward 计算后填写，
  callback 在 on_log 时读取并清空。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


# ---------------------------------------------------------------------------
# 全局缓冲：reward fn → callback 的单向通道
# ---------------------------------------------------------------------------

step_metrics_buf: Dict[str, float] = {}


def _flush_buf() -> Dict[str, float]:
    """返回当前缓冲内容并清空，线程不安全但单卡训练足够用。"""
    snapshot = dict(step_metrics_buf)
    step_metrics_buf.clear()
    return snapshot


# ---------------------------------------------------------------------------
# TRL → TensorBoard 键名映射
# TRL 0.29.x GRPOTrainer 在 logs dict 里使用的内部键名
# ---------------------------------------------------------------------------

_TRL_KEY_MAP: Dict[str, str] = {
    # loss
    "loss":                    "loss/policy",
    # policy
    "kl":                      "policy/approx_kl",
    "entropy":                 "policy/entropy",
    "grad_norm":               "policy/grad_norm",
    # reward（TRL 汇总后的总奖励）
    "reward":                  "reward/total",
    "reward_std":              "reward/std_within_group",
    # 生成长度（TRL 0.29.x 用这个键）
    "completions/mean_length": "gen/completion_length",
    "completion_length":       "gen/completion_length",
    # DAPO 专属指标（DAPOTrainer.compute_loss 写入 self._metrics，
    # TRL log() 会把它们直接放进 logs dict）
    "dapo_ratio_mean":         "dapo/ratio_mean",
    "dapo_clip_high_ratio":    "dapo/clip_high_ratio",
    "dapo_clip_low_ratio":     "dapo/clip_low_ratio",
    "dapo_n_zero_adv_groups":  "dapo/n_zero_adv_groups",
}


# ---------------------------------------------------------------------------
# 核心日志类
# ---------------------------------------------------------------------------

class GRPOTensorBoardLogger:
    """
    接收一个指标字典，按预定义分组写入 TensorBoard。

    用法::

        logger = GRPOTensorBoardLogger(log_dir="runs/medical_grpo")

        # 在每个 step 结束时调用
        logger.log_step(step=100, metrics={
            "policy/entropy":        1.23,
            "policy/approx_kl":      0.02,
            "reward/accuracy":       0.65,
            "reward/format":         0.88,
            "reward/total":          0.71,
            "reward/mean_advantage": 0.0001,
            "gen/completion_length": 312.4,
            "gen/eos_ratio":         0.92,
            "loss/policy":           0.034,
        })

        logger.close()  # 训练结束后调用
    """

    # 所有合法的指标键（用于校验 + 文档）
    EXPECTED_KEYS = {
        "policy/entropy",
        "policy/approx_kl",
        "policy/grad_norm",
        "reward/accuracy",
        "reward/format",
        "reward/total",
        "reward/mean_advantage",
        "reward/std_within_group",
        "gen/completion_length",
        "gen/length_std",
        "gen/n_unique_completions",
        "gen/eos_ratio",
        "loss/policy",
        # DAPO 专属
        "dapo/overlong_penalty_mean",
        "dapo/clip_high_ratio",
        "dapo/clip_low_ratio",
        "dapo/n_zero_adv_groups",
        "dapo/ratio_mean",
    }

    def __init__(self, log_dir: str, comment: str = ""):
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
        self._warned: set = set()
        print(f"[TensorBoard] log_dir = {log_dir}")
        print(f"[TensorBoard] launch : tensorboard --logdir {log_dir}")

    # ------------------------------------------------------------------

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """
        将 metrics dict 写入 TensorBoard。

        - 已知键直接写入对应 tag
        - 未知键以原名写入，并打印一次 warning
        - NaN / Inf 自动跳过
        - eval/<known> 视为已知（不打 warning）
        """
        for key, val in metrics.items():
            if val is None:
                continue
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue

            tag = key  # 未知键原样写入
            check_key = key[5:] if key.startswith("eval/") else key
            if check_key not in self.EXPECTED_KEYS and key not in self._warned:
                print(f"[TensorBoard] unknown metric key: '{key}' — written as-is")
                self._warned.add(key)

            self.writer.add_scalar(tag, float(val), global_step=step)

        self.writer.flush()

    # ------------------------------------------------------------------

    def close(self) -> None:
        self.writer.close()


# ---------------------------------------------------------------------------
# TRL TrainerCallback
# ---------------------------------------------------------------------------

class GRPOTensorBoardCallback(TrainerCallback):
    """
    注册到 TRL GRPOTrainer 的 TensorBoard 回调。

    职责：
      1. 在 on_log 时把 TRL 的内部日志键映射到 policy/ reward/ gen/ loss/ 分组
      2. 从 step_metrics_buf 读取 MedicalRewardFn 写入的细粒度指标
      3. 计算 reward/mean_advantage（基于 reward fn 缓存的 raw rewards）
      4. 合并后统一交给 GRPOTensorBoardLogger 写入

    参数
    ----
    log_dir       TensorBoard 日志目录
    num_generations  GRPO 组大小 G，用于计算 mean_advantage
    """

    def __init__(self, log_dir: str, num_generations: int = 4):
        self.logger = GRPOTensorBoardLogger(log_dir=log_dir)
        self.num_generations = num_generations

    # ------------------------------------------------------------------

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return

        step = state.global_step
        metrics: Dict[str, float] = {}

        # 检测 eval：TRL 在 evaluate() 内 fire on_log 时，logs 会带 eval_* 前缀键
        # （eval_runtime / eval_samples_per_second 等）；train 阶段没有这些键
        is_eval = any(k.startswith("eval_") for k in logs)

        # ── 1. TRL 内部日志 → 标准键名 ───────────────────────────────
        # eval 阶段 TRL key 形如 'eval_loss'，先去掉 'eval_' 前缀再查映射
        for trl_key, tb_key in _TRL_KEY_MAP.items():
            src_key = f"eval_{trl_key}" if is_eval else trl_key
            if src_key in logs:
                val = logs[src_key]
                if isinstance(val, (int, float)):
                    metrics[tb_key] = float(val)

        # ── 2. 从 reward fn 全局缓冲读取细粒度指标 ───────────────────
        buf = _flush_buf()
        metrics.update(buf)   # reward/accuracy, reward/format, gen/completion_length,
                               # gen/eos_ratio, _raw_rewards（内部使用，见下）

        # ── 3. 计算 reward/mean_advantage ────────────────────────────
        # raw_rewards 由 MedicalRewardFn 写入缓冲，形如 [r1, r2, ..., rG*N]
        # 每连续 G 个属于同一 prompt 的一组
        raw_rewards: Optional[List[float]] = metrics.pop("_raw_rewards", None)
        if raw_rewards and len(raw_rewards) >= self.num_generations:
            G = self.num_generations
            t = torch.tensor(raw_rewards, dtype=torch.float32)
            # 对齐到 G 的整数倍
            n_groups = len(t) // G
            t = t[: n_groups * G].reshape(n_groups, G)
            means = t.mean(dim=1, keepdim=True)
            stds  = t.std(dim=1, keepdim=True).clamp(min=1e-8)
            adv   = (t - means) / stds          # 理论均值 = 0
            metrics["reward/mean_advantage"] = adv.mean().item()

        # ── 3b. 组内 completion 长度标准差 ──────────────────────────
        comp_lens: Optional[List[int]] = metrics.pop("_completion_lengths", None)
        if comp_lens and len(comp_lens) >= self.num_generations:
            G = self.num_generations
            t = torch.tensor(comp_lens, dtype=torch.float32)
            n_groups = len(t) // G
            t = t[: n_groups * G].reshape(n_groups, G)
            metrics["gen/length_std"] = t.std(dim=1).mean().item()

        # ── 3c. 组内去重 completion 数（多样性） ────────────────────
        comp_hashes: Optional[List[int]] = metrics.pop("_completion_hashes", None)
        if comp_hashes and len(comp_hashes) >= self.num_generations:
            G = self.num_generations
            n_groups = len(comp_hashes) // G
            uniq_counts = []
            for i in range(n_groups):
                grp = comp_hashes[i * G : (i + 1) * G]
                uniq_counts.append(len(set(grp)))
            metrics["gen/n_unique_completions"] = sum(uniq_counts) / n_groups

        # ── 4. eval 阶段统一加 'eval/' 前缀，避免与 train 同 step 撞写 ─
        if is_eval and metrics:
            metrics = {f"eval/{k}": v for k, v in metrics.items()}

        # ── 5. 写入 TensorBoard ───────────────────────────────────────
        if metrics:
            self.logger.log_step(step, metrics)

    # ------------------------------------------------------------------

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self.logger.close()
