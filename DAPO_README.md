# DAPO Stage 2 中文文档

本文档介绍 `DAPO_stage2_single_gpu.py` 的设计、与 GRPO 的差异、运行方法和指标解读。

---

## 1. DAPO 是什么

DAPO（**D**ecoupled-clip & **D**ynamic s**A**mpling **P**olicy **O**ptimization）
是字节跳动 Seed 团队 2025-03 发布的论文 *DAPO: An Open-Source LLM RL System
at Scale* 提出的 GRPO 改进版本。相对于 GRPO，DAPO 有 4 项核心改动 + 1 项
KL 简化：

| # | 改动 | 论文原话 | 本仓库实现 |
|---|------|----------|-----------|
| 1 | **Clip-Higher** | `clamp(ratio, 1−ε_low, 1+ε_high)`，ε_low=0.2，ε_high=0.28 | ✅ K≥2 时生效（见 §9）|
| 2 | **Token-level Loss** | 全 batch token 加和 / token 总数 | ✅ |
| 3 | **Overlong Reward Shaping** | 长度接近 max_completion_length 时线性扣分 | ✅ |
| 4 | **Dynamic Sampling** | 多采样后过滤掉 advantage 全 0 的组 | ⚠️ 仅记录指标 `dapo/n_zero_adv_groups`，未实际重采样 |
| 5 | **β = 0** | 移除 KL 项 | ✅ 通过 `--kl_coef 0.0` 关闭 |

> **v2（off-policy）已上线**：通过 sampler 重复 + `_prepare_inputs` 缓存的方式
> 让同一批 rollout 跑 K 次 PPO inner-epoch（默认 K=2）。从第 2 个 inner-epoch
> 起 `new_logp ≠ old_logp`，`ratio = exp(new − old)` 偏离 1.0，Clip-Higher
> 真正起作用。详见 §9。

所以现在激活的是 **Clip-Higher + token-level loss + β=0 + overlong shaping** 共 4 项。

---

## 2. 与 GRPO 的对比

| 维度 | GRPO（`GRPO_stage2_single_gpu.py`） | DAPO（`DAPO_stage2_single_gpu.py`） |
|------|-----|-----|
| Loss 聚合 | `((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()` 按序列再按 batch 平均 | `(per_token_loss * mask).sum() / mask.sum()` 全 batch token 直接加和 |
| KL 系数 β | 默认 0.001 | 默认 0.0（`--kl_coef 0.0`） |
| Reward 截断 | 无 | 长度 > 90% 时线性扣分（最多 -1.0） |
| Clip-Higher | 无 | 非对称 [1−0.2, 1+0.28]，K=2 时生效 |
| PPO inner-epoch | 1（on-policy） | K（默认 2，off-policy；同一批 rollout 反复用）|
| Rollout 利用率 | 1×（rollout 完做 1 次 backward 就丢）| K×（同一批做 K 次 backward）|
| Reward 模型 | DeepSeek-R1（API） | DeepSeek-R1（API，**完全复用**） |
| 数据集 | medical-o1-verifiable-problem | 同上 |
| LoRA / 量化 / 优化器 | 4-bit NF4 + LoRA r=16 + paged AdamW 8-bit | 同上 |

**Token-level vs sequence-level 的直观差异：**
长完成的每个 token 对 loss 的贡献，sequence-level 下是 `1/L`（被序列长度归一化），
token-level 下是 `1/total_tokens`（统一权重）。所以 token-level 让长序列在
loss 里占的"分量"更重，论文报告这能让长尾推理样本梯度信号更稳。

---

## 3. 关键代码位置

### 3.1 `DAPO_stage2_single_gpu.py`

```
class DAPOTrainer(GRPOTrainerWithEntropy)
    __init__(eps_low, eps_high, num_ppo_epochs=K)
        # 强制 gradient_accumulation_steps == 1（K-cycle 与 grad accum 不能错位）

    _get_train_sampler()                            # 重复 G*K 次（GRPO 是 G）
        return RepeatRandomSampler(ds, num_generations * num_ppo_epochs, ...)

    _prepare_inputs(inputs)                         # K-cycle 缓存
        cycle = self._inner_epoch_counter % K
        if cycle == 0:
            result = super()._prepare_inputs(inputs)        # 真做 generate + reward
            with torch.inference_mode():
                result["old_per_token_logps"] = self._get_per_token_logps(...)
            self._cached_inputs = result
        else:
            result = self._cached_inputs                    # 复用，不再调 reward API
        self._inner_epoch_counter += 1
        return result

    compute_loss()
        per_token_logps     = self._get_per_token_logps(model, ...)   # new_logp，带梯度
        old_per_token_logps = inputs["old_per_token_logps"]            # 缓存的 rollout-时 logp
        ratio = exp(per_token_logps - old_per_token_logps)             # 真 ratio：cycle=0 时=1
        ratio_clipped = clamp(ratio, 1-ε_low, 1+ε_high)                # Clip-Higher 非对称
        per_token_loss = min(ratio*A, ratio_clipped*A)                 # PPO surrogate
        per_token_loss = -(per_token_loss - β * KL)
        loss = (per_token_loss * mask).sum() / mask.sum()              # token-level（DAPO §3.2）

class MedicalRewardFnAPI
    __init__(..., overlong_buffer_len, overlong_factor, max_completion_length)
    __call__()
        # 调 DeepSeek-R1 API 打分（与 GRPO 一致）
        # 之后再按长度做 overlong shaping（DAPO §3.3）：
        if toks > L - buf:
            penalty = min(1.0, (toks - (L-buf)) / buf)
            rewards[i] = max(0.0, rewards[i] - penalty * overlong_factor)
```

### 3.2 `grpo_utils/tb_logger.py`

新增了 5 个 `dapo/*` 指标到 `EXPECTED_KEYS`，避免训练时打 warning：

```python
"dapo/overlong_penalty_mean",   # reward fn 写入 step_metrics_buf
"dapo/clip_high_ratio",          # DAPOTrainer.compute_loss 写入 self._metrics
"dapo/clip_low_ratio",
"dapo/n_zero_adv_groups",
"dapo/ratio_mean",
```

`_TRL_KEY_MAP` 中也加了 `dapo_ratio_mean` → `dapo/ratio_mean` 等映射，
让 TRL log() 路径上的下划线键能被正确翻译。

### 3.3 `run_dapo.sh`

与 `run_grpo.sh` 多 5 个开关，并把若干步数翻 K 倍：
```bash
--num_ppo_epochs 2      # K：PPO inner-epoch 数；K=1 退化到 on-policy
--kl_coef 0.0           # 关 KL（β=0）
--epsilon_low 0.2       # Clip-Higher 下界
--epsilon_high 0.28     # Clip-Higher 上界（非对称，比下界宽）
--overlong_buffer_len 51  # 缓冲区长度（≈ 10% × 512）
--overlong_factor 1.0     # 最大扣分

# 配合 K=2 的步数翻倍（保持 rollout 数量与 grpo-426 可比）：
--max_steps 400         # 内层步数 = 200 × K
--save_steps 100        # 50 × K
--eval_steps 100        # 50 × K
--logging_steps 2       # 1 × K
--warmup_steps 40       # 20 × K
```

---

## 4. 运行方法

### 4.1 准备

确认：
- `./models/Qwen2.5-7B-sft-merged/` 存在（GRPO 阶段已生成）
- `SILICONFLOW_API_KEY` 已填入 `run_dapo.sh`（或 `export` 后再 source 脚本）
- 显卡空闲（A6000 49 GB）

### 4.2 冒烟测试（10 内层步 = 5 次 rollout，确认 off-policy 真生效）

把 `run_dapo.sh` 改成：
```bash
--num_ppo_epochs 2 --max_steps 10 --eval_steps 10 --eval_size 4
```

执行：
```bash
bash run_dapo.sh
```

确认：
- 不报错，写出 LoRA adapter
- TB 里 `dapo/ratio_mean` 的轨迹应是：**奇数 step 上 = 1.0**（每个 rollout 周期的
  inner-epoch 0），**偶数 step 上偏离 1.0**（inner-epoch 1，权重已更新过一次）
- `dapo/clip_high_ratio` 在偶数 step 上有非零值；奇数 step 上 = 0
- 若有完成长度超过 461 个词（512 × 0.9），`dapo/overlong_penalty_mean` 会非零

> **K=1 退化测试（可选）**：把 `--num_ppo_epochs 1`，`dapo/ratio_mean` 应**全程恒为 1.0**，
> 等价于当前 MVP 行为；用于验证缓存逻辑没破坏 single-epoch 路径。

### 4.3 完整训练（400 内层步 = 200 次 rollout）

按默认 `run_dapo.sh`（`--max_steps 400 --num_ppo_epochs 2`）执行：
```bash
bash run_dapo.sh
```

预计：单卡 A6000 **~28h**（比 MVP 多 ~50%，因为每次 rollout 多做一次 forward+backward）。
**Reward API 调用次数与 GRPO/MVP 持平**——同一批 rollout 在 K 个 inner-epoch 之间共享，
不会重复调 DeepSeek-R1。

### 4.4 监控

```bash
tensorboard --logdir train_logs --port 6006
```

可同时看到 `grpo-426` 和 `dapo-427` 两条曲线。eval 数据由 `tb_logger.py` 的 `eval/` 前缀
patch 自动落到独立 tag 下，不会和 train 互相覆盖。

### 4.5 合并 LoRA → 完整模型

```bash
python merge_grpo_lora.py \
  --base_model ./models/Qwen2.5-7B-sft-merged \
  --adapter ./grpo_ckpts/dapo-qwen2.5-7b-427/medical_dapo_qwen25_sft \
  --output_dir ./models/Qwen2.5-7B-dapo-merged
```

之后 `chat.py --model_path ./models/Qwen2.5-7B-dapo-merged` 即可对话测试。

---

## 5. TensorBoard 指标速查

### Train 阶段（与 GRPO 共有）

| Tag | 含义 | 期望走势 |
|-----|------|---------|
| `loss/policy` | 策略损失 | 平稳，DAPO 下数值约比 GRPO 小一个数量级（token-level 归一化导致）|
| `policy/approx_kl` | β=0 时仅作监控 | 可以高于 GRPO（无 KL 拉回） |
| `policy/entropy` | 策略熵 | 应缓慢下降，太快=过收敛 |
| `policy/grad_norm` | 梯度范数 | < 1.0 健康 |
| `reward/total` | 平均 reward（含 overlong 扣分后）| 单调上升 |
| `reward/accuracy` | reward ≥ 0.5 占比 | 上升 |
| `reward/format` | 格式正确占比 | 应迅速到 ~1.0 |
| `gen/completion_length` | 平均完成长度（words） | 在 overlong shaping 下应**不再贪心增长** |

### DAPO 专属

| Tag | 含义 | 解读 |
|-----|------|------|
| `dapo/ratio_mean` | exp(new_logp − old_logp) 的 token 平均 | inner-epoch 0 时 = 1.0（健康）；inner-epoch 1+ 通常在 0.95~1.05；持续 > 2 或 < 0.5 = LR 太大 |
| `dapo/clip_high_ratio` | ratio > 1+ε_high 的 token 占比 | < 5% 健康；> 20% 表示策略偏移过快，可调小 LR 或 ε_high 调大 |
| `dapo/clip_low_ratio` | ratio < 1−ε_low 的 token 占比 | 同上，对称解读 |
| `dapo/overlong_penalty_mean` | 当前 batch 平均扣分（0~1）| 0 = 没人超长；> 0 表示有 completion 触发 ramp |
| `dapo/n_zero_adv_groups` | 当前 batch 中 advantage std=0 的组数 | 0 健康；持续 > 20% 说明 reward 区分度不够，要调 reward fn |

---

## 6. 设计权衡 / 已知限制

1. **长度估计用 word-count，不是 token**
   原因：避免引入 policy tokenizer 依赖。误差 ±15%，对 ramp 形扣分够用。
   想要更准 → 改用 `tokenizer.encode(c)` 替换 `c.split()`。

2. **`max_steps` 是"内层步数"，不是 rollout 次数**
   `num_ppo_epochs=2` 时，`max_steps=400` 等于 200 次真 rollout。
   `save_steps`、`eval_steps`、`logging_steps`、`warmup_steps` 都是同样口径，
   全部按 K 翻倍才能保持时间间隔与 GRPO 一致。

3. **Dynamic Sampling 仅记录指标，未真正重采样**
   实现完整版需要在 `_prepare_inputs` 里 oversample 后过滤，单卡上代价较大。
   连续 reward (DS-R1, 0~1) 让 group 内方差很难全 0。
   实际 GRPO-426 跑下来 `reward/std_within_group` 长期 > 0，
   `dapo/n_zero_adv_groups` 当作"早期预警指标"够用了。

4. **`gradient_accumulation_steps` 必须 = 1**
   K-cycle 计数器假设每个 batch 一次 optimizer.step()。如果你需要更大的 effective
   batch，请加大 `--per_device_train_batch_size` 或 `--num_generations`，不要动 grad accum。
   脚本启动时会校验这一点，否则报错。

5. **公平性**
   想直接把 dapo-427 vs grpo-426 对比，请确认：
   - 同一 base model（`Qwen2.5-7B-sft-merged`）
   - 同一数据集 + 默认 shuffle 种子
   - 同一 eval 子集（`--eval_size 20` 取前 20）
   不一致就只能算"两次独立训练"，差异不能归因到 DAPO。

---

## 7. 故障排查

| 现象 | 原因 | 处理 |
|------|------|-----|
| `reward_api_key is empty` | shell 变量没传进去 | `run_dapo.sh` 里直接写 `SILICONFLOW_API_KEY="sk-..."` |
| TB 看不到 `dapo/*` | tb_logger 没更新 | 检查 `grpo_utils/tb_logger.py` 的 `EXPECTED_KEYS` 和 `_TRL_KEY_MAP` 是否包含 `dapo_*` |
| `loss/policy` 比 GRPO 小很多 | 正常 | token-level 归一化分母变大；看 reward/total 才是真信号 |
| `dapo/overlong_penalty_mean` 一直 0 | 完成都不超长 | 可能 max_completion_length 设得太宽；不影响训练 |
| OOM | per_device_batch * num_generations 过大 | 把 `--per_device_train_batch_size` 从 8 降到 4，或 `--num_generations` 从 8 降到 4 |

---

## 8. 文件清单

| 文件 | 作用 |
|------|------|
| `DAPO_stage2_single_gpu.py` | DAPO 训练主脚本 |
| `run_dapo.sh` | 启动命令（含所有超参）|
| `grpo_utils/tb_logger.py` | TB 日志，已扩展 `dapo/*` 键 |
| `merge_grpo_lora.py` | LoRA → 完整模型合并（DAPO/GRPO 通用）|
| `chat.py` | 终端交互式推理（DAPO/GRPO 通用，改 `--model_path` 即可）|

---

## 9. v2 off-policy 改造细节

### 9.1 为什么要这么做

DAPO 论文里 Clip-Higher 的核心公式 `clamp(ratio, 1−ε_low, 1+ε_high)` 中，
`ratio = π_new(a|s) / π_old(a|s)` 必须真的偏离 1.0 才能裁。但 TRL 0.15.2 的
`GRPOTrainer.compute_loss` 第 711 行写的是：

```python
per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
```

`logp − logp.detach()` 数值上恒为 0，`exp(0) = 1`——这是 TRL 用来**保留梯度**
而**伪造 ratio = 1** 的小技巧（等价于直接做 REINFORCE）。所以原版 GRPO 是
on-policy 单 epoch，rollout 完做一次 backward 就丢掉。

要做"真 DAPO"，必须让 π_old 和 π_new 来自不同时刻的策略——也就是论文里的
**多 epoch PPO**：rollout 一次后，用同一批样本反复更新模型 K 次。

### 9.2 实现方式：sampler 重复 + `_prepare_inputs` 缓存

TRL 0.15.2 没有 `num_iterations` 配置项（>=0.16 才加），我们用一个 trick 等效：

1. **Sampler 把每条 prompt 重复 G×K 次**（GRPO 原本 G 次）
   `RepeatRandomSampler(ds, num_generations * num_ppo_epochs)`
   于是 dataloader 会**连续吐 K 个相同的 batch**（同一组 G 条 prompt）。

2. **`_prepare_inputs` 里加 K-cycle 缓存**：
   ```python
   cycle = self._inner_epoch_counter % K
   if cycle == 0:
       result = super()._prepare_inputs(inputs)        # 真做 generate + reward
       result["old_per_token_logps"] = self._get_per_token_logps(...)  # rollout 时刻的 logp
       self._cached_inputs = result
   else:
       result = self._cached_inputs                    # 复用，不再调 reward API
   self._inner_epoch_counter += 1
   ```

3. **`compute_loss` 用缓存的 old_logp 算真 ratio**：
   ```python
   ratio = exp(per_token_logps - inputs["old_per_token_logps"])
   ```

效果：
- inner-epoch 0：模型还没更新，`new_logp == old_logp` → ratio ≈ 1.0
- inner-epoch 1+：上一个 inner-epoch 的 backward 已经把权重推走了 → ratio 偏离 1.0
- Clip-Higher 在偶数 step 上真正生效

### 9.3 关键代价 & 收益

| 指标 | MVP（K=1）| v2（K=2）|
|------|-----------|---------|
| 真 off-policy | ❌ | ✅ |
| Clip-Higher 生效 | ❌ | ✅ |
| Reward API 调用次数 | 200 | 200（缓存复用，不翻倍）|
| Forward + backward 次数 | 200 | 400（每批 rollout 用 2 次）|
| Wall-clock（200 rollout）| ~18h | ~28h |

### 9.4 Eval 阶段的特殊处理

eval 走 `prediction_step → _prepare_inputs → compute_loss`，但**不应**进入 K-cycle
缓存逻辑（每个 eval batch 都是独立的）。`_prepare_inputs` 用 `self.model.training`
区分：训练时进缓存逻辑，eval 时直接走 super()。`compute_loss` 也对 eval 做兼容
——`old_per_token_logps` 不在 inputs 里时退化为 `per_token_logps.detach()`，
此时 ratio = 1.0，clip 不会触发，eval loss 与训练定义一致。

### 9.5 已知坑 / 限制

1. **`gradient_accumulation_steps` 必须 = 1**
   K-cycle 与 grad accum 的边界不能错位。`__init__` 会校验。

2. **`_inner_epoch_counter` 是单卡逻辑**
   多卡 / DDP 下需要每个 rank 独立计数；本仓库是单卡所以无需考虑。

3. **K=2 是经验值**
   论文 K=1（off-policy 但只跑 1 epoch，靠 importance ratio 做修正）。
   本实现 K=2 让 ratio 一定偏离 1，更易观测 Clip-Higher。K=3 也能跑，
   wall-clock 再 +50%，收益不明显。

4. **Dynamic Sampling 没真正重采样**
   见 §6 第 3 条。
