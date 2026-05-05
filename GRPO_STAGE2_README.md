# GRPO Stage 2 训练说明

本文档介绍如何使用 **verl 框架** 对 HuatuoGPT-o1 进行 GRPO（Group Relative Policy Optimization）强化学习训练。

---

## 文件结构

```
GRPO_stage2.py                  # 主训练入口脚本
grpo_utils/
  __init__.py
  grpo_reward_medo1.py          # 医疗奖励函数模块（verl RewardManager）
configs/
  grpo_config.yaml              # verl GRPO 配置模板
```

---

## GRPO vs PPO 对比

| 维度 | PPO（RL_stage2.py） | GRPO（GRPO_stage2.py） |
|------|---------------------|------------------------|
| 框架 | TRL + DeepSpeed ZeRO-3 | verl + Ray + FSDP/vLLM |
| Value model | 需要（单独 3B 模型） | **不需要** |
| 推理引擎 | HuggingFace generate | **vLLM**（更快） |
| 每个 prompt 的回复数 | 1 | G 个（默认 8）|
| Advantage 计算 | GAE（依赖 value model） | 组内相对奖励归一化 |
| GPU 利用 | Actor/Ref/Reward/Value 共享 | Actor、Rollout、Ref 分组部署 |

### GRPO 核心思路

对每个 prompt，并行生成 G 个回复 $\{y_1, \ldots, y_G\}$，计算各自奖励 $r_i$，然后以组内均值/标准差做归一化：

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \epsilon}
$$

用归一化后的 advantage 更新策略，无需 critic/value model。

---

## 安装依赖

```bash
# 基础依赖
pip install verl omegaconf ray[default]

# 或从源码安装最新版 verl
pip install git+https://github.com/volcengine/verl.git

# vLLM（rollout 加速）
pip install vllm>=0.6.4
```

其余依赖与原项目 `requirements.txt` 一致。

---

## 快速开始

### 方式一：使用 GRPO_stage2.py（推荐）

脚本会自动完成数据准备、配置生成和训练启动：

```bash
python GRPO_stage2.py \
    --model_name_or_path FreedomIntelligence/HuatuoGPT-o1-8B \
    --reward_model_path  FreedomIntelligence/medical_o1_verifier_3B \
    --dataset_name       FreedomIntelligence/medical-o1-verifiable-problem \
    --output_dir         ./grpo_ckpts \
    --run_name           medical_grpo_8b \
    --total_episodes     20000 \
    --grpo_group_size    8 \
    --n_gpus_per_node    8
```

也可以使用本地 JSON 数据集（与 RL_stage2.py 格式相同）：

```bash
python GRPO_stage2.py \
    --dataset_name data/medical_o1_verifiable_problem.json \
    --model_name_or_path ./ckpts/your-sft-checkpoint \
    ...
```

### 方式二：直接使用 verl Hydra 入口

先手动准备 parquet 数据，再直接调用 verl 的训练入口：

```bash
# 1. 准备数据（仅运行 GRPO_stage2.py 中的 prepare_parquet 逻辑）
python - <<'EOF'
import json, random
from transformers import AutoTokenizer
from GRPO_stage2 import prepare_parquet

tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/HuatuoGPT-o1-8B")
prepare_parquet("data/medical_o1_verifiable_problem.json", tokenizer, 512, "data/grpo_train.parquet")
EOF

# 2. 启动训练（使用 configs/grpo_config.yaml）
python -m verl.trainer.main_ppo \
    --config-path configs \
    --config-name grpo_config \
    trainer.experiment_name=my_run \
    actor_rollout_ref.model.path=FreedomIntelligence/HuatuoGPT-o1-8B
```

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--grpo_group_size` | 8 | 每个 prompt 生成的回复数 G，越大 variance 越低但显存越多 |
| `--train_batch_size` | 256 | 每步消耗的 **prompt 数**（实际 token batch = 256 × 8 = 2048） |
| `--max_response_length` | 4096 | 生成上限；HuatuoGPT-o1 输出含 Thinking，需要较长上限 |
| `--kl_coef` | 0.001 | KL 惩罚系数，防止策略偏离参考模型太远 |
| `--lr` | 1e-6 | 学习率，建议比 SFT 低一个数量级 |
| `--actor_gpu_frac` | 0.5 | Actor+Ref 占用的 GPU 比例，其余给 vLLM rollout |

---

## 奖励函数说明

奖励逻辑与 PPO 版本（`ppo_utils/ppo_trainer_medo1.py`）完全一致：

```
生成回复
    │
    ├─ 无 "## Thinking" + "## Final Response" 结构 → 奖励 0.0
    │
    └─ 有正确结构
           │
           └─ 调用 medical_o1_verifier_3B（2分类：True/False）
                  │
                  ├─ P(True) > 0.4 → 奖励 1.0
                  └─ P(True) ≤ 0.4 → 奖励 0.1
```

奖励模型（`MedicalRewardManager`）在 verl worker 初始化时加载一次，后续复用缓存。

---

## GPU 显存需求参考

| 模型规模 | GPU 数量 | 推荐配置 |
|----------|----------|---------|
| 8B（LLaMA-3.1） | 8 × A100 80G | `--n_gpus_per_node 8 --actor_gpu_frac 0.5` |
| 70B（LLaMA-3.1） | 16 × A100 80G | 2 节点，`tensor_model_parallel_size: 4` |
| 7B（Qwen2.5） | 8 × A100 80G | 同 8B 配置 |
| 72B（Qwen2.5） | 16 × A100 80G | 同 70B 配置 |

`grpo_group_size` 增大时显存线性增加，可通过减小 `ppo_micro_batch_size` 抵消。

---

## 训练流程图

```
GRPO_stage2.py
│
├─ 1. 数据准备
│     JSON → parquet（verl 格式）
│     字段：data_source / prompt / ability / reward_model / extra_info
│
├─ 2. 配置生成
│     build_config() → OmegaConf → grpo_run_config.yaml
│
└─ 3. verl RayPPOTrainer
       │
       ├─ Actor Worker（FSDP）
       │     接收 advantages，执行 clipped surrogate 更新
       │
       ├─ Rollout Worker（vLLM）
       │     每个 prompt 生成 G=8 个回复
       │
       ├─ Reference Worker（FSDP, CPU offload）
       │     计算 log prob for KL
       │
       └─ Reward Manager（MedicalRewardManager）
             medical_o1_verifier_3B → scalar reward → token-level scores
             GRPO advantage = (r_i - mean) / std  [within group]
```

---

## 常见问题

**Q: 为什么不需要 value model？**  
A: GRPO 用同一 prompt 的多个回复之间的相对奖励代替了价值函数估计，避免训练额外的 critic，简化流程。

**Q: `ppo_mini_batch_size` 如何设置？**  
A: 应等于 `train_batch_size × grpo_group_size`（即每步处理的所有回复总数）。默认 256 × 8 = 2048。

**Q: 如何监控训练进度？**  
A: verl 默认写入 W&B（离线模式）。关注指标：  
- `reward/mean`：平均奖励（目标：趋近 1.0）  
- `reward/frac_correct`：P(True)>0.4 的比例  
- `reward/frac_no_format`：格式违规比例（目标：趋近 0）  
- `actor/kl`：KL 散度（过大则适当增加 `kl_coef`）

**Q: 遇到 OOM 怎么办？**  
1. 减小 `grpo_group_size`（如从 8→4）  
2. 减小 `ppo_micro_batch_size`（如从 4→2）  
3. 对 actor 开启 `optimizer_offload: true`  
4. 减小 `max_response_length`

**Q: 能否使用 LoRA 训练？**  
A: 支持，需在 `actor_rollout_ref.model` 下添加 peft 相关配置（verl 0.3+ 支持 LoRA）。本代码暂未集成，如需使用请参考 verl 官方 LoRA 示例。
