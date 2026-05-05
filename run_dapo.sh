#!/bin/bash
# ===== Reward judge API (硅基流动 / DeepSeek-R1) =====
# 用前先在终端 export SILICONFLOW_API_KEY=sk-xxx，或直接在下面填 key
SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:?please export SILICONFLOW_API_KEY first}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python DAPO_stage2_single_gpu.py \
  --model_name_or_path ./models/Qwen2.5-7B-sft-merged \
  --reward_api_base https://api.siliconflow.cn/v1 \
  --reward_api_key "${SILICONFLOW_API_KEY}" \
  --reward_model_id deepseek-ai/DeepSeek-R1 \
  --output_dir ./grpo_ckpts/dapo-qwen2.5-7b-427 \
  --run_name medical_dapo_qwen25_sft \
  --num_ppo_epochs 2 \
  --max_steps 400 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_generations 8 \
  --max_prompt_length 512 \
  --max_completion_length 512 \
  --lr 5e-5 \
  --warmup_steps 40 \
  --save_steps 100 \
  --logging_steps 2 \
  --tb_log_dir ./train_logs/dapo-427 \
  --eval_steps 50 \
  --eval_size 10 \
  --kl_coef 0.0 \
  --epsilon_low 0.2 \
  --epsilon_high 0.28 \
  --overlong_buffer_len 51 \
  --overlong_factor 1.0
# 备注：max_steps 是"内层步数"。num_ppo_epochs=2 时，
#   实际 rollout 次数 = max_steps / 2 = 200，与 grpo-426 可比。
#   warmup/save/eval/logging 均按 K=2 倍率上调，保持时间间隔一致。
