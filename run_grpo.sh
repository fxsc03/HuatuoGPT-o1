#!/bin/bash
# ===== Reward judge API (硅基流动 / DeepSeek-R1) =====
# 用前先在终端 export SILICONFLOW_API_KEY=sk-xxx，或直接在下面填 key
SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:?please export SILICONFLOW_API_KEY first}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python GRPO_stage2_single_gpu.py \
  --model_name_or_path ./models/Qwen2.5-7B-sft-merged \
  --reward_api_base https://api.siliconflow.cn/v1 \
  --reward_api_key "${SILICONFLOW_API_KEY}" \
  --reward_model_id deepseek-ai/DeepSeek-R1 \
  --output_dir ./grpo_ckpts/grpo-qwen2.5-7b-422 \
  --run_name medical_grpo_qwen25_sft \
  --max_steps 200 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_generations 8 \
  --max_prompt_length 512 \
  --max_completion_length 512 \
  --lr 5e-5 \
  --warmup_steps 20 \
  --save_steps 50 \
  --logging_steps 1 \
  --tb_log_dir ./train_logs/grpo-426 \
  --eval_steps 50 \
  --eval_size 20
