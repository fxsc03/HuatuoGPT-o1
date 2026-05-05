#!/bin/bash
# DAPO-430 从 checkpoint-500 恢复训练
SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:?please export SILICONFLOW_API_KEY first}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python DAPO_stage2_single_gpu.py \
  --model_name_or_path ./models/Qwen2.5-7B-sft-merged \
  --reward_api_base https://api.siliconflow.cn/v1 \
  --reward_api_key "${SILICONFLOW_API_KEY}" \
  --reward_model_id deepseek-ai/DeepSeek-R1 \
  --output_dir ./grpo_ckpts/dapo-430 \
  --run_name medical_dapo_qwen25_430 \
  --num_ppo_epochs 5 \
  --max_steps 1000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_generations 8 \
  --max_prompt_length 512 \
  --max_completion_length 512 \
  --lr 1e-5 \
  --warmup_steps 100 \
  --save_steps 500 \
  --logging_steps 5 \
  --tb_log_dir ./train_logs/dapo-430 \
  --eval_steps 250 \
  --eval_size 10 \
  --kl_coef 0.0 \
  --epsilon_low 0.2 \
  --epsilon_high 0.28 \
  --overlong_buffer_len 51 \
  --overlong_factor 1.0 \
  --dynamic_sampling_threshold 0.01 \
  --resume_from_checkpoint ./grpo_ckpts/dapo-430/medical_dapo_qwen25_430/checkpoint-500
