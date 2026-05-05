#!/bin/bash
# ==============================================================
# 第 0 步：下载模型（只需跑一次，之后会跳过已下载的文件）
# ==============================================================
bash download_model.sh

# ==============================================================
# 第 1 步：本地 GPU 推理（默认）
# ==============================================================
CUDA_VISIBLE_DEVICES=0 uv run python construct_verifiable_medical_problems.py \
  --data_path data/init/medmcqa_train.json \
  --model_name "models/DeepSeek-R1-Distill-Qwen-7B" \
  --num_process 2 \
  --filter_data \
  --limit_num 5


CUDA_VISIBLE_DEVICES=0 uv run python construct_verifiable_medical_problems.py \
  --data_path data/init/medmcqa_train.json \
  --model_name "models/Qwen2.5-7B-Instruct" \
  --filter_data \
  --batch_size 128 \
  --max_tokens_filter 1024 \
  --max_tokens_rewrite 1024


  
  python -u search_for_complex_reasoning_path.py \
    --data_path data/medical_o1_verifiable_problem_2000.json \
    --generator_model_name Qwen/QwQ-32B \
    --small_judge_model_name Qwen/Qwen2.5-7B-Instruct \
    --large_judge_model_name deepseek-ai/DeepSeek-V3 \
    --summary_model_name deepseek-ai/DeepSeek-V3 \
    --num_process 2

- 生成思维链模型：Qwen/QwQ-32B
- summary 模型：Qwen/Qwen2.5-7B-Instruct
- 验证模型：Qwen/Qwen2.5-7B-Instruct


1 把 judge prompt 改成“明确知道自己只在审当前 chunk，不要求中间步骤提前写最终答案”；二
2 把 chunk 的小模型验证改成最多 3 次独立审查，3 次都不过就对该样本硬拒绝，不再继续回炉生成。









  export SILICONFLOW_API_KEY="${SILICONFLOW_API_KEY:?please export SILICONFLOW_API_KEY first}"

  ./.venv312/bin/python search_for_complex_reasoning_path.py \
      --data_path data/medical_o1_verifiable_problem.json \
      --limit_num 4000 \
      --api_key "$SILICONFLOW_API_KEY" \
      --generator_model_name Qwen/QwQ-32B \
      --summary_model_name Qwen/QwQ-32B \
      --judge_backend local \
      --local_judge_model_path models/Qwen2.5-7B-Instruct \
      --local_judge_max_gpu_memory_gib 20 \
      --local_judge_batch_size 8 \
      --generator_backend batch \
      --batch_completion_window 24h \
      --batch_poll_interval 60 \
      --num_process 16 \
      --max_reasoning_chain_attempts 3 \
      --max_chunk_retries 3 \
      --max_chunk_judge_attempts 3 \
      --compression_trigger_fail_depth 2 \
      --use_json_mode \
      --mode_suffix prod_4000_templategate_localjudge_batch &>> sft-data-416.log