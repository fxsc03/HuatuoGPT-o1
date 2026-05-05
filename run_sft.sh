#!/bin/bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --config_file ./configs/accelerate_single_gpu.yaml \
  SFT_stage1.py \
  --model_path /workspace/HuatuoGPT-o1/models/Qwen2.5-7B-Instruct \
  --data_path ./data/medical_o1_reasoning_sft_en_mix.json \
  --output_dir ./ckpts/sft-qwen2.5-7b-qlora-425-mild \
  --max_seq_len 2048 \
  --train_bsz_per_gpu 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_rates 0.05 \
  --max_steps 1000 \
  --lora_rank 8 \
  --eval_steps 200 \
  --val_ratio 0.02 \
  --max_ckpts 3 
