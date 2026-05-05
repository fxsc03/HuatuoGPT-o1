#!/bin/bash
cd /workspace/HuatuoGPT-o1/evaluation
python eval.py \
  --model_name /workspace/HuatuoGPT-o1/models/Qwen2.5-7B-sft-merged \
  --eval_file data/eval_data.json \
  --port 30000 \
  --max_new_tokens 4096 \
  --strict_prompt \
  --batch_size 64
