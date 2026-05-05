#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /workspace/HuatuoGPT-o1/models/Qwen2.5-7B-sft-merged \
  --served-model-name default \
  --port 30000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
