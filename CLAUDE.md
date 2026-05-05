# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HuatuoGPT-o1 is a medical LLM trained for complex reasoning using a two-stage pipeline:
1. **Stage 1 (SFT)**: Supervised fine-tuning on Complex Chain-of-Thought (CoT) data
2. **Stage 2 (RL)**: PPO reinforcement learning guided by a medical verifier reward model

The model outputs in a structured format:
```
## Thinking
[reasoning process]

## Final Response
[answer]
```

## Environment Setup

```bash
# Preferred
uv sync

# Fallback
pip install -r requirements.txt
```

Python 3.10+. Key dependencies: `torch==2.5.1`, `transformers==4.46.2`, `trl==0.13.0`, `accelerate==0.34.2`, `deepspeed==0.15.4`, `vllm==0.6.4`.

## Common Commands

### Stage 1 — SFT Training
```bash
accelerate launch \
  --config_file ./configs/deepspeed_zero3.yaml \
  --num_processes 8 \
  SFT_stage1.py \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --data_path FreedomIntelligence/medical-o1-reasoning-SFT \
  --output_dir ./ckpts \
  --max_seq_len 8192 \
  --train_bsz_per_gpu 2 \
  --learning_rate 5e-6
```

### Stage 2 — PPO/RL Training
```bash
accelerate launch \
  --config_file ./configs/deepspeed_zero3.yaml \
  --num_processes 8 \
  RL_stage2.py \
  --model_name_or_path FreedomIntelligence/HuatuoGPT-o1-8B \
  --reward_model_path FreedomIntelligence/medical_o1_verifier_3B \
  --value_model_path meta-llama/Llama-3.2-3B-Instruct \
  --dataset_name FreedomIntelligence/medical-o1-verifiable-problem \
  --total_episodes 20000
```

### Evaluation
```bash
# Deploy model
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path FreedomIntelligence/HuatuoGPT-o1-8B \
  --port 30000

# Run eval
python evaluation/eval.py \
  --model_name FreedomIntelligence/HuatuoGPT-o1-8B \
  --eval_file evaluation/data/eval_data.json \
  --port 30000

# Stop server
bash evaluation/kill_sglang_server.sh
```

### Data Pipeline
```bash
python download_datasets.py

python construct_verifiable_medical_problems.py \
  --data_path data/demo_data.json --filter_data \
  --model_name gpt-4o --api_key YOUR_KEY

python cot_pipeline_accelerated.py \
  --data_path data/medical_o1_verifiable_problem.json \
  --api_url https://api.siliconflow.cn/v1/chat/completions \
  --generator_model_name Qwen/QwQ-32B \
  --num_process 8
```

## Architecture

### Data Flow
```
download_datasets.py
  → data/{medqa_usmle_train, medmcqa_train, mmlu_pro}.json

construct_verifiable_medical_problems.py
  → data/medical_o1_verifiable_problem.json  (open-ended Q + ground-truth A)

cot_pipeline_accelerated.py  (3-chunk reasoning via external API or local model)
  → output_data/...json  (adds Long_CoT field)

postprocess_verified_long_cot.py
  → SFT data: {Question, Complex_CoT, Response}

SFT_stage1.py → ckpts/
RL_stage2.py  → final model
evaluation/eval.py → accuracy metrics
```

### Key Files

| File | Role |
|------|------|
| `SFT_stage1.py` | Stage 1 entry point; formats data as `## Thinking\n{Complex_CoT}\n\n## Final Response\n{Response}` |
| `RL_stage2.py` | Stage 2 PPO entry point; uses TRL + custom trainer |
| `ppo_utils/ppo_trainer_medo1.py` | Custom PPO trainer; `get_reward_o1()` extracts `## Final Response` section and scores with verifier |
| `ppo_utils/ppo_config_medo1.py` | PPO hyperparameter dataclass |
| `cot_pipeline_accelerated.py` | Chunked CoT generation (symptom extraction → differential diagnosis → conclusion) |
| `construct_verifiable_medical_problems.py` | GPT-4o–based conversion of MCQ → open-ended verifiable problems |
| `evaluation/eval.py` | Calls SGLang-served model via OpenAI-compatible API |
| `evaluation/scorer.py` | Multi-strategy answer extraction (regex, text search, similarity) with per-dataset breakdown |
| `configs/deepspeed_zero3.yaml` | ZeRO-3, BF16, CPU offload config for 8+ GPU runs |

### SFT Data Format
```json
{"Question": "...", "Complex_CoT": "...", "Response": "..."}
```

### Verifiable Problem Format
```json
{"Open-ended Verifiable Question": "...", "Ground-True Answer": "..."}
```

## Coding Conventions

- Root-level scripts (no `src/` package), `argparse` for CLI, paths resolved from repo root
- 4-space indent, `snake_case` for functions/args, `PascalCase` for classes
- No `black`/`ruff`/`pytest` configured — keep imports, arg names, and JSON paths consistent; avoid large reformats
- Smoke-test data/training changes with a small JSON file before full runs
- New automated tests go in `tests/test_*.py` with lightweight fixture data
- Never commit API keys, model weights paths, cache files, or temporary logs
