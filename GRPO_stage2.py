"""
GRPO Stage 2 Training with verl Framework
==========================================
Group Relative Policy Optimization (GRPO) for HuatuoGPT-o1 medical reasoning.

Key differences from PPO (RL_stage2.py):
  - No value model / critic needed
  - Generates G responses per prompt, normalises rewards within the group
  - Uses verl for distributed training (Ray + FSDP/vLLM) instead of TRL + DeepSpeed

Usage:
  python GRPO_stage2.py \
      --model_name_or_path FreedomIntelligence/HuatuoGPT-o1-8B \
      --reward_model_path  FreedomIntelligence/medical_o1_verifier_3B \
      --dataset_name       FreedomIntelligence/medical-o1-verifiable-problem \
      --output_dir         ./grpo_ckpts \
      --run_name           medical_grpo_8b \
      --total_episodes     20000 \
      --grpo_group_size    8 \
      --n_gpus_per_node    8
"""

import os
import json
import random
import argparse
import subprocess
import sys

import pandas as pd
from transformers import AutoTokenizer

os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_parquet(data_path: str, tokenizer, max_prompt_length: int, out_path: str):
    """
    Convert the project's JSON format into the parquet schema expected by verl.

    Input JSON schema  : [{"Open-ended Verifiable Question": ..., "Ground-True Answer": ...}, ...]
    Output parquet cols: data_source | prompt | ability | reward_model | extra_info
    """
    if os.path.isfile(data_path):
        with open(data_path) as f:
            raw = json.load(f)
    else:
        # Try HuggingFace Hub
        from datasets import load_dataset
        ds = load_dataset(data_path, split="train")
        raw = list(ds)

    records = []
    for i, item in enumerate(raw):
        question = item.get("Open-ended Verifiable Question", "").strip()
        answer   = item.get("Ground-True Answer", "").strip()
        if not question or not answer:
            continue

        # Apply chat template so the model sees the same format as during SFT
        messages = [{"role": "user", "content": question}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # verl uses the token count to filter / truncate prompts
        token_len = len(tokenizer.encode(prompt_str, add_special_tokens=False))
        if token_len > max_prompt_length:
            continue

        records.append({
            "data_source":   "medical_o1",
            "prompt":        prompt_str,
            "ability":       "medical_reasoning",
            "reward_model":  json.dumps({"style": "model", "ground_truth": answer}),
            "extra_info":    json.dumps({"index": i}),
        })

    print(f"  {len(raw)} raw → {len(records)} usable records → {out_path}")
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GRPO Stage-2 training with verl")

    # Model
    p.add_argument("--model_name_or_path",  default="FreedomIntelligence/HuatuoGPT-o1-8B")
    p.add_argument("--reward_model_path",   default="FreedomIntelligence/medical_o1_verifier_3B")

    # Data
    p.add_argument("--dataset_name",        default="FreedomIntelligence/medical-o1-verifiable-problem",
                   help="Local JSON file path or HuggingFace dataset id")
    p.add_argument("--data_dir",            default="./data")
    p.add_argument("--eval_ratio",          type=float, default=0.05)
    p.add_argument("--max_prompt_length",   type=int, default=512)
    p.add_argument("--max_response_length", type=int, default=4096)

    # Output
    p.add_argument("--output_dir",          default="./grpo_ckpts")
    p.add_argument("--run_name",            default="medical_grpo")

    # Training
    p.add_argument("--total_episodes",      type=int, default=20000)
    p.add_argument("--train_batch_size",    type=int, default=256,
                   help="Number of prompts consumed per global update step")
    p.add_argument("--grpo_group_size",     type=int, default=8,
                   help="G – responses generated per prompt for group advantage")
    p.add_argument("--lr",                  type=float, default=1e-6)
    p.add_argument("--kl_coef",             type=float, default=0.001)
    p.add_argument("--clip_ratio",          type=float, default=0.2)
    p.add_argument("--temperature",         type=float, default=0.8)
    p.add_argument("--save_freq",           type=int, default=200)
    p.add_argument("--test_freq",           type=int, default=100)

    # Hardware
    p.add_argument("--n_gpus_per_node",     type=int, default=8)
    p.add_argument("--n_nodes",             type=int, default=1)
    p.add_argument("--actor_gpu_frac",      type=float, default=0.5,
                   help="Fraction of GPUs dedicated to actor (rest go to rollout / ref)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# verl config builder
# ---------------------------------------------------------------------------

def build_config(args, train_parquet: str, eval_parquet: str, output_dir: str) -> dict:
    """
    Return an OmegaConf-compatible nested dict that will be serialised to YAML
    and passed to verl's RayPPOTrainer.
    """
    total_gpus = args.n_gpus_per_node * args.n_nodes

    # Split GPUs: actor+ref on half, rollout (vLLM) on the other half
    actor_gpus  = max(1, int(total_gpus * args.actor_gpu_frac))
    rollout_gpus = max(1, total_gpus - actor_gpus)

    config = {
        "trainer": {
            "total_epochs":        -1,           # driven by total_training_steps
            "total_training_steps": args.total_episodes,
            "project_name":        "HuatuoGPT-o1-GRPO",
            "experiment_name":     args.run_name,
            "logger":              ["console", "wandb"],
            "default_local_dir":   output_dir,
            "default_hdfs_dir":    None,
            "save_freq":           args.save_freq,
            "test_freq":           args.test_freq,
        },
        "data": {
            "train_files":         train_parquet,
            "val_files":           eval_parquet,
            "train_batch_size":    args.train_batch_size,
            "max_prompt_length":   args.max_prompt_length,
            "max_response_length": args.max_response_length,
            # raw prompt/chat strings are forwarded to the reward manager
            "return_raw_input_ids": True,
            "return_raw_chat":     True,
        },
        "actor_rollout_ref": {
            "model": {
                "path":                        args.model_name_or_path,
                "enable_gradient_checkpointing": True,
                "use_remove_padding":          True,
            },
            "actor": {
                "strategy":          "fsdp",
                "optim": {
                    "lr":               args.lr,
                    "weight_decay":     0.01,
                    "warmup_steps":     10,
                    "lr_scheduler_type": "cosine",
                },
                # ppo_mini_batch_size = prompts × group_size slices per mini-batch
                "ppo_mini_batch_size": args.train_batch_size * args.grpo_group_size,
                "ppo_micro_batch_size": 4,
                "use_kl_loss":       True,
                "kl_loss_coef":      args.kl_coef,
                "kl_loss_type":      "low_var_kl",
                "entropy_coeff":     0.001,
                "clip_ratio":        args.clip_ratio,
                "fsdp_config": {
                    "param_offload":     False,
                    "grad_offload":      False,
                    "optimizer_offload": False,
                },
            },
            "rollout": {
                "name":                     "vllm",
                "temperature":              args.temperature,
                "top_p":                    0.95,
                "top_k":                    -1,
                # GRPO: generate G responses per prompt
                "n":                        args.grpo_group_size,
                "max_model_len":            args.max_prompt_length + args.max_response_length,
                "gpu_memory_utilization":   0.85,
                "tensor_model_parallel_size": max(1, rollout_gpus // args.n_nodes),
                "ignore_eos":               False,
                "enforce_eager":            False,
                "free_cache_engine":        True,
            },
            "ref": {
                "strategy":   "fsdp",
                "fsdp_config": {"param_offload": True},
                "log_prob_micro_batch_size": 4,
            },
        },
        "algorithm": {
            "adv_estimator":          "grpo",
            "kl_ctrl": {
                "type":    "fixed",
                "kl_coef": args.kl_coef,
            },
            # normalise advantages across the whole batch (not per-prompt group)
            "adv_normlize_type":      "batch",
        },
        "reward_model": {
            "enable":          True,
            # our custom reward manager (see grpo_utils/grpo_reward_medo1.py)
            "reward_manager":  "grpo_utils.grpo_reward_medo1.MedicalRewardManager",
            "reward_kwargs": {
                "reward_model_path": args.reward_model_path,
                "max_reward_length": 4000,
            },
        },
        "resource_pool_manager": {
            "process_on_nodes": [args.n_gpus_per_node] * args.n_nodes,
        },
    }
    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve output dir
    output_dir = args.output_dir
    if args.run_name not in output_dir:
        output_dir = os.path.join(output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Prepare parquet datasets
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if '<|eot_id|>' in tokenizer.vocab:
        tokenizer.pad_token = '<|end_of_text|>'
        tokenizer.pad_token_id = tokenizer.encode('<|end_of_text|>', add_special_tokens=False)[0]

    print("Preparing training data …")
    # Temporarily load all records to do train/eval split
    if os.path.isfile(args.dataset_name):
        with open(args.dataset_name) as f:
            raw = json.load(f)
    else:
        from datasets import load_dataset
        raw = list(load_dataset(args.dataset_name, split="train"))

    random.shuffle(raw)
    eval_num = min(int(len(raw) * args.eval_ratio), 200)

    tmp_train = os.path.join(args.data_dir, "_grpo_raw_train.json")
    tmp_eval  = os.path.join(args.data_dir, "_grpo_raw_eval.json")
    with open(tmp_train, "w") as f:
        json.dump(raw[eval_num:], f)
    with open(tmp_eval,  "w") as f:
        json.dump(raw[:eval_num], f)

    train_parquet = os.path.join(args.data_dir, "grpo_train.parquet")
    eval_parquet  = os.path.join(args.data_dir, "grpo_eval.parquet")
    prepare_parquet(tmp_train, tokenizer, args.max_prompt_length, train_parquet)
    prepare_parquet(tmp_eval,  tokenizer, args.max_prompt_length, eval_parquet)

    # ------------------------------------------------------------------
    # 2. Build and save verl YAML config
    # ------------------------------------------------------------------
    try:
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError("omegaconf is required: pip install omegaconf")

    cfg_dict  = build_config(args, train_parquet, eval_parquet, output_dir)
    cfg       = OmegaConf.create(cfg_dict)
    cfg_path  = os.path.join(output_dir, "grpo_run_config.yaml")
    OmegaConf.save(cfg, cfg_path)
    print(f"Config saved → {cfg_path}")

    # ------------------------------------------------------------------
    # 3. Launch verl GRPO trainer
    # ------------------------------------------------------------------
    try:
        import ray
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    except ImportError:
        raise ImportError(
            "verl is required. Install with:\n"
            "  pip install verl\n"
            "or from source:\n"
            "  pip install git+https://github.com/volcengine/verl.git"
        )

    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "false"}},
            ignore_reinit_error=True,
        )

    trainer = RayPPOTrainer(config=cfg)
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
