"""
合并 GRPO LoRA adapter 到 SFT base，得到完整可用模型。

注意：base 必须用 bf16 加载（不能 4-bit），否则 LoRA 权重写不回去，会掉点。
"""

import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_model",
        default="./models/Qwen2.5-7B-sft-merged",
        help="SFT base model 路径（bf16 加载）",
    )
    p.add_argument(
        "--adapter",
        default="./grpo_ckpts/grpo-qwen2.5-7b-422/medical_grpo_qwen25_sft",
        help="GRPO LoRA adapter 路径（含 adapter_config.json + adapter_model.safetensors）",
    )
    p.add_argument(
        "--output_dir",
        default="./models/Qwen2.5-7B-grpo-merged",
        help="合并后模型输出目录",
    )
    p.add_argument(
        "--max_shard_size",
        default="4GB",
        help="分片大小（safetensors）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── 0. 路径校验，防止跑了一半发现路径不对 ───────────────────
    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(f"base_model 不存在: {args.base_model}")
    adapter_cfg = os.path.join(args.adapter, "adapter_config.json")
    if not os.path.isfile(adapter_cfg):
        raise FileNotFoundError(f"adapter_config.json 不存在: {adapter_cfg}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] loading base in bf16  →  {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    print(f"[2/4] attaching LoRA adapter  →  {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[3/4] merging LoRA into base weights")
    model = model.merge_and_unload()

    print(f"[4/4] saving merged model  →  {args.output_dir}")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    # tokenizer 优先取 adapter 目录下的（与训练时一致），否则回落到 base
    tok_src = args.adapter if os.path.isfile(
        os.path.join(args.adapter, "tokenizer_config.json")
    ) else args.base_model
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(args.output_dir)

    # 拷贝 chat_template.jinja（如果有），sglang/vLLM 需要
    src_tpl = os.path.join(args.base_model, "chat_template.jinja")
    if os.path.isfile(src_tpl):
        shutil.copy2(src_tpl, args.output_dir)

    print(f"\ndone. merged model at: {args.output_dir}")
    print(f"size:")
    os.system(f"du -sh {args.output_dir}")


if __name__ == "__main__":
    main()
