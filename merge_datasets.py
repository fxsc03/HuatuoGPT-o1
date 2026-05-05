"""
按照 HuatuoGPT-o1 论文 Section 4.1 的实验设置合并数据集。

论文原文：
  - 40K medical verification dataset D from MedQA-USMLE + MedMCQA
    - 20K for SFT (stage 1)
    - 20K for RL  (stage 2)
  - 4K unconverted data (原始选择题格式) from D, to enhance generalization
  - 5K general verification questions from MMLU-Pro (non-medical tracks)

输出文件 (全部存到 data/ 目录):
  1. medical_40k.json          — 40K 医学题，送入 construct_verifiable_medical_problems.py 转换
  2. mmlu_pro_5k.json          — 5K 通用题，送入 construct_verifiable_medical_problems.py 转换
  3. medical_sft_20k.json      — 转换后用: SFT stage1 的 20K 医学子集
  4. medical_rl_20k.json       — 转换后用: RL  stage2 的 20K 医学子集
  5. medical_unconverted_4k.json — 4K 原始选择题 (不经过转换, 直接混入训练)
"""

import json
import random
import argparse
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {len(data):,} items to {path}")


def main():
    parser = argparse.ArgumentParser(description="Merge datasets per HuatuoGPT-o1 paper §4.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medical_total", type=int, default=40000,
                        help="Total medical samples (MedQA + MedMCQA)")
    parser.add_argument("--mmlu_total", type=int, default=5000,
                        help="General MMLU-Pro samples")
    parser.add_argument("--unconverted", type=int, default=4000,
                        help="Unconverted MCQ samples from medical data")
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load raw data ──
    medqa = load_json("data/medqa_usmle_train.json")
    medmcqa = load_json("data/medmcqa_train.json")
    mmlu = load_json("data/mmlu_pro_non_medical.json")

    print(f"Loaded: MedQA-USMLE={len(medqa):,}  MedMCQA={len(medmcqa):,}  MMLU-Pro={len(mmlu):,}")

    # ── 1. Build 40K medical pool ──
    # Use all MedQA-USMLE, sample the rest from MedMCQA
    for item in medqa:
        item["source"] = "medqa_usmle"
    for item in medmcqa:
        item["source"] = "medmcqa"

    need_from_medmcqa = args.medical_total - len(medqa)
    if need_from_medmcqa > len(medmcqa):
        print(f"WARNING: need {need_from_medmcqa} from MedMCQA but only {len(medmcqa)} available")
        need_from_medmcqa = len(medmcqa)

    medmcqa_sampled = random.sample(medmcqa, need_from_medmcqa)
    medical_pool = medqa + medmcqa_sampled
    random.shuffle(medical_pool)
    print(f"\nMedical pool: {len(medical_pool):,} "
          f"(MedQA={len(medqa):,} + MedMCQA sampled={len(medmcqa_sampled):,})")

    # ── 2. Split: 4K unconverted + remaining for conversion ──
    unconverted = medical_pool[:args.unconverted]
    to_convert = medical_pool[args.unconverted:]
    print(f"  Unconverted (raw MCQ): {len(unconverted):,}")
    print(f"  To convert (open-ended): {len(to_convert):,}")

    # ── 3. Split convertible medical data: 20K SFT + 20K RL ──
    half = len(to_convert) // 2
    sft_split = to_convert[:half]
    rl_split = to_convert[half:]

    # ── 4. Sample 5K from MMLU-Pro ──
    if args.mmlu_total > len(mmlu):
        print(f"WARNING: need {args.mmlu_total} from MMLU-Pro but only {len(mmlu)} available")
        mmlu_sampled = mmlu
    else:
        mmlu_sampled = random.sample(mmlu, args.mmlu_total)
    for item in mmlu_sampled:
        item["source"] = "mmlu_pro"

    # ── Save all files ──
    print("\nSaving files...")

    # Files for construct_verifiable_medical_problems.py
    save_json(to_convert, os.path.join(args.output_dir, "medical_40k.json"))
    save_json(mmlu_sampled, os.path.join(args.output_dir, "mmlu_pro_5k.json"))

    # Pre-split files (use after conversion)
    save_json(sft_split, os.path.join(args.output_dir, "medical_sft_20k.json"))
    save_json(rl_split, os.path.join(args.output_dir, "medical_rl_20k.json"))

    # Unconverted MCQ
    save_json(unconverted, os.path.join(args.output_dir, "medical_unconverted_4k.json"))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary — use these files as follows:")
    print("=" * 60)
    print()
    print("Step 1: Convert to open-ended verifiable problems")
    print("  bash run_construct_verifiable_problems.sh  # DATA_PATH=data/medical_40k.json")
    print("  bash run_construct_verifiable_problems.sh  # DATA_PATH=data/mmlu_pro_5k.json")
    print()
    print("Step 2: Training")
    print(f"  SFT stage 1:  data/medical_sft_20k.json  ({len(sft_split):,} items)")
    print(f"  RL  stage 2:  data/medical_rl_20k.json   ({len(rl_split):,} items)")
    print(f"  Unconverted:  data/medical_unconverted_4k.json ({len(unconverted):,} items)")
    print(f"  General:      data/mmlu_pro_5k.json      ({len(mmlu_sampled):,} items)")
    print()
    print("SFT training data = converted(medical_sft_20k) + unconverted_4k + converted(mmlu_pro_5k)")


if __name__ == "__main__":
    main()