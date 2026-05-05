"""
Download and format datasets for construct_verifiable_medical_problems.py

Datasets (per HuatuoGPT-o1 paper Section 4.1):
  - MedQA-USMLE training set
  - MedMCQA training set
  - MMLU-Pro (non-medical tracks)

Output format: JSON list of dicts with keys:
  question, options (dict), answer_idx, answer
"""

import json
import os
import random
import string
from datasets import load_dataset

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

MEDICAL_CATEGORIES = {
    "health", "biology", "clinical_knowledge", "medical_genetics",
    "anatomy", "professional_medicine", "college_biology", "college_medicine",
    "nutrition",
}


def download_medqa_usmle():
    print("=" * 60)
    print("Downloading MedQA-USMLE (4-options) training set...")
    print("=" * 60)
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    print(f"  Raw size: {len(ds)}")

    formatted = []
    for item in ds:
        options = item["options"]
        if isinstance(options, str):
            options = json.loads(options)

        answer_text = item["answer"]
        answer_idx = item.get("answer_idx", "")
        if not answer_idx:
            for k, v in options.items():
                if v == answer_text:
                    answer_idx = k
                    break

        formatted.append({
            "question": item["question"],
            "options": options,
            "answer_idx": answer_idx,
            "answer": answer_text,
        })

    out_path = os.path.join(DATA_DIR, "medqa_usmle_train.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(formatted)} items -> {out_path}")
    return formatted


def download_medmcqa():
    print("=" * 60)
    print("Downloading MedMCQA training set...")
    print("=" * 60)
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    print(f"  Raw size: {len(ds)}")

    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    formatted = []
    for item in ds:
        options = {
            "A": item["opa"],
            "B": item["opb"],
            "C": item["opc"],
            "D": item["opd"],
        }
        cop = item["cop"]
        answer_idx = idx_to_letter.get(cop, "A")
        answer_text = options.get(answer_idx, "")

        formatted.append({
            "question": item["question"],
            "options": options,
            "answer_idx": answer_idx,
            "answer": answer_text,
        })

    out_path = os.path.join(DATA_DIR, "medmcqa_train.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(formatted)} items -> {out_path}")
    return formatted


def download_mmlu_pro():
    print("=" * 60)
    print("Downloading MMLU-Pro (non-medical tracks)...")
    print("=" * 60)
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    print(f"  Raw size: {len(ds)}")

    letters = list(string.ascii_uppercase)
    formatted = []
    skipped_categories = set()
    for item in ds:
        category = item.get("category", "").lower().replace(" ", "_")
        if category in MEDICAL_CATEGORIES:
            skipped_categories.add(category)
            continue

        opts_list = item["options"]
        options = {}
        for i, opt in enumerate(opts_list):
            options[letters[i]] = opt

        answer_letter = item.get("answer", "")
        answer_idx_num = item.get("answer_index")
        if answer_idx_num is not None and not answer_letter:
            answer_letter = letters[answer_idx_num]

        answer_text = options.get(answer_letter, "")

        formatted.append({
            "question": item["question"],
            "options": options,
            "answer_idx": answer_letter,
            "answer": answer_text,
            "category": item.get("category", ""),
        })

    print(f"  Skipped medical categories: {skipped_categories}")
    out_path = os.path.join(DATA_DIR, "mmlu_pro_non_medical.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(formatted)} items -> {out_path}")
    return formatted


if __name__ == "__main__":
    random.seed(42)

    medqa = download_medqa_usmle()
    medmcqa = download_medmcqa()
    mmlu_pro = download_mmlu_pro()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  MedQA-USMLE train : {len(medqa)} items")
    print(f"  MedMCQA train     : {len(medmcqa)} items")
    print(f"  MMLU-Pro (general): {len(mmlu_pro)} items")
    print(f"\nAll files saved to {DATA_DIR}/")
