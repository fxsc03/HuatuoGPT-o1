from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_path   = "./models/Qwen2.5-7B-sft-merged"
lora_path   = "./grpo_ckpts/dapo-430/medical_dapo_qwen25_430/checkpoint-1000"
output_path = "./models/Qwen2.5-7B-dapo-merged-504"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="cpu")
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_path)
print("Merging...")
model = model.merge_and_unload()
model.save_pretrained(output_path)
AutoTokenizer.from_pretrained(base_path).save_pretrained(output_path)
print(f"Done → {output_path}")
