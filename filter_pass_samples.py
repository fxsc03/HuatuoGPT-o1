import json

def prepare_cot_dataset(input_filepath, output_filepath):
    # 1. 读取原始 JSON 数据
    with open(input_filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    processed_data = []
    
    # 2. 遍历并过滤数据
    for item in raw_data:
        # 只保留被 GPT 判定为 Pass 的高质量数据
        if item.get("filter_evaluation") == "Pass":
            
            # --- 核心步骤：按照推荐的方式重组 Input ---
            # 拼接 题干 (question) 和 选项 (options_str)
            # 并在中间加一个换行符，保证格式清晰
            model_input = f"{item['question']}\n{item['options_str']}"
            
            # 提取正确的选项字母和文本
            correct_idx = item.get('answer_idx', '')
            correct_text = item.get('answer', '')
            
            # 3. 构建新的数据字典
            new_item = {
                "process_id": item["process_id"],
                "input": model_input,
                "ground_truth_idx": correct_idx,
                "ground_truth_text": correct_text,
                # 预留 output 字段。下一步你将调用大模型，把生成的 CoT 填入这里
                "output": "" 
            }
            
            processed_data.append(new_item)

    # 4. 将提取后的数据保存为新的 JSON 文件
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 数据清洗完成！")
    print(f"📊 原始数据总量: {len(raw_data)} 条")
    print(f"🎯 成功提取 Pass 数据: {len(processed_data)} 条")
    print(f"💾 已保存至: {output_filepath}")

# ==========================================
# 使用方法：
# 将 "raw_data.json" 替换为你当前数据文件的名字
# ==========================================
if __name__ == "__main__":
    input_file = "/home/fxs/LLM1.30/HuatuoGPT-o1/medmcqa_train_final_182822.json"        # 你的原始带 filter_evaluation 的 JSON
    output_file = "cot_ready_data.json" # 提取后准备用来生成 CoT 的新 JSON
    
    prepare_cot_dataset(input_file, output_file)