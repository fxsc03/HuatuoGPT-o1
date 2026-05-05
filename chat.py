"""
终端交互式推理脚本（多轮对话 + 流式输出）

用法:
    python chat.py                                       # 默认加载合并后的 GRPO 模型
    python chat.py --model_path ./models/xxx             # 指定其他模型
    python chat.py --temperature 0.5 --max_new_tokens 1024

会话内置命令（按 Enter 提交后生效）:
    /reset    清空历史，开始新对话
    /show     打印当前历史
    /exit     退出（也可 Ctrl-D / Ctrl-C）


    python chat.py --model_path ./models/Qwen2.5-7B-dapo-merged-504
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


SYSTEM_PROMPT = (
    "You are a medical expert. When answering questions, you must follow this exact format:\n\n"
    "## Thinking\n"
    "[Your detailed reasoning process here]\n\n"
    "## Final Response\n"
    "[Your final answer here]\n\n"
    "Always use exactly these two section headers."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./models/Qwen2.5-7B-grpo-merged-427")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--system", default=SYSTEM_PROMPT, help="系统提示，--system '' 可关闭")
    p.add_argument("--no_stream", action="store_true", help="禁用流式输出")
    return p.parse_args()


def load(model_path):
    print(f"[load] {model_path} in bf16 …", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


def read_user_input():
    """支持多行输入：单独一行 '\\\\' 结尾继续，空行提交。普通单行直接提交。"""
    try:
        line = input("\n>>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return line


@torch.inference_mode()
def generate(tok, model, history, args):
    prompt = tok.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    streamer = None if args.no_stream else TextStreamer(
        tok, skip_prompt=True, skip_special_tokens=True
    )

    print()  # 留一行空白
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    if args.no_stream:
        print(text)
    return text


def main():
    args = parse_args()
    tok, model = load(args.model_path)

    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    print(
        "\n=== Medical Chat ===\n"
        "命令: /reset 清空历史 | /show 显示历史 | /exit 退出"
    )

    while True:
        user = read_user_input()
        if user is None or user == "/exit":
            print("bye.")
            return
        if not user:
            continue
        if user == "/reset":
            history = [{"role": "system", "content": args.system}] if args.system else []
            print("[history cleared]")
            continue
        if user == "/show":
            for m in history:
                print(f"--- {m['role']} ---")
                print(m["content"])
            continue

        history.append({"role": "user", "content": user})
        try:
            reply = generate(tok, model, history, args)
        except KeyboardInterrupt:
            print("\n[generation interrupted]")
            history.pop()  # 撤回失败的 user turn
            continue

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
