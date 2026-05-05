import os

# 必须在 import transformers / huggingface_hub 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import random
import json
import threading

# =============================================================================
# 配置区
# =============================================================================
LLM_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# 远程 API 模式的默认值（加 --api_mode 时才使用）
CHAT_COMPLETIONS_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = ""

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from retrying import retry
import argparse
import traceback
import re
import requests


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _strip_think_blocks(text: str) -> str:
    """去掉 DeepSeek-R1 常见的 <think>…</think> 包裹块，便于后续 JSON 解析。"""
    if not text:
        return text
    out = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)
    return out.strip()


def extract_bracket_content(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


# 筛题结论（四类，须与 filter_prompt 一致）：Pass / Too Simple / Ambiguous Answer / Not Reformulatable
_FILTER_EVAL_LINE = re.compile(
    r"Evaluation:\s*\*?\*?\s*(Pass|Too\s+Simple|Ambiguous\s+Answer|Not\s+Reformulatable)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_filter_label(fragment: str) -> str | None:
    t = re.sub(r"\*+", "", fragment).strip().lower()
    if t == "pass":
        return "Pass"
    if t == "too simple":
        return "Too Simple"
    if t == "ambiguous answer":
        return "Ambiguous Answer"
    if t == "not reformulatable":
        return "Not Reformulatable"
    return None


def parse_filter_evaluation(response: str | None) -> str | None:
    """
    从筛题回复中解析四类结论之一。须在 assistant 段内匹配 **Evaluation:** 行，
    避免把提示词里的 bullet「Pass」误判为通过。
    """
    if not response or not str(response).strip():
        return None
    text = str(response)
    if re.search(r"(?i)\bassistant\b", text):
        parts = re.split(r"(?i)\bassistant\b", text)
        text = parts[-1]
    # 去掉 markdown 加粗 **，便于匹配 **Evaluation: Pass**
    text = text.replace("*", "")
    matches = list(_FILTER_EVAL_LINE.finditer(text))
    if matches:
        return _normalize_filter_label(matches[-1].group(1))
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"Evaluation:\s*\*?\*?\s*(.+)$", stripped, re.IGNORECASE)
        if m:
            norm = _normalize_filter_label(m.group(1))
            if norm:
                return norm
    return None


def _is_placeholder_reformat_value(s: str) -> bool:
    if not isinstance(s, str):
        return True
    t = s.strip()
    return t == "" or t == "..." or t == "…"


# ---------------------------------------------------------------------------
# 本地 GPU 推理后端（默认）
# ---------------------------------------------------------------------------

class LocalLLM:
    """直接用 transformers 在本机 GPU 上跑推理，无需外部服务。"""

    def __init__(self, model_name: str, allow_cpu_offload: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        print(f"[LocalLLM] 正在加载模型 {model_name} ...")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到可用 CUDA。通常是 torch CUDA 版本与驱动不匹配导致。"
                "请先安装与驱动兼容的 PyTorch（例如 cu121），确保 torch.cuda.is_available() 为 True。"
            )

        # P100 (sm60) 不支持 bfloat16，优先使用 float16
        major, minor = torch.cuda.get_device_capability(0)
        use_bf16 = (major, minor) >= (8, 0)  # Ampere+ 才可靠支持 bf16
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        
        # =====================================================================
        # [核心修复 1] 强制左侧填充，解决 Decoder-only 模型批量推理生成乱码的问题
        # =====================================================================
        self.tokenizer.padding_side = "left"

        # 避免每次 generate 打印 pad_token 警告，并支持 batch padding
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 尽量把模型放进 GPU（两张 16GB 的 P100 足够容纳 7B FP16 权重；CPU offload 会极慢）
        device_count = torch.cuda.device_count()
        if device_count >= 2:
            max_memory = {i: "15GiB" for i in range(device_count)}
        else:
            max_memory = {0: "15GiB"}
        if not allow_cpu_offload:
            max_memory["cpu"] = "0GiB"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch_dtype,
            device_map="balanced",
            max_memory=max_memory,
        )
        self.model.eval()
        device_map = getattr(self.model, "hf_device_map", {})
        print(f"[LocalLLM] 模型加载完成，device_map: {device_map if device_map else 'N/A'}")
        if not allow_cpu_offload and any(v == "cpu" for v in (device_map or {}).values()):
            raise RuntimeError(
                "检测到模型被 offload 到 CPU（会导致推理极慢）。"
                "请减小模型/使用量化/或在命令中加 --allow_cpu_offload 继续（不推荐）。"
            )

    def call_batch(self, contents: list[str], additional_args: dict | None = None) -> list[str]:
        """批量推理：一次性生成多条输出，显著提升 GPU 吞吐。"""
        import torch

        additional_args = additional_args or {}
        max_new_tokens = int(additional_args.get("max_tokens", 256))

        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": c}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for c in contents
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 每条的“新生成部分”起点不同（因为 padding），需要按 attention_mask 计算
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        res: list[str] = []
        for i, in_len in enumerate(input_lens):
            gen = outputs[i][int(in_len):]
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            res.append(_strip_think_blocks(text))
        return res

    def call(self, content: str, additional_args: dict | None = None):
        return self.call_batch([content], additional_args=additional_args)[0]

    @retry(wait_fixed=1000, stop_max_attempt_number=2)
    def retry_call(self, content: str, additional_args: dict | None = None):
        return self.call(content, additional_args or {"max_tokens": 8192})


# ---------------------------------------------------------------------------
# 远程 API 后端（--api_mode 时使用）
# ---------------------------------------------------------------------------

class RemoteAPI:
    def __init__(self, model_name, api_url, api_key):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        print(f"[RemoteAPI] model={model_name}  url={api_url}")

    def call(self, content, additional_args={}):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            **additional_args,
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            raise ValueError(
                f"API returned non-JSON (status={response.status_code}): {response.text[:800]}"
            )
        if response.status_code != 200:
            raise ValueError(f"API HTTP {response.status_code}: {response_data}")
        if "error" in response_data:
            raise ValueError(f"API Error: {response_data}")
        choices = response_data.get("choices") or []
        if not choices:
            raise ValueError(f"API response has no choices: {response_data}")
        raw = choices[0].get("message") or {}
        text = raw.get("content")
        if text is None:
            raise ValueError(f"API message has no content: {response_data}")
        return _strip_think_blocks(str(text))

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call(self, content, additional_args={"max_tokens": 8192}):
        return self.call(content, additional_args)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="将选择题转化为开放式可验证问题。默认在本机 GPU 推理，加 --api_mode 走远程 API。",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--filter_data", action="store_true",
                        help="启用 LLM 过滤（额外一次调用 / 推理）。")
    parser.add_argument("--model_name", type=str, default=LLM_MODEL_ID,
                        help=f"模型 HF ID（默认 {LLM_MODEL_ID}）。")
    parser.add_argument("--allow_cpu_offload", action="store_true",
                        help="允许将部分模型参数 offload 到 CPU（会非常慢，不建议）。")

    # 远程 API 模式
    parser.add_argument("--api_mode", action="store_true",
                        help="使用远程 OpenAI 兼容 API 而非本机 GPU。")
    parser.add_argument("--api_url", type=str, default=CHAT_COMPLETIONS_URL)
    parser.add_argument("--api_key", type=str, default=None)

    parser.add_argument("--num_process", type=int, default=10,
                        help="并发数。本地 GPU 模式受显存限制，建议设 1~4；远程 API 可更高。")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="本地 GPU 批大小（一次喂给模型的 query 数）。建议 2/4/8 逐步试探；过大可能 OOM。")
    parser.add_argument("--limit_num", type=int, help="限制处理条数（调试用）。")
    parser.add_argument("--max_tokens_filter", type=int, default=128,
                        help="筛题阶段生成上限（本地为 max_new_tokens / 远端为 max_tokens）。")
    parser.add_argument("--max_tokens_rewrite", type=int, default=256,
                        help="改写阶段生成上限（本地为 max_new_tokens / 远端为 max_tokens）。")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 数据处理
# ---------------------------------------------------------------------------

def _validate_reformat_dict(parsed_data: dict) -> dict | None:
    q = parsed_data.get("Open-ended Verifiable Question")
    a = parsed_data.get("Ground-True Answer")
    if not isinstance(q, str) or not isinstance(a, str):
        return None
    if _is_placeholder_reformat_value(q) or _is_placeholder_reformat_value(a):
        return None
    return parsed_data


def _try_parse_reformat_blob(blob: str) -> dict | None:
    """从一段文本中解析改写结果 dict（从首个 '{' 起 raw_decode）；失败返回 None。"""
    s = blob.strip()
    if "{" not in s:
        return None
    decoder = json.JSONDecoder()
    start = s.find("{")
    frag = s[start:].replace("\n", " ").strip()
    try:
        parsed_data, _ = decoder.raw_decode(frag)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed_data, dict):
        return None
    return _validate_reformat_dict(parsed_data)


def _iter_reformat_json_candidates(text: str):
    """
    从模型输出中收集可能的 JSON 对象。优先使用最后一个 ```json``` 代码块，
    避免误把 prompt 里的示例占位符 {"...": "..."} 当成答案。
    """
    if not text:
        return
    # 1) fenced 代码块全文（避免非贪婪 \{ \} 截断字符串内的花括号）
    for m in reversed(list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE))):
        got = _try_parse_reformat_blob(m.group(1))
        if got is not None:
            yield got
    # 2) 全文中的 JSON 对象（自后向前尝试每个 '{' 起点）
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(starts):
        decoder = json.JSONDecoder()
        frag = text[start:].replace("\n", " ").strip()
        try:
            obj, _ = decoder.raw_decode(frag)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            got = _validate_reformat_dict(obj)
            if got is not None:
                yield got


def parse_gpt_response(response):
    try:
        if response is None:
            return False, None

        text = response.strip()
        text = text.replace("```json", "```").replace("```JSON", "```")

        for parsed_data in _iter_reformat_json_candidates(text):
            if "Open-ended Verifiable Question" not in parsed_data or "Ground-True Answer" not in parsed_data:
                continue
            v = _validate_reformat_dict(parsed_data)
            if v is not None:
                return True, v
        return False, None
    except Exception as e:
        print(f"Error parsing response: {e}")
        return False, None


def process_single_item(item, llm, save_directory, filter_prompt, reformat_prompt, filter_enabled):
    try:
        max_retries = 2
        save_path = os.path.join(save_directory, f"{item['process_id']}.json")

        item["options_str"] = "\n".join(
            [f"{k}. {v}" for k, v in item["options"].items()]
        )
        question_text = f"{item['question']}\n{item['options_str']}"

        if filter_enabled:
            filter_query = filter_prompt.format(question_text, item["answer"])
            item["gpt_filter_query"] = filter_query
            response = llm.retry_call(filter_query, {"max_tokens": item.get("_max_tokens_filter", 128)})
            item["gpt_filter_response"] = response
            feval = parse_filter_evaluation(response)
            item["filter_evaluation"] = feval
            # 仅当筛题结论明确为非 Pass 时跳过改写；解析失败(None)时仍尝试改写以免漏题
            if feval is not None and feval != "Pass":
                item["Open-ended Verifiable Question"] = ""
                item["Ground-True Answer"] = ""
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
                return 1
        else:
            item["filter_evaluation"] = None

        reformat_query = reformat_prompt.format(question_text, item["answer"])
        item["gpt_reformat_query"] = reformat_query

        rewrite_ok = False
        for _ in range(max_retries):
            response = llm.retry_call(reformat_query, {"max_tokens": item.get("_max_tokens_rewrite", 256)})
            item["gpt_reformat_response"] = response
            valid, parsed_data = parse_gpt_response(response)
            if valid:
                item["Open-ended Verifiable Question"] = parsed_data["Open-ended Verifiable Question"]
                item["Ground-True Answer"] = parsed_data["Ground-True Answer"]
                rewrite_ok = True
                break
        if not rewrite_ok:
            item["Open-ended Verifiable Question"] = ""
            item["Ground-True Answer"] = ""

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error processing item {item['process_id']}: {e}")
    return 1


def merge_saved_files(directory):
    _, _, filenames = next(os.walk(directory))
    json_files = [f for f in filenames if f.endswith(".json")]
    merged = []
    for fn in json_files:
        try:
            with open(os.path.join(directory, fn), "r", encoding="utf-8") as f:
                data = json.load(f)
                assert (
                    "Open-ended Verifiable Question" in data
                    or "gpt_filter_response" in data
                    or "gpt4_response_filter" in data
                )
                merged.append(data)
        except Exception as e:
            print(f"Error merging {fn}: {e}")
    return merged


def deduplicate_data(data, processed_data):
    done = {item["process_id"] for item in processed_data}
    return [item for item in data if item["process_id"] not in done]


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    # ---- 选择后端 ----
    if args.api_mode:
        api_key = (
            args.api_key
            or (API_KEY.strip() if API_KEY else "")
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("SILICONFLOW_API_KEY", "")
        )
        if not api_key:
            raise SystemExit(
                "远程 API 模式需要密钥：--api_key 或 export OPENAI_API_KEY / SILICONFLOW_API_KEY"
            )
        llm = RemoteAPI(model_name=args.model_name, api_url=args.api_url, api_key=api_key)
    else:
        llm = LocalLLM(model_name=args.model_name, allow_cpu_offload=args.allow_cpu_offload)

    # ---- 加载数据 ----
    with open(args.data_path, "r") as f:
        input_data = json.load(f)
    for idx, item in enumerate(input_data, start=1):
        item["process_id"] = idx
        # 传递生成长度配置给每条样本（避免改动多处函数签名）
        item["_max_tokens_filter"] = args.max_tokens_filter
        item["_max_tokens_rewrite"] = args.max_tokens_rewrite
    if args.limit_num:
        input_data = input_data[: args.limit_num]
    print(f"Loaded {len(input_data)} items.")

    task_name = os.path.splitext(os.path.basename(args.data_path))[0]
    save_directory = os.path.join("output_data", task_name)
    os.makedirs(save_directory, exist_ok=True)

    filter_prompt = """<Multiple-choice Question>
{}
Correct Answer: {}
</Multiple-choice Question>

You are an expert in filtering and evaluating multiple-choice questions for advanced reasoning tasks. Your job is to evaluate a given question and determine whether it meets the following criteria: 
1. **Depth of Reasoning:** The question should require deeper reasoning. If the question appears too simple, mark it as "Too Simple".
2. **Unambiguous Correct Answer:** The question must have a unique and unambiguous correct answer. If the question asks for "incorrect options" or allows for multiple correct answers, mark it as "Ambiguous Answer".
3. **Open-Ended Reformulation Feasibility:** The question should be suitable for reformatting into an open-ended format. If the question cannot be easily reformulated into an open-ended problem and a clear ground-truth answer, mark it as "Not Reformulatable".

You must choose exactly one conclusion. The first line of your reply MUST be exactly one of:
Evaluation: Pass
Evaluation: Too Simple
Evaluation: Ambiguous Answer
Evaluation: Not Reformulatable

Then you may briefly explain. Do not repeat the instruction bullet list."""

    reformat_prompt = """I will provide you with a multiple-choice question, and your task is to rewrite it into an open-ended question, along with a Ground-True Answer. The requirements are:

1. The question must be specific, targeting the point being tested in the original multiple-choice question. Ensure it is open-ended, meaning no options are provided.
2. CRITICAL RULE FOR QUESTION: Do NOT incorporate the answer into the question. Do NOT create a Yes/No or True/False question. The question must require the solver to generate the answer from scratch.
3. CRITICAL RULE FOR ANSWER: The Ground-True Answer must be exactly the text of the correct option, stripped of any letters (like A, B, C) and explanations. Keep it as short as possible for exact string matching. Do NOT explain why.

Here is the multiple-choice question for you to rewrite:
<Multiple-choice Question>
{}
Correct Answer: {}
</Multiple-choice Question>

Please output ONLY one JSON object in a ```json code block. Fill the two strings with your real rewritten question and real short answer — do NOT echo the prompt template, do NOT use the literal three dots (...) or placeholder text as the values.

```json
{{
"Open-ended Verifiable Question": "<your open-ended question here>",
"Ground-True Answer": "<exact correct option text here>"
}}
```"""

    # ---- 断点续跑 ----
    processed_data = merge_saved_files(save_directory)
    print(f"Previously processed items: {len(processed_data)}")
    input_data = deduplicate_data(input_data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    # ---- 处理 ----
    if (not args.api_mode) and args.batch_size > 1:
        # 本地 GPU：用 batch 提升吞吐（比多线程更有效）
        bs = int(args.batch_size)

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        for batch in tqdm(list(chunks(input_data, bs)), desc="Processing Batches", unit="batch"):
            # 1) filter（可选）
            passed: list[dict] = []
            if args.filter_data:
                filter_queries = []
                for it in batch:
                    it["options_str"] = "\n".join([f"{k}. {v}" for k, v in it["options"].items()])
                    question_text = f"{it['question']}\n{it['options_str']}"
                    q = filter_prompt.format(question_text, it["answer"])
                    it["gpt_filter_query"] = q
                    filter_queries.append(q)

                responses = llm.call_batch(filter_queries, {"max_tokens": args.max_tokens_filter})
                for it, resp in zip(batch, responses):
                    it["gpt_filter_response"] = resp
                    feval = parse_filter_evaluation(resp)
                    it["filter_evaluation"] = feval
                    if feval is None or feval == "Pass":
                        passed.append(it)
                    else:
                        it["Open-ended Verifiable Question"] = ""
                        it["Ground-True Answer"] = ""
                        save_path = os.path.join(save_directory, f"{it['process_id']}.json")
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(it, f, ensure_ascii=False, indent=2)
            else:
                # 不筛题：全部进入改写
                for it in batch:
                    it["options_str"] = "\n".join([f"{k}. {v}" for k, v in it["options"].items()])
                    it["filter_evaluation"] = None
                    passed.append(it)

            # 2) rewrite（对 passed 做 batch）
            if not passed:
                continue

            rewrite_queries = []
            for it in passed:
                question_text = f"{it['question']}\n{it['options_str']}"
                rq = reformat_prompt.format(question_text, it["answer"])
                it["gpt_reformat_query"] = rq
                rewrite_queries.append(rq)

            rewrite_responses = llm.call_batch(rewrite_queries, {"max_tokens": args.max_tokens_rewrite})
            for it, resp in zip(passed, rewrite_responses):
                it["gpt_reformat_response"] = resp
                valid, parsed = parse_gpt_response(resp)
                if valid:
                    it["Open-ended Verifiable Question"] = parsed["Open-ended Verifiable Question"]
                    it["Ground-True Answer"] = parsed["Ground-True Answer"]
                else:
                    it["Open-ended Verifiable Question"] = ""
                    it["Ground-True Answer"] = ""
                save_path = os.path.join(save_directory, f"{it['process_id']}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(it, f, ensure_ascii=False, indent=2)
    else:
        # 远端 API 或 batch_size=1：走原线程池逻辑
        with ThreadPoolExecutor(max_workers=args.num_process) as executor:
            list(tqdm(
                executor.map(
                    lambda item: process_single_item(
                        item, llm, save_directory,
                        filter_prompt, reformat_prompt, args.filter_data,
                    ),
                    input_data,
                ),
                total=len(input_data),
                desc="Processing Items",
                unit="item",
            ))

    # ---- 合并输出 ----
    final_data = merge_saved_files(save_directory)
    output_path = f"{task_name}_final_{len(final_data)}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()