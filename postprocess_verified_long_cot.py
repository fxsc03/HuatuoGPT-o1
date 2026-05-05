import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from cot_pipeline_accelerated import (
    BatchChatRequest,
    DEFAULT_BATCH_COMPLETION_WINDOW,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_HTTP_POOL_SIZE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SILICONFLOW_API_URL,
    DEFAULT_SILICONFLOW_BASE_URL,
    JSON_RESPONSE_FORMAT,
    POSTPROCESS_REFORMAT_PROMPT,
    POSTPROCESS_RESPONSE_PROMPT,
    RemoteChatClient,
    SiliconFlowBatchClient,
    append_usage,
    build_batch_cache_key,
    compact_json,
    ensure_dir,
    extract_json_object,
    get_stream_of_search,
    resolve_api_key,
)


DEFAULT_COMPLEX_MAX_TOKENS = 900
DEFAULT_RESPONSE_MAX_TOKENS = 320
DEFAULT_MAX_WORKERS = 16


def get_question(item: Dict[str, Any]) -> str:
    return item.get("Question") or item.get("Open-ended Verifiable Question") or ""


def build_complex_messages(question: str, long_cot: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": POSTPROCESS_REFORMAT_PROMPT.format(
                get_stream_of_search(long_cot),
                question,
            ),
        }
    ]


def build_response_messages(question: str, complex_cot: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": POSTPROCESS_RESPONSE_PROMPT.format(
                complex_cot,
                question,
            ),
        }
    ]


def parse_complex_response(text: str) -> str:
    payload = extract_json_object(text)
    reasoning = payload.get("NaturalReasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Missing NaturalReasoning in complex CoT response")
    return reasoning.strip()


def load_items(input_path: str, limit_num: int = 0) -> List[Dict[str, Any]]:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input file must be a JSON list.")
    for idx, item in enumerate(data, start=1):
        item.setdefault("process_id", idx)
    if limit_num > 0:
        data = data[:limit_num]
    return data


def save_items(path: str, items: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def run_online_complex_stage(items: List[Dict[str, Any]], client: RemoteChatClient, args: argparse.Namespace) -> None:
    def worker(item: Dict[str, Any]) -> None:
        question = get_question(item)
        messages = build_complex_messages(question, item["Long_CoT"])
        result = client.retry_call_messages(
            messages,
            max_tokens=args.complex_max_tokens,
            temperature=args.temperature,
            response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
        )
        item.setdefault("postprocess_trace", {}).setdefault("complex", []).append(
            {
                "messages": messages,
                "response": result.content,
            }
        )
        append_usage(
            item,
            "postprocess_complex",
            result.usage,
            fallback_query=compact_json(messages),
            fallback_response=result.content,
        )
        item["Complex_CoT"] = parse_complex_response(result.content)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            future.result()


def run_online_response_stage(items: List[Dict[str, Any]], client: RemoteChatClient, args: argparse.Namespace) -> None:
    def worker(item: Dict[str, Any]) -> None:
        question = get_question(item)
        messages = build_response_messages(question, item["Complex_CoT"])
        result = client.retry_call_messages(
            messages,
            max_tokens=args.response_max_tokens,
            temperature=args.temperature,
        )
        item.setdefault("postprocess_trace", {}).setdefault("response", []).append(
            {
                "messages": messages,
                "response": result.content,
            }
        )
        append_usage(
            item,
            "postprocess_response",
            result.usage,
            fallback_query=compact_json(messages),
            fallback_response=result.content,
        )
        item["Response"] = result.content.strip()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            future.result()


def run_batch_complex_stage(
    items: List[Dict[str, Any]],
    client: SiliconFlowBatchClient,
    model_name: str,
    args: argparse.Namespace,
    cache_dir: str,
) -> None:
    requests_list: List[BatchChatRequest] = []
    messages_map: Dict[str, List[Dict[str, str]]] = {}
    for item in items:
        custom_id = f"complex-{item['process_id']}"
        messages = build_complex_messages(get_question(item), item["Long_CoT"])
        messages_map[custom_id] = messages
        requests_list.append(
            BatchChatRequest(
                custom_id=custom_id,
                messages=messages,
                max_tokens=args.complex_max_tokens,
                temperature=args.temperature,
                response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
            )
        )
    stage_metadata = {"stage": "complex_cot_postprocess"}
    cache_key = build_batch_cache_key("complex_cot_postprocess", model_name, requests_list, stage_metadata)
    results = client.run_chat_batch(
        requests_list,
        model_name=model_name,
        completion_window=args.batch_completion_window,
        poll_interval=args.batch_poll_interval,
        metadata=stage_metadata,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    for item in items:
        custom_id = f"complex-{item['process_id']}"
        result = results[custom_id]
        item.setdefault("postprocess_trace", {}).setdefault("complex", []).append(
            {
                "messages": messages_map[custom_id],
                "response": result.content,
                "backend": "batch",
            }
        )
        append_usage(
            item,
            "postprocess_complex",
            result.usage,
            fallback_query=compact_json(messages_map[custom_id]),
            fallback_response=result.content,
        )
        item["Complex_CoT"] = parse_complex_response(result.content)


def run_batch_response_stage(
    items: List[Dict[str, Any]],
    client: SiliconFlowBatchClient,
    model_name: str,
    args: argparse.Namespace,
    cache_dir: str,
) -> None:
    requests_list: List[BatchChatRequest] = []
    messages_map: Dict[str, List[Dict[str, str]]] = {}
    for item in items:
        custom_id = f"response-{item['process_id']}"
        messages = build_response_messages(get_question(item), item["Complex_CoT"])
        messages_map[custom_id] = messages
        requests_list.append(
            BatchChatRequest(
                custom_id=custom_id,
                messages=messages,
                max_tokens=args.response_max_tokens,
                temperature=args.temperature,
                response_format=None,
            )
        )
    stage_metadata = {"stage": "final_response_postprocess"}
    cache_key = build_batch_cache_key("final_response_postprocess", model_name, requests_list, stage_metadata)
    results = client.run_chat_batch(
        requests_list,
        model_name=model_name,
        completion_window=args.batch_completion_window,
        poll_interval=args.batch_poll_interval,
        metadata=stage_metadata,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    for item in items:
        custom_id = f"response-{item['process_id']}"
        result = results[custom_id]
        item.setdefault("postprocess_trace", {}).setdefault("response", []).append(
            {
                "messages": messages_map[custom_id],
                "response": result.content,
                "backend": "batch",
            }
        )
        append_usage(
            item,
            "postprocess_response",
            result.usage,
            fallback_query=compact_json(messages_map[custom_id]),
            fallback_response=result.content,
        )
        item["Response"] = result.content.strip()


def filter_stage_items(items: List[Dict[str, Any]], stage: str, force: bool) -> List[Dict[str, Any]]:
    if stage == "complex":
        return [item for item in items if item.get("Long_CoT") and (force or not item.get("Complex_CoT"))]
    if stage == "response":
        return [item for item in items if item.get("Complex_CoT") and (force or not item.get("Response"))]
    raise ValueError(f"Unsupported stage: {stage}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_url", type=str, default=DEFAULT_SILICONFLOW_API_URL)
    parser.add_argument("--base_url", type=str, default=DEFAULT_SILICONFLOW_BASE_URL)
    parser.add_argument("--generator_model_name", type=str, default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--response_model_name", type=str, default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--backend", choices=["online", "batch"], default="online")
    parser.add_argument("--stage", choices=["all", "complex", "response"], default="all")
    parser.add_argument("--limit_num", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--complex_max_tokens", type=int, default=DEFAULT_COMPLEX_MAX_TOKENS)
    parser.add_argument("--response_max_tokens", type=int, default=DEFAULT_RESPONSE_MAX_TOKENS)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--request_timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--connect_timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT)
    parser.add_argument("--http_pool_size", type=int, default=DEFAULT_HTTP_POOL_SIZE)
    parser.add_argument("--batch_completion_window", type=str, default=DEFAULT_BATCH_COMPLETION_WINDOW)
    parser.add_argument("--batch_poll_interval", type=int, default=60)
    parser.add_argument("--use_json_mode", action="store_true", default=True)
    parser.add_argument("--disable_json_mode", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.disable_json_mode:
        args.use_json_mode = False

    api_key = resolve_api_key(args.api_key)
    items = load_items(args.input_path, limit_num=args.limit_num)
    output_path = args.output_path or os.path.splitext(args.input_path)[0] + "_postprocessed.json"
    batch_cache_dir = os.path.splitext(output_path)[0] + "_batch_cache"

    online_client = RemoteChatClient(
        model_name=args.generator_model_name,
        api_url=args.api_url,
        api_key=api_key,
        http_pool_size=args.http_pool_size,
        request_timeout=args.request_timeout,
        connect_timeout=args.connect_timeout,
    )
    batch_client = None
    if args.backend == "batch":
        batch_client = SiliconFlowBatchClient(
            api_key=api_key,
            base_url=args.base_url,
            request_timeout=args.request_timeout,
            connect_timeout=args.connect_timeout,
        )

    if args.stage in {"all", "complex"}:
        complex_items = filter_stage_items(items, "complex", args.force)
        if complex_items:
            if args.backend == "batch":
                assert batch_client is not None
                run_batch_complex_stage(complex_items, batch_client, args.generator_model_name, args, batch_cache_dir)
            else:
                run_online_complex_stage(complex_items, online_client, args)
            save_items(output_path, items)

    if args.stage in {"all", "response"}:
        response_items = filter_stage_items(items, "response", args.force)
        if response_items:
            if args.backend == "batch":
                assert batch_client is not None
                run_batch_response_stage(response_items, batch_client, args.response_model_name, args, batch_cache_dir)
            else:
                if args.response_model_name != args.generator_model_name:
                    online_client = RemoteChatClient(
                        model_name=args.response_model_name,
                        api_url=args.api_url,
                        api_key=api_key,
                        http_pool_size=args.http_pool_size,
                        request_timeout=args.request_timeout,
                        connect_timeout=args.connect_timeout,
                    )
                run_online_response_stage(response_items, online_client, args)
            save_items(output_path, items)

    save_items(output_path, items)
    print(f"Saved postprocessed data to {output_path}")


if __name__ == "__main__":
    main()
