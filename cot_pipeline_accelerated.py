import argparse
import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from retrying import retry
from tqdm import tqdm
from urllib3.util import Retry


DEFAULT_SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_GENERATOR_MODEL = "Qwen/QwQ-32B"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_JUDGE_MODEL_PATH = "models/Qwen2.5-7B-Instruct"
DEFAULT_NUM_PROCESS = 16
DEFAULT_HTTP_POOL_SIZE = 64
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_GENERATOR_MAX_TOKENS = 900
DEFAULT_JUDGE_MAX_TOKENS = 180
DEFAULT_SUMMARY_MAX_TOKENS = 240
DEFAULT_TEMPLATE_GATE_MAX_TOKENS = 180
DEFAULT_MAX_CHAIN_ATTEMPTS = 3
DEFAULT_MAX_CHUNK_RETRIES = 3
DEFAULT_MAX_CHUNK_JUDGE_ATTEMPTS = 3
DEFAULT_COMPRESSION_TRIGGER = 2
DEFAULT_LOCAL_JUDGE_GPU_GIB = 12
DEFAULT_LOCAL_JUDGE_CPU_GIB = 96
DEFAULT_LOCAL_JUDGE_BATCH_SIZE = 8
DEFAULT_BATCH_COMPLETION_WINDOW = "24h"
DEFAULT_BATCH_POLL_INTERVAL = 60

JSON_RESPONSE_FORMAT = {"type": "json_object"}

CHUNK_PLAN = [
    {
        "chunk_id": 1,
        "chunk_name": "Symptom Extraction & Pathology Mapping",
        "goal": "Extract the medically relevant findings from the question and map them to plausible pathology mechanisms.",
        "requirements": [
            "List only symptoms, signs, labs, history, and context explicitly supported by the question.",
            "Map each key finding to a plausible medical mechanism without committing to the final answer.",
            "Do not invent tests, demographics, exam findings, or diagnoses not grounded in the question.",
        ],
    },
    {
        "chunk_id": 2,
        "chunk_name": "Differential Diagnosis & Elimination",
        "goal": "Construct a focused differential and eliminate weaker candidates using the validated findings.",
        "requirements": [
            "Compare plausible diagnoses against the confirmed findings from previous chunks.",
            "State why alternative diagnoses are weaker or inconsistent.",
            "Do not provide the final diagnosis as a settled conclusion yet.",
        ],
    },
    {
        "chunk_id": 3,
        "chunk_name": "Final Conclusion",
        "goal": "Produce the final conclusion strictly grounded in the validated history and differential analysis.",
        "requirements": [
            "State the final diagnosis or conclusion clearly.",
            "Tie the conclusion back to the strongest validated findings.",
            "Do not introduce any new facts that were not present in the original question or validated chunks.",
        ],
    },
]

GENERATOR_SYSTEM_PROMPT = """You are generating one medical reasoning chunk for a verified CoT dataset.

Rules:
1. Generate only the requested chunk.
2. Use only the question, validated history, and compressed lessons provided.
3. Do not use hidden labels.
4. Return valid JSON only.
5. Keep reasoning concise but medically faithful."""

GENERATOR_USER_TEMPLATE = """QUESTION:
{question}

CONFIRMED_CHUNKS:
{confirmed_chunks}

CURRENT_CHUNK:
id={chunk_id}
name={chunk_name}
goal={chunk_goal}
requirements={chunk_requirements}

FAILED_CHAIN_HINT:
{failed_reasoning}

LESSONS:
{lesson_summary}

Return one JSON object with this schema:
{schema}
"""

CHUNK_JUDGE_SYSTEM_PROMPT = """You are a process-supervision judge for medical reasoning.

Judge only whether the candidate chunk is acceptable for the current chunk.
Return JSON only."""

CHUNK_JUDGE_TEMPLATE = """QUESTION:
{question}

CURRENT_CHUNK:
id={chunk_id}
name={chunk_name}
goal={chunk_goal}
requirements={chunk_requirements}

RULES:
- Judge the current chunk only.
- For chunk 1 and chunk 2, do not require the final answer.
- If no reference answer is provided, judge only logic, medical plausibility, and chunk-scope compliance.
- Reject hallucinations, contradictions, unsupported jumps, or future-chunk leakage.
- When uncertain, reject.

REFERENCE_ANSWER:
{reference_answer}

CONFIRMED_CHUNKS:
{confirmed_chunks}

CANDIDATE_CHUNK:
{candidate_chunk}

Return:
{{
  "pass": true/false,
  "error_type": "none|fact|logic|scope|format",
  "critique": "...",
  "fix_hint": "..."
}}
"""

FINAL_VERIFY_SYSTEM_PROMPT = """You are verifying a completed medical reasoning chain.

Judge whether the final answer and the chain are fully consistent with the question and the reference answer.
Return JSON only."""

FINAL_VERIFY_TEMPLATE = """QUESTION:
{question}

REFERENCE_ANSWER:
{reference_answer}

COMPLETED_CHAIN:
{completed_reasoning_chain}

Return:
{{
  "pass": true/false,
  "error_type": "none|fact|logic|answer|scope|format",
  "critique": "...",
  "fix_hint": "..."
}}
"""

SUMMARY_SYSTEM_PROMPT = """You compress failed medical reasoning chains into concise lessons.
Return JSON only."""

SUMMARY_TEMPLATE = """QUESTION:
{question}

FAILED_REASONING_CHAINS:
{failed_reasoning_chains}

Return:
{{
  "confirmed_clues": ["..."],
  "avoid_errors": ["..."],
  "next_chunk_focus": "..."
}}
"""

TEMPLATE_GATE_SYSTEM_PROMPT = """You are screening whether a question fits a three-stage medical reasoning template.

Return JSON only."""

TEMPLATE_GATE_TEMPLATE = """QUESTION:
{question}

TARGET_TEMPLATE:
- Chunk 1: Symptom Extraction & Pathology Mapping
- Chunk 2: Differential Diagnosis & Elimination
- Chunk 3: Final Conclusion

SUITABLE IF:
- The question is mainly a clinical case-style reasoning task.
- The question contains enough patient findings, symptoms, history, labs, imaging, or exam context to support the three chunks.
- A meaningful differential or stepwise clinical conclusion is possible.

NOT SUITABLE IF:
- It is mainly direct fact recall, definition, or term matching.
- It is mainly a pure calculation, formula application, or statistics question.
- It is mainly public health policy, program administration, or guideline trivia.
- It is mainly an isolated anatomy/pathology/microbiology fact without real case reasoning.

BORDERLINE RULE:
- If it is a patient vignette asking for likely diagnosis, pathogen, mechanism, best test, initial management, or treatment, mark it suitable.
- If there is no real patient case context and no meaningful differential reasoning path, mark it unsuitable.

Return:
{{
  "fit_template": true/false,
  "skip_recommended": true/false,
  "question_type": "clinical_case|fact_recall|calculation|policy|other",
  "reason": "short justification"
}}
"""

POSTPROCESS_REFORMAT_PROMPT = """<Thought Process>
{thought_process}
</Thought Process>

<Question>
{question}
</Question>

Rewrite the thought process above into a richer but faithful natural chain of thought.
Return JSON only:
{{
  "NaturalReasoning": "..."
}}
"""

POSTPROCESS_RESPONSE_PROMPT = """<Internal Thinking>
{complex_cot}
</Internal Thinking>

<Question>
{question}
</Question>

Generate the final user-facing response based on the internal thinking above. Output only the final response."""


class HardRejectError(Exception):
    pass


@dataclass
class GenerationResult:
    content: str
    usage: Dict[str, Any]
    raw: Dict[str, Any]


@dataclass
class BatchChatRequest:
    custom_id: str
    messages: List[Dict[str, str]]
    max_tokens: int
    temperature: Optional[float]
    response_format: Optional[Dict[str, Any]]


@dataclass
class PendingChunkJudge:
    item: Dict[str, Any]
    chunk_spec: Dict[str, Any]
    attempt_record: Dict[str, Any]
    candidate_chunk: Dict[str, Any]


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_cache_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._")
    return sanitized or "batch"


def build_batch_cache_key(
    stage_name: str,
    model_name: str,
    requests_list: List["BatchChatRequest"],
    metadata: Dict[str, str],
) -> str:
    canonical_requests = []
    for req in sorted(requests_list, key=lambda item: item.custom_id):
        canonical_requests.append(
            {
                "custom_id": req.custom_id,
                "messages": req.messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "response_format": req.response_format,
            }
        )
    payload = {
        "stage_name": stage_name,
        "model_name": model_name,
        "metadata": metadata,
        "requests": canonical_requests,
    }
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()[:24]
    return f"{sanitize_cache_name(stage_name)}_{digest}"


def ensure_transformers_cache_dir() -> str:
    cache_dir = os.environ.get("TRANSFORMERS_CACHE")
    if cache_dir:
        ensure_dir(cache_dir)
        return cache_dir
    cache_dir = "/tmp/hf-cache"
    ensure_dir(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ.setdefault("HF_HOME", cache_dir)
    return cache_dir


def model_likely_supports_json_mode(model_name: str) -> bool:
    return "qwq" not in model_name.lower()


def resolve_api_key(cli_api_key: Optional[str]) -> str:
    if cli_api_key:
        return cli_api_key
    env_api_key = os.environ.get("SILICONFLOW_API_KEY")
    if env_api_key:
        return env_api_key
    raise ValueError("Missing API key. Pass --api_key or set SILICONFLOW_API_KEY.")


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")
    if text[0] != "{":
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found: {text[:200]}")
        text = match.group(0)
    return json.loads(text)


def parse_chunk_response(response_text: str, chunk_spec: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    try:
        chunk = extract_json_object(response_text)
        assert chunk["chunk_id"] == chunk_spec["chunk_id"], "chunk_id mismatch"
        assert chunk["chunk_name"] == chunk_spec["chunk_name"], "chunk_name mismatch"
        assert chunk["status"] == "chunk_complete", "status must be chunk_complete"
        assert isinstance(chunk["content"], list) and chunk["content"], "content must be non-empty"
        assert isinstance(chunk["chunk_summary"], str) and chunk["chunk_summary"].strip(), "chunk_summary missing"
        for item in chunk["content"]:
            assert isinstance(item, dict), "content item must be dict"
            for field in ["title", "claim", "evidence", "medical_rationale"]:
                assert isinstance(item.get(field), str) and item[field].strip(), f"missing {field}"
        if chunk_spec["chunk_id"] == 3:
            assert isinstance(chunk.get("final_answer"), str) and chunk["final_answer"].strip(), "final_answer missing"
        return True, chunk, ""
    except Exception as exc:
        return False, None, str(exc)


def parse_judge_response(response_text: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        result = extract_json_object(response_text)
        assert isinstance(result["pass"], bool)
        assert isinstance(result["error_type"], str)
        assert isinstance(result["critique"], str)
        assert isinstance(result["fix_hint"], str)
        return True, result
    except Exception as exc:
        return False, {
            "pass": False,
            "error_type": "format",
            "critique": f"Judge parse failed: {exc}",
            "fix_hint": "Return valid JSON only.",
        }


def parse_summary_response(response_text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        summary = extract_json_object(response_text)
        assert isinstance(summary["confirmed_clues"], list)
        assert isinstance(summary["avoid_errors"], list)
        assert isinstance(summary["next_chunk_focus"], str)
        return True, summary
    except Exception:
        return False, None


def parse_template_gate_response(response_text: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        payload = extract_json_object(response_text)
        assert isinstance(payload["fit_template"], bool)
        assert isinstance(payload["skip_recommended"], bool)
        assert isinstance(payload["question_type"], str)
        assert isinstance(payload["reason"], str)
        return True, payload
    except Exception as exc:
        return False, {
            "fit_template": True,
            "skip_recommended": False,
            "question_type": "unknown",
            "reason": f"Template gate parse failed open: {exc}",
        }


def get_chunk_schema(chunk_spec: Dict[str, Any]) -> str:
    schema = {
        "chunk_id": chunk_spec["chunk_id"],
        "chunk_name": chunk_spec["chunk_name"],
        "status": "chunk_complete",
        "content": [
            {
                "title": "...",
                "claim": "...",
                "evidence": "...",
                "medical_rationale": "...",
            }
        ],
        "chunk_summary": "...",
    }
    if chunk_spec["chunk_id"] == 3:
        schema["final_answer"] = "..."
    return compact_json(schema)


def compact_reasoning_chain(chain: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chain_attempt_index": chain.get("chain_attempt_index"),
        "context_mode": chain.get("context_mode"),
        "used_lesson_summary": chain.get("used_lesson_summary"),
        "chunk_trace": chain.get("chunk_trace", []),
        "final_answer": chain.get("final_answer", ""),
        "final_verification": chain.get("final_verification"),
    }


def flatten_confirmed_chunks(confirmed_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    long_cot: List[Dict[str, str]] = []
    for chunk in confirmed_chunks:
        for idx, item in enumerate(chunk["content"], start=1):
            long_cot.append(
                {
                    "action": "Inner Thinking",
                    "title": f"{chunk['chunk_name']} Step {idx}: {item['title']}",
                    "content": "\n".join(
                        [
                            f"Claim: {item['claim']}",
                            f"Evidence: {item['evidence']}",
                            f"Medical rationale: {item['medical_rationale']}",
                        ]
                    ),
                }
            )
        long_cot.append(
            {
                "action": "Inner Thinking",
                "title": f"{chunk['chunk_name']} Summary",
                "content": chunk["chunk_summary"],
            }
        )
    if confirmed_chunks:
        last_chunk = confirmed_chunks[-1]
        long_cot.append({"action": "Final Conclusion", "content": last_chunk.get("final_answer", last_chunk["chunk_summary"])})
        long_cot.append({"action": "Verification", "content": "Verified by chunk checks and final-chain verification."})
    return long_cot


def get_stream_of_search(long_cot: List[Dict[str, str]]) -> str:
    pieces = []
    for item in long_cot:
        if "title" in item:
            pieces.append(f"### {item['title']}\n{item['content']}")
        else:
            pieces.append(f"### {item['action']}\n{item['content']}")
    return "\n\n".join(pieces).strip()


def build_generator_messages(
    question: str,
    confirmed_chunks: List[Dict[str, Any]],
    chunk_spec: Dict[str, Any],
    previous_failed_reasoning: Optional[Dict[str, Any]],
    lesson_summary: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    user_prompt = GENERATOR_USER_TEMPLATE.format(
        question=question,
        confirmed_chunks=compact_json(confirmed_chunks),
        chunk_id=chunk_spec["chunk_id"],
        chunk_name=chunk_spec["chunk_name"],
        chunk_goal=chunk_spec["goal"],
        chunk_requirements=" | ".join(chunk_spec["requirements"]),
        failed_reasoning=compact_json(previous_failed_reasoning) if previous_failed_reasoning else "null",
        lesson_summary=compact_json(lesson_summary) if lesson_summary else "null",
        schema=get_chunk_schema(chunk_spec),
    )
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_chunk_judge_messages(
    question: str,
    reference_answer: str,
    confirmed_chunks: List[Dict[str, Any]],
    candidate_chunk: Dict[str, Any],
    chunk_spec: Dict[str, Any],
) -> List[Dict[str, str]]:
    reference_answer_value = reference_answer if chunk_spec["chunk_id"] == 3 else "null"
    user_prompt = CHUNK_JUDGE_TEMPLATE.format(
        question=question,
        chunk_id=chunk_spec["chunk_id"],
        chunk_name=chunk_spec["chunk_name"],
        chunk_goal=chunk_spec["goal"],
        chunk_requirements=" | ".join(chunk_spec["requirements"]),
        reference_answer=reference_answer_value,
        confirmed_chunks=compact_json(confirmed_chunks),
        candidate_chunk=compact_json(candidate_chunk),
    )
    return [
        {"role": "system", "content": CHUNK_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_final_verify_messages(question: str, reference_answer: str, chain_attempt: Dict[str, Any]) -> List[Dict[str, str]]:
    user_prompt = FINAL_VERIFY_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        completed_reasoning_chain=compact_json(compact_reasoning_chain(chain_attempt)),
    )
    return [
        {"role": "system", "content": FINAL_VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_summary_messages(question: str, failed_reasoning_chains: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    payload = [compact_reasoning_chain(chain) for chain in failed_reasoning_chains]
    user_prompt = SUMMARY_TEMPLATE.format(
        question=question,
        failed_reasoning_chains=compact_json(payload),
    )
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_template_gate_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": TEMPLATE_GATE_SYSTEM_PROMPT},
        {"role": "user", "content": TEMPLATE_GATE_TEMPLATE.format(question=question)},
    ]


class RemoteChatClient:
    def __init__(
        self,
        model_name: str,
        api_url: str,
        api_key: str,
        http_pool_size: int,
        request_timeout: int,
        connect_timeout: int,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.connect_timeout = connect_timeout
        self.http_pool_size = http_pool_size
        self._thread_local = threading.local()

    def _build_session(self) -> requests.Session:
        retry_strategy = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST", "GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=self.http_pool_size,
            pool_maxsize=self.http_pool_size,
            max_retries=retry_strategy,
        )
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = self._build_session()
            self._thread_local.session = session
        return session

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _parse_response(self, response: requests.Response) -> GenerationResult:
        response_data = response.json()
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code}: {response_data}")
        if "error" in response_data:
            raise ValueError(f"API Error: {response_data}")
        choices = response_data.get("choices") or []
        if not choices:
            raise ValueError(f"Invalid API response: {response_data}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            raise ValueError(f"Invalid message content: {response_data}")
        usage = response_data.get("usage") or {}
        return GenerationResult(content=content, usage=usage, raw=response_data)

    def _json_mode_unsupported(self, response: requests.Response) -> bool:
        try:
            response_data = response.json()
        except Exception:
            return False
        text = json.dumps(response_data, ensure_ascii=False).lower()
        if response_data.get("code") == 20024:
            return True
        return "json mode is not supported" in text

    def call_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        if response_format is not None and not model_likely_supports_json_mode(self.model_name):
            response_format = None
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if response_format is not None:
            payload["response_format"] = response_format
        if stop:
            payload["stop"] = stop
        response = self._get_session().post(
            self.api_url,
            headers=self._headers(),
            json=payload,
            timeout=(self.connect_timeout, self.request_timeout),
        )
        if response_format is not None and self._json_mode_unsupported(response):
            payload = dict(payload)
            payload.pop("response_format", None)
            response = self._get_session().post(
                self.api_url,
                headers=self._headers(),
                json=payload,
                timeout=(self.connect_timeout, self.request_timeout),
            )
        return self._parse_response(response)

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    def retry_call_messages(self, messages: List[Dict[str, str]], **kwargs: Any) -> GenerationResult:
        return self.call_messages(messages, **kwargs)


class SiliconFlowBatchClient:
    def __init__(self, api_key: str, base_url: str, request_timeout: int, connect_timeout: int):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.connect_timeout = connect_timeout
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _unwrap_data_object(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = payload.get("data")
        if isinstance(data, dict):
            if any(key in data for key in ("id", "status", "output_file_id", "output_file_url", "result_url")):
                return data
        return payload

    def _extract_nested_id(self, payload: Dict[str, Any], *, id_key: str = "id") -> str:
        payload = self._unwrap_data_object(payload)
        direct = payload.get(id_key)
        if isinstance(direct, str) and direct:
            return direct
        data = payload.get("data")
        if isinstance(data, dict):
            nested = data.get(id_key)
            if isinstance(nested, str) and nested:
                return nested
        raise ValueError(f"Missing `{id_key}` in batch API response: {payload}")

    def _extract_string_field(self, payload: Dict[str, Any], *keys: str) -> Optional[str]:
        payload = self._unwrap_data_object(payload)
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    def _looks_like_url(self, value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    def _batch_cache_paths(self, cache_dir: str, cache_key: str) -> Tuple[str, str]:
        safe_key = sanitize_cache_name(cache_key)
        return (
            os.path.join(cache_dir, f"{safe_key}.meta.json"),
            os.path.join(cache_dir, f"{safe_key}.result.jsonl"),
        )

    def _load_json_file(self, path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _write_json_file(self, path: str, payload: Dict[str, Any]) -> None:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_text_file(self, path: str, content: str) -> None:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _read_text_file(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            url,
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=(self.connect_timeout, self.request_timeout),
        )
        data = response.json()
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code}: {data}")
        return data

    def upload_batch_file(self, requests_jsonl: str) -> str:
        file_obj = io.BytesIO(requests_jsonl.encode("utf-8"))
        response = self.session.post(
            f"{self.base_url}/files",
            headers=self._headers(),
            files={"file": ("batch_input.jsonl", file_obj, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=(self.connect_timeout, self.request_timeout),
        )
        data = response.json()
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code}: {data}")
        return self._extract_nested_id(data, id_key="id")

    def create_batch(self, input_file_id: str, completion_window: str, metadata: Dict[str, str]) -> str:
        data = self._post_json(
            f"{self.base_url}/batches",
            {
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": completion_window,
                "metadata": metadata,
            },
        )
        return self._extract_nested_id(data, id_key="id")

    def retrieve_batch(self, batch_id: str) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/batches/{batch_id}",
            headers=self._headers(),
            timeout=(self.connect_timeout, self.request_timeout),
        )
        data = response.json()
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code}: {data}")
        return self._unwrap_data_object(data)

    def download_file_content(self, file_locator: str) -> str:
        candidate_urls: List[Tuple[str, Dict[str, str]]] = []
        if self._looks_like_url(file_locator):
            candidate_urls.append((file_locator, {}))
        else:
            candidate_urls.append((f"{self.base_url}/files/{file_locator}/content", self._headers()))
            candidate_urls.append((f"{self.base_url}/files/{file_locator}", self._headers()))

        errors: List[str] = []
        for url, headers in candidate_urls:
            response = self.session.get(
                url,
                headers=headers,
                timeout=(self.connect_timeout, self.request_timeout),
            )
            if response.status_code >= 400:
                errors.append(f"{url} -> HTTP {response.status_code}: {response.text[:300]}")
                continue

            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type and not self._looks_like_url(file_locator):
                try:
                    payload = response.json()
                except ValueError:
                    return response.text
                nested_url = self._extract_string_field(
                    payload,
                    "download_url",
                    "content_url",
                    "file_url",
                    "url",
                    "signed_url",
                )
                if nested_url and nested_url != file_locator:
                    candidate_urls.insert(0, (nested_url, {}))
                    continue
            return response.text

        raise ValueError("Unable to download batch file content. " + " | ".join(errors))

    def _resolve_output_file_locator(self, batch: Dict[str, Any]) -> Optional[str]:
        return self._extract_string_field(
            batch,
            "output_file_url",
            "result_url",
            "download_url",
            "output_file_id",
        )

    def _resolve_error_file_locator(self, batch: Dict[str, Any]) -> Optional[str]:
        return self._extract_string_field(
            batch,
            "error_file_url",
            "error_url",
            "error_file_id",
        )

    def _parse_batch_results_content(self, content: str) -> Dict[str, GenerationResult]:
        results: Dict[str, GenerationResult] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item["custom_id"]
            body = ((item.get("response") or {}).get("body")) or item.get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                results[custom_id] = GenerationResult(
                    content="",
                    usage=body.get("usage") or {},
                    raw=item,
                )
                continue
            message = choices[0].get("message") or {}
            content_text = message.get("content")
            if isinstance(content_text, list):
                content_text = "".join(part.get("text", "") for part in content_text if isinstance(part, dict))
            results[custom_id] = GenerationResult(
                content=content_text or "",
                usage=body.get("usage") or {},
                raw=item,
            )
        return results

    def _download_batch_results_with_cache(
        self,
        batch: Dict[str, Any],
        *,
        manifest_path: Optional[str],
        result_path: Optional[str],
        manifest: Optional[Dict[str, Any]],
    ) -> Dict[str, GenerationResult]:
        output_file_locator = self._resolve_output_file_locator(batch)
        error_file_locator = self._resolve_error_file_locator(batch)
        content_parts: List[str] = []
        if output_file_locator:
            content_parts.append(self.download_file_content(output_file_locator).rstrip("\n"))
        if error_file_locator:
            content_parts.append(self.download_file_content(error_file_locator).rstrip("\n"))
        if not content_parts:
            raise ValueError(f"Batch completed without downloadable output locator: {batch}")
        content = "\n".join(part for part in content_parts if part.strip()) + "\n"
        if result_path:
            self._write_text_file(result_path, content)
        if manifest_path:
            manifest_payload = dict(manifest or {})
            manifest_payload.update(
                {
                    "status": "completed",
                    "completed_at_local": int(time.time()),
                    "output_file_locator": output_file_locator,
                    "error_file_locator": error_file_locator,
                    "result_path": result_path,
                    "batch": batch,
                }
            )
            self._write_json_file(manifest_path, manifest_payload)
        return self._parse_batch_results_content(content)

    def _poll_existing_batch(
        self,
        batch_id: str,
        poll_interval: int,
        *,
        manifest_path: Optional[str],
        manifest: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        while True:
            batch = self.retrieve_batch(batch_id)
            if manifest_path:
                manifest_payload = dict(manifest or {})
                manifest_payload.update(
                    {
                        "batch_id": batch_id,
                        "status": batch.get("status"),
                        "last_polled_at": int(time.time()),
                        "batch": batch,
                    }
                )
                self._write_json_file(manifest_path, manifest_payload)
            status = batch.get("status")
            if status == "completed":
                return batch
            if status in {"failed", "cancelled", "expired"}:
                raise ValueError(f"Batch failed: {batch}")
            time.sleep(poll_interval)

    def run_chat_batch(
        self,
        requests_list: List[BatchChatRequest],
        *,
        model_name: str,
        completion_window: str,
        poll_interval: int,
        metadata: Dict[str, str],
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, GenerationResult]:
        manifest_path: Optional[str] = None
        result_path: Optional[str] = None
        manifest: Optional[Dict[str, Any]] = None
        if cache_dir and cache_key:
            ensure_dir(cache_dir)
            manifest_path, result_path = self._batch_cache_paths(cache_dir, cache_key)
            manifest = self._load_json_file(manifest_path)
            if result_path and os.path.exists(result_path):
                return self._parse_batch_results_content(self._read_text_file(result_path))
            if manifest and manifest.get("batch_id"):
                try:
                    batch = self.retrieve_batch(manifest["batch_id"])
                except Exception as exc:
                    manifest = dict(manifest)
                    manifest["resume_error"] = repr(exc)
                    manifest["last_resume_error_at"] = int(time.time())
                    self._write_json_file(manifest_path, manifest)
                else:
                    status = batch.get("status")
                    if status == "completed":
                        return self._download_batch_results_with_cache(
                            batch,
                            manifest_path=manifest_path,
                            result_path=result_path,
                            manifest=manifest,
                        )
                    if status not in {"failed", "cancelled", "expired"}:
                        batch = self._poll_existing_batch(
                            manifest["batch_id"],
                            poll_interval,
                            manifest_path=manifest_path,
                            manifest=manifest,
                        )
                        return self._download_batch_results_with_cache(
                            batch,
                            manifest_path=manifest_path,
                            result_path=result_path,
                            manifest=manifest,
                        )

        jsonl_lines = []
        for item in requests_list:
            body: Dict[str, Any] = {
                "model": model_name,
                "messages": item.messages,
                "max_tokens": item.max_tokens,
            }
            if item.temperature is not None:
                body["temperature"] = item.temperature
            if item.response_format is not None and model_likely_supports_json_mode(model_name):
                body["response_format"] = item.response_format
            jsonl_lines.append(
                compact_json(
                    {
                        "custom_id": item.custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
            )
        file_id = self.upload_batch_file("\n".join(jsonl_lines) + "\n")
        batch_id = self.create_batch(file_id, completion_window=completion_window, metadata=metadata)
        if manifest_path:
            manifest_payload = {
                "cache_key": cache_key,
                "created_at_local": int(time.time()),
                "model_name": model_name,
                "metadata": metadata,
                "request_count": len(requests_list),
                "input_file_id": file_id,
                "batch_id": batch_id,
                "status": "submitted",
            }
            self._write_json_file(manifest_path, manifest_payload)
            manifest = manifest_payload
        batch = self._poll_existing_batch(
            batch_id,
            poll_interval,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        return self._download_batch_results_with_cache(
            batch,
            manifest_path=manifest_path,
            result_path=result_path,
            manifest=manifest,
        )


class LocalJudgeClient:
    def __init__(
        self,
        model_path: str,
        *,
        dtype: str,
        max_gpu_memory_gib: int,
        max_cpu_memory_gib: int,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.max_gpu_memory_gib = max_gpu_memory_gib
        self.max_cpu_memory_gib = max_cpu_memory_gib
        self.supports_batch = True
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        ensure_transformers_cache_dir()

    def _dtype_obj(self):
        import torch

        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float32":
            return torch.float32
        return "auto"

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            max_memory: Dict[Any, str] = {"cpu": f"{self.max_cpu_memory_gib}GiB"}
            if torch.cuda.is_available():
                max_memory[0] = f"{self.max_gpu_memory_gib}GiB"
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self._dtype_obj(),
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
            )
            model.eval()
            self._model = model
            self._tokenizer = tokenizer

    def _generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        *,
        max_tokens: int,
    ) -> List[GenerationResult]:
        self._load()
        import torch

        assert self._model is not None and self._tokenizer is not None
        prompts = [
            self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        with self._infer_lock, torch.inference_mode():
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            input_seq_len = inputs["input_ids"].shape[1]
            model_device = next(self._model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            outputs = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated_rows = outputs[:, input_seq_len:]
        results: List[GenerationResult] = []
        for idx, generated in enumerate(generated_rows):
            text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
            usage = {
                "prompt_tokens": int(prompt_lengths[idx]),
                "completion_tokens": estimate_tokens(text),
                "total_tokens": int(prompt_lengths[idx] + estimate_tokens(text)),
            }
            results.append(GenerationResult(content=text, usage=usage, raw={"backend": "local_hf_batch"}))
        return results

    def call_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        del temperature, response_format, stop
        return self._generate_batch([messages], max_tokens=max_tokens)[0]

    def call_messages_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        *,
        max_tokens: int,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        batch_size: int = DEFAULT_LOCAL_JUDGE_BATCH_SIZE,
    ) -> List[GenerationResult]:
        del temperature, response_format, stop
        if not messages_batch:
            return []
        all_results: List[GenerationResult] = []
        for start in range(0, len(messages_batch), batch_size):
            sub_batch = messages_batch[start : start + batch_size]
            all_results.extend(self._generate_batch(sub_batch, max_tokens=max_tokens))
        return all_results

    @retry(wait_fixed=1000, stop_max_attempt_number=2)
    def retry_call_messages(self, messages: List[Dict[str, str]], **kwargs: Any) -> GenerationResult:
        return self.call_messages(messages, **kwargs)


def append_usage(record: Dict[str, Any], bucket: str, usage: Dict[str, Any], *, fallback_query: str = "", fallback_response: str = "") -> None:
    usage_bucket = record.setdefault("usage_stats", {}).setdefault(
        bucket,
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        },
    )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is None:
        prompt_tokens = estimate_tokens(fallback_query)
    if completion_tokens is None:
        completion_tokens = estimate_tokens(fallback_response)
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    usage_bucket["prompt_tokens"] += int(prompt_tokens)
    usage_bucket["completion_tokens"] += int(completion_tokens)
    usage_bucket["total_tokens"] += int(total_tokens)
    usage_bucket["request_count"] += 1


def init_sample_state(item: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(item)
    item["Question"] = item["Open-ended Verifiable Question"]
    item["Final_Answer"] = ""
    item["Long_CoT"] = []
    item["chunk_trace"] = []
    item["chunk_attempts"] = []
    item["reasoning_chain_attempts"] = []
    item["judge_trace"] = []
    item["lesson_summaries"] = []
    item["sample_status"] = "running"
    item["failure_reason"] = ""
    item["verified_long_cot_ready"] = False
    item["generator_query_cot"] = []
    item["generator_response_cot"] = []
    item["small_judge_query_cot"] = []
    item["small_judge_response_cot"] = []
    item["summary_query_cot"] = []
    item["summary_response_cot"] = []
    item["template_gate_query_cot"] = []
    item["template_gate_response_cot"] = []
    item["template_gate_decision"] = None
    item["final_verification_query_cot"] = []
    item["final_verification_response_cot"] = []
    item["usage_stats"] = {}
    item["_runtime"] = {
        "failed_reasoning_chains": [],
        "reasoning_lesson_summary": None,
        "confirmed_chunks": [],
        "current_chain_attempt_index": 1,
    }
    return item


def heuristic_lesson_summary(failed_reasoning_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
    critiques: List[str] = []
    for chain in failed_reasoning_chains:
        final_verification = chain.get("final_verification") or {}
        critique = final_verification.get("critique")
        fix_hint = final_verification.get("fix_hint")
        if critique:
            critiques.append(critique)
        if fix_hint:
            critiques.append(fix_hint)
    unique = []
    for item in critiques:
        if item not in unique:
            unique.append(item)
    return {
        "confirmed_clues": ["Preserve only findings directly supported by the question and accepted chunks."],
        "avoid_errors": unique[:4] or ["Do not repeat unsupported evidence or wrong final answers from earlier full-chain attempts."],
        "next_chunk_focus": "Restart from chunk 1 and tighten evidence-to-conclusion alignment.",
    }


def finalize_template_gate_decision(item: Dict[str, Any], gate_payload: Dict[str, Any], save_dir: str) -> None:
    item["template_gate_decision"] = gate_payload
    if not gate_payload.get("fit_template", True) or gate_payload.get("skip_recommended", False):
        item["sample_status"] = "template_skip"
        item["failure_reason"] = f"template gate rejected: {gate_payload.get('reason', '').strip()}"
    write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)


def run_template_gate_online(
    items: List[Dict[str, Any]],
    generator_client: RemoteChatClient,
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    def worker(item: Dict[str, Any]) -> None:
        messages = build_template_gate_messages(item["Question"])
        query_text = pretty_json(messages)
        item["template_gate_query_cot"].append(query_text)
        try:
            result = generator_client.retry_call_messages(
                messages,
                max_tokens=args.template_gate_max_tokens,
                temperature=0.0,
                response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
            )
            item["template_gate_response_cot"].append(result.content)
            append_usage(item, "template_gate", result.usage, fallback_query=query_text, fallback_response=result.content)
            _, gate_payload = parse_template_gate_response(result.content)
        except Exception as exc:
            item["template_gate_response_cot"].append(f"TEMPLATE_GATE_CALL_FAILED: {exc}")
            gate_payload = {
                "fit_template": True,
                "skip_recommended": False,
                "question_type": "unknown",
                "reason": f"Template gate failed open: {exc}",
            }
        finalize_template_gate_decision(item, gate_payload, save_dir)

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        futures = [executor.submit(worker, item) for item in items if item["sample_status"] == "running"]
        for future in as_completed(futures):
            future.result()


def run_template_gate_batch(
    items: List[Dict[str, Any]],
    generator_client: RemoteChatClient,
    batch_client: SiliconFlowBatchClient,
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    running_items = [item for item in items if item["sample_status"] == "running"]
    if not running_items:
        return
    batch_cache_dir = os.path.join(save_dir, "_batch_cache")
    requests_list: List[BatchChatRequest] = []
    requests_by_id: Dict[str, Dict[str, Any]] = {}
    for item in running_items:
        messages = build_template_gate_messages(item["Question"])
        custom_id = f"template-gate-{item['process_id']}"
        requests_by_id[custom_id] = {"item": item, "messages": messages}
        requests_list.append(
            BatchChatRequest(
                custom_id=custom_id,
                messages=messages,
                max_tokens=args.template_gate_max_tokens,
                temperature=0.0,
                response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
            )
        )
    stage_metadata = {"stage": "template_gate"}
    stage_name = "template_gate"
    cache_key = build_batch_cache_key(stage_name, generator_client.model_name, requests_list, stage_metadata)
    results = batch_client.run_chat_batch(
        requests_list,
        model_name=generator_client.model_name,
        completion_window=args.batch_completion_window,
        poll_interval=args.batch_poll_interval,
        metadata=stage_metadata,
        cache_dir=batch_cache_dir,
        cache_key=cache_key,
    )
    for custom_id, meta in requests_by_id.items():
        item = meta["item"]
        messages = meta["messages"]
        query_text = pretty_json(messages)
        item["template_gate_query_cot"].append(query_text)
        result = results[custom_id]
        item["template_gate_response_cot"].append(result.content)
        append_usage(item, "template_gate", result.usage, fallback_query=query_text, fallback_response=result.content)
        _, gate_payload = parse_template_gate_response(result.content)
        finalize_template_gate_decision(item, gate_payload, save_dir)


def run_template_gate(
    items: List[Dict[str, Any]],
    generator_client: RemoteChatClient,
    batch_client: Optional[SiliconFlowBatchClient],
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    if not args.enable_template_gate:
        return
    if args.generator_backend == "batch" and batch_client is not None:
        run_template_gate_batch(items, generator_client, batch_client, args, save_dir)
        return
    run_template_gate_online(items, generator_client, args, save_dir)


def write_json(path: str, data: Dict[str, Any]) -> None:
    payload = {k: v for k, v in data.items() if k != "_runtime"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def merge_saved_verified_files(save_dir: str) -> List[Dict[str, Any]]:
    if not os.path.exists(save_dir):
        return []
    merged = []
    for file_name in sorted(os.listdir(save_dir)):
        if not file_name.endswith(".json"):
            continue
        try:
            with open(os.path.join(save_dir, file_name), encoding="utf-8") as f:
                item = json.load(f)
            if item.get("sample_status") == "success" and item.get("verified_long_cot_ready"):
                merged.append(item)
        except Exception:
            continue
    return merged


def build_save_dir(data_path: str, mode_suffix: str) -> Tuple[str, str]:
    task_name = f"{os.path.splitext(os.path.basename(data_path))[0]}_{mode_suffix}"
    save_dir = os.path.join("output_data", task_name)
    return task_name, save_dir


def build_generator_requests(
    samples: List[Dict[str, Any]],
    chunk_spec: Dict[str, Any],
) -> Dict[int, List[Dict[str, str]]]:
    requests_map = {}
    for item in samples:
        runtime = item["_runtime"]
        failed_chains = runtime["failed_reasoning_chains"]
        lesson_summary = runtime["reasoning_lesson_summary"]
        previous_failed_reasoning = compact_reasoning_chain(failed_chains[-1]) if failed_chains and lesson_summary is None else None
        requests_map[item["process_id"]] = build_generator_messages(
            question=item["Question"],
            confirmed_chunks=runtime["confirmed_chunks"],
            chunk_spec=chunk_spec,
            previous_failed_reasoning=previous_failed_reasoning,
            lesson_summary=lesson_summary,
        )
    return requests_map


def local_batch_judge_enabled(judge_client: Any, args: argparse.Namespace) -> bool:
    return args.judge_backend == "local" and hasattr(judge_client, "call_messages_batch")


def build_missing_batch_result(custom_id: str, *, reason: str) -> GenerationResult:
    return GenerationResult(
        content="",
        usage={},
        raw={
            "custom_id": custom_id,
            "response": None,
            "error": reason,
        },
    )


def generate_chunk_online_without_judge(
    item: Dict[str, Any],
    chunk_spec: Dict[str, Any],
    generator_client: RemoteChatClient,
    args: argparse.Namespace,
) -> PendingChunkJudge:
    runtime = item["_runtime"]
    confirmed_chunks = runtime["confirmed_chunks"]
    failed_chains = runtime["failed_reasoning_chains"]
    lesson_summary = runtime["reasoning_lesson_summary"]
    previous_failed_reasoning = compact_reasoning_chain(failed_chains[-1]) if failed_chains and lesson_summary is None else None

    for parse_retry_index in range(1, args.max_chunk_retries + 1):
        messages = build_generator_messages(
            question=item["Question"],
            confirmed_chunks=confirmed_chunks,
            chunk_spec=chunk_spec,
            previous_failed_reasoning=previous_failed_reasoning,
            lesson_summary=lesson_summary,
        )
        query_text = pretty_json(messages)
        item["generator_query_cot"].append(query_text)

        result = generator_client.retry_call_messages(
            messages,
            max_tokens=args.generator_max_tokens,
            temperature=args.generator_temperature,
            response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
        )
        item["generator_response_cot"].append(result.content)
        append_usage(item, "generator", result.usage, fallback_query=query_text, fallback_response=result.content)

        attempt_record: Dict[str, Any] = {
            "chain_attempt_index": runtime["current_chain_attempt_index"],
            "chunk_id": chunk_spec["chunk_id"],
            "chunk_name": chunk_spec["chunk_name"],
            "attempt_index": parse_retry_index,
            "generator_messages": messages,
            "generator_response": result.content,
        }
        parse_ok, candidate_chunk, parse_error = parse_chunk_response(result.content, chunk_spec)
        attempt_record["parse_ok"] = parse_ok
        attempt_record["parse_error"] = parse_error
        if not parse_ok or candidate_chunk is None:
            attempt_record["result"] = "parse_failed"
            item["chunk_attempts"].append(attempt_record)
            continue

        attempt_record["candidate_chunk"] = candidate_chunk
        return PendingChunkJudge(
            item=item,
            chunk_spec=chunk_spec,
            attempt_record=attempt_record,
            candidate_chunk=candidate_chunk,
        )

    raise HardRejectError(f"chunk_{chunk_spec['chunk_id']} exceeded max_chunk_retries={args.max_chunk_retries}")


def collect_chunk_candidates_online(
    candidates: List[Dict[str, Any]],
    chunk_spec: Dict[str, Any],
    generator_client: RemoteChatClient,
    args: argparse.Namespace,
    save_dir: str,
) -> List[PendingChunkJudge]:
    pending_judges: List[PendingChunkJudge] = []
    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        future_to_item = {
            executor.submit(generate_chunk_online_without_judge, item, chunk_spec, generator_client, args): item
            for item in candidates
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                pending_judges.append(future.result())
            except HardRejectError as exc:
                item["sample_status"] = "hard_reject"
                item["failure_reason"] = str(exc)
            except Exception as exc:
                traceback.print_exc()
                item["sample_status"] = "error"
                item["failure_reason"] = repr(exc)
            finally:
                write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
    return pending_judges


def collect_chunk_candidates_batch(
    samples: List[Dict[str, Any]],
    chunk_spec: Dict[str, Any],
    generator_client: RemoteChatClient,
    batch_client: SiliconFlowBatchClient,
    args: argparse.Namespace,
    save_dir: str,
) -> List[PendingChunkJudge]:
    pending = list(samples)
    pending_judges: List[PendingChunkJudge] = []
    batch_cache_dir = os.path.join(save_dir, "_batch_cache")
    for parse_retry_index in range(1, args.max_chunk_retries + 1):
        if not pending:
            break
        batch_requests: List[BatchChatRequest] = []
        requests_by_id: Dict[str, Dict[str, Any]] = {}
        for item in pending:
            runtime = item["_runtime"]
            failed_chains = runtime["failed_reasoning_chains"]
            lesson_summary = runtime["reasoning_lesson_summary"]
            previous_failed_reasoning = compact_reasoning_chain(failed_chains[-1]) if failed_chains and lesson_summary is None else None
            messages = build_generator_messages(
                question=item["Question"],
                confirmed_chunks=runtime["confirmed_chunks"],
                chunk_spec=chunk_spec,
                previous_failed_reasoning=previous_failed_reasoning,
                lesson_summary=lesson_summary,
            )
            custom_id = f"chunk-{chunk_spec['chunk_id']}-sample-{item['process_id']}-try-{parse_retry_index}"
            requests_by_id[custom_id] = {"item": item, "messages": messages}
            batch_requests.append(
                BatchChatRequest(
                    custom_id=custom_id,
                    messages=messages,
                    max_tokens=args.generator_max_tokens,
                    temperature=args.generator_temperature,
                    response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
                )
            )

        stage_name = f"chunk_{chunk_spec['chunk_id']}_try_{parse_retry_index}"
        stage_metadata = {"stage": f"chunk_{chunk_spec['chunk_id']}"}
        cache_key = build_batch_cache_key(stage_name, generator_client.model_name, batch_requests, stage_metadata)
        batch_results = batch_client.run_chat_batch(
            batch_requests,
            model_name=generator_client.model_name,
            completion_window=args.batch_completion_window,
            poll_interval=args.batch_poll_interval,
            metadata=stage_metadata,
            cache_dir=batch_cache_dir,
            cache_key=cache_key,
        )

        next_pending: List[Dict[str, Any]] = []
        for custom_id, request_meta in requests_by_id.items():
            item = request_meta["item"]
            messages = request_meta["messages"]
            query_text = pretty_json(messages)
            item["generator_query_cot"].append(query_text)
            result = batch_results.get(
                custom_id,
                build_missing_batch_result(custom_id, reason="Missing result row in batch output."),
            )
            item["generator_response_cot"].append(result.content)
            append_usage(item, "generator", result.usage, fallback_query=query_text, fallback_response=result.content)
            attempt_record: Dict[str, Any] = {
                "chain_attempt_index": item["_runtime"]["current_chain_attempt_index"],
                "chunk_id": chunk_spec["chunk_id"],
                "chunk_name": chunk_spec["chunk_name"],
                "attempt_index": parse_retry_index,
                "generator_messages": messages,
                "generator_response": result.content,
                "generator_backend": "batch",
                "generator_raw": result.raw,
            }
            parse_ok, candidate_chunk, parse_error = parse_chunk_response(result.content, chunk_spec)
            attempt_record["parse_ok"] = parse_ok
            attempt_record["parse_error"] = parse_error or str((result.raw or {}).get("error", ""))
            if not parse_ok or candidate_chunk is None:
                attempt_record["result"] = "parse_failed"
                item["chunk_attempts"].append(attempt_record)
                next_pending.append(item)
                write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
                continue

            attempt_record["candidate_chunk"] = candidate_chunk
            pending_judges.append(
                PendingChunkJudge(
                    item=item,
                    chunk_spec=chunk_spec,
                    attempt_record=attempt_record,
                    candidate_chunk=candidate_chunk,
                )
            )
            write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
        pending = next_pending

    for item in pending:
        item["sample_status"] = "hard_reject"
        item["failure_reason"] = f"chunk_{chunk_spec['chunk_id']} exceeded max_chunk_retries={args.max_chunk_retries}"
        write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)

    return pending_judges


def run_local_chunk_judge_batches(
    pending_judges: List[PendingChunkJudge],
    judge_client: Any,
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    remaining = list(pending_judges)
    for judge_attempt_index in range(1, args.max_chunk_judge_attempts + 1):
        if not remaining:
            return
        messages_batch: List[List[Dict[str, str]]] = []
        query_texts: List[str] = []
        for pending in remaining:
            judge_messages = build_chunk_judge_messages(
                question=pending.item["Question"],
                reference_answer=pending.item["Ground-True Answer"],
                confirmed_chunks=pending.item["_runtime"]["confirmed_chunks"],
                candidate_chunk=pending.candidate_chunk,
                chunk_spec=pending.chunk_spec,
            )
            query_text = pretty_json(judge_messages)
            pending.item["small_judge_query_cot"].append(query_text)
            messages_batch.append(judge_messages)
            query_texts.append(query_text)

        judge_results = judge_client.call_messages_batch(
            messages_batch,
            max_tokens=args.judge_max_tokens,
            temperature=0.0,
            response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
            batch_size=args.local_judge_batch_size,
        )

        next_remaining: List[PendingChunkJudge] = []
        for pending, judge_result, query_text in zip(remaining, judge_results, query_texts):
            pending.item["small_judge_response_cot"].append(judge_result.content)
            append_usage(
                pending.item,
                "judge",
                judge_result.usage,
                fallback_query=query_text,
                fallback_response=judge_result.content,
            )
            parsed_ok, judge_payload = parse_judge_response(judge_result.content)
            judge_trace = {
                "judge_role": "chunk",
                "backend": "local-batch",
                "chunk_id": pending.chunk_spec["chunk_id"],
                "chunk_name": pending.chunk_spec["chunk_name"],
                "reference_answer_included": pending.chunk_spec["chunk_id"] == 3,
                "judge_attempt_index": judge_attempt_index,
                "query": query_text,
                "response": judge_result.content,
                "parsed_ok": parsed_ok,
                "result": judge_payload,
            }
            pending.item["judge_trace"].append(judge_trace)
            pending.attempt_record.setdefault("small_judge_rounds", []).append(
                {"judge_attempt_index": judge_attempt_index, "result": judge_payload}
            )
            pending.attempt_record["small_judge"] = judge_payload

            if judge_payload["pass"]:
                pending.attempt_record["result"] = "accepted"
                pending.item["chunk_attempts"].append(pending.attempt_record)
                pending.item["_runtime"]["confirmed_chunks"].append(pending.candidate_chunk)
            elif judge_attempt_index >= args.max_chunk_judge_attempts:
                pending.attempt_record["result"] = "small_reject_hard"
                pending.item["chunk_attempts"].append(pending.attempt_record)
                pending.item["sample_status"] = "hard_reject"
                pending.item["failure_reason"] = (
                    f"small-model judge rejected chunk_{pending.chunk_spec['chunk_id']} "
                    f"{args.max_chunk_judge_attempts} times"
                )
            else:
                next_remaining.append(pending)

            write_json(os.path.join(save_dir, f"{pending.item['process_id']}.json"), pending.item)

        remaining = next_remaining


def generate_chunk_online(
    item: Dict[str, Any],
    chunk_spec: Dict[str, Any],
    generator_client: RemoteChatClient,
    judge_client: Any,
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    runtime = item["_runtime"]
    confirmed_chunks = runtime["confirmed_chunks"]
    failed_chains = runtime["failed_reasoning_chains"]
    lesson_summary = runtime["reasoning_lesson_summary"]
    previous_failed_reasoning = compact_reasoning_chain(failed_chains[-1]) if failed_chains and lesson_summary is None else None

    for parse_retry_index in range(1, args.max_chunk_retries + 1):
        messages = build_generator_messages(
            question=item["Question"],
            confirmed_chunks=confirmed_chunks,
            chunk_spec=chunk_spec,
            previous_failed_reasoning=previous_failed_reasoning,
            lesson_summary=lesson_summary,
        )
        query_text = pretty_json(messages)
        item["generator_query_cot"].append(query_text)

        result = generator_client.retry_call_messages(
            messages,
            max_tokens=args.generator_max_tokens,
            temperature=args.generator_temperature,
            response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
        )
        item["generator_response_cot"].append(result.content)
        append_usage(item, "generator", result.usage, fallback_query=query_text, fallback_response=result.content)

        attempt_record: Dict[str, Any] = {
            "chain_attempt_index": runtime["current_chain_attempt_index"],
            "chunk_id": chunk_spec["chunk_id"],
            "chunk_name": chunk_spec["chunk_name"],
            "attempt_index": parse_retry_index,
            "generator_messages": messages,
            "generator_response": result.content,
        }
        parse_ok, candidate_chunk, parse_error = parse_chunk_response(result.content, chunk_spec)
        attempt_record["parse_ok"] = parse_ok
        attempt_record["parse_error"] = parse_error
        if not parse_ok or candidate_chunk is None:
            attempt_record["result"] = "parse_failed"
            item["chunk_attempts"].append(attempt_record)
            continue

        attempt_record["candidate_chunk"] = candidate_chunk
        judge_rounds = []
        final_judge_result = None
        pass_chunk = False
        for judge_attempt_index in range(1, args.max_chunk_judge_attempts + 1):
            judge_messages = build_chunk_judge_messages(
                question=item["Question"],
                reference_answer=item["Ground-True Answer"],
                confirmed_chunks=confirmed_chunks,
                candidate_chunk=candidate_chunk,
                chunk_spec=chunk_spec,
            )
            judge_query_text = pretty_json(judge_messages)
            item["small_judge_query_cot"].append(judge_query_text)
            judge_result = judge_client.retry_call_messages(
                judge_messages,
                max_tokens=args.judge_max_tokens,
                temperature=0.0,
                response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
            )
            item["small_judge_response_cot"].append(judge_result.content)
            append_usage(item, "judge", judge_result.usage, fallback_query=judge_query_text, fallback_response=judge_result.content)
            parsed_ok, judge_payload = parse_judge_response(judge_result.content)
            judge_trace = {
                "judge_role": "chunk",
                "backend": "local" if args.judge_backend == "local" else "remote",
                "chunk_id": chunk_spec["chunk_id"],
                "chunk_name": chunk_spec["chunk_name"],
                "reference_answer_included": chunk_spec["chunk_id"] == 3,
                "judge_attempt_index": judge_attempt_index,
                "query": judge_query_text,
                "response": judge_result.content,
                "parsed_ok": parsed_ok,
                "result": judge_payload,
            }
            item["judge_trace"].append(judge_trace)
            judge_rounds.append({"judge_attempt_index": judge_attempt_index, "result": judge_payload})
            final_judge_result = judge_payload
            if judge_payload["pass"]:
                pass_chunk = True
                break

        attempt_record["small_judge_rounds"] = judge_rounds
        attempt_record["small_judge"] = final_judge_result
        if not pass_chunk:
            attempt_record["result"] = "small_reject_hard"
            item["chunk_attempts"].append(attempt_record)
            raise HardRejectError(f"small-model judge rejected chunk_{chunk_spec['chunk_id']} {args.max_chunk_judge_attempts} times")

        attempt_record["result"] = "accepted"
        item["chunk_attempts"].append(attempt_record)
        confirmed_chunks.append(candidate_chunk)
        write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
        return

    raise HardRejectError(f"chunk_{chunk_spec['chunk_id']} exceeded max_chunk_retries={args.max_chunk_retries}")


def generate_chunk_batch(
    samples: List[Dict[str, Any]],
    chunk_spec: Dict[str, Any],
    generator_client: RemoteChatClient,
    judge_client: Any,
    batch_client: SiliconFlowBatchClient,
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    pending = list(samples)
    batch_cache_dir = os.path.join(save_dir, "_batch_cache")
    for parse_retry_index in range(1, args.max_chunk_retries + 1):
        batch_requests: List[BatchChatRequest] = []
        requests_by_id: Dict[str, Dict[str, Any]] = {}
        for item in pending:
            runtime = item["_runtime"]
            failed_chains = runtime["failed_reasoning_chains"]
            lesson_summary = runtime["reasoning_lesson_summary"]
            previous_failed_reasoning = compact_reasoning_chain(failed_chains[-1]) if failed_chains and lesson_summary is None else None
            messages = build_generator_messages(
                question=item["Question"],
                confirmed_chunks=runtime["confirmed_chunks"],
                chunk_spec=chunk_spec,
                previous_failed_reasoning=previous_failed_reasoning,
                lesson_summary=lesson_summary,
            )
            custom_id = f"chunk-{chunk_spec['chunk_id']}-sample-{item['process_id']}-try-{parse_retry_index}"
            requests_by_id[custom_id] = {"item": item, "messages": messages}
            batch_requests.append(
                BatchChatRequest(
                    custom_id=custom_id,
                    messages=messages,
                    max_tokens=args.generator_max_tokens,
                    temperature=args.generator_temperature,
                    response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
                )
            )
        stage_name = f"chunk_{chunk_spec['chunk_id']}_try_{parse_retry_index}"
        stage_metadata = {"stage": f"chunk_{chunk_spec['chunk_id']}"}
        cache_key = build_batch_cache_key(stage_name, generator_client.model_name, batch_requests, stage_metadata)
        batch_results = batch_client.run_chat_batch(
            batch_requests,
            model_name=generator_client.model_name,
            completion_window=args.batch_completion_window,
            poll_interval=args.batch_poll_interval,
            metadata=stage_metadata,
            cache_dir=batch_cache_dir,
            cache_key=cache_key,
        )
        next_pending = []
        for custom_id, request_meta in requests_by_id.items():
            item = request_meta["item"]
            messages = request_meta["messages"]
            query_text = pretty_json(messages)
            item["generator_query_cot"].append(query_text)
            result = batch_results.get(
                custom_id,
                build_missing_batch_result(custom_id, reason="Missing result row in batch output."),
            )
            item["generator_response_cot"].append(result.content)
            append_usage(item, "generator", result.usage, fallback_query=query_text, fallback_response=result.content)
            attempt_record: Dict[str, Any] = {
                "chain_attempt_index": item["_runtime"]["current_chain_attempt_index"],
                "chunk_id": chunk_spec["chunk_id"],
                "chunk_name": chunk_spec["chunk_name"],
                "attempt_index": parse_retry_index,
                "generator_messages": messages,
                "generator_response": result.content,
                "generator_backend": "batch",
                "generator_raw": result.raw,
            }
            parse_ok, candidate_chunk, parse_error = parse_chunk_response(result.content, chunk_spec)
            attempt_record["parse_ok"] = parse_ok
            attempt_record["parse_error"] = parse_error or str((result.raw or {}).get("error", ""))
            if not parse_ok or candidate_chunk is None:
                attempt_record["result"] = "parse_failed"
                item["chunk_attempts"].append(attempt_record)
                next_pending.append(item)
                continue

            attempt_record["candidate_chunk"] = candidate_chunk
            judge_rounds = []
            final_judge_result = None
            pass_chunk = False
            for judge_attempt_index in range(1, args.max_chunk_judge_attempts + 1):
                judge_messages = build_chunk_judge_messages(
                    question=item["Question"],
                    reference_answer=item["Ground-True Answer"],
                    confirmed_chunks=item["_runtime"]["confirmed_chunks"],
                    candidate_chunk=candidate_chunk,
                    chunk_spec=chunk_spec,
                )
                judge_query_text = pretty_json(judge_messages)
                item["small_judge_query_cot"].append(judge_query_text)
                judge_result = judge_client.retry_call_messages(
                    judge_messages,
                    max_tokens=args.judge_max_tokens,
                    temperature=0.0,
                    response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
                )
                item["small_judge_response_cot"].append(judge_result.content)
                append_usage(item, "judge", judge_result.usage, fallback_query=judge_query_text, fallback_response=judge_result.content)
                parsed_ok, judge_payload = parse_judge_response(judge_result.content)
                item["judge_trace"].append(
                    {
                        "judge_role": "chunk",
                        "backend": "local" if args.judge_backend == "local" else "remote",
                        "chunk_id": chunk_spec["chunk_id"],
                        "chunk_name": chunk_spec["chunk_name"],
                        "reference_answer_included": chunk_spec["chunk_id"] == 3,
                        "judge_attempt_index": judge_attempt_index,
                        "query": judge_query_text,
                        "response": judge_result.content,
                        "parsed_ok": parsed_ok,
                        "result": judge_payload,
                    }
                )
                judge_rounds.append({"judge_attempt_index": judge_attempt_index, "result": judge_payload})
                final_judge_result = judge_payload
                if judge_payload["pass"]:
                    pass_chunk = True
                    break
            attempt_record["small_judge_rounds"] = judge_rounds
            attempt_record["small_judge"] = final_judge_result
            if not pass_chunk:
                attempt_record["result"] = "small_reject_hard"
                item["chunk_attempts"].append(attempt_record)
                item["sample_status"] = "hard_reject"
                item["failure_reason"] = f"small-model judge rejected chunk_{chunk_spec['chunk_id']} {args.max_chunk_judge_attempts} times"
                write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
                continue
            attempt_record["result"] = "accepted"
            item["chunk_attempts"].append(attempt_record)
            item["_runtime"]["confirmed_chunks"].append(candidate_chunk)
            write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
        pending = [item for item in next_pending if item.get("sample_status") == "running"]
        if not pending:
            return
    for item in pending:
        item["sample_status"] = "hard_reject"
        item["failure_reason"] = f"chunk_{chunk_spec['chunk_id']} exceeded max_chunk_retries={args.max_chunk_retries}"
        write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)


def run_chunk_stage(
    items: List[Dict[str, Any]],
    chunk_spec: Dict[str, Any],
    generator_client: RemoteChatClient,
    judge_client: Any,
    batch_client: Optional[SiliconFlowBatchClient],
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    candidates = [
        item
        for item in items
        if item["sample_status"] == "running" and len(item["_runtime"]["confirmed_chunks"]) == chunk_spec["chunk_id"] - 1
    ]
    if not candidates:
        return
    if local_batch_judge_enabled(judge_client, args):
        if args.generator_backend == "batch":
            if batch_client is None:
                raise ValueError("Batch backend requested but batch client is not initialized.")
            pending_judges = collect_chunk_candidates_batch(
                candidates,
                chunk_spec,
                generator_client,
                batch_client,
                args,
                save_dir,
            )
        else:
            pending_judges = collect_chunk_candidates_online(
                candidates,
                chunk_spec,
                generator_client,
                args,
                save_dir,
            )
        run_local_chunk_judge_batches(pending_judges, judge_client, args, save_dir)
        return
    if args.generator_backend == "batch":
        if batch_client is None:
            raise ValueError("Batch backend requested but batch client is not initialized.")
        generate_chunk_batch(candidates, chunk_spec, generator_client, judge_client, batch_client, args, save_dir)
        return

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        future_to_item = {
            executor.submit(generate_chunk_online, item, chunk_spec, generator_client, judge_client, args, save_dir): item
            for item in candidates
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                future.result()
            except HardRejectError as exc:
                item["sample_status"] = "hard_reject"
                item["failure_reason"] = str(exc)
            except Exception as exc:
                traceback.print_exc()
                item["sample_status"] = "error"
                item["failure_reason"] = repr(exc)
            finally:
                write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)


def summarize_lessons_for_item(item: Dict[str, Any], summary_client: RemoteChatClient, args: argparse.Namespace) -> Dict[str, Any]:
    failed_chains = item["_runtime"]["failed_reasoning_chains"]
    messages = build_summary_messages(item["Question"], failed_chains)
    query_text = pretty_json(messages)
    item["summary_query_cot"].append(query_text)
    try:
        result = summary_client.retry_call_messages(
            messages,
            max_tokens=args.summary_max_tokens,
            temperature=0.0,
            response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
        )
        item["summary_response_cot"].append(result.content)
        append_usage(item, "summary", result.usage, fallback_query=query_text, fallback_response=result.content)
        parsed_ok, summary = parse_summary_response(result.content)
    except Exception as exc:
        item["summary_response_cot"].append(f"SUMMARY_CALL_FAILED: {exc}")
        parsed_ok, summary = False, None
    if not parsed_ok or summary is None:
        summary = heuristic_lesson_summary(failed_chains)
    item["lesson_summaries"].append(
        {
            "summary_type": "reasoning_chain",
            "trigger_fail_depth": len(failed_chains),
            "summary": summary,
        }
    )
    return summary


def verify_final_chain(item: Dict[str, Any], judge_client: Any, args: argparse.Namespace) -> None:
    confirmed_chunks = item["_runtime"]["confirmed_chunks"]
    if len(confirmed_chunks) != len(CHUNK_PLAN):
        return
    chain_attempt = {
        "chain_attempt_index": item["_runtime"]["current_chain_attempt_index"],
        "context_mode": "summary" if item["_runtime"]["reasoning_lesson_summary"] is not None else ("raw_failed_chain" if item["_runtime"]["failed_reasoning_chains"] else "none"),
        "used_lesson_summary": item["_runtime"]["reasoning_lesson_summary"],
        "chunk_trace": confirmed_chunks,
        "final_answer": confirmed_chunks[-1].get("final_answer", ""),
    }
    messages = build_final_verify_messages(item["Question"], item["Ground-True Answer"], chain_attempt)
    query_text = pretty_json(messages)
    item["final_verification_query_cot"].append(query_text)
    result = judge_client.retry_call_messages(
        messages,
        max_tokens=args.judge_max_tokens,
        temperature=0.0,
        response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
    )
    item["final_verification_response_cot"].append(result.content)
    append_usage(item, "judge", result.usage, fallback_query=query_text, fallback_response=result.content)
    parsed_ok, final_verification = parse_judge_response(result.content)
    item["judge_trace"].append(
        {
            "judge_role": "final-chain",
            "backend": "local" if args.judge_backend == "local" else "remote",
            "query": query_text,
            "response": result.content,
            "parsed_ok": parsed_ok,
            "result": final_verification,
        }
    )
    chain_attempt["final_verification"] = final_verification
    item["reasoning_chain_attempts"].append(chain_attempt)
    if final_verification["pass"]:
        item["chunk_trace"] = confirmed_chunks
        item["Long_CoT"] = flatten_confirmed_chunks(confirmed_chunks)
        item["Final_Answer"] = confirmed_chunks[-1].get("final_answer", "")
        item["sample_status"] = "success"
        item["verified_long_cot_ready"] = True
        return

    item["_runtime"]["failed_reasoning_chains"].append(chain_attempt)
    if len(item["_runtime"]["failed_reasoning_chains"]) >= args.compression_trigger_fail_depth:
        item["_runtime"]["reasoning_lesson_summary"] = summarize_lessons_for_item(item, args.summary_client, args)
    else:
        item["_runtime"]["reasoning_lesson_summary"] = None
    item["_runtime"]["confirmed_chunks"] = []
    item["_runtime"]["current_chain_attempt_index"] += 1
    if item["_runtime"]["current_chain_attempt_index"] > args.max_reasoning_chain_attempts:
        item["sample_status"] = "hard_reject"
        item["failure_reason"] = (
            "reasoning_chain final verification kept failing and exceeded "
            f"max_reasoning_chain_attempts={args.max_reasoning_chain_attempts}"
        )


def verify_final_chains_local_batch(
    items: List[Dict[str, Any]],
    judge_client: Any,
    args: argparse.Namespace,
    save_dir: str,
) -> None:
    if not items:
        return

    messages_batch: List[List[Dict[str, str]]] = []
    metadata: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    for item in items:
        confirmed_chunks = item["_runtime"]["confirmed_chunks"]
        if len(confirmed_chunks) != len(CHUNK_PLAN):
            continue
        chain_attempt = {
            "chain_attempt_index": item["_runtime"]["current_chain_attempt_index"],
            "context_mode": "summary"
            if item["_runtime"]["reasoning_lesson_summary"] is not None
            else ("raw_failed_chain" if item["_runtime"]["failed_reasoning_chains"] else "none"),
            "used_lesson_summary": item["_runtime"]["reasoning_lesson_summary"],
            "chunk_trace": confirmed_chunks,
            "final_answer": confirmed_chunks[-1].get("final_answer", ""),
        }
        messages = build_final_verify_messages(item["Question"], item["Ground-True Answer"], chain_attempt)
        query_text = pretty_json(messages)
        item["final_verification_query_cot"].append(query_text)
        messages_batch.append(messages)
        metadata.append((item, chain_attempt, query_text))

    if not messages_batch:
        return

    judge_results = judge_client.call_messages_batch(
        messages_batch,
        max_tokens=args.judge_max_tokens,
        temperature=0.0,
        response_format=JSON_RESPONSE_FORMAT if args.use_json_mode else None,
        batch_size=args.local_judge_batch_size,
    )

    for judge_result, (item, chain_attempt, query_text) in zip(judge_results, metadata):
        item["final_verification_response_cot"].append(judge_result.content)
        append_usage(item, "judge", judge_result.usage, fallback_query=query_text, fallback_response=judge_result.content)
        parsed_ok, final_verification = parse_judge_response(judge_result.content)
        item["judge_trace"].append(
            {
                "judge_role": "final-chain",
                "backend": "local-batch",
                "query": query_text,
                "response": judge_result.content,
                "parsed_ok": parsed_ok,
                "result": final_verification,
            }
        )
        chain_attempt["final_verification"] = final_verification
        item["reasoning_chain_attempts"].append(chain_attempt)
        if final_verification["pass"]:
            confirmed_chunks = item["_runtime"]["confirmed_chunks"]
            item["chunk_trace"] = confirmed_chunks
            item["Long_CoT"] = flatten_confirmed_chunks(confirmed_chunks)
            item["Final_Answer"] = confirmed_chunks[-1].get("final_answer", "")
            item["sample_status"] = "success"
            item["verified_long_cot_ready"] = True
            write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
            continue

        item["_runtime"]["failed_reasoning_chains"].append(chain_attempt)
        if len(item["_runtime"]["failed_reasoning_chains"]) >= args.compression_trigger_fail_depth:
            item["_runtime"]["reasoning_lesson_summary"] = summarize_lessons_for_item(item, args.summary_client, args)
        else:
            item["_runtime"]["reasoning_lesson_summary"] = None
        item["_runtime"]["confirmed_chunks"] = []
        item["_runtime"]["current_chain_attempt_index"] += 1
        if item["_runtime"]["current_chain_attempt_index"] > args.max_reasoning_chain_attempts:
            item["sample_status"] = "hard_reject"
            item["failure_reason"] = (
                "reasoning_chain final verification kept failing and exceeded "
                f"max_reasoning_chain_attempts={args.max_reasoning_chain_attempts}"
            )
        write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)


def run_verified_long_cot_pipeline(
    input_data: List[Dict[str, Any]],
    *,
    save_dir: str,
    generator_client: RemoteChatClient,
    judge_client: Any,
    summary_client: RemoteChatClient,
    batch_client: Optional[SiliconFlowBatchClient],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    args.summary_client = summary_client
    items = [init_sample_state(item) for item in input_data]
    for item in items:
        write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)

    run_template_gate(items, generator_client, batch_client, args, save_dir)

    while True:
        running = [item for item in items if item["sample_status"] == "running"]
        if not running:
            break
        current_chain_attempt = min(item["_runtime"]["current_chain_attempt_index"] for item in running)
        current_wave = [
            item
            for item in running
            if item["_runtime"]["current_chain_attempt_index"] == current_chain_attempt and not item["_runtime"]["confirmed_chunks"]
        ]
        if not current_wave:
            current_wave = [item for item in running if item["_runtime"]["current_chain_attempt_index"] == current_chain_attempt]

        for chunk_spec in CHUNK_PLAN:
            stage_items = [
                item
                for item in items
                if item["sample_status"] == "running"
                and item["_runtime"]["current_chain_attempt_index"] == current_chain_attempt
                and len(item["_runtime"]["confirmed_chunks"]) == chunk_spec["chunk_id"] - 1
            ]
            if not stage_items:
                continue
            run_chunk_stage(stage_items, chunk_spec, generator_client, judge_client, batch_client, args, save_dir)

        final_candidates = [
            item
            for item in items
            if item["sample_status"] == "running"
            and item["_runtime"]["current_chain_attempt_index"] == current_chain_attempt
            and len(item["_runtime"]["confirmed_chunks"]) == len(CHUNK_PLAN)
        ]
        if final_candidates:
            if local_batch_judge_enabled(judge_client, args):
                try:
                    verify_final_chains_local_batch(final_candidates, judge_client, args, save_dir)
                except Exception as exc:
                    traceback.print_exc()
                    for item in final_candidates:
                        item["sample_status"] = "error"
                        item["failure_reason"] = repr(exc)
                        write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
            else:
                for item in final_candidates:
                    try:
                        verify_final_chain(item, judge_client, args)
                    except Exception as exc:
                        traceback.print_exc()
                        item["sample_status"] = "error"
                        item["failure_reason"] = repr(exc)
                    finally:
                        write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)

        exhausted = [
            item
            for item in items
            if item["sample_status"] == "running"
            and item["_runtime"]["current_chain_attempt_index"] > args.max_reasoning_chain_attempts
        ]
        for item in exhausted:
            item["sample_status"] = "hard_reject"
            item["failure_reason"] = (
                "reasoning_chain final verification kept failing and exceeded "
                f"max_reasoning_chain_attempts={args.max_reasoning_chain_attempts}"
            )
            write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)

        stalled = [
            item
            for item in items
            if item["sample_status"] == "running"
            and item["_runtime"]["current_chain_attempt_index"] == current_chain_attempt
            and not item["_runtime"]["confirmed_chunks"]
        ]
        if not any(item["sample_status"] == "success" for item in items) and not stalled and not final_candidates:
            break

    for item in items:
        write_json(os.path.join(save_dir, f"{item['process_id']}.json"), item)
    return items


def deduplicate_input(data: List[Dict[str, Any]], processed_successes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_ids = {item["process_id"] for item in processed_successes}
    return [item for item in data if item["process_id"] not in processed_ids]


def filter_input_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for item in raw_data:
        if "Open-ended Verifiable Question" not in item or "Ground-True Answer" not in item:
            continue
        filtered.append(item)
    return filtered


def write_batch_export_jsonl(
    items: List[Dict[str, Any]],
    output_path: str,
    *,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> None:
    lines = []
    for item in items:
        lines.append(
            compact_json(
                {
                    "custom_id": f"postprocess-{item['process_id']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": item["messages"],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "response_format": JSON_RESPONSE_FORMAT,
                    },
                }
            )
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--api_url", type=str, default=DEFAULT_SILICONFLOW_API_URL)
    parser.add_argument("--base_url", type=str, default=DEFAULT_SILICONFLOW_BASE_URL)
    parser.add_argument("--generator_model_name", type=str, default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--summary_model_name", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--small_judge_model_name", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge_backend", choices=["local", "remote"], default="local")
    parser.add_argument("--generator_backend", choices=["online", "batch"], default="online")
    parser.add_argument("--local_judge_model_path", type=str, default=DEFAULT_LOCAL_JUDGE_MODEL_PATH)
    parser.add_argument("--local_judge_dtype", choices=["float16", "bfloat16", "float32", "auto"], default="float16")
    parser.add_argument("--local_judge_max_gpu_memory_gib", type=int, default=DEFAULT_LOCAL_JUDGE_GPU_GIB)
    parser.add_argument("--local_judge_max_cpu_memory_gib", type=int, default=DEFAULT_LOCAL_JUDGE_CPU_GIB)
    parser.add_argument("--local_judge_batch_size", type=int, default=DEFAULT_LOCAL_JUDGE_BATCH_SIZE)
    parser.add_argument("--num_process", type=int, default=DEFAULT_NUM_PROCESS)
    parser.add_argument("--limit_num", type=int)
    parser.add_argument("--max_reasoning_chain_attempts", type=int, default=DEFAULT_MAX_CHAIN_ATTEMPTS)
    parser.add_argument("--max_chunk_retries", type=int, default=DEFAULT_MAX_CHUNK_RETRIES)
    parser.add_argument("--max_chunk_judge_attempts", type=int, default=DEFAULT_MAX_CHUNK_JUDGE_ATTEMPTS)
    parser.add_argument("--compression_trigger_fail_depth", type=int, default=DEFAULT_COMPRESSION_TRIGGER)
    parser.add_argument("--generator_max_tokens", type=int, default=DEFAULT_GENERATOR_MAX_TOKENS)
    parser.add_argument("--judge_max_tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    parser.add_argument("--summary_max_tokens", type=int, default=DEFAULT_SUMMARY_MAX_TOKENS)
    parser.add_argument("--template_gate_max_tokens", type=int, default=DEFAULT_TEMPLATE_GATE_MAX_TOKENS)
    parser.add_argument("--generator_temperature", type=float, default=0.1)
    parser.add_argument("--request_timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--connect_timeout", type=int, default=DEFAULT_CONNECT_TIMEOUT)
    parser.add_argument("--http_pool_size", type=int, default=DEFAULT_HTTP_POOL_SIZE)
    parser.add_argument("--use_json_mode", action="store_true", default=True)
    parser.add_argument("--disable_json_mode", action="store_true")
    parser.add_argument("--disable_template_gate", action="store_true")
    parser.add_argument("--batch_completion_window", type=str, default=DEFAULT_BATCH_COMPLETION_WINDOW)
    parser.add_argument("--batch_poll_interval", type=int, default=DEFAULT_BATCH_POLL_INTERVAL)
    parser.add_argument("--mode_suffix", type=str, default="verified_LongCoT_stage")
    parser.add_argument("--postprocess_after_verify", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.disable_json_mode:
        args.use_json_mode = False
    args.enable_template_gate = not args.disable_template_gate
    ensure_transformers_cache_dir()
    api_key = resolve_api_key(args.api_key)

    with open(args.data_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    for idx, item in enumerate(raw_data, start=1):
        item["process_id"] = idx
    data = filter_input_data(raw_data)
    if args.limit_num:
        data = data[: args.limit_num]

    task_name, save_dir = build_save_dir(args.data_path, args.mode_suffix)
    ensure_dir(save_dir)
    processed_successes = merge_saved_verified_files(save_dir)
    input_data = deduplicate_input(data, processed_successes)

    generator_client = RemoteChatClient(
        model_name=args.generator_model_name,
        api_url=args.api_url,
        api_key=api_key,
        http_pool_size=args.http_pool_size,
        request_timeout=args.request_timeout,
        connect_timeout=args.connect_timeout,
    )
    summary_client = RemoteChatClient(
        model_name=args.summary_model_name,
        api_url=args.api_url,
        api_key=api_key,
        http_pool_size=args.http_pool_size,
        request_timeout=args.request_timeout,
        connect_timeout=args.connect_timeout,
    )
    if args.judge_backend == "local":
        judge_client: Any = LocalJudgeClient(
            model_path=args.local_judge_model_path,
            dtype=args.local_judge_dtype,
            max_gpu_memory_gib=args.local_judge_max_gpu_memory_gib,
            max_cpu_memory_gib=args.local_judge_max_cpu_memory_gib,
        )
    else:
        judge_client = RemoteChatClient(
            model_name=args.small_judge_model_name,
            api_url=args.api_url,
            api_key=api_key,
            http_pool_size=args.http_pool_size,
            request_timeout=args.request_timeout,
            connect_timeout=args.connect_timeout,
        )
    batch_client = None
    if args.generator_backend == "batch":
        batch_client = SiliconFlowBatchClient(
            api_key=api_key,
            base_url=args.base_url,
            request_timeout=args.request_timeout,
            connect_timeout=args.connect_timeout,
        )

    print(f"Original data size: {len(raw_data)}, Filtered data size: {len(data)}")
    print(f"read data: {len(data)}")
    print(f"Previously processed verified Long_CoT items: {len(processed_successes)}")
    print(f"Items remaining for processing: {len(input_data)}")
    print(f"Generator backend: {args.generator_backend}")
    print(f"Judge backend: {args.judge_backend}")
    if args.judge_backend == "local":
        print(f"Local judge batch size: {args.local_judge_batch_size}")
    print(f"Template gate enabled: {args.enable_template_gate}")
    print(f"Save dir: {save_dir}")

    started_at = time.time()
    run_verified_long_cot_pipeline(
        input_data,
        save_dir=save_dir,
        generator_client=generator_client,
        judge_client=judge_client,
        summary_client=summary_client,
        batch_client=batch_client,
        args=args,
    )
    final_data = merge_saved_verified_files(save_dir)
    status_counter: Dict[str, int] = {}
    for file_name in sorted(os.listdir(save_dir)):
        if not file_name.endswith(".json"):
            continue
        try:
            with open(os.path.join(save_dir, file_name), encoding="utf-8") as f:
                payload = json.load(f)
            status = payload.get("sample_status", "missing")
            status_counter[status] = status_counter.get(status, 0) + 1
        except Exception:
            status_counter["bad_json"] = status_counter.get("bad_json", 0) + 1
    output_path = f"{task_name}_{len(final_data)}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    elapsed = time.time() - started_at
    print(f"Processed {len(final_data)} verified Long_CoT items. Saving to {output_path}")
    print(f"Status counts: {status_counter}")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
