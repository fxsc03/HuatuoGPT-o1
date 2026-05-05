"""
Medical Reward Manager for HuatuoGPT-o1 GRPO training.

Adapted from ppo_utils/ppo_trainer_medo1.py :: get_reward_o1().

Scoring logic (same as PPO stage):
  - If response lacks the required "## Thinking / ## Final Response" structure → 0.0
  - If verifier probability P(True) > 0.4 → 1.0
  - Otherwise                             → 0.1

The MedicalRewardManager class follows verl's reward-manager contract:
    __call__(data: DataProto, return_dict=False) → DataProto
"""

import json
import random
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Verifier prompt template (identical to PPO trainer)
# ---------------------------------------------------------------------------
_VERIFIER_PROMPT = (
    "<Model Response>\n"
    "{}\n"
    "</Model Response>\n\n"
    "<Reference Answer>\n"
    "{}\n"
    "</Reference Answer>\n\n"
    "Your task is to evaluate the model response by comparing it to the reference "
    "answer. If the model response is correct and aligns with the reference answer, "
    'output "True" . If it is incorrect or fails to select the correct option (if '
    'options are provided), output "False" . {}'
)

_OUTPUT_PATTERN = re.compile(r"## Final Response\n\n(.*)", re.S)

# Running reward statistics for monitoring
_accumulate_rewards: List[float] = []


# ---------------------------------------------------------------------------
# Reward model singleton cache (one instance per (path, device) pair)
# ---------------------------------------------------------------------------
_model_cache: Dict[str, Tuple] = {}


def _get_reward_model(path: str, device: torch.device):
    key = f"{path}_{device}"
    if key not in _model_cache:
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=2,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        tok = AutoTokenizer.from_pretrained(path)
        _model_cache[key] = (model, tok)
    return _model_cache[key]


# ---------------------------------------------------------------------------
# Core batch reward computation
# ---------------------------------------------------------------------------

def compute_reward_batch(
    responses: List[str],
    ground_truths: List[str],
    reward_model_path: str,
    device: torch.device,
    max_length: int = 4000,
) -> torch.Tensor:
    """
    Score a batch of model responses against ground-truth answers.

    Args:
        responses:         Decoded response strings from the policy.
        ground_truths:     Reference answers for each response.
        reward_model_path: Path to the 3B classifier verifier.
        device:            Target device.
        max_length:        Max token length for verifier input.

    Returns:
        rewards: Float32 tensor of shape (batch_size,).
                 Values are in {0.0, 0.1, 1.0}.
    """
    reward_model, reward_tokenizer = _get_reward_model(reward_model_path, device)

    processed_texts: List[str] = []
    format_ok: List[bool] = []

    for response, gt in zip(responses, ground_truths):
        # Structural validation: exactly one ## Thinking + one ## Final Response
        n_final    = response.count("## Final Response\n\n")
        n_thinking = response.count("## Thinking")
        ok = (n_final == 1 and n_thinking == 1)
        format_ok.append(ok)

        if ok:
            m = _OUTPUT_PATTERN.search(response)
            final_ans = m.group(1).strip() if m else "I do not know the answer."
        else:
            final_ans = "I do not know the answer."

        text = _VERIFIER_PROMPT.format(final_ans, gt, reward_tokenizer.eos_token)
        processed_texts.append(text)

    inputs = reward_tokenizer(
        processed_texts,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_length,
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        logits = reward_model(**inputs, return_dict=True).logits
        probs  = F.softmax(logits, dim=-1)   # (B, 2)

    rewards_list: List[float] = []
    for i, ok in enumerate(format_ok):
        if not ok:
            rewards_list.append(0.0)
        else:
            p_true = probs[i, 1].item()
            rewards_list.append(1.0 if p_true > 0.4 else 0.1)

    rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)

    # Running average monitoring
    global _accumulate_rewards
    _accumulate_rewards.append(rewards.mean().item())

    if random.random() < 0.05:
        for i in range(min(2, len(processed_texts))):
            print(f"[reward_input] {processed_texts[i][:300]}…", flush=True)
            print(f"[reward]       {rewards_list[i]}", flush=True)
        window = _accumulate_rewards[-50:]
        print(
            f"[avg_reward_50] {sum(window) / max(1, len(window)):.4f}",
            flush=True,
        )

    return rewards


# ---------------------------------------------------------------------------
# verl RewardManager interface
# ---------------------------------------------------------------------------

class MedicalRewardManager:
    """
    verl-compatible reward manager for HuatuoGPT-o1 GRPO training.

    verl instantiates this class once per worker and calls it with every
    rollout batch.  The manager:
      1. Decodes the generated response tokens.
      2. Reads ground-truth answers from non-tensor batch data.
      3. Scores each response with the medical verifier (3B classifier).
      4. Writes sparse token-level rewards to data.batch['token_level_scores'].

    Constructor arguments are forwarded from the YAML config under
    ``reward_model.reward_kwargs``.
    """

    def __init__(
        self,
        tokenizer,
        reward_model_path: str = "FreedomIntelligence/medical_o1_verifier_3B",
        max_reward_length:  int = 4000,
        num_examine:        int = 0,   # kept for verl API compatibility
    ):
        self.tokenizer         = tokenizer
        self.reward_model_path = reward_model_path
        self.max_reward_length = max_reward_length

    # ------------------------------------------------------------------

    def __call__(self, data, return_dict: bool = False):
        """
        Compute token-level rewards for a rollout batch.

        Args:
            data:        verl DataProto containing the rollout batch.
            return_dict: If True, also return a metrics dict.

        Returns:
            data         (with data.batch['token_level_scores'] populated)
            metrics dict (only when return_dict=True)
        """
        responses_ids: torch.Tensor = data.batch["responses"]  # (B, resp_len)
        device = responses_ids.device

        # Decode responses to strings
        responses: List[str] = self.tokenizer.batch_decode(
            responses_ids, skip_special_tokens=True
        )

        # Ground-truth answers are stored as JSON strings in non-tensor batch
        reward_model_meta: List[str] = data.non_tensor_batch.get(
            "reward_model", ["{}"] * len(responses)
        )
        ground_truths: List[str] = []
        for meta in reward_model_meta:
            if isinstance(meta, str):
                meta = json.loads(meta)
            ground_truths.append(meta.get("ground_truth", ""))

        # Scalar reward per response  (B,)
        scalar_rewards = compute_reward_batch(
            responses=responses,
            ground_truths=ground_truths,
            reward_model_path=self.reward_model_path,
            device=device,
            max_length=self.max_reward_length,
        )

        # Convert to token-level: place reward on the last non-pad token
        resp_len = responses_ids.shape[1]
        token_level_scores = torch.zeros(
            len(responses), resp_len, device=device, dtype=scalar_rewards.dtype
        )
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        for i, resp_ids in enumerate(responses_ids):
            non_pad = (resp_ids != pad_id).nonzero(as_tuple=True)[0]
            last_pos = non_pad[-1].item() if len(non_pad) > 0 else (resp_len - 1)
            token_level_scores[i, last_pos] = scalar_rewards[i]

        data.batch["token_level_scores"] = token_level_scores

        if return_dict:
            metrics = {
                "reward/mean": scalar_rewards.mean().item(),
                "reward/max":  scalar_rewards.max().item(),
                "reward/min":  scalar_rewards.min().item(),
                "reward/frac_correct":   (scalar_rewards == 1.0).float().mean().item(),
                "reward/frac_no_format": (scalar_rewards == 0.0).float().mean().item(),
            }
            return data, metrics

        return data
