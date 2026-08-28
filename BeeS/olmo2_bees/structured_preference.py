from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REQUESTED_VARIANTS = ("A", "B-DPO", "B-VDPO", "C-DPO", "C-VDPO")
SUPPORTED_VARIANTS = ("DPO", *REQUESTED_VARIANTS)

VARIANT_DESCRIPTIONS = {
    "DPO": "Optional response-level DPO control (not part of the five requested structured runs).",
    "A": (
        "Score-gap-weighted Plackett-Luce log-ratio structural utility only; "
        "there is no response-level DPO term."
    ),
    "B-DPO": (
        "Standard DPO response core plus an independent Plackett-Luce structural "
        "coherence term over score-weighted segment log-ratios."
    ),
    "B-VDPO": (
        "VDPO score-weighted segment-density core plus the Method B structural "
        "coherence term."
    ),
    "C-DPO": (
        "Standard DPO response core plus Method A's score-gap-weighted, "
        "reference-baselined Plackett-Luce utility."
    ),
    "C-VDPO": (
        "VDPO score-weighted segment-density core plus Method A's score-gap-weighted, "
        "reference-baselined Plackett-Luce utility."
    ),
}

VARIANT_FORMULAS = {
    "DPO": "O(y) = beta * sum_s h_s",
    "A": "O(y) = weighted_PL(beta * ell_policy, beta * ell_ref, score_gaps)",
    "B-DPO": "O(y) = beta * sum_s h_s + PL(score_s * beta * h_s)",
    "B-VDPO": (
        "O(y) = beta * sum_s(score_s * h_s / length_s) / sum_s(score_s) "
        "+ PL(score_s * beta * h_s)"
    ),
    "C-DPO": (
        "O(y) = beta * sum_s h_s "
        "+ weighted_PL(beta * ell_policy, beta * ell_ref, score_gaps)"
    ),
    "C-VDPO": (
        "O(y) = beta * sum_s(score_s * h_s / length_s) / sum_s(score_s) "
        "+ weighted_PL(beta * ell_policy, beta * ell_ref, score_gaps)"
    ),
}


def _non_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _validate_segments(response: str, segments: list[dict[str, Any]]) -> tuple[list[float], list[int], list[int]]:
    if not isinstance(segments, list) or not segments:
        raise ValueError("A segmented response must contain at least one segment")

    texts: list[str] = []
    scores: list[float] = []
    ranks: list[int] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TypeError(f"Segment {index} is not a mapping")
        text = segment.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Segment {index} has no text")
        score = float(segment.get("value_score"))
        rank = int(segment.get("rank"))
        if not math.isfinite(score) or not 0.01 <= score <= 1.0:
            raise ValueError(f"Segment {index} has invalid score {score}")
        texts.append(text)
        scores.append(score)
        ranks.append(rank)

    expected_ranks = list(range(1, len(segments) + 1))
    if sorted(ranks) != expected_ranks:
        raise ValueError(f"Ranks must be exactly 1..N, got {sorted(ranks)}")
    if len(set(scores)) != len(scores):
        raise ValueError("Segment scores must be unique")
    rank_order = sorted(range(len(ranks)), key=ranks.__getitem__)
    ordered_scores = [scores[index] for index in rank_order]
    if any(left <= right for left, right in zip(ordered_scores, ordered_scores[1:])):
        raise ValueError("Scores must decrease strictly as rank worsens")

    joined = "".join(texts)
    exact = joined == response
    if not exact and _non_whitespace(joined) != _non_whitespace(response):
        raise ValueError("Segments do not reconstruct the response, even after whitespace normalization")

    # Assign every response character to a source segment. The final dataset has
    # only 23 whitespace-normalized (rather than byte-exact) sides. For those
    # sides, non-whitespace characters provide an unambiguous monotonic alignment;
    # whitespace adopts the next segment at boundaries.
    if exact:
        char_segment_ids: list[int] = []
        for segment_id, text in enumerate(texts):
            char_segment_ids.extend([segment_id] * len(text))
    else:
        joined_ids: list[int] = []
        joined_nonspace: list[str] = []
        for segment_id, text in enumerate(texts):
            for character in text:
                if not character.isspace():
                    joined_nonspace.append(character)
                    joined_ids.append(segment_id)
        response_nonspace = [character for character in response if not character.isspace()]
        if response_nonspace != joined_nonspace:
            raise ValueError("Whitespace-normalized character alignment failed")
        char_segment_ids = [-1] * len(response)
        ordinal = 0
        for position, character in enumerate(response):
            if not character.isspace():
                char_segment_ids[position] = joined_ids[ordinal]
                ordinal += 1
        next_id = -1
        for position in range(len(response) - 1, -1, -1):
            if char_segment_ids[position] >= 0:
                next_id = char_segment_ids[position]
            elif next_id >= 0:
                char_segment_ids[position] = next_id
        previous_id = 0
        for position in range(len(response)):
            if char_segment_ids[position] >= 0:
                previous_id = char_segment_ids[position]
            else:
                char_segment_ids[position] = previous_id

    if len(char_segment_ids) != len(response) or any(index < 0 for index in char_segment_ids):
        raise RuntimeError("Incomplete response character-to-segment alignment")
    return scores, ranks, char_segment_ids


def encode_segmented_side(
    tokenizer: Any,
    prompt: list[dict[str, str]],
    completion: list[dict[str, str]],
    segments: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    if (
        not isinstance(completion, list)
        or len(completion) != 1
        or completion[0].get("role") != "assistant"
        or not isinstance(completion[0].get("content"), str)
    ):
        raise ValueError("Completion must be one assistant message")
    response = completion[0]["content"]
    scores, ranks, response_char_segment_ids = _validate_segments(response, segments)

    prompt_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("Full chat rendering does not begin with the generation prompt")
    response_start = len(prompt_text)
    response_end = response_start + len(response)
    if full_text[response_start:response_end] != response:
        raise ValueError("Assistant content is not contiguous in the rendered chat")

    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    canonical_ids = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if input_ids != canonical_ids:
        raise ValueError("Offset-bearing tokenization differs from apply_chat_template")
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenized completion does not preserve the prompt prefix")
    if len(input_ids) > max_length:
        raise ValueError(f"Lossless preparation requires <= {max_length} tokens, got {len(input_ids)}")

    token_segment_ids = [-1] * len(input_ids)
    token_segment_candidates: dict[int, Counter[int]] = {}
    boundary_crossing_tokens = 0
    for token_index, (token_start, token_end) in enumerate(offsets):
        overlap_start = max(token_start, response_start)
        overlap_end = min(token_end, response_end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - response_start
        local_end = overlap_end - response_start
        overlapping_ids = response_char_segment_ids[local_start:local_end]
        counts = Counter(overlapping_ids)
        token_segment_candidates[token_index] = counts
        # Prefer the later segment on an exact tie, matching causal ownership of
        # a boundary-spanning token by the content it finishes predicting.
        segment_id = max(counts, key=lambda item: (counts[item], item))
        token_segment_ids[token_index] = segment_id
        if len(counts) > 1:
            boundary_crossing_tokens += 1

    covered = set(index for index in token_segment_ids if index >= 0)
    expected = set(range(len(segments)))
    # A one-character segment can share a single BPE token with its neighbor.
    # Reassign that boundary token only when its current owner has another token,
    # so every ranked segment remains represented without duplicating log-probs.
    for missing_segment in sorted(expected - covered):
        owner_counts = Counter(index for index in token_segment_ids if index >= 0)
        candidates = [
            (counts[missing_segment], token_index, token_segment_ids[token_index])
            for token_index, counts in token_segment_candidates.items()
            if missing_segment in counts
            and owner_counts[token_segment_ids[token_index]] > 1
        ]
        if candidates:
            _, token_index, _ = max(candidates)
            token_segment_ids[token_index] = missing_segment
            covered.add(missing_segment)
    if covered != expected:
        raise ValueError(
            f"Every segment must own at least one token; missing={sorted(expected - covered)}"
        )
    content_token_count = sum(index >= 0 for index in token_segment_ids)
    if content_token_count <= 0:
        raise ValueError("Completion contains no aligned content tokens")

    return {
        "input_ids": input_ids,
        "segment_ids": token_segment_ids,
        "segment_scores": scores,
        "segment_ranks": ranks,
        "prompt_tokens": len(prompt_ids),
        "content_tokens": content_token_count,
        "boundary_crossing_tokens": boundary_crossing_tokens,
        "segments_exact": "".join(segment["text"] for segment in segments) == response,
    }


def prepare_segmented_dataset(
    source_jsonl: str | Path,
    output_dir: str | Path,
    model_id: str,
    model_revision: str | None,
    max_length: int = 1024,
    force: bool = False,
) -> dict[str, Any]:
    from datasets import Dataset, DatasetDict
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    from .common import package_versions, sha256_file, write_json

    source_jsonl = Path(source_jsonl).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir.parent / f"{output_dir.name}.manifest.json"
    source_sha256 = sha256_file(source_jsonl)
    expected_identity = {
        "source_jsonl": str(source_jsonl),
        "source_sha256": source_sha256,
        "model_id": model_id,
        "model_revision": model_revision,
        "max_length": max_length,
    }
    if output_dir.exists() and manifest_path.is_file() and not force:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (current.get(key), value)
            for key, value in expected_identity.items()
            if current.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Prepared dataset identity mismatch: {mismatches}; use --force")
        print(f"Prepared segmented dataset already exists: {output_dir}")
        return current
    if (output_dir.exists() or manifest_path.exists()) and not force:
        raise RuntimeError(f"Incomplete prepared dataset at {output_dir}; use --force to rebuild")
    if force:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        manifest_path.unlink(missing_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise RuntimeError("OLMo preparation requires its fast tokenizer and official chat template")

    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    stats = {
        "rows": 0,
        "exact_segment_sides": 0,
        "whitespace_normalized_segment_sides": 0,
        "boundary_crossing_tokens": 0,
        "max_pair_tokens": 0,
        "max_segments_per_side": 0,
    }
    with source_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(tqdm(handle, desc="Tokenizing segmented OLMo pairs"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split")
            if split not in records_by_split:
                raise ValueError(f"Unsupported split at line {line_number}: {split!r}")
            prompt = row.get("prompt")
            if (
                not isinstance(prompt, list)
                or not prompt
                or prompt[-1].get("role") != "user"
            ):
                raise ValueError(f"Invalid prompt at line {line_number}")
            chosen = encode_segmented_side(
                tokenizer, prompt, row["chosen"], row["chosen_segments"], max_length
            )
            rejected = encode_segmented_side(
                tokenizer, prompt, row["rejected"], row["rejected_segments"], max_length
            )
            if "chosen_tokens" in row and len(chosen["input_ids"]) != int(row["chosen_tokens"]):
                raise ValueError(f"Chosen token count drift at line {line_number}")
            if "rejected_tokens" in row and len(rejected["input_ids"]) != int(row["rejected_tokens"]):
                raise ValueError(f"Rejected token count drift at line {line_number}")

            record = {
                "dataset_index": len(records_by_split[split]),
                "row_id": int(row["row_id"]),
                "source_index": int(row["source_index"]),
                "chosen_input_ids": chosen["input_ids"],
                "chosen_segment_ids": chosen["segment_ids"],
                "chosen_segment_scores": chosen["segment_scores"],
                "chosen_segment_ranks": chosen["segment_ranks"],
                "rejected_input_ids": rejected["input_ids"],
                "rejected_segment_ids": rejected["segment_ids"],
                "rejected_segment_scores": rejected["segment_scores"],
                "rejected_segment_ranks": rejected["segment_ranks"],
            }
            records_by_split[split].append(record)
            stats["rows"] += 1
            stats["exact_segment_sides"] += int(chosen["segments_exact"]) + int(
                rejected["segments_exact"]
            )
            stats["whitespace_normalized_segment_sides"] += int(
                not chosen["segments_exact"]
            ) + int(not rejected["segments_exact"])
            stats["boundary_crossing_tokens"] += chosen["boundary_crossing_tokens"] + rejected[
                "boundary_crossing_tokens"
            ]
            stats["max_pair_tokens"] = max(
                stats["max_pair_tokens"], len(chosen["input_ids"]), len(rejected["input_ids"])
            )
            stats["max_segments_per_side"] = max(
                stats["max_segments_per_side"],
                len(chosen["segment_scores"]),
                len(rejected["segment_scores"]),
            )

    if not records_by_split["train"]:
        raise RuntimeError("The segmented source contains no training rows")
    prepared = DatasetDict(
        {split: Dataset.from_list(records) for split, records in records_by_split.items() if records}
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temp:
        staged = Path(temp) / output_dir.name
        prepared.save_to_disk(staged)
        staged.replace(output_dir)

    manifest = {
        **expected_identity,
        "lossless": True,
        "split_counts": {split: len(records) for split, records in records_by_split.items()},
        "statistics": stats,
        "columns": prepared["train"].column_names,
        "requested_variants": list(REQUESTED_VARIANTS),
        "packages": package_versions(
            ["torch", "transformers", "datasets", "accelerate", "trl", "bitsandbytes"]
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def pad_1d(sequences: Iterable[Any], padding_value: int | float, dtype: Any):
    import torch

    tensors = [torch.as_tensor(sequence, dtype=dtype) for sequence in sequences]
    if not tensors:
        raise ValueError("Cannot pad an empty batch")
    width = max(tensor.numel() for tensor in tensors)
    output = torch.full((len(tensors), width), padding_value, dtype=dtype)
    for row, tensor in enumerate(tensors):
        output[row, : tensor.numel()] = tensor
    return output


class StructuredPreferenceCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        sides = ["chosen", "rejected"]
        input_ids = [example[f"{side}_input_ids"] for side in sides for example in examples]
        segment_ids = [example[f"{side}_segment_ids"] for side in sides for example in examples]
        segment_scores = [
            example[f"{side}_segment_scores"] for side in sides for example in examples
        ]
        segment_ranks = [
            example[f"{side}_segment_ranks"] for side in sides for example in examples
        ]
        reference_segment_logps = [
            example[f"ref_{side}_segment_logps"] for side in sides for example in examples
        ]

        padded_ids = pad_1d(input_ids, self.pad_token_id, torch.long)
        padded_segment_ids = pad_1d(segment_ids, -1, torch.long)
        padded_scores = pad_1d(segment_scores, 0.0, torch.float32)
        padded_ranks = pad_1d(segment_ranks, -1, torch.long)
        padded_reference = pad_1d(reference_segment_logps, 0.0, torch.float32)
        segment_mask = pad_1d(
            [[True] * len(sequence) for sequence in segment_scores], False, torch.bool
        )
        if padded_scores.shape != padded_ranks.shape or padded_scores.shape != padded_reference.shape:
            raise RuntimeError("Segment score/rank/reference shapes differ")
        return {
            "input_ids": padded_ids,
            "attention_mask": padded_ids.ne(self.pad_token_id).long(),
            "token_segment_ids": padded_segment_ids,
            "segment_scores": padded_scores,
            "segment_ranks": padded_ranks,
            "reference_segment_logps": padded_reference,
            "segment_mask": segment_mask,
        }


def aggregate_segment_logps(per_token_logps, token_segment_ids, segment_count: int):
    import torch

    # per_token_logps predicts input_ids[:, 1:], so ownership metadata shifts too.
    shifted_segment_ids = token_segment_ids[:, 1:]
    valid = shifted_segment_ids.ge(0)
    safe_ids = shifted_segment_ids.clamp_min(0)
    segment_logps = torch.zeros(
        (per_token_logps.shape[0], segment_count),
        dtype=per_token_logps.dtype,
        device=per_token_logps.device,
    )
    segment_lengths = torch.zeros_like(segment_logps)
    segment_logps.scatter_add_(1, safe_ids, per_token_logps * valid)
    segment_lengths.scatter_add_(1, safe_ids, valid.to(per_token_logps.dtype))
    return segment_logps, segment_lengths


def pl_step_log_terms_ordered(utilities):
    import torch

    if utilities.numel() <= 1:
        return utilities.new_empty((0,))
    return torch.stack(
        [
            utilities[start] - torch.logsumexp(utilities[start:], dim=0)
            for start in range(utilities.numel() - 1)
        ]
    )


def pl_log_prob_ordered(utilities):
    terms = pl_step_log_terms_ordered(utilities)
    return terms.sum() if terms.numel() else utilities.sum() * 0.0


def omega_gap_weights_ordered(scores, eps: float = 1e-8):
    import torch

    if scores.numel() <= 1:
        return scores.new_empty((0,))
    omega = torch.stack(
        [
            (scores[start] - scores[start + 1 :].mean()).clamp_min(0.0)
            for start in range(scores.numel() - 1)
        ]
    )
    if omega.detach().sum() <= eps:
        omega = torch.ones_like(omega)
    return omega


def omega_weighted_pl_log_ratio_ordered(
    policy_utilities,
    reference_utilities,
    scores,
    eps: float = 1e-8,
):
    policy_terms = pl_step_log_terms_ordered(policy_utilities)
    reference_terms = pl_step_log_terms_ordered(reference_utilities)
    if policy_terms.numel() == 0:
        return policy_utilities.sum() * 0.0
    delta = policy_terms - reference_terms
    omega = omega_gap_weights_ordered(scores.to(delta.dtype), eps=eps)
    return (omega * delta).sum() / omega.sum().clamp_min(eps)


def response_terms(
    policy_segment_logps,
    reference_segment_logps,
    segment_scores,
    segment_ranks,
    segment_lengths,
    beta: float,
    eps: float = 1e-8,
):
    import torch

    order = torch.argsort(segment_ranks, stable=True)
    ell_policy = policy_segment_logps[order]
    ell_reference = reference_segment_logps[order].to(ell_policy.dtype)
    scores = segment_scores[order].to(ell_policy.dtype)
    lengths = segment_lengths[order].to(ell_policy.dtype).clamp_min(1.0)
    h = ell_policy - ell_reference

    standard_core = beta * h.sum()
    vdpo_core = beta * (scores * (h / lengths)).sum() / scores.sum().clamp_min(eps)
    method_a_omega_utility = omega_weighted_pl_log_ratio_ordered(
        beta * ell_policy,
        beta * ell_reference,
        scores,
        eps=eps,
    )
    method_b_log_sc = pl_log_prob_ordered(scores * beta * h)
    return {
        "standard_core": standard_core,
        "vdpo_core": vdpo_core,
        "method_a_omega_utility": method_a_omega_utility,
        "method_b_log_sc": method_b_log_sc,
    }


def response_objective(
    policy_segment_logps,
    reference_segment_logps,
    segment_scores,
    segment_ranks,
    segment_lengths,
    variant: str,
    beta: float,
):
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant {variant!r}; choose from {SUPPORTED_VARIANTS}")
    terms = response_terms(
        policy_segment_logps,
        reference_segment_logps,
        segment_scores,
        segment_ranks,
        segment_lengths,
        beta,
    )
    if variant == "DPO":
        return terms["standard_core"]
    if variant == "A":
        return terms["method_a_omega_utility"]
    if variant == "B-DPO":
        return terms["standard_core"] + terms["method_b_log_sc"]
    if variant == "B-VDPO":
        return terms["vdpo_core"] + terms["method_b_log_sc"]
    if variant == "C-DPO":
        return terms["standard_core"] + terms["method_a_omega_utility"]
    if variant == "C-VDPO":
        return terms["vdpo_core"] + terms["method_a_omega_utility"]
    raise AssertionError(variant)


def structured_pair_loss(
    policy_segment_logps,
    reference_segment_logps,
    segment_scores,
    segment_ranks,
    segment_lengths,
    segment_mask,
    variant: str,
    beta: float,
):
    import torch
    import torch.nn.functional as functional

    response_count = policy_segment_logps.shape[0]
    if response_count % 2:
        raise ValueError("The response batch must contain chosen rows followed by rejected rows")
    objectives = []
    for row in range(response_count):
        valid = segment_mask[row]
        if not torch.any(valid):
            raise ValueError("Every response must contain at least one valid segment")
        objectives.append(
            response_objective(
                policy_segment_logps[row, valid],
                reference_segment_logps[row, valid],
                segment_scores[row, valid],
                segment_ranks[row, valid],
                segment_lengths[row, valid],
                variant=variant,
                beta=beta,
            )
        )
    objectives = torch.stack(objectives)
    pair_count = response_count // 2
    chosen_objectives = objectives[:pair_count]
    rejected_objectives = objectives[pair_count:]
    margins = chosen_objectives - rejected_objectives
    loss = -functional.logsigmoid(margins).mean()
    return loss, chosen_objectives, rejected_objectives, margins


def run_loss_self_tests() -> dict[str, Any]:
    import torch

    reference = torch.tensor(
        [[-2.0, -4.0, -3.0], [-3.2, -1.5, -2.4]], dtype=torch.float32
    )
    policy = (reference + torch.tensor(
        [[0.3, -0.1, 0.1], [-0.2, 0.05, -0.3]], dtype=torch.float32
    )).requires_grad_(True)
    scores = torch.tensor([[0.9, 0.2, 0.6], [0.5, 0.95, 0.1]], dtype=torch.float32)
    ranks = torch.tensor([[1, 3, 2], [2, 1, 3]], dtype=torch.long)
    lengths = torch.tensor([[2.0, 4.0, 3.0], [3.0, 2.0, 5.0]], dtype=torch.float32)
    mask = torch.ones_like(scores, dtype=torch.bool)

    results: dict[str, Any] = {}
    total = policy.sum() * 0.0
    for variant in SUPPORTED_VARIANTS:
        loss, chosen, rejected, margin = structured_pair_loss(
            policy,
            reference,
            scores,
            ranks,
            lengths,
            mask,
            variant=variant,
            beta=0.1,
        )
        if not all(torch.isfinite(value).all() for value in (loss, chosen, rejected, margin)):
            raise AssertionError(f"Non-finite output for {variant}")
        total = total + loss
        results[variant] = {
            "loss": float(loss.detach()),
            "margin": float(margin.detach().mean()),
        }
    total.backward()
    if policy.grad is None or not torch.isfinite(policy.grad).all() or policy.grad.abs().sum() == 0:
        raise AssertionError("Structured losses did not produce finite non-zero gradients")

    for variant in ("DPO", "A", "C-DPO", "C-VDPO"):
        loss, chosen, rejected, margin = structured_pair_loss(
            reference,
            reference,
            scores,
            ranks,
            lengths,
            mask,
            variant=variant,
            beta=0.1,
        )
        if variant in {"DPO", "A", "C-DPO", "C-VDPO"} and not torch.allclose(
            margin, torch.zeros_like(margin), atol=1e-6
        ):
            raise AssertionError(f"{variant} should recover a zero margin at the reference policy")
        if not torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-6):
            raise AssertionError(f"{variant} should recover log(2) at a zero margin")
    return results
