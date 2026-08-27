from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .common import configure_workspace, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score BeeS implicit reward margins on two GPUs")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--reference-revision", default=None)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--flush-every", type=int, default=25)
    return parser.parse_args()


def encode_pair(tokenizer, prompt: list[dict[str, str]], completion: list[dict[str, str]], max_length: int):
    prompt_ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(
        prompt + completion, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("The full conversation does not begin with the chat-templated prompt")
    if len(full_ids) > max_length:
        raise ValueError(f"Lossless scoring requires <= {max_length} tokens, got {len(full_ids)}")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Completion contains no loss-bearing tokens")
    return full_ids, len(prompt_ids)


def collate_preference(tokenizer, row: dict[str, Any], max_length: int, device):
    import torch

    tokenized_schema = all(
        key in row
        for key in (
            "chosen_input_ids",
            "chosen_segment_ids",
            "rejected_input_ids",
            "rejected_segment_ids",
        )
    )
    if tokenized_schema:
        encoded = []
        for side in ("chosen", "rejected"):
            ids = list(row[f"{side}_input_ids"])
            segment_ids = list(row[f"{side}_segment_ids"])
            if len(ids) != len(segment_ids):
                raise ValueError(f"{side.title()} token/segment lengths differ")
            if len(ids) > max_length:
                raise ValueError(
                    f"Lossless scoring requires <= {max_length} tokens, got {len(ids)}"
                )
            completion_mask = [segment_id >= 0 for segment_id in segment_ids]
            if not any(completion_mask):
                raise ValueError(f"{side.title()} response contains no loss-bearing tokens")
            encoded.append((ids, completion_mask))
    else:
        raw_required = {"prompt", "chosen", "rejected"}
        missing = raw_required.difference(row)
        if missing:
            raise ValueError(
                "Preference row has neither the conversational nor segmented-token schema; "
                f"missing raw columns={sorted(missing)}"
            )
        encoded = []
        for completion in (row["chosen"], row["rejected"]):
            ids, prompt_length = encode_pair(
                tokenizer, row["prompt"], completion, max_length
            )
            encoded.append((ids, [index >= prompt_length for index in range(len(ids))]))

    max_tokens = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full(
        (2, max_tokens), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((2, max_tokens), dtype=torch.long, device=device)
    labels = torch.full((2, max_tokens), -100, dtype=torch.long, device=device)
    completion_lengths: list[int] = []
    for index, (ids, completion_mask) in enumerate(encoded):
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        mask = torch.tensor(completion_mask, dtype=torch.bool, device=device)
        labels[index, :length][mask] = input_ids[index, :length][mask]
        completion_lengths.append(int(mask.sum().item()))
    return input_ids, attention_mask, labels, completion_lengths


def sequence_logps(model, input_ids, attention_mask, labels):
    import torch
    import torch.nn.functional as functional

    with torch.inference_mode():
        # Materializing [chosen/rejected, sequence, vocabulary] logits and then
        # making the shifted view contiguous needs multiple ~400 MiB allocations
        # at max_length=1024.  Both FP16 models already occupy most of the 8 GiB
        # scoring rank, so project a bounded number of positions at a time.
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state[:, :-1, :]
        shift_labels = labels[:, 1:].contiguous()
        loss_positions = shift_labels.ne(-100).nonzero(as_tuple=False)
        loss_hidden = hidden_states[loss_positions[:, 0], loss_positions[:, 1]]
        loss_labels = shift_labels[loss_positions[:, 0], loss_positions[:, 1]]
        sequence_totals = torch.zeros(input_ids.shape[0], dtype=torch.float32, device=input_ids.device)
        for start in range(0, loss_hidden.shape[0], 64):
            stop = start + 64
            logits = functional.linear(
                loss_hidden[start:stop],
                model.lm_head.weight,
                model.lm_head.bias,
            ).float()
            label_chunk = loss_labels[start:stop]
            selected_logits = torch.gather(logits, -1, label_chunk.unsqueeze(-1)).squeeze(-1)
            token_logps = selected_logits - torch.logsumexp(logits, dim=-1)
            sequence_totals.scatter_add_(0, loss_positions[start:stop, 0], token_logps)
        return sequence_totals


def load_completed(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["row_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid resume record at {path}:{line_number}") from error
    return completed


def merge_parts(output_dir: Path, expected_ids: set[int], world_size: int) -> Path:
    records: dict[int, dict[str, Any]] = {}
    for rank in range(world_size):
        path = output_dir / f"part-rank-{rank:02d}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                row_id = int(record["row_id"])
                if row_id in records:
                    raise RuntimeError(f"Duplicate implicit score for row_id={row_id}")
                records[row_id] = record
    actual_ids = set(records)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise RuntimeError(f"Implicit score coverage mismatch; missing={missing}, extra={extra}")
    merged = output_dir / "implicit_scores.jsonl"
    temporary = merged.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row_id in sorted(records):
            handle.write(json.dumps(records[row_id], sort_keys=True) + "\n")
    temporary.replace(merged)
    return merged


def main() -> None:
    args = parse_args()
    configure_workspace(args.workspace)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import torch
    from accelerate import Accelerator
    from datasets import Dataset, DatasetDict, load_from_disk
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerator = Accelerator(mixed_precision="fp16")
    if accelerator.num_processes != 2:
        raise RuntimeError("Implicit scoring must be launched with exactly two processes")
    device = accelerator.device

    loaded = load_from_disk(str(args.dataset_path.resolve()))
    if isinstance(loaded, DatasetDict):
        dataset = loaded[args.split]
    elif isinstance(loaded, Dataset):
        dataset = loaded
    else:
        raise TypeError(type(loaded))
    if "row_id" not in dataset.column_names:
        raise RuntimeError("Dataset must contain stable row_id values")
    raw_schema = {"prompt", "chosen", "rejected"}.issubset(dataset.column_names)
    segmented_schema = {
        "chosen_input_ids",
        "chosen_segment_ids",
        "rejected_input_ids",
        "rejected_segment_ids",
    }.issubset(dataset.column_names)
    if not raw_schema and not segmented_schema:
        raise RuntimeError(
            "Dataset must contain either conversational preference columns or segmented token IDs"
        )
    input_schema = "segmented_tokens" if segmented_schema else "conversational"

    tokenizer = AutoTokenizer.from_pretrained(
        args.reference_model,
        revision=args.reference_revision,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = {
        "dtype": torch.float16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    reference = AutoModelForCausalLM.from_pretrained(
        args.reference_model,
        revision=args.reference_revision,
        **load_kwargs,
    ).to(device)
    policy = AutoModelForCausalLM.from_pretrained(args.policy_model, **load_kwargs).to(device)
    for model in (reference, policy):
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            raise RuntimeError("Implicit scoring loaded quantized weights")
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = False

    shard = dataset.shard(
        num_shards=accelerator.num_processes,
        index=accelerator.process_index,
        contiguous=True,
    )
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_path = args.output_dir / f"part-rank-{accelerator.process_index:02d}.jsonl"
    completed = load_completed(part_path)
    expected_shard_ids = {int(value) for value in shard["row_id"]}
    unexpected = completed - expected_shard_ids
    if unexpected:
        raise RuntimeError(f"Resume file contains rows assigned to another rank: {sorted(unexpected)[:10]}")

    remaining_indices = [
        index for index, row_id in enumerate(shard["row_id"]) if int(row_id) not in completed
    ]
    progress = tqdm(
        remaining_indices,
        total=len(shard),
        initial=len(completed),
        desc=f"Implicit margins GPU/rank {accelerator.process_index}",
        position=accelerator.process_index,
        disable=not accelerator.is_local_main_process,
    )
    with part_path.open("a", encoding="utf-8", buffering=1) as handle:
        for completed_this_run, index in enumerate(progress, start=1):
            row = shard[index]
            input_ids, attention_mask, labels, completion_lengths = collate_preference(
                tokenizer, row, args.max_length, device
            )
            policy_logps = sequence_logps(policy, input_ids, attention_mask, labels)
            reference_logps = sequence_logps(reference, input_ids, attention_mask, labels)
            implicit_rewards = policy_logps - reference_logps
            record = {
                "row_id": int(row["row_id"]),
                "policy_chosen_logp": float(policy_logps[0].item()),
                "policy_rejected_logp": float(policy_logps[1].item()),
                "reference_chosen_logp": float(reference_logps[0].item()),
                "reference_rejected_logp": float(reference_logps[1].item()),
                "implicit_chosen_reward": float(implicit_rewards[0].item()),
                "implicit_rejected_reward": float(implicit_rewards[1].item()),
                "implicit_margin": float((implicit_rewards[0] - implicit_rewards[1]).item()),
                "chosen_completion_tokens": int(completion_lengths[0]),
                "rejected_completion_tokens": int(completion_lengths[1]),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            if completed_this_run % args.flush_every == 0:
                handle.flush()
                os.fsync(handle.fileno())
            progress.set_postfix(margin=f"{record['implicit_margin']:.3f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged = merge_parts(
            args.output_dir,
            {int(value) for value in dataset["row_id"]},
            accelerator.num_processes,
        )
        write_json(
            args.output_dir / "score_manifest.json",
            {
                "dataset_path": str(args.dataset_path.resolve()),
                "split": args.split,
                "reference_model": args.reference_model,
                "reference_revision": args.reference_revision,
                "policy_model": args.policy_model,
                "max_length": args.max_length,
                "world_size": accelerator.num_processes,
                "rows": len(dataset),
                "input_schema": input_schema,
                "merged_scores": str(merged),
                "weight_dtype": "float16",
                "weight_quantization": None,
            },
        )
        print(f"Merged {len(dataset):,} scores into {merged}")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
