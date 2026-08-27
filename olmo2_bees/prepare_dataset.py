from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import (
    BEES_UPSTREAM_COMMIT,
    DATASET_ID,
    MODEL_ID,
    assistant_completion,
    configure_workspace,
    package_versions,
    read_json,
    valid_prompt,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UltraFeedback for BeeS selection")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_preference(example: dict, index: int) -> dict:
    chosen = example["chosen"]
    rejected = example["rejected"]
    if not isinstance(chosen, list) or not isinstance(rejected, list) or len(chosen) < 2 or len(rejected) < 2:
        return {"valid_schema": False, "row_id": index}
    prompt = chosen[:-1]
    same_prompt = prompt == rejected[:-1]
    chosen_completion = [chosen[-1]]
    rejected_completion = [rejected[-1]]
    schema_ok = (
        same_prompt
        and valid_prompt(prompt)
        and assistant_completion(chosen_completion)
        and assistant_completion(rejected_completion)
    )
    return {
        "row_id": index,
        "prompt": prompt,
        "chosen": chosen_completion,
        "rejected": rejected_completion,
        "external_margin": float(example["score_chosen"] - example["score_rejected"]),
        "valid_schema": schema_ok,
    }


def token_lengths(example: dict, tokenizer) -> dict:
    prompt_ids = tokenizer.apply_chat_template(
        example["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    chosen_ids = tokenizer.apply_chat_template(
        example["prompt"] + example["chosen"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    rejected_ids = tokenizer.apply_chat_template(
        example["prompt"] + example["rejected"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    prefix_ok = (
        chosen_ids[: len(prompt_ids)] == prompt_ids
        and rejected_ids[: len(prompt_ids)] == prompt_ids
    )
    return {
        "prompt_tokens": len(prompt_ids),
        "chosen_tokens": len(chosen_ids),
        "rejected_tokens": len(rejected_ids),
        "max_pair_tokens": max(len(chosen_ids), len(rejected_ids)),
        "template_prefix_ok": prefix_ok,
    }


def main() -> None:
    args = parse_args()
    configure_workspace(args.workspace)

    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    args.output_root = args.output_root.resolve()
    prepared_path = args.output_root / "prepared"
    seed_path = args.output_root / "proxy_seed"
    metadata_path = args.output_root / "prepare_metadata.json"
    if args.force:
        for path in (prepared_path, seed_path):
            if path.exists():
                shutil.rmtree(path)
        metadata_path.unlink(missing_ok=True)
    if prepared_path.exists() and seed_path.exists() and metadata_path.exists():
        existing = read_json(metadata_path)
        expected = {
            "dataset_id": args.dataset_id,
            "model_id": args.model_id,
            "max_length": args.max_length,
            "proxy_seed_size": args.seed_size,
            "seed": args.seed,
        }
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if args.dataset_revision and existing.get("dataset_revision") != args.dataset_revision:
            mismatches["dataset_revision"] = (
                existing.get("dataset_revision"),
                args.dataset_revision,
            )
        if args.model_revision and existing.get("model_revision") != args.model_revision:
            mismatches["model_revision"] = (existing.get("model_revision"), args.model_revision)
        if mismatches:
            raise RuntimeError(f"Prepared data configuration mismatch: {mismatches}; use --force")
        print(f"Prepared data already exists at {args.output_root}; use --force to rebuild")
        return
    if prepared_path.exists() or seed_path.exists() or metadata_path.exists():
        print(f"Removing incomplete prepared-data outputs under {args.output_root}")
        for path in (prepared_path, seed_path):
            if path.exists():
                shutil.rmtree(path)
        metadata_path.unlink(missing_ok=True)

    api = HfApi()
    dataset_revision = args.dataset_revision or api.dataset_info(args.dataset_id).sha
    model_revision = args.model_revision or api.model_info(args.model_id).sha
    print(f"Dataset revision: {dataset_revision}")
    print(f"Model revision:   {model_revision}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=model_revision, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise RuntimeError("The OLMo SFT tokenizer must contain its official chat template")

    prepared_splits: dict[str, Dataset] = {}
    source_splits = {"train": "train_prefs", "test": "test_prefs"}
    source_counts: dict[str, int] = {}
    prepared_counts: dict[str, int] = {}
    for output_split, source_split in source_splits.items():
        print(f"Loading {args.dataset_id}:{source_split}")
        dataset = load_dataset(
            args.dataset_id,
            revision=dataset_revision,
            split=source_split,
        )
        source_counts[output_split] = len(dataset)
        dataset = dataset.map(
            normalize_preference,
            with_indices=True,
            desc=f"Normalizing {source_split}",
            num_proc=args.num_proc,
        )
        dataset = dataset.filter(
            lambda row: row["valid_schema"],
            desc=f"Checking {source_split} schemas",
            num_proc=args.num_proc,
        )
        dataset = dataset.map(
            token_lengths,
            fn_kwargs={"tokenizer": tokenizer},
            desc=f"Measuring OLMo token lengths for {source_split}",
            num_proc=args.num_proc,
        )
        dataset = dataset.filter(
            lambda row: (
                row["template_prefix_ok"]
                and row["prompt_tokens"] < args.max_length
                and row["max_pair_tokens"] <= args.max_length
                and row["chosen_tokens"] > row["prompt_tokens"]
                and row["rejected_tokens"] > row["prompt_tokens"]
            ),
            desc=f"Keeping lossless <= {args.max_length}-token pairs in {source_split}",
            num_proc=args.num_proc,
        )
        removable = ["messages", "valid_schema", "template_prefix_ok"]
        dataset = dataset.remove_columns([column for column in removable if column in dataset.column_names])
        prepared_counts[output_split] = len(dataset)
        prepared_splits[output_split] = dataset

    if len(prepared_splits["train"]) < args.seed_size:
        raise RuntimeError("Not enough valid rows for the requested BeeS proxy seed")
    prepared = DatasetDict(prepared_splits)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(prepared_path)

    seed_dataset = prepared["train"].shuffle(seed=args.seed).select(range(args.seed_size))
    seed_dataset.save_to_disk(seed_path)

    external_positive = sum(value > 0 for value in prepared["train"]["external_margin"])
    metadata = {
        "method": "BeeS (Bayesian Aggregation for Preference data Selection)",
        "bees_upstream_commit": BEES_UPSTREAM_COMMIT,
        "dataset_id": args.dataset_id,
        "dataset_revision": dataset_revision,
        "model_id": args.model_id,
        "model_revision": model_revision,
        "max_length": args.max_length,
        "lossless_length_filter": True,
        "seed": args.seed,
        "proxy_seed_size": args.seed_size,
        "source_counts": source_counts,
        "prepared_counts": prepared_counts,
        "positive_external_margin_count": external_positive,
        "packages": package_versions(
            ["torch", "transformers", "datasets", "huggingface-hub", "tqdm"]
        ),
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
