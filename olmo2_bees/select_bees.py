from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from .common import BEES_UPSTREAM_COMMIT, configure_workspace, read_json, write_json


LOWER_BOUND = -2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply faithful BeeS Bayesian aggregation")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--prepared-path", type=Path, required=True)
    parser.add_argument("--scores-path", type=Path, required=True)
    parser.add_argument("--prepare-metadata", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--selection-size", type=int, default=6000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dynamic_upper(values: np.ndarray, source: str) -> float:
    """Match the released BeeS search and the two Appendix A.1 stopping conditions."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"No finite {source} margins")
    maximum = float(finite.max())
    divisor = 2.0 if source == "implicit" else 1.5
    upper = float(int(maximum / divisor))
    upper = max(upper, LOWER_BOUND + 1.0)
    while upper <= maximum:
        tail_count = int(np.sum((finite >= upper) & (finite <= maximum)))
        if tail_count < 30 or tail_count < (maximum - upper):
            break
        upper += 1.0
    if upper > maximum:
        upper = maximum
    if upper <= LOWER_BOUND:
        raise ValueError(f"Invalid BeeS bounds for {source}: L={LOWER_BOUND}, U={upper}")
    return upper


def project_margin(values: np.ndarray, upper: float) -> np.ndarray:
    return (np.clip(values, LOWER_BOUND, upper) - LOWER_BOUND) / (upper - LOWER_BOUND)


def bayesian_aggregate(probabilities: list[np.ndarray]) -> np.ndarray:
    """Numerically stable Eq. (3): product(p) / (product(p) + product(1-p))."""
    epsilon = np.finfo(np.float64).eps
    log_odds = np.zeros_like(probabilities[0], dtype=np.float64)
    for probability in probabilities:
        safe = np.clip(probability.astype(np.float64), epsilon, 1.0 - epsilon)
        log_odds += np.log(safe) - np.log1p(-safe)
    positive = log_odds >= 0
    result = np.empty_like(log_odds)
    result[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
    exp_values = np.exp(log_odds[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def read_scores(path: Path) -> dict[int, dict]:
    scores: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            row_id = int(record["row_id"])
            if row_id in scores:
                raise RuntimeError(f"Duplicate row_id={row_id} at line {line_number}")
            scores[row_id] = record
    return scores


def main() -> None:
    args = parse_args()
    configure_workspace(args.workspace)

    from datasets import DatasetDict, load_from_disk

    args.output_path = args.output_path.resolve()
    if args.output_path.exists():
        complete_output = (
            (args.output_path / "metadata.json").is_file()
            and (args.output_path / "dataset_dict.json").is_file()
            and (args.output_path / "train.parquet").is_file()
            and (args.output_path / "test.parquet").is_file()
        )
        if not args.force and complete_output:
            print(f"Selected dataset already exists at {args.output_path}; use --force to rebuild")
            return
        if not args.force:
            print(f"Removing incomplete selected dataset at {args.output_path}")
        shutil.rmtree(args.output_path)

    prepared = load_from_disk(str(args.prepared_path.resolve()))
    if not isinstance(prepared, DatasetDict):
        raise TypeError("Prepared data must be a DatasetDict")
    train = prepared["train"]
    score_map = read_scores(args.scores_path.resolve())
    row_ids = [int(value) for value in train["row_id"]]
    if set(row_ids) != set(score_map):
        raise RuntimeError("Implicit scores do not cover the prepared training dataset exactly")

    external = np.asarray(train["external_margin"], dtype=np.float64)
    implicit = np.asarray([score_map[row_id]["implicit_margin"] for row_id in row_ids], dtype=np.float64)
    external_upper = dynamic_upper(external, "external")
    implicit_upper = dynamic_upper(implicit, "implicit")
    external_probability = project_margin(external, external_upper)
    implicit_probability = project_margin(implicit, implicit_upper)
    bees_probability = bayesian_aggregate([external_probability, implicit_probability])

    eligible = np.isfinite(bees_probability) & (external > 0.0) & (implicit > 0.0)
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < args.selection_size:
        raise RuntimeError(
            f"Only {len(eligible_indices):,} rows have positive margins from both sources; "
            f"cannot select {args.selection_size:,} without violating the BeeS threshold rule"
        )
    ranked = eligible_indices[np.argsort(-bees_probability[eligible_indices], kind="stable")]
    selected_indices = ranked[: args.selection_size]

    enriched = train.add_column("implicit_margin", implicit.tolist())
    enriched = enriched.add_column("external_preference_probability", external_probability.tolist())
    enriched = enriched.add_column("implicit_preference_probability", implicit_probability.tolist())
    enriched = enriched.add_column("bees_probability", bees_probability.tolist())
    selected = enriched.select(selected_indices.tolist())
    selected = selected.add_column("bees_rank", list(range(1, len(selected) + 1)))
    output = DatasetDict({"train": selected, "test": prepared["test"]})
    output.save_to_disk(args.output_path)
    output["train"].to_parquet(args.output_path / "train.parquet")
    output["test"].to_parquet(args.output_path / "test.parquet")

    source_metadata = read_json(args.prepare_metadata)
    metadata = {
        "method": "BeeS (Bayesian Aggregation for Preference data Selection)",
        "paper_equation": "Eq. (3)",
        "bees_upstream_commit": BEES_UPSTREAM_COMMIT,
        "source": source_metadata,
        "prepared_train_rows": len(train),
        "selected_train_rows": len(selected),
        "test_rows": len(prepared["test"]),
        "selection_fraction": len(selected) / len(train),
        "positive_both_sources": int(eligible.sum()),
        "bounds": {
            "lower": LOWER_BOUND,
            "external_upper": external_upper,
            "implicit_upper": implicit_upper,
        },
        "selection_rule": "top BeeS probability among rows with external_margin > 0 and implicit_margin > 0",
        "margin_summary": {
            "external_min": float(np.min(external)),
            "external_median": float(np.median(external)),
            "external_max": float(np.max(external)),
            "implicit_min": float(np.min(implicit)),
            "implicit_median": float(np.median(implicit)),
            "implicit_max": float(np.max(implicit)),
            "selected_bees_probability_min": float(np.min(selected["bees_probability"])),
            "selected_bees_probability_max": float(np.max(selected["bees_probability"])),
        },
        "scores_path": str(args.scores_path.resolve()),
    }
    write_json(args.output_path / "metadata.json", metadata)
    card = f"""---
dataset_info:
  features:
  - name: prompt
  - name: chosen
  - name: rejected
  splits:
  - name: train
    num_examples: {len(selected)}
  - name: test
    num_examples: {len(prepared['test'])}
license: mit
task_categories:
- text-generation
---

# BeeS-selected UltraFeedback Binarized for OLMo 2 1B DPO

This local dataset was selected from `{source_metadata['dataset_id']}` at revision
`{source_metadata['dataset_revision']}`. It follows BeeS: a 2,000-row in-distribution proxy DPO
model supplies the implicit reward margin, the original UltraFeedback judge scores supply an
independent external margin, and Eq. (3) combines their clipped linear projections.

Only pairs that fit losslessly within {source_metadata['max_length']} OLMo chat-template tokens
and have positive external and implicit margins are eligible. The train split contains the top
{len(selected):,} eligible pairs. No response text is truncated during dataset creation.
"""
    (args.output_path / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
