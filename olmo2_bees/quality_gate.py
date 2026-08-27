from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import configure_workspace, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reject a DPO checkpoint if quality regresses")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--preference-scores", type=Path, required=True)
    parser.add_argument("--baseline-lm-eval", type=Path, default=None)
    parser.add_argument("--candidate-lm-eval", type=Path, default=None)
    parser.add_argument(
        "--required-benchmark-task",
        action="append",
        default=[],
        help="Task that must appear in both lm-eval outputs; repeat for every required task",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-task-drop", type=float, default=0.01)
    parser.add_argument("--max-macro-drop", type=float, default=0.002)
    parser.add_argument("--max-direct-preference-drop", type=float, default=0.005)
    parser.add_argument("--min-dpo-reward-accuracy", type=float, default=0.55)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records in {path}")
    return records


def preference_metrics(path: Path) -> dict[str, float]:
    records = read_jsonl(path)
    implicit = np.asarray([row["implicit_margin"] for row in records], dtype=np.float64)
    policy_chosen = np.asarray(
        [row["policy_chosen_logp"] / row["chosen_completion_tokens"] for row in records]
    )
    policy_rejected = np.asarray(
        [row["policy_rejected_logp"] / row["rejected_completion_tokens"] for row in records]
    )
    reference_chosen = np.asarray(
        [row["reference_chosen_logp"] / row["chosen_completion_tokens"] for row in records]
    )
    reference_rejected = np.asarray(
        [row["reference_rejected_logp"] / row["rejected_completion_tokens"] for row in records]
    )
    return {
        "rows": float(len(records)),
        "dpo_reward_accuracy": float(np.mean(implicit > 0.0)),
        "implicit_margin_mean": float(np.mean(implicit)),
        "implicit_margin_median": float(np.median(implicit)),
        "policy_length_normalized_preference_accuracy": float(
            np.mean(policy_chosen > policy_rejected)
        ),
        "reference_length_normalized_preference_accuracy": float(
            np.mean(reference_chosen > reference_rejected)
        ),
    }


def latest_result(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        [item for item in path.rglob("*.json") if "sample" not in item.name.lower()],
        key=lambda item: item.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No lm-eval result JSON under {path}")
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
            return candidate
    raise RuntimeError(f"No valid lm-eval results object under {path}")


def task_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(latest_result(path).read_text(encoding="utf-8"))
    output: dict[str, float] = {}
    priorities = (
        "prompt_level_strict_acc,none",
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "acc_norm,none",
        "acc,none",
        "f1,none",
    )
    for task, metrics in payload["results"].items():
        if not isinstance(metrics, dict):
            continue
        for metric in priorities:
            value = metrics.get(metric)
            if isinstance(value, (float, int)) and math_is_finite(float(value)):
                output[task] = float(value)
                break
    if not output:
        raise RuntimeError(f"No comparable metrics found in {path}")
    return output


def math_is_finite(value: float) -> bool:
    return not (np.isnan(value) or np.isinf(value))


def main() -> None:
    args = parse_args()
    configure_workspace(args.workspace)
    preference = preference_metrics(args.preference_scores.resolve())
    checks = {
        "positive_mean_implicit_margin": preference["implicit_margin_mean"] > 0.0,
        "dpo_reward_accuracy": (
            preference["dpo_reward_accuracy"] >= args.min_dpo_reward_accuracy
        ),
        "direct_preference_not_regressed": (
            preference["policy_length_normalized_preference_accuracy"]
            >= preference["reference_length_normalized_preference_accuracy"]
            - args.max_direct_preference_drop
        ),
    }
    report: dict[str, Any] = {
        "preference_metrics": preference,
        "preference_checks": checks,
        "benchmark_checks_enabled": bool(args.baseline_lm_eval and args.candidate_lm_eval),
    }

    if args.required_benchmark_task and not (
        args.baseline_lm_eval and args.candidate_lm_eval
    ):
        raise ValueError("Required benchmark tasks need both lm-eval paths")
    if bool(args.baseline_lm_eval) != bool(args.candidate_lm_eval):
        raise ValueError("Provide both lm-eval paths or neither")
    if args.baseline_lm_eval and args.candidate_lm_eval:
        baseline = task_metrics(args.baseline_lm_eval.resolve())
        candidate = task_metrics(args.candidate_lm_eval.resolve())
        required = set(args.required_benchmark_task)
        missing_baseline = sorted(required - set(baseline))
        missing_candidate = sorted(required - set(candidate))
        if missing_baseline or missing_candidate:
            raise RuntimeError(
                "Required lm-eval tasks are missing: "
                f"baseline={missing_baseline}, candidate={missing_candidate}"
            )
        common = sorted(required or (set(baseline) & set(candidate)))
        if not common:
            raise RuntimeError("Baseline and candidate lm-eval outputs share no tasks")
        deltas = {task: candidate[task] - baseline[task] for task in common}
        task_checks = {task: delta >= -args.max_task_drop for task, delta in deltas.items()}
        macro_delta = float(np.mean(list(deltas.values())))
        report["benchmarks"] = {
            "baseline": {task: baseline[task] for task in common},
            "candidate": {task: candidate[task] for task in common},
            "delta": deltas,
            "macro_delta": macro_delta,
            "max_task_drop": args.max_task_drop,
            "max_macro_drop": args.max_macro_drop,
            "task_checks": task_checks,
            "macro_check": macro_delta >= -args.max_macro_drop,
        }
        checks["all_benchmark_tasks_within_tolerance"] = all(task_checks.values())
        checks["benchmark_macro_within_tolerance"] = macro_delta >= -args.max_macro_drop

    report["passed"] = all(checks.values())
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Quality gate failed; keep the trained checkpoint separate from the approved model")


if __name__ == "__main__":
    main()
