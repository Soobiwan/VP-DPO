from __future__ import annotations

from pathlib import Path

import nbformat as nbf


WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT = WORKSPACE / "olmo_bees_tidpo_simpo_sampo_train_eval.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10+"},
    }
    notebook["cells"] = [
        markdown(
            r"""
# OLMo 2 1B — TIDPO, SimPO, and SamPO training + evaluation on two local GPUs

Run this notebook from top to bottom. It creates three independent full-parameter models from the
same pinned `allenai/OLMo-2-0425-1B-SFT` checkpoint and the same 6,000-row OLMo BeeS training split,
then evaluates all three on the untouched 1,891-row test split and a shared `lm-eval` suite.

- **TIDPO** comes from the imported, commit-pinned
  [gracefulning/TIDPO](https://github.com/gracefulning/TIDPO) implementation.
- **SimPO** follows the [official SimPO formulation](https://github.com/princeton-nlp/SimPO): a
  reference-free, length-normalized reward with a target margin.
- **SamPO** follows the [official SamPO implementation](https://github.com/LuJunru/SamPO): uniformly
  down-sample the longer side of each preference pair to the shorter side's token count.

Every training run uses both GPUs through FSDP2 full sharding, paged FP32 AdamW states, FP32 master
weights, FP16 loss-scaled compute, and activation checkpointing/offloading. There is no LoRA, PEFT,
QLoRA, or weight quantization.
"""
        ),
        code(
            r"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time


def locate_workspace() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "BeeS" / "olmo2_bees" / "train_preference_suite.py").is_file():
            return candidate
    raise FileNotFoundError("Open this notebook from the VPDPO workspace or one of its children")


WORKSPACE = locate_workspace()
PROJECT_ROOT = WORKSPACE / "BeeS"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from olmo2_bees.common import (
    assert_two_turing_gpus,
    configure_workspace,
    package_versions,
    run_streaming,
    sha256_file,
    write_json,
)
from olmo2_bees.preference_suite_losses import (
    METHOD_DESCRIPTIONS,
    METHOD_FORMULAS,
    METHODS,
)

CACHE_ENV = configure_workspace(WORKSPACE)
print("Workspace:", WORKSPACE)
print("BeeS package:", PROJECT_ROOT)
"""
        ),
        markdown(
            r"""
## Runtime environment

The existing OLMo environment is reused. The imported TIDPO repository's older dependency pins are
intentionally not installed over it; its loss is adapted to the already validated modern OLMo/FSDP2
stack. Set `INSTALL_DEPENDENCIES=True` only if this environment has not already been prepared.
"""
        ),
        code(
            r"""
INSTALL_DEPENDENCIES = False

if INSTALL_DEPENDENCIES:
    run_streaming(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements-olmo2.txt")],
        cwd=PROJECT_ROOT,
    )

versions = package_versions(
    ["torch", "transformers", "datasets", "accelerate", "trl", "bitsandbytes", "safetensors", "lm_eval"]
)
print(json.dumps(versions, indent=2))
"""
        ),
        markdown(
            r"""
## Configuration

Defaults run all training and all evaluations. Set either run switch to `False` only when using the
notebook to inspect existing artifacts. Reference token log-probabilities and TIDPO anchors are
computed once and reused. `SAVE_STEPS=0` avoids very large optimizer/FSDP restart checkpoints.

Expected wall time on this machine is roughly **24–36 hours**: about 3–7 hours for token references,
top-k KL support, and fixed TIDPO anchors; 15–22 hours across the three trainers; and 4–7 hours for
held-out and `lm-eval` scoring. TIDPO is the slowest because gradient attribution adds a model pass
and the triplet term adds an anchor pass. Disk usage is roughly **21–25 GB** for three FP32 final
checkpoints plus caches/evaluation output. This estimate is anchored by the existing 7.19-hour,
two-epoch OLMo DPO run on this PC; actual time still depends on response lengths, GPU thermals, and
benchmark generation lengths.
"""
        ),
        code(
            r"""
MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
MODEL_REVISION = "0d85a3d037876ce6ac7d4311d994400fc66ac27f"
SOURCE_JSONL = WORKSPACE / "ultrafeedback_bees_olmo2_1b_segmented_final.jsonl"

METHODS_TO_RUN = list(METHODS)
RUN_TRAINING = True
RUN_EVALUATIONS = True
MAX_LENGTH = 1024
EPOCHS = 1.0
LEARNING_RATE = 5e-7
GRADIENT_ACCUMULATION_STEPS = 8
LOGGING_STEPS = 1
SAVE_STEPS = 0
RESUME = False
SEED = 42

# Imported TIDPO defaults. Fixed base-policy anchors replace an online resident reference model.
TIDPO_BETA = 0.2
TIDPO_ALPHA = 0.5
TIDPO2 = True
TIDPO_KL_TOP_K = 32
TIDPO_LAMBDA_IMPORTANCE = 0.2
TIDPO_PRIOR_SIGMA_DIV = 8.0
TIDPO_TRIPLET_GAMMA = 0.001
TIDPO_TRIPLET_MARGIN = 0.001
ANCHOR_MAX_NEW_TOKENS = 64

# Official SimPO starting recommendations for an instruction-tuned model.
SIMPO_BETA = 2.0
SIMPO_GAMMA_BETA_RATIO = 0.5

# SamPO retains the usual DPO beta after equal-token down-sampling.
SAMPO_BETA = 0.1

ARTIFACT_ROOT = WORKSPACE / "artifacts" / "olmo2_bees" / "tidpo_simpo_sampo"
PREPARED_DATASET = (
    WORKSPACE / "artifacts" / "olmo2_bees" / "structured_all_methods" / "prepared"
)
REFERENCE_CACHE = ARTIFACT_ROOT / "token_reference_with_anchors"
RUN_ROOT = ARTIFACT_ROOT / "runs"
EVAL_ROOT = ARTIFACT_ROOT / "evaluations"
BASELINE_LM_EVAL = EVAL_ROOT / "lm_eval_baseline" / MODEL_REVISION[:16]
SUMMARY_PATH = ARTIFACT_ROOT / "comparison_summary.json"

for path in (ARTIFACT_ROOT, RUN_ROOT, EVAL_ROOT):
    path.mkdir(parents=True, exist_ok=True)
assert SOURCE_JSONL.is_file(), SOURCE_JSONL
assert METHODS_TO_RUN and set(METHODS_TO_RUN).issubset(METHODS)
assert len(METHODS_TO_RUN) == len(set(METHODS_TO_RUN))
print(json.dumps({
    "methods": METHODS_TO_RUN,
    "prepared_dataset": str(PREPARED_DATASET),
    "reference_cache": str(REFERENCE_CACHE),
    "run_root": str(RUN_ROOT),
    "eval_root": str(EVAL_ROOT),
}, indent=2))
"""
        ),
        markdown(
            r"""
## Hardware and upstream provenance audit

Both visible GPUs are required. The smaller 8 GB device determines the memory settings. The TIDPO
snapshot is recorded at commit `e04a0926869a8f9fe9c9e9ce395394fd2c697fe2`; its original source and
license remain under `third_party/TIDPO`.
"""
        ),
        code(
            r"""
gpus = assert_two_turing_gpus()
assert min(gpu["memory_gib"] for gpu in gpus[:2]) >= 7.5
assert all(not gpu["bf16_supported"] for gpu in gpus[:2])
print(json.dumps(gpus[:2], indent=2))
print("Effective global pair batch:", 2 * GRADIENT_ACCUMULATION_STEPS)

upstream = json.loads((WORKSPACE / "third_party" / "TIDPO" / "UPSTREAM.json").read_text())
assert upstream["commit"] == "e04a0926869a8f9fe9c9e9ce395394fd2c697fe2"
assert (WORKSPACE / "third_party" / "TIDPO" / "LICENSE").is_file()
print(json.dumps(upstream, indent=2))
"""
        ),
        markdown(
            r"""
## Objective audit and CPU self-tests

TIDPO retains the imported implementation's last-valid-position gradient target, L1 embedding
gradient attribution, gradient/Gaussian mixture, TDPO2 correction, and mean-one sequence-token
weights (applied only at response positions). The exact full-vocabulary position KL cannot be cached
practically, so this adapter uses a documented lower-bound projection: each reference top-k token is
retained and all remaining vocabulary mass becomes one bucket. Cached anchors are sampled
reproducibly from the pinned base OLMo policy instead of the changing online policy; these two
bounded adaptations avoid a resident reference model beside a full trainable model on the 8 GB GPU.
SimPO is reference-free. SamPO samples equal numbers of tokens from the two completions and uses
tokenwise cached reference ratios at exactly those sampled positions.
"""
        ),
        code(
            r"""
for method in METHODS_TO_RUN:
    print(f"{method:6s} | {METHOD_FORMULAS[method]}")
    print(f"         {METHOD_DESCRIPTIONS[method]}")

run_streaming([
    sys.executable, "-m", "olmo2_bees.train_preference_suite", "self-test",
    "--workspace", str(WORKSPACE),
], cwd=PROJECT_ROOT)
"""
        ),
        markdown(
            r"""
## Stage 1 — prepare the shared segmented OLMo dataset

This reuses the lossless OLMo chat tokenization and segment alignment from the combined structured
notebook. The stage is idempotent and will reuse an already validated 6,000/1,891 cache.
"""
        ),
        code(
            r"""
prepare_command = [
    sys.executable, "-m", "olmo2_bees.train_structured", "prepare",
    "--workspace", str(WORKSPACE),
    "--source-jsonl", str(SOURCE_JSONL),
    "--output-dir", str(PREPARED_DATASET),
    "--model-id", MODEL_ID,
    "--model-revision", MODEL_REVISION,
    "--max-length", str(MAX_LENGTH),
]
run_streaming(prepare_command, cwd=PROJECT_ROOT)

from datasets import load_from_disk

prepared = load_from_disk(str(PREPARED_DATASET))
assert len(prepared["train"]) == 6000
assert len(prepared["test"]) == 1891
required_segmented_columns = {
    "row_id",
    "chosen_input_ids",
    "chosen_segment_ids",
    "rejected_input_ids",
    "rejected_segment_ids",
}
assert required_segmented_columns.issubset(prepared["test"].column_names)
print(prepared)
"""
        ),
        markdown(
            r"""
## Stage 2 — shared token reference cache and TIDPO anchors

Both GPUs process disjoint rows. Per-rank JSONL files flush regularly and resume safely. The merged
cache contains aligned reference log-probabilities for chosen/rejected completion tokens, reference
top-k IDs/log-probabilities for the projected TDPO2 position-KL term, plus a fixed sampled anchor and
its reference token log-probabilities for TIDPO's triplet term.
"""
        ),
        code(
            r"""
if RUN_TRAINING:
    reference_command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
        "--mixed_precision", "fp16",
        "-m", "olmo2_bees.train_preference_suite", "reference",
        "--workspace", str(WORKSPACE),
        "--dataset-path", str(PREPARED_DATASET),
        "--split", "train",
        "--model-id", MODEL_ID,
        "--model-revision", MODEL_REVISION,
        "--output-dir", str(REFERENCE_CACHE),
        "--max-length", str(MAX_LENGTH),
        "--tidpo-kl-top-k", str(TIDPO_KL_TOP_K),
        "--with-tidpo-anchors",
        "--anchor-max-new-tokens", str(ANCHOR_MAX_NEW_TOKENS),
        "--anchor-seed", str(SEED + 1000),
    ]
    run_streaming(reference_command, cwd=PROJECT_ROOT)
else:
    print("RUN_TRAINING=False; reusing existing reference cache")

reference_manifest = json.loads((REFERENCE_CACHE / "reference_manifest.json").read_text())
assert reference_manifest["rows"] == 6000
assert reference_manifest["world_size"] == 2
assert reference_manifest["cache_schema_version"] == 2
assert reference_manifest["with_tidpo_anchors"] is True
assert reference_manifest["tidpo_kl_top_k"] == TIDPO_KL_TOP_K
print(json.dumps(reference_manifest, indent=2))
"""
        ),
        markdown(
            r"""
## Stage 3 — train TIDPO, SimPO, and SamPO sequentially

Each method starts from a clean copy of the same pinned SFT checkpoint and uses both GPUs. Existing
validated final runs are skipped. With restart checkpoints disabled, interrupting a method restarts
that method but does not disturb completed methods or the shared cache.
"""
        ),
        code(
            r"""
def run_dir(method: str) -> Path:
    return RUN_ROOT / method.lower()


def final_model(method: str) -> Path:
    return run_dir(method) / "final"


def completed_training(method: str) -> bool:
    manifest_path = run_dir(method) / "training_manifest.json"
    if not manifest_path.is_file() or not final_model(method).is_dir():
        return False
    manifest = json.loads(manifest_path.read_text())
    expected_signature = {
        "method": method,
        "epochs": EPOCHS,
        "max_steps": -1,
        "learning_rate": LEARNING_RATE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "seed": SEED,
        "transformer_layer_class": "Olmo2DecoderLayer",
        "activation_offloading": True,
        "tidpo_beta": TIDPO_BETA,
        "tidpo_alpha": TIDPO_ALPHA,
        "tidpo2": TIDPO2,
        "tidpo_kl_top_k": TIDPO_KL_TOP_K,
        "tidpo_lambda_importance": TIDPO_LAMBDA_IMPORTANCE,
        "tidpo_prior_sigma_div": TIDPO_PRIOR_SIGMA_DIV,
        "tidpo_triplet_gamma": TIDPO_TRIPLET_GAMMA,
        "tidpo_triplet_margin": TIDPO_TRIPLET_MARGIN,
        "simpo_beta": SIMPO_BETA,
        "simpo_gamma_beta_ratio": SIMPO_GAMMA_BETA_RATIO,
        "sampo_beta": SAMPO_BETA,
    }
    return (
        manifest.get("method") == method
        and manifest.get("model_revision") == MODEL_REVISION
        and manifest.get("world_size") == 2
        and manifest.get("saved_weight_dtypes") == ["F32"]
        and manifest.get("training_signature") == expected_signature
    )


if RUN_TRAINING:
    for method in METHODS_TO_RUN:
        if completed_training(method):
            print(f"{method}: completed model exists; skipping")
            continue
        command = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
            "--mixed_precision", "fp16",
            "-m", "olmo2_bees.train_preference_suite", "train",
            "--workspace", str(WORKSPACE),
            "--dataset-path", str(PREPARED_DATASET),
            "--reference-cache", str(REFERENCE_CACHE),
            "--train-split", "train",
            "--model-id", MODEL_ID,
            "--model-revision", MODEL_REVISION,
            "--output-dir", str(run_dir(method)),
            "--run-name", f"olmo2-bees-{method.lower()}",
            "--method", method,
            "--epochs", str(EPOCHS),
            "--learning-rate", str(LEARNING_RATE),
            "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
            "--logging-steps", str(LOGGING_STEPS),
            "--save-steps", str(SAVE_STEPS),
            "--seed", str(SEED),
            "--tidpo-beta", str(TIDPO_BETA),
            "--tidpo-alpha", str(TIDPO_ALPHA),
            "--tidpo-kl-top-k", str(TIDPO_KL_TOP_K),
            "--tidpo-lambda-importance", str(TIDPO_LAMBDA_IMPORTANCE),
            "--tidpo-prior-sigma-div", str(TIDPO_PRIOR_SIGMA_DIV),
            "--tidpo-triplet-gamma", str(TIDPO_TRIPLET_GAMMA),
            "--tidpo-triplet-margin", str(TIDPO_TRIPLET_MARGIN),
            "--simpo-beta", str(SIMPO_BETA),
            "--simpo-gamma-beta-ratio", str(SIMPO_GAMMA_BETA_RATIO),
            "--sampo-beta", str(SAMPO_BETA),
        ]
        command.append("--tidpo2" if TIDPO2 else "--no-tidpo2")
        if RESUME and SAVE_STEPS > 0:
            command.append("--resume")
        run_streaming(command, cwd=PROJECT_ROOT)
else:
    print("RUN_TRAINING=False; reusing existing method runs")
"""
        ),
        markdown(
            r"""
## Saved-model integrity audit

This verifies the full-parameter/precision guarantees and hashes every final weight file before any
evaluation begins.
"""
        ),
        code(
            r"""
training_manifests = {}
for method in METHODS_TO_RUN:
    path = run_dir(method) / "training_manifest.json"
    assert path.is_file(), path
    manifest = json.loads(path.read_text())
    assert manifest["method"] == method
    assert manifest["full_parameter_training"] is True
    assert manifest["peft_or_lora"] is False
    assert manifest["weight_quantization"] is None
    assert manifest["optimizer_is_paged"] is True
    assert manifest["optimizer_state_bits"] == 32
    assert manifest["saved_weight_dtypes"] == ["F32"]
    assert manifest["parallelism"] == "FSDP2_FULL_SHARD"
    assert manifest["world_size"] == 2
    for filename, digest in manifest["weight_files_sha256"].items():
        assert sha256_file(final_model(method) / filename) == digest
    training_manifests[method] = manifest
    print(method, manifest["metrics"])
"""
        ),
        markdown(
            r"""
## Stage 4 — held-out implicit preference evaluation

For every test pair, this computes the standard reference-adjusted reward margin using the pinned
SFT model as reference. It provides one common measurement even though the training objectives
differ. Each two-GPU run resumes from per-rank JSONL files.
"""
        ),
        code(
            r"""
def preference_eval_dir(method: str) -> Path:
    return EVAL_ROOT / method.lower() / "preference"


if RUN_EVALUATIONS:
    for method in METHODS_TO_RUN:
        output = preference_eval_dir(method)
        if (output / "implicit_scores.jsonl").is_file() and (output / "score_manifest.json").is_file():
            print(f"{method}: held-out preference scores exist; skipping")
            continue
        command = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
            "--mixed_precision", "fp16",
            "-m", "olmo2_bees.score_implicit",
            "--workspace", str(WORKSPACE),
            "--dataset-path", str(PREPARED_DATASET),
            "--split", "test",
            "--reference-model", MODEL_ID,
            "--reference-revision", MODEL_REVISION,
            "--policy-model", str(final_model(method)),
            "--output-dir", str(output),
            "--max-length", str(MAX_LENGTH),
        ]
        run_streaming(command, cwd=PROJECT_ROOT)
else:
    print("RUN_EVALUATIONS=False; reusing existing preference scores")
"""
        ),
        markdown(
            r"""
## Stage 5 — shared deterministic `lm-eval` comparison

The pinned baseline is evaluated once, then each trained model is evaluated with identical tasks,
chat template, seeds, precision, and two-GPU launch. These are full benchmark runs, not limits or
smoke tests.
"""
        ),
        code(
            r"""
BENCHMARK_TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "gsm8k",
    "leaderboard_ifeval",
]

from olmo2_bees.quality_gate import task_metrics


def candidate_lm_eval(method: str) -> Path:
    return EVAL_ROOT / method.lower() / "lm_eval"


def benchmark_complete(output_dir: Path) -> bool:
    try:
        return set(BENCHMARK_TASKS).issubset(task_metrics(output_dir))
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
        return False


def run_lm_eval(model_args: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_streaming([
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
        "--mixed_precision", "fp16",
        "-m", "lm_eval", "run",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", *BENCHMARK_TASKS,
        "--batch_size", "1",
        "--apply_chat_template",
        "--seed", str(SEED),
        "--output_path", str(output_dir),
    ], cwd=PROJECT_ROOT)


if RUN_EVALUATIONS:
    if not benchmark_complete(BASELINE_LM_EVAL):
        run_lm_eval(
            f"pretrained={MODEL_ID},revision={MODEL_REVISION},dtype=float16",
            BASELINE_LM_EVAL,
        )
    else:
        print("Baseline lm-eval exists; skipping")
    for method in METHODS_TO_RUN:
        output = candidate_lm_eval(method)
        if benchmark_complete(output):
            print(f"{method}: lm-eval exists; skipping")
        else:
            run_lm_eval(f"pretrained={final_model(method)},dtype=float16", output)
else:
    print("RUN_EVALUATIONS=False; reusing existing lm-eval results")
"""
        ),
        markdown(
            r"""
## Stage 6 — per-method quality gates and comparison summary

Each gate writes its report before returning failure. A failing method is recorded and evaluation
continues; only passing methods receive an `approved_model.json` pointer. The final summary makes
side-by-side preference and benchmark metrics available without selecting a winner automatically.
"""
        ),
        code(
            r"""
from olmo2_bees.quality_gate import preference_metrics

comparison = {
    "base_model": MODEL_ID,
    "base_model_revision": MODEL_REVISION,
    "dataset": str(PREPARED_DATASET.resolve()),
    "test_rows": len(prepared["test"]),
    "benchmark_tasks": BENCHMARK_TASKS,
    "baseline_benchmarks": task_metrics(BASELINE_LM_EVAL),
    "methods": {},
}

for method in METHODS_TO_RUN:
    method_eval = EVAL_ROOT / method.lower()
    report_path = method_eval / "quality_gate.json"
    scores_path = preference_eval_dir(method) / "implicit_scores.jsonl"
    gate_command = [
        sys.executable, "-m", "olmo2_bees.quality_gate",
        "--workspace", str(WORKSPACE),
        "--preference-scores", str(scores_path),
        "--baseline-lm-eval", str(BASELINE_LM_EVAL),
        "--candidate-lm-eval", str(candidate_lm_eval(method)),
        "--output", str(report_path),
        "--max-task-drop", "0.01",
        "--max-macro-drop", "0.002",
    ]
    for task in BENCHMARK_TASKS:
        gate_command.extend(["--required-benchmark-task", task])
    try:
        run_streaming(gate_command, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        # The gate deliberately returns nonzero on regression, after writing its evidence report.
        print(f"{method}: quality gate rejected this checkpoint; continuing comparison")

    report = json.loads(report_path.read_text())
    entry = {
        "approved": bool(report["passed"]),
        "final_model": str(final_model(method).resolve()),
        "training_manifest": str((run_dir(method) / "training_manifest.json").resolve()),
        "preference": preference_metrics(scores_path),
        "benchmarks": task_metrics(candidate_lm_eval(method)),
        "quality_report": str(report_path.resolve()),
    }
    comparison["methods"][method] = entry
    if report["passed"]:
        write_json(method_eval / "approved_model.json", {
            "approved": True,
            "method": method,
            "model_path": str(final_model(method).resolve()),
            "base_model": MODEL_ID,
            "base_model_revision": MODEL_REVISION,
            "training_manifest": entry["training_manifest"],
            "quality_report": entry["quality_report"],
            "weight_files_sha256": training_manifests[method]["weight_files_sha256"],
            "weight_quantization": None,
            "peft_or_lora": False,
        })

write_json(SUMMARY_PATH, comparison)
print(json.dumps(comparison, indent=2))
print("Comparison summary:", SUMMARY_PATH)
"""
        ),
        markdown(
            r"""
## Optional compact table

This cell displays the primary held-out and benchmark results after all evidence has been written.
"""
        ),
        code(
            r"""
import pandas as pd

rows = []
for method, result in comparison["methods"].items():
    row = {
        "method": method,
        "approved": result["approved"],
        "preference_accuracy": result["preference"]["dpo_reward_accuracy"],
        "implicit_margin_mean": result["preference"]["implicit_margin_mean"],
        "length_norm_accuracy": result["preference"]["policy_length_normalized_preference_accuracy"],
    }
    row.update({f"eval/{task}": value for task, value in result["benchmarks"].items() if task in BENCHMARK_TASKS})
    rows.append(row)

display(pd.DataFrame(rows).set_index("method"))
"""
        ),
    ]
    return notebook


def main() -> None:
    nbf.write(build_notebook(), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
