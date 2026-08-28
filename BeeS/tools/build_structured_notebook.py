from __future__ import annotations

from pathlib import Path

import nbformat as nbf


WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT = WORKSPACE / "olmo_bees_all_methods_dual_gpu.ipynb"


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
# OLMo 2 1B — all segmented BeeS preference methods on two local GPUs

This is the combined local training notebook for the final segmented OLMo BeeS dataset. It trains
the five requested objectives from clean copies of `allenai/OLMo-2-0425-1B-SFT`:

1. Method A
2. Method B-DPO
3. Method B-VDPO
4. Method C-DPO
5. Method C-VDPO

The runs are sequential; each run uses both GPUs simultaneously through FSDP2 full sharding. The
notebook uses the proven dual-GPU OLMo DPO architecture from `BeeS/notebooks/02_train_olmo2_1b_dpo.ipynb`,
but replaces TRL's scalar DPO loss with the segment-level formulations preserved from the five
method notebooks.

Reference segment log-probabilities are computed once and shared by all runs. There is no LoRA,
PEFT, QLoRA, or weight quantization: every OLMo policy parameter trains, master weights and AdamW
states remain FP32, and FP16 is used only for loss-scaled compute.
"""
        ),
        code(
            r"""
from pathlib import Path
import json
import os
import sys


def locate_workspace() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "BeeS" / "olmo2_bees" / "train_structured.py").is_file():
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
)
from olmo2_bees.structured_preference import (
    REQUESTED_VARIANTS,
    VARIANT_DESCRIPTIONS,
    VARIANT_FORMULAS,
)

CACHE_ENV = configure_workspace(WORKSPACE)
print("Workspace:", WORKSPACE)
print("BeeS package:", PROJECT_ROOT)
"""
        ),
        markdown(
            r"""
## Runtime

The pinned environment matches the already validated OLMo DPO notebook. PyTorch is not installed
by this cell because its CUDA wheel is machine-specific. Set `INSTALL_DEPENDENCIES=True` only when
the current environment does not already contain the packages.
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

print(json.dumps(package_versions(
    ["torch", "transformers", "datasets", "accelerate", "trl", "bitsandbytes", "safetensors"]
), indent=2))
"""
        ),
        markdown(
            r"""
## Configuration

`MAX_LENGTH=1024` uses the complete final dataset without truncation. The default optimization
settings (`1` epoch, `2e-6`, `beta=0.1`) preserve the method notebooks; the multi-GPU memory and
precision settings come from the OLMo DPO notebook.

With `SAVE_STEPS=0`, each run saves only its final roughly 6 GB FP32 model. Set a positive interval
to enable restartable paged-Adam/FSDP checkpoints and `RESUME=True`; those checkpoints are large,
so five fully restartable runs can require hundreds of gigabytes.
"""
        ),
        code(
            r"""
MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
MODEL_REVISION = "0d85a3d037876ce6ac7d4311d994400fc66ac27f"
SOURCE_JSONL = WORKSPACE / "ultrafeedback_bees_olmo2_1b_segmented_final.jsonl"
SOURCE_MANIFEST = WORKSPACE / "ultrafeedback_bees_olmo2_1b_segmented_final.manifest.json"

MAX_LENGTH = 1024
VARIANTS_TO_RUN = list(REQUESTED_VARIANTS)
EPOCHS = 1.0
LEARNING_RATE = 2e-6
BETA = 0.1
GRADIENT_ACCUMULATION_STEPS = 8
LOGGING_STEPS = 1
SAVE_STEPS = 0       # 0 = final model only; >0 = large restart checkpoints
RESUME = False       # meaningful only when SAVE_STEPS > 0
SEED = 42

ARTIFACT_ROOT = WORKSPACE / "artifacts" / "olmo2_bees" / "structured_all_methods"
PREPARED_DATASET = ARTIFACT_ROOT / "prepared"
PREPARED_MANIFEST = ARTIFACT_ROOT / "prepared.manifest.json"
REFERENCE_CACHE = ARTIFACT_ROOT / "reference_segments"
RUN_ROOT = ARTIFACT_ROOT / "runs"

for path in (ARTIFACT_ROOT, RUN_ROOT):
    path.mkdir(parents=True, exist_ok=True)

assert SOURCE_JSONL.is_file(), SOURCE_JSONL
assert SOURCE_MANIFEST.is_file(), SOURCE_MANIFEST
assert VARIANTS_TO_RUN and set(VARIANTS_TO_RUN).issubset(REQUESTED_VARIANTS)
assert len(VARIANTS_TO_RUN) == len(set(VARIANTS_TO_RUN))

print(json.dumps({
    "source": str(SOURCE_JSONL),
    "prepared": str(PREPARED_DATASET),
    "reference_cache": str(REFERENCE_CACHE),
    "run_root": str(RUN_ROOT),
    "variants": VARIANTS_TO_RUN,
}, indent=2))
"""
        ),
        markdown(
            r"""
## Hardware audit

Both GPUs must be visible in this kernel. The smaller GPU controls the per-device batch size. The
training subprocesses use one process per GPU and an effective batch of
`1 pair × 2 GPUs × 8 accumulation = 16 pairs`.
"""
        ),
        code(
            r"""
gpus = assert_two_turing_gpus()
assert len(gpus) >= 2
assert min(gpu["memory_gib"] for gpu in gpus[:2]) >= 7.5
print(json.dumps(gpus[:2], indent=2))
print("Effective global pair batch:", 2 * GRADIENT_ACCUMULATION_STEPS)
"""
        ),
        markdown(
            r"""
## Unified loss formulation

For a response segment `s`, let `ell_s` and `ell_ref_s` be the policy and frozen-reference sums of
token log-probabilities, `h_s = ell_s - ell_ref_s`, `q_s` the supplied segment score, and `n_s` its
token count. Segments are sorted by the supplied rank before every Plackett-Luce (PL) calculation.

- **DPO core:** `beta * sum_s h_s`
- **VDPO core:** `beta * sum_s(q_s * h_s / n_s) / sum_s q_s`
- **Method A structural utility:** the score-gap-weighted difference between policy and reference
  PL step log-probabilities, with utilities `beta * ell_s`
- **Method B structural coherence:** `PL(q_s * beta * h_s)`

The response objectives are:

- **A:** Method A structural utility
- **B-DPO:** DPO core + Method B coherence
- **B-VDPO:** VDPO core + Method B coherence
- **C-DPO:** DPO core + Method A structural utility
- **C-VDPO:** VDPO core + Method A structural utility

Every pair finally uses `-logsigmoid(O(chosen) - O(rejected))`. These are the formulations encoded
in the five source notebooks; the optional plain `DPO` control is supported by the module but is
not included in `VARIANTS_TO_RUN`.
"""
        ),
        code(
            r"""
for variant in REQUESTED_VARIANTS:
    print(f"{variant:7s} | {VARIANT_FORMULAS[variant]}")
    print(f"          {VARIANT_DESCRIPTIONS[variant]}")

self_test_command = [
    sys.executable, "-m", "olmo2_bees.train_structured", "self-test",
    "--workspace", str(WORKSPACE),
]
run_streaming(self_test_command, cwd=PROJECT_ROOT)
"""
        ),
        markdown(
            r"""
## Stage 1 — lossless OLMo chat tokenization and segment alignment

The old individual method notebooks expected Intel-Orca string fields. The available final OLMo
dataset instead stores conversational `prompt`, `chosen`, and `rejected` messages plus
`chosen_segments` and `rejected_segments`. This stage uses OLMo's official chat template, verifies
the prompt prefix, assigns content tokens to segment IDs, checks ranks/scores, and refuses to
truncate. A one-character segment that shares a BPE boundary token is retained through deterministic
boundary ownership.
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

prepared_manifest = json.loads(PREPARED_MANIFEST.read_text())
assert prepared_manifest["lossless"] is True
assert prepared_manifest["split_counts"] == {"train": 6000, "test": 1891}
assert prepared_manifest["statistics"]["max_pair_tokens"] <= MAX_LENGTH
assert prepared_manifest["requested_variants"] == list(REQUESTED_VARIANTS)
print(json.dumps(prepared_manifest, indent=2))
"""
        ),
        code(
            r"""
from datasets import load_from_disk

prepared = load_from_disk(str(PREPARED_DATASET))
required_columns = {
    "dataset_index", "row_id", "source_index",
    "chosen_input_ids", "chosen_segment_ids", "chosen_segment_scores", "chosen_segment_ranks",
    "rejected_input_ids", "rejected_segment_ids", "rejected_segment_scores", "rejected_segment_ranks",
}
assert required_columns.issubset(prepared["train"].column_names)
assert len(prepared["train"]) == 6000 and len(prepared["test"]) == 1891

sample = prepared["train"][0]
for side in ("chosen", "rejected"):
    assert len(sample[f"{side}_input_ids"]) == len(sample[f"{side}_segment_ids"])
    segment_count = len(sample[f"{side}_segment_scores"])
    assert sorted(sample[f"{side}_segment_ranks"]) == list(range(1, segment_count + 1))
    assert set(index for index in sample[f"{side}_segment_ids"] if index >= 0) == set(range(segment_count))

print(prepared)
print("Sample row:", sample["row_id"], "chosen tokens:", len(sample["chosen_input_ids"]),
      "rejected tokens:", len(sample["rejected_input_ids"]))
"""
        ),
        markdown(
            r"""
## Stage 2 — shared frozen-reference segment statistics on both GPUs

All five methods start from the same pinned SFT reference. This two-process pass computes each
segment's reference log-probability once, using a chunked vocabulary projection so full
`sequence × 100,352-vocabulary` logits are never materialized. Each rank appends its own resumable
JSONL shard; the completed shards are coverage-checked and merged into one shared Arrow cache.
"""
        ),
        code(
            r"""
reference_command = [
    sys.executable, "-m", "accelerate.commands.launch",
    "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
    "--mixed_precision", "fp16",
    "-m", "olmo2_bees.train_structured", "reference",
    "--workspace", str(WORKSPACE),
    "--dataset-path", str(PREPARED_DATASET),
    "--split", "train",
    "--model-id", MODEL_ID,
    "--model-revision", MODEL_REVISION,
    "--output-dir", str(REFERENCE_CACHE),
]
run_streaming(reference_command, cwd=PROJECT_ROOT)

reference_manifest = json.loads((REFERENCE_CACHE / "reference_manifest.json").read_text())
assert reference_manifest["rows"] == 6000
assert reference_manifest["world_size"] == 2
assert reference_manifest["compute_dtype"] == "float16"
assert reference_manifest["lossless_segment_coverage"] is True
print(json.dumps(reference_manifest, indent=2))
"""
        ),
        markdown(
            r"""
## Stage 3 — train all five variants sequentially

Each subprocess starts from a fresh copy of the pinned OLMo SFT model and occupies both GPUs. The
policy is FSDP2 full-sharded; activation checkpointing/offload and independently sharded vocabulary
head slices keep the smaller GPU within budget. Runs never initialize from a preceding variant.

Completed manifests are skipped. If you enable restart checkpoints, keep GPU order `0,1` unchanged
when resuming because paged optimizer shards are rank-local.
"""
        ),
        code(
            r"""
run_directories = {}
for variant in VARIANTS_TO_RUN:
    slug = variant.lower().replace("-", "_")
    output_dir = RUN_ROOT / slug
    final_model = output_dir / "final"
    manifest_path = output_dir / "training_manifest.json"
    run_directories[variant] = output_dir

    if manifest_path.is_file() and (final_model / "model.safetensors").is_file():
        print(f"[{variant}] complete; skipping: {final_model}")
        continue

    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
        "--mixed_precision", "fp16",
        "-m", "olmo2_bees.train_structured", "train",
        "--workspace", str(WORKSPACE),
        "--dataset-path", str(PREPARED_DATASET),
        "--reference-cache", str(REFERENCE_CACHE),
        "--train-split", "train",
        "--model-id", MODEL_ID,
        "--model-revision", MODEL_REVISION,
        "--output-dir", str(output_dir),
        "--run-name", f"olmo2-1b-segmented-{slug}",
        "--variant", variant,
        "--epochs", str(EPOCHS),
        "--learning-rate", str(LEARNING_RATE),
        "--beta", str(BETA),
        "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--logging-steps", str(LOGGING_STEPS),
        "--save-steps", str(SAVE_STEPS),
        "--seed", str(SEED),
    ]
    if RESUME and SAVE_STEPS > 0:
        command.append("--resume")
    run_streaming(command, cwd=PROJECT_ROOT)
"""
        ),
        markdown(
            r"""
## Verify and compare run manifests

The training process fails unless all policy parameters are trainable, the optimizer is paged
32-bit AdamW, the final saved tensors are FP32, and the distributed world size is exactly two.
Set `VERIFY_WEIGHT_HASHES=True` for a complete reread of every final model file.
"""
        ),
        code(
            r"""
VERIFY_WEIGHT_HASHES = False
manifests = {}

for variant in VARIANTS_TO_RUN:
    output_dir = run_directories[variant]
    manifest = json.loads((output_dir / "training_manifest.json").read_text())
    final_model = Path(manifest["final_model"])
    assert manifest["variant"] == variant
    assert manifest["objective_formula"] == VARIANT_FORMULAS[variant]
    assert manifest["full_parameter_training"] is True
    assert manifest["parameters"]["trainable_parameters"] == manifest["parameters"]["total_parameters"]
    assert manifest["peft_or_lora"] is False and manifest["weight_quantization"] is None
    assert manifest["optimizer_is_paged"] is True and manifest["optimizer_state_bits"] == 32
    assert manifest["master_parameter_dtype"] == "float32"
    assert manifest["compute_dtype"] == "float16"
    assert manifest["saved_weight_dtypes"] == ["F32"]
    assert manifest["parallelism"] == "FSDP2_FULL_SHARD" and manifest["world_size"] == 2
    assert manifest["reference_segment_log_probs_precomputed"] is True
    assert manifest["weight_files_sha256"]
    if VERIFY_WEIGHT_HASHES:
        for filename, digest in manifest["weight_files_sha256"].items():
            assert sha256_file(final_model / filename) == digest
    manifests[variant] = manifest

print("Verified variants:", list(manifests))
"""
        ),
        code(
            r"""
import pandas as pd
from IPython.display import display

summary_rows = []
for variant, manifest in manifests.items():
    metrics = manifest["metrics"]
    peaks = manifest["peak_cuda_memory"]
    summary_rows.append({
        "variant": variant,
        "train_loss": metrics.get("train_loss"),
        "runtime_minutes": (metrics.get("train_runtime") or 0.0) / 60.0,
        "samples_per_second": metrics.get("train_samples_per_second"),
        "gpu0_peak_reserved_GiB": peaks[0]["reserved_gib"],
        "gpu1_peak_reserved_GiB": peaks[1]["reserved_gib"],
        "final_model": manifest["final_model"],
    })

summary = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)
display(summary)
summary.to_csv(ARTIFACT_ROOT / "all_methods_training_summary.csv", index=False)
print("Saved:", ARTIFACT_ROOT / "all_methods_training_summary.csv")
"""
        ),
        markdown(
            r"""
## Outputs

Each variant has an independent full FP32 model under
`artifacts/olmo2_bees/structured_all_methods/runs/<variant>/final` and a manifest containing the
exact objective, hyperparameters, package versions, memory peaks, metrics, and weight hashes.

Training loss is not enough to select a winner across different objectives. Evaluate every final
model on the same held-out preference and general-capability suites before choosing a deployment
checkpoint; the existing OLMo DPO notebook shows the workspace's fail-closed evaluation pattern.
"""
        ),
    ]
    return notebook


def main() -> None:
    notebook = build_notebook()
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
