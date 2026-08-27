from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def notebook_one():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10+"},
    }
    nb["cells"] = [
        markdown(
            r"""
# 1 — Create a BeeS-selected UltraFeedback dataset for OLMo 2 1B

This notebook reproduces the three BeeS stages for
`HuggingFaceH4/ultrafeedback_binarized`: in-distribution proxy DPO, implicit/external
margin calculation, and Bayesian aggregation. It uses the original UltraFeedback GPT-4 judge-score
margin as the external signal and a full-parameter OLMo 2 1B proxy as the implicit signal.

The workflow is intentionally conservative:

- no LoRA, PEFT, QLoRA, or 4/8-bit model weights;
- no response truncation—over-length pairs are removed before scoring;
- exact dataset/model revisions are recorded;
- proxy training and scoring use both GPUs;
- long stages are checkpointed or append-only and can be resumed;
- every cache and artifact stays inside the current `VPDPO` workspace.

Expect proxy training and scoring all ~61k preference pairs to take a long time on RTX 20-series
GPUs. That is normal; do not interrupt a stage merely because it is slow.
"""
        ),
        code(
            r"""
from pathlib import Path
import json
import os
import subprocess
import sys


def locate_bees() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd / "BeeS", cwd.parent, cwd.parent / "BeeS"]
    for candidate in candidates:
        if (candidate / "ReadME.md").exists() and (candidate / "sampler.py").exists():
            return candidate
    raise FileNotFoundError("Open this notebook from VPDPO, BeeS, or BeeS/notebooks")


PROJECT_ROOT = locate_bees()
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from olmo2_bees.common import configure_workspace, run_streaming, sha256_file, write_json

CACHE_ENV = configure_workspace(WORKSPACE)
print("BeeS repo:", PROJECT_ROOT)
print("Workspace:", WORKSPACE)
print(json.dumps(CACHE_ENV, indent=2))
"""
        ),
        markdown(
            r"""
## Install the pinned runtime

The cache is workspace-local. PyTorch is deliberately not reinstalled because CUDA wheels are
system-specific; the next cell verifies the existing CUDA build. Set `INSTALL_DEPENDENCIES=False`
after the first successful installation if desired.
"""
        ),
        code(
            r"""
INSTALL_DEPENDENCIES = True

if INSTALL_DEPENDENCIES:
    run_streaming(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements-olmo2.txt")],
        cwd=PROJECT_ROOT,
    )
"""
        ),
        code(
            r"""
from olmo2_bees.common import assert_two_turing_gpus, package_versions

gpus = assert_two_turing_gpus()
versions = package_versions(
    ["torch", "transformers", "datasets", "accelerate", "trl", "bitsandbytes", "tqdm"]
)
print(json.dumps({"gpus": gpus, "packages": versions}, indent=2))

assert all(not gpu["bf16_supported"] for gpu in gpus), (
    "This notebook is tuned for the installed RTX 20-series pair. Review precision settings if hardware changed."
)
"""
        ),
        markdown(
            r"""
## Configuration

`MAX_LENGTH=1024` is sized to the smaller 8 GB GPU. Pairs above that limit are excluded losslessly;
the notebook never clips an answer. `SELECTION_SIZE=6000` is about 10% of the unfiltered training
split, matching the high-data-efficiency UltraFeedback setting discussed in the BeeS paper.
"""
        ),
        code(
            r"""
MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
MAX_LENGTH = 1024
PROXY_SEED_SIZE = 2_000
SELECTION_SIZE = 6_000
SEED = 42
NUM_PROC = min(8, os.cpu_count() or 1)

ARTIFACTS = WORKSPACE / "artifacts" / "olmo2_bees"
DATA_ROOT = ARTIFACTS / "dataset_work"
PREPARED = DATA_ROOT / "prepared"
PROXY_SEED = DATA_ROOT / "proxy_seed"
PREPARE_METADATA = DATA_ROOT / "prepare_metadata.json"
PROXY_RUN = ARTIFACTS / "proxy_dpo"
PROXY_MODEL = PROXY_RUN / "final"
SCORES_DIR = ARTIFACTS / "implicit_scores"
SCORES_FILE = SCORES_DIR / "implicit_scores.jsonl"
SELECTED_DATASET = ARTIFACTS / "ultrafeedback_bees_olmo2_1b"

for directory in (ARTIFACTS, DATA_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

print(json.dumps({
    "prepared": str(PREPARED),
    "proxy_seed": str(PROXY_SEED),
    "proxy_model": str(PROXY_MODEL),
    "scores": str(SCORES_FILE),
    "selected_dataset": str(SELECTED_DATASET),
}, indent=2))
"""
        ),
        markdown(
            r"""
## Stage A — Download, normalize, validate, and tokenize

The source revision is resolved once and written to `prepare_metadata.json`. The source's full
dialogues are converted to TRL's `{prompt, chosen-completion, rejected-completion}` conversational
schema before OLMo's official tokenizer/chat template measures lengths.
"""
        ),
        code(
            r"""
prepare_command = [
    sys.executable, "-m", "olmo2_bees.prepare_dataset",
    "--workspace", str(WORKSPACE),
    "--output-root", str(DATA_ROOT),
    "--model-id", MODEL_ID,
    "--dataset-id", DATASET_ID,
    "--max-length", str(MAX_LENGTH),
    "--seed-size", str(PROXY_SEED_SIZE),
    "--seed", str(SEED),
    "--num-proc", str(NUM_PROC),
]
run_streaming(prepare_command, cwd=PROJECT_ROOT)

prepare_metadata = json.loads(PREPARE_METADATA.read_text())
print(json.dumps(prepare_metadata, indent=2))
"""
        ),
        markdown(
            r"""
## Prefetch the pinned OLMo checkpoint once

The upstream checkpoint is downloaded in this single notebook process before either distributed
job starts. Both ranks then reuse the same workspace-local Hub snapshot instead of racing to fetch a
roughly 6 GB model.
"""
        ),
        code(
            r"""
from huggingface_hub import snapshot_download

model_snapshot = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=prepare_metadata["model_revision"],
))
assert str(model_snapshot).startswith(str((WORKSPACE / ".cache").resolve()))
print("Pinned model snapshot:", model_snapshot)
"""
        ),
        code(
            r"""
from datasets import load_from_disk
from tqdm.auto import tqdm

prepared = load_from_disk(str(PREPARED))
proxy_seed = load_from_disk(str(PROXY_SEED))
print(prepared)
print(proxy_seed)

# Explicit tqdm audit in addition to the progress bars emitted by datasets.map/filter.
for split in ("train", "test"):
    for row in tqdm(prepared[split], desc=f"Auditing {split}"):
        assert row["max_pair_tokens"] <= MAX_LENGTH
        assert row["prompt_tokens"] < row["chosen_tokens"]
        assert row["prompt_tokens"] < row["rejected_tokens"]
        assert row["prompt"][-1]["role"] == "user"
        assert row["chosen"][0]["role"] == "assistant"
        assert row["rejected"][0]["role"] == "assistant"

print("Lossless schema/length audit passed.")
"""
        ),
        markdown(
            r"""
## Stage B — Train the in-distribution proxy with full-parameter DPO

Both GPUs run FSDP full sharding simultaneously. The effective batch size is
`1 × 2 GPUs × 8 accumulation = 16`. Parameters and updates remain FP32; FP16 is used only for
loss-scaled mixed-precision compute. The SFT reference log-probabilities are computed before
optimization, allowing the reference model to be released. Paged AdamW uses 32-bit optimizer
states; it does not quantize the model.

Restartable FSDP checkpoints are intentionally large because they preserve full FP32 model and
optimizer state. Only two are retained; budget roughly 55–65 GiB for this proxy run.
"""
        ),
        code(
            r"""
if (PROXY_RUN / "training_manifest.json").is_file() and (PROXY_MODEL / "model.safetensors").is_file():
    print("Proxy final model already exists; skipping training:", PROXY_MODEL)
else:
    proxy_command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
        "--mixed_precision", "fp16",
        "-m", "olmo2_bees.train_dpo",
        "--workspace", str(WORKSPACE),
        "--dataset-path", str(PROXY_SEED),
        "--model-id", MODEL_ID,
        "--model-revision", prepare_metadata["model_revision"],
        "--output-dir", str(PROXY_RUN),
        "--run-name", "olmo2-1b-bees-proxy",
        "--max-length", str(MAX_LENGTH),
        "--epochs", "2",
        "--learning-rate", "5e-7",
        "--beta", "0.1",
        "--gradient-accumulation-steps", "8",
        "--save-steps", "100",
        "--num-proc", "4",
        "--seed", str(SEED),
        "--resume",
    ]
    run_streaming(proxy_command, cwd=PROJECT_ROOT)

manifest = json.loads((PROXY_RUN / "training_manifest.json").read_text())
assert manifest["full_parameter_training"] is True
assert manifest["peft_or_lora"] is False
assert manifest["weight_quantization"] is None
assert manifest["optimizer_is_paged"] is True
assert manifest["optimizer_state_bits"] == 32
assert manifest["optimizer"] == "bitsandbytes.optim.PagedAdamW32bit"
assert manifest["fp16_initial_loss_scale"] == 32.0
assert manifest["master_parameter_dtype"] == "float32"
assert manifest["compute_dtype"] == "float16"
assert manifest["saved_weight_dtypes"] == ["F32"]
assert manifest["weight_files_sha256"]
assert manifest["parallelism"] == "FSDP2_FULL_SHARD"
assert manifest["world_size"] == 2
for filename, expected_digest in tqdm(
    manifest["weight_files_sha256"].items(), desc="Verifying proxy FP32 weights"
):
    assert sha256_file(PROXY_MODEL / filename) == expected_digest
print(json.dumps(manifest, indent=2))
"""
        ),
        markdown(
            r"""
## Stage C — Score the implicit reward margin on both GPUs

For each pair, the script computes
`[log π_proxy(chosen|prompt) − log π_SFT(chosen|prompt)] −
 [log π_proxy(rejected|prompt) − log π_SFT(rejected|prompt)]`.
Each rank appends and flushes its own JSONL file, so rerunning the cell resumes from completed rows.
"""
        ),
        code(
            r"""
if SCORES_FILE.is_file() and (SCORES_DIR / "score_manifest.json").is_file():
    print("Merged implicit scores already exist; skipping scoring:", SCORES_FILE)
else:
    score_command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
        "--mixed_precision", "fp16",
        "-m", "olmo2_bees.score_implicit",
        "--workspace", str(WORKSPACE),
        "--dataset-path", str(PREPARED),
        "--split", "train",
        "--reference-model", MODEL_ID,
        "--reference-revision", prepare_metadata["model_revision"],
        "--policy-model", str(PROXY_MODEL),
        "--output-dir", str(SCORES_DIR),
        "--max-length", str(MAX_LENGTH),
    ]
    run_streaming(score_command, cwd=PROJECT_ROOT)

score_manifest = json.loads((SCORES_DIR / "score_manifest.json").read_text())
assert score_manifest["rows"] == len(prepared["train"])
assert score_manifest["world_size"] == 2
assert score_manifest["weight_quantization"] is None
print(json.dumps(score_manifest, indent=2))
"""
        ),
        markdown(
            r"""
## Stage D — BeeS projection, Bayesian aggregation, and selection

This uses BeeS Appendix A.1's lower bound `L = −2`, dynamic upper-bound stopping conditions, the
paper's linear margin-to-probability projection, and Eq. (3). Only rows with positive margins from
both independent sources are eligible; the highest aggregated probabilities become the train split.
"""
        ),
        code(
            r"""
select_command = [
    sys.executable, "-m", "olmo2_bees.select_bees",
    "--workspace", str(WORKSPACE),
    "--prepared-path", str(PREPARED),
    "--scores-path", str(SCORES_FILE),
    "--prepare-metadata", str(PREPARE_METADATA),
    "--output-path", str(SELECTED_DATASET),
    "--selection-size", str(SELECTION_SIZE),
]
run_streaming(select_command, cwd=PROJECT_ROOT)

selected = load_from_disk(str(SELECTED_DATASET))
selection_metadata = json.loads((SELECTED_DATASET / "metadata.json").read_text())
assert len(selected["train"]) == SELECTION_SIZE
assert min(selected["train"]["external_margin"]) > 0
assert min(selected["train"]["implicit_margin"]) > 0
assert selected["train"]["bees_rank"] == list(range(1, SELECTION_SIZE + 1))
print(selected)
print(json.dumps(selection_metadata, indent=2))
"""
        ),
        code(
            r"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(selected["train"]["external_margin"], bins=40)
axes[0].set_title("Selected external margin")
axes[1].hist(selected["train"]["implicit_margin"], bins=40)
axes[1].set_title("Selected implicit margin")
axes[2].hist(selected["train"]["bees_probability"], bins=40)
axes[2].set_title("Selected BeeS probability")
for axis in axes:
    axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print("Dataset ready:", SELECTED_DATASET)
print("Next: run 02_train_olmo2_1b_dpo.ipynb")
"""
        ),
    ]
    return nb


def notebook_two():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10+"},
    }
    nb["cells"] = [
        markdown(
            r"""
# 2 — Full-parameter DPO of OLMo 2 1B on the BeeS dataset

This notebook DPO post-trains `allenai/OLMo-2-0425-1B-SFT`; it does not perform another SFT
objective. Every model parameter is updated. The two GPUs train simultaneously with FSDP full
sharding, sized for the smaller 8 GB RTX 2080 SUPER.

Memory reductions are training-system techniques, not lossy model compression: FP32 master
parameters and updates with loss-scaled FP16 compute, full parameter/gradient sharding, activation
checkpointing/offloading, reference-log-probability precomputation, and paged **32-bit** AdamW
states. LoRA, PEFT, QLoRA, and 4/8-bit weight loading are explicitly rejected by the training script.
The conservative dynamic loss scale starts at 32, which was validated with actual 1,024-token pairs
on these GPUs. Every manifest records peak CUDA allocation/reservation for both ranks.

The original SFT model is never overwritten. A checkpoint is only recorded as approved after it
passes held-out preference checks and a mandatory multi-GPU general benchmark comparison.
"""
        ),
        code(
            r"""
from pathlib import Path
import hashlib
import json
import os
import sys


def locate_bees() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd / "BeeS", cwd.parent, cwd.parent / "BeeS"):
        if (candidate / "ReadME.md").exists() and (candidate / "sampler.py").exists():
            return candidate
    raise FileNotFoundError("Open this notebook from VPDPO, BeeS, or BeeS/notebooks")


PROJECT_ROOT = locate_bees()
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from olmo2_bees.common import (
    assert_two_turing_gpus,
    configure_workspace,
    package_versions,
    run_streaming,
    sha256_file,
    write_json,
)

configure_workspace(WORKSPACE)
print("BeeS repo:", PROJECT_ROOT)
print("Workspace:", WORKSPACE)
"""
        ),
        code(
            r"""
INSTALL_DEPENDENCIES = True

if INSTALL_DEPENDENCIES:
    run_streaming(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements-olmo2.txt")],
        cwd=PROJECT_ROOT,
    )

gpus = assert_two_turing_gpus()
versions = package_versions(
    ["torch", "transformers", "datasets", "accelerate", "trl", "bitsandbytes", "lm_eval"]
)
print(json.dumps({"gpus": gpus, "packages": versions}, indent=2))
"""
        ),
        markdown(
            r"""
## Configuration and immutable inputs

The model and dataset revisions captured by notebook 1 are reused. The conservative `5e-7`,
`β=0.1`, two-epoch recipe follows the upstream BeeS defaults and established OLMo DPO scale.
"""
        ),
        code(
            r"""
MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
MAX_LENGTH = 1024
SEED = 42

ARTIFACTS = WORKSPACE / "artifacts" / "olmo2_bees"
DATA_ROOT = ARTIFACTS / "dataset_work"
SELECTED_DATASET = ARTIFACTS / "ultrafeedback_bees_olmo2_1b"
FINAL_RUN = ARTIFACTS / "olmo2_1b_dpo_full"
FINAL_MODEL = FINAL_RUN / "final"
PREFERENCE_EVAL = ARTIFACTS / "preference_eval"
BASELINE_LM_EVAL_ROOT = ARTIFACTS / "lm_eval_baseline"
CANDIDATE_LM_EVAL_ROOT = ARTIFACTS / "lm_eval_candidate"
QUALITY_REPORT = ARTIFACTS / "quality_gate.json"
APPROVED_POINTER = ARTIFACTS / "approved_model.json"

prepare_metadata = json.loads((DATA_ROOT / "prepare_metadata.json").read_text())
selection_metadata = json.loads((SELECTED_DATASET / "metadata.json").read_text())
MODEL_REVISION = prepare_metadata["model_revision"]
assert prepare_metadata["model_id"] == MODEL_ID
assert prepare_metadata["max_length"] == MAX_LENGTH
print(json.dumps({
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dataset": str(SELECTED_DATASET),
    "output": str(FINAL_MODEL),
}, indent=2))
"""
        ),
        markdown(
            r"""
## Fail-closed dataset and hardware audit

The selected pairs must fit without truncation and must have positive external and implicit margins.
GPU 1 (8 GB) determines the per-device batch and sequence limits for both FSDP ranks.
"""
        ),
        code(
            r"""
from datasets import load_from_disk
from tqdm.auto import tqdm

dataset = load_from_disk(str(SELECTED_DATASET))
required = {"prompt", "chosen", "rejected", "external_margin", "implicit_margin", "bees_probability"}
assert required.issubset(dataset["train"].column_names)
assert len(dataset["train"]) == selection_metadata["selected_train_rows"]

for row in tqdm(dataset["train"], desc="Final pre-training audit"):
    assert row["max_pair_tokens"] <= MAX_LENGTH
    assert row["external_margin"] > 0.0
    assert row["implicit_margin"] > 0.0
    assert row["chosen"][0]["role"] == "assistant"
    assert row["rejected"][0]["role"] == "assistant"

minimum_gpu_gib = min(gpu["memory_gib"] for gpu in gpus)
assert minimum_gpu_gib >= 7.5
print(dataset)
print(f"Smaller GPU usable capacity: {minimum_gpu_gib:.2f} GiB")
print("Audit passed: no selected example requires truncation.")
"""
        ),
        markdown(
            r"""
## Train on both GPUs

The global batch is 16 preference pairs. Reference log-probabilities are precomputed while the model
still equals the pinned SFT checkpoint, then the same model becomes the trainable policy. This is
mathematically the normal fixed-reference DPO objective without retaining a second 1B model during
backpropagation.
"""
        ),
        code(
            r"""
if (FINAL_RUN / "training_manifest.json").is_file() and (FINAL_MODEL / "model.safetensors").is_file():
    print("Final model already exists; skipping training:", FINAL_MODEL)
else:
    train_command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
        "--mixed_precision", "fp16",
        "-m", "olmo2_bees.train_dpo",
        "--workspace", str(WORKSPACE),
        "--dataset-path", str(SELECTED_DATASET),
        "--train-split", "train",
        "--eval-split", "test",
        "--model-id", MODEL_ID,
        "--model-revision", MODEL_REVISION,
        "--output-dir", str(FINAL_RUN),
        "--run-name", "olmo2-1b-ultrafeedback-bees-dpo",
        "--max-length", str(MAX_LENGTH),
        "--epochs", "2",
        "--learning-rate", "5e-7",
        "--beta", "0.1",
        "--gradient-accumulation-steps", "8",
        "--logging-steps", "1",
        "--save-steps", "100",
        "--eval-steps", "100",
        "--num-proc", "4",
        "--seed", str(SEED),
        "--resume",
    ]
    run_streaming(train_command, cwd=PROJECT_ROOT)

manifest = json.loads((FINAL_RUN / "training_manifest.json").read_text())
assert manifest["full_parameter_training"] is True
assert manifest["parameters"]["trainable_parameters"] == manifest["parameters"]["total_parameters"]
assert manifest["peft_or_lora"] is False
assert manifest["weight_quantization"] is None
assert manifest["optimizer_is_paged"] is True
assert manifest["optimizer_state_bits"] == 32
assert manifest["optimizer"] == "bitsandbytes.optim.PagedAdamW32bit"
assert manifest["fp16_initial_loss_scale"] == 32.0
assert manifest["master_parameter_dtype"] == "float32"
assert manifest["compute_dtype"] == "float16"
assert manifest["saved_weight_dtypes"] == ["F32"]
assert manifest["weight_files_sha256"]
assert manifest["parallelism"] == "FSDP2_FULL_SHARD"
assert manifest["world_size"] == 2
for filename, expected_digest in tqdm(
    manifest["weight_files_sha256"].items(), desc="Verifying final FP32 weights"
):
    assert sha256_file(FINAL_MODEL / filename) == expected_digest

weight_manifest = json.dumps(manifest["weight_files_sha256"], sort_keys=True).encode("utf-8")
model_fingerprint = hashlib.sha256(weight_manifest).hexdigest()
BASELINE_LM_EVAL = BASELINE_LM_EVAL_ROOT / MODEL_REVISION[:16]
CANDIDATE_LM_EVAL = CANDIDATE_LM_EVAL_ROOT / model_fingerprint[:16]
print("Verified model fingerprint:", model_fingerprint)
print(json.dumps(manifest, indent=2))
"""
        ),
        markdown(
            r"""
## Held-out preference evaluation

Both GPUs compare the trained policy with the untouched pinned SFT reference on the official filtered
`test_prefs` split. The result supplies DPO reward accuracy, mean reward margin, and direct
length-normalized preference accuracy for both models.
"""
        ),
        code(
            r"""
preference_scores = PREFERENCE_EVAL / "implicit_scores.jsonl"
if preference_scores.is_file() and (PREFERENCE_EVAL / "score_manifest.json").is_file():
    print("Held-out preference scores already exist:", preference_scores)
else:
    eval_command = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
        "--mixed_precision", "fp16",
        "-m", "olmo2_bees.score_implicit",
        "--workspace", str(WORKSPACE),
        "--dataset-path", str(SELECTED_DATASET),
        "--split", "test",
        "--reference-model", MODEL_ID,
        "--reference-revision", MODEL_REVISION,
        "--policy-model", str(FINAL_MODEL),
        "--output-dir", str(PREFERENCE_EVAL),
        "--max-length", str(MAX_LENGTH),
    ]
    run_streaming(eval_command, cwd=PROJECT_ROOT)

# Early gate before spending time on the full benchmark suite.
preference_gate_command = [
    sys.executable, "-m", "olmo2_bees.quality_gate",
    "--workspace", str(WORKSPACE),
    "--preference-scores", str(preference_scores),
    "--output", str(ARTIFACTS / "preference_gate.json"),
]
run_streaming(preference_gate_command, cwd=PROJECT_ROOT)
"""
        ),
        markdown(
            r"""
## General-capability regression suite

This runs the same deterministic `lm-eval` tasks against the pinned SFT baseline and the DPO
candidate, using both GPUs in data-parallel inference. It is mandatory because preference accuracy
alone cannot establish that general or instruction-following accuracy was maintained. Full evaluation
is slow.

The gate permits at most a 1 percentage-point drop on any task and at most a 0.2 point macro drop.
Change the task list only if you have a documented alternative acceptance suite.
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


def benchmark_complete(output_dir: Path) -> bool:
    try:
        return set(BENCHMARK_TASKS).issubset(task_metrics(output_dir))
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
        return False


def run_lm_eval(model_args: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
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
    ]
    run_streaming(command, cwd=PROJECT_ROOT)


if not benchmark_complete(BASELINE_LM_EVAL):
    run_lm_eval(
        f"pretrained={MODEL_ID},revision={MODEL_REVISION},dtype=float16",
        BASELINE_LM_EVAL,
    )
else:
    print("Baseline lm-eval results already exist:", BASELINE_LM_EVAL)

if not benchmark_complete(CANDIDATE_LM_EVAL):
    run_lm_eval(f"pretrained={FINAL_MODEL},dtype=float16", CANDIDATE_LM_EVAL)
else:
    print("Candidate lm-eval results already exist:", CANDIDATE_LM_EVAL)
"""
        ),
        markdown(
            r"""
## Final fail-closed approval gate

This cell stops with an error on regression. Only a passing run writes `approved_model.json`; the
pointer references the full, unquantized saved model and its evidence report without duplicating
several gigabytes of weights.
"""
        ),
        code(
            r"""
gate_command = [
    sys.executable, "-m", "olmo2_bees.quality_gate",
    "--workspace", str(WORKSPACE),
    "--preference-scores", str(preference_scores),
    "--output", str(QUALITY_REPORT),
]
gate_command.extend([
    "--baseline-lm-eval", str(BASELINE_LM_EVAL),
    "--candidate-lm-eval", str(CANDIDATE_LM_EVAL),
    "--max-task-drop", "0.01",
    "--max-macro-drop", "0.002",
])
for task in BENCHMARK_TASKS:
    gate_command.extend(["--required-benchmark-task", task])
run_streaming(gate_command, cwd=PROJECT_ROOT)

quality = json.loads(QUALITY_REPORT.read_text())
assert quality["passed"] is True
write_json(APPROVED_POINTER, {
    "approved": True,
    "model_path": str(FINAL_MODEL.resolve()),
    "base_model": MODEL_ID,
    "base_model_revision": MODEL_REVISION,
    "dataset_path": str(SELECTED_DATASET.resolve()),
    "training_manifest": str((FINAL_RUN / "training_manifest.json").resolve()),
    "model_fingerprint": model_fingerprint,
    "weight_files_sha256": manifest["weight_files_sha256"],
    "quality_report": str(QUALITY_REPORT.resolve()),
    "weight_quantization": None,
    "peft_or_lora": False,
})
print("Approved full-parameter model:", FINAL_MODEL)
print("Approval evidence:", APPROVED_POINTER)
"""
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        NOTEBOOK_DIR / "01_create_bees_ultrafeedback_olmo2.ipynb": notebook_one(),
        NOTEBOOK_DIR / "02_train_olmo2_1b_dpo.ipynb": notebook_two(),
    }
    for path, notebook in outputs.items():
        nbf.write(notebook, path)
        print(path)


if __name__ == "__main__":
    main()
