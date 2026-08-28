from __future__ import annotations

import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "olmo_bees_sampo_kaggle_t4x2.ipynb"

NOTEBOOKS = (
    {
        "variant": "B_norm-DPO",
        "display": "Method B_norm-DPO",
        "slug": "b_norm_dpo",
        "filename": "Method B_norm-DPO training notebook - OLMo2-1B-SFT Full T4x2.ipynb",
    },
    {
        "variant": "B_norm-VDPO",
        "display": "Method B_norm-VDPO",
        "slug": "b_norm_vdpo",
        "filename": "Method B_norm-VDPO training notebook - OLMo2-1B-SFT Full T4x2.ipynb",
    },
)


def source_lines(source: str) -> list[str]:
    return (dedent(source).strip("\n") + "\n").splitlines(keepends=True)


def set_cell(notebook: dict, index: int, source: str) -> None:
    notebook["cells"][index]["source"] = source_lines(source)
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["execution_count"] = None
        notebook["cells"][index]["outputs"] = []


def render(source: str, config: dict[str, str]) -> str:
    replacements = {
        "@@VARIANT@@": config["variant"],
        "@@DISPLAY@@": config["display"],
        "@@SLUG@@": config["slug"],
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def repair_mojibake(value):
    if isinstance(value, dict):
        return {key: repair_mojibake(item) for key, item in value.items()}
    if isinstance(value, list):
        return [repair_mojibake(item) for item in value]
    if isinstance(value, str):
        for broken, correct in {
            "â€”": "—",
            "â†’": "→",
            "Ã—": "×",
        }.items():
            value = value.replace(broken, correct)
    return value


def build_notebook(template: dict, config: dict[str, str]) -> dict:
    notebook = repair_mojibake(copy.deepcopy(template))
    set_cell(
        notebook,
        0,
        render(
            r"""
            # OLMo 2 1B — @@DISPLAY@@ on Kaggle T4 x2

            This standalone notebook trains one full-parameter
            `allenai/OLMo-2-0425-1B-SFT` policy with **both** Kaggle T4 GPUs through
            two-process FSDP2 full sharding. It keeps FP32 master weights and paged FP32
            AdamW states, uses FP16 only for compute, and does not use LoRA, PEFT, QLoRA,
            or weight quantization.

            This is the normalized Method B formulation. For ranked segment `k`, `a_k` is
            its score, `h_k` is its policy/reference summed token log-ratio, and `n_k` is
            its number of owned response tokens. The sole Method B change is:

            \[
            \boxed{v_k^{\mathrm{norm}}=\beta a_k\bar h_k
            =\beta a_k\frac{h_k}{n_k}.}
            \]

            The DPO or VDPO response core is unchanged. Only the utilities passed to the
            Method B Plackett–Luce structural-coherence term use `h_k / n_k`, preventing
            segment utility from growing additively with token count.

            Before running, choose **GPU T4 x2** under *Settings → Accelerator* and enable
            Internet, or attach this repository and dataset as a Kaggle Dataset input.
            Following the current repository notebooks, the design budget assumes a 12-hour GPU
            session and less than 20 GB under `/kaggle/working`. The notebook can either compute
            the compact reference-segment cache and train in one session, or persist the reference
            cache first and consume it from a second session.

            Storage policy:

            - `/kaggle/temp`: repository clone, package/Hugging Face caches, prepared dataset,
              and transient reference/cache files.
            - `/kaggle/working`: one reusable reference cache, one final FP32 model, or small
              evaluation results—never restart optimizer/FSDP state.
            - Restart checkpoints are disabled and stale optimizer, scheduler, scaler, RNG,
              and FSDP checkpoint files are removed before committing output.
            """,
            config,
        ),
    )
    set_cell(
        notebook,
        1,
        """
        ## 1. Run switches

        The default performs reference preparation plus training. If that approaches Kaggle's
        session limit, first set `BUILD_REFERENCE_ONLY=True` and `RUN_TRAINING=False`, commit the
        output, attach it to a fresh notebook session, then reverse those switches. Training and
        evaluation are deliberately separated.
        """,
    )
    set_cell(
        notebook,
        2,
        render(
            r"""
            VARIANT = "@@VARIANT@@"

            INSTALL_DEPENDENCIES = True
            BUILD_REFERENCE_ONLY = False
            RUN_TRAINING = True
            RUN_HELDOUT_EVAL = False
            RUN_LM_EVAL = False

            REPO_URL = "https://github.com/Soobiwan/VP-DPO.git"
            REPO_REF = "main"

            if not any((BUILD_REFERENCE_ONLY, RUN_TRAINING, RUN_HELDOUT_EVAL, RUN_LM_EVAL)):
                raise ValueError("Enable reference preparation, training, or evaluation")
            if BUILD_REFERENCE_ONLY and (RUN_TRAINING or RUN_HELDOUT_EVAL or RUN_LM_EVAL):
                raise ValueError("BUILD_REFERENCE_ONLY must run by itself")
            if RUN_TRAINING and (RUN_HELDOUT_EVAL or RUN_LM_EVAL):
                raise ValueError(
                    "Kaggle safety guard: train and evaluate in separate sessions. "
                    "Commit the training output, attach it as an input, then rerun with "
                    "RUN_TRAINING=False."
                )
            """,
            config,
        ),
    )
    set_cell(notebook, 3, "## 2. Locate or clone the repository, then install the pinned environment")
    set_cell(
        notebook,
        4,
        render(
            r"""
            from pathlib import Path
            import gc
            import json
            import os
            import shutil
            import subprocess
            import sys
            import tempfile


            ON_KAGGLE = Path("/kaggle").is_dir()
            SCRATCH_PARENT = Path("/kaggle/temp") if ON_KAGGLE else Path(tempfile.gettempdir())
            SCRATCH_ROOT = SCRATCH_PARENT / "vpdpo-@@SLUG@@-scratch"
            SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)


            def is_repo(path: Path) -> bool:
                return (path / "BeeS" / "olmo2_bees" / "train_structured.py").is_file()


            def locate_repo() -> Path | None:
                cwd = Path.cwd().resolve()
                for candidate in (cwd, *cwd.parents):
                    if is_repo(candidate):
                        return candidate
                kaggle_input = Path("/kaggle/input")
                if kaggle_input.is_dir():
                    for match in kaggle_input.rglob("train_structured.py"):
                        candidate = match.resolve().parents[2]
                        if is_repo(candidate):
                            return candidate
                return None


            SOURCE_REPO = locate_repo()
            if SOURCE_REPO is None:
                clone_target = SCRATCH_ROOT / "source"
                if clone_target.exists() and not is_repo(clone_target):
                    shutil.rmtree(clone_target)
                if not clone_target.exists():
                    subprocess.check_call([
                        "git", "clone", "--depth", "1", "--branch", REPO_REF,
                        REPO_URL, str(clone_target),
                    ])
                SOURCE_REPO = clone_target

            PROJECT_ROOT = SOURCE_REPO / "BeeS"
            requirements = PROJECT_ROOT / "requirements-olmo2.txt"
            if INSTALL_DEPENDENCIES:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
                    "--no-cache-dir", "-r", str(requirements),
                ])

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

            from olmo2_bees.common import (
                assert_two_turing_gpus,
                configure_workspace,
                package_versions,
                run_streaming,
                sha256_file,
                write_json,
            )
            from olmo2_bees.structured_preference import (
                NORMALIZED_METHOD_B_VARIANTS,
                VARIANT_DESCRIPTIONS,
                VARIANT_FORMULAS,
            )

            if VARIANT not in NORMALIZED_METHOD_B_VARIANTS:
                raise ValueError(f"Repository does not provide normalized Method B variant {VARIANT!r}")

            CACHE_ENV = configure_workspace(SCRATCH_ROOT)

            source_name = "ultrafeedback_bees_olmo2_1b_segmented_final.jsonl"
            source_candidates = [SOURCE_REPO / source_name]
            if ON_KAGGLE:
                source_candidates.extend(Path("/kaggle/input").rglob(source_name))
            SOURCE_JSONL = max(
                (path for path in source_candidates if path.is_file()),
                key=lambda path: path.stat().st_size,
                default=None,
            )
            if SOURCE_JSONL is None or SOURCE_JSONL.stat().st_size < 1_000_000:
                raise FileNotFoundError(
                    f"Could not find the full {source_name}. Attach the repository/data or enable Internet."
                )

            print("Repository:", SOURCE_REPO)
            print("Source dataset:", SOURCE_JSONL)
            print(json.dumps(package_versions([
                "torch", "transformers", "datasets", "accelerate", "trl",
                "bitsandbytes", "safetensors", "lm_eval",
            ]), indent=2))
            """,
            config,
        ),
    )
    set_cell(notebook, 5, "## 3. Kaggle T4 x2 and host-resource audit")
    set_cell(
        notebook,
        6,
        r"""
        import torch

        gpus = assert_two_turing_gpus()
        assert len(gpus) == 2
        if ON_KAGGLE:
            assert all("T4" in gpu["name"] for gpu in gpus), gpus
            assert min(gpu["memory_gib"] for gpu in gpus) >= 14.5, gpus
            meminfo = Path("/proc/meminfo").read_text()
            host_kib = int(next(
                line.split()[1] for line in meminfo.splitlines() if line.startswith("MemTotal:")
            ))
            assert host_kib / 2**20 >= 25, "Expected a Kaggle GPU worker with about 29 GB RAM"
        assert all(not gpu["bf16_supported"] for gpu in gpus)
        print(json.dumps(gpus, indent=2))
        print("Host RAM GiB:", round(host_kib / 2**20, 2) if ON_KAGGLE else "not enforced")
        print("Effective global pair batch: 16 (1 × 2 GPUs × 8 accumulation)")
        """,
    )
    set_cell(notebook, 7, "## 4. Objective, hyperparameters, and storage lifecycle")
    set_cell(
        notebook,
        8,
        render(
            r"""
            MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
            MODEL_REVISION = "0d85a3d037876ce6ac7d4311d994400fc66ac27f"
            MAX_LENGTH = 1024
            EPOCHS = 1.0
            LEARNING_RATE = 5e-7
            BETA = 0.1
            GRADIENT_ACCUMULATION_STEPS = 8
            LOGGING_STEPS = 1
            SAVE_STEPS = 0  # mandatory on Kaggle: final model only, no optimizer/FSDP restart state
            SEED = 42

            assert SAVE_STEPS == 0, "Kaggle output cannot safely retain full FSDP optimizer checkpoints"

            if ON_KAGGLE:
                PERSIST_ROOT = Path("/kaggle/working") / "olmo2_bees_@@SLUG@@"
            else:
                PERSIST_ROOT = SOURCE_REPO / "artifacts" / "kaggle_t4x2" / "@@SLUG@@"
            RUN_DIR = PERSIST_ROOT / "run"
            FINAL_MODEL = RUN_DIR / "final"
            EVAL_ROOT = PERSIST_ROOT / "evaluations"
            PREPARED_DATASET = SCRATCH_ROOT / "prepared"
            SCRATCH_REFERENCE_CACHE = SCRATCH_ROOT / "reference"
            PERSISTED_REFERENCE_CACHE = PERSIST_ROOT / "reference_cache"
            REFERENCE_CACHE = (
                PERSISTED_REFERENCE_CACHE if BUILD_REFERENCE_ONLY else SCRATCH_REFERENCE_CACHE
            )
            BASELINE_LM_EVAL = EVAL_ROOT / "lm_eval_baseline"
            CANDIDATE_LM_EVAL = EVAL_ROOT / "lm_eval_@@SLUG@@"
            PREFERENCE_EVAL = EVAL_ROOT / "preference"
            for path in (PERSIST_ROOT, RUN_DIR, EVAL_ROOT):
                path.mkdir(parents=True, exist_ok=True)


            def safe_remove_tree(path: Path, allowed_root: Path) -> None:
                path = path.resolve()
                allowed_root = allowed_root.resolve()
                if path == allowed_root:
                    raise RuntimeError(f"Refusing to remove the allowed root itself: {path}")
                path.relative_to(allowed_root)
                if path.exists():
                    shutil.rmtree(path)
                    print("Deleted:", path)


            def delete_training_transients() -> None:
                for checkpoint in RUN_DIR.glob("checkpoint-*"):
                    safe_remove_tree(checkpoint, RUN_DIR)
                patterns = (
                    "paged_optimizer_rank_*.pt", "optimizer.pt", "scheduler.pt",
                    "scaler.pt", "rng_state_*.pth", "pytorch_model_fsdp.bin",
                )
                final_resolved = FINAL_MODEL.resolve()
                for pattern in patterns:
                    for path in RUN_DIR.rglob(pattern):
                        if final_resolved not in path.resolve().parents:
                            path.unlink(missing_ok=True)
                            print("Deleted transient:", path)


            def tree_size_gib(path: Path) -> float:
                if not path.exists():
                    return 0.0
                return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**30


            def training_complete() -> bool:
                manifest_path = RUN_DIR / "training_manifest.json"
                if not manifest_path.is_file() or not FINAL_MODEL.is_dir():
                    return False
                manifest = json.loads(manifest_path.read_text())
                return (
                    manifest.get("variant") == VARIANT
                    and manifest.get("objective_formula") == VARIANT_FORMULAS[VARIANT]
                    and manifest.get("model_revision") == MODEL_REVISION
                    and manifest.get("world_size") == 2
                    and manifest.get("restart_checkpoints_enabled") is False
                    and manifest.get("saved_weight_dtypes") == ["F32"]
                )


            delete_training_transients()
            TRAIN_ALREADY_COMPLETE = training_complete()
            NEEDS_DATASET = (
                BUILD_REFERENCE_ONLY
                or (RUN_TRAINING and not TRAIN_ALREADY_COMPLETE)
                or RUN_HELDOUT_EVAL
            )
            print(json.dumps({
                "variant": VARIANT,
                "formula": VARIANT_FORMULAS[VARIANT],
                "description": VARIANT_DESCRIPTIONS[VARIANT],
                "scratch": str(SCRATCH_ROOT),
                "persisted_output": str(PERSIST_ROOT),
                "build_reference_only": BUILD_REFERENCE_ONLY,
                "training_already_complete": TRAIN_ALREADY_COMPLETE,
                "persisted_gib": round(tree_size_gib(PERSIST_ROOT), 3),
                "hyperparameters": {
                    "max_length": MAX_LENGTH,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "beta": BETA,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                },
            }, indent=2))
            """,
            config,
        ),
    )
    set_cell(notebook, 9, "## 5. Normalized Method B objective self-tests")
    set_cell(
        notebook,
        10,
        r"""
        print("Normalized segment utility: v_k^norm = beta * a_k * h_k / n_k")
        print("Active objective:", VARIANT_FORMULAS[VARIANT])
        run_streaming([
            sys.executable, "-m", "olmo2_bees.train_structured", "self-test",
            "--workspace", str(SCRATCH_ROOT),
        ], cwd=PROJECT_ROOT)
        """,
    )
    set_cell(
        notebook,
        11,
        """
        ## 6. Prepare the lossless segmented dataset in scratch space

        This produces the same validated 6,000-row train and 1,891-row untouched test splits as
        the current OLMo BeeS dual-GPU notebooks. It is skipped for an lm-eval-only session.
        """,
    )
    set_cell(
        notebook,
        12,
        r"""
        if NEEDS_DATASET:
            run_streaming([
                sys.executable, "-m", "olmo2_bees.train_structured", "prepare",
                "--workspace", str(SCRATCH_ROOT),
                "--source-jsonl", str(SOURCE_JSONL),
                "--output-dir", str(PREPARED_DATASET),
                "--model-id", MODEL_ID,
                "--model-revision", MODEL_REVISION,
                "--max-length", str(MAX_LENGTH),
            ], cwd=PROJECT_ROOT)

            from datasets import load_from_disk

            prepared = load_from_disk(str(PREPARED_DATASET))
            assert len(prepared["train"]) == 6000
            assert len(prepared["test"]) == 1891
            print(prepared)
            del prepared
            gc.collect()
        else:
            print("Prepared dataset is not needed for this phase")
        """,
    )
    set_cell(notebook, 13, "## 7. Frozen-reference segment log-probability stage")
    set_cell(
        notebook,
        14,
        r"""
        def find_attached_reference_cache() -> Path | None:
            kaggle_input = Path("/kaggle/input")
            if not kaggle_input.is_dir():
                return None
            candidates = []
            for manifest_path in kaggle_input.rglob("reference_manifest.json"):
                try:
                    manifest = json.loads(manifest_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                cache = manifest_path.parent
                if (
                    (cache / "dataset").is_dir()
                    and manifest.get("rows") == 6000
                    and manifest.get("dataset_path") == str(PREPARED_DATASET.resolve())
                    and manifest.get("model_revision") == MODEL_REVISION
                    and manifest.get("world_size") == 2
                    and manifest.get("lossless_segment_coverage") is True
                ):
                    candidates.append(cache)
            candidates.sort(key=lambda path: (len(path.parts), str(path)))
            return candidates[0] if candidates else None


        NEEDS_REFERENCE = BUILD_REFERENCE_ONLY or (RUN_TRAINING and not TRAIN_ALREADY_COMPLETE)
        attached_cache = find_attached_reference_cache() if RUN_TRAINING else None
        if attached_cache is not None:
            REFERENCE_CACHE = attached_cache
            print("Using attached read-only reference cache:", REFERENCE_CACHE)
        elif NEEDS_REFERENCE:
            reference_command = [
                sys.executable, "-m", "accelerate.commands.launch",
                "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
                "--mixed_precision", "fp16",
                "-m", "olmo2_bees.train_structured", "reference",
                "--workspace", str(SCRATCH_ROOT),
                "--dataset-path", str(PREPARED_DATASET),
                "--split", "train",
                "--model-id", MODEL_ID,
                "--model-revision", MODEL_REVISION,
                "--output-dir", str(REFERENCE_CACHE),
            ]
            run_streaming(reference_command, cwd=PROJECT_ROOT)
            reference_manifest = json.loads(
                (REFERENCE_CACHE / "reference_manifest.json").read_text()
            )
            assert reference_manifest["rows"] == 6000
            assert reference_manifest["world_size"] == 2
            assert reference_manifest["compute_dtype"] == "float16"
            assert reference_manifest["lossless_segment_coverage"] is True

            # The merged Arrow dataset is authoritative; rank JSONL files only duplicate it.
            for part in REFERENCE_CACHE.glob("part-rank-*.jsonl"):
                part.unlink()
                print("Deleted merged rank shard:", part)
            print("Reference cache GiB:", round(tree_size_gib(REFERENCE_CACHE), 3))
        else:
            print("Reference stage not needed")
        """,
    )
    set_cell(
        notebook,
        15,
        render(
            """
            ## 8. Train @@DISPLAY@@ with both T4 GPUs

            Each Accelerate rank owns one T4. FSDP2 full-shards the 1B policy, activation
            checkpointing and activation offload control memory, the vocabulary head is projected
            in bounded shards, and the frozen reference is not resident during training because its
            segment log-probabilities were cached above. `SAVE_STEPS=0` permits only the consolidated
            FP32 final model to be persisted.
            """,
            config,
        ),
    )
    set_cell(
        notebook,
        16,
        render(
            r"""
            if RUN_TRAINING and not TRAIN_ALREADY_COMPLETE:
                if FINAL_MODEL.exists():
                    safe_remove_tree(FINAL_MODEL, RUN_DIR)
                command = [
                    sys.executable, "-m", "accelerate.commands.launch",
                    "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
                    "--mixed_precision", "fp16",
                    "-m", "olmo2_bees.train_structured", "train",
                    "--workspace", str(SCRATCH_ROOT),
                    "--dataset-path", str(PREPARED_DATASET),
                    "--reference-cache", str(REFERENCE_CACHE),
                    "--train-split", "train",
                    "--model-id", MODEL_ID,
                    "--model-revision", MODEL_REVISION,
                    "--output-dir", str(RUN_DIR),
                    "--run-name", "kaggle-olmo2-bees-@@SLUG@@",
                    "--variant", VARIANT,
                    "--epochs", str(EPOCHS),
                    "--learning-rate", str(LEARNING_RATE),
                    "--beta", str(BETA),
                    "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
                    "--logging-steps", str(LOGGING_STEPS),
                    "--save-steps", str(SAVE_STEPS),
                    "--seed", str(SEED),
                ]
                run_streaming(command, cwd=PROJECT_ROOT)
            elif RUN_TRAINING:
                print("Validated final training run already exists; skipping")
            else:
                print("RUN_TRAINING=False")

            delete_training_transients()
            """,
            config,
        ),
    )
    set_cell(notebook, 17, "## 9. Locate and verify the final model")
    set_cell(
        notebook,
        18,
        r"""
        def attached_final_model(variant: str) -> tuple[Path, Path]:
            local_manifest = RUN_DIR / "training_manifest.json"
            if local_manifest.is_file() and FINAL_MODEL.is_dir():
                return FINAL_MODEL, local_manifest
            candidates = []
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.is_dir():
                for manifest_path in kaggle_input.rglob("training_manifest.json"):
                    try:
                        manifest = json.loads(manifest_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    candidate = manifest_path.parent / "final"
                    if manifest.get("variant") == variant and candidate.is_dir():
                        candidates.append((candidate, manifest_path))
            if not candidates:
                raise FileNotFoundError(
                    f"No attached {variant} training output. Commit the training notebook, "
                    "add its output under Kaggle Inputs, and rerun this evaluation session."
                )
            candidates.sort(key=lambda item: (len(item[0].parts), str(item[0])))
            return candidates[0]


        if BUILD_REFERENCE_ONLY:
            POLICY_MODEL = None
            TRAINING_MANIFEST_PATH = None
            training_manifest = None
            reference_manifest_path = REFERENCE_CACHE / "reference_manifest.json"
            assert reference_manifest_path.is_file()
            assert (REFERENCE_CACHE / "dataset").is_dir()
            print("Verified reusable reference output:", REFERENCE_CACHE)
        else:
            POLICY_MODEL, TRAINING_MANIFEST_PATH = attached_final_model(VARIANT)
            training_manifest = json.loads(TRAINING_MANIFEST_PATH.read_text())
            assert training_manifest["variant"] == VARIANT
            assert training_manifest["objective_formula"] == VARIANT_FORMULAS[VARIANT]
            assert training_manifest["full_parameter_training"] is True
            assert training_manifest["peft_or_lora"] is False
            assert training_manifest["weight_quantization"] is None
            assert training_manifest["optimizer_is_paged"] is True
            assert training_manifest["optimizer_state_bits"] == 32
            assert training_manifest["master_parameter_dtype"] == "float32"
            assert training_manifest["compute_dtype"] == "float16"
            assert training_manifest["saved_weight_dtypes"] == ["F32"]
            assert training_manifest["parallelism"] == "FSDP2_FULL_SHARD"
            assert training_manifest["world_size"] == 2
            assert training_manifest["global_batch_size_pairs"] == 16
            assert training_manifest["activation_checkpointing"] is True
            assert training_manifest["restart_checkpoints_enabled"] is False
            for filename, digest in training_manifest["weight_files_sha256"].items():
                assert sha256_file(POLICY_MODEL / filename) == digest
            print("Verified policy:", POLICY_MODEL)
            print(json.dumps(training_manifest["metrics"], indent=2))
        """,
    )
    # Cells 19-23 are generic evaluation cells inherited from the current Kaggle notebook.
    set_cell(
        notebook,
        24,
        r"""
        from olmo2_bees.quality_gate import preference_metrics

        summary = {
            "variant": VARIANT,
            "formula": VARIANT_FORMULAS[VARIANT],
            "normalization": "v_k^norm = beta * a_k * h_k / n_k",
            "phase": "reference" if BUILD_REFERENCE_ONLY else (
                "training" if RUN_TRAINING else "evaluation"
            ),
            "base_model": MODEL_ID,
            "base_model_revision": MODEL_REVISION,
            "reference_cache": str(REFERENCE_CACHE) if BUILD_REFERENCE_ONLY else None,
            "policy_model": str(POLICY_MODEL) if POLICY_MODEL is not None else None,
            "training_manifest": (
                str(TRAINING_MANIFEST_PATH) if TRAINING_MANIFEST_PATH is not None else None
            ),
            "heldout_preference": None,
            "baseline_benchmarks": None,
            "candidate_benchmarks": None,
            "quality_gate": None,
        }
        if RUN_HELDOUT_EVAL:
            summary["heldout_preference"] = preference_metrics(
                PREFERENCE_EVAL / "implicit_scores.jsonl"
            )
        if RUN_LM_EVAL:
            summary["baseline_benchmarks"] = task_metrics(BASELINE_LM_EVAL)
            summary["candidate_benchmarks"] = task_metrics(CANDIDATE_LM_EVAL)
        if RUN_HELDOUT_EVAL and RUN_LM_EVAL:
            report_path = EVAL_ROOT / "quality_gate.json"
            gate_command = [
                sys.executable, "-m", "olmo2_bees.quality_gate",
                "--workspace", str(SCRATCH_ROOT),
                "--preference-scores", str(PREFERENCE_EVAL / "implicit_scores.jsonl"),
                "--baseline-lm-eval", str(BASELINE_LM_EVAL),
                "--candidate-lm-eval", str(CANDIDATE_LM_EVAL),
                "--output", str(report_path),
                "--max-task-drop", "0.01",
                "--max-macro-drop", "0.002",
            ]
            for task in BENCHMARK_TASKS:
                gate_command.extend(["--required-benchmark-task", task])
            try:
                run_streaming(gate_command, cwd=PROJECT_ROOT)
            except subprocess.CalledProcessError:
                print("Quality gate rejected the checkpoint after writing its report")
            summary["quality_gate"] = json.loads(report_path.read_text())

        write_json(PERSIST_ROOT / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        """,
    )
    set_cell(
        notebook,
        27,
        render(
            """
            ## Next session

            Use **Save Version → Save & Run All** after a completed phase. If reference preparation
            and training do not fit comfortably in one Kaggle session, persist only the reference
            cache first, attach that output to a fresh run, then train. After training, commit and
            attach the model output to an evaluation copy with `RUN_TRAINING=False`.

            Attached caches and models remain read-only under `/kaggle/input`; each phase writes
            only its own @@DISPLAY@@ output under `/kaggle/working`.
            """,
            config,
        ),
    )
    notebook["metadata"]["vpdpo_method"] = config["variant"]
    notebook["metadata"]["accelerator"] = "GPU T4 x2"
    return notebook


def main() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for config in NOTEBOOKS:
        output = ROOT / config["filename"]
        notebook = build_notebook(template, config)
        output.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(output)


if __name__ == "__main__":
    main()
