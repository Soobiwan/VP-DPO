from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks/kaggle"
MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
MODEL_REVISION = "0d85a3d037876ce6ac7d4311d994400fc66ac27f"

METHODS = {
    "TIDPO": {
        "display": "TI-DPO repository approximation",
        "filename": "olmo_bees_tidpo_kaggle_t4x2.ipynb",
        "short": "TI-DPO repository approximation",
        "reference_note": (
            "This is the older memory-saving approximation, not the citation-grade official-repo "
            "baseline: it uses a top-32-plus-remainder KL projection and fixed cached anchors. "
            "Use `olmo_bees_tidpo_official_repo_exact_kaggle_t4x2.ipynb` when reporting TI-DPO."
        ),
        "top_k": 32,
        "anchors": True,
    },
    "TIDPOOfficialRepoExact": {
        "method": "TIDPO",
        "display": "TI-DPO official-repo exact",
        "filename": "olmo_bees_tidpo_official_repo_exact_kaggle_t4x2.ipynb",
        "short": "TI-DPO official-repo exact",
        "reference_note": (
            "This is the citation-grade TI-DPO baseline. It ports the objective from the pinned "
            "[gracefulning/TIDPO repository](https://github.com/gracefulning/TIDPO/tree/"
            "e04a0926869a8f9fe9c9e9ce395394fd2c697fe2) exactly: weighted TDPO2, exhaustive "
            "full-vocabulary KL(ref||policy), full-sequence last-logit gradient/Gaussian token "
            "weights, and a same-prompt anchor sampled live from the current policy. The notebook "
            "hash-verifies the vendored upstream source and writes the commit, exactness flags, "
            "and adapter hash into the training manifest. It uses the repository-default PyTorch "
            "RMSprop optimizer; OLMo, BeeS data, schedule, precision, and FSDP2 are the controlled "
            "comparison recipe. This is an exact TI-DPO method port rather than a reproduction of "
            "the authors' Llama/Mistral runs."
        ),
        "top_k": 0,
        "anchors": False,
        "repo_exact": True,
    },
    "TIDPOPaperExact": {
        "method": "TIDPO",
        "display": "TI-DPO paper-exact",
        "filename": "olmo_bees_tidpo_paper_exact_kaggle_t4x2.ipynb",
        "short": "TI-DPO paper-exact",
        "reference_note": (
            "This variant follows [arXiv:2505.19653v3](https://arxiv.org/abs/2505.19653) "
            "Eqs. 5-14 and Appendix B.5: last-logit L1 gradient attribution mixed with the "
            "centered Gaussian prior, weighted DPO without a TDPO position-KL term, and an "
            "intermediate response sampled live from the current policy. A frozen FP16 "
            "reference model supplies exact selected-token log-probabilities during training; "
            "there is no approximation cache. This ports the published objective and reported "
            "objective hyperparameters to OLMo and this BeeS dataset; it does not claim to "
            "reproduce the paper's Llama/Mistral checkpoints."
        ),
        "top_k": 0,
        "anchors": False,
        "paper_exact": True,
    },
    "SimPO": {
        "filename": "olmo_bees_simpo_kaggle_t4x2.ipynb",
        "short": "SimPO",
        "reference_note": (
            "SimPO is reference-free. This notebook does not create or load the multi-gigabyte "
            "reference cache used by TIDPO."
        ),
        "top_k": 0,
        "anchors": False,
    },
    "SamPO": {
        "filename": "olmo_bees_sampo_kaggle_t4x2.ipynb",
        "short": "SamPO",
        "reference_note": (
            "SamPO caches only aligned chosen/rejected token log-probabilities. TIDPO top-k "
            "support and anchors are disabled, substantially reducing Kaggle disk and host-RAM use."
        ),
        "top_k": 0,
        "anchors": False,
    },
}


def source_lines(source: str) -> list[str]:
    text = dedent(source).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


def build_notebook(method_key: str, config: dict) -> dict:
    method = config.get("method", method_key)
    display = config.get("display", method)
    paper_exact = bool(config.get("paper_exact", False))
    repo_exact = bool(config.get("repo_exact", False))
    live_tidpo = paper_exact or repo_exact
    tidpo_variant = (
        "official_repo_exact"
        if repo_exact
        else (
            "paper_exact"
            if paper_exact
            else ("repository_approximation" if method == "TIDPO" else None)
        )
    )
    needs_reference_stage = method != "SimPO" and not live_tidpo
    top_k = config["top_k"]
    anchors_flag = "--with-tidpo-anchors" if config["anchors"] else "--no-with-tidpo-anchors"
    lower = method_key.lower()
    scope_note = (
        "This notebook is training-only; paper evaluation belongs in a separate immutable-model "
        "session."
        if repo_exact
        else "Evaluation runs after the trained model is attached as a read-only input."
    )
    switch_note = (
        "This citation-grade artifact has one executable phase: full training. The two evaluation "
        "flags remain hard-disabled guards used by the shared storage code."
        if repo_exact
        else (
            "Keep reference preparation, training, and evaluation in separate Kaggle sessions when "
            "configured below. The notebook rejects incompatible phase combinations."
        )
    )
    dataset_note = (
        "This produces the validated 6,000-row training split used by the controlled OLMo run."
        if repo_exact
        else (
            "This produces the same validated 6,000-row train and 1,891-row untouched test splits "
            "as the combined notebook. It is skipped for an lm-eval-only session."
        )
    )
    requirements_file = (
        "requirements-olmo2-training.txt" if repo_exact else "requirements-olmo2.txt"
    )
    optimizer_description = (
        "FP32 PyTorch RMSprop state (the pinned TI-DPO repository default)"
        if repo_exact
        else "paged FP32 AdamW states"
    )
    optimizer_cleanup_description = "optimizer state" if repo_exact else "paged optimizer files"
    restart_optimizer_description = (
        "RMSprop optimizer state" if repo_exact else "paged optimizer states"
    )
    audited_packages = [
        "torch", "transformers", "datasets", "accelerate", "trl", "bitsandbytes", "safetensors"
    ]
    if not repo_exact:
        audited_packages.append("lm_eval")
    if repo_exact:
        next_step = (
            "After training, commit this notebook output. The FP32 model and its hash-verified "
            "`training_manifest.json` are the complete outputs of this training-only notebook."
        )
    elif method == "TIDPO" and not live_tidpo:
        next_step = (
            "For TIDPO, first commit the default reference-only output and attach it to a new copy "
            "of this notebook. Set `BUILD_REFERENCE_ONLY=False` and `RUN_TRAINING=True`. After that "
            "training run, commit and attach the model output to an evaluation session."
        )
    else:
        next_step = (
            "After training, commit the model output and attach it to an evaluation copy of this "
            "notebook. Set `RUN_TRAINING=False` and enable the desired evaluation switch."
        )
    cells = [
        markdown(
            f"""
            # OLMo 2 1B — {display} on Kaggle T4 x2

            This standalone notebook trains one full-parameter `{MODEL_ID}` policy with **both**
            Kaggle T4 GPUs through FSDP2 full sharding. It keeps FP32 master weights and
            {optimizer_description}, uses FP16 only for compute, and does not use LoRA, PEFT,
            QLoRA, or weight quantization.

            Before running, choose **GPU T4 x2** under *Settings → Accelerator* and enable Internet,
            or attach this repository as a Kaggle Dataset input. Kaggle currently limits GPU notebook
            sessions to 12 hours and persisted `/kaggle/working` output to 20 GB. The legacy
            repository-approximation TIDPO notebook persists a reusable top-k/anchor cache. The
            official-repo-exact and paper-exact TI-DPO variants train with a live frozen reference;
            SimPO is reference-free. {scope_note}

            {config['reference_note']}

            Storage policy:

            - `/kaggle/temp`: repository clone, package/Hugging Face caches, prepared dataset, and
              resumable rank shards; all are deleted after the selected phase.
            - `/kaggle/working`: either the reusable TIDPO reference cache, one final FP32 model, or
              small evaluation results—never all transient copies at once.
            - Restart checkpoints are disabled. Any stale `checkpoint-*`, scheduler, scaler, RNG, or
              {optimizer_cleanup_description} under this run are explicitly removed.

            Resource reference: [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks).
            """
        ),
        markdown(
            f"""
            ## 1. Run switches

            {switch_note}
            """
        ),
        code(
            f"""
            METHOD = {method!r}
            TIDPO_VARIANT = {tidpo_variant!r}
            NEEDS_REFERENCE_STAGE = {needs_reference_stage}

            INSTALL_DEPENDENCIES = True
            BUILD_REFERENCE_ONLY = {method == 'TIDPO' and not live_tidpo}
            RUN_TRAINING = {method != 'TIDPO' or live_tidpo}
            RUN_HELDOUT_EVAL = False
            RUN_LM_EVAL = False

            REPO_URL = "https://github.com/Soobiwan/VP-DPO.git"
            REPO_REF = "main"

            if BUILD_REFERENCE_ONLY and METHOD == "SimPO":
                raise ValueError("SimPO is reference-free")
            if BUILD_REFERENCE_ONLY and TIDPO_VARIANT in ("paper_exact", "official_repo_exact"):
                raise ValueError("Exact TI-DPO variants use a live reference model, not a cache")
            if not any((BUILD_REFERENCE_ONLY, RUN_TRAINING, RUN_HELDOUT_EVAL, RUN_LM_EVAL)):
                raise ValueError("Enable reference preparation, training, or evaluation")
            if BUILD_REFERENCE_ONLY and (RUN_TRAINING or RUN_HELDOUT_EVAL or RUN_LM_EVAL):
                raise ValueError("BUILD_REFERENCE_ONLY must run by itself")
            if RUN_TRAINING and (RUN_HELDOUT_EVAL or RUN_LM_EVAL):
                raise ValueError(
                    "Kaggle safety guard: train and evaluate in separate 12-hour sessions. "
                    "Commit the training output, attach it as an input, then rerun with "
                    "RUN_TRAINING=False."
                )
            """
        ),
        markdown("## 2. Locate or clone the repository, then install the pinned environment"),
        code(
            f"""
            from pathlib import Path
            import gc
            import hashlib
            import json
            import os
            import shutil
            import subprocess
            import sys
            import tempfile
            import time


            def normalized_source_sha256(path: Path) -> str:
                content = path.read_bytes().replace(b"\\r\\n", b"\\n")
                return hashlib.sha256(content).hexdigest()


            ON_KAGGLE = Path("/kaggle").is_dir()
            SCRATCH_PARENT = Path("/kaggle/temp") if ON_KAGGLE else Path(tempfile.gettempdir())
            SCRATCH_ROOT = SCRATCH_PARENT / "vpdpo-{lower}-scratch"
            SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)


            def is_repo(path: Path) -> bool:
                return (path / "BeeS" / "olmo2_bees" / "train_preference_suite.py").is_file()


            def locate_repo() -> Path | None:
                cwd = Path.cwd().resolve()
                for candidate in (cwd, *cwd.parents):
                    if is_repo(candidate):
                        return candidate
                kaggle_input = Path("/kaggle/input")
                if kaggle_input.is_dir():
                    for match in kaggle_input.rglob("train_preference_suite.py"):
                        candidate = match.resolve().parents[2]
                        if is_repo(candidate):
                            return candidate
                return None


            clone_target = SCRATCH_ROOT / "source"


            def clone_source_repository() -> None:
                subprocess.check_call([
                    "git", "clone", "--depth", "1", "--branch", REPO_REF,
                    REPO_URL, str(clone_target),
                ])


            SOURCE_REPO = locate_repo()
            if SOURCE_REPO is None:
                if clone_target.exists() and not is_repo(clone_target):
                    shutil.rmtree(clone_target)
                if not clone_target.exists():
                    clone_source_repository()
                SOURCE_REPO = clone_target

            PROJECT_ROOT = SOURCE_REPO / "BeeS"
            requirements = PROJECT_ROOT / {requirements_file!r}
            if TIDPO_VARIANT == "official_repo_exact":
                expected_adapter_hash = (
                    "aca6d02db9f494e890b7629bf754c935343d912ec5a4be9436971628b6e3b694"
                )
                actual_adapter_hash = normalized_source_sha256(
                    PROJECT_ROOT / "olmo2_bees" / "train_preference_suite.py"
                )
                if (
                    actual_adapter_hash != expected_adapter_hash
                    and SOURCE_REPO.resolve() == clone_target.resolve()
                ):
                    # A failed Kaggle run can leave an older shallow clone under /kaggle/temp.
                    # Refresh that disposable clone once before rejecting the source snapshot.
                    shutil.rmtree(clone_target)
                    clone_source_repository()
                    SOURCE_REPO = clone_target
                    PROJECT_ROOT = SOURCE_REPO / "BeeS"
                    requirements = PROJECT_ROOT / {requirements_file!r}
                    actual_adapter_hash = normalized_source_sha256(
                        PROJECT_ROOT / "olmo2_bees" / "train_preference_suite.py"
                    )
                if actual_adapter_hash != expected_adapter_hash:
                    raise RuntimeError(
                        "This checkout does not contain the audited official-repo-exact OLMo "
                        f"adapter: expected {{expected_adapter_hash}}, got {{actual_adapter_hash}}. "
                        "Attach the repository snapshot shipped with this notebook."
                    )
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
            from olmo2_bees.preference_suite_losses import METHOD_DESCRIPTIONS, METHOD_FORMULAS

            CACHE_ENV = configure_workspace(SCRATCH_ROOT)

            source_name = "ultrafeedback_bees_olmo2_1b_segmented_final.jsonl"
            source_candidates = [
                SOURCE_REPO / "data" / "processed" / source_name,
                SOURCE_REPO / source_name,
            ]
            if ON_KAGGLE:
                source_candidates.extend(Path("/kaggle/input").rglob(source_name))
            SOURCE_JSONL = max(
                (path for path in source_candidates if path.is_file()),
                key=lambda path: path.stat().st_size,
                default=None,
            )
            if SOURCE_JSONL is None or SOURCE_JSONL.stat().st_size < 1_000_000:
                raise FileNotFoundError(
                    f"Could not find the full {{source_name}}. Expected it under "
                    f"{{SOURCE_REPO / 'data' / 'processed'}} or a Kaggle Input. "
                    "Attach the repository dataset containing data/processed; Internet-only "
                    "clones work only when that tracked data file is present in the selected ref."
                )

            print("Repository:", SOURCE_REPO)
            print("Source dataset:", SOURCE_JSONL)
            print(json.dumps(package_versions({audited_packages!r}), indent=2))
            """
        ),
        markdown("## 3. Kaggle T4 x2 and host-resource audit"),
        code(
            """
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
            """
        ),
        markdown("## 4. Method configuration and storage lifecycle"),
        code(
            f"""
            MODEL_ID = {MODEL_ID!r}
            MODEL_REVISION = {MODEL_REVISION!r}
            MAX_LENGTH = 1024
            EPOCHS = 1.0
            LEARNING_RATE = 5e-7
            GRADIENT_ACCUMULATION_STEPS = 8
            LOGGING_STEPS = 1
            SAVE_STEPS = 0  # mandatory on Kaggle: final model only, no optimizer/FSDP restart state
            SEED = 42

            TIDPO_BETA = {0.1 if paper_exact else 0.2}
            TIDPO_ALPHA = 0.5
            TIDPO2 = {not paper_exact}
            TIDPO_KL_TOP_K = {top_k}
            TIDPO_LAMBDA_IMPORTANCE = {0.7 if paper_exact else 0.2}
            TIDPO_PRIOR_SIGMA_DIV = {4.0 if paper_exact else 8.0}
            TIDPO_TRIPLET_GAMMA = {0.1 if paper_exact else 0.001}
            TIDPO_TRIPLET_MARGIN = {0.5 if paper_exact else 0.001}
            ANCHOR_MAX_NEW_TOKENS = 64
            ANCHOR_TOP_K = 50
            ANCHOR_TOP_P = 0.95
            ANCHOR_TEMPERATURE = 0.8
            SIMPO_BETA = 2.0
            SIMPO_GAMMA_BETA_RATIO = 0.5
            SAMPO_BETA = 0.1

            assert SAVE_STEPS == 0, "Kaggle output cannot safely retain full FSDP optimizer checkpoints"

            if TIDPO_VARIANT == "official_repo_exact":
                upstream_root = SOURCE_REPO / "third_party" / "TIDPO"
                upstream = json.loads((upstream_root / "UPSTREAM.json").read_text())
                assert upstream["repository"] == "https://github.com/gracefulning/TIDPO"
                assert upstream["commit"] == "e04a0926869a8f9fe9c9e9ce395394fd2c697fe2"
                expected_upstream_hashes = (
                    (
                        "trainers.py",
                        "5fb907eecc2d00a6b97d7ac45db4bd86ce4d58197b7a58363233e40040ae113f",
                    ),
                    (
                        "config/config.yaml",
                        "515b74cf7e7461049bcbf78519564acae0ec36516194e723b8f372e6ef3c4f88",
                    ),
                    (
                        "config/loss/tidpo.yaml",
                        "e77343adb00fa27a0d54d9c806a12510291b2c779be1639ea5e86da74149a1e8",
                    ),
                )
                for relative, expected_hash in expected_upstream_hashes:
                    assert normalized_source_sha256(upstream_root / relative) == expected_hash
                print("Verified pinned TI-DPO source:", upstream["commit"])

            if ON_KAGGLE:
                PERSIST_ROOT = Path("/kaggle/working") / "olmo2_bees_{lower}"
            else:
                PERSIST_ROOT = SOURCE_REPO / "artifacts" / "kaggle_t4x2" / "{lower}"
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
            CANDIDATE_LM_EVAL = EVAL_ROOT / "lm_eval_{lower}"
            PREFERENCE_EVAL = EVAL_ROOT / "preference"
            for path in (PERSIST_ROOT, RUN_DIR, EVAL_ROOT):
                path.mkdir(parents=True, exist_ok=True)


            def safe_remove_tree(path: Path, allowed_root: Path) -> None:
                path = path.resolve()
                allowed_root = allowed_root.resolve()
                if path == allowed_root:
                    raise RuntimeError(f"Refusing to remove the allowed root itself: {{path}}")
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
                    manifest.get("method") == METHOD
                    and (
                    manifest.get("tidpo_variant")
                        or ("repository_approximation" if METHOD == "TIDPO" else None)
                    ) == TIDPO_VARIANT
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
            print(json.dumps({{
                "method": METHOD,
                "tidpo_variant": TIDPO_VARIANT,
                "formula": (
                    "Pinned repo: full-sequence weighted TDPO2 with exact full-vocabulary "
                    "KL(ref||policy) + live-anchor triplet"
                    if TIDPO_VARIANT == "official_repo_exact"
                    else (
                        "TI-DPO v3 Eqs. 5-14: gradient/Gaussian-weighted DPO + live-anchor triplet"
                        if TIDPO_VARIANT == "paper_exact"
                        else METHOD_FORMULAS[METHOD]
                    )
                ),
                "description": (
                    "Exact objective port of gracefulning/TIDPO commit e04a092 to OLMo"
                    if TIDPO_VARIANT == "official_repo_exact"
                    else (
                        "TI-DPO paper-equation variant from arXiv:2505.19653v3"
                        if TIDPO_VARIANT == "paper_exact"
                        else METHOD_DESCRIPTIONS[METHOD]
                    )
                ),
                "scratch": str(SCRATCH_ROOT),
                "persisted_output": str(PERSIST_ROOT),
                "build_reference_only": BUILD_REFERENCE_ONLY,
                "training_already_complete": TRAIN_ALREADY_COMPLETE,
                "persisted_gib": round(tree_size_gib(PERSIST_ROOT), 3),
            }}, indent=2))
            """
        ),
        markdown("## 5. Objective self-tests"),
        code(
            """
            run_streaming([
                sys.executable, "-m", "olmo2_bees.train_preference_suite", "self-test",
                "--workspace", str(SCRATCH_ROOT),
            ], cwd=PROJECT_ROOT)
            """
        ),
        markdown(
            f"""
            ## 6. Prepare the lossless segmented dataset in scratch space

            {dataset_note}
            """
        ),
        code(
            """
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
            """
        ),
        markdown(f"## 7. {display} reference stage"),
        code(
            f"""
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
                        and manifest.get("model_revision") == MODEL_REVISION
                        and manifest.get("tidpo_kl_top_k") == TIDPO_KL_TOP_K
                        and manifest.get("with_tidpo_anchors") is {config['anchors']}
                    ):
                        candidates.append(cache)
                candidates.sort(key=lambda path: (len(path.parts), str(path)))
                return candidates[0] if candidates else None


            if RUN_TRAINING and not TRAIN_ALREADY_COMPLETE and NEEDS_REFERENCE_STAGE and METHOD == "TIDPO":
                attached_cache = find_attached_reference_cache()
                if attached_cache is None:
                    raise FileNotFoundError(
                        "TIDPO training requires its reference-only notebook output as a Kaggle "
                        "Input. Run the default BUILD_REFERENCE_ONLY phase first, commit it, then "
                        "attach that output and set BUILD_REFERENCE_ONLY=False, RUN_TRAINING=True."
                    )
                REFERENCE_CACHE = attached_cache
                print("Using attached read-only TIDPO reference cache:", REFERENCE_CACHE)
            elif (
                (BUILD_REFERENCE_ONLY or (RUN_TRAINING and not TRAIN_ALREADY_COMPLETE))
                and NEEDS_REFERENCE_STAGE
            ):
                reference_command = [
                    sys.executable, "-m", "accelerate.commands.launch",
                    "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
                    "--mixed_precision", "fp16",
                    "-m", "olmo2_bees.train_preference_suite", "reference",
                    "--workspace", str(SCRATCH_ROOT),
                    "--dataset-path", str(PREPARED_DATASET),
                    "--split", "train",
                    "--model-id", MODEL_ID,
                    "--model-revision", MODEL_REVISION,
                    "--output-dir", str(REFERENCE_CACHE),
                    "--max-length", str(MAX_LENGTH),
                    "--tidpo-kl-top-k", str(TIDPO_KL_TOP_K),
                    {anchors_flag!r},
                    "--anchor-max-new-tokens", str(ANCHOR_MAX_NEW_TOKENS),
                    "--anchor-seed", str(SEED + 1000),
                ]
                run_streaming(reference_command, cwd=PROJECT_ROOT)
                reference_manifest = json.loads(
                    (REFERENCE_CACHE / "reference_manifest.json").read_text()
                )
                assert reference_manifest["rows"] == 6000
                assert reference_manifest["world_size"] == 2
                assert reference_manifest["tidpo_kl_top_k"] == TIDPO_KL_TOP_K
                assert reference_manifest["with_tidpo_anchors"] is {config['anchors']}

                # The merged Arrow dataset is authoritative; JSONL rank shards only duplicate it.
                for part in REFERENCE_CACHE.glob("part-rank-*.jsonl"):
                    part.unlink()
                    print("Deleted merged rank shard:", part)
                print("Reference cache GiB:", round(tree_size_gib(REFERENCE_CACHE), 3))
            elif not NEEDS_REFERENCE_STAGE:
                print("No approximation reference-cache stage is required for this variant")
            else:
                print("Reference stage not needed")
            """
        ),
        markdown(
            f"""
            ## 8. Train {display} with both T4 GPUs

            `SAVE_STEPS=0` prevents FSDP weights and {restart_optimizer_description} from being written
            as restart checkpoints. The trainer saves only the consolidated FP32 final model.
            """
        ),
        code(
            """
            if RUN_TRAINING and not TRAIN_ALREADY_COMPLETE:
                if FINAL_MODEL.exists():
                    safe_remove_tree(FINAL_MODEL, RUN_DIR)
                command = [
                    sys.executable, "-m", "accelerate.commands.launch",
                    "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
                    "--mixed_precision", "fp16",
                    "-m", "olmo2_bees.train_preference_suite", "train",
                    "--workspace", str(SCRATCH_ROOT),
                    "--dataset-path", str(PREPARED_DATASET),
                    "--train-split", "train",
                    "--model-id", MODEL_ID,
                    "--model-revision", MODEL_REVISION,
                    "--output-dir", str(RUN_DIR),
                    "--max-length", str(MAX_LENGTH),
                    "--run-name", f"kaggle-olmo2-bees-{METHOD.lower()}",
                    "--method", METHOD,
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
                    "--tidpo-anchor-max-new-tokens", str(ANCHOR_MAX_NEW_TOKENS),
                    "--tidpo-anchor-top-k", str(ANCHOR_TOP_K),
                    "--tidpo-anchor-top-p", str(ANCHOR_TOP_P),
                    "--tidpo-anchor-temperature", str(ANCHOR_TEMPERATURE),
                    "--simpo-beta", str(SIMPO_BETA),
                    "--simpo-gamma-beta-ratio", str(SIMPO_GAMMA_BETA_RATIO),
                    "--sampo-beta", str(SAMPO_BETA),
                    "--tidpo2" if TIDPO2 else "--no-tidpo2",
                    "--tidpo-paper-exact" if TIDPO_VARIANT == "paper_exact" else "--no-tidpo-paper-exact",
                    "--tidpo-repo-exact" if TIDPO_VARIANT == "official_repo_exact" else "--no-tidpo-repo-exact",
                ]
                if NEEDS_REFERENCE_STAGE:
                    command.extend(["--reference-cache", str(REFERENCE_CACHE)])
                run_streaming(command, cwd=PROJECT_ROOT)
            elif RUN_TRAINING:
                print("Validated final training run already exists; skipping")
            else:
                print("RUN_TRAINING=False")

            # This is intentionally unconditional: even a library regression cannot leave restart
            # optimizer/FSDP state in the persisted output after a successful training subprocess.
            delete_training_transients()
            """
        ),
        markdown("## 9. Locate and verify the final model"),
        code(
            """
            def attached_final_model(method: str) -> tuple[Path, Path]:
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
                        if (
                            manifest.get("method") == method
                            and (
                                manifest.get("tidpo_variant")
                                or ("repository_approximation" if method == "TIDPO" else None)
                            ) == TIDPO_VARIANT
                            and candidate.is_dir()
                        ):
                            candidates.append((candidate, manifest_path))
                if not candidates:
                    raise FileNotFoundError(
                        f"No attached {method} training output. Commit the training notebook, "
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
                POLICY_MODEL, TRAINING_MANIFEST_PATH = attached_final_model(METHOD)
                training_manifest = json.loads(TRAINING_MANIFEST_PATH.read_text())
                assert training_manifest["method"] == METHOD
                assert (
                    training_manifest.get("tidpo_variant")
                    or ("repository_approximation" if METHOD == "TIDPO" else None)
                ) == TIDPO_VARIANT
                assert training_manifest["full_parameter_training"] is True
                assert training_manifest["peft_or_lora"] is False
                assert training_manifest["weight_quantization"] is None
                assert training_manifest["optimizer_state_bits"] == 32
                assert training_manifest["saved_weight_dtypes"] == ["F32"]
                assert training_manifest["parallelism"] == "FSDP2_FULL_SHARD"
                assert training_manifest["world_size"] == 2
                assert training_manifest["restart_checkpoints_enabled"] is False
                if TIDPO_VARIANT == "official_repo_exact":
                    assert training_manifest["optimizer"] == "torch.optim.RMSprop"
                    assert training_manifest["optimizer_is_paged"] is False
                    assert training_manifest["optimizer_foreach"] is False
                    assert training_manifest["tidpo_fidelity"] == "official_repository_objective_exact"
                    assert training_manifest["tidpo_upstream_commit"] == (
                        "e04a0926869a8f9fe9c9e9ce395394fd2c697fe2"
                    )
                    assert training_manifest["tidpo_position_kl"] == {
                        "kind": "exact_full_vocabulary_reference_to_policy",
                        "top_k": None,
                        "exact_full_vocabulary": True,
                        "support_projection_activation_checkpointing": None,
                        "support_projection_activation_offloading": None,
                        "reason": (
                            "Pinned repository TDPO2 computes KL(ref||policy) over every "
                            "vocabulary token"
                        ),
                    }
                    assert training_manifest["tidpo_live_policy_anchor"] is True
                    assert training_manifest["tidpo_fixed_anchor_cache"] is False
                    assert training_manifest["tidpo_official_defaults_enforced"] is True
                    assert training_manifest["tidpo_official_objective"] == {
                        "base": "weighted_tdpo2",
                        "beta": 0.2,
                        "alpha": 0.5,
                        "if_tdpo2": True,
                        "position_kl_direction": "KL(reference||policy)",
                        "position_kl_vocabulary": "full",
                        "triplet_gamma": 0.001,
                        "triplet_margin": 0.001,
                    }
                    assert training_manifest["tidpo_official_importance"]["gaussian_sigma"] == (
                        "max(1, valid_token_count/8)"
                    )
                    assert training_manifest["tidpo_official_importance"][
                        "normalization_scope"
                    ] == "all_nonpadding_prompt_and_response_tokens"
                    assert training_manifest["tidpo_official_anchor"]["source"] == (
                        "current_policy_same_prompt_live"
                    )
                    assert training_manifest["tidpo_source_evidence"]["commit"] == (
                        "e04a0926869a8f9fe9c9e9ce395394fd2c697fe2"
                    )
                    assert training_manifest["tidpo_source_evidence"]["optimizer"] == {{
                        "implementation": "torch.optim.RMSprop",
                        "source": "config/config.yaml",
                    }}
                    assert training_manifest["tidpo_source_evidence"][
                        "olmo_adapter_normalized_source_sha256"
                    ] == "aca6d02db9f494e890b7629bf754c935343d912ec5a4be9436971628b6e3b694"
                else:
                    assert training_manifest["optimizer_is_paged"] is True
                for filename, digest in training_manifest["weight_files_sha256"].items():
                    assert sha256_file(POLICY_MODEL / filename) == digest
                print("Verified policy:", POLICY_MODEL)
                print(json.dumps(training_manifest["metrics"], indent=2))
            """
        ),
        markdown(
            """
            ## 10. Optional held-out implicit preference evaluation

            Run this in an evaluation session with the training output attached. Per-rank score files
            are deleted immediately after the merged score file and manifest exist.
            """
        ),
        code(
            """
            if RUN_HELDOUT_EVAL:
                score_path = PREFERENCE_EVAL / "implicit_scores.jsonl"
                score_manifest = PREFERENCE_EVAL / "score_manifest.json"
                if not (score_path.is_file() and score_manifest.is_file()):
                    run_streaming([
                        sys.executable, "-m", "accelerate.commands.launch",
                        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
                        "--mixed_precision", "fp16",
                        "-m", "olmo2_bees.score_implicit",
                        "--workspace", str(SCRATCH_ROOT),
                        "--dataset-path", str(PREPARED_DATASET),
                        "--split", "test",
                        "--reference-model", MODEL_ID,
                        "--reference-revision", MODEL_REVISION,
                        "--policy-model", str(POLICY_MODEL),
                        "--output-dir", str(PREFERENCE_EVAL),
                        "--max-length", str(MAX_LENGTH),
                    ], cwd=PROJECT_ROOT)
                assert score_path.is_file() and score_manifest.is_file()
                for part in PREFERENCE_EVAL.glob("part-rank-*.jsonl"):
                    part.unlink()
                    print("Deleted merged evaluation shard:", part)
            else:
                print("Held-out evaluation disabled for this session")
            """
        ),
        markdown("## 11. Optional deterministic lm-eval comparison"),
        code(
            """
            BENCHMARK_TASKS = [
                "arc_challenge", "hellaswag", "winogrande", "gsm8k", "leaderboard_ifeval",
            ]

            from olmo2_bees.quality_gate import task_metrics


            def benchmark_complete(output_dir: Path) -> bool:
                try:
                    return set(BENCHMARK_TASKS).issubset(task_metrics(output_dir))
                except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
                    return False


            def run_lm_eval(model_args: str, output_dir: Path) -> None:
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


            if RUN_LM_EVAL:
                if not benchmark_complete(BASELINE_LM_EVAL):
                    run_lm_eval(
                        f"pretrained={MODEL_ID},revision={MODEL_REVISION},dtype=float16",
                        BASELINE_LM_EVAL,
                    )
                if not benchmark_complete(CANDIDATE_LM_EVAL):
                    run_lm_eval(f"pretrained={POLICY_MODEL},dtype=float16", CANDIDATE_LM_EVAL)
            else:
                print("lm-eval disabled for this session")
            """
        ),
        markdown("## 12. Write the available evidence and quality gate"),
        code(
            """
            from olmo2_bees.quality_gate import preference_metrics

            summary = {
                "method": METHOD,
                "tidpo_variant": TIDPO_VARIANT,
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
            """
        ),
        markdown(
            f"## {10 if repo_exact else 13}. Final cleanup and Kaggle output-budget assertion"
        ),
        code(
            """
            delete_training_transients()

            # Prepared data, scratch references, model/package caches, and paged files are
            # reproducible and must never become Kaggle notebook output. A reference-only phase
            # writes its validated cache directly under PERSIST_ROOT, outside this scratch tree.
            if SCRATCH_ROOT.exists():
                safe_remove_tree(SCRATCH_ROOT, SCRATCH_PARENT)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            persisted_gib = tree_size_gib(PERSIST_ROOT)
            working_gib = tree_size_gib(Path("/kaggle/working")) if ON_KAGGLE else persisted_gib
            assert working_gib < 19.0, (
                f"Total /kaggle/working output is {working_gib:.2f} GiB; keep at least 1 GiB below "
                "Kaggle's 20 GB output limit"
            )
            print(f"Ready to commit: {PERSIST_ROOT} ({persisted_gib:.3f} GiB)")
            print(f"Total persisted /kaggle/working: {working_gib:.3f} GiB")
            print("Scratch/cache/reference/optimizer state removed:", not SCRATCH_ROOT.exists())
            """
        ),
        markdown(
            f"""
            ## Next session

            Use **Save Version → Save & Run All** after each completed phase. {next_step}

            Attached caches and models remain read-only under `/kaggle/input`; the next phase writes
            only its own output, so the {display} FP32 checkpoint is never duplicated into
            `/kaggle/working`.
            """
        ),
    ]
    if repo_exact:
        # The requested citation-grade artifact is deliberately training-only. Evals belong in a
        # separate paper evaluation notebook/session and must not mutate or obscure this run.
        filtered_cells = []
        skip_next_code = False
        replace_next_reference_code = False
        for cell in cells:
            source = "".join(cell["source"])
            if cell["cell_type"] == "markdown" and source.startswith("## 7."):
                filtered_cells.append(
                    markdown(
                        """
                        ## 7. Live frozen reference (no approximation cache)

                        The exact full-vocabulary reference distribution is computed online for
                        every labeled position. The same frozen base model also scores the live
                        current-policy anchor. No top-k projection or fixed-anchor cache is built.
                        """
                    )
                )
                replace_next_reference_code = True
                continue
            if replace_next_reference_code and cell["cell_type"] == "code":
                filtered_cells.append(
                    code(
                        """
                        assert NEEDS_REFERENCE_STAGE is False
                        assert TIDPO_KL_TOP_K == 0
                        print("Exact online reference enabled; no approximation cache is used")
                        """
                    )
                )
                replace_next_reference_code = False
                continue
            if cell["cell_type"] == "markdown" and source.startswith(
                (
                    "## 10. Optional held-out",
                    "## 11. Optional deterministic lm-eval",
                    "## 12. Write the available evidence",
                )
            ):
                skip_next_code = True
                continue
            if skip_next_code and cell["cell_type"] == "code":
                skip_next_code = False
                continue
            filtered_cells.append(cell)
        cells = filtered_cells

    for index, cell in enumerate(cells):
        cell["id"] = f"{lower}-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for method, config in METHODS.items():
        destination = NOTEBOOK_DIR / config["filename"]
        notebook = build_notebook(method, config)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{destination.name}:cell-{index}", "exec")
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    main()
