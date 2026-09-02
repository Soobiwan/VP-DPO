"""Build one Kaggle T4 x2 Llama 3.2 training notebook per non-TIDPO method."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = ROOT / "notebooks" / "kaggle"
DATASET_FILENAME = "ultrafeedback_bees_olmo2_1b_segmented_final.jsonl"
DATASET_SHA256 = "6cddda3cedd1c078ba1f2cc3c3e798d5eeb79968478730b593a206c8ff4eb013"

MODELS = (
    {
        "size": "1b",
        "display": "Llama 3.2 1B Instruct",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "revision": "9213176726f574b556790deb65791e0c5aa438b6",
    },
    {
        "size": "3b",
        "display": "Llama 3.2 3B Instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    },
)

# Keep the order aligned with the repository's per-model evaluation suite.
METHODS = (
    {
        "number": 1,
        "key": "B_norm-DPO",
        "display": "VPDPO B_norm-DPO",
        "slug": "b_norm_dpo",
        "trainer": "structured",
        "learning_rate": "5e-7",
    },
    {
        "number": 2,
        "key": "B_norm-VDPO",
        "display": "VPDPO B_norm-VDPO",
        "slug": "b_norm_vdpo",
        "trainer": "structured",
        "learning_rate": "5e-7",
    },
    {
        "number": 3,
        "key": "B-DPO",
        "display": "VPDPO B-DPO",
        "slug": "b_dpo",
        "trainer": "structured",
        "learning_rate": "2e-6",
    },
    {
        "number": 4,
        "key": "B-VDPO",
        "display": "VPDPO B-VDPO",
        "slug": "b_vdpo",
        "trainer": "structured",
        "learning_rate": "2e-6",
    },
    {
        "number": 5,
        "key": "C-DPO",
        "display": "VPDPO C-DPO",
        "slug": "c_dpo",
        "trainer": "structured",
        "learning_rate": "2e-6",
    },
    {
        "number": 6,
        "key": "C-VDPO",
        "display": "VPDPO C-VDPO",
        "slug": "c_vdpo",
        "trainer": "structured",
        "learning_rate": "2e-6",
    },
    {
        "number": 7,
        "key": "DPO",
        "display": "Simple DPO",
        "slug": "simple_dpo",
        "trainer": "structured",
        "learning_rate": "5e-7",
    },
    {
        "number": 8,
        "key": "A",
        "display": "VPDPO A",
        "slug": "vpdpo_a",
        "trainer": "structured",
        "learning_rate": "2e-6",
    },
    {
        "number": 9,
        "key": "SimPO",
        "display": "SimPO",
        "slug": "simpo",
        "trainer": "preference",
        "learning_rate": "5e-7",
    },
    {
        "number": 10,
        "key": "SamPO",
        "display": "SamPO",
        "slug": "sampo",
        "trainer": "preference",
        "learning_rate": "5e-7",
    },
)


def source_lines(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def render(text: str, model: dict[str, str], method: dict[str, str | int]) -> str:
    replacements = {
        "@@MODEL_SIZE@@": str(model["size"]),
        "@@MODEL_DISPLAY@@": str(model["display"]),
        "@@MODEL_ID@@": str(model["model_id"]),
        "@@MODEL_REVISION@@": str(model["revision"]),
        "@@METHOD_NUMBER@@": f"{int(method['number']):02d}",
        "@@METHOD_KEY@@": str(method["key"]),
        "@@VARIANT_REPR@@": repr(method["key"] if method["trainer"] == "structured" else None),
        "@@METHOD_REPR@@": repr(method["key"] if method["trainer"] == "preference" else None),
        "@@METHOD_DISPLAY@@": str(method["display"]),
        "@@METHOD_SLUG@@": str(method["slug"]),
        "@@TRAINER@@": str(method["trainer"]),
        "@@LEARNING_RATE@@": str(method["learning_rate"]),
        "@@DATASET_FILENAME@@": DATASET_FILENAME,
        "@@DATASET_SHA256@@": DATASET_SHA256,
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    if "@@" in text:
        raise RuntimeError(f"Unrendered notebook marker in {text[:200]!r}")
    return text


def build_notebook(model: dict[str, str], method: dict[str, str | int]) -> dict:
    needs_reference = method["trainer"] == "structured" or method["key"] == "SamPO"
    reference_description = (
        "This objective uses a frozen-reference cache. The default run builds it in scratch and "
        "then trains. If runtime is tight, use the reference-only switch and attach that committed "
        "output to a fresh training session."
        if needs_reference
        else "SimPO is reference-free, so no frozen-reference cache is created."
    )
    cells = [
        markdown(
            render(
                f"""
                # @@MODEL_DISPLAY@@ — @@METHOD_DISPLAY@@ on Kaggle T4 x2

                This training-only notebook starts from the pinned, instruction-tuned
                `@@MODEL_ID@@` checkpoint and trains **every parameter** with two Kaggle T4 GPUs
                through FSDP2 full sharding. It uses FP32 master weights and paged FP32 AdamW
                states, FP16 compute, activation checkpointing/offloading, and a memory-bounded
                vocabulary projection. It does not use LoRA, PEFT, QLoRA, or weight quantization.

                It consumes the repository's existing BeeS-selected, segmented UltraFeedback file
                (`data/processed/@@DATASET_FILENAME@@`). The historical filename says `olmo2`, but
                the conversational text and segment annotations are model-independent; this
                notebook retokenizes them losslessly with Llama's official chat template.

                {reference_description}

                Before running:

                1. Select **GPU T4 x2** and enable Internet.
                2. Accept the Llama 3.2 license for the configured Hugging Face account.
                3. Add a Kaggle secret named `Huggingface` (or `HF_TOKEN`) with read access.
                4. Attach this repository as a Kaggle Dataset, or let the notebook clone it.

                The final FP32 checkpoint is written under `/kaggle/working`; repositories,
                packages, model downloads, prepared Arrow data, and non-persisted caches stay under
                `/kaggle/temp`. Restart checkpoints are disabled to remain within Kaggle's output
                budget.
                """,
                model,
                method,
            )
        ),
        markdown("## 1. Run configuration"),
        code(
            render(
                f"""
                MODEL_SIZE = "@@MODEL_SIZE@@"
                MODEL_ID = "@@MODEL_ID@@"
                MODEL_REVISION = "@@MODEL_REVISION@@"
                TRAINER_KIND = "@@TRAINER@@"
                VARIANT = @@VARIANT_REPR@@
                METHOD = @@METHOD_REPR@@
                METHOD_SLUG = "@@METHOD_SLUG@@"

                INSTALL_DEPENDENCIES = True
                BUILD_REFERENCE_ONLY = False
                RUN_TRAINING = True

                # Optional explicit Kaggle path. Leave empty to search the repository and
                # /kaggle/input recursively for the hash-verified canonical file.
                DATASET_JSONL_OVERRIDE = ""

                REPO_URL = "https://github.com/Soobiwan/VP-DPO.git"
                REPO_REF = "main"

                MAX_LENGTH = 2048  # upper bound only; batches remain dynamically padded
                EPOCHS = 1.0
                MAX_STEPS = -1
                LEARNING_RATE = @@LEARNING_RATE@@
                BETA = 0.1
                SIMPO_BETA = 2.0
                SIMPO_GAMMA_BETA_RATIO = 0.5
                SAMPO_BETA = 0.1
                GRADIENT_ACCUMULATION_STEPS = 8
                LOGGING_STEPS = 1
                SAVE_STEPS = 0
                SEED = 42

                if BUILD_REFERENCE_ONLY and not {needs_reference!r}:
                    raise ValueError("This objective is reference-free")
                if BUILD_REFERENCE_ONLY and RUN_TRAINING:
                    raise ValueError("Reference-only preparation and training must be separate phases")
                if not BUILD_REFERENCE_ONLY and not RUN_TRAINING:
                    raise ValueError("Enable BUILD_REFERENCE_ONLY or RUN_TRAINING")
                if SAVE_STEPS != 0:
                    raise ValueError("Kaggle notebooks persist only the final model")
                """,
                model,
                method,
            )
        ),
        markdown("## 2. Locate the repository, authenticate, install, and verify BeeS data"),
        code(
            render(
                r"""
                from pathlib import Path
                import gc
                import hashlib
                import json
                import os
                import shutil
                import subprocess
                import sys
                import tempfile


                ON_KAGGLE = Path("/kaggle").is_dir()
                SCRATCH_PARENT = Path("/kaggle/temp") if ON_KAGGLE else Path(tempfile.gettempdir())
                SCRATCH_ROOT = SCRATCH_PARENT / f"vpdpo-llama32-{MODEL_SIZE}-{METHOD_SLUG}"
                SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)


                def is_repo(path: Path) -> bool:
                    return (
                        (path / "BeeS" / "olmo2_bees" / "train_structured.py").is_file()
                        and (path / "BeeS" / "olmo2_bees" / "train_preference_suite.py").is_file()
                    )


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

                SOURCE_PROJECT_ROOT = SOURCE_REPO / "BeeS"
                if INSTALL_DEPENDENCIES:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
                        "--no-cache-dir", "-r",
                        str(SOURCE_PROJECT_ROOT / "requirements-olmo2-training.txt"),
                    ])

                # Kaggle users often upload only this notebook and let it clone remote main.
                # Stage a writable package copy so the notebook remains compatible with a remote
                # revision that still treats the source JSONL's OLMo token counts as universal.
                # The original checkout/input and canonical BeeS JSONL remain untouched.
                RUNTIME_REPO = SCRATCH_ROOT / "runtime_source"
                RUNTIME_PROJECT_ROOT = RUNTIME_REPO / "BeeS"
                if RUNTIME_REPO.exists():
                    shutil.rmtree(RUNTIME_REPO)
                shutil.copytree(
                    SOURCE_PROJECT_ROOT,
                    RUNTIME_PROJECT_ROOT,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"),
                )


                def apply_llama_runtime_compatibility(project_root: Path) -> list[str]:
                    applied = []
                    structured_path = project_root / "olmo2_bees" / "structured_preference.py"
                    structured_source = structured_path.read_text(encoding="utf-8")
                    old_token_guard = (
                        '            if "chosen_tokens" in row and '
                        'len(chosen["input_ids"]) != int(row["chosen_tokens"]):\n'
                        '                raise ValueError(f"Chosen token count drift at line '
                        '{line_number}")\n'
                        '            if "rejected_tokens" in row and '
                        'len(rejected["input_ids"]) != int(row["rejected_tokens"]):\n'
                        '                raise ValueError(f"Rejected token count drift at line '
                        '{line_number}")\n'
                    )
                    if old_token_guard in structured_source:
                        structured_source = structured_source.replace(
                            old_token_guard,
                            '            # Source token counts belong to the tokenizer that '
                            'created the JSONL.\n'
                            '            # Active Llama lengths are validated losslessly above.\n',
                            1,
                        )
                        applied.append("model-specific token-count provenance")
                    if (
                        "Chosen token count drift at line" in structured_source
                        or "Rejected token count drift at line" in structured_source
                    ):
                        raise RuntimeError(
                            "Could not remove the stale OLMo token-count guard from the runtime copy"
                        )
                    old_response_alignment = (
                        "    response_start = len(prompt_text)\n"
                        "    response_end = response_start + len(response)\n"
                        "    if full_text[response_start:response_end] != response:\n"
                        '        raise ValueError("Assistant content is not contiguous in the rendered chat")\n'
                    )
                    if old_response_alignment in structured_source:
                        structured_source = structured_source.replace(
                            old_response_alignment,
                            "    response_start = len(prompt_text)\n"
                            "    if full_text.startswith(response, response_start):\n"
                            "        rendered_response = response\n"
                            "        rendered_char_segment_ids = response_char_segment_ids\n"
                            "    else:\n"
                            "        rendered_response = response.strip()\n"
                            "        leading_trim = len(response) - len(response.lstrip())\n"
                            "        trailing_trim = len(response) - len(response.rstrip())\n"
                            "        rendered_stop = (\n"
                            "            len(response) - trailing_trim if trailing_trim else len(response)\n"
                            "        )\n"
                            "        rendered_char_segment_ids = response_char_segment_ids[\n"
                            "            leading_trim:rendered_stop\n"
                            "        ]\n"
                            "        if not rendered_response or not full_text.startswith(\n"
                            "            rendered_response, response_start\n"
                            "        ):\n"
                            "            raise ValueError(\n"
                            '                "Assistant content is neither preserved nor "\n'
                            '                "outer-whitespace-trimmed in the rendered chat"\n'
                            "            )\n"
                            "    if len(rendered_char_segment_ids) != len(rendered_response):\n"
                            '        raise RuntimeError("Rendered response/segment character alignment differs")\n'
                            "    response_end = response_start + len(rendered_response)\n",
                            1,
                        )
                        structured_source = structured_source.replace(
                            "        overlapping_ids = response_char_segment_ids[local_start:local_end]\n",
                            "        overlapping_ids = rendered_char_segment_ids[local_start:local_end]\n",
                            1,
                        )
                        applied.append("chat-template outer-whitespace alignment")
                    if (
                        "Assistant content is not contiguous in the rendered chat"
                        in structured_source
                        or "rendered_char_segment_ids" not in structured_source
                    ):
                        raise RuntimeError(
                            "Could not make assistant alignment compatible with Llama's chat template"
                        )
                    structured_source = structured_source.replace(
                        "Tokenizing segmented OLMo pairs", "Tokenizing segmented pairs"
                    )
                    structured_path.write_text(structured_source, encoding="utf-8")

                    trainer_path = project_root / "olmo2_bees" / "train_structured.py"
                    trainer_source = trainer_path.read_text(encoding="utf-8")
                    old_reference_identity = (
                        '    expected_reference = {\n'
                        '        "dataset_path": str(args.dataset_path.resolve()),\n'
                        '        "dataset_fingerprint": train_dataset._fingerprint,\n'
                    )
                    if old_reference_identity in trainer_source:
                        trainer_source = trainer_source.replace(
                            old_reference_identity,
                            '    expected_reference = {\n'
                            '        "dataset_fingerprint": train_dataset._fingerprint,\n',
                            1,
                        )
                        applied.append("remount-safe structured reference identity")
                    run_train_source = trainer_source.split("def run_train", 1)[-1]
                    if old_reference_identity in run_train_source:
                        raise RuntimeError(
                            "Could not make the structured reference cache remount-safe"
                        )
                    trainer_path.write_text(trainer_source, encoding="utf-8")
                    return applied


                compatibility_patches = apply_llama_runtime_compatibility(
                    RUNTIME_PROJECT_ROOT
                )
                PROJECT_ROOT = RUNTIME_PROJECT_ROOT
                print("Runtime compatibility patches:", compatibility_patches or "already present")

                # A rerun in the same kernel may have imported the read-only/original package.
                for module_name in list(sys.modules):
                    if module_name == "olmo2_bees" or module_name.startswith("olmo2_bees."):
                        del sys.modules[module_name]
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                os.environ["PYTHONPATH"] = (
                    str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
                )

                from huggingface_hub import login
                from olmo2_bees.common import (
                    assert_two_turing_gpus,
                    configure_workspace,
                    package_versions,
                    run_streaming,
                    sha256_file,
                )
                from olmo2_bees.structured_preference import (
                    SUPPORTED_VARIANTS,
                    VARIANT_DESCRIPTIONS,
                    VARIANT_FORMULAS,
                )
                from olmo2_bees.preference_suite_losses import (
                    METHOD_DESCRIPTIONS,
                    METHOD_FORMULAS,
                    sampo_pair_loss,
                    simpo_pair_loss,
                )

                CACHE_ENV = configure_workspace(SCRATCH_ROOT)


                def get_hugging_face_token() -> str:
                    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                        value = os.environ.get(name, "").strip()
                        if value:
                            return value
                    if ON_KAGGLE:
                        from kaggle_secrets import UserSecretsClient

                        client = UserSecretsClient()
                        for name in ("Huggingface", "HF_TOKEN"):
                            try:
                                value = client.get_secret(name).strip()
                            except Exception:
                                continue
                            if value:
                                return value
                    raise RuntimeError(
                        "A Hugging Face token with accepted Llama 3.2 access is required. "
                        "Add the Kaggle secret 'Huggingface' or 'HF_TOKEN'."
                    )


                HF_TOKEN = get_hugging_face_token()
                os.environ["HF_TOKEN"] = HF_TOKEN
                login(token=HF_TOKEN, add_to_git_credential=False)

                source_name = "@@DATASET_FILENAME@@"
                expected_source_sha256 = "@@DATASET_SHA256@@"


                def normalized_text_sha256(path: Path) -> str:
                    # Git may check the JSONL out as CRLF on Windows. Its canonical dataset
                    # identity is the SHA-256 after normalizing record separators to LF.
                    content = path.read_bytes().replace(b"\r\n", b"\n")
                    return hashlib.sha256(content).hexdigest()


                source_candidates = []
                if DATASET_JSONL_OVERRIDE:
                    source_candidates.append(Path(DATASET_JSONL_OVERRIDE).expanduser())
                source_candidates.extend([
                    SOURCE_REPO / "data" / "processed" / source_name,
                    SOURCE_REPO / source_name,
                    SOURCE_REPO / "BeeS" / "data" / "processed" / source_name,
                ])
                if ON_KAGGLE:
                    source_candidates.extend(Path("/kaggle/input").rglob(source_name))

                unique_candidates = []
                seen = set()
                for candidate in source_candidates:
                    try:
                        resolved = candidate.resolve()
                    except OSError:
                        continue
                    if resolved not in seen and resolved.is_file():
                        seen.add(resolved)
                        unique_candidates.append(resolved)
                matching_sources = [
                    path for path in unique_candidates
                    if normalized_text_sha256(path) == expected_source_sha256
                ]
                if not matching_sources:
                    raise FileNotFoundError(
                        "Could not find the canonical BeeS JSONL. Expected repository path "
                        f"data/processed/{source_name} or an attached Kaggle input with "
                        "LF-normalized SHA-256 "
                        f"{expected_source_sha256}. Set DATASET_JSONL_OVERRIDE if it was renamed."
                    )
                SOURCE_JSONL = min(matching_sources, key=lambda path: (len(path.parts), str(path)))

                print("Repository:", SOURCE_REPO)
                print("BeeS dataset:", SOURCE_JSONL)
                print("BeeS normalized SHA-256:", normalized_text_sha256(SOURCE_JSONL))
                print("Base model:", MODEL_ID, "@", MODEL_REVISION)
                print(json.dumps(package_versions([
                    "torch", "transformers", "datasets", "accelerate", "trl",
                    "bitsandbytes", "huggingface_hub", "safetensors",
                ]), indent=2))
                """,
                model,
                method,
            )
        ),
        markdown("## 3. Kaggle T4 x2 audit and output layout"),
        code(
            render(
                r"""
                import torch

                gpus = assert_two_turing_gpus()
                if len(gpus) != 2:
                    raise RuntimeError(f"Expected exactly two GPUs, found {len(gpus)}")
                if ON_KAGGLE:
                    if not all("T4" in gpu["name"] for gpu in gpus):
                        raise RuntimeError(f"Select Kaggle GPU T4 x2, found {gpus}")
                    if min(gpu["memory_gib"] for gpu in gpus) < 14.5:
                        raise RuntimeError(f"Expected two 16 GB-class T4 GPUs, found {gpus}")
                if any(gpu["bf16_supported"] for gpu in gpus):
                    raise RuntimeError("This audited recipe expects Turing FP16, not BF16")

                if ON_KAGGLE:
                    PERSIST_ROOT = (
                        Path("/kaggle/working")
                        / f"llama32_{MODEL_SIZE}_bees_{METHOD_SLUG}"
                    )
                else:
                    PERSIST_ROOT = (
                        SOURCE_REPO / "artifacts" / "kaggle_t4x2"
                        / f"llama32_{MODEL_SIZE}" / METHOD_SLUG
                    )
                RUN_DIR = PERSIST_ROOT / "run"
                FINAL_MODEL = RUN_DIR / "final"
                PERSISTED_REFERENCE_CACHE = PERSIST_ROOT / "reference_cache"
                SCRATCH_REFERENCE_CACHE = SCRATCH_ROOT / "reference_cache"
                PREPARED_DATASET = SCRATCH_ROOT / "prepared_llama32"
                PREPARED_MANIFEST = SCRATCH_ROOT / "prepared_llama32.manifest.json"
                for path in (PERSIST_ROOT, RUN_DIR):
                    path.mkdir(parents=True, exist_ok=True)


                def tree_size_gib(path: Path) -> float:
                    if not path.exists():
                        return 0.0
                    return sum(
                        item.stat().st_size for item in path.rglob("*") if item.is_file()
                    ) / 2**30


                def safe_remove_tree(path: Path, allowed_root: Path) -> None:
                    target = path.resolve()
                    root = allowed_root.resolve()
                    if target == root:
                        raise RuntimeError(f"Refusing to remove allowed root itself: {root}")
                    target.relative_to(root)
                    if target.exists():
                        shutil.rmtree(target)
                        print("Deleted:", target)


                def delete_training_transients() -> None:
                    for checkpoint in RUN_DIR.glob("checkpoint-*"):
                        safe_remove_tree(checkpoint, RUN_DIR)
                    patterns = (
                        "paged_optimizer_rank_*.pt", "optimizer.pt", "scheduler.pt",
                        "scaler.pt", "rng_state_*.pth", "pytorch_model_fsdp.bin",
                    )
                    final_resolved = FINAL_MODEL.resolve()
                    for pattern in patterns:
                        for item in RUN_DIR.rglob(pattern):
                            if final_resolved not in item.resolve().parents:
                                item.unlink(missing_ok=True)


                print(json.dumps(gpus, indent=2))
                print("Output:", PERSIST_ROOT)
                print("Effective global pair batch:", 2 * GRADIENT_ACCUMULATION_STEPS)
                """,
                model,
                method,
            )
        ),
        markdown("## 4. Objective self-test and lossless Llama retokenization"),
        code(
            render(
                r"""
                if TRAINER_KIND == "structured":
                    if VARIANT not in SUPPORTED_VARIANTS:
                        raise ValueError(f"Unsupported structured variant: {VARIANT}")
                    print("Objective:", VARIANT_FORMULAS[VARIANT])
                    print("Description:", VARIANT_DESCRIPTIONS[VARIANT])
                    run_streaming([
                        sys.executable, "-m", "olmo2_bees.train_structured", "self-test",
                        "--workspace", str(SCRATCH_ROOT),
                    ], cwd=PROJECT_ROOT)
                else:
                    if METHOD not in {"SimPO", "SamPO"}:
                        raise ValueError(f"Unsupported preference method: {METHOD}")
                    print("Objective:", METHOD_FORMULAS[METHOD])
                    print("Description:", METHOD_DESCRIPTIONS[METHOD])
                    synthetic_policy = torch.tensor([
                        [-0.2, -0.3, -0.4, 0.0],
                        [-0.5, -0.2, -0.1, 0.0],
                        [-0.6, -0.4, 0.0, 0.0],
                        [-0.7, -0.2, -0.2, 0.0],
                    ], requires_grad=True)
                    synthetic_mask = torch.tensor([
                        [1, 1, 1, 0], [1, 1, 1, 0],
                        [1, 1, 0, 0], [1, 1, 1, 0],
                    ], dtype=torch.bool)
                    if METHOD == "SimPO":
                        objective_test = simpo_pair_loss(
                            synthetic_policy, synthetic_mask,
                            beta=SIMPO_BETA,
                            gamma_beta_ratio=SIMPO_GAMMA_BETA_RATIO,
                        )
                    else:
                        synthetic_reference = synthetic_policy.detach() - 0.05
                        objective_test = sampo_pair_loss(
                            synthetic_policy,
                            synthetic_reference,
                            synthetic_mask,
                            beta=SAMPO_BETA,
                            generator=torch.Generator().manual_seed(SEED),
                        )
                    objective_test[0].backward()
                    if not torch.isfinite(objective_test[0]):
                        raise FloatingPointError(f"Non-finite {METHOD} self-test")
                    if synthetic_policy.grad is None or not torch.isfinite(synthetic_policy.grad).all():
                        raise FloatingPointError(f"Invalid {METHOD} self-test gradient")
                    print(f"{METHOD} objective self-test loss:", float(objective_test[0]))

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
                prepared_manifest = json.loads(PREPARED_MANIFEST.read_text())
                if len(prepared["train"]) != 6000 or len(prepared["test"]) != 1891:
                    raise RuntimeError(f"Unexpected BeeS split sizes: {prepared}")
                if normalized_text_sha256(Path(prepared_manifest["source_jsonl"])) != "@@DATASET_SHA256@@":
                    raise RuntimeError("Prepared data came from the wrong BeeS JSONL")
                if prepared_manifest["model_id"] != MODEL_ID:
                    raise RuntimeError("Prepared data used the wrong tokenizer")
                TRAIN_FINGERPRINT = prepared["train"]._fingerprint
                print(json.dumps(prepared_manifest["statistics"], indent=2))
                del prepared
                gc.collect()
                """,
                model,
                method,
            )
        ),
        markdown("## 5. Frozen-reference phase (when required)"),
        code(
            render(
                f"""
                NEEDS_REFERENCE = {needs_reference!r}


                def reference_manifest_matches(manifest: dict) -> bool:
                    common = (
                        manifest.get("dataset_fingerprint") == TRAIN_FINGERPRINT
                        and manifest.get("split") == "train"
                        and manifest.get("rows") == 6000
                        and manifest.get("model_id") == MODEL_ID
                        and manifest.get("model_revision") == MODEL_REVISION
                        and manifest.get("world_size") == 2
                        and manifest.get("compute_dtype") == "float16"
                    )
                    if not common:
                        return False
                    if TRAINER_KIND == "structured":
                        return manifest.get("lossless_segment_coverage") is True
                    return (
                        manifest.get("cache_schema_version") == 2
                        and manifest.get("max_length") == MAX_LENGTH
                        and manifest.get("tidpo_kl_top_k") == 0
                        and manifest.get("with_tidpo_anchors") is False
                    )


                def find_attached_reference_cache() -> Path | None:
                    candidates = []
                    local_manifest = PERSISTED_REFERENCE_CACHE / "reference_manifest.json"
                    if local_manifest.is_file():
                        candidates.append(local_manifest)
                    kaggle_input = Path("/kaggle/input")
                    if kaggle_input.is_dir():
                        candidates.extend(kaggle_input.rglob("reference_manifest.json"))
                    matches = []
                    for manifest_path in candidates:
                        try:
                            manifest = json.loads(manifest_path.read_text())
                        except (OSError, json.JSONDecodeError):
                            continue
                        cache = manifest_path.parent
                        if (cache / "dataset").is_dir() and reference_manifest_matches(manifest):
                            matches.append(cache)
                    matches.sort(key=lambda path: (len(path.parts), str(path)))
                    return matches[0] if matches else None


                REFERENCE_CACHE = None
                if NEEDS_REFERENCE:
                    if BUILD_REFERENCE_ONLY:
                        REFERENCE_CACHE = PERSISTED_REFERENCE_CACHE
                    else:
                        REFERENCE_CACHE = find_attached_reference_cache()
                        if REFERENCE_CACHE is None:
                            REFERENCE_CACHE = SCRATCH_REFERENCE_CACHE

                    manifest_path = REFERENCE_CACHE / "reference_manifest.json"
                    have_reference = False
                    if manifest_path.is_file() and (REFERENCE_CACHE / "dataset").is_dir():
                        have_reference = reference_manifest_matches(
                            json.loads(manifest_path.read_text())
                        )

                    if not have_reference:
                        if TRAINER_KIND == "structured":
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
                        else:
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
                                "--tidpo-kl-top-k", "0",
                                "--no-with-tidpo-anchors",
                            ]
                        run_streaming(reference_command, cwd=PROJECT_ROOT)

                    reference_manifest = json.loads(
                        (REFERENCE_CACHE / "reference_manifest.json").read_text()
                    )
                    if not reference_manifest_matches(reference_manifest):
                        raise RuntimeError("Reference cache failed its identity checks")
                    for part in REFERENCE_CACHE.glob("part-rank-*.jsonl"):
                        part.unlink()
                    print("Reference cache:", REFERENCE_CACHE)
                    print("Reference cache GiB:", round(tree_size_gib(REFERENCE_CACHE), 3))
                else:
                    print("Reference-free objective; skipping this phase")
                """,
                model,
                method,
            )
        ),
        markdown("## 6. Full-parameter training on both T4 GPUs"),
        code(
            render(
                r"""
                def training_complete() -> bool:
                    manifest_path = RUN_DIR / "training_manifest.json"
                    if not manifest_path.is_file() or not FINAL_MODEL.is_dir():
                        return False
                    manifest = json.loads(manifest_path.read_text())
                    identity_ok = (
                        manifest.get("model_id") == MODEL_ID
                        and manifest.get("model_revision") == MODEL_REVISION
                        and manifest.get("world_size") == 2
                        and manifest.get("full_parameter_training") is True
                        and manifest.get("restart_checkpoints_enabled") is False
                    )
                    method_ok = (
                        manifest.get("variant") == VARIANT
                        if TRAINER_KIND == "structured"
                        else manifest.get("method") == METHOD
                    )
                    return identity_ok and method_ok


                delete_training_transients()
                TRAIN_ALREADY_COMPLETE = training_complete()
                if RUN_TRAINING and not TRAIN_ALREADY_COMPLETE:
                    if FINAL_MODEL.exists():
                        safe_remove_tree(FINAL_MODEL, RUN_DIR)
                    common = [
                        sys.executable, "-m", "accelerate.commands.launch",
                        "--multi_gpu", "--num_processes", "2", "--gpu_ids", "0,1",
                        "--mixed_precision", "fp16",
                    ]
                    if TRAINER_KIND == "structured":
                        command = common + [
                            "-m", "olmo2_bees.train_structured", "train",
                            "--workspace", str(SCRATCH_ROOT),
                            "--dataset-path", str(PREPARED_DATASET),
                            "--reference-cache", str(REFERENCE_CACHE),
                            "--train-split", "train",
                            "--model-id", MODEL_ID,
                            "--model-revision", MODEL_REVISION,
                            "--output-dir", str(RUN_DIR),
                            "--run-name", f"llama32-{MODEL_SIZE}-bees-{METHOD_SLUG}",
                            "--variant", VARIANT,
                            "--epochs", str(EPOCHS),
                            "--max-steps", str(MAX_STEPS),
                            "--learning-rate", str(LEARNING_RATE),
                            "--beta", str(BETA),
                            "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
                            "--logging-steps", str(LOGGING_STEPS),
                            "--save-steps", str(SAVE_STEPS),
                            "--seed", str(SEED),
                            "--transformer-layer-class", "LlamaDecoderLayer",
                        ]
                    else:
                        command = common + [
                            "-m", "olmo2_bees.train_preference_suite", "train",
                            "--workspace", str(SCRATCH_ROOT),
                            "--dataset-path", str(PREPARED_DATASET),
                            "--train-split", "train",
                            "--model-id", MODEL_ID,
                            "--model-revision", MODEL_REVISION,
                            "--output-dir", str(RUN_DIR),
                            "--max-length", str(MAX_LENGTH),
                            "--run-name", f"llama32-{MODEL_SIZE}-bees-{METHOD_SLUG}",
                            "--method", METHOD,
                            "--epochs", str(EPOCHS),
                            "--max-steps", str(MAX_STEPS),
                            "--learning-rate", str(LEARNING_RATE),
                            "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
                            "--logging-steps", str(LOGGING_STEPS),
                            "--save-steps", str(SAVE_STEPS),
                            "--seed", str(SEED),
                            "--transformer-layer-class", "LlamaDecoderLayer",
                            "--simpo-beta", str(SIMPO_BETA),
                            "--simpo-gamma-beta-ratio", str(SIMPO_GAMMA_BETA_RATIO),
                            "--sampo-beta", str(SAMPO_BETA),
                        ]
                        if METHOD == "SamPO":
                            command.extend(["--reference-cache", str(REFERENCE_CACHE)])
                    run_streaming(command, cwd=PROJECT_ROOT)
                elif RUN_TRAINING:
                    print("Verified final training output already exists; skipping")
                else:
                    print("Reference-only phase complete; training intentionally skipped")

                delete_training_transients()
                """,
                model,
                method,
            )
        ),
        markdown("## 7. Verify the final artifact and clean scratch storage"),
        code(
            render(
                r"""
                if RUN_TRAINING:
                    if not training_complete():
                        raise RuntimeError("Training did not produce a complete validated artifact")
                    training_manifest_path = RUN_DIR / "training_manifest.json"
                    training_manifest = json.loads(training_manifest_path.read_text())
                    config_path = FINAL_MODEL / "config.json"
                    config = json.loads(config_path.read_text())

                    # The trainers replace the monolithic output projection with independent
                    # checkpointed vocabulary slices, then consolidate those slices back to
                    # lm_head.weight. Llama starts with tied input/output embeddings; leaving its
                    # original flag set would make Transformers retie on reload and silently
                    # discard that trained output head. Record and persist the actual save layout.
                    source_output_embeddings_tied = training_manifest.get(
                        "source_output_embeddings_tied",
                        bool(config.get("tie_word_embeddings")),
                    )
                    config["tie_word_embeddings"] = False
                    config_path.write_text(
                        json.dumps(config, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    training_manifest.update({
                        "source_output_embeddings_tied": source_output_embeddings_tied,
                        "saved_output_embeddings_tied": False,
                        "output_head_strategy": "independent_checkpointed_vocabulary_shards",
                    })
                    training_manifest_path.write_text(
                        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                    if training_manifest["optimizer"] != "bitsandbytes.optim.PagedAdamW32bit":
                        raise RuntimeError("Unexpected optimizer")
                    if training_manifest["optimizer_is_paged"] is not True:
                        raise RuntimeError("Expected paged optimizer state")
                    if training_manifest["optimizer_state_bits"] != 32:
                        raise RuntimeError("Expected FP32 optimizer state")
                    if training_manifest["master_parameter_dtype"] != "float32":
                        raise RuntimeError("Expected FP32 master parameters")
                    if training_manifest["compute_dtype"] != "float16":
                        raise RuntimeError("Expected T4 FP16 compute")
                    if training_manifest["saved_weight_dtypes"] != ["F32"]:
                        raise RuntimeError("Final weights are not all FP32")
                    if training_manifest["parallelism"] != "FSDP2_FULL_SHARD":
                        raise RuntimeError("Expected FSDP2 full sharding")
                    if training_manifest["global_batch_size_pairs"] != 16:
                        raise RuntimeError("Unexpected global pair batch")
                    if training_manifest["train_rows"] != 6000:
                        raise RuntimeError("Training did not consume the full BeeS train split")
                    if training_manifest["saved_output_embeddings_tied"] is not False:
                        raise RuntimeError("Chunked output-head save contract was not recorded")
                    if config.get("model_type") != "llama":
                        raise RuntimeError(f"Unexpected saved architecture: {config}")
                    if config.get("tie_word_embeddings") is not False:
                        raise RuntimeError(
                            "Saved Llama config would retie and overwrite the trained output head"
                        )
                    for filename, digest in training_manifest["weight_files_sha256"].items():
                        if sha256_file(FINAL_MODEL / filename) != digest:
                            raise RuntimeError(f"Weight hash mismatch: {filename}")

                    persisted_gib = tree_size_gib(PERSIST_ROOT)
                    if ON_KAGGLE and persisted_gib >= 19.5:
                        raise RuntimeError(
                            f"Kaggle output is too large ({persisted_gib:.3f} GiB); "
                            "remove non-final artifacts before saving a version"
                        )
                    print("Verified final model:", FINAL_MODEL)
                    print("Persisted GiB:", round(persisted_gib, 3))
                    print(json.dumps(training_manifest["metrics"], indent=2))
                else:
                    if REFERENCE_CACHE != PERSISTED_REFERENCE_CACHE:
                        raise RuntimeError("Reference-only output was not written persistently")
                    print("Verified reusable reference cache:", REFERENCE_CACHE)

                # This is the notebook-owned disposable directory only; never /kaggle/temp itself.
                safe_remove_tree(SCRATCH_ROOT, SCRATCH_PARENT)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                """,
                model,
                method,
            )
        ),
        markdown(
            """
            ## Saved output

            Use **Save Version → Save & Run All**. For a reference-only run, attach the committed
            output to a fresh copy of this same notebook, set `BUILD_REFERENCE_ONLY=False`, and set
            `RUN_TRAINING=True`. A training run persists only its final FP32 model, tokenizer,
            training manifest, metrics, and hashes.

            GPU execution is intentionally not combined with evaluation here. Evaluate the
            immutable committed model in a separate Kaggle session.
            """
        ),
    ]

    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        cell["id"] = hashlib.sha256(f"{index}:{source}".encode("utf-8")).hexdigest()[:8]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU T4 x2",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "vpdpo_base_model": model["model_id"],
            "vpdpo_method": method["key"],
            "vpdpo_training_only": True,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_notebook(
    notebook: dict, model: dict[str, str], method: dict[str, str | int]
) -> None:
    serialized = json.dumps(notebook, ensure_ascii=False)
    required = (
        model["model_id"],
        model["revision"],
        method["key"],
        "GPU T4 x2",
        "LlamaDecoderLayer",
        DATASET_FILENAME,
        DATASET_SHA256,
        "data/processed/",
        "apply_llama_runtime_compatibility",
        "BUILD_REFERENCE_ONLY",
        "RUN_TRAINING",
        "training_manifest.json",
    )
    missing = [value for value in required if str(value) not in serialized]
    if missing:
        raise RuntimeError(f"Notebook is missing required values: {missing}")
    if "METHOD = 'TIDPO'" in serialized or 'METHOD = "TIDPO"' in serialized:
        raise RuntimeError("TIDPO must not be generated")
    if notebook["metadata"].get("accelerator") != "GPU T4 x2":
        raise RuntimeError("Notebook accelerator metadata is wrong")
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError("Notebook cell IDs are not unique")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            if cell["execution_count"] is not None or cell["outputs"]:
                raise RuntimeError(f"Generated code cell {index} contains execution state")
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")


def build_readme(model: dict[str, str]) -> str:
    rows = "\n".join(
        f"| {int(method['number'])} | {method['display']} | "
        f"`{int(method['number']):02d}_llama32_{model['size']}_{method['slug']}_kaggle_t4x2.ipynb` |"
        for method in METHODS
    )
    return dedent(
        f"""
        # Llama 3.2 {model['size'].upper()} BeeS training notebooks

        These notebooks train one non-TIDPO preference method per Kaggle **GPU T4 x2** session
        from `{model['model_id']}` at pinned revision `{model['revision']}`. TIDPO is deliberately
        excluded while its original notebook is under correction.

        All notebooks use the same canonical BeeS file:
        `data/processed/{DATASET_FILENAME}` (LF-normalized SHA-256 `{DATASET_SHA256}`). The old `olmo2` filename
        is dataset provenance only; every notebook retokenizes the unchanged text and segment
        annotations with the pinned Llama tokenizer.

        ## Kaggle setup

        1. Select **GPU T4 x2** and enable Internet.
        2. Accept Meta's Llama 3.2 license on Hugging Face.
        3. Add a Kaggle secret named `Huggingface` or `HF_TOKEN`.
        4. Attach the repository/dataset input, or allow the notebook to clone the repository.
        5. Run one notebook at a time. Objectives requiring a reference cache can be split into a
           reference-only version followed by a training version.

        Standalone notebook uploads stage a writable copy of the training package and apply the
        Llama token-count/remounted-cache compatibility changes there. The attached or cloned
        repository and canonical BeeS dataset are never modified.

        ## Notebook index

        | # | Method | Notebook |
        |---:|---|---|
        @@ROWS@@

        The recipe is full-parameter FSDP2 training with FP32 master weights and paged FP32 AdamW
        state, FP16 compute, no LoRA/PEFT/QLoRA, no quantization, and no restart checkpoints. The
        3B recipe is materially heavier than the 1B recipe; use the split reference phase and
        preserve the final-only output policy. These generated notebooks have static validation,
        but must still be runtime-validated on Kaggle's current image before results are reported.

        Regenerate both model-size suites from the repository root with:

        ```powershell
        python .\\BeeS\\tools\\build_kaggle_llama32_training_notebooks.py
        ```
        """
    ).replace("@@ROWS@@", rows).strip() + "\n"


def main() -> None:
    for model in MODELS:
        output_dir = NOTEBOOK_ROOT / f"llama32_{model['size']}_training"
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_outputs = set()
        for method in METHODS:
            filename = (
                f"{int(method['number']):02d}_llama32_{model['size']}_"
                f"{method['slug']}_kaggle_t4x2.ipynb"
            )
            expected_outputs.add(filename)
            notebook = build_notebook(model, method)
            validate_notebook(notebook, model, method)
            output = output_dir / filename
            output.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            print(output)
        (output_dir / "README.md").write_text(build_readme(model), encoding="utf-8")
        actual_outputs = {path.name for path in output_dir.glob("*.ipynb")}
        if actual_outputs != expected_outputs:
            raise RuntimeError(
                f"Unexpected notebook set under {output_dir}: "
                f"missing={sorted(expected_outputs - actual_outputs)}, "
                f"extra={sorted(actual_outputs - expected_outputs)}"
            )


if __name__ == "__main__":
    main()
