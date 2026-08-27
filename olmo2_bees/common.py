from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
BEES_UPSTREAM_COMMIT = "749faf478e2827dd72835d574693623926a2e444"


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the imported BeeS repository from a notebook or module path."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve().parents[1]])
    for candidate in candidates:
        for path in (candidate, *candidate.parents):
            if (path / "ReadME.md").exists() and (path / "sampler.py").exists():
                return path
            nested = path / "BeeS"
            if (nested / "ReadME.md").exists() and (nested / "sampler.py").exists():
                return nested
    raise FileNotFoundError("Could not locate the BeeS repository")


def configure_workspace(workspace: str | Path | None = None) -> dict[str, str]:
    """Put every cache, temporary file, run log, and output under the workspace."""
    if workspace is None:
        workspace_path = find_project_root().parent
    else:
        workspace_path = Path(workspace).resolve()

    cache = workspace_path / ".cache"
    mapping = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "HF_DATASETS_CACHE": cache / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "hub",
        "TORCH_HOME": cache / "torch",
        "XDG_CACHE_HOME": cache,
        "PIP_CACHE_DIR": cache / "pip",
        "TRITON_CACHE_DIR": cache / "triton",
        "WANDB_CACHE_DIR": cache / "wandb" / "cache",
        "WANDB_DIR": workspace_path / "artifacts" / "wandb",
        "TMPDIR": workspace_path / ".tmp",
    }
    for key, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        # The user explicitly requested workspace-local storage. Override inherited
        # machine-wide cache variables so subprocesses cannot spill into the home disk.
        os.environ[key] = str(path)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # PyTorch 2.10 uses PYTORCH_ALLOC_CONF; retain the older alias for compatible
    # workers launched from environments that still consume it.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_ALLOC_CONF"])
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Transformers otherwise launches a non-daemon Hub conversion thread for the
    # upstream OLMo `.bin` checkpoint.  It is unnecessary (our outputs are already
    # safetensors) and can keep every distributed rank alive after training ends.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    return {key: os.environ[key] for key in mapping}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: Iterable[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def run_streaming(command: list[str], cwd: str | Path | None = None) -> None:
    """Run a long subprocess while preserving tqdm/log output in a notebook."""
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=None if cwd is None else str(cwd),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def python_command() -> str:
    return sys.executable


def assert_two_turing_gpus() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This workflow requires two visible CUDA GPUs")
    devices: list[dict[str, Any]] = []
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        if (props.major, props.minor) < (7, 0):
            raise RuntimeError(f"GPU {index} does not provide reliable FP16 tensor-core training")
        devices.append(
            {
                "index": index,
                "name": props.name,
                "memory_gib": round(props.total_memory / 2**30, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                # Native BF16 tensor cores begin with Ampere (SM 8.x). Recent PyTorch
                # releases may report software-emulated BF16 as supported on Turing.
                "bf16_supported": props.major >= 8,
            }
        )
    return devices


def ensure_full_parameter_model(model: Any) -> dict[str, int]:
    """Fail if the model is quantized, adapter-wrapped, or partially frozen."""
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise RuntimeError("Quantized model loading is forbidden for this workflow")
    if hasattr(model, "peft_config"):
        raise RuntimeError("PEFT/LoRA adapters are forbidden for this workflow")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if total != trainable:
        raise RuntimeError(f"Expected every parameter to train, got {trainable:,}/{total:,}")
    return {"total_parameters": total, "trainable_parameters": trainable}


def assistant_completion(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and len(messages) == 1
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "assistant"
        and isinstance(messages[0].get("content"), str)
        and bool(messages[0]["content"].strip())
    )


def valid_prompt(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and len(messages) >= 1
        and all(
            isinstance(message, dict)
            and message.get("role") in {"system", "user", "assistant"}
            and isinstance(message.get("content"), str)
            for message in messages
        )
        and messages[-1]["role"] == "user"
    )
