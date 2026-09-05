"""Notebook helpers for evaluating Hugging Face causal language models.

Heavy dependencies are imported lazily so configuration and result handling can
be checked without installing PyTorch or downloading models.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


HARNESS_URL = "https://github.com/EleutherAI/lm-evaluation-harness.git"
HARNESS_REF = "v0.4.11"
HARNESS_COMMIT = "27988a293647d5853e48edea291640b4af54740c"


@dataclass(frozen=True)
class Benchmark:
    tasks: tuple[str, ...]
    fewshot: int
    description: str


BENCHMARKS = {
    "MMLU": Benchmark(("mmlu",), 5, "All 57 subjects; accuracy"),
    "GSM8K": Benchmark(("gsm8k",), 5, "Strict and flexible exact match"),
    "GPQA": Benchmark(("gpqa_main_zeroshot",), 0, "Main split; acc and acc_norm"),
    "HumanEval": Benchmark(("humaneval",), 0, "Greedy code completion; pass@1"),
    "TruthfulQA": Benchmark(("truthfulqa_mc1", "truthfulqa_mc2"), 0, "MC1 and MC2"),
    "IFEval": Benchmark(("ifeval",), 0, "Prompt/instruction strict and loose accuracy"),
}


@dataclass
class EvalConfig:
    model: str = "meta-llama/Llama-3.2-1B"
    revision: str = "main"
    tokenizer: str | None = None
    benchmarks: tuple[str, ...] = tuple(BENCHMARKS)
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int | str = 1
    max_length: int = 4096
    limit: int | float | None = None
    seed: int = 42
    apply_chat_template: bool = False
    trust_remote_code: bool = False
    allow_humaneval_execution: bool = True
    log_samples: bool = True
    bootstrap_iters: int = 1000

    def validate(self) -> None:
        if not self.benchmarks or len(set(self.benchmarks)) != len(self.benchmarks):
            raise ValueError("Choose at least one benchmark, without duplicates.")
        unknown = set(self.benchmarks) - BENCHMARKS.keys()
        if unknown:
            raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
        if self.limit is not None:
            valid_count = type(self.limit) is int and self.limit >= 1
            valid_fraction = type(self.limit) is float and 0 < self.limit < 1
            if not (valid_count or valid_fraction):
                raise ValueError("limit must be None, a positive integer, or a fraction between 0 and 1.")
        if not ((type(self.batch_size) is int and self.batch_size > 0) or self.batch_size == "auto"):
            raise ValueError("batch_size must be a positive integer or 'auto'.")
        if self.max_length <= 1280:
            raise ValueError("max_length must exceed IFEval's 1280-token generation budget.")
        if self.bootstrap_iters < 0:
            raise ValueError("bootstrap_iters must be nonnegative.")
        if "HumanEval" in self.benchmarks and not self.allow_humaneval_execution:
            raise ValueError("HumanEval requires code execution. Enable it or remove HumanEval from benchmarks.")


def normalize_model_source(source: str | Path) -> str:
    """Accept Hub IDs, Hub model-card URLs, and save_pretrained directories."""
    text = str(source).strip()
    if not text:
        raise ValueError("Provide a Hugging Face model ID, model-card URL, or checkpoint directory.")
    if text.startswith(("https://", "http://")):
        url = urlparse(text)
        parts = [unquote(part) for part in url.path.strip("/").split("/")]
        if url.hostname not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            raise ValueError("Model-card URLs must point to huggingface.co or hf.co.")
        if len(parts) < 2 or parts[0] in {"datasets", "spaces"}:
            raise ValueError("Use a model-card URL such as https://huggingface.co/meta-llama/Llama-3.2-1B.")
        if len(parts) > 2:
            raise ValueError("Use the root model-card URL; put branch/tag/commit in revision separately.")
        text = "/".join(parts)
    path = Path(text).expanduser()
    if path.exists():
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("Local models need a save_pretrained directory containing config.json and weights.")
        return str(path.resolve())
    if text.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {text}")
    if not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", text):
        raise ValueError(f"Invalid Hugging Face model ID: {text!r}")
    return text


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def prepare_harness(destination: str | Path, source: str = HARNESS_URL,
                    ref: str | None = HARNESS_REF, *, install: bool = True) -> Path:
    """Use an existing checkout unchanged, or clone a URL into destination.

    Set ref=None to accept the current commit of a supplied local checkout.
    Existing checkouts are never reset, cleaned, or switched automatically.
    """
    candidate = Path(source).expanduser()
    if candidate.exists():
        repo = candidate.resolve()
    else:
        if not source.startswith(("https://", "ssh://", "git@")):
            raise FileNotFoundError(f"Harness checkout does not exist: {source}")
        repo = Path(destination).expanduser().resolve()
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            command = ["git", "clone", "--depth", "1"]
            if ref:
                command += ["--branch", ref]
            subprocess.run([*command, source, str(repo)], check=True)
    if not (repo / "lm_eval" / "evaluator.py").is_file():
        raise ValueError(f"Not an lm-evaluation-harness checkout: {repo}")
    commit = git_output(repo, "rev-parse", "HEAD")
    if ref:
        expected = git_output(repo, "rev-parse", f"{ref}^{{commit}}")
        if commit != expected:
            raise ValueError(f"Harness is at {commit}, not {ref}. Use another destination or set ref=None.")
    if source == HARNESS_URL and ref == HARNESS_REF and commit != HARNESS_COMMIT:
        raise ValueError("The upstream release tag no longer matches the verified commit.")
    if install:
        requirements = Path(__file__).with_name("requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements),
                        "-e", f"{repo}[hf,ifeval]"], check=True)
    print(f"Harness: {repo}\nCommit: {commit}")
    return repo


def verify_harness_import(repo: Path) -> None:
    import lm_eval

    actual = Path(lm_eval.__file__).resolve().parent
    if actual != repo.resolve() / "lm_eval":
        raise RuntimeError(f"Kernel imported {actual}, but {repo} was requested. Restart the kernel after setup.")


def resolve_device_dtype(device: str, dtype: str) -> tuple[str, str]:
    import torch

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU kernel/runtime or set device='cpu'.")
    if dtype == "auto":
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                # Recent PyTorch releases may report BF16 as supported through
                # emulation on Turing. Use it automatically only where the GPU
                # has native BF16 tensor-core support (Ampere or newer).
                major, _ = torch.cuda.get_device_capability(device)
                dtype = "bfloat16" if major >= 8 and torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype = "float32"
    return device, dtype


def preflight(config: EvalConfig, harness_dir: Path, *, loaded_model: Any = None) -> dict:
    """Check the selected checkout, registry, gated downloads, and metric data."""
    config.validate()
    verify_harness_import(harness_dir)
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager()
    requested = [task for name in config.benchmarks for task in BENCHMARKS[name].tasks]
    missing = set(requested) - set(task_manager.all_tasks)
    if missing:
        raise RuntimeError(f"Selected harness is missing tasks: {sorted(missing)}")
    details = {"tasks": requested}
    if loaded_model is None:
        from huggingface_hub import HfApi, hf_hub_download

        source = normalize_model_source(config.model)
        if not Path(source).is_dir():
            revision = HfApi().model_info(source, revision=config.revision).sha
            hf_hub_download(source, "config.json", revision=revision)
            # Freeze the actual weights/tokenizer revision used by the run.
            config.revision = revision
            details["resolved_model_revision"] = revision
    if "GPQA" in config.benchmarks:
        from datasets import load_dataset

        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        details["gpqa_rows"] = len(dataset)
        del dataset
    if "IFEval" in config.benchmarks:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            if not nltk.download("punkt_tab", quiet=True, raise_on_error=True):
                raise RuntimeError("Could not download NLTK punkt_tab for IFEval.")
    if "HumanEval" in config.benchmarks:
        if sys.platform == "win32":
            raise RuntimeError("Run HumanEval in Linux/WSL; the code_eval metric does not support Windows.")
        # Must be set BEFORE importing the task's utils: it tests code_eval on import.
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        import evaluate

        evaluate.load("code_eval")
    return details


def load_model(config: EvalConfig, *, loaded_model: Any = None, tokenizer: Any = None):
    from lm_eval.models.huggingface import HFLM

    kwargs = dict(batch_size=config.batch_size, max_batch_size=8,
                  max_length=config.max_length, backend="causal")
    if loaded_model is not None:
        if tokenizer is None:
            raise ValueError("Supply the matching tokenizer with a loaded Transformers model.")
        loaded_model.eval()
        lm = HFLM(pretrained=loaded_model, tokenizer=tokenizer, **kwargs)
    else:
        device, dtype = resolve_device_dtype(config.device, config.dtype)
        kwargs.update(pretrained=normalize_model_source(config.model), revision=config.revision,
                      device=device, dtype=dtype, trust_remote_code=config.trust_remote_code)
        if tokenizer is not None or config.tokenizer is not None:
            tokenizer = tokenizer if tokenizer is not None else config.tokenizer
            if isinstance(tokenizer, (str, Path)):
                from transformers import AutoTokenizer

                # A separate tokenizer repository does not share the model commit.
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer), trust_remote_code=config.trust_remote_code)
            kwargs["tokenizer"] = tokenizer
        lm = HFLM(**kwargs)
    if config.apply_chat_template and not lm.tokenizer.chat_template:
        raise ValueError("Chat templates are enabled, but this tokenizer has no chat template.")
    return lm


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if callable(value):
        return f"{value.__module__}.{value.__name__}"
    if isinstance(value, set):
        return sorted(value)
    # The harness also stringifies types such as torch.dtype in model metadata.
    return str(value)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric_rows(benchmark: str, result: dict, *, limited: bool) -> list[dict]:
    rows = []
    # Preserve the harness's weighted group aggregates; do not average subject scores.
    for task, metrics in {**result.get("groups", {}), **result.get("results", {})}.items():
        for key, value in metrics.items():
            if "," not in key:
                continue
            metric, filter_name = key.split(",", 1)
            if metric.endswith("_stderr") or not isinstance(value, (int, float)):
                continue
            rows.append({"benchmark": benchmark, "task": task, "metric": metric,
                         "filter": filter_name, "value": value,
                         "stderr": metrics.get(f"{metric}_stderr,{filter_name}"),
                         "fewshot": result.get("n-shot", {}).get(task),
                         "evaluated_samples": result.get("n-samples", {}).get(task, {}).get("effective"),
                         "limited_run": limited})
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_suite(config: EvalConfig, harness_dir: Path, output_root: Path, *,
              loaded_model: Any = None, tokenizer: Any = None) -> Path:
    """Run each selected benchmark, retain completed artifacts, raise on failure.

    Each invocation gets a fresh directory. Full and limited runs cannot mix.
    """
    config.validate()
    verify_harness_import(harness_dir)
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    if "HumanEval" in config.benchmarks:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    mode = "full" if config.limit is None else "smoke"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root).resolve() / f"{stamp}_{mode}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "config": asdict(config), "limited_run": config.limit is not None,
                "harness_path": str(harness_dir.resolve()),
                "harness_commit": git_output(harness_dir, "rev-parse", "HEAD"),
                "harness_dirty": bool(git_output(harness_dir, "status", "--porcelain")),
                "python": platform.python_version(), "model_input": "loaded" if loaded_model is not None else "source",
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                             if d.metadata["Name"]},
                "benchmarks": {name: {"status": "pending"} for name in config.benchmarks}}
    write_json(run_dir / "manifest.json", manifest)
    print(f"Artifacts: {run_dir}")
    rows = []
    active = None
    try:
        lm = load_model(config, loaded_model=loaded_model, tokenizer=tokenizer)
        manifest["resolved_device"] = str(lm.device)
        manifest["resolved_dtype"] = str(lm.model.dtype)
        manifest["model_class"] = type(lm.model).__name__
        manifest["model_revision"] = getattr(lm.model.config, "_commit_hash", None)
        manager = TaskManager()
        for name in config.benchmarks:
            active = name
            spec = BENCHMARKS[name]
            manifest["benchmarks"][name] = {"status": "running", "tasks": spec.tasks, "fewshot": spec.fewshot}
            write_json(run_dir / "manifest.json", manifest)
            print(f"\n{name}: {', '.join(spec.tasks)} ({spec.fewshot}-shot)")
            # HumanEval's stock task expects raw function continuation, even if
            # chat templates are requested for the other benchmarks.
            result = simple_evaluate(
                model=lm, tasks=list(spec.tasks), num_fewshot=spec.fewshot,
                task_manager=manager, limit=config.limit, log_samples=config.log_samples,
                bootstrap_iters=config.bootstrap_iters,
                apply_chat_template=config.apply_chat_template and name != "HumanEval",
                fewshot_as_multiturn=False,
                random_seed=config.seed, numpy_random_seed=config.seed,
                torch_random_seed=config.seed, fewshot_random_seed=config.seed,
                confirm_run_unsafe_code=name == "HumanEval" and config.allow_humaneval_execution,
            )
            if result is None or not result.get("results"):
                raise RuntimeError(f"{name} returned no scores; run this notebook in one process.")
            destination = run_dir / name.lower()
            destination.mkdir()
            samples = result.pop("samples", {})
            write_json(destination / "results.json", result)
            if config.log_samples:
                for task, task_samples in samples.items():
                    safe_task = re.sub(r"[^\w.-]", "_", task)
                    with (destination / f"samples_{safe_task}.jsonl").open("w", encoding="utf-8") as handle:
                        for sample in task_samples:
                            handle.write(json.dumps(sample, default=_json_default) + "\n")
            rows.extend(metric_rows(name, result, limited=config.limit is not None))
            write_summary(run_dir / "summary.csv", rows)
            manifest["benchmarks"][name]["status"] = "completed"
            write_json(run_dir / "manifest.json", manifest)
            del result, samples
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if active:
            manifest["benchmarks"][active]["status"] = manifest["status"]
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "manifest.json", manifest)
    return run_dir
