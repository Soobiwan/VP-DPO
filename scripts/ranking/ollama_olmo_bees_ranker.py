"""Resumable local Ollama segmenter/ranker for the OLMo BeeS preference dataset.

The production notebook ``ollama_ranking_olmo_bees.ipynb`` is the intended entry point.
This module owns canonical-prompt rendering, exact validation, item-level caching, the
Q3-first/Q4-failure-only pipeline, and Method A/B/C-compatible export.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import threading
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests


PROMPT_VERSION = "olmo-bees-kaggle-prompts-v3"
PRODUCTION_PIPELINE_VERSION = "olmo-bees-two-pass-v1"
ITEM_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?ITEM\s*#?\s*(\d+)\s*:?\s*(?:\n|$)",
    flags=re.IGNORECASE,
)
SEGMENT_LINE_RE = re.compile(
    r"^(.+)\s*->\s*([0-9]*\.?[0-9]+)\s*->\s*([0-9]+)\s*$"
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


KAGGLE_PROMPT_NAMES = {
    "POSITIVE_PROMPT": "chosen",
    "NEGATIVE_PROMPT": "rejected",
}
LOCAL_BATCH_CONTRACT = """

LOCAL OUTPUT CONTRACT:
Return only segmented lines grouped under ITEM headers. Output exactly one matching ITEM
header for every input block, using the same ITEM number. Preserve every source character;
do not rewrite, omit, duplicate, or reorder response text. Scores must be unique, ranks must
be the complete sequence 1..N, and higher scores must have better (smaller) rank numbers.
"""


class OllamaError(RuntimeError):
    """Base exception for this module."""


class OllamaRequestError(OllamaError):
    """An Ollama request failed after all configured HTTP retries."""

    def __init__(self, message: str, attempts: int, errors: list[str], wall_seconds: float):
        super().__init__(message)
        self.attempts = attempts
        self.errors = errors
        self.wall_seconds = wall_seconds


class OutputValidationError(OllamaError):
    """The model output could not be projected losslessly onto the source response."""


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.8:27b-q3"
    num_ctx: int = 12288
    num_predict: int = 2560
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    keep_alive: str | int = "30m"
    timeout_seconds: float = 600.0
    http_max_attempts: int = 3
    validation_max_attempts: int = 3
    backoff_base_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.num_ctx < 512:
            raise ValueError("num_ctx must be at least 512")
        if not 1 <= self.num_predict < self.num_ctx:
            raise ValueError("num_predict must be positive and smaller than num_ctx")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.http_max_attempts < 1 or self.validation_max_attempts < 1:
            raise ValueError("retry counts must be positive")


@dataclass(frozen=True)
class Segment:
    text: str
    value_score: float
    rank: int


@dataclass(frozen=True)
class RankingItem:
    item_id: str
    dataset_index: int
    side: str
    prompt: str
    response: str

    def __post_init__(self) -> None:
        if self.side not in {"chosen", "rejected"}:
            raise ValueError(f"side must be chosen or rejected, got {self.side!r}")
        if not self.prompt.strip() or not self.response.strip():
            raise ValueError("prompt and response must both contain text")


def _jsonable(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def message_text(value: Any) -> str:
    """Convert HF conversational messages (or a string) to plain task text."""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(f"Expected a string or message list, got {type(value).__name__}")
    parts: list[str] = []
    for message in value:
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise TypeError("Each conversational message must have a string content field")
        content = message["content"].strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def load_olmo_bees_dataset(
    *,
    split: str = "train",
    dataset_id: str | None = None,
    local_path: str | os.PathLike[str] | None = None,
):
    """Load a Hub dataset when configured, otherwise the workspace-local HF artifact.

    ``dataset_id`` is intentionally not guessed.  If it is supplied but unavailable, an
    existing ``local_path`` is used and the Hub failure is printed.  This makes the notebook
    usable before or after the local BeeS artifact is published to a Hub namespace.
    """
    from datasets import load_dataset, load_from_disk

    hub_error: Exception | None = None
    if dataset_id:
        try:
            dataset = load_dataset(dataset_id, split=split)
            source = f"hf://datasets/{dataset_id}/{split}"
            return dataset, source
        except Exception as exc:  # Hub auth/offline failures should permit the exact local copy.
            hub_error = exc

    if local_path is None:
        if hub_error is not None:
            raise RuntimeError(f"Could not load Hugging Face dataset {dataset_id!r}") from hub_error
        raise ValueError("Set dataset_id or local_path")

    path = Path(local_path).expanduser().resolve()
    if not path.exists():
        if hub_error is not None:
            raise RuntimeError(
                f"Hub dataset {dataset_id!r} failed and local fallback does not exist: {path}"
            ) from hub_error
        raise FileNotFoundError(path)
    if hub_error is not None:
        print(f"Hub load failed ({type(hub_error).__name__}); using local HF artifact: {path}")

    if path.is_dir() and (path / "dataset_dict.json").is_file():
        dataset_dict = load_from_disk(str(path))
        if split not in dataset_dict:
            raise KeyError(f"Split {split!r} is absent; available: {list(dataset_dict)}")
        return dataset_dict[split], f"local://{path}#{split}"
    if path.is_dir() and (path / "state.json").is_file():
        return load_from_disk(str(path)), f"local://{path}"

    data_files: str | list[str]
    if path.is_dir():
        matches = sorted(str(candidate) for candidate in path.glob("*.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No Dataset save or JSONL files found under {path}")
        data_files = matches
    else:
        data_files = str(path)
    dataset = load_dataset("json", data_files=data_files, split="train")
    if "split" in dataset.column_names:
        dataset = dataset.filter(lambda row: row["split"] == split)
    return dataset, f"local-json://{path}#{split}"


def make_ranking_items(
    dataset: Any,
    indices: Iterable[int],
    sides: Sequence[str] = ("chosen",),
    split: str = "train",
) -> list[RankingItem]:
    """Materialize stable chosen/rejected ranking inputs from a HF Dataset."""
    missing = {"prompt", *sides} - set(dataset.column_names)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")
    items: list[RankingItem] = []
    for raw_index in indices:
        index = int(raw_index)
        if not 0 <= index < len(dataset):
            raise IndexError(f"Dataset index {index} is outside [0, {len(dataset)})")
        row = dataset[index]
        prompt = message_text(row["prompt"])
        for side in sides:
            response = message_text(row[side])
            items.append(
                RankingItem(
                    item_id=f"{split}:{index}:{side}",
                    dataset_index=index,
                    side=side,
                    prompt=prompt,
                    response=response,
                )
            )
    return items


@lru_cache(maxsize=4)
def load_kaggle_prompt_templates(
    notebook_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Load the exact chosen/rejected prompt literals from kaggle_ranking.ipynb.

    The assignments are parsed with ``ast.literal_eval``; notebook code is never executed.
    Keeping one prompt source prevents the local and Kaggle ranking criteria from drifting.
    """
    path = (
        Path(notebook_path).expanduser().resolve()
        if notebook_path is not None
        else Path(__file__).resolve().parents[2] / "notebooks/kaggle/kaggle_ranking.ipynb"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"The canonical ranking prompt notebook is missing: {path}"
        )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    templates: dict[str, str] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not any(name in source for name in KAGGLE_PROMPT_NAMES):
            continue
        tree = ast.parse(source, filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in KAGGLE_PROMPT_NAMES:
                continue
            value_node = node.value
            if (
                isinstance(value_node, ast.Call)
                and isinstance(value_node.func, ast.Attribute)
                and value_node.func.attr == "dedent"
                and len(value_node.args) == 1
            ):
                value_node = value_node.args[0]
            literal = ast.literal_eval(value_node)
            if not isinstance(literal, str):
                raise TypeError(f"{target.id} must be a string literal")
            template = textwrap.dedent(literal)
            if template.count("__BATCH_INPUT__") != 1:
                raise ValueError(f"{target.id} must contain exactly one batch placeholder")
            templates[KAGGLE_PROMPT_NAMES[target.id]] = template
    missing = {"chosen", "rejected"} - templates.keys()
    if missing:
        raise KeyError(f"Missing canonical Kaggle prompt(s): {sorted(missing)}")
    return templates


def build_ranking_prompt(items: Sequence[RankingItem]) -> str:
    if not items:
        raise ValueError("Cannot build an empty prompt")
    sides = {item.side for item in items}
    if len(sides) != 1:
        raise ValueError("A packed Kaggle prompt must contain only one response side")
    side = next(iter(sides))
    blocks = []
    for position, item in enumerate(items):
        blocks.append(
            f"\nITEM {position}\n\n"
            f"USER PROMPT:\n\n{item.prompt}\n\n"
            f"MODEL RESPONSE:\n\n{item.response}\n"
        )
    batch_input = "\n\n".join(blocks)
    canonical = load_kaggle_prompt_templates()[side].replace(
        "__BATCH_INPUT__", batch_input
    )
    return canonical + LOCAL_BATCH_CONTRACT


def approximate_tokens(text: str) -> int:
    """A conservative context guard; server counters remain the benchmark truth."""
    return max(1, math.ceil(len(text) / 3.0))


def approximate_output_tokens(items: Sequence[RankingItem]) -> int:
    """Conservative allowance for copied response text plus score/rank annotations."""
    return sum(max(1, math.ceil(len(item.response) / 2.5) + 64) for item in items)


def prompt_fits(prompt: str, config: OllamaConfig) -> bool:
    return approximate_tokens(prompt) + config.num_predict <= config.num_ctx


def pack_items(
    items: Sequence[RankingItem], config: OllamaConfig, batch_size: int
) -> list[list[RankingItem]]:
    """Pack up to batch_size items without exceeding the conservative context guard."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batches: list[list[RankingItem]] = []
    # Canonical Kaggle uses distinct chosen-usefulness and rejected-damage prompts.
    # Grouping by side permits real packed batches even when source items alternate sides;
    # execute_strategy restores the caller's original order after inference.
    side_order = list(dict.fromkeys(item.side for item in items))
    for side in side_order:
        pending: list[RankingItem] = []
        for item in (candidate for candidate in items if candidate.side == side):
            single_prompt = build_ranking_prompt([item])
            single_output = approximate_output_tokens([item])
            if not prompt_fits(single_prompt, config) or single_output > config.num_predict:
                estimate = approximate_tokens(single_prompt)
                raise ValueError(
                    f"{item.item_id} needs about {estimate} input tokens and "
                    f"{single_output} output tokens; configured num_ctx={config.num_ctx}, "
                    f"num_predict={config.num_predict}. Choose a larger tier."
                )
            candidate = [*pending, item]
            if pending and (
                len(candidate) > batch_size
                or not prompt_fits(build_ranking_prompt(candidate), config)
                or approximate_output_tokens(candidate) > config.num_predict
            ):
                batches.append(pending)
                pending = [item]
            else:
                pending = candidate
        if pending:
            batches.append(pending)
    return batches


def strip_thinking(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


def parse_segment_lines(raw_text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SEGMENT_LINE_RE.match(line)
        if match:
            segments.append(
                {
                    "text": match.group(1).rstrip(),
                    "value_score": float(match.group(2)),
                    "rank": int(match.group(3)),
                }
            )
    return segments


def split_batch_output(raw_text: str, n_items: int) -> list[list[dict[str, Any]]]:
    """Split 0-based or 1-based ITEM blocks, retaining the Kaggle parser behavior."""
    if n_items < 1:
        return []
    text = strip_thinking(raw_text)
    parts = ITEM_HEADER_RE.split(text)
    raw_blocks: dict[int, str] = {}
    for position in range(1, len(parts) - 1, 2):
        raw_blocks[int(parts[position])] = parts[position + 1]
    if not raw_blocks:
        return [parse_segment_lines(text)] if n_items == 1 else [[] for _ in range(n_items)]

    keys = set(raw_blocks)
    one_based = 0 not in keys and min(keys) >= 1 and max(keys) <= n_items
    normalized: dict[int, str] = {}
    for index, block in raw_blocks.items():
        target = index - 1 if one_based else index
        if 0 <= target < n_items:
            normalized[target] = block
    return [parse_segment_lines(normalized.get(index, "")) for index in range(n_items)]


def _normalized_with_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            if chars and end < len(text):
                chars.append(" ")
                spans.append((index, end))
            index = end
        else:
            chars.append(text[index])
            spans.append((index, index + 1))
            index += 1
    return "".join(chars), spans


def project_segments_to_source(
    parsed: Sequence[Mapping[str, Any]], original: str
) -> list[Segment]:
    """Restore exact source whitespace while requiring lossless ordered model spans."""
    if not parsed:
        raise OutputValidationError("No segment lines were parsed")
    normalized_original, source_spans = _normalized_with_spans(original)
    normalized_segments = [" ".join(str(part["text"]).split()) for part in parsed]
    if any(not text for text in normalized_segments):
        raise OutputValidationError("The model returned an empty segment")

    locations: list[tuple[int, int]] = []
    cursor = 0
    for index, needle in enumerate(normalized_segments):
        found = normalized_original.find(needle, cursor)
        if found < 0:
            raise OutputValidationError(
                f"Segment {index} does not occur after the previous span: {needle[:100]!r}"
            )
        gap = normalized_original[cursor:found]
        if gap.strip():
            raise OutputValidationError(
                f"Segment {index} omitted or changed source text: {gap[:100]!r}"
            )
        locations.append((found, found + len(needle)))
        cursor = found + len(needle)
    if normalized_original[cursor:].strip():
        raise OutputValidationError(
            f"Model omitted trailing source text: {normalized_original[cursor:cursor + 100]!r}"
        )

    source_starts = [source_spans[start][0] for start, _ in locations]
    projected: list[Segment] = []
    for index, part in enumerate(parsed):
        start = 0 if index == 0 else source_starts[index]
        end = source_starts[index + 1] if index + 1 < len(parsed) else len(original)
        projected.append(
            Segment(
                text=original[start:end],
                value_score=float(part["value_score"]),
                rank=int(part["rank"]),
            )
        )
    return projected


def validate_segments(segments: Sequence[Segment], original: str) -> dict[str, Any]:
    scores = [segment.value_score for segment in segments]
    ranks = [segment.rank for segment in segments]
    reconstructed = "".join(segment.text for segment in segments)
    issues: list[str] = []
    if not segments:
        issues.append("no segments")
    if reconstructed != original:
        issues.append("segments do not exactly reconstruct the source response")
    if any(not 0.01 <= score <= 1.0 for score in scores):
        issues.append("scores must be in [0.01, 1.00]")
    if len(set(scores)) != len(scores):
        issues.append("scores must be unique")
    if sorted(ranks) != list(range(1, len(segments) + 1)):
        issues.append("ranks must be a unique complete 1..N sequence")
    by_rank = sorted(segments, key=lambda segment: segment.rank)
    if any(
        former.value_score <= latter.value_score
        for former, latter in zip(by_rank, by_rank[1:])
    ):
        issues.append("scores must strictly decrease as rank numbers increase")
    return {
        "valid": not issues,
        "exact_match": reconstructed == original,
        "n_segments": len(segments),
        "issues": issues,
    }


def normalize_ranks_from_scores(segments: Sequence[Segment]) -> tuple[list[Segment], bool]:
    """Derive the redundant rank column from unique scores when a model miscounts N.

    Some otherwise lossless outputs assign ranks as if un-emitted candidate fragments still
    existed.  The score already defines the requested total order, so repairing that redundant
    column is deterministic and avoids an expensive regeneration.  Duplicate scores are not
    repaired because they do not define a unique ordering and will still fail validation.
    """
    scores = [segment.value_score for segment in segments]
    if len(set(scores)) != len(scores):
        return list(segments), False
    order = sorted(range(len(segments)), key=lambda index: scores[index], reverse=True)
    corrected = {segment_index: rank for rank, segment_index in enumerate(order, start=1)}
    repaired = any(segment.rank != corrected[index] for index, segment in enumerate(segments))
    if not repaired:
        return list(segments), False
    return [
        Segment(
            text=segment.text,
            value_score=segment.value_score,
            rank=corrected[index],
        )
        for index, segment in enumerate(segments)
    ], True


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def version(self) -> str:
        response = requests.get(f"{self.base_url}/api/version", timeout=10)
        response.raise_for_status()
        return str(response.json().get("version", "unknown"))

    def models(self) -> list[dict[str, Any]]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=30)
        response.raise_for_status()
        return list(response.json().get("models", []))

    def ensure_model(self) -> dict[str, Any]:
        available = self.models()
        for model in available:
            if model.get("name") == self.config.model or model.get("model") == self.config.model:
                return model
        names = [str(model.get("name")) for model in available]
        raise OllamaError(
            f"Model {self.config.model!r} is not installed. Run the model setup cell "
            f"in ollama_ranking_olmo_bees.ipynb. Available models: {names}"
        )

    def running_models(self) -> list[dict[str, Any]]:
        response = requests.get(f"{self.base_url}/api/ps", timeout=30)
        response.raise_for_status()
        return list(response.json().get("models", []))

    def unload(self) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.config.model, "keep_alive": 0},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "seed": self.config.seed,
            },
        }
        errors: list[str] = []
        started = time.perf_counter()
        for attempt in range(1, self.config.http_max_attempts + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=body,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not payload.get("done", False):
                    raise OllamaError("Non-streaming Ollama response did not finish")
                payload["_client_wall_seconds"] = time.perf_counter() - started
                payload["_http_attempts"] = attempt
                payload["_http_errors"] = errors
                payload["response"] = strip_thinking(str(payload.get("response", "")))
                return payload
            except (requests.RequestException, ValueError, OllamaError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < self.config.http_max_attempts:
                    time.sleep(self.config.backoff_base_seconds * (2 ** (attempt - 1)))
        wall_seconds = time.perf_counter() - started
        raise OllamaRequestError(
            f"Ollama failed after {self.config.http_max_attempts} HTTP attempt(s)",
            attempts=self.config.http_max_attempts,
            errors=errors,
            wall_seconds=wall_seconds,
        )


class ResultCache:
    """Persistent cache of fully validated per-item results, keyed by exact inputs/options."""

    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    @staticmethod
    def key(item: RankingItem, config: OllamaConfig) -> str:
        canonical_prompt = load_kaggle_prompt_templates()[item.side] + LOCAL_BATCH_CONTRACT
        identity = {
            "prompt_version": PROMPT_VERSION,
            "prompt_template_sha256": hashlib.sha256(
                canonical_prompt.encode("utf-8")
            ).hexdigest(),
            "model": config.model,
            "num_ctx": config.num_ctx,
            "num_predict": config.num_predict,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
            "think": False,
            "item": asdict(item),
        }
        encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, item: RankingItem, config: OllamaConfig) -> dict[str, Any] | None:
        path = self.directory / f"{self.key(item, config)}.json"
        if not path.is_file():
            return None
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not result.get("validation", {}).get("valid"):
            return None
        result["disk_cache_hit"] = True
        return result

    def put(self, item: RankingItem, config: OllamaConfig, result: Mapping[str, Any]) -> Path:
        if not result.get("validation", {}).get("valid"):
            raise ValueError("Refusing to cache an invalid result")
        path = self.directory / f"{self.key(item, config)}.json"
        temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        content = json.dumps(result, ensure_ascii=False, indent=2, default=_jsonable) + "\n"
        with self._write_lock:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
        return path


def _call_record(
    *,
    response: Mapping[str, Any] | None,
    request_kind: str,
    item_ids: Sequence[str],
    validation_round: int,
    prompt_chars: int,
    error: OllamaRequestError | None = None,
) -> dict[str, Any]:
    if response is None:
        return {
            "request_kind": request_kind,
            "item_ids": list(item_ids),
            "validation_round": validation_round,
            "prompt_chars": prompt_chars,
            "output_chars": 0,
            "success": False,
            "client_wall_seconds": error.wall_seconds if error else 0.0,
            "http_attempts": error.attempts if error else 0,
            "http_errors": error.errors if error else [],
        }
    fields = (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
        "done_reason",
    )
    record = {
        "request_kind": request_kind,
        "item_ids": list(item_ids),
        "validation_round": validation_round,
        "prompt_chars": prompt_chars,
        "output_chars": len(str(response.get("response", ""))),
        "success": True,
        "client_wall_seconds": float(response.get("_client_wall_seconds", 0.0)),
        "http_attempts": int(response.get("_http_attempts", 1)),
        "http_errors": list(response.get("_http_errors", [])),
    }
    record.update({field: response.get(field, 0) for field in fields})
    return record


def _valid_result(
    item: RankingItem,
    segments: Sequence[Segment],
    raw_text: str,
    validation_round: int,
    producer_strategy: str,
    rank_repaired: bool = False,
) -> dict[str, Any]:
    validation = validate_segments(segments, item.response)
    return {
        **asdict(item),
        "segments": [asdict(segment) for segment in segments],
        "validation": validation,
        "validation_attempts": validation_round,
        "raw_text": raw_text,
        "producer_strategy": producer_strategy,
        "rank_repaired": rank_repaired,
        "disk_cache_hit": False,
    }


def _failed_result(
    item: RankingItem, errors: Sequence[str], validation_attempts: int, strategy: str
) -> dict[str, Any]:
    return {
        **asdict(item),
        "segments": [],
        "validation": {"valid": False, "exact_match": False, "n_segments": 0, "issues": list(errors)},
        "validation_attempts": validation_attempts,
        "raw_text": "",
        "producer_strategy": strategy,
        "rank_repaired": False,
        "disk_cache_hit": False,
    }


def _parse_one(item: RankingItem, parsed: Sequence[Mapping[str, Any]], raw_text: str, attempt: int, strategy: str) -> dict[str, Any]:
    segments = project_segments_to_source(parsed, item.response)
    segments, rank_repaired = normalize_ranks_from_scores(segments)
    result = _valid_result(
        item,
        segments,
        raw_text,
        attempt,
        strategy,
        rank_repaired=rank_repaired,
    )
    if not result["validation"]["valid"]:
        raise OutputValidationError("; ".join(result["validation"]["issues"]))
    return result


def _process_one(
    client: OllamaClient,
    item: RankingItem,
    *,
    strategy: str,
    start_round: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    prompt = build_ranking_prompt([item])
    if not prompt_fits(prompt, client.config):
        raise ValueError(f"Single prompt for {item.item_id} exceeds the configured context guard")
    for validation_round in range(start_round, client.config.validation_max_attempts + 1):
        try:
            response = client.generate(prompt)
            calls.append(
                _call_record(
                    response=response,
                    request_kind="single",
                    item_ids=[item.item_id],
                    validation_round=validation_round,
                    prompt_chars=len(prompt),
                )
            )
            raw_text = str(response.get("response", ""))
            parsed = split_batch_output(raw_text, 1)[0]
            return _parse_one(item, parsed, raw_text, validation_round, strategy), calls
        except OllamaRequestError as exc:
            calls.append(
                _call_record(
                    response=None,
                    error=exc,
                    request_kind="single",
                    item_ids=[item.item_id],
                    validation_round=validation_round,
                    prompt_chars=len(prompt),
                )
            )
            errors.append(f"round {validation_round} request: {exc}; {' | '.join(exc.errors)}")
        except OutputValidationError as exc:
            errors.append(f"round {validation_round} validation: {exc}")
    return _failed_result(item, errors, client.config.validation_max_attempts, strategy), calls


def execute_strategy(
    client: OllamaClient,
    items: Sequence[RankingItem],
    *,
    strategy: str,
    batch_size: int = 2,
    concurrent_workers: int = 2,
    result_cache: ResultCache | None = None,
    use_result_cache: bool = False,
    progress_callback: Callable[[RankingItem, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a strategy, caching and reporting each item as soon as it finishes."""
    supported = {"single_sequential", "single_concurrent", "packed_batch"}
    if strategy not in supported:
        raise ValueError(f"strategy must be one of {sorted(supported)}")
    if concurrent_workers < 1:
        raise ValueError("concurrent_workers must be positive")
    started = time.perf_counter()
    outcomes: dict[str, dict[str, Any]] = {}
    pending: list[RankingItem] = []

    def store_outcome(item: RankingItem, outcome: dict[str, Any]) -> None:
        outcomes[item.item_id] = outcome
        if (
            result_cache is not None
            and outcome["validation"]["valid"]
            and not outcome.get("disk_cache_hit")
        ):
            result_cache.put(item, client.config, outcome)
        if progress_callback is not None:
            progress_callback(item, outcome)

    for item in items:
        cached = result_cache.get(item, client.config) if use_result_cache and result_cache else None
        if cached is not None:
            store_outcome(item, cached)
        else:
            pending.append(item)

    calls: list[dict[str, Any]] = []
    if strategy == "single_sequential":
        for item in pending:
            outcome, item_calls = _process_one(client, item, strategy=strategy)
            store_outcome(item, outcome)
            calls.extend(item_calls)
    elif strategy == "single_concurrent":
        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = {
                executor.submit(_process_one, client, item, strategy=strategy): item
                for item in pending
            }
            for future in as_completed(futures):
                outcome, item_calls = future.result()
                store_outcome(futures[future], outcome)
                calls.extend(item_calls)
    else:
        for batch in pack_items(pending, client.config, batch_size):
            prompt = build_ranking_prompt(batch)
            packed_errors: dict[str, list[str]] = {item.item_id: [] for item in batch}
            try:
                response = client.generate(prompt)
                calls.append(
                    _call_record(
                        response=response,
                        request_kind="packed",
                        item_ids=[item.item_id for item in batch],
                        validation_round=1,
                        prompt_chars=len(prompt),
                    )
                )
                raw_text = str(response.get("response", ""))
                parsed_items = split_batch_output(raw_text, len(batch))
                for item, parsed in zip(batch, parsed_items):
                    try:
                        store_outcome(item, _parse_one(item, parsed, raw_text, 1, strategy))
                    except OutputValidationError as exc:
                        packed_errors[item.item_id].append(f"packed validation: {exc}")
            except OllamaRequestError as exc:
                calls.append(
                    _call_record(
                        response=None,
                        error=exc,
                        request_kind="packed",
                        item_ids=[item.item_id for item in batch],
                        validation_round=1,
                        prompt_chars=len(prompt),
                    )
                )
                for item in batch:
                    packed_errors[item.item_id].append(
                        f"packed request: {exc}; {' | '.join(exc.errors)}"
                    )

            # A malformed packed item is retried by itself, preventing one bad item from
            # forcing successful batch members through the model again.
            for item in batch:
                if item.item_id in outcomes:
                    continue
                outcome, retry_calls = _process_one(
                    client, item, strategy=strategy, start_round=2
                )
                if not outcome["validation"]["valid"]:
                    outcome["validation"]["issues"] = [
                        *packed_errors[item.item_id],
                        *outcome["validation"]["issues"],
                    ]
                store_outcome(item, outcome)
                calls.extend(retry_calls)

    ordered = [outcomes[item.item_id] for item in items]
    return {
        "strategy": strategy,
        "model": client.config.model,
        "config": asdict(client.config),
        "prompt_version": PROMPT_VERSION,
        "wall_seconds": time.perf_counter() - started,
        "outcomes": ordered,
        "calls": sorted(calls, key=lambda call: (call["validation_round"], call["item_ids"])),
    }


def canonical_prompt_fingerprints() -> dict[str, dict[str, Any]]:
    """Return auditable hashes for the untouched prompt literals in kaggle_ranking.ipynb."""
    return {
        side: {
            "characters": len(prompt),
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "batch_placeholders": prompt.count("__BATCH_INPUT__"),
        }
        for side, prompt in load_kaggle_prompt_templates().items()
    }


def validate_production_items_fit(
    items: Sequence[RankingItem], configs: Sequence[OllamaConfig]
) -> dict[str, Any]:
    """Fail before a long run if any source row would be truncated by either model."""
    if not configs:
        raise ValueError("At least one model configuration is required")
    largest_input = ("", 0)
    largest_output = ("", 0)
    checked_tiers: set[tuple[int, int]] = set()
    for config in configs:
        tier = (config.num_ctx, config.num_predict)
        if tier in checked_tiers:
            continue
        checked_tiers.add(tier)
        for item in items:
            prompt = build_ranking_prompt([item])
            input_estimate = approximate_tokens(prompt)
            output_estimate = approximate_output_tokens([item])
            if input_estimate > largest_input[1]:
                largest_input = (item.item_id, input_estimate)
            if output_estimate > largest_output[1]:
                largest_output = (item.item_id, output_estimate)
            if input_estimate + config.num_predict > config.num_ctx:
                raise ValueError(
                    f"{item.item_id} needs about {input_estimate} input tokens plus the "
                    f"{config.num_predict}-token reserve; num_ctx={config.num_ctx} is too small"
                )
            if output_estimate > config.num_predict:
                raise ValueError(
                    f"{item.item_id} needs about {output_estimate} output tokens; "
                    f"num_predict={config.num_predict} is too small"
                )
    return {
        "items": len(items),
        "context_tiers": sorted(checked_tiers),
        "largest_input_item": largest_input[0],
        "largest_input_tokens_estimate": largest_input[1],
        "largest_output_item": largest_output[0],
        "largest_output_tokens_estimate": largest_output[1],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_jsonable)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")
    os.replace(temporary, path)
    return path


class ProductionLedger:
    """Append-only, item-level pass ledger used to resume multi-day production runs."""

    def __init__(self, directory: str | os.PathLike[str], manifest: Mapping[str, Any]):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "run_manifest.json"
        self.events_path = self.directory / "pass_events.jsonl"
        expected = json.loads(json.dumps(manifest, sort_keys=True, default=_jsonable))
        if self.manifest_path.is_file():
            observed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if observed != expected:
                raise RuntimeError(
                    "The production output directory belongs to a different dataset, prompt, "
                    "model, or generation configuration. Choose a new OUTPUT_DIR instead of "
                    "mixing checkpoints."
                )
        else:
            _atomic_write_json(self.manifest_path, expected)
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        self._load_events()

    def _load_events(self) -> None:
        if not self.events_path.is_file():
            return
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        valid_events: list[dict[str, Any]] = []
        trailing_partial = False
        for position, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if position != len(lines) - 1:
                    raise RuntimeError(
                        f"Corrupt production ledger at line {position + 1}: {self.events_path}"
                    )
                trailing_partial = True
                break
            valid_events.append(event)
            self.records[(str(event["pass"]), str(event["item_id"]))] = event
        if trailing_partial:
            _atomic_write_jsonl(self.events_path, valid_events)

    def get(self, pass_name: str, item_id: str) -> dict[str, Any] | None:
        return self.records.get((pass_name, item_id))

    def record(
        self,
        pass_name: str,
        item: RankingItem,
        outcome: Mapping[str, Any],
        model: str,
    ) -> dict[str, Any]:
        validation = outcome.get("validation", {})
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pass": pass_name,
            "item_id": item.item_id,
            "dataset_index": item.dataset_index,
            "side": item.side,
            "model": model,
            "valid": bool(validation.get("valid")),
            "exact_match": bool(validation.get("exact_match")),
            "segments": int(validation.get("n_segments", 0) or 0),
            "validation_attempts": int(outcome.get("validation_attempts", 0) or 0),
            "rank_repaired": bool(outcome.get("rank_repaired")),
            "disk_cache_hit": bool(outcome.get("disk_cache_hit")),
            "issues": list(validation.get("issues", [])),
        }
        encoded = json.dumps(event, ensure_ascii=False, default=_jsonable) + "\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self.records[(pass_name, item.item_id)] = event
        return event


def _inference_identity(config: OllamaConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "num_ctx": config.num_ctx,
        "num_predict": config.num_predict,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "think": False,
        "http_max_attempts": config.http_max_attempts,
        "validation_max_attempts": config.validation_max_attempts,
    }


def _production_manifest(
    items: Sequence[RankingItem],
    datasets: Mapping[str, Any],
    primary_config: OllamaConfig,
    fallback_config: OllamaConfig,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for item in items:
        digest.update(
            json.dumps(asdict(item), sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "pipeline_version": PRODUCTION_PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "canonical_prompts": canonical_prompt_fingerprints(),
        "local_output_contract_sha256": hashlib.sha256(
            LOCAL_BATCH_CONTRACT.encode("utf-8")
        ).hexdigest(),
        "dataset_splits": {str(split): len(dataset) for split, dataset in datasets.items()},
        "dataset_items_sha256": digest.hexdigest(),
        "items": len(items),
        "primary": _inference_identity(primary_config),
        "fallback": _inference_identity(fallback_config),
    }


class _SilentProgress:
    def update(self, amount: int = 1) -> None:
        del amount

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def close(self) -> None:
        return None


def _progress_bar(*, total: int, initial: int, description: str, enabled: bool):
    if not enabled:
        return _SilentProgress()
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("Install tqdm before starting the production run") from exc
    return tqdm(
        total=total,
        initial=initial,
        desc=description,
        unit="item",
        dynamic_ncols=True,
        smoothing=0.05,
    )


def _chunks(items: Sequence[RankingItem], chunk_size: int) -> Iterable[list[RankingItem]]:
    for offset in range(0, len(items), chunk_size):
        yield list(items[offset : offset + chunk_size])


def run_two_pass_production(
    datasets: Mapping[str, Any],
    primary_client: OllamaClient,
    fallback_client: OllamaClient,
    result_cache: ResultCache,
    output_dir: str | os.PathLike[str],
    *,
    primary_strategy: str = "single_concurrent",
    fallback_strategy: str = "single_sequential",
    chunk_size: int = 8,
    concurrent_workers: int = 2,
    batch_size: int = 2,
    show_progress: bool = True,
    require_complete: bool = True,
    retry_failed_fallback_on_resume: bool = True,
    unload_when_done: bool = True,
) -> dict[str, Any]:
    """Run all Q3 items first, then send only Q3 failures through Q4.

    Each valid result is atomically cached before the progress callback advances. Every
    attempted item is also appended to a durable pass ledger, so restarting this function
    skips completed work. Final JSONL files use the Method A/B/C combined-pair schema.
    """
    if not datasets:
        raise ValueError("At least one dataset split is required")
    if primary_client.config.model == fallback_client.config.model:
        raise ValueError("Primary and fallback models must be different")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    ordered_datasets = {str(split): dataset for split, dataset in datasets.items()}
    items: list[RankingItem] = []
    for split, dataset in ordered_datasets.items():
        items.extend(
            make_ranking_items(
                dataset,
                range(len(dataset)),
                sides=("chosen", "rejected"),
                split=split,
            )
        )
    fit_report = validate_production_items_fit(
        items, [primary_client.config, fallback_client.config]
    )
    output_path = Path(output_dir).expanduser().resolve()
    manifest = _production_manifest(
        items, ordered_datasets, primary_client.config, fallback_client.config
    )
    ledger = ProductionLedger(output_path, manifest)

    primary_client.ensure_model()
    fallback_client.ensure_model()
    fallback_client.unload()

    def completed_items(
        pass_name: str, candidates: Sequence[RankingItem], client: OllamaClient
    ) -> list[RankingItem]:
        completed: list[RankingItem] = []
        for item in candidates:
            event = ledger.get(pass_name, item.item_id)
            if event is None:
                continue
            if (
                pass_name == "fallback"
                and not event["valid"]
                and retry_failed_fallback_on_resume
            ):
                continue
            if event["valid"] and result_cache.get(item, client.config) is None:
                continue
            completed.append(item)
        return completed

    def run_pass(
        pass_name: str,
        candidates: Sequence[RankingItem],
        client: OllamaClient,
        strategy: str,
    ) -> None:
        already_complete = completed_items(pass_name, candidates, client)
        completed_ids = {item.item_id for item in already_complete}
        remaining = [item for item in candidates if item.item_id not in completed_ids]
        existing_events = [ledger.get(pass_name, item.item_id) for item in already_complete]
        counters = {
            "valid": sum(bool(event and event["valid"]) for event in existing_events),
            "failed": sum(bool(event and not event["valid"]) for event in existing_events),
            "cached": sum(bool(event and event.get("disk_cache_hit")) for event in existing_events),
        }
        bar = _progress_bar(
            total=len(candidates),
            initial=len(already_complete),
            description=f"{pass_name} {client.config.model}",
            enabled=show_progress,
        )

        def on_outcome(item: RankingItem, outcome: Mapping[str, Any]) -> None:
            event = ledger.record(pass_name, item, outcome, client.config.model)
            key = "valid" if event["valid"] else "failed"
            counters[key] += 1
            counters["cached"] += int(event["disk_cache_hit"])
            bar.update(1)
            bar.set_postfix(
                valid=counters["valid"],
                failed=counters["failed"],
                cache=counters["cached"],
                last=item.item_id,
                refresh=False,
            )

        try:
            for chunk in _chunks(remaining, chunk_size):
                execute_strategy(
                    client,
                    chunk,
                    strategy=strategy,
                    batch_size=batch_size,
                    concurrent_workers=concurrent_workers,
                    result_cache=result_cache,
                    use_result_cache=True,
                    progress_callback=on_outcome,
                )
        finally:
            bar.close()

    # Keep identical canonical prefixes adjacent so Ollama can reuse prompt/KV state. The
    # exporter restores split/index/side order independently of inference completion order.
    side_order = {"chosen": 0, "rejected": 1}
    inference_items = sorted(items, key=lambda item: side_order[item.side])

    # This pass ordering is intentional: Q4 cannot start until every Q3 item has a ledger event.
    run_pass("primary", inference_items, primary_client, primary_strategy)
    primary_client.unload()
    primary_events = {item.item_id: ledger.get("primary", item.item_id) for item in items}
    if any(event is None for event in primary_events.values()):
        raise RuntimeError("Primary pass ended without recording every item")
    fallback_items = [
        item
        for item in inference_items
        if not bool(primary_events[item.item_id]["valid"])
    ]
    if fallback_items:
        run_pass("fallback", fallback_items, fallback_client, fallback_strategy)

    selected: dict[str, tuple[dict[str, Any], str]] = {}
    unresolved: list[dict[str, Any]] = []
    primary_valid = 0
    fallback_valid = 0
    for item in items:
        primary_event = ledger.get("primary", item.item_id)
        result = (
            result_cache.get(item, primary_client.config)
            if primary_event and primary_event["valid"]
            else None
        )
        selected_model = primary_client.config.model
        if result is not None:
            primary_valid += 1
        else:
            fallback_event = ledger.get("fallback", item.item_id)
            result = (
                result_cache.get(item, fallback_client.config)
                if fallback_event and fallback_event["valid"]
                else None
            )
            selected_model = fallback_client.config.model
            if result is not None:
                fallback_valid += 1
        if result is None:
            unresolved.append(
                {
                    "item_id": item.item_id,
                    "dataset_index": item.dataset_index,
                    "side": item.side,
                    "primary": primary_event,
                    "fallback": ledger.get("fallback", item.item_id),
                }
            )
        else:
            selected[item.item_id] = (result, selected_model)

    export_paths: dict[str, str] = {}
    complete_rows = 0
    for split, dataset in ordered_datasets.items():
        rows: list[dict[str, Any]] = []
        split_complete = True
        for index in range(len(dataset)):
            chosen = selected.get(f"{split}:{index}:chosen")
            rejected = selected.get(f"{split}:{index}:rejected")
            if chosen is None or rejected is None:
                split_complete = False
                continue
            chosen_result, _ = chosen
            rejected_result, _ = rejected
            rows.append(
                {
                    "index": index,
                    "prompt": chosen_result["prompt"],
                    "positive_response": chosen_result["response"],
                    "positive_segments": chosen_result["segments"],
                    "negative_response": rejected_result["response"],
                    "negative_segments": rejected_result["segments"],
                }
            )
        complete_rows += len(rows)
        filename = f"method_abc_{split}.jsonl" if split_complete else f"method_abc_{split}.partial.jsonl"
        destination = _atomic_write_jsonl(output_path / filename, rows)
        export_paths[split] = str(destination)

    unresolved_path = _atomic_write_jsonl(output_path / "unresolved_items.jsonl", unresolved)
    summary = {
        "status": "complete" if not unresolved else "incomplete",
        "pipeline_version": PRODUCTION_PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "canonical_prompts": canonical_prompt_fingerprints(),
        "items": len(items),
        "source_rows": sum(len(dataset) for dataset in ordered_datasets.values()),
        "complete_rows": complete_rows,
        "primary_model": primary_client.config.model,
        "primary_valid_items": primary_valid,
        "primary_failed_items": len(items) - primary_valid,
        "fallback_model": fallback_client.config.model,
        "fallback_valid_items": fallback_valid,
        "unresolved_items": len(unresolved),
        "fit_report": fit_report,
        "exports": export_paths,
        "unresolved_path": str(unresolved_path),
        "manifest_path": str(ledger.manifest_path),
        "events_path": str(ledger.events_path),
    }
    summary_path = _atomic_write_json(output_path / "production_summary.json", summary)
    summary["summary_path"] = str(summary_path)
    if unload_when_done:
        fallback_client.unload()
    if unresolved and require_complete:
        raise RuntimeError(
            f"{len(unresolved)} item(s) remain invalid after Q4; see {unresolved_path}"
        )
    return summary


__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "OllamaError",
    "OutputValidationError",
    "RankingItem",
    "ResultCache",
    "Segment",
    "canonical_prompt_fingerprints",
    "approximate_output_tokens",
    "approximate_tokens",
    "build_ranking_prompt",
    "execute_strategy",
    "load_kaggle_prompt_templates",
    "load_olmo_bees_dataset",
    "make_ranking_items",
    "normalize_ranks_from_scores",
    "pack_items",
    "parse_segment_lines",
    "prompt_fits",
    "project_segments_to_source",
    "run_two_pass_production",
    "split_batch_output",
    "validate_segments",
    "validate_production_items_fit",
]
