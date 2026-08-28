#!/usr/bin/env python
"""Generate OLMo responses for AlpacaEval 2 and execute the judge notebook."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_APPROVAL = WORKSPACE_ROOT / "artifacts/olmo2_bees/approved_model.json"
DEFAULT_MODEL = WORKSPACE_ROOT / "artifacts/olmo2_bees/olmo2_1b_dpo_full/final"
DEFAULT_NOTEBOOK = WORKSPACE_ROOT / "alpaca-eval-2-judge.ipynb"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "artifacts/olmo2_bees/alpaca_eval2/model_outputs.json"

# Keep package and model caches inside the workspace, consistent with the OLMo notebooks.
os.environ.setdefault("HF_HOME", str(WORKSPACE_ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(WORKSPACE_ROOT / ".cache/huggingface/datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(WORKSPACE_ROOT / ".cache/huggingface/transformers"))
os.environ.setdefault("TMPDIR", str(WORKSPACE_ROOT / ".tmp"))


def approved_model_path() -> Path:
    if DEFAULT_APPROVAL.is_file():
        approval = json.loads(DEFAULT_APPROVAL.read_text(encoding="utf-8"))
        if approval.get("approved") and approval.get("model_path"):
            return Path(approval["model_path"]).expanduser().resolve()
    return DEFAULT_MODEL.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all 805 AlpacaEval responses with the approved OLMo checkpoint, "
            "then execute alpaca-eval-2-judge.ipynb. Existing generations are resumed."
        )
    )
    parser.add_argument("--model", type=Path, default=approved_model_path())
    parser.add_argument("--model-name", default="olmo2-1b-bees-dpo")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--env-file", type=Path, default=WORKSPACE_ROOT / ".env")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N prompts for a smoke test; evaluation is skipped.",
    )
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generation-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_input_tokens < 1 or args.max_new_tokens < 1:
        parser.error("token limits must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit is not None and args.eval_only:
        parser.error("--limit cannot be combined with --eval-only")
    if (
        args.limit is not None
        and args.output.expanduser().resolve() == DEFAULT_OUTPUT.resolve()
    ):
        args.output = WORKSPACE_ROOT / ".tmp/alpaca_eval2_smoke.json"
    return args


def load_environment(env_file: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "python-dotenv is missing; run `.venv/bin/python -m pip install "
            "-r BeeS/requirements-olmo2.txt`"
        ) from exc

    env_file = env_file.expanduser().resolve()
    if not env_file.is_file():
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    load_dotenv(env_file, override=False)
    os.environ["ALPACA_ENV_FILE"] = str(env_file)


def require_judge_environment() -> None:
    required = (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    )
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Populate these values in .env before running the judge: " + ", ".join(missing)
        )


def load_eval_rows(limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("The `huggingface-hub` package is required for generation") from exc

    # Loading the raw official JSON works with datasets>=5, which no longer executes
    # the dataset repository's legacy alpaca_eval.py loading script.
    dataset_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise RuntimeError(f"Expected a JSON list in {dataset_path}")
    if limit is not None:
        dataset = dataset[:limit]
    rows = [dict(row) for row in dataset]
    if not rows or any(not row.get("instruction") for row in rows):
        raise RuntimeError("The AlpacaEval dataset did not contain usable instructions")
    return rows


def load_resumable_outputs(
    output_path: Path,
    eval_rows: list[dict[str, Any]],
    model_name: str,
    overwrite: bool,
) -> list[dict[str, str]]:
    if overwrite or not output_path.is_file():
        return []
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(existing, list):
        raise RuntimeError(f"Expected a JSON list in {output_path}")
    if len(existing) > len(eval_rows):
        raise RuntimeError(f"{output_path} has more rows than the selected evaluation set")
    for index, row in enumerate(existing):
        if row.get("instruction") != eval_rows[index]["instruction"]:
            raise RuntimeError(
                f"Cannot resume {output_path}: instruction {index} does not match AlpacaEval"
            )
        if row.get("generator") != model_name or not isinstance(row.get("output"), str):
            raise RuntimeError(
                f"Cannot resume {output_path}: row {index} has the wrong generator/schema"
            )
    return existing


def save_outputs(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)


def generate_outputs(args: argparse.Namespace) -> tuple[Path, int]:
    try:
        import torch
        from tqdm.auto import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Generation requires torch, tqdm, and transformers") from exc

    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")

    eval_rows = load_eval_rows(args.limit)
    generated = load_resumable_outputs(
        output_path, eval_rows, args.model_name, args.overwrite
    )
    if len(generated) == len(eval_rows):
        print(f"Generations already complete: {output_path}")
        return output_path, len(eval_rows)

    print(f"Loading {model_path} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    context_limit = int(getattr(model.config, "max_position_embeddings", 4096))
    max_input_tokens = min(args.max_input_tokens, context_limit - args.max_new_tokens)
    if max_input_tokens < 1:
        raise ValueError(
            f"--max-new-tokens={args.max_new_tokens} leaves no room in the "
            f"{context_limit}-token context window"
        )

    start = len(generated)
    pending = eval_rows[start:]
    progress = tqdm(total=len(eval_rows), initial=start, desc="AlpacaEval generations")
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["instruction"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(args.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

        with torch.inference_mode():
            sequences = model.generate(**inputs, **generation_kwargs)
        new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for row, completion in zip(batch, completions, strict=True):
            generated.append(
                {
                    "instruction": row["instruction"],
                    "output": completion.strip(),
                    "generator": args.model_name,
                }
            )
        save_outputs(output_path, generated)
        progress.update(len(batch))
    progress.close()
    print(f"Saved {len(generated)} generations to {output_path}")
    return output_path, len(eval_rows)


def validate_outputs(output_path: Path, expected_rows: int = 805) -> int:
    if not output_path.is_file():
        raise FileNotFoundError(f"Generation file not found: {output_path}")
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("instruction"), str)
        or not isinstance(row.get("output"), str)
        for row in rows
    ):
        raise RuntimeError(f"Invalid AlpacaEval output schema in {output_path}")
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} complete AlpacaEval generations, found {len(rows)}"
        )
    return len(rows)


def execute_judge_notebook(args: argparse.Namespace, output_path: Path) -> Path:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError("Notebook execution requires nbformat and nbclient") from exc

    notebook_path = args.notebook.expanduser().resolve()
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Judge notebook not found: {notebook_path}")

    eval_dir = output_path.parent / "judge_results"
    executed_path = eval_dir / f"{notebook_path.stem}.executed.ipynb"
    eval_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "ALPACA_GENERATIONS_PATH": str(output_path),
            "ALPACA_MODEL_NAME": args.model_name,
            "ALPACA_OUTPUT_DIR": str(eval_dir),
        }
    )

    annotation_cache = (
        eval_dir / "custom_annotator" / "annotations_seed0_configs.json"
    )
    if annotation_cache.is_file():
        cached_rows = json.loads(annotation_cache.read_text(encoding="utf-8"))
        if not isinstance(cached_rows, list):
            raise RuntimeError(f"Invalid AlpacaEval annotation cache: {annotation_cache}")
        generated_rows = json.loads(output_path.read_text(encoding="utf-8"))
        generated_pairs = {
            (row["instruction"], row["output"]) for row in generated_rows
        }
        reusable_pairs = {
            (row.get("instruction"), row.get("output_2"))
            for row in cached_rows
            if (row.get("instruction"), row.get("output_2")) in generated_pairs
        }
        print(
            f"Resuming judge with {len(reusable_pairs)} matching cached annotations; "
            f"{len(generated_pairs) - len(reusable_pairs)} remain."
        )

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="vpdpo-olmo",
        resources={"metadata": {"path": str(WORKSPACE_ROOT)}},
    )
    print(f"Executing judge notebook: {notebook_path}")
    try:
        client.execute()
    finally:
        nbformat.write(notebook, executed_path)
    print(f"Saved executed notebook to {executed_path}")
    return executed_path


def main() -> None:
    args = parse_args()
    load_environment(args.env_file)

    will_evaluate = not args.generation_only and args.limit is None
    if will_evaluate:
        require_judge_environment()

    if args.eval_only:
        output_path = args.output.expanduser().resolve()
        validate_outputs(output_path)
    else:
        output_path, selected_rows = generate_outputs(args)
        if args.limit is None:
            validate_outputs(output_path, expected_rows=selected_rows)

    if will_evaluate:
        execute_judge_notebook(args, output_path)
    elif args.limit is not None:
        print("Smoke-test generation complete; skipping evaluation because --limit was used.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Completed batches are saved and will resume next time.", file=sys.stderr)
        raise SystemExit(130)
