from __future__ import annotations

import argparse
import functools
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any

from .common import configure_workspace, package_versions, sha256_file, write_json
from .structured_preference import (
    REQUESTED_VARIANTS,
    SUPPORTED_VARIANTS,
    VARIANT_DESCRIPTIONS,
    VARIANT_FORMULAS,
    StructuredPreferenceCollator,
    aggregate_segment_logps,
    prepare_segmented_dataset,
    run_loss_self_tests,
    structured_pair_loss,
)


FP16_INITIAL_SCALE = 32.0
VOCABULARY_SHARD_SIZE = 8192
REFERENCE_PROJECTION_CHUNK_SIZE = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and train all segmented OLMo structured-preference variants"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Tokenize and align the segmented JSONL")
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--source-jsonl", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--model-revision", default=None)
    prepare.add_argument("--max-length", type=int, default=1024)
    prepare.add_argument("--force", action="store_true")

    reference = subparsers.add_parser(
        "reference", help="Precompute reusable reference segment log-probabilities"
    )
    reference.add_argument("--workspace", type=Path, required=True)
    reference.add_argument("--dataset-path", type=Path, required=True)
    reference.add_argument("--split", default="train")
    reference.add_argument("--model-id", required=True)
    reference.add_argument("--model-revision", default=None)
    reference.add_argument("--output-dir", type=Path, required=True)
    reference.add_argument("--flush-every", type=int, default=25)

    train = subparsers.add_parser("train", help="Train one structured-preference variant")
    train.add_argument("--workspace", type=Path, required=True)
    train.add_argument("--dataset-path", type=Path, required=True)
    train.add_argument("--reference-cache", type=Path, required=True)
    train.add_argument("--train-split", default="train")
    train.add_argument("--model-id", required=True)
    train.add_argument("--model-revision", default=None)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--run-name", default=None)
    train.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--learning-rate", type=float, default=2e-6)
    train.add_argument("--beta", type=float, default=0.1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--logging-steps", type=int, default=1)
    train.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="0 saves only the final model; a positive value enables large restart checkpoints",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--transformer-layer-class", default="Olmo2DecoderLayer")
    train.add_argument(
        "--activation-offloading",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--resume", action="store_true")

    test = subparsers.add_parser("self-test", help="Run CPU loss-formulation tests")
    test.add_argument("--workspace", type=Path, required=True)
    return parser


def _load_split(dataset_path: Path, split: str):
    from datasets import Dataset, DatasetDict, load_from_disk

    loaded = load_from_disk(str(dataset_path.resolve()))
    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            raise KeyError(f"Dataset has no split {split!r}")
        return loaded[split]
    if isinstance(loaded, Dataset):
        if split != "train":
            raise KeyError(f"A single Dataset can only be addressed as 'train', not {split!r}")
        return loaded
    raise TypeError(f"Unsupported dataset object: {type(loaded)!r}")


def _reference_identity(args: argparse.Namespace, dataset: Any) -> dict[str, Any]:
    return {
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_fingerprint": dataset._fingerprint,
        "split": args.split,
        "rows": len(dataset),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
    }


def _read_reference_part(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                dataset_index = int(record["dataset_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid reference cache record at {path}:{line_number}") from error
            if dataset_index in completed:
                raise RuntimeError(f"Duplicate dataset_index={dataset_index} in {path}")
            completed[dataset_index] = record
    return completed


def _pad_reference_pair(row: dict[str, Any], pad_token_id: int, device: Any):
    import torch

    sequences = [row["chosen_input_ids"], row["rejected_input_ids"]]
    segment_sequences = [row["chosen_segment_ids"], row["rejected_segment_ids"]]
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((2, width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((2, width), dtype=torch.long, device=device)
    segment_ids = torch.full((2, width), -1, dtype=torch.long, device=device)
    for index, (sequence, ownership) in enumerate(zip(sequences, segment_sequences, strict=True)):
        length = len(sequence)
        input_ids[index, :length] = torch.as_tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        segment_ids[index, :length] = torch.as_tensor(
            ownership, dtype=torch.long, device=device
        )
    return input_ids, attention_mask, segment_ids


def _reference_segment_logps(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    segment_ids: Any,
    segment_counts: list[int],
):
    import torch
    import torch.nn.functional as functional

    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state[:, :-1, :]
        labels = input_ids[:, 1:]
        shifted_segment_ids = segment_ids[:, 1:]
        positions = shifted_segment_ids.ge(0).nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise RuntimeError("Reference batch has no segment-owned tokens")
        loss_hidden = hidden_states[positions[:, 0], positions[:, 1]]
        loss_labels = labels[positions[:, 0], positions[:, 1]]
        token_logps: list[Any] = []
        for start in range(0, loss_hidden.shape[0], REFERENCE_PROJECTION_CHUNK_SIZE):
            stop = start + REFERENCE_PROJECTION_CHUNK_SIZE
            logits = functional.linear(
                loss_hidden[start:stop], model.lm_head.weight, model.lm_head.bias
            ).float()
            label_chunk = loss_labels[start:stop]
            selected = torch.gather(logits, -1, label_chunk.unsqueeze(-1)).squeeze(-1)
            token_logps.append(selected - torch.logsumexp(logits, dim=-1))
        token_logps = torch.cat(token_logps)
        max_segments = max(segment_counts)
        segment_logps = torch.zeros(
            (2, max_segments), dtype=torch.float32, device=input_ids.device
        )
        segment_logps.index_put_(
            (positions[:, 0], shifted_segment_ids[positions[:, 0], positions[:, 1]]),
            token_logps,
            accumulate=True,
        )
        return [
            segment_logps[row, :segment_count].cpu().tolist()
            for row, segment_count in enumerate(segment_counts)
        ]


def run_reference(args: argparse.Namespace) -> None:
    configure_workspace(args.workspace)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import torch
    from accelerate import Accelerator
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerator = Accelerator(mixed_precision="fp16")
    if accelerator.num_processes != 2:
        raise RuntimeError("Reference precomputation requires exactly two Accelerate processes")
    dataset = _load_split(args.dataset_path, args.split)
    required = {
        "dataset_index",
        "row_id",
        "chosen_input_ids",
        "chosen_segment_ids",
        "chosen_segment_scores",
        "rejected_input_ids",
        "rejected_segment_ids",
        "rejected_segment_scores",
    }
    missing = required.difference(dataset.column_names)
    if missing:
        raise RuntimeError(f"Prepared dataset is missing columns: {sorted(missing)}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dataset_path = args.output_dir / "dataset"
    manifest_path = args.output_dir / "reference_manifest.json"
    identity = _reference_identity(args, dataset)
    if cache_dataset_path.is_dir() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in identity.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Reference cache identity mismatch: {mismatches}")
        if accelerator.is_main_process:
            print(f"Reference segment log-probabilities already exist: {cache_dataset_path}")
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return
    if cache_dataset_path.exists() or manifest_path.exists():
        raise RuntimeError(
            f"Incomplete merged reference cache under {args.output_dir}; retain part files but "
            "remove the incomplete merged dataset/manifest before resuming"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(accelerator.device)
    model.config.use_cache = False
    model.eval()

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    part_path = args.output_dir / f"part-rank-{rank:02d}.jsonl"
    completed = _read_reference_part(part_path)
    assigned_indices = list(range(rank, len(dataset), world_size))
    unexpected = set(completed).difference(assigned_indices)
    if unexpected:
        raise RuntimeError(f"Reference part {part_path} contains indices owned by another rank")

    from tqdm.auto import tqdm

    pending = [index for index in assigned_indices if index not in completed]
    with part_path.open("a", encoding="utf-8") as handle:
        for completed_since_flush, dataset_index in enumerate(
            tqdm(
                pending,
                desc=f"Reference segments rank {rank}",
                disable=not accelerator.is_local_main_process and rank != 1,
            ),
            1,
        ):
            row = dataset[dataset_index]
            input_ids, attention_mask, segment_ids = _pad_reference_pair(
                row, tokenizer.pad_token_id, accelerator.device
            )
            segment_counts = [
                len(row["chosen_segment_scores"]),
                len(row["rejected_segment_scores"]),
            ]
            with accelerator.autocast():
                chosen_logps, rejected_logps = _reference_segment_logps(
                    model,
                    input_ids,
                    attention_mask,
                    segment_ids,
                    segment_counts,
                )
            if not all(math.isfinite(value) for value in chosen_logps + rejected_logps):
                raise RuntimeError(f"Non-finite reference statistics at dataset index {dataset_index}")
            handle.write(
                json.dumps(
                    {
                        "dataset_index": dataset_index,
                        "row_id": int(row["row_id"]),
                        "ref_chosen_segment_logps": chosen_logps,
                        "ref_rejected_segment_logps": rejected_logps,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            if completed_since_flush % args.flush_every == 0:
                handle.flush()
                os.fsync(handle.fileno())
        handle.flush()
        os.fsync(handle.fileno())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        records: dict[int, dict[str, Any]] = {}
        for process_rank in range(world_size):
            part_records = _read_reference_part(
                args.output_dir / f"part-rank-{process_rank:02d}.jsonl"
            )
            overlap = set(records).intersection(part_records)
            if overlap:
                raise RuntimeError(f"Duplicate reference indices across ranks: {sorted(overlap)[:10]}")
            records.update(part_records)
        expected_indices = set(range(len(dataset)))
        if set(records) != expected_indices:
            raise RuntimeError(
                "Reference cache coverage mismatch; "
                f"missing={sorted(expected_indices - set(records))[:10]}, "
                f"extra={sorted(set(records) - expected_indices)[:10]}"
            )
        ordered_records = [records[index] for index in range(len(dataset))]
        for index, record in enumerate(ordered_records):
            source_row = dataset[index]
            if record["row_id"] != int(source_row["row_id"]):
                raise RuntimeError(f"Reference row identity mismatch at dataset index {index}")
            if len(record["ref_chosen_segment_logps"]) != len(
                source_row["chosen_segment_scores"]
            ) or len(record["ref_rejected_segment_logps"]) != len(
                source_row["rejected_segment_scores"]
            ):
                raise RuntimeError(f"Reference segment count mismatch at dataset index {index}")
        cache_dataset = Dataset.from_list(ordered_records)
        with tempfile.TemporaryDirectory(
            prefix=".reference-segments-", dir=args.output_dir
        ) as temp:
            staged = Path(temp) / "dataset"
            cache_dataset.save_to_disk(staged)
            staged.replace(cache_dataset_path)
        write_json(
            manifest_path,
            {
                **identity,
                "world_size": world_size,
                "compute_dtype": "float16",
                "lossless_segment_coverage": True,
                "columns": cache_dataset.column_names,
                "packages": package_versions(
                    ["torch", "transformers", "datasets", "accelerate", "safetensors"]
                ),
            },
        )
        print(f"Saved shared reference segment cache: {cache_dataset_path}")
    accelerator.wait_for_everyone()
    del model
    torch.cuda.empty_cache()
    accelerator.end_training()


def last_complete_checkpoint(output_dir: Path, world_size: int = 2) -> str | None:
    candidates = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        reverse=True,
    )
    for checkpoint in candidates:
        required = [
            checkpoint / "pytorch_model_fsdp.bin",
            checkpoint / "model.safetensors",
            checkpoint / "scheduler.pt",
            checkpoint / "scaler.pt",
            checkpoint / "trainer_state.json",
            *(checkpoint / f"paged_optimizer_rank_{rank:02d}.pt" for rank in range(world_size)),
            *(checkpoint / f"rng_state_{rank}.pth" for rank in range(world_size)),
        ]
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            return str(checkpoint)
    if candidates:
        raise RuntimeError(
            f"No complete checkpoint under {output_dir}; incomplete candidates: "
            f"{[path.name for path in candidates]}"
        )
    return None


def run_train(args: argparse.Namespace) -> None:
    configure_workspace(args.workspace)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_functional
    import torch
    from accelerate.utils import GradScalerKwargs, save_fsdp_model
    from bitsandbytes.utils import sync_gpu
    from datasets import load_from_disk
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from transformers.distributed.fsdp import get_fsdp_ckpt_kwargs
    from transformers.trainer import SCHEDULER_NAME
    from trl.models.activation_offloading import get_act_offloading_ctx_manager

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("WORLD_SIZE", "1") != "2":
        raise RuntimeError("Launch structured training with exactly two Accelerate processes")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    set_seed(args.seed)

    train_dataset = _load_split(args.dataset_path, args.train_split)
    reference_dataset_path = args.reference_cache.resolve() / "dataset"
    reference_manifest_path = args.reference_cache.resolve() / "reference_manifest.json"
    if not reference_dataset_path.is_dir() or not reference_manifest_path.is_file():
        raise FileNotFoundError(
            f"Run the shared reference stage first: {reference_dataset_path} / "
            f"{reference_manifest_path}"
        )
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    # A Kaggle reference-only output is remounted under /kaggle/input in the training
    # session, so its absolute dataset path necessarily changes.  Content identity is
    # enforced by the deterministic Dataset fingerprint plus the remaining fields.
    expected_reference = {
        "dataset_fingerprint": train_dataset._fingerprint,
        "split": args.train_split,
        "rows": len(train_dataset),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
    }
    reference_mismatches = {
        key: (reference_manifest.get(key), value)
        for key, value in expected_reference.items()
        if reference_manifest.get(key) != value
    }
    if reference_mismatches:
        raise RuntimeError(f"Reference cache does not match this run: {reference_mismatches}")
    reference_dataset = load_from_disk(str(reference_dataset_path))
    if len(reference_dataset) != len(train_dataset):
        raise RuntimeError("Reference cache row count differs from the training dataset")
    if reference_dataset["dataset_index"] != train_dataset["dataset_index"]:
        raise RuntimeError("Reference cache order differs from the training dataset")
    if reference_dataset["row_id"] != train_dataset["row_id"]:
        raise RuntimeError("Reference cache row IDs differ from the training dataset")
    train_dataset = train_dataset.add_column(
        "ref_chosen_segment_logps", reference_dataset["ref_chosen_segment_logps"]
    ).add_column(
        "ref_rejected_segment_logps", reference_dataset["ref_rejected_segment_logps"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    class VocabularyShardLinear(torch.nn.Linear):
        """An independently FSDP-sharded slice of the OLMo vocabulary head."""

    class CheckpointedChunkedLMHead(torch.nn.Module):
        """Project/reduce bounded vocabulary slices without retaining full logits."""

        def __init__(self, source):
            super().__init__()
            self.in_features = source.in_features
            self.out_features = source.out_features
            self.vocabulary_shards = torch.nn.ModuleList()
            self.vocabulary_offsets: list[int] = []
            for start in range(0, source.out_features, VOCABULARY_SHARD_SIZE):
                stop = min(start + VOCABULARY_SHARD_SIZE, source.out_features)
                shard = VocabularyShardLinear(
                    source.in_features,
                    stop - start,
                    bias=source.bias is not None,
                    device=source.weight.device,
                    dtype=source.weight.dtype,
                )
                shard.weight = torch.nn.Parameter(
                    source.weight[start:stop].detach().clone(),
                    requires_grad=source.weight.requires_grad,
                )
                if source.bias is not None:
                    shard.bias = torch.nn.Parameter(
                        source.bias[start:stop].detach().clone(),
                        requires_grad=source.bias.requires_grad,
                    )
                self.vocabulary_shards.append(shard)
                self.vocabulary_offsets.append(start)

        def forward(self, hidden_states, labels):
            selected_logits = torch.zeros_like(hidden_states[..., 0], dtype=torch.float32)
            log_normalizer = None
            for start, shard in zip(
                self.vocabulary_offsets, self.vocabulary_shards, strict=True
            ):
                stop = start + shard.out_features

                def project_and_reduce(hidden, target, head=shard, offset=start, limit=stop):
                    logits = head(hidden).float()
                    shard_normalizer = torch.logsumexp(logits, dim=-1)
                    belongs_here = target.ge(offset) & target.lt(limit)
                    local_target = (target - offset).clamp(0, head.out_features - 1)
                    selected = torch.gather(
                        logits, -1, local_target.unsqueeze(-1)
                    ).squeeze(-1)
                    return shard_normalizer, selected * belongs_here

                shard_normalizer, shard_selected = activation_checkpoint(
                    project_and_reduce,
                    hidden_states,
                    labels,
                    use_reentrant=False,
                )
                selected_logits = selected_logits + shard_selected
                log_normalizer = (
                    shard_normalizer
                    if log_normalizer is None
                    else torch.logaddexp(log_normalizer, shard_normalizer)
                )
            return selected_logits - log_normalizer

    class DTensorPagedAdamW32bit(bnb.optim.PagedAdamW32bit):
        """Run bitsandbytes kernels on each FSDP2 DTensor's owned local shard."""

        @staticmethod
        def _local(tensor):
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        def _init_local_state(
            self, group, parameter, local_parameter, group_index, param_index
        ):
            config = self.get_config(group_index, param_index, group)
            state = self.state[parameter]
            state["step"] = 0
            state["state1"] = self.get_state_buffer(local_parameter, dtype=torch.float32)
            state["state2"] = self.get_state_buffer(local_parameter, dtype=torch.float32)
            if config["percentile_clipping"] < 100:
                state["gnorm_vec"] = torch.zeros(100, device=local_parameter.device)
            if config["max_unorm"] > 0.0:
                state["unorm_vec"] = torch.zeros(1, device=local_parameter.device)

        def _update_local(
            self, group, parameter, local_parameter, local_grad, group_index, param_index
        ):
            state = self.state[parameter]
            config = self.get_config(group_index, param_index, group)
            state["step"] += 1
            step = state["step"]
            if config["percentile_clipping"] < 100:
                _, _, gnorm_scale = bnb_functional.percentile_clipping(
                    local_grad,
                    state["gnorm_vec"],
                    step,
                    config["percentile_clipping"],
                )
            else:
                gnorm_scale = 1.0
            bnb_functional.optimizer_update_32bit(
                self.optimizer_name,
                local_grad,
                local_parameter,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                config["lr"],
                state["state2"],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config.get("alpha", 0.0),
                config["weight_decay"],
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            if not self.initialized:
                self.check_overrides()
                self.to_gpu()
                self.initialized = True
            last_local_parameter = None
            for group_index, group in enumerate(self.param_groups):
                for param_index, parameter in enumerate(group["params"]):
                    if parameter.grad is None:
                        continue
                    local_parameter = self._local(parameter)
                    local_grad = self._local(parameter.grad).contiguous()
                    if not self.state[parameter]:
                        self._init_local_state(
                            group, parameter, local_parameter, group_index, param_index
                        )
                    self.prefetch_state(parameter)
                    self._update_local(
                        group,
                        parameter,
                        local_parameter,
                        local_grad,
                        group_index,
                        param_index,
                    )
                    sync_gpu(local_parameter)
                    last_local_parameter = local_parameter
            if self.is_paged and last_local_parameter is not None:
                sync_gpu(last_local_parameter)
            return loss

    def consolidate_saved_vocabulary_head(model_dir: Path) -> None:
        from safetensors.torch import load_file, save_file

        weight_files = sorted(model_dir.glob("*.safetensors"))
        tensors = {}
        head_slices: dict[int, torch.Tensor] = {}
        prefix = "lm_head.vocabulary_shards."
        for weight_file in weight_files:
            for name, tensor in load_file(weight_file, device="cpu").items():
                if name.startswith(prefix) and name.endswith(".weight"):
                    shard_index = int(name[len(prefix) : -len(".weight")])
                    head_slices[shard_index] = tensor
                else:
                    tensors[name] = tensor
        if not head_slices:
            return
        if sorted(head_slices) != list(range(len(head_slices))):
            raise RuntimeError(f"Incomplete vocabulary head slices: {sorted(head_slices)}")
        tensors["lm_head.weight"] = torch.cat(
            [head_slices[index] for index in range(len(head_slices))], dim=0
        )
        temporary = model_dir / "model.consolidated.safetensors.tmp"
        save_file(tensors, temporary, metadata={"format": "pt"})
        for weight_file in weight_files:
            weight_file.unlink()
        (model_dir / "model.safetensors.index.json").unlink(missing_ok=True)
        temporary.replace(model_dir / "model.safetensors")

    class StructuredFSDPTrainer(Trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.maybe_activation_offload_context = (
                get_act_offloading_ctx_manager(model=self.model, max_fwd_stash_size=1)
                if args.activation_offloading
                else None
            )
            self.structured_metrics: dict[str, list[float]] = {
                "chosen_objective": [],
                "rejected_objective": [],
                "preference_margin": [],
            }

        def training_step(self, *step_args, **step_kwargs):
            if self.maybe_activation_offload_context is None:
                return super().training_step(*step_args, **step_kwargs)
            with self.maybe_activation_offload_context:
                return super().training_step(*step_args, **step_kwargs)

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            per_token_logps = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                structured_labels=inputs["input_ids"][..., 1:],
            )
            segment_count = inputs["segment_scores"].shape[1]
            policy_segment_logps, segment_lengths = aggregate_segment_logps(
                per_token_logps,
                inputs["token_segment_ids"],
                segment_count,
            )
            loss, chosen, rejected, margins = structured_pair_loss(
                policy_segment_logps,
                inputs["reference_segment_logps"],
                inputs["segment_scores"],
                inputs["segment_ranks"],
                segment_lengths,
                inputs["segment_mask"],
                variant=args.variant,
                beta=args.beta,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {args.variant} loss: {loss.detach()}")
            gathered = {
                "chosen_objective": self.accelerator.gather(chosen.detach()).mean().item(),
                "rejected_objective": self.accelerator.gather(rejected.detach()).mean().item(),
                "preference_margin": self.accelerator.gather(margins.detach()).mean().item(),
            }
            for key, value in gathered.items():
                self.structured_metrics[key].append(value)
            torch.cuda.empty_cache()
            outputs = {
                "chosen_objective": chosen.detach(),
                "rejected_objective": rejected.detach(),
                "preference_margin": margins.detach(),
            }
            return (loss, outputs) if return_outputs else loss

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            if self.model.training:
                for key, values in self.structured_metrics.items():
                    if values:
                        logs[key] = sum(values) / len(values)
                        values.clear()
            super().log(logs, start_time)

        def create_optimizer(self):
            super().create_optimizer()
            optimizer = self.optimizer
            optimizer_args = getattr(optimizer, "args", None)
            is_paged_32bit = bool(
                getattr(optimizer, "is_paged", False)
                or getattr(optimizer_args, "is_paged", False)
            ) and int(
                getattr(optimizer, "optim_bits", 0)
                or getattr(optimizer_args, "optim_bits", 0)
            ) == 32
            if is_paged_32bit and not isinstance(optimizer, DTensorPagedAdamW32bit):
                optimizer.__class__ = DTensorPagedAdamW32bit
            return optimizer

        def _build_accelerator_args(self, **kwargs):
            accelerator_args = super()._build_accelerator_args(**kwargs)
            accelerator_args.setdefault("kwargs_handlers", []).append(
                GradScalerKwargs(init_scale=FP16_INITIAL_SCALE, growth_interval=1000)
            )
            return accelerator_args

        def _paged_optimizer_path(self, checkpoint: str | Path) -> Path:
            return Path(checkpoint) / f"paged_optimizer_rank_{self.args.process_index:02d}.pt"

        def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
            if not self.is_fsdp_enabled:
                return super()._save_optimizer_and_scheduler(output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model,
                output_dir,
                **get_fsdp_ckpt_kwargs(),
            )
            torch.save(self.optimizer.state_dict(), self._paged_optimizer_path(output_dir))
            self.accelerator.wait_for_everyone()
            if self.args.should_save:
                torch.save(self.lr_scheduler.state_dict(), Path(output_dir) / SCHEDULER_NAME)
            self.accelerator.wait_for_everyone()

        def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
            if checkpoint is None or not self.is_fsdp_enabled:
                return super()._load_optimizer_and_scheduler(checkpoint)
            optimizer_path = self._paged_optimizer_path(checkpoint)
            scheduler_path = Path(checkpoint) / SCHEDULER_NAME
            if not optimizer_path.exists() or not scheduler_path.exists():
                raise FileNotFoundError(
                    f"Incomplete paged-FSDP checkpoint: {optimizer_path} / {scheduler_path}"
                )
            optimizer_state = torch.load(
                optimizer_path, map_location="cpu", weights_only=True
            )
            bnb_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
            bnb_optimizer.load_state_dict(optimizer_state, move_to_device=False)
            for group in bnb_optimizer.param_groups:
                for parameter in group["params"]:
                    state = bnb_optimizer.state.get(parameter)
                    if not state:
                        continue
                    for state_name in ("state1", "state2"):
                        saved = state.get(state_name)
                        if not isinstance(saved, torch.Tensor):
                            continue
                        if parameter.numel() >= 100_000:
                            restored = bnb_functional.get_paged(
                                *saved.shape,
                                dtype=saved.dtype,
                                device=parameter.device,
                            )
                            bnb_functional.fill(restored, 0)
                            restored.copy_(saved, non_blocking=False)
                            bnb_optimizer.page_mng.paged_tensors.append(restored)
                        else:
                            restored = saved.to(parameter.device)
                        state[state_name] = restored
            scheduler_state = torch.load(
                scheduler_path, map_location="cpu", weights_only=True
            )
            self.lr_scheduler.load_state_dict(scheduler_state)
            self.accelerator.wait_for_everyone()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.lm_head = CheckpointedChunkedLMHead(model.lm_head)
    parameter_info = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    if parameter_info["total_parameters"] != parameter_info["trainable_parameters"]:
        raise RuntimeError(f"Full-parameter training check failed: {parameter_info}")

    standard_model_forward = model.forward

    @functools.wraps(standard_model_forward)
    def memory_efficient_model_forward(
        self, *model_args, structured_labels=None, **model_kwargs
    ):
        if structured_labels is None:
            return standard_model_forward(*model_args, **model_kwargs)
        model_kwargs.pop("labels", None)
        model_kwargs["return_dict"] = True
        outputs = self.model(*model_args, **model_kwargs)
        hidden_states = outputs.last_hidden_state[..., :-1, :]
        return self.lm_head(hidden_states, structured_labels)

    model.forward = MethodType(memory_efficient_model_forward, model)

    save_enabled = args.save_steps > 0
    run_name = args.run_name or f"olmo2-1b-segmented-{args.variant.lower()}"
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=run_name,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",
        fp16=True,
        bf16=False,
        tf32=False,
        fsdp=True,
        fsdp_config={
            "version": 2,
            "reshard_after_forward": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": [
                args.transformer_layer_class,
                "VocabularyShardLinear",
            ],
            "activation_checkpointing": True,
            "cpu_offload": False,
            "cpu_ram_efficient_loading": False,
            "state_dict_type": "FULL_STATE_DICT",
            "use_orig_params": True,
            "sync_module_states": True,
            "limit_all_gathers": True,
            "forward_prefetch": False,
            "backward_prefetch": "NO_PREFETCH",
        },
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps" if save_enabled else "no",
        save_steps=args.save_steps if save_enabled else 500,
        save_total_limit=1,
        eval_strategy="no",
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        torch_empty_cache_steps=10,
    )
    trainer = StructuredFSDPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=StructuredPreferenceCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.create_optimizer()
    optimizer_class = (
        f"{trainer.optimizer.__class__.__module__}.{trainer.optimizer.__class__.__name__}"
    )
    optimizer_args = getattr(trainer.optimizer, "args", None)
    optimizer_is_paged = bool(
        getattr(trainer.optimizer, "is_paged", False)
        or getattr(optimizer_args, "is_paged", False)
        or isinstance(trainer.optimizer, bnb.optim.PagedAdamW32bit)
    )
    optimizer_bits = int(
        getattr(trainer.optimizer, "optim_bits", 0)
        or getattr(optimizer_args, "optim_bits", 0)
        or (32 if isinstance(trainer.optimizer, bnb.optim.PagedAdamW32bit) else 0)
    )
    if not optimizer_is_paged or optimizer_bits != 32:
        raise RuntimeError(
            f"Expected paged 32-bit AdamW, got {optimizer_class} "
            f"(is_paged={optimizer_is_paged}, bits={optimizer_bits})"
        )

    checkpoint = last_complete_checkpoint(args.output_dir) if args.resume and save_enabled else None
    result = trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    local_memory = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.int64,
        device=trainer.accelerator.device,
    )
    all_memory = trainer.accelerator.gather(local_memory).cpu().reshape(-1, 2)
    peak_cuda_memory = [
        {
            "rank": rank,
            "allocated_gib": round(int(values[0]) / 2**30, 3),
            "reserved_gib": round(int(values[1]) / 2**30, 3),
        }
        for rank, values in enumerate(all_memory)
    ]
    if trainer.accelerator.is_main_process:
        consolidate_saved_vocabulary_head(final_dir)
        tokenizer.save_pretrained(final_dir)
        weight_files = sorted(final_dir.glob("*.safetensors"))
        if not weight_files:
            raise RuntimeError(f"No safetensors weights saved under {final_dir}")
        saved_weight_dtypes: set[str] = set()
        saved_tensor_count = 0
        for weight_file in weight_files:
            with safe_open(weight_file, framework="pt", device="cpu") as handle:
                for tensor_name in handle.keys():
                    saved_weight_dtypes.add(str(handle.get_slice(tensor_name).get_dtype()))
                    saved_tensor_count += 1
        if saved_weight_dtypes != {"F32"}:
            raise RuntimeError(f"Final checkpoint is not all FP32: {saved_weight_dtypes}")
        weight_files_sha256 = {
            weight_file.name: sha256_file(weight_file) for weight_file in weight_files
        }
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        write_json(
            args.output_dir / "training_manifest.json",
            {
                "variant": args.variant,
                "variant_description": VARIANT_DESCRIPTIONS[args.variant],
                "objective_formula": VARIANT_FORMULAS[args.variant],
                "pair_loss": "-logsigmoid(O(chosen) - O(rejected))",
                "requested_structured_variants": list(REQUESTED_VARIANTS),
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "dataset_path": str(args.dataset_path.resolve()),
                "dataset_fingerprint": expected_reference["dataset_fingerprint"],
                "reference_cache": str(args.reference_cache.resolve()),
                "output_dir": str(args.output_dir),
                "final_model": str(final_dir),
                "optimizer": "bitsandbytes.optim.PagedAdamW32bit",
                "optimizer_class": optimizer_class,
                "optimizer_is_paged": optimizer_is_paged,
                "optimizer_state_bits": optimizer_bits,
                "fp16_initial_loss_scale": FP16_INITIAL_SCALE,
                "master_parameter_dtype": "float32",
                "compute_dtype": "float16",
                "saved_weight_dtypes": sorted(saved_weight_dtypes),
                "saved_tensor_count": saved_tensor_count,
                "weight_files_sha256": weight_files_sha256,
                "weight_quantization": None,
                "peft_or_lora": False,
                "full_parameter_training": True,
                "reference_segment_log_probs_precomputed": True,
                "parallelism": "FSDP2_FULL_SHARD",
                "activation_checkpointing": True,
                "activation_offloading": args.activation_offloading,
                "parameters": parameter_info,
                "world_size": trainer.accelerator.num_processes,
                "global_batch_size_pairs": 2 * args.gradient_accumulation_steps,
                "train_rows": len(train_dataset),
                "peak_cuda_memory": peak_cuda_memory,
                "restart_checkpoints_enabled": save_enabled,
                "hyperparameters": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "metrics": result.metrics,
                "packages": package_versions(
                    ["torch", "transformers", "datasets", "accelerate", "trl", "bitsandbytes"]
                ),
            },
        )
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        configure_workspace(args.workspace)
        prepare_segmented_dataset(
            source_jsonl=args.source_jsonl,
            output_dir=args.output_dir,
            model_id=args.model_id,
            model_revision=args.model_revision,
            max_length=args.max_length,
            force=args.force,
        )
    elif args.command == "reference":
        run_reference(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "self-test":
        configure_workspace(args.workspace)
        print(json.dumps(run_loss_self_tests(), indent=2))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
