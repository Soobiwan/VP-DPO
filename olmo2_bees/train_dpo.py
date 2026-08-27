from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
from types import MethodType

from .common import (
    configure_workspace,
    ensure_full_parameter_model,
    package_versions,
    sha256_file,
    write_json,
)


FP16_INITIAL_SCALE = 32.0
LOG_SOFTMAX_CHUNK_SIZE = 32
VOCABULARY_SHARD_SIZE = 8192


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter, two-GPU OLMo 2 DPO training")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="olmo2-bees-dpo")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transformer-layer-class", default="Olmo2DecoderLayer")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_workspace(args.workspace)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_functional
    import torch
    import torch.nn.functional as torch_functional
    from bitsandbytes.utils import sync_gpu
    from torch.distributed.tensor import DTensor
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
    from accelerate.utils import GradScalerKwargs, save_fsdp_model
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.distributed.fsdp import get_fsdp_ckpt_kwargs
    from transformers.trainer import SCHEDULER_NAME
    from trl import DPOConfig, DPOTrainer
    from trl.models.activation_offloading import get_act_offloading_ctx_manager
    from trl.trainer import dpo_trainer as dpo_trainer_module

    def chunked_selective_log_softmax(logits, index):
        """Gather token log-probs without a sequence-sized CUDA reduction workspace."""
        squeeze = index.ndim == logits.ndim - 1
        if squeeze:
            index = index.unsqueeze(-1)

        vocabulary_size = logits.shape[-1]
        index_width = index.shape[-1]
        flat_logits = logits.reshape(-1, vocabulary_size)
        flat_index = index.reshape(-1, index_width)
        chunks = []
        for start in range(0, flat_logits.shape[0], LOG_SOFTMAX_CHUNK_SIZE):
            stop = start + LOG_SOFTMAX_CHUNK_SIZE
            logits_chunk = flat_logits[start:stop]
            index_chunk = flat_index[start:stop]
            if logits.dtype in (torch.float32, torch.float64):
                selected = torch.gather(logits_chunk, dim=-1, index=index_chunk)
                normalizer = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)
                chunks.append(selected - normalizer)
            else:
                logps = torch_functional.log_softmax(logits_chunk, dim=-1)
                chunks.append(torch.gather(logps, dim=-1, index=index_chunk))

        per_token_logps = torch.cat(chunks).reshape(*index.shape)
        if squeeze:
            per_token_logps = per_token_logps.squeeze(-1)
        return per_token_logps

    def chunked_entropy_from_logits(logits, chunk_size=LOG_SOFTMAX_CHUNK_SIZE):
        """Compute TRL's entropy metric without copying a non-contiguous logits slice."""
        if logits.ndim != 3:
            raise ValueError(f"Expected [batch, sequence, vocabulary] logits, got {logits.shape}")
        batch_entropies = []
        for row in logits.unbind(dim=0):
            row_entropies = []
            for logits_chunk in row.split(chunk_size, dim=0):
                logps = torch_functional.log_softmax(logits_chunk, dim=-1)
                probabilities = torch.exp(logps)
                row_entropies.append(-(probabilities.mul_(logps)).sum(dim=-1))
            batch_entropies.append(torch.cat(row_entropies))
        return torch.stack(batch_entropies)

    # TRL's float32 implementation reduces a complete sequence-width vocabulary
    # tensor at once.  On the 8 GiB rank that creates a ~206 MiB temporary during
    # the first gradient-tracked DPO forward.  Chunking only that reduction keeps
    # the operation mathematically identical while bounding its temporary storage.
    dpo_trainer_module.selective_log_softmax = chunked_selective_log_softmax
    # `shift_logits` is a non-contiguous view. TRL's entropy helper reshapes it and
    # thereby copies the complete 412+ MiB tensor before its own chunking begins.
    dpo_trainer_module.entropy_from_logits = chunked_entropy_from_logits

    class VocabularyShardLinear(torch.nn.Linear):
        """An independently FSDP-sharded slice of the OLMo vocabulary head."""

    class CheckpointedChunkedLMHead(torch.nn.Module):
        """OLMo vocabulary projection split into bounded FSDP/all-gather units."""

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

        def forward(self, hidden_states, labels=None):
            if labels is None:
                return torch.cat(
                    [shard(hidden_states) for shard in self.vocabulary_shards],
                    dim=-1,
                )

            selected_logits = torch.zeros_like(hidden_states[..., 0], dtype=torch.float32)
            log_normalizer = None
            for start, shard in zip(
                self.vocabulary_offsets,
                self.vocabulary_shards,
                strict=True,
            ):
                stop = start + shard.out_features

                def project_and_reduce(hidden, target, head=shard, offset=start, limit=stop):
                    # Reducing inside the checkpoint means neither vocabulary
                    # logits nor a full vocabulary-head gradient is retained.
                    # Each independently wrapped head slice all-gathers only
                    # about 32 MiB in FP16 on the smaller rank.
                    logits = head(hidden).float()
                    shard_normalizer = torch.logsumexp(logits, dim=-1)
                    belongs_here = target.ge(offset) & target.lt(limit)
                    local_target = (target - offset).clamp(0, head.out_features - 1)
                    selected = torch.gather(
                        logits,
                        -1,
                        local_target.unsqueeze(-1),
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

        def _init_local_state(self, group, parameter, local_parameter, group_index, param_index):
            config = self.get_config(group_index, param_index, group)
            state = self.state[parameter]
            state["step"] = 0
            state["state1"] = self.get_state_buffer(local_parameter, dtype=torch.float32)
            state["state2"] = self.get_state_buffer(local_parameter, dtype=torch.float32)
            if config["percentile_clipping"] < 100:
                state["gnorm_vec"] = torch.zeros(100, device=local_parameter.device)
            if config["max_unorm"] > 0.0:
                state["unorm_vec"] = torch.zeros(1, device=local_parameter.device)

        def _update_local(self, group, parameter, local_parameter, local_grad, group_index, param_index):
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
                            group,
                            parameter,
                            local_parameter,
                            group_index,
                            param_index,
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
        """Restore standard `lm_head.weight` keys for AutoModel compatibility."""
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
        expected_indices = list(range(len(head_slices)))
        if sorted(head_slices) != expected_indices:
            raise RuntimeError(
                f"Incomplete saved vocabulary head: found slices {sorted(head_slices)}"
            )
        tensors["lm_head.weight"] = torch.cat(
            [head_slices[index] for index in expected_indices],
            dim=0,
        )
        temporary = model_dir / "model.consolidated.safetensors.tmp"
        save_file(tensors, temporary, metadata={"format": "pt"})
        for weight_file in weight_files:
            weight_file.unlink()
        (model_dir / "model.safetensors.index.json").unlink(missing_ok=True)
        temporary.replace(model_dir / "model.safetensors")

    class PagedFSDPDPOTrainer(DPOTrainer):
        """Save paged-AdamW FSDP optimizer shards without PyTorch's incompatible gather."""

        def _precompute_ref_logps(self, dataset, name, batch_size):
            if {"ref_chosen_logps", "ref_rejected_logps"}.issubset(dataset.column_names):
                return dataset
            return super()._precompute_ref_logps(dataset, name, batch_size)

        def __init__(self, *trainer_args, **trainer_kwargs):
            dpo_args = trainer_kwargs.get("args")
            use_activation_offloading = bool(
                dpo_args is not None and dpo_args.activation_offloading
            )
            if use_activation_offloading:
                # Build the tighter context ourselves so the default five-tensor
                # transfer stash cannot consume the last few MiB on rank 1.
                dpo_args.activation_offloading = False
            try:
                super().__init__(*trainer_args, **trainer_kwargs)
            finally:
                if use_activation_offloading:
                    dpo_args.activation_offloading = True
            if use_activation_offloading:
                self.maybe_activation_offload_context = get_act_offloading_ctx_manager(
                    model=self.model,
                    max_fwd_stash_size=1,
                )

        def _compute_loss(self, model, inputs, return_outputs):
            """Compute sigmoid DPO from checkpointed, chunked LM-head log-probs."""
            if self.loss_types != ["sigmoid"] or self.f_divergence_type != "reverse_kl":
                raise RuntimeError(
                    "The memory-efficient OLMo path currently supports sigmoid reverse-KL DPO only"
                )
            input_ids = inputs["input_ids"]
            completion_mask = inputs["completion_mask"][..., 1:]
            per_token_logps = model(
                input_ids=input_ids,
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
                dpo_labels=input_ids[..., 1:],
            )
            per_token_logps = per_token_logps * completion_mask
            chosen_logps, rejected_logps = per_token_logps.sum(dim=1).chunk(2, dim=0)
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            delta = chosen_logratios - rejected_logratios
            per_sequence_loss = (
                -(1 - self.label_smoothing) * torch_functional.logsigmoid(self.beta * delta)
                - self.label_smoothing * torch_functional.logsigmoid(-self.beta * delta)
            )
            loss = per_sequence_loss.mean()

            mode = "train" if self.model.training else "eval"
            if mode == "train":
                num_tokens = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum())
                self._total_train_tokens += num_tokens.sum().item()
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            chosen_rewards = self.beta * chosen_logratios.detach()
            rejected_rewards = self.beta * rejected_logratios.detach()
            gathered_chosen = self.accelerator.gather(chosen_rewards)
            gathered_rejected = self.accelerator.gather(rejected_rewards)
            self._metrics[mode]["rewards/chosen"].append(gathered_chosen.mean().item())
            self._metrics[mode]["rewards/rejected"].append(gathered_rejected.mean().item())
            self._metrics[mode]["rewards/accuracies"].append(
                (gathered_chosen > gathered_rejected).float().mean().item()
            )
            self._metrics[mode]["rewards/margins"].append(
                (gathered_chosen - gathered_rejected).mean().item()
            )
            self._metrics[mode]["logps/chosen"].append(
                self.accelerator.gather(chosen_logps.detach()).mean().item()
            )
            self._metrics[mode]["logps/rejected"].append(
                self.accelerator.gather(rejected_logps.detach()).mean().item()
            )
            small_outputs = {
                "chosen_logps": chosen_logps.detach(),
                "rejected_logps": rejected_logps.detach(),
            }
            # The vocabulary-head gradient is a transient ~392 MiB allocation.
            # Return cached, currently unused blocks to CUDA so the smaller rank
            # can satisfy it as one allocation during the ensuing backward pass.
            if mode == "train":
                torch.cuda.empty_cache()
            return (loss, small_outputs) if return_outputs else loss

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
            # OLMo 2 can overflow Accelerate's generic 65536 initial FP16 scale on
            # Turing.  A conservative dynamic scale avoids silently skipped early
            # updates while retaining loss scaling for small gradients.
            accelerator_args["kwargs_handlers"].append(
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
            # bitsandbytes' paged state contains device-local metadata that the generic
            # FSDP optimizer-state gather cannot compare across heterogeneous CUDA devices.
            # Each rank is already a complete, resumable local shard for a fixed 2-GPU run.
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
                optimizer_path,
                map_location="cpu",
                weights_only=True,
            )
            # A serialized CUDA tensor is no longer CUDA managed ("paged") memory
            # after torch.load.  Loading it directly into PagedAdamW32bit makes the
            # next optimizer kernel operate on an ordinary allocation and can fail
            # inside bitsandbytes.  First restore the logical state on CPU, then
            # recreate each large moment tensor with bnb's managed-memory allocator.
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
            scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
            self.lr_scheduler.load_state_dict(scheduler_state)
            self.accelerator.wait_for_everyone()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("WORLD_SIZE", "1") != "2":
        raise RuntimeError("Launch this script with exactly two Accelerate processes")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)

    set_seed(args.seed)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_from_disk(str(args.dataset_path.resolve()))
    if isinstance(loaded, DatasetDict):
        if args.train_split not in loaded:
            raise KeyError(f"Missing training split {args.train_split!r}")
        train_dataset = loaded[args.train_split]
        eval_dataset = loaded.get(args.eval_split)
    elif isinstance(loaded, Dataset):
        train_dataset = loaded
        eval_dataset = None
    else:
        raise TypeError(f"Unsupported dataset type: {type(loaded)!r}")

    required = {"prompt", "chosen", "rejected"}
    missing = required.difference(train_dataset.column_names)
    if missing:
        raise RuntimeError(f"Preference dataset is missing columns: {sorted(missing)}")

    reference_cache = args.output_dir / "reference_logps"
    reference_cache_metadata = args.output_dir / "reference_logps.json"
    loaded_reference_cache = False
    if reference_cache.is_dir() and reference_cache_metadata.is_file():
        cached_metadata = json.loads(reference_cache_metadata.read_text(encoding="utf-8"))
        expected_metadata = {
            "dataset_path": str(args.dataset_path.resolve()),
            "train_split": args.train_split,
            "rows": len(train_dataset),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "max_length": args.max_length,
        }
        mismatches = {
            key: (cached_metadata.get(key), value)
            for key, value in expected_metadata.items()
            if cached_metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Reference-logp cache mismatch: {mismatches}")
        cached_logps = load_from_disk(str(reference_cache))
        if cached_logps.column_names != ["ref_chosen_logps", "ref_rejected_logps"]:
            raise RuntimeError(
                f"Unexpected reference-logp cache columns: {cached_logps.column_names}"
            )
        train_dataset = concatenate_datasets([train_dataset, cached_logps], axis=1)
        loaded_reference_cache = True
        if local_rank == 0:
            print(f"Using persistent reference log-probs from {reference_cache}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # TRL precomputes reference log-probabilities during trainer construction. Under
    # FSDP the Trainer otherwise leaves the unwrapped model on CPU at that moment,
    # while the batches are already on CUDA. Build the vocabulary slices on CPU,
    # then place one full FP32 model on each GPU temporarily; FSDP shards it as soon
    # as training starts.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.lm_head = CheckpointedChunkedLMHead(model.lm_head)
    model.to(torch.cuda.current_device())
    parameter_info = ensure_full_parameter_model(model)

    standard_model_forward = model.forward

    @functools.wraps(standard_model_forward)
    def memory_efficient_model_forward(self, *model_args, dpo_labels=None, **model_kwargs):
        if dpo_labels is None:
            return standard_model_forward(*model_args, **model_kwargs)

        # Run the OLMo backbone normally, but checkpoint small slices of the very
        # wide vocabulary projection. This prevents both full logits and their
        # equally large backward gradient from residing on the 8 GiB GPU at once.
        model_kwargs.pop("labels", None)
        model_kwargs["return_dict"] = True
        outputs = self.model(*model_args, **model_kwargs)
        hidden_states = outputs.last_hidden_state[..., :-1, :]
        return self.lm_head(hidden_states, dpo_labels)

    model.forward = MethodType(memory_efficient_model_forward, model)

    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    config = DPOConfig(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        beta=args.beta,
        loss_type="sigmoid",
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",
        fp16=True,
        bf16=False,
        tf32=False,
        max_length=args.max_length,
        truncation_mode="keep_start",
        gradient_checkpointing=False,
        activation_offloading=True,
        use_cache=False,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        dataset_num_proc=args.num_proc,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
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
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=True,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=False,
        torch_empty_cache_steps=10,
    )

    trainer = PagedFSDPDPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    if not loaded_reference_cache:
        if trainer.accelerator.is_main_process:
            trainer.train_dataset.select_columns(
                ["ref_chosen_logps", "ref_rejected_logps"]
            ).save_to_disk(reference_cache)
            write_json(
                reference_cache_metadata,
                {
                    "dataset_path": str(args.dataset_path.resolve()),
                    "train_split": args.train_split,
                    "rows": len(train_dataset),
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                    "max_length": args.max_length,
                },
            )
        trainer.accelerator.wait_for_everyone()
    ensure_full_parameter_model(trainer.model)
    trainer.create_optimizer()
    optimizer_class = (
        f"{trainer.optimizer.__class__.__module__}.{trainer.optimizer.__class__.__name__}"
    )
    optimizer_name = "bitsandbytes.optim.PagedAdamW32bit"
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
            f"(is_paged={optimizer_is_paged}, optim_bits={optimizer_bits})"
        )
    if trainer.ref_model is not None:
        raise RuntimeError("Reference model should be released when reference log-probs are precomputed")

    # Reference precomputation requires a complete model on each GPU. Move that
    # same model back to host memory before FSDP creates device-local shards;
    # otherwise FSDP2 must allocate each shard alongside the full 5.9 GiB copy.
    trainer.model.to("cpu")
    torch.cuda.empty_cache()

    if trainer.accelerator.is_main_process:
        print(
            json.dumps(
                {
                    **parameter_info,
                    "optimizer": optimizer_name,
                    "optimizer_class": optimizer_class,
                    "optimizer_is_paged": optimizer_is_paged,
                    "optimizer_state_bits": optimizer_bits,
                    "weight_quantization": None,
                    "adapter": None,
                    "parallelism": "FSDP2_FULL_SHARD",
                    "world_size": trainer.accelerator.num_processes,
                    "global_batch_size": 2 * args.gradient_accumulation_steps,
                    "train_rows": len(train_dataset),
                    "eval_rows": 0 if eval_dataset is None else len(eval_dataset),
                },
                indent=2,
            )
        )

    checkpoint = last_complete_checkpoint(args.output_dir) if args.resume else None
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
        from safetensors import safe_open

        consolidate_saved_vocabulary_head(final_dir)
        tokenizer.save_pretrained(final_dir)
        weight_files = sorted(final_dir.glob("*.safetensors"))
        if not weight_files:
            raise RuntimeError(f"No safetensors model weights were saved under {final_dir}")
        saved_weight_dtypes: set[str] = set()
        saved_tensor_count = 0
        for weight_file in weight_files:
            with safe_open(weight_file, framework="pt", device="cpu") as handle:
                for tensor_name in handle.keys():
                    saved_weight_dtypes.add(str(handle.get_slice(tensor_name).get_dtype()))
                    saved_tensor_count += 1
        if saved_weight_dtypes != {"F32"}:
            raise RuntimeError(
                f"Final checkpoint is not entirely FP32: {sorted(saved_weight_dtypes)}"
            )
        weight_files_sha256 = {
            weight_file.name: sha256_file(weight_file) for weight_file in weight_files
        }
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        write_json(
            args.output_dir / "training_manifest.json",
            {
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "dataset_path": str(args.dataset_path.resolve()),
                "output_dir": str(args.output_dir),
                "final_model": str(final_dir),
                "optimizer": optimizer_name,
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
                "reference_log_probs_precomputed": True,
                "parallelism": "FSDP2_FULL_SHARD",
                "activation_checkpointing": True,
                "activation_offloading": True,
                "parameters": parameter_info,
                "world_size": trainer.accelerator.num_processes,
                "peak_cuda_memory": peak_cuda_memory,
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
    if os.environ.get("OLMO2_BEES_DEBUG_THREADS") == "1":
        import threading

        print(
            f"rank={local_rank} threads_at_exit="
            f"{[(thread.name, thread.daemon) for thread in threading.enumerate()]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
