from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any, Tuple

from .common import configure_workspace, package_versions, sha256_file, write_json
from .preference_suite_losses import (
    METHOD_DESCRIPTIONS,
    METHOD_FORMULAS,
    METHODS,
    packed_triplet_loss,
    paper_tidpo_importance_weights,
    paper_tidpo_pair_loss,
    run_loss_self_tests,
    sampo_pair_loss,
    simpo_pair_loss,
    tidpo_importance_weights,
    tidpo_pair_loss,
    topk_bucket_position_kl,
)


FP16_INITIAL_SCALE = 32.0
VOCABULARY_SHARD_SIZE = 8192
REFERENCE_PROJECTION_CHUNK_SIZE = 64
TIDPO_REPOSITORY = "https://github.com/gracefulning/TIDPO"
TIDPO_COMMIT = "e04a0926869a8f9fe9c9e9ce395394fd2c697fe2"
TIDPO_UPSTREAM_SOURCE_HASHES = {
    "trainers.py": "5fb907eecc2d00a6b97d7ac45db4bd86ce4d58197b7a58363233e40040ae113f",
    "config/config.yaml": (
        "515b74cf7e7461049bcbf78519564acae0ec36516194e723b8f372e6ef3c4f88"
    ),
    "config/loss/tidpo.yaml": (
        "e77343adb00fa27a0d54d9c806a12510291b2c779be1639ea5e86da74149a1e8"
    ),
}


def _normalized_source_sha256(path: Path) -> str:
    """Hash source reproducibly across Git's LF/CRLF checkout conversion."""
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _upstream_objective_equivalence_self_test() -> dict[str, Any]:
    """Execute the pinned repo's pure objective functions and compare this OLMo adapter."""
    import torch
    import torch.nn.functional as functional

    upstream_root = Path(__file__).resolve().parents[2] / "third_party" / "TIDPO"
    actual_hashes = {
        relative: _normalized_source_sha256(upstream_root / relative)
        for relative in TIDPO_UPSTREAM_SOURCE_HASHES
    }
    if actual_hashes != TIDPO_UPSTREAM_SOURCE_HASHES:
        raise AssertionError(f"Pinned TI-DPO source hash mismatch: {actual_hashes}")

    parsed = ast.parse((upstream_root / "trainers.py").read_text(encoding="utf-8"))
    wanted = {"tdpo_loss", "_weighted_tdpo_get_batch_logps"}
    definitions = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    if {node.name for node in definitions} != wanted:
        raise AssertionError("Could not locate the pinned TI-DPO objective functions")
    namespace: dict[str, Any] = {
        "torch": torch,
        "F": functional,
        "Tuple": Tuple,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "upstream/trainers.py", "exec"), namespace)

    generator = torch.Generator().manual_seed(20250823)
    policy_logits = torch.randn(4, 6, 13, generator=generator, requires_grad=True)
    reference_logits = torch.randn(4, 6, 13, generator=generator)
    labels = torch.tensor(
        [
            [-100, -100, 2, 3, 4, -100],
            [-100, -100, 5, 6, 7, 8],
            [-100, -100, 1, 9, -100, -100],
            [-100, -100, 10, 11, 12, -100],
        ],
        dtype=torch.long,
    )
    full_weights = torch.rand(4, 6, generator=generator) + 0.25
    upstream_margin, upstream_kl, _ = namespace["_weighted_tdpo_get_batch_logps"](
        policy_logits,
        reference_logits,
        labels,
        full_weights,
        average_log_prob=False,
    )
    upstream_losses, upstream_chosen, upstream_rejected = namespace["tdpo_loss"](
        upstream_margin[:2],
        upstream_margin[2:],
        upstream_kl[:2],
        upstream_kl[2:],
        beta=0.2,
        alpha=0.5,
        if_tdpo2=True,
    )

    shifted_labels = labels[:, 1:].clone()
    completion_mask = shifted_labels.ne(-100)
    shifted_labels.masked_fill_(~completion_mask, 0)
    policy_vocab_logps = policy_logits[:, :-1].log_softmax(dim=-1)
    reference_probabilities = reference_logits[:, :-1].softmax(dim=-1)
    reference_vocab_logps = reference_probabilities.log()
    policy_token_logps = torch.gather(
        policy_vocab_logps, -1, shifted_labels.unsqueeze(-1)
    ).squeeze(-1)
    reference_token_logps = torch.gather(
        reference_vocab_logps, -1, shifted_labels.unsqueeze(-1)
    ).squeeze(-1)
    adapter = tidpo_pair_loss(
        policy_token_logps,
        reference_token_logps,
        completion_mask,
        full_weights[:, 1:],
        beta=0.2,
        position_kl=upstream_kl,
        alpha=0.5,
        if_tdpo2=True,
    )
    adapter_loss, adapter_chosen, adapter_rejected = adapter[0], adapter[1], adapter[2]
    if not torch.allclose(adapter_loss, upstream_losses.mean(), atol=1e-6, rtol=1e-6):
        raise AssertionError("OLMo TI-DPO loss differs from pinned trainers.py")
    if not torch.allclose(adapter_chosen, upstream_chosen, atol=1e-6, rtol=1e-6):
        raise AssertionError("OLMo TI-DPO chosen reward differs from pinned trainers.py")
    if not torch.allclose(adapter_rejected, upstream_rejected, atol=1e-6, rtol=1e-6):
        raise AssertionError("OLMo TI-DPO rejected reward differs from pinned trainers.py")
    upstream_gradient = torch.autograd.grad(
        upstream_losses.mean(), policy_logits, retain_graph=True
    )[0]
    adapter_gradient = torch.autograd.grad(adapter_loss, policy_logits)[0]
    if not torch.allclose(adapter_gradient, upstream_gradient, atol=2e-6, rtol=2e-6):
        max_error = (adapter_gradient - upstream_gradient).abs().max()
        raise AssertionError(f"OLMo TI-DPO gradient differs from pinned trainers.py: {max_error}")
    return {
        "commit": TIDPO_COMMIT,
        "normalized_source_sha256": actual_hashes,
        "loss_equal": True,
        "rewards_equal": True,
        "policy_gradient_equal": True,
    }


def _chunked_exact_kl_self_test() -> dict[str, Any]:
    """Compare the actual T4 chunked head against a dense full-vocabulary calculation."""
    import torch
    import torch.nn.functional as functional

    parsed = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    run_train_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_train"
    )
    wanted = {"VocabularyShardLinear", "CheckpointedChunkedLMHead"}
    definitions = [
        node
        for node in run_train_node.body
        if isinstance(node, ast.ClassDef) and node.name in wanted
    ]
    if {node.name for node in definitions} != wanted:
        raise AssertionError("Could not locate the chunked TI-DPO vocabulary head")

    def direct_checkpoint(function, *checkpoint_args, **checkpoint_kwargs):
        checkpoint_kwargs.pop("use_reentrant", None)
        if checkpoint_kwargs:
            raise AssertionError(f"Unexpected checkpoint kwargs: {checkpoint_kwargs}")
        return function(*checkpoint_args)

    namespace: dict[str, Any] = {
        "torch": torch,
        "activation_checkpoint": direct_checkpoint,
        "VOCABULARY_SHARD_SIZE": 4,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), __file__, "exec"), namespace)
    head_class = namespace["CheckpointedChunkedLMHead"]

    generator = torch.Generator().manual_seed(19653)
    policy_source = torch.nn.Linear(5, 13, bias=False)
    reference_source = torch.nn.Linear(5, 13, bias=False)
    with torch.no_grad():
        policy_source.weight.copy_(torch.randn(13, 5, generator=generator))
        reference_source.weight.copy_(torch.randn(13, 5, generator=generator))
    policy_head = head_class(policy_source)
    reference_head = head_class(reference_source)
    reference_head.requires_grad_(False)
    policy_hidden = torch.randn(2, 4, 5, generator=generator, requires_grad=True)
    reference_hidden = torch.randn(2, 4, 5, generator=generator)
    labels = torch.tensor([[0, 3, 7, 12], [1, 5, 8, 11]], dtype=torch.long)

    chunked_policy, chunked_reference, chunked_kl = policy_head(
        policy_hidden,
        labels=labels,
        exact_reference_hidden_states=reference_hidden,
        exact_reference_head=reference_head,
    )
    policy_weight = torch.cat(
        [shard.weight for shard in policy_head.vocabulary_shards], dim=0
    )
    reference_weight = torch.cat(
        [shard.weight for shard in reference_head.vocabulary_shards], dim=0
    )
    dense_policy_hidden = policy_hidden.detach().clone().requires_grad_(True)
    dense_policy_logps = functional.linear(
        dense_policy_hidden, policy_weight
    ).log_softmax(dim=-1)
    dense_reference_logps = functional.linear(
        reference_hidden, reference_weight
    ).log_softmax(dim=-1)
    dense_policy = torch.gather(
        dense_policy_logps, -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    dense_reference = torch.gather(
        dense_reference_logps, -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    dense_kl = (
        dense_reference_logps.exp() * (dense_reference_logps - dense_policy_logps)
    ).sum(dim=-1)
    comparisons = {
        "selected_policy_logps": (chunked_policy, dense_policy),
        "selected_reference_logps": (chunked_reference, dense_reference),
        "full_vocabulary_kl": (chunked_kl, dense_kl),
    }
    for name, (chunked, dense) in comparisons.items():
        if not torch.allclose(chunked, dense, atol=2e-6, rtol=2e-6):
            error = (chunked - dense).abs().max()
            raise AssertionError(f"Chunked {name} differs from dense calculation: {error}")
    chunked_gradient = torch.autograd.grad(
        (chunked_policy + chunked_kl).sum(), policy_hidden
    )[0]
    dense_gradient = torch.autograd.grad(
        (dense_policy + dense_kl).sum(), dense_policy_hidden
    )[0]
    if not torch.allclose(chunked_gradient, dense_gradient, atol=3e-6, rtol=3e-6):
        error = (chunked_gradient - dense_gradient).abs().max()
        raise AssertionError(f"Chunked exact-KL policy gradient differs from dense: {error}")
    return {
        "vocabulary_size": 13,
        "chunk_size": 4,
        "selected_logps_equal": True,
        "full_vocabulary_kl_equal": True,
        "policy_hidden_gradient_equal": True,
    }


def _optimizer_update_32bit(
    functional,
    optimizer_name,
    local_grad,
    local_parameter,
    state,
    config,
    step,
    gnorm_scale,
) -> None:
    """Call the bitsandbytes kernel without shifting its positional tensor arguments."""
    functional.optimizer_update_32bit(
        optimizer_name,
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


def _optimizer_contract_self_test() -> dict[str, Any]:
    class Recorder:
        def __init__(self):
            self.args = None
            self.kwargs = None

        def optimizer_update_32bit(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    recorder = Recorder()
    state1, state2, unorm = object(), object(), object()
    config = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "lr": 5e-7,
        "alpha": 0.0,
        "weight_decay": 0.0,
        "max_unorm": 1.0,
        "skip_zeros": False,
    }
    _optimizer_update_32bit(
        recorder,
        "adam",
        "gradient",
        "parameter",
        {"state1": state1, "state2": state2, "unorm_vec": unorm},
        config,
        3,
        1.0,
    )
    assert recorder.args is not None and recorder.kwargs is not None
    assert recorder.args[7] == config["lr"]
    assert recorder.args[8] is state2
    assert recorder.args[14] is unorm
    assert recorder.kwargs == {"max_unorm": 1.0, "skip_zeros": False}
    return {
        "learning_rate_argument_index": 7,
        "state2_argument_index": 8,
        "unorm_argument_index": 14,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-GPU full-parameter OLMo training for TIDPO, SimPO, and SamPO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser(
        "reference", help="Precompute token-level OLMo reference statistics and optional anchors"
    )
    reference.add_argument("--workspace", type=Path, required=True)
    reference.add_argument("--dataset-path", type=Path, required=True)
    reference.add_argument("--split", default="train")
    reference.add_argument("--model-id", required=True)
    reference.add_argument("--model-revision", default=None)
    reference.add_argument("--output-dir", type=Path, required=True)
    reference.add_argument("--max-length", type=int, default=1024)
    reference.add_argument(
        "--tidpo-kl-top-k",
        type=int,
        default=32,
        help="Reference top-k support retained per response position for the TDPO2 KL term",
    )
    reference.add_argument("--flush-every", type=int, default=25)
    reference.add_argument(
        "--with-tidpo-anchors", action=argparse.BooleanOptionalAction, default=True
    )
    reference.add_argument("--anchor-max-new-tokens", type=int, default=64)
    reference.add_argument("--anchor-top-k", type=int, default=50)
    reference.add_argument("--anchor-top-p", type=float, default=0.95)
    reference.add_argument("--anchor-temperature", type=float, default=0.8)
    reference.add_argument("--anchor-seed", type=int, default=314159)

    train = subparsers.add_parser("train", help="Train one preference method")
    train.add_argument("--workspace", type=Path, required=True)
    train.add_argument("--dataset-path", type=Path, required=True)
    train.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help="Required for TIDPO and SamPO; SimPO is reference-free",
    )
    train.add_argument("--train-split", default="train")
    train.add_argument("--model-id", required=True)
    train.add_argument("--model-revision", default=None)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--max-length", type=int, default=1024)
    train.add_argument("--run-name", default=None)
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--learning-rate", type=float, default=5e-7)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--logging-steps", type=int, default=1)
    train.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="0 saves only the final model; a positive value enables restart checkpoints",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--transformer-layer-class", default="Olmo2DecoderLayer")
    train.add_argument(
        "--activation-offloading", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument("--resume", action="store_true")
    train.add_argument("--tidpo-beta", type=float, default=0.2)
    train.add_argument("--tidpo-alpha", type=float, default=0.5)
    train.add_argument("--tidpo-kl-top-k", type=int, default=32)
    train.add_argument(
        "--tidpo2", action=argparse.BooleanOptionalAction, default=True
    )
    train.add_argument("--tidpo-lambda-importance", type=float, default=0.2)
    train.add_argument("--tidpo-prior-sigma-div", type=float, default=8.0)
    train.add_argument("--tidpo-triplet-gamma", type=float, default=0.001)
    train.add_argument("--tidpo-triplet-margin", type=float, default=0.001)
    train.add_argument(
        "--tidpo-paper-exact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use arXiv:2505.19653v3 Eqs. 5-14: last-logit gradient attribution mixed "
            "with a Gaussian prior, weighted DPO without TDPO position-KL, and a live "
            "current-policy triplet anchor"
        ),
    )
    train.add_argument(
        "--tidpo-repo-exact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the pinned gracefulning/TIDPO implementation at commit e04a092: "
            "full-sequence gradient/Gaussian weights, exact full-vocabulary TDPO2 "
            "position-KL, and a live current-policy triplet anchor"
        ),
    )
    train.add_argument("--tidpo-anchor-max-new-tokens", type=int, default=64)
    train.add_argument("--tidpo-anchor-top-k", type=int, default=50)
    train.add_argument("--tidpo-anchor-top-p", type=float, default=0.95)
    train.add_argument("--tidpo-anchor-temperature", type=float, default=0.8)
    train.add_argument("--simpo-beta", type=float, default=2.0)
    train.add_argument("--simpo-gamma-beta-ratio", type=float, default=0.5)
    train.add_argument("--sampo-beta", type=float, default=0.1)

    test = subparsers.add_parser("self-test", help="Run CPU objective tests")
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
        "cache_schema_version": 2,
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_fingerprint": dataset._fingerprint,
        "split": args.split,
        "rows": len(dataset),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "max_length": args.max_length,
        "tidpo_kl_top_k": args.tidpo_kl_top_k,
        "with_tidpo_anchors": args.with_tidpo_anchors,
        "anchor_max_new_tokens": args.anchor_max_new_tokens if args.with_tidpo_anchors else None,
        "anchor_top_k": args.anchor_top_k if args.with_tidpo_anchors else None,
        "anchor_top_p": args.anchor_top_p if args.with_tidpo_anchors else None,
        "anchor_temperature": args.anchor_temperature if args.with_tidpo_anchors else None,
        "anchor_seed": args.anchor_seed if args.with_tidpo_anchors else None,
    }


def _read_jsonl_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                index = int(record["dataset_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid cache record at {path}:{line_number}") from error
            if index in records:
                raise RuntimeError(f"Duplicate dataset_index={index} in {path}")
            records[index] = record
    return records


def _pad_reference_pair(row: dict[str, Any], pad_token_id: int, device: Any):
    import torch

    ids = [row["chosen_input_ids"], row["rejected_input_ids"]]
    ownership = [row["chosen_segment_ids"], row["rejected_segment_ids"]]
    width = max(map(len, ids))
    input_ids = torch.full((2, width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((2, width), dtype=torch.long, device=device)
    completion_mask = torch.zeros((2, width - 1), dtype=torch.bool, device=device)
    for index, (sequence, segment_ids) in enumerate(zip(ids, ownership, strict=True)):
        length = len(sequence)
        input_ids[index, :length] = torch.as_tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        completion_mask[index, : length - 1] = torch.as_tensor(
            segment_ids[1:], dtype=torch.long, device=device
        ).ge(0)
    if torch.any(completion_mask.sum(-1) == 0):
        raise RuntimeError("A reference pair has an empty completion mask")
    return input_ids, attention_mask, completion_mask


def _selected_token_statistics(
    model,
    input_ids,
    attention_mask,
    completion_mask,
    *,
    top_k: int = 0,
):
    import torch
    import torch.nn.functional as functional

    with torch.inference_mode():
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state[:, :-1, :]
        labels = input_ids[:, 1:]
        positions = completion_mask.nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise RuntimeError("No loss-bearing tokens in reference batch")
        selected_hidden = hidden[positions[:, 0], positions[:, 1]]
        selected_labels = labels[positions[:, 0], positions[:, 1]]
        chunks = []
        support_id_chunks = []
        support_logp_chunks = []
        for start in range(0, selected_hidden.shape[0], REFERENCE_PROJECTION_CHUNK_SIZE):
            stop = start + REFERENCE_PROJECTION_CHUNK_SIZE
            logits = functional.linear(
                selected_hidden[start:stop], model.lm_head.weight, model.lm_head.bias
            ).float()
            targets = selected_labels[start:stop]
            normalizer = torch.logsumexp(logits, dim=-1)
            selected = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
            chunks.append(selected - normalizer)
            if top_k:
                if top_k > logits.shape[-1]:
                    raise ValueError(f"top_k={top_k} exceeds vocabulary size {logits.shape[-1]}")
                support_logits, support_ids = torch.topk(logits, k=top_k, dim=-1)
                support_id_chunks.append(support_ids.to(torch.int32))
                support_logp_chunks.append(support_logits - normalizer.unsqueeze(-1))
        token_logps = torch.cat(chunks)
        aligned = torch.zeros_like(completion_mask, dtype=torch.float32)
        aligned[positions[:, 0], positions[:, 1]] = token_logps
        if not top_k:
            return aligned, None, None
        support_ids = torch.zeros(
            (*completion_mask.shape, top_k), dtype=torch.int32, device=input_ids.device
        )
        support_logps = torch.zeros(
            (*completion_mask.shape, top_k), dtype=torch.float32, device=input_ids.device
        )
        support_ids[positions[:, 0], positions[:, 1]] = torch.cat(support_id_chunks)
        support_logps[positions[:, 0], positions[:, 1]] = torch.cat(support_logp_chunks)
        return aligned, support_ids, support_logps


def _prompt_prefix(row: dict[str, Any]) -> list[int]:
    segment_ids = row["chosen_segment_ids"]
    try:
        completion_start = next(index for index, value in enumerate(segment_ids) if value >= 0)
    except StopIteration as error:
        raise RuntimeError("Chosen response contains no segment-owned token") from error
    prefix = list(row["chosen_input_ids"][:completion_start])
    if not prefix:
        raise RuntimeError("Cannot generate a TIDPO anchor from an empty prompt")
    return prefix


def _generate_reference_anchor(model, tokenizer, row, args, dataset_index, device):
    import torch

    prompt = _prompt_prefix(row)
    remaining = args.max_length - len(prompt)
    if remaining <= 0:
        raise RuntimeError(f"No room for anchor at dataset index {dataset_index}")
    max_new_tokens = min(args.anchor_max_new_tokens, remaining)
    prompt_tensor = torch.as_tensor([prompt], dtype=torch.long, device=device)
    attention = torch.ones_like(prompt_tensor)
    # Per-row seeding makes the cache reproducible even when a partially completed rank resumes.
    torch.manual_seed(args.anchor_seed + dataset_index)
    torch.cuda.manual_seed_all(args.anchor_seed + dataset_index)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prompt_tensor,
            attention_mask=attention,
            do_sample=True,
            top_k=args.anchor_top_k,
            top_p=args.anchor_top_p,
            temperature=args.anchor_temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if generated.shape[1] <= len(prompt):
        raise RuntimeError(f"Anchor generation produced no tokens at dataset index {dataset_index}")
    # Generation is unpadded (batch size one); pad==EOS for OLMo, so token-value masking would
    # incorrectly hide a real terminal EOS token.
    anchor_attention = torch.ones_like(generated, dtype=torch.long)
    anchor_completion_mask = torch.zeros(
        (1, generated.shape[1] - 1), dtype=torch.bool, device=device
    )
    # Shifted position j predicts input token j+1; generated tokens start at input index len(prompt).
    anchor_completion_mask[:, max(0, len(prompt) - 1) :] = True
    anchor_completion_mask &= anchor_attention[:, 1:].to(torch.bool)
    anchor_logps, _, _ = _selected_token_statistics(
        model, generated, anchor_attention, anchor_completion_mask
    )
    return (
        generated[0].cpu().tolist(),
        anchor_completion_mask[0].cpu().tolist(),
        anchor_logps[0].cpu().tolist(),
    )


def run_reference(args: argparse.Namespace) -> None:
    configure_workspace(args.workspace)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import torch
    from accelerate import Accelerator
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerator = Accelerator(mixed_precision="fp16")
    if accelerator.num_processes != 2:
        raise RuntimeError("Reference preparation requires exactly two Accelerate processes")
    dataset = _load_split(args.dataset_path, args.split)
    if args.tidpo_kl_top_k < 0:
        raise ValueError("--tidpo-kl-top-k cannot be negative")
    required = {
        "dataset_index",
        "row_id",
        "chosen_input_ids",
        "chosen_segment_ids",
        "rejected_input_ids",
        "rejected_segment_ids",
    }
    missing = required.difference(dataset.column_names)
    if missing:
        raise RuntimeError(f"Prepared segmented dataset is missing: {sorted(missing)}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "dataset"
    manifest_path = args.output_dir / "reference_manifest.json"
    identity_path = args.output_dir / "reference_identity.json"
    identity = _reference_identity(args, dataset)
    if cache_path.is_dir() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in identity.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Reference-cache identity mismatch: {mismatches}")
        if accelerator.is_main_process:
            print(f"Token reference cache already exists: {cache_path}")
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return
    if cache_path.exists() or manifest_path.exists():
        raise RuntimeError(
            f"Incomplete merged reference cache under {args.output_dir}; retain rank part files "
            "but move the incomplete dataset/manifest before resuming"
        )
    if identity_path.is_file():
        recorded_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded_identity != identity:
            raise RuntimeError(
                f"Partial reference-cache identity mismatch: recorded={recorded_identity}, "
                f"requested={identity}"
            )
    else:
        partial_parts = sorted(args.output_dir.glob("part-rank-*.jsonl"))
        if partial_parts:
            raise RuntimeError(
                "Reference rank parts exist without reference_identity.json; move those stale "
                f"parts before restarting: {[path.name for path in partial_parts]}"
            )
        if accelerator.is_main_process:
            write_json(identity_path, identity)
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, use_fast=True
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
    model.config.use_cache = True
    model.requires_grad_(False)
    model.eval()

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    part_path = args.output_dir / f"part-rank-{rank:02d}.jsonl"
    completed = _read_jsonl_records(part_path)
    assigned = list(range(rank, len(dataset), world_size))
    unexpected = set(completed).difference(assigned)
    if unexpected:
        raise RuntimeError(f"Reference part includes another rank's indices: {sorted(unexpected)[:10]}")
    pending = [index for index in assigned if index not in completed]

    from tqdm.auto import tqdm

    with part_path.open("a", encoding="utf-8") as handle:
        for completed_since_flush, dataset_index in enumerate(
            tqdm(pending, desc=f"Token reference rank {rank}", position=rank), 1
        ):
            row = dataset[dataset_index]
            input_ids, attention_mask, completion_mask = _pad_reference_pair(
                row, tokenizer.pad_token_id, accelerator.device
            )
            with accelerator.autocast():
                pair_logps, support_ids, support_logps = _selected_token_statistics(
                    model,
                    input_ids,
                    attention_mask,
                    completion_mask,
                    top_k=args.tidpo_kl_top_k,
                )
            chosen_width = len(row["chosen_input_ids"]) - 1
            rejected_width = len(row["rejected_input_ids"]) - 1
            record = {
                "dataset_index": dataset_index,
                "row_id": int(row["row_id"]),
                "ref_chosen_token_logps": pair_logps[0, :chosen_width].cpu().tolist(),
                "ref_rejected_token_logps": pair_logps[1, :rejected_width].cpu().tolist(),
            }
            if args.tidpo_kl_top_k:
                # Store only response positions. The segmented completion mask restores alignment
                # in the collator and avoids serializing K zeros at every prompt position.
                record.update(
                    {
                        "ref_chosen_support_ids": support_ids[0][completion_mask[0]].cpu().tolist(),
                        "ref_rejected_support_ids": support_ids[1][completion_mask[1]].cpu().tolist(),
                        "ref_chosen_support_logps": support_logps[0][completion_mask[0]].cpu().tolist(),
                        "ref_rejected_support_logps": support_logps[1][completion_mask[1]].cpu().tolist(),
                    }
                )
            if args.with_tidpo_anchors:
                anchor_ids, anchor_mask, anchor_logps = _generate_reference_anchor(
                    model, tokenizer, row, args, dataset_index, accelerator.device
                )
                record.update(
                    {
                        "anchor_input_ids": anchor_ids,
                        "anchor_completion_mask": anchor_mask,
                        "ref_anchor_token_logps": anchor_logps,
                    }
                )
            arrays = [record["ref_chosen_token_logps"], record["ref_rejected_token_logps"]]
            nested_arrays = (
                [
                    record["ref_chosen_support_logps"],
                    record["ref_rejected_support_logps"],
                ]
                if args.tidpo_kl_top_k
                else []
            )
            if args.with_tidpo_anchors:
                arrays.append(record["ref_anchor_token_logps"])
            if not all(math.isfinite(value) for array in arrays for value in array):
                raise FloatingPointError(f"Non-finite reference value at index {dataset_index}")
            if not all(
                math.isfinite(value)
                for array in nested_arrays
                for position in array
                for value in position
            ):
                raise FloatingPointError(
                    f"Non-finite top-k reference support at index {dataset_index}"
                )
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            if completed_since_flush % args.flush_every == 0:
                handle.flush()
                os.fsync(handle.fileno())
        handle.flush()
        os.fsync(handle.fileno())

    accelerator.wait_for_everyone()
    # Release both reference-model replicas before rank 0 converts the potentially multi-GB
    # JSONL cache.  The conversion below interleaves rank files one row at a time instead of
    # materializing millions of nested Python numbers in host RAM (important on Kaggle's 29 GB
    # GPU workers).
    del model
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        part_paths = [
            args.output_dir / f"part-rank-{process_rank:02d}.jsonl"
            for process_rank in range(world_size)
        ]
        handles = [path.open("r", encoding="utf-8") for path in part_paths]
        try:
            with tempfile.TemporaryDirectory(prefix=".token-reference-", dir=args.output_dir) as temp:
                temporary_root = Path(temp)
                merged_jsonl = temporary_root / "ordered.jsonl"
                with merged_jsonl.open("w", encoding="utf-8") as merged:
                    for index in range(len(dataset)):
                        handle = handles[index % world_size]
                        line = handle.readline()
                        while line and not line.strip():
                            line = handle.readline()
                        if not line:
                            raise RuntimeError(f"Reference cache ends before dataset index {index}")
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise RuntimeError(
                                f"Invalid reference JSON for dataset index {index}"
                            ) from error
                        if int(record.get("dataset_index", -1)) != index:
                            raise RuntimeError(
                                f"Reference order mismatch: expected {index}, "
                                f"got {record.get('dataset_index')}"
                            )
                        row = dataset[index]
                        if record["row_id"] != int(row["row_id"]):
                            raise RuntimeError(f"Reference row mismatch at index {index}")
                        if len(record["ref_chosen_token_logps"]) != len(row["chosen_input_ids"]) - 1:
                            raise RuntimeError(f"Chosen reference width mismatch at index {index}")
                        if len(record["ref_rejected_token_logps"]) != len(row["rejected_input_ids"]) - 1:
                            raise RuntimeError(f"Rejected reference width mismatch at index {index}")
                        for side in ("chosen", "rejected") if args.tidpo_kl_top_k else ():
                            expected_width = sum(
                                value >= 0 for value in row[f"{side}_segment_ids"][1:]
                            )
                            support_ids = record[f"ref_{side}_support_ids"]
                            support_logps = record[f"ref_{side}_support_logps"]
                            if (
                                len(support_ids) != expected_width
                                or len(support_logps) != expected_width
                            ):
                                raise RuntimeError(
                                    f"{side.title()} response top-k support width mismatch "
                                    f"at index {index}"
                                )
                            if any(len(values) != args.tidpo_kl_top_k for values in support_ids):
                                raise RuntimeError(
                                    f"{side.title()} top-k ID count mismatch at index {index}"
                                )
                            if any(len(values) != args.tidpo_kl_top_k for values in support_logps):
                                raise RuntimeError(
                                    f"{side.title()} top-k logp count mismatch at index {index}"
                                )
                        merged.write(line.rstrip("\r\n") + "\n")
                    for process_rank, handle in enumerate(handles):
                        if any(line.strip() for line in handle):
                            raise RuntimeError(
                                f"Reference rank {process_rank} contains unexpected extra rows"
                            )
                cache = Dataset.from_json(str(merged_jsonl), keep_in_memory=False)
                if cache["dataset_index"] != list(range(len(dataset))):
                    raise RuntimeError("Merged reference cache order is invalid")
                staged = temporary_root / "dataset"
                cache.save_to_disk(staged)
                cache_columns = cache.column_names
                staged.replace(cache_path)
        finally:
            for handle in handles:
                handle.close()
        write_json(
            manifest_path,
            {
                **identity,
                "world_size": world_size,
                "compute_dtype": "float16",
                "columns": cache_columns,
                "packages": package_versions(
                    ["torch", "transformers", "datasets", "accelerate", "safetensors"]
                ),
            },
        )
        print(f"Saved shared token reference cache: {cache_path}")
    accelerator.wait_for_everyone()
    accelerator.end_training()


class PreferenceTokenCollator:
    def __init__(
        self,
        pad_token_id: int,
        *,
        include_reference: bool,
        include_support: bool,
        include_anchors: bool,
        include_live_prompts: bool = False,
    ):
        self.pad_token_id = int(pad_token_id)
        self.include_reference = include_reference
        self.include_support = include_support
        self.include_anchors = include_anchors
        self.include_live_prompts = include_live_prompts

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        rows = [
            (feature, side)
            for side in ("chosen", "rejected")
            for feature in features
        ]
        width = max(len(feature[f"{side}_input_ids"]) for feature, side in rows)
        input_ids = torch.full(
            (len(rows), width), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        completion_mask = torch.zeros((len(rows), width - 1), dtype=torch.bool)
        reference_logps = torch.zeros((len(rows), width - 1), dtype=torch.float32)
        support_width = (
            len(features[0]["ref_chosen_support_ids"][0]) if self.include_support else 0
        )
        reference_support_ids = torch.zeros(
            (len(rows), width - 1, support_width), dtype=torch.long
        )
        reference_support_logps = torch.zeros(
            (len(rows), width - 1, support_width), dtype=torch.float32
        )
        for index, (feature, side) in enumerate(rows):
            ids = feature[f"{side}_input_ids"]
            segment_ids = feature[f"{side}_segment_ids"]
            length = len(ids)
            if len(segment_ids) != length:
                raise RuntimeError(f"Malformed {side} token cache for row {feature['row_id']}")
            completion_mask[index, : length - 1] = torch.as_tensor(
                segment_ids[1:], dtype=torch.long
            ).ge(0)
            completion_positions = torch.nonzero(
                completion_mask[index, : length - 1], as_tuple=False
            ).squeeze(-1)
            input_ids[index, :length] = torch.as_tensor(ids, dtype=torch.long)
            attention_mask[index, :length] = 1
            if self.include_reference:
                ref = feature[f"ref_{side}_token_logps"]
                if len(ref) != length - 1:
                    raise RuntimeError(
                        f"Malformed {side} reference cache for row {feature['row_id']}"
                    )
                reference_logps[index, : length - 1] = torch.as_tensor(
                    ref, dtype=torch.float32
                )
            if self.include_support:
                support_ids = feature[f"ref_{side}_support_ids"]
                support_logps = feature[f"ref_{side}_support_logps"]
                support_shape_ok = (
                    len(support_ids) == int(completion_positions.numel())
                    and len(support_logps) == int(completion_positions.numel())
                    and all(len(values) == support_width for values in support_ids)
                    and all(len(values) == support_width for values in support_logps)
                )
                if not support_shape_ok:
                    raise RuntimeError(
                        f"Malformed {side} TIDPO support for row {feature['row_id']}"
                    )
                reference_support_ids[index, completion_positions] = torch.as_tensor(
                    support_ids, dtype=torch.long
                )
                reference_support_logps[index, completion_positions] = torch.as_tensor(
                    support_logps, dtype=torch.float32
                )
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "reference_token_logps": reference_logps,
            "reference_support_ids": reference_support_ids,
            "reference_support_logps": reference_support_logps,
            "row_ids": torch.as_tensor([feature["row_id"] for feature in features]),
        }
        if self.include_live_prompts:
            prompts = []
            for feature in features:
                segment_ids = feature["chosen_segment_ids"]
                try:
                    completion_start = next(
                        index for index, segment_id in enumerate(segment_ids) if segment_id >= 0
                    )
                except StopIteration as error:
                    raise RuntimeError(
                        f"Chosen response has no completion for row {feature['row_id']}"
                    ) from error
                prompt = feature["chosen_input_ids"][:completion_start]
                if not prompt:
                    raise RuntimeError(f"Empty prompt for row {feature['row_id']}")
                prompts.append(prompt)
            prompt_width = max(map(len, prompts))
            prompt_ids = torch.full(
                (len(prompts), prompt_width), self.pad_token_id, dtype=torch.long
            )
            prompt_attention = torch.zeros((len(prompts), prompt_width), dtype=torch.long)
            for index, prompt in enumerate(prompts):
                prompt_ids[index, : len(prompt)] = torch.as_tensor(prompt, dtype=torch.long)
                prompt_attention[index, : len(prompt)] = 1
            output.update(
                {
                    "prompt_input_ids": prompt_ids,
                    "prompt_attention_mask": prompt_attention,
                }
            )
        if self.include_anchors:
            anchor_width = max(len(feature["anchor_input_ids"]) for feature in features)
            anchor_ids = torch.full(
                (len(features), anchor_width), self.pad_token_id, dtype=torch.long
            )
            anchor_attention = torch.zeros((len(features), anchor_width), dtype=torch.long)
            anchor_mask = torch.zeros((len(features), anchor_width - 1), dtype=torch.bool)
            anchor_reference = torch.zeros(
                (len(features), anchor_width - 1), dtype=torch.float32
            )
            for index, feature in enumerate(features):
                ids = feature["anchor_input_ids"]
                mask = feature["anchor_completion_mask"]
                ref = feature["ref_anchor_token_logps"]
                length = len(ids)
                if len(mask) != length - 1 or len(ref) != length - 1:
                    raise RuntimeError(f"Malformed anchor cache for row {feature['row_id']}")
                anchor_ids[index, :length] = torch.as_tensor(ids, dtype=torch.long)
                anchor_attention[index, :length] = 1
                anchor_mask[index, : length - 1] = torch.as_tensor(mask, dtype=torch.bool)
                anchor_reference[index, : length - 1] = torch.as_tensor(
                    ref, dtype=torch.float32
                )
            output.update(
                {
                    "anchor_input_ids": anchor_ids,
                    "anchor_attention_mask": anchor_attention,
                    "anchor_completion_mask": anchor_mask,
                    "reference_anchor_token_logps": anchor_reference,
                }
            )
        return output


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
        raise RuntimeError(f"No complete checkpoint under {output_dir}: {[p.name for p in candidates]}")
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
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.distributed.fsdp import get_fsdp_ckpt_kwargs
    from transformers.trainer import SCHEDULER_NAME
    from trl.models.activation_offloading import get_act_offloading_ctx_manager

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("WORLD_SIZE", "1") != "2":
        raise RuntimeError("Launch preference-suite training with exactly two Accelerate processes")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    set_seed(args.seed)

    train_dataset = _load_split(args.dataset_path, args.train_split)
    if args.tidpo_paper_exact and args.tidpo_repo_exact:
        raise ValueError("Choose only one of --tidpo-paper-exact and --tidpo-repo-exact")
    if (args.tidpo_paper_exact or args.tidpo_repo_exact) and args.method != "TIDPO":
        raise ValueError("TI-DPO exact-variant flags are valid only with --method TIDPO")
    paper_exact = args.method == "TIDPO" and args.tidpo_paper_exact
    repo_exact = args.method == "TIDPO" and args.tidpo_repo_exact
    live_tidpo = paper_exact or repo_exact
    upstream_evidence = None
    if repo_exact:
        upstream_root = Path(__file__).resolve().parents[2] / "third_party" / "TIDPO"
        upstream_manifest_path = upstream_root / "UPSTREAM.json"
        upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
        actual_upstream_hashes = {
            relative: _normalized_source_sha256(upstream_root / relative)
            for relative in TIDPO_UPSTREAM_SOURCE_HASHES
        }
        if (
            upstream_manifest.get("repository") != TIDPO_REPOSITORY
            or upstream_manifest.get("commit") != TIDPO_COMMIT
            or actual_upstream_hashes != TIDPO_UPSTREAM_SOURCE_HASHES
        ):
            raise RuntimeError(
                "Vendored TI-DPO source does not match pinned upstream commit "
                f"{TIDPO_COMMIT}: manifest={upstream_manifest}, hashes={actual_upstream_hashes}"
            )
        upstream_config = (upstream_root / "config" / "config.yaml").read_text(
            encoding="utf-8"
        )
        if "\noptimizer: RMSprop\n" not in f"\n{upstream_config}":
            raise RuntimeError("Pinned TI-DPO configuration no longer selects RMSprop")
        upstream_evidence = {
            "repository": TIDPO_REPOSITORY,
            "commit": TIDPO_COMMIT,
            "verified_normalized_source_sha256": actual_upstream_hashes,
            "optimizer": {
                "implementation": "torch.optim.RMSprop",
                "source": "config/config.yaml",
            },
            "olmo_adapter_normalized_source_sha256": _normalized_source_sha256(
                Path(__file__).resolve()
            ),
        }
    if paper_exact:
        published_objective_hyperparameters = {
            "tidpo_beta": 0.1,
            "tidpo_lambda_importance": 0.7,
            "tidpo_prior_sigma_div": 4.0,
            "tidpo_triplet_gamma": 0.1,
            "tidpo_triplet_margin": 0.5,
        }
        mismatched = {
            name: (float(getattr(args, name)), expected_value)
            for name, expected_value in published_objective_hyperparameters.items()
            if not math.isclose(
                float(getattr(args, name)), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        }
        if mismatched:
            raise ValueError(
                "--tidpo-paper-exact requires the arXiv:2505.19653v3 objective "
                f"hyperparameters (actual, expected): {mismatched}"
            )
    if repo_exact:
        official_repo_hyperparameters = {
            "tidpo_beta": 0.2,
            "tidpo_alpha": 0.5,
            "tidpo_lambda_importance": 0.2,
            "tidpo_prior_sigma_div": 8.0,
            "tidpo_triplet_gamma": 0.001,
            "tidpo_triplet_margin": 0.001,
            "tidpo_anchor_max_new_tokens": 64.0,
            "tidpo_anchor_top_k": 50.0,
            "tidpo_anchor_top_p": 0.95,
            "tidpo_anchor_temperature": 0.8,
        }
        mismatched = {
            name: (float(getattr(args, name)), expected_value)
            for name, expected_value in official_repo_hyperparameters.items()
            if not math.isclose(
                float(getattr(args, name)), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        }
        if mismatched or not args.tidpo2:
            raise ValueError(
                "--tidpo-repo-exact requires gracefulning/TIDPO commit e04a092 defaults "
                f"and TDPO2 enabled (actual, expected): {mismatched}"
            )
    if live_tidpo:
        if args.tidpo_anchor_max_new_tokens < 1:
            raise ValueError("--tidpo-anchor-max-new-tokens must be positive")
        if args.tidpo_anchor_top_k < 1:
            raise ValueError("--tidpo-anchor-top-k must be positive")
        if not 0.0 < args.tidpo_anchor_top_p <= 1.0:
            raise ValueError("--tidpo-anchor-top-p must be in (0, 1]")
        if args.tidpo_anchor_temperature <= 0.0:
            raise ValueError("--tidpo-anchor-temperature must be positive")
    needs_reference = args.method in {"TIDPO", "SamPO"} and not live_tidpo
    expected = {"dataset_fingerprint": train_dataset._fingerprint}
    if needs_reference:
        if args.reference_cache is None:
            raise ValueError(f"--reference-cache is required for {args.method}")
        cache_path = args.reference_cache.resolve() / "dataset"
        manifest_path = args.reference_cache.resolve() / "reference_manifest.json"
        if not cache_path.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"Run the token reference stage first: {cache_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The content fingerprint is authoritative. Do not compare the absolute dataset path:
        # Kaggle mounts a reference-only notebook's validated cache under /kaggle/input in the
        # subsequent training session.
        expected.update(
            {
                "cache_schema_version": 2,
                "split": args.train_split,
                "rows": len(train_dataset),
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "tidpo_kl_top_k": args.tidpo_kl_top_k,
            }
        )
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Reference cache does not match this training run: {mismatches}")
        reference = load_from_disk(str(cache_path))
        if len(reference) != len(train_dataset):
            raise RuntimeError("Reference cache length differs from training data")
        if reference["dataset_index"] != train_dataset["dataset_index"]:
            raise RuntimeError("Reference cache order differs from training data")
        if reference["row_id"] != train_dataset["row_id"]:
            raise RuntimeError("Reference cache row IDs differ from training data")
        cache_columns = [
            column
            for column in reference.column_names
            if column not in {"dataset_index", "row_id"}
        ]
        for column in cache_columns:
            train_dataset = train_dataset.add_column(column, reference[column])

    support_columns = {
        "ref_chosen_support_ids",
        "ref_rejected_support_ids",
        "ref_chosen_support_logps",
        "ref_rejected_support_logps",
    }
    include_support = args.method == "TIDPO" and not live_tidpo
    if include_support and not support_columns.issubset(train_dataset.column_names):
        raise RuntimeError(
            "Reference cache has no TIDPO top-k KL support; rerun the reference command"
        )

    include_anchors = (
        args.method == "TIDPO" and not live_tidpo and args.tidpo_triplet_gamma > 0.0
    )
    if live_tidpo:
        base_config = AutoConfig.from_pretrained(args.model_id, revision=args.model_revision)
        context_limit = int(getattr(base_config, "max_position_embeddings", args.max_length))
        if args.max_length + args.tidpo_anchor_max_new_tokens > context_limit:
            raise ValueError(
                "Live TI-DPO synchronized anchor decoding requires max_length + "
                f"anchor_max_new_tokens <= model context ({context_limit})"
            )
    anchor_columns = {
        "anchor_input_ids", "anchor_completion_mask", "ref_anchor_token_logps"
    }
    if include_anchors and not anchor_columns.issubset(train_dataset.column_names):
        raise RuntimeError(
            "TIDPO triplet loss is enabled but the reference cache has no anchors; rerun the "
            "reference command with --with-tidpo-anchors or set --tidpo-triplet-gamma 0"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    class VocabularyShardLinear(torch.nn.Linear):
        """An independently FSDP-sharded slice of the OLMo vocabulary head."""

    class CheckpointedChunkedLMHead(torch.nn.Module):
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

        def forward(
            self,
            hidden_states,
            labels=None,
            reference_support_ids=None,
            maximum_only=False,
            top_k_only=None,
            exact_reference_hidden_states=None,
            exact_reference_head=None,
        ):
            if exact_reference_hidden_states is not None or exact_reference_head is not None:
                if (
                    exact_reference_hidden_states is None
                    or exact_reference_head is None
                    or labels is None
                    or reference_support_ids is not None
                    or maximum_only
                    or top_k_only is not None
                ):
                    raise ValueError("Exact reference KL requires hidden states, head, and labels only")
                return self.exact_reference_statistics(
                    hidden_states,
                    exact_reference_hidden_states,
                    labels,
                    exact_reference_head,
                )
            if top_k_only is not None:
                if labels is not None or reference_support_ids is not None or maximum_only:
                    raise ValueError("top_k_only does not accept labels, support, or maximum_only")
                return self.top_k_logits(hidden_states, int(top_k_only))
            if maximum_only:
                if labels is not None or reference_support_ids is not None:
                    raise ValueError("maximum_only does not accept labels or reference support")
                return self.maximum_logit(hidden_states)
            if labels is None:
                raise ValueError("labels are required for token log-probability projection")
            selected_logits = torch.zeros_like(hidden_states[..., 0], dtype=torch.float32)
            support_selected_logits = (
                torch.zeros(
                    (*labels.shape, reference_support_ids.shape[-1]),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                if reference_support_ids is not None
                else None
            )
            log_normalizer = None
            for start, shard in zip(
                self.vocabulary_offsets, self.vocabulary_shards, strict=True
            ):
                stop = start + shard.out_features

                if reference_support_ids is None:
                    def project_and_reduce(hidden, target, head=shard, offset=start, limit=stop):
                        logits = head(hidden).float()
                        normalizer = torch.logsumexp(logits, dim=-1)
                        belongs = target.ge(offset) & target.lt(limit)
                        local_target = (target - offset).clamp(0, head.out_features - 1)
                        selected = torch.gather(
                            logits, -1, local_target.unsqueeze(-1)
                        ).squeeze(-1)
                        return normalizer, selected * belongs

                    shard_normalizer, shard_selected = activation_checkpoint(
                        project_and_reduce,
                        hidden_states,
                        labels,
                        use_reentrant=False,
                    )
                else:
                    def project_and_reduce_support(
                        hidden,
                        target,
                        support,
                        head=shard,
                        offset=start,
                        limit=stop,
                    ):
                        logits = head(hidden).float()
                        normalizer = torch.logsumexp(logits, dim=-1)
                        belongs = target.ge(offset) & target.lt(limit)
                        local_target = (target - offset).clamp(0, head.out_features - 1)
                        selected = torch.gather(
                            logits, -1, local_target.unsqueeze(-1)
                        ).squeeze(-1)
                        support_belongs = support.ge(offset) & support.lt(limit)
                        local_support = (support - offset).clamp(0, head.out_features - 1)
                        support_selected = torch.gather(logits, -1, local_support)
                        return (
                            normalizer,
                            selected * belongs,
                            support_selected * support_belongs,
                        )

                    # TIDPO's extra support output keeps this checkpoint region live across the
                    # later importance and anchor forwards. On FSDP2, its backward recomputation
                    # can then observe a resharded DTensor weight with a plain saved activation.
                    # Compute this one projection directly; activation offloading still moves its
                    # saved tensors to host RAM, and the T4s have ample headroom for one shard.
                    shard_normalizer, shard_selected, shard_support_selected = (
                        project_and_reduce_support(
                            hidden_states,
                            labels,
                            reference_support_ids,
                        )
                    )
                    support_selected_logits = support_selected_logits + shard_support_selected
                selected_logits = selected_logits + shard_selected
                log_normalizer = (
                    shard_normalizer
                    if log_normalizer is None
                    else torch.logaddexp(log_normalizer, shard_normalizer)
                )
            selected_logps = selected_logits - log_normalizer
            if support_selected_logits is None:
                return selected_logps
            return selected_logps, support_selected_logits - log_normalizer.unsqueeze(-1)

        def maximum_logit(self, hidden_states):
            maximum = None
            for shard in self.vocabulary_shards:
                def project_max(hidden, head=shard):
                    return head(hidden).float().amax(dim=-1)

                shard_maximum = activation_checkpoint(
                    project_max, hidden_states, use_reentrant=False
                )
                maximum = shard_maximum if maximum is None else torch.maximum(maximum, shard_maximum)
            return maximum

        def exact_reference_statistics(
            self,
            policy_hidden_states,
            reference_hidden_states,
            labels,
            reference_head,
        ):
            """Selected log-probs plus exact per-position KL(ref||policy), vocabulary-chunked."""
            if policy_hidden_states.shape != reference_hidden_states.shape:
                raise ValueError(
                    "Policy/reference hidden-state mismatch: "
                    f"{policy_hidden_states.shape} vs {reference_hidden_states.shape}"
                )
            if labels.shape != policy_hidden_states.shape[:-1]:
                raise ValueError(f"Label/hidden mismatch: {labels.shape} vs {policy_hidden_states.shape}")
            if (
                self.out_features != reference_head.out_features
                or self.vocabulary_offsets != reference_head.vocabulary_offsets
                or len(self.vocabulary_shards) != len(reference_head.vocabulary_shards)
            ):
                raise ValueError("Policy and reference vocabulary heads are not identically sharded")

            reference_selected = torch.zeros_like(labels, dtype=torch.float32)
            reference_normalizer = None
            # The reference is frozen, so obtain its exact global log-normalizer first without
            # retaining a graph. This lets the policy use every FSDP-sharded vocabulary slice
            # exactly once in the gradient-bearing pass below.
            for start, reference_shard in zip(
                self.vocabulary_offsets,
                reference_head.vocabulary_shards,
                strict=True,
            ):
                stop = start + reference_shard.out_features
                with torch.no_grad():
                    reference_logits = reference_shard(reference_hidden_states).float()
                    belongs = labels.ge(start) & labels.lt(stop)
                    local_target = (labels - start).clamp(
                        0, reference_shard.out_features - 1
                    )
                    shard_reference_normalizer = torch.logsumexp(
                        reference_logits, dim=-1
                    )
                    shard_reference_selected = torch.gather(
                        reference_logits, -1, local_target.unsqueeze(-1)
                    ).squeeze(-1) * belongs
                reference_selected = reference_selected + shard_reference_selected
                reference_normalizer = (
                    shard_reference_normalizer
                    if reference_normalizer is None
                    else torch.logaddexp(reference_normalizer, shard_reference_normalizer)
                )

            policy_selected = torch.zeros_like(labels, dtype=torch.float32)
            policy_normalizer = None
            reference_logp_expectation = torch.zeros_like(
                reference_normalizer, dtype=torch.float32
            )
            policy_logit_expectation = torch.zeros_like(
                reference_normalizer, dtype=torch.float32
            )
            reference_probability_mass = torch.zeros_like(
                reference_normalizer, dtype=torch.float32
            )
            for start, policy_shard, reference_shard in zip(
                self.vocabulary_offsets,
                self.vocabulary_shards,
                reference_head.vocabulary_shards,
                strict=True,
            ):
                stop = start + policy_shard.out_features

                def project_exact_statistics(
                    policy_hidden,
                    reference_hidden,
                    reference_log_normalizer,
                    target,
                    p_head=policy_shard,
                    r_head=reference_shard,
                    offset=start,
                    limit=stop,
                ):
                    policy_logits = p_head(policy_hidden).float()
                    with torch.no_grad():
                        reference_logps = (
                            r_head(reference_hidden).float()
                            - reference_log_normalizer.unsqueeze(-1)
                        )
                        reference_probabilities = reference_logps.exp()
                    belongs = target.ge(offset) & target.lt(limit)
                    local_target = (target - offset).clamp(0, p_head.out_features - 1)
                    return (
                        torch.logsumexp(policy_logits, dim=-1),
                        torch.gather(
                            policy_logits, -1, local_target.unsqueeze(-1)
                        ).squeeze(-1)
                        * belongs,
                        (reference_probabilities * reference_logps).sum(dim=-1),
                        (reference_probabilities * policy_logits).sum(dim=-1),
                        reference_probabilities.sum(dim=-1),
                    )

                (
                    shard_policy_normalizer,
                    shard_policy_selected,
                    shard_reference_logp_expectation,
                    shard_policy_logit_expectation,
                    shard_reference_mass,
                ) = activation_checkpoint(
                    project_exact_statistics,
                    policy_hidden_states,
                    reference_hidden_states,
                    reference_normalizer,
                    labels,
                    use_reentrant=False,
                )
                policy_selected = policy_selected + shard_policy_selected
                policy_normalizer = (
                    shard_policy_normalizer
                    if policy_normalizer is None
                    else torch.logaddexp(policy_normalizer, shard_policy_normalizer)
                )
                reference_logp_expectation = (
                    reference_logp_expectation + shard_reference_logp_expectation
                )
                policy_logit_expectation = (
                    policy_logit_expectation + shard_policy_logit_expectation
                )
                reference_probability_mass = (
                    reference_probability_mass + shard_reference_mass
                )

            per_position_kl = (
                reference_logp_expectation
                - policy_logit_expectation
                + policy_normalizer
            )
            if not getattr(self, "_exact_kl_audited", False):
                if not bool(torch.isfinite(per_position_kl).all()):
                    raise FloatingPointError("Non-finite exact full-vocabulary TI-DPO KL")
                mass_error = reference_probability_mass.sub(1.0).abs().max()
                if float(mass_error) > 5e-5:
                    raise RuntimeError(
                        "Exact reference vocabulary probability mass did not sum to one: "
                        f"max error={float(mass_error)}"
                    )
                self._exact_kl_audited = True

            policy_logps = policy_selected - policy_normalizer
            reference_logps = reference_selected - reference_normalizer
            return policy_logps, reference_logps, per_position_kl

        def top_k_logits(self, hidden_states, top_k: int):
            """Return exact global top-k logits without materializing the full vocabulary."""
            if top_k < 1 or top_k > self.out_features:
                raise ValueError(f"top_k must be in [1, {self.out_features}], got {top_k}")
            candidate_logits = []
            candidate_ids = []
            for start, shard in zip(
                self.vocabulary_offsets, self.vocabulary_shards, strict=True
            ):
                logits = shard(hidden_states).float()
                local_k = min(top_k, shard.out_features)
                local_logits, local_ids = torch.topk(logits, k=local_k, dim=-1)
                candidate_logits.append(local_logits)
                candidate_ids.append(local_ids + start)
            logits = torch.cat(candidate_logits, dim=-1)
            ids = torch.cat(candidate_ids, dim=-1)
            selected_logits, selected = torch.topk(logits, k=top_k, dim=-1)
            return selected_logits, torch.gather(ids, -1, selected)

    class DTensorPagedAdamW32bit(bnb.optim.PagedAdamW32bit):
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
                    local_grad, state["gnorm_vec"], step, config["percentile_clipping"]
                )
            else:
                gnorm_scale = 1.0
            _optimizer_update_32bit(
                bnb_functional,
                self.optimizer_name,
                local_grad,
                local_parameter,
                state,
                config,
                step,
                gnorm_scale,
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
                        group, parameter, local_parameter, local_grad, group_index, param_index
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
        slices: dict[int, torch.Tensor] = {}
        prefix = "lm_head.vocabulary_shards."
        for weight_file in weight_files:
            for name, tensor in load_file(weight_file, device="cpu").items():
                if name.startswith(prefix) and name.endswith(".weight"):
                    index = int(name[len(prefix) : -len(".weight")])
                    slices[index] = tensor
                else:
                    tensors[name] = tensor
        if not slices:
            return
        if sorted(slices) != list(range(len(slices))):
            raise RuntimeError(f"Incomplete vocabulary-head slices: {sorted(slices)}")
        tensors["lm_head.weight"] = torch.cat(
            [slices[index] for index in range(len(slices))], dim=0
        )
        temporary = model_dir / "model.consolidated.safetensors.tmp"
        save_file(tensors, temporary, metadata={"format": "pt"})
        for weight_file in weight_files:
            weight_file.unlink()
        (model_dir / "model.safetensors.index.json").unlink(missing_ok=True)
        temporary.replace(model_dir / "model.safetensors")

    class PreferenceSuiteTrainer(Trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            live_reference_model = trainer_kwargs.pop("live_reference_model", None)
            super().__init__(*trainer_args, **trainer_kwargs)
            self.live_reference_model = live_reference_model
            if self.live_reference_model is not None:
                self.live_reference_model.to(self.accelerator.device)
                self.live_reference_model.eval()
            self.fsdp_root_initialized = False
            self.maybe_activation_offload_context = (
                get_act_offloading_ctx_manager(model=self.model, max_fwd_stash_size=1)
                if args.activation_offloading
                else None
            )
            self.preference_metrics: dict[str, list[float]] = {
                "chosen_score": [],
                "rejected_score": [],
                "preference_logit": [],
                "reward_accuracy": [],
                "triplet_loss": [],
            }

        def training_step(self, *step_args, **step_kwargs):
            if self.maybe_activation_offload_context is None:
                return super().training_step(*step_args, **step_kwargs)
            with self.maybe_activation_offload_context:
                return super().training_step(*step_args, **step_kwargs)

        def _gradient_importance(self, model, input_ids, attention_mask):
            original_training = model.training
            model.eval()
            try:
                embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
                targets = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    use_cache=False,
                    importance_only=True,
                )
                gradients = torch.autograd.grad(
                    targets.sum(),
                    embeddings,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                return gradients.detach().abs().sum(dim=-1)
            finally:
                model.train(original_training)

        def _initialize_fsdp_root(self, model, input_ids):
            """Run one root forward before attribution accesses the sharded embedding module."""
            if self.fsdp_root_initialized:
                return
            original_training = model.training
            model.eval()
            try:
                warmup_ids = input_ids[:1, :1]
                with torch.no_grad():
                    target = model(
                        input_ids=warmup_ids,
                        attention_mask=torch.ones_like(warmup_ids),
                        use_cache=False,
                        importance_only=True,
                    )
                if target.shape != (1,) or not bool(torch.isfinite(target).all()):
                    raise RuntimeError(f"Invalid FSDP root warm-up target: {target}")
                self.fsdp_root_initialized = True
            finally:
                model.train(original_training)

        def _paper_gradient_importance(
            self, model, input_ids, attention_mask, completion_mask
        ):
            """Implement TI-DPO v3 Eqs. 5-6 on the observed response tokens."""
            original_training = model.training
            model.eval()
            try:
                embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
                targets = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    use_cache=False,
                    importance_only=True,
                )
                gradients = torch.autograd.grad(
                    targets.sum(),
                    embeddings,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                # Log-probabilities for input_ids[:, 1:] are predicted one position earlier, but
                # attribution I_i belongs to token embedding e_i itself (paper Eq. 6).
                importances = gradients[:, 1:].detach().abs().sum(dim=-1)
                if importances.shape != completion_mask.shape:
                    raise RuntimeError(
                        f"Paper TI-DPO attribution/mask mismatch: {importances.shape} vs "
                        f"{completion_mask.shape}"
                    )
                return importances
            finally:
                model.train(original_training)

        def _sample_live_anchor(self, model, prompt_ids, prompt_attention):
            """Sample the intermediate y from the current policy (TI-DPO Algorithm 1)."""
            if prompt_ids.shape[0] != 1:
                raise RuntimeError("Live-anchor TI-DPO requires per-device pair batch size 1")
            if not bool(prompt_attention.to(torch.bool).all()):
                raise RuntimeError("Live-anchor TI-DPO expects one unpadded local prompt")
            prompt_length = int(prompt_ids.shape[1])
            local_limit = min(
                int(args.tidpo_anchor_max_new_tokens), int(args.max_length) - prompt_length
            )
            # Fail collectively: one rank raising while another enters an FSDP forward deadlocks.
            synchronized_limit = torch.tensor([local_limit], dtype=torch.long, device=prompt_ids.device)
            torch.distributed.all_reduce(
                synchronized_limit, op=torch.distributed.ReduceOp.MIN
            )
            if int(synchronized_limit.item()) < 1:
                raise RuntimeError("Prompt leaves no room for a live TI-DPO anchor response")
            # Every rank performs the configured number of decoding forwards for FSDP collective
            # symmetry. Tokens beyond a rank's local 1024-token allowance are dummy EOS inputs and
            # are discarded before the triplet forward, so another rank's prompt length cannot
            # shorten this sample's anchor.
            decode_steps = int(args.tidpo_anchor_max_new_tokens)
            eos_id = tokenizer.eos_token_id
            filler_id = eos_id if eos_id is not None else tokenizer.pad_token_id
            generated = []
            generated_mask = []
            finished = False
            original_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    top_logits, top_ids, past = model(
                        input_ids=prompt_ids,
                        attention_mask=prompt_attention,
                        use_cache=True,
                        generation_top_k=args.tidpo_anchor_top_k,
                    )
                    running_attention = prompt_attention
                    for step in range(decode_steps):
                        within_local_limit = step < local_limit
                        if finished or not within_local_limit:
                            next_token = torch.full(
                                (1,), filler_id, dtype=torch.long, device=prompt_ids.device
                            )
                            is_valid = False
                        else:
                            logits = top_logits / float(args.tidpo_anchor_temperature)
                            probabilities = torch.softmax(logits, dim=-1)
                            cumulative = probabilities.cumsum(dim=-1)
                            remove = (cumulative - probabilities) >= float(
                                args.tidpo_anchor_top_p
                            )
                            probabilities = probabilities.masked_fill(remove, 0.0)
                            probabilities = probabilities / probabilities.sum(
                                dim=-1, keepdim=True
                            )
                            sampled = torch.multinomial(probabilities, num_samples=1)
                            next_token = torch.gather(top_ids, -1, sampled).squeeze(-1)
                            is_valid = True
                        generated.append(next_token)
                        generated_mask.append(is_valid)
                        if eos_id is not None and is_valid and int(next_token.item()) == eos_id:
                            finished = True
                        if step + 1 == decode_steps:
                            break
                        running_attention = torch.cat(
                            [
                                running_attention,
                                torch.ones(
                                    (1, 1),
                                    dtype=running_attention.dtype,
                                    device=running_attention.device,
                                ),
                            ],
                            dim=1,
                        )
                        top_logits, top_ids, past = model(
                            input_ids=next_token[:, None],
                            attention_mask=running_attention,
                            past_key_values=past,
                            use_cache=True,
                            generation_top_k=args.tidpo_anchor_top_k,
                        )
            finally:
                model.train(original_training)
            generated = generated[:local_limit]
            generated_mask = generated_mask[:local_limit]
            generated_ids = torch.stack(generated, dim=1)
            anchor_ids = torch.cat([prompt_ids, generated_ids], dim=1)
            anchor_attention = torch.ones_like(anchor_ids)
            anchor_mask = torch.zeros(
                (1, anchor_ids.shape[1] - 1), dtype=torch.bool, device=anchor_ids.device
            )
            start = prompt_length - 1
            anchor_mask[0, start : start + len(generated_mask)] = torch.as_tensor(
                generated_mask, dtype=torch.bool, device=anchor_ids.device
            )
            if not bool(anchor_mask.any()):
                raise RuntimeError("Current-policy anchor generation produced no response token")
            return anchor_ids, anchor_attention, anchor_mask

        @staticmethod
        def _combine_pair_and_anchor(inputs, anchor_ids, anchor_attention, anchor_mask):
            pair_ids = inputs["input_ids"]
            pair_attention = inputs["attention_mask"]
            pair_mask = inputs["completion_mask"]
            width = max(pair_ids.shape[1], anchor_ids.shape[1])
            rows = pair_ids.shape[0] + anchor_ids.shape[0]
            combined_ids = torch.full(
                (rows, width),
                tokenizer.pad_token_id,
                dtype=pair_ids.dtype,
                device=pair_ids.device,
            )
            combined_attention = torch.zeros(
                (rows, width), dtype=pair_attention.dtype, device=pair_ids.device
            )
            combined_mask = torch.zeros(
                (rows, width - 1), dtype=torch.bool, device=pair_ids.device
            )
            combined_ids[: pair_ids.shape[0], : pair_ids.shape[1]] = pair_ids
            combined_attention[: pair_ids.shape[0], : pair_ids.shape[1]] = pair_attention
            combined_mask[: pair_ids.shape[0], : pair_mask.shape[1]] = pair_mask
            combined_ids[pair_ids.shape[0] :, : anchor_ids.shape[1]] = anchor_ids
            combined_attention[pair_ids.shape[0] :, : anchor_ids.shape[1]] = anchor_attention
            combined_mask[pair_ids.shape[0] :, : anchor_mask.shape[1]] = anchor_mask
            return combined_ids, combined_attention, combined_mask

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            extra: dict[str, Any] = {}
            if repo_exact:
                if self.live_reference_model is None:
                    raise RuntimeError(
                        "Official-repo-exact TI-DPO requires a live frozen reference model"
                    )
                self._initialize_fsdp_root(model, inputs["input_ids"])
                importances = self._gradient_importance(
                    model, inputs["input_ids"], inputs["attention_mask"]
                )
                full_importance_weights = tidpo_importance_weights(
                    importances,
                    inputs["attention_mask"].to(torch.bool),
                    lambda_importance=args.tidpo_lambda_importance,
                    prior_sigma_div=args.tidpo_prior_sigma_div,
                )
                # The pinned repository normalizes over the full non-padding sequence, shifts
                # once for next-token predictions, then masks the prompt in the TDPO2 loss.
                pair_importance_weights = full_importance_weights[:, 1:]

                anchor_ids, anchor_attention, anchor_mask = self._sample_live_anchor(
                    model, inputs["prompt_input_ids"], inputs["prompt_attention_mask"]
                )
                combined_ids, combined_attention, combined_mask = (
                    self._combine_pair_and_anchor(
                        inputs, anchor_ids, anchor_attention, anchor_mask
                    )
                )
                (
                    policy_all_logps,
                    reference_all_logps,
                    all_per_position_kl,
                ) = model(
                    input_ids=combined_ids,
                    attention_mask=combined_attention,
                    use_cache=False,
                    preference_labels=combined_ids[:, 1:],
                    exact_reference_model=self.live_reference_model,
                )
                pair_rows = inputs["input_ids"].shape[0]
                policy_logps = policy_all_logps[:pair_rows]
                reference_logps = reference_all_logps[:pair_rows]
                completion_mask = combined_mask[:pair_rows]
                importance_weights = torch.zeros_like(policy_logps, dtype=torch.float32)
                importance_weights[:, : pair_importance_weights.shape[1]] = (
                    pair_importance_weights
                )
                position_kl = (
                    all_per_position_kl[:pair_rows] * completion_mask
                ).sum(dim=-1)
                anchor_policy_logps = policy_all_logps[pair_rows:]
                anchor_reference_logps = reference_all_logps[pair_rows:]
                ratios = policy_logps - reference_logps.to(policy_logps.dtype)
                anchor_ratios = anchor_policy_logps - anchor_reference_logps.to(
                    anchor_policy_logps.dtype
                )
                pair_count = policy_logps.shape[0] // 2
                triplet = packed_triplet_loss(
                    anchor_ratios,
                    combined_mask[pair_rows:],
                    ratios[:pair_count],
                    completion_mask[:pair_count],
                    ratios[pair_count:],
                    completion_mask[pair_count:],
                    margin=args.tidpo_triplet_margin,
                )
                (
                    loss,
                    chosen,
                    rejected,
                    logits,
                    base_loss,
                    triplet,
                    chosen_kl,
                    rejected_kl,
                ) = tidpo_pair_loss(
                    policy_logps,
                    reference_logps,
                    completion_mask,
                    importance_weights,
                    beta=args.tidpo_beta,
                    position_kl=position_kl,
                    alpha=args.tidpo_alpha,
                    if_tdpo2=True,
                    triplet_loss=triplet,
                    triplet_gamma=args.tidpo_triplet_gamma,
                )
                extra["base_loss"] = base_loss.detach()
                extra["chosen_position_kl"] = chosen_kl.mean().detach()
                extra["rejected_position_kl"] = rejected_kl.mean().detach()
                extra["exact_position_kl_mean"] = position_kl.mean().detach()
                extra["importance_weight_max"] = importance_weights[
                    completion_mask
                ].max().detach()
                extra["importance_weight_min_valid"] = importance_weights[
                    completion_mask
                ].min().detach()
                extra["anchor_tokens"] = combined_mask[pair_rows:].sum().to(torch.float32)
            elif paper_exact:
                if self.live_reference_model is None:
                    raise RuntimeError("Paper-exact TI-DPO requires a live frozen reference model")
                anchor_ids, anchor_attention, anchor_mask = self._sample_live_anchor(
                    model, inputs["prompt_input_ids"], inputs["prompt_attention_mask"]
                )
                importances = self._paper_gradient_importance(
                    model,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["completion_mask"],
                )
                pair_weights = paper_tidpo_importance_weights(
                    importances,
                    inputs["completion_mask"],
                    lambda_importance=args.tidpo_lambda_importance,
                    prior_sigma_div=args.tidpo_prior_sigma_div,
                )
                combined_ids, combined_attention, combined_mask = (
                    self._combine_pair_and_anchor(
                        inputs, anchor_ids, anchor_attention, anchor_mask
                    )
                )
                policy_all_logps = model(
                    input_ids=combined_ids,
                    attention_mask=combined_attention,
                    use_cache=False,
                    preference_labels=combined_ids[:, 1:],
                )
                with torch.no_grad():
                    reference_all_logps = self.live_reference_model(
                        input_ids=combined_ids,
                        attention_mask=combined_attention,
                        use_cache=False,
                        preference_labels=combined_ids[:, 1:],
                    )
                pair_rows = inputs["input_ids"].shape[0]
                policy_logps = policy_all_logps[:pair_rows]
                reference_logps = reference_all_logps[:pair_rows]
                completion_mask = combined_mask[:pair_rows]
                importance_weights = torch.zeros_like(policy_logps, dtype=torch.float32)
                importance_weights[:, : pair_weights.shape[1]] = pair_weights
                ratios = policy_logps - reference_logps.to(policy_logps.dtype)
                anchor_ratios = policy_all_logps[pair_rows:] - reference_all_logps[
                    pair_rows:
                ].to(policy_all_logps.dtype)
                pair_count = pair_rows // 2
                triplet = packed_triplet_loss(
                    anchor_ratios,
                    combined_mask[pair_rows:],
                    ratios[:pair_count],
                    completion_mask[:pair_count],
                    ratios[pair_count:],
                    completion_mask[pair_count:],
                    margin=args.tidpo_triplet_margin,
                )
                loss, chosen, rejected, logits, base_loss, triplet = paper_tidpo_pair_loss(
                    policy_logps,
                    reference_logps,
                    completion_mask,
                    importance_weights,
                    beta=args.tidpo_beta,
                    triplet_loss=triplet,
                    triplet_gamma=args.tidpo_triplet_gamma,
                )
                extra["base_loss"] = base_loss.detach()
                extra["importance_weight_max"] = importance_weights[
                    completion_mask
                ].max().detach()
                extra["importance_weight_min_valid"] = importance_weights[
                    completion_mask
                ].min().detach()
                extra["anchor_tokens"] = combined_mask[pair_rows:].sum().to(torch.float32)
            else:
                policy_output = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                    preference_labels=inputs["input_ids"][:, 1:],
                    reference_support_ids=(
                        inputs["reference_support_ids"] if args.method == "TIDPO" else None
                    ),
                )
                if args.method == "TIDPO":
                    policy_logps, policy_support_logps = policy_output
                else:
                    policy_logps = policy_output
                    policy_support_logps = None
                completion_mask = inputs["completion_mask"]
                reference_logps = inputs["reference_token_logps"]
                triplet = torch.zeros((), device=policy_logps.device, dtype=policy_logps.dtype)

                if args.method == "SimPO":
                    loss, chosen, rejected, logits = simpo_pair_loss(
                        policy_logps,
                        completion_mask,
                        beta=args.simpo_beta,
                        gamma_beta_ratio=args.simpo_gamma_beta_ratio,
                    )
                elif args.method == "SamPO":
                    loss, chosen, rejected, logits, sampled_counts = sampo_pair_loss(
                        policy_logps,
                        reference_logps,
                        completion_mask,
                        beta=args.sampo_beta,
                    )
                    extra["sampled_tokens"] = sampled_counts.to(torch.float32).mean().detach()
                elif args.method == "TIDPO":
                    importances = self._gradient_importance(
                        model, inputs["input_ids"], inputs["attention_mask"]
                    )
                    full_importance_weights = tidpo_importance_weights(
                        importances,
                        inputs["attention_mask"].to(torch.bool),
                        lambda_importance=args.tidpo_lambda_importance,
                        prior_sigma_div=args.tidpo_prior_sigma_div,
                    )
                    # Match the imported implementation: normalize over all non-padding sequence
                    # tokens, shift once, and apply the resulting weights only at response positions.
                    importance_weights = full_importance_weights[:, 1:]
                    position_kl = topk_bucket_position_kl(
                        policy_support_logps,
                        inputs["reference_support_logps"],
                        completion_mask,
                    )
                    if include_anchors:
                        anchor_policy_logps = model(
                            input_ids=inputs["anchor_input_ids"],
                            attention_mask=inputs["anchor_attention_mask"],
                            use_cache=False,
                            preference_labels=inputs["anchor_input_ids"][:, 1:],
                        )
                        pair_count = policy_logps.shape[0] // 2
                        ratios = policy_logps - reference_logps.to(policy_logps.dtype)
                        anchor_ratios = anchor_policy_logps - inputs[
                            "reference_anchor_token_logps"
                        ].to(anchor_policy_logps.dtype)
                        triplet = packed_triplet_loss(
                            anchor_ratios,
                            inputs["anchor_completion_mask"],
                            ratios[:pair_count],
                            completion_mask[:pair_count],
                            ratios[pair_count:],
                            completion_mask[pair_count:],
                            margin=args.tidpo_triplet_margin,
                        )
                    (
                        loss,
                        chosen,
                        rejected,
                        logits,
                        base_loss,
                        triplet,
                        chosen_kl,
                        rejected_kl,
                    ) = tidpo_pair_loss(
                        policy_logps,
                        reference_logps,
                        completion_mask,
                        importance_weights,
                        beta=args.tidpo_beta,
                        position_kl=position_kl,
                        alpha=args.tidpo_alpha,
                        if_tdpo2=args.tidpo2,
                        triplet_loss=triplet,
                        triplet_gamma=args.tidpo_triplet_gamma,
                    )
                    extra["base_loss"] = base_loss.detach()
                    extra["chosen_position_kl"] = chosen_kl.mean().detach()
                    extra["rejected_position_kl"] = rejected_kl.mean().detach()
                    extra["importance_weight_max"] = importance_weights.max().detach()
                    extra["importance_weight_min_valid"] = importance_weights[
                        completion_mask
                    ].min().detach()
                else:
                    raise AssertionError(args.method)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {args.method} loss: {loss.detach()}")
            gathered = {
                "chosen_score": self.accelerator.gather(chosen.detach()).mean().item(),
                "rejected_score": self.accelerator.gather(rejected.detach()).mean().item(),
                "preference_logit": self.accelerator.gather(logits.detach()).mean().item(),
                "reward_accuracy": self.accelerator.gather(
                    chosen.detach().gt(rejected.detach()).to(torch.float32)
                ).mean().item(),
                "triplet_loss": self.accelerator.gather(triplet.detach().reshape(1)).mean().item(),
            }
            for key, value in gathered.items():
                self.preference_metrics[key].append(value)
            for key, value in extra.items():
                gathered_value = self.accelerator.gather(value.reshape(1)).mean().item()
                self.preference_metrics.setdefault(key, []).append(gathered_value)
            torch.cuda.empty_cache()
            outputs = {key: torch.as_tensor(value, device=loss.device) for key, value in gathered.items()}
            return (loss, outputs) if return_outputs else loss

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            if self.model.training:
                for key, values in self.preference_metrics.items():
                    if values:
                        logs[key] = sum(values) / len(values)
                        values.clear()
            super().log(logs, start_time)

        def create_optimizer(self):
            super().create_optimizer()
            optimizer = self.optimizer
            if repo_exact:
                if not isinstance(optimizer, torch.optim.RMSprop):
                    raise RuntimeError(
                        "Official-repo-exact TI-DPO requires the upstream repository's "
                        f"torch RMSprop optimizer, got {type(optimizer)}"
                    )
                # Avoid multi-tensor optimizer kernels over FSDP2 DTensors. This selects the
                # single-tensor implementation of the same RMSprop update used by the repository.
                optimizer.defaults["foreach"] = False
                for group in optimizer.param_groups:
                    group["foreach"] = False
                return optimizer
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
            optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
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
                                *saved.shape, dtype=saved.dtype, device=parameter.device
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

    def install_memory_efficient_forward(target_model):
        standard_model_forward = target_model.forward

        @functools.wraps(standard_model_forward)
        def memory_efficient_model_forward(
            self,
            *model_args,
            preference_labels=None,
            reference_support_ids=None,
            exact_reference_model=None,
            importance_only=False,
            generation_top_k=None,
            **model_kwargs,
        ):
            if (
                preference_labels is None
                and exact_reference_model is None
                and not importance_only
                and generation_top_k is None
            ):
                return standard_model_forward(*model_args, **model_kwargs)
            model_kwargs.pop("labels", None)
            model_kwargs["return_dict"] = True
            outputs = self.model(*model_args, **model_kwargs)
            if exact_reference_model is not None:
                if (
                    preference_labels is None
                    or reference_support_ids is not None
                    or importance_only
                    or generation_top_k is not None
                ):
                    raise ValueError(
                        "Exact reference statistics require labels and cannot be combined "
                        "with support, importance, or generation modes"
                    )
                with torch.no_grad():
                    reference_outputs = exact_reference_model.model(
                        *model_args, **model_kwargs
                    )
                return self.lm_head(
                    outputs.last_hidden_state[..., :-1, :],
                    labels=preference_labels,
                    exact_reference_hidden_states=(
                        reference_outputs.last_hidden_state[..., :-1, :]
                    ),
                    exact_reference_head=exact_reference_model.lm_head,
                )
            if generation_top_k is not None:
                hidden = outputs.last_hidden_state[:, -1, :]
                top_logits, top_ids = self.lm_head(
                    hidden, top_k_only=int(generation_top_k)
                )
                return top_logits, top_ids, outputs.past_key_values
            if importance_only:
                attention_mask = model_kwargs.get("attention_mask")
                if attention_mask is None:
                    raise ValueError("importance_only requires an attention mask")
                last_positions = attention_mask.to(torch.long).sum(dim=-1).sub(1).clamp_min(0)
                batch_indices = torch.arange(
                    outputs.last_hidden_state.shape[0], device=outputs.last_hidden_state.device
                )
                last_hidden = outputs.last_hidden_state[batch_indices, last_positions]
                # Invoke the module through __call__/forward so FSDP2 runs the parent head's
                # pre/post-forward hooks. Calling maximum_logit() directly bypassed those hooks and
                # left the nested vocabulary shards resharded before the main loss backward.
                return self.lm_head(last_hidden, maximum_only=True)
            hidden_states = outputs.last_hidden_state[..., :-1, :]
            return self.lm_head(hidden_states, preference_labels, reference_support_ids)

        target_model.forward = MethodType(memory_efficient_model_forward, target_model)

    install_memory_efficient_forward(model)

    live_reference_model = None
    if live_tidpo:
        live_reference_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        live_reference_model.requires_grad_(False)
        live_reference_model.config.use_cache = False
        live_reference_model.lm_head = CheckpointedChunkedLMHead(
            live_reference_model.lm_head
        )
        live_reference_model.requires_grad_(False)
        install_memory_efficient_forward(live_reference_model)
        live_reference_model.eval()

    save_enabled = args.save_steps > 0
    run_name = args.run_name or f"olmo2-1b-bees-{args.method.lower()}"
    optimizer_name = "rmsprop" if repo_exact else "paged_adamw_32bit"
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
        optim=optimizer_name,
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
    trainer = PreferenceSuiteTrainer(
        model=model,
        live_reference_model=live_reference_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PreferenceTokenCollator(
            tokenizer.pad_token_id,
            include_reference=needs_reference,
            include_support=include_support,
            include_anchors=include_anchors,
            include_live_prompts=live_tidpo,
        ),
        processing_class=tokenizer,
    )
    trainer.create_optimizer()
    optimizer_class = f"{trainer.optimizer.__class__.__module__}.{trainer.optimizer.__class__.__name__}"
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
        or (32 if isinstance(trainer.optimizer, torch.optim.RMSprop) else 0)
    )
    if repo_exact:
        if (
            not isinstance(trainer.optimizer, torch.optim.RMSprop)
            or optimizer_is_paged
            or optimizer_bits != 32
        ):
            raise RuntimeError(
                f"Expected non-paged torch RMSprop with FP32 state, got {optimizer_class} "
                f"(is_paged={optimizer_is_paged}, bits={optimizer_bits})"
            )
    elif not optimizer_is_paged or optimizer_bits != 32:
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
            raise RuntimeError(f"No safetensors weights under {final_dir}")
        saved_dtypes: set[str] = set()
        tensor_count = 0
        for weight_file in weight_files:
            with safe_open(weight_file, framework="pt", device="cpu") as handle:
                for tensor_name in handle.keys():
                    saved_dtypes.add(str(handle.get_slice(tensor_name).get_dtype()))
                    tensor_count += 1
        if saved_dtypes != {"F32"}:
            raise RuntimeError(f"Final checkpoint is not all FP32: {saved_dtypes}")
        hashes = {path.name: sha256_file(path) for path in weight_files}
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        write_json(
            args.output_dir / "training_manifest.json",
            {
                "method": args.method,
                "tidpo_variant": (
                    "official_repo_exact"
                    if repo_exact
                    else (
                        "paper_exact"
                        if paper_exact
                        else ("repository_approximation" if args.method == "TIDPO" else None)
                    )
                ),
                "method_description": (
                    "Exact objective port of gracefulning/TIDPO commit e04a092 to OLMo: "
                    "full-sequence gradient/Gaussian weighting, full-vocabulary TDPO2 KL, "
                    "and a live current-policy triplet anchor"
                    if repo_exact
                    else (
                        "TI-DPO paper-equation variant from arXiv:2505.19653v3 Eqs. 5-14"
                        if paper_exact
                        else METHOD_DESCRIPTIONS[args.method]
                    )
                ),
                "objective_formula": (
                    METHOD_FORMULAS["TIDPO"]
                    if repo_exact
                    else (
                        "-logsigmoid(beta * (sum_t w+_t log(pi/ref) - sum_t w-_t "
                        "log(pi/ref))) + triplet_gamma * L_triplet; w=lambda*normalize(L1 "
                        "last-logit gradient attribution)+(1-lambda)*Gaussian"
                        if paper_exact
                        else METHOD_FORMULAS[args.method]
                    )
                ),
                "training_signature": {
                    "method": args.method,
                    "epochs": args.epochs,
                    "max_steps": args.max_steps,
                    "learning_rate": args.learning_rate,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "seed": args.seed,
                    "transformer_layer_class": args.transformer_layer_class,
                    "activation_offloading": args.activation_offloading,
                    "max_length": args.max_length,
                    "tidpo_paper_exact": args.tidpo_paper_exact,
                    "tidpo_repo_exact": args.tidpo_repo_exact,
                    "tidpo_beta": args.tidpo_beta,
                    "tidpo_alpha": args.tidpo_alpha,
                    "tidpo2": args.tidpo2,
                    "tidpo_kl_top_k": args.tidpo_kl_top_k,
                    "tidpo_lambda_importance": args.tidpo_lambda_importance,
                    "tidpo_prior_sigma_div": args.tidpo_prior_sigma_div,
                    "tidpo_triplet_gamma": args.tidpo_triplet_gamma,
                    "tidpo_triplet_margin": args.tidpo_triplet_margin,
                    "tidpo_anchor_max_new_tokens": args.tidpo_anchor_max_new_tokens,
                    "tidpo_anchor_top_k": args.tidpo_anchor_top_k,
                    "tidpo_anchor_top_p": args.tidpo_anchor_top_p,
                    "tidpo_anchor_temperature": args.tidpo_anchor_temperature,
                    "simpo_beta": args.simpo_beta,
                    "simpo_gamma_beta_ratio": args.simpo_gamma_beta_ratio,
                    "sampo_beta": args.sampo_beta,
                },
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "dataset_path": str(args.dataset_path.resolve()),
                "dataset_fingerprint": expected["dataset_fingerprint"],
                "reference_cache": (
                    str(args.reference_cache.resolve()) if args.reference_cache is not None else None
                ),
                "final_model": str(final_dir),
                "tidpo_paper": "arXiv:2505.19653v3" if paper_exact else None,
                "tidpo_upstream_repository": TIDPO_REPOSITORY,
                "tidpo_upstream_commit": TIDPO_COMMIT,
                "tidpo_source_evidence": upstream_evidence,
                "tidpo_fidelity": (
                    "official_repository_objective_exact"
                    if repo_exact
                    else (
                        "paper_equations_exact"
                        if paper_exact
                        else ("repository_inspired_approximation" if args.method == "TIDPO" else None)
                    )
                ),
                "tidpo_controlled_port_disclosure": (
                    "The method-specific objective and pinned defaults match the official "
                    "repository. The optimizer is also the repository-default torch RMSprop; "
                    "the base model, dataset, schedule, precision, and distributed runtime are "
                    "the OLMo comparison recipe, not the authors' Llama/Mistral experiment recipe."
                    if repo_exact
                    else None
                ),
                "tidpo_fixed_anchor_cache": include_anchors,
                "tidpo_live_policy_anchor": live_tidpo,
                "tidpo_position_kl": {
                    "kind": (
                        "exact_full_vocabulary_reference_to_policy"
                        if repo_exact
                        else (
                            "not_in_paper_objective"
                            if paper_exact
                            else "reference_topk_plus_remainder_bucket_lower_bound"
                        )
                    ),
                    "top_k": None if live_tidpo else args.tidpo_kl_top_k,
                    "exact_full_vocabulary": True if repo_exact else (None if paper_exact else False),
                    "support_projection_activation_checkpointing": (
                        None if live_tidpo else False
                    ),
                    "support_projection_activation_offloading": (
                        None if live_tidpo else args.activation_offloading
                    ),
                    "reason": (
                        "Pinned repository TDPO2 computes KL(ref||policy) over every vocabulary token"
                        if repo_exact
                        else (
                            "TI-DPO paper Eqs. 11-14 contain no TDPO position-KL term"
                            if paper_exact
                            else "avoid a resident full reference model during full-parameter training"
                        )
                    ),
                },
                "tidpo_importance_normalization": (
                    "all_nonpadding_sequence_tokens_then_shift_and_apply_at_response_positions"
                    if repo_exact
                    else (
                        "per_response_sum_normalized_gradient_plus_unnormalized_gaussian_eqs_6_8"
                        if paper_exact
                        else "all_nonpadding_sequence_tokens_then_apply_at_response_positions"
                    )
                ),
                "tidpo_gaussian_prior": True,
                "tidpo_gaussian_prior_sum_normalized": False if paper_exact else True,
                "tidpo_weight_length_rescale": True if repo_exact else (False if paper_exact else None),
                "tidpo_official_defaults_enforced": repo_exact,
                "tidpo_official_objective": (
                    {
                        "base": "weighted_tdpo2",
                        "beta": 0.2,
                        "alpha": 0.5,
                        "if_tdpo2": True,
                        "position_kl_direction": "KL(reference||policy)",
                        "position_kl_vocabulary": "full",
                        "triplet_gamma": 0.001,
                        "triplet_margin": 0.001,
                    }
                    if repo_exact
                    else None
                ),
                "tidpo_official_importance": (
                    {
                        "attribution_target": "maximum_logit_at_last_valid_sequence_position",
                        "gradient_norm": "L1_over_embedding_dimension",
                        "normalization_scope": "all_nonpadding_prompt_and_response_tokens",
                        "lambda_importance": 0.2,
                        "gaussian_center": "(valid_token_count-1)/2",
                        "gaussian_sigma": "max(1, valid_token_count/8)",
                        "gaussian_sum_normalized": True,
                        "mixture_sum_normalized": True,
                        "mixture_rescaled_by_valid_token_count": True,
                        "application": "shift_once_then_mask_to_labeled_response_positions",
                    }
                    if repo_exact
                    else None
                ),
                "tidpo_official_anchor": (
                    {
                        "source": "current_policy_same_prompt_live",
                        "max_new_tokens": 64,
                        "do_sample": True,
                        "top_k": 50,
                        "top_p": 0.95,
                        "temperature": 0.8,
                        "triplet_alignment": "response_token_ordinal",
                    }
                    if repo_exact
                    else None
                ),
                "live_frozen_reference_model": live_tidpo,
                "optimizer": (
                    "torch.optim.RMSprop"
                    if repo_exact
                    else "bitsandbytes.optim.PagedAdamW32bit"
                ),
                "optimizer_class": optimizer_class,
                "optimizer_is_paged": optimizer_is_paged,
                "optimizer_state_bits": optimizer_bits,
                "optimizer_foreach": trainer.optimizer.defaults.get("foreach"),
                "fp16_initial_loss_scale": FP16_INITIAL_SCALE,
                "master_parameter_dtype": "float32",
                "compute_dtype": "float16",
                "saved_weight_dtypes": sorted(saved_dtypes),
                "saved_tensor_count": tensor_count,
                "weight_files_sha256": hashes,
                "weight_quantization": None,
                "peft_or_lora": False,
                "full_parameter_training": True,
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
    if args.command == "reference":
        run_reference(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "self-test":
        configure_workspace(args.workspace)
        results = run_loss_self_tests()
        results["optimizer_update_32bit_contract"] = _optimizer_contract_self_test()
        results["tidpo_upstream_equivalence"] = _upstream_objective_equivalence_self_test()
        results["tidpo_chunked_exact_kl"] = _chunked_exact_kl_self_test()
        print(json.dumps(results, indent=2))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
