from __future__ import annotations

import math
from typing import Any


METHODS = ("TIDPO", "SimPO", "SamPO")

METHOD_DESCRIPTIONS = {
    "TIDPO": (
        "Token-importance-weighted TDPO2 using the imported TIDPO repository's gradient-attribution "
        "mixture and position-KL correction; a cached-anchor triplet term supplies its auxiliary "
        "objective."
    ),
    "SimPO": (
        "Reference-free Bradley-Terry loss over length-normalized policy log-probabilities with a "
        "target reward margin."
    ),
    "SamPO": (
        "DPO after uniformly down-sampling the longer completion to the shorter completion's token "
        "count, eliminating the objective's direct reliance on unequal response lengths."
    ),
}

METHOD_FORMULAS = {
    "TIDPO": (
        "-logsigmoid(beta * (m_w(y+) - m_w(y-) - alpha * "
        "(KL_pos(y-) - stopgrad(KL_pos(y+))))) + triplet_gamma * L_triplet"
    ),
    "SimPO": (
        "-logsigmoid(beta * (mean_t log pi(y+_t) - mean_t log pi(y-_t)) "
        "- beta * gamma_beta_ratio)"
    ),
    "SamPO": (
        "-logsigmoid(beta * (sum_{t in S+}(log pi-log ref) - "
        "sum_{t in S-}(log pi-log ref))), |S+|=|S-|=min(T+,T-)"
    ),
}


def _validate_pair_tensors(policy_token_logps, completion_mask, reference_token_logps=None):
    import torch

    if policy_token_logps.ndim != 2:
        raise ValueError(f"Expected [2B, L] policy log-probs, got {policy_token_logps.shape}")
    if completion_mask.shape != policy_token_logps.shape:
        raise ValueError(
            f"Completion-mask mismatch: {completion_mask.shape} vs {policy_token_logps.shape}"
        )
    if policy_token_logps.shape[0] % 2:
        raise ValueError("Rows must be chosen responses followed by rejected responses")
    if reference_token_logps is not None and reference_token_logps.shape != policy_token_logps.shape:
        raise ValueError(
            f"Reference-logp mismatch: {reference_token_logps.shape} vs {policy_token_logps.shape}"
        )
    mask = completion_mask.to(torch.bool)
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("Every response must have at least one completion token")
    tensors = (policy_token_logps,) if reference_token_logps is None else (
        policy_token_logps,
        reference_token_logps,
    )
    if not all(torch.isfinite(tensor[mask]).all() for tensor in tensors):
        raise FloatingPointError("Non-finite completion-token log-probability")
    return mask, policy_token_logps.shape[0] // 2


def simpo_pair_loss(
    policy_token_logps,
    completion_mask,
    *,
    beta: float = 2.0,
    gamma_beta_ratio: float = 0.5,
):
    """SimPO Eq. 4: average policy log-probability reward plus target margin."""
    import torch
    import torch.nn.functional as functional

    mask, pair_count = _validate_pair_tensors(policy_token_logps, completion_mask)
    counts = mask.sum(dim=-1).to(policy_token_logps.dtype)
    rewards = (policy_token_logps * mask).sum(dim=-1) / counts
    chosen, rejected = rewards[:pair_count], rewards[pair_count:]
    target_margin = float(beta) * float(gamma_beta_ratio)
    logits = float(beta) * (chosen - rejected) - target_margin
    losses = -functional.logsigmoid(logits)
    return losses.mean(), chosen, rejected, logits


def sampo_pair_loss(
    policy_token_logps,
    reference_token_logps,
    completion_mask,
    *,
    beta: float = 0.1,
    generator=None,
):
    """Official SamPO down-sampling: uniformly retain min(T+, T-) tokens per pair."""
    import torch
    import torch.nn.functional as functional

    mask, pair_count = _validate_pair_tensors(
        policy_token_logps, completion_mask, reference_token_logps
    )
    ratios = policy_token_logps - reference_token_logps.to(policy_token_logps.dtype)
    chosen_scores = []
    rejected_scores = []
    sampled_counts = []
    for index in range(pair_count):
        chosen_valid = torch.nonzero(mask[index], as_tuple=False).squeeze(-1)
        rejected_valid = torch.nonzero(mask[index + pair_count], as_tuple=False).squeeze(-1)
        sample_count = min(int(chosen_valid.numel()), int(rejected_valid.numel()))
        if chosen_valid.numel() > sample_count:
            order = torch.randperm(chosen_valid.numel(), device=chosen_valid.device, generator=generator)
            chosen_valid = chosen_valid[order[:sample_count]]
        if rejected_valid.numel() > sample_count:
            order = torch.randperm(
                rejected_valid.numel(), device=rejected_valid.device, generator=generator
            )
            rejected_valid = rejected_valid[order[:sample_count]]
        chosen_scores.append(ratios[index, chosen_valid].sum())
        rejected_scores.append(ratios[index + pair_count, rejected_valid].sum())
        sampled_counts.append(sample_count)
    chosen = torch.stack(chosen_scores)
    rejected = torch.stack(rejected_scores)
    logits = float(beta) * (chosen - rejected)
    losses = -functional.logsigmoid(logits)
    counts = torch.as_tensor(sampled_counts, dtype=torch.long, device=logits.device)
    return losses.mean(), chosen, rejected, logits, counts


def tidpo_importance_weights(
    importances,
    completion_mask,
    *,
    lambda_importance: float = 0.2,
    prior_sigma_div: float = 8.0,
):
    """Reproduce the pinned TIDPO repo's gradient/Gaussian weight mixture.

    The imported implementation normalizes the mixed weights to have mean one over response
    tokens. This preserves the scale of an ordinary token-summed DPO margin.
    """
    import torch

    if importances.shape != completion_mask.shape:
        raise ValueError(f"Importance/mask mismatch: {importances.shape} vs {completion_mask.shape}")
    if not 0.0 <= lambda_importance <= 1.0:
        raise ValueError("lambda_importance must be in [0, 1]")
    if prior_sigma_div < 1.0:
        raise ValueError("prior_sigma_div must be >= 1")
    mask = completion_mask.to(torch.bool)
    weights = torch.zeros_like(importances, dtype=torch.float32)
    for row in range(importances.shape[0]):
        valid = torch.nonzero(mask[row], as_tuple=False).squeeze(-1)
        count = int(valid.numel())
        if count == 0:
            raise ValueError("Every response must have at least one completion token")
        scores = importances[row, valid].detach().to(torch.float32).clamp_min(0.0)
        normalized_scores = scores / scores.sum() if float(scores.sum()) > 0.0 else None
        positions = torch.arange(count, dtype=torch.float32, device=importances.device)
        center = (count - 1) / 2.0
        sigma = max(1.0, count / float(prior_sigma_div))
        prior = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
        prior = prior / prior.sum()
        mixed = prior if normalized_scores is None else (
            float(lambda_importance) * normalized_scores
            + (1.0 - float(lambda_importance)) * prior
        )
        mixed = mixed / mixed.sum()
        weights[row, valid] = mixed * float(count)
    return weights


def tidpo_pair_loss(
    policy_token_logps,
    reference_token_logps,
    completion_mask,
    importance_weights,
    *,
    beta: float = 0.2,
    position_kl=None,
    alpha: float = 0.5,
    if_tdpo2: bool = True,
    triplet_loss=None,
    triplet_gamma: float = 0.0,
):
    """Pinned-repository TI-DPO/TDPO core plus its optional triplet loss."""
    import torch
    import torch.nn.functional as functional

    mask, pair_count = _validate_pair_tensors(
        policy_token_logps, completion_mask, reference_token_logps
    )
    if importance_weights.shape != policy_token_logps.shape:
        raise ValueError("TIDPO importance weights must match the token log-probability tensor")
    ratios = policy_token_logps - reference_token_logps.to(policy_token_logps.dtype)
    scores = (ratios * importance_weights.to(ratios.dtype) * mask).sum(dim=-1)
    chosen_margin, rejected_margin = scores[:pair_count], scores[pair_count:]
    if position_kl is None:
        position_kl = torch.zeros_like(scores)
    if position_kl.shape != scores.shape:
        raise ValueError(f"Position-KL mismatch: {position_kl.shape} vs {scores.shape}")
    if not torch.isfinite(position_kl).all():
        raise FloatingPointError("Non-finite position KL")
    chosen_kl, rejected_kl = position_kl[:pair_count], position_kl[pair_count:]
    if if_tdpo2:
        tdpo_logits = chosen_margin - rejected_margin - float(alpha) * (
            rejected_kl - chosen_kl.detach()
        )
    else:
        tdpo_logits = chosen_margin - rejected_margin - (rejected_kl - chosen_kl)
    logits = float(beta) * tdpo_logits
    losses = -functional.logsigmoid(logits)
    base = losses.mean()
    if triplet_loss is None:
        triplet_loss = torch.zeros((), device=base.device, dtype=base.dtype)
    total = base + float(triplet_gamma) * triplet_loss
    chosen_reward = chosen_margin + chosen_kl
    rejected_reward = rejected_margin + rejected_kl
    return (
        total,
        chosen_reward,
        rejected_reward,
        logits,
        base,
        triplet_loss,
        chosen_kl,
        rejected_kl,
    )


def topk_bucket_position_kl(
    policy_support_logps,
    reference_support_logps,
    completion_mask,
):
    """Lower-bound KL(ref||policy) on reference top-k tokens plus one remainder bucket.

    The exact TIDPO repository term needs both full vocabulary distributions at every response
    position. This projection retains each reference top-k token as its own outcome and merges the
    rest of the vocabulary into one outcome, making the term cacheable on small dual GPUs.
    """
    import torch

    if policy_support_logps.ndim != 3:
        raise ValueError("Top-k policy log-probabilities must have shape [2B, L, K]")
    if reference_support_logps.shape != policy_support_logps.shape:
        raise ValueError(
            "Reference top-k log-probabilities must match policy top-k log-probabilities"
        )
    if completion_mask.shape != policy_support_logps.shape[:2]:
        raise ValueError("Completion mask does not match top-k support positions")
    mask = completion_mask.to(torch.bool)
    if not torch.isfinite(policy_support_logps[mask]).all() or not torch.isfinite(
        reference_support_logps[mask]
    ).all():
        raise FloatingPointError("Non-finite top-k support log-probability")

    policy_logps = policy_support_logps.to(torch.float32)
    reference_logps = reference_support_logps.to(torch.float32)
    reference_probs = reference_logps.exp()
    policy_probs = policy_logps.exp()
    epsilon = torch.finfo(torch.float32).eps
    reference_top_mass = reference_probs.sum(dim=-1).clamp(min=0.0, max=1.0 - epsilon)
    policy_top_mass = policy_probs.sum(dim=-1).clamp(min=0.0, max=1.0 - epsilon)
    top_tokens = (reference_probs * (reference_logps - policy_logps)).sum(dim=-1)
    reference_remainder = (1.0 - reference_top_mass).clamp_min(epsilon)
    policy_remainder = (1.0 - policy_top_mass).clamp_min(epsilon)
    remainder = reference_remainder * (
        reference_remainder.log() - policy_remainder.log()
    )
    per_position = (top_tokens + remainder).clamp_min(0.0)
    return (per_position * mask).sum(dim=-1)


def packed_triplet_loss(
    anchor_ratios,
    anchor_mask,
    chosen_ratios,
    chosen_mask,
    rejected_ratios,
    rejected_mask,
    *,
    margin: float = 0.001,
):
    """TIDPO Eq. 14 using response-token ordinal alignment, as in the imported repo."""
    import torch
    import torch.nn.functional as functional

    sequences = []
    masks = []
    for values, valid in (
        (anchor_ratios, anchor_mask),
        (chosen_ratios, chosen_mask),
        (rejected_ratios, rejected_mask),
    ):
        packed = [values[row][valid[row].to(torch.bool)] for row in range(values.shape[0])]
        width = max(int(item.numel()) for item in packed)
        sequences.append(
            torch.stack([functional.pad(item, (0, width - item.numel())) for item in packed])
        )
        masks.append(
            torch.stack(
                [
                    functional.pad(
                        torch.ones_like(item, dtype=torch.bool), (0, width - item.numel())
                    )
                    for item in packed
                ]
            )
        )
    width = max(item.shape[1] for item in sequences)
    sequences = [functional.pad(item, (0, width - item.shape[1])) for item in sequences]
    masks = [functional.pad(item, (0, width - item.shape[1])) for item in masks]
    anchor, chosen, rejected = sequences
    anchor_valid, chosen_valid, rejected_valid = masks
    positive_mask = (anchor_valid & chosen_valid).to(anchor.dtype)
    negative_mask = (anchor_valid & rejected_valid).to(anchor.dtype)
    positive_distance = ((anchor - chosen).square() * positive_mask).sum(dim=-1)
    negative_distance = ((anchor - rejected).square() * negative_mask).sum(dim=-1)
    return functional.relu(positive_distance - negative_distance + float(margin)).mean()


def run_loss_self_tests() -> dict[str, Any]:
    import torch

    policy = torch.tensor(
        [
            [-0.2, -0.3, -0.4, 0.0],
            [-0.5, -0.2, -0.1, 0.0],
            [-0.6, -0.4, 0.0, 0.0],
            [-0.7, -0.2, -0.2, 0.0],
        ],
        requires_grad=True,
    )
    reference = torch.tensor(
        [
            [-0.3, -0.3, -0.4, 0.0],
            [-0.5, -0.3, -0.2, 0.0],
            [-0.5, -0.4, 0.0, 0.0],
            [-0.6, -0.2, -0.1, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    importances = torch.tensor(
        [
            [1.0, 4.0, 2.0, 0.0],
            [3.0, 1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.0],
        ]
    )
    weights = tidpo_importance_weights(importances, mask)
    expected_sums = mask.sum(-1).to(weights.dtype)
    if not torch.allclose(weights.sum(-1), expected_sums, atol=1e-6):
        raise AssertionError("TIDPO weights do not preserve mean-one scale")

    simpo = simpo_pair_loss(policy, mask)
    sampo_generator = torch.Generator().manual_seed(42)
    sampo = sampo_pair_loss(policy, reference, mask, generator=sampo_generator)
    reference_support = torch.log(
        torch.tensor([0.35, 0.25], dtype=torch.float32)
    ).view(1, 1, 2).expand(4, 4, 2)
    policy_support = torch.log(
        torch.tensor([0.30, 0.20], dtype=torch.float32)
    ).view(1, 1, 2).expand(4, 4, 2).clone().requires_grad_(True)
    position_kl = topk_bucket_position_kl(policy_support, reference_support, mask)
    vocabulary_generator = torch.Generator().manual_seed(7)
    reference_vocab_logps = torch.randn(
        4, 4, 11, generator=vocabulary_generator
    ).log_softmax(-1)
    policy_vocab_logps = torch.randn(
        4, 4, 11, generator=vocabulary_generator
    ).log_softmax(-1)
    reference_top_logps, reference_top_ids = reference_vocab_logps.topk(4, dim=-1)
    policy_on_reference_top = torch.gather(
        policy_vocab_logps, -1, reference_top_ids
    )
    projected = topk_bucket_position_kl(
        policy_on_reference_top, reference_top_logps, mask
    )
    exact = (
        reference_vocab_logps.exp() * (reference_vocab_logps - policy_vocab_logps)
    ).sum(-1)
    exact = (exact * mask).sum(-1)
    if torch.any(projected < -1e-6) or torch.any(projected > exact + 1e-5):
        raise AssertionError("Top-k bucket KL violated the KL lower-bound property")
    tidpo = tidpo_pair_loss(
        policy,
        reference,
        mask,
        weights,
        position_kl=position_kl,
    )
    ratios = policy - reference
    triplet = packed_triplet_loss(
        ratios[2:],
        mask[2:],
        ratios[:2],
        mask[:2],
        ratios[2:],
        mask[2:],
    )
    total = simpo[0] + sampo[0] + tidpo[0] + triplet
    total.backward()
    if policy.grad is None or not torch.isfinite(policy.grad).all() or policy.grad.abs().sum() == 0:
        raise AssertionError("Preference-suite losses did not produce finite non-zero gradients")
    if policy_support.grad is None or policy_support.grad.abs().sum() == 0:
        raise AssertionError("TIDPO position-KL approximation did not produce gradients")
    if float(triplet.detach()) <= 0.0:
        raise AssertionError("TIDPO triplet test did not exercise the active hinge branch")
    outputs = {
        "SimPO": float(simpo[0].detach()),
        "SamPO": float(sampo[0].detach()),
        "TIDPO": float(tidpo[0].detach()),
        "TIDPO_triplet": float(triplet.detach()),
        "TIDPO_topk_bucket_kl": position_kl.detach().tolist(),
        "TIDPO_projected_kl_below_exact": True,
        "SamPO_sampled_counts": sampo[-1].tolist(),
    }
    scalar_values = [outputs[key] for key in ("SimPO", "SamPO", "TIDPO", "TIDPO_triplet")]
    if not all(math.isfinite(value) for value in scalar_values):
        raise AssertionError(outputs)
    return outputs
