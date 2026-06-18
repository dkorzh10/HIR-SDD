"""
Focal loss for binary (and optionally multiclass) classification.
Down-weights easy examples and focuses training on hard examples (e.g. real vs fake).
Ref: Lin et al., "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal cross-entropy for next-token prediction (e.g. causal LM completion tokens).
    Down-weights easy token predictions by (1 - p_t)^gamma.

    Args:
        logits: (N, vocab_size) unnormalized scores (e.g. flattened from batch*seq).
        labels: (N,) long tensor of target token ids; use ignore_index for masked positions.
        gamma: focusing parameter; higher = more down-weight on easy examples.
        ignore_index: label value to ignore (e.g. -100).
        reduction: 'mean' | 'sum' | 'none'. Mean/sum over valid (non-ignored) positions only.

    Returns:
        Scalar (or per-element if reduction='none') over valid positions only.
    """
    if logits.dim() != 2 or labels.dim() != 1:
        raise ValueError("focal_cross_entropy expects logits (N, V) and labels (N,)")
    mask = labels != ignore_index
    if mask.sum() == 0:
        return logits.sum() * 0.0  # return 0 with same device/dtype

    log_probs = F.log_softmax(logits, dim=-1)
    ce_per_token = F.nll_loss(log_probs, labels, reduction="none", ignore_index=ignore_index)
    # Safe labels for gather (ignore_index would be out of bounds)
    labels_safe = labels.clone()
    labels_safe[~mask] = 0
    probs = log_probs.exp()
    p_t = probs.gather(1, labels_safe.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
    focal_weight = (1 - p_t).pow(gamma)
    loss_per_token = focal_weight * ce_per_token

    if reduction == "none":
        return loss_per_token
    if reduction == "mean":
        return loss_per_token[mask].sum() / mask.sum().clamp(min=1)
    return loss_per_token[mask].sum()


def focal_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal loss for binary classification (replaces BCE with logits).
    Easy examples (high confidence correct) get down-weighted by (1 - p_t)^gamma.

    Args:
        logits: (N,) or (N, 1) unnormalized scores.
        targets: (N,) float in [0, 1] (e.g. is_bonafide).
        gamma: focusing parameter; higher = more down-weight on easy examples. Default 2.0.
        alpha: weight for positive class in [0, 1]; None = no class weighting.
        reduction: 'mean' | 'sum' | 'none'.

    Returns:
        Scalar (or per-element if reduction='none').
    """
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    targets = targets.to(logits.dtype).to(logits.device)
    if targets.dim() > 1:
        targets = targets.squeeze(-1)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    # p_t = prob of correct class: p if y=1 else (1-p)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t).clamp(min=1e-8).pow(gamma)

    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = alpha_t * focal_weight

    loss = focal_weight * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class FocalBCELoss(nn.Module):
    """
    Module wrapper for focal binary cross-entropy with configurable gamma and alpha.
    Use when you want a stateful loss (e.g. gamma/alpha from config).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return focal_binary_cross_entropy_with_logits(
            logits,
            targets,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
        )
