import torch
import torch.nn.functional as F
from typing import Optional, Iterable


def bpr_loss(pos_scores: torch.Tensor,
             neg_scores: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bayesian Personalized Ranking (BPR) loss.

    Args:
        pos_scores: Tensor of shape (B,) or (B, 1) — scores for positive items.
        neg_scores: Tensor of shape (B, N) or broadcastable with pos_scores — scores for negative items.
        mask: Optional boolean/float mask with same shape as neg_scores to ignore some negatives.

    Returns:
        Scalar tensor (mean over batch and negatives).
    """
    # ensure shapes broadcast to (B, N)
    pos = pos_scores.unsqueeze(-1) if pos_scores.dim() == 1 else pos_scores
    x = pos - neg_scores
    loss = -F.logsigmoid(x)
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        denom = mask.to(loss.dtype).sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


def pairwise_hinge_loss(pos_scores: torch.Tensor,
                        neg_scores: torch.Tensor,
                        margin: float = 1.0,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pairwise hinge loss: max(0, margin - (pos - neg))."""
    pos = pos_scores.unsqueeze(-1) if pos_scores.dim() == 1 else pos_scores
    loss = F.relu(margin - (pos - neg_scores))
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        denom = mask.to(loss.dtype).sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


def bce_with_logits_loss(logits: torch.Tensor,
                         targets: torch.Tensor,
                         pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Binary cross-entropy with logits for multi-label setups.

    Args:
        logits: (B, D) raw scores.
        targets: (B, D) in {0,1}.
        pos_weight: Optional tensor of shape (D,) to reweight positive labels.
    """
    return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight, reduction="mean")


def cross_entropy_loss(logits: torch.Tensor,
                       target_indices: torch.Tensor) -> torch.Tensor:
    """Multi-class cross-entropy from logits.

    Args:
        logits: (B, C)
        target_indices: (B,) with values in [0, C)
    """
    return F.cross_entropy(logits, target_indices, reduction="mean")


def info_nce_loss(logits: torch.Tensor,
                  target_indices: Optional[torch.Tensor] = None,
                  temperature: float = 1.0) -> torch.Tensor:
    """InfoNCE loss from logits.

    Typical usage in recommendation: each row corresponds to a query and columns to
    candidate items (1 positive + k negatives). If target_indices is None, assumes
    positives are at column 0.
    """
    z = logits / max(temperature, 1e-12)
    if target_indices is None:
        target_indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(z, target_indices, reduction="mean")


def l2_regularization(parameters: Iterable[torch.nn.Parameter], weight: float) -> torch.Tensor:
    """L2 regularization over a collection of parameters."""
    if weight <= 0:
        return torch.tensor(0.0)
    total = torch.zeros((), device=next(iter(parameters)).device)
    for p in parameters:
        if p is not None:
            total = total + p.pow(2).sum()
    return weight * total


__all__ = [
    "bpr_loss",
    "pairwise_hinge_loss",
    "bce_with_logits_loss",
    "cross_entropy_loss",
    "info_nce_loss",
    "l2_regularization",
]


