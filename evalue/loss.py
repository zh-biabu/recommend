import torch
import torch.nn.functional as F
from typing import Optional, Iterable
import numpy as np


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


def compute_info_bpr_loss(a_embeddings, b_embeddings, pos_edges, neg_items, reduction='mean', hard_negs=None):

    if isinstance(pos_edges, list):
        pos_edges = np.array(pos_edges)

    device = a_embeddings.device

    a_indices = pos_edges[:, 0]
    b_indices = pos_edges[:, 1]
    num_pos_edges = pos_edges.size(0)

    embedded_a = a_embeddings[a_indices]
    embedded_b = b_embeddings[b_indices]
    embedded_neg_b = b_embeddings[neg_items]
    

    embedded_combined_b = torch.cat([embedded_b.unsqueeze(1), embedded_neg_b], 1)

    logits = (embedded_combined_b @ embedded_a.unsqueeze(-1)).squeeze(-1)

    info_bpr_loss = F.cross_entropy(logits, torch.zeros([num_pos_edges], dtype=torch.int64).to(device), reduction=reduction)

    return info_bpr_loss

def compute_l2_loss(params):
    """
    Compute l2 loss for a list of parameters/tensors
    """
    l2_loss = 0.0
    for param in params:
        l2_loss += param.pow(2).sum() * 0.5
    return l2_loss


def mig_loss_func(outputs, batch):

    user_h = outputs["user_embeddings"]
    item_h = outputs["item_embeddings"]
    z_memory_h = outputs["z_memory_h"]
    user_ids = batch.get('user_ids', torch.tensor([], dtype=torch.long))
    item_ids = batch.get('item_ids', torch.tensor([], dtype=torch.long))
    neg_items = batch.get('neg_items', torch.tensor([], dtype=torch.long))
    batch = torch.cat([user_ids.unsqueeze(1), item_ids.unsqueeze(1)],dim=1)
    num_users = user_h.size(0)
    


    mf_losses = compute_info_bpr_loss(user_h, item_h, batch, neg_items, reduction="none")
    l2_loss = compute_l2_loss([user_h, item_h])

    loss = mf_losses.sum() + l2_loss * 1e-5
    pos_user_h = user_h[batch[:, 0]]
    pos_z_memory_h = z_memory_h[batch[:, 1] + num_users]  
    unsmooth_logits = (pos_user_h.unsqueeze(1) @ pos_z_memory_h.permute(0, 2, 1)).squeeze(1)
    unsmooth_loss = F.cross_entropy(unsmooth_logits, torch.zeros([batch.size(0)], dtype=torch.long).to(unsmooth_logits.device), reduction="none").sum()
    loss = loss + unsmooth_loss
    return loss

def mmgcn_loss(outputs, batch):
    for v in outputs.values():
        v.to("cpu")
    user_tensor = batch.get('user_ids', torch.tensor([], dtype=torch.long))
    item_tensor = batch.get('item_ids', torch.tensor([], dtype=torch.long))
    neg_tensor = batch.get('neg_items', torch.tensor([], dtype=torch.long))
    user_h = outputs["user_embeddings"]
    item_h = outputs["item_embeddings"]
    id_embedding = outputs["id_embedding"]
    user_score = user_h[user_tensor]
    item_score = item_h[item_tensor]
    neg_score = item_h[torch.cat(neg_tensor, dim=0)]
    score = torch.sum(user_score*item_score, dim=1)
    neg_score = torch.sum(user_score*neg_score, dim=1)
    loss = -torch.mean(torch.log(torch.sigmoid(score - neg_score)))
    reg_embedding_loss = (id_embedding[user_tensor]**2 + id_embedding[item_tensor]**2).mean()
    # for preference in outputs["pres"]:
    reg_embedding_loss += (outputs["pres"][0]**2).mean()
    print(reg_embedding_loss)
    reg_loss =  1e-5 * (reg_embedding_loss)
    return loss+reg_loss



__all__ = [
    "bpr_loss",
    "pairwise_hinge_loss",
    "bce_with_logits_loss",
    "cross_entropy_loss",
    "info_nce_loss",
    "l2_regularization",
    "compute_info_bpr_loss",
    "compute_l2_loss",
    "mig_loss_func",
    "mmgcn_loss"
]


