import torch


def _validate_inputs(scores: torch.Tensor, targets: torch.Tensor, k: int) -> None:
    if scores.dim() != 2 or targets.dim() != 2:
        raise ValueError("scores and targets must be 2D tensors: (batch_size, num_items)")
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have the same shape")
    if k <= 0 or k > scores.shape[1]:
        raise ValueError("k must be in the range [1, num_items]")


def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    # returns boolean mask of shape (batch, num_items) where True indicates item in top-k
    topk_indices = torch.topk(scores, k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    return mask


def precision_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    _validate_inputs(scores, targets, k)
    topk = _topk_mask(scores, k)
    hits = (targets.bool() & topk).sum(dim=1).float()
    precision = hits / float(k)
    return precision.mean()


def recall_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    _validate_inputs(scores, targets, k)
    topk = _topk_mask(scores, k)
    true_positives = (targets.bool() & topk).sum(dim=1).float()
    positives = targets.sum(dim=1).clamp_min(1).float()
    recall = true_positives / positives
    return recall.mean()


def hit_rate_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    _validate_inputs(scores, targets, k)
    topk = _topk_mask(scores, k)
    hits_per_user = (targets.bool() & topk).any(dim=1).float()
    return hits_per_user.mean()


def ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    _validate_inputs(scores, targets, k)
    # get ranks within top-k (sorted to compute DCG)
    topk = torch.topk(scores, k, dim=1, largest=True, sorted=True).indices
    batch_idx = torch.arange(scores.size(0), device=scores.device).unsqueeze(1).expand_as(topk)
    rel = targets[batch_idx, topk].float()
    positions = torch.arange(1, k + 1, device=scores.device).float()
    discounts = 1.0 / torch.log2(positions + 1.0)
    dcg = (rel * discounts).sum(dim=1)
    # ideal DCG
    ideal_k = torch.minimum(targets.sum(dim=1), torch.tensor(k, device=scores.device)).long()
    ideal_rel = torch.zeros_like(rel)
    # fill first ideal_k positions with 1
    for i in range(scores.size(0)):
        if ideal_k[i] > 0:
            ideal_rel[i, : ideal_k[i]] = 1.0
    idcg = (ideal_rel * discounts).sum(dim=1)
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
    return ndcg.mean()


def map_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    _validate_inputs(scores, targets, k)
    topk = torch.topk(scores, k, dim=1, largest=True, sorted=True).indices
    batch_idx = torch.arange(scores.size(0), device=scores.device).unsqueeze(1).expand_as(topk)
    rel = targets[batch_idx, topk].float()
    # precision at each position
    cum_rels = rel.cumsum(dim=1)
    positions = torch.arange(1, k + 1, device=scores.device).float()
    prec_at_pos = cum_rels / positions
    ap = (prec_at_pos * rel).sum(dim=1) / targets.sum(dim=1).clamp_min(1).float()
    return ap.mean()


def mrr_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    _validate_inputs(scores, targets, k)
    topk = torch.topk(scores, k, dim=1, largest=True, sorted=True).indices
    batch_idx = torch.arange(scores.size(0), device=scores.device).unsqueeze(1).expand_as(topk)
    rel = targets[batch_idx, topk].bool()
    # find first relevant position
    no_hit = (~rel).all(dim=1)
    # default rank = k+1 (will map to 0 reciprocal)
    first_pos = torch.full((scores.size(0),), k + 1, device=scores.device, dtype=torch.float32)
    # for users with any hit, compute first index
    has_hit_idx = (~no_hit).nonzero(as_tuple=False).flatten()
    if has_hit_idx.numel() > 0:
        first_hit_pos = rel[has_hit_idx].float().argmax(dim=1) + 1  # 1-based rank
        first_pos[has_hit_idx] = first_hit_pos.float()
    mrr = torch.where(first_pos <= k, 1.0 / first_pos, torch.zeros_like(first_pos))
    return mrr.mean()


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "ndcg_at_k",
    "map_at_k",
    "mrr_at_k",
]

def evaluate_all_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> dict:
    """Compute Top-K metrics in a single pass (computes Top-K once).

    Returns a dict with keys: precision, recall, hit_rate, ndcg, map, mrr.
    """
    _validate_inputs(scores, targets, k)

    # Precompute Top-K indices and relevance
    topk = torch.topk(scores, k, dim=1, largest=True, sorted=True).indices
    batch_idx = torch.arange(scores.size(0), device=scores.device).unsqueeze(1).expand_as(topk)
    rel = targets[batch_idx, topk].float()  # (B, K) in {0,1}

    # Precision@K
    hits = rel.sum(dim=1)
    precision = (hits / float(k)).mean()

    # Recall@K
    positives = targets.sum(dim=1).clamp_min(1).float()
    recall = (hits / positives).mean()

    # HitRate@K
    hit_rate = (hits > 0).float().mean()

    # NDCG@K
    positions = torch.arange(1, k + 1, device=scores.device).float()
    discounts = 1.0 / torch.log2(positions + 1.0)
    dcg = (rel * discounts).sum(dim=1)
    ideal_k = torch.minimum(targets.sum(dim=1), torch.tensor(k, device=scores.device)).long()
    ideal_rel = torch.zeros_like(rel)
    for i in range(scores.size(0)):
        if ideal_k[i] > 0:
            ideal_rel[i, : ideal_k[i]] = 1.0
    idcg = (ideal_rel * discounts).sum(dim=1)
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg)).mean()

    # MAP@K
    cum_rels = rel.cumsum(dim=1)
    prec_at_pos = cum_rels / positions
    ap = (prec_at_pos * rel).sum(dim=1) / positives
    mapk = ap.mean()

    # MRR@K
    rel_bool = rel.bool()
    no_hit = (~rel_bool).all(dim=1)
    first_pos = torch.full((scores.size(0),), k + 1, device=scores.device, dtype=torch.float32)
    has_hit_idx = (~no_hit).nonzero(as_tuple=False).flatten()
    if has_hit_idx.numel() > 0:
        first_hit_pos = rel_bool[has_hit_idx].float().argmax(dim=1) + 1
        first_pos[has_hit_idx] = first_hit_pos.float()
    mrr = torch.where(first_pos <= k, 1.0 / first_pos, torch.zeros_like(first_pos)).mean()

    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "ndcg": ndcg,
        "map": mapk,
        "mrr": mrr,
    }


