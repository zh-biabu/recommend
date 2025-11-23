import torch.nn as nn 
import numpy as np
import torch
import dgl
import dgl.function as fn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from .gcn import II_GCN,IU_GCN

class Graph(nn.Module):
    """Constructs and manages recommendation graphs with multi-modal features."""

    def __init__(
        self,
        num_users: int = 0,
        num_items: int = 0,
        device: str = "cpu",
        v_feat: Optional[torch.Tensor] = None, 
        t_feat: Optional[torch.Tensor] = None, 
        v_k: int = 3,
        t_k: int = 3,
        emb_dim: int = 256,
        gcn_v_k: int = 4,
        gcn_t_k: int = 4,
        k: int = 2,
        edge_drop_rate: float = 0.2,
        feat_drop_rate: float = 0.1,
        x_drop_rate: float = 0.3,
        z_drop_rate: float = 0.3,
        alpha: float = 0.9,
        beta: float = 0.1,
        hidden_unit: int = 256
        ):
        """
        Initialize graph constructor.
        
        Args:
            user_features: Dictionary of user feature tensors
            item_features: Dictionary of item feature tensors
            edge_weight_type: Type of edge weight computation ("cosine", "dot", "uniform")
            add_self_loops: Whether to add self-loops
            normalize_adj: Whether to normalize adjacency matrix
            max_neighbors: Maximum number of neighbors to keep
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.device = device
        self.emb_dim = emb_dim
        self.weight_cache = "weight"
        self.input_feat_dropout = nn.Dropout(feat_drop_rate)
        self.v_k = v_k
        self.t_k = t_k

        self.v_feat = v_feat
        self.t_feat = t_feat

        self.v_ffn = nn.Sequential(
            nn.Linear(v_feat.size(1), hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim)
            )

        self.t_ffn = nn.Sequential(
            nn.Linear(t_feat.size(1), hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim)
            )

        self.v_gcn = II_GCN(
            gcn_v_k,
            alpha,
            beta,
            edge_drop_rate,
            x_drop_rate, 
            z_drop_rate,
            self.weight_cache,
            )
        self.t_gcn = II_GCN(
            gcn_t_k,
            alpha,
            beta,
            edge_drop_rate,
            x_drop_rate, 
            z_drop_rate,
            self.weight_cache,
            )
        self.iu_gcn = IU_GCN(
            k,
            edge_drop_rate,
            x_drop_rate, 
            z_drop_rate,
            self.weight_cache,
        )
        self.activate = nn.ReLU()
    
    def build_graph(
        self,
        interactions: List[Tuple[int, int, float]],
    ) -> dgl.DGLGraph:
        self.num_inter = len(interactions)

        inter = torch.tensor(interactions, dtype = torch.long)[: , :2].T.contiguous()
        inter[1] += self.num_users
        inter = torch.cat([inter, inter[[1, 0]], torch.arange(self.num_nodes).unsqueeze(0).repeat(2, 1)], dim = 1)
        self.g = dgl.graph((inter[0], inter[1])).to(self.device)
        self.v_g = self.build_item_g(self.v_feat, self.v_k)
        self.t_g = self.build_item_g(self.t_feat, self.t_k)
        return self.g

    def build_item_g(self, feat, k):
        dig = torch.sqrt(torch.sum(feat** 2, dim=1, keepdim=True))
        normalized_feat = feat / dig
        score = normalized_feat @ normalized_feat.T
        mask = torch.zeros_like(score, dtype=torch.bool, device=self.device)
        _ , k_index = torch.topk(score, k)
        n = k_index.size(0)
        row_indices = torch.arange(n).unsqueeze(1).repeat(1, k)
        mask[row_indices, k_index] = True
        score = score * mask
        degree = score.sum(dim=1)
        degree_safe = degree + 1e-8
        degree_inv = torch.diag(1.0 / degree_safe)
        score = degree_inv @ score
        dst, src = score.nonzero().t()
        edge_weights = score[dst, src]
        g = dgl.graph((src, dst)).to(self.device)
        g.edata[self.weight_cache] = edge_weights
        return g

    def creat_graph_weight(self):
        self.norm_adj()

    def norm_adj(self):

        degs = self.g.in_degrees()
        src_norm = degs.pow(-0.5)
        dst_norm = src_norm

        with self.g.local_scope():
            self.g.ndata["src_norm"] = src_norm
            self.g.ndata["dst_norm"] = dst_norm
            self.g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", self.weight_cache))
            gcn_weight = self.g.edata[self.weight_cache]

        self.g.edata[self.weight_cache] = gcn_weight

    def forward(self, user_emb, item_emb):
        v_feat = self.input_feat_dropout(self.v_feat)
        t_feat = self.input_feat_dropout(self.t_feat)
        
        encode_v = self.v_ffn(v_feat)
        encode_t = self.t_ffn(t_feat)

        v_h = self.v_gcn(encode_v, self.v_g)
        t_h = self.t_gcn(encode_t, self.t_g)

        node_emb = torch.cat([user_emb, item_emb], dim=0)
        node_h = self.iu_gcn(node_emb, self.g)

        combine_i_h = v_h + t_h

        node_h[self.num_users:] += combine_i_h

        return node_h





