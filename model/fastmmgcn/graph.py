import torch.nn as nn 
import numpy as np
import torch
import dgl
import dgl.function as fn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class Graph(nn.Module):
    """Constructs and manages recommendation graphs with multi-modal features."""

    def __init__(
        self,
        num_users: int = 0,
        num_items: int = 0,
        device: str = "cpu",
        user_features: Optional[Dict[str, torch.Tensor]] = None, 
        item_features: Optional[Dict[str, torch.Tensor]] = None, 
        user_ks: Tuple[int] = (), 
        item_ks: Tuple[int] = (3, 3),
        emb_dim: int = 256,
        ks: Tuple[int] = (4, 2)
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
        self.user_features = user_features or {}
        self.item_features = item_features or {}
        self.user_ks = user_ks
        self.item_ks = item_ks
        self.emb_dim = emb_dim
        self.ks = ks


        self.edge_dropout = nn.Dropout(0.2)
        self.x_dropout = nn.Dropout(0.3)
        self.z_dropout = nn.Dropout(0.3)
        self.beta = 0.9
        self.alpha = 0.1


        l=0
        self.feats = []
        for k, feat in self.item_features.items():
            # self.item_features[k] = feat.to(self.device)
            l += feat.size(-1)
            self.feats.append(feat.to(self.device))
        
        # self.feats = torch.stack(self.feats, dim=0)

        self.item_feats_name = list(self.item_features.keys())
        self.user_feats_name = list(self.user_features.keys())

        self.weight_cache = "weight"
        
        self.g = None
        self.user_g = None
        self.item_g = None

        self.trans = nn.Sequential(
                        nn.Linear(l, 256),
                        nn.ReLU(),                   
                        nn.Linear(256, emb_dim)  
                    )
        self.activate = nn.ReLU()
        # self.alphas = nn.Parameter(torch.randn(len(self.item_features)))


    def compute_gamma(self, k):
        return np.power(self.beta, k) + self.alpha * np.sum([np.power(self.beta, i) for i in range(k)])

    
    def build_graph(
        self,
        interactions: List[Tuple[int, int, float]],
    ) -> dgl.DGLGraph:
        self.num_inter = len(interactions)
        # adjacency_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        # for u, v, w in interactions:
        #     adjacency_matrix[u][v] = 1

        # adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        

        inter = torch.tensor(interactions, dtype = torch.long)[: , :2].T.contiguous()
        inter[1] += self.num_users
        inter = torch.cat([inter, inter[[1, 0]]], dim = 1)
        self.g = dgl.graph((inter[0], inter[1])).to(self.device)
        self.item_gs = self.build_item_g()
        self.norm_adj(self.g)


        return self.g

    def build_item_g(self):
        gs = []
        for j, feat in enumerate(self.feats):
            dig = torch.sqrt(torch.sum(feat** 2, dim=1, keepdim=True))
            normalized_feat = feat / dig
            score = normalized_feat @ normalized_feat.T
            mask = torch.zeros_like(score, dtype=torch.bool, device=self.device)
            _ , k_index = torch.topk(score, self.item_ks[j])
            n, k = k_index.shape
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
            g.edata['weight'] = edge_weights
            gs.append(g)
        return gs


    def forward(self, user_emb, item_emb):
        user_emb = user_emb.weight
        item_emb = item_emb.weight
        node_emb = torch.cat([user_emb, item_emb], dim=0)
        hs = []
        for i,feat in enumerate(self.feats):
            h = self.gcn_ii(feat, i)

        h = torch.cat(hs, dim=1)
        emb = torch.cat(torch.zeros_like(user_emb, device=self.device), self.activate(self.trans(h))) + self.gcn(node_emb)
        
        return emb

    def norm_adj(self, g):

        CACHE_KEY = "weight"

        degs = g.in_degrees()
        src_norm = degs.pow(-0.5)
        dst_norm = src_norm

        with g.local_scope():
            g.ndata["src_norm"] = src_norm
            g.ndata["dst_norm"] = dst_norm
            g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", CACHE_KEY))
            gcn_weight = g.edata[CACHE_KEY]

        g.edata[CACHE_KEY] = gcn_weight
    
    def gcn_ii(self, feat, i):
        k = self.ks[i]
        g = self.item_gs[i]
        edge_weight = g.edata["weight"]
        dropped_edge_weight = self.edge_dropout(edge_weight)
        h0 = self.x_dropout(feat)
        h = h0

        with g.local_scope():
            g.edata["weight"] = dropped_edge_weight
            for _ in range(k):
                g.ndata["h"] = h
                g.update_all(fn.u_mul_e("h", "weight", "m"), fn.sum("m", "h"))
                h = g.ndata.pop("h")
                h = h * self.beta + h0 * self.alpha
        gamma = torch.tensor(self.compute_gamma(k)).float()
        h = h / gamma
        h = self.z_dropout(h)

        return h

    def gcn(self, feat):
        k = 2
        g = self.g
        edge_weight = g.edata["weight"]
        dropped_edge_weight = self.edge_dropout(edge_weight)
        h0 = self.x_dropout(feat)
        h = h0

        with g.local_scope():
            g.edata["weight"] = dropped_edge_weight
            for _ in range(k):
                g.ndata["h"] = h
                g.update_all(fn.u_mul_e("h", "weight", "m"), fn.sum("m", "h"))
                h = g.ndata.pop("h")
                h = h * self.beta + h0 * self.alpha
        gamma = torch.tensor(self.compute_gamma(k)).float()
        h = h / gamma
        h = self.z_dropout(h)

        return h

    
    # def move_to_device(self, device):
    #     self.g = self.g.to(device)
    #     self.user_g = self.user_g.to(device)
    #     self.item_g = self.item_g.to(device)
    #     return

    # def get_graph_statistics(self) -> Dict[str, Any]:
    #     """Get graph statistics."""
    #     return {
    #         "num_users": self.num_users,
    #         "num_items": self.num_items,
    #         "num_nodes": self.num_nodes,
    #         "num_interactions": self.num_interactions
    #     }

        
