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
        emb_dim: int = 256
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
        self.edge_dropout = nn.Dropout(0.5)
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
        self.v_g, self.t_g = self.build_item_g()
        


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
            degree_matrix = torch.diag(degree)
            degree_safe = degree + 1e-8
            degree_inv = torch.diag(1.0 / degree_safe)
            score = degree_inv @ score
            dst, src = score.nonzero().t()
            edge_weights = score[dst, src]
            g = dgl.graph((src, dst)).to(self.device)
            g.edata['weight'] = edge_weights
            gs.append(g)
        return gs


    def func_train(self, item_emb, k):
        # self.item_g = self.item_g.to(self.device)
        
        # alphas = torch.softmax(self.alphas, dim=0)
        hs = []
        for feat in self.feats:
            h = self.gcn(feat)
            hs.append(h)
        h = torch.cat(hs, dim=1)
        feat_emb = self.activate(self.trans(h)) + self.gcn(item_emb)
        return feat_emb
    
    def 

    def reduce_func(self, nodes):
        return {"h": torch.sum(nodes.mailbox["m"], dim=1)/self.k}

    # def apply_func(self, nodes):
    #     return {"feat": nodes.data["feat"] + nodes.data["h"]}


    def func_test(self, user_emb, item_emb):
        user_emb = user_emb.weight
        item_emb = item_emb.weight
        # self.g = self.g.to(self.device)
        alphas = torch.softmax(self.alphas, dim=0)
        feats = self.activate(self.trans(torch.cat([self.feats[i].to(self.device) * alphas[i] for i in range(len(self.feats))], dim=1)))
        item_emb = item_emb + feats
        node_emb = torch.cat([user_emb, item_emb], dim=0)
        
        with self.g.local_scope():
            self.g.ndata["emb"] = node_emb
            self.g.update_all(fn.copy_u("emb", "m"), fn.mean("m", "h"))
            self.g.apply_nodes(self._conbin)
            emb = self.g.ndata["emb"]
        return emb
    def _conbin(self, nodes):
        return {"emb": nodes["emb"] + nodes["h"]}

        




    
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

        
