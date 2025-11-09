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
        self.user_ks = user_ks, 
        self.item_ks = item_ks,
        self.emb_dim = emb_dim

        l=0
        self.feats = []
        for k, feat in self.item_features.items():
            self.item_features[k] = feat.to(self.device)
            l += feat.size(-1)
            self.feats.append(feat)
        
        # self.feats = torch.stack(self.feats, dim=0)

        self.item_feats_name = list(self.item_features.keys())
        self.user_feats_name = list(self.user_features.keys())

        self.weight_cache = "weight"
        
        self.g = None
        self.user_g = None
        self.item_g = None

        self.trans = nn.Linear(l, emb_dim)

        self.alphas = nn.Parameter(torch.randn(len(self.item_features)))




    
    def build_graph(
        self,
        interactions: List[Tuple[int, int, float]],
    ) -> dgl.DGLGraph:
        self.num_inter = len(interactions)
        adjacency_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        for u, v, w in interactions:
            adjacency_matrix[u][v] = 1

        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        

        inter = torch.tensor(interactions, dtype = torch.long)[: , :2].T.contiguous()
        inter[1] += self.num_users
        inter = torch.cat([inter, inter[[1, 0]]], dim = 1)
        self.g = dgl.graph((inter[0], inter[1]))

        double_adjacency_matrix_u = torch.matmul(adjacency_matrix, adjacency_matrix.T)
        u_src, u_dst = torch.nonzero(double_adjacency_matrix_u, as_tuple=True)
        double_adjacency_matrix_i = torch.matmul(adjacency_matrix.T, adjacency_matrix)
        i_src, i_dst = torch.nonzero(double_adjacency_matrix_i, as_tuple=True)

        self.user_g = dgl.graph((u_src, u_dst), num_nodes=self.num_users)
        # self.user_g.edata["times"] = double_adjacency_matrix_u[u_src, u_dst]
        self.item_g = dgl.graph((i_src, i_dst), num_nodes=self.num_items)
        # self.item_g.edata["times"] = double_adjacency_matrix_i[i_src, i_dst]

        # self.g = self.g.to(self.device)
        # self.user_g = self.user_g.to(self.device)
        # self.item_g = self.item_g.to(self.device)

        return self.g, self.user_g, self.item_g
    
    def creat_feature_weight(self):

        print("1")
        i=0
        j=0
        s_u = []
        s_i = []
        # self.user_g = self.user_g.to(self.device)
        # with self.user_g.local_scope():
        #     for modal_name, features in self.user_features.items():
        #         self.user_g.ndata[modal_name] = features
        #         weight = self._adj_norm("user", modal_name, self.user_ks[i])                
        #         s_u.append(weight)
        #         i += 1
        # self.user_g = self.user_g.to("cpu")

        self.item_g = self.item_g.to(self.device)
        with self.item_g.local_scope():
            for modal_name, features in self.item_features.items():
                self.item_g.ndata[modal_name] = features
                weight = self._adj_norm("item", modal_name, self.item_ks[j])
                s_i.append(weight)
                j+=1
        s_i = torch.stack(s_i, dim=1)
        self.item_g.edata["weight"] = s_i
        self.item_g = self.item_g.to("cpu")
    
        return
    
    def _adj_norm(self, graph_name, modal_name, k):
        self.cur_modal_name = modal_name
        self.k=k
        self.weight_key = f"{self.cur_modal_name}_weight"

        if graph_name == "user":
            
            self.user_g.apply_edges(self._edge_weight_fn)
            self.user_g.apply_edges(self._clip)
            self.user_g.update_all(fn.copy_e(self.weight_key, "m"), fn.sum("m", "in_weight"))
            self.user_g.apply_edges(self._norm)
            return self.user_g.edata[self.weight_key]
            

        else:
            print("adj_norm")
            self.item_g.apply_edges(self._edge_weight_fn)
            self.item_g.apply_edges(self._clip)
            self.item_g.update_all(fn.copy_e(self.weight_key, "m"), fn.sum("m", "in_weight"))
            self.item_g.apply_edges(self._norm)
            return self.item_g.edata[self.weight_key]

            


        
        
    def _edge_weight_fn(self, edges):
        # 1. 计算点积（原始score）
        dot_product = torch.sum(edges.src[self.cur_modal_name] * edges.dst[self.cur_modal_name], dim=1).squeeze()  # 形状：(num_edges,)
        print("dot succsses")
        # 2. 计算源节点和目标节点特征的L2范数（模长）
        norm_u = torch.norm(edges.src[self.cur_modal_name], dim=1)  # 源节点模长，形状：(num_edges,)
        norm_v = torch.norm(edges.dst[self.cur_modal_name], dim=1)  # 目标节点模长，形状：(num_edges,)
        
        # 3. 除以模长乘积（加epsilon避免除零错误）
        epsilon = 1e-8  # 微小值，防止分母为0
        score = torch.sigmoid(dot_product) / (norm_u * norm_v + epsilon)
        print("edge success")
        return {f"{self.cur_modal_name}_weight": torch.sigmoid(score)}
                 
    
    def _clip(self, edges):
        print("cliping")
        weights = edges.edata[self.weight_key]
        num_edges = weights.shape[0]

        if num_edges > self.k:

            mask = torch.zeros(num_edges, dtype=torch.bool)

            _, index = torch.topk(edges.edata[self.weight_key], k=self.k)

            mask[index] = True
            weights = weights * mask.float()

        return {self.weight_key: weights}
    
    def _norm(self, edges):
        return {self.weight_key: self.weight_key/(torch.sqrt(edges.src["in_weight"]+1e-8)*torch.sqrt(edges.dst["in_weight"]+1e-8))}


    def func_train(self, item_emb, k):
        self.item_g = self.item_g.to(self.device)
        alphas = torch.softmax(self.alphas, dim=0)
        feats = self.trans(torch.cat([feats[i] * alphas[i] for i in range(len(feats))], dim=1))
        self.k = k
        with self.item_g.local_scope():
            self.item_g.edata["weight"] = torch. sum(self.item_g.edata["weight"].view((self.num_inter, -1)) * alphas, dim=1)
            self.item_g.ndata["feat"] = feats + item_emb
            self.item_g.update_all(fn.u_mul_e("feat", "weight", "m"), self.reduce_func)
            feat_emb = self.item_g.ndata["h"]
        self.item_g.to("cpu")
        return feat_emb

    def reduce_func(self, nodes):
        return {"h": torch.sum(nodes.mailbox["m"], dim=1)/self.k}

    # def apply_func(self, nodes):
    #     return {"feat": nodes.data["feat"] + nodes.data["h"]}


    def func_test(self, user_emb, item_emb):
        user_emb = user_emb.weight
        item_emb = item_emb.weight
        self.g = self.g.to(self.device)
        alphas = torch.softmax(self.alphas, dim=0)
        feats = self.trans(torch.cat([feats[i] * alphas[i] for i in range(len(feats))], dim=1))
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

        
