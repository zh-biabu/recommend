import torch.nn as nn 
import numpy as np
import torch
import dgl
import dgl.function as fn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class Graph:
    """Constructs and manages recommendation graphs with multi-modal features."""

    def __init__(
        self,
        num_users: int = 0,
        num_items: int = 0
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
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items

        self.weight_cache = "weight"
        
        self.g = None
        self.user_g = None
        self.item_g = None

    
    def build_graph(
        self,
        interactions: List[Tuple[int, int, float]],
    ) -> dgl.DGLGraph:

        adjacency_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        for u, v, w in interactions:
            adjacency_matrix[u][v] = 1

        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.long)
        
        
        inter = torch.tensor(interactions, dtype = torch.long)[: , :2].T.contiguous()
        inter[1] += self.num_users
        inter = torch.cat([inter, inter[[1, 0]]], dim = 1)
        self.g = dgl.graph(inter)

        double_adjacency_matrix_u = torch.matmul(adjacency_matrix, adjacency_matrix.T)
        u_src, u_dst = torch.nonzero(double_adjacency_matrix_u, as_tuple=True)
        double_adjacency_matrix_i = torch.matmul(adjacency_matrix.T, adjacency_matrix)
        i_src, i_dst = torch.nonzero(double_adjacency_matrix_i, as_tuple=True)

        self.user_g = dgl.graph((u_src, u_dst), num_nodes=self.num_users)
        self.user_g.edata["times"] = double_adjacency_matrix_u[u_src, u_dst]
        self.item_g = dgl.graph((i_src, i_dst), num_nodes=self.num_items)
        self.item_g.edata["times"] = double_adjacency_matrix_i[i_src, i_dst]

        return self.g, self.user_g, self.item_g
    
    def creat_feature_weight(self, user_features: Dict[str, torch.Tensor], item_features: Dict[str, torch.Tensor], user_ks, item_ks):
        self.user_features = user_features
        self.item_features = item_features

        self.user_modals_name = []
        self.item_modals_name = []

        self.user_ks = user_ks
        self.item_ks = item_ks

        for modal_name, features in user_features.items():
            self.user_g.ndata[modal_name] = features
            self.user_modals_name.append(modal_name)
        for modal_name, features in item_features.items():
            self.item_g.ndata[modal_name] = features
            self.item_modals_name.append(modal_name)
        
        for i, modal_name in enumerate(self.user_modals_name):
            self._adj_norm("user", modal_name, self.user_ks[i])
        for i, modal_name in enumerate(self.item_modals_name):
            self._adj_norm("item", modal_name, self.item_ks[i])
        return
    
    def _adj_norm(self, graph_name, modal_name, k):
        self.cur_modal_name = modal_name
        self.k=k
        self.weight_key = f"{self.cur_modal_name}_weight"

        if graph_name == "user":
            
            self.user_g.apply_edges(self._edge_weight_fn)
            self.user_g.apply_edges(self._clip)
            self.user_g.update_all(
                fn.copy_u(self.weight_key, "m"), 
                fn.sum("m", "in_edge_sum") 
            )
            reversed_graph = dgl.reverse(self.user_g, copy_ndata=True)
            reversed_graph.update_all(
                fn.copy_u(self.weight_key, "m"), 
                fn.sum("m", "out_edge_sum") 
            )
            self.user_g.ndata["out_edge_sum"] = reversed_graph.ndata["out_edge_sum"]
            del reversed_graph
            self.user_g.apply_edges(self._norm)
            del self.user_g.ndata["in_edge_sum"]
            del self.user_g.ndata["out_edge_sum"]
        else:
            
            self.item_g.apply_edges(self._edge_weight_fn)
            self.item_g.apply_edges(self._clip)
            self.item_g.update_all(
                fn.copy_u(self.weight_key, "m"), 
                fn.sum("m", "in_edge_sum") 
            )
            reversed_graph = dgl.reverse(self.item_g, copy_ndata=True)
            reversed_graph.update_all(
                fn.copy_u(self.weight_key, "m"), 
                fn.sum("m", "out_edge_sum") 
            )
            self.item_g.ndata["out_edge_sum"] = reversed_graph.ndata["out_edge_sum"]
            del reversed_graph
            self.item_g.apply_edges(self._norm)
            del self.item_g.ndata["in_edge_sum"]
            del self.item_g.ndata["out_edge_sum"]
            


        
        
    def _edge_weight_fn(self, edges):
        # 获取源节点和目标节点的特征
        u = edges.src[self.cur_modal_name]  # 源节点特征，形状：(num_edges, d)
        v = edges.dst[self.cur_modal_name]  # 目标节点特征，形状：(num_edges, d)
        
        # 1. 计算点积（原始score）
        dot_product = torch.sum(u * v, dim=1).squeeze()  # 形状：(num_edges,)
        
        # 2. 计算源节点和目标节点特征的L2范数（模长）
        norm_u = torch.norm(u, dim=1)  # 源节点模长，形状：(num_edges,)
        norm_v = torch.norm(v, dim=1)  # 目标节点模长，形状：(num_edges,)
        
        # 3. 除以模长乘积（加epsilon避免除零错误）
        epsilon = 1e-8  # 微小值，防止分母为0
        score = dot_product / (norm_u * norm_v + epsilon)
        
        return {f"{self.cur_modal_name}_weight": torch.sigmoid(score)}  
                 
    
    def _clip(self, edges):

        weights = edges.edata[self.weight_key]
        num_edges = weights.shape[0]

        if num_edges > self.k:

            mask = torch.zeros(num_edges, dtype=torch.bool)

            _, index = torch.topk(edges.edata[self.weight_key], k=self.k)

            mask[index] = True
            weights = weights * mask.float()

        return {self.weight_key: weights}

    def _norm(self, edges):

        weights = edges.edata[self.weight_key]
        D_u = edges.src["out_edge_sum"]
        D_v = edges.dst["in_edge_sum"]
        epsilon = 1e-8
        weights = weights / torch.sqrt((D_u+ epsilon) * (D_v+ epsilon))

        return {self.weight_key: weights}

    def func_train(self, item_emb, feats_name, alphas, k):
        feats = []
        eweights = []
        alphas = torch.softmax(alphas)
        self.k = k
        
        for i,name in enumerate(feats_name):
            feats.append(self.item_g.ndata[name] * alphas[i])
            eweights.append(self.item_g.ndata[f"{name}_weight"] * alphas[i])
        feats_emb = torch.cat(feats, dim=1)
        eweight = sum(eweights)
        with self.item_g.local_scope():
            self.item_g.edata["weight"] = eweight
            self.item_g.ndata["feat"] = feats_emb
            self.item_g.update_all(fn.u_mul_e("feat", "m"), self.reduce_func, self.apply_func)
            feat_emb = self.item_g.ndata["feat"]
        return feat_emb

    def reduce_func(self, nodes):
        return {"h": torch.sum(nodes.mailbox["m"], dim=1)/self.k}

    def apply_func(self, nodes):
        return {"feat": nodes.data["feat"] + nodes.data["h"]}


    def func_test(self, user_emb, item_emb, feats_name, alphas, w):
        user_emb = user_emb.weight
        item_emb = item_emb.weight
        feats = []
        # node_emb = torch.cat([user_emb, item_emb], dim=0)
        for i,name in enumerate(feats_name):
            feats.append(self.item_g.ndata[name] * alphas[i])
        feats_emb = torch.matmul(torch.cat(feats, dim=1), w)
        item_emb = item_emb + feats_emb
        node_emb = torch.cat([user_emb, item_emb], dim=0)
        
        with self.g.local_scope():
            self.g.ndata["emb"] = node_emb
            self.g.update_all(fn.copy_u("emb", "m"), fn.sum("m", "h"))
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

        
