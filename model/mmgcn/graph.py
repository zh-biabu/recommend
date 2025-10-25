"""
Graph construction module for multi-modal recommendation systems.
Handles bipartite graph construction with multi-modal features and edge weights.
"""

from code import interact
from hmac import new
import torch
import torch.nn as nn 
import numpy as np
import dgl
import dgl.function as fn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class Graph:
    """Constructs and manages recommendation graphs with multi-modal features."""
    
    def __init__(
        self,
        add_self_loops: bool = True,
        normalize_adj: bool = True,
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
        self.add_self_loops = add_self_loops
        self.normalize_adj = normalize_adj

        self.weight_cache = "weight"
        
        self.g = None


    
    def build_graph(
        self,
        interactions: List[Tuple[int, int, float]],
        num_users: int,
        num_items: int
    ) -> dgl.DGLGraph:
        """
        Build bipartite graph from interactions.
        
        Args:
            interactions: List of (user_id, item_id, rating) tuples
            num_users: Number of users
            num_items: Number of items
            
        Returns:
            Constructed DGL graph
        """
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.num_interactions = len(interactions)
        
        # Create edge lists
        user_indices = []
        item_indices = []
        
        for user_id, item_id, rating in interactions:
            # Add user-item edge
            user_indices.append(user_id)
            item_indices.append(item_id + num_users)  # Offset items
            
            # Add item-user edge (bipartite)
            user_indices.append(item_id + num_users)
            item_indices.append(user_id)
        
        # Add self-loops if requested
        if self.add_self_loops:
            for i in range(self.num_nodes):
                user_indices.append(i)
                item_indices.append(i)
        
        # Create graph
        src = torch.tensor(user_indices, dtype=torch.long)
        dst = torch.tensor(item_indices, dtype=torch.long)
        
        self.g = dgl.graph((src, dst), num_nodes=self.num_nodes)
        
        return self.g

    def move_to_device(self, device):
        self.g = self.g.to(device)
    
    def func(self, feat, w):
        self.w = w
        # print(self.w.device, feat.device)
        with self.g.local_scope():
            self.g.ndata["feat"] = feat 
            self.g.update_all(self.message, fn.mean("h", "feat"))
            new_feat = self.g.ndata["feat"]
        return new_feat


    def message(self, edges):
        return {'h': torch.matmul(edges.src['feat'], self.w)}




    # @torch.no_grad()
    # def norm_adj(self, g):
        
    #     degs = self.g.in_degrees()
    #     src_norm = degs.pow(-0.5)
    #     dst_norm = src_norm

    #     with g.local_scope():
    #         g.ndata["src_norm"] = src_norm
    #         g.ndata["dst_norm"] = dst_norm
    #         g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", CACHE_KEY))
    #         gcn_weight = g.edata[CACHE_KEY]

    #     g.edata[CACHE_KEY] = gcn_weight
    
    # def move_to_device(self,device="cpu"):
    #     self.graph=self.graph.to(device)
    

    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if self.graph is None:
            return {}
        
        stats = {
            'num_nodes': self.graph.num_nodes(),
            'num_edges': self.graph.num_edges(),
            'num_users': self.num_users,
            'num_items': self.num_items,
            'edge_weight_stats': {
                'mean': self.graph.edata[self.weight].mean().item(),
                'std': self.graph.edata[self.weight].std().item(),
                'min': self.graph.edata[self.weight].min().item(),
                'max': self.graph.edata[self.weight].max().item()
            }
        }
        
        return stats
    
    def save_graph(self, filepath: str):
        """Save graph to file."""
        if self.graph is None:
            raise ValueError("No graph to save")
        
        dgl.save_graphs(filepath, [self.graph])
    
    def load_graph(self, filepath: str):
        """Load graph from file."""
        graphs, _ = dgl.load_graphs(filepath)
        self.graph = graphs[0]
        self.num_nodes = self.graph.num_nodes()
        
        # Infer num_users and num_items from features
        if 'user_features' in self.graph.ndata:
            self.num_users = self.graph.ndata['user_features'].shape[0]
            self.num_items = self.num_nodes - self.num_users