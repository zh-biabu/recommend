"""
Graph construction module for multi-modal recommendation systems.
Handles bipartite graph construction with multi-modal features and edge weights.
"""

from code import interact
import torch
import numpy as np
import dgl
import dgl.function as fn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class GraphConstructor:
    """Constructs and manages recommendation graphs with multi-modal features."""
    
    def __init__(
        self,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None,
        edge_weight_type: str = "uniform",
        feature: str = "text",
        add_self_loops: bool = True,
        normalize_adj: bool = True,
        max_neighbors: int = 50
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
        self.user_features = user_features or {}
        self.item_features = item_features or {}
        self.edge_weight_type = edge_weight_type
        self.add_self_loops = add_self_loops
        self.normalize_adj = normalize_adj
        self.max_neighbors = max_neighbors
        
        self.graph = None
        self.num_users = 0
        self.num_items = 0
        self.num_nodes = 0
    
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
        edge_weights = []

        max_rating = 0 
        
        for user_id, item_id, rating in interactions:
            # Add user-item edge
            user_indices.append(user_id)
            item_indices.append(item_id + num_users)  # Offset items
            edge_weights.append(rating)
            
            # Add item-user edge (bipartite)
            user_indices.append(item_id + num_users)
            item_indices.append(user_id)
            edge_weights.append(rating)

            max_rating = max(rating, max_rating)
        
        # Add self-loops if requested
        if self.add_self_loops:
            for i in range(self.num_nodes):
                user_indices.append(i)
                item_indices.append(i)
                edge_weights.append(max_rating)
        
        # Create graph
        src = torch.tensor(user_indices, dtype=torch.long)
        dst = torch.tensor(item_indices, dtype=torch.long)
        
        self.graph = dgl.graph((src, dst), num_nodes=self.num_nodes)
        self.graph.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
        
        # Add node features
        self._add_node_features()
        
        # Compute edge weights based on features
        # self._compute_edge_weights()
        
        # Normalize if requested
        # if self.normalize_adj:
        #     self._normalize_graph()
        
        return self.graph
    
    def move_to_device(self,device="cpu"):
        self.graph=self.graph.to(device)
    
    def creat_feature_weight(self, feature=None):
        # Compute edge weights based on features
        self._compute_edge_weights(feature=feature)

        # Normalize if requested
        if self.normalize_adj:
            self._normalize_graph(feature=feature)

    def _add_node_features(self):
        for modal_name, features in self.user_features.items():
            self.graph.ndata[modal_name] = torch.cat([features,torch.zeros([self.num_items, features.shape[1]], dtype=features.dtype, device=features.device)], dim = 0)
        
        for modal_name, features in self.item_features.items():
            self.graph.ndata[modal_name] = torch.cat([torch.zeros([self.num_users, features.shape[1]], dtype=features.dtype, device=features.device), features], dim = 0)
    
    def _compute_edge_weights(self, feature=None):
        """Compute edge weights based on node features."""
        if self.edge_weight_type == "uniform" or feature == None:
            return  # Already set to uniform weights
        
        # Get features for edge weight computation
        with self.graph.local_scope():
            if self.edge_weight_type == "cosine":
                # Cosine similarity
                self.graph.ndata["norm"] = torch.norm(self.graph.ndata[feature], dim=1) + 1e-12
                self.graph.apply_edges(lambda edges: {"similarity": torch.sum(edges.src[feature]*edges.dst[feature], dim=1)/(edges.src["norm"]*edges.dst["norm"])})
                weights = torch.clamp(self.graph.edata["similarity"], 0, 1)  # Ensure non-negative
            elif self.edge_weight_type == "dot":
                # Dot product
                self.graph.apply_edges(lambda edges: {"similarity": torch.sum(edges.src[feature]*edges.dst[feature], dim=1)})
                weights = torch.sigmoid(self.graph.edata["similarity"])  # Normalize to [0, 1]
            else:
                raise ValueError(f"Unknown edge weight type: {self.edge_weight_type}")
        
        self.graph.edata[f'{feature}_weight'] = weights
    
    def _normalize_graph(self, feature=None):
        """Normalize graph adjacency matrix."""
        # Compute in-degrees
        # in_degrees = self.graph.in_degrees().float()
        # in_degrees[in_degrees == 0] = 1  # Avoid division by zero
        
        # Normalize edge weights
        # src_nodes = self.graph.edges()[0]
        # normalized_weights = self.graph.edata['weight'] / torch.sqrt(in_degrees[src_nodes])
        # self.graph.edata['weight'] = normalized_weights

        # 标准化权重: w_ij / (sqrt(sum_in_i) * sqrt(sum_in_j))
        # compute in_weight = sqrt(sum of incoming edge weights)
        if self.edge_weight_type == "uniform" or feature == None:
            self.weight = "weight"
        else:
            self.weight = f"{feature}_weight"


        self.graph.update_all(fn.copy_e(self.weight, "m"), fn.sum("m", "in_weight"))
        self.graph.ndata["in_weight"] = torch.clamp(self.graph.ndata["in_weight"], min=1e-12).sqrt()
        # apply symmetric normalization (this must be outside of local_scope to persist)
        self.graph.apply_edges(self._norm)
    # remove temporary node feature
        del self.graph.ndata["in_weight"]

    def _norm(self, edges):
        return {
            self.weight: edges.data[self.weight] / (edges.src["in_weight"] * edges.dst["in_weight"]) 
        }
    
    def aggrate(self, node_x, k, alpha):
        h0 = node_x
        self.graph.ndata["h"] = node_x
        with self.graph.local_scope():
            for _ in range(k):
                self.graph.update_all(fn.u_mul_e("h", "weight", "m"), fn.sum("m", "h"))
                self.graph.ndata["h"] = alpha*h0 + (1-alpha)*self.graph.ndata["h"]
            hk = self.graph.ndata["h"]
        return hk

    def agg_func(self, node_x, k, f, dropout, active):
        h0 = node_x
        self.graph.ndata["h"] = node_x
        with self.graph.local_scope():
            for _ in range(k):
                self.graph.ndata["h"] = dropout(active(f(self.graph.ndata["h"])))
                self.graph.update_all(fn.u_mul_e("h", "weight", "m"), fn.sum("m", "h"))
                # self.graph.ndata["h"] = alpha*h0 + (1-alpha)*self.graph.ndata["h"]
            hk = self.graph.ndata["h"]
        return hk
    
    def get_node_features(self, modal_name: Optional[str] = None) -> torch.Tensor:
        """Get node features for a specific modality or combined features."""
        if modal_name:
            if modal_name.startswith('user_'):
                return self.graph.ndata[modal_name][:self.num_users]
            elif modal_name.startswith('item_'):
                return self.graph.ndata[modal_name][self.num_users:]
            else:
                return self.graph.ndata[modal_name]
        else:
            return self.graph.ndata.keys()
    
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