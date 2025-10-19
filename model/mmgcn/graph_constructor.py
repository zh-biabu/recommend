"""
Graph construction module for multi-modal recommendation systems.
Handles bipartite graph construction with multi-modal features and edge weights.
"""

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
        
        # Create edge lists
        user_indices = []
        item_indices = []
        edge_weights = []
        
        for user_id, item_id, rating in interactions:
            # Add user-item edge
            user_indices.append(user_id)
            item_indices.append(item_id + num_users)  # Offset items
            edge_weights.append(rating)
            
            # Add item-user edge (bipartite)
            user_indices.append(item_id + num_users)
            item_indices.append(user_id)
            edge_weights.append(rating)
        
        # Add self-loops if requested
        if self.add_self_loops:
            for i in range(self.num_nodes):
                user_indices.append(i)
                item_indices.append(i)
                edge_weights.append(5.0)
        
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
    # def _add_node_features(self):
    #     """Add multi-modal features to nodes."""
    #     # Initialize feature matrices
    #     feature_dim = 0
        
    #     # Determine feature dimension
    #     for features in self.user_features.values():
    #         feature_dim = max(feature_dim, features.shape[1])
    #     for features in self.item_features.values():
    #         feature_dim = max(feature_dim, features.shape[1])
        
    #     if feature_dim == 0:
    #         # No features available, use one-hot encoding
    #         feature_dim = self.num_nodes
    #         user_features = torch.eye(self.num_users, self.num_nodes)
    #         item_features = torch.eye(self.num_items, self.num_nodes)
    #     else:
    #         # Combine features
    #         user_features = self._combine_features(self.user_features, self.num_users, feature_dim)
    #         item_features = self._combine_features(self.item_features, self.num_items, feature_dim)
        
    #     # Concatenate user and item features
    #     all_features = torch.cat([user_features, item_features], dim=0)
    #     self.graph.ndata['features'] = all_features
        
    #     # Add individual modal features if available
    #     for modal_name, features in self.user_features.items():
    #         padded_features = torch.zeros(self.num_users, features.shape[1])
    #         padded_features[:features.shape[0]] = features
    #         self.graph.ndata[f'user_{modal_name}'] = padded_features
        
    #     for modal_name, features in self.item_features.items():
    #         padded_features = torch.zeros(self.num_items, features.shape[1])
    #         padded_features[:features.shape[0]] = features
    #         self.graph.ndata[f'item_{modal_name}'] = padded_features
    
    # def _combine_features(self, features_dict: Dict[str, torch.Tensor], num_entities: int, target_dim: int) -> torch.Tensor:
    #     """Combine multiple feature modalities."""
    #     if not features_dict:
    #         return torch.zeros(num_entities, target_dim)
        
    #     combined_features = []
    #     for modal_name, features in features_dict.items():
    #         # Pad or truncate to target dimension
    #         if features.shape[1] < target_dim:
    #             padding = torch.zeros(features.shape[0], target_dim - features.shape[1])
    #             features = torch.cat([features, padding], dim=1)
    #         elif features.shape[1] > target_dim:
    #             features = features[:, :target_dim]
            
    #         # Pad to num_entities
    #         if features.shape[0] < num_entities:
    #             padding = torch.zeros(num_entities - features.shape[0], target_dim)
    #             features = torch.cat([features, padding], dim=0)
            
    #         combined_features.append(features)
        
    #     # Average the features
    #     return torch.stack(combined_features, dim=0).mean(dim=0)
    
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


class GraphManager:
    """Manages multiple graphs and provides utilities for graph operations."""
    
    def __init__(self):
        self.graphs = {}
        self.constructors = {}
    
    def add_graph(
        self,
        name: str,
        interactions: List[Tuple[int, int, float]],
        num_users: int,
        num_items: int,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ):
        """Add a new graph to the manager."""
        constructor = GraphConstructor(
            user_features=user_features,
            item_features=item_features,
            **kwargs
        )
        
        graph = constructor.build_graph(interactions, num_users, num_items)
        
        self.graphs[name] = graph
        self.constructors[name] = constructor
    
    def get_graph(self, name: str) -> dgl.DGLGraph:
        """Get a graph by name."""
        if name not in self.graphs:
            raise KeyError(f"Graph '{name}' not found")
        return self.graphs[name]
    
    def get_all_graphs(self) -> Dict[str, dgl.DGLGraph]:
        """Get all graphs."""
        return self.graphs
    
    def remove_graph(self, name: str):
        """Remove a graph from the manager."""
        if name in self.graphs:
            del self.graphs[name]
            del self.constructors[name]
    
    def clear(self):
        """Clear all graphs."""
        self.graphs.clear()
        self.constructors.clear()


if __name__ == "__main__":
    # Test graph construction
    import torch
    
    # Create sample data
    num_users, num_items = 100, 50
    interactions = [
        (i % num_users, i % num_items, 1.0)
        for i in range(1000)
    ]
    
    # Create features
    user_features = {
        'image': torch.randn(num_users, 64),
        'text': torch.randn(num_users, 128)
    }
    item_features = {
        'image': torch.randn(num_items, 64),
        'text': torch.randn(num_items, 128)
    }
    
    # Build graph
    constructor = GraphConstructor(
        user_features=user_features,
        item_features=item_features,
        edge_weight_type="cosine",
        feature="text",
        normalize_adj=True
    )
    
    graph = constructor.build_graph(interactions, num_users, num_items)
    
    print("Graph statistics:")
    stats = constructor.get_graph_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"Graph nodes: {graph.num_nodes()}")
    print(f"Graph edges: {graph.num_edges()}")
    print(f"Node features shape: {graph.ndata.keys()}")
