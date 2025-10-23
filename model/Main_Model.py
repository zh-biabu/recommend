"""
Enhanced MMFCN model wrapper with proper integration for the training pipeline.
Handles multi-modal features and provides a unified interface.
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Dict, List, Tuple, Optional, Any

# Add model paths to sys.path for imports
from .mmgcn.graph_constructor import GraphConstructor
from .mmgcn.out_Layer import MMGCN

from .mig.mirf_gt import MIGGT
from .mig.mgdcf import MGDCF


class MMGCNModel(nn.Module):
    """Enhanced MMFCN model wrapper for graph-based recommendation."""
    
    def __init__(
        self,
        config,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize MMFCN model.
        
        Args:
            config: Configuration object
            num_users: Number of users
            num_items: Number of items
            user_features: User feature tensors
            item_features: Item feature tensors
        """
        super().__init__()
        
        self.config = config
        self.num_users = config.data.num_users
        self.num_items = config.data.num_items
        self.num_nodes = self.num_users + self.num_items
        
        # Store features
        self.user_features = user_features or {}
        self.item_features = item_features or {}
        self.feats = []

        for feat in self.user_features.values():
            self.feats.append(torch.cat([feat, torch.zeros(self.num_items,feat.size(1))]).to(config.system.device))
        for feat in self.item_features.values():
            self.feats.append(torch.cat([torch.zeros(self.num_users,feat.size(1)), feat]).to(config.system.device))

        self.linears = nn.ModuleList([
            nn.Linear(x.shape[1], config.model.emb_dim) if x.shape[1] != config.model.emb_dim
            else nn.Identity() 
            for x in self.feats
        ])
            
        self.ks= [5]*config.model.modal_num
        self.alphas = torch.randn(config.model.modal_num, requires_grad=True, device=config.system.device)
        self.emb = nn.Embedding(self.num_nodes, config.model.emb_dim, device=config.system.device)
        self.embs = self.emb.weight.unsqueeze(0).expand(config.model.modal_num, -1, -1)

        # Initialize MMGCN model
        self.mmgcn = MMGCN(
            modal_num=config.model.modal_num,
            emb_num=config.model.emb_dim,
            layer_num=config.model.layer_num,
            dropout=config.model.dropout
        )
        
        # Graph constructor
        self.graph_constructor = GraphConstructor(
            user_features=self.user_features,
            item_features=self.item_features,
            edge_weight_type=config.graph.edge_weight_type,
            add_self_loops=config.graph.add_self_loops,
            normalize_adj=config.graph.normalize_adj
        )

        self.emb = nn.Embedding(self.num_nodes, embedding_dim=config.model.emb_dim)

        
        
        # Final projection layer
        self.final_projection = nn.Linear(
            config.model.emb_dim,
            config.model.emb_dim
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.model.dropout)
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Cache for graph
        self._graph_cache = None
        self._graph_features_cache = None
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)
    
    def build_graph(self, interactions: List[Tuple[int, int, float]]):
        """Build the recommendation graph."""
        self._graph_cache = self.graph_constructor.build_graph(
            interactions, self.num_users, self.num_items
        )
        return self._graph_cache
    
    def get_graph(self):
        """Get the constructed graph."""
        if self._graph_cache is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        return self._graph_cache
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        if self._graph_cache is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        g = self.graph_constructor
        g.move_to_device(self.config.system.device)

        self.Xs = [linear(x) for linear, x in zip(self.linears, self.feats)]
        # try:
        node_embeddings = self.mmgcn(g, self.Xs, self.embs, self.ks, self.alphas)
        # print(node_embeddings)
        # print(input())
        # except Exception as e:
        #     print(f"MMGCN forward failed: {e}. Using fallback.")
            # node_embeddings = self._fallback_forward(g)
        node_embeddings = self.final_projection(node_embeddings)
        node_embeddings = self.dropout(node_embeddings)
        user_embeddings = node_embeddings[:self.num_users]
        item_embeddings = node_embeddings[self.num_users:]
        user_ids = batch.get('user_ids', torch.tensor([], dtype=torch.long))
        item_ids = batch.get('item_ids', torch.tensor([], dtype=torch.long))
        # 支持正负样本embedding输出
        if 'neg_items' in batch:
            neg_item_ids = batch['neg_items']
            # print(neg_item_ids)
            pos_user_emb = user_embeddings[user_ids]
            pos_item_emb = item_embeddings[item_ids]
            neg_item_emb = item_embeddings[neg_item_ids]  # shape: (N, neg_ratio, emb_dim)
            return {
                'user_embeddings': pos_user_emb,
                'pos_item_embeddings': pos_item_emb,
                'neg_item_embeddings': neg_item_emb,
                'embeddings': node_embeddings
            }
        # 仅正样本
        if len(user_ids) > 0 and len(item_ids) > 0:
            user_emb = user_embeddings[user_ids]
            item_emb = item_embeddings[item_ids]
            scores = torch.sum(user_emb * item_emb, dim=1)
        else:
            scores = torch.empty(0, dtype=torch.float32, device=node_embeddings.device)
        result = {'scores': scores}
        if return_embeddings:
            result['embeddings'] = node_embeddings
            result['user_embeddings'] = user_embeddings
            result['item_embeddings'] = item_embeddings
        return result
    
    def _prepare_modal_features(self) -> List[torch.Tensor]:
        """Prepare multi-modal features for MMGCN."""
        Xs = []
        same_feats=set()
        # Combine user and item features for each modality
        for modal_name in self.user_features.keys():
            user_feat = self.user_features[modal_name]
            item_feat = self.item_features.get(modal_name, torch.zeros_like(user_feat))
            
            # Concatenate user and item features
            modal_features = torch.cat([user_feat, item_feat], dim=0)
            Xs.append(modal_features)
            same_feats.add(modal_name)

        for modal_name in self.item_features.keys():
            if modal_name in same_feats:
                continue
            user_feat = self.user_features[modal_name]
            item_feat = self.item_features.get(modal_name, torch.zeros_like(user_feat))
            
            # Concatenate user and item features
            modal_features = torch.cat([user_feat, item_feat], dim=0)
            Xs.append(modal_features)
        
        # If no features available, use random features
        if not Xs:
            random_feat = torch.randn(self.num_nodes, self.config.model.emb_dim)
            Xs = [random_feat, random_feat]  # Two modalities
        
        return Xs
    
    def _prepare_embeddings(self) -> List[torch.Tensor]:
        """Prepare initial embeddings for MMGCN."""
        # Create initial embeddings (could be learnable parameters)
        initial_emb = torch.randn(self.num_nodes, self.config.model.emb_dim)
        return [initial_emb, initial_emb]  # Two modalities
    
    def _prepare_k_values(self) -> List[int]:
        """Prepare k values for graph aggregation."""
        return [2, 2]  # Two modalities, k=2 for each
    
    def _prepare_alpha_weights(self) -> torch.Tensor:
        """Prepare alpha weights for modality fusion."""
        # Equal weights for all modalities
        num_modalities = max(len(self.user_features), 2)
        return torch.ones(num_modalities) / num_modalities
    
    def _fallback_forward(self, g) -> torch.Tensor:
        """Fallback forward pass when MMGCN fails."""
        # Simple embedding lookup
        embeddings = torch.randn(self.num_nodes, self.config.model.emb_dim)
        return embeddings
    
    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get user and item embeddings."""
        if self._graph_cache is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        with torch.no_grad():
            # Create dummy batch for forward pass
            dummy_batch = {
                'user_ids': torch.tensor([0], dtype=torch.long),
                'item_ids': torch.tensor([self.num_users], dtype=torch.long)
            }
            
            result = self.forward(dummy_batch, return_embeddings=True)
            return result['user_embeddings'], result['item_embeddings']
    
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Predict scores for user-item pairs."""
        batch = {
            'user_ids': user_ids,
            'item_ids': item_ids
        }
        
        with torch.no_grad():
            result = self.forward(batch)
            return result['scores']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MMFCN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_nodes': self.num_nodes,
            'embedding_dim': self.config.model.emb_dim,
            'num_modalities': self.config.model.modal_num,
            'user_features': list(self.user_features.keys()),
            'item_features': list(self.item_features.keys())
        }


class MIG(nn.Module):
    def __init__(
        self,
        config,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize MMFCN model.
        
        Args:
            config: Configuration object
            num_users: Number of users
            num_items: Number of items
            user_features: User feature tensors
            item_features: Item feature tensors
        """
        super().__init__()

        self.config = config
        self.num_users = config.data.num_users
        self.num_items = config.data.num_items
        self.num_nodes = self.num_users + self.num_items
        self.device = config.system.device
        self.use_rp = True
        self.embedding_size = config.model.emb_dim

        self.user_features = user_features or {}
        self.item_features = item_features or {}

        self.v_feat = item_features["image_feat"]
        self.t_feat = item_features["text_feat"]
        if self.use_rp:
            self.v_feat = self.random_project(self.v_feat, self.t_feat.size(-1))

        self.v_feat = self.v_feat.to(self.device)
        self.t_feat = self.t_feat.to(self.device)
        self.user_embeddings = torch.randn((self.num_users, config.model.emb_dim), device=self.device)
        self.item_embeddings = torch.randn((self.num_items, config.model.emb_dim), device=self.device)
 
        self.k_e = 4
        self.k_t = 2
        self.k_v = 1
        self.alpha, self.beta = 0.1, 0.9
        self.input_feat_drop_rate=0.3
        self.feat_drop_rate=0.3
        self.user_x_drop_rate=0.3
        self.item_x_drop_rate=0.3
        self.edge_drop_rate=0.2
        self.z_drop_rate=0.2
        self.use_rp=True
        self.use_item_emb=False
        self.num_clusters=5
        self.num_samples=10
        self.feat_hidden_units = 512

        


        self.model =  MIGGT(
        # k=config.k,

        k_e=self.k_e,
        k_t=self.k_t,
        k_v=self.k_v,

        alpha=self.alpha, 
        beta=self.beta, 

        input_feat_drop_rate=self.input_feat_drop_rate,
        feat_drop_rate=self.feat_drop_rate,
        user_x_drop_rate=self.user_x_drop_rate,
        item_x_drop_rate=self.item_x_drop_rate, 
        edge_drop_rate=self.edge_drop_rate, 
        z_drop_rate=self.z_drop_rate,
        user_in_channels=self.embedding_size,
        item_v_in_channels=self.v_feat.size(-1),
        item_v_hidden_channels_list=[self.feat_hidden_units, self.embedding_size], 
        item_t_in_channels=self.t_feat.size(-1), 
        item_t_hidden_channels_list=[self.feat_hidden_units, self.embedding_size], 

        bn=True,
        num_clusters=self.num_clusters,
        num_samples=self.num_samples
    ).to(self.device)

        self._graph_cache = None


    def random_project(self, x, units):
        
        weight_shape = (x.shape[-1], units)
        weight = torch.randn(weight_shape)
        h = x @ weight
        h = self.l2_normalize(h, dim=-1)
        return h

    def l2_normalize(self, x, dim=-1):
        return x / (torch.norm(x, dim=dim, keepdim=True) + 1e-8)
    
    def build_graph(self, interactions: List[Tuple[int, int, float]]):
        """Build the recommendation graph."""
        interactions = torch.tensor(interactions,dtype=torch.long)[:,:-1]
        self._graph_cache = MGDCF.build_sorted_homo_graph(
            interactions, self.num_users, self.num_items
        )
        return self._graph_cache
    
    def forward(self,batch, return_embeddings=True):
        result = {}
        self._graph_cache = self._graph_cache.to(self.device)
        print(self._graph_cache.device, self.v_feat.device, self.t_feat.device)
        virtual_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h = self.model(self._graph_cache, self.user_embeddings, self.v_feat, self.t_feat, 
                                                                                item_embeddings=self.item_embeddings if self.use_item_emb else None, 
                                                                                return_all=True)
        result["user_h"] = virtual_h[:self.num_users]
        result["item_h"] = virtual_h[self.num_users:]

        result["user_emb_h"] = emb_h[:self.num_users]
        result["item_emb_h"] = emb_h[self.num_users:]

        result["user_t_h"] = t_h[:self.num_users]
        result["item_t_h"] = t_h[self.num_users:]

        if v_h is not None:
            result["user_v_h"] = v_h[:self.num_users]
            result["item_v_h"] = v_h[self.num_users:]
        else:
            result["user_v_h"] = None
            result["item_v_h"] = None
        
        result["encoded_t"] = encoded_t
        result["encoded_v"] = encoded_v
        result["z_memory_h"] = z_memory_h

        return result



class ModelFactory:
    """Factory for creating recommendation models."""
    
    @staticmethod
    def create_Model(
        config,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """
        Create a recommendation model.
        
        Args:
            model_name: Name of the model to create
            config: Configuration object
            num_users: Number of users
            num_items: Number of items
            user_features: User feature tensors
            item_features: Item feature tensors
            
        Returns:
            Initialized model
        """
        if config.model.model_name.lower() == 'mmgcn':
            return MMGCNModel(
                config=config,
                user_features=user_features,
                item_features=item_features
            )
        if config.model.model_name.lower() == "mig":
            return MIG(
                config=config,
                user_features=user_features,
                item_features=item_features
            )
        else:
            raise ValueError(f"Unknown model: {config.model.model_name}")


if __name__ == "__main__":
    # Test the model
    import sys
    sys.path.append(r"F:\project")
    from recommend.config import get_config
    
    config = get_config('baby')
    model = MMGCNModel(
        config=config,
        num_users=1000,
        num_items=500,
        user_features={'image': torch.randn(1000, 64), 'text': torch.randn(1000, 128)},
        item_features={'image': torch.randn(500, 64), 'text': torch.randn(500, 128)}
    )
    
    # Build graph
    interactions = [(i % 1000, i % 500, 1.0) for i in range(10000)]
    graph = model.build_graph(interactions)
    
    # Test forward pass
    batch = {
        'user_ids': torch.tensor([0, 1, 2]),
        'item_ids': torch.tensor([0, 1, 2])  # Add offset
    }
    
    result = model.forward(batch)
    print("Model test successful!")
    print(f"Output shape: {result['scores']}")
    print(f"Model info: {model.get_model_info()}")
