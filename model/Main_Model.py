"""
Enhanced MMFCN model wrapper with proper integration for the training pipeline.
Handles multi-modal features and provides a unified interface.
"""

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Add model paths to sys.path for imports
# from .test.graph_constructor import GraphConstructor
# from .test.out_Layer import TEST

from .mig.mirf_gt import MIGGT
from .mig.mgdcf import MGDCF

# from .mmgcn.graph import Graph
# from .mmgcn.net import Net

from .mmgcn_rec.net import Net_rec

# from .fastmmgcn.graph import Graph

from .sgrec.graph import Graph


class TESTModel(nn.Module):
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
        self.modal_num = config.model.modal_num
        self.weight_feature = config.graph.weight_feature
        
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
            
        self.ks= [5] * self.modal_num
        self.alphas = torch.randn(self.modal_num, requires_grad=True, device=config.system.device)
        self.emb = nn.Embedding(self.num_nodes, config.model.emb_dim, device=config.system.device)
        self.embs = self.emb.weight.unsqueeze(0).expand(self.modal_num, -1, -1)

        # Initialize MMGCN model
        self.test = TEST(
            modal_num=self.modal_num,
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
    
    def creat_feature_weight(self):
        for feature in self.weight_feature:
            self.graph_constructor.creat_feature_weight(feature = feature)
    
    def get_graph(self):
        """Get the constructed graph."""
        if self._graph_cache is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        return self._graph_cache
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._graph_cache is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        g = self.graph_constructor
        g.move_to_device(self.config.system.device)

        self.Xs = [linear(x) for linear, x in zip(self.linears, self.feats)]
        # try:
        node_embeddings = self.test(g, self.Xs, self.embs, self.ks, self.alphas)
        # print(node_embeddings)
        # print(input())
        # except Exception as e:
        #     print(f"MMGCN forward failed: {e}. Using fallback.")
            # node_embeddings = self._fallback_forward(g)
        node_embeddings = self.final_projection(node_embeddings)
        node_embeddings = self.dropout(node_embeddings)
        user_embeddings = node_embeddings[:self.num_users]
        item_embeddings = node_embeddings[self.num_users:]

        result = {}
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
            'model_name': 'TEST',
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
        mig_gt
        """
        super().__init__()

        self.config = config
        self.num_users = config.data.num_users
        self.num_items = config.data.num_items
        self.num_nodes = self.num_users + self.num_items
        self.use_rp = True
        self.embedding_size = config.model.emb_dim
        self.device = config.system.device

        self.user_features = user_features or {}
        self.item_features = item_features or {}

        self.v_feat = item_features["image_feat"]
        self.t_feat = item_features["text_feat"]
        if self.use_rp:
            self.v_feat = self.random_project(self.v_feat, self.t_feat.size(-1))

        self.v_feat = self.v_feat.to(self.device)
        self.t_feat  = self.t_feat.to(self.device)
        self.user_embeddings = np.random.randn(self.num_users, self.embedding_size) / np.sqrt(self.embedding_size)
        self.user_embeddings = torch.tensor(self.user_embeddings, dtype=torch.float32, requires_grad=True, device=self.device)
        self.item_embeddings = np.random.randn(self.num_items, self.embedding_size) / np.sqrt(self.embedding_size)
        self.item_embeddings = torch.tensor(self.item_embeddings, dtype=torch.float32, requires_grad=True, device=self.device)
 
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
        self.feat_hidden_units = 64


        


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
        ).to(self.device)
        return self._graph_cache

    def creat_feature_weight(self):
        MGDCF.norm_adj(self._graph_cache)
    
    def forward(self,batch):
        result = {}
        virtual_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h = self.model(self._graph_cache, self.user_embeddings, self.v_feat, self.t_feat, 
                                                                                item_embeddings=self.item_embeddings if self.use_item_emb else None, 
                                                                                return_all=True)
        result["user_embeddings"] = virtual_h[:self.num_users]
        result["item_embeddings"] = virtual_h[self.num_users:]

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

    def loss_func(self, outputs, batch):
        user_h = outputs["user_embeddings"]
        item_h = outputs["item_embeddings"]
        z_memory_h = outputs["z_memory_h"]
        user_ids = batch.get('user_ids', torch.tensor([], dtype=torch.long))
        item_ids = batch.get('item_ids', torch.tensor([], dtype=torch.long))
        neg_items = batch.get('neg_items', torch.tensor([], dtype=torch.long))
        batch = torch.cat([user_ids.unsqueeze(1), item_ids.unsqueeze(1)],dim=1)
        num_users = user_h.size(0)
        


        mf_losses = self.compute_info_bpr_loss(user_h, item_h, batch, neg_items, reduction="none")
        l2_loss = self.compute_l2_loss([user_h, item_h])
        # print(l2_loss, mf_losses.mean())
        loss = mf_losses.sum() + l2_loss * 1e-5
        pos_user_h = user_h[batch[:, 0]]
        pos_z_memory_h = z_memory_h[batch[:, 1] + num_users]  
        unsmooth_logits = (pos_user_h.unsqueeze(1) @ pos_z_memory_h.permute(0, 2, 1)).squeeze(1)
        unsmooth_loss = F.cross_entropy(unsmooth_logits, torch.zeros([batch.size(0)], dtype=torch.long).to(unsmooth_logits.device), reduction="none").sum()
        loss = loss + unsmooth_loss
        return loss

    def compute_info_bpr_loss(self, a_embeddings, b_embeddings, pos_edges, neg_items, reduction='mean', hard_negs=None):

        if isinstance(pos_edges, list):
            pos_edges = np.array(pos_edges)

        device = a_embeddings.device

        a_indices = pos_edges[:, 0]
        b_indices = pos_edges[:, 1]
        num_pos_edges = pos_edges.size(0)

        embedded_a = a_embeddings[a_indices]
        embedded_b = b_embeddings[b_indices]
        embedded_neg_b = b_embeddings[neg_items]
        # print(embedded_a.shape, embedded_b.shape, embedded_neg_b.shape)

        embedded_combined_b = torch.cat([embedded_b.unsqueeze(1), embedded_neg_b], 1)

        logits = (embedded_combined_b @ embedded_a.unsqueeze(-1)).squeeze(-1)

        info_bpr_loss = F.cross_entropy(logits, torch.zeros([num_pos_edges], dtype=torch.int64).to(device), reduction=reduction)

        return info_bpr_loss

    def compute_l2_loss(self, params):
        """
        Compute l2 loss for a list of parameters/tensors
        """
        l2_loss = 0.0
        for param in params:
            l2_loss += param.pow(2).sum() * 0.5
        return l2_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MIG',
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

class MMGCN(nn.Module):
    """
    MMGCN
    """
    def __init__(
        self,
        config,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        self.config = config
        self.num_users = config.data.num_users
        self.num_items = config.data.num_items
        self.num_nodes = self.num_users + self.num_items
        self.modal_num = config.model.modal_num
        self.device = config.system.device
        self.hidden_dim = config.model.hidden_dim
        self.emb_dim = config.model.emb_dim
        self.concat = config.model.concat
        self.k = config.model.k

        self.user_features = user_features or {}
        self.item_features = item_features or {}

        self.node_emb = nn.init.xavier_normal_(torch.randn((self.num_nodes, self.emb_dim), dtype=torch.float32, requires_grad=True)).to(self.device)

        self.dim_feats = []
        self.feats =[]
        for feat in item_features.values():
            self.dim_feats.append(feat.size(1))
            self.feats.append(feat.to(self.device))

        self.model = Net(
            modal_num = self.modal_num, 
            dim_feats = self.dim_feats, 
            hidden_dim = self.hidden_dim, 
            emb_dim = self.emb_dim, 
            num_users = self.num_users, 
            num_items = self.num_items, 
            concat = self.concat, 
            k=self.k,
            device = self.device
        )

        self._initialize_parameters()


        self.graph = Graph(
            add_self_loops=config.graph.add_self_loops,
            normalize_adj=config.graph.normalize_adj
        )


        self._graph_cache = None
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)

    def build_graph(self, interactions):
        self._graph_cache = self.graph.build_graph(interactions, self.num_users, self.num_items)
        return self._graph_cache

    def creat_feature_weight(self):
        self.graph.move_to_device(self.device)
        return

    def forward(self, batch):
        for name, p in self.named_parameters():
            if torch.isnan(p).any():
                print(f"参数 {name} 存在 NaN，形状: {p.shape}")
                raise Exception("参数包含 NaN")
        result = {}
        emb, pres = self.model(self.feats, self.node_emb, self.graph)

        result["user_embeddings"] = emb[: self.num_users]
        result["item_embeddings"] = emb[self.num_users: ]
        result["id_embeddings"] = self.node_emb
        result["pres"] = pres
        return result

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


class MMGCN_rec(nn.Module):
    """
    MMGCN_rec
    """
    def __init__(
        self,
        config,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        self.config = config
        self.num_users = config.data.num_users
        self.num_items = config.data.num_items
        self.num_nodes = self.num_users + self.num_items
        self.modal_num = config.model.modal_num
        self.device = config.system.device
        self.hidden_dim = config.model.hidden_dim
        self.emb_dim = config.model.emb_dim
        self.concat = config.model.concat
        self.k = config.model.k

        self.user_features = user_features or {}
        self.item_features = item_features or {}

        self.node_emb = nn.init.xavier_normal_(torch.rand((self.num_nodes, self.emb_dim), requires_grad=True)).to(self.device)

        self.dim_feats = []
        self.feats =[]
        for feat in item_features.values():
            self.dim_feats.append(feat.size(1))
            self.feats.append(feat.to(self.device))
        
        self.model = Net_rec(
            # num_users, num_items, emb_dim, dim_feats, device
            dim_feats = self.dim_feats, 
            # hidden_dim = self.hidden_dim, 
            emb_dim = self.emb_dim, 
            num_users = self.num_users, 
            num_items = self.num_items, 
            # concat = self.concat, 
            # k=self.k,
            device = self.device
        )

        # self._initialize_parameters()


        # self.graph = Graph(
        #     add_self_loops=config.graph.add_self_loops,
        #     normalize_adj=config.graph.normalize_adj
        # )


        self._graph_cache = None
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)

    def build_graph(self, interactions):
        # self._graph_cache = self.graph.build_graph(interactions, self.num_users, self.num_items)
        edge_index = torch.tensor(interactions, dtype = torch.long)[: , :2].T.contiguous().to(self.device)
        edge_index[1] += self.num_users
        self.edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim = 1)
        return None

    def creat_feature_weight(self):
        # self.graph.move_to_device(self.device)
        return

    def forward(self, batch):
        result = {}
        emb, pres = self.model(self.edge_index, self.feats, self.node_emb)

        result["user_embeddings"] = emb[: self.num_users]
        result["item_embeddings"] = emb[self.num_users: ]
        result["id_embeddings"] = self.node_emb
        result["pres"] = pres
        return result

    def loss_func(self,outputs, batch):
        batch_users = batch.get('user_ids', torch.tensor([], dtype=torch.long))
        pos_items = batch.get('item_ids', torch.tensor([], dtype=torch.long))
        neg_items = batch.get('neg_items', torch.tensor([], dtype=torch.long)).reshape(-1)
        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)
        user_h = outputs["user_embeddings"]
        item_h = outputs["item_embeddings"]
        id_embedding = outputs["id_embeddings"]
        user_emb = user_h[user_tensor]
        item_emb = item_h[item_tensor]
        score = torch.sum(user_emb*item_emb, dim=1).view((-1,2))
        device = score.device
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, torch.tensor([[1.0], [-1.0]]).to(device)))))
        reg_embedding_loss = (id_embedding[user_tensor]**2 + id_embedding[item_tensor]**2).mean()
        # for preference in outputs["pres"]:
        reg_embedding_loss += (outputs["pres"]**2).mean()
        reg_loss =  0 * (reg_embedding_loss)
        # print(reg_loss)
        return loss + reg_loss

    
    
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

class FastMMGCN(nn.Module):
    def __init__(
        self,        
        config,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        
        self.config = config
        self.user_features = user_features or {}
        self.item_features = item_features or {}
        self.num_users = config.data.num_users
        self.num_items = config.data.num_items
        self.emb_dim = config.model.emb_dim
        self.device = config.system.device

        self.user_ks = config.graph.user_ks
        self.item_ks = config.graph.item_ks
        self.ks = config.graph.ks
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)
        
        self.graph = Graph(
            self.num_users, 
            self.num_items, 
            self.device, 
            self.user_features, 
            self.item_features, 
            self.user_ks, 
            self.item_ks, 
            self.emb_dim,
            self.ks
            )

        self._initialize_parameters()

    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)


    def build_graph(self, interactions):
        self.g = self.graph.build_graph(interactions)
        return self.g
        
    
    def creat_feature_weight(self):
        # self.graph.creat_feature_weight()
        return


    def forward(self, batch):
        result = {}
        emb = self.graph(self.user_emb, self.item_emb)
        result["user_embeddings"] = emb[: self.num_users]
        result["item_embeddings"] = emb[self.num_users: ]

        return result

    def loss_func(self, result, batch):

        user_emb, item_emb = result["user_embeddings"], result["item_embeddings"]
        batch_users = batch.get('user_ids', torch.tensor([], dtype=torch.long))
        pos_items = batch.get('item_ids', torch.tensor([], dtype=torch.long))
        neg_items = batch.get('neg_items', torch.tensor([], dtype=torch.long)).reshape(-1)
        users = user_emb[batch_users]
        items = item_emb[pos_items]
        negs = item_emb[neg_items]
        pos_score = torch.sum(users * items, dim=1)
        neg_score = torch.sum(users * negs, dim=1)
        assert not torch.isnan(pos_score).any(), "pos_score contains NaN"
        assert not torch.isinf(pos_score).any(), "pos_score contains Inf"
        assert not torch.isnan(neg_score).any(), "neg_score contains NaN"
        assert not torch.isinf(neg_score).any(), "neg_score contains Inf"
        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))

        return loss

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
            # 'num_nodes': self.num_nodes,
            'embedding_dim': self.config.model.emb_dim,
            'num_modalities': self.config.model.modal_num,
            'user_features': list(self.user_features.keys()),
            'item_features': list(self.item_features.keys())
        }



class SGrec(nn.Module):
    def __init__(
        self,        
        config,
        user_features: Optional[Dict[str, torch.Tensor]] = None,
        item_features: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        
        self.config = config
        self.user_features = user_features or {}
        self.item_features = item_features or {}
        self.num_users = config.data.num_users
        self.num_items = config.data.num_items
        self.emb_dim = config.model.emb_dim
        self.device = config.system.device
        self.v_feat = self.item_features["image_feat"].to(self.device)
        self.t_feat = self.item_features["text_feat"].to(self.device)
        self.k = config.model.k
        self.edge_drop_rate = config.model.edge_drop_rate
        self.feat_drop_rate = config.model.feat_drop_rate
        self.x_drop_rate = config.model.x_drop_rate
        self.z_drop_rate = config.model.z_drop_rate
        self.hidden_unit = config.model.hidden_dim
        self.v_layer = config.model.v_layer
        self.t_layer = config.model.t_layer
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)

        self.reg_weight = config.training.weight_decay

        self.graph = Graph(
            self.num_users,
            self.num_items,
            self.device,
            self.v_feat,
            self.t_feat,
            self.emb_dim,
            self.k,
            self.edge_drop_rate,
            self.feat_drop_rate,
            self.x_drop_rate,
            self.z_drop_rate,
            self.hidden_unit,
            self.v_layer,
            self.t_layer
            )

        self._initialize_parameters()

    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)


    def build_graph(self, interactions):
        self.g = self.graph.build_graph(interactions)
        return self.g
        
    
    def creat_feature_weight(self):
        self.graph.creat_graph_weight()
        return


    def forward(self, batch):
        result = {}
        emb = self.graph(self.user_emb.weight, self.item_emb.weight)
        result["user_embeddings"] = emb[: self.num_users]
        result["item_embeddings"] = emb[self.num_users: ]
        result["ori_u_emb"] = self.user_emb.weight
        result["ori_i_emb"] = self.item_emb.weight

        return result

    def loss_func(self, result, batch):
        
        ori_u = result["ori_u_emb"]
        ori_i = result["ori_i_emb"]
        user_emb, item_emb = result["user_embeddings"], result["item_embeddings"]
        batch_users = batch.get('user_ids', torch.tensor([], dtype=torch.long))
        pos_items = batch.get('item_ids', torch.tensor([], dtype=torch.long))
        neg_items = batch.get('neg_items', torch.tensor([], dtype=torch.long)).reshape(-1)
        users = user_emb[batch_users]
        items = item_emb[pos_items]
        negs = item_emb[neg_items]
        sample_items = item_emb[torch.randint(low=0, high=self.num_items, size=(batch_users.size(0), 20))]
        pos_score = torch.sum(users * items, dim=1)
        neg_score = torch.sum(users * negs, dim=1)
        assert not torch.isnan(pos_score).any(), "pos_score contains NaN"
        assert not torch.isinf(pos_score).any(), "pos_score contains Inf"
        assert not torch.isnan(neg_score).any(), "neg_score contains NaN"
        assert not torch.isinf(neg_score).any(), "neg_score contains Inf"
        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))

        unsmooth_logits = torch.cat([pos_score.unsqueeze(1), torch.sum(users.unsqueeze(1) * sample_items, dim=2)], dim=1)

        unsmooth_loss = F.cross_entropy(unsmooth_logits, torch.zeros([batch_users.size(0)], dtype=torch.long).to(unsmooth_logits.device), reduction="none").mean()
        
        reg_loss = torch.mean(ori_u**2) + torch.mean(ori_i**2)

        return loss + self.reg_weight * reg_loss

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.config.model.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_users': self.num_users,
            'num_items': self.num_items,
            # 'num_nodes': self.num_nodes,
            'embedding_dim': self.config.model.emb_dim,
            'num_modalities': self.config.model.modal_num,
            'user_features': list(self.user_features.keys()),
            'item_features': list(self.item_features.keys())
        }




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
        if config.model.model_name.lower() == 'test':
            return TESTModel(
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
        if config.model.model_name.lower() == "mmgcn":
            return MMGCN(
                config=config,
                user_features=user_features,
                item_features=item_features
            )
        if config.model.model_name.lower() == "mmgcn_rec":
            return MMGCN_rec(
                config=config,
                user_features=user_features,
                item_features=item_features
            )
        if config.model.model_name.lower() == "fastmmgcn":
            return FastMMGCN(
                config=config,
                user_features=user_features,
                item_features=item_features
            )
        if config.model.model_name.lower() == "sgrec":
            return SGrec(
                config=config,
                user_features=user_features,
                item_features=item_features
            )
        else:
            raise ValueError(f"Unknown model: {config.model.model_name}")
