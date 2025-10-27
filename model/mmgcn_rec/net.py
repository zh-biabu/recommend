# coding: utf-8
"""
MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. 
In ACM MM`19,
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

# from common.abstract_recommender import GeneralRecommender
# from common.loss import BPRLoss, EmbLoss
# from common.init import xavier_uniform_initialization


class Net_rec(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, dim_feats, device):
        super().__init__()
        self.num_user = num_users
        self.num_item = num_users
        dim_x = emb_dim
        self.aggr_mode = 'mean'
        self.concate = 'False'
        has_id = True
        self.dim_feats = dim_feats
        self.device = device
        # self.edge_index = self.edge_index
        self.num_modal= 2

        self.v_gcn = GCN(self.num_user, self.num_item, self.dim_feats[0], dim_x, self.aggr_mode,
                            self.concate, has_id=has_id, dim_latent=256, device=self.device)

        self.t_gcn = GCN(self.num_user, self.num_item, self.dim_feats[1], dim_x,
                            self.aggr_mode, self.concate, has_id=has_id, device=self.device)

        # self.id_embedding = nn.init.xavier_normal_(torch.rand((self.num_user+self.num_item, dim_x), requires_grad=True)).to(self.device)
        # self.result = nn.init.xavier_normal_(torch.rand((self.num_user+self.num_item, dim_x))).to(self.device)

    # def pack_edge_index(self, inter_mat):
    #     rows = inter_mat.row
    #     cols = inter_mat.col + self.n_users
    #     # ndarray([598918, 2]) for ml-imdb
    #     return np.column_stack((rows, cols))

    def forward(self, edge_index, feats, id_embedding):
        representation = None
        self.v_feat = feats[0]
        self.t_feat = feats[1]
        representation, preference = self.v_gcn(edge_index, self.v_feat, id_embedding)
        representation += self.t_gcn(edge_index, self.t_feat, id_embedding)[0]

        representation /= self.num_modal

        return representation, preference

    # def calculate_loss(self, interaction):
    #     batch_users = interaction[0]
    #     pos_items = interaction[1] + self.n_users
    #     neg_items = interaction[2] + self.n_users

    #     user_tensor = batch_users.repeat_interleave(2)
    #     stacked_items = torch.stack((pos_items, neg_items))
    #     item_tensor = stacked_items.t().contiguous().view(-1)

    #     out = self.forward()
    #     user_score = out[user_tensor]
    #     item_score = out[item_tensor]
    #     score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
    #     loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
    #     reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
    #     if self.v_feat is not None:
    #         reg_embedding_loss += (self.v_gcn.preference**2).mean()
    #     reg_loss = self.reg_weight * reg_embedding_loss
    #     return loss + reg_loss

    # def full_sort_predict(self, interaction):
    #     user_tensor = self.result[:self.n_users]
    #     item_tensor = self.result[self.n_users:]

    #     temp_user_tensor = user_tensor[interaction[0], :]
    #     score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
    #     return score_matrix


class GCN(nn.Module):
    def __init__(self, num_user, num_item, dim_feat, dim_id, aggr_mode, concate,
                 has_id, dim_latent=None, device='cpu'):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        # self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.has_id = has_id
        self.device = device

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(self.device)
            #self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent))))

            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(self.device)
            #self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat))))

            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

    def forward(self, edge_index, features, id_embedding):
        self.edge_index = edge_index
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer1(x))  # equation 5
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer1(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer2(x))  # equation 5
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer2(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer3(x))  # equation 5
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer3(h) + x_hat)

        return x, self.preference


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, size=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)