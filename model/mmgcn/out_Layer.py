import torch
import torch.nn as nn
import dgl
import numpy
import dgl.function as fn
from .Agg_Layer import AGG
from .Combin_Layer import Combine

import os
import sys

# ensure `recommend/model/common` is importable for `my_mlp`
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_CURRENT_DIR, '..', 'common'))
sys.path.append(_COMMON_DIR)

from my_mlp import MyLinear,MyPReLU


class MGCN(nn.Module):
    def __init__(self, emb_num, layer_num, dropout):
        super().__init__()
        self.agg = AGG(emb_num, dropout, "mean")
        self.W_o = MyLinear(emb_num, emb_num)
        self.combine = nn.ModuleList()
        for _ in range(layer_num):
            self.combine.append(Combine(emb_num, dropout))

        self.dropout = nn.Dropout(dropout)
        self.active = MyPReLU()

    def forward(self, g, X, emb, k):
        agg_X = self.agg(X, g, k)
        for f in self.combine:
            agg_X = f(X, agg_X)
        return agg_X + self.dropout(self.active(self.W_o(emb)))

class MMGCN(nn.Module):
    def __init__(self, modal_num, emb_num, layer_num, dropout):
        super().__init__()
        self.modal_num = modal_num
        self.MGCNs = nn.ModuleList()
        for _ in range(modal_num):
            self.MGCNs.append(MGCN(emb_num, layer_num, dropout))

    def forward(self, g, Xs, embs, ks, alphas):
        # print(Xs, embs, ks, alphas, "\n")
        # print(input())
        modal_features = []
        for i in range(self.modal_num):
            modal_features.append(self.MGCNs[i](g, Xs[i], embs[i], ks[i]))
        modal_features = torch.stack(modal_features, dim=0)
        new_alphas = torch.unsqueeze(torch.softmax(alphas, dim=0), dim=-1).unsqueeze(-1)
        return torch.sum(new_alphas * modal_features, dim=0)