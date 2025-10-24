import torch
import torch.nn as nn
import dgl
import numpy
import dgl.function as fn
import os
import sys

# ensure `recommend/model/common` is importable for `my_mlp`
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_CURRENT_DIR, '..', 'common'))
sys.path.append(_COMMON_DIR)

from my_mlp import MyLinear,MyPReLU

class AGG(nn.Module):
    def __init__(self, emb_num, dropout, agg_type):
        super().__init__()
        self.W = MyLinear(emb_num, emb_num)
        self.dropout = nn.Dropout(dropout)
        self.active = MyPReLU()
        self.agg_type = agg_type

    def forward(self, x, g, k):
        if self.agg_type == "mean":
            return g.agg_func(x, k, self.W, self.dropout, self.active)
    
