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

class Combine(nn.Module):
    def __init__(self, emb_num, dropout):
        super().__init__()
        self.W_ori_f = MyLinear(emb_num, emb_num)
        self.W_f = MyLinear(emb_num, emb_num)
        # self.W_e = MyLinear(emb_num, emb_num)
        self.W_o = MyLinear(emb_num, emb_num)
        self.dropout = nn.Dropout(dropout)
        self.active = MyPReLU()

    def forward(self, ori_feature_x, feature_x):
        y1 = self.W_ori_f(ori_feature_x)
        y2 = self.W_f(feature_x)
        o = self.dropout(self.active(self.W_o(y1 + y2)))
        return o