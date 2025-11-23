import torch
import torch.nn as nn
import torch.nn.functional as f

class Cross(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(
        self,
        item_emb,
        v_h,
        t_h,
    ):
        



