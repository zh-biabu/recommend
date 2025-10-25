import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import dim


class GCN(nn.Module):
    def __init__(self, dim_feat, hidden_dim, emb_dim, num_users, num_items, concat, k=3, device="cpu"):
        super().__init__()

        self.concat = concat
        self.k = k

        self.user_feat_emb = nn.init.xavier_normal_(torch.randn((num_users, hidden_dim), dtype=torch.float32, requires_grad=True)).to(device)
        self.trans = nn.Linear(dim_feat, hidden_dim)
        
        self.ws = [nn.init.xavier_normal_(torch.randn((hidden_dim, hidden_dim), dtype=torch.float32, requires_grad=True)).to(device)]
        self.des = nn.ModuleList([nn.Linear(hidden_dim, emb_dim)])
        if concat:
            self.outs = nn.ModuleList([nn.Linear(hidden_dim + emb_dim, emb_dim)])
        else:
            self.outs = nn.ModuleList([nn.Linear(hidden_dim, emb_dim)])
        for _ in range(k-1):
            self.ws.append(nn.init.xavier_normal_(torch.randn((emb_dim,emb_dim), dtype=torch.float32, requires_grad=True)).to(device))
            self.des.append(nn.Linear(emb_dim, emb_dim))
            if concat:
                self.outs.append(nn.Linear(emb_dim + emb_dim, emb_dim))
            else:
                self.outs.append(nn.Linear(emb_dim, emb_dim))
    
    def forward(self, feat, node_emb, g):
        feat = self.trans(feat)
        feat = torch.cat([self.user_feat_emb, feat], dim=0)
        feat = F.normalize(feat)

        for i in range(self.k):
            h = F.leaky_relu(g.func(feat, self.ws[i]))
            u = F.leaky_relu(self.des[i](h) + node_emb)
            if self.concat:
                feat = self.outs[i](torch.cat([h,u], dim=1))
            else:
                feat = self.outs[i](h) + u
            feat = F.leaky_relu(feat)
        return feat, self.user_feat_emb



class Net(nn.Module):
    def __init__(self, modal_num, dim_feats, hidden_dim, emb_dim, num_users, num_items, concat, k=3, device="cpu"):
        super().__init__()
        self.gcns = nn.ModuleList()
        for i in range(modal_num):
            self.gcns.append(
                GCN(
                    dim_feats[i], hidden_dim, emb_dim, num_users, num_items, concat, k, device
                )
            )
    
    def forward(self, feats, node_emb, g):
        results= []
        embs = []
        for i, feat in enumerate(feats):
            res, emb = self.gcns[i](feat, node_emb, g)
            results.append(res.unsqueeze(0))
            embs.append(emb)
        
        results = torch.cat(results, dim=0)
        return results.mean(dim = 0), embs