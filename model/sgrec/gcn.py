import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class II_GCN(nn.Module):
    def __init__(
        self,
        k,
        alpha,
        beta,
        edge_drop_rate,
        x_drop_rate, 
        z_drop_rate,
        cache_key,
        ):
        super().__init__()
        # self.g = g
        self.k = k
        self.alpah = alpha
        self.beta = beta
        self.alpha = alpha
        self.gamma = pow(self.beta, k) + self.alpha * sum(pow(self.beta, i) for i in range(k))

        self.x_dropout = nn.Dropout(x_drop_rate)
        self.edge_dropout = nn.Dropout(edge_drop_rate)
        self.z_dropout = nn.Dropout(z_drop_rate)

        self.cache_key = cache_key

    def forward(self,x,g):
        self.g = g
        h0 = self.x_dropout(x)
        h = h0

        edge_weight = self.edge_dropout(self.g.edata[self.cache_key])

        with self.g.local_scope():
            self.g.edata[self.cache_key] = edge_weight
            for _ in range(self.k):
                g.ndata["h"] = h
                g.update_all(fn.u_mul_e("h", self.cache_key, "m"), fn.sum("m", "h"))
                h = g.ndata.pop("h")

                h = h * self.beta + h0 * self.alpha
                h = h / self.gamma
                h = self.z_dropout(h)
            return h
        



class IU_GCN(nn.Module):
    def __init__(
        self,
        k,
        edge_drop_rate,
        x_drop_rate, 
        z_drop_rate,
        cache_key,
        ):
        super().__init__()
        self.k = k

        self.x_dropout = nn.Dropout(x_drop_rate)
        self.edge_dropout = nn.Dropout(edge_drop_rate)
        self.z_dropout = nn.Dropout(z_drop_rate)

        self.cache_key = cache_key


    def forward(self,x,g):
        self.g = g
        h0 = self.x_dropout(x)
        h = h0

        edge_weight = self.edge_dropout(self.g.edata[self.cache_key])

        with self.g.local_scope():
            self.g.edata[self.cache_key] = edge_weight
            for _ in range(self.k):
                g.ndata["h"] = h
                g.update_all(fn.u_mul_e("h", self.cache_key, "m"), fn.sum("m", "h"))
                h = g.ndata.pop("h")
                h = self.z_dropout(h)
            return h




