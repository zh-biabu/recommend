import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from mig_gt.layers.common import MyLinear, MyMLP, get_activation

from mig_gt.layers.mgdcf import MGDCF
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):
    def __init__(self, in_units, att_units, out_units, 
                 ff_units_list,
                 num_heads=1,
                 drop_rate=0.0,
                 att_h_drop_rate=0.0,
                 att_drop_rate=0.0,
                 ff_drop_rate=0.0,
                 output_activation=None,
                 output_drop_rate=0.0,
                 output_ln=False,
                 att_residual=True,
                 ff_residual=True,
                 ln=True,
                 ) -> None:
        super().__init__()


        self.q_linear = MyLinear(in_units, att_units)
        self.k_linear = MyLinear(in_units, att_units)
        # self.v_linear = MyLinear(in_units, out_units)
        self.ff_units_list = ff_units_list

        # ff_units_list = [out_units] + ff_units_list

        if ln:
            self.ln = nn.LayerNorm(out_units)
        else:
            self.ln = None

        self.att_h_dropout = nn.Dropout(att_h_drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.att_dropout = nn.Dropout(att_drop_rate)

        if len(ff_units_list) == 0:
            self.ff = None
        else:
            self.ff = MyMLP(out_units, ff_units_list, 
                            activation="prelu", 
                            drop_rate=drop_rate,
                            bn=True,
                            output_activation="prelu",
                            output_drop_rate=0.0,
                            output_bn=True,
                            
                            # ln=True,
                            # ln=True,
                            # output_ln=True
                            
                            )

        
        self.output_activation = get_activation(output_activation)
        self.output_dropout = nn.Dropout(ff_drop_rate)
        # self.output_bn = nn.BatchNorm1d(ff_units_list[-1]) if output_bn else None
        if len(ff_units_list) > 0:
            self.output_ln = nn.LayerNorm(ff_units_list[-1]) if output_ln else None

        self.att_residual = att_residual
        self.ff_residual = ff_residual

        self.num_heads = num_heads



        

    def forward(self, q, k, return_all=False):

        Q = self.q_linear(q)
        K = self.k_linear(k)
        V = k

        Q_ = torch.concat(Q.split(Q.size(-1) // self.num_heads, dim=-1), dim=0)
        K_ = torch.concat(K.split(K.size(-1) // self.num_heads, dim=-1), dim=0)
        V_ = torch.concat(V.split(V.size(-1) // self.num_heads, dim=-1), dim=0)

        sim = Q_ @ K_.transpose(-2, -1)
        sim /= Q_.size(-1) ** 0.5


        sim_logits = sim

        sim = F.softmax(sim, dim=-1)

        sim = self.att_dropout(sim)

        att_h = sim @ V_

        att_h = torch.concat(att_h.split(q.size(0), dim=0), dim=-1)

        no_residual_att_h = att_h



        if self.att_residual:
            # att_h = att_h + q
            att_h = att_h * 0.1 + q * 0.9
            # att_h = q
    

        if self.ln is not None:
            att_h = self.ln(att_h)


        att_h = self.att_h_dropout(att_h)

        
        if self.ff is None:
            ff_h = att_h
        else:
            # ff_h = self.ff(att_h)

            
            ff_h = self.ff(att_h.squeeze(1)).unsqueeze(1)

            if self.ff_residual:
                ff_h = ff_h + att_h


            if self.output_ln:
                ff_h = self.output_ln(ff_h)

            ff_h = self.output_activation(ff_h)
            ff_h = self.output_dropout(ff_h)

        if return_all:
            return ff_h, no_residual_att_h, sim_logits
        else:
            return ff_h
        


class MMMLP(nn.Module):
    def __init__(self, 
                 feat_drop_rate, 
                 item_v_in_channels=None,
                 item_v_hidden_channels_list=None,
                 item_t_in_channels=None,
                 item_t_hidden_channels_list=None,
                 item_hidden_channels_list=None,
                 *args, **kwargs):
        
        super().__init__()

        self.user_input_dropout = nn.Dropout(p=feat_drop_rate)
        self.item_input_dropout = nn.Dropout(p=feat_drop_rate)
        

        self.item_v_mlp = MyMLP(item_v_in_channels, item_v_hidden_channels_list,
                                activation="prelu", drop_rate=feat_drop_rate, bn=True, 
                                output_activation="prelu", output_drop_rate=feat_drop_rate, output_bn=True)
        self.item_v_input_dropout = nn.Dropout(p=feat_drop_rate)

        self.item_t_mlp = MyMLP(item_t_in_channels, item_t_hidden_channels_list,
                                activation="prelu", drop_rate=feat_drop_rate, bn=True, 
                                output_activation="prelu", output_drop_rate=feat_drop_rate, output_bn=True)
        self.item_t_input_dropout = nn.Dropout(p=feat_drop_rate)

        item_v_out_channels = item_v_hidden_channels_list[-1] if len(item_v_hidden_channels_list) > 0 else item_v_in_channels
        item_t_out_channels = item_t_hidden_channels_list[-1] if len(item_t_hidden_channels_list) > 0 else item_t_in_channels

        self.item_mlp = MyMLP(item_v_out_channels + item_t_out_channels, 
                              item_hidden_channels_list,
                              activation="prelu", drop_rate=feat_drop_rate, bn=True, 
                              output_activation="prelu", output_drop_rate=0.0, output_bn=True)


    def forward(self, item_v_feat, item_t_feat):

        item_v_h = self.item_v_input_dropout(item_v_feat)
        item_v_h = self.item_v_mlp(item_v_h)

        item_t_h = self.item_t_input_dropout(item_t_feat)
        item_t_h = self.item_t_mlp(item_t_h)

        item_h = torch.cat([item_v_h, item_t_h], dim=-1)
        item_h = self.item_mlp(item_h)      

        return item_h      




class MIGGT(nn.Module):
    def __init__(self, 
                #  k, 
                 k_e,
                 k_t,
                 k_v,
                 alpha, beta, 
                 input_feat_drop_rate,
                 feat_drop_rate,
                 user_x_drop_rate,
                 item_x_drop_rate, 
                 edge_drop_rate, z_drop_rate, 
                 user_in_channels=None,
                 user_hidden_channels_list=None,
                 item_v_in_channels=None,
                 item_v_hidden_channels_list=None,
                 item_t_in_channels=None,
                 item_t_hidden_channels_list=None,
                #  item_hidden_channels_list=None,
                 bn=True,
                 num_clusters=0,
                 num_samples=0,
                 use_dual=False,
                 *args, **kwargs):
        
        super().__init__()


     

        self.emb_mgdcf = MGDCF(k_e, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_e >= 0 else None
        self.t_mgdcf = MGDCF(k_t, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_t >= 0 else None
        self.v_mgdcf = MGDCF(k_v, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_v >= 0 else None


        self.z_dropout = nn.Dropout(z_drop_rate)


        if user_hidden_channels_list is not None:
            raise Exception()

        self.user_gnn_input_dropout = nn.Dropout(user_x_drop_rate)
        self.item_gnn_input_dropout = nn.Dropout(item_x_drop_rate)


        self.use_dual = use_dual
        self.num_samples = num_samples


        self.emb_transformer = Transformer(
            user_in_channels,
            16,
            user_in_channels,
            ff_units_list=[],
            att_residual=True,
            ff_residual=False,
            drop_rate=0.0,
            ln=False
        )


        self.t_transformer = Transformer(
            item_t_in_channels,
            4,
            item_t_in_channels,
            ff_units_list=item_t_hidden_channels_list,
            att_residual=True,
            ff_residual=False,
            att_h_drop_rate=input_feat_drop_rate,
            drop_rate=feat_drop_rate,
            ln=False
        )

        self.v_transformer = Transformer(
            item_v_in_channels,
            4,
            item_v_in_channels,
            ff_units_list=item_v_hidden_channels_list,
            att_residual=True,
            ff_residual=False,
            att_h_drop_rate=input_feat_drop_rate,
            drop_rate=feat_drop_rate,
            ln=False
        )


        
        self.z_transformer = Transformer(
            user_in_channels,
            4,
            user_in_channels,
            ff_units_list=[],
            att_residual=True,
            ff_residual=False,
            att_h_drop_rate=0.0,
            drop_rate=feat_drop_rate,
            ln=False
        )

        self.t_memory = nn.parameter.Parameter(
            torch.randn([1, num_clusters, item_t_in_channels]) / np.sqrt(item_t_in_channels), requires_grad=False
        )

        self.v_memory = nn.parameter.Parameter(
            torch.randn([1, num_clusters, item_v_in_channels]) / np.sqrt(item_v_in_channels), requires_grad=False
        )

        self.z_memory = nn.parameter.Parameter(
            torch.randn([1, num_clusters, user_in_channels]) / np.sqrt(user_in_channels), requires_grad=False
        )

        self.t_ff = nn.Sequential(


            MyMLP(
                item_t_in_channels, 
                item_t_hidden_channels_list,
                activation="prelu",
                drop_rate=feat_drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=True,
            )
        )

        self.v_ff = nn.Sequential(
            
            MyMLP(
                item_v_in_channels, 
                item_v_hidden_channels_list,
                activation="prelu",
                drop_rate=feat_drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=True
            )
        )

       
        self.input_feat_dropout = nn.Dropout(input_feat_drop_rate)

        self.ema_alpha = 0.1


      

    def forward(self, g, user_embeddings, item_v_feat, item_t_feat, item_embeddings=None, return_all=False):



        item_v_feat = self.input_feat_dropout(item_v_feat)
        item_t_feat = self.input_feat_dropout(item_t_feat)

      

        encoded_t = self.t_ff(item_t_feat)
        encoded_v = self.v_ff(item_v_feat)


        num_users = user_embeddings.size(0)
        user_h = user_embeddings



        if self.emb_mgdcf is not None:
            emb_h = self.emb_mgdcf(
                g, 
                torch.cat([
                    self.user_gnn_input_dropout(user_h), 
                    torch.zeros_like(encoded_t) if item_embeddings is None else self.item_gnn_input_dropout(item_embeddings)
                ], dim=0)
            )
        else:
            emb_h = None


        if self.t_mgdcf is not None:
            t_h = self.t_mgdcf(
                g, 
                torch.cat([
                    torch.zeros_like(user_h),
                    self.item_gnn_input_dropout(encoded_t),
                ], dim=0)
            )
        else:
            t_h = None




        if self.v_mgdcf is not None:
            v_h = self.v_mgdcf(
                g,
                torch.cat([
                    torch.zeros_like(user_h),
                    self.item_gnn_input_dropout(encoded_v),
                ], dim=0)
            )
        else:
            v_h = None



      
        combined_h = None
        for i, h in enumerate([emb_h, t_h, v_h]):
            if h is not None:
                # print("combine index:", i)
                if combined_h is None:
                    combined_h = h
                else:
                    combined_h  = combined_h + h


        combined_h = self.z_dropout(combined_h)

        emb_h = self.z_dropout(emb_h) if emb_h is not None else None
        t_h = self.z_dropout(t_h) if t_h is not None else None
        v_h = self.z_dropout(v_h) if v_h is not None else None




        num_items = item_v_feat.size(0)
        item_h = combined_h[num_users:]


        memory_index = torch.randint(0, num_items, [combined_h.size(0), self.num_samples])

        memory = item_h[memory_index]

        memory = torch.concat([combined_h.unsqueeze(1), memory], dim=1)

        z_memory_h = self.z_transformer(memory, memory)
        combined_h = z_memory_h[:, 0]

       

        if return_all:
            return combined_h, emb_h, t_h, v_h, encoded_t, encoded_v, z_memory_h
        else:
            return combined_h



            

        
