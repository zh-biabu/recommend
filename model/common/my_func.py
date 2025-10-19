from torch import nn
import torch
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()

        self.func = func

    def forward(self, x):
        return self.func(x)
        
def create_act(name=None):
    if name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return Lambda(lambda x: x)
    else:
        raise Exception()

def ROPE(x):
    seq_len, dim = x.size()[-2:]
    inv = 1.0 / (10000**(torch.arange(0, dim, 2).float() / dim)) 
    pos = torch.arange(0, seq_len).float()
    freqs = torch.einsum('i,j->ij', pos, inv)
    emb = torch.stack((freqs.sin(), freqs.cos()), dim=-1).reshape(x.shape[-2:])#L*d
    emb = emb.unsqueeze(0)
    x_rot = x[..., ::2] * emb[..., ::2] - x[..., 1::2] * emb[..., 1::2]
    x_pass = x[..., ::2] * emb[..., 1::2] + x[..., 1::2] * emb[..., ::2]
    x = torch.stack([x_rot, x_pass], dim=-1).reshape(x.shape)

    return x

# 测试用例
if __name__ == "__main__":
    # 测试5: ROPE函数
    print("测试1: ROPE (Rotary Position Embedding)")
    print("-" * 30)
    
    test_input = torch.randn(seq_len, qkv_num)
    print(f"ROPE输入形状: {test_input.shape}")
    
    rope_output = ROPE(test_input)
    print(f"ROPE输出形状: {rope_output.shape}")
    print(f"ROPE输出范围: [{rope_output.min().item():.4f}, {rope_output.max().item():.4f}]")
    print()