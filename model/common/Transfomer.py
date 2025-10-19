from torch import nn
import torch
import torch.nn.functional as F
from my_mlp import MyLinear,MyNormalLinear,MyPReLU


class MultiHeadAttention(nn.Module):
    def __init__(self, qkv_num, n_heads, dropout = 0.1):
        super().__init__()
        assert qkv_num % n_heads == 0
        
        self.qkv_num = qkv_num
        self.n_heads = n_heads
        self.d_k = qkv_num // n_heads
        
        self.W_q = MyLinear(qkv_num, qkv_num)
        self.W_k = MyLinear(qkv_num, qkv_num)
        self.W_v = MyLinear(qkv_num, qkv_num)

        self.W_o = MyLinear(qkv_num, qkv_num)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.qkv_num
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, qkv_num, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = MyLinear(qkv_num, d_ff)
        self.linear2 = MyLinear(d_ff, qkv_num)
        self.dropout = nn.Dropout(dropout)
        self.activation = MyPReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, qkv_num, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(qkv_num, n_heads, dropout)
        self.feed_forward = FeedForward(qkv_num, d_ff, dropout)
        self.norm1 = nn.LayerNorm(qkv_num)
        self.norm2 = nn.LayerNorm(qkv_num)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights



class Transformer(nn.Module):
    def __init__(self, qkv_num, n_heads, n_layers, d_ff, vocab_size=None, max_seq_len=None, dropout=0.1):
        super().__init__()
        self.qkv_num = qkv_num
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Embedding layers (if vocab_size is provided)
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, qkv_num)
            self.position_embedding = nn.Embedding(max_seq_len, qkv_num) if max_seq_len else None
        else:
            self.token_embedding = None
            self.position_embedding = None

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(qkv_num, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, x, pad_token_id=0):
        """Create padding mask for sequences"""
        return (x != pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def forward(self, x, mask=None, return_attention=False):
        # If input is token indices, convert to embeddings
        if self.token_embedding is not None:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
            
            x = self.token_embedding(x) * (self.qkv_num ** 0.5)
            
            if self.position_embedding is not None:
                x += self.position_embedding(positions)
                
            # Create padding mask if not provided
            if mask is None:
                mask = self.create_padding_mask(x.argmax(dim=-1) if x.dim() > 2 else x)
        
        x = self.dropout(x)
        
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x, mask)
            if return_attention:
                attention_weights_list.append(attention_weights)
        
        if return_attention:
            return x, attention_weights_list
        return x


# 测试用例
if __name__ == "__main__":
    print("=" * 50)
    print("Transformer 测试用例")
    print("=" * 50)
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 测试参数（与最新接口保持一致）
    batch_size = 2
    seq_len = 10
    qkv_num = 256
    n_heads = 8
    n_layers = 3
    d_ff = 512
    vocab_size = 1000
    max_seq_len = 512
    
    print(f"测试参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  qkv_num: {qkv_num}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_layers: {n_layers}")
    print(f"  d_ff: {d_ff}")
    print()
    
    # 测试1: 使用预嵌入输入 (直接输入特征，形状为 B x L x qkv_num)
    print("测试1: 使用预嵌入输入 (B x L x qkv_num)")
    print("-" * 30)
    
    transformer1 = Transformer(
        qkv_num=qkv_num,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # 创建随机输入
    x1 = torch.randn(batch_size, seq_len, qkv_num)
    print(f"输入形状: {x1.shape}")
    
    # 前向传播
    output1 = transformer1(x1)
    print(f"输出形状: {output1.shape}")
    print(f"输出范围: [{output1.min().item():.4f}, {output1.max().item():.4f}]")
    print()
    
    # 测试2: 使用token索引输入
    print("测试2: 使用token索引输入")
    print("-" * 30)
    
    transformer2 = Transformer(
        qkv_num=qkv_num,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    # 创建随机token序列
    x2 = torch.randint(1, vocab_size, (batch_size, seq_len))
    print(f"输入形状: {x2.shape}")
    print(f"输入范围: [{x2.min().item()}, {x2.max().item()}]")
    
    # 前向传播
    output2 = transformer2(x2)
    print(f"输出形状: {output2.shape}")
    print(f"输出范围: [{output2.min().item():.4f}, {output2.max().item():.4f}]")
    print()
    
    # 测试3: 返回注意力权重
    print("测试3: 返回注意力权重")
    print("-" * 30)
    
    output3, attention_weights = transformer1(x1, return_attention=True)
    print(f"输出形状: {output3.shape}")
    print(f"注意力权重数量: {len(attention_weights)}")
    print(f"每个注意力权重形状: {attention_weights[0].shape}")
    print()
    
    # 测试4: 使用掩码
    print("测试4: 使用掩码")
    print("-" * 30)
    
    # 创建掩码 (模拟padding)，形状 B x 1 x 1 x L
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 8:] = 0  # 最后2个位置为padding
    print(f"掩码形状: {mask.shape}")
    
    output4 = transformer1(x1, mask=mask)
    print(f"输出形状: {output4.shape}")
    print()
    

    
    # 测试6: 梯度计算
    print("测试6: 梯度计算")
    print("-" * 30)
    
    # 创建简单的损失函数
    target = torch.randn_like(output1)
    loss = nn.MSELoss()(output1, target)
    print(f"损失值: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    total_params = sum(p.numel() for p in transformer1.parameters())
    params_with_grad = sum(p.numel() for p in transformer1.parameters() if p.grad is not None)
    print(f"总参数数量: {total_params}")
    print(f"有梯度的参数数量: {params_with_grad}")
    print()
    
    # 测试7: 模型参数统计
    print("测试7: 模型参数统计")
    print("-" * 30)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params1 = count_parameters(transformer1)
    params2 = count_parameters(transformer2)
    
    print(f"Transformer1 参数数量: {params1:,}")
    print(f"Transformer2 参数数量: {params2:,}")
    print()
    
    # 测试8: 设备兼容性
    print("测试8: 设备兼容性")
    print("-" * 30)
    
    if torch.cuda.is_available():
        print("CUDA可用，测试GPU计算")
        device = torch.device('cuda')
        transformer1_gpu = transformer1.to(device)
        x1_gpu = x1.to(device)
        
        output1_gpu = transformer1_gpu(x1_gpu)
        print(f"GPU输出形状: {output1_gpu.shape}")
        print(f"GPU输出设备: {output1_gpu.device}")
    else:
        print("CUDA不可用，跳过GPU测试")
    print()
    
    print("=" * 50)
    print("所有测试完成！")
    print("=" * 50)


