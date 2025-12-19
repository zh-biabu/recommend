import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Spatial Multi-head Self-Attention module.
    
    This module performs multi-head self-attention with two pooling strategies:
    - "mean": Average pooling across attention heads
    - "cat": Concatenation of all attention heads
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1, 
        pool_method: str = "mean",
        ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.pool_method = pool_method
        
        # Linear projections for query, key, value, and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.o_linear = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        Forward pass of multi-head attention.
        
        Args:
            q: Query tensor of shape (batch_size, d_model)
            k: Key tensor of shape (batch_size, d_model)
            v: Value tensor of shape (batch_size, d_model)
            
        Returns:
            output: Tensor of shape (batch_size, d_model)
        """
        batch_size = q.shape[0]
        
        # Project and reshape q, k to (num_heads, batch_size, head_dim)
        q = self.q_linear(q).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)#H*N*e
        k = self.k_linear(k).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)#H*N*e
        
        # Compute attention scores: (num_heads, batch_size, batch_size)
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        
        # Process value tensor based on pooling method
        if self.pool_method == "mean":
            v = self.v_linear(v)#N*E
            output = scores @ v#H*N*E
            output = output.mean(dim=0)#N*E
        elif self.pool_method == "cat":
            v = self.v_linear(v).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)#H*N*e
            output = scores @ v
            # Concatenate heads: (batch_size, d_model)
            output = output.transpose(0, 1).contiguous().view(batch_size, self.d_model)
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
        
        # Final output projection and dropout
        output = self.o_linear(output)
        output = self.output_dropout(output)
        return output

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    
    A two-layer MLP with ReLU activation and dropout.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, d_model)
            
        Returns:
            output: Tensor of shape (batch_size, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class SpatialTransformerBlock(nn.Module):
    """
    A single spatial transformer block consisting of:
    1. Multi-head self-attention
    2. Layer normalization
    3. Feed-forward network
    4. Layer normalization
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, pool_method: str = "mean"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.pool_method = pool_method
        
        # Sub-modules
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout, pool_method)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization layers
        self.norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        Forward pass of transformer block.
        
        Args:
            q: Query tensor of shape (batch_size, d_model)
            k: Key tensor of shape (batch_size, d_model)
            v: Value tensor of shape (batch_size, d_model)
            
        Returns:
            output: Tensor of shape (batch_size, d_model)
        """
        # Multi-head self-attention
        x = self.attention(q, k, v)
        # Layer normalization
        x = self.norm(x)
        # Feed-forward network
        x = self.feed_forward(x)
        # Layer normalization
        x = self.norm2(x)
        # Output dropout
        x = self.output_dropout(x)
        return x

class SpatialTransformer(nn.Module):
    """
    Spatial Transformer consisting of multiple transformer blocks.
    
    The first layer uses separate q, k, v inputs, while subsequent layers
    use the output from the previous layer as q, k, v (self-attention).
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, pool_method: str = "mean"):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.pool_method = pool_method
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SpatialTransformerBlock(d_model, num_heads, d_ff, dropout, pool_method) 
            for _ in range(num_layers)
        ])
    
    def forward(self, q, k, v):
        """
        Forward pass through all transformer blocks.
        
        Args:
            q: Query tensor of shape (batch_size, d_model)
            k: Key tensor of shape (batch_size, d_model)
            v: Value tensor of shape (batch_size, d_model)
            
        Returns:
            output: Tensor of shape (batch_size, d_model)
        """
        # First layer uses original q, k, v
        for i, transformer_block in enumerate(self.transformer_blocks):
            if i == 0:
                # First block: use provided q, k, v
                v = transformer_block(q, k, v)
            else:
                # Subsequent blocks: use previous output as q, k, v (self-attention)
                v = transformer_block(v, v, v)
        return v


if __name__ == "__main__":
    """
    Test cases for Spatial Transformer components.
    """
    print("=" * 60)
    print("Testing Spatial Transformer Components")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 4
    d_model = 64
    num_heads = 4
    d_ff = 128
    dropout = 0.1
    
    print(f"\nTest Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  dropout: {dropout}")
    
    # Create dummy input tensors
    q = torch.randn(batch_size, d_model)
    k = torch.randn(batch_size, d_model)
    v = torch.randn(batch_size, d_model)
    
    print(f"\nInput shapes:")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    
    # Test 1: MultiHeadSelfAttention with mean pooling
    print("\n" + "-" * 60)
    print("Test 1: MultiHeadSelfAttention (mean pooling)")
    print("-" * 60)
    try:
        attn_mean = MultiHeadSelfAttention(d_model, num_heads, dropout, pool_method="mean")
        output_mean = attn_mean(q, k, v)
        print(f"✓ Output shape: {output_mean.shape}")
        assert output_mean.shape == (batch_size, d_model), f"Expected ({batch_size}, {d_model}), got {output_mean.shape}"
        print("✓ Shape check passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: MultiHeadSelfAttention with cat pooling
    print("\n" + "-" * 60)
    print("Test 2: MultiHeadSelfAttention (cat pooling)")
    print("-" * 60)
    try:
        attn_cat = MultiHeadSelfAttention(d_model, num_heads, dropout, pool_method="cat")
        output_cat = attn_cat(q, k, v)
        print(f"✓ Output shape: {output_cat.shape}")
        assert output_cat.shape == (batch_size, d_model), f"Expected ({batch_size}, {d_model}), got {output_cat.shape}"
        print("✓ Shape check passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: FeedForward
    print("\n" + "-" * 60)
    print("Test 3: FeedForward")
    print("-" * 60)
    try:
        ffn = FeedForward(d_model, d_ff, dropout)
        x_test = torch.randn(batch_size, d_model)
        output_ffn = ffn(x_test)
        print(f"✓ Input shape: {x_test.shape}")
        print(f"✓ Output shape: {output_ffn.shape}")
        assert output_ffn.shape == (batch_size, d_model), f"Expected ({batch_size}, {d_model}), got {output_ffn.shape}"
        print("✓ Shape check passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: SpatialTransformerBlock
    print("\n" + "-" * 60)
    print("Test 4: SpatialTransformerBlock (mean pooling)")
    print("-" * 60)
    try:
        block_mean = SpatialTransformerBlock(d_model, num_heads, d_ff, dropout, pool_method="mean")
        output_block = block_mean(q, k, v)
        print(f"✓ Output shape: {output_block.shape}")
        assert output_block.shape == (batch_size, d_model), f"Expected ({batch_size}, {d_model}), got {output_block.shape}"
        print("✓ Shape check passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 5: SpatialTransformerBlock with cat pooling
    print("\n" + "-" * 60)
    print("Test 5: SpatialTransformerBlock (cat pooling)")
    print("-" * 60)
    try:
        block_cat = SpatialTransformerBlock(d_model, num_heads, d_ff, dropout, pool_method="cat")
        output_block_cat = block_cat(q, k, v)
        print(f"✓ Output shape: {output_block_cat.shape}")
        assert output_block_cat.shape == (batch_size, d_model), f"Expected ({batch_size}, {d_model}), got {output_block_cat.shape}"
        print("✓ Shape check passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 6: SpatialTransformer (single layer)
    print("\n" + "-" * 60)
    print("Test 6: SpatialTransformer (1 layer, mean pooling)")
    print("-" * 60)
    try:
        transformer_1 = SpatialTransformer(1, d_model, num_heads, d_ff, dropout, pool_method="mean")
        output_trans = transformer_1(q, k, v)
        print(f"✓ Output shape: {output_trans.shape}")
        assert output_trans.shape == (batch_size, d_model), f"Expected ({batch_size}, {d_model}), got {output_trans.shape}"
        print("✓ Shape check passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 7: SpatialTransformer (multiple layers)
    print("\n" + "-" * 60)
    print("Test 7: SpatialTransformer (3 layers, cat pooling)")
    print("-" * 60)
    try:
        transformer_3 = SpatialTransformer(3, d_model, num_heads, d_ff, dropout, pool_method="cat")
        output_trans_3 = transformer_3(q, k, v)
        print(f"✓ Output shape: {output_trans_3.shape}")
        assert output_trans_3.shape == (batch_size, d_model), f"Expected ({batch_size}, {d_model}), got {output_trans_3.shape}"
        print("✓ Shape check passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 8: Gradient flow test
    print("\n" + "-" * 60)
    print("Test 8: Gradient Flow Check")
    print("-" * 60)
    try:
        transformer = SpatialTransformer(2, d_model, num_heads, d_ff, dropout, pool_method="mean")
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        
        output = transformer(q, k, v)
        loss = output.sum()
        loss.backward()
        
        print(f"✓ q.grad is not None: {q.grad is not None}")
        print(f"✓ k.grad is not None: {k.grad is not None}")
        print(f"✓ v.grad is not None: {v.grad is not None}")
        assert q.grad is not None and k.grad is not None and v.grad is not None
        print("✓ Gradient flow check passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 9: Error handling - invalid d_model/num_heads
    print("\n" + "-" * 60)
    print("Test 9: Error Handling (invalid d_model/num_heads)")
    print("-" * 60)
    try:
        try:
            attn_invalid = MultiHeadSelfAttention(65, 4, dropout)  # 65 % 4 != 0
            print("✗ Should have raised ValueError")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 10: Error handling - invalid pool_method
    print("\n" + "-" * 60)
    print("Test 10: Error Handling (invalid pool_method)")
    print("-" * 60)
    try:
        attn_invalid = MultiHeadSelfAttention(d_model, num_heads, dropout, pool_method="invalid")
        output = attn_invalid(q, k, v)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("All Tests Completed!")
    print("=" * 60)