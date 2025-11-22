import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding that can be added on top of token embeddings.
    This module is stateless and therefore works for variable sequence lengths.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self attention implemented using torch.matmul for clarity.
    Supports optional key padding masks to ignore padded positions.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            mask: Optional tensor broadcastable to (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer feed-forward network with configurable hidden size and activation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with pre-norm configuration for stability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(x)
        attn_output, attn_weights = self.self_attn(attn_input, mask)
        x = x + self.dropout(attn_output)

        ff_input = self.norm2(x)
        ff_output = self.ffn(ff_input)
        x = x + self.dropout(ff_output)
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers with optional attention weight return.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        attn_list = [] if return_attn else None
        for layer in self.layers:
            x, attn = layer(x, mask)
            if return_attn:
                attn_list.append(attn)
        x = self.norm(x)
        return x, attn_list


class Transformer(nn.Module):
    """
    A lightweight Transformer encoder model that can work with either
    raw embeddings (shape B x L x D) or token indices (shape B x L).
    When token indices are provided, an internal embedding layer plus
    positional encoding are applied automatically.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        *,
        vocab_size: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pooling: str = "cls"
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling

        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.token_embedding = None

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        elif pooling == "mean":
            self.cls_token = None
        else:
            raise ValueError("pooling must be either 'cls' or 'mean'")

    def _embed_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_embedding is None:
            return x
        embeddings = self.token_embedding(x)
        return embeddings * math.sqrt(self.d_model)

    def _prepare_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Tensor of shape (batch, seq_len) when vocab_size is provided,
               otherwise (batch, seq_len, d_model).
            mask: Optional tensor marking valid tokens (1) vs padding (0).
        """
        if self.token_embedding is not None:
            x = self._embed_inputs(x)

        batch_size = x.size(0)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                mask = torch.cat(
                    [torch.ones(batch_size, 1, device=x.device, dtype=mask.dtype), mask],
                    dim=1
                )

        x = self.pos_encoding(self.dropout(x))
        attn_mask = self._prepare_mask(mask)
        encoded, attn_list = self.encoder(x, attn_mask, return_attn)

        if self.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            if mask is not None:
                valid_lens = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / valid_lens
            else:
                pooled = encoded.mean(dim=1)

        return pooled, attn_list

