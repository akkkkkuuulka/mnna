import math

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadMaskedAttention(nn.Module):
    # Многоголовое внимание. block_mask (B,1,L,L, bool) приходит снаружи:
    # реализует block-masked attention для packed batching.
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # общая проекция в Q,K,V
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, block_mask):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # делим на головы: (B, n_heads, L, d_head)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~block_mask, float("-inf"))  # запрещённые позиции -> -inf
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)  # обратно склеиваем головы
        return self.resid_dropout(self.proj(out))
