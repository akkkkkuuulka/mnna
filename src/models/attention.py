import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    # GQA: голов Q много (n_heads), а K/V мало (n_kv_heads) -> меньше KV-кэш на инференсе.
    # Каждая группа из n_heads/n_kv_heads Q-голов делит одну K/V-голову.
    def __init__(self, d_model, n_heads, n_kv_heads, dropout, use_flash=True):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.d_head = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # сколько Q-голов на одну KV-голову
        self.use_flash = use_flash
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head)  # узкая проекция K
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head)  # узкая проекция V
        self.proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, past_kv=None, use_cache=False):
        # attn_mask: bool (B,1,Tq,Tk), True=внимание разрешено. past_kv=(k,v) для KV-кэша
        B, Tq, D = x.shape
        q = self.q_proj(x).view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, Tq, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, Tq, self.n_kv_heads, self.d_head).transpose(1, 2)

        if past_kv is not None:  # дозаписываем новые K/V к кэшу
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v) if use_cache else None

        # размножаем KV-головы до числа Q-голов (GQA -> по форме обычное MHA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self.use_flash:
            # Flash Attention через SDPA: bool-маска True=участвует в внимании
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            scores = scores.masked_fill(~attn_mask, float("-inf"))
            out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.resid_dropout(self.proj(out)), new_kv
