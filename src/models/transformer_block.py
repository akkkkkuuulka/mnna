import torch.nn as nn

from src.models.attention import GroupedQueryAttention
from src.models.ffn import FeedForward


class TransformerBlock(nn.Module):
    # Post-norm слой: z1 = LN(x + Attn(x)); z2 = LN(z1 + FFN(z1)).
    # Внимание — GQA; пробрасываем KV-кэш для инференса.
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout, use_flash=True):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, use_flash)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask, past_kv=None, use_cache=False):
        a, new_kv = self.attn(x, attn_mask, past_kv=past_kv, use_cache=use_cache)
        z1 = self.ln1(x + a)
        z2 = self.ln2(z1 + self.ffn(z1))
        return z2, new_kv
