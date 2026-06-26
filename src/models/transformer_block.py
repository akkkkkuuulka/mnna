import torch.nn as nn

from src.models.attention import MultiHeadMaskedAttention
from src.models.ffn import FeedForward


class TransformerBlock(nn.Module):
    # Post-norm слой (по заданию): нормализация ПОСЛЕ residual.
    # z1 = LN(x + Attn(x)); z2 = LN(z1 + FFN(z1))
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadMaskedAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, block_mask):
        z1 = self.ln1(x + self.attn(x, block_mask))
        z2 = self.ln2(z1 + self.ffn(z1))
        return z2
