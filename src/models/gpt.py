import torch
import torch.nn as nn

from src.models.positional import SinusoidalPositionalEncoding
from src.models.transformer_block import TransformerBlock
from src.training.masks import build_block_mask, build_position_ids


class GPTBackbone(nn.Module):
    # Чистая torch-модель GPT (без Lightning). Все размеры — из конфига.
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, n_heads,
                 d_ff, dropout, tie_weights=True, pad_token_id=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)  # GPT-init: иначе огромные логиты и нестабильность
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # связываем веса эмбеддинга и LM-head

    @staticmethod
    def _init_weights(module):
        # нормальная инициализация std=0.02 (как в GPT-2)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, segment_ids):
        # segment_ids задают позиции и блок-маску внимания
        pos_ids = build_position_ids(segment_ids)
        block_mask = build_block_mask(segment_ids)
        x = self.drop(self.tok_emb(input_ids) + self.pos_enc(pos_ids))
        for blk in self.blocks:
            x = blk(x, block_mask)
        return self.lm_head(x)  # сырые логиты, softmax не применяем (его делает cross_entropy)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=None, eos_id=None):
        # Авторегрессионная генерация одной последовательности (инференс)
        self.eval()
        ids = prompt_ids  # (1, t)
        for _ in range(max_new_tokens):
            ctx = ids[:, -self.max_seq_len:]  # обрезаем контекст до max_seq_len
            seg = torch.ones_like(ctx)  # один объект -> обычная causal-маска
            logits = self.forward(ctx, seg)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")  # оставляем только top-k
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return ids
