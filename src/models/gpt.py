import torch
import torch.nn as nn

from src.models.positional import SinusoidalPositionalEncoding
from src.models.transformer_block import TransformerBlock
from src.training.masks import build_block_mask, build_position_ids


class GPTBackbone(nn.Module):
    # GPT с GQA. Обучение — через segment_ids (block-mask, позиции per-object).
    # Инференс — инкрементально с KV-кэшем (generate).
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, n_heads, n_kv_heads,
                 d_ff, dropout, tie_weights=True, pad_token_id=0, use_flash=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, dropout, use_flash)
             for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)  # GPT-init: иначе огромные логиты и нестабильность
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # tie эмбеддинга и LM-head

    @staticmethod
    def _init_weights(module):
        # нормальная инициализация std=0.02 (как в GPT-2)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, segment_ids=None, position_ids=None,
                attn_mask=None, past_kvs=None, use_cache=False):
        if segment_ids is not None:  # путь обучения: всё из segment_ids
            position_ids = build_position_ids(segment_ids)
            attn_mask = build_block_mask(segment_ids)
        x = self.drop(self.tok_emb(input_ids) + self.pos_enc(position_ids))
        new_kvs = []
        for i, blk in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, kv = blk(x, attn_mask, past_kv=past, use_cache=use_cache)
            new_kvs.append(kv)
        logits = self.lm_head(x)  # сырые логиты (softmax внутри cross_entropy)
        return (logits, new_kvs) if use_cache else logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=None, eos_id=None):
        # Инференс с KV-кэшем: префилл промпта, затем по одному токену
        self.eval()
        device = prompt_ids.device
        ids = prompt_ids
        t0 = ids.shape[1]
        pos = torch.arange(t0, device=device).unsqueeze(0)
        causal = torch.tril(torch.ones(t0, t0, dtype=torch.bool, device=device))[None, None]
        logits, kvs = self.forward(ids, position_ids=pos, attn_mask=causal, use_cache=True)
        cur_len = t0
        for _ in range(max_new_tokens):
            nxt = self._sample(logits[:, -1, :], temperature, top_k)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
            if cur_len >= self.max_seq_len:  # PE ограничено max_seq_len
                break
            pos = torch.tensor([[cur_len]], device=device)
            # новый токен видит все прошлые ключи -> маска целиком True
            mask = torch.ones(1, 1, 1, cur_len + 1, dtype=torch.bool, device=device)
            logits, kvs = self.forward(nxt, position_ids=pos, attn_mask=mask,
                                       past_kvs=kvs, use_cache=True)
            cur_len += 1
        return ids

    @staticmethod
    def _sample(logits, temperature, top_k):
        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
