import math

import torch
import torch.nn.functional as F
import lightning as L

from src.models.gpt import GPTBackbone
from src.training.masks import build_loss_mask
from src.training.scheduler import warmup_cosine_lambda


class GPTLitModule(L.LightningModule):
    # Сборка модели + обучение. Все гиперпараметры приходят из конфига.
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        m = cfg.model
        self.model = GPTBackbone(
            vocab_size=vocab_size, max_seq_len=m.max_seq_len, d_model=m.d_model,
            n_layers=m.n_layers, n_heads=m.n_heads, d_ff=m.d_ff, dropout=m.dropout,
            tie_weights=m.tie_weights, pad_token_id=m.pad_token_id,
        )
        if cfg.trainer.get("compile", False):  # ускорение; при ошибке остаёмся в eager
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print("torch.compile failed:", e)
        self._val_loss_sum = 0.0
        self._val_tokens = 0

    def forward(self, input_ids, segment_ids):
        return self.model(input_ids, segment_ids)

    def _masked_loss(self, batch):
        # Кросс-энтропия только на валидных переходах (loss-маска packed batching)
        ids, seg = batch["input_ids"], batch["segment_ids"]
        logits = self.model(ids, seg)
        mask = build_loss_mask(seg)
        sl = logits[:, :-1, :][mask]      # логиты позиции i
        st = ids[:, 1:][mask]             # цель = токен i+1
        loss = F.cross_entropy(sl, st)
        return loss, int(mask.sum())

    def training_step(self, batch, _):
        loss, _ = self._masked_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_perplexity", torch.exp(loss), prog_bar=True, on_step=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_loss_sum, self._val_tokens = 0.0, 0

    def validation_step(self, batch, _):
        loss, n = self._masked_loss(batch)
        # копим взвешенно по числу токенов -> точная перплексия за эпоху
        self._val_loss_sum += loss.item() * n
        self._val_tokens += n

    def on_validation_epoch_end(self):
        mean = self._val_loss_sum / max(1, self._val_tokens)
        self.log("val_loss", mean, prog_bar=True)
        self.log("val_perplexity", math.exp(min(mean, 50)), prog_bar=True)

    def configure_optimizers(self):
        o = self.cfg.optim
        opt = torch.optim.AdamW(self.parameters(), lr=o.lr,
                                betas=tuple(o.betas), weight_decay=o.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, warmup_cosine_lambda(o.warmup_steps, o.max_steps, o.min_lr_ratio))
        # шаг шедулера на каждом батче (warm-up считается в шагах)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    @torch.no_grad()
    def generate(self, prompt_ids, **kw):
        return self.model.generate(prompt_ids, **kw)
