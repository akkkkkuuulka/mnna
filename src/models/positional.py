import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    # Синусоидальное PE. Индексируем по position_ids, а не по arange,
    # чтобы у packed-объектов позиции считались с нуля внутри каждого объекта.
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # чётные каналы — sin
        pe[:, 1::2] = torch.cos(position * div_term)  # нечётные — cos
        self.register_buffer("pe", pe, persistent=False)  # фиксированный, не обучается

    def forward(self, position_ids):
        return self.pe[position_ids]  # (B, L) -> (B, L, d_model)
