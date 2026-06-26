import torch

# Маски для packed batching. segment_ids (B, L): 0=паддинг, 1=первый объект, 2=второй и т.д.
# Из них строим позиции, блок-маску внимания и маску для loss.


def build_position_ids(segment_ids):
    # Позиции сбрасываются на 0 в начале каждого объекта (а не сквозные по строке)
    B, L = segment_ids.shape
    ar = torch.arange(L, device=segment_ids.device).unsqueeze(0).expand(B, L)
    is_start = torch.ones_like(segment_ids, dtype=torch.bool)
    is_start[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]  # смена объекта = новый старт
    start_idx, _ = torch.cummax(torch.where(is_start, ar, torch.zeros_like(ar)), dim=1)
    pos = ar - start_idx
    return pos * (segment_ids != 0)  # у паддинга позиция 0 (всё равно маскируется)


def build_block_mask(segment_ids):
    # M[i,j] = (один объект) и (j<=i) и (не паддинг). True = внимание разрешено
    B, L = segment_ids.shape
    same = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=segment_ids.device))
    nonpad = (segment_ids != 0).unsqueeze(2)
    allowed = same & causal & nonpad
    eye = torch.eye(L, dtype=torch.bool, device=segment_ids.device).unsqueeze(0)
    allowed = allowed | eye  # диагональ всегда True — иначе строка-паддинг даёт NaN после softmax
    return allowed.unsqueeze(1)  # (B, 1, L, L)


def build_loss_mask(segment_ids):
    # Позиция i учится предсказывать i+1 только если оба токена из одного объекта
    cur, nxt = segment_ids[:, :-1], segment_ids[:, 1:]
    return (cur == nxt) & (cur != 0)
