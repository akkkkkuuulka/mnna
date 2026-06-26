import math


def warmup_cosine_lambda(warmup_steps, max_steps, min_lr_ratio):
    # Множитель LR: линейный warm-up, затем cosine-спад до min_lr_ratio
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # разогрев: 0 -> 1
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        prog = min(prog, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))  # 1 -> 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return fn
