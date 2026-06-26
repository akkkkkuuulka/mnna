import math


def perplexity(mean_loss):
    # Перплексия = exp(средняя кросс-энтропия). Порог задания: val_perplexity <= 30
    return math.exp(min(mean_loss, 50))  # clamp от переполнения на ранних шагах
