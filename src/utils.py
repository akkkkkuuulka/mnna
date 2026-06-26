import random
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
