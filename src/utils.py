import os

from dotenv import load_dotenv
from omegaconf import OmegaConf


def load_config(path):
    # Конфиг через OmegaConf; .env подхватывает пути/секреты (на GitHub .env не кладём)
    load_dotenv()
    return OmegaConf.load(path)


def env(key, default=None):
    return os.environ.get(key, default)
