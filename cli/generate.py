import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer

from src.training.lightning_module import GPTLitModule
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model.yaml")
    ap.add_argument("--ckpt", required=True, help="чекпоинт обученной модели")
    ap.add_argument("--prompt", default="The history of")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = Tokenizer.from_file(cfg.data.bpe_tokenizer_path)
    # инициализация обученной модели из сохранённых весов
    model = GPTLitModule.load_from_checkpoint(
        args.ckpt, cfg=cfg, vocab_size=tok.get_vocab_size(), map_location="cpu")
    model.eval()

    ids = torch.tensor([tok.encode(args.prompt).ids])
    out = model.model.generate(ids, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k)
    print("PROMPT:", args.prompt)
    print("GEN   :", tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
