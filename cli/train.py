import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.datamodule import LMDataModule
from src.training.callbacks import GradNormCallback
from src.training.lightning_module import GPTLitModule
from src.utils import env, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model.yaml")
    ap.add_argument("--resume", default=None, help="чекпоинт для продолжения обучения")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = env("CHECKPOINT_DIR", "./checkpoints")

    # ClearML: сохраняет конфиг запуска и авто-подхватывает TensorBoard-метрики
    try:
        from clearml import Task
        Task.init(project_name=cfg.logging.clearml_project,
                  task_name=cfg.logging.clearml_task)
    except Exception as e:
        print("ClearML отключён:", e)

    dm = LMDataModule(cfg)
    cfg.model.vocab_size = dm.vocab_size  # размер словаря берём из токенизатора ЛР1
    model = GPTLitModule(cfg, dm.vocab_size)

    ckpt_cb = ModelCheckpoint(            # сохраняем лучший по val_perplexity + last для resume
        dirpath=ckpt_dir, monitor="val_perplexity", mode="min",
        save_top_k=1, save_last=True, filename="gpt-{step}-{val_perplexity:.2f}")
    callbacks = [ckpt_cb, LearningRateMonitor("step"),
                 GradNormCallback(cfg.trainer.track_layer_grad_norms)]

    trainer = L.Trainer(
        max_steps=cfg.optim.max_steps,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.optim.grad_clip,          # обрезка по глобальной норме
        gradient_clip_algorithm="norm",
        logger=TensorBoardLogger(env("CHECKPOINT_DIR", "./checkpoints"), name="tb"),
        callbacks=callbacks,
    )
    trainer.fit(model, dm, ckpt_path=args.resume)

    # Демонстрация инференса по лучшему чекпоinту
    import torch
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(cfg.data.bpe_tokenizer_path)
    prompt = cfg.generate.prompt
    ids = torch.tensor([tok.encode(prompt).ids], device=model.device)
    out = model.generate(ids, max_new_tokens=cfg.generate.max_new_tokens,
                         temperature=cfg.generate.temperature, top_k=cfg.generate.top_k)
    print("PROMPT:", prompt)
    print("GENERATION:", tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
