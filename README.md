# ЛР2. Обучение GPT-like модели

GPT-декодер с нуля (PyTorch + Lightning), данные и BPE-токенизатор из ЛР1.

- **val_perplexity = 29.0** (порог ≤ 30 — выполнен).
- Финальный чекпоинт: [`checkpoints/model_final.ckpt`](checkpoints/model_final.ckpt) (~499 МБ, нужен Git LFS).
- Запуск обучения: `python cli/train.py --config configs/model.yaml`
- Результаты прогона: [`notebooks/lab2.ipynb`](notebooks/lab2.ipynb).
