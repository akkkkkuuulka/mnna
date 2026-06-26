import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
from tokenizers import Tokenizer

from src.data.text import (
    clean_wikitext,
    encode_documents,
    load_wikitext_articles,
    packed_batching,
)


class PackedDataset(Dataset):
    # Готовые packed-строки: input_ids и segment-маска одинаковой длины
    def __init__(self, inputs, masks):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return {"input_ids": self.inputs[i], "segment_ids": self.masks[i]}


class LMDataModule(L.LightningDataModule):
    # Данные и токенизатор из ЛР1; на выходе packed-батчи фиксированной длины
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = Tokenizer.from_file(cfg.data.bpe_tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("<PAD>") or 0
        self.vocab_size = self.tokenizer.get_vocab_size()

    def setup(self, stage=None):
        d = self.cfg.data
        if d.dataset == "wikitext":
            articles = load_wikitext_articles("train")
            docs = clean_wikitext(articles, max_documents=d.get("max_documents"))
        else:
            raise ValueError(f"unknown dataset {d.dataset}")

        seqs = encode_documents(docs, self.tokenizer)
        inputs, masks = packed_batching(seqs, d.target_len, self.pad_id)

        # Валидационный сплит по packed-строкам
        n_val = max(1, int(len(inputs) * d.val_fraction))
        self.train_ds = PackedDataset(inputs[:-n_val], masks[:-n_val])
        self.val_ds = PackedDataset(inputs[-n_val:], masks[-n_val:])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.data.batch_size,
                          shuffle=True, num_workers=self.cfg.data.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.data.batch_size,
                          shuffle=False, num_workers=self.cfg.data.num_workers)
