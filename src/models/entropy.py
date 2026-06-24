import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from configs.config import MAX_LENGTH


def load_gpt2(device):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(device)
    return model, tokenizer


def compute_entropy_batch(texts, model, tokenizer, device,
                          batch_size=8, max_length=MAX_LENGTH):
    entropies = []
    token_counts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Entropy batches"):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        use_amp = str(device).startswith("cuda")
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            mask = attention_mask[:, 1:]
            for j in range(input_ids.shape[0]):
                m = mask[j]
                n_val = m.sum().item()
                if n_val > 0:
                    # cast per-sample only (full-batch .float() would need ~13GB at large batch)
                    loss_j = F.cross_entropy(
                        logits[j].float(), targets[j], reduction="none"
                    )
                    ent = (loss_j * m).sum().item() / n_val
                    entropies.append(ent)
                    token_counts.append(int(n_val))
                else:
                    entropies.append(None)
                    token_counts.append(0)
    return entropies, token_counts


def compute_info_density(entropies, token_counts):
    valid_ent = np.array([e for e in entropies if e is not None])
    valid_tc = np.array([t for e, t in zip(entropies, token_counts) if e is not None])
    return np.average(valid_ent, weights=valid_tc)


def filter_by_entropy(documents, entropies, token_counts, low_pct=5, high_pct=95):
    valid_ent = np.array([e for e in entropies if e is not None])
    low = np.percentile(valid_ent, low_pct)
    high = np.percentile(valid_ent, high_pct)

    filtered = []
    for doc, ent in zip(documents, entropies):
        if ent is not None and low <= ent <= high:
            filtered.append(doc)
    return filtered, (low, high)


def unload_gpt2(model, device):
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
