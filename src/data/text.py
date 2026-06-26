import re
import unicodedata

# Пайплайн данных из ЛР1 (вендорнут, чтобы lab-2 был самодостаточным на Kaggle):
# загрузка wikitext -> очистка -> BPE -> packed batching.


def load_wikitext_articles(split="train"):
    # Склеиваем построчный wikitext обратно в статьи по заголовкам "= ... ="
    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")[split]["text"]
    articles, cur = [], []
    for line in raw:
        s = line.strip()
        if s.startswith("= ") and s.endswith(" =") and not s.startswith("= ="):
            if cur:
                t = "\n".join(cur).strip()
                if len(t) > 50:
                    articles.append(t)
            cur = []
        elif s:
            cur.append(s)
    if cur:
        t = "\n".join(cur).strip()
        if len(t) > 50:
            articles.append(t)
    return articles


def _normalize(text):
    # Нормализация Unicode + схлопывание пробелов
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def _split_long(text, max_words=700, min_words=20):
    # Длинные статьи режем на куски ~max_words по абзацам
    words = text.split()
    if len(words) <= max_words:
        return [text] if len(words) >= min_words else []
    chunks, cur, n = [], [], 0
    for para in text.split("\n"):
        pw = len(para.split())
        if n + pw > max_words and cur:
            chunks.append("\n".join(cur)); cur, n = [], 0
        cur.append(para); n += pw
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if len(c.split()) >= min_words]


def clean_wikitext(articles, max_documents=None):
    # Артефакты wikitext (@-@ и т.п.) -> нормальные символы, затем нормализация и нарезка
    docs = []
    for a in articles:
        a = a.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
        docs.extend(_split_long(_normalize(a)))
        if max_documents and len(docs) >= max_documents:
            return docs[:max_documents]
    return docs


def encode_documents(docs, tokenizer):
    # BPE-кодирование (tokenizer из ЛР1)
    return [tokenizer.encode(d).ids for d in docs]


def packed_batching(token_sequences, target_len, pad_id):
    # Склеиваем короткие объекты в строки длиной target_len; маска = segment id
    # (0=PAD, 1=первый объект, 2=второй, ...) — её и использует block-masked attention в ЛР2
    packed_inputs, packed_masks = [], []
    cur_tok, cur_mask, doc_id = [], [], 1
    for seq in token_sequences:
        seq = seq[:target_len]
        if len(cur_tok) + len(seq) > target_len:
            pad = target_len - len(cur_tok)
            cur_tok += [pad_id] * pad; cur_mask += [0] * pad
            packed_inputs.append(cur_tok); packed_masks.append(cur_mask)
            cur_tok, cur_mask, doc_id = [], [], 1
        cur_tok += seq; cur_mask += [doc_id] * len(seq); doc_id += 1
    if cur_tok:
        pad = target_len - len(cur_tok)
        cur_tok += [pad_id] * pad; cur_mask += [0] * pad
        packed_inputs.append(cur_tok); packed_masks.append(cur_mask)
    return packed_inputs, packed_masks
