import json
import re
from collections import Counter


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
_PRETOK = re.compile(r"\s+|\w+|[^\w\s]+", re.UNICODE)


class Encoding:
    def __init__(self, ids, tokens):
        self.ids = ids
        self.tokens = tokens


class BPETokenizer:
    def __init__(self):
        self.merges = {}     # пара (a, b) -> ранг (порядок слияния)
        self.vocab = {}      # id -> bytes
        self.token2id = {}   # bytes/спецтокен -> id

    def train(self, documents, vocab_size):
        word_freqs = Counter()
        for doc in documents:
            for chunk in _PRETOK.findall(doc):
                word_freqs[chunk] += 1
        words = {w: [bytes([b]) for b in w.encode("utf-8")] for w in word_freqs}

        tokens = list(SPECIAL_TOKENS) + [bytes([b]) for b in range(256)]
        self.merges = {}
        n_merges = max(0, vocab_size - len(tokens))

        for _ in range(n_merges):
            # считаем частоты пар соседних токенов по всем словам 
            pairs = Counter()
            for w, syms in words.items():
                f = word_freqs[w]
                for i in range(len(syms) - 1):
                    pairs[(syms[i], syms[i + 1])] += f
            if not pairs:
                break
            best = max(pairs, key=pairs.get)  # самая частая пара
            merged = best[0] + best[1]
            self.merges[best] = len(self.merges)
            tokens.append(merged)
            # применяем слияние во всех словах
            for w, syms in words.items():
                if len(syms) < 2:
                    continue
                new, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                        new.append(merged); i += 2
                    else:
                        new.append(syms[i]); i += 1
                words[w] = new

        self.vocab = {i: tok for i, tok in enumerate(tokens)}
        self.token2id = {tok: i for i, tok in enumerate(tokens)}
        return self

    def _merge_word(self, syms):
        while len(syms) >= 2:
            best, best_rank = None, None
            for i in range(len(syms) - 1):
                r = self.merges.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            syms[best:best + 2] = [syms[best] + syms[best + 1]]
        return syms

    def encode(self, text):
        ids, toks = [], []
        for chunk in _PRETOK.findall(text):
            syms = self._merge_word([bytes([b]) for b in chunk.encode("utf-8")])
            for s in syms:
                ids.append(self.token2id.get(s, self.token2id["<UNK>"]))
                toks.append(s.decode("utf-8", errors="replace"))
        return Encoding(ids, toks)

    def decode(self, ids):
        specials = set(range(len(SPECIAL_TOKENS)))
        data = b"".join(self.vocab[i] for i in ids if i in self.vocab and i not in specials)
        return data.decode("utf-8", errors="replace")

    def get_vocab_size(self):
        return len(self.vocab)

    def token_to_id(self, token):
        if token in SPECIAL_TOKENS:
            return self.token2id[token]
        return self.token2id.get(token.encode("utf-8"))

    def save(self, path):
        obj = {
            "special_tokens": SPECIAL_TOKENS,
            "vocab": [[i, list(tok) if isinstance(tok, bytes) else tok]
                      for i, tok in self.vocab.items()],
            "merges": [[list(a), list(b)] for (a, b) in self.merges],
        }
        with open(path, "w") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            obj = json.load(f)
        t = cls()
        t.vocab = {i: (bytes(tok) if isinstance(tok, list) else tok) for i, tok in obj["vocab"]}
        t.token2id = {tok: i for i, tok in t.vocab.items()}
        t.merges = {(bytes(a), bytes(b)): r for r, (a, b) in enumerate(obj["merges"])}
        return t


def train_bpe(documents, vocab_size, save_path):
    tok = BPETokenizer().train(documents, vocab_size)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(save_path))
    return tok


def load_bpe(path):
    return BPETokenizer.load(str(path))
