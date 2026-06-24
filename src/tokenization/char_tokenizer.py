class CharTokenizer:
    def __init__(self):
        self.char2id = {}
        self.id2char = {}

    def build_vocab(self, documents):
        all_chars = set()
        for doc in documents:
            all_chars.update(doc)
        vocab = sorted(all_chars)
        self.char2id = {ch: i for i, ch in enumerate(vocab)}
        self.id2char = {i: ch for ch, i in self.char2id.items()}

    @property
    def vocab_size(self):
        return len(self.char2id)

    def encode(self, text):
        return [self.char2id[ch] for ch in text if ch in self.char2id]

    def decode(self, ids):
        return "".join(self.id2char[i] for i in ids)
