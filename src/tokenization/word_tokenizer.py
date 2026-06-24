import re
from collections import Counter

from tqdm.auto import tqdm

WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class WordTokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.unk_id = 0
        self.total_unique = 0

    def build_vocab(self, documents, vocab_size=50000):
        counter = Counter()
        for doc in tqdm(documents, desc="Building word vocab"):
            tokens = WORD_PATTERN.findall(doc.lower())
            counter.update(tokens)
        self.total_unique = len(counter)

        special = ["<PAD>", "<UNK>"]
        most_common = [w for w, _ in counter.most_common(vocab_size)]
        vocab = special + most_common
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.unk_id = self.word2id["<UNK>"]

    @property
    def vocab_size(self):
        return len(self.word2id)

    def encode(self, text):
        tokens = WORD_PATTERN.findall(text.lower())
        return [self.word2id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids):
        return " ".join(self.id2word.get(i, "<UNK>") for i in ids)
