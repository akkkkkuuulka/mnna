from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_bpe(documents, vocab_size, save_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(documents, trainer=trainer)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    return tokenizer


def load_bpe(path):
    return Tokenizer.from_file(str(path))
