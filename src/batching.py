import torch
from torch.utils.data import Dataset


def packed_batching(token_sequences, target_len, pad_id):
    packed_inputs = []
    packed_masks = []

    current_tokens = []
    current_mask = []
    doc_id = 1

    for seq in token_sequences:
        if len(seq) > target_len:
            seq = seq[:target_len]

        if len(current_tokens) + len(seq) > target_len:
            padding_len = target_len - len(current_tokens)
            current_tokens.extend([pad_id] * padding_len)
            current_mask.extend([0] * padding_len)
            packed_inputs.append(current_tokens)
            packed_masks.append(current_mask)
            current_tokens = []
            current_mask = []
            doc_id = 1

        current_tokens.extend(seq)
        current_mask.extend([doc_id] * len(seq))
        doc_id += 1

    if current_tokens:
        padding_len = target_len - len(current_tokens)
        current_tokens.extend([pad_id] * padding_len)
        current_mask.extend([0] * padding_len)
        packed_inputs.append(current_tokens)
        packed_masks.append(current_mask)

    return packed_inputs, packed_masks


class PackedDataset(Dataset):
    def __init__(self, packed_inputs, packed_masks):
        self.inputs = torch.tensor(packed_inputs, dtype=torch.long)
        self.masks = torch.tensor(packed_masks, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "attention_mask": self.masks[idx]}
