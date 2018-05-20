# coding: utf-8

import pandas as pd
import torch
from torch.autograd import Variable

def read_corpus(filepath, gender="M"):
    df = pd.read_csv(filepath, usecols=["Name", "Gender"])
    boy_names = df.loc[df["Gender"] == gender]
    boy_names = boy_names["Name"].str.lower().unique()
    return boy_names

def get_char2id(names, pad_mark="0"):
    char2id = {"\n": 0, pad_mark: 1}
    for name in names:
        for x in name:
            if x not in char2id:
                char2id[x] = len(char2id)
    return char2id

def pad_sequence(seq, max_len, pad_mark="0"):
    seq += (pad_mark * max(max_len - len(seq), 0))
    return seq

def make_tensor(seq, char2id):
    tensor = torch.zeros(len(seq)).long()
    for idx, x in enumerate(seq):
        tensor[idx] = char2id[x]
    return tensor

def yield_batch(data, batch_size, char2id):
    batches = [data[k: k+batch_size] for k in range(0, len(data), batch_size)]
    for batch in batches:
        batch_sorted = sorted(batch, key=lambda seq: len(seq), reverse=True)
        seq_lengths = [len(seq) for seq in batch_sorted]
        max_len = seq_lengths[0]

        sequences_padded = [pad_sequence(seq, max_len) for seq in batch_sorted]
        inputs_padded = [make_tensor(seq[: -1], char2id) for seq in sequences_padded]
        targets_padded = [make_tensor(seq[1: ], char2id) for seq in sequences_padded]
        yield Variable(torch.stack(inputs_padded)).transpose(0, 1), Variable(torch.stack(targets_padded)).transpose(0, 1), seq_lengths

