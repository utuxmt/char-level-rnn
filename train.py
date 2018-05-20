# coding: utf-8

import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM
from utils import read_corpus, get_char2id, pad_sequence, make_tensor, yield_batch


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--hidden_size", type=int, default=50)
    argparser.add_argument("-b", "--batch_size", type=int, default=128)
    argparser.add_argument("-e", "--num_epoch", type=int, default=3)
    argparser.add_argument("-l", "--lr", type=float, default=0.01)
    args = argparser.parse_args()

    boy_names = read_corpus("./data/NationalNames.csv", gender="M")
    try:
        with open("char2id.pkl", "rb") as handle:
            char2id = pickle.load(handle)
    except FileNotFoundError as err:
        print("Creating char2id...")
        char2id = get_char2id(boy_names)
        with open("./char2id.pkl", "wb") as handle:
            pickle.dump(char2id, handle)
    print('Character to id:\n', char2id)
    print("Distinct boys' name count:\n", len(boy_names))

    model = BiLSTM(input_size=len(char2id), hidden_size=args.hidden_size, output_size=len(char2id))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    batch_size = args.batch_size

    avg_loss = 0
    for epoch in range(args.num_epoch):
        batches = yield_batch(boy_names, batch_size, char2id)
        for i, (input_, target_, seq_lengths) in enumerate(batches):
            loss = 0
            seq_len = len(input_[:, 0])
            batch_size_ = len(input_[0, :])
            hidden = model.init_hidden(batch_size_)
            model.zero_grad()

            for x in range(seq_len):
                output, hidden = model(input_[x, :], hidden)
                loss += loss_func(output, target_[x, :])
            loss.backward()
            optimizer.step()
            avg_loss += (loss.item() / seq_len)
            if i % 100 == 0:
                print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))

    torch.save(model, './name.pt')


