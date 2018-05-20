# coding: utf-8

import sys
import pickle
import argparse
import torch
from torch.autograd import Variable
from utils import get_char2id, make_tensor

def generate(model, char2id, id2char, args):
    prime_str = args.starts_with
    inputs_ = [make_tensor(seq[: -1], char2id) for seq in [prime_str]]
    prime_input = Variable(torch.stack(inputs_)).transpose(0, 1)
    predicted = prime_str

    hidden = model.init_hidden(1)
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)

    inp = prime_input[-1]
    for p in range(args.predict_len):
        output, _ = model(inp, hidden)
        output_dist = output.data.view(-1).div(args.temperature).exp()
        top_id = torch.multinomial(output_dist, 1)[0]
        predicted_char = id2char[top_id.item()]
        predicted += predicted_char
        inp = make_tensor(predicted_char, char2id)
    return predicted.replace('0', '')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--starts_with", type=str, default="ha")
    argparser.add_argument("-l", "--predict_len", type=int, default=2)
    argparser.add_argument("-t", "--temperature", type=float, default=0.8)
    args = argparser.parse_args()

    try:
        model = torch.load("./name.pt")
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)

    try:
        with open("./char2id.pkl", "rb") as handle:
            char2id = pickle.load(handle)
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)

    id2char = {id_: char for (char, id_) in char2id.items()}

    print("Names generated:")
    for _ in range(10):
        predicted = generate(model, char2id, id2char, args)
        print('', predicted)
