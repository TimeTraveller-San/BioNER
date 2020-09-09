import torch
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F

from model import Net
from train import get_model
from dataloader import NerDataset, pad_collate

import pickle
from utils import seed_everything

seed_everything(42) #for reproducibility

def load_data(model_path, vocab_dict, labels_dict):

    vocab_size = len(vocab_dict) + 1
    number_of_tags = len(labels_dict)
    
    model = get_model(vocab_size, number_of_tags).to(device)
    model.load_state_dict(torch.load(model_path))
    return model       

def infer(sentence, model, vocab_dict, labels_dict, device):
    model.eval()
    s = [vocab_dict[w] if w in vocab_dict else vocab_dict["UNK"] for w in sentence.split()]
    model_inp = torch.LongTensor(s).view(1, -1).to(device) #bs of 1
    model.eval()
    with torch.no_grad():
        y = model(model_inp).detach().cpu().numpy()
    preds = y.argmax(axis=1)
    mask = (preds!=labels_dict['O'])
    return np.array(sentence.split())[mask]


def load(fname):
    with open(fname, "rb") as handle:
        return pickle.load(handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="Bi-LSTM", 
                    type=str,
                    help="type of model (Bi-LSTM, Transformer etc).")
    parser.add_argument('--f', default="data/train_all.tsv", 
                    type=str,
                    help="path to the pyTorch model.")
    parser.add_argument('--s', default="Lipoxygenases (EC 1.13.11.-) are a family of (non-heme) iron-containing enzymes most of which catalyze the dioxygenation", 
                    type=str, 
                    help="string to identify NER for.")                    
    parser.add_argument('--vd', default="models/vocab_dict.pkl", 
                    type=str, 
                    help="Vocab dict pkl file.") 
    parser.add_argument('--ld', default="models/labels_dict.pkl", 
                    type=str, 
                    help="Label dict pkl file.") 

    args = parser.parse_args()    
    model_type = args.model_type
    model_path = args.f
    sentence = args.s
    vd = load(args.vd)
    ld = load(args.ld)
    device = "cuda"
    model = load_data(model_path, vd, ld)
    model = model.to(device)
    nes = infer(sentence, model, vd, ld, device)
    print(f"Identifed named entities are: {nes}")



