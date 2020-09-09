import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


PAD = -1
def build_vocab(file):
    all_words = []
    all_labels = []
    with open(file, "r") as handle:
        lines = handle.readlines()
        for l in lines:
            l = l.split()
            if len(l):
                all_words.append(l[0])
                all_labels.append(l[1])   
            
    all_words = list(set(all_words))                    
    all_labels = list(set(all_labels)) 
    vocab = dict(zip(all_words, [i for i in range(len(all_words))]))
    labels = dict(zip(all_labels, [i for i in range(len(all_labels))]))    
    l = len(vocab)
    vocab['UNK'] = l
    vocab['PAD'] = l + 1
    global PAD
    PAD = vocab['PAD']
    return vocab, labels

def get_data_examples(fname, vocab_dict, labels_dict):
    sentences = []
    labels = []

    sentence = []
    label = []

    with open(fname) as handle:
        for l in handle.readlines():
            l = l.split()
            if len(l)==0 and len(sentence):
                assert len(sentence) == len(label)
                sentences.append(np.array(sentence))
                labels.append(np.array(label))
                sentence = []
                label = []
            else:
                try:
                    sentence.append(vocab_dict[l[0]])
                except:
                    sentence.append(vocab_dict["UNK"])
                label.append(labels_dict[l[1]])
                    
    assert len(sentences) == len(labels) 
    return sentences, labels


class NerDataset(Dataset):
    def __init__(self, fname, vocab_dict=None, labels_dict=None):
        if vocab_dict is None:
            self.vocab_dict, self.labels_dict = build_vocab(fname)
        else:
            assert labels_dict is not None, "weird.. why would you do this?"
            self.vocab_dict, self.labels_dict = vocab_dict, labels_dict
        self.sentences, self.labels = get_data_examples(fname, 
                        self.vocab_dict, 
                        self.labels_dict
                        )
        
    def __getitem__(self, idx):
        s = torch.LongTensor(self.sentences[idx])
        l = torch.LongTensor(self.labels[idx])
        return s, l
    
    def __len__(self):
        return len(self.sentences)

    
def pad_collate(batch):
    """For equal length sequences. Code taken from [1]
    [1]: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html"""
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=PAD)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1)
    
    return xx_pad, yy_pad, x_lens, y_lens        