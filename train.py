from dataloader import NerDataset, pad_collate
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Net
import argparse
from datetime import datetime
import pickle
from utils import seed_everything
seed_everything(42)

def plot(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), dpi=300)
    ax1.plot(metrics['train_loss'])
    ax1.plot(metrics['val_loss'])
    ax1.legend(['train_loss', 'val_loss'])

    ax2.plot(metrics['train_acc'])
    ax2.plot(metrics['val_acc'])
    ax2.legend(['train_acc', 'val_acc'])

    plt.show()

def acc(ypreds, ytrue):
    mask = (ytrue >= 0)
    ypreds = ypreds[mask]
    ytrue = ytrue[mask]
    return (ypreds==ytrue).sum()/len(ytrue)


def loss_fn(outputs, labels):
    labels = labels.flatten()
    mask = (labels >= 0)
    return F.nll_loss(outputs[mask], labels[mask])


def get_model(vocab_size, number_of_tags):
    embedding_dim = 64
    lstm_hidden_dim = 16
    n_stack = 1
    dropout = 0.0
    bidirectional = True
    seed_everything(42)
    return Net(vocab_size,
                embedding_dim,
                lstm_hidden_dim,
                number_of_tags,
                n_stack,
                dropout,
                bidirectional)

def train(train_f, dev_f):
    shuffle = True
    bs = 64
    train_dataset = NerDataset(train_f)
    val_dataset = NerDataset(dev_f, 
                            train_dataset.vocab_dict, 
                            train_dataset.labels_dict
                            )

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=bs, 
                            shuffle=shuffle, 
                            collate_fn=pad_collate
                            )
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=bs,
                            shuffle=shuffle, 
                            collate_fn=pad_collate
                            )
    lr = 3e-2
    n_epochs = 5
    device = "cuda"
    vocab_size = len(train_dataset.vocab_dict) + 1
    number_of_tags = len(train_dataset.labels_dict)
    
    vocab_dict = train_dataset.vocab_dict
    labels_dict = train_dataset.labels_dict
    
    model = get_model(vocab_size, number_of_tags).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = loss_fn    
    metrics = defaultdict(list)

    #Lets' go! Start the training!
    for epoch in tqdm(range(n_epochs)):
        running_train_loss = 0
        running_val_loss = 0
        running_train_acc = 0
        running_val_acc = 0
        model.train()
        for i, data in enumerate(train_loader):
            X, y, _, _ = data
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = crit(output, y.flatten())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_train_loss += loss.item()
                ypreds = output.argmax(axis=1).detach().cpu().numpy()
                ytrue = y.flatten().detach().cpu().numpy()
                running_train_acc += acc(ypreds, ytrue)
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y, _, _ = data
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = crit(output, y.flatten())
                running_val_loss += loss.item()   
                ypreds = output.argmax(axis=1).detach().cpu().numpy()
                ytrue = y.flatten().detach().cpu().numpy()
                running_val_acc += acc(ypreds, ytrue)
                
        print(f"Epoch: [{epoch}/{n_epochs}] |" +
                f"Train Loss: {running_train_loss/len(train_loader):.3f} | " +
                f"Train Acc: {running_train_acc/len(train_loader):.3f} | " +
                f"Val Loss: {running_val_loss/len(val_loader):.3f} | " +
                f"Val Acc: {running_val_acc/len(val_loader):.3f}"
                )    
        
        metrics['train_loss'].append(running_train_loss/len(train_loader))
        metrics['train_acc'].append(running_train_acc/len(train_loader))    
        metrics['val_loss'].append(running_val_loss/len(val_loader))    
        metrics['val_acc'].append(running_val_acc/len(val_loader))       
    
      
    return model, metrics, vocab_dict, labels_dict               


def save(obj, fname):
    with open(fname, "wb") as handle:
        pickle.dump(obj, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_f', default="data/train_all.tsv", 
                    type=str,
                    help="path to file containing the training data.")
    parser.add_argument('--dev_f', default="data/devel_all.tsv", 
                    type=str,
                    help="path to file containing the devel data.")
    parser.add_argument('--save_fname', default="lstm_model", 
                    type=str, 
                    help="output model path.") 


    args = parser.parse_args()    
    train_f = args.train_f
    dev_f = args.dev_f
    save_fname = args.save_fname
    model, metrics, vocab_dict, labels_dict = train(train_f, dev_f)
    tstamp = str(datetime.now())
    save_fname = f"models/{save_fname}_{tstamp}.pth"
    f1 = "models/vocab_dict.pkl"
    f2 = "models/labels_dict.pkl"
    save(vocab_dict, f1)
    save(labels_dict, f2)
    torch.save(model.state_dict(), save_fname)
    print(f"Model saved in {save_fname}")
    print(f"vocab_dict saved in {f1}")
    print(f"labels_dict saved in {f2}")

    plot(metrics)
