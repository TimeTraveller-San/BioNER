import os
import numpy as np
import random
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def path(*args): 
    """My custom function to quickly combine paths in a safe manner"""
    return os.path.join(*args)

def read_data(input_file):
    """a simple helper function to read tsv files. Simply copy pasted 
    from [1].
    [1]: https://github.com/dmis-lab/biobert/blob/master/run_ner.py#L143
    """
    inpFilept = open(input_file)
    lines = []
    words = []
    labels = []
    for lineIdx, line in enumerate(inpFilept):
        contents = line.splitlines()[0]
        lineList = contents.split()
        if len(lineList) == 0: # For blank line
            assert len(words) == len(labels), "Error in reading."
            if len(words) != 0:
                wordSent = " ".join(words)
                labelSent = " ".join(labels)
                lines.append((labelSent, wordSent))
                words = []
                labels = []
            else: 
                print("Two continual empty lines detected!")
        else:
            words.append(lineList[0])
            labels.append(lineList[-1])
    if len(words) != 0:
        wordSent = " ".join(words)
        labelSent = " ".join(labels)
        lines.append((labelSent, wordSent))
        words = []
        labels = []

    inpFilept.close()
    return lines    