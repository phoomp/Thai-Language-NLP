import argparse

import torch
from torch import embedding, nn
import torchtext

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from zmq import device

from BEST import BESTDataset, CharacterTokenizer
from BESTModels import SimpleLSTM


parser = argparse.ArgumentParser('Train a word segmentation model')

parser.add_argument('train_path', type=str)
parser.add_argument('test_path', type=str)


def main():
    print('\n')
    args = parser.parse_args()
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    
    surround = 64
    batch_size = 2
    device = 'mps'

    dataset = BESTDataset(TRAIN_PATH, TEST_PATH, surround=surround)
    
    char_remove_list = ['^', '=', '+', '~', '\\\\', r'\ufeff', '_', '$', '?', '@', '#']
    
    dataset.clean(remove_list=char_remove_list) 
    features, labels = dataset.generate(mode='minimal')
    
    print(features[-3:-1])
    print(labels[-3:-1])

    dataset.tokenizer = CharacterTokenizer(onehot=True)

    dataset.tokenizer.fit(features, print_dict=True)

    print('ready')
    
    vocab_size = len(dataset.tokenizer)
    embedding_dim = surround
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = SimpleLSTM(vocab_size, embedding_dim, batch_size).to('mps')
    
    # TODO: Define a loss function for contrastive loss
    # TODO: Generate len(loader) negative pairs
    
    print(len(dataset))

    for batch, (left_frame, right_frame, label) in enumerate(loader):
        left_frame = left_frame.to(device)
        right_frame = right_frame.to(device)
        
        label = label.to(torch.int64).to(device)
        x = model(left_frame, right_frame)
        print(label)
        if batch == 2:
            break

    

if __name__ == '__main__':
    main()