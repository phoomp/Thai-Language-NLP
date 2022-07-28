import argparse

import torch
from torch import nn
import torchtext

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from BEST import BESTDataset, CharacterTokenizer


parser = argparse.ArgumentParser('Train a word segmentation model')

parser.add_argument('train_path', type=str)
parser.add_argument('test_path', type=str)


def main():
    print('\n')
    args = parser.parse_args()
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path

    dataset = BESTDataset(TRAIN_PATH, TEST_PATH)
    
    char_remove_list = ['^', '=', '+', '~', r'\\', r'\ufeff']
    
    dataset.clean(remove_list=char_remove_list) 
    features, labels = dataset.generate(mode='minimal')
    
    print(features[-3:-1])
    print(labels[-3:-1])

    dataset.tokenizer = CharacterTokenizer(onehot=True)

    dataset.tokenizer.fit(features, print_dict=True)

    print('ready')

    while True:
        input_string = input()

        print(dataset.tokenizer(input_string))
        print(dataset.tokenizer.to_text(dataset.tokenizer(input_string)))

    

if __name__ == '__main__':
    main()