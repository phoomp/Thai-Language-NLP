import argparse

import torch
from torch import nn
import torchtext

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from BEST import BESTDataset, BESTText, CharacterTokenizer


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

    tokenizer = CharacterTokenizer(onehot=True)

    tokenizer.fit(features, print_dict=True)
    
    dataset = BESTText(features, labels, surround=500)
    
    

    print('ready')

    while True:
        input_string = input()

        print(tokenizer(input_string))
        print(tokenizer.to_text(tokenizer(input_string)))

    

if __name__ == '__main__':
    main()