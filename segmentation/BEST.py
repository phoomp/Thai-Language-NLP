import os
import re

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

import pickle


class CharacterTokenizer:
    def __init__(self, onehot=True):
        self.use_onehot = onehot 
        self.dictionary = {}
        self.oov_token = -1
        self.oov_text = '<OOV>'


    def __get_key_from_value(self, value):
        for k, v in self.dictionary.items():
            if v == value:
                return k
        
        return 'None'


    def fit(self, text, print_dict=False):
        for char in text:
            if char not in self.dictionary.keys():
                self.dictionary[char] = len(self.dictionary)

        self.dictionary[self.oov_text] = len(self.dictionary)
        
        if print_dict:
            print(self.dictionary)


    def load(self, load_path):
        self.dictionary = pickle.load(open(load_path, 'rb'))


    def save(self, save_path):
        pickle.dump(self.dictionary, open(save_path, 'wb'))


    def to_text(self, tokens):
        text_str = ''
        if self.use_onehot:
            for i in range(tokens.shape[0]):
                idx = np.argwhere(tokens[i] == 1)

                if idx == len(self.dictionary) - 1:
                    text_str += self.oov_text
                else:
                    text_str += self.__get_key_from_value(idx)
        else:
            for token in tokens:
                if token == len(self.dictionary) - 1:
                    text_str += self.oov_text
                else:
                    text_str += self.__get_key_from_value(token)

        return text_str


    def __onehot(self, x):
        print(f'Generating one-hot array of size ({len(x)}, {len(self.dictionary)})')
        arr = np.zeros((len(x), len(self.dictionary)))
        for i, e in enumerate(x):
            arr[i][e] = 1

        return arr


    def __len__(self):
        return len(self.dictionary)


    def __call__(self, text):
        tokens = []
        for char in text:
            try:
                tokens.append(self.dictionary[char])
            except KeyError as e:
                tokens.append(self.oov_token)

        if self.use_onehot:
            tokens = self.__onehot(tokens)

        return np.array(tokens)


class BESTText(Dataset):
    def __init__(self, features, labels, surround):
        self.features = features
        self.labels = labels
        self.surround = surround

        print(self.labels[:20])
        print(self.features[:20])
    
    def __len__(self):
        return len(self.labels)
    
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        label = self.labels[idx]
        
        back_label = label - self.surround if label - self.surround > 0 else 0
        forward_label = label + self.surround if label + self.surround < len(self.labels) - 1 else len(self.labels) - 1
        
        print(back_label)
        print(forward_label)
        
        return feat, 
            

class BESTDataset(Dataset):
    def __init__(self, train_path=None, test_path=None):
        self.train_path = train_path
        self.test_path = test_path

        self.train_str = ''
        self.test_str = ''

        # self.train_str = self.train_str.decode()
        # self.test_str = self.test_str.decode()

        if self.train_path is None or self.test_path is None:
            return

        print(f'Gathering text from {self.train_path} and {self.test_path}')
        
        for directory in next(os.walk(self.train_path))[1]:
            print(f'Found training text directory: {directory}')

            text_dir_path = os.path.join(self.train_path, directory)

            _, _, files = next(os.walk(text_dir_path))     

            for file in files:
                file = os.path.join(text_dir_path, file)
                with open(file, 'r') as f:
                    self.train_str += f.read()

        print(f'Train dataset size: {len(self.train_str)}')

        _, _, files = next(os.walk(self.test_path))

        print(f'Found testing file(s): {files}')

        for file in files:
            if file[0] == '.':
                continue
            with open(os.path.join(self.test_path, file), 'r') as f:
                self.test_str += f.read()

        print(f'Test dataset size: {len(self.test_str)}')
        print(f'Sucessfully loaded BEST Dataset')

    
    def filter(self, string, char):
        string = string.replace(char, '')
        return string


    def clean(self, remove_list=[]):
        """
        Clean the data
        """
        
        # self.train_str = self.train_str.replace('\n', ' ')
        # self.test_str = self.test_str.replace('\n', ' ')
        
        # Remove duplicate spaces
        self.train_str = self.train_str.split(' ')
        self.train_str = ' '.join(self.train_str)

        self.test_str = self.test_str.split(' ')
        self.test_str = ' '.join(self.test_str)

        # Remove chars in remove_list
        for char in remove_list:
            print(f'Removing char: "{char}"')
            self.train_str = self.filter(self.train_str, char)
            self.test_str = self.filter(self.test_str, char)

    
    def generate(self, mode='minimal'):
        """
        Generate the text and labels to be tokenized.
        modes = 'minimal', 'full'
        
        Minimal: all word separators are treated as 1, else 0
        Full: Refer to dictionary below. Make sure to filter these characters
        before inputting data
        
        {
            '^': beginning of named entity,
            '=': end of named entity,
            '+': beginnin of abbreviation,
            '~': end of abbreviation
        }
        """
        
        if mode == 'minimal':
            self.train_str = self.train_str.replace('<NE>', '')
            self.train_str = self.train_str.replace('</NE>', '')
            self.train_str = self.train_str.replace('<AB>', '')
            self.train_str = self.train_str.replace('</AB>', '')
            
            self.test_str = self.test_str.replace('<NE>', '')
            self.test_str = self.test_str.replace('</NE>', '')
            self.test_str = self.test_str.replace('<AB>', '')
            self.test_str = self.test_str.replace('</AB>', '')
            
            train_bar_idx = [m.start() for m in re.finditer(re.escape('|'), self.train_str)]
            test_bar_idx = [m.start() for m in re.finditer(re.escape('|'), self.test_str)]
            
            # Indicates the end of previous word and start of next word
            for i, idx in enumerate(train_bar_idx):
                train_bar_idx[i] = idx - (i + 1)
            
            self.train_str = self.train_str.replace('|', '')
            
            for i, idx in enumerate(test_bar_idx):
                test_bar_idx[i] = idx - (i + 1)
            
            self.train_str = self.train_str.replace('|', '')
            self.test_str = self.test_str.replace('|', '')
            
            # 1-0 label whether the word should be separated at position or not
            train_labels = np.zeros(len(self.train_str) - 1)
            test_labels = np.zeros(len(self.test_str) - 1)
            
            for idx in train_bar_idx:
                train_labels[idx] = 1
            
            for idx in test_bar_idx:
                test_labels[idx] = 1
                
            print('processed train features')
            print(self.test_str[:40])
            print('processed train labels')
            print(test_labels[:40])
            
            self.train_ds = (self.train_str, train_labels)
            self.test_ds = (self.test_ds, test_labels)
        else:
            raise NotImplementedError  


    def __len__(self):
        return len(self.train_str)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()