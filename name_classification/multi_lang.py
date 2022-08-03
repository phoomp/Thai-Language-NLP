#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import unicode_literals

import glob
import os
import string
import unicodedata
import random


# In[3]:


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt


# In[4]:


batch_size = 1


# In[5]:


def find_files(path):
    return glob.glob(path)

print(find_files('/Users/phoom/Documents/thai_intent/datasets/names/*.txt'))


# In[6]:


all_letters = string.ascii_letters + ".,;"
n_letters = len(all_letters)


# In[7]:


def unicode_to_ascii(s):
    return ''.join(char for char in unicodedata.normalize('NFD', s)
                   if unicodedata.category(char) != 'Mn' and char in all_letters
                   )
    
print(unicode_to_ascii('Ślusàrski'))


# In[8]:


category_lines = {}
category_map = {}
all_categories = []

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

for filename in find_files('/Users/phoom/Documents/thai_intent/datasets/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines
    
n_categories = len(all_categories)

print(all_categories)
print(all_letters)


# In[9]:


class NameDataset(Dataset):
    def __init__(self, n_letters, all_letters, all_categories, category_lines):
        self.n_letters = n_letters
        self.all_letters = all_letters
        self.all_categories = all_categories
        self.category_map = {}
        
        self.names = []
        self.labels = []
        
        for i, category in enumerate(all_categories):
            self.category_map[category] = i
            
        print(self.category_map)
        
        for category, names in category_lines.items():
            for name in names:
                self.names.append(name)
                self.labels.append(self.category_map[category])
                
                
        self.names = np.array(self.names)
        self.labels = np.array(self.labels)
                
        
    def __len__(self):
        assert len(self.names) == len(self.labels)
        return len(self.labels)
    
    
    def __getitem__(self, idx):
        idx = np.random.choice(self.labels.shape[0], size=1, replace=False)        
        ret_name = self.names[idx]
        ret_label = self.labels[idx]

        ret_name = self.line_to_tensor(ret_name)
        ret_label = torch.Tensor(ret_label)
        
        return ret_name, ret_label
        
    
    def letter_to_idx(self, letter):
        return self.all_letters.find(letter)
    
    
    def line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for i, letter in enumerate(line):
            tensor[i][0][self.letter_to_idx(letter)] = 1
            
        return tensor
            


# In[10]:


class SimpleLSTM(nn.Module):
    def __init__(self, batch_size, vocab_size, hidden_size, num_layers):
        super().__init__()        
        self.lstm = nn.LSTM(batch_size * vocab_size, hidden_size, num_layers)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, 10)
        )
        
        self.sm = nn.LogSoftmax(dim=1)
        
    def forward(self, x, h_n, c_n):
        output, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        x = self.fc1(output)
        x = self.sm(x)
        
        return x, (h_n, c_n)


# In[ ]:


hidden_size = 32
num_layers = 2

device = 'cpu'

dataset = NameDataset(n_letters, all_letters, all_categories, category_lines)
model = SimpleLSTM(batch_size, n_letters, hidden_size, num_layers).to(device)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

h_n = torch.randn((2, hidden_size)).to(device)
c_n = torch.randn((2, hidden_size)).to(device)

epochs = 100

loss_avg = []

for epoch in range(epochs):
    for batch, (name, category) in enumerate(dataset):
        name = name.to(device)
        category = category.to(device)
        
        for i in range(name.size()[0]):
            output, (h_n, c_n) = model(name[i], h_n.detach(), c_n.detach())     
            
        loss = loss_fn(output.to('cpu'), category.to('cpu').type(torch.LongTensor))
        print(batch)
        print(torch.argmax(output, dim=1).item())
        print(category.item())
        print(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        break

