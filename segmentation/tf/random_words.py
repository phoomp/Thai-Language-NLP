import tensorflow as tf
from tensorflow import keras

import numpy as np

import os


def clean_words(words):
    # Remove dataset tags
    words = words.replace('<NE>', '|')
    words = words.replace('</NE>', '|')
    words = words.replace('<AB>', '|')
    words = words.replace('</AB>', '|')
    
    # Remove unwanted characters
    words = words.replace('\n', '|')
    words = words.replace(':', '|')
    words = words.replace('"', '|')
    words = words.replace('(', '|')
    words = words.replace(')', '|')
    words = words.replace('[', '|')
    words = words.replace(']', '|')
    words = words.replace('{', '|')
    words = words.replace('}', '|')
    
    # Remove extraneous spaces
    words = words.split(' ')
    words = ''.join(words)
    
    # Remove extraneous bars
    words = words.split('|')
    words = filter(lambda word: len(word) > 0, words)
    words = '|'.join(words)
    return words


def generate_random_words(file_path='/data/thai-datasets/datasets/BEST/BEST-TrainingSet'):
    words = ''
    for dir in next(os.walk(f'{file_path}'))[1]:
        path = os.path.join(file_path, dir)
        for file in next(os.walk(path))[2]:
            if file[0] == '.':
                continue
            else:
                txt_path = os.path.join(path, file)
                with open(txt_path, 'r') as f:
                    words += f.read()
                    
    words = clean_words(words)
    words = words.split('|')
    words = list(set(words))
    
    print(words[1:30])
    print(len(words))
    