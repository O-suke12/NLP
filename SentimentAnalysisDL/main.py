import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import re
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, wordnet

nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

train = pd.read_csv('train.tsv',sep='\t')
test = pd.read_csv('train.tsv',sep='\t')

def cleaning(df):
    cleaned = []
    for sentence in train["Phrase"]:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z0-9]', ' ', sentence)
        tokens = word_tokenize(sentence)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        cleaned.append(tokens)
    return cleaned

train_sentences = cleaning(train)
test_sentences = cleaning(test)
target = nn.functional.one_hot(torch.tensor(train.Sentiment.values))
num_class = target.size()[1]

X_train, X_valid, y_train, y_valid = train_test_split(train_sentences, target, test_size=0.2)

unique_words = set()
len_max = 0
for sentence in X_train:
    unique_words.update(sentence)
    if len_max<len(sentence):
        len_max = len(sentence)


unique_words.add("<unk>")
unique_words.add("<pad>")
unique_words = sorted(list(unique_words))
word_to_id = { word: i for i, word in enumerate(unique_words)}
pid = word_to_id["<pad>"]

import torch

class CustomDataset(Dataset):
    def __init__(self, vocabulary, tokens, label, window_size):
        self.vocab = vocabulary
        self.X = tokens
        self.y = label
        self.pad_token_id = word_to_id["<pad>"]
        self.window_size = window_size
        
    def __len__(self):
        return len(self.X)
    
    def pad_window(self, tokens):
        pad = self.window_size * ["<pad>"]
        tokens = pad + tokens + pad
        return tokens
        
    def __getitem__(self, id):
        length = len(self.X[id])
        y = self.y
        X_padded = self.pad_window(self.X[id])
        X_padded = [word_to_id[token] for token in X_padded]
        X_padded = nn.utils.rnn.pad_sequence(X_padded, batch_first=True, padding_value=self.pad_token_id)
        return X_padded, y, length