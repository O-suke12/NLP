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