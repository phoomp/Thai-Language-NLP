import torch
from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.embed_dim = 16
        
        self.embeddings = nn.Embedding(vocab_size, self.embed_dim)
        
        print('vocab')
        print(vocab_size)
        print('embedding_dim')
        print(embedding_dim)
        print('batch_size')
        print(batch_size)
        
        self.lstm_input_size = int(embedding_dim * vocab_size * self.embed_dim)
        
        self.left_lstm = nn.Sequential(
            nn.LSTM(self.lstm_input_size, 128, batch_first=True)
        )
        
        self.right_lstm = nn.Sequential(
            nn.LSTM(self.lstm_input_size, 128, batch_first=True)
        )
        
        self.fc1 = nn.Linear(256, 1)
        
    
    def forward(self, left, right):
        embeds_left = self.embeddings(left)
        embeds_right = self.embeddings(right)
        
        print(embeds_left.size())
        
        embeds_left = embeds_left.view(len(left), 1, -1)
        embeds_right = embeds_right.view(len(right), 1, -1)
        
        left_out, _ = self.left_lstm(embeds_left)
        right_out, _ = self.right_lstm(embeds_right)
        
        left_out = left_out.transpose(0, 1).view(self.batch_size, -1)
        right_out = right_out.transpose(0, 1).view(self.batch_size, -1)
        
        x = torch.cat((left_out, right_out), dim=1)
        
        x = self.fc1(x)

        return x
