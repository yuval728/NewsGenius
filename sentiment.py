import numpy as np
import torch
from torch import nn
import torch.utils
import torch.utils.data
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.data.utils import get_tokenizer




class Sentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(Sentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False, dropout=0.5)
        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        # out = self.fc(out)
        return 
    
    def predict(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        out = torch.softmax(out, dim=1)
        return out
    
def sentiment_predictor(text, model, vocab, tokenizer):
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    model.eval()
    with torch.inference_mode():
        text=tokenizer(text)['input_ids']
        for i in range(len(text)):
            if text[i] not in vocab.get_stoi().values():
                text[i]=0 # replace with <unk> token
                # print('not in vocab')
        text=torch.nn.utils.rnn.pad_sequence([torch.tensor(text)], batch_first=True)
        # print(text)
        
        output=model.predict(text)
        print(output)
        _, predicted = torch.max(output, 1)
        return predicted.item()

    

def load_vocab_tokenizer_model(model_path, vocab_path, embedding_dim, hidden_dim, num_layers, num_classes):
    tokenizer = get_tokenizer("basic_english")
    vocab=torch.load(vocab_path)
    model = Sentiment(len(vocab), embedding_dim, hidden_dim, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model, vocab, tokenizer

