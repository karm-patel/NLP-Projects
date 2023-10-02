import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
# from torchmetrics.functional import multiclass_f1_score

class LSTM(nn.Module):
    def __init__(self, n_classes=3, emb_size=300, hidden_size=200):
        super(LSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.lstm_layer = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
        self.layer_1 = nn.Linear(hidden_size,128)
        self.layer_2 = nn.Linear(128,n_classes)
        
    def forward(self, inputs):
        x, h, c = inputs
        # x.shape = (B, L, 300)
        # h.shape = c.shape = (B, 200)
        _, (h, c) = self.lstm_layer(x, (h,c)) #c.shape = (Num layers, B, 200)
        c = c.mean(dim = 0) # (B, 200)
        x = self.layer_1(c)
        x = F.relu(x)
        
        x = self.layer_2(x)
        x = F.softmax(x)
        
        return x # (B,3)
    
    def get_accuracy(self, y_true, y_pred):
        y_true_cpu = y_true.to(torch.device("cpu"))
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        return accuracy_score(y_true_cpu, y_pred_cpu)
    
    def get_f1_score(self, y_true, y_pred):
        y_true_cpu = y_true.to(torch.device("cpu"))
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        f1 = f1_score(y_pred_cpu, y_true_cpu, average='macro')
        return f1


