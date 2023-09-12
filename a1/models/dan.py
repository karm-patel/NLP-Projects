import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
# from torchmetrics.functional import multiclass_f1_score

class DAN(nn.Module):
    def __init__(self, d=300, n_classes=3, drop_prob=0.3):
        super(DAN, self).__init__()
        self.layer_1 = nn.Linear(d,128)
        self.dropout = nn.Dropout(drop_prob)
        self.layer_2 = nn.Linear(128,n_classes)
        
    def forward(self, x):
        # x.shape = (b, 300)
        # import pdb; pdb.set_trace()
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = F.softmax(x)
        
        return x # (b,3)
    
    def get_accuracy(self, y_true, y_pred):
        y_true_cpu = y_true.to(torch.device("cpu"))
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        return accuracy_score(y_true_cpu, y_pred_cpu)
    
    def get_f1_score(self, y_true, y_pred, average="micro"):
        y_true_cpu = y_true.to(torch.device("cpu"))
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        f1 = f1_score(y_pred_cpu, y_true_cpu, average=average)
        return f1


