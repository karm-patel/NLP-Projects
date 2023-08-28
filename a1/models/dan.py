import torch
import torch.nn as nn
import torch.nn.functional as F

class DAN(nn.Module):
    def __init__(self, d=300, n_classes=3):
        super(DAN, self).__init__()
        self.layer_1 = nn.Linear(d,128)
        self.layer_2 = nn.Linear(128,n_classes)
        
    def forward(self, x):
        # x.shape = (b, 300)
        # import pdb; pdb.set_trace()
        x = self.layer_1(x)
        x = F.relu(x)
        
        x = self.layer_2(x)
        x = F.log_softmax(x)
        
        return x # (b,3)
    
    def get_accuracy(self, logits, y_true):
        return (torch.argmax(logits,axis=-1) == y_true).to(torch.float).mean()