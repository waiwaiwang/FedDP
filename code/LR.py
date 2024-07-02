import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(20, 2) 
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x):
        x = x.to(self.linear.weight.dtype)
        logits = self.linear(x)
        output = logits
        return output