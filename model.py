import torch
import torch.nn as nn


class BaseModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self):
        super(BaseModel, self).__init__()
        
        self.layer1 = nn.Linear(3072, 256)
        self.layer2 = nn.Linear(256, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.log_softmax(x)       
        
        return out