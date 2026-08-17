from torch import nn
import torch

class QNet(nn.Module):

    def __init__(self, inDim, outDim):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(inDim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, outDim)
    
    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.fc3(x)