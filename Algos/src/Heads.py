from typing import *
from torch import nn
import torch



class Policy_Head(nn.Module):
    def __init__(self, num_actions: int, embed_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class Value_Head(nn.Module): #Value head network.
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256, num_quantiles: int = 8):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # (bs, num_quantiles)