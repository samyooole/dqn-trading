import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F

class qNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(qNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        #self.fc4 = nn.Linear(64,64)
        self.fclast = nn.Linear(8, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #x = torch.relu(self.fc4(x))
        x = self.fclast(x)
        #return F.softmax(x, dim=-1)
        return x
    

class ExperienceReplayBuffer:
    def __init__(self, batch_size):
        self.buffer = []
        self.batch_size = batch_size

    def add(self, item_to_add):
        self.buffer.append(item_to_add)

    def sampleBuffer(self):
        return random.sample(self.buffer, self.batch_size)