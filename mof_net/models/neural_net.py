# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:41:59 2024

@author: ssnaik
"""
import torch.nn as nn
import torch

torch.manual_seed(42)
#Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size,hidden,  output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
       
        return x

class ModerateNN(nn.Module):
    def __init__(self, input_size,hidden,  output_size):
        super(ModerateNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden[0])
        self.bn1 = nn.BatchNorm1d(hidden[0])
        
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.bn3 = nn.BatchNorm1d(hidden[2])
        
        self.fc4 = nn.Linear(hidden[2], output_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        
        x = self.relu(x)
        
        x = self.fc4(x)
       
        return x