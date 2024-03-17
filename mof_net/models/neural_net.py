# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:41:59 2024

@author: ssnaik
"""
import torch.nn as nn
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
