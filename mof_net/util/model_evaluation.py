# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:10:34 2024

@author: ssnaik
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import R2Score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def model_eval(model, test_loader, criterion):
    # Step 5: Evaluate the Model
    r2_score = {}
    model.eval()
    total_loss = 0.0
    metric = R2Score()
    plt.figure()
    # Generate x values
    pp_x = np.linspace(0, 1)
    
    # Compute y values
    pp_y = [i for i in pp_x]
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            plt.plot(outputs, 'ro')
            plt.plot(labels, 'bx')
            
            plt.figure()
            plt.plot(outputs, labels, 'o')
            plt.plot(pp_x, pp_y)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            for dim in range(1):
                metric.update(outputs[:,dim], labels[:,dim])
                r2_score[dim] = metric.compute()
                print("R2-SCORE = ", metric.compute())
                

    average_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {average_loss}")
    return r2_score, average_loss
    
   