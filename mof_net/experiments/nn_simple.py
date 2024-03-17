# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:43:24 2024

@author: ssnaik
"""

from mof_net.data.reading_files import get_data
from mof_net.models.neural_net import SimpleNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import R2Score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def simple_nn_baseline(hidden_layers, learning_rate):
    data_path = r"C:\Users\ssnaik\Documents\Courses\Homeworks\adv_deep_learning\Project\deep-learning\mof_net\data"
    features_tensor, targets_tensor = get_data(data_path, features_file = "zeopp.csv" , label_file = "N2_SSL.csv")
    input_size = features_tensor.shape[1]  
    output_size = targets_tensor.shape[1]  
    hidden = hidden_layers
    model = SimpleNN(input_size, hidden, output_size)
    
    # Step 3: Define Loss Function and Optimizer
    criterion = nn.MSELoss()  # Using Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #Total number of trainable paramters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Step 4: Train the Neural Network
    # Splitting the data into training and testing sets
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_tensor, targets_tensor, test_size=0.2, random_state=42)

    # Split train data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # Creating DataLoader for training and testing sets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size = 500, shuffle = False)
    # Training the model
    training_loss = []
    validation_loss = []
    r2_score = {}
    num_epochs = 30000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        val_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_loss.append(running_loss/len(train_loader))
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        for inputs, labels in val_loader:
            #Validation loss
            val_out = model(inputs)
            loss = criterion(val_out, labels)
            val_loss += loss.item()
        validation_loss.append(val_loss/len(val_loader))
        print(f"Epoch {epoch+1}, Loss validation: {val_loss/len(val_loader)}")
        
    # Step 5: Evaluate the Model
    model.eval()
    total_loss = 0.0
    metric = R2Score()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            for dim in range(y_test.shape[1]):
                metric.update(outputs[:,dim], labels[:,dim])
                r2_score[dim] = metric.compute()
                print("R2-SCORE = ", metric.compute())
                

    average_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {average_loss}")
    
    plt.figure()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('log loss')
    plt.title('2 HL network (%d parameters, lr = %0.3f)'%(pytorch_total_params, learning_rate))
    return r2_score, average_loss
    
if __name__ == "__main__":
    r2_score = {}
    test_loss = {}
    hidden_layers = [[30, 10], [50, 20], [100, 100]]
    learning_rate = [0.1, 0.01, 0.001]
    for h in hidden_layers:
        for l in learning_rate:
             print("######"*5)
             print("hidden = ", h)
             print("learning_rate = ", l)
             r2, loss = simple_nn_baseline(h, l)
             r2_score[(tuple(h),l)] = r2
             test_loss[(tuple(h),l)] = loss