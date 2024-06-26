# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:19:53 2024

@author: ssnaik
"""

from mof_net.data.reading_files import get_data
from mof_net.models.neural_net import SimpleNN, ModerateNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import R2Score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from mof_net.util.model_evaluation import model_eval
from mof_net.util.perform_pca import get_principal_components

torch.manual_seed(42)
def append_feature_vector_1(features_tensor, targets_tensor):
    #Load trained model on feature 1
    #Data for loading model outputs from first target network
    features_target1 = features_tensor
    input_size_t1 = features_target1.shape[1]  
    output_size_t1 = targets_tensor[:,0:1].shape[1]  
    hidden_t1 = [150, 200]
    
    #training features for target 1
    model_target1 = SimpleNN(input_size_t1, hidden_t1, output_size_t1)
    model_target1.load_state_dict(torch.load(r"C:\Users\ssnaik\Documents\Courses\Homeworks\adv_deep_learning\Project\deep-learning\mof_net\experiments\saved_models\new_results\baseline\nn_target_1_model_150_200_001_mse.pkl"))
    
    #Evaluate target 1 by passing the features through trained model
   
    with torch.no_grad():
        outputs = model_target1(features_target1)
        
    #Scale the outputs
    #outputs_scaled = (outputs - min(outputs))/(max(outputs) - min(outputs))
    
    #Append the predicted output to the features tensor as an input to the second NN
    features_tensor = torch.cat((features_tensor, outputs), 1)
    return features_tensor
    
    
def simple_nn_baseline(hidden_layers, learning_rate):
    data_path = r"C:\Users\ssnaik\Documents\Courses\Homeworks\adv_deep_learning\Project\deep-learning\mof_net\data"
    features_tensor_orig, targets_tensor = get_data(data_path, features_file = "zeopp.csv" , 
                                               label_file = "N2_SSL_R299up.csv")
    
    #Append real labels for target 1 to the features tensor
    features_tensor = torch.cat((features_tensor_orig, targets_tensor[:,0:1]), 1)
    

    #Make another features tensor for testing which has predictions from the first NN
    #as a feature
    features_tensor_test = append_feature_vector_1(features_tensor_orig, targets_tensor)
    
    targets_tensor = targets_tensor[:,1:2]
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
    X_train, X_test_fake, y_train, y_test_fake = train_test_split(features_tensor, targets_tensor, test_size=0.2, random_state=1, shuffle = True)
    
    #We can't use this test data for testing since it has the real y 
    #We need to create the test set from the features_tensor_test
    X_train_fake, X_test, y_train_fake, y_test = train_test_split(features_tensor_test, targets_tensor, test_size=0.2, random_state=1, shuffle = True)
    
    # Split train data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1, shuffle = True)

    # Creating DataLoader for training and testing sets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size = 500, shuffle = False)
    # Training the model
    training_loss = []
    training_loss_to_eval = []
    validation_loss = []
    validation_loss_to_eval = []
    
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_loss_eval = 0.0
        val_loss = 0.0
        val_loss_to_eval = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_to_eval = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_eval += loss_to_eval.item()
        training_loss.append(running_loss/len(train_loader))
        training_loss_to_eval.append(running_loss_eval/len(train_loader))
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        for inputs, labels in val_loader:
            #Validation loss
            val_out = model(inputs)
            loss = criterion(val_out, labels)
            loss_to_eval = criterion(val_out, labels)
            val_loss += loss.item()
            val_loss_to_eval += loss_to_eval.item()
            
        validation_loss.append(val_loss/len(val_loader))
        validation_loss_to_eval.append(val_loss_to_eval/len(val_loader))
        print(f"Epoch {epoch+1}, Loss validation: {val_loss/len(val_loader)}")
    
    torch.save(model.state_dict(), 'nn_target_2_model_teachersforcing.pkl') # Save the model
    plt.figure()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('log loss')
    plt.title('2 HL network (%d parameters, lr = %0.3f)'%(pytorch_total_params, learning_rate))
    
    plt.figure()
    plt.plot(training_loss_to_eval, label = 'training loss')
    plt.plot(validation_loss_to_eval, label = 'validation loss')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('log loss')
    plt.title('[Loss Eval MSE] 2 HL network (%d parameters, lr = %0.3f)'%(pytorch_total_params, learning_rate))
    r2_score_training, average_loss_training = model_eval(model, train_loader, criterion)
    r2_score, average_loss = model_eval(model, test_loader, criterion)
    return r2_score, average_loss, r2_score_training, average_loss_training
    
if __name__ == "__main__":
    r2_score_test = {}
    r2_score_train = {}
    test_loss = {}
    train_loss = {}
    hidden_layers = [[50, 70]]
    learning_rate = [0.001]
    for h in hidden_layers:
        for l in learning_rate:
             print("######"*5)
             print("hidden = ", h)
             print("learning_rate = ", l)
             r2_test, loss_test, r2_train, loss_train = simple_nn_baseline(h, l)
             r2_score_test[(tuple(h),l)] = r2_test
             test_loss[(tuple(h),l)] = loss_test
             r2_score_train[(tuple(h),l)] = r2_train
             train_loss[(tuple(h),l)] = loss_train
             



