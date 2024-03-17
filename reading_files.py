import csv
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torcheval.metrics import R2Score, MeanSquaredError
import torchvision.transforms as transforms
import numpy as np
def get_data(features_file = "zeopp.csv" , label_file = "N2_SSL.csv"):
    path = os.getcwd()
    
    #Read features
    file_path = os.path.join(path,"data", features_file)
    df_x = pd.read_csv(file_path)
    
    #Read labels
    file_path = os.path.join(path, "data", label_file)
    df_y = pd.read_csv(file_path)
    
    filtered_df_x = df_x[df_x['REFCODE'].isin(df_y['REFCODE'])]
    
    # Sort both DataFrames based on the 'ChildName' column
    dfx_sorted = filtered_df_x.sort_values(by='REFCODE')
    dfy_sorted = df_y.sort_values(by='REFCODE')
    
    # Normalize the Features
    x_values = dfx_sorted.iloc[:, 1:].values
    y_values = dfy_sorted.iloc[:, 1:4].values
    
    # scaler = MinMaxScaler()
    # x_scaled = scaler.fit_transform(x_values)  # Reshape to ensure correct shape
    min_x = np.min(x_values, axis= 0)
    max_x = np.max(x_values, axis = 0)
    x_scaled = (x_values-min_x)/(max_x-min_x + 1e-10)
    
    
    min_y = np.min(y_values, axis= 0)
    max_y = np.max(y_values, axis = 0)
    y_scaled = (y_values-min_y)/(max_y-min_y)
    
    # Converting data to torch tensors
    X = torch.tensor(x_scaled, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32)
    return X, y

#Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
       
        return x

features_tensor, targets_tensor = get_data()
input_size = features_tensor.shape[1]  
output_size = targets_tensor.shape[1]  
model = SimpleNN(input_size, output_size)

# Step 3: Define Loss Function and Optimizer
criterion = nn.MSELoss()  # Using Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
            print("R2-SCORE = ", metric.compute())
            

average_loss = total_loss / len(test_loader)
print(f"Average Test Loss: {average_loss}")
    

import matplotlib.pyplot as plt
plt.plot(training_loss[1:])
plt.plot(validation_loss[1:])
plt.yscale('log')





