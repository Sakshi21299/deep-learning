import os
import pandas as pd
import torch
import numpy as np

def get_data(data_path, features_file = "zeopp.csv" , label_file = "N2_SSL_R299up.csv", column_name = "REFCODE"):
    path = data_path
    
    #Read features
    file_path = os.path.join(path, features_file)
    df_x = pd.read_csv(file_path)
    
    #Read labels
    file_path = os.path.join(path, label_file)
    df_y = pd.read_csv(file_path)
    
    filtered_df_x = df_x[df_x[column_name].isin(df_y[column_name])]
    
    # Sort both DataFrames based on the 'ChildName' column
    dfx_sorted = filtered_df_x.sort_values(by=column_name)
    dfy_sorted = df_y.sort_values(by=column_name)
    
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
    y = torch.tensor(y_scaled, dtype=torch.float32)
    return X, y

    