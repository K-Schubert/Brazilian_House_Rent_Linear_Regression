import torch
import torch.nn as nn
import numpy as np
import pandas as pd

torch.manual_seed(42)

'''
- Predictors (8): nb rooms, nb bathrooms, nb parking spaces, floor, 
homeowners association tax ($R), rent amount ($R), property tax ($R), fire insurance ($R).
- Target (1): total rent ($R).
- Model: Multiple Linear Regression trained with mini-batch SGD.
- Remarks: The linear regression assumptions are not thorougly checked, thus resulting model quality is not optimal. 
Features are normalized with scale/location centering. SGD hyper-parameters are not tuned to be optimal.
'''

# Data Processing
x = pd.read_csv("./Data/houses_to_rent_v2.csv", sep=',', header=0)
y = x.iloc[:, -1]
y = y.apply(pd.to_numeric, errors='coerce')
x = x.iloc[:, [2,3,4,5,8,9,10,11]]
x = x.apply(pd.to_numeric, errors='coerce')

# Remove NaNs
y = y[~x['floor'].isin(['NaN'])]
x = x[~x['floor'].isin(['NaN'])]

# Data Normalization
from sklearn import preprocessing

'''
x = preprocessing.scale(x, axis=0)
x = pd.DataFrame(x)

y = preprocessing.scale(y, axis=0)
y = pd.DataFrame(y)
'''

scaler_x = preprocessing.StandardScaler().fit(x)
x = scaler_x.transform(x)
y = y.values.reshape(-1,1)
scaler_y = preprocessing.StandardScaler().fit(y)
y = scaler_y.transform(y)

x = torch.from_numpy(x).type(torch.FloatTensor).view(-1,8)
y = torch.from_numpy(y).type(torch.FloatTensor).view(-1,1)

from torch.utils.data import TensorDataset

# TensorDataset allows to access a small portion of the training data with indexing as a tuple (x, y)
trn = TensorDataset(x, y)

from torch.utils.data import DataLoader

# DataLoader splits the data into batches of predefined size and allows to shuffle/sample the data
batch_size = 500
trn_dl = DataLoader(trn, batch_size, shuffle=True)

# Model Preparation
# Define model with nn.Linear (initializes weights and biases automatically)
model = nn.Linear(8, 1)

# Package contains loss functions
import torch.nn.functional as F

# Define loss function
loss_fn = F.mse_loss

# Define optimizer (SGD)
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Function to train the model
def fit(num_epochs, model, loss_fn, opt, trn_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in trn_dl:
            
            # 1. Generate predictions
            yhat = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(yhat, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Training
fit(100, model, loss_fn, opt, trn_dl)

# Make Predictions
yhat = model(x)

# Inverse Transform
yhat = scaler_y.inverse_transform(yhat.detach())
y = scaler_y.inverse_transform(y)

# Plot fitted values against targets (should sit on 45° line if the fitted values are close to the targets)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
ax.plot(y, yhat , 'o', alpha=0.5)
xx = np.linspace(*ax.get_xlim())
ax.plot(xx, xx, '--')
plt.show()