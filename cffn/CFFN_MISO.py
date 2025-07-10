#!/usr/bin/env python
# coding: utf-8

# In[217]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# In[218]:


X_scaler = StandardScaler()
y_scaler = StandardScaler()


# In[219]:


# Load data
df = pd.read_csv("../Stud_Hansa_Sim_Flight_data.csv")


# In[220]:


# Features and labels
X = df[['Alpha', 'q', 'delta_e']].values
y = df[['CD']].values
X, y


# In[221]:


X = X_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)
X, y


# In[222]:


# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# In[223]:


# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)


# In[224]:


# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)


# In[234]:


# Define the CFFN model
class CascadeFeedForwardNet(nn.Module):
    def __init__(self, input_size=3, hidden1=20, hidden2=15, output_size=1):
        super(CascadeFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(input_size + hidden1, hidden2)
        self.out = nn.Linear(input_size + hidden1 + hidden2, output_size)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))  # MATLAB-like activation
        h2_input = torch.cat([x, h1], dim=1)
        h2 = torch.tanh(self.fc2(h2_input))
        final_input = torch.cat([x, h1, h2], dim=1)
        return self.out(final_input)


# In[237]:


# Instantiate model in float64 for precision
model = CascadeFeedForwardNet().double()
criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=500, history_size=100)

# Closure required for LBFGS
def closure():
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    return loss


# Run optimization
model.train()
optimizer.step(closure)


# In[233]:


epochs = 5000
patience = 100  # how many epochs to wait before stopping if no improvement
min_delta = 1e-6  # minimum change to be considered as improvement

best_val_loss = float('inf')
trigger_times = 0
best_epoch = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    # Check for convergence
    if val_loss.item() + min_delta < best_val_loss:
        best_val_loss = val_loss.item()
        trigger_times = 0
        best_epoch = epoch + 1
    else:
        trigger_times += 1

    # Print progress
    if (epoch + 1) % 100 == 0 or trigger_times == 1:
        print(f"Epoch {epoch+1} | Train Loss: {loss.item():.10f} | Val Loss: {val_loss.item():.10f}")

    # Early stopping
    if trigger_times >= patience:
        print(f"Early stopping at epoch {epoch+1} — no improvement for {patience} consecutive epochs")
        break


# In[231]:


# Evaluation
model.eval()
with torch.no_grad():
    # Predictions
    train_preds = model(X_train)
    test_preds = model(X_test)

    # Compute MSE loss
    train_loss = criterion(train_preds, y_train)
    test_loss = criterion(test_preds, y_test)

    # Convert to numpy for sklearn R²
    train_r2 = r2_score(y_train.numpy(), train_preds.numpy())
    test_r2 = r2_score(y_test.numpy(), test_preds.numpy())

    print(f"\nTrain MSE Loss: {train_loss.item():.10f}, R²: {train_r2:.10f}")
    print(f"Test  MSE Loss: {test_loss.item():.10f}, R²: {test_r2:.10f}")


# In[216]:


torch.save(model.state_dict(), "cffn_model.pth")


# In[ ]:




