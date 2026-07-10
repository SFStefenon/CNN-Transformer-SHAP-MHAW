import torch
import os
device = torch.device("cuda:0")
print(f"Using device: {device}")

window_size = 15
epochs = 100

import pandas as pd
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

def performance(y_true, y_pred, time_s):
    # Calculate metrics
    rmse_v = rmse(y_true, y_pred)
    mae_v = mean_absolute_error(y_true, y_pred)
    mape_v = mean_absolute_percentage_error(y_true, y_pred)
    msle_v = mean_squared_log_error(y_true, abs(y_pred))
    # RMSE & MAE & MAPE & MSLE
    result = (f'{rmse_v:.2E} & {mae_v:.2E} & {mape_v:.2E} & {msle_v:.2E} & {time_s:.2E}\\\\')
    return result

CNN_layers = 3
num_heads = 4
filters = 238
kernel_size = 4

df = pd.read_csv("tucurui.csv", sep=";")
df.columns = [col.strip() for col in df.columns]
df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)
df["UPH610010000"] = df["UPH610010000"].str.replace(",", ".").astype(float)
df["Natural Flow"] = df["Natural Flow"].str.replace(",", ".").astype(float)

df = df.sort_values("Data").reset_index(drop=True)
df["time_idx"] = df.index
df["group"] = "tucurui"
df = df.rename(columns={"Natural Flow": "y", "UPH610010000": "precipitation"})
#data = df[["y", "precipitation"]]
data = df[["y"]]

# convert y and precipitation to numpy arrays
data = data.to_numpy()
features = data.shape[1]

# create_dataset function
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

# Load Dataset
X, y = create_dataset(data, window_size)

# Split into train/test sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, features)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, features)).reshape(X_test.shape)
y_train = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Reshape inputs for Conv1D (samples, timesteps, features)
X_train = X_train.reshape((-1, window_size, features))
X_test = X_test.reshape((-1, window_size, features))

inputs = layers.Input(shape=(window_size, features))
x = layers.Conv1D(filters, kernel_size, activation='relu', padding='causal')(inputs)
for _ in range(CNN_layers):
  x = layers.Conv1D(filters, kernel_size, activation='relu', padding='causal')(x)
attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
x = layers.Concatenate()([x, attention])
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)
model = models.Model(inputs, outputs)
start = time.time()
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                    validation_split=0.2, verbose=1)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {test_loss[1]:.4f}")
test_part = X_test[0].reshape(1, window_size, features)
test_predictions = model.predict(test_part)
test_part = np.append(test_part, test_predictions)
end = time.time()
time_s = end - start

# Adding window size to the predictions
for i in range(1, window_size):
    test_predictions = model.predict(test_part[-window_size:].reshape(1, window_size, features))
    test_part = np.append(test_part, test_predictions)
test_part = test_part[-window_size:].flatten()

# Calculate performance metrics
test_part = scaler.inverse_transform(test_part.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
X_test, y_test = X[split:], y[split:]
y_true = y_test[:window_size]
y_pred = test_part

# Calculate performance metrics
performance_metrics = performance(y_test[:window_size], test_part, time_s)
print(f"Performance metrics: {performance_metrics}")
