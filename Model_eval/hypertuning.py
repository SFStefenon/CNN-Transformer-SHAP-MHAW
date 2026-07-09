import torch
import os
device = torch.device("cuda:0")
print(f"Using device: {device}")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

window_size = 15
epochs = 20

import pandas as pd
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
import time
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error

seed = 1
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

def performance(y_true, y_pred, time_s):
    # Calculate metrics
    rmse_v = rmse(y_true, y_pred)
    mae_v = mean_absolute_error(y_true, y_pred)
    mape_v = mean_absolute_percentage_error(y_true, y_pred)
    msle_v = mean_squared_log_error(y_true, abs(y_pred))
    # RMSE & MAE & MAPE & MSLE & time
    result = (f'{rmse_v:.2E} & {mae_v:.2E} & {mape_v:.2E} & {msle_v:.2E} & {time_s:.2E} \\\\')
    return result

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

from bayes_opt import BayesianOptimization

def objective_function(CNN_layers, num_heads, filters, kernel_size):
    CNN_layers = int(CNN_layers)
    num_heads = int(num_heads)
    filters = int(filters)
    kernel_size = int(kernel_size)

    inputs = layers.Input(shape=(window_size, features))
    x = layers.Conv1D(filters, kernel_size, activation='relu', padding='causal')(inputs)
    for _ in range(CNN_layers):
      x = layers.Conv1D(filters, kernel_size, activation='relu', padding='causal')(x)
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
    x = layers.Concatenate()([x, attention])
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
    val_mse = min(history.history["val_mse"])
    return -np.sqrt(val_mse)

# Define parameter bounds
pbounds = {
    'CNN_layers': (1, 12),
    'num_heads': (2, 5),
    'filters': (16, 256),
    'kernel_size': (2, 5),
}

# Initialize Bayesian Optimizer
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=1,
)

# Perform optimization
optimizer.maximize(init_points=5, n_iter=50)

# Get best parameters
best_params = optimizer.max
print("Best Parameters:", best_params)
Best = [int(best_params['params']['CNN_layers']), int(best_params['params']['num_heads']), int(best_params['params']['filters']), int(best_params['params']['kernel_size'])]

df = pd.DataFrame(data=Best, index=["CNN_layers", "num_heads", "filters", "kernel_size"])
df.to_csv('best_params.csv')

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
from scipy.interpolate import griddata
from matplotlib import cm

# Process optimization results
results = []
for i, res in enumerate(optimizer.res):
    # Cast parameters to correct types
    params = {
        'CNN_layers': int(res['params']['CNN_layers']),
        'num_heads': int(res['params']['num_heads']),
        'filters': res['params']['filters'],
        'kernel_size': res['params']['kernel_size'],
        'rmse': -res['target']  # Convert back to positive RMSE
    }
    results.append(params)

df = pd.DataFrame(results)

# 1. Convergence Plot
plt.figure(figsize=(5, 3))
df['best_rmse'] = df['rmse'].cummin()
plt.plot(df.index, df['best_rmse'], 'b-', label='Best RMSE')
plt.scatter(df.index, df['rmse'], c='r', alpha=0.3, label='Trials')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.grid(linestyle='--', linewidth=0.5)
#plt.title('Bayesian Optimization Convergence')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('Bayesian-Optimization-Convergence.pdf', bbox_inches='tight')
plt.show()

# Parameters to visualize
x_param = 'CNN_layers'
y_param = 'num_heads'

# Prepare data for contour plot
x = np.array([res['params'][x_param] for res in optimizer.res])
y = np.array([res['params'][y_param] for res in optimizer.res])
z = np.array([-res['target'] for res in optimizer.res])  # RMSE values

# Create grid for smooth contour plot
xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate z values using Gaussian process surrogate model
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Create plot
plt.figure(figsize=(5, 3.5))
contour = plt.contourf(xi, yi, zi, levels=150, cmap=cm.viridis)
plt.colorbar(contour, label='Loss')

# Plot actual sampled points
plt.scatter(x, y, c=z, s=50, edgecolor='black', cmap=cm.viridis,
            linewidth=0.5, label='Sampled Points')

plt.xlabel(x_param.capitalize().replace('_', ' '))
plt.ylabel(y_param.capitalize().replace('_', ' '))
plt.tight_layout()
plt.savefig('Hyperparameter-Contour-Plot1.pdf', bbox_inches='tight')
plt.show()

# Parameters to visualize
x_param = 'filters'
y_param = 'kernel_size'

# Prepare data for contour plot
x = np.array([res['params'][x_param] for res in optimizer.res])
y = np.array([res['params'][y_param] for res in optimizer.res])
z = np.array([-res['target'] for res in optimizer.res])  # RMSE values

# Create grid for smooth contour plot
xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate z values using Gaussian process surrogate model
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Create plot
plt.figure(figsize=(5, 3.5))
contour = plt.contourf(xi, yi, zi, levels=150, cmap=cm.viridis)
plt.colorbar(contour, label='Loss')

# Plot actual sampled points
plt.scatter(x, y, c=z, s=50, edgecolor='black', cmap=cm.viridis,
            linewidth=0.5, label='Sampled Points')

plt.xlabel(x_param.capitalize().replace('_', ' '))
plt.ylabel(y_param.capitalize().replace('_', ' '))
plt.tight_layout()
plt.savefig('Hyperparameter-Contour-Plot2.pdf', bbox_inches='tight')
plt.show()
