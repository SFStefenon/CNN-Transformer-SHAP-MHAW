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

def performance(y_true, y_pred):
    # Calculate metrics
    rmse_v = rmse(y_true, y_pred)
    mae_v = mean_absolute_error(y_true, y_pred)
    mape_v = mean_absolute_percentage_error(y_true, y_pred)
    msle_v = mean_squared_log_error(y_true, abs(y_pred))
    # RMSE & MAE & MAPE & MSLE
    result = (f'{rmse_v:.2E} & {mae_v:.2E} & {mape_v:.2E} & {msle_v:.2E} \\\\')
    return result

CNN_layers = 3
num_heads = 4
filters = 238
kernel_size = 4

rmse_v =[]
mae_v = []
mape_v = []
msle_v = []

for i in range(100):
    print(f"Run {i}: seed = {i}")
    seed = i
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

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
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                        validation_split=0.2, verbose=1)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {test_loss[1]:.4f}")
    test_part = X_test[0].reshape(1, window_size, features)
    test_predictions = model.predict(test_part)
    test_part = np.append(test_part, test_predictions)
    
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
    
    rmse_v.append(rmse(y_true, y_pred))
    mae_v.append(mean_absolute_error(y_true, y_pred))
    mape_v.append(mean_absolute_percentage_error(y_true, y_pred))
    msle_v.append(mean_squared_log_error(y_true, abs(y_pred)))
    
    # Calculate performance metrics
    performance_metrics = performance(y_test[:window_size], test_part)
    print(f"Performance metrics: {performance_metrics}")

pd.DataFrame(rmse_v).to_csv('rmse_v1.csv', index=False)
pd.DataFrame(mae_v).to_csv('mae_v1.csv', index=False)
pd.DataFrame(mape_v).to_csv('mape_v1.csv', index=False)
pd.DataFrame(msle_v).to_csv('msle_v1.csv', index=False)

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

rmse = np.array(pd.read_csv('rmse_v1.csv')).flatten()
mae = np.array(pd.read_csv('mae_v1.csv')).flatten()
mape = np.array(pd.read_csv('mape_v1.csv')).flatten()
msle = np.array(pd.read_csv('msle_v1.csv')).flatten()

# Compute detailed statistics
def compute_stats(values):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    return {
        'Mean': np.mean(values),
        'Std': np.std(values),
        'Min': np.min(values),
        'Max': np.max(values),
        'Median': np.median(values),
        'Q1 (25\\%)': q1,
        'Q3 (75\\%)': q3,
        'Range': np.max(values) - np.min(values),
        'IQR': q3 - q1,
        'Skewness': skew(values),
        'Kurtosis': kurtosis(values),
    }

rmse_stats = compute_stats(rmse)
mae_stats = compute_stats(mae)
mape_stats = compute_stats(mape)
msle_stats = compute_stats(msle)

# Prepare LaTeX table (metrics as columns, stats as rows)
latex_table = r'''
\begin{table}[!ht]
\centering
\caption{Extended Statistical Results over 50 Runs (Engineering Notation)}
\begin{tabular}{lccc}
\toprule
Statistic & RMSE & MAE & MAPE & MSLE \\
\midrule
'''

for stat in rmse_stats.keys():
    latex_table += f"{stat} & {rmse_stats[stat]:.4e} & {mae_stats[stat]:.4e} & {mape_stats[stat]:.4e} & {msle_stats[stat]:.4e} \\\\\n"

latex_table += r'''\bottomrule
\end{tabular}
\label{tab:extended_stats_eng}
\end{table}
'''

print(latex_table)

df = pd.DataFrame({
    'RMSE': rmse,
    'MAE': mae,
    'MAPE': mape,
    'MSLE': msle
})

plt.figure(figsize=(5, 3), facecolor='white')  # Pure white background
ax = plt.gca()
ax.set_facecolor('white')  # White plot area

# Create high-contrast boxplot with clear elements
bp = plt.boxplot([df[col] for col in df.columns], 
                 labels=df.columns,
                 patch_artist=True,  # Allow style modifications
                 boxprops=dict(facecolor='white', color='black', linewidth=1),
                 whiskerprops=dict(color='black', linestyle='-', linewidth=1),
                 capprops=dict(color='black', linewidth=1),
                 medianprops=dict(color='red', linewidth=1),
                 flierprops=dict(marker='o', markersize=3,
                               markerfacecolor='black', markeredgecolor='none'))

plt.ylabel('Values', fontsize=10, labelpad=8, fontweight='medium')
plt.grid(True, linestyle='--', linewidth=0.7, color='#e0e0e0')

# Clean axis styling
plt.xticks(fontsize=9, rotation=0, ha='center')
plt.yticks(fontsize=9)
plt.tick_params(axis='both', which='major', length=4, width=0.8, color='#404040')

# Top and right borders
ax.spines['top'].set_color('#808080')
ax.spines['right'].set_color('#808080')
ax.spines['bottom'].set_color('#808080')
ax.spines['left'].set_color('#808080')

plt.tight_layout(pad=2)
plt.savefig('sta1a.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Line plot
plt.figure(figsize=(5, 3))
plt.plot(df, marker='o')
#plt.title('Line Plot of 3 Variables over 50 Runs')
plt.xlabel('Run')
plt.ylabel('Value')
plt.legend(df.columns)
plt.grid(True)
plt.savefig('sta2a.pdf', bbox_inches='tight')
plt.show()
