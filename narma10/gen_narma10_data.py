"""
Generate the NARMA10 dataset once and save it to disk.

Saves four arrays to narma10/data/:
    X_train.npy  — shape (n_batch, n_time_in, n_states)
    X_test.npy   — shape (n_batch, n_time_in, n_states)
    y_train.npy  — shape (n_batch, n_time_out, 1)
    y_test.npy   — shape (n_batch, n_time_out, 1)

Run this script once before any experiment scripts.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ----------------------
# Output directory
# ----------------------
DATA_DIR = os.path.dirname(__file__)
os.makedirs(DATA_DIR, exist_ok=True)

# Fixed seed for reproducibility
np.random.seed(42)


# ----------------------
# Parameters (match your experiment)
# ----------------------
n_batch = 200          # number of sequences
n_time_in = 600        # input sequence length
n_time_out = 10        # output sequence length
n_states = 2           # number of input channels
narma_order = 10
washout = 150


# ----------------------
# NARMA10 generator
# ----------------------
def generate_narma10_data(n_samples=25000, random_seed=None, warmup=2000):
    if random_seed is not None:
        np.random.seed(random_seed)

    u = np.random.uniform(0, 0.25, size=n_samples + warmup)
    y = np.zeros(n_samples + warmup)

    for t in range(narma_order, n_samples + warmup):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * np.sum(y[t - 10:t])
            + 1.5 * u[t - 10] * u[t - 1]
            + 0.1
        )

    y = np.clip(y, 0, 5)

    u = u[warmup:].reshape(-1, 1)
    y = y[warmup:].reshape(-1, 1)

    return u, y


# ----------------------
# Dataset builder
# ----------------------
def build_dataset():
    total_required = n_batch + n_time_in + n_time_out + washout + narma_order

    u, y = generate_narma10_data(total_required)

    # --- Multi-channel input (delayed versions) ---
    features = [u]
    for k in range(1, n_states):
        delayed = np.roll(u, shift=narma_order * k)
        delayed[:narma_order * k] = u[:narma_order * k]
        features.append(delayed)

    X_multi = np.concatenate(features, axis=1)

    # --- Rolling windows ---
    X, Y = [], []
    for i in range(n_batch):
        start = washout + i
        X.append(X_multi[start : start + n_time_in])
        Y.append(y[start + n_time_in : start + n_time_in + n_time_out])

    X = np.array(X)
    Y = np.array(Y)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # --- Normalize inputs ---
    n_features = X.shape[2]
    x_scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
    X_train = x_scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # --- Normalize outputs ---
    n_out = Y.shape[2]
    y_scaler = MinMaxScaler().fit(y_train.reshape(-1, n_out))
    y_train = y_scaler.transform(y_train.reshape(-1, n_out)).reshape(y_train.shape)
    y_test  = y_scaler.transform(y_test.reshape(-1, n_out)).reshape(y_test.shape)

    return X_train, X_test, y_train, y_test


# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    print("Generating NARMA10 dataset...")

    X_train, X_test, y_train, y_test = build_dataset()

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

    print(f"Saved dataset to {DATA_DIR}")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}  y_test: {y_test.shape}")