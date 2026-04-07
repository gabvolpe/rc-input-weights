"""
Generate the sine-to-cosine² dataset once and save it to disk.

Saves four arrays to sin-to-cos2/data/:
    X_train.npy  — shape (n_batch, n_time, n_states)
    X_test.npy   — shape (n_batch, n_time, n_states)
    y_train.npy  — shape (n_batch, n_time, n_states)
    y_test.npy   — shape (n_batch, n_time, n_states)

Run this script once before any experiment scripts.
"""

import os
import numpy as np
from pyreco.utils_data import sequence_to_sequence

DATA_DIR = os.path.dirname(__file__)

# Fixed seed for reproducibility
np.random.seed(42)

n_samples = 200  # Number of sequences
n_timesteps = 1000  # Length of each sequence
n_states = 1  # Number of features (sine and cosine² are 1

print("Generating sine-to-cosine² dataset...")
X_train, X_test, y_train, y_test = sequence_to_sequence(
    name="sin_to_cos2", n_states=n_states, n_batch=n_samples, n_time=n_timesteps
)

np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

print(f"Saved dataset to {DATA_DIR}")
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}  y_test: {y_test.shape}")
