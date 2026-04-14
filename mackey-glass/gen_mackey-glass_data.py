"""
Generate the Mackey-Glass dataset once and save it to disk.

Saves four arrays to mackey-glass/data/:
    X_train.npy  — shape (n_batch, n_time_in, 2)
    X_test.npy   — shape (n_batch, n_time_in, 2)
    y_train.npy  — shape (n_batch, n_time_out, 2)
    y_test.npy   — shape (n_batch, n_time_out, 2)

Run this script once before any experiment scripts.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import brainpy as bp
import brainpy.math as bm


# ----------------------
# Output directory
# ----------------------
DATA_DIR = os.path.dirname(__file__)
os.makedirs(DATA_DIR, exist_ok=True)


# ----------------------
# Fixed seed
# ----------------------
np.random.seed(42)


# ----------------------
# Mackey-Glass generator
# ----------------------
def generate_mackey_glass_data(n_samples=30000, beta=0.2, gamma=0.1, tau=17, n=10):
    class MackeyGlassEq(bp.Dynamic):
        def __init__(self, num):
            super().__init__(num)
            self.beta, self.gamma, self.tau, self.n = beta, gamma, tau, n
            self.delay_len = int(self.tau / bm.get_dt())

            self.x = bm.Variable(bm.zeros(num))
            self.x_delay = bm.LengthDelay(
                self.x,
                delay_len=self.delay_len,
                initial_delay_data=lambda sh, dtype: 1.2 + 0.2 * (bm.random.random(sh) - 0.5)
            )
            self.x_oldest = bm.Variable(self.x_delay(self.delay_len))

            self.integral = bp.odeint(
                lambda x, t, x_tau: self.beta * x_tau / (1 + x_tau ** n) - self.gamma * x,
                method='exp_auto'
            )

        def update(self):
            self.x.value = self.integral(
                self.x.value, bp.share['t'], self.x_oldest.value, bp.share['dt']
            )
            self.x_delay.update(self.x.value)
            self.x_oldest.value = self.x_delay(self.delay_len)

    runner = bp.DSRunner(MackeyGlassEq(1), monitors=['x', 'x_oldest'])
    runner.run(n_samples * bm.get_dt())

    # Build 2D state: [x(t), x(t - tau)]
    data = np.column_stack([
        runner.mon.x.reshape(-1),
        runner.mon.x_oldest.reshape(-1)
    ])

    # Normalize BOTH channels to [-1, 1]
    for i in range(data.shape[1]):
        d = data[:, i]
        d_min, d_max = d.min(), d.max()
        data[:, i] = 2 * (d - d_min) / (d_max - d_min + 1e-12) - 1.0

    assert not np.any(np.isnan(data)), "NaN detected in Mackey-Glass data."

    return data


# ----------------------
# Sequence construction
# ----------------------
def create_dataset(n_batch=200, n_time_in=1000, n_time_out=100):
    data = generate_mackey_glass_data()

    # 2 states: [x(t), x(t - tau)]
    n_states = 2
    time_series = data[:, :n_states]

    X = np.array([
        time_series[i:i + n_time_in]
        for i in range(n_batch)
    ])

    y = np.array([
        time_series[i + n_time_in:i + n_time_in + n_time_out]
        for i in range(n_batch)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize using training statistics (per channel)
    xflat = X_train.reshape(-1, n_states)
    yflat = y_train.reshape(-1, n_states)

    X_mean, X_std = xflat.mean(axis=0), xflat.std(axis=0)
    Y_mean, Y_std = yflat.mean(axis=0), yflat.std(axis=0)

    X_train = (X_train - X_mean) / (X_std + 1e-12)
    X_test  = (X_test  - X_mean) / (X_std + 1e-12)
    y_train = (y_train - Y_mean) / (Y_std + 1e-12)
    y_test  = (y_test  - Y_mean) / (Y_std + 1e-12)

    return X_train, X_test, y_train, y_test


# ----------------------
# Generate and save
# ----------------------
print("Generating Mackey-Glass dataset (2 states)...")

X_train, X_test, y_train, y_test = create_dataset(
    n_batch=200,
    n_time_in=1000,
    n_time_out=100
)

np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

print(f"Saved dataset to {DATA_DIR}")
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}  y_test: {y_test.shape}")