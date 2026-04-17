"""
Generate a Lorenz multi-step dataset once and save it to disk.

Saves four arrays to lorenz/data/:
    X_train.npy  — shape (n_samples_train, n_time_in, n_states)
    X_test.npy   — shape (n_samples_test, n_time_in, n_states)
    y_train.npy  — shape (n_samples_train, n_time_out, n_states)
    y_test.npy   — shape (n_samples_test, n_time_out, n_states)

This dataset is intended for multi-step prediction only.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.dirname(__file__)

np.random.seed(42)

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0


def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def generate_lorenz_data(n_trajectories=6, n_steps=5000, dt=0.01):
    t_eval = np.linspace(0.0, dt * (n_steps - 1), n_steps)
    rng = np.random.default_rng(42)
    initials = rng.uniform(-15, 15, size=(n_trajectories, 3))
    all_trajs = []

    for init in initials:
        sol = solve_ivp(
            lorenz,
            [t_eval[0], t_eval[-1]],
            init,
            t_eval=t_eval,
            method="RK45",
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed for initial condition {init}")
        all_trajs.append(sol.y.T)

    return np.asarray(all_trajs)


def make_windows(data, n_time_in=100, n_time_out=10):
    X, y, traj_ids = [], [], []
    for traj_id, traj in enumerate(data):
        n = len(traj)
        for i in range(n - n_time_in - n_time_out + 1):
            X.append(traj[i:i + n_time_in, :])
            y.append(traj[i + n_time_in:i + n_time_in + n_time_out, :])
            traj_ids.append(traj_id)

    return np.asarray(X), np.asarray(y), np.asarray(traj_ids)


def main():
    n_trajectories = 6
    n_steps = 5000
    dt = 0.01
    n_time_in = 100
    n_time_out = 10
    test_size = 0.33

    print("Generating Lorenz dataset...")
    data = generate_lorenz_data(n_trajectories=n_trajectories, n_steps=n_steps, dt=dt)
    X, y, traj_ids = make_windows(data, n_time_in=n_time_in, n_time_out=n_time_out)

    unique_traj_ids = np.unique(traj_ids)
    train_traj, test_traj = train_test_split(unique_traj_ids, test_size=test_size, random_state=42)

    train_idx = np.isin(traj_ids, train_traj)
    test_idx = np.isin(traj_ids, test_traj)

    scaler = StandardScaler().fit(X[train_idx].reshape(-1, 3))
    X_scaled = scaler.transform(X.reshape(-1, 3)).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, 3)).reshape(y.shape)

    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]
    y_train = y_scaled[train_idx]
    y_test = y_scaled[test_idx]

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

    print(f"Saved dataset to {DATA_DIR}")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")


if __name__ == "__main__":
    main()