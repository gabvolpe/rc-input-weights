"""
Generate the Lorenz autoregressive dataset once and save it to disk.

Saves four arrays to lorenz/data/:
    X_train.npy  — shape (n_samples, 1, 3)
    X_test.npy   — shape (n_samples, 1, 3)
    y_train.npy  — shape (n_samples, 1, 3)
    y_test.npy   — shape (n_samples, 1, 3)

Each sample is a one-step transition:
    X(t) -> X(t+1)

Run this script once before any experiment scripts.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split

# ----------------------
# Output directory
# ----------------------
DATA_DIR = os.path.dirname(__file__)
os.makedirs(DATA_DIR, exist_ok=True)

# Reproducibility
np.random.seed(42)

# ----------------------
# Lorenz system
# ----------------------
sigma, beta, rho = 10, 8/3, 28


def lorenz(t, s):
    x, y, z = s
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]


def generate_lorenz(n_traj=5, n_steps=2000, dt=0.01):
    t = np.linspace(0, dt * (n_steps - 1), n_steps)

    data = []
    initials = np.random.uniform(-10, 10, (n_traj, 3))

    for x0 in initials:
        sol = solve_ivp(lorenz, [t[0], t[-1]], x0, t_eval=t)
        data.append(sol.y.T)

    return np.array(data)  # (traj, time, 3)


# ----------------------
# Build autoregressive dataset
# ----------------------
def build_autoreg_dataset(n_traj=5, n_steps=2000, dt=0.01):
    data = generate_lorenz(n_traj=n_traj, n_steps=n_steps, dt=dt)

    train_data, test_data = train_test_split(
        data, test_size=0.3, random_state=42
    )

    def make_pairs(trajs):
        X_list, y_list = [], []
        for traj in trajs:
            for t in range(len(traj) - 1):
                X_list.append(traj[t])
                y_list.append(traj[t + 1])

        X = np.array(X_list)[:, None, :]  # (samples, 1, 3)
        y = np.array(y_list)[:, None, :]
        return X, y

    X_train, y_train = make_pairs(train_data)
    X_test, y_test = make_pairs(test_data)

    return X_train, X_test, y_train, y_test


# ----------------------
# Main
# ----------------------
def main():
    print("Generating Lorenz autoregressive dataset...")

    X_train, X_test, y_train, y_test = build_autoreg_dataset(
        n_traj=5,
        n_steps=2000,
        dt=0.01
    )

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

    print(f"Saved dataset to {DATA_DIR}")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test : {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test : {y_test.shape}")


if __name__ == "__main__":
    main()