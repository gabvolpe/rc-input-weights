"""
Lorenz RC - Autoregressive with 1-step prediction

Adds:
1. Saving rollout MSE results
2. Saving read-in weights
3. Saving reservoir weights

Notes:
- This version fixes trajectory alignment for rollout evaluation.
- It evaluates autoregressive forecasting on contiguous Lorenz trajectories.
- It scales state variables using StandardScaler fit on training trajectories only.
- It saves one reservoir per outer trial and read-in weights per inner trial and distribution.
"""

import os
import numpy as np
import time
import argparse
import pickle
import threading
import concurrent.futures

from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.optimizers import RidgeSK


# ----------------------
# Output
# ----------------------
OUTPUT_DIR = "lorenz/outputs/fixed-reservoir"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------
# Args
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=2)
parser.add_argument("--n_inner", type=int, default=3)
parser.add_argument("--reservoir_nodes", type=int, default=300)
parser.add_argument("--density", type=float, default=0.1)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.5)
parser.add_argument("--fraction_input", type=float, default=1.0)
parser.add_argument("--ridge_alpha", type=float, default=1e-3)
parser.add_argument("--rollout_steps", type=int, default=5)
args = parser.parse_args()


# ----------------------
# Lorenz system
# ----------------------
sigma, beta, rho = 10, 8/3, 28


def lorenz(t, s):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def generate_lorenz(n_traj=5, n_steps=2000, dt=0.01):
    t = np.linspace(0, dt * (n_steps - 1), n_steps)
    data = []
    initials = np.random.uniform(-10, 10, (n_traj, 3))
    for x0 in initials:
        sol = solve_ivp(lorenz, [t[0], t[-1]], x0, t_eval=t)
        data.append(sol.y.T)
    return np.array(data)  # (traj, time, 3)


# ----------------------
# AUTOREGRESSIVE DATA
# ----------------------
def lorenz_autoreg(n_traj=5, n_steps=2000, dt=0.01):
    """
    Build a one-step autoregressive dataset from contiguous Lorenz trajectories.

    Each sample is:
        X_t -> X_{t+1}

    Returns:
        X_train, X_test, y_train, y_test, traj_train, traj_test, scaler
    where X/y are shaped (samples, 1, 3).
    """
    data = generate_lorenz(n_traj=n_traj, n_steps=n_steps, dt=dt)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    X_train_list, y_train_list, train_ids = [], [], []
    X_test_list, y_test_list, test_ids = [], [], []

    for traj_id, traj in enumerate(train_data):
        for t in range(len(traj) - 1):
            X_train_list.append(traj[t])
            y_train_list.append(traj[t + 1])
            train_ids.append(traj_id)

    for traj_id, traj in enumerate(test_data):
        for t in range(len(traj) - 1):
            X_test_list.append(traj[t])
            y_test_list.append(traj[t + 1])
            test_ids.append(traj_id)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    y_train = scaler.transform(y_train)
    X_test = scaler.transform(X_test)
    y_test = scaler.transform(y_test)

    X_train = X_train[:, None, :]
    y_train = y_train[:, None, :]
    X_test = X_test[:, None, :]
    y_test = y_test[:, None, :]

    return X_train, X_test, y_train, y_test, scaler


def get_test_trajectory(n_traj=5, n_steps=2000, dt=0.01):
    """
    Build one fresh Lorenz test trajectory for rollout evaluation.
    """
    data = generate_lorenz(n_traj=n_traj, n_steps=n_steps, dt=dt)
    _, test_data = train_test_split(data, test_size=0.3, random_state=42)
    return test_data[0]


# ----------------------
# MODEL
# ----------------------
def create_model():
    model = RC()

    # RC expects (time_steps, features)
    model.add(InputLayer(input_shape=(1, 3)))

    reservoir = RandomReservoirLayer(
        nodes=args.reservoir_nodes,
        density=args.density,
        activation="tanh",
        spec_rad=args.spectral_radius,
        leakage_rate=args.leakage_rate,
        fraction_input=args.fraction_input
    )

    model.add(reservoir)

    # Predict the next 3D state
    model.add(ReadoutLayer(output_shape=(1, 3), fraction_out=1.0))

    model.compile(
        optimizer=RidgeSK(alpha=args.ridge_alpha),
        metrics=["mse"]
    )

    return model, reservoir


# ----------------------
# WEIGHTS
# ----------------------
def create_weights(method):
    shape = (args.reservoir_nodes, 3)

    if method == "uniform":
        return np.random.uniform(-1, 1, size=shape)

    if method == "gaussian":
        return np.random.normal(0, 1, size=shape)

    if method == "double_gaussian":
        g1 = np.random.normal(-1.5, 0.5, size=shape)
        g2 = np.random.normal(1.5, 0.5, size=shape)
        mask = np.random.rand(*shape) > 0.5
        return np.where(mask, g1, g2)

    if method == "laplace":
        return np.random.laplace(0, 0.5, size=shape)

    if method == "powerlaw":
        return np.random.power(2.0, size=shape) * np.sign(np.random.randn(*shape))

    raise ValueError(f"Unknown method: {method}")


# ----------------------
# ROLLOUT
# ----------------------
def rollout(model, x0, steps=200):
    x = x0.copy()
    preds = []
    for _ in range(steps):
        y = model.predict(x[None, None, :])[0, 0]
        preds.append(y)
        x = y
    return np.array(preds)


def rollout_mse(model, true_traj, scaler, steps=200):
    """
    Compare recursive prediction against the corresponding true continuation
    from the same trajectory.
    """
    steps = min(steps, len(true_traj) - 1)
    x0 = scaler.transform(true_traj[0][None, :])[0]
    true = scaler.transform(true_traj[1:steps + 1])
    pred = rollout(model, x0, steps=steps)
    return float(np.mean((pred - true) ** 2))


# ----------------------
# INNER TRIAL
# ----------------------
def run_inner(model_serial, X_train, y_train, true_rollout_traj, scaler, method):
    model = pickle.loads(model_serial)

    W = create_weights(method)
    model._set_readin_weights(W)

    model.fit(X_train, y_train)

    mse = rollout_mse(model, true_rollout_traj, scaler, steps=args.rollout_steps)

    return mse, W


# ----------------------
# MAIN
# ----------------------
def main():
    np.random.seed(42)
    start = time.time()

    X_train, X_test, y_train, y_test, scaler = lorenz_autoreg()

    # Use a fresh, aligned trajectory for rollout evaluation
    true_rollout_traj = get_test_trajectory()

    distributions = ["uniform", "gaussian", "double_gaussian", "laplace", "powerlaw"]

    readin_records = {d: [] for d in distributions}
    reservoir_records = []
    results = []

    lock = threading.Lock()

    for outer in range(args.n_trials):
        print(f"Outer {outer + 1}")

        model, reservoir = create_model()
        model_serial = pickle.dumps(model)

        reservoir_records.append((outer + 1, reservoir.weights.flatten().copy()))

        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = [
                ex.submit(
                    run_inner,
                    model_serial,
                    X_train,
                    y_train,
                    true_rollout_traj,
                    scaler,
                    dist
                )
                for dist in distributions
                for _ in range(args.n_inner)
            ]

            for i, f in enumerate(futures):
                mse, W = f.result()
                dist = distributions[i // args.n_inner]
                with lock:
                    readin_records[dist].append((outer + 1, W.flatten().copy()))
                    results.append((outer + 1, dist, mse))

    # ----------------------
    # SAVE RESULTS
    # ----------------------
    results_arr = np.array(results, dtype=object)
    np.save(os.path.join(OUTPUT_DIR, "sc1_results_fixed-reservoir.npy"), results_arr)

    # ----------------------
    # SAVE READ-IN WEIGHTS
    # ----------------------
    for dist, rec in readin_records.items():
        arr = np.array([
            np.concatenate(([outer], w))
            for outer, w in rec
        ], dtype=object)
        np.save(os.path.join(OUTPUT_DIR, f"sc1_readin_weights_{dist}.npy"), arr)

    # ----------------------
    # SAVE RESERVOIR
    # ----------------------
    arr = np.array([
        np.concatenate(([outer], w))
        for outer, w in reservoir_records
    ], dtype=object)
    np.save(os.path.join(OUTPUT_DIR, "sc1_reservoir_weights.npy"), arr)

    print("\nSaved ALL required files.")
    print("Done in", time.time() - start)


if __name__ == "__main__":
    main()