"""
Lorenz RC - Fixed Read-in / Variable Reservoir (TRUE AUTOREGRESSIVE)

Structure:
- OUTER trials = fixed READ-IN matrices (5 distributions)
- INNER trials = different RANDOM reservoirs

Evaluation:
- One-step prediction (teacher forced)
- Autoregressive rollout prediction (closed loop)

Outputs (UNCHANGED FORMAT):
1) sc1_readin_weights_{dist}.npy
2) sc1_reservoir_weights.npy
3) sc1_results_fixed-readin.npy
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
# OUTPUT
# ----------------------
OUTPUT_DIR = "lorenz/outputs/fixed-readin"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------
# ARGUMENTS
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
# LORENZ SYSTEM
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

    return np.array(data)


# ----------------------
# AUTOREGRESSIVE DATA (ONE-STEP TRAINING)
# ----------------------
def lorenz_autoreg(n_traj=6, n_steps=2000, dt=0.01):
    data = generate_lorenz(n_traj, n_steps, dt)

    train, test = train_test_split(data, test_size=0.3, random_state=42)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for traj in train:
        for t in range(len(traj) - 1):
            X_train.append(traj[t])
            y_train.append(traj[t + 1])

    for traj in test:
        for t in range(len(traj) - 1):
            X_test.append(traj[t])
            y_test.append(traj[t + 1])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)[:, None, :]
    y_train = scaler.transform(y_train)[:, None, :]
    X_test = scaler.transform(X_test)[:, None, :]
    y_test = scaler.transform(y_test)[:, None, :]

    return X_train, X_test, y_train, y_test, scaler


# ----------------------
# MODEL
# ----------------------
def create_model():
    model = RC()

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

    model.add(ReadoutLayer(output_shape=(1, 3), fraction_out=1.0))

    model.compile(
        optimizer=RidgeSK(alpha=args.ridge_alpha),
        metrics=["mse"]
    )

    return model, reservoir


# ----------------------
# READ-IN WEIGHTS
# ----------------------
def create_weights(method):
    shape = (args.reservoir_nodes, 3)

    if method == "uniform":
        return np.random.uniform(-1, 1, shape)

    if method == "gaussian":
        return np.random.normal(0, 1, shape)

    if method == "double_gaussian":
        g1 = np.random.normal(-1.5, 0.5, shape)
        g2 = np.random.normal(1.5, 0.5, shape)
        mask = np.random.rand(*shape) > 0.5
        return np.where(mask, g1, g2)

    if method == "laplace":
        return np.random.laplace(0, 0.5, shape)

    if method == "powerlaw":
        return np.random.power(2.0, shape) * np.sign(np.random.randn(*shape))

    raise ValueError(method)


# ----------------------
# AUTOREGRESSIVE ROLLOUT (CRITICAL FIX)
# ----------------------
def rollout(model, x0, steps):
    x = x0.copy()
    preds = []

    for _ in range(steps):
        y = model.predict(x[None, None, :])[0, 0]
        preds.append(y)
        x = y  # closed loop

    return np.array(preds)


def evaluate_autoregressive(model, traj, scaler, steps):
    """
    True autoregressive evaluation:
    - start from first point of trajectory
    - recursively predict future
    - compare against true continuation
    """

    steps = min(steps, len(traj) - 1)

    x0 = scaler.transform(traj[0][None, :])[0]

    true = scaler.transform(traj[1:steps + 1])
    pred = rollout(model, x0, steps)

    return float(np.mean((pred - true) ** 2))


# ----------------------
# ONE-STEP LOSS
# ----------------------
def one_step_mse(model, X, y):
    pred = model.predict(X)
    return float(np.mean((pred - y) ** 2))


# ----------------------
# INNER TRIAL
# ----------------------
def run_inner_trial(
    X_train, y_train, X_test, y_test,
    true_traj,
    trial_outer,
    trial_inner,
    readin_mats,
    reservoir_records,
    lock
):
    outer_id = trial_outer + 1
    inner_id = trial_inner + 1

    model, reservoir = create_model()

    with lock:
        reservoir_records.append(
            (outer_id, inner_id, reservoir.weights.flatten().copy())
        )

    def run(W):
        m = pickle.loads(pickle.dumps(model))
        m._set_readin_weights(W)
        m.fit(X_train, y_train)

        loss_1 = one_step_mse(m, X_test, y_test)
        loss_roll = evaluate_autoregressive(
            m,
            true_traj,
            scaler_global,
            args.rollout_steps
        )

        return 0.5 * loss_1 + 0.5 * loss_roll

    return (
        run(readin_mats["uniform"]),
        run(readin_mats["gaussian"]),
        run(readin_mats["double_gaussian"]),
        run(readin_mats["laplace"]),
        run(readin_mats["powerlaw"]),
    )


# ----------------------
# GLOBAL SCALER HOLDER (needed for rollout consistency)
# ----------------------
scaler_global = None


# ----------------------
# MAIN
# ----------------------
def main():
    global scaler_global

    np.random.seed(42)
    start = time.time()

    X_train, X_test, y_train, y_test, scaler = lorenz_autoreg()
    scaler_global = scaler

    # true trajectory for autoregressive rollout alignment
    true_traj = generate_lorenz(n_traj=6, n_steps=2000)[0]

    dtype = np.dtype([
        ("outer", "i4"),
        ("gt_uniform", "f8"), ("pred_uniform", "f8"),
        ("gt_gauss", "f8"), ("pred_gauss", "f8"),
        ("gt_dbgauss", "f8"), ("pred_dbgauss", "f8"),
        ("gt_laplace", "f8"), ("pred_laplace", "f8"),
        ("gt_powlaw", "f8"), ("pred_powlaw", "f8"),
    ])

    results = []

    readin_records = {
        k: [] for k in
        ["uniform", "gaussian", "double_gaussian", "laplace", "powerlaw"]
    }

    reservoir_records = []
    lock = threading.Lock()

    for outer in range(args.n_trials):
        print(f"Outer {outer+1}")

        readin_mats = {
            k: create_weights(k)
            for k in readin_records.keys()
        }

        for k, v in readin_mats.items():
            readin_records[k].append((outer + 1, v.flatten().copy()))

        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = [
                ex.submit(
                    run_inner_trial,
                    X_train, y_train, X_test, y_test,
                    true_traj,
                    outer, i,
                    readin_mats,
                    reservoir_records,
                    lock
                )
                for i in range(args.n_inner)
            ]

            for f in futures:
                u, g, d, l, p = f.result()
                results.append((outer + 1, u, u, g, g, d, d, l, l, p, p))

    # ----------------------
    # SAVE RESULTS
    # ----------------------
    np.save(
        os.path.join(OUTPUT_DIR, "sc1_results_fixed-readin.npy"),
        np.array(results, dtype=object)
    )

    # ----------------------
    # SAVE READ-IN
    # ----------------------
    for k, rec in readin_records.items():
        arr = np.array([np.concatenate(([o], w)) for o, w in rec], dtype=object)
        np.save(os.path.join(OUTPUT_DIR, f"sc1_readin_weights_{k}.npy"), arr)

    # ----------------------
    # SAVE RESERVOIR
    # ----------------------
    arr = np.array([
        np.concatenate(([o, i], w))
        for o, i, w in reservoir_records
    ], dtype=object)

    np.save(os.path.join(OUTPUT_DIR, "sc1_reservoir_weights.npy"), arr)

    print("Done in", time.time() - start)


if __name__ == "__main__":
    main()