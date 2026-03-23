"""
Lorenz
Unconditional Variability Extraction with fixed reservoir.
For each fixed reservoir, many inner trials are run, each with different read-in matrices sampled from the five distributions.

Set Contraint 3: input without masking 100%, zero or near-zero read-in values allowed.

1) The reservoir weights are saved in the corresponding .npy, in the form: outer_ID (reservoir_ID), reservoir_weights
2) The readin weights are saved in the corresponding .npy, in the form:
   outer_ID (reservoir_ID), inner_ID (read-in_ID), readin_weights
3) The results (predictions and ground truths) are saved in the corresponding .npy, in the form:
   outer_ID (reservoir_ID), gt-uniform, pred-uniform, gt-gauss, pred-gauss, gt-dbgauss, pred-dbgauss,
   gt-laplace, pred-laplace, gt-powerlaw, pred-powerlaw
"""


import os
import numpy as np
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_data import sequence_to_sequence
from pyreco.optimizers import RidgeSK
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
import brainpy as bp
import brainpy.math as bm
import concurrent.futures
import pickle
import matplotlib.pyplot as plt
import threading


# ----------------------
# Output directories
# ----------------------
OUTPUT_DIR = "lorenz/outputs/fixed-reservoir"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------
# --- User Arguments ---
# ----------------------
parser = argparse.ArgumentParser(description="Run RC model with customizable hyperparameters.")

parser.add_argument("--n_trials", type=int, default=2,
                    help="Number of outer trials, each trial has a different fixed reservoir")
parser.add_argument("--n_inner", type=int, default=3,
                    help="Number of inner trials, each inner trial with different read-ins from the 5 distributions")

parser.add_argument("--reservoir_nodes", type=int, default=300)
parser.add_argument("--density", type=float, default=0.1)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.5)
parser.add_argument("--fraction_input", type=float, default=1.0,
                    help="Percentage of input without masking")

parser.add_argument("--ridge_alpha", type=float, default=1e-3)

parser.add_argument("--set_threshold", type=bool, default=False,
                    help="Set to False if no threshold wished")
parser.add_argument("--readin_threshold", type=float, default=1e-3)

parser.add_argument(
    "--sd_list",
    type=str,
    default="[0.1, 0.25, 0.5, 0.75, 1.0]",
    help="List of Gaussian SD values, format: [0.1,0.25,...]"
)
parser.add_argument(
    "--task",
    type=str,
    default="sequence_to_sequence_autoregressive",
    choices=["sequence_to_sequence_autoregressive", "sequence_to_sequence_non_autoregressive"]
)

parser.add_argument(
    "--constraint_set",
    type=str, default="1",
    choices=["1", "2", "3"],
    help="(Unused now; only kept for compatibility)"
)

# Parse
args = parser.parse_args()
sd_list = eval(args.sd_list)


# ----------------------
# --- Functions --------
# ----------------------
def create_base_model(input_shape, output_shape):
    model_rc = RC()
    model_rc.add(InputLayer(input_shape=input_shape))
    reservoir_layer = RandomReservoirLayer(
        nodes=args.reservoir_nodes,
        density=args.density,
        activation="tanh",
        spec_rad=args.spectral_radius,
        leakage_rate=args.leakage_rate,
        fraction_input=args.fraction_input
    )
    model_rc.add(reservoir_layer)
    model_rc.add(ReadoutLayer(output_shape, fraction_out=1.0))
    optim = RidgeSK(alpha=args.ridge_alpha)
    model_rc.compile(optimizer=optim, metrics=["mean_squared_error"])
    return model_rc, reservoir_layer

sigma, beta, rho = 10, 8/3, 28

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def generate_lorenz_data(n_trajectories=3, n_steps=2000, dt=0.01): #trajectories are number of independent Lorenz system trajectories (simulations with different initial conditions). Each being one complete dynamical evolution of the L. system starting from a distinct random initial state.
    t_eval = np.linspace(0, dt * (n_steps - 1), n_steps)
    all_trajs = []
    rng = np.random.default_rng(42)
    initials = rng.uniform(-15, 15, size=(n_trajectories, 3))
    for init in initials:
        sol = solve_ivp(lorenz, [t_eval[0], t_eval[-1]], init, t_eval=t_eval, method="RK45")
        all_trajs.append(sol.y.T)
    return np.array(all_trajs) # [num_traj, n_steps, 3]

def lorenz_pred(
    n_trajectories=6, n_steps=2000, dt=0.01,
    n_time_in=10, n_time_out=5, n_states=3
):
    data = generate_lorenz_data(n_trajectories, n_steps, dt)
    X, y, traj_ids = [], [], []
    for idx, traj in enumerate(data):
        for i in range(len(traj) - n_time_in - n_time_out):
            X.append(traj[i:i+n_time_in, :n_states])
            y.append(traj[i+n_time_in:i+n_time_in+n_time_out, :n_states])
            traj_ids.append(idx)
    X, y, traj_ids = np.array(X), np.array(y), np.array(traj_ids)

    # --- Improved train/test split: different trajectories only! ---
    unique_traj_ids = np.unique(traj_ids)
    train_traj, test_traj = train_test_split(
        unique_traj_ids, test_size=0.33, random_state=42
    )
    train_idx = np.isin(traj_ids, train_traj)
    test_idx = np.isin(traj_ids, test_traj)
    # --- Normalize on train only, then apply to test ---
    scaler = StandardScaler().fit(X[train_idx].reshape(-1, n_states))
    X = scaler.transform(X.reshape(-1, n_states)).reshape(X.shape)
    y = scaler.transform(y.reshape(-1, n_states)).reshape(y.shape)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], scaler

# --- Autoregressive and non autoregressive ---
def sequence_to_sequence_autoregressive(
    n_batch=2000, n_trajectories=6, dt=0.01,
    n_states=3, n_time_in=100, n_time_out=10
):
    X_train, X_test, y_train, y_test, scaler = lorenz_pred(
        n_trajectories=n_trajectories, n_steps=n_batch, dt=dt,
        n_time_in=n_time_in, n_time_out=n_time_out, n_states=n_states
    )
    return X_train, X_test, y_train, y_test

def sequence_to_sequence_non_autoregressive(
    n_batch=2000, n_trajectories=6, dt=0.01,
    n_states=3, n_time_in=100, n_time_out=1
):
    X_train, X_test, y_train, y_test, scaler = lorenz_pred(
        n_trajectories=n_trajectories, n_steps=n_batch, dt=dt,
        n_time_in=n_time_in, n_time_out=n_time_out, n_states=n_states
    )
    return X_train, X_test, y_train, y_test

task_functions = {
    "sequence_to_sequence_autoregressive": sequence_to_sequence_autoregressive,
    "sequence_to_sequence_non_autoregressive": sequence_to_sequence_non_autoregressive
}

# Map string to function
#args.task = task_functions[args.task]


def create_weights(shape, method, dynamic_sd=None):
    # Ensure shape is always [nodes, n_states_in = 3]
    nodes = args.reservoir_nodes
    if len(shape) == 2:
        n_states = shape[1]
        shape = (nodes, n_states)
    else:
        shape = (nodes, 2)

    if args.set_threshold:
        threshold = args.readin_threshold
        if method == "random_uniform":
            return np.random.uniform(-1, 1, size=shape)
        if method == "random_normal":
            mu = 0.0
            sd = dynamic_sd if dynamic_sd is not None else 1.0
            w = np.random.normal(mu, sd, size=shape)
            while np.any(np.abs(w) < threshold):
                idx = np.abs(w) < threshold
                w[idx] = np.random.normal(mu, sd, size=np.sum(idx))
            return w
        if method == "double_gaussian":
            mu1, mu2 = -1.5, 1.5
            sigma1, sigma2 = dynamic_sd, dynamic_sd
            amp1, amp2 = 0.5, 0.5
            n_elements = np.prod(shape)
            choices = np.random.choice([0, 1], size=n_elements, p=[amp1, amp2])
            g1 = np.random.normal(mu1, sigma1, size=n_elements)
            g2 = np.random.normal(mu2, sigma2, size=n_elements)
            w = np.where(choices == 0, g1, g2)
            while np.any(np.abs(w) < threshold):
                idx = np.abs(w) < threshold
                n_idx = np.sum(idx)
                g1_new = np.random.normal(mu1, sigma1, size=n_idx)
                g2_new = np.random.normal(mu2, sigma2, size=n_idx)
                w[idx] = np.where(np.random.rand(n_idx) < amp1, g1_new, g2_new)
            return w.reshape(shape)
        if method == "laplace":
            w = np.random.laplace(loc=0.0, scale=0.5, size=shape).flatten()
            while np.any(np.abs(w) < threshold):
                idx = np.abs(w) < threshold
                w[idx] = np.random.laplace(loc=0.0, scale=0.5, size=np.sum(idx))
            return w.reshape(shape)
        if method == "power_law":
            a = 2.0
            positive = np.random.power(a, size=np.prod(shape))
            negative = -positive.copy()
            combined = np.concatenate([positive, negative])
            w = np.random.choice(combined, size=np.prod(shape), replace=False)
            while np.any(np.abs(w) < threshold):
                idx = np.abs(w) < threshold
                n_new = np.sum(idx)
                new_pos = np.random.power(a, size=n_new // 2 + n_new % 2)
                new_neg = -np.random.power(a, size=n_new // 2)
                new_vals = np.concatenate([new_pos, new_neg])
                np.random.shuffle(new_vals)
                w[idx] = new_vals[:n_new]
            return w.reshape(shape)
        raise ValueError(f"Unknown weight initialization method: {method}")
    else:
        if method == "random_uniform":
            return np.random.uniform(-1, 1, size=shape)
        if method == "random_normal":
            mu = 0.0
            sd = dynamic_sd if dynamic_sd is not None else 1.0
            w = np.random.normal(mu, sd, size=shape)
            return w
        if method == "double_gaussian":
            mu1, mu2 = -1.5, 1.5
            sigma1, sigma2 = dynamic_sd, dynamic_sd
            amp1, amp2 = 0.5, 0.5
            n_elements = np.prod(shape)
            choices = np.random.choice([0, 1], size=n_elements, p=[amp1, amp2])
            g1 = np.random.normal(mu1, sigma1, size=n_elements)
            g2 = np.random.normal(mu2, sigma2, size=n_elements)
            w = np.where(choices == 0, g1, g2)
            return w.reshape(shape)
        if method == "laplace":
            w = np.random.laplace(loc=0.0, scale=0.5, size=shape).flatten()
            return w.reshape(shape)
        if method == "power_law":
            a = 2.0
            positive = np.random.power(a, size=np.prod(shape))
            negative = -positive.copy()
            combined = np.concatenate([positive, negative])
            w = np.random.choice(combined, size=np.prod(shape), replace=False)
            return w.reshape(shape)
        raise ValueError(f"Unknown weight initialization method: {method}")
    

def scalar_from_model(fitted_model, X_test, y_test, washout=200):
    y_pred = fitted_model.predict(X_test)
    n_time_out = y_test.shape[1]

    # Debug print
    print("y_pred shape:", y_pred.shape)
    print("y_pred first 10 values:", y_pred[0, :10])
    
    # Ensure washout is not longer than the output length
    if washout >= n_time_out:
        washout = 0  # or n_time_out // 2

    y_true_seq = np.nan_to_num(y_test[0, washout:, 0], nan=0.0)
    y_pred_seq = np.nan_to_num(y_pred[0, washout:, 0], nan=0.0)

    y_true_scalar = float(np.mean(y_true_seq))
    y_pred_scalar = float(np.mean(y_pred_seq))

    return y_true_scalar, y_pred_scalar

def run_inner_trial(
    model_serialized,
    sd_list,
    X_train,
    y_train,
    X_test,
    y_test,
    best_sd,
    trial_outer,
    trial_inner,
    readin_records,
    record_lock,
):
    """Run one inner trial for all distributions and return scalars."""
    washout = min(50, y_test.shape[1] // 4) # initial timesteps (transients) to discard. Ensure that you never exceed the length of the sequence
    outer_idx = trial_outer + 1
    inner_idx = trial_inner + 1

    # ---- Uniform ----
    model_rc_uniform = pickle.loads(model_serialized)
    weights_uniform = create_weights((args.reservoir_nodes, 3), "random_uniform")
    model_rc_uniform._set_readin_weights(weights_uniform)
    model_rc_uniform.fit(X_train, y_train)
    gt_uniform, pred_uniform = scalar_from_model(model_rc_uniform, X_test, y_test, washout)

    # ---- Gaussian with best_sd ----
    model_rc_gauss = pickle.loads(model_serialized)
    weights_gauss = create_weights((args.reservoir_nodes, 3), "random_normal", dynamic_sd=best_sd)
    model_rc_gauss._set_readin_weights(weights_gauss)
    model_rc_gauss.fit(X_train, y_train)
    gt_gauss, pred_gauss = scalar_from_model(model_rc_gauss, X_test, y_test, washout)

    # ---- Double-Gaussian with best_sd ----
    model_rc_dbgauss = pickle.loads(model_serialized)
    weights_dbgauss = create_weights((args.reservoir_nodes, 3), "double_gaussian", dynamic_sd=best_sd)
    model_rc_dbgauss._set_readin_weights(weights_dbgauss)
    model_rc_dbgauss.fit(X_train, y_train)
    gt_dbgauss, pred_dbgauss = scalar_from_model(model_rc_dbgauss, X_test, y_test, washout)

    # ---- Laplace ----
    model_rc_laplace = pickle.loads(model_serialized)
    weights_laplace = create_weights((args.reservoir_nodes, 3), "laplace")
    model_rc_laplace._set_readin_weights(weights_laplace)
    model_rc_laplace.fit(X_train, y_train)
    gt_laplace, pred_laplace = scalar_from_model(model_rc_laplace, X_test, y_test, washout)

    # ---- Power-law ----
    model_rc_powlaw = pickle.loads(model_serialized)
    weights_powlaw = create_weights((args.reservoir_nodes, 3), "power_law")
    model_rc_powlaw._set_readin_weights(weights_powlaw)
    model_rc_powlaw.fit(X_train, y_train)
    gt_powlaw, pred_powlaw = scalar_from_model(model_rc_powlaw, X_test, y_test, washout)

    # Record read-in weights (thread-safe)
    with record_lock:
        readin_records["uniform"].append(
            (outer_idx, inner_idx, weights_uniform.flatten().copy())
        )
        readin_records["gaussian"].append(
            (outer_idx, inner_idx, weights_gauss.flatten().copy())
        )
        readin_records["double_gaussian"].append(
            (outer_idx, inner_idx, weights_dbgauss.flatten().copy())
        )
        readin_records["laplace"].append(
            (outer_idx, inner_idx, weights_laplace.flatten().copy())
        )
        readin_records["power_law"].append(
            (outer_idx, inner_idx, weights_powlaw.flatten().copy())
        )

    return (
        gt_uniform, pred_uniform,
        gt_gauss, pred_gauss,
        gt_dbgauss, pred_dbgauss,
        gt_laplace, pred_laplace,
        gt_powlaw, pred_powlaw,
    )

# ----------------------
# --- Main Program------
# ----------------------
def main():
    # Fix the seed for reproducibility
    np.random.seed(42)

    start_time = time.time()

    # Structured dtype: each row is (outer, gt_u, pr_u, gt_g, pr_g, ..., gt_p, pr_p)
    dtype = np.dtype([
        ("outer", "i4"),
        ("gt_uniform_inner", "f8"),
        ("pred_uniform_inner", "f8"),
        ("gt_gauss_inner", "f8"),
        ("pred_gauss_inner", "f8"),
        ("gt_dbgauss_inner", "f8"),
        ("pred_dbgauss_inner", "f8"),
        ("gt_laplace_inner", "f8"),
        ("pred_laplace_inner", "f8"),
        ("gt_powlaw_inner", "f8"),
        ("pred_powlaw_inner", "f8"),
    ])

    all_rows = []

    # Buffers for read-in weights: per distribution, list of (outer, inner, weights_1d)
    readin_records = {
        "uniform": [],
        "gaussian": [],
        "double_gaussian": [],
        "laplace": [],
        "power_law": [],
    }
    record_lock = threading.Lock()

    # Buffer for reservoir weights across outer trials: (outer, weights_1d)
    reservoir_records = []

    for trial_outer in range(args.n_trials):
        print(f"Outer Trial {trial_outer + 1}/{args.n_trials} - Creating fresh model")
        task_func = task_functions[args.task]
        X_train, X_test, y_train, y_test = task_func(n_batch=500, n_states=3)

        # Sanity check: no NaN in data splits
        assert not np.any(np.isnan(X_train)), "X_train contains NaN."
        assert not np.any(np.isnan(X_test)), "X_test contains NaN."
        assert not np.any(np.isnan(y_train)), "y_train contains NaN."
        assert not np.any(np.isnan(y_test)), "y_test contains NaN."

        model_rc, reservoir_layer = create_base_model(
            (X_train.shape[1], X_train.shape[2]),
            (y_train.shape[1], y_train.shape[2])
        )
        model_serialized = pickle.dumps(model_rc)

        # Record reservoir weights (one row per outer trial)
        reservoir_weights = reservoir_layer.weights
        reservoir_records.append(
            (trial_outer + 1, reservoir_weights.flatten().copy())
        )
        print(f"Recorded reservoir weights for outer trial {trial_outer + 1}")

        # 1) Find best_sd for this outer trial
        gauss_losses_per_sd = {sd: [] for sd in sd_list}

        for sd in sd_list:
            for _ in range(args.n_inner):
                model_rc_g = pickle.loads(model_serialized)
                weights_gauss = create_weights((args.reservoir_nodes, 3), "random_normal", dynamic_sd=sd)
                model_rc_g._set_readin_weights(weights_gauss)
                model_rc_g.fit(X_train, y_train)

                y_pred = model_rc_g.predict(X_test)
                washout = 200
                y_true_seq = y_test[0, washout:, 0]
                y_pred_seq = y_pred[0, washout:, 0]
                mae = float(np.mean(np.abs(y_true_seq - y_pred_seq)))
                gauss_losses_per_sd[sd].append(mae)

        avg_gauss_losses = {sd: np.mean(gauss_losses_per_sd[sd]) for sd in sd_list}
        best_sd = min(avg_gauss_losses, key=avg_gauss_losses.get)
        best_sd_numeric = float(best_sd)
        print(f"Best Gaussian SD for outer trial {trial_outer + 1}: {best_sd_numeric}")

        # 2) Now run your actual inner trials using this best_sd
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    run_inner_trial,
                    model_serialized,
                    sd_list,
                    X_train, y_train, X_test, y_test,
                    best_sd_numeric,
                    trial_outer,
                    inner_idx,
                    readin_records,
                    record_lock,
                )
                for inner_idx in range(args.n_inner)
            ]

            for future in concurrent.futures.as_completed(futures):
                (
                    gt_uniform, pred_uniform,
                    gt_gauss, pred_gauss,
                    gt_dbgauss, pred_dbgauss,
                    gt_laplace, pred_laplace,
                    gt_powlaw, pred_powlaw,
                ) = future.result()

                row = (
                    trial_outer + 1,
                    gt_uniform, pred_uniform,
                    gt_gauss, pred_gauss,
                    gt_dbgauss, pred_dbgauss,
                    gt_laplace, pred_laplace,
                    gt_powlaw, pred_powlaw,
                )
                all_rows.append(row)

    # Save scalar results
    all_rows = np.array(all_rows, dtype=dtype)
    out_path = os.path.join(OUTPUT_DIR, "sc3_results_fixed-reservoir.npy")
    np.save(out_path, all_rows)
    print(f"Saved results to {out_path}")

    # ----------------------
    # Save read-in weights: one .npy per distribution
    # ----------------------
    for dist, records in readin_records.items():
        if not records:
            continue

        # records: list of (outer, inner, weights_1d)
        n_samples = len(records)
        outer = np.array([r[0] for r in records], dtype=np.int32)
        inner = np.array([r[1] for r in records], dtype=np.int32)
        weights = np.stack([r[2] for r in records], axis=0)

        # Build array: [outer, inner, w0, w1, ...]
        arr = np.zeros((n_samples, 2 + weights.shape[1]), dtype=np.float64)
        arr[:, 0] = outer
        arr[:, 1] = inner
        arr[:, 2:] = weights

        fname = os.path.join(OUTPUT_DIR, f"sc3_readin_weights_{dist}.npy")
        np.save(fname, arr)
        print(f"Saved read-in weights for {dist} to {fname}")

    # ----------------------
    # Save all reservoir weights in one file
    # ----------------------
    if reservoir_records:
        n_outer = len(reservoir_records)
        outer_idx = np.array([r[0] for r in reservoir_records], dtype=np.int32)
        res_weights = np.stack([r[1] for r in reservoir_records], axis=0)

        # Array: [outer_trial, w0, w1, ...]
        res_arr = np.zeros((n_outer, 1 + res_weights.shape[1]), dtype=np.float64)
        res_arr[:, 0] = outer_idx
        res_arr[:, 1:] = res_weights

        res_path = os.path.join(OUTPUT_DIR, "sc3_reservoir_weights.npy")
        np.save(res_path, res_arr)
        print(f"Saved all reservoir weights to {res_path}")

    print(f"Total time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    main()

