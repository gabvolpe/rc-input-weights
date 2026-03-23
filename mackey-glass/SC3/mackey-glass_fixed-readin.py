"""
Mackey-Glass
Unconditional Variability Extraction with fixed read-in.
For each of the five fixed read-in matrices, many inner trials are run, each with different reservoir matrices are generated.

Set Contraint 3: input without masking 100%, zero or near-zero read-in values allowed.

1)The readin weights are saved in the corresponding .npy, in the form: outer_ID (readin_ID), readin_weights
2)The reservoir weights are saved in the corresponding .npy, in the form: outer_ID (readin_ID), inner_ID (reservoir_ID), reservoir_weights
3)The results (predictions and ground truths) are saved in the corresponding .npy, in the form: 
outer_trial (readin_ID), gt-uniform, pred-uniform, gt-gauss, pred-gauss, gt-dbgauss, pred-dbgauss, gt-laplace, pred-laplace, gt-powerlaw, pred-powerlaw
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
import brainpy as bp
import brainpy.math as bm
import concurrent.futures
import pickle
import matplotlib.pyplot as plt
import threading


# ----------------------
# Output directories
# ----------------------
OUTPUT_DIR = "mackey-glass/outputs/fixed-readin"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------
# --- User Arguments ---
# ----------------------
parser = argparse.ArgumentParser(description="Run RC model with customizable hyperparameters.")

parser.add_argument("--n_trials", type=int, default=2, help="Number of outer trials, each trial has different fixed read-ins from the 5 distributions")
parser.add_argument("--n_inner", type=int, default=3, help="Number of inner trials, each inner trial with a different reservoir")

parser.add_argument("--reservoir_nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.2)
parser.add_argument("--spectral_radius", type=float, default=0.8)
parser.add_argument("--leakage_rate", type=float, default=0.5)
parser.add_argument("--fraction_input", type=float, default=1.0,
                    help="Percentage of input without masking")

parser.add_argument("--ridge_alpha", type=float, default=0.1)

parser.add_argument("--set_threshold", type=bool, default=False,
                    help="Set to False if no threshold wished")
parser.add_argument("--readin_threshold", type=float, default=1e-3)

parser.add_argument(
    "--sd_list",
    type=str,
    default="[0.1, 0.25, 0.5, 0.75, 1.0]",
    help="List of Gaussian SD values, format: [0.1,0.25,...]"
)
parser.add_argument("--task", type=str, default='sequence_to_sequence')

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


def generate_mackey_glass_data(n_samples=20000, beta=0.2, gamma=0.1, tau=17, n=10):
    class MackeyGlassEq(bp.Dynamic):
        def __init__(self, num):
            super().__init__(num)
            self.beta, self.gamma, self.tau, self.n = beta, gamma, tau, n
            self.delay_len = int(self.tau / bm.get_dt())
            self.x = bm.Variable(bm.zeros(num))
            self.x_delay = bm.LengthDelay(
                self.x, delay_len=self.delay_len,
                initial_delay_data=lambda sh, dtype: 1.2 + 0.2 * (bm.random.random(sh) - 0.5)
            )
            self.x_oldest = bm.Variable(self.x_delay(self.delay_len))
            self.integral = bp.odeint(
                lambda x, t, x_tau: self.beta * x_tau / (1 + x_tau ** n) - self.gamma * x,
                method='exp_auto'
            )

        def update(self):
            self.x.value = self.integral(self.x.value, bp.share['t'], self.x_oldest.value, bp.share['dt'])
            self.x_delay.update(self.x.value)
            self.x_oldest.value = self.x_delay(self.delay_len)

    runner = bp.DSRunner(MackeyGlassEq(1), monitors=['x', 'x_oldest'])
    runner.run(n_samples * bm.get_dt())
    data = np.column_stack([runner.mon.ts, runner.mon.x, runner.mon.x_oldest])

    # Normalize x to [-1, 1] to avoid large values
    x_vals = data[:, 1].reshape(-1, 1)
    x_min, x_max = x_vals.min(), x_vals.max()
    x_scaled = 2 * (x_vals - x_min) / (x_max - x_min + 1e-12) - 1.0
    data[:, 1] = x_scaled.ravel()

    assert not np.any(np.isnan(data)), "Mackey-Glass data contains NaN after normalization."

    return data


def mackey_glass_pred(n_batch, n_time_in, n_time_out, n_states=2):
    data = generate_mackey_glass_data(n_samples=30000)
    time_series = data[:, 1:1 + n_states]

    X = np.array([time_series[i:i + n_time_in] for i in range(n_batch)])
    y = np.array([time_series[i + n_time_in:i + n_time_in + n_time_out]
                  for i in range(n_batch)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize X and y using train only
    xflat = X_train.reshape(-1, X_train.shape[-1]).copy()
    yflat = y_train.reshape(-1, y_train.shape[-1]).copy()

    X_mean, X_std = xflat.mean(axis=0), xflat.std(axis=0)
    Y_mean, Y_std = yflat.mean(axis=0), yflat.std(axis=0)

    X_train = (X_train - X_mean) / (X_std + 1e-12)
    X_test  = (X_test  - X_mean) / (X_std + 1e-12)
    y_train = (y_train - Y_mean) / (Y_std + 1e-12)
    y_test  = (y_test  - Y_mean) / (Y_std + 1e-12)

    return X_train, X_test, y_train, y_test


# Sequence-to-Sequence
def sequence_to_sequence(n_batch=200, n_states=2):
    return mackey_glass_pred(n_batch=n_batch, n_states=n_states, n_time_in=1000, n_time_out=100)


task_functions = {
    "sequence_to_sequence": sequence_to_sequence
}


def create_weights(shape, method, dynamic_sd=None):
    # Ensure shape is always [nodes, n_states_in = 2]
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
    X_train,
    y_train,
    X_test,
    y_test,
    trial_outer,
    trial_inner,
    readin_mats_for_outer,
    reservoir_records,
    reservoir_lock,
):
    """
    For a given outer trial (fixed read-in weights):
    - Create a new random reservoir (new model) for this inner trial.
    - For each distribution, set its fixed read-in weights, train, and evaluate.
    - Record the reservoir matrix as (outer, inner, flattened_reservoir_weights).
    Returns scalar gt/pred per distribution.
    """
    washout = 200
    outer_idx = trial_outer + 1
    inner_idx = trial_inner + 1

    # Create a model with a fresh random reservoir for this inner trial
    model_rc, reservoir_layer = create_base_model(
        (X_train.shape[1], X_train.shape[2]),
        (y_train.shape[1], y_train.shape[2])
    )
    reservoir_weights = reservoir_layer.weights  # matrix

    # Record reservoir weights (thread-safe)
    with reservoir_lock:
        reservoir_records.append(
            (outer_idx, inner_idx, reservoir_weights.flatten().copy())
        )

    # Helper to run one distribution with given read-in weights on this reservoir
    def run_with_readin_weights(W_in):
        # Clone model_rc via serialization so each run starts from same reservoir
        model = pickle.loads(pickle.dumps(model_rc))
        model._set_readin_weights(W_in)
        model.fit(X_train, y_train)
        gt, pred = scalar_from_model(model, X_test, y_test, washout)
        return gt, pred

    gt_uniform, pred_uniform = run_with_readin_weights(readin_mats_for_outer["uniform"])
    gt_gauss, pred_gauss = run_with_readin_weights(readin_mats_for_outer["gaussian"])
    gt_dbgauss, pred_dbgauss = run_with_readin_weights(readin_mats_for_outer["double_gaussian"])
    gt_laplace, pred_laplace = run_with_readin_weights(readin_mats_for_outer["laplace"])
    gt_powlaw, pred_powlaw = run_with_readin_weights(readin_mats_for_outer["power_law"])

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
    # fix the seed for reproducibility
    np.random.seed(42)
    
    if args.task == "sequence_to_sequence":
        print("\nMackey-Glass is ready\n")
        X_train, X_test, y_train, y_test = sequence_to_sequence(
            n_states=2, n_batch=200,
        )
    else:
        raise NotImplementedError(f"Task {args.task} not implemented")

    start_time = time.time()

    # Structured dtype with named columns for scalar outputs
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

    # Buffers for read-in weights: per distribution, list of (outer, weights_1d)
    readin_weight_records = {
        "uniform": [],
        "gaussian": [],
        "double_gaussian": [],
        "laplace": [],
        "power_law": [],
    }

    # Buffer for reservoir weights: list of (outer, inner, weights_1d)
    reservoir_records = []
    reservoir_lock = threading.Lock()

    for trial_outer in range(args.n_trials):
        print(f"Outer Trial {trial_outer + 1}/{args.n_trials} - Generating fixed read-in weights")

        # --- For this outer trial, generate 5 fixed read-in weight vectors ---
        readin_mats_for_outer = {}

        W_uniform = create_weights((args.reservoir_nodes, 2), "random_uniform")
        W_gauss = create_weights((args.reservoir_nodes, 2), "random_normal", dynamic_sd=1.0)
        W_dbgauss = create_weights((args.reservoir_nodes, 2), "double_gaussian", dynamic_sd=1.0)
        W_laplace = create_weights((args.reservoir_nodes, 2), "laplace")
        W_powlaw = create_weights((args.reservoir_nodes, 2), "power_law")

        readin_mats_for_outer["uniform"] = W_uniform
        readin_mats_for_outer["gaussian"] = W_gauss
        readin_mats_for_outer["double_gaussian"] = W_dbgauss
        readin_mats_for_outer["laplace"] = W_laplace
        readin_mats_for_outer["power_law"] = W_powlaw

        # Append these read-in weights to their corresponding records
        for dist, W in readin_mats_for_outer.items():
            readin_weight_records[dist].append(
                (trial_outer + 1, W.flatten().copy())
            )

        # -------------------------------------------------
        # Inner trials for this outer read-in set
        # -------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    run_inner_trial,
                    X_train, y_train, X_test, y_test,
                    trial_outer,
                    inner_idx,
                    readin_mats_for_outer,
                    reservoir_records,
                    reservoir_lock,
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

                # Results format:
                # outer_trial, gt-uniform, pred-uniform, gt-gauss, pred-gauss,
                # gt-dbgauss, pred-dbgauss, gt-laplace, pred-laplace, gt-powerlaw, pred-powerlaw
                row = (
                    trial_outer + 1,
                    gt_uniform, pred_uniform,
                    gt_gauss, pred_gauss,
                    gt_dbgauss, pred_dbgauss,
                    gt_laplace, pred_laplace,
                    gt_powlaw, pred_powlaw,
                )
                all_rows.append(row)

    # ----------------------
    # Save scalar results (unchanged format)
    # ----------------------
    all_rows = np.array(all_rows, dtype=dtype)
    out_path = os.path.join(OUTPUT_DIR, "sc3_results_fixed-readin.npy")
    np.save(out_path, all_rows)
    print(f"Saved results to {out_path}")

    # ----------------------
    # Save read-in weights: one .npy per distribution
    # format: [outer_trial, readin_weights...]
    # ----------------------
    for dist, records in readin_weight_records.items():
        if not records:
            continue

        # records: list of (outer, weights_1d)
        n_samples = len(records)
        outer = np.array([r[0] for r in records], dtype=np.int32)
        weights = np.stack([r[1] for r in records], axis=0)  # (n_samples, n_params)

        arr = np.zeros((n_samples, 1 + weights.shape[1]), dtype=np.float64)
        arr[:, 0] = outer
        arr[:, 1:] = weights

        fname = os.path.join(OUTPUT_DIR, f"sc3_readin_weights_{dist}.npy")
        np.save(fname, arr)
        print(f"Saved read-in weights for {dist} to {fname}")

    # ----------------------
    # Save all reservoir weights in one file
    # format: [outer_trial, inner_trial, reservoir_weights...]
    # ----------------------
    if reservoir_records:
        n_rows = len(reservoir_records)
        outer_idx = np.array([r[0] for r in reservoir_records], dtype=np.int32)
        inner_idx = np.array([r[1] for r in reservoir_records], dtype=np.int32)
        res_weights = np.stack([r[2] for r in reservoir_records], axis=0)

        res_arr = np.zeros((n_rows, 2 + res_weights.shape[1]), dtype=np.float64)
        res_arr[:, 0] = outer_idx
        res_arr[:, 1] = inner_idx
        res_arr[:, 2:] = res_weights

        res_path = os.path.join(OUTPUT_DIR, "sc3_reservoir_weights.npy")
        np.save(res_path, res_arr)
        print(f"Saved all reservoir weights to {res_path}")

    print(f"Total time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    main()
