"""
Lorenz — Unconditional Variability Extraction, fixed Read-In | Variable Reservoir.
Constraint Set 1: full input (no masking), no near-zero read-in weights.
Gaussian SD is fixed at 1.0.

Note that this code is more expensive than the one with fixed reservoirs as outer trials.
The reason is simply that there:
    - 1 reservoir build
    - many read-ins reuse it
    i.e: cost per experiment=C_reservoir​ + N_readin​⋅ C_fit​
Here:
    - 1 read-in set
    - many reservoirs, each one triggers full pipeline
    i.e: cost per experiment=N_reservoir ​⋅ (C_reservoir ​+ C_fit​)

Memory-optimised variant (_fast):
    - Pre-allocated memmap arrays written row-by-row; no load/concatenate/save.
    - No parallelism — single-threaded to keep peak RAM minimal.
    - Output files are named identically to the original so downstream evaluation
      scripts work unchanged.
"""

import os
import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.helpers import (
    load_dataset,
    create_model,
    predict_sequences,
    sample_readin_weights,
    assert_weights_above_threshold,
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs",
    "fixed-readin"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# ARGS
# ------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--n_trials", type=int, default=25, help="Outer trials = fixed read-ins")
parser.add_argument("--n_inner",  type=int, default=25, help="Inner trials = variable reservoirs")

parser.add_argument("--reservoir_nodes",  type=int,   default=200)
parser.add_argument("--density",          type=float, default=0.1)
parser.add_argument("--spectral_radius",  type=float, default=0.9)
parser.add_argument("--leakage_rate",     type=float, default=0.5)
parser.add_argument("--fraction_input",   type=float, default=1.0)
parser.add_argument("--ridge_alpha",      type=float, default=1e-3)

parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--set_threshold",    type=bool,  default=True)

args = parser.parse_args()

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
GAUSS_SD  = 1.0
THRESHOLD = args.readin_threshold if args.set_threshold else None

dist_keys = ["uniform", "gaussian", "double_gaussian", "laplace", "power_law"]

# ------------------------------------------------------------
# CORE MODEL RUN
# ------------------------------------------------------------
def run_model(model, W_in, X_train, y_train, X_test, y_test):
    model._set_readin_weights(W_in)
    model.fit(X_train, y_train)
    return predict_sequences(model, X_test, y_test, channels=0)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_dataset("lorenz")

    n_states_in  = X_train.shape[2]
    readin_shape = (args.reservoir_nodes, n_states_in)
    n_rows       = args.n_trials * args.n_inner

    # determine timeseries length from data
    ts_len      = y_test.shape[1]
    res_w_len   = args.reservoir_nodes ** 2
    readin_w_len = int(np.prod(readin_shape))

    print(f"Dataset loaded. Input Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"output train shape: {y_train.shape}, Test shape: {y_test.shape}")
    
    # make a plot of the ground truth for sanity check (inputs and outputs)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_test[0], label=["x", "y", "z"])
    plt.title("Inputs (Test Set)")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(y_test[0], label=["x", "y", "z"])
    plt.title("Ground Truth Outputs (Test Set)")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ground truth written once up front
    np.save(os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"), y_test[0])

    # ----------------------------------------------------------
    # Pre-allocate memmap arrays
    # Layout: [outer_id, inner_id, ...data...]
    # Writing directly into these; no RAM accumulation at all.
    # ----------------------------------------------------------
    def make_mm(fname, n_cols):
        path = os.path.join(OUTPUT_DIR, fname)
        return np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(n_rows, n_cols)
        )

    mm_reservoir = make_mm("sc1_reservoir_weights.npy", 2 + res_w_len)
    mm_readin    = {k: make_mm(f"sc1_readin_weights_{k}.npy", 2 + readin_w_len) for k in dist_keys}
    mm_timeseries= {k: make_mm(f"sc1_timeseries_{k}.npy",    2 + ts_len)       for k in dist_keys}

    print("\nLorenz — Fixed Read-In | Variable Reservoir (memory-optimised)\n")

    row = 0

    # --------------------------------------------------------
    # OUTER LOOP = FIXED READ-INS
    # --------------------------------------------------------
    for outer in range(args.n_trials):
        print(f"[Outer {outer+1}/{args.n_trials}] sampling fixed read-ins")

        readin_set = {
            "uniform":        sample_readin_weights(readin_shape, "random_uniform",  threshold=THRESHOLD),
            "gaussian":       sample_readin_weights(readin_shape, "random_normal",   sd=GAUSS_SD, threshold=THRESHOLD),
            "double_gaussian":sample_readin_weights(readin_shape, "double_gaussian", sd=GAUSS_SD, threshold=THRESHOLD),
            "laplace":        sample_readin_weights(readin_shape, "laplace",         threshold=THRESHOLD),
            "power_law":      sample_readin_weights(readin_shape, "power_law",       threshold=THRESHOLD),
        }

        for k, w in readin_set.items():
            assert_weights_above_threshold(w, THRESHOLD, k)

        # --------------------------------------------------------
        # INNER LOOP = VARIABLE RESERVOIRS
        # --------------------------------------------------------
        for inner in range(args.n_inner):
            outer_id = outer + 1
            inner_id = inner + 1

            model_base, reservoir_layer = create_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_shape=(y_train.shape[1], y_train.shape[2]),
                nodes=args.reservoir_nodes,
                density=args.density,
                spectral_radius=args.spectral_radius,
                leakage_rate=args.leakage_rate,
                fraction_input=args.fraction_input,
                ridge_alpha=args.ridge_alpha,
            )

            reservoir_w = reservoir_layer.weights.flatten()

            # write reservoir row
            mm_reservoir[row, 0] = outer_id
            mm_reservoir[row, 1] = inner_id
            mm_reservoir[row, 2:] = reservoir_w.astype(np.float32)

            for k in dist_keys:
                _, pred = run_model(model_base, readin_set[k], X_train, y_train, X_test, y_test)
                pred_f32 = np.asarray(pred, dtype=np.float32).flatten()

                mm_readin[k][row, 0] = outer_id
                mm_readin[k][row, 1] = inner_id
                mm_readin[k][row, 2:] = readin_set[k].flatten().astype(np.float32)

                mm_timeseries[k][row, 0] = outer_id
                mm_timeseries[k][row, 1] = inner_id
                mm_timeseries[k][row, 2:] = pred_f32

            row += 1

        print(f"[Outer {outer+1}/{args.n_trials}] done")

    # flush memmaps to disk
    del mm_reservoir
    for k in dist_keys:
        del mm_readin[k]
        del mm_timeseries[k]

    print("\nDONE")


if __name__ == "__main__":
    main()
