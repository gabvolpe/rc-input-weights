"""
Mackey-Glass — Unconditional Variability Extraction, fixed Read-In | Variable Reservoir.
Constraint Set 2: 50% input (50% masking), no near-zero read-in weights.
Gaussian SD is fixed at 1.0.
"""

import os
import sys
import numpy as np
import argparse
import concurrent.futures
import threading

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

parser.add_argument("--n_trials", type=int, default=50,     help="Number of outer trials (read-in)") 
parser.add_argument("--n_inner",  type=int, default=100,    help="Number of inner trials per read-in") 

parser.add_argument("--reservoir_nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.2)
parser.add_argument("--spectral_radius", type=float, default=0.8)
parser.add_argument("--leakage_rate", type=float, default=0.5)
parser.add_argument("--fraction_input", type=float, default=0.5)

parser.add_argument("--ridge_alpha", type=float, default=0.1)

parser.add_argument("--set_threshold", type=bool, default=True)
parser.add_argument("--readin_threshold", type=float, default=1e-3)

parser.add_argument("--parallel", action="store_true", default=True)

args = parser.parse_args()

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
GAUSS_SD = 1.0
THRESHOLD = args.readin_threshold if args.set_threshold else None

dist_keys = ["uniform", "gaussian", "double_gaussian", "laplace", "power_law"]

lock = threading.Lock()

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def run_model(model, W_in, X_train, y_train, X_test, y_test):
    model._set_readin_weights(W_in)
    model.fit(X_train, y_train)
    pred = predict_sequences(model, X_test, y_test)
    return pred


# ------------------------------------------------------------
# INNER LOOP
# ------------------------------------------------------------
def run_inner_trial(
    readin_set,
    X_train, y_train, X_test, y_test,
    outer, inner,
    readin_records,
    reservoir_records,
    timeseries_records
):

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

    reservoir_weights = reservoir_layer.weights.flatten().copy()

    results = {}

    for k in dist_keys:
        _, pred = run_model(
            model_base,
            readin_set[k],
            X_train, y_train,
            X_test, y_test
        )
        results[k] = pred

    with lock:
        reservoir_records.append((outer + 1, inner + 1, reservoir_weights))

        for k in dist_keys:
            readin_records[k].append(
                (outer + 1, inner + 1, readin_set[k].flatten().copy())
            )
            timeseries_records[k].append(
                (outer + 1, inner + 1, results[k].copy())
            )


# ------------------------------------------------------------
# SAVE UTIL
# ------------------------------------------------------------
def save_array(path, records):
    if not records:
        return

    outer = np.array([r[0] for r in records], dtype=np.int32)
    inner = np.array([r[1] for r in records], dtype=np.int32)
    data = np.stack([r[2] for r in records], axis=0)

    arr = np.zeros((len(records), 2 + data.shape[1]), dtype=np.float64)
    arr[:, 0] = outer
    arr[:, 1] = inner
    arr[:, 2:] = data

    np.save(path, arr)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    np.random.seed(42)

    # --------------------------------------------------------
    # DATA (helper handles Mackey-Glass internally)
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = load_dataset("mackey-glass")

    # ground truth (single source of truth)
    np.save(
        os.path.join(OUTPUT_DIR, "sc2_ground_truth.npy"),
        y_test[0]
    )

    readin_records = {k: [] for k in dist_keys}
    reservoir_records = []
    timeseries_records = {k: [] for k in dist_keys}

    readin_shape = (args.reservoir_nodes, 2)

    print("\nMackey-Glass Fixed Read-In | Variable Reservoir (Code B)\n")

    # --------------------------------------------------------
    # OUTER LOOP (fixed read-in)
    # --------------------------------------------------------
    for outer in range(args.n_trials):

        print(f"[Outer {outer+1}] sampling read-in weights")

        readin_set = {
            "uniform": sample_readin_weights(readin_shape, "random_uniform", threshold=THRESHOLD),
            "gaussian": sample_readin_weights(readin_shape, "random_normal", sd=GAUSS_SD, threshold=THRESHOLD),
            "double_gaussian": sample_readin_weights(readin_shape, "double_gaussian", sd=GAUSS_SD, threshold=THRESHOLD),
            "laplace": sample_readin_weights(readin_shape, "laplace", threshold=THRESHOLD),
            "power_law": sample_readin_weights(readin_shape, "power_law", threshold=THRESHOLD),
        }

        for k, w in readin_set.items():
            assert_weights_above_threshold(w, THRESHOLD, k)

        # ----------------------------------------------------
        # INNER LOOP (variable reservoir)
        # ----------------------------------------------------
        if args.parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                futures = [
                    ex.submit(
                        run_inner_trial,
                        readin_set,
                        X_train, y_train, X_test, y_test,
                        outer, inner,
                        readin_records,
                        reservoir_records,
                        timeseries_records,
                    )
                    for inner in range(args.n_inner)
                ]
                for f in concurrent.futures.as_completed(futures):
                    f.result()
        else:
            for inner in range(args.n_inner):
                run_inner_trial(
                    readin_set,
                    X_train, y_train, X_test, y_test,
                    outer, inner,
                    readin_records,
                    reservoir_records,
                    timeseries_records,
                )

        print(f"[Outer {outer+1}] done")

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    save_array(
        os.path.join(OUTPUT_DIR, "sc2_reservoir_weights.npy"),
        reservoir_records
    )

    for k in dist_keys:
        save_array(
            os.path.join(OUTPUT_DIR, f"sc2_readin_weights_{k}.npy"),
            readin_records[k]
        )

        save_array(
            os.path.join(OUTPUT_DIR, f"sc2_timeseries_{k}.npy"),
            timeseries_records[k]
        )

    print("\nSaved all Mackey-Glass SC2 outputs successfully.")


if __name__ == "__main__":
    main()