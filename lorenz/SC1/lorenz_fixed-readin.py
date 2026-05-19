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
"""

import os
import sys
import numpy as np
import argparse
import concurrent.futures
import threading
import pickle

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
parser.add_argument("--n_inner", type=int,  default=25, help="Inner trials = variable reservoirs") 

parser.add_argument("--reservoir_nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.1)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.5)
parser.add_argument("--fraction_input", type=float, default=1.0)
parser.add_argument("--ridge_alpha", type=float, default=1e-3)

parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--set_threshold", type=bool, default=True)

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
# CORE MODEL RUN
# ------------------------------------------------------------
def run_model(model, W_in, X_train, y_train, X_test, y_test):
    model._set_readin_weights(W_in)
    model.fit(X_train, y_train)
    return predict_sequences(model, X_test, y_test, channels=0)

# ------------------------------------------------------------
# INNER TRIAL (VARIABLE RESERVOIR)
# ------------------------------------------------------------
def run_inner_trial(
    model_serialized,
    X_train, y_train, X_test, y_test,
    outer, inner,
    readin_set,
    readin_records,
    reservoir_records,
    timeseries_records
):
    outer_id = outer + 1
    inner_id = inner + 1

    # fresh reservoir per inner trial
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

    reservoir_w = reservoir_layer.weights.flatten().copy()

    results = {}

    # evaluate all read-ins on same reservoir
    for k in dist_keys:
        _, pred = run_model(
            model_base,
            readin_set[k],
            X_train, y_train, X_test, y_test
        )
        results[k] = np.asarray(pred)

    with lock:
        reservoir_records.append((outer_id, inner_id, reservoir_w))

        for k in dist_keys:
            readin_records[k].append(
                (outer_id, inner_id, readin_set[k].flatten().copy())
            )

        for k in dist_keys:
            timeseries_records[k].append(
                (outer_id, inner_id, results[k].flatten().copy())
            )

# ------------------------------------------------------------
# SAVE HELPERS
# ------------------------------------------------------------
def save_records(records, path_template):
    for k, entries in records.items():
        if not entries:
            continue

        outer = np.array([e[0] for e in entries], dtype=np.int32)
        inner = np.array([e[1] for e in entries], dtype=np.int32)
        data  = np.stack([e[2] for e in entries], axis=0)

        arr = np.zeros((len(entries), 2 + data.shape[1]), dtype=np.float64)
        arr[:, 0] = outer
        arr[:, 1] = inner
        arr[:, 2:] = data

        np.save(path_template.format(k), arr)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_dataset("lorenz")

    readin_records = {k: [] for k in dist_keys}
    reservoir_records = []
    timeseries_records = {k: [] for k in dist_keys}

    n_states_in = X_train.shape[2]
    readin_shape = (args.reservoir_nodes, n_states_in)

    print("\nLorenz — Fixed Read-In | Variable Reservoir\n")

    # --------------------------------------------------------
    # OUTER LOOP = FIXED READ-INS
    # --------------------------------------------------------
    for outer in range(args.n_trials):
        print(f"[Outer {outer+1}] sampling fixed read-ins")

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
        # INNER LOOP = VARIABLE RESERVOIRS
        # ----------------------------------------------------
        if args.parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                futures = [
                    ex.submit(
                        run_inner_trial,
                        None,  # no serialized reuse needed
                        X_train, y_train, X_test, y_test,
                        outer, inner,
                        readin_set,
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
                    None,
                    X_train, y_train, X_test, y_test,
                    outer, inner,
                    readin_set,
                    readin_records,
                    reservoir_records,
                    timeseries_records,
                )

        print(f"[Outer {outer+1}] done")

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    # ground truth (unchanged structure)
    np.save(
        os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"),
        y_test[0]
    )

    # reservoir weights
    outer = np.array([r[0] for r in reservoir_records], dtype=np.int32)
    inner = np.array([r[1] for r in reservoir_records], dtype=np.int32)
    res_w = np.stack([r[2] for r in reservoir_records], axis=0)

    arr = np.zeros((len(reservoir_records), 2 + res_w.shape[1]), dtype=np.float64)
    arr[:, 0] = outer
    arr[:, 1] = inner
    arr[:, 2:] = res_w

    np.save(os.path.join(OUTPUT_DIR, "sc1_reservoir_weights.npy"), arr)

    # read-in + timeseries per distribution
    save_records(readin_records, os.path.join(OUTPUT_DIR, "sc1_readin_weights_{}.npy"))
    save_records(timeseries_records, os.path.join(OUTPUT_DIR, "sc1_timeseries_{}.npy"))

    print("\nDONE")


if __name__ == "__main__":
    main()