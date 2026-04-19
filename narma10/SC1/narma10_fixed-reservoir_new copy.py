"""
NARMA-10 — Unconditional Variability Extraction, fixed reservoir.
Constraint Set 1: full input (no masking), no near-zero read-in weights.
Gaussian SD is fixed at 1.0; no SD optimisation is performed.
"""

import os
import sys
import numpy as np
import time
import argparse
import concurrent.futures
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.helpers import (
    load_dataset,
    create_model,
    predict_sequences,
    sample_readin_weights,
    assert_weights_above_threshold
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs",
    "fixed-reservoir"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISTS = ["random_uniform", "random_normal", "double_gaussian", "laplace", "power_law"]

# ------------------------------------------------------------
# ARGS
# ------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--n_trials", type=int, default=2)
parser.add_argument("--n_inner", type=int, default=3)

parser.add_argument("--nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.4)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.2)
parser.add_argument("--fraction_input", type=float, default=1.0)
parser.add_argument("--ridge_alpha", type=float, default=1e-6)

parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--parallel", action="store_true")

args = parser.parse_args()

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = load_dataset("narma10")

X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test  = y_test.astype(np.float32)

np.save(os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"), y_test)

# ------------------------------------------------------------
# INNER RUN (PURE FUNCTION STYLE)
# ------------------------------------------------------------
def run_inner(model_bytes, outer_id, inner_id):
    """
    Returns all results instead of mutating shared state.
    """

    model_template = model_bytes  # serialized once

    results = []
    gt_store = []
    readin_store_local = {d: None for d in DISTS}

    for dist in DISTS:

        model = pickle.loads(model_template)

        W = sample_readin_weights(
            shape=(args.nodes, X_train.shape[2]),
            method=dist,
            threshold=args.readin_threshold
        )

        assert_weights_above_threshold(W, args.readin_threshold, dist)

        model._set_readin_weights(W)
        model.fit(X_train, y_train)

        gt, pred = predict_sequences(model, X_test, y_test)

        results.append(pred)
        gt_store.append(gt)

        readin_store_local[dist] = W.copy()

    return outer_id, inner_id, results, gt_store, readin_store_local


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    np.random.seed(42)

    readin_store = {d: [] for d in DISTS}
    timeseries_store = {"gt": []}
    timeseries_store.update({d: [] for d in DISTS})
    reservoir_store = []

    all_rows = []

    for outer in range(args.n_trials):

        print(f"Outer {outer+1}/{args.n_trials}")

        model, reservoir = create_model(
            input_shape=X_train.shape[1:],
            output_shape=y_train.shape[1:],
            nodes=args.nodes,
            density=args.density,
            spectral_radius=args.spectral_radius,
            leakage_rate=args.leakage_rate,
            fraction_input=args.fraction_input,
            ridge_alpha=args.ridge_alpha
        )

        reservoir_store.append((outer, reservoir.weights.copy()))
        model_bytes = pickle.dumps(model)

        # ----------------------------------------------------
        # INNER LOOP
        # ----------------------------------------------------
        if args.parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                futures = [
                    ex.submit(run_inner, model_bytes, outer, i)
                    for i in range(args.n_inner)
                ]

                for f in concurrent.futures.as_completed(futures):
                    outer_id, inner_id, preds, gts, readins = f.result()

                    # store predictions
                    for i, pred in enumerate(preds):
                        all_rows.append((outer_id, pred))

                    # store ground truth (IMPORTANT FIX)
                    for i, gt in enumerate(gts):
                        timeseries_store["gt"].append((outer_id, inner_id, gt.copy()))

                    # store readins
                    for d in DISTS:
                        readin_store[d].append(
                            (outer_id, inner_id, readins[d])
                        )

        else:
            for i in range(args.n_inner):
                outer_id, inner_id, preds, gts, readins = run_inner(
                    model_bytes, outer, i
                )

                for i, gt in enumerate(gts):
                    timeseries_store["gt"].append((outer_id, inner_id, gt.copy()))

                for d in DISTS:
                    readin_store[d].append(
                        (outer_id, inner_id, readins[d])
                    )

        print(f"Outer {outer+1} done")

    # ------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------
    for d in DISTS:
        np.save(
            os.path.join(OUTPUT_DIR, f"sc1_readin_weights_{d}.npy"),
            np.array(readin_store[d], dtype=object)
        )

        np.save(
            os.path.join(OUTPUT_DIR, f"sc1_timeseries_{d}.npy"),
            np.array(timeseries_store[d], dtype=object)
        )
    
    np.save(
        os.path.join(OUTPUT_DIR, "sc1_timeseries_gt.npy"),
        np.array(timeseries_store["gt"], dtype=object)
    )

    np.save(
        os.path.join(OUTPUT_DIR, "sc1_reservoir_weights.npy"),
        np.array(reservoir_store, dtype=object)
    )

    print("DONE")


if __name__ == "__main__":
    main()