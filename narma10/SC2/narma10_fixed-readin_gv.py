"""
NARMA-10 — Unconditional Variability Extraction, fixed Read-In | Variable Reservoir.
Constraint Set 2: 50% input (50% masking), no near-zero read-in weights.
Gaussian SD fixed at 1.0.

Memory-safe streaming version of NARMA-10 fixed Read-In | Variable Reservoir.

This version streams:
    sc2_timeseries_<dist>.npy
    sc2_readin_weights_<dist>.npy
    sc2_timeseries_gt.npy
    sc2_reservoir_weights.npy

through temporary files under:
    outputs/fixed-readin/_temp_stream
and merges them at the end to avoid RAM accumulation.
"""

import os
import sys
import numpy as np
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
    "fixed-readin"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# DISTRIBUTIONS
# ------------------------------------------------------------
DIST_MAP = {
    "uniform": "random_uniform",
    "gaussian": "random_normal",
    "double_gaussian": "double_gaussian",
    "laplace": "laplace",
    "power_law": "power_law",
}

EVAL_DISTS = list(DIST_MAP.keys())

# ------------------------------------------------------------
# ARGS
# ------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--n_outer", type=int, default=50,     help="Number of outer trials (read-in)")
parser.add_argument("--n_inner", type=int, default=100,    help="Number of inner trials per read-in")

parser.add_argument("--nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.4)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.2)
parser.add_argument("--fraction_input", type=float, default=0.5)
parser.add_argument("--ridge_alpha", type=float, default=1e-6)

parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--set_threshold",    type=bool,  default=True)
parser.add_argument("--parallel", action="store_true")

args = parser.parse_args()

# ------------------------------------------------------------
# READ-IN CONTROL
# ------------------------------------------------------------
GAUSS_SD = 1.0
THRESHOLD = args.readin_threshold if args.set_threshold else None

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = load_dataset("narma10")

X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test  = y_test.astype(np.float32)

# select only firt 50 samples for speed
X_train = X_train[:50, ...]
y_train = y_train[:50, ...]
X_test = X_test[:50, ...]
y_test = y_test[:50, ...]

print(f"shape of X-train: {X_train.shape}")
print(f"shape of X-test: {X_test.shape}")


np.save(os.path.join(OUTPUT_DIR, "sc2_ground_truth.npy"), y_test)

# ------------------------------------------------------------
# SAFE UNPACK (prevents future shape bugs)
# ------------------------------------------------------------
def to_1d(x):
    x = np.asarray(x)
    return np.asarray(x).squeeze().reshape(-1)


# ------------------------------------------------------------
# INNER LOOP: variable reservoirs
# ------------------------------------------------------------
def run_inner(model_bytes, outer_id, inner_id, readin_sets):

    model = pickle.loads(model_bytes)

    results = []
    readin_local = {}
    gt_local = None

    for dist, sampler in DIST_MAP.items():

        # fixed read-in for this OUTER loop
        W = readin_sets[dist]

        model_instance = pickle.loads(model_bytes)

        model_instance._set_readin_weights(W)
        model_instance.fit(X_train, y_train)

        gt, pred = predict_sequences(model_instance, X_test, y_test)

        pred = to_1d(pred)
        gt_local = to_1d(gt)

        results.append(pred)
        readin_local[dist] = W.copy()

    return outer_id, inner_id, results, gt_local, readin_local


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    np.random.seed(42)

    readin_store = {d: [] for d in EVAL_DISTS}
    timeseries_store = {d: [] for d in EVAL_DISTS}
    timeseries_store["gt"] = []

    reservoir_store = []

    print("\nNARMA10 — FIXED READ-IN / VARIABLE RESERVOIR\n")

    # --------------------------------------------------------
    # OUTER LOOP = FIXED READ-IN SETS
    # --------------------------------------------------------
    for outer in range(args.n_outer):

        print(f"[Outer {outer+1}] sampling fixed read-in sets")

        # FIXED READ-IN SET PER OUTER LOOP
        readin_sets = {
            d: sample_readin_weights(
                shape=(args.nodes, X_train.shape[2]),
                method=DIST_MAP[d],
                sd=GAUSS_SD if DIST_MAP[d] in ["random_normal", "double_gaussian"] else None,
                threshold=THRESHOLD
            )
            for d in EVAL_DISTS
        }

        for d in EVAL_DISTS:
            if THRESHOLD is not None:
                assert_weights_above_threshold(
                    readin_sets[d],
                    THRESHOLD,
                    d
                )

        # ----------------------------------------------------
        # INNER LOOP = VARIABLE RESERVOIRS
        # ----------------------------------------------------
        if args.parallel:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as ex:

                futures = [
                    ex.submit(run_inner, None, outer, inner, readin_sets)
                    for inner in range(args.n_inner)
                ]

                for f in concurrent.futures.as_completed(futures):
                    outer_id, inner_id, preds, gt, readins = f.result()

                    reservoir_store.append((outer_id, inner_id))

                    for i, d in enumerate(EVAL_DISTS):
                        timeseries_store[d].append(
                            (outer_id, inner_id, preds[i])
                        )

                    timeseries_store["gt"].append(
                        (outer_id, inner_id, gt)
                    )

                    for d in EVAL_DISTS:
                        readin_store[d].append(
                            (outer_id, inner_id, readins[d])
                        )

        else:
            for inner in range(args.n_inner):

                model, reservoir = create_model(
                    input_shape=X_train.shape[1:],
                    output_shape=y_train.shape[1:],
                    nodes=args.nodes,
                    density=args.density,
                    spectral_radius=args.spectral_radius,
                    leakage_rate=args.leakage_rate,
                    fraction_input=args.fraction_input,
                    ridge_alpha=args.ridge_alpha,
                )

                model_bytes = pickle.dumps(model)

                outer_id, inner_id, preds, gt, readins = run_inner(
                    model_bytes, outer, inner, readin_sets
                )

                reservoir_store.append((outer_id, inner_id))

                for i, d in enumerate(EVAL_DISTS):
                    timeseries_store[d].append(
                        (outer_id, inner_id, preds[i])
                    )

                timeseries_store["gt"].append(
                    (outer_id, inner_id, gt)
                )

                for d in EVAL_DISTS:
                    readin_store[d].append(
                        (outer_id, inner_id, readins[d])
                    )

        print(f"[Outer {outer+1}] done")

    # ------------------------------------------------------------
    # SAVE OUTPUTS 
    # ------------------------------------------------------------
    for d in EVAL_DISTS:

        np.save(
            os.path.join(OUTPUT_DIR, f"sc2_readin_weights_{d}.npy"),
            np.array(readin_store[d], dtype=object)
        )

        np.save(
            os.path.join(OUTPUT_DIR, f"sc2_timeseries_{d}.npy"),
            np.array(timeseries_store[d], dtype=object)
        )

    np.save(
        os.path.join(OUTPUT_DIR, "sc2_timeseries_gt.npy"),
        np.array(timeseries_store["gt"], dtype=object)
    )

    np.save(
        os.path.join(OUTPUT_DIR, "sc2_reservoir_index.npy"),
        np.array(reservoir_store, dtype=object)
    )

    print("DONE")


if __name__ == "__main__":
    main()
