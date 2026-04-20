"""
NARMA-10 — Unconditional Variability Extraction, fixed reservoir.
Constraint Set 3: full input (no masking), near-zero read-in weights allowed.
Gaussian SD is fixed at 1.0; no SD optimisation is performed.
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
    "fixed-reservoir"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# CLEAN DISTRIBUTION MAPPING
# evaluation_name -> sampler_name
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

parser.add_argument("--n_trials", type=int, default=2,     help="Number of outer trials (reservoirs)")
parser.add_argument("--n_inner",          type=int,   default=3,    help="Number of inner trials per reservoir")
parser.add_argument("--nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.4)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.2)
parser.add_argument("--fraction_input", type=float, default=1.0)
parser.add_argument("--ridge_alpha", type=float, default=1e-6)

parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--set_threshold",    type=bool,  default=False)
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

np.save(os.path.join(OUTPUT_DIR, "sc3_ground_truth.npy"), y_test)

# ------------------------------------------------------------
# INNER RUN
# ------------------------------------------------------------
def run_inner(model_bytes, outer_id, inner_id):

    results = []
    gt_store = []
    readin_store_local = {}

    for eval_dist, sampler_dist in DIST_MAP.items():

        model = pickle.loads(model_bytes)

        W = sample_readin_weights(
            shape=(args.nodes, X_train.shape[2]),
            method=sampler_dist,
            sd=GAUSS_SD if sampler_dist in ["random_normal", "double_gaussian"] else None,
            threshold=THRESHOLD
        )

        if THRESHOLD is not None and THRESHOLD is not False:
            assert_weights_above_threshold(W, THRESHOLD, sampler_dist)

        model._set_readin_weights(W)
        model.fit(X_train, y_train)

        gt, pred = predict_sequences(model, X_test, y_test)

        results.append(pred)
        gt_store.append(gt)

        readin_store_local[eval_dist] = W.copy()

    return outer_id, inner_id, results, gt_store, readin_store_local


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    np.random.seed(42)

    readin_store = {d: [] for d in EVAL_DISTS}
    timeseries_store = {d: [] for d in EVAL_DISTS}
    timeseries_store["gt"] = []

    reservoir_store = []

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

        if args.parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                futures = [
                    ex.submit(run_inner, model_bytes, outer, i)
                    for i in range(args.n_inner)
                ]

                for f in concurrent.futures.as_completed(futures):
                    outer_id, inner_id, preds, gts, readins = f.result()

                    # store predictions per distribution
                    for i, eval_dist in enumerate(EVAL_DISTS):
                        timeseries_store[eval_dist].append(
                            (outer_id, inner_id, preds[i].copy())
                        )

                    # store GT once per run
                    for gt in gts:
                        timeseries_store["gt"].append((outer_id, inner_id, gt.copy()))

                    # store read-ins
                    for d in EVAL_DISTS:
                        readin_store[d].append(
                            (outer_id, inner_id, readins[d])
                        )

        else:
            for i in range(args.n_inner):
                outer_id, inner_id, preds, gts, readins = run_inner(
                    model_bytes, outer, i
                )

                for j, eval_dist in enumerate(EVAL_DISTS):
                    timeseries_store[eval_dist].append(
                        (outer_id, inner_id, preds[j].copy())
                    )

                for gt in gts:
                    timeseries_store["gt"].append((outer_id, inner_id, gt.copy()))

                for d in EVAL_DISTS:
                    readin_store[d].append(
                        (outer_id, inner_id, readins[d])
                    )

        print(f"Outer {outer+1} done")

    # ------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------
    for eval_dist in EVAL_DISTS:
        np.save(
            os.path.join(OUTPUT_DIR, f"sc3_readin_weights_{eval_dist}.npy"),
            np.array(readin_store[eval_dist], dtype=object)
        )

        np.save(
            os.path.join(OUTPUT_DIR, f"sc3_timeseries_{eval_dist}.npy"),
            np.array(timeseries_store[eval_dist], dtype=object)
        )

    np.save(
        os.path.join(OUTPUT_DIR, "sc3_timeseries_gt.npy"),
        np.array(timeseries_store["gt"], dtype=object)
    )

    np.save(
        os.path.join(OUTPUT_DIR, "sc3_reservoir_weights.npy"),
        np.array(reservoir_store, dtype=object)
    )

    print("DONE")


if __name__ == "__main__":
    main()