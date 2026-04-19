"""
NARMA-10 — Unconditional Variability Extraction, fixed Read-In | Variable Reservoir.
Constraint Set 1: full input (no masking), no near-zero read-in weights.
Gaussian SD fixed at 1.0.
"""

import os
import sys
import numpy as np
import argparse
import concurrent.futures
import pickle
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

parser.add_argument("--n_trials", type=int, default=2)
parser.add_argument("--n_inner", type=int, default=3)

parser.add_argument("--nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.4)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.2)
parser.add_argument("--fraction_input", type=float, default=1.0)
parser.add_argument("--ridge_alpha", type=float, default=1e-6)

parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--parallel", action="store_true", default=True)

args = parser.parse_args()

GAUSS_SD = 1.0
THRESHOLD = args.readin_threshold
lock = threading.Lock()

# ------------------------------------------------------------
# HELPERS 
# ------------------------------------------------------------
def _flat(x):
    return np.asarray(x).squeeze().astype(np.float32).ravel()

def _safe_stack(records):
    return np.stack([_flat(r[2]) for r in records], axis=0)

def _safe_inner(records):
    return np.array([r[1] for r in records], dtype=np.int32)

def _safe_outer(records):
    return np.array([r[0] for r in records], dtype=np.int32)

def save_array(path, records):
    if not records:
        return

    outer = _safe_outer(records)
    inner = _safe_inner(records)
    data  = _safe_stack(records)

    arr = np.zeros((len(records), 2 + data.shape[1]), dtype=np.float32)
    arr[:, 0] = outer
    arr[:, 1] = inner
    arr[:, 2:] = data

    np.save(path, arr)

# ------------------------------------------------------------
# INNER TRIAL
# ------------------------------------------------------------
def run_inner(model_bytes, outer_id, inner_id, readin_set):

    model, reservoir = pickle.loads(model_bytes)

    results = {}
    gt_ref = None

    for dist, sampler in DIST_MAP.items():

        W = sample_readin_weights(
            shape=(args.nodes, X_train.shape[2]),
            method=sampler,
            threshold=THRESHOLD
        )

        assert_weights_above_threshold(W, THRESHOLD, dist)

        m = pickle.loads(model_bytes)[0]
        m._set_readin_weights(W)
        m.fit(X_train, y_train)

        gt, pred = predict_sequences(m, X_test, y_test)

        pred = _flat(pred)   # 🔥 FIX
        gt_ref = _flat(gt)

        results[dist] = pred

    return outer_id, inner_id, reservoir.weights.copy(), results, gt_ref


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    global X_train, X_test, y_train, y_test

    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_dataset("narma10")

    readin_store = {d: [] for d in EVAL_DISTS}
    timeseries_store = {d: [] for d in EVAL_DISTS}
    timeseries_store["gt"] = []
    reservoir_store = []

    print("\nNARMA-10 | FIXED READ-IN / VARIABLE RESERVOIR (FIXED)\n")

    for outer in range(args.n_trials):

        print(f"[Outer {outer+1}] sampling read-in sets")

        readin_set = {
            k: sample_readin_weights(
                (args.nodes, X_train.shape[2]),
                DIST_MAP[k],
                threshold=THRESHOLD
            )
            for k in EVAL_DISTS
        }

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

        model_bytes = pickle.dumps((model, reservoir))

        if args.parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                futures = [
                    ex.submit(run_inner, model_bytes, outer, i, readin_set)
                    for i in range(args.n_inner)
                ]

                for f in concurrent.futures.as_completed(futures):
                    outer_id, inner_id, res_w, preds, gt = f.result()

                    reservoir_store.append((outer_id, inner_id, _flat(res_w)))

                    for d in EVAL_DISTS:
                        timeseries_store[d].append((outer_id, inner_id, _flat(preds[d])))
                        readin_store[d].append((outer_id, inner_id, readin_set[d].ravel()))

                    timeseries_store["gt"].append((outer_id, inner_id, gt))

        else:
            for i in range(args.n_inner):
                outer_id, inner_id, res_w, preds, gt = run_inner(
                    model_bytes, outer, i, readin_set
                )

                reservoir_store.append((outer_id, inner_id, _flat(res_w)))

                for d in EVAL_DISTS:
                    timeseries_store[d].append((outer_id, inner_id, _flat(preds[d])))
                    readin_store[d].append((outer_id, inner_id, readin_set[d].ravel()))

                timeseries_store["gt"].append((outer_id, inner_id, gt))

        print(f"[Outer {outer+1}] done")

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------
    for d in EVAL_DISTS:
        save_array(os.path.join(OUTPUT_DIR, f"sc1_readin_weights_{d}.npy"),
                   readin_store[d])

        save_array(os.path.join(OUTPUT_DIR, f"sc1_timeseries_{d}.npy"),
                   timeseries_store[d])

    save_array(os.path.join(OUTPUT_DIR, "sc1_timeseries_gt.npy"),
               timeseries_store["gt"])

    save_array(os.path.join(OUTPUT_DIR, "sc1_reservoir_weights.npy"),
               reservoir_store)

    print("DONE")


if __name__ == "__main__":
    main()