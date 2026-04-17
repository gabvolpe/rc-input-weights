"""
Sine-to-Cosine^2 — Unconditional Variability Extraction, fixed reservoir.
Constraint Set 1: full input (no masking), no near-zero read-in weights.
Gaussian SD is fixed at 1.0; no SD optimisation is performed.

Optimizations:
- Training subsampling
- Clean shape handling
- Minimal overhead per trial
"""

import os
import sys
import numpy as np
import time
import argparse
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
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs",
    "fixed-reservoir"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--n_trials", type=int, default=2,     help="Number of outer trials (reservoirs)") #50
parser.add_argument("--n_inner",          type=int,   default=2,    help="Number of inner trials per reservoir") #100

parser.add_argument("--reservoir_nodes", type=int, default=200)  # reduced default
parser.add_argument("--density", type=float, default=0.1)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.5)
parser.add_argument("--fraction_input", type=float, default=1.0)
parser.add_argument("--ridge_alpha", type=float, default=1e-3)

parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--set_threshold", action="store_true")

# Training subsample size
parser.add_argument("--max_train_samples", type=int, default=5000)

args = parser.parse_args()

GAUSS_SD = 1.0
THRESHOLD = args.readin_threshold if args.set_threshold else None

# ------------------------------------------------------------
def flatten_ts(x):
    """
    Convert any (samples, T, dim) → flat 1D signal (like evaluation expects)
    """
    return np.asarray(x).reshape(-1)

# ------------------------------------------------------------
def _fit_and_predict(model_serialized, weights, X_train, y_train, X_test, y_test):
    model = pickle.loads(model_serialized)

    model._set_readin_weights(weights)
    model.fit(X_train, y_train)

    gt_seq, pred_seq = predict_sequences(model, X_test, y_test, channels=0)

    return flatten_ts(gt_seq), flatten_ts(pred_seq)

# ------------------------------------------------------------
def run_inner_trial(
    model_serialized,
    X_train, y_train, X_test, y_test,
    outer, inner,
    readin_records, pred_records, lock
):
    outer_id = outer + 1
    inner_id = inner + 1

    shape = (args.reservoir_nodes, 3)

    weights = {
        "uniform": sample_readin_weights(shape, "random_uniform", threshold=THRESHOLD),
        "gaussian": sample_readin_weights(shape, "random_normal", sd=GAUSS_SD, threshold=THRESHOLD),
        "double_gaussian": sample_readin_weights(shape, "double_gaussian", sd=GAUSS_SD, threshold=THRESHOLD),
        "laplace": sample_readin_weights(shape, "laplace", threshold=THRESHOLD),
        "power_law": sample_readin_weights(shape, "power_law", threshold=THRESHOLD),
    }

    for k, w in weights.items():
        assert_weights_above_threshold(w, THRESHOLD, k)

    gt_seq, pred_u = _fit_and_predict(model_serialized, weights["uniform"], X_train, y_train, X_test, y_test)
    _, pred_g = _fit_and_predict(model_serialized, weights["gaussian"], X_train, y_train, X_test, y_test)
    _, pred_d = _fit_and_predict(model_serialized, weights["double_gaussian"], X_train, y_train, X_test, y_test)
    _, pred_l = _fit_and_predict(model_serialized, weights["laplace"], X_train, y_train, X_test, y_test)
    _, pred_p = _fit_and_predict(model_serialized, weights["power_law"], X_train, y_train, X_test, y_test)

    with lock:
        for k, w in weights.items():
            readin_records[k].append((outer_id, inner_id, w.flatten().copy()))

        pred_records["gt"].append((outer_id, inner_id, gt_seq))
        pred_records["uniform"].append((outer_id, inner_id, pred_u))
        pred_records["gaussian"].append((outer_id, inner_id, pred_g))
        pred_records["double_gaussian"].append((outer_id, inner_id, pred_d))
        pred_records["laplace"].append((outer_id, inner_id, pred_l))
        pred_records["power_law"].append((outer_id, inner_id, pred_p))

# ------------------------------------------------------------
def save_records(records, path_template):
    for key, entries in records.items():
        if not entries:
            continue

        outer = np.array([e[0] for e in entries])
        inner = np.array([e[1] for e in entries])
        data = np.stack([e[2] for e in entries])

        arr = np.zeros((len(entries), 2 + data.shape[1]))
        arr[:, 0] = outer
        arr[:, 1] = inner
        arr[:, 2:] = data

        np.save(path_template.format(key), arr)

# ------------------------------------------------------------
def main():
    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_dataset("lorenz")

    # SPEEDUP: subsample training data
    if len(X_train) > args.max_train_samples:
        idx = np.random.choice(len(X_train), args.max_train_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    dist_keys = ["uniform", "gaussian", "double_gaussian", "laplace", "power_law"]

    readin_records = {k: [] for k in dist_keys}
    pred_records = {k: [] for k in ["gt"] + dist_keys}
    reservoir_records = []

    lock = threading.Lock()

    print(f"\nUsing {len(X_train)} training samples")

    for outer in range(args.n_trials):
        print(f"[Reservoir {outer+1}/{args.n_trials}]")

        model_rc, reservoir_layer = create_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_shape=(y_train.shape[1], y_train.shape[2]),
            nodes=args.reservoir_nodes,
            density=args.density,
            spectral_radius=args.spectral_radius,
            leakage_rate=args.leakage_rate,
            fraction_input=args.fraction_input,
            ridge_alpha=args.ridge_alpha,
        )

        model_serialized = pickle.dumps(model_rc)

        reservoir_records.append(
            (outer + 1, reservoir_layer.weights.flatten().copy())
        )

        for inner in range(args.n_inner):
            run_inner_trial(
                model_serialized,
                X_train, y_train, X_test, y_test,
                outer, inner,
                readin_records, pred_records, lock
            )
            print(f"  inner {inner+1}/{args.n_inner}")

    # --------------------------------------------------------
    # SAVE EVERYTHING (same format as sin-to-cos2)
    # --------------------------------------------------------
    #np.save(os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"), flatten_ts(y_test[:, :, 0]))
    
    np.save(os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"), y_test[0])

    outer = np.array([r[0] for r in reservoir_records])
    res_w = np.stack([r[1] for r in reservoir_records])

    arr = np.zeros((len(reservoir_records), 1 + res_w.shape[1]))
    arr[:, 0] = outer
    arr[:, 1:] = res_w

    np.save(os.path.join(OUTPUT_DIR, "sc1_reservoir_weights.npy"), arr)

    save_records(readin_records, os.path.join(OUTPUT_DIR, "sc1_readin_weights_{}.npy"))
    save_records(pred_records, os.path.join(OUTPUT_DIR, "sc1_timeseries_{}.npy"))

    print("\nDONE")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()