"""
NARMA-10 — Unconditional Variability Extraction, fixed reservoir.

Constraint Set 1:
- Full input (no masking)
- No near-zero read-in weights

Gaussian SD is fixed at 1.0 (no SD optimisation).
"""

import os
import sys
import numpy as np
import time
import argparse
import concurrent.futures
import pickle
import threading

# --- helpers import ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.helpers import (
    load_dataset,
    create_model,
    predict_sequences,
    sample_readin_weights,
    assert_weights_above_threshold
)

# ----------------------
# Output directories
# ----------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "outputs", "fixed-reservoir")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# --- User Arguments ---
# ----------------------
parser = argparse.ArgumentParser()

parser.add_argument("--n_trials", type=int, default=50)
parser.add_argument("--n_inner", type=int, default=100)

parser.add_argument("--reservoir_nodes", type=int, default=50)
parser.add_argument("--density", type=float, default=0.4)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.2)
parser.add_argument("--fraction_input", type=float, default=1.0)

parser.add_argument("--ridge_alpha", type=float, default=1e-6)

parser.add_argument("--set_threshold", type=bool, default=True)
parser.add_argument("--readin_threshold", type=float, default=1e-3)

parser.add_argument("--task", type=str, default="narma10_seq2seq",
                    choices=["narma10_seq2seq", "narma10_seq2scalar"])

parser.add_argument("--parallel", action="store_true", default=True)

args = parser.parse_args()

GAUSS_SD  = 1.0
THRESHOLD = args.readin_threshold if args.set_threshold else None

# ----------------------
# Profiling
# ----------------------
_timing = {"sample": 0.0, "deserialize": 0.0, "set_weights": 0.0,
           "fit": 0.0, "predict": 0.0}
_timing_lock = threading.Lock()

# ----------------------
# Core fit/predict
# ----------------------
def _fit_and_predict(model_serialized, weights, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    model = pickle.loads(model_serialized)
    t1 = time.perf_counter()

    model._set_readin_weights(weights)
    t2 = time.perf_counter()

    model.fit(X_train, y_train)
    t3 = time.perf_counter()

    gt_seq, pred_seq = predict_sequences(model, X_test, y_test)
    t4 = time.perf_counter()

    return gt_seq, pred_seq, {
        "deserialize": t1 - t0,
        "set_weights": t2 - t1,
        "fit": t3 - t2,
        "predict": t4 - t3,
    }

# ----------------------
# Inner trial
# ----------------------
def run_inner_trial(model_serialized, X_train, y_train, X_test, y_test,
                    trial_outer, trial_inner,
                    readin_records, pred_records, record_lock):

    outer_idx = trial_outer + 1
    inner_idx = trial_inner + 1

    readin_shape = (args.reservoir_nodes, X_train.shape[2])

    # ---- sample weights ----
    t0 = time.perf_counter()
    weights = {
        "uniform":         sample_readin_weights(readin_shape, "random_uniform",  threshold=THRESHOLD),
        "gaussian":        sample_readin_weights(readin_shape, "random_normal",   sd=GAUSS_SD, threshold=THRESHOLD),
        "double_gaussian": sample_readin_weights(readin_shape, "double_gaussian", sd=GAUSS_SD, threshold=THRESHOLD),
        "laplace":         sample_readin_weights(readin_shape, "laplace",         threshold=THRESHOLD),
        "power_law":       sample_readin_weights(readin_shape, "power_law",       threshold=THRESHOLD),
    }
    t_sample = time.perf_counter() - t0

    for k, w in weights.items():
        assert_weights_above_threshold(w, THRESHOLD, k)

    # ---- fit models ----
    gt_seq, pred_uniform,  t_u  = _fit_and_predict(model_serialized, weights["uniform"],         X_train, y_train, X_test, y_test)
    _,      pred_gauss,    t_g  = _fit_and_predict(model_serialized, weights["gaussian"],        X_train, y_train, X_test, y_test)
    _,      pred_dbgauss,  t_dg = _fit_and_predict(model_serialized, weights["double_gaussian"], X_train, y_train, X_test, y_test)
    _,      pred_laplace,  t_l  = _fit_and_predict(model_serialized, weights["laplace"],         X_train, y_train, X_test, y_test)
    _,      pred_powlaw,   t_p  = _fit_and_predict(model_serialized, weights["power_law"],       X_train, y_train, X_test, y_test)

    # ---- accumulate timing ----
    with _timing_lock:
        _timing["sample"] += t_sample
        for t in (t_u, t_g, t_dg, t_l, t_p):
            _timing["deserialize"] += t["deserialize"]
            _timing["set_weights"] += t["set_weights"]
            _timing["fit"] += t["fit"]
            _timing["predict"] += t["predict"]

    # ---- record ----
    with record_lock:
        for dist, w in weights.items():
            readin_records[dist].append((outer_idx, inner_idx, w.flatten().copy()))

        pred_records["gt"].append((outer_idx, inner_idx, gt_seq.copy()))
        pred_records["uniform"].append((outer_idx, inner_idx, pred_uniform.copy()))
        pred_records["gaussian"].append((outer_idx, inner_idx, pred_gauss.copy()))
        pred_records["double_gaussian"].append((outer_idx, inner_idx, pred_dbgauss.copy()))
        pred_records["laplace"].append((outer_idx, inner_idx, pred_laplace.copy()))
        pred_records["power_law"].append((outer_idx, inner_idx, pred_powlaw.copy()))

# ----------------------
# Save helpers
# ----------------------
def _save_records(records, template):
    for key, entries in records.items():
        if not entries:
            continue
        outer = np.array([e[0] for e in entries])
        inner = np.array([e[1] for e in entries])
        data  = np.stack([e[2] for e in entries], axis=0)

        arr = np.zeros((len(entries), 2 + data.shape[1]))
        arr[:,0], arr[:,1] = outer, inner
        arr[:,2:] = data

        np.save(template.format(key), arr)

def save_checkpoint(readin_records, pred_records, reservoir_records, y_test):
    np.save(os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"), y_test[0])

    if reservoir_records:
        outer = np.array([r[0] for r in reservoir_records])
        w     = np.stack([r[1] for r in reservoir_records])

        arr = np.zeros((len(reservoir_records), 1 + w.shape[1]))
        arr[:,0] = outer
        arr[:,1:] = w

        np.save(os.path.join(OUTPUT_DIR, "sc1_reservoir_weights.npy"), arr)

    _save_records(readin_records, os.path.join(OUTPUT_DIR, "sc1_readin_weights_{}.npy"))
    _save_records(pred_records,   os.path.join(OUTPUT_DIR, "sc1_timeseries_{}.npy"))

# ----------------------
# Main
# ----------------------
def main():
    np.random.seed(42)

    # ---- load dataset via helpers ----
    X_train, X_test, y_train, y_test = load_dataset(args.task)

    start_time = time.time()

    dist_keys = ["uniform", "gaussian", "double_gaussian", "laplace", "power_law"]

    readin_records = {k: [] for k in dist_keys}
    pred_records   = {k: [] for k in ["gt"] + dist_keys}
    reservoir_records = []

    record_lock = threading.Lock()

    for trial_outer in range(args.n_trials):

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
            (trial_outer + 1, reservoir_layer.weights.flatten().copy())
        )

        # ---- inner trials ----
        if args.parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(run_inner_trial,
                                    model_serialized,
                                    X_train, y_train, X_test, y_test,
                                    trial_outer, i,
                                    readin_records, pred_records, record_lock)
                    for i in range(args.n_inner)
                ]
                for f in concurrent.futures.as_completed(futures):
                    f.result()
        else:
            for i in range(args.n_inner):
                run_inner_trial(model_serialized,
                                X_train, y_train, X_test, y_test,
                                trial_outer, i,
                                readin_records, pred_records, record_lock)

        save_checkpoint(readin_records, pred_records, reservoir_records, y_test)

    print(f"\nTotal time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()