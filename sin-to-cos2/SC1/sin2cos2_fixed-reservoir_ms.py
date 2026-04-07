"""
Sine-to-Cosine^2 — Unconditional Variability Extraction, fixed reservoir.
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
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.helpers import load_dataset, create_model, predict_sequences, sample_readin_weights, assert_weights_above_threshold

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "fixed-reservoir")
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Run RC model with customizable hyperparameters.")
parser.add_argument("--n_trials",         type=int,   default=3,    help="Number of outer trials (reservoirs)")
parser.add_argument("--n_inner",          type=int,   default=10,    help="Number of inner trials per reservoir")
parser.add_argument("--reservoir_nodes",  type=int,   default=200)
parser.add_argument("--density",          type=float, default=0.15)
parser.add_argument("--spectral_radius",  type=float, default=0.9)
parser.add_argument("--leakage_rate",     type=float, default=0.1)
parser.add_argument("--fraction_input",   type=float, default=1.0)
parser.add_argument("--ridge_alpha",      type=float, default=0.1)
parser.add_argument("--set_threshold",    type=bool,  default=True)
parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--task",             type=str,   default="sin_to_cos2")
parser.add_argument("--constraint_set",   type=str,   default="1", choices=["1", "2", "3"],
                    help="Unused; kept for compatibility")

args = parser.parse_args()

GAUSS_SD  = 1.0
THRESHOLD = args.readin_threshold if args.set_threshold else None


def _fit_and_predict(model_serialized, weights, X_train, y_train, X_test, y_test):
    """Inject read-in weights into a fresh model copy, fit, and return predictions."""
    model = pickle.loads(model_serialized)
    model._set_readin_weights(weights)
    model.fit(X_train, y_train)
    return predict_sequences(model, X_test, y_test)


def run_inner_trial(model_serialized, X_train, y_train, X_test, y_test,
                    trial_outer, trial_inner, readin_records, pred_records, record_lock):
    """
    Run one inner trial for all distributions.

    Samples read-in weights, fits a fresh model copy per distribution, and
    records the full time series into pred_records (thread-safe).
    """
    outer_idx = trial_outer + 1  # 1-based for storage
    inner_idx = trial_inner + 1
    readin_shape = (args.reservoir_nodes, 1)

    weights = {
        "uniform":         sample_readin_weights(readin_shape, "random_uniform",  threshold=THRESHOLD),
        "gaussian":        sample_readin_weights(readin_shape, "random_normal",   sd=GAUSS_SD, threshold=THRESHOLD),
        "double_gaussian": sample_readin_weights(readin_shape, "double_gaussian", sd=GAUSS_SD, threshold=THRESHOLD),
        "laplace":         sample_readin_weights(readin_shape, "laplace",         threshold=THRESHOLD),
        "power_law":       sample_readin_weights(readin_shape, "power_law",       threshold=THRESHOLD),
    }

    # Assert that all weights meet the threshold requirement before fitting any models.
    for dist, w in weights.items():
        assert_weights_above_threshold(w, THRESHOLD, dist)

    # Fit and predict for each distribution; gt_seq is the same for all, so we can reuse it.
    gt_seq, pred_uniform  = _fit_and_predict(model_serialized, weights["uniform"],         X_train, y_train, X_test, y_test)
    _,      pred_gauss    = _fit_and_predict(model_serialized, weights["gaussian"],        X_train, y_train, X_test, y_test)
    _,      pred_dbgauss  = _fit_and_predict(model_serialized, weights["double_gaussian"], X_train, y_train, X_test, y_test)
    _,      pred_laplace  = _fit_and_predict(model_serialized, weights["laplace"],         X_train, y_train, X_test, y_test)
    _,      pred_powlaw   = _fit_and_predict(model_serialized, weights["power_law"],       X_train, y_train, X_test, y_test)

    with record_lock:
        for dist, w in weights.items():
            readin_records[dist].append((outer_idx, inner_idx, w.flatten().copy()))

        pred_records["gt"].append((outer_idx, inner_idx, gt_seq.copy()))  # gt is distribution-independent
        pred_records["uniform"].append((outer_idx, inner_idx, pred_uniform.copy()))
        pred_records["gaussian"].append((outer_idx, inner_idx, pred_gauss.copy()))
        pred_records["double_gaussian"].append((outer_idx, inner_idx, pred_dbgauss.copy()))
        pred_records["laplace"].append((outer_idx, inner_idx, pred_laplace.copy()))
        pred_records["power_law"].append((outer_idx, inner_idx, pred_powlaw.copy()))


def _save_records(records, path_template):
    """Serialise a dict of (outer, inner, array) records to one .npy file each."""
    for key, entries in records.items():
        if not entries:
            continue
        outer   = np.array([e[0] for e in entries], dtype=np.int32)
        inner   = np.array([e[1] for e in entries], dtype=np.int32)
        data    = np.stack([e[2] for e in entries], axis=0)
        arr     = np.zeros((len(entries), 2 + data.shape[1]), dtype=np.float64)
        arr[:, 0]  = outer
        arr[:, 1]  = inner
        arr[:, 2:] = data
        np.save(path_template.format(key), arr)


def save_checkpoint(output_dir, readin_records, pred_records, reservoir_records, y_test):
    """
    Write all accumulated experiment data to disk, overwriting existing files.

    Call after each outer trial so the on-disk state always reflects a consistent,
    complete snapshot of everything collected so far. If the run is interrupted,
    all data up to the last completed outer trial is recoverable.

    Files written
    -------------
    sc1_ground_truth.npy          shape (n_time, n_states)  — saved every call but never changes
    sc1_reservoir_weights.npy     shape (n_outer_so_far, 1 + n_reservoir_weights)
    sc1_readin_weights_{dist}.npy shape (n_rows, 2 + n_nodes)   columns: outer, inner, w...
    sc1_timeseries_{dist}.npy     shape (n_rows, 2 + n_time)    columns: outer, inner, t...
    sc1_timeseries_gt.npy         shape (n_rows, 2 + n_time)    columns: outer, inner, t...
    """
    # Ground truth — constant, written once (overwrite is a no-op after first call)
    np.save(os.path.join(output_dir, "sc1_ground_truth.npy"), y_test[0])

    # Reservoir weights — one row per completed outer trial
    if reservoir_records:
        outer      = np.array([r[0] for r in reservoir_records], dtype=np.int32)
        res_w      = np.stack([r[1] for r in reservoir_records], axis=0)
        arr        = np.zeros((len(reservoir_records), 1 + res_w.shape[1]), dtype=np.float64)
        arr[:, 0]  = outer
        arr[:, 1:] = res_w
        np.save(os.path.join(output_dir, "sc1_reservoir_weights.npy"), arr)

    # Read-in weights and predictions — one file per distribution
    _save_records(readin_records, os.path.join(output_dir, "sc1_readin_weights_{}.npy"))
    _save_records(pred_records,   os.path.join(output_dir, "sc1_timeseries_{}.npy"))

    n_outer = len(reservoir_records)
    print(f"  [checkpoint] Saved after outer trial {n_outer}")


def main():
    np.random.seed(42)

    if args.task != "sin_to_cos2":
        raise NotImplementedError(f"Task {args.task} not implemented")

    X_train, X_test, y_train, y_test = load_dataset("sin-to-cos2")

    start_time = time.time()

    dist_keys = ["uniform", "gaussian", "double_gaussian", "laplace", "power_law"]
    readin_records = {k: [] for k in dist_keys}
    pred_records   = {k: [] for k in ["gt"] + dist_keys}
    reservoir_records = []
    record_lock = threading.Lock()

    print(
        f"\n{'='*60}\n"
        f"  Experiment : Fixed-Reservoir | Constraint Set 1 | {args.task}\n"
        f"  Reservoir  : {args.reservoir_nodes} nodes | density {args.density} | "
        f"spec_rad {args.spectral_radius} | leakage {args.leakage_rate}\n"
        f"  Read-ins   : {len(dist_keys)} distributions | SD {GAUSS_SD} (Gaussian) | "
        f"threshold {'off' if THRESHOLD is None else THRESHOLD}\n"
        f"  Trials     : {args.n_trials} reservoirs × {args.n_inner} read-in samples each\n"
        f"{'='*60}\n"
    )

    for trial_outer in range(args.n_trials):
        print(f"[Reservoir {trial_outer + 1}/{args.n_trials}] Drawing new reservoir — "
              f"weights fixed for this outer trial")
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
        reservoir_records.append((trial_outer + 1, reservoir_layer.weights.flatten().copy()))

        print(f"  → {args.n_inner} inner trials | reservoir fixed, "
              f"read-in weights resampled per trial per distribution")

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(run_inner_trial, model_serialized,
                                X_train, y_train, X_test, y_test,
                                trial_outer, inner_idx,
                                readin_records, pred_records, record_lock)
                for inner_idx in range(args.n_inner)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                print(f"  [{completed}/{args.n_inner}] inner trials complete")

        print(f"  Outer trial {trial_outer + 1}/{args.n_trials} done")
        save_checkpoint(OUTPUT_DIR, readin_records, pred_records, reservoir_records, y_test)
        print()

    print(f"Total time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    main()
