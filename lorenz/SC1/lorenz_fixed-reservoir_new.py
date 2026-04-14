"""
Lorenz — Unconditional Variability Extraction, fixed reservoir.
Constraint Set 1: full input (no masking), no near-zero read-in weights.
Gaussian SD is fixed at 1.0; no SD optimisation is performed.
"""

import os
import sys
import time
import numpy as np
import argparse
import pickle
import threading
import concurrent.futures
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.helpers import (
    load_dataset,
    create_model,
    predict_sequences,
    sample_readin_weights,
    assert_weights_above_threshold
)

# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs",
    "fixed-reservoir"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--n_trials", type=int, default=2,     help="Number of outer trials (reservoirs)") #50
parser.add_argument("--n_inner", type=int,   default=3,    help="Number of inner trials per reservoir") #100

parser.add_argument("--reservoir_nodes", type=int, default=300)
parser.add_argument("--density", type=float, default=0.1)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.5)
parser.add_argument("--fraction_input", type=float, default=1.0)

parser.add_argument("--ridge_alpha", type=float, default=1e-3)
parser.add_argument("--rollout_steps", type=int, default=5)

parser.add_argument("--task", type=str, default="lorenz")

parser.add_argument("--set_threshold", type=bool, default=True)
parser.add_argument("--readin_threshold", type=float, default=1e-3)

parser.add_argument("--parallel", action="store_true", default=True)

args = parser.parse_args()

GAUSS_SD = 1.0
THRESHOLD = args.readin_threshold if args.set_threshold else None

# ------------------------------------------------------------------
def build_model():
    return create_model(
        input_shape=(1, 3),
        output_shape=(1, 3),
        nodes=args.reservoir_nodes,
        density=args.density,
        spectral_radius=args.spectral_radius,
        leakage_rate=args.leakage_rate,
        fraction_input=args.fraction_input,
        ridge_alpha=args.ridge_alpha,
    )

# ------------------------------------------------------------------
def run_inner_trial(model_serialized,
                    X_train, y_train,
                    X_test, y_test,
                    true_traj, scaler,
                    outer, inner,
                    readin_records,
                    pred_records,
                    lock):

    outer_id = outer + 1
    inner_id = inner + 1
    shape = (args.reservoir_nodes, 3)

    # -----------------------
    # sample read-in weights
    # -----------------------
    weights = {
        "uniform": sample_readin_weights(shape, "random_uniform", threshold=THRESHOLD),
        "gaussian": sample_readin_weights(shape, "random_normal", sd=GAUSS_SD, threshold=THRESHOLD),
        "double_gaussian": sample_readin_weights(shape, "double_gaussian", sd=GAUSS_SD, threshold=THRESHOLD),
        "laplace": sample_readin_weights(shape, "laplace", threshold=THRESHOLD),
        "power_law": sample_readin_weights(shape, "power_law", threshold=THRESHOLD),
    }

    for k, w in weights.items():
        assert_weights_above_threshold(w, THRESHOLD, k)

    # -----------------------
    # store full sequences (SC1-style)
    # -----------------------
    def run_model(W):
        m = pickle.loads(model_serialized)
        m._set_readin_weights(W)
        m.fit(X_train, y_train)
        gt, pred = predict_sequences(m, X_test, y_test)
        return gt, pred

    gt, pred_u = run_model(weights["uniform"])
    _,  pred_g = run_model(weights["gaussian"])
    _,  pred_d = run_model(weights["double_gaussian"])
    _,  pred_l = run_model(weights["laplace"])
    _,  pred_p = run_model(weights["power_law"])

    # -----------------------
    # SC1-style prediction storage
    # -----------------------
    with lock:
        for k, w in weights.items():
            readin_records[k].append((outer_id, inner_id, w.flatten().copy()))

        pred_records["gt"].append((outer_id, inner_id, gt.copy()))
        pred_records["uniform"].append((outer_id, inner_id, pred_u.copy()))
        pred_records["gaussian"].append((outer_id, inner_id, pred_g.copy()))
        pred_records["double_gaussian"].append((outer_id, inner_id, pred_d.copy()))
        pred_records["laplace"].append((outer_id, inner_id, pred_l.copy()))
        pred_records["power_law"].append((outer_id, inner_id, pred_p.copy()))

# ------------------------------------------------------------------
def save_records(records, path_template):
    for key, entries in records.items():
        if not entries:
            continue

        outer = np.array([e[0] for e in entries], dtype=np.int32)
        inner = np.array([e[1] for e in entries], dtype=np.int32)
        data  = np.stack([e[2] for e in entries], axis=0)

        arr = np.zeros((len(entries), 2 + data.shape[1]), dtype=np.float64)
        arr[:, 0] = outer
        arr[:, 1] = inner
        arr[:, 2:] = data

        np.save(path_template.format(key), arr)

# ------------------------------------------------------------------
def main():
    np.random.seed(42)
    start = time.time()

    # -----------------------
    # dataset + Option A scaler
    # -----------------------
    X_train, X_test, y_train, y_test = load_dataset("lorenz")

    scaler = StandardScaler().fit(X_train.reshape(-1, 3))

    X_train = scaler.transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
    X_test  = scaler.transform(X_test.reshape(-1, 3)).reshape(X_test.shape)
    y_train = scaler.transform(y_train.reshape(-1, 3)).reshape(y_train.shape)
    y_test  = scaler.transform(y_test.reshape(-1, 3)).reshape(y_test.shape)

    true_traj = X_test.reshape(-1, 3)

    # -----------------------
    # model
    # -----------------------
    model, reservoir = build_model()
    model_serialized = pickle.dumps(model)

    # -----------------------
    # storage
    # -----------------------
    dist_keys = ["uniform", "gaussian", "double_gaussian", "laplace", "power_law"]

    readin_records = {k: [] for k in dist_keys}
    pred_records = {k: [] for k in ["gt"] + dist_keys}
    reservoir_records = []
    lock = threading.Lock()

    # -----------------------
    # ground truth (SC1 required)
    # -----------------------
    np.save(os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"), true_traj)

    # -----------------------
    # outer loop
    # -----------------------
    for outer in range(args.n_trials):
        print(f"[Outer {outer+1}]")

        model, reservoir = build_model()
        model_serialized = pickle.dumps(model)

        reservoir_records.append(
            (outer + 1, reservoir.weights.flatten().copy())
        )

        if args.parallel:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                futures = [
                    ex.submit(
                        run_inner_trial,
                        model_serialized,
                        X_train, y_train,
                        X_test, y_test,
                        true_traj, scaler,
                        outer, i,
                        readin_records,
                        pred_records,
                        lock
                    )
                    for i in range(args.n_inner)
                ]
                for f in futures:
                    f.result()
        else:
            for i in range(args.n_inner):
                run_inner_trial(
                    model_serialized,
                    X_train, y_train,
                    X_test, y_test,
                    true_traj, scaler,
                    outer, i,
                    readin_records,
                    pred_records,
                    lock
                )

    # -----------------------
    # SAVE SC1-COMPLETE OUTPUT SET
    # -----------------------
    np.save(
        os.path.join(OUTPUT_DIR, "sc1_results_fixed-reservoir.npy"),
        np.array([])  # optional placeholder (you can extend later)
    )

    save_records(readin_records, os.path.join(OUTPUT_DIR, "sc1_readin_weights_{}.npy"))
    save_records(pred_records,   os.path.join(OUTPUT_DIR, "sc1_timeseries_{}.npy"))

    arr = np.array([
        np.concatenate(([o], w))
        for o, w in reservoir_records
    ], dtype=object)

    np.save(os.path.join(OUTPUT_DIR, "sc1_reservoir_weights.npy"), arr)

    print("\nDONE")
    print("Time:", time.time() - start)


if __name__ == "__main__":
    main()