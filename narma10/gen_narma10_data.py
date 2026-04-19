"""
Generate the NARMA10 dataset once and save it to disk.

Saves four arrays to narma10/data/:
    X_train.npy  — shape (n_batch, n_time_in, n_states)
    X_test.npy   — shape (n_batch, n_time_in, n_states)
    y_train.npy  — shape (n_batch, n_time_out, 1)
    y_test.npy   — shape (n_batch, n_time_out, 1)

Run this script once before any experiment scripts.
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ------------------------------------------------------------------
# Output directory: project_root/narma10/
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "narma10")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate and save NARMA-10 dataset")

parser.add_argument(
    "--task",
    type=str,
    default="sequence_to_sequence",
)

parser.add_argument("--n_batch", type=int, default=2000)
parser.add_argument("--n_states", type=int, default=2)
parser.add_argument("--narma_order", type=int, default=10)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--test_size", type=float, default=0.2)

args = parser.parse_args()


# ------------------------------------------------------------------
# Core generator (from original experiment script)
# ------------------------------------------------------------------
def generate_narma10_data(n_samples=25000, random_seed=None, warmup=2000, order=10):
    """
    Generate NARMA-10 input/output series.

    Returns
    -------
    u : shape (n_samples,1)
    y : shape (n_samples,1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    u = np.random.uniform(0, 0.25, size=n_samples + warmup)
    y = np.zeros(n_samples + warmup)

    for t in range(order, n_samples + warmup):
        y[t] = (
            0.3 * y[t-1]
            + 0.05 * y[t-1] * np.sum(y[t-10:t])
            + 1.5 * u[t-10] * u[t-1]
            + 0.1
        )

    y = np.clip(y, 0, 5)

    u = u[warmup:].reshape(-1,1)
    y = y[warmup:].reshape(-1,1)

    return u, y


def narma10_pred(
    n_batch,
    n_time_in,
    n_time_out,
    n_states=2,
    washout=150,
    random_seed=None,
    order=10,
    test_size=0.2,
):
    """
    Build delayed-input multi-channel windows and return:

        X_train, X_test, y_train, y_test
    """
    total_required = n_batch + n_time_in + n_time_out + washout + order

    u, y = generate_narma10_data(
        total_required,
        random_seed=random_seed,
        order=order,
    )

    # Delayed channels
    features = [u]
    for k in range(1, n_states):
        delayed = np.roll(u, shift=order * k)
        delayed[:order*k] = u[:order*k]
        features.append(delayed)

    X_multi = np.concatenate(features, axis=1)

    # Rolling windows
    X, Y = [], []
    for i in range(n_batch):
        start = washout + i

        X.append(
            X_multi[
                start : start + n_time_in
            ]
        )

        Y.append(
            y[
                start + n_time_in : start + n_time_in + n_time_out
            ]
        )

    X = np.array(X)
    Y = np.array(Y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=42,
    )

    # Input normalization
    n_features = X.shape[2]

    x_scaler = StandardScaler().fit(
        X_train.reshape(-1, n_features)
    )

    X_train = x_scaler.transform(
        X_train.reshape(-1, n_features)
    ).reshape(X_train.shape)

    X_test = x_scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(X_test.shape)

    # Output normalization
    n_out = Y.shape[2]

    y_scaler = MinMaxScaler().fit(
        y_train.reshape(-1, n_out)
    )

    y_train = y_scaler.transform(
        y_train.reshape(-1, n_out)
    ).reshape(y_train.shape)

    y_test = y_scaler.transform(
        y_test.reshape(-1, n_out)
    ).reshape(y_test.shape)

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# Task wrappers (match original code)
# ------------------------------------------------------------------
def sequence_to_scalar(n_batch=2000, n_states=2, order=10, seed=42, test_size=0.2):
    return narma10_pred(
        n_batch=n_batch,
        n_time_in=100,
        n_time_out=1,
        n_states=n_states,
        random_seed=seed,
        order=order,
        test_size=test_size,
    )


def sequence_to_sequence(n_batch=10000, n_states=2, order=10, seed=42, test_size=0.2):
    return narma10_pred(
        n_batch=n_batch,
        n_time_in=600,
        n_time_out=10,
        n_states=n_states,
        random_seed=seed,
        order=order,
        test_size=test_size,
    )


TASKS = {
    "sequence_to_sequence": sequence_to_sequence,
    "sequence_to_scalar": sequence_to_scalar,
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    np.random.seed(args.random_seed)

    X_train, X_test, y_train, y_test = TASKS[args.task](
        n_batch=args.n_batch,
        n_states=args.n_states,
        order=args.narma_order,
        seed=args.random_seed,
        test_size=args.test_size,
    )

    # Safety checks
    assert not np.any(np.isnan(X_train)), "X_train contains NaN"
    assert not np.any(np.isnan(X_test)), "X_test contains NaN"
    assert not np.any(np.isnan(y_train)), "y_train contains NaN"
    assert not np.any(np.isnan(y_test)), "y_test contains NaN"

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print("Saved NARMA-10 dataset to:", OUTPUT_DIR)
    print("Shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)
    print("  y_train:", y_train.shape)
    print("  y_test :", y_test.shape)
    print("Task:", args.task)


if __name__ == "__main__":
    main()
