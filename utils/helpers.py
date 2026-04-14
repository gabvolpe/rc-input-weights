"""
Shared utilities for all RC experiment scripts.

Public API
----------
load_dataset(name)
    Load a pre-generated dataset by name and return (X_train, X_test, y_train, y_test).

create_model(input_shape, output_shape, nodes, density, spectral_radius,
             leakage_rate, fraction_input, ridge_alpha)
    Build, compile and return an RC model instance together with its reservoir layer.

r2_score(y_true, y_pred, washout=0)
    Compute R² between prediction and ground truth, with optional washout.

predict_sequences(fitted_model, X_test, y_test, channels=None)
    Run prediction and return full time series for the first batch sample.
    Optionally select a subset of output channels.

sample_readin_weights(shape, method, sd=1.0, threshold=None)
    Sample a weight matrix from the given distribution, with optional
    iterative resampling to enforce a near-zero exclusion threshold.

assert_weights_above_threshold(weights, threshold, label="weights")
    Raise ValueError if any weight violates the threshold constraint.

Supported methods
-----------------
    "random_uniform"    Uniform[-1, 1]
    "random_normal"     Gaussian(0, sd)
    "double_gaussian"   Bimodal Gaussian at means ±1.5, equal mixing, width sd
    "laplace"           Laplace(loc=0, scale=0.5)
    "power_law"         Symmetric power-law, exponent=2, values in (-1,0)∪(0,1]

Adding a new distribution
--------------------------
Implement a sampler with signature  (shape: tuple) -> np.ndarray
and register it in the ``_samplers`` dict inside ``sample_readin_weights``.
No other changes required.
"""

import os
import numpy as np
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.optimizers import RidgeSK


# Project root is one level above this file (utils/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Maps dataset name → subdirectory inside the project root
_DATASET_DIRS = {
    "sin-to-cos2":  "sin-to-cos2",
    "lorenz":       "lorenz",
    "mackey-glass": "mackey-glass",
    "narma10":      "narma10",
}


def load_dataset(name):
    """
    Load a pre-generated dataset by name.

    Expects four files in the dataset's directory:
        X_train.npy, X_test.npy, y_train.npy, y_test.npy

    Args:
        name: Dataset name, one of: 'sin-to-cos2', 'lorenz', 'mackey-glass', 'narma10'.

    Returns:
        X_train, X_test, y_train, y_test as numpy arrays.

    Raises:
        ValueError if the name is not recognised.
        FileNotFoundError if the .npy files are missing (run the corresponding
        gen_*_data.py script first).
    """
    if name not in _DATASET_DIRS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {sorted(_DATASET_DIRS)}")

    data_dir = os.path.join(_PROJECT_ROOT, _DATASET_DIRS[name])
    files = {"X_train": None, "X_test": None, "y_train": None, "y_test": None}

    try:
        return tuple(np.load(os.path.join(data_dir, f"{k}.npy")) for k in files)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset '{name}' not found in {data_dir}. "
            "Run the corresponding gen_*_data.py script first."
        ) from exc


# ------------------------------------------------------------------ #
# RC model factory                                                    #
# ------------------------------------------------------------------ #

def create_model(
    input_shape,
    output_shape,
    nodes,
    density,
    spectral_radius,
    leakage_rate,
    fraction_input,
    ridge_alpha,
):
    """
    Build and compile an RC model with a random reservoir.

    Args:
        input_shape:      Tuple (n_time, n_features) of the input data.
        output_shape:     Tuple (n_time, n_features) of the target data.
        nodes:            Number of reservoir nodes.
        density:          Connection density of the reservoir graph.
        spectral_radius:  Spectral radius of the reservoir weight matrix.
        leakage_rate:     Leakage rate (α) of the reservoir neurons.
        fraction_input:   Fraction of reservoir nodes that receive input (1.0 = all).
        ridge_alpha:      Regularisation strength for the Ridge readout.

    Returns:
        model_rc:         Compiled RC model instance.
        reservoir_layer:  The RandomReservoirLayer, giving access to its weights.
    """
    model_rc = RC()
    model_rc.add(InputLayer(input_shape=input_shape))
    reservoir_layer = RandomReservoirLayer(
        nodes=nodes,
        density=density,
        activation="tanh",
        spec_rad=spectral_radius,
        leakage_rate=leakage_rate,
        fraction_input=fraction_input,
    )
    model_rc.add(reservoir_layer)
    model_rc.add(ReadoutLayer(output_shape, fraction_out=1.0))
    optim = RidgeSK(alpha=ridge_alpha)
    model_rc.compile(optimizer=optim, metrics=["mean_squared_error"])
    return model_rc, reservoir_layer


# ------------------------------------------------------------------ #
# Private per-distribution samplers                                   #
# Signature: (shape: tuple) -> np.ndarray                            #
# ------------------------------------------------------------------ #

def _sample_uniform(shape):
    return np.random.uniform(-1, 1, size=shape)


def _sample_gaussian(sd):
    """Return a zero-mean Gaussian sampler with the given SD."""
    def _sample(shape):
        return np.random.normal(0.0, sd, size=shape)
    return _sample


def _sample_double_gaussian(sd):
    """Return a bimodal Gaussian sampler (means ±1.5, equal mixing, width sd)."""
    def _sample(shape):
        n = int(np.prod(shape))
        choices = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
        g1 = np.random.normal(-1.5, sd, size=n)
        g2 = np.random.normal( 1.5, sd, size=n)
        return np.where(choices == 0, g1, g2).reshape(shape)
    return _sample


def _sample_laplace(shape):
    return np.random.laplace(loc=0.0, scale=0.5, size=shape)


def _sample_power_law(shape):
    """Symmetric power-law (exponent=2). Draws positives, mirrors to negatives,
    then samples without replacement to avoid duplicate values."""
    a = 2.0
    n = int(np.prod(shape))
    positive = np.random.power(a, size=n)
    combined = np.concatenate([positive, -positive])
    return np.random.choice(combined, size=n, replace=False).reshape(shape)


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def sample_readin_weights(shape, method, sd=1.0, threshold=None):
    """
    Sample a read-in weight matrix from the specified distribution.

    Args:
        shape:     Tuple defining the weight matrix shape, e.g. (n_nodes, 1).
        method:    Distribution name (see module docstring for options).
        sd:        Standard deviation for Gaussian-based methods (default 1.0).
        threshold: If provided, iteratively resample weights with |w| < threshold
                   until all weights satisfy |w| >= threshold. Pass None to
                   disable (no constraint enforced).

    Returns:
        np.ndarray of the requested shape.

    Raises:
        ValueError for unknown method names.
    """
    _samplers = {
        "random_uniform":  _sample_uniform,
        "random_normal":   _sample_gaussian(sd),
        "double_gaussian": _sample_double_gaussian(sd),
        "laplace":         _sample_laplace,
        "power_law":       _sample_power_law,
    }

    if method not in _samplers:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {sorted(_samplers)}"
        )

    sampler = _samplers[method]
    w = sampler(shape).flatten()

    if threshold is not None and threshold is not False:
        while np.any(np.abs(w) < threshold):
            idx = np.abs(w) < threshold
            w[idx] = sampler((int(np.sum(idx)),)).flatten()

    return w.reshape(shape)


def r2_score(y_true, y_pred, washout=0):
    """
    Compute R² (coefficient of determination) between prediction and ground truth.

    Args:
        y_true:  1-D ground-truth array of shape (n_time,).
        y_pred:  1-D prediction array of shape (n_time,).
        washout: Number of initial timesteps to discard before scoring.

    Returns:
        float R² score. Returns 1.0 if both signals are constant and identical,
        0.0 if the prediction is no better than the mean of y_true.
    """
    y_true = np.asarray(y_true, dtype=float)[washout:]
    y_pred = np.asarray(y_pred, dtype=float)[washout:]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def predict_sequences(fitted_model, X_test, y_test, channels=None):
    """
    Run prediction and return the full time series for the first batch sample.

    Args:
        fitted_model: A fitted RC model with a .predict() method.
        X_test:       Input array of shape (n_batch, n_time, n_features).
        y_test:       Ground-truth array of shape (n_batch, n_time, n_states).
        channels:     Which output channel(s) to return.
                        None  — return only channel 0, as a 1-D array (n_time,).
                        int   — return that single channel, as a 1-D array (n_time,).
                        list  — return those channels, as a 2-D array (n_time, len(channels)).

    Returns:
        y_true_seq: np.ndarray sliced according to `channels`.
        y_pred_seq: np.ndarray sliced according to `channels`.
    """
    y_pred = fitted_model.predict(X_test)

    if channels is None or isinstance(channels, int):
        ch = 0 if channels is None else channels
        return y_test[0, :, ch], y_pred[0, :, ch]

    # list of channels → 2-D output (n_time, n_channels)
    return y_test[0, :, channels], y_pred[0, :, channels]


def assert_weights_above_threshold(weights, threshold, label="weights"):
    """
    Assert that all read-in weights satisfy the near-zero exclusion constraint.

    Checks that no weight has absolute value below threshold. This corresponds
    to Constraint Set 1 (no near-zero read-in values). Other constraint types
    (e.g. masking, sign constraints) are checked separately.

    Args:
        weights:   np.ndarray of read-in weights to validate.
        threshold: Minimum allowed absolute weight value. Pass None to skip.
        label:     Human-readable distribution name used in the error message.

    Raises:
        ValueError if any weight violates the threshold constraint.
    """
    if threshold is None:
        return

    violations = np.sum(np.abs(weights) < threshold)
    if violations > 0:
        raise ValueError(
            f"{label}: {violations} weight(s) below threshold {threshold}. "
            f"Min |w| = {np.min(np.abs(weights)):.2e}"
        )
