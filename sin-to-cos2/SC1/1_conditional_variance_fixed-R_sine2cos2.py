"""
Sine-to-Cosine^2
Conditional Variability Extraction (Fixed Reservoir)
"""
import os
import numpy as np
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_data import sequence_to_sequence
from pyreco.optimizers import RidgeSK
import time
import argparse
import pickle

# ----------------------
# --- User Arguments ---
# ----------------------
parser = argparse.ArgumentParser(description="Run RC model with customizable hyperparameters.")

parser.add_argument("--n_trials", type=int, default=3, help="Number of trials (no outer trials anymore)")

parser.add_argument("--reservoir_nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.15)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.1)
parser.add_argument("--fraction_input", type=float, default=1.0, help="Percentage of input without masking")

parser.add_argument("--ridge_alpha", type=float, default=0.1)

parser.add_argument("--set_threshold", type=bool, default=True, help="Set to False if no threshold wished")
parser.add_argument("--readin_threshold", type=float, default=1e-3)

parser.add_argument(
    "--sd_list",
    type=str,
    default="[0.1, 0.25, 0.5, 0.75, 1.0]",
    help="List of Gaussian SD values, format: [0.1,0.25,...]"
)
parser.add_argument("--task", type=str, default="sin_to_cos2")

parser.add_argument(
    "--constraint_set",
    type=str, default="1",
    choices=["1", "2", "3"],
    help="(Unused now; only kept for compatibility)"
)

# Parse
args = parser.parse_args()
sd_list = eval(args.sd_list)

# ----------------------
# --- Functions --------
# ----------------------
def create_base_model(input_shape, output_shape):
    """
    Create a RC model with specified input and output shapes.
    
    input_shape: Shape of the input data
    output_shape: Shape of the output data
    """
    model_rc = RC()
    model_rc.add(InputLayer(input_shape=input_shape))
    reservoir_layer = RandomReservoirLayer(
        nodes=args.reservoir_nodes,
        density=args.density,
        activation="tanh",
        spec_rad=args.spectral_radius,
        leakage_rate=args.leakage_rate,
        fraction_input=args.fraction_input
    )
    model_rc.add(reservoir_layer)
    model_rc.add(ReadoutLayer(output_shape, fraction_out=1.0))
    optim = RidgeSK(alpha=args.ridge_alpha)
    model_rc.compile(optimizer=optim, metrics=["mean_squared_error"])
    return model_rc, reservoir_layer

# --- Create Weights -----
def create_weights(shape, method, dynamic_sd=None):
    if args.set_threshold is True:
        threshold = args.readin_threshold
        if method == "random_uniform":
            return np.random.uniform(-1, 1, size=shape)
        if method == "random_normal":
            mu = 0.0
            sd = dynamic_sd if dynamic_sd is not None else 1.0
            w = np.random.normal(mu, sd, size=shape)
            while np.any(np.abs(w) < threshold):
                idx = np.abs(w) < threshold
                w[idx] = np.random.normal(mu, sd, size=np.sum(idx))
            return w
        if method == "double_gaussian":
            mu1, mu2 = -1.5, 1.5
            sigma1, sigma2 = dynamic_sd, dynamic_sd
            amp1, amp2 = 0.5, 0.5
            n_elements = np.prod(shape)
            choices = np.random.choice([0, 1], size=n_elements, p=[amp1, amp2])
            g1 = np.random.normal(mu1, sigma1, size=n_elements)
            g2 = np.random.normal(mu2, sigma2, size=n_elements)
            w = np.where(choices == 0, g1, g2)
            while np.any(np.abs(w) < threshold):
                idx = np.abs(w) < threshold
                n_idx = np.sum(idx)
                g1_new = np.random.normal(mu1, sigma1, size=n_idx)
                g2_new = np.random.normal(mu2, sigma2, size=n_idx)
                w[idx] = np.where(np.random.rand(n_idx) < amp1, g1_new, g2_new)
            return w.reshape(shape)
        if method == "laplace":
            w = np.random.laplace(loc=0.0, scale=0.5, size=shape).flatten()
            while np.any(np.abs(w) < threshold):
                idx = np.abs(w) < threshold
                w[idx] = np.random.laplace(loc=0.0, scale=0.5, size=np.sum(idx))
            return w.reshape(shape)
        if method == "power_law":
            a = 2.0
            positive = np.random.power(a, size=np.prod(shape))
            negative = -positive.copy()
            combined = np.concatenate([positive, negative])
            w = np.random.choice(combined, size=np.prod(shape), replace=False)
            while np.any(np.abs(w) < threshold):
                idx = np.abs(w) < threshold
                n_new = np.sum(idx)
                new_pos = np.random.power(a, size=n_new // 2 + n_new % 2)
                new_neg = -np.random.power(a, size=n_new // 2)
                new_vals = np.concatenate([new_pos, new_neg])
                np.random.shuffle(new_vals)
                w[idx] = new_vals[:n_new]
            return w.reshape(shape)
        raise ValueError(f"Unknown weight initialization method: {method}")
    else:
        if method == "random_uniform":
            return np.random.uniform(-1, 1, size=shape)
        if method == "random_normal":
            mu = 0.0
            sd = dynamic_sd if dynamic_sd is not None else 1.0
            w = np.random.normal(mu, sd, size=shape)
            return w
        if method == "double_gaussian":
            mu1, mu2 = -1.5, 1.5
            sigma1, sigma2 = dynamic_sd, dynamic_sd
            amp1, amp2 = 0.5, 0.5
            n_elements = np.prod(shape)
            choices = np.random.choice([0, 1], size=n_elements, p=[amp1, amp2])
            g1 = np.random.normal(mu1, sigma1, size=n_elements)
            g2 = np.random.normal(mu2, sigma2, size=n_elements)
            w = np.where(choices == 0, g1, g2)
            return w.reshape(shape)
        if method == "laplace":
            w = np.random.laplace(loc=0.0, scale=0.5, size=shape).flatten()
            return w.reshape(shape)
        if method == "power_law":
            a = 2.0
            positive = np.random.power(a, size=np.prod(shape))
            negative = -positive.copy()
            combined = np.concatenate([positive, negative])
            w = np.random.choice(combined, size=np.prod(shape), replace=False)
            return w.reshape(shape)
        raise ValueError(f"Unknown weight initialization method: {method}")

def scalar_from_model(fitted_model, X_test, y_test, washout=200):
    """Return scalar gt/pred from first sample, first feature, post-washout."""
    y_pred = fitted_model.predict(X_test)
    y_true_seq = y_test[0, washout:, 0]
    y_pred_seq = y_pred[0, washout:, 0]
    y_true_scalar = float(np.mean(y_true_seq))
    y_pred_scalar = float(np.mean(y_pred_seq))
    return y_true_scalar, y_pred_scalar

# ----------------------
# --- Main Program------
# ----------------------
def main():
    if args.task == "sin_to_cos2":
        print("\nSine-to-cosine is ready\n")
        X_train, X_test, y_train, y_test = sequence_to_sequence(
            name="sin_to_cos2", n_states=1, n_batch=200, n_time=1000
        )
    else:
        raise NotImplementedError(f"Task {args.task} not implemented")

    start_time = time.time()

    # Structured dtype with named columns
    # Note: first column is now `trial`, not outer trial.
    dtype = np.dtype([
        ("trial", "i4"),
        ("gt_uniform_inner", "f8"),
        ("pred_uniform_inner", "f8"),
        ("gt_gauss_inner", "f8"),
        ("pred_gauss_inner", "f8"),
        ("gt_dbgauss_inner", "f8"),
        ("pred_dbgauss_inner", "f8"),
        ("gt_laplace_inner", "f8"),
        ("pred_laplace_inner", "f8"),
        ("gt_powlaw_inner", "f8"),
        ("pred_powlaw_inner", "f8"),
    ])

    all_rows = []

    # Create ONE reservoir (fixed reservoir) and reuse it across trials
    print("Creating fixed reservoir model")
    model_rc, reservoir_layer = create_base_model(
        (X_train.shape[1], X_train.shape[2]),
        (y_train.shape[1], y_train.shape[2])
    )
    model_serialized = pickle.dumps(model_rc)

    # If you still want a best_sd, you can compute it once here.
    # For simplicity, we compute best_sd over the sd_list using a few runs
    # on the fixed reservoir.
    gauss_losses_per_sd = {sd: [] for sd in sd_list}
    washout = 200

    for sd in sd_list:
        # You can decide how many runs to use here; n_trials is a reasonable choice.
        for _ in range(args.n_trials):
            model_rc_g = pickle.loads(model_serialized)
            weights_gauss = create_weights((200, 1), "random_normal", dynamic_sd=sd)
            model_rc_g._set_readin_weights(weights_gauss)
            model_rc_g.fit(X_train, y_train)

            y_pred = model_rc_g.predict(X_test)
            y_true_seq = y_test[0, washout:, 0]
            y_pred_seq = y_pred[0, washout:, 0]
            mae = float(np.mean(np.abs(y_true_seq - y_pred_seq)))
            gauss_losses_per_sd[sd].append(mae)

    avg_gauss_losses = {sd: np.mean(gauss_losses_per_sd[sd]) for sd in sd_list}
    best_sd = min(avg_gauss_losses, key=avg_gauss_losses.get)
    best_sd_numeric = float(best_sd)
    print(f"Best Gaussian SD (fixed reservoir): {best_sd_numeric}")

    # Now run the actual trials, looping only over read-ins for the fixed reservoir
    for trial_idx in range(args.n_trials):
        print(f"Trial {trial_idx+1}/{args.n_trials}")

        # ----- Uniform -----
        model_rc_uniform = pickle.loads(model_serialized)
        weights_uniform = create_weights((200, 1), "random_uniform")
        model_rc_uniform._set_readin_weights(weights_uniform)
        model_rc_uniform.fit(X_train, y_train)
        gt_uniform, pred_uniform = scalar_from_model(model_rc_uniform, X_test, y_test, washout)

        # ----- Gaussian (best_sd) -----
        model_rc_gauss = pickle.loads(model_serialized)
        weights_gauss = create_weights((200, 1), "random_normal", dynamic_sd=best_sd_numeric)
        model_rc_gauss._set_readin_weights(weights_gauss)
        model_rc_gauss.fit(X_train, y_train)
        gt_gauss, pred_gauss = scalar_from_model(model_rc_gauss, X_test, y_test, washout)

        # ----- Double-Gaussian (best_sd) -----
        model_rc_dbgauss = pickle.loads(model_serialized)
        weights_dbgauss = create_weights((200, 1), "double_gaussian", dynamic_sd=best_sd_numeric)
        model_rc_dbgauss._set_readin_weights(weights_dbgauss)
        model_rc_dbgauss.fit(X_train, y_train)
        gt_dbgauss, pred_dbgauss = scalar_from_model(model_rc_dbgauss, X_test, y_test, washout)

        # ----- Laplace -----
        model_rc_laplace = pickle.loads(model_serialized)
        weights_laplace = create_weights((200, 1), "laplace")
        model_rc_laplace._set_readin_weights(weights_laplace)
        model_rc_laplace.fit(X_train, y_train)
        gt_laplace, pred_laplace = scalar_from_model(model_rc_laplace, X_test, y_test, washout)

        # ----- Power-law -----
        model_rc_powlaw = pickle.loads(model_serialized)
        weights_powlaw = create_weights((200, 1), "power_law")
        model_rc_powlaw._set_readin_weights(weights_powlaw)
        model_rc_powlaw.fit(X_train, y_train)
        gt_powlaw, pred_powlaw = scalar_from_model(model_rc_powlaw, X_test, y_test, washout)

        row = (
            trial_idx + 1,
            gt_uniform,    pred_uniform,
            gt_gauss,      pred_gauss,
            gt_dbgauss,    pred_dbgauss,
            gt_laplace,    pred_laplace,
            gt_powlaw,     pred_powlaw,
        )
        all_rows.append(row)

    all_rows = np.array(all_rows, dtype=dtype)

    out_path = os.path.join(os.getcwd(), "sin2cos2_conditional_variance_fixed-R.npy")
    np.save(out_path, all_rows)
    print(f"Saved results to {out_path}")
    print(f"Total time: {time.time() - start_time:.2f} sec")

if __name__ == "__main__":
    main()
