"""
Sine-to-Cosine^2
Conditional Variability Extraction (Fixed Reservoir)
All .npy files saved to SC1/1_conditional_variance_fixed-R_sine2cos2/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_data import sequence_to_sequence
from pyreco.optimizers import RidgeSK
import time
import argparse
import pickle
import copy

# Define output directory
OUTPUT_DIR = "sin-to-cos2/SC1/OUTPUT_1_conditional_variance_fixed-R"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ----------------------
# --- Main Program------
# ----------------------
def main():
    # fix the seed for reproducibility
    np.random.seed(42)

    # whether to generate data or analyse only
    analyze_only = False  # Set to True to skip generation and run post-processing only

    if not analyze_only:
        if args.task == "sin_to_cos2":
            print("\nSine-to-cosine is ready\n")
            X_train, X_test, y_train, y_test = sequence_to_sequence(
                name="sin_to_cos2", n_states=1, n_batch=200, n_time=1000
            )
        else:
            raise NotImplementedError(f"Task {args.task} not implemented")

        # save data to specific directory
        np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
        
        print(f"Data saved to {OUTPUT_DIR}/")

        start_time = time.time()

        input_shape = X_train.shape[1:]
        output_shape = y_train.shape[1:]

        # Create ONE reservoir (fixed reservoir) and reuse it across trials
        print("Creating fixed reservoir model")
        model_rc, reservoir_layer = create_base_model(input_shape, output_shape)
        model_serialized = pickle.dumps(model_rc)

        # extract reservoir weights and save as npy file
        reservoir_weights = reservoir_layer.weights
        np.save(os.path.join(OUTPUT_DIR, "reservoir_weights.npy"), reservoir_weights)
        print(f"Reservoir weights saved to {OUTPUT_DIR}/")

        # Compute best_sd for Gaussian
        gauss_losses_per_sd = {sd: [] for sd in sd_list}
        washout = 200

        for sd in sd_list:
            for _ in range(args.n_trials):
                model_rc_g = pickle.loads(model_serialized)
                weights_gauss = create_weights((args.reservoir_nodes, 1), "random_normal", dynamic_sd=sd)
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

        # SAVE best_sd to .npy file
        np.save(os.path.join(OUTPUT_DIR, "best_sd.npy"), best_sd_numeric)
        print(f"Best SD saved to {OUTPUT_DIR}/best_sd.npy")

        # Run trials for all distributions
        readin_weights = {
            'uniform': [], 'gaussian': [], 'double_gaussian': [],
            'laplace': [], 'power_law': []
        }
        predictions = {key: [] for key in readin_weights}

        for trial_idx in range(args.n_trials):
            print(f"Trial {trial_idx+1}/{args.n_trials}")

            # Uniform
            model_rc_uniform = pickle.loads(model_serialized)
            weights_uniform = create_weights((args.reservoir_nodes, 1), "random_uniform")
            model_rc_uniform._set_readin_weights(weights_uniform)
            model_rc_uniform.fit(X_train, y_train)
            y_pred_uniform = model_rc_uniform.predict(X_test)
            readin_weights['uniform'].append(weights_uniform)
            predictions['uniform'].append(y_pred_uniform)

            # Gaussian (best_sd)
            model_rc_gauss = pickle.loads(model_serialized)
            weights_gauss = create_weights((args.reservoir_nodes, 1), "random_normal", dynamic_sd=best_sd_numeric)
            model_rc_gauss._set_readin_weights(weights_gauss)
            model_rc_gauss.fit(X_train, y_train)
            y_pred_gauss = model_rc_gauss.predict(X_test)
            readin_weights['gaussian'].append(weights_gauss)
            predictions['gaussian'].append(y_pred_gauss)

            # Double-Gaussian (best_sd)
            model_rc_dbgauss = pickle.loads(model_serialized)
            weights_dbgauss = create_weights((args.reservoir_nodes, 1), "double_gaussian", dynamic_sd=best_sd_numeric)
            model_rc_dbgauss._set_readin_weights(weights_dbgauss)
            model_rc_dbgauss.fit(X_train, y_train)
            y_pred_dbgauss = model_rc_dbgauss.predict(X_test)
            readin_weights['double_gaussian'].append(weights_dbgauss)
            predictions['double_gaussian'].append(y_pred_dbgauss)

            # Laplace
            model_rc_laplace = pickle.loads(model_serialized)
            weights_laplace = create_weights((args.reservoir_nodes, 1), "laplace")
            model_rc_laplace._set_readin_weights(weights_laplace)
            model_rc_laplace.fit(X_train, y_train)
            y_pred_laplace = model_rc_laplace.predict(X_test)
            readin_weights['laplace'].append(weights_laplace)
            predictions['laplace'].append(y_pred_laplace)

            # Power-law
            model_rc_powlaw = pickle.loads(model_serialized)
            weights_powlaw = create_weights((args.reservoir_nodes, 1), "power_law")
            model_rc_powlaw._set_readin_weights(weights_powlaw)
            model_rc_powlaw.fit(X_train, y_train)
            y_pred_powlaw = model_rc_powlaw.predict(X_test)
            readin_weights['power_law'].append(weights_powlaw)
            predictions['power_law'].append(y_pred_powlaw)

        # Save all results to specific directory
        for dist in readin_weights:
            readin_weights[dist] = np.array(readin_weights[dist])
            predictions[dist] = np.array(predictions[dist])
            np.save(os.path.join(OUTPUT_DIR, f"readin_weights_{dist}.npy"), readin_weights[dist])
            np.save(os.path.join(OUTPUT_DIR, f"predictions_{dist}.npy"), predictions[dist])
        
        print(f"All readin weights and predictions saved to {OUTPUT_DIR}/")
        print(f"Total time: {time.time() - start_time:.2f} sec")

    # ----------------------
    # Post-Processing
    # ----------------------
    
    print("\n" + "="*60)
    print("POST-PROCESSING ANALYSIS")
    print("="*60)

    # Load data from specific directory
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    
    distributions = ['uniform', 'gaussian', 'double_gaussian', 'laplace', 'power_law']
    all_r2_scores = {}
    
    
    for dist in distributions:
        print(f"\nAnalyzing {dist} distribution...")
        
        # Load predictions for this distribution
        predictions_dist = np.load(os.path.join(OUTPUT_DIR, f"predictions_{dist}.npy"))
        readin_weights_dist = np.load(os.path.join(OUTPUT_DIR, f"readin_weights_{dist}.npy"))
        n_trials = readin_weights_dist.shape[0]
        '''
        # Example: plot prediction of first trial
        plt.figure(figsize=(10,4))
        plt.plot(y_test[0,:,0], label=r'True $\cos^2(t)$', linestyle='--')
        plt.plot(predictions_dist[0,0,:,0], label=r'Predicted $\cos^2(t)$', alpha=0.7)
        plt.legend()
        plt.xlabel(r'time steps')
        plt.ylabel(r'amplitude $x$')
        plt.title(f'Prediction of first trial - {dist} readin weights')
        plt.show()'''

        # Compute R^2 across all trials
        r2_scores = []
        for trial in range(n_trials):
            r2 = r2_score(y_test.reshape(-1), predictions_dist[trial].reshape(-1))
            r2_scores.append(r2)
        
        r2_scores = np.array(r2_scores)
        all_r2_scores[dist] = r2_scores
        '''
        # Plot histogram of R^2 scores
        plt.figure(figsize=(8,4))
        plt.hist(r2_scores, bins=10, alpha=0.7, density=True)
        plt.xlabel(r'$R^2$ Score')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of $R^2$ Scores - {dist} Readin Weights')
        plt.show()'''

        r2_stats = {
            'min': float(np.min(r2_scores)),
            'max': float(np.max(r2_scores)),
            'mean': float(np.mean(r2_scores)),
            'median': float(np.median(r2_scores)),
            'std': float(np.std(r2_scores)),
            'r2_scores': r2_scores  # Save full array too
        }
         # Save R2 stats to npy file
        np.save(os.path.join(OUTPUT_DIR, f"r2_stats_{dist}.npy"), r2_stats)   

        # Print statistics
        print(f"{dist.upper()} R² Stats:")
        print(f"  Range: {min(r2_scores):.4f} to {max(r2_scores):.4f}")
        print(f"  Mean:  {np.mean(r2_scores):.4f}")
        print(f"  Median:{np.median(r2_scores):.4f}")
        print(f"  Std:   {np.std(r2_scores):.4f}")



    # Summary comparison across all distributions
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    summary_stats = {}
    for dist, r2_scores in all_r2_scores.items():
        summary_stats[dist] = {
            'mean': np.mean(r2_scores),
            'std': np.std(r2_scores),
            'min': np.min(r2_scores),
            'max': np.max(r2_scores)
        }
    '''
    # Create comparison table
    dist_names = list(summary_stats.keys())
    means = [summary_stats[d]['mean'] for d in dist_names]
    stds = [summary_stats[d]['std'] for d in dist_names]
    
    plt.figure(figsize=(10,6))
    x = np.arange(len(dist_names))
    width = 0.35
    plt.bar(x, means, width, yerr=stds, capsize=5, label='Mean R² ± Std')
    plt.xlabel('Weight Distribution')
    plt.ylabel(r'$R^2$ Score')
    plt.title('Comparison of R² Scores Across Weight Distributions')
    plt.xticks(x, dist_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()'''
    
    # Print ranked results
    ranked = sorted(summary_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    print("\nRANKING (by mean R²):")
    for i, (dist, stats) in enumerate(ranked, 1):
        print(f"{i}. {dist:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

if __name__ == "__main__":
    main()
