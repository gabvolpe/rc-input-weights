"""
NARMA-10 — Unconditional Variability Extraction, fixed Reservoir | Variable Read-in.
Constraint Set 1: full input (no masking), no near-zero read-in weights.
Gaussian SD is fixed at 1.0.

Memory-safe streaming implementation:
Intermediate results are streamed to temporary files during execution
to avoid RAM accumulation. After completion, temporary files are
merged back into the original output structure:

    sc1_ground_truth.npy
    sc1_timeseries_<dist>.npy
    sc1_timeseries_gt.npy
    sc1_readin_weights_<dist>.npy
    sc1_reservoir_weights.npy
"""

import os
import sys
import gc
import glob
import shutil
import pickle
import argparse
import concurrent.futures

import numpy as np

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
# TEMP STREAMING DIRECTORIES
# ------------------------------------------------------------
TEMP_DIR = os.path.join(OUTPUT_DIR, "_temp_stream")

TEMP_TIMESERIES_DIR = os.path.join(TEMP_DIR, "timeseries")
TEMP_READIN_DIR = os.path.join(TEMP_DIR, "readins")
TEMP_GT_DIR = os.path.join(TEMP_DIR, "gt")
TEMP_RESERVOIR_DIR = os.path.join(TEMP_DIR, "reservoirs")

os.makedirs(TEMP_TIMESERIES_DIR, exist_ok=True)
os.makedirs(TEMP_READIN_DIR, exist_ok=True)
os.makedirs(TEMP_GT_DIR, exist_ok=True)
os.makedirs(TEMP_RESERVOIR_DIR, exist_ok=True)

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

parser.add_argument(
    "--n_trials",
    type=int,
    default=50, 
    help="Number of outer trials (reservoirs)"
)

parser.add_argument(
    "--n_inner",
    type=int,
    default=100,
    help="Number of inner trials per reservoir"
)

parser.add_argument("--nodes", type=int, default=200)
parser.add_argument("--density", type=float, default=0.4)
parser.add_argument("--spectral_radius", type=float, default=0.9)
parser.add_argument("--leakage_rate", type=float, default=0.2)
parser.add_argument("--fraction_input", type=float, default=1.0)
parser.add_argument("--ridge_alpha", type=float, default=1e-6)

parser.add_argument("--readin_threshold", type=float, default=1e-3)
parser.add_argument("--set_threshold", type=bool, default=True)

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

# select only firt 50 samples for speed
X_train = X_train[:50, ...]
y_train = y_train[:50, ...]
X_test = X_test[:50, ...]
y_test = y_test[:50, ...]

print(f"shape of X-train: {X_train.shape}")
print(f"shape of X-test: {X_test.shape}")

# ------------------------------------------------------------
# SAVE GLOBAL GROUND TRUTH
# ------------------------------------------------------------
np.save(
    os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"),
    y_test
)

# ------------------------------------------------------------
# TEMP SAVE HELPERS
# ------------------------------------------------------------
def save_temp_prediction(pred, eval_dist, outer_id, inner_id):

    path = os.path.join(
        TEMP_TIMESERIES_DIR,
        f"{eval_dist}_outer{outer_id}_inner{inner_id}.npy"
    )

    np.save(path, pred)


def save_temp_readin(W, eval_dist, outer_id, inner_id):

    path = os.path.join(
        TEMP_READIN_DIR,
        f"{eval_dist}_outer{outer_id}_inner{inner_id}.npy"
    )

    np.save(path, W)


def save_temp_gt(gt, outer_id, inner_id):

    path = os.path.join(
        TEMP_GT_DIR,
        f"gt_outer{outer_id}_inner{inner_id}.npy"
    )

    np.save(path, gt)


def save_temp_reservoir(weights, outer_id):

    path = os.path.join(
        TEMP_RESERVOIR_DIR,
        f"reservoir_outer{outer_id}.npy"
    )

    np.save(path, weights)


# ------------------------------------------------------------
# INNER RUN
# ------------------------------------------------------------
def run_inner(model_bytes, outer_id, inner_id):

    for eval_dist, sampler_dist in DIST_MAP.items():

        # ----------------------------------------------------
        # Restore model
        # ----------------------------------------------------
        model = pickle.loads(model_bytes)

        # ----------------------------------------------------
        # Sample read-in weights
        # ----------------------------------------------------
        W = sample_readin_weights(
            shape=(args.nodes, X_train.shape[2]),
            method=sampler_dist,
            sd=GAUSS_SD if sampler_dist in ["random_normal", "double_gaussian"] else None,
            threshold=THRESHOLD
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------
        if THRESHOLD is not None and THRESHOLD is not False:
            assert_weights_above_threshold(W, THRESHOLD, sampler_dist)

        # ----------------------------------------------------
        # Fit model
        # ----------------------------------------------------
        model._set_readin_weights(W)
        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------
        gt, pred = predict_sequences(model, X_test, y_test)

        # ----------------------------------------------------
        # STREAM RESULTS DIRECTLY TO DISK
        # ----------------------------------------------------
        save_temp_prediction(
            pred,
            eval_dist,
            outer_id,
            inner_id
        )

        save_temp_readin(
            W,
            eval_dist,
            outer_id,
            inner_id
        )

        # Save GT once per inner run
        if eval_dist == EVAL_DISTS[0]:
            save_temp_gt(
                gt,
                outer_id,
                inner_id
            )

        # ----------------------------------------------------
        # Explicit cleanup
        # ----------------------------------------------------
        del model
        del W
        del pred
        del gt

        gc.collect()

    return outer_id, inner_id


# ------------------------------------------------------------
# FINAL MERGE HELPERS
# ------------------------------------------------------------
def merge_timeseries(eval_dist):

    files = sorted(
        glob.glob(
            os.path.join(
                TEMP_TIMESERIES_DIR,
                f"{eval_dist}_outer*_inner*.npy"
            )
        )
    )

    merged = []

    for f in files:

        basename = os.path.basename(f)

        outer_id = int(
            basename.split("_outer")[1].split("_")[0]
        )

        inner_id = int(
            basename.split("_inner")[1].split(".npy")[0]
        )

        arr = np.load(f)

        merged.append(
            (outer_id, inner_id, arr)
        )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"sc1_timeseries_{eval_dist}.npy"
        ),
        np.array(merged, dtype=object)
    )


def merge_readins(eval_dist):

    files = sorted(
        glob.glob(
            os.path.join(
                TEMP_READIN_DIR,
                f"{eval_dist}_outer*_inner*.npy"
            )
        )
    )

    merged = []

    for f in files:

        basename = os.path.basename(f)

        outer_id = int(
            basename.split("_outer")[1].split("_")[0]
        )

        inner_id = int(
            basename.split("_inner")[1].split(".npy")[0]
        )

        arr = np.load(f)

        merged.append(
            (outer_id, inner_id, arr)
        )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"sc1_readin_weights_{eval_dist}.npy"
        ),
        np.array(merged, dtype=object)
    )


def merge_gt():

    files = sorted(
        glob.glob(
            os.path.join(
                TEMP_GT_DIR,
                "gt_outer*_inner*.npy"
            )
        )
    )

    merged = []

    for f in files:

        basename = os.path.basename(f)

        outer_id = int(
            basename.split("_outer")[1].split("_")[0]
        )

        inner_id = int(
            basename.split("_inner")[1].split(".npy")[0]
        )

        arr = np.load(f)

        merged.append(
            (outer_id, inner_id, arr)
        )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "sc1_timeseries_gt.npy"
        ),
        np.array(merged, dtype=object)
    )


def merge_reservoirs():

    files = sorted(
        glob.glob(
            os.path.join(
                TEMP_RESERVOIR_DIR,
                "reservoir_outer*.npy"
            )
        )
    )

    merged = []

    for f in files:

        basename = os.path.basename(f)

        outer_id = int(
            basename.split("_outer")[1].split(".npy")[0]
        )

        arr = np.load(f)

        merged.append(
            (outer_id, arr)
        )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "sc1_reservoir_weights.npy"
        ),
        np.array(merged, dtype=object)
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    np.random.seed(42)

    for outer in range(args.n_trials):

        print(f"\nOuter {outer+1}/{args.n_trials}")

        # ----------------------------------------------------
        # Create fixed reservoir
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Save reservoir immediately
        # ----------------------------------------------------
        save_temp_reservoir(
            reservoir.weights.copy(),
            outer
        )

        # ----------------------------------------------------
        # Serialize model
        # ----------------------------------------------------
        model_bytes = pickle.dumps(model)

        # ----------------------------------------------------
        # Cleanup originals
        # ----------------------------------------------------
        del model
        del reservoir

        gc.collect()

        # ----------------------------------------------------
        # INNER LOOP
        # ----------------------------------------------------
        if args.parallel:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as ex:

                futures = [
                    ex.submit(
                        run_inner,
                        model_bytes,
                        outer,
                        inner
                    )
                    for inner in range(args.n_inner)
                ]

                for f in concurrent.futures.as_completed(futures):

                    outer_id, inner_id = f.result()

                    print(
                        f"Completed outer={outer_id} inner={inner_id}"
                    )

        else:

            for inner in range(args.n_inner):

                outer_id, inner_id = run_inner(
                    model_bytes,
                    outer,
                    inner
                )

                print(
                    f"Completed outer={outer_id} inner={inner_id}"
                )

        # ----------------------------------------------------
        # Cleanup serialized model
        # ----------------------------------------------------
        del model_bytes

        gc.collect()

        print(f"Outer {outer+1} done")

    # ------------------------------------------------------------
    # FINAL MERGE
    # ------------------------------------------------------------
    print("\nMerging temporary files...")

    for eval_dist in EVAL_DISTS:

        print(f"Merging timeseries: {eval_dist}")
        merge_timeseries(eval_dist)

        print(f"Merging readins: {eval_dist}")
        merge_readins(eval_dist)

    print("Merging GT")
    merge_gt()

    print("Merging reservoirs")
    merge_reservoirs()

    # ------------------------------------------------------------
    # CLEAN TEMP FILES
    # ------------------------------------------------------------
    print("\nCleaning temporary files...")

    shutil.rmtree(TEMP_DIR)

    print("\nDONE")


if __name__ == "__main__":
    main()
