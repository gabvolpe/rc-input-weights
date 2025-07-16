'''
Investigating different clipping values for the read-in weights
'''

import os
import sys
import copy
import pickle
from matplotlib import pyplot as plt
import numpy as np
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_data import sequence_to_scalar
from pyreco.optimizers import RidgeSK
from pyreco.metrics import r2, mae, mse
import time
import argparse
from openpyxl import load_workbook, Workbook


# as its not a library, we need to add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from helper_files.utils import create_weights
from helper_files.utils import compute_median_and_iqr_loss


def create_base_model(input_shape, output_shape):
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(nodes=200, 
                                   density=0.1, 
                                   activation="tanh", 
                                   leakage_rate=0.1, 
                                   fraction_input=1.0)) # no fraction_input
    model.add(ReadoutLayer(output_shape, fraction_out=1.0)) #fraction_out either 1.0 or 0.5
    optim = RidgeSK(alpha=0.5)
    model.compile(optimizer=optim, metrics=["mean_squared_error"])
    return model




def main():
    parser = argparse.ArgumentParser(description="Run RC model with read-in weights variations.")
    parser.add_argument('--n_trials', type=int, default=200, help='Number of trials')
    args = parser.parse_args()


    X_train, X_test, y_train, y_test = sequence_to_scalar(
        name="sine_prediction",
        n_states=1,
        n_batch=200,
        n_time_in=20,
    )
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = (y_train.shape[1], y_train.shape[2])

    # grid of clipping values to test
    thresholds = np.linspace(0.000001, 0.99, 20)  # from 0.0001 to 0.9 in 20 steps

    # dict of ways to create the read-in weights (random normal, laplace, fourier, ...)
    weight_methods = {
        "RandomUniform": "random_uniform",
        "RandomNormal": "random_normal",
        "Laplace": "laplace",
    }

    # error metric to use
    metric = "r2"

    # create a based RC model using a random reservoir layer
    model_rc = create_base_model(input_shape, output_shape)

    losses = {}

    # we will use only a single loop in this case here (i.e. keep the reservoir fixed)
    for method_name, method in weight_methods.items():
        print(f"Using {method_name} weights")

        losses[method_name] = np.zeros((len(thresholds), args.n_trials))

        for i, _thresh in enumerate(thresholds):
            print(f"Testing threshold: {_thresh}")

            for j in range(args.n_trials):
                print(f"trial {j+1}/{args.n_trials}, threshold {i+1}/{len(thresholds)}")

                # create a clean copy for every inner trial
                _model = copy.deepcopy(model_rc)

                # create read-in weights
                readin_weights = create_weights((200, 1), method, threshold=_thresh)
                _model._set_readin_weights(readin_weights)

                # fit the model
                _model.fit(X_train, y_train)

                # evaluate the model
                loss = _model.evaluate(X_test, y_test, metrics=[metric])[0]

                losses[method_name][i, j] = loss

    # evaluate the losses as a function of the clipping threshold

    # plot of the box plots over the clipping thresholds
    for method_name in list(weight_methods.keys()):
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [losses[method_name][i, :] for i in range(len(thresholds))]
        ax.boxplot(box_data, positions=thresholds, widths=0.05,) # tick_labels=[method_name])
        ax.set_xlabel("Clipping Threshold")
        ax.set_ylabel(metric)
        ax.set_title("Effect of Clipping Threshold on Model Performance, Method: " + method_name)
        ax.legend()
        plt.savefig(f"clipping_threshold_effect_{method_name}.png")
        plt.show()

if __name__ == "__main__":
    main()
