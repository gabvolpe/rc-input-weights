import os
import numpy as np
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_data import sequence_to_sequence
from pyreco.optimizers import RidgeSK
import time
import matplotlib.pyplot as plt
import copy


def create_base_model(input_shape: tuple, output_shape: tuple, args: dict) -> RC:
    """
    Create a RC model with specified input and output shapes.
    
    input_shape: Shape of the input data
    output_shape: Shape of the output data
    """

    # Define optimizer
    optim = RidgeSK(alpha=args["ridge_alpha"])

    # Build model
    model_rc = RC()
    model_rc.add(InputLayer(input_shape=input_shape))
    reservoir_layer = RandomReservoirLayer(
        nodes=args["reservoir_nodes"],
        density=args["density"],
        activation=args["activation"],
        spec_rad=args["spectral_radius"],
        leakage_rate=args["leakage_rate"],
        fraction_input=args["fraction_input"]
    )
    model_rc.add(reservoir_layer)
    model_rc.add(ReadoutLayer(output_shape, fraction_out=1.0))
    
    model_rc.compile(optimizer=optim, 
                     metrics=["mean_squared_error"])
    return model_rc



if __name__ == "__main__":

    """
    We would need to run this script now many times 
    to vary the reservoir graph as well, but for now we keep it fixed.

    We will then look at the variation of the variation per reservoir graph.
    """


    # fix the seed for reproducibility
    np.random.seed(42)

    # whether to analyse only (should be part of a different script)
    analyze_only = True

    if not analyze_only:

        #     # # Generate data (ONLY ONCE)
        # X_train, X_test, y_train, y_test = sequence_to_sequence(
        #             name="sin_to_cos2", n_states=1, n_batch=200, n_time=1000
        #         )
        
        # # save data to local files
        # np.save("X_train.npy", X_train)
        # np.save("X_test.npy", X_test)
        # np.save("y_train.npy", y_train)
        # np.save("y_test.npy", y_test)

        # load data from local files
        X_train, X_test = np.load("X_train.npy"), np.load("X_test.npy")
        y_train, y_test = np.load("y_train.npy"), np.load("y_test.npy")
        
        # plot some examples
        plt.figure(figsize=(10,4))
        plt.plot(X_train[0,:,0], label=r'Input $\sin(t)$')
        plt.plot(y_train[0,:,0], label=r'Target $\cos^2(t)$')
        plt.legend()
        plt.xlabel(r'time steps')
        plt.ylabel(r'amplitude $x$')
        plt.title('Example of input and target sequences')
        plt.show()

        # create RC model for specified hyperparameters
        input_shape = X_train.shape[1:]  # (n_time, n_states)
        output_shape = y_train.shape[1:]  # (n_time, n_states)

        # hyperparameters
        args = {
            "reservoir_nodes": 200,
            "density": 0.15,
            "activation": "tanh",
            "spectral_radius": 0.9,
            "leakage_rate": 0.1,
            "fraction_input": 1.0,
            "ridge_alpha": 0.1,
        }

        # build baseline model 
        model = create_base_model(input_shape, output_shape, args)

        # extract reservoir weights and save as npy file
        reservoir_weights = model.reservoir_layer.weights
        np.save("reservoir_weights.npy", reservoir_weights)

        # central loop: every run, we will sample a new readin weight matrix, 
        # train the model, run predictions and store the results
        n_trials = 100
        readin_weights = []
        predictions = []
        readin_shape = (model.reservoir_layer.nodes, input_shape[1])    
        start_time = time.time()
        for trial in range(n_trials):
            print(f"Trial {trial+1}/{n_trials}")

            _model = copy.deepcopy(model)  # avoid any contamination between trials

            # sample new readin weights of readin_shape
            w_in = np.random.rand(*readin_shape)

            # set new readin weights in the model
            _model.input_layer.weights = w_in
            # train and predict
            _model.fit(X_train, y_train)
            y_pred = _model.predict(X_test)

            # store readin weights and predictions
            readin_weights.append(w_in)
            predictions.append(y_pred)

        end_time = time.time()
        print(f"Total time for {n_trials} trials: {end_time - start_time:.2f} seconds")

        # convert lists to arrays and save as np arrays
        readin_weights = np.array(readin_weights)
        predictions = np.array(predictions)
        np.save("readin_weights.npy", readin_weights)
        np.save("predictions.npy", predictions)


    # ----------------------
    # Post-Processing
    # ----------------------

    # read data from npy files
    readin_weights = np.load("readin_weights.npy")
    predictions = np.load("predictions.npy")
    y_test = np.load("y_test.npy")
    n_trials = readin_weights.shape[0]

    # Example: plot prediction of first trial
    plt.figure(figsize=(10,4))
    plt.plot(y_test[0,:,0], label=r'True $\cos^2(t)$', linestyle='--')
    plt.plot(predictions[0,0,:,0], label=r'Predicted $\cos^2(t)$', alpha=0.7)
    plt.legend()
    plt.xlabel(r'time steps')
    plt.ylabel(r'amplitude $x$')
    plt.title('Prediction of first trial with random readin weights')
    plt.show()

    # global analysis: compute R^2 across all trials
    from sklearn.metrics import r2_score
    r2_scores = []
    for trial in range(n_trials):
        r2 = r2_score(y_test.reshape(-1), predictions[trial].reshape(-1))
        r2_scores.append(r2)

    # concatenate into one array
    r2_scores = np.array(r2_scores)

    # plot histogram of R^2 scores: this is the impact of the readin weights
    plt.figure(figsize=(8,4))
    plt.hist(r2_scores, bins=10, alpha=0.7, density=True)
    plt.xlabel(r'$R^2$ Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of $R^2$ Scores across Trials with Random Readin Weights')
    plt.show()

    # print to console: range, mean, median and std of R^2 scores
    print(f"\n\nR^2 Score Range: {min(r2_scores):.4f} to {max(r2_scores):.4f}")
    print(f"Mean R^2 Score: {np.mean(r2_scores):.4f}")
    print(f"Median R^2 Score: {np.median(r2_scores):.4f}")
    print(f"Standard Deviation of R^2 Scores: {np.std(r2_scores):.4f}")



