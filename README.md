# Read-in Weight Matrix Sensitivity Experiments

This repository contains code for a set of experiments evaluating how changes to the read-in weight matrix of a Reservoir Computing (RC) affect model performance across four distinct tasks: **Sine-to-Cosine<sup>2</sup>, Mackey-Glass, Lorenz, and NARMA-10**.

## Overview

For each task, we compare the performance across modified versions of the model that differ only in the design of the read-in matrix. Considering the results for the four different tasks help analyze how architectural changes in the read-in weight matrix influence overall performance and robustness. All experiments are run by keeping the reservoir structure and all hyperparameters than the read-in matrix fixed. Inside the `evaluation` folder there are the python files to evaluate and plot the experiments.

### Goals

- Assess the impact of read-in weight matrix modifications across different systems.
- Quantify changes in predictive performance using metrics such as MSE (Mean Squared Error).
- Measure stability across runs using the IQR (Interquartile Range) of MSEs.

### Evaluation

Each model is evaluated over multiple runs (e.g., 1000 trials). For each run, the Mean Squared Error (MSE) is recorded. Final evaluation is based on:  
`•	Median MSE: performance measure`  
`•	IQR of MSE: robustness and consistency indicator`

## Requirements

Install the required dependencies using:  
`pip install -r requirements.txt`
