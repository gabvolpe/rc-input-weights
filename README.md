# Read-in Weight Matrix Sensitivity Experiments

This repository contains code for a set of three experiments evaluating how changes to the read-in weight matrix of a Reservoir Computin (RC) affect model performance across three distinct systems: Sine Wave, Mackey-Glass, Lorenz.

## Overview

We use a baseline model (**Model A**) and compare it against three modified versions (**Models B, C, and D**) that differ only in the design of the read-in matrix. Each of the three experiments is conducted on a different system, with the goal of analyzing how architectural changes in the read-in weight matrix influence overall performance and robustness. All experiments are run by keeping the reservoir structure and all hyperparameters than the read-in matrix fixed. Inside the helper folder there are the python files to evaluate and plot the experiments.

### Goals

- Assess the impact of read-in weight matrix modifications across different systems.
- Quantify changes in predictive performance using metrics such as MSE (Mean Squared Error).
- Measure stability across runs using the IQR (Interquartile Range) of MSEs.

### Evaluation

Each model is evaluated over multiple runs (e.g., 1000 trials). For each run, the Mean Squared Error (MSE) is recorded. Final evaluation is based on:
•	Median MSE: performance measure
•	IQR of MSE: robustness and consistency indicator

## Requirements

Install the required dependencies using:
`pip install -r requirements.txt`
