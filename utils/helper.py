import numpy as np
import os

OUTPUT_DIR = "lorenz/outputs/fixed-readin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(os.path.join(OUTPUT_DIR, "sc2_results_fixed-readin.npy"))
#data = np.load("sin2cos2_unconditional_variance.npy", allow_pickle=True)
print(data)
