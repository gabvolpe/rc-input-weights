'''
Read .npy files
'''

import numpy as np
import os

OUTPUT_DIR = "narma10/outputs/fixed-reservoir"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(os.path.join(OUTPUT_DIR, "sc2_results_fixed-reservoir.npy"))

print(data)
