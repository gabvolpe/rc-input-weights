'''
Read .npy files
'''

import numpy as np
import os

OUTPUT_DIR = "lorenz/outputs/fixed-reservoir"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(os.path.join(OUTPUT_DIR, "sc1_timeseries_gt.npy"), allow_pickle=True)

print(data)
