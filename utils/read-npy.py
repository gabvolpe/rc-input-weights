'''
Read .npy files
'''

import numpy as np
import os

OUTPUT_DIR = "sin-to-cos2/outputs/fixed-readin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(os.path.join(OUTPUT_DIR, "sc1_ground_truth.npy"))

print(data)
