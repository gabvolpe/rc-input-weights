"""
Evaluation: SC1 Fixed-Reservoir experiment — sin-to-cos2.

Loads results from sin-to-cos2/outputs/, computes R² scores per
(reservoir, read-in sample, distribution), and produces a grouped violin
plot showing per-reservoir score distributions coloured by distribution.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import r2_score

# ------------------------------------------------------------------ #
# Settings
# ------------------------------------------------------------------ #
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
WASHOUT     = 200  # transient timesteps excluded from scoring

DISTRIBUTIONS = ["uniform", "gaussian", "double_gaussian", "laplace", "power_law"]

DIST_LABELS = {
    "uniform":         "Uniform",
    "gaussian":        "Gaussian",
    "double_gaussian": "Double Gaussian",
    "laplace":         "Laplace",
    "power_law":       "Power Law",
}
DIST_COLORS = {
    "uniform":         "steelblue",
    "gaussian":        "seagreen",
    "double_gaussian": "darkorange",
    "laplace":         "mediumpurple",
    "power_law":       "crimson",
}

# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #
gt_path = os.path.join(RESULTS_DIR, "sc1_ground_truth.npy")
gt = np.load(gt_path)          # shape: (n_time, n_states)
gt_seq = gt[:, 0]              # channel 0, full length — washout applied in r2_score

preds = {}
for dist in DISTRIBUTIONS:
    path = os.path.join(RESULTS_DIR, f"sc1_timeseries_{dist}.npy")
    preds[dist] = np.load(path)  # shape: (n_rows, 2 + n_time); cols: outer, inner, t0...

print(f"Ground truth shape : {gt.shape}")
for dist in DISTRIBUTIONS:
    print(f"  {dist:20s}: {preds[dist].shape[0]} rows")

# ------------------------------------------------------------------ #
# Compute R² per (outer trial, inner trial, distribution)
# ------------------------------------------------------------------ #
outer_ids = sorted(set(preds[DISTRIBUTIONS[0]][:, 0].astype(int)))

# r2_data[dist][outer_id] = 1-D array of R² scores (one per inner trial)
r2_data = {dist: {} for dist in DISTRIBUTIONS}

for dist in DISTRIBUTIONS:
    data = preds[dist]
    for oid in outer_ids:
        rows  = data[data[:, 0] == oid]           # all inner trials for this reservoir
        scores = np.array([
            r2_score(gt_seq, row[2:], washout=WASHOUT)
            for row in rows
        ])
        r2_data[dist][oid] = scores

# ------------------------------------------------------------------ #
# Violin plot: one group per reservoir, one violin per distribution
# ------------------------------------------------------------------ #
n_dists   = len(DISTRIBUTIONS)
n_outer   = len(outer_ids)
group_gap = 1                          # extra space between reservoir groups
group_w   = n_dists + group_gap

fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_outer * n_dists), 5))

for j, dist in enumerate(DISTRIBUTIONS):
    positions   = [i * group_w + j for i in range(n_outer)]
    score_lists = [r2_data[dist][oid] for oid in outer_ids]

    vp = ax.violinplot(score_lists, positions=positions, widths=0.8,
                       showmedians=True, showextrema=True)

    color = DIST_COLORS[dist]
    for pc in vp["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for part in ("cmedians", "cmaxes", "cmins", "cbars"):
        vp[part].set_color(color)
        vp[part].set_linewidth(1.5)

# x-axis: tick at the centre of each reservoir group
group_centers = [i * group_w + (n_dists - 1) / 2 for i in range(n_outer)]
ax.set_xticks(group_centers)
ax.set_xticklabels([f"Reservoir {oid}" for oid in outer_ids])

ax.set_ylabel("R²")
ax.set_title("SC1 Fixed-Reservoir — R² per reservoir and read-in distribution\n"
             f"(washout = {WASHOUT} timesteps, inner trials shown as violins)")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_ylim(bottom=min(-0.1, ax.get_ylim()[0]))

legend_patches = [
    mpatches.Patch(facecolor=DIST_COLORS[d], alpha=0.6, label=DIST_LABELS[d])
    for d in DISTRIBUTIONS
]
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sc1_r2_violin.png"), dpi=150)
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, 'sc1_r2_violin.png')}")
