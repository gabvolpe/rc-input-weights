"""
Evaluation: Fixed-Reservoir experiment for specified SC — NARMA10.

Loads results from sin-to-cos2/outputs/, computes R² scores per
(reservoir, read-in sample, distribution), and produces a grouped violin
plot showing per-reservoir score distributions coloured by distribution.

Please specify which set constraint (SC1, SC2, or SC3) you want to evaluate by inizializing the following variable SET_CONSTRAINT (e.g. "1" for SC1).
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
SET_CONSTRAINT = "1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "fixed-reservoir")
WASHOUT     = 0  # transient timesteps excluded from scoring

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
gt_path = os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_ground_truth.npy")
gt = np.load(gt_path)          # shape: (n_time, n_states)
gt_seq = gt[:, 0]              # channel 0, full length — washout applied in r2_score

preds = {}
for dist in DISTRIBUTIONS:
    path = os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+f"_timeseries_{dist}.npy")
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
group_gap = 3                          # extra space between reservoir groups
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
ax.set_title(f"SC{SET_CONSTRAINT} Fixed-Reservoir — R² per reservoir and read-in distribution\n"
             f"(washout = {WASHOUT} timesteps, inner trials shown as violins)")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
all_r2 = np.concatenate([
    scores for dist in DISTRIBUTIONS for scores in r2_data[dist].values()
])
r2_min = all_r2.min()
ax.set_ylim(bottom=r2_min - 0.01 * abs(r2_min), top=1.0)

legend_patches = [
    mpatches.Patch(facecolor=DIST_COLORS[d], alpha=0.6, label=DIST_LABELS[d])
    for d in DISTRIBUTIONS
]
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_violin.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_violin.pdf"))
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_violin.png")}")

# ------------------------------------------------------------------ #
# Best-prediction plot: ground truth + best prediction per distribution
# with shaded range and median across all trials
# ------------------------------------------------------------------ #
# For each distribution:
#   - best prediction: row with highest R² across all outer/inner trials
#   - shaded band: min–max range across all rows per timestep (post-washout)
#   - dashed line: median across all rows per timestep (post-washout)
best_preds = {}
for dist in DISTRIBUTIONS:
    data   = preds[dist]                    # (n_rows, 2 + n_time)
    ts_all = data[:, 2:]                    # (n_rows, n_time)
    scores = np.array([
        r2_score(gt_seq, row, washout=WASHOUT)
        for row in ts_all
    ])
    best_preds[dist] = {
        "best":   (ts_all[np.argmax(scores)], scores.max()),
        "median": np.median(ts_all, axis=0),
        "lo":     ts_all.min(axis=0),
        "hi":     ts_all.max(axis=0),
    }

timesteps = np.arange(len(gt_seq))
t = timesteps[WASHOUT:]

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(t, gt_seq[WASHOUT:],
        color="black", linewidth=1.5, label="Ground truth", zorder=10)

for dist in DISTRIBUTIONS:
    d     = best_preds[dist]
    color = DIST_COLORS[dist]
    ts, r2 = d["best"]

    # shaded range: full spread of all predictions
    ax.fill_between(t, d["lo"][WASHOUT:], d["hi"][WASHOUT:],
                    color=color, alpha=0.15)

    # median across all trials
    ax.plot(t, d["median"][WASHOUT:],
            color=color, linewidth=1.0, linestyle="--", alpha=0.7)

    # best single prediction
    ax.plot(t, ts[WASHOUT:],
            color=color, linewidth=1.0, alpha=0.9,
            label=f"{DIST_LABELS[dist]} (best R²={r2:.3f})")

ax.set_xlabel("Timestep")
ax.set_ylabel("Output")
ax.set_title(f"SC{SET_CONSTRAINT} Fixed-Reservoir — Ground truth vs. predictions per distribution\n"
             "solid: best | dashed: median | shaded: full range")
ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_best_predictions.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_best_predictions.pdf"))
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_best_predictions.png")}")

# ------------------------------------------------------------------ #
# Variance decomposition: reservoir vs. read-in contribution
#
# One-way ANOVA decomposition per distribution:
#   var_between — variance of per-reservoir mean R² across reservoirs
#                 (how much the reservoir choice shifts average performance)
#   var_within  — mean of per-reservoir R² variances across inner trials
#                 (how much the read-in weights shift performance for a
#                  fixed reservoir)
# Both are normalised to their sum so the bars reach 100%.
# ------------------------------------------------------------------ #
var_between = {}  # reservoir effect
var_within  = {}  # read-in effect

for dist in DISTRIBUTIONS:
    outer_means = np.array([r2_data[dist][oid].mean() for oid in outer_ids])
    outer_vars  = np.array([r2_data[dist][oid].var()  for oid in outer_ids])
    var_between[dist] = outer_means.var()
    var_within[dist]  = outer_vars.mean()

# --- Plot 1: grouped bar chart of variance fractions ---
fig, ax = plt.subplots(figsize=(9, 4))

x      = np.arange(len(DISTRIBUTIONS))
width  = 0.35

for i, (label, key, hatch) in enumerate([
    ("Reservoir (between)", var_between, ""),
    ("Read-in (within)",    var_within,  "///"),
]):
    totals    = np.array([var_between[d] + var_within[d] for d in DISTRIBUTIONS])
    values    = np.array([key[d] for d in DISTRIBUTIONS])
    fractions = np.where(totals > 0, values / totals * 100, 0)
    bars = ax.bar(x + (i - 0.5) * width, fractions,
                  width=width,
                  color=[DIST_COLORS[d] for d in DISTRIBUTIONS],
                  alpha=0.85 if i == 0 else 0.45,
                  hatch=hatch,
                  label=label,
                  edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([DIST_LABELS[d] for d in DISTRIBUTIONS])
ax.set_ylabel("Variance explained (%)")
ax.set_ylim(0, 100)
ax.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_title(f"SC{SET_CONSTRAINT} Fixed-Reservoir — Variance decomposition: reservoir vs. read-in weights\n"
             "solid fill = reservoir effect | hatched = read-in effect")

legend_handles = [
    mpatches.Patch(
        facecolor="dimgray",   
        alpha=0.9,
        label= "Reservoir effect (between)" # instead of "Read-in effect (between)"
    ),
    mpatches.Patch(
        facecolor="0.6",   # lighter for contrast
        hatch="///",
        edgecolor="white",
        alpha=0.7,
        label= "Read-in effect (within)" # instead of "Reservoir effect (within)"
    ),
]
#ax.legend(framealpha=0.9)
ax.legend(handles=legend_handles, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_variance_decomposition.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_variance_decomposition.pdf"))
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_variance_decomposition.png")}")

# --- Plot 2: R² heatmap (outer × inner) per distribution ---
n_inner_max = max(len(r2_data[DISTRIBUTIONS[0]][oid]) for oid in outer_ids)

fig, axes = plt.subplots(1, len(DISTRIBUTIONS),
                         figsize=(3 * len(DISTRIBUTIONS), max(3, 0.5 * len(outer_ids))),
                         sharey=True)

# Shared colour scale across all distributions for comparability
all_vals = np.concatenate([
    r2_data[dist][oid] for dist in DISTRIBUTIONS for oid in outer_ids
])
vmin, vmax = all_vals.min(), 1.0

im = None
for ax, dist in zip(axes, DISTRIBUTIONS):
    # Build matrix: rows = reservoirs, columns = inner trials
    matrix = np.full((len(outer_ids), n_inner_max), np.nan)
    for i, oid in enumerate(outer_ids):
        scores = r2_data[dist][oid]
        matrix[i, :len(scores)] = scores

    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax,
                   cmap="RdYlGn", interpolation="nearest")
    ax.set_title(DIST_LABELS[dist], fontsize=9, color=DIST_COLORS[dist], fontweight="bold")
    ax.set_xlabel("Read-in samples", fontsize=8)
    ax.set_xticks(range(n_inner_max))
    #ax.set_xticklabels(range(1, n_inner_max + 1), fontsize=7)
    ax.set_xticklabels([])

axes[0].set_ylabel("Reservoirs")
axes[0].set_yticks(range(len(outer_ids)))
axes[0].set_yticklabels([f"R{oid}" for oid in outer_ids], fontsize=8)

assert im is not None, "No heatmap was created — DISTRIBUTIONS must not be empty"
fig.colorbar(im, ax=axes[-1], label="R²", shrink=0.8)
fig.suptitle(f"SC{SET_CONSTRAINT} — R² heatmap: each row = one reservoir, each column = one read-in sample\n"
             "Row-to-row variation → reservoir effect  |  Column-to-column variation → read-in effect",
             fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_heatmap.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_heatmap.pdf"))
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_heatmap.png")}")
