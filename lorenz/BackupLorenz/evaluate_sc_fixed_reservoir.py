"""
Evaluation: Fixed-Reservoir experiment for specified SC — Lorenz.

Loads results from lorenz/outputs/fixed-reservoir/, computes R² scores per
(reservoir, read-in sample, distribution), and produces:
- grouped violin plot
- best-prediction plot
- variance decomposition
- R² heatmap

This version assumes CODE 2 saves:
- sc1_ground_truth.npy                shape (n_time, 3)
- sc1_reservoir_weights.npy           shape (n_outer, 1 + n_reservoir_weights)
- sc1_readin_weights_{dist}.npy       shape (n_rows, 2 + n_nodes * n_inputs)
- sc1_timeseries_{dist}.npy           shape (n_rows, 2 + n_time * n_states)
- sc1_timeseries_gt.npy               shape (n_rows, 2 + n_time * n_states)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import r2_score

SET_CONSTRAINT = "1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "fixed-reservoir")
WASHOUT = 0

DISTRIBUTIONS = ["uniform", "gaussian", "double_gaussian", "laplace", "power_law"]

DIST_LABELS = {
    "uniform": "Uniform",
    "gaussian": "Gaussian",
    "double_gaussian": "Double Gaussian",
    "laplace": "Laplace",
    "power_law": "Power Law",
}
DIST_COLORS = {
    "uniform": "steelblue",
    "gaussian": "seagreen",
    "double_gaussian": "darkorange",
    "laplace": "mediumpurple",
    "power_law": "crimson",
}

gt_path = os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_ground_truth.npy")
gt = np.load(gt_path)
gt_seq = gt[:, 0]

preds = {}
for dist in DISTRIBUTIONS:
    path = os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_timeseries_{dist}.npy")
    preds[dist] = np.load(path)

print(f"Ground truth shape : {gt.shape}")
for dist in DISTRIBUTIONS:
    print(f"  {dist:20s}: {preds[dist].shape}")

outer_ids = sorted(set(preds[DISTRIBUTIONS[0]][:, 0].astype(int)))
r2_data = {dist: {} for dist in DISTRIBUTIONS}

for dist in DISTRIBUTIONS:
    data = preds[dist]
    for oid in outer_ids:
        rows = data[data[:, 0] == oid]
        scores = np.array([
            r2_score(gt_seq, row[2:], washout=WASHOUT)
            for row in rows
        ])
        r2_data[dist][oid] = scores

n_dists = len(DISTRIBUTIONS)
n_outer = len(outer_ids)
group_gap = 3
group_w = n_dists + group_gap

fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_outer * n_dists), 5))

for j, dist in enumerate(DISTRIBUTIONS):
    positions = [i * group_w + j for i in range(n_outer)]
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

group_centers = [i * group_w + (n_dists - 1) / 2 for i in range(n_outer)]
ax.set_xticks(group_centers)
ax.set_xticklabels([f"Reservoir {oid}" for oid in outer_ids])

ax.set_ylabel("R²")
ax.set_title(
    f"SC{SET_CONSTRAINT} Fixed-Reservoir — R² per reservoir and read-in distribution\n"
    f"(washout = {WASHOUT} timesteps, inner trials shown as violins)"
)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

all_r2 = np.concatenate([scores for dist in DISTRIBUTIONS for scores in r2_data[dist].values()])
r2_min = all_r2.min()
ax.set_ylim(bottom=r2_min - 0.01 * abs(r2_min), top=1.0)

legend_patches = [
    mpatches.Patch(facecolor=DIST_COLORS[d], alpha=0.6, label=DIST_LABELS[d])
    for d in DISTRIBUTIONS
]
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_r2_violin.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_r2_violin.pdf"))
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, f'sc{SET_CONSTRAINT}_r2_violin.png')}")


best_preds = {}
for dist in DISTRIBUTIONS:
    data = preds[dist]
    ts_all = data[:, 2:].copy()

    if ts_all.ndim == 1:
        ts_all = ts_all[None, :]

    n_rows, n_flat = ts_all.shape
    n_time = gt_seq.shape[0]
    n_states = gt.shape[1]

    if n_flat == n_time:
        ts_all_1d = ts_all
    elif n_flat == n_time * n_states:
        ts_all_1d = ts_all.reshape(n_rows, n_time, n_states)[:, :, 0]
    else:
        raise ValueError(
            f"{dist}: prediction length {n_flat} does not match gt length {n_time} "
            f"and is not equal to n_time*n_states={n_time * n_states}"
        )

    scores = np.array([
        r2_score(gt_seq, row, washout=WASHOUT)
        for row in ts_all_1d
    ])
    best_idx = np.argmax(scores)

    best_preds[dist] = {
        "best": (ts_all_1d[best_idx], scores[best_idx]),
        "median": np.median(ts_all_1d, axis=0),
        "lo": np.min(ts_all_1d, axis=0),
        "hi": np.max(ts_all_1d, axis=0),
    }

t = np.arange(gt_seq.shape[0])[WASHOUT:]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, gt_seq[WASHOUT:], color="black", linewidth=1.5, label="Ground truth", zorder=10)

for dist in DISTRIBUTIONS:
    d = best_preds[dist]
    color = DIST_COLORS[dist]
    ts, r2 = d["best"]

    ax.fill_between(t, d["lo"][WASHOUT:], d["hi"][WASHOUT:], color=color, alpha=0.15)
    ax.plot(t, d["median"][WASHOUT:], color=color, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot(t, ts[WASHOUT:], color=color, linewidth=1.0, alpha=0.9,
            label=f"{DIST_LABELS[dist]} (best R²={r2:.3f})")

ax.set_xlabel("Timestep")
ax.set_ylabel("Output")
ax.set_title(
    f"SC{SET_CONSTRAINT} Fixed-Reservoir — Ground truth vs. predictions per distribution\n"
    "solid: best | dashed: median | shaded: full range"
)
ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_best_predictions.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_best_predictions.pdf"))
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, f'sc{SET_CONSTRAINT}_best_predictions.png')}")


var_between = {}
var_within = {}

for dist in DISTRIBUTIONS:
    outer_means = np.array([r2_data[dist][oid].mean() for oid in outer_ids])
    outer_vars = np.array([r2_data[dist][oid].var() for oid in outer_ids])
    var_between[dist] = outer_means.var()
    var_within[dist] = outer_vars.mean()

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(DISTRIBUTIONS))
width = 0.35

for i, (label, key, hatch) in enumerate([
    ("Reservoir (between)", var_between, ""),
    ("Read-in (within)", var_within, "///"),
]):
    totals = np.array([var_between[d] + var_within[d] for d in DISTRIBUTIONS])
    values = np.array([key[d] for d in DISTRIBUTIONS])
    fractions = np.where(totals > 0, values / totals * 100, 0)
    ax.bar(x + (i - 0.5) * width, fractions,
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
ax.set_title(
    f"SC{SET_CONSTRAINT} Fixed-Reservoir — Variance decomposition: reservoir vs. read-in weights\n"
    "solid fill = reservoir effect | hatched = read-in effect"
)

legend_handles = [
    mpatches.Patch(facecolor="dimgray", alpha=0.9, label="Reservoir effect (between)"),
    mpatches.Patch(facecolor="0.6", hatch="///", edgecolor="white", alpha=0.7, label="Read-in effect (within)"),
]
ax.legend(handles=legend_handles, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_variance_decomposition.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_variance_decomposition.pdf"))
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, f'sc{SET_CONSTRAINT}_variance_decomposition.png')}")


n_inner_max = max(len(r2_data[DISTRIBUTIONS[0]][oid]) for oid in outer_ids)

fig, axes = plt.subplots(1, len(DISTRIBUTIONS),
                         figsize=(3 * len(DISTRIBUTIONS), max(3, 0.5 * len(outer_ids))),
                         sharey=True)

all_vals = np.concatenate([r2_data[dist][oid] for dist in DISTRIBUTIONS for oid in outer_ids])
vmin, vmax = all_vals.min(), 1.0

im = None
for ax, dist in zip(axes, DISTRIBUTIONS):
    matrix = np.full((len(outer_ids), n_inner_max), np.nan)
    for i, oid in enumerate(outer_ids):
        scores = r2_data[dist][oid]
        matrix[i, :len(scores)] = scores

    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax,
                   cmap="RdYlGn", interpolation="nearest")
    ax.set_title(DIST_LABELS[dist], fontsize=9, color=DIST_COLORS[dist], fontweight="bold")
    ax.set_xlabel("Read-in samples", fontsize=8)
    ax.set_xticks(range(n_inner_max))
    ax.set_xticklabels([])

axes[0].set_ylabel("Reservoirs")
axes[0].set_yticks(range(len(outer_ids)))
axes[0].set_yticklabels([f"R{oid}" for oid in outer_ids], fontsize=8)

assert im is not None
fig.colorbar(im, ax=axes[-1], label="R²", shrink=0.8)
fig.suptitle(
    f"SC{SET_CONSTRAINT} — R² heatmap: each row = one reservoir, each column = one read-in sample\n"
    "Row-to-row variation → reservoir effect  |  Column-to-column variation → read-in effect",
    fontsize=9
)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_r2_heatmap.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_r2_heatmap.pdf"))
plt.show()
print(f"Saved plot → {os.path.join(RESULTS_DIR, f'sc{SET_CONSTRAINT}_r2_heatmap.png')}")