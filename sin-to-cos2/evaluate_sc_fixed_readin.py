"""
Evaluation: Fixed Read-In | Variable Reservoir (sin-to-cos2) for specified SC

Loads SC outputs from:
    outputs/fixed-readin/

Computes:
- R² per (read-in set, reservoir, distribution)
- Violin plot grouped by read-in (outer)
- Best prediction plots
- Variance decomposition (read-in vs reservoir)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import r2_score

# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------
SET_CONSTRAINT = "1"
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "fixed-readin"
)

WASHOUT = 0

DISTRIBUTIONS = [
    "uniform",
    "gaussian",
    "double_gaussian",
    "laplace",
    "power_law",
]

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

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
gt_path = os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_ground_truth.npy")
gt = np.load(gt_path)[:, 0]

preds = {
    d: np.load(os.path.join(RESULTS_DIR, f"sc{SET_CONSTRAINT}_timeseries_{d}.npy"))
    for d in DISTRIBUTIONS
}

print("Loaded:")
print("GT:", gt.shape)
for d in DISTRIBUTIONS:
    print(d, preds[d].shape)

# ------------------------------------------------------------
# IDS (outer = read-in, inner = reservoir)
# ------------------------------------------------------------
outer_ids = np.unique(preds[DISTRIBUTIONS[0]][:, 0]).astype(int)

# ------------------------------------------------------------
# R² COMPUTATION
# ------------------------------------------------------------
r2_data = {d: {} for d in DISTRIBUTIONS}

for d in DISTRIBUTIONS:
    data = preds[d]

    for oid in outer_ids:
        rows = data[data[:, 0] == oid]

        scores = np.array([
            r2_score(gt, row[2:], washout=WASHOUT)
            for row in rows
        ])

        r2_data[d][oid] = scores

# ------------------------------------------------------------
# VIOLIN PLOT (outer = read-in, inner = reservoirs)
# ------------------------------------------------------------
n_dists = len(DISTRIBUTIONS)
n_outer = len(outer_ids)

group_gap = 3
group_w = n_dists + group_gap

fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_outer * n_dists), 5))

for j, dist in enumerate(DISTRIBUTIONS):
    positions = [i * group_w + j for i in range(n_outer)]
    score_lists = [r2_data[dist][oid] for oid in outer_ids]

    vp = ax.violinplot(
        score_lists,
        positions=positions,
        widths=0.8,
        showmedians=True,
        showextrema=True
    )

    color = DIST_COLORS[dist]

    for pc in vp["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    for part in ("cmedians", "cmaxes", "cmins", "cbars"):
        vp[part].set_color(color)
        vp[part].set_linewidth(1.5)

centers = [i * group_w + (n_dists - 1) / 2 for i in range(n_outer)]
ax.set_xticks(centers)
ax.set_xticklabels([f"Read-in set {i}" for i in outer_ids])

ax.set_ylabel("R²")
ax.set_title(
    f"SC{SET_CONSTRAINT} Fixed-Read-in — R² per read-in distribution and reservoirs\n"
    f"(washout = {WASHOUT} timesteps, outer trials shown as violins)"
)

ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)

legend = [
    mpatches.Patch(color=DIST_COLORS[d], label=DIST_LABELS[d], alpha=0.6)
    for d in DISTRIBUTIONS
]
ax.legend(handles=legend, loc="lower right", framealpha=0.9)

all_r2 = np.concatenate([
    scores for dist in DISTRIBUTIONS for scores in r2_data[dist].values()
])
r2_min = all_r2.min()
ax.set_ylim(bottom=r2_min - 0.01 * abs(r2_min), top=1.0)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_violin.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_violin.pdf"))
plt.show()

print("Saved violin plot")

# ------------------------------------------------------------
# BEST PREDICTIONS
# ------------------------------------------------------------
best_preds = {}

for d in DISTRIBUTIONS:
    data = preds[d]
    ts = data[:, 2:]

    scores = np.array([
        r2_score(gt, row, washout=WASHOUT)
        for row in ts
    ])

    best_idx = np.argmax(scores)

    best_preds[d] = {
        "best": ts[best_idx],
        "score": scores[best_idx],
        "median": np.median(ts, axis=0),
        "lo": ts.min(axis=0),
        "hi": ts.max(axis=0),
    }

t = np.arange(len(gt))[WASHOUT:]
gt_plot = gt[WASHOUT:]

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(t, gt_plot, color="black", linewidth=1.5, label="Ground truth")

for d in DISTRIBUTIONS:
    color = DIST_COLORS[d]
    dres = best_preds[d]

    ax.fill_between(t, dres["lo"][WASHOUT:], dres["hi"][WASHOUT:], alpha=0.15, color=color)
    ax.plot(t, dres["median"][WASHOUT:], linestyle="--", color=color, alpha=0.7)
    ax.plot(t, dres["best"][WASHOUT:], color=color,
            label=f"{DIST_LABELS[d]} (best {dres['score']:.3f})")

ax.set_title(f"SC{SET_CONSTRAINT} Fixed-Read-in — Ground truth vs. predictions per distribution\n"
             "solid: best | dashed: median | shaded: full range")
ax.set_xlabel("Timestep")
ax.set_ylabel("Output")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_best_predictions.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_best_prediction.pdf"))
plt.show()

print("Saved best prediction plot")

# ------------------------------------------------------------
# VARIANCE DECOMPOSITION (CORRECTED)
# ------------------------------------------------------------
readin_effect = {}      # BETWEEN read-in sets
reservoir_effect = {}   # WITHIN read-in sets

for d in DISTRIBUTIONS:

    # mean performance per read-in set (averaged over reservoirs)
    outer_means = np.array([r2_data[d][oid].mean() for oid in outer_ids])

    # variability across read-in sets
    readin_effect[d] = outer_means.var()

    # variability across reservoirs within each read-in
    outer_vars = np.array([r2_data[d][oid].var() for oid in outer_ids])
    reservoir_effect[d] = outer_vars.mean()

# ------------------------------------------------------------
# PLOT VARIANCE DECOMPOSITION
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))

x = np.arange(len(DISTRIBUTIONS))
width = 0.35

for i, (label, source, hatch) in enumerate([
    ("Read-in effect (between)", readin_effect, ""),
    ("Reservoir effect (within)", reservoir_effect, "///"),
]):
    totals = np.array([
        readin_effect[d] + reservoir_effect[d]
        for d in DISTRIBUTIONS
    ])

    vals = np.array([source[d] for d in DISTRIBUTIONS])

    frac = np.where(totals > 0, vals / totals * 100, 0)

    ax.bar(
        x + (i - 0.5) * width,
        frac,
        width=width,
        alpha=0.85 if i == 0 else 0.5,
        hatch=hatch,
        label=label,
        color=[DIST_COLORS[d] for d in DISTRIBUTIONS],
        edgecolor="white"
    )

ax.set_xticks(x)
ax.set_xticklabels([DIST_LABELS[d] for d in DISTRIBUTIONS])
ax.set_ylabel("Variance explained (%)")
ax.set_ylim(0, 100)
ax.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_title(f"SC{SET_CONSTRAINT} Fixed-Read-in set — Variance decomposition: read-in vs. reservoir weights\n"
             "solid fill = read-in effect | hatched = reservoir effect")

legend_handles = [
    mpatches.Patch(
        facecolor="dimgray",   
        alpha=0.9,
        label="Read-in effect (between)"
    ),
    mpatches.Patch(
        facecolor="0.6",   # lighter for contrast
        hatch="///",
        edgecolor="white",
        alpha=0.7,
        label="Reservoir effect (within)"
    ),
]
#ax.legend(framealpha=0.9)
ax.legend(handles=legend_handles, framealpha=0.9)

ax.legend(handles=legend_handles, framealpha=0.9)
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_variance_decomposition.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_variance_decomposition.pdf"))
plt.show()

print("Saved variance plot")

# ------------------------------------------------------------
# HEATMAP (outer × inner)
# ------------------------------------------------------------
n_inner = max(len(r2_data[DISTRIBUTIONS[0]][oid]) for oid in outer_ids)

fig, axes = plt.subplots(
    1, len(DISTRIBUTIONS),
    figsize=(3 * len(DISTRIBUTIONS), 3 + len(outer_ids)),
    sharey=True
)

all_vals = np.concatenate([
    r2_data[d][oid] for d in DISTRIBUTIONS for oid in outer_ids
])

vmin, vmax = all_vals.min(), 1.0

for ax, d in zip(axes, DISTRIBUTIONS):

    mat = np.full((len(outer_ids), n_inner), np.nan)

    for i, oid in enumerate(outer_ids):
        vals = r2_data[d][oid]
        mat[i, :len(vals)] = vals

    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdYlGn")

    ax.set_title(DIST_LABELS[d], color=DIST_COLORS[d], fontsize=9)
    ax.set_xlabel("Reservoirs")
    ax.set_xticks(range(n_inner))
    #ax.set_xticklabels(range(1, n_inner + 1), fontsize=7)
    ax.set_xticklabels([])

axes[0].set_ylabel("Read-in samples")
axes[0].set_yticks(range(len(outer_ids)))
axes[0].set_yticklabels([f"R-in{oid}" for oid in outer_ids], fontsize=7)

fig.colorbar(im, ax=axes[-1], label="R²")
fig.suptitle(f"SC{SET_CONSTRAINT} — R² heatmap: each row = one read-in set, each column = one reservoir\n"
             "Row-to-row variation → read-in effect  |  Column-to-column variation → reservoir effect",
             fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_heatmap.png"), dpi=150)
plt.savefig(os.path.join(RESULTS_DIR, "sc"+SET_CONSTRAINT+"_r2_heatmap.pdf"))
plt.show()

print("Saved heatmap")