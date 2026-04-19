"""
Evaluation: Fixed Read-In | Variable Reservoir (NARMA10) for specified SC

Loads results from narma10/outputs/, computes R² scores per
(reservoir, read-in sample, distribution), and produces a grouped violin
plot showing per-reservoir score distributions coloured by distribution.

Please specify which set constraint (SC1, SC2, or SC3) you want to evaluate by inizializing the following variable SET_CONSTRAINT (e.g. "1" for SC1).

This evaluation script is structurally slightly different from the others, in order to match the NARMA-10 generated .npy shapes.
"""

"""
Evaluation: Fixed Read-In | Variable Reservoir (NARMA10)

Corrected for actual saved structure:

sc1_ground_truth.npy:
    shape (400,10)

sc1_timeseries_{dist}.npy:
    rows:
       (outer_id, inner_id, pred)

where:
    pred.shape == (10,)

Thus evaluation is done against one matching GT sequence:
    gt[0]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import r2_score


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
SET_CONSTRAINT="1"

RESULTS_DIR=os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "fixed-readin"
)

WASHOUT=0


DISTRIBUTIONS=[
    "uniform",
    "gaussian",
    "double_gaussian",
    "laplace",
    "power_law"
]


DIST_LABELS={
    "uniform":"Uniform",
    "gaussian":"Gaussian",
    "double_gaussian":"Double Gaussian",
    "laplace":"Laplace",
    "power_law":"Power Law"
}


DIST_COLORS={
    "uniform":"steelblue",
    "gaussian":"seagreen",
    "double_gaussian":"darkorange",
    "laplace":"mediumpurple",
    "power_law":"crimson"
}



# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
gt=np.load(

    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_ground_truth.npy"
    ),

    allow_pickle=True

)

# correct GT shape -> (10,)
gt=np.asarray(
    gt[0]
).reshape(-1)


preds={

    d:np.load(

        os.path.join(
            RESULTS_DIR,
            f"sc{SET_CONSTRAINT}_timeseries_{d}.npy"
        ),

        allow_pickle=True

    )

    for d in DISTRIBUTIONS
}


print("Loaded:")
print("GT:",gt.shape)

for d in DISTRIBUTIONS:
    print(
        d,
        preds[d].shape
    )



# ------------------------------------------------------------
# OUTER IDS
# ------------------------------------------------------------
outer_ids=np.unique(
    [row[0] for row in preds[DISTRIBUTIONS[0]]]
).astype(int)



# ------------------------------------------------------------
# R² COMPUTATION
# ------------------------------------------------------------
r2_data={
    d:{} for d in DISTRIBUTIONS
}


for d in DISTRIBUTIONS:

    rows=preds[d]

    for oid in outer_ids:

        these=[

            row for row in rows

            if row[0]==oid

        ]

        scores=[]

        for row in these:

            pred=np.asarray(
                row[2]
            ).reshape(-1)

            scores.append(

                r2_score(
                    gt,
                    pred,
                    washout=WASHOUT
                )

            )

        r2_data[d][oid]=np.array(
            scores
        )



# ------------------------------------------------------------
# VIOLIN PLOT
# ------------------------------------------------------------
n_dists=len(DISTRIBUTIONS)
n_outer=len(outer_ids)

group_gap=3
group_w=n_dists+group_gap

fig,ax=plt.subplots(
    figsize=(max(
        8,
        1.5*n_outer*n_dists
    ),5)
)


for j,dist in enumerate(
    DISTRIBUTIONS
):

    positions=[
        i*group_w+j
        for i in range(n_outer)
    ]

    vals=[

        r2_data[dist][oid]

        for oid in outer_ids

    ]

    vp=ax.violinplot(
        vals,
        positions=positions,
        widths=.8,
        showmedians=True,
        showextrema=True
    )

    c=DIST_COLORS[dist]

    for pc in vp["bodies"]:

        pc.set_facecolor(c)
        pc.set_alpha(.6)

    for part in (
        "cmedians",
        "cmaxes",
        "cmins",
        "cbars"
    ):
        vp[part].set_color(c)


centers=[
    i*group_w+(n_dists-1)/2
    for i in range(n_outer)
]


ax.set_xticks(
    centers
)

ax.set_xticklabels(
    [
        f"Read-in {i}"
        for i in outer_ids
    ]
)

ax.set_ylabel("R²")

ax.set_title(
    f"SC{SET_CONSTRAINT} Fixed Read-In\n"
    "R² across reservoirs"
)


legend=[

    mpatches.Patch(
        color=DIST_COLORS[d],
        label=DIST_LABELS[d]
    )

    for d in DISTRIBUTIONS

]

ax.legend(
    handles=legend
)


all_r2=np.concatenate([

    vals

    for d in DISTRIBUTIONS
    for vals in r2_data[d].values()

])


ax.set_ylim(
    all_r2.min()-0.01*abs(
        all_r2.min()
    ),
    1.0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_r2_violin.png"
    ),
    dpi=150
)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_r2_violin.pdf"
    )
)

plt.show()

print(
    "Saved violin plot"
)



# ------------------------------------------------------------
# BEST PREDICTIONS
# ------------------------------------------------------------
best_preds={}


for d in DISTRIBUTIONS:

    ts=np.array([

        np.asarray(
            row[2]
        ).reshape(-1)

        for row in preds[d]

    ])


    scores=np.array([

        r2_score(
            gt,
            t,
            washout=WASHOUT
        )

        for t in ts

    ])


    best_idx=np.argmax(
        scores
    )


    best_preds[d]={

        "best":ts[best_idx],

        "score":scores[best_idx],

        "median":np.median(
            ts,
            axis=0
        ),

        "lo":ts.min(axis=0),

        "hi":ts.max(axis=0)

    }



t=np.arange(
    len(gt)
)


fig,ax=plt.subplots(
    figsize=(12,4)
)

ax.plot(
    t,
    gt,
    color="black",
    label="Ground truth"
)


for d in DISTRIBUTIONS:

    c=DIST_COLORS[d]

    r=best_preds[d]

    ax.fill_between(
        t,
        r["lo"],
        r["hi"],
        alpha=.15,
        color=c
    )

    ax.plot(
        t,
        r["median"],
        linestyle="--",
        color=c
    )

    ax.plot(
        t,
        r["best"],
        color=c,
        label=f"{DIST_LABELS[d]} ({r['score']:.3f})"
    )


ax.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_best_predictions.png"
    ),
    dpi=150
)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_best_predictions.pdf"
    )
)

plt.show()

print(
    "Saved best predictions"
)



# ------------------------------------------------------------
# VARIANCE DECOMPOSITION
# ------------------------------------------------------------
readin_effect = {}      # variance BETWEEN read-in sets
reservoir_effect = {}   # variance WITHIN read-in sets

for d in DISTRIBUTIONS:

    # mean performance per read-in set (averaged over reservoirs)
    outer_means = np.array([
        r2_data[d][oid].mean()
        for oid in outer_ids
    ])

    # variability across read-in sets
    readin_effect[d] = outer_means.var()

    # variability across reservoirs within each read-in
    outer_vars = np.array([
        r2_data[d][oid].var()
        for oid in outer_ids
    ])

    reservoir_effect[d] = outer_vars.mean()


# ------------------------------------------------------------
# PLOT VARIANCE DECOMPOSITION (MATCHED STYLE)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))

x = np.arange(len(DISTRIBUTIONS))
width = 0.35


for i, (label, source, hatch) in enumerate([
    ("Read-in (between)", readin_effect, ""),
    ("Reservoir (within)", reservoir_effect, "///"),
]):

    totals = np.array([
        readin_effect[d] + reservoir_effect[d]
        for d in DISTRIBUTIONS
    ])

    vals = np.array([
        source[d]
        for d in DISTRIBUTIONS
    ])

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

ax.axhline(
    50,
    color="black",
    linewidth=0.8,
    linestyle="--",
    alpha=0.4
)

ax.set_title(
    f"SC{SET_CONSTRAINT} Fixed-Read-in — Variance decomposition: read-in vs. reservoir\n"
    "solid fill = read-in effect | hatched = reservoir effect"
)


# ------------------------------------------------------------
# LEGEND (EXACT STYLE AS REQUESTED)
# ------------------------------------------------------------
legend_handles = [
    mpatches.Patch(
        facecolor="dimgray",
        alpha=0.9,
        label="Read-in effect (between)"
    ),
    mpatches.Patch(
        facecolor="0.6",
        hatch="///",
        edgecolor="white",
        alpha=0.7,
        label="Reservoir effect (within)"
    ),
]

ax.legend(handles=legend_handles, framealpha=0.9)


plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_variance_decomposition.png"
    ),
    dpi=150
)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_variance_decomposition.pdf"
    )
)

plt.show()

print("Saved variance plot")


# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------
n_inner=max(

    len(
        r2_data[DISTRIBUTIONS[0]][oid]
    )

    for oid in outer_ids

)


fig,axes=plt.subplots(
    1,
    len(DISTRIBUTIONS),
    figsize=(15,6),
    sharey=True
)


all_vals=np.concatenate([

    r2_data[d][oid]

    for d in DISTRIBUTIONS
    for oid in outer_ids

])


vmin=all_vals.min()
vmax=1.0


for ax,d in zip(
    axes,
    DISTRIBUTIONS
):

    mat=np.full(
        (
            len(outer_ids),
            n_inner
        ),
        np.nan
    )

    for i,oid in enumerate(
        outer_ids
    ):

        vals=r2_data[d][oid]

        mat[i,:len(vals)]=vals


    im=ax.imshow(
        mat,
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax
    )


    ax.set_title(
        DIST_LABELS[d]
    )

    ax.set_xlabel(
        "Reservoir"
    )


axes[0].set_ylabel(
    "Read-in"
)


fig.colorbar(
    im,
    ax=axes[-1],
    label="R²"
)


plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_heatmap.png"
    ),
    dpi=150
)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"sc{SET_CONSTRAINT}_heatmap.pdf"
    )
)

plt.show()

print(
    "Saved heatmap"
)

print(
    "DONE"
)