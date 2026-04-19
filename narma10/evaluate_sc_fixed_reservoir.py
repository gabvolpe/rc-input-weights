"""
Evaluation: Fixed-Reservoir experiment for specified SC — NARMA10.

Loads results from sin-to-cos2/outputs/, computes R² scores per
(reservoir, read-in sample, distribution), and produces a grouped violin
plot showing per-reservoir score distributions coloured by distribution.

Please specify which set constraint (SC1, SC2, or SC3) you want to evaluate by inizializing the following variable SET_CONSTRAINT (e.g. "1" for SC1).

This evaluation script is structurally slightly different from the others, in order to match the NARMA-10 generated .npy shapes.
"""

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from utils.helpers import r2_score

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SET_CONSTRAINT="1"
RESULTS_DIR=os.path.join(os.path.dirname(__file__),"outputs","fixed-reservoir")
WASHOUT=0

DISTRIBUTIONS=["uniform","gaussian","double_gaussian","laplace","power_law"]

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

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

gt=np.load(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_ground_truth.npy"),allow_pickle=True)
gt_seq=gt[0].squeeze()

preds={}
for dist in DISTRIBUTIONS:
    preds[dist]=np.load(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_timeseries_{dist}.npy"),allow_pickle=True)

print("Ground truth shape:",gt.shape)
for dist in DISTRIBUTIONS:
    print(f"{dist:20s}:",preds[dist].shape[0],"rows")

# ---------------------------------------------------------
# R²
# ---------------------------------------------------------

outer_ids=sorted(set(preds[DISTRIBUTIONS[0]][:,0].astype(int)))
r2_data={d:{} for d in DISTRIBUTIONS}

for dist in DISTRIBUTIONS:
    data=preds[dist]
    for oid in outer_ids:
        rows=data[data[:,0]==oid]
        r2_data[dist][oid]=np.array([
            r2_score(gt_seq,row[2].squeeze(),washout=WASHOUT)
            for row in rows
        ])

# ---------------------------------------------------------
# VIOLINS
# ---------------------------------------------------------

n_dists=len(DISTRIBUTIONS)
n_outer=len(outer_ids)
group_gap=3
group_w=n_dists+group_gap

fig,ax=plt.subplots(figsize=(10,5))

for j,dist in enumerate(DISTRIBUTIONS):
    pos=[i*group_w+j for i in range(n_outer)]
    vals=[r2_data[dist][oid] for oid in outer_ids]

    vp=ax.violinplot(vals,positions=pos,widths=.8,showmedians=True,showextrema=True)

    c=DIST_COLORS[dist]
    for b in vp["bodies"]:
        b.set_facecolor(c)
        b.set_alpha(.6)

    for p in ["cmedians","cmins","cmaxes","cbars"]:
        vp[p].set_color(c)
        vp[p].set_linewidth(1.5)

centers=[i*group_w+(n_dists-1)/2 for i in range(n_outer)]
ax.set_xticks(centers)
ax.set_xticklabels([f"Reservoir {i}" for i in outer_ids])
ax.set_ylabel("R²")
ax.axhline(0,color="black",linestyle="--",alpha=.4)

patches=[mpatches.Patch(facecolor=DIST_COLORS[d],alpha=.6,label=DIST_LABELS[d]) for d in DISTRIBUTIONS]
ax.legend(handles=patches,loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_r2_violin.png"),dpi=150); plt.savefig(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_r2_violin.pdf"))
plt.show()

# ---------------------------------------------------------
# BEST PREDICTIONS
# ---------------------------------------------------------

best_preds={}

for dist in DISTRIBUTIONS:
    ts_all=np.stack([row[2].squeeze() for row in preds[dist]])
    scores=np.array([r2_score(gt_seq,ts,washout=WASHOUT) for ts in ts_all])

    best_preds[dist]={
        "best":(ts_all[np.argmax(scores)],scores.max()),
        "median":np.median(ts_all,axis=0),
        "lo":ts_all.min(axis=0),
        "hi":ts_all.max(axis=0)
    }

t=np.arange(len(gt_seq))
fig,ax=plt.subplots(figsize=(12,4))

ax.plot(t,gt_seq,color="black",linewidth=1.5,label="Ground Truth",zorder=10)

for dist in DISTRIBUTIONS:
    d=best_preds[dist]
    c=DIST_COLORS[dist]
    ts,r2=d["best"]

    ax.fill_between(t,d["lo"],d["hi"],alpha=.15,color=c)
    ax.plot(t,d["median"],linestyle="--",linewidth=1.0,alpha=.7,color=c)
    ax.plot(t,ts,color=c,linewidth=1.0,alpha=.9,label=f"{DIST_LABELS[dist]} (R²={r2:.3f})")

ax.legend(loc="upper right",fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_best_predictions.png"),dpi=150); plt.savefig(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_best_predictions.pdf"))
plt.show()

# ---------------------------------------------------------
# VARIANCE
# ---------------------------------------------------------

var_between={}
var_within={}

for dist in DISTRIBUTIONS:
    means=np.array([r2_data[dist][o].mean() for o in outer_ids])
    vars_=np.array([r2_data[dist][o].var() for o in outer_ids])

    var_between[dist]=means.var()
    var_within[dist]=vars_.mean()

fig,ax=plt.subplots(figsize=(9,4))

x=np.arange(len(DISTRIBUTIONS))
w=.35

totals=np.array([var_between[d]+var_within[d] for d in DISTRIBUTIONS])

between=np.array([100*var_between[d]/totals[i] for i,d in enumerate(DISTRIBUTIONS)])
within=np.array([100*var_within[d]/totals[i] for i,d in enumerate(DISTRIBUTIONS)])

ax.bar(x-w/2,between,w,color=[DIST_COLORS[d] for d in DISTRIBUTIONS],alpha=.85,label="Reservoir effect",edgecolor="white")
ax.bar(x+w/2,within,w,color=[DIST_COLORS[d] for d in DISTRIBUTIONS],alpha=.45,hatch="///",label="Read-in effect",edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([DIST_LABELS[d] for d in DISTRIBUTIONS])
ax.set_ylim(0,100)
ax.axhline(50,color="black",linestyle="--",alpha=.4)
import matplotlib.patches as mpatches

legend_handles = [
    mpatches.Patch(facecolor="0.2", alpha=0.9, label="Reservoir effect (between)"),
    mpatches.Patch(facecolor="0.6", alpha=0.7, hatch="///", edgecolor="white", label="Read-in effect (within)")
]

ax.legend(handles=legend_handles)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_variance_decomposition.png"),dpi=150); plt.savefig(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_variance_decomposition.pdf"))
plt.show()

# ---------------------------------------------------------
# HEATMAPS
# ---------------------------------------------------------

n_inner=max(len(r2_data[DISTRIBUTIONS[0]][o]) for o in outer_ids)

fig,axes=plt.subplots(1,len(DISTRIBUTIONS),figsize=(15,4),sharey=True)

allvals=np.concatenate([r2_data[d][o] for d in DISTRIBUTIONS for o in outer_ids])
vmin=allvals.min()
vmax=1.0

for ax,dist in zip(axes,DISTRIBUTIONS):
    M=np.full((len(outer_ids),n_inner),np.nan)

    for i,o in enumerate(outer_ids):
        s=r2_data[dist][o]
        M[i,:len(s)]=s

    im=ax.imshow(M,aspect="auto",vmin=vmin,vmax=vmax,cmap="RdYlGn",interpolation="nearest")

    ax.set_title(DIST_LABELS[dist],fontsize=9,color=DIST_COLORS[dist],fontweight="bold")
    ax.set_xticks([])

axes[0].set_yticks(range(len(outer_ids)))
axes[0].set_yticklabels([f"R{o}" for o in outer_ids])

fig.colorbar(im,ax=axes[-1],label="R²",shrink=.8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_r2_heatmap.png"),dpi=150); plt.savefig(os.path.join(RESULTS_DIR,f"sc{SET_CONSTRAINT}_r2_heatmap.pdf"))
plt.show()