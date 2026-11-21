"""Journal Figures 5 and 6: evolution of the juvenile period (12 lifetimes; 36 lifetimes and fixed periods).

Curves were recovered from the vector paths of the notebook's fig12_juvenile_evolution.pdf and
fig13_juvenile_long.pdf (their end points match evolution.csv); the bars come from evolution.csv.
Run: python journal_fig5.py journal_data journal_figs
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DATA, OUT = sys.argv[1], sys.argv[2]
S = np.load(os.path.join(DATA, "evolution_series.npz"), allow_pickle=False)
EV = pd.read_csv(os.path.join(DATA, "evolution.csv"))
WIKI, NOTE, ALT, GRAY, DARK, LIGHT = "#534AB7", "#BA7517", "#D85A30", "#888780", "#26215C", "#AFA9EC"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.32)
A, B, C = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
xl = lambda u: [f"{int(v / 300)}" for v in u]


def lifetimes(ax, total):
    ticks = np.arange(0, total + 1, 300 * (2 if total <= 3600 else 6))
    ax.set_xticks(ticks); ax.set_xticklabels(xl(ticks)); ax.set_xlim(0, total)
    ax.set_xlabel("lifetimes elapsed")


# (a) private notebooks, 12 lifetimes
for (i0, i150), color, label in (((0, 2), NOTE, "notebook, frozen world"),
                                 ((4, 5), ALT, "notebook, every 2,000 steps"),
                                 ((6, 7), WIKI, "kept features, frozen world")):
    A.plot(*S[f"f12a_{i0}"], color=color, lw=1.5, label=label)
    A.plot(*S[f"f12a_{i150}"], color=color, lw=1.5, ls="--")
A.plot(*S["f12a_1"], color=GRAY, lw=0.9, label="no selection (expected)")
A.plot(*S["f12a_3"], color=GRAY, lw=0.9, ls="--")
lifetimes(A, 3600); A.set_ylim(0, 315); A.set_ylabel("mean juvenile period (updates)")
A.set_title("(a) Private notebooks, 12 lifetimes", loc="left")
h = A.get_legend_handles_labels()
A.legend(h[0] + [Line2D([], [], color="k", lw=1.2), Line2D([], [], color="k", lw=1.2, ls="--")],
         h[1] + ["started at 0", "started at 150"], frameon=False, fontsize=6.2, loc="upper left", ncol=2,
         columnspacing=0.8, handlelength=1.6, borderaxespad=0.1)

# (b) shared wiki and die-offs, 12 lifetimes
for (i0, i150), color, label in (((0, 2), WIKI, "shared wiki, staggered deaths"),
                                 ((4, 5), NOTE, "die-offs, individual selection"),
                                 ((6, 7), DARK, "die-offs, group selection")):
    B.plot(*S[f"f12b_{i0}"], color=color, lw=1.5, label=label, drawstyle="default")
    B.plot(*S[f"f12b_{i150}"], color=color, lw=1.5, ls="--")
B.plot(*S["f12b_1"], color=GRAY, lw=0.9, label="no selection (expected)")
B.plot(*S["f12b_3"], color=GRAY, lw=0.9, ls="--")
lifetimes(B, 3600); B.set_ylim(0, 165); B.set_ylabel("mean juvenile period (updates)")
B.set_title("(b) Shared records, 12 lifetimes", loc="left")
B.legend(frameon=False, fontsize=6.8, loc="center left", bbox_to_anchor=(0.0, 0.55))

# (c) final juvenile period after 12 lifetimes vs neutral
short = {"notebook, facts never change": "notebook,\nfrozen", "notebook, facts change every 2,000 steps": "notebook,\nevery 2,000",
         "notebook, heir keeps features, facts never change": "kept\nfeatures,\nfrozen", "shared wiki, staggered deaths": "shared\nwiki",
         "mass die-offs, facts never change, individual selection": "die-offs,\nindividual", "mass die-offs, facts never change, group selection": "die-offs,\ngroup"}
w = 0.38
for j, (start, color) in enumerate(((0, WIKI), (150, ALT))):
    for i, (name, lab) in enumerate(short.items()):
        r = EV[(EV["setting"] == name) & (EV["start"] == start)].iloc[0]
        lo, hi = [float(x) for x in r["95% CI"].strip("[]").split(",")]
        x = i + (j - 0.5) * w
        C.bar(x, r["final juvenile period"], w * 0.92, color=color, yerr=[[r["final juvenile period"] - lo], [hi - r["final juvenile period"]]],
              capsize=2, error_kw={"lw": 0.8}, label=f"started at {start}" if i == 0 else None)
        C.plot([x - w * 0.46, x + w * 0.46], [r["neutral expectation"]] * 2, color="k", lw=1.2,
               label="no selection (expected)" if i == 0 and j == 0 else None)
C.set_xticks(range(len(short))); C.set_xticklabels(short.values(), fontsize=6.6)
C.set_ylim(0, 255); C.set_ylabel("final juvenile period (updates)")
C.set_title("(c) After 12 lifetimes", loc="left"); C.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=3)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig5_evolution.{ext}"), bbox_inches="tight", dpi=200)
print("wrote fig5_evolution")

# Figure 6: 36 lifetimes (a-d) and the fitness of fixed juvenile periods (e)
fig = plt.figure(figsize=(7.2, 7.9))
gs = fig.add_gridspec(3, 2, hspace=0.75, wspace=0.32, height_ratios=[1, 1, 1.15])
D = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
E = fig.add_subplot(gs[2, :])
titles = ("(a) Fresh network, facts change\nevery 2,000 steps", "(b) Fresh network, facts change\nevery 8,000 steps",
          "(c) Kept features, facts change\nevery 2,000 steps", "(d) Kept features, facts change\nevery 8,000 steps")
for k, ax in enumerate(D):
    for i in range(3):
        ax.plot(*S[f"f13{k}_{i}"], color=LIGHT, lw=0.5)
        ax.plot(*S[f"f13{k}_{5 + i}"], color="#F0997B", lw=0.5)
    ax.plot(*S[f"f13{k}_3"], color=WIKI, lw=1.7, label="started at 0")
    ax.plot(*S[f"f13{k}_8"], color=ALT, lw=1.7, label="started at 150")
    ax.plot(*S[f"f13{k}_4"], color=GRAY, lw=0.9, ls="--", label="no selection (expected)")
    ax.plot(*S[f"f13{k}_9"], color=GRAY, lw=0.9, ls="--")
    lifetimes(ax, 10800); ax.set_ylim(0, 158); ax.set_title(titles[k], loc="left", fontsize=8.5)
    if k == 0:
        ax.set_ylabel("mean juvenile period (updates)")
        ax.legend(frameon=False, fontsize=6.8, loc="upper right", bbox_to_anchor=(1.0, 0.9))
FC = pd.read_csv(os.path.join(DATA, "batch8", "fitness_curves_full.csv"))
LONG = (("notebook, facts change every 2,000 steps", "#EF9F27", "fresh network, every 2,000 steps", "o"),
        ("notebook, facts change every 8,000 steps", NOTE, "fresh network, every 8,000 steps", "s"),
        ("heir keeps features, facts change every 2,000 steps", ALT, "kept features, every 2,000 steps", "o"),
        ("heir keeps features, facts change every 8,000 steps", "#993C1D", "kept features, every 8,000 steps", "s"))
for name, color, label, mk in LONG:
    d = FC[FC["setting"] == name].sort_values("juvenile period")
    lo_hi = np.array([[float(x) for x in t.strip("[]").split(",")] for t in d["95% CI"]])
    y = d["fitness"].to_numpy()
    E.errorbar(d["juvenile period"], y, yerr=[y - lo_hi[:, 0], lo_hi[:, 1] - y], color=color, marker=mk, ms=4, capsize=2,
               lw=1.3, label=label)
    for start, mfc in ((0, color), (150, "white")):            # evolved values after 36 lifetimes, on the curve
        r = EV[(EV["setting"] == name + " (36 lifetimes)") & (EV["start"] == start)].iloc[0]
        J = float(r["final juvenile period"])
        E.plot([J], [np.interp(J, d["juvenile period"], y)], marker="D", ms=6, mfc=mfc, mec=color, mew=1.2, ls="none", zorder=5)
h, l = E.get_legend_handles_labels()
h += [Line2D([], [], color=GRAY, marker="D", ls="none", ms=5.5), Line2D([], [], color=GRAY, marker="D", ls="none", ms=5.5, mfc="white")]
l += ["selected, started at 0", "selected, started at 150"]
E.legend(h, l, frameon=False, fontsize=6.8, ncol=3, loc="lower left", columnspacing=1.0)
E.set_xticks([0, 10, 25, 50, 100, 150, 200]); E.set_xlim(-4, 204); E.set_ylim(0.46, 0.87)
E.set_xlabel("fixed juvenile period (updates)"); E.set_ylabel("lifetime reward per pull")
E.set_title("(e) Reward with the juvenile period fixed (6 lifetimes), and the selected lengths", loc="left")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig6_evolution_long.{ext}"), bbox_inches="tight", dpi=200)
print("wrote fig6_evolution_long")
