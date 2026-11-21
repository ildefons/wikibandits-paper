"""Journal version of the notebook's survival figure (wiki majority right over six lifetimes).

The curves were recovered from the vector paths of the notebook's fig5_survival.pdf
(mean of three seeds, smoothed over 15 updates); their late-run means match summary.csv.
Run: python journal_fig_survival.py journal_data/survival_series.npz journal_figs
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src, out = sys.argv[1], sys.argv[2]
d = np.load(src)
WIKI, NOTE, ALT, GRAY = "#534AB7", "#BA7517", "#D85A30", "#888780"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.5, 3.2))
for u in range(300, 1800, 300):
    ax.axvline(u, color="#D3D1C7", lw=0.6, ls=":", zorder=0)
for key, label, color, ls in (("individual_cited", "shared wiki, credit for reuse", WIKI, "-"),
                              ("wipe_cited", "shared wiki, credit, dead authors' pages erased", ALT, "--"),
                              ("individual", "shared wiki, no credit", GRAY, "-"),
                              ("rival_info", "shared wiki, rivalry (c = −0.5)", NOTE, "-")):
    x, y = d[key]
    ax.plot(x, y, color=color, ls=ls, lw=1.6, label=label)
ax.set_xlim(0, 1800); ax.set_ylim(0, 1.02)
ax.set_xticks(range(0, 1801, 300))
ax.set_xticklabels([f"{u:,}\n({u // 300})" if u else "0\n(0)" for u in range(0, 1801, 300)])
ax.set_xlabel("update (lifetimes elapsed; each agent lives about 300 updates)")
ax.set_ylabel("machines whose wiki\nmajority names the good arm")
ax.axhline(0.2, color=GRAY, ls=":", lw=0.8); ax.text(1790, 0.215, "chance", ha="right", fontsize=7, color=GRAY)
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
os.makedirs(out, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig_survival.{ext}"), bbox_inches="tight", dpi=200)
print("wrote fig_survival")
