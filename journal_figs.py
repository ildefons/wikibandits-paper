"""Journal figures drawn from the result tables (summary.csv and the comparison files).

Run:  python journal_figs.py journal_data journal_figs
Writes PDF (vector, for submission) and PNG (for the draft) versions of Figures 2, 3, 4, 7, 8 and 9.
Figures that need per-update logs (evolution trajectories, symbols over time) come from the notebook.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

DATA, OUT = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("journal_data", "journal_figs")
os.makedirs(OUT, exist_ok=True)

WIKI, NOTE, ALT, GRAY, DARK, LIGHT = "#534AB7", "#BA7517", "#D85A30", "#888780", "#26215C", "#AFA9EC"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

S = pd.read_csv(os.path.join(DATA, "summary.csv")).set_index("setting")
w = 0.36  # width of a bar in grouped bar charts


def val(name, col):
    """(mean, standard error) of a summary cell such as '0.719 ± 0.001'."""
    m = re.match(r"\s*([-\d.]+)\s*±\s*([-\d.]+)", str(S.loc[name, col]))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def ci(name, col, seeds=None):
    m, se = val(name, col)
    n = int(S.loc[name, "seeds"]) if seeds is None else seeds
    return m, stats.t.ppf(0.975, n - 1) * se


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def hbars(ax, items, xlabel, xlim=(0, 1)):
    """Horizontal bars: items = [(label, value, error, colour, note)], drawn top to bottom."""
    for i, (label, v, e, color, note) in enumerate(items):
        y = len(items) - 1 - i
        ax.barh(y, v, height=0.7, color=color, xerr=e if e else None, capsize=2, error_kw={"lw": 0.8})
        ax.text(v + (e or 0) + 0.012, y, f"{v:.2f}" + (f"  {note}" if note else ""), va="center", fontsize=7.5)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items][::-1])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)


# Figure 2: adults reach the best strategy for their memory; newborns need others' pages
fig, ax = plt.subplots(2, 1, figsize=(6.6, 6.4), gridspec_kw={"hspace": 0.45})
hbars(ax[0], [
    ("Random", 0.26, 0, GRAY, ""),
    ("Best player without memory", 0.61, 0, GRAY, ""),
    ("Learners, no record", *val("no_wiki", "adult reward"), DARK, ""),
    ("Learners, private notebook", *val("notebook_cited", "adult reward"), NOTE, ""),
    ("Learners, shared wiki", *val("individual_cited", "adult reward"), WIKI, ""),
    ("Bayesian, remembers every machine", 0.87, 0, GRAY, ""),
    ("Bayesian, pools everything", 0.90, 0, GRAY, ""),
    ("Omniscient", 0.90, 0, GRAY, "")], "adult reward per pull")
ax[0].set_title("(a) Adults against the best strategies", loc="left")
t = lambda n: f"({int(val(n, 't to 0.61')[0])} updates)" if np.isfinite(val(n, "t to 0.61")[0]) else "(never)"
hbars(ax[1], [
    ("No record", *val("no_wiki", "newborn reward"), DARK, t("no_wiki")),
    ("Own notebook, inherited", *val("notebook_cited", "newborn reward"), NOTE, t("notebook_cited")),
    ("Own notebook, erased at death", *val("notebook_wiped_cited", "newborn reward"), NOTE, t("notebook_wiped_cited")),
    ("Own entry repeated 16 times", *val("notebook_replicated_cited", "newborn reward"), LIGHT, t("notebook_replicated_cited")),
    ("Own page + 1 partner's", *val("commons1_cited", "newborn reward"), LIGHT, t("commons1_cited")),
    ("Own page + 3 partners'", *val("commons3_cited", "newborn reward"), LIGHT, t("commons3_cited")),
    ("Shared wiki (all 16 pages)", *val("individual_cited", "newborn reward"), WIKI, t("individual_cited")),
    ("Perfect record (oracle)", *val("oracle", "newborn reward"), GRAY, t("oracle"))], "newborn reward per pull")
ax[1].set_title("(b) Newborns, by what they can read", loc="left")
save(fig, "fig2_newcomers")

# Figure 3: when and where the record pays
fig, ax = plt.subplots(2, 2, figsize=(7, 6.4), gridspec_kw={"wspace": 0.5, "hspace": 0.55})
ax = ax.flatten()
DR, LF = [500, 2000, 8000], [100, 300, 900]


def drift_pair(d, life):
    if d == 2000:
        return ("no_wiki", "individual_cited") if life == 300 else (f"no_wiki_life{life}", f"cited_life{life}")
    return f"no_wiki_drift{d}_life{life}", f"cited_drift{d}_life{life}"


seq = LinearSegmentedColormap.from_list("seq", ["#EEEDFE", WIKI, DARK])
for a, col, title in ((ax[0], "newborn reward", "(a) Gain for newborns"), (ax[1], "population reward", "(b) Gain for the population")):
    M = np.array([[val(drift_pair(d, life)[1], col)[0] - val(drift_pair(d, life)[0], col)[0] for life in LF] for d in DR])
    a.imshow(M, cmap=seq, vmin=0.22, vmax=0.46, origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            r, g, b, _ = seq((M[i, j] - 0.22) / 0.24)
            a.text(j, i, f"+{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                   color="white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else "#2C2C2A")
    a.set_xticks(range(3)); a.set_xticklabels(LF); a.set_xlabel("lifespan (updates)")
    a.set_yticks(range(3)); a.set_yticklabels([f"{d:,}" for d in DR]); a.set_ylabel("steps between changes")
    a.set_title(title, loc="left")
    a.spines[["left", "bottom"]].set_visible(False)
a = ax[2]
for label, names, color in (("no record", ["no_wiki_K4", "no_wiki", "no_wiki_K32"], DARK),
                            ("private notebook", ["notebook_cited_K4", "notebook_cited", "notebook_cited_K32"], NOTE),
                            ("shared wiki", ["cited_K4", "individual_cited", "cited_K32"], WIKI)):
    m = [ci(n, "newborn reward") for n in names]
    a.errorbar([4, 8, 32], [x for x, _ in m], yerr=[e for _, e in m], marker="o", ms=4, color=color, capsize=2, label=label)
a.set_xscale("log", base=2); a.set_xticks([4, 8, 32]); a.set_xticklabels([4, 8, 32]); a.set_ylim(0.25, 0.8)
a.set_xlabel("machines"); a.set_ylabel("newborn reward per pull"); a.set_title("(c) Bigger worlds", loc="left")
a.legend(frameon=False, loc="center right")
a = ax[3]
for label, names, color in (("no record", ["no_wiki_N4", "no_wiki_N8", "no_wiki", "no_wiki_N32"], DARK),
                            ("shared wiki", ["cited_N4", "cited_N8", "individual_cited", "cited_N32"], WIKI)):
    m = [ci(n, "newborn reward") for n in names]
    a.errorbar([4, 8, 16, 32], [x for x, _ in m], yerr=[e for _, e in m], marker="o", ms=4, color=color, capsize=2, label=label)
a.set_xscale("log", base=2); a.set_xticks([4, 8, 16, 32]); a.set_xticklabels([4, 8, 16, 32]); a.set_ylim(0.25, 0.8)
a.set_xlabel("agents"); a.set_ylabel("newborn reward per pull"); a.set_title("(d) Bigger populations", loc="left")
a.legend(frameon=False, loc="center right")
save(fig, "fig3_robustness")

# Figure 4: heirs who write before they read
C3 = pd.read_csv(os.path.join(DATA, "contrasts3.csv"))
EV = pd.read_csv(os.path.join(DATA, "events.csv")).set_index("setting")


def contrast(a, b, measure="newborn reward"):
    r = C3[(C3["a"] == a) & (C3["b"] == b) & (C3["measure"] == measure)]
    if not len(r):
        return None
    lo, hi = [float(x) for x in r.iloc[0]["95% CI"].strip("[]").split(",")]
    return float(r.iloc[0]["a - b"]), lo, hi


fig, _ax = plt.subplots(2, 2, figsize=(7, 6.9), gridspec_kw={"hspace": 0.5, "wspace": 0.38})
ax = [_ax[0, 0], _ax[0, 1], _ax[1, 0], _ax[1, 1]]
a = ax[0]
TAGS = [("2,000", ""), ("8,000", "_drift8000"), ("never", "_static")]
PAIRS = [("notebook, fresh heir", "notebook_cited", "notebook_wiped_cited", NOTE, -0.12),
         ("notebook, heir keeps features", "notebook_torso_cited", "notebook_wiped_torso_cited", ALT, 0.0),
         ("pages after a mass die-off", "sync_cited", "sync_wipe_cited", WIKI, 0.12)]
for juv, ls, mk in (("", ":", "o"), ("_juv", "-", "s")):
    for label, A, B, color, off in PAIRS:
        xs, ys, lo, hi = [], [], [], []
        for i, (_, tag) in enumerate(TAGS):
            c = contrast(A + tag + juv, B + tag + juv)
            if c:
                xs.append(i + off); ys.append(c[0]); lo.append(c[0] - c[1]); hi.append(c[2] - c[0])
        if xs:
            a.errorbar(xs, ys, yerr=[lo, hi], color=color, ls="none", marker="o", ms=5, capsize=2,
                       mfc="white" if not juv else color)
a.axhspan(-0.02, 0.02, color="#E8E6E0", zorder=0); a.axhline(0, color=GRAY, lw=0.6)
a.set_xticks(range(3)); a.set_xticklabels([t for t, _ in TAGS]); a.set_xlabel("steps between changes of a machine")
a.set_ylabel("inherited minus erased, newborn reward"); a.set_title("(a) Inheritance pays with a juvenile period", loc="left")
from matplotlib.lines import Line2D
hand = [Line2D([], [], color=GRAY, marker="o", ls="none", ms=5, mfc="white", label="no juvenile period"),
        Line2D([], [], color=GRAY, marker="o", ls="none", ms=5, label="juvenile period, 50 updates")]
a.legend(handles=hand, frameon=False, fontsize=7, loc="upper left")
a.set_ylim(-0.05, 0.62)
# (b) newborns that rarely write: the keep prior (facts never change)
KP = pd.read_csv(os.path.join(DATA, "batch7", "keep_prior.csv"))
a = ax[1]
KREC = (("notebook, fresh heir", NOTE, -0.08), ("notebook, heir keeps features", ALT, 0.0), ("mass die-off pages", WIKI, 0.08))
KLEV = ["0.90", "0.97", "0.99", "0.90 + juvenile period 50"]
for rec, color, off in KREC:
    xs, ys, lo, hi = [], [], [], []
    for i, lev in enumerate(KLEV):
        r = KP[(KP["record"] == rec) & (KP["keep prior"].astype(str) == lev)]
        if len(r):
            l, h = [float(x) for x in r.iloc[0]["95% CI"].strip("[]").split(",")]
            xs.append(i + off); ys.append(float(r.iloc[0]["difference"])); lo.append(ys[-1] - l); hi.append(h - ys[-1])
    a.errorbar(xs[:3], ys[:3], yerr=[lo[:3], hi[:3]], color=color, marker="o", ms=5, capsize=2, mfc="white", lw=1.2,
               label={"mass die-off pages": "pages after a die-off", "notebook, fresh heir": "notebook, fresh network", "notebook, heir keeps features": "notebook, kept features"}.get(rec, rec))
    a.errorbar(xs[3:], ys[3:], yerr=[lo[3:], hi[3:]], color=color, marker="o", ms=5, capsize=2, ls="none")
a.axhspan(-0.02, 0.02, color="#E8E6E0", zorder=0); a.axhline(0, color=GRAY, lw=0.6)
a.axvline(2.5, color=GRAY, lw=0.6, ls=":")
a.set_xticks(range(4)); a.set_xticklabels(["0.90", "0.97", "0.99", "0.90 +\njuvenile 50"])
a.set_xlabel("newborns' initial probability of keeping an entry")
a.set_ylabel("inherited minus erased, newborn reward"); a.set_title("(b) Newborns that rarely write, facts never change", loc="left")
a.legend(frameon=False, fontsize=7, loc="upper left")
a.set_ylim(-0.05, 0.62)
a = ax[2]
J = [0, 10, 25, 50, 100]
sfx = lambda j: "" if j == 0 else ("_juv" if j == 50 else f"_juv{j}")
for label, kept, erased, color in (("pages after a mass die-off", "sync_cited_static", "sync_wipe_cited_static", WIKI),
                                   ("private notebook", "notebook_cited_static", "notebook_wiped_cited_static", NOTE)):
    for name, ls, mfc, tag in ((kept, "-", color, "inherited"), (erased, "--", "white", "erased")):
        m = [ci(name + sfx(j), "newborn reward") for j in J]
        a.errorbar(J, [x for x, _ in m], yerr=[e for _, e in m], color=color, ls=ls, marker="o", ms=4, mfc=mfc,
                   capsize=2, label=f"{label}, {tag}")
for ref, label in (("oracle", "perfect record"), ("no_wiki", "no record")):
    y = val(ref, "newborn reward")[0]
    a.axhline(y, color=GRAY, ls=":", lw=0.8); a.text(100, y + 0.008, label, ha="right", va="bottom", fontsize=7, color=GRAY)
a.set_xticks(J); a.set_ylim(0.2, 0.85); a.set_xlabel("juvenile period (updates without writing)")
a.set_ylabel("newborn reward per pull"); a.set_title("(c) Longer juvenile period, facts never change", loc="left")
a.legend(frameon=False, fontsize=6.3, loc="center right", bbox_to_anchor=(1.0, 0.6))
a = ax[3]
# (d) value of the inheritance against the juvenile period, for heirs that learn to read at different speeds
TF = pd.read_csv(os.path.join(DATA, "batch10", "timescale_forward.csv"))
for cond, color, lab in (("learning rate x2", "#26215C", "faster learning (18)"),
                         ("default (paper)", NOTE, "default (41)"),
                         ("learning rate x0.5", "#AFA9EC", "slower learning (150)"),
                         ("facts change every 8,000 steps", ALT, "facts every 8,000 steps (36)")):
    d = TF[TF["condition"] == cond].sort_values("juvenile period")
    a.plot(d["juvenile period"], d["gain"], color=color, marker="o", ms=3.5, lw=1.3, label=lab,
           ls="--" if cond.startswith("facts") else "-")
a.axhline(0, color=GRAY, lw=0.6)
a.set_xticks([0, 10, 25, 50, 100]); a.set_xlim(-4, 104); a.set_ylim(-0.03, 0.34)
a.set_xlabel("juvenile period (updates)"); a.set_ylabel("newborn reward, inherited − erased")
a.set_title("(d) Faster learners gain more", loc="left")
a.legend(frameon=False, fontsize=6.2, loc="lower right", bbox_to_anchor=(1.10, 0.25), handlelength=1.3)
save(fig, "fig4_inheritance")

# Supplementary Figure S2: a bottleneck halfway through the run
fig, a = plt.subplots(figsize=(3.6, 2.9))
surv = [(0, "bottleneck16_cited"), (1, "bottleneck15_cited"), (4, "bottleneck12_cited"), (8, "bottleneck8_cited")]
a.plot([s for s, _ in surv], [EV.loc[n, "cohort newborn reward"] for _, n in surv], color=WIKI, marker="o", ms=4, label="pages kept")
a.plot([0], [EV.loc["bottleneck16_wipe_cited", "cohort newborn reward"]], color=ALT, marker="s", ms=5, ls="none", label="pages erased")
for ref, label in (("individual_cited", "staggered deaths"), ("no_wiki", "no record")):
    y = val(ref, "newborn reward")[0]
    a.axhline(y, color=GRAY, ls=":", lw=0.8); a.text(10.8, y + 0.008, label, ha="right", va="bottom", fontsize=7, color=GRAY)
a.set_xlim(-0.5, 11); a.set_ylim(0.25, 0.8); a.set_xticks([0, 1, 4, 8])
a.set_xlabel("experienced survivors (of 16)"); a.set_ylabel("newborn reward of the new agents")
a.legend(frameon=False, loc="center right")
save(fig, "figS_bottleneck")

# Figure 7: what keeps the record honest, and whether it depends on history
TH = pd.read_csv(os.path.join(DATA, "honesty_threshold.csv"))
HY = pd.read_csv(os.path.join(DATA, "batch7", "hysteresis.csv"))
HZ = np.load(os.path.join(DATA, "history_series.npz"))   # smoothed time courses saved by the notebook (section 22)
AH = pd.read_csv(os.path.join(DATA, "adult_history.csv"))


def courses(name, kind="all"):
    # every seed's (x, y) for one setting; kind is "all", "adults" or "follows"
    out, j = [], 0
    while f"{name}__{j}__x" in HZ.files:
        out.append((HZ[f"{name}__{j}__x"], HZ[f"{name}__{j}__{kind}"]))
        j += 1
    return out


def hname(pages, c, start):
    return f"hyst_{pages}_c{c:+g}_from_{start}"
fig = plt.figure(figsize=(7.2, 7.6))
_gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 0.8, 0.8], hspace=0.62, wspace=1.1)
ax = [fig.add_subplot(_gs[0, :3]), fig.add_subplot(_gs[0, 3:])]
a = ax[0]
for pages, color, label in (("readable", WIKI, "own page readable"), ("hidden", ALT, "own page hidden")):
    d = TH[TH["own page"] == pages].sort_values("credit c")
    lo_hi = [tuple(float(x) for x in s.strip("[]").split(",")) for s in d["95% CI"]]
    y = d["honest writes"].to_numpy()
    a.errorbar(d["credit c"], y, yerr=[[v - l for v, (l, h) in zip(y, lo_hi)], [h - v for v, (l, h) in zip(y, lo_hi)]],
               marker="o", ms=4, capsize=2, color=color, label=label)
a.axhline(0.5, color=GRAY, ls=":", lw=0.8); a.axvline(0, color=GRAY, lw=0.6)
a.set_xscale("symlog", linthresh=0.02, linscale=0.6)
a.set_xticks([-0.5, -0.1, -0.02, 0, 0.02, 0.1, 0.5]); a.set_xticklabels(["−0.5", "−0.1", "−0.02", "0", "0.02", "0.1", "0.5"])
for pages, color, mk in (("readable", WIKI, "o"), ("hidden", ALT, "s")):     # end points after a switched credit
    for start, mfc in (("honest", "white"), ("lying", "white")):
        d = HY[(HY["own page"] == pages) & (HY["start"] == start)]
        a.plot(d["credit after switch"] + (0.0015 if start == "lying" else -0.0015), d["honest writes"], ls="none",
               marker="^" if start == "honest" else "v", ms=4.5, mfc=mfc, mec=color, mew=0.9, zorder=4)
a.set_ylim(0, 1.02); a.set_xlabel("credit c (negative: rivalry)"); a.set_ylabel("honest writes")
a.set_title("(a) Honesty follows the credit", loc="left")
h, l = a.get_legend_handles_labels()
h += [Line2D([], [], color=GRAY, marker="^", ls="none", ms=4.5, mfc="white"), Line2D([], [], color=GRAY, marker="v", ls="none", ms=4.5, mfc="white")]
l += ["after an honest start", "after a lying start"]
a.legend(h, l, frameon=False, loc="lower right", fontsize=6.4, handlelength=1.5)
a = ax[1]
groups = [("no credit\nindividual", "individual", "hidden_own"), ("no credit\ncompetitive", "competitive_nocite", "hidden_own_competitive"),
          ("credit\nindividual", "individual_cited", "hidden_own_cited"), ("credit\ncompetitive", "competitive_cited", "hidden_own_competitive_cited"),
          ("credit, comp.\n4 agents", "competitive_N4", "hidden_own_competitive_cited_N4")]
for j, (lab, color) in enumerate((("own page readable", WIKI), ("own page hidden", ALT))):
    for i, g in enumerate(groups):
        m, e = ci(g[1 + j], "honest writes")
        a.bar(i + (j - 0.5) * w, m, w * 0.92, yerr=e, color=color, capsize=2, label=lab if i == 0 else None)
a.axhline(0.5, color=GRAY, ls=":", lw=0.8)
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0].replace("competitive", "compet.") for g in groups], fontsize=5.8)
a.set_ylim(0, 1.3); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a.set_ylabel("honest writes"); a.set_title("(b) Credit and one's own page", loc="left")
a.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.4, columnspacing=1.0)
# (c) honest writes over time after the credit switches from +0.5 or -0.5 to a value near the threshold
TARGETS = [("readable", -0.05), ("readable", -0.02), ("readable", 0.0), ("hidden", -0.02), ("hidden", 0.0), ("hidden", 0.02)]
CONST = {("readable", -0.05): "rival_0.05", ("readable", -0.02): "rival_0.02", ("readable", 0.0): "individual",
         ("hidden", -0.02): "hidden_own_c-0.02", ("hidden", 0.0): "hidden_own", ("hidden", 0.02): "hidden_own_c+0.02"}
for k, (pages, c) in enumerate(TARGETS):
    a = fig.add_subplot(_gs[1 + k // 3, 2 * (k % 3): 2 * (k % 3) + 2])
    for start, color in (("honest", "#1D9E75"), ("lying", ALT)):
        for i, (x, y) in enumerate(courses(hname(pages, c, start), "adults")):
            a.plot(x / 300, y, color=color, lw=0.5, alpha=0.7, label=f"{start} start" if i == 0 else None)
        for i, (x, y) in enumerate(courses(hname(pages, c, start) + "_immortal")):
            a.plot(x / 300, y, color=color, lw=0.6, ls=(0, (3, 1.5)), alpha=0.8,
                   label="never die" if (i == 0 and start == "lying") else None)
    tc = AH[(AH["own page"] == pages) & (np.isclose(AH["credit after switch"], c))]["adults, constant credit"]
    if len(tc):
        a.axhline(float(tc.iloc[0]), color="k", ls="--", lw=0.7, label="constant credit")
    a.axvline(3, color=GRAY, ls=":", lw=0.8)
    a.set_ylim(0, 1.02); a.set_xlim(0, 9); a.set_xticks([0, 3, 6, 9])
    a.set_title(f"({'cdefgh'[k]}) own page {pages}, c = {c:+g}".replace("+0 ", "0 ").replace("= +0", "= 0").replace("-", "−"),
                loc="left", fontsize=7.6)
    if k % 3 == 0:
        a.set_ylabel("honest writes\n(mortal: adults only)", fontsize=7.5)
    else:
        a.set_yticklabels([])
    if k // 3 == 1:
        a.set_xlabel("lifetimes elapsed", fontsize=8)
    if k == 0:
        _h, _l = a.get_legend_handles_labels()
_l = [{"honest start": "after an honest start", "lying start": "after a lying start", "never die": "learners that never die (dashed)",
       "constant credit": "constant credit (adults)"}.get(x, x) for x in _l]
fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=4, frameon=False, fontsize=7, handlelength=2.2)
save(fig, "fig7_honesty")

# Figure 8: how much turnover, and which side holds the lock
DOSE = pd.read_csv(os.path.join(DATA, "batch9", "turnover_dose.csv"))
LOCK = pd.read_csv(os.path.join(DATA, "batch9", "lock_mechanism.csv"))
LIVES = [(100, "100"), (300, "300"), (900, "900"), (2700, "2,700"), (None, "never")]
LIFE_COLS = ["#CECBF6", "#AFA9EC", "#7F77DD", "#534AB7", "#26215C"]
TARGETS8 = (("readable", 0.0, "own page readable, c = 0"), ("hidden", 0.02, "own page hidden, c = +0.02"))
SHORT8 = {"readable": "own page readable, c = 0", "hidden": "hidden, c = +0.02"}


def life_name(pages, c, start, life):
    if life is None:
        return hname(pages, c, start) + "_immortal"
    return hname(pages, c, start) + ("" if life == 300 else f"_life{life}")


def mean_course(name, kind="all"):
    cs = courses(name, kind)
    k = min(len(y) for _, y in cs)
    return cs[0][0][:k], np.mean([y[:k] for _, y in cs], 0)


fig = plt.figure(figsize=(7.2, 8.6))
_gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.85, 1], hspace=0.62, wspace=0.32)
for k, (pages, c, title) in enumerate(TARGETS8):
    a = fig.add_subplot(_gs[0, k])
    for (life, lab), col in zip(LIVES, LIFE_COLS):
        x, y = mean_course(life_name(pages, c, "lying", life))
        a.plot(x / 300, y, color=col, lw=1.3, label=f"lives of {lab}" if life else "never die")
    a.axvline(3, color=GRAY, ls=":", lw=0.8)
    a.set_xlim(0, 9); a.set_ylim(0, 1.02); a.set_xticks([0, 3, 6, 9]); a.set_xlabel("lifetimes of 300 updates elapsed")
    a.set_title(f"({'ab'[k]}) lying start, {SHORT8[pages]}", loc="left", fontsize=8.6)
    if k == 0:
        a.set_ylabel("honest writes")
        a.legend(frameon=False, fontsize=6.6, loc="upper left", bbox_to_anchor=(0.0, 0.93), handlelength=1.6)
xs = np.arange(len(LIVES))
for k, (col_name, ylab, title) in enumerate((("honest - lying", "honest minus lying start", "(c) Effect of history, last third"),
                                             ("recovery time (updates)", "updates after the switch", "(d) How long the lie lasted"))):
    a = fig.add_subplot(_gs[1, k])
    for j, (pages, c, lab) in enumerate(TARGETS8):
        d = DOSE[(DOSE["own page"] == pages)]
        ys, lo, hi = [], [], []
        for _, life in LIVES:
            r = d[d["lifespan"] == life]
            ci_col = "95% CI" if k == 0 else "recovery 95% CI"
            l_, h_ = (float(v) for v in r[ci_col].iloc[0].strip("[]").split(","))
            v = float(r[col_name].iloc[0])
            never = k == 1 and life == "never" and int(r["never recovered"].iloc[0]) > 0
            ys.append(np.nan if never and pages == "readable" else v); lo.append(v - l_); hi.append(h_ - v)
        color = WIKI if pages == "readable" else ALT
        a.errorbar(xs + 0.08 * (j - 0.5), ys, yerr=[lo, hi], marker="o", ms=3.5, capsize=2, color=color, lw=1.1, label=lab)
    a.set_xticks(xs); a.set_xticklabels([lab for _, lab in LIVES]); a.set_xlabel("lifespan (updates)")
    a.set_ylabel(ylab); a.set_title(title, loc="left")
    if k == 0:
        a.axhline(0, color=GRAY, lw=0.6)
        a.legend(frameon=False, fontsize=6.6, loc="upper left")
    else:
        a.text(4, 60, "readable:\nnever", ha="center", va="bottom", fontsize=6.5, color=WIKI)
PARTS8 = [("nothing", "nothing", GRAY), ("readers", "readers", "#378ADD"), ("writers", "writers", ALT),
          ("both heads", "both", WIKI), ("whole network", "whole network", "#1D9E75")]
for k, (pages, c, title) in enumerate(TARGETS8):
    a = fig.add_subplot(_gs[2, k])
    for lab, short, color in PARTS8:
        name = hname(pages, c, "lying") + "_immortal" + ("" if lab == "nothing" else
               "_reset_" + {"readers": "pull", "writers": "write", "both heads": "heads", "whole network": "all"}[lab])
        for i, (x, y) in enumerate(courses(name)):
            a.plot(x / 300, y, color=color, lw=0.4, alpha=0.35)
        x, y = mean_course(name)
        a.plot(x / 300, y, color=color, lw=1.4, label=f"{short} reset" if lab != "nothing" else "nothing reset")
    ref = float(DOSE[(DOSE["own page"] == pages) & (DOSE["lifespan"] == "300")]["lying start"].iloc[0])
    a.axhline(ref, color="k", ls="--", lw=0.7, label="mortal, lying start")
    a.axvline(3, color=GRAY, ls=":", lw=0.8)
    a.set_xlim(0, 9); a.set_ylim(0, 1.02); a.set_xticks([0, 3, 6, 9]); a.set_xlabel("lifetimes of 300 updates elapsed")
    a.set_title(f"({'ef'[k]}) never die, lying start, {SHORT8[pages]}", loc="left", fontsize=8.6)
    if k == 0:
        a.set_ylabel("honest writes")
        _h, _l = a.get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=3, frameon=False, fontsize=7, handlelength=2.0)
fig.subplots_adjust(bottom=0.1)
save(fig, "fig_turnover")

# Figure 5 (shortened paper): honesty follows weak incentives only through turnover
HZ10 = np.load(os.path.join(DATA, "batch10", "batch10_series.npz"))
def courses10(name):
    out, j = [], 0
    while f"{name}__{j}__x" in HZ10.files:
        out.append((HZ10[f"{name}__{j}__x"], HZ10[f"{name}__{j}__all"])); j += 1
    return out
def mean_of(cs):
    k = min(len(y) for _, y in cs)
    return cs[0][0][:k] / 300, np.mean([y[:k] for _, y in cs], 0)
HON, LIE, NOISE, HIDC = "#1D9E75", "#D85A30", "#2C2C2A", "#378ADD"
fig = plt.figure(figsize=(7.2, 8.4))
_gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.32)
ax = [fig.add_subplot(_gs[i, j]) for i in range(3) for j in range(2)]
a = ax[0]
for pages, color, label in (("readable", WIKI, "own page readable"), ("hidden", HIDC, "own page hidden")):
    d = TH[TH["own page"] == pages].sort_values("credit c")
    lo_hi = [tuple(float(x) for x in t.strip("[]").split(",")) for t in d["95% CI"]]
    y = d["honest writes"].to_numpy()
    a.errorbar(d["credit c"], y, yerr=[[v - l for v, (l, h) in zip(y, lo_hi)], [h - v for v, (l, h) in zip(y, lo_hi)]],
               marker="o", ms=3.5, capsize=2, color=color, label=label)
a.axhline(0.5, color=GRAY, ls=":", lw=0.8)
a.set_xscale("symlog", linthresh=0.02, linscale=0.6)
a.set_xticks([-0.5, -0.1, -0.02, 0, 0.02, 0.1, 0.5]); a.set_xticklabels(["−0.5", "−0.1", "−0.02", "0", "0.02", "0.1", "0.5"], fontsize=6.5)
for xb in (-0.02, 0.02):                      # mark where the axis changes from logarithmic to linear
    a.plot([xb], [0], marker=(2, 0, 60), ms=7, color="black", mew=0.9, clip_on=False, zorder=5, transform=a.get_xaxis_transform())
a.set_ylim(0, 1.02); a.set_xlabel("credit c (negative: rivalry)"); a.set_ylabel("honest writes")
a.set_title("(a) Honesty follows the credit", loc="left")
a.legend(frameon=False, loc="lower right", fontsize=6.4)
for a, (pages, c, title) in zip(ax[1:3], (("readable", 0.0, "(b) Own page readable, c = 0"), ("hidden", 0.02, "(c) Own page hidden, c = +0.02"))):
    for name, color, ls, lab in ((hname(pages, c, "honest"), HON, "-", "mortal, honest start"),
                                 (hname(pages, c, "lying"), LIE, "-", "mortal, lying start"),
                                 (hname(pages, c, "honest") + "_immortal", HON, "--", "never die, honest start"),
                                 (hname(pages, c, "lying") + "_immortal", LIE, "--", "never die, lying start")):
        x, y = mean_of(courses(name)); a.plot(x, y, color=color, ls=ls, lw=1.3, label=lab)
    x, y = mean_of(courses10(hname(pages, c, "lying") + "_immortal_noise0.1"))
    a.plot(x, y, color=NOISE, ls=":", lw=1.6, label="never die, lying start,\n10% random writes")
    a.axvline(3, color=GRAY, ls=":", lw=0.8); a.set_ylim(-0.02, 1.02); a.set_xlim(0, 9)
    a.set_xlabel("lifetimes"); a.set_title(title, loc="left")
ax[1].set_ylabel("honest writes")
a = ax[3]
for pages, c, color in (("readable", 0.0, WIKI), ("hidden", 0.02, HIDC)):
    x, y = mean_of(courses(hname(pages, c, "lying") + "_immortal")); a.plot(x, y, color=color, ls="--", lw=1.2)
    x, y = mean_of(courses10(f"hyst_{pages}_c+0.5_from_lying_immortal")); a.plot(x, y, color=color, lw=1.6)
    x, y = mean_of(courses10(f"hyst_{pages}_c+0.5_from_lying")); a.plot(x, y, color=color, lw=0.8, alpha=0.6)
a.axvline(3, color=GRAY, ls=":", lw=0.8); a.set_ylim(-0.02, 1.02); a.set_xlim(0, 9)
a.set_xlabel("lifetimes"); a.set_ylabel("honest writes")
a.set_title("(d) Lying start, strong credit", loc="left")
a.legend(handles=[Line2D([], [], color="black", lw=1.6, label="never die, c = +0.5"),
                  Line2D([], [], color="black", lw=1.2, ls="--", label="never die, weak credit"),
                  Line2D([], [], color="black", lw=0.8, alpha=0.6, label="mortal, c = +0.5"),
                  Line2D([], [], color=WIKI, lw=2, label="own page readable"), Line2D([], [], color=HIDC, lw=2, label="own page hidden")],
         frameon=False, fontsize=5.3, loc="center left", bbox_to_anchor=(-0.02, 0.63), handlelength=1.2, handletextpad=0.4)
a = ax[4]
for pages, c, color, lab in (("readable", 0.0, WIKI, "own page readable, c = 0"), ("hidden", 0.02, HIDC, "own page hidden, c = +0.02")):
    d = DOSE[(DOSE["own page"] == pages)]
    xs = np.arange(len(LIVES)); ys, lo, hi = [], [], []
    for _, lab_l in LIVES:
        r = d[d["lifespan"].astype(str) == lab_l].iloc[0]
        l, h = (float(v) for v in r["95% CI"].strip("[]").split(","))
        ys.append(r["honest - lying"]); lo.append(r["honest - lying"] - l); hi.append(h - r["honest - lying"])
    a.errorbar(xs, ys, yerr=[lo, hi], color=color, marker="o", ms=3.5, capsize=2, label=lab)
a.axhline(0, color=GRAY, lw=0.6); a.set_xticks(range(len(LIVES))); a.set_xticklabels([l for _, l in LIVES])
a.set_xlabel("lifespan (updates)"); a.set_ylabel("honest − lying start, last third")
a.set_title("(e) Longer lives, longer lies", loc="left"); a.legend(frameon=False, fontsize=6.2, loc="upper left")
a = ax[5]
PARTS5 = ["nothing", "readers", "writers", "both heads", "whole network"]
w5 = 0.38
for j, (pages, color, lab) in enumerate((("readable", WIKI, "own page readable, c = 0"), ("hidden", HIDC, "own page hidden, c = +0.02"))):
    d = LOCK[LOCK["own page"] == pages].set_index("re-initialised at the switch")
    ys = [d.loc[p, "honest writes"] for p in PARTS5]
    errs = [[y - float(d.loc[p, "95% CI"].strip("[]").split(",")[0]) for y, p in zip(ys, PARTS5)],
            [float(d.loc[p, "95% CI"].strip("[]").split(",")[1]) - y for y, p in zip(ys, PARTS5)]]
    a.bar(np.arange(5) + (j - 0.5) * w5, ys, w5 * 0.92, yerr=errs, color=color, capsize=2, label=lab)
a.set_xticks(range(5)); a.set_xticklabels(["nothing", "readers", "writers", "both\nheads", "whole\nnetwork"], fontsize=6.5)
a.set_ylim(0, 1.28); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); a.set_ylabel("honest writes, never die")
a.set_title("(f) Reset at the switch", loc="left"); a.legend(frameon=False, fontsize=6.0, loc="upper left", ncol=2, columnspacing=0.8)
_h, _l = ax[1].get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False, fontsize=6.6)
save(fig, "fig5_turnover_v3")

# Figure 8: liars and identity
fig, ax = plt.subplots(2, 1, figsize=(6.8, 6.8), gridspec_kw={"hspace": 0.5})
a = ax[0]
ks = [0, 2, 4, 6, 8, 10, 12]
for ident, color, base in (("named", WIKI, "individual_cited"), ("anonymous", ALT, "clean_anonymous")):
    names = [base] + [f"deceivers{k}_{ident}" for k in ks[1:]]
    m = [ci(n, "newborn reward") for n in names]
    a.errorbar(ks, [x for x, _ in m], yerr=[e for _, e in m], marker="o", ms=4, capsize=2, color=color, label=f"{ident} authors")
a.axhline(val("no_wiki", "newborn reward")[0], color=GRAY, ls=":", lw=0.8)
a.text(0, val("no_wiki", "newborn reward")[0] + 0.008, "no record", ha="left", va="bottom", fontsize=7, color=GRAY)
a.set_xticks(ks); a.set_xlabel("coordinated liars among 16 authors"); a.set_ylabel("newborn reward per pull")
a.set_ylim(0.2, 0.8); a.set_title("(a) Named authors against organised liars", loc="left"); a.legend(frameon=False, loc="upper right")
a = ax[1]
items = [("liars,\nfixed names", "deceivers8_named"), ("liars,\nrenamed\nonce a life", "deceivers8_rotate_life"),
         ("liars,\nrenamed\n10× a life", "deceivers8_rotate_short"), ("liars,\npositions differ\nby copy", "deceivers8_seats_per_copy"),
         ("liars,\nanonymous", "deceivers8_anonymous"),
         ("sleepers,\nnamed", "deceivers8_sleeper_named"), ("sleepers,\nanonymous", "deceivers8_sleeper_anonymous"),
         ("saboteurs,\nnamed", "saboteurs8_named"), ("saboteurs,\nanonymous", "saboteurs8_anonymous")]
for j, (col, color, lab) in enumerate((("newborn reward", LIGHT, "newborns"), ("adult reward", WIKI, "adults"))):
    for i, (_, n) in enumerate(items):
        m, e = ci(n, col)
        a.bar(i + (j - 0.5) * w, m, w * 0.92, yerr=e, color=color, capsize=2, label=lab if i == 0 else None)
a.set_xticks(range(len(items))); a.set_xticklabels([x for x, _ in items], fontsize=6.2, rotation=0)
a.set_ylim(0, 1.1); a.set_ylabel("reward per pull")
a.set_title("(b) Eight bad authors among 16", loc="left")
a.legend(frameon=False, loc="upper right", ncol=2)
save(fig, "fig8_identity")
print("wrote", sorted(os.listdir(OUT)))

# Figure 9: free symbols never become a working code
fig, ax = plt.subplots(2, 1, figsize=(6.6, 6.4), gridspec_kw={"hspace": 0.5})
conds = [("built-in\nreports, credit", "individual_cited"), ("free\nsymbols", "symbol"), ("free symbols,\ncredit", "symbol_cited"),
         ("free symbols,\n18 lifetimes", "symbol_long"), ("free symbols,\ncredit,\n18 lifetimes", "symbol_cited_long")]
for a, pairs, ylabel, title in (
        (ax[0], (("wiki right", WIKI, "wiki majority names the good arm"), ("follows", LIGHT, "pulls on the majority's arm")),
         "share", "(a) A code emerges only partly, and only with credit"),
        (ax[1], (("adult reward", WIKI, "adults"), ("newborn reward", LIGHT, "newborns")),
         "reward per pull", "(b) The partial code does not pay")):
    for j, (col, color, lab) in enumerate(pairs):
        for i, (_, n) in enumerate(conds):
            m, e = ci(n, col)
            a.bar(i + (j - 0.5) * w, m, w * 0.92, yerr=e, color=color, capsize=2, label=lab if i == 0 else None)
    a.set_xticks(range(len(conds))); a.set_xticklabels([c for c, _ in conds], fontsize=7)
    a.set_ylim(0, 1.15); a.set_ylabel(ylabel); a.set_title(title, loc="left"); a.legend(frameon=False, loc="upper right", fontsize=7.5)
ax[0].axhline(0.2, color=GRAY, ls=":", lw=0.8); ax[0].text(2.5, 0.215, "chance", ha="center", fontsize=7, color=GRAY)
for ref, label in (("no_wiki", "adults, no record"),):
    y = val(ref, "adult reward")[0]
    ax[1].axhline(y, color=GRAY, ls=":", lw=0.8); ax[1].text(2.5, y + 0.015, label, ha="center", fontsize=7, color=GRAY)
save(fig, "fig9_symbols")
print("wrote fig9_symbols")
