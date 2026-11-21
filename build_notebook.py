"""Assemble WikiBandits.ipynb. python build_notebook.py [--quick]"""
import re
import sys

import nbformat as nbf

QUICK = "--quick" in sys.argv

INTRO = r"""# WikiBandits: do mortal agents keep honest wikis for each other?

Sixteen learning agents play a set of slot machines ("bandits"). On each machine
one arm is good and the rest are bad; which one is hidden, and it changes now and
then. Each agent owns a **public wiki page** with one entry per machine, and while
playing a machine it reads **everyone's entry for that machine**. It can keep its
own entry, report what it believes, or report something false.

**When an agent dies, its neural network is reset to random**; only the wikis
survive, and the newborn inherits its predecessor's page.

The questions: how much faster do newborns become good players thanks to the
wikis? Under which incentives do wikis stay honest, and when do they fill with
lies?

**How to run it**: upload this notebook to Google Colab
(colab.research.google.com → File → Upload notebook), choose *Runtime → Change
runtime type → T4 GPU*, then *Runtime → Run all*. Results are saved to your
Google Drive after each run; if Colab disconnects, *Run all* again and it resumes.

**On your own Windows PC with an NVIDIA GPU.** JAX uses the GPU only from Linux,
which Windows provides as WSL2. Put `GPU_on_Windows.ipynb` in this folder, open it
in your usual Jupyter and *Run all*: it installs what's needed (the first time) and
opens this notebook in a Jupyter running on your GPU. Results are saved in a
`wikibandits_results` folder next to this file. On Linux, install
`jax[cuda13]` (or `jax[cuda12]` for NVIDIA drivers older than 580) and open this
file in Jupyter.

**This version is set up for the paper runs (sections 13–14).** The exploratory
grids in sections 8 and 11 are switched off (`RUN_MAIN`, `RUN_ADV`), so *Run all*
goes straight to the clean GPU runs. **Check that section 1 says `GPU`**; the
paper runs refuse to start on a CPU.

Sections: 1 setup · 2 the world · 3 the learners · 4 running many generations ·
5 the baseline ladder · 6 analysis tools · 7 quick demo · 8 the experiment ·
9 comparing settings · 11–12 deceivers, incompetents and anonymity ·
10 how to read the results (at the end).
"""

PILOT_MD = r"""### What the CPU pilots showed (8 Nov 2026, one seed, no deaths)

Defaults: 16 agents, 8 machines with 5 arms (good arm pays 90% of the time, bad
arms 10%), visits of 8 pulls, arms drifting every ~2,000 steps. Reward per pull:

| | Reward per pull |
|---|---|
| random | 0.26 |
| learners without a wiki | 0.36 |
| smart lone player, no memory (Thompson sampling; see the note below) | 0.40 |
| learners with a perfect wiki (oracle) | 0.90 (reached in ~100 updates) |
| learners writing freely (any arm number), any incentive | ≈ random: writing dies out |
| learners reporting (keep / honest / false), no citation credit | ≈ random: writing dies out |
| **learners reporting + citation credit** | **0.41–0.49**, entries correct 45–64% (chance 20%) |

So: reading a good wiki is easy to learn; **writing one is not**, unless authors
earn credit when others use their entries successfully (like karma or being
cited). Even then writing slowly fades in the pilots.

**Correction (later the same day).** Those pilots, and sections 8 and 11, ran
*without reward centering*, which hampers learning with a long horizon. With
centering (section 13), learners write a useful wiki even without citation;
citation's effect is to make it fully honest. Rivalry over information produces
a wiki of lies that readers learn to read in reverse. Sections 8 and 11 are kept
as an exploratory record; **sections 13–14 are the clean runs.**

**Second correction (first GPU runs).** The "smart lone player" above didn't know the
rules of this world (one good arm per machine), so learners beat it. Section 5 now
uses the best possible strategies: without memory 0.61, with perfect own memory
0.87, with perfectly shared memory 0.90.
"""

SETUP = r"""# 1. Setup: checks that JAX is available and whether an accelerator is in use.
import importlib, subprocess, sys, os
for pkg in ("jax", "optax", "matplotlib", "pandas"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
import jax
print("JAX", jax.__version__, "| devices:", jax.devices())
ON_GPU = jax.default_backend() in ("gpu", "tpu")
print("Accelerator in use:", jax.default_backend().upper() if ON_GPU else "none (CPU)")

RESULTS_DIR = "wikibandits_results"
try:
    from google.colab import drive  # only exists on Colab
    drive.mount("/content/drive")
    RESULTS_DIR = "/content/drive/MyDrive/wikibandits_results"
except ImportError:
    pass
os.makedirs(RESULTS_DIR, exist_ok=True)
print("results go to", RESULTS_DIR)
"""

LADDER_CODE = r"""LADDER = simulate_baselines(**{k: v for k, v in BanditConfig()._asdict().items()
                               if k in ("n_agents", "n_bandits", "n_arms", "p_high", "p_low", "visit_len", "drift_period")},
                            steps=STEPS_LADDER)
for name, v in LADDER.items():
    print(f"{name:>22}: {v:.3f} reward per pull")
"""

DEMO_MD = r"""## 7. Quick demo

Skipped unless you set `RUN_DEMO = True`. One world with individual rewards plus citation credit, the combination that
worked in the pilots, for a few generations. A couple of minutes on a GPU.
"""

DEMO = r"""RUN_DEMO = False   # set to True to run the demo
if RUN_DEMO:
    demo_world = BanditConfig(reward_mode="individual", citation=0.5)
    demo_learn = LearnConfig(num_envs=256 if ON_GPU else 64)
    demo_log, demo_runner = run_world(demo_world, demo_learn, life_updates=DEMO_LIFE, generations=DEMO_GENS, seed=0)
    plot_run(demo_log, "demo: individual + citation credit", LADDER)
    print("wiki use test (reward per pull):", evaluate_wiki_use(demo_world, demo_learn, demo_runner))
"""

EXP_MD = r"""## 8. The experiment

Nine settings, several seeds each. All use *report* writing (keep / honest /
false) except the last, which lets agents write any arm number.

| Setting | Reward | Citation credit | Wiki |
|---|---|---|---|
| no_wiki | own | – | none (control) |
| oracle | own | – | always correct, agents can't write (upper bound) |
| individual / cooperative / competitive | own / own + others' average / own − others' average | no | learned |
| …_cited | same three | yes (0.5 per successful use) | learned |
| symbol_cited | own | yes | learned, free symbols |

Results are saved after each run; rerunning the cell resumes.

**`QUICK_CHECK = True` (default)** runs one seed, 3 generations and only the
five settings that answer the main questions: `no_wiki`, `oracle`, `individual`,
`individual_cited` and `competitive_cited`. In section 11 it also runs only
4 bad agents. That's about 9 runs. Set it to False for the full grid
(9 settings × 3 seeds here, plus 13 × 3 in section 11).
"""

EXP = r"""SETTINGS = {
    "no_wiki":           dict(wiki=False),
    "oracle":            dict(oracle=True),
    "individual":        dict(reward_mode="individual"),
    "cooperative":       dict(reward_mode="cooperative"),
    "competitive":       dict(reward_mode="competitive"),
    "individual_cited":  dict(reward_mode="individual", citation=0.5),
    "cooperative_cited": dict(reward_mode="cooperative", citation=0.5),
    "competitive_cited": dict(reward_mode="competitive", citation=0.5),
    "symbol_cited":      dict(reward_mode="individual", citation=0.5, write_mode="symbol"),
}
SEEDS = SEEDS_
LIFE_UPDATES = LIFE_     # learning updates per lifetime
GENERATIONS = GENS_      # lifetimes per agent
NUM_ENVS = 256 if ON_GPU else 64

QUICK_CHECK = True   # one seed, the key settings, shorter runs; set False for the full grid
if QUICK_CHECK:
    SEEDS = [0]
    GENERATIONS = min(GENERATIONS, 3)
    SETTINGS = {k: SETTINGS[k] for k in ("no_wiki", "oracle", "individual", "individual_cited", "competitive_cited")}

import pickle

def run_setting(name, seed, settings=None):
    settings = SETTINGS if settings is None else settings
    path = os.path.join(RESULTS_DIR, f"{name}_s{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    wc, lc = BanditConfig(**settings[name]), LearnConfig(num_envs=NUM_ENVS)
    log, runner = run_world(wc, lc, LIFE_UPDATES, GENERATIONS, seed, log_every=100)
    log["wiki_use"] = evaluate_wiki_use(wc, lc, runner) if (wc.wiki and not wc.oracle) else None
    with open(path, "wb") as f:
        pickle.dump(log, f)
    return log

RUN_MAIN = False   # True: run (or reload) this exploratory grid; the paper runs are in section 13
RESULTS = {}
if RUN_MAIN:
    for seed in SEEDS:
        for name in SETTINGS:
            print(f"=== {name}, seed {seed}")
            RESULTS[(name, seed)] = run_setting(name, seed)
"""

COMPARE_MD = r"""## 9. Comparing settings

Averages over seeds (± standard error across seeds). "Late" means the last third
of the run.

- **newborn reward**: mean reward per pull in the first 50 updates of each newborn's life.
- **adult reward**: mean reward per pull in the last 50 updates of complete lives.
- **wiki right**: share of machines whose entries' majority names the good arm (late).
- **correct / honest writes**: share of writes naming the good arm / the writer's own belief (late).
- **writes**: writes per pull (late). **follows**: share of pulls on the others' majority arm (late).
- **real − scrambled**: how much the final agents lose when shown wikis from another world copy.
"""

COMPARE = r"""import pandas as pd

def summarise(results):
    rows = []
    for name in SETTINGS:
        logs = [results[(n, s)] for n, s in results if n == name]
        if not logs:
            continue
        def ms(vals, d=3):
            vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
            if len(vals) == 0:
                return "-"
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            return f"{vals.mean():.{d}f} ± {se:.{d}f}"
        early = min(50, LIFE_UPDATES // 4)
        has_wiki = name not in ("no_wiki", "oracle")
        wk = (lambda key, d=2: ms([late(l, key) for l in logs], d)) if has_wiki else (lambda key, d=2: "-")
        rows.append({
            "setting": name, "seeds": len(logs),
            "newborn reward": ms([newborn_reward(l, early) for l in logs]),
            "adult reward": ms([adult_reward(l, early) for l in logs]),
            "wiki right": wk("majority_ok"),
            "correct writes": wk("correct_writes"),
            "honest writes": wk("honest_writes"),
            "writes": wk("write_rate", 4),
            "follows": ms([late(l, "follow_rate") for l in logs], 2),
            "real − scrambled": ms([l["wiki_use"]["real"] - l["wiki_use"]["scrambled"] if l.get("wiki_use") else np.nan for l in logs]),
        })
    return pd.DataFrame(rows).set_index("setting")

if not RESULTS:
    print("Section 8 was skipped (RUN_MAIN = False).")
else:
  table = summarise(RESULTS)
  try:
      display(table)
  except NameError:
      print(table.to_string())

  names = [n for n in SETTINGS if any(k[0] == n for k in RESULTS)]
  fig, ax = plt.subplots(figsize=(12, 3.8))
  vals = [[adult_reward(RESULTS[k], min(50, LIFE_UPDATES // 4)) for k in RESULTS if k[0] == n] for n in names]
  ax.bar(range(len(names)), [np.nanmean(v) for v in vals], color="#7F77DD")
  for i, v in enumerate(vals):
      ax.scatter([i] * len(v), v, color="k", s=10, zorder=3)
  for name, v in LADDER.items():
      ax.axhline(v, lw=0.8, ls=":", color="gray")
      ax.text(len(names) - 0.4, v, f" {name}", fontsize=8, va="center", color="gray")
  ax.set_xticks(range(len(names)))
  ax.set_xticklabels(names, rotation=20, ha="right")
  ax.set_ylabel("adult reward per pull")
  ax.set_title("where each setting lands on the baseline ladder (dots: seeds)")
  ax.spines[["top", "right"]].set_visible(False)
  fig.tight_layout()

  for (name, seed), l in RESULTS.items():
      if seed == SEEDS[0] and name not in ("no_wiki", "oracle"):
          plot_run(l, f"{name}, seed {seed}", LADDER)
"""


ADV_MD = r"""## 11. Deceivers, incompetents and anonymity

What if some authors lie or simply don't know? Here some slots are taken by
**scripted bad authors** who play normally but, at the end of every visit, write a
bad entry, and stay bad across generations:

- **deceivers** name a bad arm, all naming the *same* bad arm on a machine, so
  together they can outvote honest authors;
- **incompetents** name a random arm.

**Identity** decides whether readers can learn whom to trust:
- **named**: authors always appear in the same order;
- **anonymous**: each reader sees its own entry first and the others in a fresh
  random order every step (like sockpuppets or sybils).

Everything else is the `individual_cited` setting. The grid is 2 types × 3 shares
(2, 4 or 8 of 16 agents) × 2 identity modes, plus clean named and clean anonymous
controls. The clean named control reuses `individual_cited` from section 8.

CPU pilot (one seed, 4 deceivers, no deaths, 500 updates):
- **named:** learners learned to avoid the deceivers (agreement with their entries
  0.08, below chance) and still reached 0.47 reward per pull;
- **anonymous:** learners followed deceivers as much as anyone and sank to 0.34,
  *below* learners with no wiki at all (0.36).
"""

ADV_EXP = r"""BASE = dict(reward_mode="individual", citation=0.5)
ADV_SETTINGS = {"clean_anonymous": dict(BASE, identity="anonymous")}
ADV_SHARES = (4,) if QUICK_CHECK else (2, 4, 8)
for kind in ("deceivers", "incompetent"):
    for n_bad in ADV_SHARES:
        for ident in ("named", "anonymous"):
            key = "n_deceivers" if kind == "deceivers" else "n_incompetent"
            ADV_SETTINGS[f"{kind}{n_bad}_{ident}"] = dict(BASE, identity=ident, **{key: n_bad})
ADV_SEEDS = SEEDS   # reduce (e.g. [0, 1]) to save time: 13 runs per seed

RUN_ADV = False   # True: run (or reload) this exploratory grid; the paper runs are in section 13
ADV = {}
for seed in (ADV_SEEDS if RUN_ADV else []):
    ADV[("clean_named", seed)] = RESULTS.get(("individual_cited", seed)) or run_setting("individual_cited", seed)
    for name in ADV_SETTINGS:
        print(f"=== {name}, seed {seed}")
        ADV[(name, seed)] = run_setting(name, seed, ADV_SETTINGS)
"""

ADV_CMP_MD = r"""## 12. Who keeps the wiki useful?

Numbers are for the **learners only** (scripted slots excluded), averaged over
seeds:

- **adult reward** per pull;
- **agrees w/ bad**: share of learners' pulls that match a bad author's entry
  when one is present. Chance is about 0.2; below chance means learners actively
  avoid bad authors;
- **agrees w/ learners**: the same for entries by other learners;
- **wiki right**: share of machines whose majority entry is right;
- **honest writes**: learners' writes that match their own belief.

The plots show learners' reward and their agreement with bad authors as the
number of bad agents grows. The dotted line is learners without any wiki
(`no_wiki`, section 8), so a point below it means the wiki has become a liability.
"""

ADV_CMP = r"""def adv_table(adv):
    rows = []
    for name in ["clean_named"] + list(ADV_SETTINGS):
        logs = [adv[(n, s)] for n, s in adv if n == name]
        if not logs:
            continue
        def ms(vals, d=3):
            vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
            if len(vals) == 0:
                return "-"
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            return f"{vals.mean():.{d}f} ± {se:.{d}f}"
        early = min(50, LIFE_UPDATES // 4)
        rows.append({"setting": name, "seeds": len(logs),
                     "adult reward": ms([adult_reward(l, early) for l in logs]),
                     "newborn reward": ms([newborn_reward(l, early) for l in logs]),
                     "agrees w/ bad": ms([late(l, "agree_bad") for l in logs], 2),
                     "agrees w/ learners": ms([late(l, "agree_good") for l in logs], 2),
                     "wiki right": ms([late(l, "majority_ok") for l in logs], 2),
                     "honest writes": ms([late(l, "honest_writes") for l in logs], 2)})
    return pd.DataFrame(rows).set_index("setting")

atable = adv_table(ADV) if ADV else None
if atable is None:
    print("Section 11 was skipped (RUN_ADV = False).")
else:
    try:
        display(atable)
    except NameError:
        print(atable.to_string())

early = min(50, LIFE_UPDATES // 4)
def mean_over_seeds(name, fn):
    v = [fn(ADV[k]) for k in ADV if k[0] == name]
    v = [x for x in v if np.isfinite(x)]
    return np.mean(v) if v else np.nan

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8)) if ADV else (None, [])
for kind, ls in ((("deceivers", "-"), ("incompetent", "--")) if ADV else ()):
    for ident, color in (("named", "#534AB7"), ("anonymous", "#D85A30")):
        clean = "clean_named" if ident == "named" else "clean_anonymous"
        xs = [0, 2, 4, 8]
        names = [clean] + [f"{kind}{n}_{ident}" for n in (2, 4, 8)]
        ax[0].plot(xs, [mean_over_seeds(n, lambda l: adult_reward(l, early)) for n in names], ls=ls, color=color,
                   marker="o", label=f"{kind}, {ident}")
        ax[1].plot(xs[1:], [mean_over_seeds(n, lambda l: late(l, "agree_bad")) for n in names[1:]], ls=ls,
                   color=color, marker="o", label=f"{kind}, {ident}")
nowiki = [adult_reward(l, early) for (n, s), l in RESULTS.items() if n == "no_wiki"]
if nowiki and ADV:
    ax[0].axhline(np.mean(nowiki), color="gray", ls=":", lw=1)
    ax[0].text(8, np.mean(nowiki), " no wiki", fontsize=8, color="gray", va="center")
if ADV:
    ax[1].axhline(1 / BanditConfig().n_arms, color="gray", ls=":", lw=1)
    ax[1].text(8, 1 / BanditConfig().n_arms, " chance", fontsize=8, color="gray", va="center")
    ax[0].set_title("learners' adult reward per pull")
    ax[1].set_title("how often learners agree with bad authors")
for a in ax:
    a.set_xlabel("number of bad agents (of 16)")
    a.legend(fontsize=8, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
if ADV:
    fig.tight_layout()

for name in ("deceivers4_named", "deceivers4_anonymous"):
    if (name, ADV_SEEDS[0]) in ADV:
        plot_run(ADV[(name, ADV_SEEDS[0])], f"{name}, seed {ADV_SEEDS[0]}", LADDER)
"""


PAPER_MD = r"""## 13. Paper runs: clean, GPU only

The runs behind the paper's claims, done cleanly: all on the GPU with the same
batch size (256 world copies), 3 seeds, 300-update lives × 6 generations,
**reward centering on**, saved to a separate Drive folder
(`wikibandits_results/paper`) so nothing exploratory gets mixed in. The cell
refuses to run on a CPU.

**Why reward centering.** The shuffled control showed that the earlier
"competitive" advantage was arithmetic, not rivalry. Subtracting *any*
average helped because, with a long horizon (γ = 0.99), uncentred rewards make
value targets large and slow learning down. Centering every agent's reward
removes that handicap in all settings, so they can be compared fairly. CPU pilots
with centering (one seed, no deaths, 400 updates) changed the picture:

| | reward | honest writes | wiki right |
|---|---|---|---|
| no wiki | 0.53 | – | – |
| individual (no citation) | **0.87** | 78% | 95% |
| individual + citation | **0.87** | **100%** | 96% |
| informational rivalry (−0.5 when a rival uses your entry) | 0.56 | **0%** | 1% (readers follow the majority 6% of the time: they read the lies in reverse) |
| free symbols (± citation) | 0.36–0.41 | ≈ chance | ≈ chance |
| 8 of 16 deceivers, named / anonymous | 0.87 / 0.76 | | |

**Tier 1 (default, 11 settings × 3 seeds)**, one row per claim:

| Setting | Question |
|---|---|
| no_wiki, oracle | floor and ceiling |
| individual, individual_cited | does a wiki work without any incentive, and what does citation add (honesty)? |
| symbol, symbol_cited | can agents invent what symbols mean? |
| **rival_info** | if others' use of your entry costs you, do agents lie, and do readers adapt? |
| **wipe_cited** | does the wiki survive because pages outlive their authors? (pages erased at death) |
| clean_anonymous, deceivers8_named, deceivers8_anonymous | does stable identity protect against many liars? |

**Tier 2 (`RUN_TIER2`, on by default, 10 more):** competitive and shuffled controls,
**blame** (authors lose credit when a follower loses, also combined with
rivalry), **noisier machines** (0.8 vs 0.2), 4 deceivers, and 8 incompetents.

**Added after the first GPU runs (always on):**

| Setting | Question |
|---|---|
| notebook_cited | agents read **only their own page** (inherited by the next holder of the slot): how much of the wiki's value is just private memory plus inheritance? |
| notebook_wiped_cited | own page only, erased at death: pure individual memory |
| symbol_long, symbol_cited_long | free symbols over 3× more generations: does a convention finish forming across generations? |

**Robustness and mechanisms (`RUN_ROBUST`, on by default, 36 more; analysed in section 15):**

| Settings | Question |
|---|---|
| commons1_cited, commons3_cited, notebook_replicated_cited | why are others' pages so much easier for newborns than their own: is it how many entries they see, or that others keep them current? |
| no_wiki_torso, notebook_torso_cited, wiki_torso_cited | does an inherited notebook help an heir that keeps its predecessor's learned features? |
| saboteurs8_named / anonymous | liars that learn: 8 agents rewarded when the honest learners lose |
| deceivers2/6/10/12, named / anonymous | how many coordinated liars it takes before anonymity fails |
| no_wiki_N4/8/32, cited_N4/8/32, competitive_N4/8/32 | does competition corrupt small groups, where one liar can sway the majority? |
| cited_0.1/0.25/1.0, rival_0.1/0.25 | how honesty depends on the size of the credit or the rivalry |
| no_wiki / notebook / cited, at 1/3 and 3× the lifespan | does the newcomer advantage depend on how long agents live? |

**Second round (`RUN_EXTRA2`, on by default, 34 more; analysed in section 16):**

| Settings | Question |
|---|---|
| notebook_wiped_torso_cited | heirs keep their predecessor's hidden layers but the notebook is erased: is inheritance the features, the notebook, or both? |
| hidden_own, hidden_own_cited, hidden_own_competitive(_cited), competitive_nocite, …_N4 | authors cannot read their own page: is honesty under competition explained by the own page being the author's own memory? |
| sync_no_wiki, sync_notebook_cited, sync_cited, sync_wipe_cited | mass die-off: everyone dies at the same moment, so no experienced agent is left to keep the record current |
| deceivers8_rotate_life, deceivers8_rotate_short | 8 liars whose names are reshuffled once or ten times a lifetime |
| deceivers8_sleeper_named / anonymous | 8 liars who tell the truth for the first half of the run, then lie |
| no_wiki / cited × drift 500 / 8000 × three lifespans | when is a record worth keeping: how fast the world changes against how long agents live |
| no_wiki / notebook_cited / cited × 4 and 32 machines | smaller and bigger worlds |

**Third round, for the journal version (`RUN_EXTRA3`, on by default, 29 more; analysed in section 17).**
Predictions were written down before these runs (project note "ALife journal plan").

| Settings | Question |
|---|---|
| notebook, wiped notebook, heirs with kept features, die-offs with pages kept or erased, each with drift every 8,000 steps or never (`_drift8000`, `_static`) | do inherited notes fail only because they go stale before a newborn can read them? |
| cited_0.02 / 0.05, rival_0.02 / 0.05, hidden_own_c±0.02 / ±0.05 / ±0.1 | where exactly does honesty switch, with own pages readable or hidden? |
| bottleneck8 / 12 / 15 / 16_cited, bottleneck16_wipe_cited | a single bottleneck halfway through the run: how many experienced survivors does the record need? |

**Juvenile period (`RUN_EXTRA4`, on by default, 11 more, about 25 minutes).** Model v1 says inherited notes are
worthless because a newborn overwrites them (it writes on about 10% of steps, half of them false) long before it
can read them. Here newborns cannot write during their first 50 updates (`_juv`). If the model is right,
inherited notes should now help, most when facts never change.

**Evolution of the juvenile period (`RUN_EXTRA5`, on by default, 24 more, about 1.3 hours; analysed in section 18).**
The juvenile period becomes heritable: each agent in each world copy has its own, and heirs copy that of a
fitter copy, with mutation. Six settings (private notebooks with facts that change or not, heirs that keep
their features, a shared wiki with staggered deaths, mass die-offs with individual or group selection), each
started with no juvenile period and with 150 updates, for 12 lifetimes. Plus fixed juvenile periods of 10, 25
and 100 updates for the inheritance comparisons when facts never change.

**Longer evolution (`RUN_EXTRA6`, on by default, 8 more, about 1.5 hours; section 18, Fig. 13).** 36 lifetimes
instead of 12, for notebooks with facts changing every 2,000 or 8,000 steps, with fresh heirs or heirs that keep
their features, each from no juvenile period and from 150 updates: do both starts meet at the same value?

**The reviewer's tests (`RUN_EXTRA7`, on by default, 41 more, about 1.7 hours; section 19, Figs. 14–15).**
(K) newborns that start keeping their entry with probability 0.97 or 0.99 instead of 0.9, for the inheritance
pairs when facts never change and for the shared wiki; (F) juvenile periods fixed at 10, 25, 50 or 100 updates
in the four 36-lifetime settings, to find the best fixed value; (H) the credit set to +0.5 or −0.5 for 3
lifetimes and then switched to a value near the honesty threshold for 6 more: do both starts end in the same
place?

**Second round of review tests (`RUN_EXTRA8`, on by default, 34 more, about 1.8 hours; section 20, Figs. 16–17).**
(R) liars in seats drawn separately for each world copy, so readers cannot store their trust in weights shared
by all copies; (I) the credit switch of section 19 with learners that never die; (G) the juvenile period in the
shared wiki under group selection, 36 lifetimes; (F) a finer and wider grid of fixed juvenile periods; (M) agents
with a memory of their own past visits.

**Turnover and the lock (`RUN_EXTRA9`, on by default, 20 more, about 0.9 hours for seeds 0–2; section 22,
Figs. 19–20).** (D) the credit switch of section 19 with lives of 100, 900 or 2,700 updates instead of 300, in runs of
the same length (2,700 updates); (X) learners that never die, starting lying, whose readers (pull heads), writers
(write heads), both heads, or whole networks are re-initialised once at the switch.

**The fourth review's tests (`RUN_EXTRA10`, on by default, 53 more, about 2.2 hours for seeds 0–2; section 23,
Figs. 21–23).** (N) learners that never die with random writes at 2% or 10% of steps; (S) a lying start switched to
strong credit (+0.5); (J) the credit switch with juvenile periods of 50 and 150 updates; (R) rivalry over any reader's
win; (T) the timescale rule tested forward, with half or double the learning rate and with slow drift; (C) 64 copies
of the world instead of 256.

**Corrections after the independent review (`RUN_EXTRA11`, on by default, 14 more, about 0.7 hours; section 24).**
(G) group selection in the shared wiki with values taken at birth; (V) private notebooks with credit only for entries
the reader could see.

**Why group selection lagged (`RUN_EXTRA12`, on by default, 2 more, about 20 minutes for seeds 0–2; section 25,
Fig. 24).** Group selection in the shared wiki at every death: the heir copies its slot's juvenile period from the
fitter of two copies, judged by the whole copy's reward over the dead agent's life.

**More seeds (`RUN_EXTRA_SEEDS`, on by default, run last).** Seeds 3–9 for the settings behind the registered
comparisons: the 85 settings of earlier rounds (already done if you ran them before), plus the 21 settings of
`RUN_EXTRA10` that test registered predictions with ten seeds (about 2.5 hours). Set it to `False` to skip; everything else still
works with 3 seeds.

Runs go seed by seed, so after the first pass every setting already has one seed
for section 14. Finished runs are skipped, so *Run all* only runs what's missing.
Each saved run records its minutes and backend. On an RTX 3070 or a T4 a normal
run takes about 0.7 minutes and a long one about 2. The second round adds about
1.5 hours, the third about 1.3 hours, the evolution round about 1.3 hours.
"""

PAPER_RUNS = r"""PAPER_DIR = os.path.join(RESULTS_DIR, "paper")
os.makedirs(PAPER_DIR, exist_ok=True)
REQUIRE_GPU = REQ_
if REQUIRE_GPU and not ON_GPU:
    raise RuntimeError(
        "Paper runs need a GPU, but JAX only sees: " + str(jax.devices()) + ". "
        "On Colab: Runtime -> Change runtime type -> T4 GPU, then Run all. "
        "On your own Windows PC: open GPU_on_Windows.ipynb (same folder) in your usual Jupyter and Run all; "
        "it opens this notebook on the GPU. "
        "To run on the CPU anyway (very slow), set REQUIRE_GPU = False.")
PAPER_SEEDS = PSEEDS_
PAPER_LIFE, PAPER_GENS, PAPER_ENVS = PLIFE_, PGENS_, PENVS_
RUN_TIER2 = TIER2_

CITE = dict(citation=0.5)
NOISY = dict(p_high=0.8, p_low=0.2)
TIER1 = {
    "no_wiki":              dict(wiki=False),
    "oracle":               dict(oracle=True),
    "individual":           dict(),
    "individual_cited":     dict(CITE),
    "symbol":               dict(write_mode="symbol"),
    "symbol_cited":         dict(CITE, write_mode="symbol"),
    "rival_info":           dict(citation=-0.5),
    "wipe_cited":           dict(CITE, wipe_on_death=True),
    "clean_anonymous":      dict(CITE, identity="anonymous"),
    "deceivers8_named":     dict(CITE, n_deceivers=8),
    "deceivers8_anonymous": dict(CITE, n_deceivers=8, identity="anonymous"),
}
EXTRA = {   # added after the first GPU runs
    "notebook_cited":        dict(CITE, read_others=False),                       # own page only, inherited
    "notebook_wiped_cited":  dict(CITE, read_others=False, wipe_on_death=True),   # own page only, erased at death
    "symbol_long":           dict(write_mode="symbol", gens=3 * PAPER_GENS),      # 3x more generations
    "symbol_cited_long":     dict(CITE, write_mode="symbol", gens=3 * PAPER_GENS),
}
TIER2 = {
    "competitive_cited":      dict(CITE, reward_mode="competitive"),
    "shuffled_cited":         dict(CITE, reward_mode="shuffled"),
    "blamed":                 dict(CITE, blame=0.5),
    "rival_info_blamed":      dict(citation=-0.5, blame=0.5),
    "no_wiki_noisy":          dict(NOISY, wiki=False),
    "cited_noisy":            dict(NOISY, **CITE),
    "deceivers4_named":       dict(CITE, n_deceivers=4),
    "deceivers4_anonymous":   dict(CITE, n_deceivers=4, identity="anonymous"),
    "incompetent8_named":     dict(CITE, n_incompetent=8),
    "incompetent8_anonymous": dict(CITE, n_incompetent=8, identity="anonymous"),
}
RUN_ROBUST = True   # robustness and mechanism runs (added after the first draft; section 15)
ROBUST = {
    # what makes others' pages useful to a newborn
    "commons1_cited":            dict(CITE, read_k=1),           # own page + 1 partner's page
    "commons3_cited":            dict(CITE, read_k=3),           # own page + 3 partners' pages
    "notebook_replicated_cited": dict(CITE, replicate_own=True), # own entry repeated in every slot
    # heirs that keep their predecessor's hidden layers
    "no_wiki_torso":             dict(wiki=False, keep_torso=True),
    "notebook_torso_cited":      dict(CITE, read_others=False, keep_torso=True),
    "wiki_torso_cited":          dict(CITE, keep_torso=True),
    # learning saboteurs (rewarded when honest learners lose)
    "saboteurs8_named":          dict(CITE, n_saboteurs=8),
    "saboteurs8_anonymous":      dict(CITE, n_saboteurs=8, identity="anonymous"),
}
for k in (2, 6, 10, 12):                         # how many coordinated deceivers
    for ident in ("named", "anonymous"):
        ROBUST[f"deceivers{k}_{ident}"] = dict(CITE, n_deceivers=k, identity=ident)
for n in (4, 8, 32):                             # population size
    ROBUST[f"no_wiki_N{n}"] = dict(wiki=False, n_agents=n)
    ROBUST[f"cited_N{n}"] = dict(CITE, n_agents=n)
    ROBUST[f"competitive_N{n}"] = dict(CITE, reward_mode="competitive", n_agents=n)
for c in (0.1, 0.25, 1.0):                       # size of the citation credit
    ROBUST[f"cited_{c}"] = dict(citation=c)
for c in (0.1, 0.25):                            # size of the informational rivalry
    ROBUST[f"rival_{c}"] = dict(citation=-c)
for life in (PAPER_LIFE // 3, 3 * PAPER_LIFE):   # lifespan (same number of lifetimes)
    ROBUST[f"no_wiki_life{life}"] = dict(wiki=False, life=life)
    ROBUST[f"notebook_life{life}"] = dict(CITE, read_others=False, life=life)
    ROBUST[f"cited_life{life}"] = dict(CITE, life=life)

RUN_EXTRA2 = True   # second round of mechanism and attack runs (section 16)
STEPS = LearnConfig().num_steps                  # environment steps per update
EXTRA2 = {
    # inheritance: heir keeps its predecessor's hidden layers, but the notebook is erased
    "notebook_wiped_torso_cited":      dict(CITE, read_others=False, wipe_on_death=True, keep_torso=True),
    # is honesty explained by the own page being the author's own memory? hide it from its author
    "competitive_nocite":              dict(reward_mode="competitive"),
    "hidden_own":                      dict(hide_own=True),
    "hidden_own_cited":                dict(CITE, hide_own=True),
    "hidden_own_competitive":          dict(reward_mode="competitive", hide_own=True),
    "hidden_own_competitive_cited":    dict(CITE, reward_mode="competitive", hide_own=True),
    "hidden_own_cited_N4":             dict(CITE, hide_own=True, n_agents=4),
    "hidden_own_competitive_cited_N4": dict(CITE, reward_mode="competitive", hide_own=True, n_agents=4),
    # mass die-offs: everyone dies at the same moment
    "sync_no_wiki":                    dict(wiki=False, sync_deaths=True),
    "sync_notebook_cited":             dict(CITE, read_others=False, sync_deaths=True),
    "sync_cited":                      dict(CITE, sync_deaths=True),
    "sync_wipe_cited":                 dict(CITE, wipe_on_death=True, sync_deaths=True),
    # identity attacks on 8 coordinated liars
    "deceivers8_rotate_life":          dict(CITE, n_deceivers=8, identity_period=PAPER_LIFE * STEPS),       # names reshuffled once a lifetime
    "deceivers8_rotate_short":         dict(CITE, n_deceivers=8, identity_period=PAPER_LIFE * STEPS // 10), # ten times a lifetime
    "deceivers8_sleeper_named":        dict(CITE, n_deceivers=8, deceive_after=PAPER_LIFE * PAPER_GENS * STEPS // 2),
    "deceivers8_sleeper_anonymous":    dict(CITE, n_deceivers=8, identity="anonymous",
                                            deceive_after=PAPER_LIFE * PAPER_GENS * STEPS // 2),
}
DRIFTS, LIFES = (500, 2000, 8000), (PAPER_LIFE // 3, PAPER_LIFE, 3 * PAPER_LIFE)
for d in (500, 8000):                            # how fast the world changes x how long agents live
    for life in LIFES:
        EXTRA2[f"no_wiki_drift{d}_life{life}"] = dict(wiki=False, drift_period=d, life=life)
        EXTRA2[f"cited_drift{d}_life{life}"] = dict(CITE, drift_period=d, life=life)
for K in (4, 32):                                # number of machines (default 8)
    EXTRA2[f"no_wiki_K{K}"] = dict(wiki=False, n_bandits=K)
    EXTRA2[f"notebook_cited_K{K}"] = dict(CITE, read_others=False, n_bandits=K)
    EXTRA2[f"cited_K{K}"] = dict(CITE, n_bandits=K)

RUN_EXTRA3 = True   # third round, for the journal version (section 17)
FROZEN = 10 ** 9                                 # drift period so long that facts never change
EXTRA3 = {
    "no_wiki_static": dict(wiki=False, drift_period=FROZEN),
    "cited_static":   dict(CITE, drift_period=FROZEN),
}
for tag, d in (("drift8000", 8000), ("static", FROZEN)):     # does inheritance help when facts last?
    EXTRA3[f"notebook_cited_{tag}"] = dict(CITE, read_others=False, drift_period=d)
    EXTRA3[f"notebook_wiped_cited_{tag}"] = dict(CITE, read_others=False, wipe_on_death=True, drift_period=d)
    EXTRA3[f"notebook_torso_cited_{tag}"] = dict(CITE, read_others=False, keep_torso=True, drift_period=d)
    EXTRA3[f"notebook_wiped_torso_cited_{tag}"] = dict(CITE, read_others=False, wipe_on_death=True,
                                                         keep_torso=True, drift_period=d)
    EXTRA3[f"sync_cited_{tag}"] = dict(CITE, sync_deaths=True, drift_period=d)
    EXTRA3[f"sync_wipe_cited_{tag}"] = dict(CITE, wipe_on_death=True, sync_deaths=True, drift_period=d)
for c in (0.02, 0.05):                           # where exactly does honesty switch?
    EXTRA3[f"cited_{c}"] = dict(citation=c)
    EXTRA3[f"rival_{c}"] = dict(citation=-c)
for c in (-0.1, -0.05, -0.02, 0.02, 0.05, 0.1):  # the same with authors' own pages hidden from them
    EXTRA3[f"hidden_own_c{c:+g}"] = dict(citation=c, hide_own=True)
BOTTLENECK_AT = PAPER_LIFE * PAPER_GENS // 2     # a single bottleneck halfway through the run
for k in (8, 12, 15, 16):                        # k of 16 agents die at once
    EXTRA3[f"bottleneck{k}_cited"] = dict(CITE, bottleneck_at=BOTTLENECK_AT, bottleneck_frac=k / 16)
EXTRA3["bottleneck16_wipe_cited"] = dict(CITE, wipe_on_death=True, bottleneck_at=BOTTLENECK_AT, bottleneck_frac=1.0)

RUN_EXTRA4 = True   # juvenile period: newborns cannot write during their first updates (section 17)
JUV = dict(juvenile_updates=PAPER_LIFE // 6)    # 50 updates with 300-update lives
EXTRA4 = {"individual_cited_juv": dict(CITE, **JUV)}
for tag, d in (("", 2000), ("_static", FROZEN)):
    EXTRA4[f"notebook_cited{tag}_juv"] = dict(CITE, read_others=False, drift_period=d, **JUV)
    EXTRA4[f"notebook_wiped_cited{tag}_juv"] = dict(CITE, read_others=False, wipe_on_death=True, drift_period=d, **JUV)
    EXTRA4[f"sync_cited{tag}_juv"] = dict(CITE, sync_deaths=True, drift_period=d, **JUV)
    EXTRA4[f"sync_wipe_cited{tag}_juv"] = dict(CITE, wipe_on_death=True, sync_deaths=True, drift_period=d, **JUV)
EXTRA4["notebook_torso_cited_static_juv"] = dict(CITE, read_others=False, keep_torso=True, drift_period=FROZEN, **JUV)
EXTRA4["notebook_wiped_torso_cited_static_juv"] = dict(CITE, read_others=False, wipe_on_death=True, keep_torso=True,
                                                       drift_period=FROZEN, **JUV)

RUN_EXTRA5 = True   # evolution of the juvenile period (section 18)
EVO = dict(juvenile_evolve=True, juvenile_sigma=10.0, gens=12)
EVO_BASES = {
    "notebook_static":       dict(CITE, read_others=False, drift_period=FROZEN),
    "notebook":              dict(CITE, read_others=False),
    "notebook_torso_static": dict(CITE, read_others=False, keep_torso=True, drift_period=FROZEN),
    "wiki":                  dict(CITE),
    "sync_static":           dict(CITE, sync_deaths=True, drift_period=FROZEN),
    "sync_static_group":     dict(CITE, sync_deaths=True, drift_period=FROZEN, juvenile_group=True),
}
EXTRA5 = {}
for base, spec in EVO_BASES.items():                 # evolve from no juvenile period and from half a lifetime
    for j0 in (0, PAPER_LIFE // 2):
        EXTRA5[f"evo_{base}_from{j0}"] = dict(spec, juvenile_updates=j0, **EVO)
for J in (PAPER_LIFE // 30, PAPER_LIFE // 12, PAPER_LIFE // 3):   # fixed juvenile periods of 10, 25 and 100 updates
    for name in ("notebook_cited_static", "notebook_wiped_cited_static", "sync_cited_static", "sync_wipe_cited_static"):
        EXTRA5[f"{name}_juv{J}"] = dict(EXTRA3[name], juvenile_updates=J)

RUN_EXTRA6 = True   # longer evolution: does the juvenile period settle? (section 18, Fig. 13)
LONG_BASES = {
    "notebook":                 dict(CITE, read_others=False),
    "notebook_drift8000":       dict(CITE, read_others=False, drift_period=8000),
    "notebook_torso":           dict(CITE, read_others=False, keep_torso=True),
    "notebook_torso_drift8000": dict(CITE, read_others=False, keep_torso=True, drift_period=8000),
}
EXTRA6 = {f"evo_{base}_from{j0}_long": dict(spec, juvenile_updates=j0, juvenile_evolve=True, juvenile_sigma=10.0, gens=36)
          for base, spec in LONG_BASES.items() for j0 in (0, PAPER_LIFE // 2)}

RUN_EXTRA7 = True   # the reviewer's tests: write prior, fixed juvenile periods, hysteresis (section 19, Figs. 14-15)
EXTRA7 = {}
# K: newborns start keeping their entry with probability 0.97 or 0.99 instead of 0.9 (write prior 0.03 or 0.01)
KEEP_PRIORS = {"keep97": 0.03, "keep99": 0.01}
KEEP_BASES = ("notebook_cited_static", "notebook_wiped_cited_static", "notebook_torso_cited_static",
              "notebook_wiped_torso_cited_static", "sync_cited_static", "sync_wipe_cited_static")
for tag, p in KEEP_PRIORS.items():
    for name in KEEP_BASES:
        EXTRA7[f"{name}_{tag}"] = dict(EXTRA3[name], p_write=p)
    EXTRA7[f"individual_cited_{tag}"] = dict(TIER1["individual_cited"], p_write=p)
PAPER_SETTINGS_SO_FAR = dict(TIER1, **EXTRA, **(TIER2 if RUN_TIER2 else {}), **(ROBUST if RUN_ROBUST else {}),
                             **(EXTRA2 if RUN_EXTRA2 else {}), **(EXTRA3 if RUN_EXTRA3 else {}),
                             **(EXTRA4 if RUN_EXTRA4 else {}), **(EXTRA5 if RUN_EXTRA5 else {}),
                             **(EXTRA6 if RUN_EXTRA6 else {}))
# F: fixed juvenile periods in the four 36-lifetime settings (6 lifetimes each)
FIX_JS = (0, PAPER_LIFE // 30, PAPER_LIFE // 12, PAPER_LIFE // 6, PAPER_LIFE // 3)   # 0, 10, 25, 50, 100 updates
FIX_REUSE = {("notebook", 0): "notebook_cited", ("notebook_drift8000", 0): "notebook_cited_drift8000",
             ("notebook_torso", 0): "notebook_torso_cited", ("notebook_torso_drift8000", 0): "notebook_torso_cited_drift8000",
             ("notebook", PAPER_LIFE // 6): "notebook_cited_juv"}       # runs that already exist
def fix_name(base, J):
    return FIX_REUSE.get((base, J), f"fixJ_{base}_J{J}")
for base, spec in LONG_BASES.items():
    for J in FIX_JS:
        name = fix_name(base, J)
        if name.startswith("fixJ_") or name not in PAPER_SETTINGS_SO_FAR:
            EXTRA7[name] = dict(spec, juvenile_updates=J) if J else dict(spec)
# H: hysteresis near the honesty threshold: credit +0.5 or -0.5 for 3 lifetimes, then the target for 6 more
SWITCH_AT = 3 * PAPER_LIFE * STEPS               # environment steps
HYST_TARGETS = {"readable": (-0.05, -0.02, 0.0), "hidden": (-0.02, 0.0, 0.02)}
HYST_STARTS = {"honest": 0.5, "lying": -0.5}
def hyst_name(pages, c, start):
    return f"hyst_{pages}_c{c:+g}_from_{start}"
for pages, targets in HYST_TARGETS.items():
    for c in targets:
        for start, c0 in HYST_STARTS.items():
            EXTRA7[hyst_name(pages, c, start)] = dict(citation=c, citation_early=c0, citation_switch_at=SWITCH_AT,
                                                      gens=PAPER_GENS + 3, **({"hide_own": True} if pages == "hidden" else {}))

RUN_EXTRA8 = True   # second round of review tests (section 20, Figs. 16-17)
EXTRA8 = {
    # R: where liars sit. Own entry first, others in an order shared by all copies / drawn per copy
    "deceivers8_ownfirst_fixed":             dict(CITE, n_deceivers=8, identity_period=FROZEN),
    "deceivers8_seats_per_copy":             dict(CITE, n_deceivers=8, seat_per_copy=True),
    "deceivers8_seats_per_copy_rotate_life": dict(CITE, n_deceivers=8, seat_per_copy=True, identity_period=PAPER_LIFE * STEPS),
    "clean_seats_per_copy":                  dict(CITE, seat_per_copy=True),
}
for pages, targets in HYST_TARGETS.items():          # I: the credit switch with learners that never die
    for c in targets:
        for start in HYST_STARTS:
            EXTRA8[hyst_name(pages, c, start) + "_immortal"] = dict(EXTRA7[hyst_name(pages, c, start)], immortal=True)
for j0 in (0, PAPER_LIFE // 2):                        # G: shared wiki, 36 lifetimes, individual vs group selection
    EXTRA8[f"evo_wiki_from{j0}_long"] = dict(CITE, juvenile_updates=j0, juvenile_evolve=True, juvenile_sigma=10.0, gens=36)
    EXTRA8[f"evo_wiki_group_from{j0}_long"] = dict(CITE, juvenile_updates=j0, juvenile_evolve=True, juvenile_sigma=10.0,
                                                   juvenile_group=True, gens=36)
FIX_JS_MORE = {"notebook": (PAPER_LIFE // 20, PAPER_LIFE * 7 // 60),            # 15, 35 updates
               "notebook_torso": (PAPER_LIFE // 60, PAPER_LIFE // 20),           # 5, 15
               "notebook_drift8000": (PAPER_LIFE // 2, PAPER_LIFE * 2 // 3),     # 150, 200
               "notebook_torso_drift8000": (PAPER_LIFE // 20, PAPER_LIFE * 7 // 60)}
for base, Js in FIX_JS_MORE.items():                   # F: finer and wider grid of fixed juvenile periods
    for J in Js:
        EXTRA8[fix_name(base, J)] = dict(LONG_BASES[base], juvenile_updates=J)
MEM = dict(own_memory=True)                            # M: agents with a memory of their own past visits
EXTRA8.update({
    "mem_no_wiki":                         dict(wiki=False, **MEM),
    "mem_cited":                           dict(CITE, **MEM),
    "mem_notebook_cited_static":           dict(EXTRA3["notebook_cited_static"], **MEM),
    "mem_notebook_wiped_cited_static":     dict(EXTRA3["notebook_wiped_cited_static"], **MEM),
    "mem_notebook_cited_static_juv":       dict(EXTRA3["notebook_cited_static"], **MEM, **JUV),
    "mem_notebook_wiped_cited_static_juv": dict(EXTRA3["notebook_wiped_cited_static"], **MEM, **JUV),
})

RUN_EXTRA9 = True   # turnover dose and which side holds the lock (section 22, Figs. 19-20)
EXTRA9 = {}
RUN_LEN = (PAPER_GENS + 3) * PAPER_LIFE              # the credit-switch runs: 2,700 updates, switch at 900
DOSE_LIVES = (PAPER_LIFE // 3, PAPER_LIFE * 3, PAPER_LIFE * 9)   # D: lives of 100, 900 and 2,700 updates
DOSE_TARGETS = (("readable", 0.0), ("hidden", 0.02))
def dose_name(pages, c, start, life):
    return hyst_name(pages, c, start) + f"_life{life}"
for pages, c in DOSE_TARGETS:
    for start in HYST_STARTS:
        for life in DOSE_LIVES:
            EXTRA9[dose_name(pages, c, start, life)] = dict(EXTRA7[hyst_name(pages, c, start)], life=life, gens=RUN_LEN / life)
RESET_PARTS_NB = ("pull", "write", "heads", "all")   # X: at the switch, re-initialise the readers, writers, both, or all
def reset_name(pages, c, part):
    return hyst_name(pages, c, "lying") + f"_immortal_reset_{part}"
for pages, c in DOSE_TARGETS:
    for part in RESET_PARTS_NB:
        EXTRA9[reset_name(pages, c, part)] = dict(EXTRA7[hyst_name(pages, c, "lying")], immortal=True,
                                                  reset_at=SWITCH_AT // STEPS, reset_part=part)

RUN_EXTRA10 = True   # the fourth review's tests (section 23, Figs. 21-23)
EXTRA10 = {}
NOISES = (0.02, 0.10)                                # N: random writes without death
def noise_name(pages, c, start, eps):
    return hyst_name(pages, c, start) + f"_immortal_noise{eps:g}"
for pages, c in DOSE_TARGETS:
    for eps in NOISES:
        EXTRA10[noise_name(pages, c, "lying", eps)] = dict(EXTRA7[hyst_name(pages, c, "lying")], immortal=True, write_noise=eps)
    EXTRA10[noise_name(pages, c, "honest", 0.10)] = dict(EXTRA7[hyst_name(pages, c, "honest")], immortal=True, write_noise=0.10)
def strong_name(pages, immortal):                    # S: a lying start switched to strong credit
    return hyst_name(pages, 0.5, "lying") + ("_immortal" if immortal else "")
for pages, c in DOSE_TARGETS:
    for imm in (False, True):
        EXTRA10[strong_name(pages, imm)] = dict(EXTRA7[hyst_name(pages, c, "lying")], citation=0.5,
                                                **({"immortal": True} if imm else {}))
SILENCES = (PAPER_LIFE // 6, PAPER_LIFE // 2)        # J: juvenile periods of 50 and 150 updates in the switch runs
def silence_name(pages, c, start, J):
    return hyst_name(pages, c, start) + f"_juv{J}"
for pages, c in DOSE_TARGETS:
    for start in HYST_STARTS:
        for J in SILENCES:
            EXTRA10[silence_name(pages, c, start, J)] = dict(EXTRA7[hyst_name(pages, c, start)], juvenile_updates=J)
EXTRA10.update({                                     # R: rivalry over any reader's win, not only the named arm
    "readrival_readable_c-0.5": dict(citation=-0.5, credit_on="read"),
    "readrival_hidden_c-0.5":   dict(citation=-0.5, credit_on="read", hide_own=True),
    "hidden_own_c-0.5":         dict(citation=-0.5, hide_own=True),
})
TJS = (0, PAPER_LIFE // 30, PAPER_LIFE // 12, PAPER_LIFE // 6, PAPER_LIFE // 3)   # T: 0, 10, 25, 50, 100 updates
def tsuf(J):                                         # the names used by the juvenile-period dose series
    return "" if J == 0 else ("_juv" if J == PAPER_LIFE // 6 else f"_juv{J}")
LRS = ((0.5, "lrhalf"), (2.0, "lrdouble"))
for f, tag in LRS:
    for J in TJS:
        for base in ("notebook_cited_static", "notebook_wiped_cited_static"):
            EXTRA10[f"{base}{tsuf(J)}_{tag}"] = dict(EXTRA3[base], juvenile_updates=J, lr=LearnConfig().lr * f)
for J in TJS[1:]:
    for base in ("notebook_cited_drift8000", "notebook_wiped_cited_drift8000"):
        EXTRA10[f"{base}{tsuf(J)}"] = dict(EXTRA3[base], juvenile_updates=J)
EXTRA10.update({                                     # C: 64 copies of the world instead of 256
    "no_wiki_envs64":          dict(wiki=False, envs=64),
    "individual_cited_envs64": dict(CITE, envs=64),
    hyst_name("readable", 0.0, "lying") + "_envs64":           dict(EXTRA7[hyst_name("readable", 0.0, "lying")], envs=64),
    hyst_name("readable", 0.0, "lying") + "_immortal_envs64":  dict(EXTRA7[hyst_name("readable", 0.0, "lying")], immortal=True, envs=64),
})
TEN_SEEDS10 = [n for n in EXTRA10 if n.startswith(("hyst_", "readrival_", "hidden_own_c-0.5")) and "envs64" not in n]

RUN_EXTRA11 = True   # corrections after the independent review (section 24)
EXTRA11 = {}
for j0 in (0, PAPER_LIFE // 2):                      # G re-tested: group selection with staggered deaths, values taken at birth
    EXTRA11[f"evo_wiki_groupb_from{j0}_long"] = dict(CITE, juvenile_updates=j0, juvenile_evolve=True, juvenile_sigma=10.0,
                                                     juvenile_group_birth=True, gens=36)
SEEN_PAIRS = [("notebook_cited", "notebook_wiped_cited"), ("notebook_cited_juv", "notebook_wiped_cited_juv"),
              ("notebook_cited_static", "notebook_wiped_cited_static"),
              ("notebook_cited_static_juv", "notebook_wiped_cited_static_juv"),
              ("notebook_torso_cited_static", "notebook_wiped_torso_cited_static"),
              ("notebook_torso_cited_static_juv", "notebook_wiped_torso_cited_static_juv")]
_SRC11 = {**PAPER_SETTINGS_SO_FAR, **EXTRA7, **EXTRA8, **EXTRA9, **EXTRA10}
for a, b in SEEN_PAIRS:                              # V: private notebooks, credit only for entries readers can see
    for n in (a, b):
        EXTRA11[n + "_seen"] = dict(_SRC11[n], credit_seen_only=True)

RUN_EXTRA12 = True   # why group selection lagged: group selection at every death (section 25)
EXTRA12 = {}
for j0 in (0, PAPER_LIFE // 2):
    EXTRA12[f"evo_wiki_groupd_from{j0}_long"] = dict(CITE, juvenile_updates=j0, juvenile_evolve=True, juvenile_sigma=10.0,
                                                     juvenile_group_each_death=True, gens=36)

PAPER_SETTINGS = {**PAPER_SETTINGS_SO_FAR, **(EXTRA7 if RUN_EXTRA7 else {}), **(EXTRA8 if RUN_EXTRA8 else {}),
                  **(EXTRA9 if RUN_EXTRA9 else {}), **(EXTRA10 if RUN_EXTRA10 else {}),
                  **(EXTRA11 if RUN_EXTRA11 else {}), **(EXTRA12 if RUN_EXTRA12 else {})}

RUN_EXTRA_SEEDS = True   # more seeds for the registered comparisons, run after everything else (about 4.5 hours)
EXTRA_SEEDS = ESEEDS_
SEED_SETTINGS = ([hyst_name(p, c, st) for p, ts in HYST_TARGETS.items() for c in ts for st in HYST_STARTS]
                 + ["rival_0.05", "rival_0.02", "individual", "hidden_own_c-0.02", "hidden_own", "hidden_own_c+0.02"]
                 + [n + sfx for n in KEEP_BASES for sfx in ("", "_juv", "_keep99")]
                 + ["individual_cited", "individual_cited_juv", "individual_cited_keep99"]
                 + [fix_name(b, J) for b in ("notebook", "notebook_torso") for J in FIX_JS]
                 + ["deceivers8_named", "deceivers8_anonymous", "deceivers8_rotate_life", "deceivers8_rotate_short"]
                 + [hyst_name(p, c, st) + "_immortal" for p, ts in HYST_TARGETS.items() for c in ts for st in HYST_STARTS]
                 + list(EXTRA9) + TEN_SEEDS10)
SEED_SETTINGS = [n for n in dict.fromkeys(SEED_SETTINGS) if n in PAPER_SETTINGS]

import time

def run_paper(name, seed):
    path = os.path.join(PAPER_DIR, f"{name}_s{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    spec = dict(PAPER_SETTINGS[name])
    gens, life = spec.pop("gens", PAPER_GENS), spec.pop("life", PAPER_LIFE)
    p_write = spec.pop("p_write", LearnConfig().p_write)          # newborns' initial write probability
    lr, envs = spec.pop("lr", LearnConfig().lr), spec.pop("envs", PAPER_ENVS)   # learning rate, world copies
    wc, lc = BanditConfig(**spec), LearnConfig(num_envs=envs, reward_centering=True, p_write=p_write, lr=lr)
    t0 = time.time()
    log, runner = run_world(wc, lc, life, gens, seed, log_every=300)
    log["wiki_use"] = evaluate_wiki_use(wc, lc, runner) if (wc.wiki and not wc.oracle) else None
    log["minutes"] = (time.time() - t0) / 60
    log["backend"] = jax.default_backend()
    with open(path, "wb") as f:
        pickle.dump(log, f)
    print(f"    saved in {log['minutes']:.1f} min")
    return log

PAPER = {}
for seed in PAPER_SEEDS:
    for name in PAPER_SETTINGS:
        print(f"=== {name}, seed {seed}")
        PAPER[(name, seed)] = run_paper(name, seed)
if RUN_EXTRA_SEEDS:                                   # more seeds for the registered comparisons
    for seed in EXTRA_SEEDS:
        for name in SEED_SETTINGS:
            print(f"=== {name}, seed {seed} (extra seeds)")
            PAPER[(name, seed)] = run_paper(name, seed)
"""

FIG_MD = r"""## 14. Paper tables and figures

Everything below is saved to `wikibandits_results/paper/` in your Drive:
`summary.csv` and `figures/fig1–fig6` as PNG and PDF.

1. **Ladder**: adults in the core settings against the best possible strategies
   with no memory, with their own memory and with perfectly shared memory.
2. **Incentives and honesty**: reward, honest writes and following for individual,
   cited and informational rivalry (plus blame and the competitive controls from
   tier 2). Following below chance means readers read the lies in reverse.
3. **Identity and liars**: clean and with 4 or 8 deceivers or 8 incompetents,
   named vs anonymous, for adults and for newborns.
4. **Newborns**: reward from birth to death, averaged over every complete life:
   wiki vs private notebook vs nothing, and under liars or rivalry.
5. **Wiki survival**: share of machines whose wiki is right, over the whole run.
   Dotted lines mark one lifetime; by the end every agent has died about 6 times.
6. **Free symbols over many generations**: whether entries come to name the good
   arm and readers come to follow them.

"Newborn reward" is the average over the first 50 updates of a life, "adult
reward" over the last 50. "Population reward" is the average over all learners,
whatever their age, in the last third of the run. ± is the standard error across seeds.
"""

FIG = r"""plt.rcParams["figure.dpi"] = 150          # sharper figures in the notebook itself
pd.set_option("display.max_rows", 500)            # show every setting in the tables
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
EARLY = min(50, PAPER_LIFE // 4)

def ms(vals, d=3):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
    if len(vals) == 0:
        return "-", np.nan, np.nan
    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return f"{vals.mean():.{d}f} ± {se:.{d}f}", vals.mean(), se

def logs_of(name):
    return [PAPER[k] for k in PAPER if k[0] == name]

def has_wiki(name):
    cfg = PAPER_SETTINGS[name]
    return cfg.get("wiki", True) and not cfg.get("oracle", False)

rows, nums = [], {}
for name in PAPER_SETTINGS:
    L = logs_of(name)
    if not L:
        continue
    metrics = {
        "adult reward": [adult_reward(l, EARLY) for l in L],
        "newborn reward": [newborn_reward(l, EARLY) for l in L],
        "population reward": [late(l, "raw") for l in L],
        "t to 0.61": [recovery_time(l, 0.61) for l in L],
        "t to 90% adult": [recovery_time(l, 0.9 * adult_reward(l, EARLY)) for l in L],
        "wiki right": [late(l, "majority_ok") for l in L] if has_wiki(name) else [],
        "honest writes": [late(l, "honest_writes") for l in L] if has_wiki(name) else [],
        "correct writes": [late(l, "correct_writes") for l in L] if has_wiki(name) else [],
        "writes": [late(l, "write_rate") for l in L] if has_wiki(name) else [],
        "follows": [late(l, "follow_rate") for l in L] if has_wiki(name) else [],
        "agrees w/ bad": [late(l, "agree_bad") for l in L] if has_wiki(name) else [],
        "real − scrambled": [l["wiki_use"]["real"] - l["wiki_use"]["scrambled"] if l.get("wiki_use") else np.nan for l in L],
    }
    row = {"setting": name, "seeds": len(L)}
    for k, v in metrics.items():
        row[k], m, se = ms(v, 4 if k == "writes" else 0 if k.startswith("t to") else 3)
        nums[(name, k)] = (m, se, v)
    row["minutes/run"] = f"{np.mean([l.get('minutes', np.nan) for l in L]):.1f}"
    row["backend"] = ",".join(sorted({str(l.get('backend', '?')) for l in L}))
    rows.append(row)
summary = pd.DataFrame(rows).set_index("setting")
summary.to_csv(os.path.join(PAPER_DIR, "summary.csv"))
try:
    display(summary)
except NameError:
    print(summary.to_string())

def savefig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")

def smooth_xy(x, k):
    # moving average plotted at the centre of its window, so it lines up with the update axis
    y = smooth(x, k)
    return np.arange(len(y)) + ((k - 1) / 2 if len(x) >= k else 0), y

def bars(ax, names, key, color="#7F77DD", ref=None):
    names = [n for n in names if (n, key) in nums and np.isfinite(nums[(n, key)][0])]
    ax.bar(range(len(names)), [nums[(n, key)][0] for n in names], yerr=[nums[(n, key)][1] for n in names],
           color=color, capsize=3)
    for i, n in enumerate(names):
        ax.scatter([i] * len(nums[(n, key)][2]), nums[(n, key)][2], color="k", s=8, zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return names

# Figure 1: ladder
fig, ax = plt.subplots(figsize=(10, 3.8))
shown = bars(ax, ["no_wiki", "notebook_wiped_cited", "notebook_cited", "symbol", "symbol_cited",
                  "individual", "individual_cited", "oracle"], "adult reward")
label_y = -1.0
for nm, v in sorted(LADDER.items(), key=lambda kv: kv[1]):
    ax.axhline(v, lw=0.8, ls=":", color="gray")
    label_y = max(v, label_y + 0.045)                     # keep close labels apart
    ax.text(len(shown) - 0.45, label_y, f" {nm} ({v:.2f})", fontsize=7, va="center", color="gray")
ax.set_ylim(0, 1.0)
ax.set_ylabel("adult reward per pull")
ax.set_title("Adults against the best possible strategies with each kind of memory")
savefig(fig, "fig1_ladder")

# Figure 2: incentives -> honesty, and how readers respond
comp = ["individual", "individual_cited", "blamed", "competitive_cited", "shuffled_cited", "rival_info", "rival_info_blamed"]
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
bars(ax[0], comp, "adult reward")
ax[0].set_ylabel("adult reward per pull")
if ("no_wiki", "adult reward") in nums:
    ax[0].axhline(nums[("no_wiki", "adult reward")][0], color="gray", ls=":", lw=1)
bars(ax[1], comp, "honest writes", color="#1D9E75")
ax[1].set_ylabel("share of writes that are honest")
ax[1].set_ylim(0, 1.05)
bars(ax[2], comp, "follows", color="#BA7517")
ax[2].axhline(1 / BanditConfig().n_arms, color="gray", ls=":", lw=1)
ax[2].set_ylabel("follows majority (dotted: chance)")
ax[2].set_ylim(0, 1.05)
fig.suptitle("Incentives decide honesty; readers adapt (below chance = reading lies in reverse)")
fig.tight_layout()
savefig(fig, "fig2_competition")

# Figure 3: identity and liars, for adults and for newborns
idn = ["individual_cited", "clean_anonymous", "deceivers4_named", "deceivers4_anonymous",
       "deceivers8_named", "deceivers8_anonymous", "incompetent8_named", "incompetent8_anonymous",
       "saboteurs8_named", "saboteurs8_anonymous"]
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), sharey=True)
for a, key in zip(ax, ("adult reward", "newborn reward")):
    shown = bars(a, idn, key, color="#D85A30" if key == "adult reward" else "#F0997B")
    if ("no_wiki", key) in nums:
        a.axhline(nums[("no_wiki", key)][0], color="gray", ls=":", lw=1)
        a.text(len(shown) - 0.5, nums[("no_wiki", key)][0], " no wiki", fontsize=8, color="gray", va="center")
    a.set_title(key.replace(" reward", "s") + (" (last 50 updates of a life)" if key == "adult reward"
                                                 else f" (first {EARLY} updates of a life)"), fontsize=10)
ax[0].set_ylabel("learners' reward per pull")
fig.suptitle("Liars, identity and newcomers (individual_cited = clean wiki, named authors)")
fig.tight_layout()
savefig(fig, "fig3_identity")

# Figure 5: wiki survival across generations
fig, ax = plt.subplots(figsize=(9, 3.4))
for name, color in (("individual_cited", "#534AB7"), ("individual", "#888780"), ("wipe_cited", "#D85A30"),
                    ("rival_info", "#BA7517")):
    L = logs_of(name)
    if L:
        ax.plot(*smooth_xy(np.mean([l["majority_ok"] for l in L], 0), 15), color=color, label=name)
for g in range(1, PAPER_GENS):
    ax.axvline(g * PAPER_LIFE, color="gray", ls=":", lw=0.7)
ax.set_ylim(0, 1.02)
ax.set_xlabel("update (dotted: one lifetime)")
ax.set_ylabel("machines whose wiki majority is right")
ax.set_title("The wiki outlives its writers")
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
savefig(fig, "fig5_survival")

# Figure 4: newborns, from birth to death (complete lives only)
def life_curve(name, k=5):
    L = logs_of(name)
    curves = [r for l in L for _, _, r, done in lives(l) if done] or [r for l in L for _, _, r, _ in lives(l)]
    if not curves:
        return None
    m = min(map(len, curves))
    return smooth(np.mean([c[:m] for c in curves], 0), k)

panels = [("Wiki, private notebook or nothing",
           (("no_wiki", "#888780"), ("notebook_wiped_cited", "#B4B2A9"), ("notebook_cited", "#BA7517"),
            ("individual_cited", "#534AB7"), ("oracle", "#1D9E75"))),
          ("Liars and rivalry",
           (("no_wiki", "#888780"), ("individual_cited", "#534AB7"), ("deceivers8_named", "#D85A30"),
            ("deceivers8_anonymous", "#F0997B"), ("rival_info", "#BA7517")))]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
for a, (title, series) in zip(ax, panels):
    for name, color in series:
        c = life_curve(name)
        if c is not None:
            a.plot(c, color=color, label=name)
    a.set_xlabel("updates since birth")
    a.set_title(title, fontsize=10)
    a.legend(fontsize=8, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("reward per pull")
fig.suptitle("Newborns start with random weights: what they inherit decides how fast they recover")
fig.tight_layout()
savefig(fig, "fig4_newborns")

# Figure 6: free symbols over many generations
import warnings

def over_time(log, key):
    x = log[key]
    if x.ndim == 2:
        x = x[:, learner_mask(log)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x = np.nanmean(x, axis=1)
    return x

long_names = [n for n in ("symbol_long", "symbol_cited_long") if logs_of(n)]
if long_names:
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
    for name, color in zip(long_names, ("#B4B2A9", "#534AB7")):
        for a, key in zip(ax, ("correct_writes", "follow_rate", "raw")):
            curves = [over_time(l, key) for l in logs_of(name)]
            m = min(map(len, curves))
            a.plot(smooth(np.nanmean([c[:m] for c in curves], 0), 30), color=color, label=name)
    ax[0].axhline(1 / BanditConfig().n_arms, color="gray", ls=":", lw=1)
    ax[0].set_ylabel("entries naming the good arm (dotted: chance)")
    ax[1].set_ylabel("follows the wiki majority")
    ax[2].set_ylabel("reward per pull (all ages)")
    if logs_of("no_wiki"):
        ax[2].axhline(np.mean([over_time(l, "raw")[-PAPER_LIFE:].mean() for l in logs_of("no_wiki")]),
                      color="gray", ls=":", lw=1)
        ax[2].text(0, ax[2].get_ylim()[0], " dotted: no wiki", fontsize=8, color="gray", va="bottom")
    for a in ax:
        a.set_xlabel("update")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Free symbols over {3 * PAPER_GENS} generations: does a shared convention form?")
    fig.tight_layout()
    savefig(fig, "fig6_symbols_long")

print("figures saved to", FIG_DIR)
"""

ROBUST_MD = r"""## 15. Robustness and mechanisms

Uses the runs of `RUN_ROBUST` (section 13) together with the earlier ones.
Saved to `wikibandits_results/paper/`: `contrasts.csv` and `figures/fig7`, `fig8`.

- **Comparisons with confidence intervals.** For each key comparison: the difference
  between the two settings' means, its 95% confidence interval across seeds (Welch),
  and a verdict. *Same within ±0.02* means the whole interval lies within ±0.02.
  *Different* means it excludes zero; *unclear* means neither.
- **Newborn recovery.** *t to 0.61*: median updates from birth until a newborn's
  reward beats the best possible player without memory (0.61). *t to 90% adult*:
  until it reaches 90% of its setting's adult level. "-" means most newborns never
  get there.
- **Fig. 7**:
  - (a) honest writes against population size;
  - (b) newborn reward against the number of coordinated liars, named and anonymous;
  - (c) honest writes and newborn reward against the size of the citation credit
    (negative = rivalry);
  - (d) newborn and adult reward against lifespan.
- **Fig. 8**: why others' pages help newborns.
  - (a) newborn reward with a notebook, the notebook entry repeated in every slot,
    small commons and the full wiki;
  - (b) heirs that keep their predecessor's hidden layers;
  - (c) recovery curves.
"""

ROBUST_FIG = r"""def vals(name, key):
    return nums.get((name, key), (np.nan, np.nan, []))[2]

CONTRASTS = [
    ("wipe_cited", "individual_cited", "newborn reward", "erasing a dead author's page"),
    ("notebook_cited", "notebook_wiped_cited", "newborn reward", "inheriting a private notebook"),
    ("notebook_cited", "no_wiki", "newborn reward", "private notebook vs nothing"),
    ("individual_cited", "notebook_cited", "newborn reward", "shared wiki vs private notebook"),
    ("notebook_replicated_cited", "notebook_cited", "newborn reward", "own entry repeated in every slot"),
    ("commons1_cited", "notebook_cited", "newborn reward", "one partner's page added"),
    ("commons3_cited", "notebook_cited", "newborn reward", "three partners' pages added"),
    ("notebook_torso_cited", "no_wiki_torso", "newborn reward", "inherited notebook, heir keeps features"),
    ("individual_cited", "individual", "honest writes", "citation credit"),
    ("competitive_cited", "individual_cited", "honest writes", "competition (16 agents)"),
    ("competitive_N4", "cited_N4", "honest writes", "competition (4 agents)"),
    ("shuffled_cited", "competitive_cited", "adult reward", "competitive vs shuffled control"),
    ("rival_info", "individual", "honest writes", "informational rivalry"),
    ("rival_info", "no_wiki", "newborn reward", "rivalry vs no wiki, newborns"),
    ("clean_anonymous", "individual_cited", "newborn reward", "anonymity, no liars"),
    ("deceivers8_anonymous", "deceivers8_named", "newborn reward", "anonymity, 8 coordinated liars"),
    ("deceivers8_named", "incompetent8_named", "newborn reward", "named: liars vs random writers"),
    ("deceivers8_anonymous", "no_wiki", "newborn reward", "anonymous liars vs no wiki"),
    ("saboteurs8_anonymous", "saboteurs8_named", "newborn reward", "anonymity, 8 learning saboteurs"),
    ("symbol_cited_long", "no_wiki", "adult reward", "free symbols after 18 generations vs no wiki"),
]
rows = []
for a, b, key, what in CONTRASTS:
    if (a, key) in nums and (b, key) in nums:
        d, lo, hi, verdict = welch_diff(vals(a, key), vals(b, key))
        rows.append({"comparison": what, "a": a, "b": b, "measure": key, "a - b": round(d, 3),
                     "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "verdict": verdict})
contrasts = pd.DataFrame(rows)
contrasts.to_csv(os.path.join(PAPER_DIR, "contrasts.csv"), index=False)
try:
    display(contrasts)
except NameError:
    print(contrasts.to_string())

RECOVERY = ["no_wiki", "notebook_wiped_cited", "notebook_cited", "notebook_replicated_cited", "commons1_cited",
            "commons3_cited", "individual", "individual_cited", "wipe_cited", "oracle", "no_wiki_torso",
            "notebook_torso_cited", "wiki_torso_cited", "deceivers8_named", "deceivers8_anonymous",
            "saboteurs8_named", "saboteurs8_anonymous", "rival_info"]
rec = summary.loc[[n for n in RECOVERY if n in summary.index], ["newborn reward", "t to 0.61", "t to 90% adult"]]
try:
    display(rec)
except NameError:
    print(rec.to_string())

def series(points, key):
    xs, m, se = [], [], []
    for x, name in points:
        if (name, key) in nums and np.isfinite(nums[(name, key)][0]):
            xs.append(x); m.append(nums[(name, key)][0]); se.append(nums[(name, key)][1])
    return np.array(xs), np.array(m), np.array(se)

def no_wiki_line(a, key):
    if ("no_wiki", key) in nums:
        a.axhline(nums[("no_wiki", key)][0], color="gray", ls=":", lw=1)
        a.text(a.get_xlim()[1], nums[("no_wiki", key)][0], " no wiki", fontsize=8, color="gray", va="center")

# Figure 7: robustness
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
a = ax[0, 0]
Ns = [4, 8, 16, 32]
for label, fmt, at16, color in (("individual + citation", "cited_N{}", "individual_cited", "#534AB7"),
                                ("competitive + citation", "competitive_N{}", "competitive_cited", "#D85A30")):
    x, m, se = series([(n, at16 if n == 16 else fmt.format(n)) for n in Ns], "honest writes")
    a.errorbar(x, m, yerr=se, marker="o", color=color, label=label, capsize=3)
a.set_xscale("log", base=2); a.set_xticks(Ns); a.set_xticklabels(Ns)
a.set_ylim(0, 1.02); a.set_xlabel("number of agents"); a.set_ylabel("honest writes")
a.set_title("(a) Honesty and population size", fontsize=10)
a = ax[0, 1]
ks = [0, 2, 4, 6, 8, 10, 12]
for ident, clean, color in (("named", "individual_cited", "#D85A30"), ("anonymous", "clean_anonymous", "#F0997B")):
    x, m, se = series([(k, clean if k == 0 else f"deceivers{k}_{ident}") for k in ks], "newborn reward")
    a.errorbar(x, m, yerr=se, marker="o", color=color, label=ident, capsize=3)
a.set_xlabel("coordinated liars among 16 authors"); a.set_ylabel("newborn reward per pull")
a.set_title("(b) Liars and identity, newborns", fontsize=10)
no_wiki_line(a, "newborn reward")
a = ax[1, 0]
cs = [(-0.5, "rival_info"), (-0.25, "rival_0.25"), (-0.1, "rival_0.1"), (0.0, "individual"),
      (0.1, "cited_0.1"), (0.25, "cited_0.25"), (0.5, "individual_cited"), (1.0, "cited_1.0")]
for key, color in (("honest writes", "#1D9E75"), ("newborn reward", "#534AB7")):
    x, m, se = series(cs, key)
    a.errorbar(x, m, yerr=se, marker="o", color=color, label=key, capsize=3)
a.axvline(0, color="gray", lw=0.7)
a.set_ylim(0, 1.02); a.set_xlabel("citation credit c (negative: informational rivalry)")
a.set_title("(c) Size of the credit or rivalry", fontsize=10)
a = ax[1, 1]
lifes = [PAPER_LIFE // 3, PAPER_LIFE, 3 * PAPER_LIFE]
for base, at, color in (("no_wiki", "no_wiki", "#888780"), ("notebook", "notebook_cited", "#BA7517"),
                        ("cited", "individual_cited", "#534AB7")):
    for key, ls in (("newborn reward", "-"), ("adult reward", "--")):
        x, m, se = series([(L, at if L == PAPER_LIFE else f"{base}_life{L}") for L in lifes], key)
        a.errorbar(x, m, yerr=se, marker="o", ls=ls, color=color, capsize=3,
                   label=f"{at}, {key.split()[0]}s")
a.set_xscale("log"); a.set_xticks(lifes); a.set_xticklabels(lifes)
a.set_xlabel("lifespan (updates)"); a.set_ylabel("reward per pull")
a.set_title("(d) Lifespan (solid: newborns, dashed: adults)", fontsize=10)
for a in ax.flat:
    a.legend(fontsize=7, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Robustness: population size, number of liars, size of credit or rivalry, lifespan")
fig.tight_layout()
savefig(fig, "fig7_robustness")

# Figure 8: why others' pages help newborns
fig, ax = plt.subplots(1, 3, figsize=(18, 3.9))
shown = bars(ax[0], ["no_wiki", "notebook_wiped_cited", "notebook_cited", "notebook_replicated_cited",
                     "commons1_cited", "commons3_cited", "individual_cited", "oracle"], "newborn reward")
ax[0].set_ylabel("newborn reward per pull")
ax[0].set_title("(a) What the newborn reads", fontsize=10)
bars(ax[1], ["no_wiki", "no_wiki_torso", "notebook_cited", "notebook_torso_cited", "individual_cited",
             "wiki_torso_cited"], "newborn reward", color="#BA7517")
ax[1].set_title("(b) Heirs that keep their predecessor's hidden layers", fontsize=10)
for name, color in (("no_wiki", "#888780"), ("notebook_cited", "#BA7517"), ("notebook_replicated_cited", "#F0997B"),
                    ("commons1_cited", "#85B7EB"), ("commons3_cited", "#378ADD"), ("individual_cited", "#534AB7")):
    c = life_curve(name)
    if c is not None:
        ax[2].plot(c, color=color, label=name)
ax[2].set_xlabel("updates since birth"); ax[2].set_ylabel("reward per pull")
ax[2].set_title("(c) Recovery after birth", fontsize=10)
ax[2].legend(fontsize=7, frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Why others' pages help newborns")
fig.tight_layout()
savefig(fig, "fig8_mechanism")
print("figures saved to", FIG_DIR)
"""

EXTRA2_MD = r"""## 16. Second round: inheritance, own memory, mass die-offs, identity attacks, world size

Uses the runs of `RUN_EXTRA2` (section 13) together with the earlier ones.
Saved to `wikibandits_results/paper/`: `contrasts2.csv`, `extra2.csv` and
`figures/fig9`, `fig10`. The comparisons read like those of section 15.

- **Fig. 9**:
  - (a) honest writes when authors cannot read their own page, with and without
    competition, credit and with 4 agents;
  - (b, c) what the wiki is worth (cited − no wiki) for newborns and for the whole
    population, against how fast the world changes and how long agents live;
  - (d, e) mass die-offs: newborn reward, and the share of machines whose wiki is
    right over the run (dotted lines: the die-offs);
  - (f) newborn reward against the number of machines.
- **Fig. 10**: identity attacks on 8 coordinated liars:
  - (a) newborn and population reward with fixed names, names reshuffled once or
    ten times a lifetime, anonymity, and sleepers (honest for the first half, then
    lying);
  - (b) learners' reward over the run with sleepers (dotted line: they start
    lying);
  - (c) how often learners pull the arm the liars name.
"""

EXTRA2_FIG = r"""import matplotlib.colors as mcolors

def drift_names(d, L):
    # (no wiki, wiki) settings at drift d and lifespan L; drift 2000 is the default of the earlier runs
    if d == 2000:
        return (("no_wiki", "individual_cited") if L == PAPER_LIFE else (f"no_wiki_life{L}", f"cited_life{L}"))
    return f"no_wiki_drift{d}_life{L}", f"cited_drift{d}_life{L}"

def paired(a, cells, legend, colors, ticklabels, rotation=0):
    # cells[i] = ((name, key), (name, key)): the two bars of group i
    w, tops, done = 0.38, [], [False, False]
    for j in range(2):
        for i, pair in enumerate(cells):
            n, key = pair[j]
            if (n, key) in nums and np.isfinite(nums[(n, key)][0]):
                m, se, v = nums[(n, key)]
                a.bar(i + (j - 0.5) * w, m, w * 0.95, yerr=se, color=colors[j], capsize=3,
                      label=None if done[j] else legend[j])
                a.scatter([i + (j - 0.5) * w] * len(v), v, color="k", s=8, zorder=3)
                done[j] = True
                tops.append(max([m + se] + list(v)))
    a.set_xticks(range(len(cells)))
    a.set_xticklabels(ticklabels, fontsize=8, rotation=rotation, ha="right" if rotation else "center")
    if tops:
        a.set_ylim(0, max(tops) * 1.3)                # headroom for the legend
    a.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)

CONTRASTS2 = [
    ("notebook_wiped_torso_cited", "notebook_torso_cited", "newborn reward", "heir keeps features: notebook erased vs inherited"),
    ("notebook_wiped_torso_cited", "no_wiki_torso", "newborn reward", "heir keeps features: erased notebook vs nothing"),
    ("hidden_own", "individual", "honest writes", "own page hidden, no credit"),
    ("hidden_own_cited", "individual_cited", "honest writes", "own page hidden, credit"),
    ("competitive_nocite", "individual", "honest writes", "competition, no credit"),
    ("hidden_own_competitive", "competitive_nocite", "honest writes", "own page hidden, competition, no credit"),
    ("hidden_own_competitive_cited", "competitive_cited", "honest writes", "own page hidden, competition + credit"),
    ("hidden_own_competitive_cited_N4", "competitive_N4", "honest writes", "own page hidden, competition + credit, 4 agents"),
    ("hidden_own_cited_N4", "cited_N4", "honest writes", "own page hidden, credit, 4 agents"),
    ("hidden_own_cited", "individual_cited", "newborn reward", "own page hidden: newborns"),
    ("sync_cited", "sync_no_wiki", "newborn reward", "mass die-off: wiki vs nothing"),
    ("sync_notebook_cited", "sync_no_wiki", "newborn reward", "mass die-off: notebook vs nothing"),
    ("sync_cited", "sync_wipe_cited", "newborn reward", "mass die-off: pages kept vs erased"),
    ("sync_cited", "individual_cited", "newborn reward", "wiki: mass vs staggered deaths"),
    ("sync_no_wiki", "no_wiki", "newborn reward", "no wiki: mass vs staggered deaths"),
    ("deceivers8_rotate_life", "deceivers8_named", "newborn reward", "liars' names reshuffled once a lifetime"),
    ("deceivers8_rotate_short", "deceivers8_named", "newborn reward", "liars' names reshuffled 10x a lifetime"),
    ("deceivers8_rotate_short", "deceivers8_anonymous", "newborn reward", "fast reshuffling vs anonymity"),
    ("deceivers8_sleeper_named", "deceivers8_named", "population reward", "named: sleepers vs liars from the start"),
    ("deceivers8_sleeper_anonymous", "deceivers8_sleeper_named", "population reward", "sleepers: anonymous vs named"),
    ("cited_K4", "no_wiki_K4", "newborn reward", "4 machines: wiki vs nothing"),
    ("individual_cited", "no_wiki", "newborn reward", "8 machines: wiki vs nothing"),
    ("cited_K32", "no_wiki_K32", "newborn reward", "32 machines: wiki vs nothing"),
    ("cited_K32", "notebook_cited_K32", "newborn reward", "32 machines: wiki vs notebook"),
]
for d in DRIFTS:
    for L in LIFES:
        nw, ct = drift_names(d, L)
        CONTRASTS2.append((ct, nw, "population reward", f"drift {d}, life {L}: wiki vs nothing"))
rows = []
for a, b, key, what in CONTRASTS2:
    if (a, key) in nums and (b, key) in nums:
        d, lo, hi, verdict = welch_diff(vals(a, key), vals(b, key))
        rows.append({"comparison": what, "a": a, "b": b, "measure": key, "a - b": round(d, 3),
                     "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "verdict": verdict})
contrasts2 = pd.DataFrame(rows)
contrasts2.to_csv(os.path.join(PAPER_DIR, "contrasts2.csv"), index=False)
try:
    display(contrasts2)
except NameError:
    print(contrasts2.to_string())

cols = ["newborn reward", "population reward", "adult reward", "honest writes", "wiki right", "t to 0.61"]
EXTRA2_TABLE = ["individual", "individual_cited", "competitive_nocite", "competitive_cited", "cited_N4", "competitive_N4"] \
    + list(EXTRA2) + ["no_wiki", "notebook_cited", "deceivers8_named", "deceivers8_anonymous", "no_wiki_torso",
                      "notebook_torso_cited"]
ext = summary.loc[[n for n in dict.fromkeys(EXTRA2_TABLE) if n in summary.index], [c for c in cols if c in summary.columns]]
ext.to_csv(os.path.join(PAPER_DIR, "extra2.csv"))
try:
    display(ext)
except NameError:
    print(ext.to_string())

# Figure 9: own memory, when a record pays, mass die-offs, world size
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
a = ax[0, 0]
pairs = [("individual", "hidden_own"), ("individual_cited", "hidden_own_cited"),
         ("competitive_nocite", "hidden_own_competitive"), ("competitive_cited", "hidden_own_competitive_cited"),
         ("cited_N4", "hidden_own_cited_N4"), ("competitive_N4", "hidden_own_competitive_cited_N4")]
labels = ["individual", "credit", "competitive", "competitive\n+ credit", "credit\n4 agents", "competitive\n+ credit\n4 agents"]
paired(a, [((p, "honest writes"), (q, "honest writes")) for p, q in pairs], ("own page readable", "own page hidden"),
       ("#534AB7", "#D85A30"), labels)
a.set_ylabel("share of writes that are honest")
a.set_title("(a) Authors who cannot read their own page", fontsize=10)

def heat(a, key, title):
    M = np.full((len(DRIFTS), len(LIFES)), np.nan)
    for i, d in enumerate(DRIFTS):
        for j, L in enumerate(LIFES):
            nw, ct = drift_names(d, L)
            if (nw, key) in nums and (ct, key) in nums:
                M[i, j] = nums[(ct, key)][0] - nums[(nw, key)][0]
    lim = max(0.05, np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.05
    cmap = mcolors.LinearSegmentedColormap.from_list("div", ["#D85A30", "#E8E6E0", "#534AB7"])
    im = a.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, origin="lower", aspect="auto")
    for i in range(len(DRIFTS)):
        for j in range(len(LIFES)):
            if np.isfinite(M[i, j]):
                a.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9, color="#2C2C2A")
    a.set_xticks(range(len(LIFES))); a.set_xticklabels(LIFES)
    a.set_yticks(range(len(DRIFTS))); a.set_yticklabels(DRIFTS)
    a.set_xlabel("lifespan (updates)"); a.set_ylabel("mean steps between changes of a machine")
    a.set_title(title, fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="cited − no wiki")

heat(ax[0, 1], "newborn reward", "(b) What the wiki is worth to newborns")
heat(ax[0, 2], "population reward", "(c) What the wiki is worth to the population")
a = ax[1, 0]
groups = [("nothing", "no_wiki", "sync_no_wiki"), ("notebook", "notebook_cited", "sync_notebook_cited"),
          ("wiki", "individual_cited", "sync_cited"), ("wiki, pages erased", "wipe_cited", "sync_wipe_cited")]
paired(a, [((g[1], "newborn reward"), (g[2], "newborn reward")) for g in groups],
       ("staggered deaths", "everyone dies at once"), ("#534AB7", "#D85A30"), [g[0] for g in groups])
a.set_ylabel("newborn reward per pull")
a.set_title("(d) Mass die-offs: newborns", fontsize=10)
a = ax[1, 1]
for name, color, ls in (("individual_cited", "#534AB7", "-"), ("sync_cited", "#D85A30", "-"),
                        ("sync_wipe_cited", "#F0997B", "--"), ("sync_notebook_cited", "#BA7517", ":")):
    L = logs_of(name)
    if L:
        a.plot(*smooth_xy(np.mean([l["majority_ok"] for l in L], 0), 15), color=color, ls=ls, label=name)
for g in range(1, PAPER_GENS):
    a.axvline(g * PAPER_LIFE, color="gray", ls=":", lw=0.7)
a.set_ylim(0, 1.02); a.set_xlabel("update (dotted: die-offs)"); a.set_ylabel("machines whose wiki majority is right")
a.set_title("(e) Mass die-offs: does the record survive?", fontsize=10)
a.legend(fontsize=8, frameon=False)
a = ax[1, 2]
Ks = [4, 8, 32]
for label, fmt, at8, color in (("no wiki", "no_wiki_K{}", "no_wiki", "#888780"),
                               ("notebook", "notebook_cited_K{}", "notebook_cited", "#BA7517"),
                               ("wiki", "cited_K{}", "individual_cited", "#534AB7")):
    x, m, se = series([(K, at8 if K == 8 else fmt.format(K)) for K in Ks], "newborn reward")
    a.errorbar(x, m, yerr=se, marker="o", color=color, label=label, capsize=3)
a.set_xscale("log", base=2); a.set_xticks(Ks); a.set_xticklabels(Ks)
a.set_xlabel("number of machines"); a.set_ylabel("newborn reward per pull")
a.set_title("(f) Bigger worlds", fontsize=10)
a.legend(fontsize=8, frameon=False)
for a in ax.flat:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Own memory, when a record pays, mass die-offs and world size")
fig.tight_layout()
savefig(fig, "fig9_extra")

# Figure 10: identity attacks
att = ["individual_cited", "deceivers8_named", "deceivers8_rotate_life", "deceivers8_rotate_short",
       "deceivers8_anonymous", "deceivers8_sleeper_named", "deceivers8_sleeper_anonymous"]
fig, ax = plt.subplots(1, 3, figsize=(18, 3.9))
a = ax[0]
shown = [n for n in att if (n, "newborn reward") in nums]
paired(a, [((n, "newborn reward"), (n, "population reward")) for n in shown], ("newborn reward", "population reward"),
       ("#F0997B", "#D85A30"), shown, rotation=25)
no_wiki_line(a, "newborn reward")
a.set_ylabel("learners' reward per pull")
a.set_title("(a) Reshuffled names and sleepers (dotted: newborns, no wiki)", fontsize=10)
switch = PAPER_LIFE * PAPER_GENS // 2
for a, key, ylab in ((ax[1], "raw", "learners' reward per pull"),
                     (ax[2], "agree_bad", "pulls agreeing with the liars' entries")):
    for name, color, ls in (("individual_cited", "#534AB7", "-"), ("deceivers8_named", "#D85A30", ":"),
                            ("deceivers8_anonymous", "#F0997B", ":"), ("deceivers8_sleeper_named", "#D85A30", "-"),
                            ("deceivers8_sleeper_anonymous", "#F0997B", "-")):
        L = logs_of(name)
        if not L or (key == "agree_bad" and name == "individual_cited"):
            continue
        curves = [over_time(l, key) for l in L]
        m = min(map(len, curves))
        a.plot(*smooth_xy(np.nanmean([c[:m] for c in curves], 0), 15), color=color, ls=ls, label=name)
    a.axvline(switch, color="k", ls=":", lw=1)
    a.set_xlabel("update (dotted: sleepers start lying)"); a.set_ylabel(ylab)
    a.legend(fontsize=7, frameon=False)
ax[1].set_title("(b) Sleepers: reward over the run", fontsize=10)
ax[2].set_title("(c) Sleepers: how much readers trust the liars", fontsize=10)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Identity attacks on 8 coordinated liars among 16 authors")
fig.tight_layout()
savefig(fig, "fig10_identity_attacks")
print("figures saved to", FIG_DIR)
"""

EXTRA3_MD = r"""## 17. Third round: inheritance when facts last, the honesty threshold, bottlenecks, recovery times

Uses the runs of `RUN_EXTRA3` (section 13) with the earlier ones. Saved to
`wikibandits_results/paper/`: `contrasts3.csv`, `honesty_threshold.csv`, `events.csv`,
`extra3.csv` and `figures/fig11`.

- **Predictions** (written down before these runs):
  - **P1:** if inherited notes fail only because they go stale, inherited beats erased by more than
    0.02 when facts never change;
  - **P2:** honesty switches at a negative credit when authors read their own pages, and near zero
    when they cannot;
  - **P3:** the cohort reborn at a bottleneck does better the more experienced agents survive.
- **Events** are measured, not read off figures:
  - *wiki low* is the lowest share of right majorities after a die-off or bottleneck (smoothed over 5 updates);
  - *wiki back* is the number of updates until that share is within 0.02 of its level before the event;
  - *cohort to 0.61* is the number of updates until the reborn agents' smoothed reward beats 0.61;
  - for sleepers, *low* is the learners' lowest reward after the switch, and *back* is the number of
    updates until it is within 0.02 of its level over the last third of the run;
  - *trust gone* is the number of updates until readers' agreement with the liars' entries is within
    0.05 of its final level.
- **Fig. 11:**
  - (a) what inherited notes or pages are worth to newborns (inherited minus erased) against how long
    facts last (dashed: newborns cannot write during their first 50 updates);
  - (b) honest writes against the citation credit, with own pages readable or hidden (dotted: 50%);
  - (c) newborn reward of the cohort reborn at a bottleneck, against the number of survivors;
  - (d) share of right majorities around the bottleneck.
"""

EXTRA3_FIG = r"""from scipy import stats

def mean_ci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), v.mean() - h, v.mean() + h

def first_time(x, t0, ok, k=5):
    # updates from t0 until the smoothed series first satisfies ok (NaN if never)
    y = smooth(np.asarray(x[t0:], float), k)
    hit = np.nonzero(ok(y))[0]
    return float(hit[0] + k - 1) if len(hit) else np.nan

def ci_text(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]"

rows = []
# P1: does inheritance help when facts last?
PAIRS = {"notebook, newborn from scratch": ("notebook_cited", "notebook_wiped_cited"),
         "notebook, heir keeps features": ("notebook_torso_cited", "notebook_wiped_torso_cited"),
         "mass die-off, pages kept vs erased": ("sync_cited", "sync_wipe_cited")}
DRIFT_TAGS = [("2,000", ""), ("8,000", "_drift8000"), ("never", "_static")]
for juv, juv_label in (("", ""), ("_juv", ", juvenile period")):
    for what, (a, b) in PAIRS.items():
        for label, tag in DRIFT_TAGS:
            A, B = a + tag + juv, b + tag + juv
            for key in ("newborn reward", "t to 0.61"):
                if (A, key) in nums and (B, key) in nums:
                    d, lo, hi, verdict = welch_diff(vals(A, key), vals(B, key))
                    rows.append({"test": "P1" if not juv else "juvenile", "comparison": f"{what}, facts change every {label} steps{juv_label}",
                                 "a": A, "b": B, "measure": key, "a - b": round(d, 3), "95% CI": ci_text(lo, hi), "verdict": verdict})
for key in ("newborn reward", "wiki right"):
    if ("individual_cited_juv", key) in nums and ("individual_cited", key) in nums:
        d, lo, hi, verdict = welch_diff(vals("individual_cited_juv", key), vals("individual_cited", key))
        rows.append({"test": "juvenile", "comparison": "shared wiki: juvenile period vs none", "a": "individual_cited_juv",
                     "b": "individual_cited", "measure": key, "a - b": round(d, 3), "95% CI": ci_text(lo, hi), "verdict": verdict})

# P2: the honesty threshold
READABLE = {-0.5: "rival_info", -0.25: "rival_0.25", -0.1: "rival_0.1", -0.05: "rival_0.05", -0.02: "rival_0.02",
            0.0: "individual", 0.02: "cited_0.02", 0.05: "cited_0.05", 0.1: "cited_0.1", 0.25: "cited_0.25",
            0.5: "individual_cited", 1.0: "cited_1.0"}
HIDDEN = {c: f"hidden_own_c{c:+g}" for c in (-0.1, -0.05, -0.02, 0.02, 0.05, 0.1)}
HIDDEN.update({0.0: "hidden_own", 0.5: "hidden_own_cited"})
thr = []
for pages, table in (("readable", READABLE), ("hidden", HIDDEN)):
    for c in sorted(table):
        name = table[c]
        if (name, "honest writes") not in nums:
            continue
        m, lo, hi = mean_ci(vals(name, "honest writes"))
        side = "mostly honest" if lo > 0.5 else "mostly false" if hi < 0.5 else "unclear"
        thr.append({"own page": pages, "credit c": c, "setting": name, "honest writes": round(m, 3),
                    "95% CI": f"[{lo:.3f}, {hi:.3f}]", "side of 50%": side,
                    "newborn reward": round(nums.get((name, "newborn reward"), (np.nan,))[0], 3)})
thr = pd.DataFrame(thr)
thr.to_csv(os.path.join(PAPER_DIR, "honesty_threshold.csv"), index=False)

# P3 and events: bottlenecks, mass die-offs, sleepers
ev = []
def bottleneck_measures(l):
    t0 = l["config"]["world"]["bottleneck_at"]
    c = np.asarray(l["bottleneck_cohort"], bool)
    coh = np.nanmean(l["raw"][:, c], 1)
    out = {"cohort newborn reward": cohort_reward(l, c, t0, EARLY), "cohort to 0.61": first_time(coh, t0, lambda y: y >= 0.61)}
    base, low, back = event_dip(l["majority_ok"], t0)
    out.update({"wiki before": base, "wiki low": low, "wiki back": back})
    if (~c).any():
        sb, slow, _ = event_dip(np.nanmean(l["raw"][:, ~c], 1), t0)
        out["survivors' dip"] = sb - slow
    return out

def dieoff_measures(l):
    L = l["config"]["life_updates"]
    U = len(l["majority_ok"])
    per = []
    for t0 in range(L, U, L):
        base, low, back = event_dip(l["majority_ok"], t0)
        per.append({"cohort newborn reward": float(np.nanmean(l["raw"][t0:t0 + EARLY][:, learner_mask(l)])),
                    "cohort to 0.61": first_time(over_time(l, "raw"), t0, lambda y: y >= 0.61),
                    "wiki before": base, "wiki low": low, "wiki back": back})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # events where nobody reaches 0.61
        return {k: float(np.nanmean([p[k] for p in per])) for k in per[0]} if per else {}

def sleeper_measures(l):
    t0 = l["config"]["world"]["deceive_after"] // LearnConfig().num_steps
    r = over_time(l, "raw")
    final = float(np.nanmean(r[-max(1, len(r) // 3):]))
    base, low, back = event_dip(r, t0, post=900, target=final - 0.02)
    a = over_time(l, "agree_bad")
    a_final = float(np.nanmean(a[-max(1, len(a) // 3):]))
    return {"reward before": base, "reward low": low, "reward back": back, "final reward": final,
            "trust gone": first_time(a, t0, lambda y: y <= a_final + 0.05)}

for name in PAPER_SETTINGS:
    cfg = PAPER_SETTINGS[name]
    fn = (bottleneck_measures if cfg.get("bottleneck_at", 0) else dieoff_measures if cfg.get("sync_deaths")
          else sleeper_measures if cfg.get("deceive_after", 0) else None)
    if fn is None or not logs_of(name):
        continue
    per_seed = [fn(l) for l in logs_of(name)]
    row = {"setting": name}
    for k in per_seed[0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m, lo, hi = mean_ci([p.get(k, np.nan) for p in per_seed])
        row[k] = f"{m:.3f}" if abs(m) < 2 else f"{m:.0f}"
        nums[(name, "event: " + k)] = (m, np.nan, [p.get(k, np.nan) for p in per_seed])
    ev.append(row)
events = pd.DataFrame(ev).set_index("setting") if ev else pd.DataFrame()
events.to_csv(os.path.join(PAPER_DIR, "events.csv"))

for a, b in (("bottleneck16_cited", "bottleneck16_wipe_cited"), ("bottleneck8_cited", "bottleneck16_cited"),
             ("bottleneck15_cited", "bottleneck16_cited")):
    key = "event: cohort newborn reward"
    if (a, key) in nums and (b, key) in nums:
        d, lo, hi, verdict = welch_diff(vals(a, key), vals(b, key))
        rows.append({"test": "P3", "comparison": f"{a} vs {b}", "a": a, "b": b, "measure": "cohort newborn reward",
                     "a - b": round(d, 3), "95% CI": ci_text(lo, hi), "verdict": verdict})
if ("bottleneck8_cited", "event: cohort newborn reward") in nums and ("individual_cited", "newborn reward") in nums:
    d, lo, hi, verdict = welch_diff(vals("bottleneck8_cited", "event: cohort newborn reward"), vals("individual_cited", "newborn reward"))
    rows.append({"test": "P3", "comparison": "half survive vs staggered deaths", "a": "bottleneck8_cited",
                 "b": "individual_cited", "measure": "newborn reward", "a - b": round(d, 3),
                 "95% CI": ci_text(lo, hi), "verdict": verdict})
contrasts3 = pd.DataFrame(rows)
contrasts3.to_csv(os.path.join(PAPER_DIR, "contrasts3.csv"), index=False)
cols = ["newborn reward", "population reward", "adult reward", "honest writes", "wiki right", "t to 0.61"]
ext3 = summary.loc[[n for n in EXTRA3 if n in summary.index], [c for c in cols if c in summary.columns]]
ext3.to_csv(os.path.join(PAPER_DIR, "extra3.csv"))
for title, table in (("P1 and P3 comparisons", contrasts3), ("P2 honesty threshold", thr), ("events", events),
                     ("third-round settings", ext3)):
    print("\n" + title)
    try:
        display(table)
    except NameError:
        print(table.to_string())

# Figure 11
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
for juv, ls, mk, dx in (("", "-", "o", 0.0), ("_juv", "--", "s", 0.05)):
    for (what, (A, B)), color, off in zip(PAIRS.items(), ("#BA7517", "#D85A30", "#534AB7"), (-0.12, 0, 0.12)):
        xs, ms, los, his = [], [], [], []
        for i, (label, tag) in enumerate(DRIFT_TAGS):
            if (A + tag + juv, "newborn reward") in nums and (B + tag + juv, "newborn reward") in nums:
                d, lo, hi, _ = welch_diff(vals(A + tag + juv, "newborn reward"), vals(B + tag + juv, "newborn reward"))
                xs.append(i + off + dx); ms.append(d); los.append(d - lo); his.append(hi - d)
        if xs:
            a.errorbar(xs, ms, yerr=[los, his], marker=mk, ls=ls, capsize=3, color=color,
                       label=what + (" (juvenile period)" if juv else ""))
a.axhspan(-0.02, 0.02, color="#E8E6E0", zorder=0)
a.axhline(0, color="gray", lw=0.7)
a.set_xticks(range(len(DRIFT_TAGS))); a.set_xticklabels([l for l, _ in DRIFT_TAGS])
a.set_xlabel("mean steps between changes of a machine"); a.set_ylabel("inherited − erased, newborn reward")
a.set_title("(a) Does inheritance help when facts last? (band: ±0.02)", fontsize=10)
a.legend(fontsize=8, frameon=False)
a = ax[0, 1]
for pages, table, color in (("own page readable", READABLE, "#534AB7"), ("own page hidden", HIDDEN, "#D85A30")):
    pts = [(c, table[c]) for c in sorted(table) if (table[c], "honest writes") in nums]
    if pts:
        cs = [c for c, _ in pts]
        mci = [mean_ci(vals(n, "honest writes")) for _, n in pts]
        a.errorbar(cs, [m for m, _, _ in mci], yerr=[[m - lo for m, lo, _ in mci], [hi - m for m, _, hi in mci]],
                   marker="o", capsize=3, color=color, label=pages)
a.axhline(0.5, color="gray", ls=":", lw=1); a.axvline(0, color="gray", lw=0.7)
a.set_xscale("symlog", linthresh=0.02); a.set_xticks([-0.5, -0.1, -0.02, 0, 0.02, 0.1, 0.5])
a.set_xticklabels(["−0.5", "−0.1", "−0.02", "0", "0.02", "0.1", "0.5"])
a.set_ylim(0, 1.02); a.set_xlabel("citation credit c (negative: informational rivalry)"); a.set_ylabel("honest writes")
a.set_title("(b) Where honesty switches", fontsize=10)
a.legend(fontsize=8, frameon=False)
a = ax[1, 0]
pts = [(16 - k, f"bottleneck{k}_cited") for k in (8, 12, 15, 16)]
pts = [(s, n) for s, n in pts if (n, "event: cohort newborn reward") in nums]
if pts:
    mci = [mean_ci(vals(n, "event: cohort newborn reward")) for _, n in pts]
    a.errorbar([s for s, _ in pts], [m for m, _, _ in mci], yerr=[[m - lo for m, lo, _ in mci], [hi - m for m, _, hi in mci]],
               marker="o", capsize=3, color="#534AB7", label="pages kept")
if ("bottleneck16_wipe_cited", "event: cohort newborn reward") in nums:
    m, lo, hi = mean_ci(vals("bottleneck16_wipe_cited", "event: cohort newborn reward"))
    a.errorbar([0.15], [m], yerr=[[m - lo], [hi - m]], marker="s", capsize=3, color="#D85A30", ls="none", label="pages erased")
for ref, label in (("individual_cited", "staggered deaths"), ("sync_cited", "repeated mass die-offs"), ("no_wiki", "no wiki")):
    if (ref, "newborn reward") in nums:
        a.axhline(nums[(ref, "newborn reward")][0], color="gray", ls=":", lw=1)
        a.text(8.1, nums[(ref, "newborn reward")][0], " " + label, fontsize=7, color="gray", va="center")
a.set_xticks([0, 1, 4, 8]); a.set_xlim(-0.5, 10.5)
a.set_xlabel("experienced survivors (of 16)"); a.set_ylabel("newborn reward of the reborn cohort")
a.set_title("(c) Bottlenecks: how many survivors does the record need?", fontsize=10)
a.legend(fontsize=8, frameon=False, loc="lower right")
a = ax[1, 1]
for k, color in zip((8, 12, 15, 16), ("#AFA9EC", "#7F77DD", "#534AB7", "#26215C")):
    L = logs_of(f"bottleneck{k}_cited")
    if L:
        t0 = L[0]["config"]["world"]["bottleneck_at"]
        y = np.mean([l["majority_ok"][max(0, t0 - 50):t0 + 200] for l in L], 0)
        x, ys = smooth_xy(y, 5)
        a.plot(x - min(50, t0), ys, color=color, label=f"{k} of 16 die")
L = logs_of("bottleneck16_wipe_cited")
if L:
    t0 = L[0]["config"]["world"]["bottleneck_at"]
    x, ys = smooth_xy(np.mean([l["majority_ok"][max(0, t0 - 50):t0 + 200] for l in L], 0), 5)
    a.plot(x - min(50, t0), ys, color="#D85A30", ls="--", label="16 die, pages erased")
a.axvline(0, color="k", ls=":", lw=1)
a.set_ylim(0, 1.02); a.set_xlabel("updates since the bottleneck"); a.set_ylabel("machines whose wiki majority is right")
a.set_title("(d) The record around a bottleneck", fontsize=10)
a.legend(fontsize=8, frameon=False, loc="lower right")
for a in ax.flat:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Inheritance when facts last, the honesty threshold, and bottlenecks")
fig.tight_layout()
savefig(fig, "fig11_journal")
print("figures saved to", FIG_DIR)
"""

EXTRA5_MD = r"""## 18. Evolution of the juvenile period

Uses the runs of `RUN_EXTRA5` (section 13). Saved to `wikibandits_results/paper/`:
`evolution.csv`, `juvenile_dose.csv` and `figures/fig12`.

**How the evolution works:**
- Each agent in each of the 256 world copies has its own juvenile period: the number of updates after
  birth during which it may not write.
- When an agent dies, its heir copies the juvenile period of the more successful of two random world copies,
  then adds a random change of about 10 updates. "More successful" means the higher reward over the life that
  just ended, comparing the same slot (and so the same brain) across copies.
- With group selection (mass die-offs only), a whole world copy instead copies all the juvenile periods of
  the fitter of two random copies, judged by its agents' mean reward.
- Each setting starts once with no juvenile period and once with 150 updates, and runs for 12 lifetimes.

**Predictions** (written down before these runs, in the project note "ALife journal plan"):
- with private notebooks and facts that never change, a long juvenile period evolves from both starts;
- it is shorter when facts change;
- it evolves fastest when heirs keep their predecessor's learned features;
- with a shared wiki kept by staggered contemporaries, it stays short or shrinks;
- after mass die-offs it evolves further under group selection than under individual selection, because one
  agent's silence mostly protects a record that others read.

**Fig. 12:**
- (a, b) mean juvenile period over the run (solid: starting from none; dashed: starting from 150 updates);
- (c) the juvenile period at the end of the run. Black marks show what mutation alone would give: with
  random parents, the floor at 0 makes small juvenile periods drift upwards (about 22 updates after 11
  deaths when starting from 0). Evolution means a departure from this neutral expectation;
- (d) what inherited notes or pages are worth to newborns (inherited minus erased) against a fixed juvenile
  period, when facts never change.

**Fig. 13** (from `RUN_EXTRA6`): juvenile periods over 36 lifetimes, starting from none and from 150 updates.
"""

EXTRA5_FIG = r"""EVO_LABELS = {"notebook_static": "notebook, facts never change",
              "notebook": "notebook, facts change every 2,000 steps",
              "notebook_torso_static": "notebook, heir keeps features, facts never change",
              "wiki": "shared wiki, staggered deaths",
              "sync_static": "mass die-offs, facts never change, individual selection",
              "sync_static_group": "mass die-offs, facts never change, group selection"}
EVO_COLORS = {"notebook_static": "#BA7517", "notebook": "#EF9F27", "notebook_torso_static": "#D85A30",
              "wiki": "#888780", "sync_static": "#7F77DD", "sync_static_group": "#26215C"}
EVO_SHORT = {"notebook_static": "notebook\nfrozen", "notebook": "notebook\ndrift 2,000",
             "notebook_torso_static": "notebook, heir\nkeeps features", "wiki": "shared wiki\nstaggered",
             "sync_static": "die-offs\nindividual sel.", "sync_static_group": "die-offs\ngroup sel."}
J0S = (0, PAPER_LIFE // 2)

def final_J(l):
    return float(l["J_mean"][-PAPER_LIFE:].mean())         # mean over slots and copies, last lifetime

def neutral_curve(j0, kmax, sigma, E, reps=50, seed=0):
    # expected mean juvenile period after k deaths when parents are picked at random (no selection):
    # mutation plus the floor at 0 alone make it drift upwards from small values
    rng = np.random.default_rng(seed)
    out = np.zeros(kmax + 1)
    for _ in range(reps):
        J = np.full(E, float(j0))
        out[0] += J.mean()
        for k in range(1, kmax + 1):
            J = np.clip(np.round(J[rng.integers(E, size=E)] + rng.normal(0, sigma, E)), 0, 2 * PAPER_LIFE)
            out[k] += J.mean()
    return out / reps

def neutral_J(l):
    # the neutral expectation for this run, given how many times each slot's agent has died so far
    w = l["config"]["world"]
    nc = neutral_curve(w["juvenile_updates"], int(l["gen"].max()), w["juvenile_sigma"], l["config"]["learn"]["num_envs"])
    return nc[l["gen"]].mean(1)

rows = []
for base in EVO_BASES:
    for j0 in J0S:
        name = f"evo_{base}_from{j0}"
        L = logs_of(name)
        if not L:
            continue
        m, lo, hi = mean_ci([final_J(l) for l in L])
        neu = [float(neutral_J(l)[-PAPER_LIFE:].mean()) for l in L]
        dm, dlo, dhi = mean_ci([final_J(l) - n for l, n in zip(L, neu)])
        rows.append({"setting": EVO_LABELS[base], "start": j0, "final juvenile period": round(m, 1),
                     "95% CI": f"[{lo:.1f}, {hi:.1f}]", "neutral expectation": round(float(np.mean(neu)), 1),
                     "selection effect": round(dm, 1), "selection 95% CI": f"[{dlo:+.1f}, {dhi:+.1f}]",
                     "newborn reward": round(nums.get((name, "newborn reward"), (np.nan,))[0], 3),
                     "population reward": round(nums.get((name, "population reward"), (np.nan,))[0], 3)})
LONG_LABELS = {"notebook": "notebook, facts change every 2,000 steps",
               "notebook_drift8000": "notebook, facts change every 8,000 steps",
               "notebook_torso": "heir keeps features, facts change every 2,000 steps",
               "notebook_torso_drift8000": "heir keeps features, facts change every 8,000 steps"}
LONG_BASES_SHOWN = [b for b in LONG_LABELS if any(logs_of(f"evo_{b}_from{j0}_long") for j0 in J0S)]
for base in LONG_BASES_SHOWN:
    for j0 in J0S:
        name = f"evo_{base}_from{j0}_long"
        L = logs_of(name)
        if not L:
            continue
        m, lo, hi = mean_ci([final_J(l) for l in L])
        neu = [float(neutral_J(l)[-PAPER_LIFE:].mean()) for l in L]
        dm, dlo, dhi = mean_ci([final_J(l) - n for l, n in zip(L, neu)])
        rows.append({"setting": LONG_LABELS[base] + " (36 lifetimes)", "start": j0, "final juvenile period": round(m, 1),
                     "95% CI": f"[{lo:.1f}, {hi:.1f}]", "neutral expectation": round(float(np.mean(neu)), 1),
                     "selection effect": round(dm, 1), "selection 95% CI": f"[{dlo:+.1f}, {dhi:+.1f}]",
                     "newborn reward": round(nums.get((name, "newborn reward"), (np.nan,))[0], 3),
                     "population reward": round(nums.get((name, "population reward"), (np.nan,))[0], 3)})
evolution = pd.DataFrame(rows)
evolution.to_csv(os.path.join(PAPER_DIR, "evolution.csv"), index=False)

DOSE_PAIRS = {"notebook": ("notebook_cited_static", "notebook_wiped_cited_static"),
              "mass die-off": ("sync_cited_static", "sync_wipe_cited_static")}
def juv_name(base, J):
    return base if J == 0 else (base + "_juv" if J == PAPER_LIFE // 6 else f"{base}_juv{J}")
DOSES = [0, PAPER_LIFE // 30, PAPER_LIFE // 12, PAPER_LIFE // 6, PAPER_LIFE // 3]
dose = []
for what, (a, b) in DOSE_PAIRS.items():
    for J in DOSES:
        A, B = juv_name(a, J), juv_name(b, J)
        if (A, "newborn reward") in nums and (B, "newborn reward") in nums:
            d, lo, hi, verdict = welch_diff(vals(A, "newborn reward"), vals(B, "newborn reward"))
            dose.append({"record": what, "juvenile period": J, "inherited": round(nums[(A, "newborn reward")][0], 3),
                         "erased": round(nums[(B, "newborn reward")][0], 3), "inherited - erased": round(d, 3),
                         "95% CI": ci_text(lo, hi), "verdict": verdict})
dose = pd.DataFrame(dose)
dose.to_csv(os.path.join(PAPER_DIR, "juvenile_dose.csv"), index=False)
for title, table in (("evolved juvenile periods", evolution), ("fixed juvenile periods, facts never change", dose)):
    print("\n" + title)
    try:
        display(table)
    except NameError:
        print(table.to_string())

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for a, bases, title in ((ax[0, 0], ("notebook_static", "notebook", "notebook_torso_static"), "(a) Private notebooks"),
                        (ax[0, 1], ("wiki", "sync_static", "sync_static_group"), "(b) Shared wiki and mass die-offs")):
    for base in bases:
        for j0, ls in zip(J0S, ("-", "--")):
            L = logs_of(f"evo_{base}_from{j0}")
            if L:
                y = np.mean([l["J_mean"].mean(1) for l in L], 0)
                a.plot(np.arange(len(y)), y, color=EVO_COLORS[base], ls=ls, label=EVO_LABELS[base] if j0 == 0 else None)
                if base == bases[0]:
                    a.plot(np.arange(len(y)), neutral_J(L[0]), color="gray", lw=0.8, ls=ls,
                           label="no selection (expected)" if j0 == 0 else None)
    a.set_xlabel("update"); a.set_ylabel("mean juvenile period (updates)")
    a.set_title(title + " (solid: from none; dashed: from 150)", fontsize=10)
    a.legend(fontsize=7, frameon=False)
a = ax[1, 0]
w = 0.38
names = [b for b in EVO_BASES if any(logs_of(f"evo_{b}_from{j0}") for j0 in J0S)]
for j, (j0, color) in enumerate(zip(J0S, ("#534AB7", "#D85A30"))):
    for i, base in enumerate(names):
        L = logs_of(f"evo_{base}_from{j0}")
        if L:
            m, lo, hi = mean_ci([final_J(l) for l in L])
            a.bar(i + (j - 0.5) * w, m, w * 0.95, yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None, color=color,
                  capsize=3, label=(f"started at {j0}") if i == 0 else None)
            a.plot([i + (j - 0.5) * w - w * 0.45, i + (j - 0.5) * w + w * 0.45], [np.mean([neutral_J(l)[-PAPER_LIFE:].mean() for l in L])] * 2,
                   color="k", lw=1.5, label="no selection (expected)" if (i == 0 and j == 0) else None)
a.set_xticks(range(len(names))); a.set_xticklabels([EVO_SHORT[b] for b in names], fontsize=8)
a.set_ylabel("juvenile period at the end (updates)")
a.set_title("(c) Evolved juvenile periods (black: expected without selection)", fontsize=10)
top = a.get_ylim()[1]
a.set_ylim(0, top * 1.25)
a.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
a = ax[1, 1]
for what, color in zip(DOSE_PAIRS, ("#BA7517", "#534AB7")):
    sub = dose[dose["record"] == what] if len(dose) else dose
    if len(sub):
        lo_hi = [tuple(float(x) for x in s.strip("[]").split(", ")) for s in sub["95% CI"]]
        y = sub["inherited - erased"].to_numpy()
        a.errorbar(sub["juvenile period"], y, yerr=[[v - l for v, (l, h) in zip(y, lo_hi)], [h - v for v, (l, h) in zip(y, lo_hi)]],
                   marker="o", capsize=3, color=color, label=what)
a.axhspan(-0.02, 0.02, color="#E8E6E0", zorder=0); a.axhline(0, color="gray", lw=0.7)
a.set_xlabel("fixed juvenile period (updates)"); a.set_ylabel("inherited − erased, newborn reward")
a.set_title("(d) Inheritance against the length of the juvenile period", fontsize=10)
a.legend(fontsize=8, frameon=False)
for a in ax.flat:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Evolution of the juvenile period")
fig.tight_layout()
savefig(fig, "fig12_juvenile_evolution")

if LONG_BASES_SHOWN:                                  # Figure 13: 36 lifetimes, both starts
    fig, ax = plt.subplots(1, len(LONG_BASES_SHOWN), figsize=(4.6 * len(LONG_BASES_SHOWN), 3.9), squeeze=False)
    for a, base in zip(ax[0], LONG_BASES_SHOWN):
        for j0, color in zip(J0S, ("#534AB7", "#D85A30")):
            L = logs_of(f"evo_{base}_from{j0}_long")
            if not L:
                continue
            ys = np.array([l["J_mean"].mean(1) for l in L])
            for y in ys:
                a.plot(y, color=color, lw=0.6, alpha=0.4)
            a.plot(ys.mean(0), color=color, lw=1.8, label=f"started at {j0}")
            a.plot(neutral_J(L[0]), color="gray", lw=0.8, ls="--", label="no selection (expected)" if j0 == 0 else None)
        a.set_title(LONG_LABELS[base], fontsize=9)
        a.set_xlabel("update"); a.set_ylabel("mean juvenile period (updates)")
        a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=7, frameon=False)
    fig.suptitle("Juvenile periods over 36 lifetimes (thin lines: single runs)")
    fig.tight_layout()
    savefig(fig, "fig13_juvenile_long")
print("figures saved to", FIG_DIR)
"""

EXTRA7_MD = r"""## 19. The reviewer's tests: write prior, fixed juvenile periods, hysteresis

Uses the runs of `RUN_EXTRA7` (section 13). Saved to `wikibandits_results/paper/`:
`keep_prior.csv`, `fitness_curves.csv`, `hysteresis.csv`, `review_verdicts.csv` and `figures/fig14`, `fig15`.

**Predictions** (written down before these runs, in the project note "ALife journal plan", 12 Nov 2026):

*K, write prior.* Newborns start keeping their entry with probability 0.97 or 0.99 instead of 0.9.
- K1: inherited minus erased newborn reward rises with the keep prior, for the private notebook, the notebook
  whose heir keeps its features, and the pages left after a mass die-off (facts never change).
- K2: at 0.99 it stays below the gain from a 50-update juvenile period.
- K3: in the shared wiki with staggered deaths, newborn reward, adult reward and honest writes each stay
  within ±0.02 of the 0.9 prior.

*F, fixed juvenile periods.* J fixed at 0, 10, 25, 50 or 100 updates in the four 36-lifetime settings, for
6 lifetimes. Fitness = mean reward per pull over completed lives whose midpoint lies in the second half of the run.
- F1: in every setting the best fixed J is above zero and beats J = 0 by more than the 95% CI.
- F2: the best fixed J is at least as long when facts change every 8,000 steps as every 2,000.
- F3: with facts changing every 2,000 steps, the best fixed J is longer than the evolved one.

*H, hysteresis.* The credit is +0.5 (honest start) or −0.5 (lying start) for 3 lifetimes, then switches to the
target for 6 more. Measured over the last third of the run.
- H1: for each target, honest writes differ by less than 0.05 between the two starts.
- H2: within each run the honest-write share across the 256 world copies has one mode and an SD below 0.1.
- H3: final honest writes are within 0.05 of the constant-credit runs at the same credit.

A prediction "holds" when the 95% CI lies inside the stated bound, "fails" when it lies outside, and is
"unclear" otherwise. With 3 seeds the intervals are wide.

**Fig. 14:** (a) inheritance against the keep prior (open markers: the gain from a 50-update juvenile period);
(b) fitness against the fixed juvenile period (arrows: the evolved value after 36 lifetimes);
(c) honest writes after the switch, from both starts, against the credit (black: constant credit);
(d) honest writes in each of the 256 world copies (seed 0).
**Fig. 15:** honest writes over time from both starts; the dotted line marks the switch.
"""

EXTRA7_FIG = r"""from scipy import stats

def life_fitness(l):
    # mean reward per pull of each completed life whose midpoint lies in the second half of the run
    gen, raw = l["gen"], l["raw"]
    U = len(raw)
    out = []
    for i in np.nonzero(learner_mask(l))[0]:
        for g in np.unique(gen[:, i]):
            idx = np.nonzero(gen[:, i] == g)[0]
            if g >= 1 and idx[-1] < U - 1 and (idx[0] + idx[-1]) / 2 >= U / 2:
                out.append(float(np.nanmean(raw[idx, i])))
    return out

def bound_verdict(lo, hi, bound):
    # does the 95% CI of a difference lie inside +-bound?
    if not np.isfinite(lo):
        return "-"
    return "holds" if (lo > -bound and hi < bound) else "fails" if (lo >= bound or hi <= -bound) else "unclear"

def n_modes(x, rel=0.1):
    # number of peaks of a kernel density estimate on [0, 1] (peaks under 10% of the highest are ignored)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or x.std() < 1e-6:
        return 1
    try:
        y = stats.gaussian_kde(x)(np.linspace(0, 1, 401))
    except np.linalg.LinAlgError:
        return 1
    y = np.concatenate([[-1.0], y, [-1.0]])
    peaks = [k for k in range(1, len(y) - 1) if y[k] > y[k - 1] and y[k] >= y[k + 1] and y[k] >= rel * y.max()]
    return len(peaks)

verdicts = []
def verdict(test, what, ok, detail):
    verdicts.append({"prediction": test, "case": what, "result": ok, "detail": detail})

# K: the write prior
KEEP_PAIRS = {"notebook, fresh heir": ("notebook_cited_static", "notebook_wiped_cited_static"),
              "notebook, heir keeps features": ("notebook_torso_cited_static", "notebook_wiped_torso_cited_static"),
              "mass die-off pages": ("sync_cited_static", "sync_wipe_cited_static")}
KEEP_LEVELS = (("0.90", ""), ("0.97", "_keep97"), ("0.99", "_keep99"))
kp, gaps, juv_gain = [], {}, {}
for what, (a, b) in KEEP_PAIRS.items():
    for keep, sfx in KEEP_LEVELS:
        A, B = a + sfx, b + sfx
        if (A, "newborn reward") in nums and (B, "newborn reward") in nums:
            d, lo, hi, v = welch_diff(vals(A, "newborn reward"), vals(B, "newborn reward"))
            gaps[(what, keep)] = (d, lo, hi)
            kp.append({"record": what, "keep prior": keep, "measure": "newborn reward, inherited vs erased",
                       "value": round(nums[(A, "newborn reward")][0], 3), "reference": round(nums[(B, "newborn reward")][0], 3),
                       "difference": round(d, 3), "95% CI": ci_text(lo, hi), "verdict": v})
    A, B = a + "_juv", b + "_juv"                       # 50-update juvenile period, keep prior 0.9
    if (A, "newborn reward") in nums and (B, "newborn reward") in nums:
        d, lo, hi, v = welch_diff(vals(A, "newborn reward"), vals(B, "newborn reward"))
        juv_gain[what] = (d, lo, hi)
        kp.append({"record": what, "keep prior": f"0.90 + juvenile period {PAPER_LIFE // 6}", "measure": "newborn reward, inherited vs erased",
                   "value": round(nums[(A, "newborn reward")][0], 3), "reference": round(nums[(B, "newborn reward")][0], 3),
                   "difference": round(d, 3), "95% CI": ci_text(lo, hi), "verdict": v})
for keep, sfx in KEEP_LEVELS[1:]:
    for key in ("newborn reward", "adult reward", "honest writes"):
        A, B = "individual_cited" + sfx, "individual_cited"
        if (A, key) in nums and (B, key) in nums:
            d, lo, hi, v = welch_diff(vals(A, key), vals(B, key))
            kp.append({"record": "shared wiki, staggered deaths", "keep prior": keep, "measure": key + " (vs keep 0.90)",
                       "value": round(nums[(A, key)][0], 3), "reference": round(nums[(B, key)][0], 3),
                       "difference": round(d, 3), "95% CI": ci_text(lo, hi), "verdict": v})
            verdict("K3", f"shared wiki, keep {keep}, {key}", "holds" if v.startswith("same") else
                    ("fails" if v == "different" else "unclear"), f"{d:+.3f} {ci_text(lo, hi)}")
keep_prior = pd.DataFrame(kp)
keep_prior.to_csv(os.path.join(PAPER_DIR, "keep_prior.csv"), index=False)
for what in KEEP_PAIRS:
    g = [gaps.get((what, keep), (np.nan,))[0] for keep, _ in KEEP_LEVELS]
    if all(np.isfinite(g)):
        verdict("K1", what, "holds" if g[0] < g[1] < g[2] else "fails", " < ".join(f"{x:+.3f}" for x in g))
    if (what, "0.99") in gaps and what in juv_gain:
        g99, jg = gaps[(what, "0.99")][0], juv_gain[what][0]
        verdict("K2", what, "holds" if g99 < jg else "fails", f"keep 0.99: {g99:+.3f}; juvenile period 50: {jg:+.3f}")

# F: fixed juvenile periods
fc, best = [], {}
for base in LONG_BASES:
    per_J = {}
    for J in FIX_JS:
        name = fix_name(base, J)
        L = logs_of(name)
        if not L:
            continue
        fit = [np.mean(life_fitness(l)) for l in L]
        per_J[J] = fit
        m, lo, hi = mean_ci(fit)
        fc.append({"setting": LONG_LABELS[base], "juvenile period": J, "run": name, "fitness": round(m, 4),
                   "95% CI": f"[{lo:.4f}, {hi:.4f}]", "lives per seed": round(np.mean([len(life_fitness(l)) for l in L])),
                   "newborn reward": round(nums.get((name, "newborn reward"), (np.nan,))[0], 3),
                   "population reward": round(nums.get((name, "population reward"), (np.nan,))[0], 3)})
    if 0 in per_J and len(per_J) > 1:
        Jb = max(per_J, key=lambda J: np.mean(per_J[J]))
        best[base] = Jb
        if Jb == 0:
            verdict("F1", LONG_LABELS[base], "fails", "best fixed juvenile period is 0")
        else:
            d, lo, hi, _ = welch_diff(per_J[Jb], per_J[0])
            verdict("F1", LONG_LABELS[base], "holds" if lo > 0 else "unclear",
                    f"best J = {Jb}; fitness gain over J = 0: {d:+.4f} {ci_text(lo, hi)}")
fitness_curves = pd.DataFrame(fc)
fitness_curves.to_csv(os.path.join(PAPER_DIR, "fitness_curves.csv"), index=False)
for fast, slow in (("notebook", "notebook_drift8000"), ("notebook_torso", "notebook_torso_drift8000")):
    if fast in best and slow in best:
        verdict("F2", LONG_LABELS[fast].split(",")[0], "holds" if best[slow] >= best[fast] else "fails",
                f"best J: {best[fast]} (2,000 steps), {best[slow]} (8,000 steps)")
evolved = {}
for base in LONG_BASES:
    Ls = [l for j0 in J0S for l in logs_of(f"evo_{base}_from{j0}_long")]
    if Ls:
        evolved[base] = float(np.mean([final_J(l) for l in Ls]))
for base in ("notebook", "notebook_torso"):
    if base in best and base in evolved:
        verdict("F3", LONG_LABELS[base], "holds" if best[base] > evolved[base] else "fails",
                f"best fixed J = {best[base]}; evolved J = {evolved[base]:.1f}")

# H: hysteresis near the honesty threshold
CONST_C = {"readable": {-0.05: "rival_0.05", -0.02: "rival_0.02", 0.0: "individual"},
           "hidden": {-0.02: "hidden_own_c-0.02", 0.0: "hidden_own", 0.02: "hidden_own_c+0.02"}}
hy, hrows = [], {}
for pages, targets in HYST_TARGETS.items():
    for c in targets:
        const = CONST_C[pages].get(c)
        cv = vals(const, "honest writes") if const else []
        for start in HYST_STARTS:
            name = hyst_name(pages, c, start)
            L = logs_of(name)
            if not L:
                continue
            v = vals(name, "honest writes")
            m, lo, hi = mean_ci(v)
            copies = [l["honest_by_copy"] for l in L if "honest_by_copy" in l]
            sds = [float(np.nanstd(x)) for x in copies]
            modes = [n_modes(x) for x in copies]
            d, dlo, dhi, _ = welch_diff(v, cv) if len(cv) else (np.nan, np.nan, np.nan, "-")
            hrows[(pages, c, start)] = v
            hy.append({"own page": pages, "credit after switch": c, "start": start, "run": name,
                       "honest writes": round(m, 3), "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                       "per seed": ", ".join(f"{x:.3f}" for x in v),
                       "copy SD (max over seeds)": round(max(sds), 3) if sds else np.nan,
                       "modes (max over seeds)": max(modes) if modes else np.nan,
                       "copies above 0.5": round(float(np.mean([np.nanmean(x > 0.5) for x in copies])), 3) if copies else np.nan,
                       "constant credit": round(float(np.mean(cv)), 3) if len(cv) else np.nan,
                       "minus constant": round(d, 3), "minus constant 95% CI": ci_text(dlo, dhi)})
            if copies:
                verdict("H2", f"{pages}, c = {c:+g}, from {start}",
                        "holds" if max(sds) < 0.1 and max(modes) == 1 else "fails",
                        f"copy SD up to {max(sds):.3f}, up to {max(modes)} mode(s)")
            if len(cv):
                verdict("H3", f"{pages}, c = {c:+g}, from {start}", bound_verdict(dlo, dhi, 0.05),
                        f"{d:+.3f} {ci_text(dlo, dhi)} vs constant credit")
        if (pages, c, "honest") in hrows and (pages, c, "lying") in hrows:
            d, lo, hi, _ = welch_diff(hrows[(pages, c, "honest")], hrows[(pages, c, "lying")])
            verdict("H1", f"{pages}, c = {c:+g}", bound_verdict(lo, hi, 0.05), f"honest start - lying start: {d:+.3f} {ci_text(lo, hi)}")
hysteresis = pd.DataFrame(hy)
hysteresis.to_csv(os.path.join(PAPER_DIR, "hysteresis.csv"), index=False)
review_verdicts = pd.DataFrame(verdicts)
review_verdicts.to_csv(os.path.join(PAPER_DIR, "review_verdicts.csv"), index=False)
for title, table in (("write prior", keep_prior), ("fixed juvenile periods", fitness_curves),
                     ("hysteresis", hysteresis), ("verdicts on the registered predictions", review_verdicts)):
    print("\n" + title)
    try:
        display(table)
    except NameError:
        print(table.to_string())

# Figure 14
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
a = ax[0, 0]
KEEP_COLORS = {"notebook, fresh heir": "#BA7517", "notebook, heir keeps features": "#D85A30", "mass die-off pages": "#534AB7"}
for k, (what, color) in enumerate(KEEP_COLORS.items()):
    xs = [i for i, (keep, _) in enumerate(KEEP_LEVELS) if (what, keep) in gaps]
    if xs:
        g = np.array([gaps[(what, KEEP_LEVELS[i][0])] for i in xs])
        a.errorbar(np.array(xs) + (k - 1) * 0.06, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]], marker="o",
                   capsize=3, color=color, label=what)
    if what in juv_gain:
        d, lo, hi = juv_gain[what]
        a.errorbar([3 + (k - 1) * 0.06], [d], yerr=[[d - lo], [hi - d]], marker="o", mfc="white", capsize=3, color=color)
a.axhspan(-0.02, 0.02, color="#E8E6E0", zorder=0); a.axhline(0, color="gray", lw=0.7)
a.set_xticks(range(4)); a.set_xticklabels(["keep 0.90", "keep 0.97", "keep 0.99", "keep 0.90\n+ juvenile 50"])
a.set_ylabel("inherited − erased, newborn reward")
a.set_title("(a) Inheritance against newborns' keep prior (facts never change)", fontsize=10)
a.legend(fontsize=8, frameon=False)
a = ax[0, 1]
FIX_COLORS = {"notebook": "#EF9F27", "notebook_drift8000": "#BA7517", "notebook_torso": "#D85A30", "notebook_torso_drift8000": "#993C1D"}
for base, color in FIX_COLORS.items():
    sub = fitness_curves[fitness_curves["setting"] == LONG_LABELS[base]] if len(fitness_curves) else fitness_curves
    if len(sub):
        lo_hi = np.array([[float(x) for x in s.strip("[]").split(", ")] for s in sub["95% CI"]])
        y = sub["fitness"].to_numpy()
        a.errorbar(sub["juvenile period"], y, yerr=[y - lo_hi[:, 0], lo_hi[:, 1] - y], marker="o", capsize=3,
                   color=color, label=LONG_LABELS[base])
        if base in evolved:
            a.annotate("", xy=(evolved[base], y.min()), xytext=(evolved[base], y.min() - 0.25 * (np.ptp(y) + 1e-3)),
                       arrowprops=dict(arrowstyle="->", color=color))
a.set_xlabel("fixed juvenile period (updates)"); a.set_ylabel("fitness (reward per pull, completed lives)")
a.set_title("(b) Fitness against a fixed juvenile period (arrows: evolved)", fontsize=10)
a.legend(fontsize=7, frameon=False)
a = ax[1, 0]
for pages, mk, dx in (("readable", "o", -0.003), ("hidden", "s", 0.003)):
    for start, color in (("honest", "#1D9E75"), ("lying", "#D85A30")):
        pts = [(c, hrows[(pages, c, start)]) for c in HYST_TARGETS[pages] if (pages, c, start) in hrows]
        for c, v in pts:
            m, lo, hi = mean_ci(v)
            a.errorbar([c + dx], [m], yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None, marker=mk, color=color,
                       capsize=3, ls="none", label=f"own page {pages}, {start} start" if c == pts[0][0] else None)
        for c in HYST_TARGETS[pages]:
            const = CONST_C[pages].get(c)
            if const and (const, "honest writes") in nums:
                a.plot([c + dx - 0.002, c + dx + 0.002], [nums[(const, "honest writes")][0]] * 2, color="k", lw=1.5)
a.axhline(0.5, color="gray", lw=0.7, ls=":")
a.set_xlabel("credit after the switch"); a.set_ylabel("honest writes (last third)"); a.set_ylim(0, 1.02)
a.set_title("(c) Both starts end at the same honesty? (black: constant credit)", fontsize=10)
a.legend(fontsize=7, frameon=False)
a = ax[1, 1]
labels = []
for pages in HYST_TARGETS:
    for c in HYST_TARGETS[pages]:
        for start, color in (("honest", "#1D9E75"), ("lying", "#D85A30")):
            L = logs_of(hyst_name(pages, c, start))
            if L and "honest_by_copy" in L[0]:
                x = L[0]["honest_by_copy"]
                i = len(labels)
                a.scatter(i + np.random.default_rng(i).uniform(-0.3, 0.3, len(x)), x, s=3, color=color, alpha=0.5)
                labels.append(f"{pages[:4]} {c:+g}\n{start}")
a.set_xticks(range(len(labels))); a.set_xticklabels(labels, fontsize=6)
a.set_ylabel("honest writes per world copy"); a.set_ylim(0, 1.02)
a.set_title("(d) Honesty in each of the 256 world copies (seed 0)", fontsize=10)
for a in ax.flat:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("The reviewer's tests")
fig.tight_layout()
savefig(fig, "fig14_review_tests")

# Figure 15: honest writes over time from both starts
fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for r, pages in enumerate(HYST_TARGETS):
    for k, c in enumerate(HYST_TARGETS[pages]):
        a = ax[r, k]
        for start, color in (("honest", "#1D9E75"), ("lying", "#D85A30")):
            for j, l in enumerate(logs_of(hyst_name(pages, c, start))):
                x, y = smooth_xy(over_time(l, "honest_writes"), 25)
                a.plot(x, y, color=color, lw=0.9, alpha=0.8, label=f"{start} start" if j == 0 else None)
        const = CONST_C[pages].get(c)
        if const and (const, "honest writes") in nums:
            a.axhline(nums[(const, "honest writes")][0], color="gray", ls="--", lw=0.8, label="constant credit")
        a.axvline(SWITCH_AT // STEPS, color="k", ls=":", lw=0.8)
        a.set_title(f"own page {pages}, credit {c:+g} after the switch", fontsize=9)
        a.set_xlabel("update")
        a.spines[["top", "right"]].set_visible(False)
    ax[r, 0].set_ylabel("honest writes")
ax[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("Honest writes from an honest and a lying start (dotted: the credit switches to the target)")
fig.tight_layout()
savefig(fig, "fig15_hysteresis")
print("figures saved to", FIG_DIR)
"""

EXTRA8_MD = r"""## 20. Second round of review tests

Uses `RUN_EXTRA8` and `RUN_EXTRA_SEEDS` (section 13). Saved to `wikibandits_results/paper/`:
`stats_verdicts.csv` (every key comparison with exclusive verdicts), `review_verdicts2.csv` (the registered
predictions R, I, G, F, M and the history test re-judged), `seats.csv`, `immortal_history.csv`,
`group_selection_wiki.csv`, `fitness_curves_full.csv`, `own_memory.csv`, `ideal_reader.csv`, and
`figures/fig16`, `fig17`.

**Predictions** (written down before these runs, in the project note "ALife journal plan", 12 Nov 2026, 19:15):
- R1: with liars in seats drawn per world copy, newborns earn within ±0.05 of anonymous authors and more than 0.2
  below "own entry first, fixed order"; R2: "own entry first, fixed order" is within ±0.05 of fixed named authors;
  R3: without liars, per-copy seats cost nothing (within ±0.02 of the shared wiki).
- I1: with learners that never die, honest writes from the two starts end less than 0.05 apart for every credit;
  I2: they take longer to converge than with deaths for at least 4 of the 6 credits.
- G1: in the shared wiki, from 150, group selection ends at a shorter juvenile period than individual selection
  (non-overlapping 95% CIs); G2: from 0, group selection's selection effect is negative and individual selection's
  is near neutral; G3: from 150, newborns earn more under group selection.
- F4: for fresh heirs with facts changing every 8,000 steps, fitness peaks at 100 updates or later;
  F5: with facts changing every 2,000 steps, the evolved lengths lie within the range of fixed lengths whose
  fitness is within 0.005 of the best.
- M1: agents with their own memory reach adult reward 0.75 without a record; M2: the shared wiki still adds at
  least 0.2 for newborns; M3: it adds less than 0.05 for adults; M4: in a frozen world, inherited minus erased
  notebook is within ±0.02 without a juvenile period and above +0.05 with one.

**Verdicts** are exclusive: *equivalent* (90% interval inside ±m and 95% interval includes 0), *small but
non-zero* (90% interval inside ±m, 95% interval excludes 0), *larger than m* (95% interval beyond ±m),
*different* (95% interval excludes 0 otherwise), *inconclusive*. m = 0.02 for rewards, 0.05 for honest writes.

**Fig. 16:** (a) newborn reward with 8 coordinated liars, by how seats are assigned; (b) juvenile periods in the
shared wiki under individual and group selection; (c) fitness against fixed juvenile periods on the full grid,
with the evolved values; (d) convergence after the credit switch, with and without deaths; (e) agents with their
own memory; (f) what the juvenile period rescues: inherited with it minus inherited without it.
**Fig. 17:** honest writes over time after the credit switch, learners that never die.
"""

EXTRA8_FIG = r"""from scipy import stats

def xverdict(a, b, m):
    # difference of means a - b (Welch), with exclusive verdicts against the margin m
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    b = np.asarray([x for x in b if x is not None and np.isfinite(x)], float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, "-"
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d, se = a.mean() - b.mean(), np.sqrt(va + vb)
    if se == 0:
        lo95 = hi95 = lo90 = hi90 = d
    else:
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        t95, t90 = stats.t.ppf(0.975, df), stats.t.ppf(0.95, df)
        lo95, hi95, lo90, hi90 = d - t95 * se, d + t95 * se, d - t90 * se, d + t90 * se
    if lo90 > -m and hi90 < m:
        v = "small but non-zero" if (lo95 > 0 or hi95 < 0) else "equivalent"
    elif lo95 >= m or hi95 <= -m:
        v = f"larger than {m:g}"
    elif lo95 > 0 or hi95 < 0:
        v = "different"
    else:
        v = "inconclusive"
    return d, lo95, hi95, v

def nseeds(name):
    return len(logs_of(name))

SV = []
def compare(family, what, a, b, measure, m):
    if (a, measure) not in nums or (b, measure) not in nums:
        return None
    d, lo, hi, v = xverdict(vals(a, measure), vals(b, measure), m)
    SV.append({"family": family, "comparison": what, "a": a, "b": b, "measure": measure, "seeds a/b": f"{nseeds(a)}/{nseeds(b)}",
               "a - b": round(d, 3), "95% CI": ci_text(lo, hi), "margin": m, "verdict": v})
    return d, lo, hi, v

V2 = []
def registered(test, case, ok, detail):
    V2.append({"prediction": test, "case": case, "result": ok, "detail": detail})

# --- inheritance, rescue and keep prior, with every seed available
for tag, label in (("", "2,000"), ("_drift8000", "8,000"), ("_static", "never")):
    for what, a, b in (("notebook, fresh heir", "notebook_cited", "notebook_wiped_cited"),
                       ("notebook, heir keeps features", "notebook_torso_cited", "notebook_wiped_torso_cited"),
                       ("pages after a die-off", "sync_cited", "sync_wipe_cited")):
        compare("inherited - erased, no juvenile period", f"{what}, facts change every {label} steps", a + tag, b + tag, "newborn reward", 0.02)
        compare("inherited - erased, juvenile period 50", f"{what}, facts change every {label} steps", a + tag + "_juv", b + tag + "_juv", "newborn reward", 0.02)
        compare("rescue: inherited with - without juvenile period", f"{what}, facts change every {label} steps", a + tag + "_juv", a + tag, "newborn reward", 0.02)
        compare("inherited - erased, keep prior 0.99", f"{what}, facts change every {label} steps", a + tag + "_keep99", b + tag + "_keep99", "newborn reward", 0.02)
for key in ("newborn reward", "adult reward", "honest writes"):
    compare("shared wiki: keep prior 0.99 vs 0.90", key, "individual_cited_keep99", "individual_cited", key, 0.02)
compare("shared wiki: juvenile period vs none", "newborn reward", "individual_cited_juv", "individual_cited", "newborn reward", 0.02)
for a, what in (("deceivers8_rotate_life", "names redrawn once a lifetime"), ("deceivers8_rotate_short", "names redrawn ten times a lifetime"),
                ("deceivers8_anonymous", "anonymous")):
    for key in ("newborn reward", "adult reward"):
        compare("identity, 8 deceivers, vs fixed names", what, a, "deceivers8_named", key, 0.02)

# --- the history test of section 19, re-judged with every seed available
for pages, targets in HYST_TARGETS.items():
    for c in targets:
        A, B = hyst_name(pages, c, "honest"), hyst_name(pages, c, "lying")
        r = compare("history: honest start - lying start", f"own page {pages}, c = {c:+g}", A, B, "honest writes", 0.05)
        if r:
            registered("H1 (all seeds)", f"{pages}, c = {c:+g}", "holds" if r[3] in ("equivalent", "small but non-zero") else
                       ("fails" if r[3].startswith("larger") else "unclear"), f"{r[0]:+.3f} {ci_text(r[1], r[2])}, {r[3]}, seeds {nseeds(A)}/{nseeds(B)}")
        const = CONST_C[pages].get(c)
        for start in HYST_STARTS:
            r = compare("history: switched - constant credit", f"own page {pages}, c = {c:+g}, from {start}",
                        hyst_name(pages, c, start), const, "honest writes", 0.05)
            if r:
                registered("H3 (all seeds)", f"{pages}, c = {c:+g}, from {start}", "holds" if r[3] in ("equivalent", "small but non-zero") else
                           ("fails" if r[3].startswith("larger") else "unclear"), f"{r[0]:+.3f} {ci_text(r[1], r[2])}, {r[3]}")

# --- R: where liars sit
SEATS = [("deceivers8_named", "fixed seats\n(own at its slot)"), ("deceivers8_ownfirst_fixed", "own first,\nfixed order"),
         ("deceivers8_seats_per_copy", "seats drawn\nper copy"), ("deceivers8_seats_per_copy_rotate_life", "per copy,\nredrawn each life"),
         ("deceivers8_anonymous", "anonymous")]
seats = []
for name, lab in SEATS + [("clean_seats_per_copy", "no liars, seats per copy"), ("individual_cited", "no liars, fixed seats")]:
    if (name, "newborn reward") in nums:
        mn, lo, hi = mean_ci(vals(name, "newborn reward"))
        ma, _, _ = mean_ci(vals(name, "adult reward"))
        seats.append({"setting": lab.replace("\n", " "), "run": name, "newborn reward": round(mn, 3), "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                      "adult reward": round(ma, 3), "seeds": nseeds(name)})
seats = pd.DataFrame(seats)
seats.to_csv(os.path.join(PAPER_DIR, "seats.csv"), index=False)
r = compare("seats", "per copy - anonymous (8 deceivers)", "deceivers8_seats_per_copy", "deceivers8_anonymous", "newborn reward", 0.05)
r2 = compare("seats", "own first fixed - per copy (8 deceivers)", "deceivers8_ownfirst_fixed", "deceivers8_seats_per_copy", "newborn reward", 0.05)
if r and r2:
    registered("R1", "seats per copy vs anonymous and vs own first, fixed", "holds" if abs(r[0]) <= 0.05 and r2[0] > 0.2 else "fails",
               f"per copy - anonymous {r[0]:+.3f} {ci_text(r[1], r[2])}; own first fixed - per copy {r2[0]:+.3f}")
r = compare("seats", "own first fixed - fixed named (8 deceivers)", "deceivers8_ownfirst_fixed", "deceivers8_named", "newborn reward", 0.05)
if r:
    registered("R2", "own first, fixed vs fixed named", "holds" if abs(r[0]) <= 0.05 else "fails", f"{r[0]:+.3f} {ci_text(r[1], r[2])}")
r = compare("seats", "no liars: per copy - fixed", "clean_seats_per_copy", "individual_cited", "newborn reward", 0.02)
if r:
    registered("R3", "no liars, seats per copy vs fixed", "holds" if abs(r[0]) <= 0.02 else "fails", f"{r[0]:+.3f} {ci_text(r[1], r[2])}, {r[3]}")

# --- I: credit switch with learners that never die
SWITCH_U = SWITCH_AT // STEPS
def conv_time(l, k=25):
    x, y = smooth_xy(over_time(l, "honest_writes"), k)
    final = np.nanmean(y[x >= len(l["raw"]) * 2 / 3])
    ok = np.nonzero((x >= SWITCH_U) & (np.abs(y - final) < 0.05))[0]
    return float(x[ok[0]] - SWITCH_U) if len(ok) else np.nan
imm, slower = [], 0
for pages, targets in HYST_TARGETS.items():
    for c in targets:
        row = {"own page": pages, "credit after switch": c}
        for mort, sfx in (("mortal", ""), ("immortal", "_immortal")):
            for start in HYST_STARTS:
                name = hyst_name(pages, c, start) + sfx
                if (name, "honest writes") in nums:
                    m_, lo, hi = mean_ci(vals(name, "honest writes"))
                    row[f"{mort}, {start} start"] = round(m_, 3)
            ct = [conv_time(l) for start in HYST_STARTS for l in logs_of(hyst_name(pages, c, start) + sfx)]
            row[f"{mort}: updates to converge"] = round(float(np.nanmean(ct)), 1) if ct and np.isfinite(ct).any() else np.nan
        A, B = hyst_name(pages, c, "honest") + "_immortal", hyst_name(pages, c, "lying") + "_immortal"
        r = compare("history, learners that never die: honest - lying start", f"own page {pages}, c = {c:+g}", A, B, "honest writes", 0.05)
        if r:
            row["immortal: honest - lying"] = round(r[0], 3); row["95% CI"] = ci_text(r[1], r[2])
            registered("I1", f"{pages}, c = {c:+g}", "holds" if abs(r[0]) < 0.05 else "fails", f"{r[0]:+.3f} {ci_text(r[1], r[2])}, {r[3]}")
        if np.isfinite(row.get("immortal: updates to converge", np.nan)) and np.isfinite(row.get("mortal: updates to converge", np.nan)):
            slower += row["immortal: updates to converge"] > row["mortal: updates to converge"]
        imm.append(row)
imm = pd.DataFrame(imm)
imm.to_csv(os.path.join(PAPER_DIR, "immortal_history.csv"), index=False)
if len(imm):
    registered("I2", "slower convergence without deaths", "holds" if slower >= 4 else "fails", f"slower in {slower} of {len(imm)} credits")

# --- G: group selection in the shared wiki
def newborn_late(l):
    # newborn reward of lives that begin in the last third of the run
    U = len(l["raw"]); v = []
    for i, g, r_, _ in lives(l):
        idx = np.nonzero(l["gen"][:, i] == g)[0]
        if idx[0] >= 2 * U / 3 and len(r_) >= EARLY:
            v.append(r_[:EARLY].mean())
    return float(np.mean(v)) if v else np.nan
grp = []
GROUPS = {(sel, j0): (f"evo_wiki_from{j0}_long" if sel == "individual" else f"evo_wiki_group_from{j0}_long")
          for sel in ("individual", "group") for j0 in J0S}
for (sel, j0), name in GROUPS.items():
    L = logs_of(name)
    if not L:
        continue
    m_, lo, hi = mean_ci([final_J(l) for l in L])
    neu = [float(neutral_J(l)[-PAPER_LIFE:].mean()) for l in L]
    dm, dlo, dhi = mean_ci([final_J(l) - n for l, n in zip(L, neu)])
    nb, nlo, nhi = mean_ci([newborn_late(l) for l in L])
    grp.append({"selection": sel, "start": j0, "final juvenile period": round(m_, 1), "95% CI": f"[{lo:.1f}, {hi:.1f}]",
                "neutral expectation": round(float(np.mean(neu)), 1), "selection effect": round(dm, 1),
                "selection 95% CI": f"[{dlo:+.1f}, {dhi:+.1f}]", "newborn reward, last third": round(nb, 3),
                "newborn 95% CI": f"[{nlo:.3f}, {nhi:.3f}]", "_lo": lo, "_hi": hi, "_dlo": dlo, "_dhi": dhi, "_nb": [newborn_late(l) for l in L]})
grp_df = pd.DataFrame([{k: v for k, v in g.items() if not k.startswith("_")} for g in grp])
grp_df.to_csv(os.path.join(PAPER_DIR, "group_selection_wiki.csv"), index=False)
G = {(g["selection"], g["start"]): g for g in grp}
j150 = J0S[1]
if ("group", j150) in G and ("individual", j150) in G:
    a_, b_ = G[("group", j150)], G[("individual", j150)]
    registered("G1", "from 150: group < individual, non-overlapping CIs", "holds" if a_["_hi"] < b_["_lo"] else "fails",
               f"group {a_['final juvenile period']} {a_['95% CI']}, individual {b_['final juvenile period']} {b_['95% CI']}")
    d, lo, hi, v = xverdict(a_["_nb"], b_["_nb"], 0.02)
    registered("G3", "from 150: newborns earn more under group selection", "holds" if lo > 0 else ("fails" if hi < 0 or d <= 0 else "unclear"),
               f"group - individual {d:+.3f} {ci_text(lo, hi)}")
if ("group", 0) in G and ("individual", 0) in G:
    a_, b_ = G[("group", 0)], G[("individual", 0)]
    ok = a_["_dhi"] < 0 and b_["_dlo"] <= 0 <= b_["_dhi"]
    registered("G2", "from 0: group negative, individual neutral", "holds" if ok else "fails",
               f"group {a_['selection effect']:+} {a_['selection 95% CI']}, individual {b_['selection effect']:+} {b_['selection 95% CI']}")

# --- F: fitness of fixed juvenile periods on the full grid
fc2, best2, curves = [], {}, {}
for base in LONG_BASES:
    Js = sorted(set(FIX_JS) | set(FIX_JS_MORE.get(base, ())))
    pts = []
    for J in Js:
        L = logs_of(fix_name(base, J))
        if not L:
            continue
        fit = [np.mean(life_fitness(l)) for l in L]
        m_, lo, hi = mean_ci(fit)
        pts.append((J, m_, lo, hi))
        fc2.append({"setting": LONG_LABELS[base], "juvenile period": J, "run": fix_name(base, J), "fitness": round(m_, 4),
                    "95% CI": f"[{lo:.4f}, {hi:.4f}]", "seeds": len(L)})
    if pts:
        curves[base] = np.array(pts)
fitness_full = pd.DataFrame(fc2)
fitness_full.to_csv(os.path.join(PAPER_DIR, "fitness_curves_full.csv"), index=False)
evolved_runs = {b: [final_J(l) for j0 in J0S for l in logs_of(f"evo_{b}_from{j0}_long")] for b in LONG_BASES}
if "notebook_drift8000" in curves:
    c_ = curves["notebook_drift8000"]
    f100 = c_[c_[:, 0] == PAPER_LIFE // 3, 1]
    later = c_[c_[:, 0] > PAPER_LIFE // 3, 1]
    if len(f100) and len(later):
        registered("F4", "fresh heirs, drift 8,000: peak at 100 or later", "holds" if later.max() >= f100[0] - 0.005 else "fails",
                   ", ".join(f"J {int(J)}: {f:.4f}" for J, f in c_[:, :2]))
for base in ("notebook", "notebook_torso"):
    if base in curves and evolved_runs.get(base):
        c_ = curves[base]
        near = c_[c_[:, 1] >= c_[:, 1].max() - 0.005, 0]
        ev = float(np.mean(evolved_runs[base]))
        registered("F5", LONG_LABELS[base], "holds" if near.min() <= ev <= near.max() else "fails",
                   f"evolved {ev:.1f}; lengths within 0.005 of the best: {', '.join(str(int(j)) for j in near)}")

# --- M: agents with a memory of their own
MEMS = [("mem_no_wiki", "no record"), ("mem_cited", "shared wiki"), ("mem_notebook_cited_static", "notebook inherited, frozen"),
        ("mem_notebook_wiped_cited_static", "notebook erased, frozen"), ("mem_notebook_cited_static_juv", "inherited + juvenile 50"),
        ("mem_notebook_wiped_cited_static_juv", "erased + juvenile 50")]
mem = []
for name, lab in MEMS:
    if (name, "adult reward") in nums:
        row = {"setting": lab, "run": name}
        for key in ("adult reward", "newborn reward", "population reward"):
            m_, lo, hi = mean_ci(vals(name, key)); row[key] = round(m_, 3); row[key + " 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        mem.append(row)
mem = pd.DataFrame(mem)
mem.to_csv(os.path.join(PAPER_DIR, "own_memory.csv"), index=False)
if ("mem_no_wiki", "adult reward") in nums:
    a_ = nums[("mem_no_wiki", "adult reward")][0]
    registered("M1", "adults without a record", "holds" if a_ >= 0.75 else "fails", f"{a_:.3f}")
r = compare("own memory", "wiki - no record, newborns", "mem_cited", "mem_no_wiki", "newborn reward", 0.02)
if r:
    registered("M2", "wiki adds at least 0.2 for newborns", "holds" if r[0] >= 0.2 else "fails", f"{r[0]:+.3f} {ci_text(r[1], r[2])}")
r = compare("own memory", "wiki - no record, adults", "mem_cited", "mem_no_wiki", "adult reward", 0.02)
if r:
    registered("M3", "wiki adds less than 0.05 for adults", "holds" if r[0] < 0.05 else "fails", f"{r[0]:+.3f} {ci_text(r[1], r[2])}")
r = compare("own memory", "inherited - erased notebook, no juvenile period", "mem_notebook_cited_static", "mem_notebook_wiped_cited_static", "newborn reward", 0.02)
r2 = compare("own memory", "inherited - erased notebook, juvenile period 50", "mem_notebook_cited_static_juv", "mem_notebook_wiped_cited_static_juv", "newborn reward", 0.02)
if r and r2:
    registered("M4", "inheritance needs a juvenile period", "holds" if abs(r[0]) <= 0.02 and r2[0] > 0.05 else "fails",
               f"no juvenile period {r[0]:+.3f} {ci_text(r[1], r[2])} ({r[3]}); with {r2[0]:+.3f} {ci_text(r2[1], r2[2])}")

# --- what an inherited record is worth to a reader that simply trusts it
def ideal(name):
    L = logs_of(name)
    if not L:
        return None
    w = PAPER_SETTINGS[name]
    ph, pl = w.get("p_high", 0.9), w.get("p_low", 0.1)
    acc = np.nanmean([late(l, "accuracy") for l in L]); cov = np.nanmean([late(l, "coverage") for l in L])
    return {"setting": name, "share of entries filled": round(cov, 3), "share of filled entries correct": round(acc, 3),
            "reader that trusts its own entry (approx.)": round(cov * (acc * ph + (1 - acc) * pl) + (1 - cov) * 0.61, 3),
            "newborn reward": round(nums.get((name, "newborn reward"), (np.nan,))[0], 3),
            "adult reward": round(nums.get((name, "adult reward"), (np.nan,))[0], 3)}
ideal_reader = pd.DataFrame([x for x in (ideal(n) for n in ("notebook_cited_static", "notebook_torso_cited_static", "notebook_cited")) if x])
ideal_reader.to_csv(os.path.join(PAPER_DIR, "ideal_reader.csv"), index=False)

stats_verdicts = pd.DataFrame(SV)
stats_verdicts.to_csv(os.path.join(PAPER_DIR, "stats_verdicts.csv"), index=False)
review_verdicts2 = pd.DataFrame(V2)
review_verdicts2.to_csv(os.path.join(PAPER_DIR, "review_verdicts2.csv"), index=False)
for title, table in (("where liars sit", seats), ("credit switch, learners that never die", imm),
                     ("group selection in the shared wiki", grp_df), ("fitness, full grid", fitness_full),
                     ("agents with their own memory", mem), ("an inherited record read by a reader that trusts it", ideal_reader),
                     ("key comparisons, exclusive verdicts", stats_verdicts), ("registered predictions", review_verdicts2)):
    print("\n" + title)
    try:
        display(table)
    except NameError:
        print(table.to_string())

# Figure 16
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
a = ax[0, 0]
xs = [(n, l) for n, l in SEATS if (n, "newborn reward") in nums]
for i, (n, l) in enumerate(xs):
    m_, lo, hi = mean_ci(vals(n, "newborn reward"))
    a.bar(i, m_, 0.7, yerr=[[m_ - lo], [hi - m_]] if np.isfinite(lo) else None, color="#534AB7" if i < 2 else "#D85A30" if i < 4 else "#888780", capsize=3)
for ref, lab in (("individual_cited", "no liars"), ("no_wiki", "no record")):
    if (ref, "newborn reward") in nums:
        a.axhline(nums[(ref, "newborn reward")][0], color="gray", ls=":", lw=0.8); a.text(len(xs) - 0.5, nums[(ref, "newborn reward")][0] + 0.01, lab, ha="right", fontsize=7)
a.set_xticks(range(len(xs))); a.set_xticklabels([l for _, l in xs], fontsize=7); a.set_ylabel("newborn reward")
a.set_title("(a) 8 coordinated liars: how seats are assigned", fontsize=10)
a = ax[0, 1]
for (sel, j0), name in GROUPS.items():
    L = logs_of(name)
    if L:
        y = np.mean([l["J_mean"].mean(1) for l in L], 0)
        a.plot(y, color="#534AB7" if sel == "individual" else "#D85A30", ls="-" if j0 == 0 else "--", label=f"{sel} selection, from {j0}")
        if sel == "individual":
            a.plot(neutral_J(L[0]), color="gray", lw=0.8, ls="-" if j0 == 0 else "--")
a.set_xlabel("update"); a.set_ylabel("mean juvenile period"); a.legend(fontsize=7, frameon=False)
a.set_title("(b) Shared wiki: individual vs group selection (grey: neutral)", fontsize=10)
a = ax[0, 2]
FCOL = {"notebook": "#EF9F27", "notebook_drift8000": "#BA7517", "notebook_torso": "#D85A30", "notebook_torso_drift8000": "#993C1D"}
for base, c_ in curves.items():
    a.errorbar(c_[:, 0], c_[:, 1], yerr=[c_[:, 1] - c_[:, 2], c_[:, 3] - c_[:, 1]], marker="o", ms=3, capsize=2, color=FCOL[base], label=LONG_LABELS[base])
    if evolved_runs.get(base):
        ev = float(np.mean(evolved_runs[base])); a.plot([ev], [np.interp(ev, c_[:, 0], c_[:, 1])], marker="D", color=FCOL[base], ms=7, mfc="white")
a.set_xlabel("fixed juvenile period"); a.set_ylabel("fitness (reward per pull, completed lives)"); a.legend(fontsize=6, frameon=False)
a.set_title("(c) Fitness, full grid (diamonds: evolved)", fontsize=10)
a = ax[1, 0]
if len(imm):
    labs = [f"{r_['own page'][:4]} {r_['credit after switch']:+g}" for _, r_ in imm.iterrows()]
    for j, (mort, color) in enumerate((("mortal", "#534AB7"), ("immortal", "#D85A30"))):
        col = f"{mort}: updates to converge"
        if col in imm:
            a.bar(np.arange(len(imm)) + (j - 0.5) * 0.38, imm[col], 0.36, color=color, label="with deaths" if mort == "mortal" else "never die")
    a.set_xticks(range(len(imm))); a.set_xticklabels(labs, fontsize=7); a.legend(fontsize=8, frameon=False)
a.set_ylabel("updates after the switch to converge"); a.set_title("(d) Convergence after the credit switch", fontsize=10)
a = ax[1, 1]
for j, key in enumerate(("adult reward", "newborn reward")):
    xs_ = [(n, l) for n, l in MEMS if (n, key) in nums]
    for i, (n, l) in enumerate(xs_):
        m_, lo, hi = mean_ci(vals(n, key))
        a.bar(i + (j - 0.5) * 0.38, m_, 0.36, yerr=[[m_ - lo], [hi - m_]] if np.isfinite(lo) else None, capsize=2,
              color="#534AB7" if j == 0 else "#BA7517", label=key if i == 0 else None)
a.set_xticks(range(len(MEMS))); a.set_xticklabels(["no record", "shared\nwiki", "notebook\ninherited", "notebook\nerased", "inherited\n+ juvenile", "erased\n+ juvenile"], fontsize=7)
a.legend(fontsize=8, frameon=False); a.set_title("(e) Agents with a memory of their own", fontsize=10)
a = ax[1, 2]
res = [r_ for r_ in SV if r_["family"].startswith("rescue") or r_["family"] == "inherited - erased, juvenile period 50"]
labs = sorted({r_["comparison"] for r_ in res})
for j, (fam, color) in enumerate((("rescue: inherited with - without juvenile period", "#1D9E75"), ("inherited - erased, juvenile period 50", "#888780"))):
    for i, lab in enumerate(labs):
        r_ = [x for x in res if x["family"] == fam and x["comparison"] == lab]
        if r_:
            lo, hi = [float(v) for v in r_[0]["95% CI"].strip("[]").split(", ")]
            d = r_[0]["a - b"]
            a.barh(i + (j - 0.5) * 0.38, d, 0.36, xerr=[[d - lo], [hi - d]], color=color, capsize=2,
                   label=("with minus without juvenile period" if j == 0 else "inherited minus erased, both with it") if i == 0 else None)
a.set_yticks(range(len(labs))); a.set_yticklabels([l.replace(", facts change every ", "\n") for l in labs], fontsize=6)
a.axvline(0, color="gray", lw=0.7); a.legend(fontsize=7, frameon=False); a.set_xlabel("newborn reward")
a.set_title("(f) What the juvenile period rescues", fontsize=10)
for a in ax.flat:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
savefig(fig, "fig16_review_tests2")

# Figure 17: credit switch, learners that never die
fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for r_, pages in enumerate(HYST_TARGETS):
    for k, c in enumerate(HYST_TARGETS[pages]):
        a = ax[r_, k]
        for start, color in (("honest", "#1D9E75"), ("lying", "#D85A30")):
            for j, l in enumerate(logs_of(hyst_name(pages, c, start) + "_immortal")):
                x, y = smooth_xy(over_time(l, "honest_writes"), 25)
                a.plot(x, y, color=color, lw=0.9, alpha=0.8, label=f"{start} start" if j == 0 else None)
        const = CONST_C[pages].get(c)
        if const and (const, "honest writes") in nums:
            a.axhline(nums[(const, "honest writes")][0], color="gray", ls="--", lw=0.8, label="constant credit, with deaths")
        a.axvline(SWITCH_U, color="k", ls=":", lw=0.8)
        a.set_title(f"own page {pages}, credit {c:+g} after the switch", fontsize=9); a.set_xlabel("update")
        a.spines[["top", "right"]].set_visible(False)
    ax[r_, 0].set_ylabel("honest writes")
ax[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("Honest writes after the credit switch, learners that never die")
fig.tight_layout()
savefig(fig, "fig17_history_immortal")
print("figures saved to", FIG_DIR)
"""


ADULT_MD = r"""## 21. History without newborns: adults only, and why immortal liars stay liars

No new runs: this section re-reads the logs of sections 19 and 20. Saved to `wikibandits_results/paper/`:
`adult_history.csv`, `adult_verdicts.csv` and `figures/fig18_adult_history`.

In mortal populations, the share of honest writes includes newborns, whose first writes are close to random.
That could pull both histories towards the same value. Here honest writes are split by age: adults (older than
50 updates) and newborns.

**Predictions** (written down before this analysis, project note "ALife journal plan", 13 Nov 2026, 12:10):
- A1: adults' honest writes after an honest and after a lying start are equivalent within ±0.05 in at least 5 of
  the 6 targets.
- A2: adults are at least as honest as newborns at every target.

**Fig. 18:** (a–f) adults' honest writes over time from both starts, mortal learners; (g) how often readers
follow the record's majority over time, learners that never die, at credit 0 with own pages readable.
"""

ADULT_FIG = r"""def split_by_age(l, key="honest_writes", frac=1 / 3):
    # mean of a per-agent quantity over the last third, for adults (age > EARLY) and newborns (age <= EARLY)
    U = len(l[key]); u0 = U - int(U * frac)
    v, age = l[key][u0:], l["age"][u0:]
    L = learner_mask(l)[None, :] & np.isfinite(v)
    adult, young = L & (age > EARLY), L & (age <= EARLY)
    return (float(v[adult].mean()) if adult.any() else np.nan, float(v[young].mean()) if young.any() else np.nan,
            float(young.sum() / max(L.sum(), 1)))

rows, AV = [], []
for pages, targets in HYST_TARGETS.items():
    for c in targets:
        row = {"own page": pages, "credit after switch": c}
        per = {}
        for start in HYST_STARTS:
            L = logs_of(hyst_name(pages, c, start))
            if not L:
                continue
            sp = [split_by_age(l) for l in L]
            per[start] = [x[0] for x in sp]
            row[f"adults, {start} start"] = round(float(np.nanmean([x[0] for x in sp])), 3)
            row[f"newborns, {start} start"] = round(float(np.nanmean([x[1] for x in sp])), 3)
            row[f"newborn share of writing cells, {start}"] = round(float(np.mean([x[2] for x in sp])), 3)
            row["seeds"] = len(L)
        const = CONST_C[pages].get(c)
        if const and logs_of(const):
            row["adults, constant credit"] = round(float(np.nanmean([split_by_age(l)[0] for l in logs_of(const)])), 3)
        if len(per) == 2:
            d, lo, hi, v = xverdict(per["honest"], per["lying"], 0.05)
            row["adults: honest - lying start"] = round(d, 3); row["95% CI"] = ci_text(lo, hi); row["verdict"] = v
            AV.append({"prediction": "A1", "case": f"{pages}, c = {c:+g}", "result": "holds" if v in ("equivalent", "small but non-zero") else
                       ("fails" if v.startswith("larger") else "unclear"), "detail": f"{d:+.3f} {ci_text(lo, hi)}, {v}"})
            ok = all(row.get(f"adults, {s} start", np.nan) >= row.get(f"newborns, {s} start", np.nan) for s in HYST_STARTS)
            AV.append({"prediction": "A2", "case": f"{pages}, c = {c:+g}", "result": "holds" if ok else "fails",
                       "detail": ", ".join(f"{s}: adults {row.get(f'adults, {s} start')} vs newborns {row.get(f'newborns, {s} start')}" for s in HYST_STARTS)})
        rows.append(row)
adult_history = pd.DataFrame(rows)
adult_history.to_csv(os.path.join(PAPER_DIR, "adult_history.csv"), index=False)
n_ok = sum(a["result"] == "holds" for a in AV if a["prediction"] == "A1")
AV.append({"prediction": "A1 (overall)", "case": "at least 5 of 6 targets", "result": "holds" if n_ok >= 5 else "fails", "detail": f"{n_ok} of 6"})
adult_verdicts = pd.DataFrame(AV)
adult_verdicts.to_csv(os.path.join(PAPER_DIR, "adult_verdicts.csv"), index=False)
for title, table in (("honest writes by age, mortal history runs", adult_history), ("verdicts", adult_verdicts)):
    print("\n" + title)
    try:
        display(table)
    except NameError:
        print(table.to_string())

def over_time_adults(l, key="honest_writes"):
    v = np.where(learner_mask(l)[None, :] & (l["age"] > EARLY), l[key], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(v, axis=1)

fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
for r_, pages in enumerate(HYST_TARGETS):
    for k, c in enumerate(HYST_TARGETS[pages]):
        a = ax[r_, k]
        for start, color in (("honest", "#1D9E75"), ("lying", "#D85A30")):
            for j, l in enumerate(logs_of(hyst_name(pages, c, start))):
                x, y = smooth_xy(over_time_adults(l), 25)
                a.plot(x, y, color=color, lw=0.7, alpha=0.6, label=f"{start} start" if j == 0 else None)
        a.axvline(SWITCH_AT // STEPS, color="k", ls=":", lw=0.8)
        a.set_ylim(0, 1.02); a.set_title(f"({'abcdef'[r_ * 3 + k]}) adults, own page {pages}, c = {c:+g}", fontsize=9)
        a.set_xlabel("update"); a.spines[["top", "right"]].set_visible(False)
    ax[r_, 0].set_ylabel("adults' honest writes")
ax[0, 0].legend(fontsize=7, frameon=False)
a = ax[0, 3]
for start, color in (("honest", "#1D9E75"), ("lying", "#D85A30")):
    for j, l in enumerate(logs_of(hyst_name("readable", 0.0, start) + "_immortal")):
        x, y = smooth_xy(over_time(l, "follow_rate"), 25)
        a.plot(x, y, color=color, lw=0.8, label=f"{start} start" if j == 0 else None)
a.axvline(SWITCH_AT // STEPS, color="k", ls=":", lw=0.8); a.set_ylim(0, 1.02)
a.set_title("(g) never die, readable, c = 0: readers follow the majority", fontsize=9); a.set_xlabel("update")
a.legend(fontsize=7, frameon=False); a.spines[["top", "right"]].set_visible(False)
ax[1, 3].axis("off")
fig.tight_layout()
savefig(fig, "fig18_adult_history")
print("figures saved to", FIG_DIR)
"""

EXTRA10_MD = r"""## 23. The fourth review's tests

Uses the runs of `RUN_EXTRA10` and the extra seeds (section 13), with the switch runs of sections 19, 20 and 22.
Predictions were registered on 13 Nov 2026 at 18:46, before any code. Saved to `batch10_tests.csv`,
`timescale_forward.csv`, `batch10_verdicts.csv` and `batch10_series.npz`; figures `fig21_noise_and_strong_credit`,
`fig22_silence_and_renewal` and `fig23_timescale_forward`.

- **N, random writes without death.** Learners that never die, whose sampled writes are replaced with probability
  0.02 or 0.10 by a random report (belief or false arm) that they train on as their own.
- **S, strong credit.** A lying start switched to c = +0.5, mortal and immortal.
- **J, silence against renewal.** The mortal switch runs with juvenile periods of 50 and 150 updates.
- **R, rivalry over any reader's win.** An author loses whenever a reader who saw its entry wins, whatever arm it pulled.
- **T, the timescale rule tested forward.** Notebook heirs in a frozen world with half or double the learning rate, and
  with facts changing every 8,000 steps; the share of the inheritance's value is predicted from the curve fitted
  before (half value at a ratio of 1.10, slope 1.50 in log ratio).
- **C, fewer copies.** 64 copies of the world instead of 256.
"""

EXTRA10_FIG = r"""# --- section 23: the fourth review's tests
V4, rows = [], []
def reg4(test, case, ok, detail):
    V4.append({"prediction": test, "case": case, "result": "holds" if ok else "fails", "detail": detail})

def m_of(name, key="honest writes"):
    v = hv(name, key)
    return float(np.mean(v)) if v else np.nan

def case_of(p, c):
    return f"{p}, c = {c:+g}"

# N: random writes without death
for pages, c in DOSE_TARGETS:
    runs = [("lying", e, noise_name(pages, c, "lying", e)) for e in NOISES] + [("honest", 0.10, noise_name(pages, c, "honest", 0.10))]
    for start, eps, n in runs:
        if not logs_of(n):
            continue
        m, lo, hi = mean_ci(hv(n))
        rows.append({"test": "N", "own page": pages, "credit after switch": c, "start": start, "noise": eps, "honest writes": round(m, 3),
                     "95% CI": ci_text(lo, hi), "follows": round(m_of(n, "follows"), 3),
                     "information": round(m_of(n, "real − scrambled"), 3), "seeds": nseeds(n)})
p, c = "hidden", 0.02
n10, n02, base = noise_name(p, c, "lying", 0.10), noise_name(p, c, "lying", 0.02), hyst_name(p, c, "lying") + "_immortal"
if logs_of(n10):
    reg4("N1", case_of(p, c) + ", noise 0.10", m_of(n10) >= 0.5, f"{m_of(n10):.3f} (at least 0.5)")
if logs_of(n02) and logs_of(base):
    d, lo, hi, v = xverdict(hv(n02), hv(base), 0.05)
    reg4("N1", case_of(p, c) + ", noise 0.02", lo > 0.05, f"{m_of(n02):.3f} against {m_of(base):.3f} without noise: {d:+.3f} {ci_text(lo, hi)}")
p, c = "readable", 0.0
for e in NOISES:
    n = noise_name(p, c, "lying", e)
    if logs_of(n):
        reg4("N2", case_of(p, c) + f", noise {e:g}", m_of(n) < 0.25, f"{m_of(n):.3f} (below 0.25)")
for p, c in DOSE_TARGETS:
    n = noise_name(p, c, "honest", 0.10)
    if logs_of(n):
        reg4("N3", case_of(p, c), m_of(n) > 0.75, f"{m_of(n):.3f} (above 0.75)")

# S: strong credit after a lying start
for p, c in DOSE_TARGETS:
    for imm in (False, True):
        n = strong_name(p, imm)
        if not logs_of(n):
            continue
        m, lo, hi = mean_ci(hv(n))
        rows.append({"test": "S", "own page": p, "credit after switch": 0.5, "start": "lying", "immortal": imm, "honest writes": round(m, 3),
                     "95% CI": ci_text(lo, hi), "follows": round(m_of(n, "follows"), 3),
                     "information": round(m_of(n, "real − scrambled"), 3), "seeds": nseeds(n)})
        if not imm:
            reg4("S1", p, m >= 0.9, f"{m:.3f} (at least 0.9)")
        elif p == "readable":
            reg4("S2", p, m < 0.25, f"{m:.3f} (below 0.25)")
        else:
            b = hyst_name(p, c, "lying") + "_immortal"
            if logs_of(b):
                d, lo2, hi2, v = xverdict(hv(n), hv(b), 0.05)
                reg4("S3", p, lo2 > 0.05, f"{m:.3f} against {m_of(b):.3f} at c = +0.02: {d:+.3f} {ci_text(lo2, hi2)}")

# J: silence against renewal
for p, c in DOSE_TARGETS:
    rec, gap = {}, {}
    for J in (0,) + SILENCES:
        A = hyst_name(p, c, "honest") if J == 0 else silence_name(p, c, "honest", J)
        B = hyst_name(p, c, "lying") if J == 0 else silence_name(p, c, "lying", J)
        if not (logs_of(A) and logs_of(B)):
            continue
        rt = [recovery(l) for l in logs_of(B)]
        rec[J], gap[J] = float(np.mean(rt)), xverdict(hv(A), hv(B), 0.05)
        rm, rlo, rhi = mean_ci(rt)
        rows.append({"test": "J", "own page": p, "credit after switch": c, "juvenile period": J,
                     "honest start": round(m_of(A), 3), "lying start": round(m_of(B), 3), "honest - lying": round(gap[J][0], 3),
                     "95% CI": ci_text(*gap[J][1:3]), "verdict": gap[J][3], "recovery time (updates)": round(rm, 1),
                     "recovery 95% CI": ci_text(rlo, rhi), "recovery median": float(np.median(rt)),
                     "never recovered": int(sum(r >= len(logs_of(B)[0]["raw"]) - SWITCH_U for r in rt)),
                     "seeds": f"{nseeds(A)}/{nseeds(B)}"})
    if len(rec) == 3:
        j1, j2 = SILENCES
        reg4("J1", case_of(p, c), rec[0] < rec[j1] < rec[j2] and rec[j2] - rec[0] >= 100,
             ", ".join(f"J {J}: {x:.0f}" for J, x in rec.items()))
        reg4("J2", case_of(p, c), all(gap[J][3] in ("equivalent", "small but non-zero") for J in SILENCES),
             "; ".join(f"J {J}: {gap[J][0]:+.3f} {ci_text(*gap[J][1:3])}, {gap[J][3]}" for J in SILENCES))
        h0, h2 = m_of(hyst_name(p, c, "honest")), m_of(silence_name(p, c, "honest", j2))
        reg4("J3", case_of(p, c), h2 >= h0 - 0.02, f"J {j2}: {h2:.3f}, J 0: {h0:.3f}")

# R: rivalry over any reader's win
for n, lab in (("readrival_readable_c-0.5", "rivalry over any win, readable"), ("readrival_hidden_c-0.5", "rivalry over any win, hidden"),
               ("rival_info", "rivalry over the named arm, readable"), ("hidden_own_c-0.5", "rivalry over the named arm, hidden"),
               ("no_wiki", "no record")):
    if logs_of(n):
        rows.append({"test": "R", "setting": lab, "honest writes": round(m_of(n), 3) if hv(n) else np.nan,
                     "information": round(m_of(n, "real − scrambled"), 3) if hv(n, "real − scrambled") else np.nan,
                     "newborn reward": round(m_of(n, "newborn reward"), 3), "adult reward": round(m_of(n, "adult reward"), 3),
                     "seeds": nseeds(n)})
for n, lim in (("readrival_hidden_c-0.5", 0.05), ("readrival_readable_c-0.5", 0.10)):
    if logs_of(n):
        reg4("R1", n, m_of(n, "real − scrambled") < lim, f"information {m_of(n, 'real − scrambled'):.3f} (below {lim})")
if logs_of("hidden_own_c-0.5"):
    reg4("R1", "hidden_own_c-0.5", m_of("hidden_own_c-0.5", "real − scrambled") >= 0.15,
         f"information {m_of('hidden_own_c-0.5', 'real − scrambled'):.3f} (at least 0.15)")
if logs_of("readrival_hidden_c-0.5"):
    nb, ad = m_of("readrival_hidden_c-0.5", "newborn reward"), m_of("readrival_hidden_c-0.5", "adult reward")
    reg4("R2", "readrival_hidden_c-0.5", abs(nb - 0.315) <= 0.03 and abs(ad - 0.585) <= 0.05, f"newborns {nb:.3f}, adults {ad:.3f}")

# T: the timescale rule tested forward (curve fixed before the runs)
MID, SLOPE = 1.10, 1.50
def share_curve(ratio):
    return 1 / (1 + np.exp(-(np.log(ratio) - np.log(MID)) * SLOPE))
def t_protect(J, keep=0.9):
    return J + 1 / (-4 * np.log(keep))
TCOND = [("default (paper)", lambda b, J: f"{b}_static{tsuf(J)}"),
         ("learning rate x0.5", lambda b, J: f"{b}_static{tsuf(J)}_lrhalf"),
         ("learning rate x2", lambda b, J: f"{b}_static{tsuf(J)}_lrdouble"),
         ("facts change every 8,000 steps", lambda b, J: f"{b}_drift8000{tsuf(J)}")]
trows, pts, trs = [], [], {}
for lab, nm in TCOND:
    if not all(logs_of(nm(b, J)) for b in ("notebook_cited", "notebook_wiped_cited") for J in TJS):
        continue
    tr = float(np.nanmean(vals(nm("notebook_cited", TJS[3]), "t to 0.61")))
    trs[lab] = tr
    gains = {J: m_of(nm("notebook_cited", J), "newborn reward") - m_of(nm("notebook_wiped_cited", J), "newborn reward") for J in TJS}
    for J in TJS:
        sh, pr = gains[J] / gains[TJS[-1]], share_curve(t_protect(J) / tr)
        trows.append({"condition": lab, "juvenile period": J, "reading time": round(tr, 1), "ratio": round(t_protect(J) / tr, 3),
                      "gain": round(gains[J], 3), "share of max": round(sh, 3), "predicted share": round(pr, 3)})
        if J < TJS[-1] and lab != "default (paper)":
            pts.append((lab, J, sh, pr))
if all(k in trs for k in ("learning rate x0.5", "learning rate x2")):
    ref = trs.get("default (paper)", 41.0)
    reg4("T1", "reading time", trs["learning rate x0.5"] > ref > trs["learning rate x2"],
         f"x0.5: {trs['learning rate x0.5']:.0f}, default: {ref:.0f}, x2: {trs['learning rate x2']:.0f}")
if pts:
    err = np.array([abs(sh - pr) for _, _, sh, pr in pts])
    reg4("T2", f"{len(pts)} points", (err <= 0.2).sum() >= 9 and err.mean() <= 0.15,
         f"{int((err <= 0.2).sum())} of {len(pts)} within 0.20; mean absolute error {err.mean():.3f}")

# C: 64 copies of the world
if logs_of("no_wiki_envs64") and logs_of("individual_cited_envs64"):
    a, b = m_of("individual_cited_envs64", "newborn reward"), m_of("no_wiki_envs64", "newborn reward")
    reg4("C1", "64 copies", a >= 2 * b, f"newborns {a:.3f} with the wiki, {b:.3f} without")
im, mo = hyst_name("readable", 0.0, "lying") + "_immortal_envs64", hyst_name("readable", 0.0, "lying") + "_envs64"
if logs_of(im) and logs_of(mo):
    reg4("C2", "64 copies", m_of(im) < 0.25 and m_of(mo) > 0.5, f"never die {m_of(im):.3f}, mortal {m_of(mo):.3f}")
for n in ("no_wiki_envs64", "individual_cited_envs64", im, mo):
    if logs_of(n):
        rows.append({"test": "C", "setting": n, "honest writes": round(m_of(n), 3) if hv(n) else np.nan,
                     "newborn reward": round(m_of(n, "newborn reward"), 3), "adult reward": round(m_of(n, "adult reward"), 3),
                     "seeds": nseeds(n)})

batch10_tests = pd.DataFrame(rows)
batch10_tests.to_csv(os.path.join(PAPER_DIR, "batch10_tests.csv"), index=False)
timescale_forward = pd.DataFrame(trows)
timescale_forward.to_csv(os.path.join(PAPER_DIR, "timescale_forward.csv"), index=False)
batch10_verdicts = pd.DataFrame(V4)
batch10_verdicts.to_csv(os.path.join(PAPER_DIR, "batch10_verdicts.csv"), index=False)
for title, table in (("tests", batch10_tests), ("timescale rule, forward", timescale_forward), ("registered predictions", batch10_verdicts)):
    print("\n" + title)
    try:
        display(table)
    except NameError:
        print(table.to_string())

series10 = {}
for n in [k for k in EXTRA10 if k.startswith("hyst_")]:
    for j, l in enumerate(logs_of(n)):
        x, y = smooth_series(l)
        series10[f"{n}__{j}__x"] = x[::3].astype(np.float32)
        series10[f"{n}__{j}__all"] = y[::3].astype(np.float32)
np.savez_compressed(os.path.join(PAPER_DIR, "batch10_series.npz"), **series10)

def mean_course10(name):
    L = logs_of(name)
    if not L:
        return None, None
    xs, ys = zip(*[smooth_series(l) for l in L])
    k = min(len(y) for y in ys)
    return np.asarray(xs[0][:k]) / PAPER_LIFE, np.nanmean([y[:k] for y in ys], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (p, c) in zip(axes, DOSE_TARGETS):
    for n, lab, col, ls in ((hyst_name(p, c, "lying"), "mortal", "#888780", "-"),
                            (hyst_name(p, c, "lying") + "_immortal", "never die", "#26215C", "-"),
                            (noise_name(p, c, "lying", 0.02), "never die, noise 0.02", "#AFA9EC", "-"),
                            (noise_name(p, c, "lying", 0.10), "never die, noise 0.10", "#534AB7", "-"),
                            (noise_name(p, c, "honest", 0.10), "never die, noise 0.10, honest start", "#534AB7", ":"),
                            (strong_name(p, True), "never die, c = +0.5", "#D85A30", "--"),
                            (strong_name(p, False), "mortal, c = +0.5", "#D85A30", ":")):
        x, y = mean_course10(n)
        if x is not None:
            ax.plot(x, y, color=col, ls=ls, lw=1.4, label=lab)
    ax.axvline(SWITCH_U / PAPER_LIFE, color="gray", ls=":", lw=0.8)
    ax.set_title(f"lying start, {case_of(p, c)}"); ax.set_xlabel("lifetimes elapsed"); ax.set_ylim(0, 1)
axes[0].set_ylabel("honest writes"); axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(PAPER_DIR, "fig21_noise_and_strong_credit.pdf")); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (p, c) in zip(axes, DOSE_TARGETS):
    for J, col in zip((0,) + SILENCES, ("#26215C", "#534AB7", "#AFA9EC")):
        for start, ls in (("lying", "-"), ("honest", ":")):
            n = hyst_name(p, c, start) if J == 0 else silence_name(p, c, start, J)
            x, y = mean_course10(n)
            if x is not None:
                ax.plot(x, y, color=col, ls=ls, lw=1.4, label=f"juvenile period {J}, {start} start")
    ax.axvline(SWITCH_U / PAPER_LIFE, color="gray", ls=":", lw=0.8)
    ax.set_title(case_of(p, c)); ax.set_xlabel("lifetimes elapsed"); ax.set_ylim(0, 1)
axes[0].set_ylabel("honest writes"); axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(PAPER_DIR, "fig22_silence_and_renewal.pdf")); plt.show()

if trows:
    fig, ax = plt.subplots(figsize=(5, 3.8))
    xx = np.exp(np.linspace(np.log(0.04), np.log(12), 200))
    ax.plot(xx, share_curve(xx), color="gray", lw=1, label="curve fixed before the runs")
    for (lab, _), col in zip(TCOND, ("#888780", "#534AB7", "#D85A30", "#1D9E75")):
        d = timescale_forward[timescale_forward["condition"] == lab]
        ax.plot(d["ratio"], d["share of max"], "o", color=col, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("protection time / time to learn to read"); ax.set_ylabel("share of the largest value")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(PAPER_DIR, "fig23_timescale_forward.pdf")); plt.show()
"""

EXTRA9_MD = r"""## 22. Turnover and the lock

Uses the runs of `RUN_EXTRA9` and the extra seeds (section 13), with the switch runs of sections 19 and 20. Saved to
`wikibandits_results/paper/`: `turnover_dose.csv`, `lock_mechanism.csv`, `immortal_all_seeds.csv`,
`batch9_verdicts.csv`, `history_series.npz` (smoothed time courses of every switch run, so the paper's figures can be
drawn without re-running) and `figures/fig19_turnover_dose`, `figures/fig20_lock_mechanism`.

**Predictions** (written down before any code or runs, project note "ALife journal plan", 13 Nov 2026, 12:58).
Honest writes over the last third (updates 1,800–2,700), all ages unless stated.
- I1, re-judged with 10 seeds: honest minus lying start beyond +0.05 (95% interval) at readable c = −0.02 and 0 and
  hidden c = 0 and +0.02; positive point estimates at readable −0.05 and hidden −0.02.
- D1: the honest-minus-lying gap grows with lifespan: 300 < 900 < 2,700 < never at both targets, and 100 no larger
  than 300 + 0.02.
- D2: at lifespans 100 and 300 the gap is equivalent within ±0.05 (90% interval); at 2,700 it exceeds +0.2.
- D3: recovery time (updates after the switch until the lying start's honest writes, smoothed over 25 updates,
  first exceed 0.35; 1,800 if never) is ordered 100 < 300 < 900 < 2,700, and 900 is at least 1.5 times 300.
- X1, readers re-initialised: stays lying (below 0.2) at both targets.
- X2, writers re-initialised: escapes (at least 0.5) at both targets.
- X3, both heads re-initialised: escapes (at least 0.5).
- X4, whole network re-initialised: escapes and ends within 0.1 of the mortal lying-start level (0.776 readable
  c = 0; 0.729 hidden +0.02).

**Fig. 19:** (a, b) honest writes over time for each lifespan, from a lying start (solid) and an honest start
(dashed), mean over seeds; (c) honest minus lying start against lifespan; (d) recovery time against lifespan.
**Fig. 20:** learners that never die, starting lying, with nothing, the readers, the writers, both heads or the
whole network re-initialised at the switch: (a, b) honest writes, (c, d) how often readers pull the majority's arm.
"""

EXTRA9_FIG = r"""# --- section 22: turnover dose and which side holds the lock
V3, S3 = [], []
def reg3(test, case, ok, detail):
    V3.append({"prediction": test, "case": case, "result": ok, "detail": detail})

def hv(name, key="honest writes"):
    return [x for x in vals(name, key) if x is not None and np.isfinite(x)]

def ffill(y):
    y = np.asarray(y, float).copy()
    for i in range(1, len(y)):
        if not np.isfinite(y[i]):
            y[i] = y[i - 1]
    return y

def smooth_series(l, key="honest_writes", k=25, adults=False):
    y = over_time_adults(l, key) if adults else over_time(l, key)
    return smooth_xy(ffill(y), k)

def recovery(l, level=0.35, k=25):
    x, y = smooth_series(l, k=k)
    ok = np.nonzero((x >= SWITCH_U) & (y > level))[0]
    return float(x[ok[0]] - SWITCH_U) if len(ok) else float(len(l["raw"]) - SWITCH_U)

# I1 with every seed
rows = []
for pages, targets in HYST_TARGETS.items():
    for c in targets:
        A, B = hyst_name(pages, c, "honest") + "_immortal", hyst_name(pages, c, "lying") + "_immortal"
        if not (logs_of(A) and logs_of(B)):
            continue
        d, lo, hi, v = xverdict(hv(A), hv(B), 0.05)
        strong = (pages, c) in (("readable", -0.02), ("readable", 0.0), ("hidden", 0.0), ("hidden", 0.02))
        ok = (lo > 0.05) if strong else (d > 0)
        reg3("I1 (all seeds)", f"{pages}, c = {c:+g}", "holds" if ok else "fails",
             f"{d:+.3f} {ci_text(lo, hi)}, {v}, seeds {nseeds(A)}/{nseeds(B)}")
        rows.append({"own page": pages, "credit after switch": c, "honest start": round(float(np.mean(hv(A))), 3),
                     "lying start": round(float(np.mean(hv(B))), 3), "honest - lying": round(d, 3), "95% CI": ci_text(lo, hi),
                     "verdict": v, "follows, lying start": round(float(np.mean(hv(B, "follows"))), 3),
                     "seeds": f"{nseeds(A)}/{nseeds(B)}"})
immortal_all = pd.DataFrame(rows)
immortal_all.to_csv(os.path.join(PAPER_DIR, "immortal_all_seeds.csv"), index=False)

# D: turnover dose
LIVES = [(PAPER_LIFE // 3, "100"), (PAPER_LIFE, "300"), (PAPER_LIFE * 3, "900"), (PAPER_LIFE * 9, "2,700"), (None, "never")]
def dose_setting(pages, c, start, life):
    if life is None:
        return hyst_name(pages, c, start) + "_immortal"
    if life == PAPER_LIFE:
        return hyst_name(pages, c, start)
    return dose_name(pages, c, start, life)

rows, gaps, recs = [], {}, {}
for pages, c in DOSE_TARGETS:
    for life, lab in LIVES:
        A, B = dose_setting(pages, c, "honest", life), dose_setting(pages, c, "lying", life)
        if not (logs_of(A) and logs_of(B)):
            continue
        d, lo, hi, v = xverdict(hv(A), hv(B), 0.05)
        rt = [recovery(l) for l in logs_of(B)]
        ad = {s: [split_by_age(l)[0] for l in logs_of(n)] for s, n in (("honest", A), ("lying", B))}
        gaps[(pages, c, lab)] = (d, lo, hi, v); recs[(pages, c, lab)] = rt
        rm, rlo, rhi = mean_ci(rt)
        rows.append({"own page": pages, "credit after switch": c, "lifespan": lab, "honest start": round(float(np.mean(hv(A))), 3),
                     "lying start": round(float(np.mean(hv(B))), 3), "honest - lying": round(d, 3), "95% CI": ci_text(lo, hi),
                     "verdict": v, "adults, honest start": round(float(np.nanmean(ad["honest"])), 3),
                     "adults, lying start": round(float(np.nanmean(ad["lying"])), 3),
                     "recovery time (updates)": round(rm, 1), "recovery 95% CI": ci_text(rlo, rhi),
                     "never recovered": int(sum(r >= len(logs_of(B)[0]["raw"]) - SWITCH_U for r in rt)),
                     "seeds": f"{nseeds(A)}/{nseeds(B)}"})
turnover_dose = pd.DataFrame(rows)
turnover_dose.to_csv(os.path.join(PAPER_DIR, "turnover_dose.csv"), index=False)
for pages, c in DOSE_TARGETS:
    case = f"{pages}, c = {c:+g}"
    g = {lab: gaps[(pages, c, lab)][0] for _, lab in LIVES if (pages, c, lab) in gaps}
    if len(g) == len(LIVES):
        ok = g["300"] < g["900"] < g["2,700"] < g["never"] and g["100"] <= g["300"] + 0.02
        reg3("D1", case, "holds" if ok else "fails", ", ".join(f"{k}: {x:+.3f}" for k, x in g.items()))
    if all((pages, c, lab) in gaps for lab in ("100", "300", "2,700")):
        eq = [gaps[(pages, c, lab)][3] in ("equivalent", "small but non-zero") for lab in ("100", "300")]
        big = gaps[(pages, c, "2,700")][0] > 0.2
        reg3("D2", case, "holds" if all(eq) and big else "fails",
             "; ".join(f"{lab}: {gaps[(pages, c, lab)][0]:+.3f} {ci_text(*gaps[(pages, c, lab)][1:3])}, {gaps[(pages, c, lab)][3]}"
                       for lab in ("100", "300", "2,700")))
    r = {lab: float(np.mean(recs[(pages, c, lab)])) for _, lab in LIVES[:4] if (pages, c, lab) in recs}
    if len(r) == 4:
        ok = r["100"] < r["300"] < r["900"] < r["2,700"] and r["900"] >= 1.5 * r["300"]
        reg3("D3", case, "holds" if ok else "fails", ", ".join(f"{k}: {x:.0f}" for k, x in r.items()))

# X: which side holds the lock
MORTAL_LYING = {("readable", 0.0): 0.776, ("hidden", 0.02): 0.729}      # registered reference levels (batch 8)
PARTS = [(None, "nothing"), ("pull", "readers"), ("write", "writers"), ("heads", "both heads"), ("all", "whole network")]
def part_setting(pages, c, part):
    return hyst_name(pages, c, "lying") + "_immortal" if part is None else reset_name(pages, c, part)
rows, xm = [], {}
for pages, c in DOSE_TARGETS:
    for part, lab in PARTS:
        n = part_setting(pages, c, part)
        if not logs_of(n):
            continue
        m, lo, hi = mean_ci(hv(n))
        xm[(pages, c, part)] = m
        rows.append({"own page": pages, "credit after switch": c, "re-initialised at the switch": lab,
                     "honest writes": round(m, 3), "95% CI": ci_text(lo, hi),
                     "follows": round(float(np.mean(hv(n, "follows"))), 3), "wiki right": round(float(np.mean(hv(n, "wiki right"))), 3),
                     "seeds": nseeds(n)})
lock_mechanism = pd.DataFrame(rows)
lock_mechanism.to_csv(os.path.join(PAPER_DIR, "lock_mechanism.csv"), index=False)
for test, part, rule, text in (("X1", "pull", lambda m, ref: m < 0.2, "below 0.2"), ("X2", "write", lambda m, ref: m >= 0.5, "at least 0.5"),
                               ("X3", "heads", lambda m, ref: m >= 0.5, "at least 0.5"),
                               ("X4", "all", lambda m, ref: abs(m - ref) <= 0.1, "within 0.1 of the mortal lying start")):
    for pages, c in DOSE_TARGETS:
        if (pages, c, part) in xm:
            m = xm[(pages, c, part)]
            reg3(test, f"{pages}, c = {c:+g}", "holds" if rule(m, MORTAL_LYING[(pages, c)]) else "fails",
                 f"{m:.3f} ({text}; mortal lying start {MORTAL_LYING[(pages, c)]})")
batch9_verdicts = pd.DataFrame(V3)
batch9_verdicts.to_csv(os.path.join(PAPER_DIR, "batch9_verdicts.csv"), index=False)
for title, table in (("learners that never die, all seeds", immortal_all), ("turnover dose", turnover_dose),
                     ("which side holds the lock", lock_mechanism), ("registered predictions", batch9_verdicts)):
    print("\n" + title)
    try:
        display(table)
    except NameError:
        print(table.to_string())

# time courses of every switch run, smoothed over 25 updates, every 3rd update kept (for the paper's figures)
series = {}
names = ([hyst_name(p, c, st) + sfx for p, ts in HYST_TARGETS.items() for c in ts for st in HYST_STARTS for sfx in ("", "_immortal")]
         + [n for n in EXTRA9])
for n in dict.fromkeys(names):
    for j, l in enumerate(logs_of(n)):
        x, y = smooth_series(l)
        series[f"{n}__{j}__x"] = x[::3].astype(np.float32)
        series[f"{n}__{j}__all"] = y[::3].astype(np.float32)
        series[f"{n}__{j}__adults"] = smooth_series(l, adults=True)[1][::3].astype(np.float32)
        series[f"{n}__{j}__follows"] = smooth_series(l, "follow_rate")[1][::3].astype(np.float32)
np.savez_compressed(os.path.join(PAPER_DIR, "history_series.npz"), **series)
print("saved", len(series) // 4, "time courses to history_series.npz")

def mean_course(n, key="honest_writes"):
    ys = [smooth_series(l, key)[1] for l in logs_of(n)]
    if not ys:
        return None, None
    k = min(len(y) for y in ys)
    return np.arange(k) + 12, np.mean([y[:k] for y in ys], 0)

# Fig. 19
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(LIVES)))
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
for k, (pages, c) in enumerate(DOSE_TARGETS):
    a = ax[0, k]
    for (life, lab), col in zip(LIVES, cols):
        for start, ls in (("lying", "-"), ("honest", "--")):
            x, y = mean_course(dose_setting(pages, c, start, life))
            if x is not None:
                a.plot(x, y, color=col, ls=ls, lw=1.2, label=f"life {lab}" if start == "lying" else None)
    a.axvline(SWITCH_U, color="k", ls=":", lw=0.8); a.set_ylim(0, 1.02); a.set_xlabel("update"); a.set_ylabel("honest writes")
    a.set_title(f"({'ab'[k]}) own page {pages}, c = {c:+g}; solid: lying start, dashed: honest start", fontsize=9)
    a.legend(fontsize=7, frameon=False); a.spines[["top", "right"]].set_visible(False)
xs = np.arange(len(LIVES))
for k, (pages, c) in enumerate(DOSE_TARGETS):
    g = [gaps.get((pages, c, lab)) for _, lab in LIVES]
    ok = [i for i, v in enumerate(g) if v is not None]
    ax[1, 0].errorbar(xs[ok] + 0.08 * (k - 0.5), [g[i][0] for i in ok], yerr=[[g[i][0] - g[i][1] for i in ok], [g[i][2] - g[i][0] for i in ok]],
                      marker="o", capsize=3, label=f"own page {pages}, c = {c:+g}")
    r = [recs.get((pages, c, lab)) for _, lab in LIVES[:4]]
    ok = [i for i, v in enumerate(r) if v is not None]
    ax[1, 1].errorbar(xs[ok] + 0.08 * (k - 0.5), [np.mean(r[i]) for i in ok],
                      yerr=[mean_ci(r[i])[2] - np.mean(r[i]) for i in ok], marker="o", capsize=3, label=f"own page {pages}, c = {c:+g}")
for a, t, yl in ((ax[1, 0], "(c) honest minus lying start, last third", "difference in honest writes"),
                 (ax[1, 1], "(d) recovery time of the lying start", "updates after the switch")):
    a.set_xticks(xs); a.set_xticklabels([lab for _, lab in LIVES]); a.set_xlabel("lifespan (updates)")
    a.set_title(t, fontsize=9); a.set_ylabel(yl); a.legend(fontsize=7, frameon=False); a.spines[["top", "right"]].set_visible(False)
ax[1, 0].axhline(0, color="gray", lw=0.6); ax[1, 1].set_xlim(-0.5, len(LIVES) - 0.5)
fig.tight_layout(); savefig(fig, "fig19_turnover_dose")

# Fig. 20
pcols = {"nothing": "#444441", "readers": "#378ADD", "writers": "#D85A30", "both heads": "#7F77DD", "whole network": "#1D9E75"}
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
for k, (pages, c) in enumerate(DOSE_TARGETS):
    for r_, key, yl in ((0, "honest_writes", "honest writes"), (1, "follow_rate", "pulls on the majority's arm")):
        a = ax[r_, k]
        for part, lab in PARTS:
            for j, l in enumerate(logs_of(part_setting(pages, c, part))):
                x, y = smooth_series(l, key)
                a.plot(x, y, color=pcols[lab], lw=0.6, alpha=0.6, label=lab if j == 0 else None)
        if r_ == 0:
            a.axhline(MORTAL_LYING[(pages, c)], color="k", ls="--", lw=0.7, label="mortal, lying start")
        a.axvline(SWITCH_U, color="k", ls=":", lw=0.8); a.set_ylim(0, 1.02); a.set_xlabel("update"); a.set_ylabel(yl)
        a.set_title(f"({'abcd'[r_ * 2 + k]}) never die, lying start, own page {pages}, c = {c:+g}", fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
ax[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout(); savefig(fig, "fig20_lock_mechanism")
print("figures saved to", FIG_DIR)
"""

EXTRA11_MD = r"""## 24. Corrections after the independent review

Uses the runs of `RUN_EXTRA11` (section 13) and the saved logs of earlier sections. Predictions were registered on
16 Nov 2026 at 23:45, before any runs. Saved to `batch11_group.csv`, `batch11_seen.csv`, `batch11_post_switch.csv`,
`batch11_evo_windows.csv` and `batch11_verdicts.csv`.

- **G re-tested, group selection in the shared wiki.** The earlier group-selection runs replaced the juvenile periods
  of all sixteen slots whenever agent 0 died, so living adults could become juveniles again. Here (`juvenile_group_birth`)
  each copy inherits the fitter copy's juvenile periods once per lifetime, judged by mean reward per agent-step since the
  last selection, and each heir takes its slot's inherited value at birth.
- **V, private notebooks with credit only for entries readers can see** (`credit_seen_only`). Earlier, an author was
  paid when another agent won on the arm its unseen notebook named.
- **W, measurement windows** (no new runs). Newborn and adult reward of the mortal credit-switch runs over lives born at
  least one lifetime after the switch, and newborn reward of the evolution runs over lives born in the last third.
"""

EXTRA11_FIG = r"""# --- section 24: corrections after the independent review
V5 = []
def reg5(test, case, ok, detail):
    V5.append({"prediction": test, "case": case, "result": "holds" if ok else "fails", "detail": detail})

# G re-tested: individual selection against group selection with values taken at birth
gb = []
for sel, fmt in (("individual", "evo_wiki_from{}_long"), ("group, at birth", "evo_wiki_groupb_from{}_long"),
                 ("group, original", "evo_wiki_group_from{}_long")):
    for j0 in J0S:
        L = logs_of(fmt.format(j0))
        if not L:
            continue
        m_, lo, hi = mean_ci([final_J(l) for l in L])
        neu = [float(neutral_J(l)[-PAPER_LIFE:].mean()) for l in L]
        dm, dlo, dhi = mean_ci([final_J(l) - n for l, n in zip(L, neu)])
        nbv = [newborn_late(l) for l in L]
        nb, nlo, nhi = mean_ci(nbv)
        gb.append({"selection": sel, "start": j0, "seeds": len(L), "final juvenile period": round(m_, 1),
                   "95% CI": f"[{lo:.1f}, {hi:.1f}]", "control": round(float(np.mean(neu)), 1),
                   "selection effect": round(dm, 1), "selection 95% CI": f"[{dlo:+.1f}, {dhi:+.1f}]",
                   "newborn reward, last third": round(nb, 3), "newborn 95% CI": f"[{nlo:.3f}, {nhi:.3f}]",
                   "_lo": lo, "_hi": hi, "_dhi": dhi, "_nb": nbv})
gb_df = pd.DataFrame([{k: v for k, v in g.items() if not k.startswith("_")} for g in gb])
gb_df.to_csv(os.path.join(PAPER_DIR, "batch11_group.csv"), index=False)
GB = {(g["selection"], g["start"]): g for g in gb}
j150 = J0S[1]
if ("group, at birth", j150) in GB and ("individual", j150) in GB:
    a_, b_ = GB[("group, at birth", j150)], GB[("individual", j150)]
    reg5("G1b", "from 150: group shorter than individual, intervals not overlapping", a_["_hi"] < b_["_lo"],
         f"group {a_['final juvenile period']} {a_['95% CI']}, individual {b_['final juvenile period']} {b_['95% CI']}")
    d, lo, hi, v = xverdict(a_["_nb"], b_["_nb"], 0.02)
    reg5("G3b", "from 150: newborns earn more under group selection", lo > 0, f"group - individual {d:+.3f} {ci_text(lo, hi)}, {v}")
if ("group, at birth", 0) in GB:
    a_ = GB[("group, at birth", 0)]
    reg5("G2b", "from 0: group-selection effect negative", a_["_dhi"] < 0,
         f"selection effect {a_['selection effect']:+} {a_['selection 95% CI']}")

# V: private notebooks, credit only for entries readers can see
sv = []
for a, b in SEEN_PAIRS:
    if not (logs_of(a + "_seen") and logs_of(b + "_seen")):
        continue
    d, lo, hi, v = xverdict(hv(a + "_seen", "newborn reward"), hv(b + "_seen", "newborn reward"), 0.02)
    d0, lo0, hi0, v0 = xverdict(hv(a, "newborn reward"), hv(b, "newborn reward"), 0.02)
    sv.append({"inherited": a, "erased": b, "inherited - erased, seen only": round(d, 3), "95% CI": ci_text(lo, hi),
               "verdict": v, "inherited - erased, original": round(d0, 3), "original 95% CI": ci_text(lo0, hi0),
               "newborn inherited, seen only": round(m_of(a + "_seen", "newborn reward"), 3),
               "newborn erased, seen only": round(m_of(b + "_seen", "newborn reward"), 3),
               "adult inherited, seen only": round(m_of(a + "_seen", "adult reward"), 3),
               "honest writes inherited, seen only": round(m_of(a + "_seen"), 3),
               "honest writes erased, seen only": round(m_of(b + "_seen"), 3),
               "honest writes inherited, original": round(m_of(a), 3), "_d": d, "_lo": lo, "_v": v, "_d0": d0})
sv_df = pd.DataFrame([{k: x for k, x in r.items() if not k.startswith("_")} for r in sv])
sv_df.to_csv(os.path.join(PAPER_DIR, "batch11_seen.csv"), index=False)
SV = {r["inherited"]: r for r in sv}
for a in ("notebook_cited", "notebook_cited_static"):
    if a in SV:
        reg5("V1", f"{a}: inherited - erased equivalent within 0.02", SV[a]["_v"] == "equivalent",
             f"{SV[a]['_d']:+.3f} {SV[a]['95% CI']}, {SV[a]['_v']}")
for a, m in (("notebook_cited_static_juv", 0.10), ("notebook_torso_cited_static_juv", 0.20)):
    if a in SV:
        reg5("V2", f"{a}: inherited - erased above +{m:g}", SV[a]["_lo"] > m, f"{SV[a]['_d']:+.3f} {SV[a]['95% CI']}")
if sv:
    worst = max(abs(r["_d"] - r["_d0"]) for r in sv)
    reg5("V3", "every pair within 0.05 of the original credit rule", worst < 0.05, f"largest change {worst:.3f}")
    hw = [r[k] for r in sv for k in ("honest writes inherited, seen only", "honest writes erased, seen only")]
    reg5("V4", "notebook honest writes between 0.60 and 0.90", all(0.60 <= x <= 0.90 for x in hw),
         f"range {min(hw):.3f} to {max(hw):.3f}")

# W: measurement windows, from the saved logs
def birth_of(l, i, g):
    return int(np.nonzero(l["gen"][:, i] == g)[0][0])

def newborn_after(l, u0):
    v = [r_[:EARLY].mean() for i, g, r_, _ in lives(l) if birth_of(l, i, g) >= u0 and len(r_) >= EARLY]
    return float(np.mean(v)) if v else np.nan

def adult_after(l, u0):
    v = [r_[-EARLY:].mean() for i, g, r_, c in lives(l) if c and birth_of(l, i, g) >= u0 and len(r_) > 2 * EARLY]
    return float(np.mean(v)) if v else np.nan

U0 = SWITCH_AT // STEPS + PAPER_LIFE                 # one lifetime after the switch
ps, n_eq = [], 0
for pages, targets in HYST_TARGETS.items():
    for c in targets:
        Lh, Ll = logs_of(hyst_name(pages, c, "honest")), logs_of(hyst_name(pages, c, "lying"))
        if not (Lh and Ll):
            continue
        nh, nl = [newborn_after(l, U0) for l in Lh], [newborn_after(l, U0) for l in Ll]
        ah, al = [adult_after(l, U0) for l in Lh], [adult_after(l, U0) for l in Ll]
        d, lo, hi, v = xverdict(nh, nl, 0.02)
        da, loa, hia, va = xverdict(ah, al, 0.02)
        n_eq += v == "equivalent" or v == "small but non-zero"
        ps.append({"own page": pages, "credit after switch": c, "seeds": f"{len(Lh)}/{len(Ll)}",
                   "newborn, honest start": round(float(np.nanmean(nh)), 3), "newborn, lying start": round(float(np.nanmean(nl)), 3),
                   "newborn difference": round(d, 3), "newborn 95% CI": ci_text(lo, hi), "newborn verdict": v,
                   "adult, honest start": round(float(np.nanmean(ah)), 3), "adult, lying start": round(float(np.nanmean(al)), 3),
                   "adult difference": round(da, 3), "adult 95% CI": ci_text(loa, hia), "adult verdict": va})
ps_df = pd.DataFrame(ps)
ps_df.to_csv(os.path.join(PAPER_DIR, "batch11_post_switch.csv"), index=False)
if ps:
    reg5("W1", "newborns equivalent after the switch at >= 5 of 6 targets", n_eq >= 5, f"{n_eq} of {len(ps)} equivalent within 0.02")
ew = []
for name in PAPER_SETTINGS:
    if not name.startswith("evo_"):
        continue
    L = logs_of(name)
    if not L:
        continue
    m_, lo, hi = mean_ci([newborn_late(l) for l in L])
    ew.append({"setting": name, "seeds": len(L), "newborn reward, last third": round(m_, 3), "95% CI": ci_text(lo, hi),
               "newborn reward, whole run": round(m_of(name, "newborn reward"), 3)})
pd.DataFrame(ew).to_csv(os.path.join(PAPER_DIR, "batch11_evo_windows.csv"), index=False)

v5_df = pd.DataFrame(V5)
v5_df.to_csv(os.path.join(PAPER_DIR, "batch11_verdicts.csv"), index=False)
for title, table in (("group selection, values taken at birth", gb_df), ("private notebooks, credit only when seen", sv_df),
                     ("after the switch", ps_df), ("evolution runs, newborns in the last third", pd.DataFrame(ew)),
                     ("registered predictions", v5_df)):
    print("\n" + title)
    display(table) if len(table) else print("  (no runs yet)")
"""

EXTRA12_MD = r"""## 25. Why group selection lagged

Uses the runs of `RUN_EXTRA12` (section 13) and the saved runs of sections 23 and 24. Predictions were registered on
21 Nov 2026 at 11:55, before any code or runs. Saved to `batch12_group.csv` and `batch12_verdicts.csv`; Fig. 24.

In section 24, group selection once a lifetime shortened a 150-update juvenile period less than individual selection.
Here group selection acts at every death, exactly as individual selection does (the heir copies its slot's value from
the fitter of two random copies and mutates it), but fitness is the copy's mean reward per agent and step over the dead
agent's life, not that agent's own reward (`juvenile_group_each_death`). Only what selection sees differs.

- **Y1:** from 150, ends shorter than group selection once a lifetime (difference below zero, 95% interval).
- **Y2:** from 150, the difference from individual selection includes zero or lies below it.
- **Y3:** from 150, newborns earn more than under group selection once a lifetime, and the difference from individual
  selection includes zero or lies above it.
- **Y4:** from 0, the selection effect is negative (95% interval below zero).
"""

EXTRA12_FIG = r"""# --- section 25: why group selection lagged
V6 = []
def reg6(test, case, ok, detail):
    V6.append({"prediction": test, "case": case, "result": "holds" if ok else "fails", "detail": detail})

KINDS12 = (("individual", "evo_wiki_from{}_long"), ("group, once a lifetime", "evo_wiki_groupb_from{}_long"),
           ("group, every death", "evo_wiki_groupd_from{}_long"))
g6 = []
for sel, fmt in KINDS12:
    for j0 in J0S:
        L = logs_of(fmt.format(j0))
        if not L:
            continue
        fj = [final_J(l) for l in L]
        m_, lo, hi = mean_ci(fj)
        neu = [float(neutral_J(l)[-PAPER_LIFE:].mean()) for l in L]
        dm, dlo, dhi = mean_ci([f - n for f, n in zip(fj, neu)])
        nbv = [newborn_late(l) for l in L]
        nb, nlo, nhi = mean_ci(nbv)
        g6.append({"selection": sel, "start": j0, "seeds": len(L), "final juvenile period": round(m_, 1),
                   "95% CI": f"[{lo:.1f}, {hi:.1f}]", "control": round(float(np.mean(neu)), 1),
                   "selection effect": round(dm, 1), "selection 95% CI": f"[{dlo:+.1f}, {dhi:+.1f}]",
                   "newborn reward, last third": round(nb, 3), "newborn 95% CI": f"[{nlo:.3f}, {nhi:.3f}]",
                   "_fj": fj, "_dhi": dhi, "_nb": nbv})
g6_df = pd.DataFrame([{k: v for k, v in g.items() if not k.startswith("_")} for g in g6])
g6_df.to_csv(os.path.join(PAPER_DIR, "batch12_group.csv"), index=False)
G6 = {(g["selection"], g["start"]): g for g in g6}

def jdiff(a, b):
    d, lo, hi, _ = xverdict(a, b, 10.0)
    return d, lo, hi, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]"

j150 = J0S[1]
D, B, I = (("group, every death", j150), ("group, once a lifetime", j150), ("individual", j150))
if D in G6 and B in G6:
    d, lo, hi, t = jdiff(G6[D]["_fj"], G6[B]["_fj"])
    reg6("Y1", "from 150: every death minus once a lifetime below zero", hi < 0, f"{t} updates")
if D in G6 and I in G6:
    d, lo, hi, t = jdiff(G6[D]["_fj"], G6[I]["_fj"])
    reg6("Y2", "from 150: every death minus individual includes zero or lies below", lo <= 0, f"{t} updates")
if D in G6 and B in G6 and I in G6:
    nd, nb_ = np.nanmean(G6[D]["_nb"]), np.nanmean(G6[B]["_nb"])
    d, lo, hi, v = xverdict(G6[D]["_nb"], G6[I]["_nb"], 0.02)
    reg6("Y3", "from 150: newborns above once a lifetime; difference from individual includes zero or lies above",
         nd > nb_ and hi >= 0, f"every death {nd:.3f}, once a lifetime {nb_:.3f}; minus individual {d:+.3f} {ci_text(lo, hi)}, {v}")
if ("group, every death", 0) in G6:
    a_ = G6[("group, every death", 0)]
    reg6("Y4", "from 0: selection effect negative", a_["_dhi"] < 0,
         f"selection effect {a_['selection effect']:+} {a_['selection 95% CI']}")
v6_df = pd.DataFrame(V6)
v6_df.to_csv(os.path.join(PAPER_DIR, "batch12_verdicts.csv"), index=False)

# Fig. 24: mean juvenile period over 36 lifetimes under the three kinds of selection
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for k, j0 in enumerate(J0S):
    a = ax[k]
    for (sel, fmt), col in zip(KINDS12, ("#2a6fdb", "#e07b39", "#3a9d5d")):
        L = logs_of(fmt.format(j0))
        if not L:
            continue
        curves = np.array([l["J_mean"].mean(1) for l in L])
        x = np.arange(curves.shape[1]) / PAPER_LIFE
        for c in curves:
            a.plot(x, c, color=col, lw=0.6, alpha=0.35)
        a.plot(x, curves.mean(0), color=col, lw=2, label=sel)
    L = logs_of(KINDS12[0][1].format(j0))
    if L:
        a.plot(np.arange(len(L[0]["J_mean"])) / PAPER_LIFE, np.mean([neutral_J(l) for l in L], 0), color="grey",
               ls="--", lw=1, label="random parents")
    a.set_title(f"({'ab'[k]}) shared wiki, start {j0} updates", fontsize=9)
    a.set_xlabel("lifetimes"); a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("mean juvenile period (updates)")
ax[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); savefig(fig, "fig24_group_timing")

for title, table in (("selection in the shared wiki, 36 lifetimes", g6_df), ("registered predictions", v6_df)):
    print("\n" + title)
    display(table) if len(table) else print("  (no runs yet)")
"""

READ_MD = r"""## 10. How to read the results

- **Does the wiki speed newborns up?** Compare newborn reward with `no_wiki`
  (floor) and `oracle` (ceiling). The gap closed by a setting is how much of a
  perfect wiki's value the agents actually build and pass on.
- **Does knowledge outlive its writers?** "Wiki right" staying high late in the
  run, after every original agent has died several times, means the wiki is
  maintained across generations.
- **Honest or deceptive?** In *competitive* settings misleading others pays: watch
  honest writes fall, and whether readers learn to stop following (follows drops,
  real − scrambled near 0).
- **Is citation credit what makes it work?** Compare each setting with its `_cited`
  twin.
- **Invention vs choice.** `symbol_cited` asks agents to invent what symbols mean;
  if it stays near `no_wiki`, invention is the hard part, as in the pilots.
- **Bad actors (sections 11–12).** Named vs anonymous at the same number of bad
  agents measures how much stable identity protects a shared memory. Deceivers
  vs incompetents separates coordinated lies from noise. A point below the no-wiki
  line means the wiki has become a liability.

**Dials to try next** (in `BanditConfig`): `p_high`/`p_low` closer together
(noisier machines, harder knowledge), `n_bandits` (more machines than one agent
can learn), `drift_period` (how fast knowledge goes stale), `citation` (size of
the credit), `write_cost`; and `life_updates` in `run_world`.
"""


def py_to_cells(path):
    text = open(path).read()
    cells = []
    for block in re.split(r"^# %%", text, flags=re.M):
        if not block.strip():
            continue
        if block.startswith(" [markdown]"):
            lines = block.split("\n")[1:]
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines).strip()
            cells.append(nbf.v4.new_markdown_cell(md))
        else:
            cells.append(nbf.v4.new_code_cell(block.split("\n", 1)[1].strip()))
    return cells


life, gens, seeds, demo_life, demo_gens, ladder_steps = (
    (30, 3, "[0]", 30, 2, 600) if QUICK else (300, 6, "[0, 1, 2]", 300, 3, 6000))
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_markdown_cell(PILOT_MD), nbf.v4.new_code_cell(SETUP)]
nb.cells += py_to_cells("wikibandits.py")
nb.cells += py_to_cells("baselines.py")
nb.cells += [nbf.v4.new_code_cell(f"STEPS_LADDER = {ladder_steps}\n" + LADDER_CODE)]
nb.cells += py_to_cells("analysis_wb.py")
nb.cells += [
    nbf.v4.new_code_cell(f"DEMO_LIFE, DEMO_GENS = {demo_life}, {demo_gens}   # demo: updates per lifetime, lifetimes per agent"),
    nbf.v4.new_markdown_cell(DEMO_MD), nbf.v4.new_code_cell(DEMO),
    nbf.v4.new_markdown_cell(EXP_MD),
    nbf.v4.new_code_cell(EXP.replace("SEEDS_", seeds).replace("LIFE_ ", f"{life} ").replace("GENS_ ", f"{gens} ")),
    nbf.v4.new_markdown_cell(COMPARE_MD), nbf.v4.new_code_cell(COMPARE),
    nbf.v4.new_markdown_cell(ADV_MD), nbf.v4.new_code_cell(ADV_EXP),
    nbf.v4.new_markdown_cell(ADV_CMP_MD), nbf.v4.new_code_cell(ADV_CMP),
    nbf.v4.new_markdown_cell(PAPER_MD),
    nbf.v4.new_code_cell(PAPER_RUNS.replace("REQ_", "False" if QUICK else "True")
                         .replace("PSEEDS_", "[0]" if QUICK else "[0, 1, 2]")
                         .replace("PLIFE_", "30" if QUICK else "300").replace("PGENS_", "3" if QUICK else "6")
                         .replace("PENVS_", "64 if not ON_GPU else 256" if QUICK else "256")
                         .replace("ESEEDS_", "[1]" if QUICK else "[3, 4, 5, 6, 7, 8, 9]")
                         .replace("TIER2_", "True")),
    nbf.v4.new_markdown_cell(FIG_MD), nbf.v4.new_code_cell(FIG),
    nbf.v4.new_markdown_cell(ROBUST_MD), nbf.v4.new_code_cell(ROBUST_FIG),
    nbf.v4.new_markdown_cell(EXTRA2_MD), nbf.v4.new_code_cell(EXTRA2_FIG),
    nbf.v4.new_markdown_cell(EXTRA3_MD), nbf.v4.new_code_cell(EXTRA3_FIG),
    nbf.v4.new_markdown_cell(EXTRA5_MD), nbf.v4.new_code_cell(EXTRA5_FIG),
    nbf.v4.new_markdown_cell(EXTRA7_MD), nbf.v4.new_code_cell(EXTRA7_FIG),
    nbf.v4.new_markdown_cell(EXTRA8_MD), nbf.v4.new_code_cell(EXTRA8_FIG),
    nbf.v4.new_markdown_cell(ADULT_MD), nbf.v4.new_code_cell(ADULT_FIG),
    nbf.v4.new_markdown_cell(EXTRA9_MD), nbf.v4.new_code_cell(EXTRA9_FIG),
    nbf.v4.new_markdown_cell(EXTRA10_MD), nbf.v4.new_code_cell(EXTRA10_FIG),
    nbf.v4.new_markdown_cell(EXTRA11_MD), nbf.v4.new_code_cell(EXTRA11_FIG),
    nbf.v4.new_markdown_cell(EXTRA12_MD), nbf.v4.new_code_cell(EXTRA12_FIG),
    nbf.v4.new_markdown_cell(READ_MD),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
out = "WikiBandits_quick.ipynb" if QUICK else "WikiBandits.ipynb"
nbf.write(nb, out)
print("wrote", out, len(nb.cells), "cells")
