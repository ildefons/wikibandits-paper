# %% [markdown]
# ## 6. Analysis tools
#
# - **Reward vs the ladder**: where learned agents land between the reference
#   strategies of section 5.
# - **Newborns**: reward by age for each generation. Later generations learning
#   faster than earlier ones would be a cultural ratchet.
# - **Wiki quality**: share of machines whose entries' majority names the right
#   arm; share of writes that are correct; share that are *honest* (match what
#   the writer itself believes).
# - **Behaviour**: how often agents write, and how often a reader pulls the arm
#   that the other authors' majority names.

# %%
import matplotlib.pyplot as plt


def smooth(x, k=10):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def learner_mask(log):
    """Slots played by learners (not scripted deceivers or incompetents)."""
    return np.asarray(log.get("learner_slots", np.ones(log["raw"].shape[1], bool)))


def lives(log, key="raw", min_gen=1):
    """Yield (agent, generation, values-by-age, complete?) for each learner's life."""
    gen, val = log["gen"], log[key]
    U, N = val.shape
    L = learner_mask(log)
    for i in range(N):
        if not L[i]:
            continue
        for g in np.unique(gen[:, i]):
            if g < min_gen:
                continue
            idx = np.where(gen[:, i] == g)[0]
            yield i, int(g), val[idx, i], idx[-1] < U - 1


def newborn_reward(log, early=50):
    v = [r[:early].mean() for _, _, r, _ in lives(log) if len(r) >= early]
    return float(np.mean(v)) if v else np.nan


def adult_reward(log, late=50):
    v = [r[-late:].mean() for _, _, r, c in lives(log) if c and len(r) > 2 * late]
    return float(np.mean(v)) if v else np.nan


def late(log, key, frac=1 / 3):
    if key not in log:                             # results saved by an older version
        return np.nan
    x = log[key]
    if x.ndim == 2:
        x = x[:, learner_mask(log)]                # learners only
    x = x[-max(1, int(len(x) * frac)):]
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def generation_curves(log):
    by_gen = {}
    for _, g, r, _ in lives(log):
        by_gen.setdefault(g, []).append(r)
    return {g: np.mean([r[:min(map(len, rs))] for r in rs], 0) for g, rs in sorted(by_gen.items())}


def recovery_time(log, level, k=5):
    """Median number of updates from birth until a newborn's smoothed reward first reaches `level`
    (complete lives after the first generation). NaN when most newborns never get there."""
    times = []
    for _, _, r, done in lives(log):
        if not done or len(r) < k:
            continue
        hit = np.nonzero(smooth(r, k) >= level)[0]
        times.append(hit[0] + k - 1 if len(hit) else np.inf)
    if not times:
        return np.nan
    m = float(np.median(times))
    return m if np.isfinite(m) else np.nan


def event_dip(series, t0, pre=50, post=300, k=5, tol=0.02, target=None):
    """Dip and recovery of a series after an event at update t0 (a die-off, a bottleneck, liars switching).
    Returns (level before, lowest smoothed level after, updates from t0 until the smoothed series is back
    to `target`, by default the level before minus `tol`). Recovery is 0 when the series never drops below
    the target, and NaN when it does not get back within `post` updates."""
    x = np.asarray(series, float)
    base = float(np.nanmean(x[max(0, t0 - pre):t0])) if t0 > 0 else np.nan
    target = base - tol if target is None else target
    y = smooth(x[t0:min(len(x), t0 + post)], k)          # y[j] averages updates t0 + j ... t0 + j + k - 1
    if len(y) == 0 or not np.isfinite(y).any():
        return base, np.nan, np.nan
    j_min = int(np.nanargmin(y))
    if y[j_min] >= target:
        return base, float(y[j_min]), 0.0
    back = np.nonzero(y[j_min:] >= target)[0]
    return base, float(y[j_min]), (float(j_min + back[0] + k - 1) if len(back) else np.nan)


def cohort_reward(log, cohort, t0, early=50):
    """Mean reward per pull of the agents in `cohort` (a boolean mask) over their first `early` updates
    after being reborn together at update t0."""
    return float(np.nanmean(log["raw"][t0:t0 + early][:, cohort]))


def welch_diff(a, b, margin=0.02):
    """Difference of means a - b across seeds with a 95% confidence interval (Welch).
    Verdict: 'same within ±margin' if the whole interval lies inside ±margin, 'different' if it
    excludes zero, otherwise 'unclear'."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, "-"
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d, se = a.mean() - b.mean(), np.sqrt(va + vb)
    if se == 0:
        lo = hi = d
    else:
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        t = stats.t.ppf(0.975, df)
        lo, hi = d - t * se, d + t * se
    if -margin <= lo and hi <= margin:
        verdict = f"same within ±{margin:g}"
    elif lo > 0 or hi < 0:
        verdict = "different"
    else:
        verdict = "unclear"
    return d, lo, hi, verdict


def plot_run(log, title="", ladder=None):
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    L = learner_mask(log)
    ax[0].plot(smooth(np.nanmean(log["raw"][:, L], 1)), color="k", lw=1.2, label="learners")
    if ladder:
        for name, v in ladder.items():
            ax[0].axhline(v, lw=0.8, ls=":", color="gray")
            ax[0].text(0, v, f" {name}", fontsize=7, va="bottom", color="gray")
    ax[0].set_title("reward per pull (dotted: baseline ladder)")
    ax[0].set_xlabel("update")
    for g, c in generation_curves(log).items():
        ax[1].plot(smooth(c), color=plt.cm.viridis(g / max(1, log["gen"].max())), label=f"generation {g}")
    ax[1].set_title("newborns: reward by age")
    ax[1].set_xlabel("updates since birth")
    ax[1].legend(fontsize=7, frameon=False)
    ax[2].plot(smooth(log["majority_ok"]), label="machines whose wiki majority is right")
    ax[2].plot(smooth(np.nanmean(log["correct_writes"][:, L], 1)), label="learners' writes that are correct")
    ax[2].plot(smooth(np.nanmean(log["honest_writes"][:, L], 1)), label="learners' writes that are honest")
    ax[2].set_ylim(0, 1.02)
    ax[2].set_title("wiki quality")
    ax[2].set_xlabel("update")
    ax[2].legend(fontsize=7, frameon=False)
    ax[3].plot(smooth(np.nanmean(log["write_rate"][:, L], 1)), label="learners' writes per pull")
    ax[3].plot(smooth(np.nanmean(log["follow_rate"][:, L], 1)), label="pulls following the others' majority")
    if (~L).any():
        ax[3].plot(smooth(np.nanmean(log["agree_bad"][:, L], 1)), label="pulls agreeing with bad authors")
        ax[3].plot(smooth(np.nanmean(log["agree_good"][:, L], 1)), label="pulls agreeing with learner authors")
    ax[3].set_ylim(0, 1.02)
    ax[3].set_title("behaviour")
    ax[3].set_xlabel("update")
    ax[3].legend(fontsize=7, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
