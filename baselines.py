# %% [markdown]
# ## 5. The baseline ladder (no learning, just reference strategies)
#
# Five hand-written strategies show how much there is to gain from shared
# knowledge in this world. All of them know its rules (one good arm per machine,
# the payout odds, how often machines change), so they are the best one can do
# with a given kind of memory, not just "reasonable" strategies:
#
# 1. **random**: pull arms at random.
# 2. **best player, no memory**: forgets a machine as soon as it leaves; every
#    visit starts from scratch. Computed exactly (dynamic programming over what it
#    can see during a visit), so it is the true ceiling for agents without a wiki.
# 3. **Bayesian player, own memory**: remembers everything it ever saw on every
#    machine and knows that old evidence goes stale as machines change.
# 4. **Bayesian shared memory**: everyone pools every observation perfectly (what a
#    perfectly honest, perfectly informative wiki could give).
# 5. **omniscient**: always pulls the good arm.
#
# Rungs 3 and 4 pull the arm most likely to be good (Bayesian greedy), which is
# close to optimal here. Learned agents without a wiki can at best reach rung 2;
# with a wiki, rung 4.

# %%
from functools import lru_cache

import numpy as np


def best_no_memory(n_arms, p_high, p_low, visit_len):
    """Exact best expected reward per pull over one visit, starting from scratch."""
    lw, ll = np.log(p_high / p_low), np.log((1 - p_high) / (1 - p_low))

    @lru_cache(maxsize=None)
    def value(state, left):                     # state: sorted (wins, losses) per arm
        if left == 0:
            return 0.0
        logit = np.array([w * lw + l * ll for w, l in state])
        b = np.exp(logit - logit.max())
        b /= b.sum()                            # chance that each arm is the good one
        best, seen = 0.0, set()
        for i, (w, l) in enumerate(state):
            if (w, l) in seen:                  # arms with the same record are interchangeable
                continue
            seen.add((w, l))
            p = p_low + b[i] * (p_high - p_low)
            win = tuple(sorted(state[:i] + ((w + 1, l),) + state[i + 1:]))
            lose = tuple(sorted(state[:i] + ((w, l + 1),) + state[i + 1:]))
            best = max(best, p * (1 + value(win, left - 1)) + (1 - p) * value(lose, left - 1))
        return best

    return value(tuple([(0, 0)] * n_arms), visit_len) / visit_len


def simulate_baselines(n_agents=16, n_bandits=8, n_arms=5, p_high=0.9, p_low=0.1,
                       visit_len=8, drift_period=2000, steps=20000, seed=0):
    """Mean reward per pull for the five reference strategies."""
    N, K, A, V = n_agents, n_bandits, n_arms, visit_len
    h = 1.0 / drift_period
    lam = 1 - h * A / (A - 1)                   # how much belief survives one step of possible drift
    out = {}
    for strat in ("random", "best player, no memory", "Bayesian player, own memory",
                  "Bayesian shared memory", "omniscient"):
        if strat == "best player, no memory":
            out[strat] = best_no_memory(A, p_high, p_low, V)
            continue
        rng = np.random.default_rng(seed)
        best = rng.integers(A, size=K)
        own = np.full((N, K, A), 1.0 / A)       # each agent's belief about each machine
        shared = np.full((K, A), 1.0 / A)       # pooled belief
        rows = np.arange(N)
        total = 0.0
        for t in range(steps):
            if t % V == 0:
                where = rng.integers(K, size=N)
            if strat == "random":
                a = rng.integers(A, size=N)
            elif strat == "omniscient":
                a = best[where]
            else:
                bel = own[rows, where] if strat == "Bayesian player, own memory" else shared[where]
                a = np.argmax(bel + 1e-9 * rng.random((N, A)), axis=1)   # random among ties
            r = rng.random(N) < np.where(a == best[where], p_high, p_low)
            total += r.sum()
            lik = np.repeat(np.where(r, p_low, 1 - p_low)[:, None], A, axis=1)
            lik[rows, a] = np.where(r, p_high, 1 - p_high)
            own[rows, where] *= lik
            np.multiply.at(shared, where, lik)
            own /= own.sum(-1, keepdims=True)
            shared /= shared.sum(-1, keepdims=True)
            own = lam * own + (1 - lam) / A
            shared = lam * shared + (1 - lam) / A
            switch = rng.random(K) < h
            best = np.where(switch, (best + rng.integers(1, A, size=K)) % A, best)
        out[strat] = total / (steps * N)
    return out
