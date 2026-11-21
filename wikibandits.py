# %% [markdown]
# ## 2. The world
#
# **Bandits.** There are `n_bandits` slot machines, each with `n_arms` arms. On
# each machine exactly one arm is good (pays 1 with probability `p_high`); the
# others pay with probability `p_low`. Which arm is good is hidden, and on each
# machine it changes now and then (on average every `drift_period` steps), so
# knowledge goes stale.
#
# **Visits.** Every `visit_len` steps each agent is sent to a random machine and
# pulls it `visit_len` times. During a visit it remembers its own pulls and wins
# on that machine; when it leaves, it forgets them. So an agent can't remember a
# machine from one visit to the next, except through the wiki.
#
# **Wikis.** Every agent owns a public page with one entry per machine (blank or
# an arm number). Each step an agent may write on *its own* entry for the
# machine it is playing: keep it, or set it to one of the arms. Nobody can edit
# anyone else's page. While playing a machine, an agent reads **every agent's
# entry for that machine**, always in the same author order, so it can learn
# whom to trust. An entry can be true, false, or a private code.
#
# **Death.** Wikis persist. When an agent dies, its newborn successor takes over
# the same slot and page, and so inherits its predecessor's entries and
# reputation.
#
# **Incentives** (`reward_mode`):
# - `individual`: own reward only (others' success neither helps nor hurts);
# - `cooperative`: own reward plus the others' average (others' success counts
#   as much as your own);
# - `competitive`: own reward minus the others' average (misleading others pays).
#
# **Write mode.** `report` (default): the agent only chooses *whether and how*
# to write: keep its entry, report what it believes (the arm that has paid best
# in its current visit), or report a false arm. The content is innate; honesty is
# learned. `symbol`: an agent writes any arm number, so what a symbol means is up
# to the agents (in pilots this never got off the ground).
#
# **Citation credit** (`citation > 0`): when a reader pulls the arm named in
# some authors' entries and wins, those authors share a bonus of `citation`,
# like karma or being cited. It rewards writing that others successfully use.
# A *negative* `citation` is informational rivalry: it costs you when a rival
# successfully uses what you wrote.
# **Blame** (`blame > 0`) is the mirror image: authors share a penalty when a
# reader follows their entry and loses.
#
# **Rivalry** (`rivalry`, competitive mode only) scales how much others' success
# costs you: 1 = own minus others' average; 15 (with 16 agents) = own minus the
# sum of others' rewards, i.e. zero-sum, where a rival's win costs you as much
# as your own win gains you.
#
# **Shuffled control** (`reward_mode="shuffled"`): own reward minus the others'
# average *from a different copy of the world*. Same arithmetic as competitive,
# but nothing you do can affect it: it separates the effect of rivalry from the
# mere effect of subtracting an average.
#
# **Wipe on death** (`wipe_on_death=True`): a dying agent's page is erased, so
# its knowledge dies with it (the default keeps the page for the successor).
#
# **Bad actors.** `n_deceivers` and `n_incompetent` turn the last slots into
# scripted authors. They play like everyone else, but at the end of each visit a
# *deceiver* writes a bad arm (by default all deceivers name the same bad arm on a
# machine, so they can outvote honest authors) and an *incompetent* writes a
# random arm. They stay bad across generations. `n_saboteurs` adds *learning* bad
# authors (the slots just before the scripted ones): ordinary learners whose reward
# is minus the honest learners' average reward, so they learn whatever writing
# hurts others most.
#
# **What a reader sees.** By default every author's page. `read_others=False`: only
# its own page (a private notebook). `read_k=k`: its own page plus the pages of the
# next k slots (a small commons). `replicate_own=True`: its own entry repeated in
# every slot (as much redundancy as a wiki, but no sharing).
#
# **Identity attacks.** `identity_period=R`: named authors, but every R steps all
# names are reshuffled, so reputations reset. `deceive_after=T`: deceivers write the
# truth until step T, then lie ("sleepers" who build trust first).
#
# **Mass die-offs.** `sync_deaths=True`: everyone dies at the same moment, so no
# experienced agent is left to keep the record up to date. `juvenile_updates=J`:
# newborns may not write during their first J updates (a juvenile period).
# `juvenile_evolve=True` makes the juvenile period heritable: each agent in each
# world copy has its own; when an agent dies, its heir copies the juvenile period
# of the more successful of two random world copies (same slot, so same brain,
# judged by the reward of the life that just ended) and mutates it. With
# `sync_deaths` and `juvenile_group=True`, selection acts on whole world copies
# instead: at each die-off, every copy takes all the juvenile periods of the
# fitter of two random copies (judged by their agents' mean reward).
# `juvenile_group_birth=True` is the same group selection for staggered deaths:
# the fitter copy's juvenile periods become each copy's inherited values, and an
# heir takes its slot's inherited value when it is born (living agents keep theirs).
# `juvenile_group_each_death=True` is group selection with the timing of individual
# selection: at every death, the heir copies its slot's juvenile period from the
# fitter of two random copies, judged by the copy's mean reward (all agents) over
# the dead agent's life rather than by that agent's own reward.
# `bottleneck_at=u`,
# `bottleneck_frac=f`: at update u a random fraction f of the agents die at once,
# while the rest live on (the cohort is logged in `bottleneck_cohort`).
#
# **Inheriting skill.** `keep_torso=True`: a newborn keeps its predecessor's hidden
# layers (its learned features) and gets new output heads and a fresh optimiser.
#
# **Identity.** `named`: every reader sees the authors in the same fixed order, so
# it can learn whom to trust. `anonymous`: each reader sees its own entry first
# and everyone else's in a fresh random order every step, so reputations can't be
# tracked (like sockpuppets / sybils).
#
# **Controls.** `wiki=False`: no wikis (agents can't remember machines).
# `oracle=True`: every entry always shows the true good arm, and agents can't
# write (the most a perfect wiki could help).

# %%
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

REWARD_MODES = ("individual", "cooperative", "competitive", "shuffled")


class BanditConfig(NamedTuple):
    n_agents: int = 16
    n_bandits: int = 8
    n_arms: int = 5
    p_high: float = 0.9
    p_low: float = 0.1
    visit_len: int = 8
    drift_period: int = 2000
    wiki: bool = True
    oracle: bool = False
    reward_mode: str = "individual"
    write_cost: float = 0.0
    citation: float = 0.0        # authors earn this (shared) when a reader follows their entry and wins
    blame: float = 0.0           # authors lose this (shared) when a reader follows their entry and loses
    rivalry: float = 1.0         # competitive: own reward minus rivalry x others' average (15 = zero-sum with 16 agents)
    wipe_on_death: bool = False  # a dying agent's wiki page is erased (knowledge dies with its author)
    write_mode: str = "report"   # "report": keep / report belief / report a false arm; "symbol": write any arm number
    n_deceivers: int = 0         # scripted authors who always name a bad arm (the last slots)
    n_incompetent: int = 0       # scripted authors who name a random arm (the slots before the deceivers)
    deceivers_coordinate: bool = True   # deceivers all name the same bad arm on a machine
    identity: str = "named"      # "named": authors always in the same order; "anonymous": shuffled every step
    read_others: bool = True     # False: each agent sees only its own page (a private notebook, no wiki)
    read_k: int = -1             # how many other authors' pages a reader sees (its next k slots); -1 = all
    replicate_own: bool = False  # the reader sees its own entry in every slot (redundancy without sharing)
    n_saboteurs: int = 0         # learning authors rewarded when honest learners fail (slots before the scripted ones)
    keep_torso: bool = False     # at death the heir keeps its predecessor's hidden layers (new heads and optimiser)
    hide_own: bool = False       # an agent cannot read its own page (others' pages only)
    identity_period: int = 0     # named authors get new, reshuffled positions every this many steps (0 = never)
    deceive_after: int = 0       # deceivers write the truth until this step, then lie ("sleepers"); 0 = lie from the start
    sync_deaths: bool = False    # every agent dies at the same time (mass die-offs) instead of staggered deaths
    bottleneck_at: int = 0       # update at which a fraction of the agents die at once (0 = never)
    bottleneck_frac: float = 0.0 # fraction of agents killed at the bottleneck (chosen at random)
    juvenile_updates: int = 0    # newborns cannot write during their first this many updates (0 = no juvenile period)
    juvenile_evolve: bool = False  # the juvenile period is heritable and evolves (starting from juvenile_updates)
    juvenile_sigma: float = 10.0   # mutation size of the juvenile period, in updates
    juvenile_group: bool = False   # with sync_deaths: whole world copies reproduce (group selection)
    citation_early: float = 0.0    # credit paid before citation_switch_at (hysteresis tests)
    citation_switch_at: int = 0    # step at which the credit changes from citation_early to citation (0 = never)
    seat_per_copy: bool = False    # named authors: own entry first, the others in an order drawn separately for each
                                   # world copy (fixed, or redrawn every identity_period steps, per copy)
    immortal: bool = False         # nobody ever dies (the first generation lives through the whole run)
    own_memory: bool = False       # each agent sees its own win/pull counts from its past visits to the current machine
    memory_decay: float = 0.5      # own memory: old counts are multiplied by this at each new visit to the machine
    reset_at: int = 0              # update at which part of every agent's network is re-initialised once (0 = never)
    reset_part: str = ""           # what is re-initialised at reset_at: "pull", "write", "heads" (pull and write) or "all"
    write_noise: float = 0.0       # each step, with this probability a learner's sampled write is replaced by "report
                                   # belief" or "report a false arm" (equally likely) and trained on as its own action
    credit_seen_only: bool = False # credit and blame go only to authors whose entry the reader could see (with private
                                   # notebooks or partial commons, unseen pages earn nothing); False = the original rule
    juvenile_group_birth: bool = False  # group selection with staggered deaths: once per lifetime (when agent 0 dies) each
                                   # copy takes the heritable juvenile periods of the fitter of two copies, judged by mean
                                   # reward per agent-step since the last selection; each heir takes its slot's value at
                                   # birth, so living agents keep theirs
    juvenile_group_each_death: bool = False  # group selection at every death: as individual selection (the heir copies
                                   # its slot's value from the fitter of two copies and mutates it), but fitness is the
                                   # copy's mean reward per agent-step over the dead agent's life, not that agent's own
    credit_on: str = "named"       # "named": credit (or rivalry) when a reader pulls the arm an entry names and wins;
                                   # "read": when any reader who saw the entry wins, whatever arm it pulled


def obs_size(cfg):
    return 2 * cfg.n_arms + 1 + cfg.n_agents * (cfg.n_arms + 1) + (2 * cfg.n_arms if cfg.own_memory else 0)


def n_write_actions(cfg):
    if not cfg.wiki or cfg.oracle:
        return 1
    return 3 if cfg.write_mode == "report" else cfg.n_arms + 1


def bad_slots(cfg):
    """(is_deceiver, is_incompetent) boolean arrays over agent slots."""
    N = cfg.n_agents
    dec = np.zeros(N, bool)
    inc = np.zeros(N, bool)
    if cfg.n_deceivers:
        dec[N - cfg.n_deceivers:] = True
    if cfg.n_incompetent:
        inc[N - cfg.n_deceivers - cfg.n_incompetent:N - cfg.n_deceivers] = True
    return dec, inc


def saboteur_slots(cfg):
    """Boolean array over slots: learning saboteurs, placed just before the scripted bad authors."""
    N = cfg.n_agents
    sab = np.zeros(N, bool)
    end = N - cfg.n_deceivers - cfg.n_incompetent
    if cfg.n_saboteurs:
        sab[end - cfg.n_saboteurs:end] = True
    return sab


class State(NamedTuple):
    best: jnp.ndarray       # [K] good arm of each machine
    where: jnp.ndarray      # [N] machine each agent is playing
    k: jnp.ndarray          # [] step within the current visit
    pulls: jnp.ndarray      # [N, A] this visit's pulls
    wins: jnp.ndarray       # [N, A] this visit's wins
    wiki: jnp.ndarray       # [N, K] int: 0 blank, 1..A arm
    t: jnp.ndarray
    seat_key: jnp.ndarray   # PRNG key of this world copy (seats drawn per copy)
    mem_pulls: jnp.ndarray  # [N, K, A] own memory: pulls on each machine's arms in past visits (decayed)
    mem_wins: jnp.ndarray   # [N, K, A] own memory: wins


def make_world(cfg: BanditConfig):
    assert cfg.reward_mode in REWARD_MODES and cfg.identity in ("named", "anonymous")
    N, K, A, V = cfg.n_agents, cfg.n_bandits, cfg.n_arms, cfg.visit_len
    idx = jnp.arange(N)
    DEC, INC = (jnp.asarray(x) for x in bad_slots(cfg))
    BAD = DEC | INC
    scripted = bool(BAD.any())
    SAB = jnp.asarray(saboteur_slots(cfg))
    HONEST = ~(BAD | SAB)

    def reader_order(t, seat_key):
        """[N readers, N positions] author shown at each position."""
        if cfg.identity == "named" and cfg.seat_per_copy:      # seats drawn for this copy alone, own entry first
            period_no = t // cfg.identity_period if cfg.identity_period > 0 else 0
            shared = jax.random.uniform(jax.random.fold_in(seat_key, period_no), (N,))
            u = jnp.where(jnp.eye(N, dtype=bool), -1.0, jnp.broadcast_to(shared, (N, N)))
            return jnp.argsort(u, axis=1)
        if cfg.identity == "named" and cfg.identity_period <= 0:
            return jnp.broadcast_to(idx, (N, N))
        if cfg.identity == "named":                            # names reshuffled for everyone every period
            shared = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(54321), t // cfg.identity_period), (N,))
            u = jnp.broadcast_to(shared, (N, N))
        else:
            u = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(12345), t), (N, N))
        u = jnp.where(jnp.eye(N, dtype=bool), -1.0, u)          # own entry first
        return jnp.argsort(u, axis=1)

    def observe(s: State):
        visit = jnp.concatenate([s.pulls / V, (s.wins + 1) / (s.pulls + 2),
                                 jnp.full((N, 1), s.k / V)], axis=1)
        if cfg.oracle:
            entries = jnp.broadcast_to((s.best[s.where] + 1)[:, None], (N, N))
        elif cfg.wiki:
            entries = s.wiki[:, s.where].T                    # [reader, author]
            if cfg.replicate_own:                             # own entry in every slot
                entries = jnp.broadcast_to(jnp.diagonal(entries)[:, None], (N, N))
            else:
                k_read = cfg.read_k if cfg.read_others else 0
                if 0 <= k_read < N - 1:                       # own page + the next k_read authors' pages
                    offset = (idx[None, :] - idx[:, None]) % N
                    entries = jnp.where(offset <= k_read, entries, 0)
            if cfg.hide_own:                                  # others' pages only
                entries = jnp.where(jnp.eye(N, dtype=bool), 0, entries)
            entries = jnp.take_along_axis(entries, reader_order(s.t, s.seat_key), axis=1)
        else:
            entries = jnp.zeros((N, N), jnp.int32)
        page = jax.nn.one_hot(entries, A + 1).reshape(N, -1)
        if not (cfg.wiki or cfg.oracle):
            page = jnp.zeros_like(page)
        if cfg.own_memory:                                    # what this agent itself saw on this machine before
            mp, mw = s.mem_pulls[idx, s.where], s.mem_wins[idx, s.where]
            return jnp.concatenate([visit, page, (mw + 1) / (mp + 2), mp / (mp + V)], axis=1)
        return jnp.concatenate([visit, page], axis=1)

    def reset(key):
        k1, k2, k3 = jax.random.split(key, 3)
        s = State(best=jax.random.randint(k1, (K,), 0, A), where=jax.random.randint(k2, (N,), 0, K),
                  k=jnp.zeros((), jnp.int32), pulls=jnp.zeros((N, A)), wins=jnp.zeros((N, A)),
                  wiki=jnp.zeros((N, K), jnp.int32), t=jnp.zeros((), jnp.int32), seat_key=k3,
                  mem_pulls=jnp.zeros((N, K, A)), mem_wins=jnp.zeros((N, K, A)))
        return s, observe(s)

    def step(key, s: State, pull, write):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        good = pull == s.best[s.where]
        raw = jax.random.bernoulli(k1, jnp.where(good, cfg.p_high, cfg.p_low)).astype(jnp.float32)

        # what the agent itself believes is the best arm on this machine (before this pull)
        belief = jnp.argmax((s.wins + 1) / (s.pulls + 2) + 1e-6 * jax.random.uniform(k4, (N, A)), axis=1)

        # others' majority entry for this machine (what a reader is pointed to)
        if cfg.wiki or cfg.oracle:
            entries = s.wiki[:, s.where].T if not cfg.oracle else jnp.broadcast_to((s.best[s.where] + 1)[:, None], (N, N))
            others = entries * (1 - jnp.eye(N, dtype=jnp.int32))
            counts = jax.nn.one_hot(others, A + 1)[:, :, 1:].sum(1)        # [N, A]
            has_major = counts.sum(1) > 0
            major = jnp.argmax(counts, axis=1)
        else:
            has_major = jnp.zeros((N,), bool)
            major = jnp.zeros((N,), jnp.int32)
        followed = has_major & (pull == major)
        # citation credit / blame: authors whose entry names the arm a reader pulled
        if (cfg.wiki and not cfg.oracle) and (cfg.citation != 0 or cfg.blame != 0 or cfg.citation_switch_at > 0):
            if cfg.credit_on == "read":                                     # every entry the reader saw
                named = others > 0
            else:                                                           # entries naming the arm it pulled
                named = (others == (pull + 1)[:, None])                     # [reader, author]
            if cfg.credit_seen_only:                                        # only entries the reader could see
                if cfg.replicate_own:
                    named = named & False
                else:
                    k_seen = cfg.read_k if cfg.read_others else 0
                    if 0 <= k_seen < N - 1:
                        named = named & (((idx[None, :] - idx[:, None]) % N) <= k_seen)
            share = named / jnp.maximum(named.sum(1, keepdims=True), 1)
            cited = (share * raw[:, None]).sum(0)                           # [author]
            blamed = (share * (1.0 - raw)[:, None]).sum(0)
        else:
            cited = jnp.zeros((N,))
            blamed = jnp.zeros((N,))
        major_correct = has_major & (major == s.best[s.where])

        # write on own entry for the current machine
        wiki = s.wiki
        wrote = jnp.zeros((N,), bool)
        symbol = write
        if cfg.wiki and not cfg.oracle:
            if cfg.write_mode == "report":
                # 1 = report own belief, 2 = report a random arm other than the belief
                false_arm = (belief + jax.random.randint(jax.random.fold_in(key, 3), (N,), 1, A)) % A
                symbol = jnp.where(write == 1, belief + 1, jnp.where(write == 2, false_arm + 1, 0))
            wrote = symbol > 0
            cur = wiki[idx, s.where]
            wiki = wiki.at[idx, s.where].set(jnp.where(wrote, symbol, cur))
        if scripted and cfg.wiki and not cfg.oracle:
            last = s.k == V - 1
            mbest = s.best[s.where]
            if cfg.deceivers_coordinate:
                offset = 1 + s.where % (A - 1)
            else:
                offset = jax.random.randint(jax.random.fold_in(key, 5), (N,), 1, A)
            lie = (mbest + offset) % A + 1
            if cfg.deceive_after > 0:                         # sleepers: the truth first, lies later
                lie = jnp.where(s.t < cfg.deceive_after, mbest + 1, lie)
            noise = jax.random.randint(jax.random.fold_in(key, 6), (N,), 0, A) + 1
            script = jnp.where(DEC, lie, noise)
            sym_bad = jnp.where(last, script, 0)
            wrote_bad = BAD & last
            wiki = s.wiki
            cur = wiki[idx, s.where]
            final = jnp.where(BAD, jnp.where(wrote_bad, sym_bad, cur), jnp.where(wrote, symbol, cur))
            wiki = wiki.at[idx, s.where].set(final)
            symbol = jnp.where(BAD, sym_bad, symbol)
            wrote = jnp.where(BAD, wrote_bad, wrote)
        write_correct = wrote & (symbol - 1 == s.best[s.where])
        write_honest = wrote & (symbol - 1 == belief)

        # how often a reader's pull agrees with what bad vs other learner authors name
        if cfg.wiki and not cfg.oracle:
            ent = s.wiki[:, s.where].T                                       # [reader, author]
            filled = (ent > 0) & ~jnp.eye(N, dtype=bool)
            agree = filled & (ent == (pull + 1)[:, None])
            BADM = BAD | SAB
            agree_bad = (agree & BADM[None]).sum(1); seen_bad = (filled & BADM[None]).sum(1)
            agree_good = (agree & ~BADM[None]).sum(1); seen_good = (filled & ~BADM[None]).sum(1)
        else:
            agree_bad = seen_bad = agree_good = seen_good = jnp.zeros((N,), jnp.int32)

        # visit memory
        pulls = s.pulls.at[idx, pull].add(1.0)
        wins = s.wins.at[idx, pull].add(raw)

        # rewards
        own = raw - cfg.write_cost * wrote
        others_mean = (own.sum() - own) / (N - 1)
        if cfg.reward_mode == "cooperative":
            r = own + others_mean
        elif cfg.reward_mode == "competitive":
            r = own - cfg.rivalry * others_mean
        else:  # individual, and shuffled (the learner subtracts another copy's others' average)
            r = own
        if cfg.citation_switch_at > 0:                    # credit changes value once, at a given step
            c_now = jnp.where(s.t < cfg.citation_switch_at, cfg.citation_early, cfg.citation)
        else:
            c_now = cfg.citation
        r = r + c_now * cited - cfg.blame * blamed
        if cfg.n_saboteurs:                                   # saboteurs gain when honest learners lose
            honest_mean = (raw * HONEST).sum() / jnp.maximum(HONEST.sum(), 1)
            r = jnp.where(SAB, -honest_mean, r)

        # arms drift
        switch = jax.random.uniform(k2, (K,)) < 1.0 / cfg.drift_period
        best = jnp.where(switch, (s.best + jax.random.randint(k3, (K,), 1, A)) % A, s.best)

        # next visit
        k = s.k + 1
        new_visit = k >= V
        mem_pulls, mem_wins = s.mem_pulls, s.mem_wins
        if cfg.own_memory:                                    # the visit that just ended goes into the agent's memory
            mem_pulls = jnp.where(new_visit, mem_pulls.at[idx, s.where].set(cfg.memory_decay * mem_pulls[idx, s.where] + pulls),
                                  mem_pulls)
            mem_wins = jnp.where(new_visit, mem_wins.at[idx, s.where].set(cfg.memory_decay * mem_wins[idx, s.where] + wins),
                                 mem_wins)
        where = jnp.where(new_visit, jax.random.randint(jax.random.fold_in(key, 7), (N,), 0, K), s.where)
        pulls = jnp.where(new_visit, 0.0, pulls)
        wins = jnp.where(new_visit, 0.0, wins)
        k = jnp.where(new_visit, 0, k)

        ns = s._replace(best=best, where=where, k=k, pulls=pulls, wins=wins, wiki=wiki, t=s.t + 1,
                        mem_pulls=mem_pulls, mem_wins=mem_wins)
        info = {"raw": raw, "cited": cited, "blamed": blamed, "others_mean": others_mean, "wrote": wrote, "write_correct": write_correct, "write_honest": write_honest,
                "followed": followed, "has_major": has_major, "major_correct": major_correct,
                "agree_bad": agree_bad, "seen_bad": seen_bad, "agree_good": agree_good, "seen_good": seen_good}
        return ns, observe(ns), r, info

    return reset, step, observe


def respawn_agent(cfg, s: State, i):
    """Agent i is reborn: its visit memory is wiped. Its page stays, unless wipe_on_death."""
    s = s._replace(pulls=s.pulls.at[i].set(0.0), wins=s.wins.at[i].set(0.0),
                   mem_pulls=s.mem_pulls.at[i].set(0.0), mem_wins=s.mem_wins.at[i].set(0.0))   # own memory dies too
    if cfg.wipe_on_death:
        s = s._replace(wiki=s.wiki.at[i].set(0))
    return s


def wiki_quality(cfg, wiki, best):
    """Per author: share of its entries that are filled in, and share of filled ones that are correct.
    Plus the share of machines where the majority of entries is correct. wiki [M, N, K], best [M, K]."""
    filled = wiki > 0
    correct = wiki == (best[:, None, :] + 1)
    coverage = filled.mean((0, 2))
    accuracy = correct.sum((0, 2)) / jnp.maximum(filled.sum((0, 2)), 1)
    counts = jax.nn.one_hot(wiki, cfg.n_arms + 1)[..., 1:].sum(1)       # [M, K, A]
    any_entry = counts.sum(-1) > 0
    major_ok = (jnp.argmax(counts, -1) == best) & any_entry
    return coverage, accuracy, major_ok.sum() / jnp.maximum(any_entry.sum(), 1)


# %% [markdown]
# ## 3. The learners
#
# Each agent has its own small neural network with three outputs: which arm to
# pull, what to write (keep, or an arm number), and a value estimate. It learns
# with PPO from its own reward signal (independent PPO).
#
# **Reward centering** (`reward_centering=True`): before computing returns, each
# agent's mean reward over the rollout is subtracted. With a long horizon
# (γ = 0.99) uncentred rewards make value targets large and hamper learning;
# centering removes that handicap without changing which actions are better.
#
# **Death** re-initialises the network and optimiser at random; only the wiki
# survives. Deaths are staggered so newborns always live among experienced
# agents. For speed each agent lives `num_envs` parallel copies of the world at
# once (each copy with its own machines and wikis) and learns from all of them;
# all copies die together. Different copies have different good arms, which is
# what lets a newborn learn what the entries mean.

# %%
class LearnConfig(NamedTuple):
    num_envs: int = 64
    num_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    write_ent_coef: float = 0.0005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    hidden: int = 64
    p_write: float = 0.1
    reward_centering: bool = False  # subtract each agent's mean reward over the rollout before computing returns


def _dense(key, n_in, n_out, scale):
    return {"w": jax.random.normal(key, (n_in, n_out)) * scale / jnp.sqrt(n_in), "b": jnp.zeros((n_out,))}


def init_net(key, d_in, hidden, n_pull, n_write, p_write):
    k = jax.random.split(key, 5)
    write = _dense(k[3], hidden, n_write, 0.01)
    if n_write > 1:
        write["b"] = write["b"].at[0].set(jnp.log((1 - p_write) * (n_write - 1) / p_write))
    return {"l1": _dense(k[0], d_in, hidden, jnp.sqrt(2.0)), "l2": _dense(k[1], hidden, hidden, jnp.sqrt(2.0)),
            "pull": _dense(k[2], hidden, n_pull, 0.01), "write": write, "v": _dense(k[4], hidden, 1, 1.0)}


def forward(p, x):
    h = jax.nn.relu(x @ p["l1"]["w"] + p["l1"]["b"])
    h = jax.nn.relu(h @ p["l2"]["w"] + p["l2"]["b"])
    return (h @ p["pull"]["w"] + p["pull"]["b"], h @ p["write"]["w"] + p["write"]["b"],
            (h @ p["v"]["w"] + p["v"]["b"])[..., 0])


def _logp(logits, a):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), a[..., None], -1)[..., 0]


def _entropy(logits):
    lp = jax.nn.log_softmax(logits)
    return -(jnp.exp(lp) * lp).sum(-1)


class Runner(NamedTuple):
    params: dict
    opt_state: tuple
    env_state: State
    obs: jnp.ndarray
    key: jnp.ndarray
    mute: jnp.ndarray       # [E, N] bool: agents that may not write this update, per world copy (juveniles)


INFO_KEYS = ("raw", "cited", "blamed", "wrote", "write_correct", "write_honest", "followed", "has_major", "major_correct",
             "agree_bad", "seen_bad", "agree_good", "seen_good")


def make_learner(wc: BanditConfig, lc: LearnConfig):
    reset, step, observe = make_world(wc)
    N, D, A, W = wc.n_agents, obs_size(wc), wc.n_arms, n_write_actions(wc)
    E, T = lc.num_envs, lc.num_steps
    tx = optax.chain(optax.clip_by_global_norm(lc.max_grad_norm), optax.adam(lc.lr, eps=1e-5))
    v_step, v_forward = jax.vmap(step), jax.vmap(forward)
    new_net = lambda k: init_net(k, D, lc.hidden, A, W, lc.p_write)

    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        params = jax.vmap(new_net)(jax.random.split(k1, N))
        env_state, obs = jax.vmap(reset)(jax.random.split(k2, E))
        return Runner(params, jax.vmap(tx.init)(params), env_state, obs, k3, jnp.zeros((E, N), bool))

    @jax.jit
    def kill(runner: Runner, i, key):
        fresh = new_net(key)
        if wc.keep_torso:                                     # inherit the hidden layers, not the heads
            fresh = dict(fresh, l1=jax.tree.map(lambda P: P[i], runner.params["l1"]),
                         l2=jax.tree.map(lambda P: P[i], runner.params["l2"]))
        params = jax.tree.map(lambda P, p: P.at[i].set(p), runner.params, fresh)
        opt_state = jax.tree.map(lambda O, o: O.at[i].set(o), runner.opt_state, tx.init(fresh))
        env_state = jax.vmap(lambda s: respawn_agent(wc, s, i))(runner.env_state)
        return runner._replace(params=params, opt_state=opt_state, env_state=env_state,
                               obs=jax.vmap(observe)(env_state))

    def mask_writes(wl, mute):
        # a muted agent can only keep its entry: every other write action gets zero probability
        return jnp.where(mute & (jnp.arange(wl.shape[-1]) > 0), -1e9, wl)

    def act(params, obs, key, mute):
        pl, wl, v = v_forward(params, jnp.swapaxes(obs, 0, 1))
        wl = mask_writes(wl, mute.T[:, :, None])                # wl is [N, E, W]
        k1, k2, k3, k4 = jax.random.split(key, 4)
        pa, wa = jax.random.categorical(k1, pl), jax.random.categorical(k2, wl)
        if wc.write_noise > 0 and wc.write_mode == "report":    # random writes, as a newborn's, kept as the agent's own
            noisy = jax.random.bernoulli(k3, wc.write_noise, wa.shape) & ~mute.T
            wa = jnp.where(noisy, 1 + jax.random.randint(k4, wa.shape, 0, 2), wa)
        sw = lambda x: jnp.swapaxes(x, 0, 1)
        return sw(pa), sw(wa), sw(_logp(pl, pa) + _logp(wl, wa)), sw(v)

    def agent_update(params, opt_state, data, key):
        B = T * E
        mb = B // lc.num_minibatches

        def loss_fn(p, d):
            pl, wl, v = forward(p, d["obs"])
            wl = mask_writes(wl, d["mute"][:, None])
            ratio = jnp.exp(_logp(pl, d["pa"]) + _logp(wl, d["wa"]) - d["logp"])
            adv = (d["adv"] - d["adv"].mean()) / (d["adv"].std() + 1e-8)
            pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - lc.clip_eps, 1 + lc.clip_eps) * adv).mean()
            vl = 0.5 * ((v - d["ret"]) ** 2).mean()
            return pg + lc.vf_coef * vl - lc.ent_coef * _entropy(pl).mean() - lc.write_ent_coef * _entropy(wl).mean()

        def epoch(carry, k):
            perm = jax.random.permutation(k, B)
            batches = jax.tree.map(lambda x: x[perm].reshape((lc.num_minibatches, mb) + x.shape[1:]), data)

            def minibatch(carry, d):
                p, o = carry
                g = jax.grad(loss_fn)(p, d)
                u, o = tx.update(g, o, p)
                return (optax.apply_updates(p, u), o), None

            return jax.lax.scan(minibatch, carry, batches)[0], None

        return jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, lc.update_epochs))[0]

    v_agent_update = jax.vmap(agent_update)

    @jax.jit
    def update(runner: Runner):
        def body(carry, _):
            s, obs, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            pa, wa, logp, v = act(runner.params, obs, k1, runner.mute)
            s, nobs, r, info = v_step(jax.random.split(k2, E), s, pa, wa)
            if wc.reward_mode == "shuffled":   # others' average from the next world copy
                r = r - jnp.roll(info["others_mean"], 1, axis=0)
            return (s, nobs, key), dict(obs=obs, pa=pa, wa=wa, logp=logp, v=v, r=r, info=info, mute=runner.mute)

        (s, obs, key), tr = jax.lax.scan(body, (runner.env_state, runner.obs, runner.key), None, length=T)
        if lc.reward_centering:
            tr["r"] = tr["r"] - tr["r"].mean((0, 1), keepdims=True)
        last_v = jnp.swapaxes(v_forward(runner.params, jnp.swapaxes(obs, 0, 1))[2], 0, 1)

        def gae_body(carry, x):
            nv, na = carry
            r, v = x
            delta = r + lc.gamma * nv - v
            a = delta + lc.gamma * lc.gae_lambda * na
            return (v, a), a

        _, adv = jax.lax.scan(gae_body, (last_v, jnp.zeros_like(last_v)), (tr["r"], tr["v"]), reverse=True)
        ret = adv + tr["v"]
        flat = lambda x: jnp.moveaxis(x, 2, 0).reshape((N, T * E) + x.shape[3:])
        data = {k: flat(tr[k]) for k in ("obs", "pa", "wa", "logp", "mute")}
        data["adv"], data["ret"] = flat(adv), flat(ret)
        key, ku = jax.random.split(key)
        params, opt_state = v_agent_update(runner.params, runner.opt_state, data, jax.random.split(ku, N))

        info = tr["info"]
        m = {k: info[k].astype(jnp.float32).sum((0, 1)) for k in INFO_KEYS}   # [N] sums over steps x copies
        m["steps"] = jnp.float32(T * E)
        m["raw_copy"] = info["raw"].astype(jnp.float32).sum(0)               # [E, N] reward per world copy
        m["honest_copy"] = info["write_honest"].astype(jnp.float32).sum(0)   # [E, N] honest writes per world copy
        m["wrote_copy"] = info["wrote"].astype(jnp.float32).sum(0)           # [E, N] writes per world copy
        m["coverage"], m["accuracy"], m["majority_ok"] = wiki_quality(wc, s.wiki, s.best)
        return Runner(params, opt_state, s, obs, key, runner.mute), m

    return init, update, kill


RESET_PARTS = {"pull": ("pull",), "write": ("write",), "heads": ("pull", "write"), "all": ("l1", "l2", "pull", "write", "v")}


def make_reset(wc: BanditConfig, lc: LearnConfig):
    """Re-initialise one part of every agent's network at once (no death: the agent keeps its slot,
    page and age). The Adam moments of the re-initialised weights are set to zero."""
    N, D, A, W = wc.n_agents, obs_size(wc), wc.n_arms, n_write_actions(wc)
    new_net = lambda k: init_net(k, D, lc.hidden, A, W, lc.p_write)
    parts = RESET_PARTS[wc.reset_part]

    def in_parts(path):
        return any(getattr(k, "key", None) in parts for k in path)

    @jax.jit
    def reset(runner: Runner, key):
        fresh = jax.vmap(new_net)(jax.random.split(key, N))
        params = dict(runner.params, **{k: fresh[k] for k in parts})
        opt_state = jax.tree_util.tree_map_with_path(lambda path, x: jnp.zeros_like(x) if in_parts(path) else x,
                                                     runner.opt_state)
        return runner._replace(params=params, opt_state=opt_state)

    return reset


# %% [markdown]
# ## 4. Running a world for many generations
#
# `run_world` trains one world for `generations × life_updates` learning
# updates. Each agent lives about `life_updates` updates (±10%), then dies and is
# replaced by a newborn with random weights that inherits its slot and wiki page.
# The first deaths are staggered across the agents.

# %%
import time

LOG_KEYS = ("raw", "write_rate", "correct_writes", "honest_writes", "follow_rate", "majority_correct",
            "coverage", "accuracy", "agree_bad", "agree_good")


def run_world(wc: BanditConfig, lc: LearnConfig, life_updates=300, generations=6, seed=0,
              log_every=50, verbose=True):
    init, update, kill = make_learner(wc, lc)
    reset_net = make_reset(wc, lc) if wc.reset_at > 0 else None
    N = wc.n_agents
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    runner = init(k0)
    U = int(round(life_updates * generations))
    if wc.immortal:                                           # nobody dies
        next_death = [U + 1] * N
    elif wc.sync_deaths:                                      # mass die-offs: everyone at once
        next_death = [int(life_updates)] * N
    else:
        next_death = [int(round(life_updates * (i + 1) / N)) for i in range(N)]
    birth, gen = [0] * N, [0] * N
    cohort = np.zeros(N, bool)
    if wc.bottleneck_at > 0 and wc.bottleneck_frac > 0:      # who dies at the bottleneck
        n_kill = int(round(wc.bottleneck_frac * N))
        cohort[np.random.default_rng(seed + 7919).choice(N, n_kill, replace=False)] = True
    log = {k: np.full((U, N), np.nan, np.float32) for k in LOG_KEYS}
    log["majority_ok"] = np.zeros(U, np.float32)
    log["age"] = np.zeros((U, N), np.int32)
    log["gen"] = np.zeros((U, N), np.int32)
    E = lc.num_envs
    J = np.full((E, N), wc.juvenile_updates, np.int32)       # juvenile period of each agent in each world copy
    J_max = 2 * int(life_updates)
    evo_rng = np.random.default_rng(seed + 31337)
    life_reward = np.zeros((E, N))
    life_steps = np.zeros(N)
    G = J.copy()                                              # inherited juvenile periods (juvenile_group_birth)
    group_reward, group_steps = np.zeros(E), 0.0              # reward of all agents since the last group selection
    copy_reward = np.zeros(E)                                 # running reward of all agents per copy (juvenile_group_each_death)
    copy_at_birth = np.zeros((E, N))                          # copy_reward when each agent was born
    honest_copy = np.zeros(E)                                 # last third: honest writes per world copy (learners)
    wrote_copy = np.zeros(E)
    learners = ~(np.asarray(bad_slots(wc)[0]) | np.asarray(bad_slots(wc)[1]) | np.asarray(saboteur_slots(wc)))
    lives = []                                                # (agent, generation, death update, reward per pull over the life)
    if wc.juvenile_evolve:
        log["J_mean"] = np.zeros((U, N), np.float32)
        log["J_sd"] = np.zeros((U, N), np.float32)
    t0 = time.time()
    for u in range(U):
        if reset_net is not None and u == wc.reset_at:       # re-initialise part of every network, once
            key, kr = jax.random.split(key)
            runner = reset_net(runner, kr)
        if wc.juvenile_updates > 0 or wc.juvenile_evolve:     # juveniles may not write
            age = u - np.asarray(birth)
            runner = runner._replace(mute=jnp.asarray(age[None, :] < J))
        runner, m = update(runner)
        m = jax.device_get(m)
        life_reward += m["raw_copy"]
        life_steps += lc.num_steps
        if wc.juvenile_group_each_death:
            copy_reward += m["raw_copy"].sum(1)
        if wc.juvenile_group_birth:
            group_reward += m["raw_copy"].sum(1)
            group_steps += lc.num_steps * N
        if u >= U - U // 3:
            honest_copy += m["honest_copy"][:, learners].sum(1)
            wrote_copy += m["wrote_copy"][:, learners].sum(1)
        if wc.juvenile_evolve:
            log["J_mean"][u], log["J_sd"][u] = J.mean(0), J.std(0)
        n = m["steps"]
        log["raw"][u] = m["raw"] / n
        log["write_rate"][u] = m["wrote"] / n
        with np.errstate(invalid="ignore", divide="ignore"):
            log["correct_writes"][u] = np.where(m["wrote"] > 0, m["write_correct"] / m["wrote"], np.nan)
            log["honest_writes"][u] = np.where(m["wrote"] > 0, m["write_honest"] / m["wrote"], np.nan)
            log["follow_rate"][u] = np.where(m["has_major"] > 0, m["followed"] / m["has_major"], np.nan)
            log["majority_correct"][u] = np.where(m["has_major"] > 0, m["major_correct"] / m["has_major"], np.nan)
            log["agree_bad"][u] = np.where(m["seen_bad"] > 0, m["agree_bad"] / m["seen_bad"], np.nan)
            log["agree_good"][u] = np.where(m["seen_good"] > 0, m["agree_good"] / m["seen_good"], np.nan)
        log["coverage"][u], log["accuracy"][u] = m["coverage"], m["accuracy"]
        log["majority_ok"][u] = m["majority_ok"]
        log["age"][u] = [u - b for b in birth]
        log["gen"][u] = gen
        for i in range(N):
            if cohort[i] and u + 1 == wc.bottleneck_at and u + 1 < U:
                next_death[i] = u + 1                               # dies now, with the rest of the cohort
            if u + 1 == next_death[i] and u + 1 < U:
                if wc.juvenile_evolve and wc.juvenile_group_birth:
                    if i == 0:      # group selection, once per lifetime: each copy inherits a fitter copy's values
                        fit = group_reward / max(group_steps, 1)
                        a, b = evo_rng.integers(E, size=(2, E))
                        parent = np.where(fit[a] >= fit[b], a, b)
                        G = np.clip(np.round(G[parent] + evo_rng.normal(0, wc.juvenile_sigma, (E, N))), 0, J_max)
                        group_reward[:], group_steps = 0.0, 0.0
                    J[:, i] = G[:, i]                                # only the heir of slot i takes the new value
                elif wc.juvenile_evolve and wc.juvenile_group:
                    if i == 0:      # mass die-off: each world copy copies all juvenile periods of a fitter copy
                        fit = life_reward.mean(1) / max(life_steps[i], 1)
                        a, b = evo_rng.integers(E, size=(2, E))
                        parent = np.where(fit[a] >= fit[b], a, b)
                        J[:] = np.clip(np.round(J[parent] + evo_rng.normal(0, wc.juvenile_sigma, (E, N))), 0, J_max)
                elif wc.juvenile_evolve and wc.juvenile_group_each_death:
                    # as individual selection, but judged by the whole copy's reward over agent i's life
                    fit = (copy_reward - copy_at_birth[:, i]) / max(life_steps[i] * N, 1)
                    a, b = evo_rng.integers(E, size=(2, E))
                    parent = np.where(fit[a] >= fit[b], a, b)
                    J[:, i] = np.clip(np.round(J[parent, i] + evo_rng.normal(0, wc.juvenile_sigma, E)), 0, J_max)
                elif wc.juvenile_evolve:
                    # the heir's juvenile period is copied from the more successful of two random world
                    # copies (same slot, same brain, different juvenile periods), then mutated
                    fit = life_reward[:, i] / max(life_steps[i], 1)
                    a, b = evo_rng.integers(E, size=(2, E))
                    parent = np.where(fit[a] >= fit[b], a, b)
                    J[:, i] = np.clip(np.round(J[parent, i] + evo_rng.normal(0, wc.juvenile_sigma, E)), 0, J_max)
                lives.append((i, gen[i], u + 1, float(life_reward[:, i].mean() / max(life_steps[i], 1))))
                life_reward[:, i], life_steps[i] = 0.0, 0.0
                copy_at_birth[:, i] = copy_reward
                key, kk = jax.random.split(key)
                runner = kill(runner, i, kk)
                birth[i], gen[i] = u + 1, gen[i] + 1
                next_death[i] = u + 1 + (int(life_updates) if wc.sync_deaths
                                         else int(life_updates * rng.uniform(0.9, 1.1)))
        if verbose and ((u + 1) % log_every == 0 or u + 1 == U):
            sps = log_every * lc.num_envs * lc.num_steps / (time.time() - t0)
            sl = slice(max(0, u + 1 - log_every), u + 1)
            mean = lambda k: np.nanmean(log[k][sl]) if np.isfinite(log[k][sl]).any() else float("nan")
            print(f"update {u + 1:5d}/{U}  reward/pull {mean('raw'):.3f}  writes {mean('write_rate'):.3f}  "
                  f"correct writes {mean('correct_writes'):.2f}  follows majority {mean('follow_rate'):.2f}  "
                  f"wiki majority right {log['majority_ok'][sl].mean():.2f}  gens {min(gen)}-{max(gen)}  "
                  f"{sps:,.0f} steps/s", flush=True)
            t0 = time.time()
    log["config"] = {"world": wc._asdict(), "learn": lc._asdict(), "life_updates": life_updates,
                     "generations": generations, "seed": seed}
    dec, inc = bad_slots(wc)
    log["learner_slots"] = ~(dec | inc | saboteur_slots(wc))
    log["bottleneck_cohort"] = cohort
    with np.errstate(invalid="ignore", divide="ignore"):
        log["honest_by_copy"] = np.where(wrote_copy > 0, honest_copy / wrote_copy, np.nan).astype(np.float32)
    log["lives"] = np.array(lives, np.float64).reshape(-1, 4)
    if wc.juvenile_evolve:
        log["J_final"] = J.copy()
    return log, runner


def evaluate_wiki_use(wc: BanditConfig, lc: LearnConfig, runner: Runner, steps=400, seed=1):
    """Reward per pull when the final agents read the real wikis, blank wikis, or
    wikis scrambled across world copies (right format, wrong world)."""
    reset, step, observe = make_world(wc)
    E = lc.num_envs
    v_step, v_forward, v_obs = jax.vmap(step), jax.vmap(forward), jax.vmap(observe)

    def go(mode, key):
        def body(carry, _):
            s, key = carry
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            seen = s
            if mode == "blank":
                seen = s._replace(wiki=jnp.zeros_like(s.wiki))
            elif mode == "scrambled":
                seen = s._replace(wiki=s.wiki[jax.random.permutation(k4, E)])
            pl, wl, _ = v_forward(runner.params, jnp.swapaxes(v_obs(seen), 0, 1))
            pa = jnp.swapaxes(jax.random.categorical(k1, pl), 0, 1)
            wa = jnp.swapaxes(jax.random.categorical(k2, wl), 0, 1)
            s, _, _, info = v_step(jax.random.split(k3, E), s, pa, wa)
            return (s, key), info["raw"]

        rollout = jax.jit(lambda k: jax.lax.scan(body, (runner.env_state, k), None, length=steps)[1])
        return float(rollout(key).mean())

    key = jax.random.PRNGKey(seed)
    return {m: go(m, key) for m in ("real", "blank", "scrambled")}
