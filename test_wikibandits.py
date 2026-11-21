import jax, jax.numpy as jnp, numpy as np
from wikibandits import *

def st(cfg, **kw):
    reset, _, _ = make_world(cfg)
    s, _ = reset(jax.random.PRNGKey(0))
    return s._replace(**kw)

def test_rewards_follow_good_arm():
    cfg = BanditConfig(n_agents=4, n_bandits=3, n_arms=3, p_high=1.0, p_low=0.0)
    _, step, _ = make_world(cfg)
    s = st(cfg, best=jnp.array([0, 1, 2]), where=jnp.array([0, 1, 2, 0]))
    _, _, r, info = step(jax.random.PRNGKey(1), s, jnp.array([0, 0, 2, 1]), jnp.zeros(4, jnp.int32))
    assert np.asarray(info["raw"]).tolist() == [1, 0, 1, 0]

def test_reward_modes():
    for mode in REWARD_MODES:
        cfg = BanditConfig(n_agents=4, n_bandits=3, n_arms=3, p_high=1.0, p_low=0.0, reward_mode=mode)
        _, step, _ = make_world(cfg)
        s = st(cfg, best=jnp.array([0, 1, 2]), where=jnp.array([0, 1, 2, 0]))
        _, _, r, _ = step(jax.random.PRNGKey(1), s, jnp.array([0, 0, 2, 1]), jnp.zeros(4, jnp.int32))
        r = np.asarray(r)
        if mode in ("individual", "shuffled"):    # shuffled subtracts in the learner, not the world
            assert r.tolist() == [1, 0, 1, 0]
        elif mode == "cooperative":
            assert np.allclose(r, [1 + 1 / 3, 2 / 3, 1 + 1 / 3, 2 / 3])
        else:
            assert abs(r.sum()) < 1e-6 and np.isclose(r[0], 1 - 1 / 3) and np.isclose(r[1], -2 / 3)

def test_writing_own_entry_and_reading_order():
    cfg = BanditConfig(n_agents=3, n_bandits=4, n_arms=3, write_mode="symbol")
    _, step, observe = make_world(cfg)
    s = st(cfg, where=jnp.array([2, 2, 1]), best=jnp.array([0, 0, 1, 0]))
    ns, obs, _, info = step(jax.random.PRNGKey(1), s, jnp.array([0, 0, 0]), jnp.array([2, 0, 3]))
    w = np.asarray(ns.wiki)
    assert w[0, 2] == 2 and w[1].sum() == 0 and w[2, 1] == 3 and w.sum() == 5
    assert np.asarray(info["write_correct"]).tolist() == [True, False, False]   # machine 2's good arm is 1 = symbol 2
    # agent 1 plays machine 2 again: it reads author 0's entry "2" in slot 0
    o = np.asarray(observe(ns._replace(where=jnp.array([2, 2, 1]))))
    page = o[:, 2 * 3 + 1:].reshape(3, 3, 4).argmax(-1)
    assert page[1].tolist() == [2, 0, 0] and page[2].tolist() == [0, 0, 3]

def test_majority_and_following():
    cfg = BanditConfig(n_agents=4, n_bandits=2, n_arms=3)
    _, step, _ = make_world(cfg)
    wiki = jnp.array([[2, 0], [2, 0], [3, 0], [0, 0]])
    s = st(cfg, wiki=wiki, where=jnp.array([0, 0, 0, 0]), best=jnp.array([1, 0]))
    _, _, _, info = step(jax.random.PRNGKey(1), s, jnp.array([1, 1, 1, 0]), jnp.zeros(4, jnp.int32))
    # every reader's majority (ties broken toward the lower arm) is arm 1 = the good arm
    assert np.asarray(info["has_major"]).all()
    assert np.asarray(info["followed"]).tolist() == [True, True, True, False]
    assert np.asarray(info["major_correct"]).tolist() == [True, True, True, True]

def test_oracle_and_nowiki():
    cfg = BanditConfig(n_agents=3, n_bandits=4, n_arms=3, oracle=True)
    _, step, observe = make_world(cfg)
    s = st(cfg, where=jnp.array([0, 1, 3]), best=jnp.array([2, 1, 0, 1]))
    page = np.asarray(observe(s))[:, 7:].reshape(3, 3, 4).argmax(-1)
    assert page.tolist() == [[3, 3, 3], [2, 2, 2], [2, 2, 2]]
    assert n_write_actions(cfg) == 1
    cfg = BanditConfig(n_agents=3, n_bandits=4, n_arms=3, wiki=False)
    _, step, observe = make_world(cfg)
    s = st(cfg)
    assert np.asarray(observe(s))[:, 7:].sum() == 0
    ns, _, _, _ = step(jax.random.PRNGKey(1), s, jnp.zeros(3, jnp.int32), jnp.zeros(3, jnp.int32))
    assert int(ns.wiki.sum()) == 0

def test_visits_reset_memory():
    cfg = BanditConfig(n_agents=2, n_bandits=50, n_arms=3, visit_len=3)
    _, step, _ = make_world(cfg)
    s = st(cfg)
    Z = jnp.zeros(2, jnp.int32)
    for _ in range(2):
        s, _, _, _ = step(jax.random.PRNGKey(_), s, Z, Z)
    assert float(s.pulls.sum()) == 4 and int(s.k) == 2
    s, _, _, _ = step(jax.random.PRNGKey(9), s, Z, Z)
    assert float(s.pulls.sum()) == 0 and int(s.k) == 0

def test_respawn_keeps_wiki():
    cfg = BanditConfig(n_agents=2, n_bandits=3, n_arms=3)
    s = st(cfg, pulls=jnp.ones((2, 3)), wins=jnp.ones((2, 3)), wiki=jnp.ones((2, 3), jnp.int32))
    ns = respawn_agent(cfg, s, 1)
    assert float(ns.pulls[1].sum()) == 0 and float(ns.pulls[0].sum()) == 3 and int(ns.wiki.sum()) == 6

def test_wiki_quality():
    cfg = BanditConfig(n_agents=2, n_bandits=2, n_arms=3)
    wiki = jnp.array([[[1, 0], [2, 3]]])          # copy 0: author0 writes arm0 on m0; author1 arm1 on m0, arm2 on m1
    best = jnp.array([[0, 2]])
    cov, acc, maj = wiki_quality(cfg, wiki, best)
    assert np.allclose(np.asarray(cov), [0.5, 1.0]) and np.allclose(np.asarray(acc), [1.0, 0.5])
    assert float(maj) == 1.0  # m0: tie between arm0 and arm1 -> argmax picks arm0 (correct); m1 correct

def test_learner_and_kill():
    wc, lc = BanditConfig(n_agents=4, n_bandits=6), LearnConfig(num_envs=8, num_steps=8)
    init, update, kill = make_learner(wc, lc)
    r = init(jax.random.PRNGKey(0))
    r, m = update(r)
    assert m["raw"].shape == (4,) and m["coverage"].shape == (4,)
    r2 = kill(r, 1, jax.random.PRNGKey(3))
    a, b = np.asarray(r.params["l1"]["w"]), np.asarray(r2.params["l1"]["w"])
    assert np.allclose(a[[0, 2, 3]], b[[0, 2, 3]]) and not np.allclose(a[1], b[1])
    assert np.array_equal(np.asarray(r.env_state.wiki), np.asarray(r2.env_state.wiki))


def test_citation_credit():
    cfg = BanditConfig(n_agents=4, n_bandits=2, n_arms=3, p_high=1.0, p_low=0.0, citation=0.5, write_mode="symbol")
    _, step, _ = make_world(cfg)
    # authors 0 and 1 name arm 1 (symbol 2) on machine 0; author 2 names arm 2
    wiki = jnp.array([[2, 0], [2, 0], [3, 0], [0, 0]])
    s = st(cfg, wiki=wiki, where=jnp.array([0, 0, 0, 0]), best=jnp.array([1, 0]))
    # agent 3 pulls arm 1 and wins: authors 0 and 1 share 0.5
    _, _, r, info = step(jax.random.PRNGKey(1), s, jnp.array([0, 0, 0, 1]), jnp.zeros(4, jnp.int32))
    c = np.asarray(info["cited"])
    assert np.allclose(c[:2], [0.5, 0.5]) and c[2] == 0 and c[3] == 0
    assert np.allclose(np.asarray(r), np.asarray(info["raw"]) + 0.5 * c)


def test_report_mode():
    cfg = BanditConfig(n_agents=3, n_bandits=2, n_arms=4, write_mode="report")
    assert n_write_actions(cfg) == 3
    _, step, _ = make_world(cfg)
    pulls = jnp.zeros((3, 4)).at[:, 2].set(3.0)
    wins = jnp.zeros((3, 4)).at[:, 2].set(3.0)          # every agent believes arm 2
    s = st(cfg, pulls=pulls, wins=wins, where=jnp.array([0, 0, 1]), best=jnp.array([2, 0]))
    ns, _, _, info = step(jax.random.PRNGKey(1), s, jnp.zeros(3, jnp.int32), jnp.array([1, 2, 0]))
    w = np.asarray(ns.wiki)
    assert w[0, 0] == 3                     # honest report of arm 2
    assert w[1, 0] not in (0, 3)            # false report: some other arm
    assert w[2].sum() == 0                  # kept (blank)
    assert np.asarray(info["write_honest"]).tolist() == [True, False, False]


def test_scripted_bad_authors():
    cfg = BanditConfig(n_agents=6, n_bandits=3, n_arms=4, visit_len=2, n_deceivers=2, n_incompetent=1)
    dec, inc = bad_slots(cfg)
    assert dec.tolist() == [False, False, False, False, True, True] and inc.tolist() == [False, False, False, True, False, False]
    _, step, _ = make_world(cfg)
    s = st(cfg, where=jnp.array([0, 0, 0, 0, 0, 0]), best=jnp.array([1, 2, 3]), k=jnp.array(0))
    Z = jnp.zeros(6, jnp.int32)
    s1, _, _, info = step(jax.random.PRNGKey(1), s, Z, Z)
    assert int(s1.wiki.sum()) == 0                          # first step of the visit: scripted agents don't write
    s2, _, _, info = step(jax.random.PRNGKey(2), s1._replace(where=jnp.zeros(6, jnp.int32)), Z, Z)
    w = np.asarray(s2.wiki)[:, 0]
    assert w[4] == w[5] and w[4] != 2 and w[4] > 0         # coordinated deceivers, never the good arm (symbol 2)
    assert w[3] > 0 and w[:3].sum() == 0                    # incompetent wrote something; learners chose to keep
    assert np.asarray(info["wrote"])[3:].all()

def test_learners_cannot_write_for_bad_slots():
    cfg = BanditConfig(n_agents=3, n_bandits=2, n_arms=3, visit_len=4, n_deceivers=1)
    _, step, _ = make_world(cfg)
    s = st(cfg, where=jnp.array([0, 0, 0]), k=jnp.array(0))
    ns, _, _, _ = step(jax.random.PRNGKey(1), s, jnp.zeros(3, jnp.int32), jnp.array([1, 1, 1]))  # all "report belief"
    w = np.asarray(ns.wiki)[:, 0]
    assert w[0] > 0 and w[1] > 0 and w[2] == 0             # the deceiver's learned action is ignored mid-visit

def test_anonymous_order():
    cfg = BanditConfig(n_agents=5, n_bandits=1, n_arms=4, identity="anonymous")
    _, _, observe = make_world(cfg)
    wiki = jnp.array([[1], [2], [3], [4], [0]])
    seen = []
    for t in range(3):
        s = st(cfg, wiki=wiki, where=jnp.zeros(5, jnp.int32), t=jnp.array(t))
        page = np.asarray(observe(s))[:, 2 * 4 + 1:].reshape(5, 5, 5).argmax(-1)
        assert page[:, 0].tolist() == [1, 2, 3, 4, 0]                       # own entry first
        assert all(sorted(page[i].tolist()) == [0, 1, 2, 3, 4] for i in range(5))
        seen.append(page[0].tolist())
    assert len({tuple(x) for x in seen}) > 1                                 # order changes over time

def test_agreement_metrics():
    cfg = BanditConfig(n_agents=4, n_bandits=1, n_arms=3, n_deceivers=1)
    _, step, _ = make_world(cfg)
    wiki = jnp.array([[2], [2], [0], [3]])                 # learners 0,1 name arm 1; deceiver 3 names arm 2
    s = st(cfg, wiki=wiki, where=jnp.zeros(4, jnp.int32), k=jnp.array(0))
    _, _, _, info = step(jax.random.PRNGKey(1), s, jnp.array([1, 1, 2, 0]), jnp.zeros(4, jnp.int32))
    # reader 2 pulled arm 2: agrees with the deceiver (1 of 1 seen), with 0 of 2 learner entries
    assert int(info["agree_bad"][2]) == 1 and int(info["seen_bad"][2]) == 1
    assert int(info["agree_good"][2]) == 0 and int(info["seen_good"][2]) == 2
    # reader 0 pulled arm 1: agrees with learner 1 (1 of 1 seen, own entry excluded)
    assert int(info["agree_good"][0]) == 1 and int(info["seen_good"][0]) == 1


def test_blame_and_rivalry():
    wiki = jnp.array([[2, 0], [2, 0], [3, 0], [0, 0]])
    base = dict(n_agents=4, n_bandits=2, n_arms=3, p_high=1.0, p_low=0.0, write_mode="symbol")
    cfg = BanditConfig(citation=0.5, blame=0.4, **base)
    _, step, _ = make_world(cfg)
    s = st(cfg, wiki=wiki, where=jnp.array([0, 0, 0, 0]), best=jnp.array([0, 0]))
    # agent 3 follows authors 0 and 1 (arm 1) and loses (good arm is 0): they share the blame
    _, _, r, info = step(jax.random.PRNGKey(1), s, jnp.array([0, 0, 0, 1]), jnp.zeros(4, jnp.int32))
    assert np.allclose(np.asarray(info["blamed"])[:2], [0.5, 0.5])
    assert np.allclose(np.asarray(r), np.asarray(info["raw"]) - 0.4 * np.asarray(info["blamed"]))
    cfg = BanditConfig(reward_mode="competitive", rivalry=3.0, **base)
    _, step, _ = make_world(cfg)
    s = st(cfg, where=jnp.array([0, 0, 0, 0]), best=jnp.array([0, 0]))
    _, _, r, info = step(jax.random.PRNGKey(1), s, jnp.array([0, 0, 1, 1]), jnp.zeros(4, jnp.int32))
    raw = np.asarray(info["raw"])
    assert np.allclose(np.asarray(r), raw - 3.0 * (raw.sum() - raw) / 3)

def test_wipe_on_death():
    cfg = BanditConfig(n_agents=2, n_bandits=3, n_arms=3, wipe_on_death=True)
    s = st(cfg, wiki=jnp.ones((2, 3), jnp.int32))
    ns = respawn_agent(cfg, s, 1)
    assert np.asarray(ns.wiki).tolist() == [[1, 1, 1], [0, 0, 0]]

def test_shuffled_learner_runs():
    wc, lc = BanditConfig(n_agents=4, n_bandits=4, reward_mode="shuffled", citation=0.5), LearnConfig(num_envs=8, num_steps=8)
    init, update, _ = make_learner(wc, lc)
    r, m = update(init(jax.random.PRNGKey(0)))
    assert m["raw"].shape == (4,)

def test_private_notebook():
    cfg = BanditConfig(n_agents=4, n_bandits=1, n_arms=4, read_others=False)
    _, _, observe = make_world(cfg)
    wiki = jnp.array([[1], [2], [3], [4]])
    s = st(cfg, wiki=wiki, where=jnp.zeros(4, jnp.int32))
    page = np.asarray(observe(s))[:, 2 * 4 + 1:].reshape(4, 4, 5).argmax(-1)
    assert page.tolist() == [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]   # only own entry visible
    cfg2 = cfg._replace(read_others=True)
    _, _, observe2 = make_world(cfg2)
    page2 = np.asarray(observe2(s))[:, 2 * 4 + 1:].reshape(4, 4, 5).argmax(-1)
    assert page2.tolist() == [[1, 2, 3, 4]] * 4

def _page(cfg, observe, s):
    N, A = cfg.n_agents, cfg.n_arms
    return np.asarray(observe(s))[:, 2 * A + 1:].reshape(N, N, A + 1).argmax(-1)

def test_read_k_commons():
    cfg = BanditConfig(n_agents=4, n_bandits=1, n_arms=4, read_k=1)
    _, _, observe = make_world(cfg)
    s = st(cfg, wiki=jnp.array([[1], [2], [3], [4]]), where=jnp.zeros(4, jnp.int32))
    assert _page(cfg, observe, s).tolist() == [[1, 2, 0, 0], [0, 2, 3, 0], [0, 0, 3, 4], [1, 0, 0, 4]]
    cfg0 = cfg._replace(read_k=0)
    _, _, observe0 = make_world(cfg0)
    assert _page(cfg0, observe0, s).tolist() == [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]

def test_replicate_own():
    cfg = BanditConfig(n_agents=3, n_bandits=2, n_arms=3, replicate_own=True)
    _, _, observe = make_world(cfg)
    s = st(cfg, wiki=jnp.array([[1, 2], [3, 0], [2, 1]]), where=jnp.array([0, 0, 1]))
    assert _page(cfg, observe, s).tolist() == [[1, 1, 1], [3, 3, 3], [1, 1, 1]]

def test_saboteur_reward_and_slots():
    cfg = BanditConfig(n_agents=4, n_bandits=1, n_arms=2, p_high=1.0, p_low=0.0, n_saboteurs=1, n_deceivers=1)
    assert saboteur_slots(cfg).tolist() == [False, False, True, False]
    _, step, _ = make_world(cfg)
    s = st(cfg, best=jnp.array([0]), where=jnp.zeros(4, jnp.int32))
    _, _, r, info = step(jax.random.PRNGKey(1), s, jnp.array([0, 1, 0, 0]), jnp.zeros(4, jnp.int32))
    # honest learners are slots 0 and 1: one wins, one loses -> saboteur gets -0.5
    assert np.asarray(info["raw"]).tolist() == [1, 0, 1, 1]
    assert np.isclose(float(r[2]), -0.5) and np.isclose(float(r[0]), 1.0)

def test_keep_torso():
    wc, lc = BanditConfig(n_agents=3, n_bandits=4, keep_torso=True), LearnConfig(num_envs=4, num_steps=4)
    init, update, kill = make_learner(wc, lc)
    r, _ = update(init(jax.random.PRNGKey(0)))
    r2 = kill(r, 1, jax.random.PRNGKey(5))
    for layer in ("l1", "l2"):
        assert np.allclose(np.asarray(r.params[layer]["w"]), np.asarray(r2.params[layer]["w"]))
    a, b = np.asarray(r.params["pull"]["w"]), np.asarray(r2.params["pull"]["w"])
    assert np.allclose(a[[0, 2]], b[[0, 2]]) and not np.allclose(a[1], b[1])

def test_hide_own():
    cfg = BanditConfig(n_agents=3, n_bandits=1, n_arms=3, hide_own=True)
    _, _, observe = make_world(cfg)
    s = st(cfg, wiki=jnp.array([[1], [2], [3]]), where=jnp.zeros(3, jnp.int32))
    assert _page(cfg, observe, s).tolist() == [[0, 2, 3], [1, 0, 3], [1, 2, 0]]

def test_identity_period():
    cfg = BanditConfig(n_agents=5, n_bandits=1, n_arms=5, identity_period=10)
    _, _, observe = make_world(cfg)
    s = st(cfg, wiki=jnp.array([[1], [2], [3], [4], [5]]), where=jnp.zeros(5, jnp.int32))
    pages = {t: _page(cfg, observe, s._replace(t=jnp.array(t))) for t in (0, 3, 9, 10, 25)}
    assert (pages[0] == pages[3]).all() and (pages[0] == pages[9]).all()     # same names within a period
    assert not all((pages[0] == pages[t]).all() for t in (10, 25))           # reshuffled in later periods
    for p in pages.values():
        assert p[:, 0].tolist() == [1, 2, 3, 4, 5]                           # own entry first
        assert all(sorted(row[1:]) == sorted(set(range(1, 6)) - {i + 1}) for i, row in enumerate(p))

def test_sleeper_deceivers():
    cfg = BanditConfig(n_agents=3, n_bandits=1, n_arms=4, visit_len=1, n_deceivers=1, deceive_after=100)
    _, step, _ = make_world(cfg)
    Z = jnp.zeros(3, jnp.int32)
    early = st(cfg, where=Z, best=jnp.array([2]), k=jnp.array(0), t=jnp.array(5))
    s1, _, _, _ = step(jax.random.PRNGKey(1), early, Z, Z)
    assert int(s1.wiki[2, 0]) == 3                           # tells the truth (arm 2 -> symbol 3)
    late = st(cfg, where=Z, best=jnp.array([2]), k=jnp.array(0), t=jnp.array(500))
    s2, _, _, _ = step(jax.random.PRNGKey(1), late, Z, Z)
    assert int(s2.wiki[2, 0]) not in (0, 3)                  # lies afterwards

def test_sync_deaths():
    wc, lc = BanditConfig(n_agents=3, n_bandits=2, sync_deaths=True), LearnConfig(num_envs=2, num_steps=4)
    log, _ = run_world(wc, lc, life_updates=3, generations=3, seed=0, verbose=False)
    g = log["gen"]
    assert (g == g[:, :1]).all() and g[-1, 0] == 2           # everyone dies together, twice

def test_bottleneck():
    wc, lc = BanditConfig(n_agents=4, n_bandits=2, bottleneck_at=5, bottleneck_frac=0.5), LearnConfig(num_envs=2, num_steps=4)
    log, _ = run_world(wc, lc, life_updates=40, generations=1, seed=0, verbose=False)
    c = log["bottleneck_cohort"]
    assert c.sum() == 2
    born_at_5 = (log["age"][5] == 0)
    assert (born_at_5 == c).all()                            # exactly the cohort is reborn at update 5
    assert (log["gen"][5][c] == 1).all() and (log["gen"][5][~c] == 0).all()

def test_juvenile_silence():
    wc, lc = BanditConfig(n_agents=4, n_bandits=2, juvenile_updates=3), LearnConfig(num_envs=4, num_steps=8)
    log, _ = run_world(wc, lc, life_updates=6, generations=2, seed=0, verbose=False)
    w, age = log["write_rate"], log["age"]
    assert (w[age < 3] == 0).all()                           # juveniles never write
    assert (w[age >= 3] > 0).any()                           # adults do

def test_juvenile_evolution():
    wc = BanditConfig(n_agents=4, n_bandits=2, juvenile_updates=5, juvenile_evolve=True, juvenile_sigma=3.0)
    lc = LearnConfig(num_envs=8, num_steps=8)
    log, _ = run_world(wc, lc, life_updates=6, generations=3, seed=0, verbose=False)
    J = log["J_final"]
    assert J.shape == (8, 4) and (J >= 0).all() and (J <= 12).all()
    assert (log["J_mean"][0] == 5).all() and not np.allclose(log["J_mean"][-1], 5)   # it changes once agents die
    w, age = log["write_rate"], log["age"]
    assert (w[age < J.min()] == 0).all()                     # nobody writes before the shortest juvenile period

def test_juvenile_group_selection():
    wc = BanditConfig(n_agents=4, n_bandits=2, juvenile_updates=5, juvenile_evolve=True, juvenile_sigma=0.0,
                      juvenile_group=True, sync_deaths=True)
    log, _ = run_world(wc, LearnConfig(num_envs=8, num_steps=8), life_updates=6, generations=2, seed=0, verbose=False)
    J = log["J_final"]
    assert (J == 5).all()                                    # no mutation: copies only copy whole rows of 5s
    wc2 = wc._replace(juvenile_sigma=4.0)
    log2, _ = run_world(wc2, LearnConfig(num_envs=8, num_steps=8), life_updates=6, generations=2, seed=0, verbose=False)
    assert not (log2["J_final"] == 5).all()

def test_citation_switch():
    cfg = BanditConfig(n_agents=4, n_bandits=2, n_arms=3, p_high=1.0, p_low=0.0, write_mode="symbol",
                       citation=-0.5, citation_early=0.5, citation_switch_at=10)
    _, step, _ = make_world(cfg)
    wiki = jnp.array([[2, 0], [2, 0], [3, 0], [0, 0]])
    pulls = jnp.array([0, 0, 0, 1])
    for t, sign in ((3, 1.0), (50, -1.0)):                   # before the switch credit, after it rivalry
        s = st(cfg, wiki=wiki, where=jnp.array([0, 0, 0, 0]), best=jnp.array([1, 0]), t=jnp.array(t))
        _, _, r, info = step(jax.random.PRNGKey(1), s, pulls, jnp.zeros(4, jnp.int32))
        c = np.asarray(info["cited"])
        assert np.allclose(c[:2], [0.5, 0.5])
        assert np.allclose(np.asarray(r), np.asarray(info["raw"]) + sign * 0.5 * c)

def test_write_prior():
    cfg = BanditConfig(n_agents=2, n_bandits=2)
    for p in (0.1, 0.01):
        net = init_net(jax.random.PRNGKey(0), obs_size(cfg), 16, cfg.n_arms, n_write_actions(cfg), p)
        logits = net["write"]["b"]
        keep = float(jax.nn.softmax(logits)[0])
        assert abs(keep - (1 - p)) < 1e-4                    # the newborn keeps its entry with probability 1 - p

def test_lives_and_honesty_by_copy():
    wc, lc = BanditConfig(n_agents=3, n_bandits=2, citation=0.5), LearnConfig(num_envs=4, num_steps=8)
    log, _ = run_world(wc, lc, life_updates=6, generations=2, seed=0, verbose=False)
    lives = log["lives"]
    assert lives.shape[1] == 4 and set(lives[:, 0].astype(int)) == {0, 1, 2}   # every agent completed a life
    assert ((lives[:, 3] >= 0) & (lives[:, 3] <= 1)).all()
    h = log["honest_by_copy"]
    assert h.shape == (4,) and np.nanmax(h) <= 1 and np.nanmin(h) >= 0

def test_seat_per_copy():
    # named authors with seats per copy: own entry first; the others' order differs between copies,
    # stays fixed within a copy, and is redrawn per copy every identity_period steps
    cfg = BanditConfig(n_agents=5, n_bandits=1, n_arms=4, citation=0.5, seat_per_copy=True)
    def order(key, t=0, cfg=cfg):
        reset, _, observe = make_world(cfg)
        s, _ = reset(key)
        s = s._replace(wiki=jnp.array([[1], [2], [3], [4], [4]], jnp.int32), t=jnp.asarray(t, jnp.int32))
        obs = np.asarray(observe(s))
        page = obs[:, 2 * 4 + 1:].reshape(5, 5, 5)                  # reader, position, one-hot of the entry
        return page.argmax(-1)                                      # entry (1..4) shown at each position
    o1, o2 = order(jax.random.PRNGKey(1)), order(jax.random.PRNGKey(2))
    entries = np.array([1, 2, 3, 4, 4])
    for o in (o1, o2):
        assert (o[:, 0] == entries).all()                           # own entry first for every reader
    orders = [tuple(order(jax.random.PRNGKey(k))[0]) for k in range(8)]
    assert len(set(orders)) > 1                                     # differs between copies
    assert (order(jax.random.PRNGKey(1), t=0) == order(jax.random.PRNGKey(1), t=5000)).all()   # fixed in a copy
    cfg2 = cfg._replace(identity_period=100)
    redrawn = [tuple(order(jax.random.PRNGKey(1), t=100 * j, cfg=cfg2)[0]) for j in range(8)]
    assert len(set(redrawn)) > 1                                    # redrawn over time

def test_own_memory():
    cfg = BanditConfig(n_agents=2, n_bandits=2, n_arms=3, visit_len=2, p_high=1.0, p_low=0.0, own_memory=True)
    reset, step, observe = make_world(cfg)
    s, obs = reset(jax.random.PRNGKey(0))
    assert obs.shape == (2, obs_size(cfg)) and obs_size(cfg) == obs_size(cfg._replace(own_memory=False)) + 6
    s = s._replace(best=jnp.array([0, 1]), where=jnp.array([0, 1]))
    z = jnp.zeros(2, jnp.int32)
    s, _, _, _ = step(jax.random.PRNGKey(1), s, jnp.array([0, 1]), z)     # both pull the good arm
    assert float(s.mem_pulls.sum()) == 0                                    # memory written only when a visit ends
    s, _, _, _ = step(jax.random.PRNGKey(2), s, jnp.array([0, 0]), z)     # visit ends
    mp, mw = np.asarray(s.mem_pulls), np.asarray(s.mem_wins)
    assert mp[0, 0].tolist() == [2, 0, 0] and mw[0, 0].tolist() == [2, 0, 0]
    assert mp[1, 1].tolist() == [1, 1, 0] and mw[1, 1].tolist() == [0, 1, 0]
    assert mp[0, 1].sum() == 0 and mp[1, 0].sum() == 0
    s2 = s._replace(where=jnp.array([0, 1]), k=jnp.zeros((), jnp.int32))  # a second visit to the same machines
    s2, _, _, _ = step(jax.random.PRNGKey(3), s2, jnp.array([1, 1]), z)
    s2, _, _, _ = step(jax.random.PRNGKey(4), s2, jnp.array([1, 1]), z)
    assert np.allclose(np.asarray(s2.mem_pulls)[0, 0], [1, 2, 0])          # halved old counts plus the new visit
    ns = respawn_agent(cfg, s, 1)
    assert float(ns.mem_pulls[1].sum()) == 0 and float(ns.mem_pulls[0].sum()) == 2   # only the dead agent forgets

def test_immortal():
    wc, lc = BanditConfig(n_agents=3, n_bandits=2, citation=0.5, immortal=True), LearnConfig(num_envs=4, num_steps=8)
    log, _ = run_world(wc, lc, life_updates=6, generations=2, seed=0, verbose=False)
    assert log["gen"].max() == 0 and len(log["lives"]) == 0

def test_reset_part():
    lc = LearnConfig(num_envs=4, num_steps=8)
    for part, changed in (("pull", {"pull"}), ("write", {"write"}), ("heads", {"pull", "write"}),
                          ("all", {"l1", "l2", "pull", "write", "v"})):
        wc = BanditConfig(n_agents=3, n_bandits=2, citation=0.5, immortal=True, reset_at=1, reset_part=part)
        init, update, _ = make_learner(wc, lc)
        runner, _ = update(init(jax.random.PRNGKey(0)))          # one update, so Adam moments are non-zero
        new = make_reset(wc, lc)(runner, jax.random.PRNGKey(5))
        for k in ("l1", "l2", "pull", "write", "v"):
            same = all(np.allclose(a, b) for a, b in zip(jax.tree.leaves(runner.params[k]), jax.tree.leaves(new.params[k])))
            assert same == (k not in changed), (part, k)
        # moments of the re-initialised weights are zero; the others are untouched
        def moments(state):
            out = {}
            jax.tree_util.tree_map_with_path(
                lambda path, x: out.setdefault(tuple(getattr(p, "key", getattr(p, "idx", getattr(p, "name", None))) for p in path), x),
                state)
            return out
        old_m, new_m = moments(runner.opt_state), moments(new.opt_state)
        for path, x in new_m.items():
            if any(k in changed for k in path):
                assert np.all(np.asarray(x) == 0), (part, path)
            else:
                assert np.allclose(np.asarray(x), np.asarray(old_m[path])), (part, path)
    # in a run, the reset happens once and nobody dies
    wc = BanditConfig(n_agents=3, n_bandits=2, citation=0.5, immortal=True, reset_at=3, reset_part="write")
    log, _ = run_world(wc, lc, life_updates=6, generations=1, seed=0, verbose=False)
    assert log["gen"].max() == 0 and np.isfinite(log["raw"]).all()


def test_credit_on_read():
    # "read" rivalry: every author whose entry the winning reader saw shares the credit, whatever arm it pulled
    cfg = BanditConfig(n_agents=4, n_bandits=2, n_arms=3, p_high=1.0, p_low=0.0, citation=0.5, write_mode="symbol",
                       credit_on="read")
    _, step, _ = make_world(cfg)
    wiki = jnp.array([[2, 0], [2, 0], [3, 0], [0, 0]])
    s = st(cfg, wiki=wiki, where=jnp.array([0, 0, 0, 0]), best=jnp.array([1, 0]))
    _, _, r, info = step(jax.random.PRNGKey(1), s, jnp.array([0, 0, 0, 1]), jnp.zeros(4, jnp.int32))
    c = np.asarray(info["cited"])
    assert np.allclose(c, [1 / 3, 1 / 3, 1 / 3, 0])           # author 2 named another arm, and still shares
    assert np.allclose(np.asarray(r), np.asarray(info["raw"]) + 0.5 * c)


def test_write_noise():
    lc = LearnConfig(num_envs=4, num_steps=8)
    # with noise 1 every learner writes at every step, honestly about half the time
    wc = BanditConfig(n_agents=3, n_bandits=2, citation=0.5, write_noise=1.0)
    log, _ = run_world(wc, lc, life_updates=4, generations=1, seed=0, verbose=False)
    assert np.allclose(log["write_rate"], 1.0)
    assert 0.3 < np.nanmean(log["honest_writes"]) < 0.7
    # a juvenile period still mutes newborns: noise never writes for a muted agent
    wc = BanditConfig(n_agents=3, n_bandits=2, citation=0.5, write_noise=1.0, juvenile_updates=100)
    log, _ = run_world(wc, lc, life_updates=4, generations=1, seed=0, verbose=False)
    assert np.allclose(log["write_rate"], 0.0)

def test_credit_seen_only():
    base = dict(n_agents=4, n_bandits=2, n_arms=3, p_high=1.0, p_low=0.0, citation=0.5, write_mode="symbol")
    wiki = jnp.array([[2, 0], [2, 0], [3, 0], [0, 0]])          # authors 0 and 1 name arm 1 on machine 0
    pulls = jnp.array([0, 0, 0, 1])                             # agent 3 pulls arm 1 and wins
    def cited(**kw):
        cfg = BanditConfig(**base, **kw)
        _, step, _ = make_world(cfg)
        s = st(cfg, wiki=wiki, where=jnp.array([0, 0, 0, 0]), best=jnp.array([1, 0]))
        return np.asarray(step(jax.random.PRNGKey(1), s, pulls, jnp.zeros(4, jnp.int32))[3]["cited"])
    assert np.allclose(cited(read_others=False), [0.5, 0.5, 0, 0])                         # original rule: unseen pages paid
    assert np.allclose(cited(read_others=False, credit_seen_only=True), 0)                 # private notebooks: nothing
    assert np.allclose(cited(read_k=1, credit_seen_only=True), [1.0, 0, 0, 0])             # reader 3 sees only authors 3 and 0
    assert np.allclose(cited(credit_seen_only=True), [0.5, 0.5, 0, 0])                    # full wiki: unchanged

def test_juvenile_group_selection_at_birth():
    wc = BanditConfig(n_agents=4, n_bandits=2, juvenile_updates=5, juvenile_evolve=True, juvenile_sigma=0.0,
                      juvenile_group_birth=True)
    lc = LearnConfig(num_envs=8, num_steps=8)
    log, _ = run_world(wc, lc, life_updates=8, generations=3, seed=0, verbose=False)
    assert (log["J_final"] == 5).all()                       # no mutation: values never change
    log2, _ = run_world(wc._replace(juvenile_sigma=4.0), lc, life_updates=8, generations=3, seed=0, verbose=False)
    Jm, age = log2["J_mean"], log2["age"]
    changed = np.abs(np.diff(Jm, axis=0)) > 0
    assert changed.any()                                     # selection and mutation do act
    assert (age[1:][changed] == 0).all()                     # a slot's value changes only when an heir is born there


def test_juvenile_group_selection_each_death():
    wc = BanditConfig(n_agents=4, n_bandits=2, juvenile_updates=5, juvenile_evolve=True, juvenile_sigma=0.0,
                      juvenile_group_each_death=True)
    lc = LearnConfig(num_envs=8, num_steps=8)
    log, _ = run_world(wc, lc, life_updates=8, generations=3, seed=0, verbose=False)
    assert (log["J_final"] == 5).all()                       # no mutation: values never change
    wc2 = wc._replace(juvenile_sigma=4.0)
    log2, _ = run_world(wc2, lc, life_updates=8, generations=3, seed=0, verbose=False)
    Jm, age = log2["J_mean"], log2["age"]
    changed = np.abs(np.diff(Jm, axis=0)) > 0
    assert changed.any()                                     # selection and mutation do act
    assert (age[1:][changed] == 0).all()                     # a slot's value changes only when an heir is born there
    log3, _ = run_world(wc2._replace(juvenile_group_each_death=False), lc, life_updates=8, generations=3, seed=0,
                        verbose=False)
    assert not np.array_equal(log2["J_final"], log3["J_final"])   # judged by the copy, not by the agent
