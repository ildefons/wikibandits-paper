"""Two-fold and leave-one-out cross-validation of the freshness model's one fitted constant (Table 2 values)."""
import math, itertools
import numpy as np
V = 8
T = [  # setting, K, D (None = frozen), q, observed plurality-right
    ("default", 8, 2000, .81, .955), ("D=500", 8, 500, .77, .879), ("D=8000", 8, 8000, .82, .980),
    ("frozen", 8, None, .84, 1.000), ("life100", 8, 2000, .76, .944), ("life900", 8, 2000, .82, .962),
    ("D500 life100", 8, 500, .72, .855), ("D500 life900", 8, 500, .80, .896), ("D8000 life100", 8, 8000, .78, .980),
    ("D8000 life900", 8, 8000, .82, .983), ("K=4", 4, 2000, .79, .975), ("K=32", 32, 2000, .74, .848),
    ("N=4", 8, 2000, .80, .932), ("N=8", 8, 2000, .81, .951), ("N=32", 8, 2000, .81, .955)]
def core(K, D, q, kind):
    if D is None: return 0.0
    L = math.log(1 / (1 - 1 / (2 * q))) if kind == "majority" else math.log((1 + q) / q)
    return K * V / D * L
for kind in ("majority", "plurality"):
    c = np.array([core(K, D, q, kind) for _, K, D, q, _ in T]); obs = 1 - np.array([o for *_, o in T])
    eps_all = np.mean(obs - c)
    loo = [abs(obs[i] - c[i] - np.mean(np.delete(obs - c, i))) for i in range(len(T))]
    rng = np.random.default_rng(0); errs = []
    for _ in range(1000):
        idx = rng.permutation(len(T)); a, b = idx[:7], idx[7:]
        e = np.mean((obs - c)[a]); errs += list(abs(obs[b] - c[b] - e))
    print(f"{kind}: eps(all)={eps_all:.3f}; in-sample mean |err|={np.mean(abs(obs-c-eps_all)):.3f}, max={np.max(abs(obs-c-eps_all)):.3f}; "
          f"leave-one-out mean |err|={np.mean(loo):.3f}, max={np.max(loo):.3f}; random halves mean |err|={np.mean(errs):.3f}, 95th pct={np.percentile(errs,95):.3f}")
