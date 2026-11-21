"""Leave-one-record-out prediction and a bootstrap interval for the timescale curve's midpoint."""
import csv, math, sys
import numpy as np
from scipy.optimize import curve_fit
rows = list(csv.DictReader(open(sys.argv[1])))
x = np.log([float(r["ratio"]) for r in rows]); y = np.array([float(r["share of max"]) for r in rows])
rec = np.array([r["record"] for r in rows])
f = lambda x, m, b: 1 / (1 + np.exp(-(x - m) * b))
p, _ = curve_fit(f, x, y, p0=[0, 2])
print("all: midpoint %.2f slope %.2f" % (math.exp(p[0]), p[1]))
for r in sorted(set(rec)):
    tr, te = rec != r, rec == r
    q, _ = curve_fit(f, x[tr], y[tr], p0=[0, 2], maxfev=10000)
    err = np.abs(y[te] - f(x[te], *q))
    print(f"leave out {r:30s}: midpoint {math.exp(q[0]):.2f}, mean abs error on it {err.mean():.3f}, max {err.max():.3f}")
rng = np.random.default_rng(0); mids = []
for _ in range(2000):
    i = rng.integers(0, len(x), len(x))
    try:
        q, _ = curve_fit(f, x[i], y[i], p0=[0, 2], maxfev=10000); mids.append(math.exp(q[0]))
    except Exception:
        pass
print("bootstrap midpoint 95%% interval [%.2f, %.2f] (%d fits)" % (*np.percentile(mids, [2.5, 97.5]), len(mids)))
