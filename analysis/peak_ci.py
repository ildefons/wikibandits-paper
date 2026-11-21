"""Best fixed juvenile period: vertex of a quadratic through the best grid length and its two neighbours,
with a parametric bootstrap from each length's mean and 95% interval."""
import re, sys
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); rng = np.random.default_rng(0)
def peak(J, f):
    i = int(np.argmax(f)); i = min(max(i, 1), len(J) - 2)
    a, b, c = np.polyfit(J[i - 1:i + 2], f[i - 1:i + 2], 2)
    return float(np.clip(-b / (2 * a), J[i - 1], J[i + 1])) if a < 0 else float(J[int(np.argmax(f))])
for s, g in d.groupby("setting", sort=False):
    g = g.sort_values("juvenile period"); J = g["juvenile period"].to_numpy(float); f = g["fitness"].to_numpy()
    lo, hi = np.array([[float(x) for x in re.findall(r"[\d.]+", t)] for t in g["95% CI"]]).T
    se = (hi - lo) / 3.92
    boots = [peak(J, rng.normal(f, se)) for _ in range(4000)]
    near = J[f >= f.max() - 0.005]
    print(f"{s:55s} peak {peak(J, f):5.1f}  95% interval [{np.percentile(boots, 2.5):.0f}, {np.percentile(boots, 97.5):.0f}]"
          f"  within 0.005 of best: {near.min():.0f}-{near.max():.0f}")
