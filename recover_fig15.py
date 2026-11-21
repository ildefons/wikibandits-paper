"""Recover the hysteresis time courses (smoothed honest writes per seed) from the notebook's vector fig15 PDF."""
import sys
import numpy as np
import pymupdf

pdf, out = sys.argv[1], sys.argv[2]
page = pymupdf.open(pdf)[0]
words = page.get_text("words")
draws = page.get_drawings()

def num(s):
    try:
        return float(s.replace("−", "-"))
    except ValueError:
        return None

# short black tick marks
ticks_v, ticks_h = [], []          # vertical tick (x axis): (x, y); horizontal tick (y axis): (x, y)
for d in draws:
    if d.get("color") != (0.0, 0.0, 0.0) or d.get("dashes") not in ("[] 0", None):
        continue
    for it in d["items"]:
        if it[0] != "l":
            continue
        a, b = it[1], it[2]
        L = abs(a.x - b.x) + abs(a.y - b.y)
        if 1 < L < 6:
            (ticks_v if abs(a.x - b.x) < 0.01 else ticks_h).append(((a.x + b.x) / 2, (a.y + b.y) / 2))

def nearest_tick(cx, cy, ticks):
    t = min(ticks, key=lambda t: (t[0] - cx) ** 2 + (t[1] - cy) ** 2)
    return t

# x-axis calibration per panel column/row; y-axis per row
xcal, ycal = {}, {}
for w in words:
    v = num(w[4])
    if v is None:
        continue
    cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
    if v >= 0 and float(v).is_integer() and v % 500 == 0 and w[4] in ("0", "500", "1000", "1500", "2000", "2500"):
        t = nearest_tick(cx, cy - 8, ticks_v)
        key = (round(cy), int(cx // 300))
        xcal.setdefault(key, []).append((t[0], v))
    elif w[4] in ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"):
        t = nearest_tick(cx + 8, cy, ticks_h)
        ycal.setdefault(round(w[3] > 250), []).append((t[1], v))

def fit(pairs):
    p, v = np.array(pairs).T
    a, b = np.polyfit(p, v, 1)
    return a, b

rows = {}
for (ry, col), pairs in xcal.items():
    rows.setdefault(ry, {})[col] = fit(pairs)
row_keys = sorted(rows)
ymap = {r: fit(ycal[r]) for r in ycal}
colors = {"honest": (0.114, 0.62, 0.459), "lying": (0.847, 0.353, 0.188)}
series = {}
for d in draws:
    col = d.get("color")
    if col is None or len(d["items"]) < 20:
        continue
    col = tuple(round(c, 3) for c in col)
    start = [k for k, c in colors.items() if c == col]
    if not start:
        continue
    pts = [d["items"][0][1]] + [it[-1] for it in d["items"]]
    xs = np.array([p.x for p in pts]); ys = np.array([p.y for p in pts])
    row = int(ys.mean() > 250)
    colx = int(xs.mean() // 300)
    ax, bx = rows[row_keys[row]][colx]
    ay, by = ymap[row]
    key = f"r{row}c{colx}_{start[0]}"
    series.setdefault(key, []).append(np.stack([ax * xs + bx, ay * ys + by]))
np.savez(out, **{f"{k}_{i}": s for k, v in series.items() for i, s in enumerate(v)})
for k, v in sorted(series.items()):
    print(k, len(v), [f"{s[0, 0]:.0f}-{s[0, -1]:.0f}" for s in v], [f"{s[1, -200:].mean():.3f}" for s in v])
