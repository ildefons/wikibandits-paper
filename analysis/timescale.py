"""Value of an inheritance against protection time / reading time (existing results only).

protection time T_p = J + 1/(-4 ln p): J = juvenile period (updates), p = newborns' keep prior.
An entry survives a visit (8 pulls) with probability p**8, and each machine is visited by its
owner about every 2 updates, so survival after u updates is p**(4u) (mean 1/(-4 ln p) updates).
reading time T_r = updates until a newborn first beats 0.61 with the inherited record intact
(the juvenile-period-50 runs, where nothing is overwritten): notebook 41, kept features 11, die-off 12.
gain = newborn reward with the record inherited minus erased; normalised by the record's largest gain.
"""
import csv, math, sys
import numpy as np
rows = {x['setting']: x for x in csv.DictReader(open(sys.argv[1]))}
def val(k, col='newborn reward'):
    m, s = rows[k][col].split('±'); return float(m), float(s), int(rows[k]['seeds'])
REC = {  # record: (inherited prefix, erased prefix, reading-time setting)
    'notebook, fresh heir': ('notebook_cited_static', 'notebook_wiped_cited_static', 'notebook_cited_static_juv'),
    'notebook, heir keeps features': ('notebook_torso_cited_static', 'notebook_wiped_torso_cited_static', 'notebook_torso_cited_static_juv'),
    'pages after a die-off': ('sync_cited_static', 'sync_wipe_cited_static', 'sync_cited_static_juv'),
}
SUFFIX = {'': (0, .90), '_keep97': (0, .97), '_keep99': (0, .99), '_juv10': (10, .90), '_juv25': (25, .90),
          '_juv': (50, .90), '_juv100': (100, .90)}
out = []
for rec, (inh, era, rd) in REC.items():
    T_r = float(rows[rd]['t to 0.61'].split('±')[0])
    for suf, (J, p) in SUFFIX.items():
        ki, ke = inh + suf, era + suf
        if rec == 'pages after a die-off' and suf == '':
            ki = 'sync_cited_static'
        if ki not in rows or ke not in rows:
            continue
        (mi, si, ni), (me, se, ne) = val(ki), val(ke)
        T_p = J + 1 / (-4 * math.log(p))
        out.append(dict(record=rec, J=J, keep=p, T_p=round(T_p, 1), T_r=T_r, ratio=round(T_p / T_r, 3),
                        gain=round(mi - me, 3), se=round(math.hypot(si, se), 4), seeds=min(ni, ne)))
for rec in REC:
    g = max(o['gain'] for o in out if o['record'] == rec)
    for o in out:
        if o['record'] == rec:
            o['share of max'] = round(o['gain'] / g, 3)
w = csv.DictWriter(open(sys.argv[2], 'w', newline=''), fieldnames=list(out[0].keys()))
w.writeheader(); w.writerows(out)
for o in sorted(out, key=lambda o: o['ratio']):
    print(f"{o['record']:30s} J={o['J']:3d} keep={o['keep']:.2f} Tp={o['T_p']:6.1f} Tr={o['T_r']:5.0f} ratio={o['ratio']:6.3f} gain={o['gain']:+.3f} share={o['share of max']:.2f}")
x = np.log([o['ratio'] for o in out]); y = np.array([o['share of max'] for o in out])
from scipy.optimize import curve_fit
f = lambda x, a, b: 1 / (1 + np.exp(-(x - a) * b))
pp, _ = curve_fit(f, x, y, p0=[0, 2])
res = y - f(x, *pp)
print('logistic in log ratio: half-value ratio %.2f, slope %.2f, R2 %.3f' % (math.exp(pp[0]), pp[1], 1 - res.var() / y.var()))
