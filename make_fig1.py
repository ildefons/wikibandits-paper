"""Figure 1 (world diagram) as SVG, converted to vector PDF with rsvg-convert."""
import subprocess
import sys

out = sys.argv[1]
edge, accent, ink, quiet, tint, grid = "#888780", "#534AB7", "#2C2C2A", "#5F5E5A", "#E8E6E0", "#D3D1C7"
s = []
a = s.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 430" width="760" height="430" font-family="Inter, DejaVu Sans, sans-serif" font-size="13">')
a(f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{edge}"/></marker></defs>')
a('<rect x="0" y="0" width="760" height="430" fill="white"/>')


def bx(x, y, w, h, main=False):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{accent if main else "none"}" fill-opacity="{0.12 if main else 1}" stroke="{accent if main else edge}" stroke-width="{2 if main else 1.25}"/>')


def t(x, y, txt, size=11.5, color=ink, weight="400", anchor="start"):
    txt = txt.replace("&", "&amp;").replace("'", "&#8217;")
    a(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{txt}</text>')


t(24, 30, "Agents die; their pages outlive them", 15, ink, "600")
bx(24, 56, 200, 200)
t(124, 80, "8 slot machines", 13, ink, "600", "middle")
for i, line in enumerate(["5 arms each; one good arm", "good pays 90%, others 10%", "good arm moves about every", "2,000 steps"]):
    t(40, 102 + 16 * i, line)
goods = [2, 0, 4, 1, 3, 2, 0, 4]
for k in range(8):
    x0, y0 = 40 + (k % 4) * 44, 172 + (k // 4) * 40
    a(f'<rect x="{x0}" y="{y0}" width="36" height="28" rx="4" fill="none" stroke="{edge}" stroke-width="1"/>')
    a(f'<rect x="{x0 + 4 + goods[k] * 6}" y="{y0 + 6}" width="4" height="16" fill="{accent}"/>')
bx(280, 56, 200, 200)
t(380, 80, "An agent's visit", 13, ink, "600", "middle")
for i, line in enumerate(["8 pulls on a random machine", "sees its own pulls and wins", "reads all 16 entries for it", "may rewrite its own entry", "forgets the visit after"]):
    t(296, 102 + 16 * i, line)
for i, line in enumerate(["an author is paid c when", "a reader follows its entry", "and wins (credit)"]):
    t(296, 198 + 16 * i, line, color=quiet)
bx(536, 56, 200, 200)
t(636, 80, "The wiki: 16 pages", 13, ink, "600", "middle")
t(552, 102, "one page per agent,"); t(552, 118, "one entry per machine")
a(f'<rect x="{564 + 2 * 18}" y="132" width="18" height="72" fill="{tint}"/>')
for r in range(6):
    for c in range(8):
        a(f'<rect x="{564 + c * 18}" y="{132 + r * 12}" width="18" height="12" fill="none" stroke="{grid}" stroke-width="1"/>')
a(f'<rect x="564" y="144" width="144" height="12" fill="none" stroke="{accent}" stroke-width="2"/>')
t(552, 224, "shaded: machine being played", color=quiet); t(552, 240, "outlined: the reader's own page", color=quiet)
for d in ("M224 156H278", "M536 132H482", "M480 176H534"):
    a(f'<path d="{d}" fill="none" stroke="{edge}" stroke-width="1.25" marker-end="url(#ar)"/>')
t(252, 148, "visits", color=quiet, anchor="middle"); t(508, 124, "reads", color=quiet, anchor="middle"); t(508, 168, "writes", color=quiet, anchor="middle")
t(24, 284, "One life, about 300 updates", 11.5, quiet, "600")
lx, ly = [24, 206, 388, 570], 296
for x, main, name, l1, l2 in zip(lx, (False, True, False, False), ("Birth", "Juvenile period", "Adult", "Death"),
                                 ("random weights", "reads, may not write", "reads and writes", "new network"),
                                 ("takes over the page", "optional; can evolve", "paid when reused", "page kept or erased")):
    bx(x, ly, 166, 72, main)
    t(x + 83, ly + 24, name, 13, ink, "600", "middle"); t(x + 83, ly + 44, l1, anchor="middle"); t(x + 83, ly + 60, l2, anchor="middle")
for i in range(3):
    a(f'<path d="M{lx[i] + 166} {ly + 36}H{lx[i + 1] - 2}" fill="none" stroke="{edge}" stroke-width="1.25" marker-end="url(#ar)"/>')
a(f'<path d="M{lx[3] + 83} {ly + 72}V{ly + 96}H{lx[0] + 83}V{ly + 74}" fill="none" stroke="{edge}" stroke-width="1.25" marker-end="url(#ar)"/>')
t(380, ly + 116, "the heir takes the same slot and page; the record persists", color=quiet, anchor="middle")
a("</svg>")
open(out + ".svg", "w").write("\n".join(s))
subprocess.run(["rsvg-convert", "-f", "pdf", "-o", out + ".pdf", out + ".svg"], check=True)
subprocess.run(["rsvg-convert", "-f", "png", "-z", "2", "-o", out + ".png", out + ".svg"], check=True)
print("wrote", out)
