import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})

nbk = np.load("nb_n400_d3.npz")
sbm = np.load("nb_sbm.npz")
fig, ax = plt.subplots(2, 2, figsize=(11, 10.5))
TOL = 0.02


def plane(a, ev, r, title, color, outlier_color=None, note=None):
    t = np.linspace(0, 2 * np.pi, 400)
    a.plot(r * np.cos(t), r * np.sin(t), color=INK2, lw=1, ls="--", zorder=1)
    ev = ev[np.abs(np.abs(ev) - 1) > 1e-6]  # drop the trivial +-1 eigenvalues of the (1-u^2)^(m-n) factor
    inside = np.abs(ev) <= r + TOL
    a.scatter(ev.real[inside], ev.imag[inside], s=8, color=color, alpha=0.6, lw=0, zorder=2)
    if outlier_color is not None:
        a.scatter(ev.real[~inside], ev.imag[~inside], s=22, color=outlier_color, lw=0.6,
                  edgecolor=SURF, zorder=3, label=f"outside the circle by > {TOL} ({(~inside).sum()})")
        a.legend(loc="upper left", frameon=False, fontsize=9, borderaxespad=0.1)
    a.set_aspect("equal")
    lim = 2.95
    a.set_ylim(-lim, lim)
    if max(abs(ev.real).max(), r) < lim: a.set_xlim(-lim, lim)
    a.set_title(title, loc="left", fontsize=10.5, color=INK)
    a.axhline(0, color=GRID, lw=0.8, zorder=0); a.axvline(0, color=GRID, lw=0.8, zorder=0)
    a.set_xlabel("Re μ"); a.set_ylabel("Im μ")
    if note:
        a.text(0.99, 0.01, note, transform=a.transAxes, ha="right", va="bottom", fontsize=9, color=INK2)


# (a) BSS sparsifier of K_400: unweighted support
eB = nbk["eB"]; rho = float(nbk["rho"])
nontriv = eB[np.abs(eB - rho) > 1e-6]
plane(ax[0, 0], nontriv, np.sqrt(rho),
      "(a) BSS sparsifier of K₄₀₀ — NB spectrum of its support\n(394 of 400 vertices have degree 6, the rest 5; weights within 1.75×)",
      BLUE, ORANGE, note=f"dashed: |μ| = √ρ = {np.sqrt(rho):.3f}\nmax nontrivial |μ| = {np.abs(nontriv).max():.3f}")

# (b) random 6-regular graph, same n and edge count
eR = nbk["eR"]; D = int(nbk["D"])
nR = eR[np.abs(eR - (D - 1)) > 1e-6]
plane(ax[0, 1], nR, np.sqrt(D - 1),
      "(b) Random 6-regular graph on 400 vertices — NB spectrum\n(same number of edges; trivial eigenvalue 5 not shown)",
      BLUE, ORANGE, note=f"dashed: |μ| = √5 = {np.sqrt(5):.3f}\nmax nontrivial |μ| = {np.abs(nR).max():.3f}")

# (c) SBM
ev = sbm["ev"]; rs = float(sbm["rho"])
nS = ev[np.abs(ev - rs) > 1e-6]
plane(ax[1, 0], nS, np.sqrt(rs),
      "(c) BSS sparsifier of a 2-community graph G (n=200)\nH approximates G well (κ=4.43), yet NB(H) has an outlier",
      BLUE, ORANGE, note="outlier ≈ 4.11 = G's community cut,\ninherited by H, not a sparsification error")

# (d) kappa vs n
rows = []
for f in glob.glob("exp_laplacian_*.json"):
    rows += json.load(open(f))
rows = sorted([r for r in rows if r["d"] == 3], key=lambda r: r["n"])
ns = [r["n"] for r in rows]
a = ax[1, 1]
a.plot(ns, [r["kappa_bss_actual"] for r in rows], "-o", color=BLUE, lw=2, ms=6, label="BSS output (actual)")
a.plot(ns, [r["kappa_random_regular"] for r in rows], "-o", color=AQUA, lw=2, ms=6, label="random 6-regular (mean of 5)")
a.axhline(rows[0]["kappa_ramanujan_asymptotic"], color=INK2, lw=1, ls="--")
a.text(ns[0], rows[0]["kappa_ramanujan_asymptotic"] + 0.15, "Ramanujan bound, n→∞ (6.85)", fontsize=9, color=INK2)
a.axhline(rows[0]["kappa_bss_theorem"], color=ORANGE, lw=1, ls="--")
a.text(ns[0], rows[0]["kappa_bss_theorem"] - 0.6, "BSS theorem guarantee (13.93)", fontsize=9, color=INK2)
a.set_xscale("log", base=2); a.set_xticks(ns); a.set_xticklabels([str(x) for x in ns])
a.set_ylim(0, 15); a.set_xlabel("n (vertices of K_n)"); a.set_ylabel("condition number κ (lower is better)")
a.set_title("(d) Approximation quality of K_n with avg degree ≈ 6", loc="left", fontsize=10.5)
a.grid(axis="y", color=GRID, lw=0.6); a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=False, fontsize=9)
for s in ["top", "right"]:
    for b in ax.flat:
        b.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("bss_nonbacktracking.png", dpi=160)
print("saved", ns)
