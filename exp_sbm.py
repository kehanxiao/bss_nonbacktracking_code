"""Experiment 3: BSS on a non-complete graph (2-block SBM). NB spectrum of H reflects G's own structure."""
import json, numpy as np, networkx as nx
from bss import bss
from nb import *
n = 200
G = nx.stochastic_block_model([100, 100], [[0.5, 0.05], [0.05, 0.5]], seed=2)
E = list(G.edges()); w = np.ones(len(E))
s, A, info = bss(n, E, w, 3)
keep = np.where(s > 0)[0]; EH = [E[k] for k in keep]
B, _ = nb_matrix(n, EH); ev = np.linalg.eigvals(B)
rho = np.abs(ev).max()
order = np.argsort(-np.abs(ev))
top = ev[order[:4]]
# how many H-edges cross the cut, vs. G
cross = lambda ED: sum((a < 100) != (b < 100) for a, b in ED)
res = dict(kappa_relative_to_G=info["kappa_actual"], m_H=len(EH), sqrt_rho=float(np.sqrt(rho)),
           top4_abs=[float(abs(z)) for z in top], top4=[str(np.round(z, 3)) for z in top],
           cross_frac_G=cross(E) / len(E), cross_frac_H=cross(EH) / len(EH))
print(json.dumps(res, indent=1))
np.savez("nb_sbm.npz", ev=ev, rho=rho)
