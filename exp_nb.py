"""Experiment 2: non-backtracking spectrum of the BSS sparsifier of K_n vs a random regular graph."""
import json, sys
import numpy as np, networkx as nx
from bss import complete_graph
from nb import *

n, d = int(sys.argv[1]), int(sys.argv[2])
E, _ = complete_graph(n)
s = np.load(f"s_n{n}_d{d}.npy")
keep = np.where(s > 0)[0]
EH = [E[k] for k in keep]
wH = s[keep] / n                       # weights of H so that L_H ~ L_{K_n} = n*Pi  ->  normalised below
m = len(EH)
A_w, Dc = adjacency_degree(n, EH, wH)
L_w = np.diag(A_w.sum(1)) - A_w
lamL = np.linalg.eigvalsh(L_w)
kappa = lamL[-1] / lamL[1]
degs = np.diag(Dc)
wdeg = A_w.sum(1)
# normalise weights so the average weighted degree equals the average combinatorial degree
c = degs.mean() / wdeg.mean()
wH = wH * c
dbar = 2 * m / n

# unweighted support
B, _ = nb_matrix(n, EH)
eB = np.linalg.eigvals(B)
rho = np.max(np.abs(eB))
nontriv = eB[np.abs(eB - rho) > 1e-6]
r_unw = np.max(np.abs(nontriv))

# weighted
Bw, _ = nb_matrix(n, EH, wH)
eBw = np.linalg.eigvals(Bw)
rho_w = np.max(np.abs(eBw))
B2, _ = nb_matrix(n, EH, wH ** 2)
rho_w2 = np.max(np.abs(np.linalg.eigvals(B2)))
nontriv_w = eBw[np.abs(eBw - rho_w) > 1e-6]
r_w = np.max(np.abs(nontriv_w))

# thesis per-eigenvalue map applied to H's (weighted) Laplacian: lambda_A := dbar - lambda_L
lamA_from_L = dbar - lamL * c
tm = thesis_map(lamA_from_L, dbar)

# baseline random regular graph with the same number of edges
D = int(round(dbar)); G = nx.random_regular_graph(D, n, seed=0)
ER = list(G.edges())
BR, _ = nb_matrix(n, ER)
eR = np.linalg.eigvals(BR)
nontrivR = eR[np.abs(eR - (D - 1)) > 1e-6]
lamR = np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray().astype(float))

res = dict(n=n, d=d, m=m, avg_deg=dbar, deg_min=int(degs.min()), deg_max=int(degs.max()),
           weight_ratio_max_min=float(wH.max() / wH.min()),
           kappa_H=float(kappa), kappa_random_regular=float(lamR[-1] / lamR[1]),
           unweighted=dict(rho=float(rho), sqrt_rho=float(np.sqrt(rho)), max_nontrivial=float(r_unw),
                           frac_outside_sqrt_rho=float(np.mean(np.abs(nontriv) > np.sqrt(rho) + 1e-9))),
           weighted=dict(rho=float(rho_w), sqrt_rho_w2=float(np.sqrt(rho_w2)), max_nontrivial=float(r_w),
                         frac_outside=float(np.mean(np.abs(nontriv_w) > np.sqrt(rho_w2) + 1e-9))),
           random_regular=dict(D=D, sqrt_Dm1=float(np.sqrt(D - 1)), max_nontrivial=float(np.max(np.abs(nontrivR)))),
           thesis_map=dict(max_abs=float(np.max(np.abs(tm))),
                           frac_on_circle=float(np.mean(np.abs(np.abs(tm) - np.sqrt(dbar - 1)) < 1e-6))))
print(json.dumps(res, indent=1))
np.savez(f"nb_n{n}_d{d}.npz", eB=eB, eBw=eBw, eR=eR, tm=tm, rho=rho, rho_w2=rho_w2, D=D, dbar=dbar,
         degs=degs, wH=wH, lamL=lamL * c, lamR=lamR)
