"""Sanity checks of the conversions on a small irregular weighted graph."""
import numpy as np, networkx as nx
from nb import *
rng = np.random.default_rng(1)
G = nx.gnp_random_graph(30, 0.2, seed=3)
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
G = nx.convert_node_labels_to_integers(G)
n = G.number_of_nodes(); E = list(G.edges())
# 1) unweighted Ihara-Bass: eig(B) = eig(companion) plus +-1 with multiplicity m-n
B, _ = nb_matrix(n, E)
eb = np.linalg.eigvals(B); ec = ihara_bass_companion(n, E)
m = len(E)
extra = np.concatenate([np.ones(m - n), -np.ones(m - n)])
a = np.sort_complex(np.round(eb, 8)); b = np.sort_complex(np.round(np.concatenate([ec, extra]), 8))
print("unweighted Ihara-Bass max mismatch:", np.abs(a - b).max(), "(n,m)=", n, m)
# 2) weighted: eigenvalues of B_w make det(I + Dt - At) vanish
w = rng.uniform(0.2, 3.0, size=m)
Bw, _ = nb_matrix(n, E, w)
ew = np.linalg.eigvals(Bw)
ratios = []
for mu in ew[np.argsort(-np.abs(ew))][:40]:
    if np.min(np.abs(mu**2 - w**2)) < 1e-3: continue
    sv = np.linalg.svd(weighted_rational_matrix(mu, n, E, w), compute_uv=False)
    ratios.append(sv[-1] / sv[0])
mu0 = 1.7 + 0.9j
sv0 = np.linalg.svd(weighted_rational_matrix(mu0, n, E, w), compute_uv=False)
print("weighted: max smin/smax at eigenvalues:", max(ratios), " at a random point:", sv0[-1] / sv0[0])
# 3) is NB spectrum a function of the weighted Laplacian spectrum? Build two weightings with identical
#    weighted Laplacian (same graph, w vs. w scaled) -> trivial; instead show regular-graph check:
H = nx.random_regular_graph(4, 20, seed=0); EH = list(H.edges())
BH, _ = nb_matrix(20, EH); eh = np.linalg.eigvals(BH)
lam = np.linalg.eigvalsh(nx.adjacency_matrix(H).toarray().astype(float))
tm = thesis_map(lam, 4)
print("regular graph: thesis map reproduces nontrivial NB eigenvalues:",
      all(np.min(np.abs(eh - z)) < 1e-6 for z in tm))
