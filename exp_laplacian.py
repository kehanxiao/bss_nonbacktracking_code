"""Experiment 1: Laplacian quality of BSS(K_n) vs random regular graphs with the same edge budget."""
import json, sys, time
import numpy as np
import networkx as nx
from bss import bss, complete_graph


def ramanujan_kappa(D):
    r = 2 * np.sqrt(D - 1)
    return (D + r) / (D - r)


def bss_theorem_kappa(d):
    sd = np.sqrt(d)
    return (d + 1 + 2 * sd) / (d + 1 - 2 * sd)


def random_regular_kappa(n, D, reps=5, seed=0):
    out = []
    for r in range(reps):
        G = nx.random_regular_graph(D, n, seed=seed + r)
        lam = np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray().astype(float))
        out.append(lam[-1] / lam[1])
    return float(np.mean(out)), float(np.min(out))


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1].split(",")]
    ds = [int(x) for x in sys.argv[2].split(",")]
    rows = []
    for d in ds:
        for n in ns:
            t0 = time.time()
            E, w = complete_graph(n)
            s, A, info = bss(n, E, w, d)
            supp = int((s > 0).sum())
            D = 2 * d  # average degree of H is ~2d
            rr_mean, rr_min = random_regular_kappa(n, D)
            row = dict(n=n, d=d, avg_deg=2 * supp / n, kappa_bss_actual=info["kappa_actual"],
                       kappa_bss_theorem=bss_theorem_kappa(d), kappa_random_regular=rr_mean,
                       kappa_random_regular_best=rr_min, kappa_ramanujan_asymptotic=ramanujan_kappa(D),
                       secs=time.time() - t0)
            rows.append(row)
            print(json.dumps(row), flush=True)
            np.save(f"s_n{n}_d{d}.npy", s)
    json.dump(rows, open(f"exp_laplacian_{sys.argv[1]}_{sys.argv[2]}.json".replace(",", "-"), "w"), indent=1)
