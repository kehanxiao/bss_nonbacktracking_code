"""Batson-Spielman-Srivastava (Twice-Ramanujan) sparsifier, straight from the barrier argument.

Works for any connected weighted graph G given as (n, edges, weights).
Vectors are whitened: v_e = sqrt(w_e) * L_G^{+/2} b_e, restricted to 1^perp (dimension N = n-1),
so that sum_e v_e v_e^T = I_N.
"""
import numpy as np


def whitened_vectors(n, edges, w):
    m = len(edges)
    Bt = np.zeros((n, m))
    for k, (a, b) in enumerate(edges):
        Bt[a, k] = 1.0
        Bt[b, k] = -1.0
    L = Bt @ (w[:, None] * Bt.T)
    # orthonormal basis of 1^perp
    Q = np.linalg.svd(np.eye(n) - np.ones((n, n)) / n)[0][:, : n - 1]
    LQ = Q.T @ L @ Q
    ev, U = np.linalg.eigh(LQ)
    Wh = U @ np.diag(ev ** -0.5) @ U.T  # LQ^{-1/2}
    V = Wh @ Q.T @ Bt * np.sqrt(w)[None, :]
    return V, L


def bss(n, edges, w, d, rule="maxgap", verbose=False):
    """Run BSS with parameter d (output has <= ceil(d*(n-1)) distinct edges).
    Returns per-edge multipliers s (weights of H are s_e * w_e), and the final A.

    Quadratic forms v_e^T f(A) v_e are computed through n x n matrices
    R f(Lambda) R^T with R = Q Wh U, so the cost per step is O(n^3 + m), not O(n^2 m)."""
    m = len(edges)
    ea = np.array([e[0] for e in edges])
    eb = np.array([e[1] for e in edges])
    Bt = np.zeros((n, m))
    Bt[ea, np.arange(m)] = 1.0
    Bt[eb, np.arange(m)] = -1.0
    L = Bt @ (w[:, None] * Bt.T)
    Q = np.linalg.svd(np.eye(n) - np.ones((n, n)) / n)[0][:, : n - 1]
    ev, UL = np.linalg.eigh(Q.T @ L @ Q)
    Wh = UL @ np.diag(ev ** -0.5) @ UL.T
    QW = Q @ Wh  # v_e = sqrt(w_e) * QW^T b_e
    N = n - 1

    def qforms(R, f):
        M = (R * f) @ R.T
        return w * (M[ea, ea] + M[eb, eb] - 2 * M[ea, eb])

    sd = np.sqrt(d)
    dL, dU = 1.0, (sd + 1) / (sd - 1)
    eL, eU = 1 / sd, (sd - 1) / (d + sd)
    l, u = -N / eL, N / eU
    A = np.zeros((N, N))
    s = np.zeros(m)
    T = int(np.ceil(d * N))
    for q in range(T):
        lam, U = np.linalg.eigh(A)
        R = QW @ U
        un, ln = u + dU, l + dL
        Pu_old, Pu_new = np.sum(1 / (u - lam)), np.sum(1 / (un - lam))
        Pl_old, Pl_new = np.sum(1 / (lam - l)), np.sum(1 / (lam - ln))
        Ua = qforms(R, 1 / (un - lam)) + qforms(R, 1 / (un - lam) ** 2) / (Pu_old - Pu_new)
        La = qforms(R, 1 / (lam - ln) ** 2) / (Pl_new - Pl_old) - qforms(R, 1 / (lam - ln))
        ok = Ua <= La
        if not ok.any():
            raise RuntimeError(f"no admissible vector at step {q}")
        if rule == "maxgap":
            idx = np.argmax(np.where(ok, La - Ua, -np.inf))
        elif rule == "ratio":
            idx = np.argmax(np.where(ok, La / Ua, -np.inf))
        else:
            raise ValueError(rule)
        t = 2.0 / (Ua[idx] + La[idx])
        v = np.sqrt(w[idx]) * (QW[ea[idx]] - QW[eb[idx]])
        A += t * np.outer(v, v)
        s[idx] += t
        u, l = un, ln
        if verbose and q % 200 == 0:
            print(q, T, flush=True)
    lam = np.linalg.eigvalsh(A)
    return s, A, dict(u=u, l=l, kappa_barrier=u / l, kappa_actual=lam[-1] / lam[0], steps=T)


def bss_slow(n, edges, w, d, rule="maxgap", verbose=False):
    """Reference implementation with explicit whitened vectors (for cross-checking)."""
    V, L = whitened_vectors(n, edges, w)
    N, m = V.shape
    sd = np.sqrt(d)
    dL, dU = 1.0, (sd + 1) / (sd - 1)
    eL, eU = 1 / sd, (sd - 1) / (d + sd)
    l, u = -N / eL, N / eU
    A = np.zeros((N, N))
    s = np.zeros(m)
    T = int(np.ceil(d * N))
    for q in range(T):
        lam, U = np.linalg.eigh(A)
        un, ln = u + dU, l + dL
        Pu_old, Pu_new = np.sum(1 / (u - lam)), np.sum(1 / (un - lam))
        Pl_old, Pl_new = np.sum(1 / (lam - l)), np.sum(1 / (lam - ln))
        P = (U.T @ V) ** 2  # N x m, squared coordinates in eigenbasis
        Ua = (1 / (un - lam)) @ P + ((1 / (un - lam) ** 2) @ P) / (Pu_old - Pu_new)
        La = ((1 / (lam - ln) ** 2) @ P) / (Pl_new - Pl_old) - (1 / (lam - ln)) @ P
        ok = Ua <= La
        if not ok.any():
            raise RuntimeError(f"no admissible vector at step {q}")
        if rule == "maxgap":
            idx = np.argmax(np.where(ok, La - Ua, -np.inf))
        elif rule == "ratio":
            idx = np.argmax(np.where(ok, La / Ua, -np.inf))
        else:
            raise ValueError(rule)
        t = 2.0 / (Ua[idx] + La[idx])
        A += t * np.outer(V[:, idx], V[:, idx])
        s[idx] += t
        u, l = un, ln
        if verbose and q % 100 == 0:
            print(q, T, idx)
    lam = np.linalg.eigvalsh(A)
    return s, A, dict(u=u, l=l, kappa_barrier=u / l, kappa_actual=lam[-1] / lam[0], steps=T)


def complete_graph(n):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return edges, np.ones(len(edges))
