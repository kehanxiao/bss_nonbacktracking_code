"""Non-backtracking (Hashimoto) matrices of a weighted graph, and the Ihara-Bass style conversions."""
import numpy as np


def directed_edges(edges):
    darts = []
    for (a, b) in edges:
        darts.append((a, b))
        darts.append((b, a))
    return darts


def nb_matrix(n, edges, weights=None):
    """B[(u->v),(v->x)] = w(v,x) if x != u.  weights=None gives the unweighted Hashimoto matrix."""
    darts = directed_edges(edges)
    M = len(darts)
    wdart = np.ones(M) if weights is None else np.repeat(np.asarray(weights, float), 2)
    out = [[] for _ in range(n)]  # out[v] = list of dart indices leaving v
    for k, (a, b) in enumerate(darts):
        out[a].append(k)
    B = np.zeros((M, M))
    for k, (u, v) in enumerate(darts):
        for j in out[v]:
            if darts[j][1] != u:
                B[k, j] = wdart[j]
    return B, darts


def adjacency_degree(n, edges, weights=None):
    A = np.zeros((n, n))
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
    for (a, b), x in zip(edges, w):
        A[a, b] += x
        A[b, a] += x
    Dcomb = np.diag((np.abs(A) > 0).sum(1).astype(float))  # combinatorial degree
    return A, Dcomb


def ihara_bass_companion(n, edges):
    """Unweighted: nontrivial eigenvalues of B are the 2n roots of det(mu^2 I - mu A + (D - I)) = 0,
    i.e. eigenvalues of the companion matrix [[A, -(D-I)], [I, 0]]."""
    A, D = adjacency_degree(n, edges)
    I = np.eye(n)
    C = np.block([[A, -(D - I)], [I, np.zeros((n, n))]])
    return np.linalg.eigvals(C)


def weighted_rational_matrix(mu, n, edges, w):
    """Weighted version (derived by eliminating darts): eigenvalue mu of B_w (mu^2 != w_e^2) iff
    det( I + Dt(mu) - At(mu) ) = 0 with At_vu = mu w_vu/(mu^2 - w_vu^2), Dt_vv = sum_u w_vu^2/(mu^2 - w_vu^2)."""
    M = np.eye(n, dtype=complex)
    for (a, b), x in zip(edges, w):
        den = mu * mu - x * x
        M[a, b] -= mu * x / den
        M[b, a] -= mu * x / den
        M[a, a] += x * x / den
        M[b, b] += x * x / den
    return M


def thesis_map(lams, dbar):
    """Thesis eq. (2.13): each adjacency eigenvalue lambda -> roots of mu^2 - lambda mu + (dbar-1) = 0."""
    out = []
    for lam in lams:
        out.extend(np.roots([1.0, -lam, dbar - 1.0]))
    return np.array(out)
