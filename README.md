# BSS sparsifier -> non-backtracking matrix

- `bss.py`       BSS (Twice-Ramanujan) barrier algorithm for any weighted graph (whitened vectors, O(n^3 + m) per step)
- `nb.py`        weighted/unweighted Hashimoto matrices; Ihara-Bass as a quadratic eigenproblem; weighted rational version; thesis eq. (2.13) map
- `checks.py`    numerical verification of the three conversions
- `exp_laplacian.py N1,N2 d1,d2`  kappa of BSS(K_n) vs random regular graphs
- `exp_nb.py n d`  NB spectra of BSS(K_n) (needs s_n{n}_d{d}.npy from exp_laplacian)
- `exp_sbm.py`   BSS on a 2-community graph
- `plot.py`      the figure

Run order: python3 checks.py; python3 exp_laplacian.py 100,200,400 3; python3 exp_nb.py 400 3; python3 exp_sbm.py; python3 plot.py
