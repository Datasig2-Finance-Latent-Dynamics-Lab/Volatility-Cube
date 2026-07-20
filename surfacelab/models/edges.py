"""
Cross-asset graph edges: builders + learners (ported faithfully from dgraph).

An edge (i → j) carries a `DeltaEdge(precision, matrix)`.  In the closed-form/scipy
graph loss it penalises the *increment* residual

    r_{i,j} = (c_i - prior_i) - M_{ij} (c_j - prior_j)        (M = I when matrix is None)

so coupled assets must move consistently relative to their own priors, optionally through
a learned linear map M.  Nodes are keyed (asset_id, maturity_index); edges connect assets
that share a maturity index.

Builders (prior structure, M = I):
  * uniform_edges  — all same-maturity pairs, one scalar weight
  * tiered_edges   — SPY↔stock asymmetry (SPY leads stocks)
  * factored_edges — tiered × exp(-λ·Δmaturity) decay across maturities

Learners (estimate edges from a per-node coefficient history, i.e. the Δθ sequences):
  * learn_edges_matrix   — full M_{i→j} by OLS:  Δc_j ≈ M Δc_i
  * learn_edges_diagonal — diagonal M (per-coefficient scalar)
  * learn_edges_scalar   — scalar precision = |corr| of the first-coefficient increments

This mirrors `learn_surface_edges`, `learn_surface_edges_diagonal`, and
`learn_scalar_pca_edges` in `dgraph/examples/vol_surface/experiments/main.py` plus the
`build_tiered_*` / `build_factored_*` builders.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np


@dataclass
class DeltaEdge:
    """Directed edge: precision (scalar or matrix) + optional linear map M."""

    precision: float | np.ndarray = 1.0
    matrix: np.ndarray | None = None


# ════════════════════════════════════════════════════════════════════════════
# Prior-structure builders (M = I)
# ════════════════════════════════════════════════════════════════════════════
def _by_maturity(nodes):
    by_mat: dict = {}
    for nd in nodes:
        by_mat.setdefault(nd[1], []).append(nd)
    return by_mat


def _pair_weights(ni, nj, asset_names, spy_to_stock, stock_to_spy, stock_to_stock, spy_name):
    """SPY-leads-stocks asymmetric (w_{i→j}, w_{j→i}) for a same-maturity pair.

    Defaults are caller-supplied so each builder keeps its own weight scale."""
    u1 = asset_names[ni[0]] if ni[0] < len(asset_names) else str(ni[0])
    u2 = asset_names[nj[0]] if nj[0] < len(asset_names) else str(nj[0])
    if u1 == spy_name:
        return spy_to_stock, stock_to_spy
    if u2 == spy_name:
        return stock_to_spy, spy_to_stock
    return stock_to_stock, stock_to_stock


def uniform_edges(nodes, weight: float = 1.0) -> dict:
    """All cross-asset pairs sharing a maturity, equal scalar precision."""
    edges: dict = {}
    for mat_nodes in _by_maturity(nodes).values():
        for ni, nj in combinations(mat_nodes, 2):
            edges[(ni, nj)] = DeltaEdge(weight)
            edges[(nj, ni)] = DeltaEdge(weight)
    return edges


def tiered_edges(nodes, asset_names, *, spy_to_stock=0.8, stock_to_spy=0.2,
                 stock_to_stock=0.4, spy_name="SPY") -> dict:
    """Same-maturity pairs with SPY-leads-stocks asymmetry (ported from build_tiered_*)."""
    edges: dict = {}
    for mat_nodes in _by_maturity(nodes).values():
        for ni, nj in combinations(mat_nodes, 2):
            w12, w21 = _pair_weights(ni, nj, asset_names, spy_to_stock,
                                     stock_to_spy, stock_to_stock, spy_name)
            edges[(ni, nj)] = DeltaEdge(w12)
            edges[(nj, ni)] = DeltaEdge(w21)
    return edges


def factored_edges(nodes, asset_names, mat_values, *, lambda_f=1.0, min_precision=0.2,
                   spy_to_stock=0.5, stock_to_spy=0.1, stock_to_stock=0.2,
                   spy_name="SPY") -> dict:
    """Cross-asset AND cross-maturity edges with exp(-λ·Δmaturity) decay."""
    edges: dict = {}
    for ni in nodes:
        for nj in nodes:
            if ni[0] >= nj[0] or ni == nj:
                continue
            dt = abs(float(mat_values[ni[1]]) - float(mat_values[nj[1]]))
            f = float(np.exp(-lambda_f * dt))
            if f < min_precision:
                continue
            w12, w21 = _pair_weights(ni, nj, asset_names, spy_to_stock,
                                     stock_to_spy, stock_to_stock, spy_name)
            edges[(ni, nj)] = DeltaEdge(w12 * f)
            edges[(nj, ni)] = DeltaEdge(w21 * f)
    return edges


# ════════════════════════════════════════════════════════════════════════════
# Learners — estimate edges from a per-node coefficient history
# `history`: dict {node_key: list[coeff_vector]} in chronological order
# Pairs are formed within a shared maturity index.
# ════════════════════════════════════════════════════════════════════════════
def _delta_seqs(history) -> dict:
    """Day-aligned increments.  history: {node: (days(T,), coeffs(T,p))} →
    {node: (ddays(M,), deltas(M,p))} where each delta is between *adjacent trading days*
    (days differing by 1) and tagged by its end-day, so increments from different nodes can
    be paired on a common calendar (see `_aligned`)."""
    out = {}
    for nd, val in history.items():
        days, coeffs = val
        days = np.asarray(days); coeffs = np.asarray(coeffs, dtype=float)
        if len(days) < 2:
            continue
        i = np.where(np.diff(days) == 1)[0]       # consecutive-day pairs only
        if len(i) == 0:
            continue
        out[nd] = (days[i + 1], coeffs[i + 1] - coeffs[i])
    return out


def _aligned(seqs, ni, nj):
    """Return (Di, Dj) increment arrays for nodes ni, nj restricted to their COMMON end-days
    (so the rows correspond to the same calendar dates), or (None, None) if too few overlap."""
    da, Da = seqs[ni]; db, Db = seqs[nj]
    common = np.intersect1d(da, db)
    if len(common) < 2:
        return None, None
    ia = {int(d): k for k, d in enumerate(da)}
    ib = {int(d): k for k, d in enumerate(db)}
    ra = [ia[int(d)] for d in common]; rb = [ib[int(d)] for d in common]
    return Da[ra], Db[rb]


def _aligned_pairs(deltas):
    """Yield (ni, nj, Di, Dj) for every same-maturity node pair with enough common-day
    increments — the shared scaffold of the matrix/diagonal/scalar/pca learners."""
    for ni, nj in combinations(sorted(deltas), 2):
        if ni[1] != nj[1]:                        # couple within a maturity index
            continue
        Di, Dj = _aligned(deltas, ni, nj)         # common-day rows only
        if Di is None:
            continue
        yield ni, nj, Di, Dj


def learn_edges_matrix(history, precision: float = 1.0) -> dict:
    """Full linear map per pair:  Δc_j ≈ M Δc_i  via OLS (ported from learn_surface_edges)."""
    deltas = _delta_seqs(history)
    edges: dict = {}
    for ni, nj, Di, Dj in _aligned_pairs(deltas):
        # _solve penalises ‖r_i − M·r_j‖² for edge (ni,nj), so M must map node j → node i.
        # lstsq(Di,Dj) gives the i→j map; the j→i map is lstsq(Dj,Di).  Hence edge (ni,nj)
        # carries Mji.T (j→i) and edge (nj,ni) carries Mij.T (i→j).
        Mij, *_ = np.linalg.lstsq(Di, Dj, rcond=None)   # Di @ Mij = Dj  (i→j)
        Mji, *_ = np.linalg.lstsq(Dj, Di, rcond=None)   # Dj @ Mji = Di  (j→i)
        edges[(ni, nj)] = DeltaEdge(precision, Mji.T)
        edges[(nj, ni)] = DeltaEdge(precision, Mij.T)
    return edges


def learn_edges_diagonal(history, precision: float = 1.0) -> dict:
    """Diagonal M: per-coefficient no-intercept OLS (ported from *_diagonal)."""
    deltas = _delta_seqs(history)
    edges: dict = {}
    for ni, nj, Di, Dj in _aligned_pairs(deltas):
        di, dj = (Di * Di).sum(0), (Dj * Dj).sum(0)
        mij = np.where(di > 0, (Di * Dj).sum(0) / di, 0.0)
        mji = np.where(dj > 0, (Dj * Di).sum(0) / dj, 0.0)
        edges[(ni, nj)] = DeltaEdge(precision, np.diag(mij))
        edges[(nj, ni)] = DeltaEdge(precision, np.diag(mji))
    return edges


def learn_edges_scalar(history) -> dict:
    """Scalar precision = |corr| of the (day-aligned) first-coefficient increments."""
    deltas = _delta_seqs(history)
    edges: dict = {}
    for ni, nj, Di, Dj in _aligned_pairs(deltas):
        d1, d2 = Di[:, 0], Dj[:, 0]
        if d1.std() < 1e-12 or d2.std() < 1e-12:
            continue
        w = abs(float(np.corrcoef(d1, d2)[0, 1]))
        edges[(ni, nj)] = DeltaEdge(w)
        edges[(nj, ni)] = DeltaEdge(w)
    return edges


def learn_market_edges(history, asset_names, spy_name: str = "SPY",
                       weight_by_corr: bool = True) -> dict:
    """Single-market-factor coupling: pull each asset's LEVEL increment toward β·(SPY's).

    Motivated by the cross-asset diagnostic — most co-movement is one market factor (SPY).
    For each (stock, maturity) we regress the stock's level increment on SPY's at the same
    maturity to get β, and add a *directed* edge stock→SPY whose penalty acts only on the
    level direction:  w·(L·Δc_stock − β·L·Δc_spy)²,  with L the unit "level" vector (a uniform
    coefficient shift).  In the DeltaEdge form ‖r_i − M r_j‖²_P this is  P = w·LᵀL (rank-1),
    M = β·I.  w = |corr| (data-grounded) so weakly-coupled names barely move.  Asymmetric:
    stocks follow SPY, not vice-versa.  One β per (asset, maturity) — cheap and estimable."""
    deltas = _delta_seqs(history)
    if not deltas or spy_name not in asset_names:
        return {}
    p = next(iter(deltas.values()))[1].shape[1]
    spy_id = asset_names.index(spy_name)
    L = np.ones(p) / np.sqrt(p)          # unit level direction (uniform coeff shift)
    LtL = np.outer(L, L)                 # rank-1 precision: penalise only the level component
    I = np.eye(p)
    by_mat: dict = {}
    for nd in deltas:
        by_mat.setdefault(nd[1], []).append(nd)
    edges: dict = {}
    for m, nds in by_mat.items():
        spy_nd = (spy_id, m)
        if spy_nd not in deltas:
            continue
        for nd in nds:
            if nd[0] == spy_id:
                continue
            Da, Ds = _aligned(deltas, nd, spy_nd)   # stock & SPY on common days
            if Da is None:
                continue
            l_a, l_s = Da @ L, Ds @ L
            if l_s.std() < 1e-12 or l_a.std() < 1e-12:
                continue
            beta = float(np.cov(l_a, l_s)[0, 1] / (l_s.var() + 1e-12))
            w = abs(float(np.corrcoef(l_a, l_s)[0, 1])) if weight_by_corr else 1.0
            edges[(nd, spy_nd)] = DeltaEdge(w * LtL, beta * I)
    return edges


def learn_pca_edges(history, n_modes: int = 2, precision: float = 1.0) -> dict:
    """Low-rank `learn_edges_matrix`: couple coefficient increments through a rank-`n_modes`
    map learned in the PCA-mode space of the pooled increments.

    The full 13×13 OLS map is hopeless to estimate from a short increment history (169 params,
    ~hundreds of samples) — it overfits and washes out.  Projecting onto the top `n_modes` modes
    (mode 1 ≈ market level, mode 2 ≈ residual/skew structure) cuts it to n_modes² parameters,
    then maps back: M = V·M_scores·Vᵀ (rank ≤ n_modes).  Same DeltaEdge contract as the other
    learners, so `_solve` is unchanged."""
    deltas = _delta_seqs(history)
    if not deltas:
        return {}
    p = next(iter(deltas.values()))[1].shape[1]
    pool = np.vstack([d for (_, d) in deltas.values()])
    pool = pool - pool.mean(0)
    try:
        _, _, Vt = np.linalg.svd(pool, full_matrices=False)
    except np.linalg.LinAlgError:
        return {}
    k = min(n_modes, Vt.shape[0])
    V = Vt[:k].T                                   # (p, k) top PCA modes of the increments
    edges: dict = {}
    for ni, nj, Di, Dj in _aligned_pairs(deltas):
        Si, Sj = Di @ V, Dj @ V                     # (n, k) mode scores
        # same j→i direction convention as learn_edges_matrix; mapped back via V (rank ≤ k)
        Mij, *_ = np.linalg.lstsq(Si, Sj, rcond=None)   # i→j in mode space
        Mji, *_ = np.linalg.lstsq(Sj, Si, rcond=None)   # j→i in mode space
        edges[(ni, nj)] = DeltaEdge(precision, V @ Mji.T @ V.T)
        edges[(nj, ni)] = DeltaEdge(precision, V @ Mij.T @ V.T)
    return edges


EDGE_LEARNERS = {
    "matrix": learn_edges_matrix,
    "diagonal": learn_edges_diagonal,
    "scalar": learn_edges_scalar,
    "pca": learn_pca_edges,
}
