"""
Temporally- and graph-regularised parametric fits.

These wrap a per-day representation with two extra loss terms:

  * **temporal** — pull today's coefficients toward the carried prior (yesterday's fit),
  * **graph**    — couple coupled assets' *increments* from their priors.

`RegularizedBSpline` exploits that B-spline IV coefficients are *linear*: the combined
loss is quadratic, so one block linear solve `H c = b` gives the exact optimum (the
headline efficiency trick, ported from `dgraph/.../time_stepping/updater.py`).  The
graph term produces off-diagonal blocks, so temporal vs temporal+graph genuinely differ.

`RegularizedParametric` handles the non-linear SVI / SSVI fits via scipy on the stacked
parameter vector with the same two regulariser terms.

The carried prior is held inside the model and advanced by the sequential harness:
`seed_prior` fits a 'perfect' prior on a full day; `predict` fits today toward that prior;
`step` does the same and then replaces the prior with today's fit.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from surfacelab.core.model import SurfaceModel
from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction, query_as_quotes
from surfacelab.models import edges as edges_mod
from surfacelab.models.parametric import representations as rep
from surfacelab.models.parametric.base import _interp_smiles


# Shared constants
_DEFAULT_IV = 0.2            # flat-vol fallback when a node has no data/prior
_IV_FLOOR = 1e-8             # IV positivity floor
_CLAMP_LO_FRAC = 0.7         # variation-diminishing coeff clamp: lower edge = frac·observed min
_CLAMP_HI_FRAC = 1.5         # ...upper edge = frac·observed max (an under-constrained wing margin)
_FLAT_ANCHOR_W = 1e-2        # weak flat-mean anchor weight in the maturity-smoothness solve
_NPARAMS = {"svi": 5, "ssvi": 6}                                  # params per node by kind
_LEARNED_KINDS = {"learned": "matrix", "learned_diag": "diagonal",
                  "learned_scalar": "scalar"}                     # edges-spec → EDGE_LEARNERS key


def _default_suffix(lambda_temporal, lambda_graph) -> str:
    """Auto model-name suffix from the active regularisers."""
    if lambda_temporal == 0:
        return "data"
    return "temporal_graph" if lambda_graph > 0 else "temporal"


def _resolve_edges_spec(spec, *, nodes, asset_names, mat_grid, edge_weight,
                        history_fn, allow_market_pca):
    """Shared edge-spec dispatch for both regularised models.

    `nodes` is precomputed by the caller; `history_fn` is a thunk so the (expensive) coeff/
    param history is built only for the learner specs.  `allow_market_pca` gates the
    market/pca learners (B-spline only)."""
    if spec == "uniform":
        return edges_mod.uniform_edges(nodes, edge_weight)
    if spec == "tiered":
        return edges_mod.tiered_edges(nodes, asset_names)
    if spec == "factored":
        return edges_mod.factored_edges(nodes, asset_names, mat_grid)
    if allow_market_pca and spec == "market":   # single-market-factor (SPY) level coupling
        return edges_mod.learn_market_edges(history_fn(), asset_names)
    if allow_market_pca and spec == "pca":      # low-rank coupling in the increments' PCA space
        return edges_mod.learn_pca_edges(history_fn())
    if spec.startswith("learned"):
        kind = _LEARNED_KINDS.get(spec, "matrix")
        return edges_mod.EDGE_LEARNERS[kind](history_fn())
    raise ValueError(f"unknown edges spec {spec!r}")


# ════════════════════════════════════════════════════════════════════════════
# Shared maturity-grid helper
# ════════════════════════════════════════════════════════════════════════════
def _maturity_grid(data, max_grid: int = 16) -> np.ndarray:
    qf = data.query_feats
    T = qf[:, :, 1][qf[:, :, 1] > 0]
    uniq = np.unique(np.round(T, 6))
    if len(uniq) <= max_grid:
        return uniq
    return np.quantile(T, np.linspace(0, 1, max_grid))


def _snap(T, grid) -> np.ndarray:
    """Index of the nearest grid maturity for each T."""
    T = np.atleast_1d(np.asarray(T, float))
    return np.argmin(np.abs(T[:, None] - grid[None, :]), axis=1)


# ════════════════════════════════════════════════════════════════════════════
# Closed-form regularised B-spline
# ════════════════════════════════════════════════════════════════════════════
class RegularizedBSpline(SurfaceModel):
    def __init__(self, lambda_data=1.0, lambda_temporal=0.0, lambda_graph=0.0,
                 lambda_maturity=0.0, lambda_rough=0.1, n_interior=9, degree=3,
                 edge_weight=1.0, edges="uniform", n_history=250, name=None,
                 graph_fallback=False):
        self.lambda_data = lambda_data
        # graph_fallback: give every present asset a prior on the FULL maturity grid, filling
        # data-less maturities from the nearest observed maturity of the same asset. This is the
        # cross-maturity anchor — it lets a cross-asset edge FIRE on maturities a sparse target
        # never quoted (the edge needs a baseline prior p_i on both endpoints; without this fill
        # those nodes have no prior and the edge is silently skipped, so peers can only refine
        # maturities the target already observes, never fill in the ones it doesn't). Default off.
        self.graph_fallback = graph_fallback
        self.lambda_temporal = lambda_temporal
        self.lambda_graph = lambda_graph
        # within-asset maturity-smoothness: couples adjacent same-underlying maturity
        # nodes' *states* (not increments); 0 = off.  See `_solve`.
        self.lambda_maturity = lambda_maturity
        # P-spline curvature penalty λ‖D2 c‖²: tames the steep-wing blow-up of a fixed-knot
        # fit on sparse data; fixed (data-independent), so it's negligible once a node is well
        # sampled → the fit still converges to the interpolant as observations grow.
        self.lambda_rough = lambda_rough          # default tuned so 13 coeffs stay tame at low n
        self.n_interior = n_interior              # bank convention: 9 interior knots
        self.degree = degree
        self._D2D2 = None                         # D2ᵀD2 curvature matrix (built in train)
        self.edge_weight = edge_weight
        # edges: "uniform" | "tiered" | "factored" | "learned"[=matrix] |
        #        "learned_diag" | "learned_scalar" | a prebuilt {(ni,nj): DeltaEdge} dict
        self.edges_spec = edges
        self.n_history = n_history
        if name:
            self.name = name
        else:
            suffix = _default_suffix(lambda_temporal, lambda_graph)
            self.name = f"bspline_{suffix}" + ("_interp" if lambda_maturity > 0 else "")
        self.knots = None
        self.mat_grid = None
        self.asset_names: list | None = None
        self._prior: dict | None = None     # node -> coeffs
        self._nodes: list | None = None
        self._edges: dict | None = None     # learned/built once in train()
        self._tmpl: rep.BSplineState | None = None

    # ── setup ──────────────────────────────────────────────────────────────────
    def train(self, data, *, saved: bool = False, force: bool = False) -> None:
        self.data_tag = data.meta.get("dgp", self.data_tag)
        qf = data.query_feats
        k = qf[:, :, 0][qf[:, :, 1] > 0]
        k_lo, k_hi = np.percentile(k, [1, 99])
        self.knots = rep.BSplineState.make_knots(k_lo, k_hi, self.n_interior, self.degree)
        self.mat_grid = _maturity_grid(data)
        self.asset_names = data.meta.get("asset_names",
                                         [f"asset_{i}" for i in range(data.n_assets)])
        p = len(self.knots) - self.degree - 1
        self._tmpl = rep.BSplineState(self.knots, self.degree, 1.0, np.zeros(p))
        D2 = rep.second_diff_matrix(p)
        self._D2D2 = D2.T @ D2                     # (p, p) curvature operator for the block solve
        if self.lambda_graph > 0:
            self._edges = self._resolve_edges(data)

    # ── edge resolution (builders + learners) ───────────────────────────────────
    def _all_nodes(self, data) -> list:
        valid = data.query_feats[:, :, 1] > 0
        aids = data.asset_ids[valid]
        mi = _snap(data.query_feats[:, :, 1][valid], self.mat_grid)
        return sorted({(int(a), int(m)) for a, m in zip(aids, mi)})

    def _coeff_history(self, data) -> dict:
        """Per-node fitted-coeff history, **day-indexed**: node -> (days, coeffs).

        days[k] is the training-day index of coeffs[k], so the edge learners can compute
        increments on a common calendar and pair assets on the SAME days.  (Storing bare
        ordered lists silently misaligned cross-asset increments — every pair then correlated
        mismatched dates, which is why the learned edges came out ≈0.)"""
        train_idx = data.train_idx()
        if self.n_history and len(train_idx) > self.n_history:
            train_idx = train_idx[-self.n_history:]
        days: dict = {}
        coeffs: dict = {}
        for t in train_idx:
            m = data.valid_mask(t)
            aids, kk = data.asset_ids[t, m], data.query_feats[t, m, 0]
            ivv, mi = data.targets[t, m], _snap(data.query_feats[t, m, 1], self.mat_grid)
            for a in np.unique(aids):
                for mm in np.unique(mi[aids == a]):
                    sel = (aids == a) & (mi == mm)
                    if sel.sum() < self.degree + 1:
                        continue
                    st = rep.fit_bspline_fixed(kk[sel], ivv[sel],
                                               float(self.mat_grid[mm]), self.knots, self.degree,
                                               curvature=self.lambda_rough)
                    nd = (int(a), int(mm))
                    days.setdefault(nd, []).append(int(t))
                    coeffs.setdefault(nd, []).append(st.coeffs)
        history: dict = {nd: (np.asarray(days[nd]), np.asarray(coeffs[nd])) for nd in days}
        return history

    def _resolve_edges(self, data) -> dict:
        spec = self.edges_spec
        if isinstance(spec, dict):
            return spec
        return _resolve_edges_spec(
            spec, nodes=self._all_nodes(data), asset_names=self.asset_names,
            mat_grid=self.mat_grid, edge_weight=self.edge_weight,
            history_fn=lambda: self._coeff_history(data), allow_market_pca=True)

    @property
    def _p(self) -> int:
        return len(self.knots) - self.degree - 1

    def _design(self, k) -> np.ndarray:
        return self._tmpl.design_matrix(np.asarray(k, float))

    # ── prior / sequence ───────────────────────────────────────────────────────
    def reset_sequence(self) -> None:
        # Clear the carried temporal state ONLY — the edge structure (built/learned in
        # train()) is static across the sequence and must survive the reset.
        self._prior = None
        self._nodes = None

    def _nodes_from(self, quotes: Quotes) -> list:
        mi = _snap(quotes.T, self.mat_grid)
        return sorted({(int(a), int(m)) for a, m in zip(quotes.asset_id, mi)})

    def _merge_today_nodes(self, quotes: Quotes) -> None:
        """Widen the active node set to include today's (asset, maturity) nodes.

        In a free-run the maturities quoted drift away from the day the prior was first
        seeded; without this, `_solve` only ever fits the seed day's maturities and silently
        drops (and mis-interpolates) any new one.  The carried prior in `self._prior` is left
        untouched, so temporal coupling on the nodes it already covers is preserved.
        """
        if not quotes.n or self._nodes is None:
            return
        merged = set(self._nodes) | set(self._nodes_from(quotes))
        if len(merged) != len(self._nodes):
            self._nodes = sorted(merged)

    def seed_prior(self, quotes: Quotes) -> None:
        if self.knots is None:
            raise RuntimeError("RegularizedBSpline needs train() before seeding")
        nodes = self._nodes_from(quotes)
        mi = _snap(quotes.T, self.mat_grid)
        prior = {}
        for (a, m) in nodes:
            sel = (quotes.asset_id == a) & (mi == m)
            st = rep.fit_bspline_fixed(quotes.k[sel], quotes.iv[sel],
                                       float(self.mat_grid[m]), self.knots, self.degree,
                                       curvature=self.lambda_rough)
            prior[(a, m)] = st.coeffs
        if self.graph_fallback:
            # Cross-maturity anchor: extend each asset's prior to EVERY grid maturity by copying
            # the nearest observed maturity's coeffs (coeffs ≈ IV values, so this is a flat-in-
            # maturity level anchor). The node then has a baseline p_i, so the cross-asset edge
            # transfers β·ΔSPY onto it and today's own quote (if any) refines it.
            grid_m = range(len(self.mat_grid))
            for a in sorted({aa for (aa, _) in nodes}):
                obs = sorted(m for (aa, m) in prior if aa == a)
                if not obs:
                    continue
                for m in grid_m:
                    if (a, m) not in prior:
                        nearest = min(obs, key=lambda om: abs(om - m))
                        prior[(a, m)] = prior[(a, nearest)].copy()
            nodes = sorted(prior.keys())
        self._prior = prior
        self._nodes = nodes
        # edges were built/learned once in train(); fall back to uniform if predict()
        # is called without train() (e.g. a bare smoke test).
        if self._edges is None and self.lambda_graph > 0:
            self._edges = edges_mod.uniform_edges(nodes, self.edge_weight)

    # ── the block solve ─────────────────────────────────────────────────────────
    def _solve(self, context: Quotes) -> dict:
        nodes = self._nodes
        idx = {nd: i for i, nd in enumerate(nodes)}
        n, p = len(nodes), self._p
        H = np.zeros((n * p, n * p))
        b = np.zeros(n * p)

        mi = _snap(context.T, self.mat_grid) if context.n else np.zeros(0, int)
        supported = set()      # nodes constrained by data or a temporal prior (see return)
        node_fb: dict = {}     # per-node flat fallback level + observed IV range (coeff clamp)
        node_max: dict = {}
        node_min: dict = {}
        for nd, i in idx.items():
            sl = slice(i * p, (i + 1) * p)
            a, m = nd
            # data term
            has_data = False
            if self.lambda_data > 0 and context.n:
                sel = (context.asset_id == a) & (mi == m)
                if sel.any():
                    iv_sel = context.iv[sel]
                    B = self._design(context.k[sel])
                    H[sl, sl] += self.lambda_data * (B.T @ B)
                    b[sl] += self.lambda_data * (B.T @ iv_sel)
                    has_data = True
                    node_fb[nd] = float(np.mean(iv_sel))
                    node_max[nd] = float(np.max(iv_sel))
                    node_min[nd] = float(np.min(iv_sel))
            # curvature (P-spline) term — fixed, so negligible once the node is well sampled
            if self.lambda_rough > 0 and self._D2D2 is not None:
                H[sl, sl] += self.lambda_rough * self._D2D2
            # temporal term
            has_prior = bool(self.lambda_temporal > 0 and self._prior and nd in self._prior)
            if has_prior:
                H[sl, sl] += self.lambda_temporal * np.eye(p)
                b[sl] += self.lambda_temporal * self._prior[nd]
                if nd not in node_fb:                      # prior coeffs ARE IV values
                    pc = self._prior[nd]
                    node_fb[nd] = float(np.mean(pc))
                    node_max[nd] = float(np.max(pc))
                    node_min[nd] = float(np.min(pc))
            if has_data or has_prior:
                supported.add(nd)

        # graph term — delta-edge coupling with optional linear map M.
        # For edge (i,j) penalising ||(c_i-p_i) - M(c_j-p_j)||^2_P (P=prec*I or matrix):
        #   H[i,i] += λ P ;  H[i,j] -= λ P M ;  b[i] += λ P (p_i - M p_j)
        if self.lambda_graph > 0 and self._edges:
            eye = np.eye(p)
            for (ni, nj), edge in self._edges.items():
                # The penalty is on INCREMENTS off the priors, so it needs both p_i and p_j;
                # skip the edge entirely if either is missing (adding only the H terms would
                # silently turn it into an absolute coupling c_i ≈ M c_j).
                if (ni not in idx or nj not in idx
                        or not self._prior or ni not in self._prior or nj not in self._prior):
                    continue
                i, j = idx[ni], idx[nj]
                si, sj = slice(i * p, (i + 1) * p), slice(j * p, (j + 1) * p)
                prec = edge.precision
                P = prec * eye if np.isscalar(prec) else np.asarray(prec)
                M = eye if edge.matrix is None else np.asarray(edge.matrix)
                PM = P @ M
                H[si, si] += self.lambda_graph * P
                H[si, sj] -= self.lambda_graph * PM
                b[si] += self.lambda_graph * (P @ self._prior[ni] - PM @ self._prior[nj])

        # maturity-smoothness term — couple the *states* (not increments) of adjacent
        # same-asset maturity nodes: λ_mat ‖c_i − c_j‖².  Within-asset only (shares the
        # underlying), so it adds NO cross-asset information.  It is the closed-form
        # analogue of "use the nearest observed maturity": a chain of empty (no-data)
        # maturity nodes anchored to one observed end collapses to that end's coeffs
        # (flat clamp in T), while fitted nodes are only lightly smoothed.
        if self.lambda_maturity > 0:
            eye = np.eye(p)
            # Weak flat-mean anchor (coeffs ARE IV values).  Without a temporal prior the
            # data term is often rank-deficient (few points, p coeffs) and the Laplacian
            # has a constant null space, so the tiny global ridge alone lets data-less
            # nodes blow up.  This anchor is negligible vs a fitted node's data term, but
            # gives data-less nodes a sane fallback level for the Laplacian to smooth from.
            iv_bar = float(np.mean(context.iv)) if context.n else _DEFAULT_IV
            for nd, i in idx.items():
                sl = slice(i * p, (i + 1) * p)
                H[sl, sl] += _FLAT_ANCHOR_W * eye
                b[sl] += _FLAT_ANCHOR_W * iv_bar
            # adjacent-maturity Laplacian, per asset (grid index order == maturity order)
            by_asset: dict = {}
            for (a, m) in nodes:
                by_asset.setdefault(a, []).append(m)
            for a, ms in by_asset.items():
                ms_sorted = sorted(ms)
                for m1, m2 in zip(ms_sorted, ms_sorted[1:]):
                    i, j = idx[(a, m1)], idx[(a, m2)]
                    si, sj = slice(i * p, (i + 1) * p), slice(j * p, (j + 1) * p)
                    H[si, si] += self.lambda_maturity * eye
                    H[sj, sj] += self.lambda_maturity * eye
                    H[si, sj] -= self.lambda_maturity * eye
                    H[sj, si] -= self.lambda_maturity * eye

        H += 1e-8 * np.eye(n * p)
        c = np.linalg.solve(H, b)
        # Drop maturity nodes that NEITHER data NOR a prior constrained: a pure-data fit leaves
        # those at ~0 (the curvature/ridge null-space), which would corrupt the across-maturity
        # interpolation in _eval (a spurious zero-smile).  With maturity-smoothness on, empty
        # nodes ARE filled (Laplacian + flat anchor), so keep them.  Net effect: fit where there
        # is support, interpolate across the supported maturities otherwise — like the prior.
        keep = set(nodes) if self.lambda_maturity > 0 else supported
        out: dict = {}
        for nd in nodes:
            if nd not in keep:
                continue
            cc = c[idx[nd] * p:(idx[nd] + 1) * p]
            if not np.all(np.isfinite(cc)):
                cc = np.full(p, max(node_fb.get(nd, _DEFAULT_IV), _IV_FLOOR))
            # Variation-diminishing clamp: a clamped B-spline lies within the convex hull of its
            # coefficients, so bounding each coeff to the node's observed IV range (with margin)
            # bounds the whole smile — an under-constrained wing can rise to ~1.5×max but never
            # blow up.  Only applied to nodes with their OWN data/prior; a maturity-Laplacian-
            # filled node (no data, no prior) is left to inherit its neighbours' level instead
            # of being forced into a fixed [0.14, 0.3] band.
            if nd in node_max:
                lo = max(node_min[nd] * _CLAMP_LO_FRAC, 1e-4)
                hi = max(node_max[nd] * _CLAMP_HI_FRAC, lo + 1e-3)
                cc = np.clip(cc, lo, hi)
            out[nd] = np.maximum(cc, _IV_FLOOR)
        return out

    def _eval(self, coeffs: dict, query: QueryPoints) -> np.ndarray:
        out = np.empty(query.n, dtype=float)
        for a in np.unique(query.asset_id):
            qm = query.asset_id == a
            # each smile MUST carry its own maturity: _interp_smiles interpolates in total
            # variance w = iv²·T, so a wrong T (e.g. the template's 1.0) inflates short-dated
            # IVs by 1/√T — catastrophic at weeklies.
            states = {float(self.mat_grid[m]): self._tmpl.with_coeffs(c, T=float(self.mat_grid[m]))
                      for (aa, m), c in coeffs.items() if aa == a}
            if not states:
                out[qm] = _DEFAULT_IV
                continue
            out[qm] = np.maximum(_interp_smiles(states, query.k[qm], query.T[qm]), _IV_FLOOR)
        return out

    # ── contract ────────────────────────────────────────────────────────────────
    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self._nodes is None:          # no prior seeded → fit context alone
            self.seed_prior(context if context.n else query_as_quotes(query))
        coeffs = self._solve(context)
        return SurfacePrediction(iv=self._eval(coeffs, query))

    def step(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self._nodes is None:
            self.seed_prior(context)
        else:
            self._merge_today_nodes(context)   # today's maturities may differ from the seed day
        coeffs = self._solve(context)
        pred = self._eval(coeffs, query)
        self._prior = coeffs             # roll today's fit forward
        return SurfacePrediction(iv=pred)


# ════════════════════════════════════════════════════════════════════════════
# Non-linear regularised SVI / SSVI
# ════════════════════════════════════════════════════════════════════════════
class RegularizedParametric(SurfaceModel):
    """SVI (smile) or SSVI (surface) with temporal + graph regularisation via scipy.

    Nodes are (asset, maturity) for SVI and (asset, 0) for SSVI.  Each node carries a
    parameter vector; the objective is per-node data MSE (total-variance space) plus
    temporal and graph penalties on the parameters / their increments.
    """

    def __init__(self, kind: str, lambda_temporal=0.0, lambda_graph=0.0,
                 edge_weight=1.0, edges="uniform", n_history=250, name=None):
        assert kind in ("svi", "ssvi")
        self.kind = kind
        self.lambda_temporal = lambda_temporal
        self.lambda_graph = lambda_graph
        self.edge_weight = edge_weight
        self.edges_spec = edges
        self.n_history = n_history
        if name:
            self.name = name
        else:
            suffix = _default_suffix(lambda_temporal, lambda_graph)
            self.name = f"{kind}_{suffix}"
        self.mat_grid = None
        self.asset_names: list | None = None
        self._prior: dict | None = None
        self._nodes: list | None = None
        self._edges: dict | None = None

    def train(self, data, *, saved: bool = False, force: bool = False) -> None:
        self.data_tag = data.meta.get("dgp", self.data_tag)
        self.mat_grid = _maturity_grid(data)
        self.asset_names = data.meta.get("asset_names",
                                         [f"asset_{i}" for i in range(data.n_assets)])
        if self.lambda_graph > 0:
            self._edges = self._resolve_edges(data)

    # ── edge resolution (builders + learners on the parameter history) ───────────
    def _param_history(self, data) -> dict:
        """Per-node fitted-parameter history, **day-indexed**: node -> (days, params).

        Same `(days, params)` tuple shape as the B-spline `_coeff_history`, so the edge
        learners (`_delta_seqs`/`_aligned_pairs`) compute increments on a common calendar and
        pair assets on the SAME days.  (Storing bare ordered lists silently misaligned
        cross-asset increments — see the matching note on `_coeff_history`.)"""
        train_idx = data.train_idx()
        if self.n_history and len(train_idx) > self.n_history:
            train_idx = train_idx[-self.n_history:]
        days: dict = {}
        params: dict = {}
        last: dict = {}                      # warm-start each node day-over-day
        for t in train_idx:
            q = data.quotes_at(t)
            for nd in self._node_list(q):
                k, T, iv = self._node_obs(q, nd)
                need = _NPARAMS[self.kind]
                if len(k) >= need:
                    mat_T = float(self.mat_grid[nd[1]]) if self.kind == "svi" else 0.0
                    p = self._fit_indep(k, T, iv, mat_T, x0=last.get(nd))
                    last[nd] = p
                    days.setdefault(nd, []).append(int(t))
                    params.setdefault(nd, []).append(p)
        return {nd: (np.asarray(days[nd]), np.asarray(params[nd])) for nd in days}

    def _resolve_edges(self, data) -> dict:
        spec = self.edges_spec
        if isinstance(spec, dict):
            return spec
        # all nodes seen across training data
        nodes = sorted({nd for t in data.train_idx()[-self.n_history:]
                        for nd in self._node_list(data.quotes_at(t))}) \
            if self.n_history else None
        return _resolve_edges_spec(
            spec, nodes=nodes, asset_names=self.asset_names, mat_grid=self.mat_grid,
            edge_weight=self.edge_weight,
            history_fn=lambda: self._param_history(data), allow_market_pca=False)

    def reset_sequence(self) -> None:
        # keep self._edges (learned/built in train); only clear carried temporal state
        self._prior = self._nodes = None

    # node param helpers
    def _fit_indep(self, k, T, iv, mat_T, x0=None):
        if self.kind == "svi":
            s = rep.fit_svi(k, iv, mat_T)
            return np.array([s.a, s.b, s.rho, s.m, s.sigma])
        # SSVI: warm-started analytic-gradient fitter (same objective as fit_ssvi, faster).
        s = rep.fit_ssvi_fast(k, T, iv, x0=x0)
        return np.array([s.v_0, s.v_inf, s.kappa, s.rho, s.eta, s.gamma])

    def _bounds(self):
        if self.kind == "svi":
            return [(1e-6, None), (1e-6, None), (-0.999, 0.999), (-0.5, 0.5), (1e-4, None)]
        return [(1e-6, None), (1e-6, None), (1e-6, None), (-0.999, 0.999), (1e-6, 2.0), (1e-6, 0.5)]

    def _state(self, params, mat_T):
        if self.kind == "svi":
            return rep.SviRawState(*params, T=mat_T)
        return rep.SSVISurfaceState(*params)

    def _w_fit(self, params, k, T, mat_T):
        st = self._state(params, mat_T)
        return st.total_variance(k) if self.kind == "svi" else st.total_variance(k, T)

    def _node_list(self, quotes: Quotes):
        if self.kind == "ssvi":
            return sorted({(int(a), 0) for a in quotes.asset_id})
        mi = _snap(quotes.T, self.mat_grid)
        return sorted({(int(a), int(m)) for a, m in zip(quotes.asset_id, mi)})

    def _node_obs(self, quotes: Quotes, node):
        a, m = node
        if self.kind == "ssvi":
            sel = quotes.asset_id == a
        else:
            mi = _snap(quotes.T, self.mat_grid)
            sel = (quotes.asset_id == a) & (mi == m)
        return quotes.k[sel], quotes.T[sel], quotes.iv[sel]

    def seed_prior(self, quotes: Quotes) -> None:
        nodes = self._node_list(quotes)
        carried = self._prior or {}          # warm-start from the previous prior if present
        prior = {}
        for nd in nodes:
            k, T, iv = self._node_obs(quotes, nd)
            mat_T = float(self.mat_grid[nd[1]]) if self.kind == "svi" else 0.0
            if len(k) >= _NPARAMS[self.kind]:
                prior[nd] = self._fit_indep(k, T, iv, mat_T, x0=carried.get(nd))
        self._prior = prior
        self._nodes = nodes
        if self._edges is None and self.lambda_graph > 0:
            self._edges = edges_mod.uniform_edges(nodes, self.edge_weight)

    def _solve(self, context: Quotes) -> dict:
        nodes = self._nodes
        d = _NPARAMS[self.kind]
        # Fast path — pure per-day SSVI data fit (no temporal pull, no graph coupling):
        # the joint objective then separates per asset, so fit each node directly with the
        # warm-started analytic-gradient fitter instead of one big numeric-gradient solve.
        if self.kind == "ssvi" and self.lambda_temporal == 0 and self.lambda_graph == 0:
            out = {}
            for nd in nodes:
                k, T, iv = self._node_obs(context, nd)
                if len(k) >= d:
                    x0 = self._prior.get(nd) if self._prior else None
                    s = rep.fit_ssvi_fast(k, T, iv, x0=x0)
                    out[nd] = np.array([s.v_0, s.v_inf, s.kappa, s.rho, s.eta, s.gamma])
                elif self._prior and nd in self._prior:
                    out[nd] = self._prior[nd]
                else:
                    out[nd] = _default_params("ssvi")
            return out
        idx = {nd: i for i, nd in enumerate(nodes)}
        # init from prior (or independent fit on context, else generic guess)
        x0 = np.zeros(len(nodes) * d)
        for nd, i in idx.items():
            if self._prior and nd in self._prior:
                x0[i * d:(i + 1) * d] = self._prior[nd]
            else:
                k, T, iv = self._node_obs(context, nd)
                mat_T = float(self.mat_grid[nd[1]]) if self.kind == "svi" else 0.0
                x0[i * d:(i + 1) * d] = (self._fit_indep(k, T, iv, mat_T)
                                         if len(k) >= d else _default_params(self.kind))
        bounds = self._bounds() * len(nodes)

        # precompute per-node context obs and total-variance targets w_obs.  For SVI the fitted
        # state is evaluated at the node's snapped grid maturity (`_w_fit` uses mat_T), so w_obs
        # must use the SAME mat_T — not each point's raw T — or the objective compares total
        # variances at different maturities.  SSVI's w_fit uses the per-point T, so w_obs = iv²·T.
        node_data = {}
        for nd in nodes:
            k, T, iv = self._node_obs(context, nd)
            wT = float(self.mat_grid[nd[1]]) if self.kind == "svi" else T
            node_data[nd] = (k, T, iv * iv * wT)

        def objective(x):
            tot = 0.0
            for nd, i in idx.items():
                pi = x[i * d:(i + 1) * d]
                k, T, w_obs = node_data[nd]
                mat_T = float(self.mat_grid[nd[1]]) if self.kind == "svi" else 0.0
                if len(k):
                    try:
                        w_fit = self._w_fit(pi, k, T, mat_T)
                        tot += float(np.mean((w_fit - w_obs) ** 2))
                    except Exception:
                        tot += 1e6
                if self.lambda_temporal > 0 and self._prior and nd in self._prior:
                    tot += self.lambda_temporal * float(np.sum((pi - self._prior[nd]) ** 2))
            if self.lambda_graph > 0 and self._edges:
                for (ni, nj), edge in self._edges.items():
                    if not (self._prior and ni in self._prior and nj in self._prior
                            and ni in idx and nj in idx):
                        continue
                    di = x[idx[ni] * d:(idx[ni] + 1) * d] - self._prior[ni]
                    dj = x[idx[nj] * d:(idx[nj] + 1) * d] - self._prior[nj]
                    M = np.eye(d) if edge.matrix is None else np.asarray(edge.matrix)
                    r = di - M @ dj                       # delta-edge residual
                    prec = edge.precision
                    quad = (float(prec) * float(r @ r) if np.isscalar(prec)
                            else float(r @ np.asarray(prec) @ r))
                    tot += self.lambda_graph * quad
            return tot

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})
        return {nd: res.x[idx[nd] * d:(idx[nd] + 1) * d] for nd in nodes}

    def _eval(self, params: dict, query: QueryPoints) -> np.ndarray:
        out = np.empty(query.n, dtype=float)
        for a in np.unique(query.asset_id):
            qm = query.asset_id == a
            if self.kind == "ssvi":
                nd = (int(a), 0)
                if nd not in params:
                    out[qm] = _DEFAULT_IV
                    continue
                st = self._state(params[nd], 0.0)
                out[qm] = np.maximum(st.implied_vol(query.k[qm], query.T[qm]), _IV_FLOOR)
            else:
                states = {float(self.mat_grid[m]): self._state(p, float(self.mat_grid[m]))
                          for (aa, m), p in params.items() if aa == a}
                if not states:
                    out[qm] = _DEFAULT_IV
                    continue
                out[qm] = np.maximum(_interp_smiles(states, query.k[qm], query.T[qm]), _IV_FLOOR)
        return out

    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self._nodes is None:
            self.seed_prior(context if context.n else query_as_quotes(query))
        return SurfacePrediction(iv=self._eval(self._solve(context), query))

    def step(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self._nodes is None:
            self.seed_prior(context)
        params = self._solve(context)
        pred = self._eval(params, query)
        self._prior = params
        return SurfacePrediction(iv=pred)


def _default_params(kind: str) -> np.ndarray:
    if kind == "svi":
        return np.array([0.04, 0.1, -0.3, 0.0, 0.1])
    return np.array([0.04, 0.032, 1.0, -0.3, 0.5, 0.25])
