"""
Name → model factory, so experiments/CLI can request models by string.

Regularised variants are pre-named (e.g. "bspline_temporal_graph"); base models take
their default `name`.  CNP factories default to loading the cached checkpoint under
trained_models/ (pass checkpoint=None + a config to train from scratch).
"""
from __future__ import annotations

from surfacelab.models.parametric import SVIModel, BSplineModel, SSVIModel, PCAModel
from surfacelab.models.kalman import KalmanFactorModel
from surfacelab.models.kalman_ssvi import KalmanSSVIModel
from surfacelab.models.prior_baseline import PriorModel
from surfacelab.models.cnp import CNPModel
from surfacelab.models.regularized import RegularizedBSpline, RegularizedParametric

REGISTRY = {
    # per-day parametric (data-only)
    "svi":      lambda **k: SVIModel(),
    "svi_jw":   lambda **k: SVIModel(jump_wing=True),
    "bspline":  lambda **k: BSplineModel(),
    "ssvi":     lambda **k: SSVIModel(),
    "pca":      lambda **k: PCAModel(**k),
    # Kalman: PCA-grid factor filter, and the SSVI-parameter filter (full cross-asset A).
    "kalman_pca":  lambda **k: KalmanFactorModel(**k),
    "kalman":      lambda **k: KalmanFactorModel(**k),   # back-compat alias for kalman_pca
    "kalman_ssvi": lambda **k: KalmanSSVIModel(**k),
    "kalman_ssvi_inc": lambda **k: KalmanSSVIModel(transition_mode="increments", **k),
    # Kalman on the coefficients of a richer LINEAR basis (same filter, swappable basis):
    #   pca_inc — AR(1) on PCA factor INCREMENTS (cross-asset only; modes mix assets)
    #   bspline[_inc][_nox] — fixed tensor B-spline coeffs; per-asset blocks give a clean
    #     cross/non-cross (nox) ablation. transition="full" so the cross-asset coupling is
    #     learnable (nox zeroes the cross blocks). Richer than SSVI's 6 params/asset.
    "kalman_pca_inc":         lambda **k: KalmanFactorModel(**{"basis_kind": "pca", "transition_mode": "increments", **k}),
    "kalman_bspline":         lambda **k: KalmanFactorModel(**{"basis_kind": "bspline", "transition": "full", "transition_mode": "levels", **k}),
    "kalman_bspline_nox":     lambda **k: KalmanFactorModel(**{"basis_kind": "bspline", "transition": "full", "transition_mode": "levels", "cross_asset": False, **k}),
    "kalman_bspline_inc":     lambda **k: KalmanFactorModel(**{"basis_kind": "bspline", "transition": "full", "transition_mode": "increments", **k}),
    "kalman_bspline_inc_nox": lambda **k: KalmanFactorModel(**{"basis_kind": "bspline", "transition": "full", "transition_mode": "increments", "cross_asset": False, **k}),
    # persistence baseline: yesterday's surface (delta-CNP's B-spline prior), no model
    "prior":    lambda **k: PriorModel(),
    # neural — base CNP (mean-pool encoder) and the Attentive-NP variant (cross-attn decoder)
    "cnp":            lambda **k: CNPModel(delta=False, **k),
    "cnp_delta":      lambda **k: CNPModel(delta=True, **k),
    # regularised B-spline (closed form). Graph edges: uniform | tiered | factored |
    # learned (OLS matrix M) | learned_diag | learned_scalar.
    "bspline_data":           lambda **k: RegularizedBSpline(lambda_temporal=0.0, name="bspline_data"),
    "bspline_temporal":       lambda **k: RegularizedBSpline(name="bspline_temporal", **{"lambda_temporal": 1.0, **k}),
    "bspline_temporal_graph": lambda **k: RegularizedBSpline(edges="uniform", name="bspline_temporal_graph", **{"lambda_temporal": 1.0, "lambda_graph": 0.5, **k}),
    "bspline_tiered_graph":   lambda **k: RegularizedBSpline(lambda_temporal=1.0, lambda_graph=0.5, edges="tiered", name="bspline_tiered_graph"),
    "bspline_factored_graph": lambda **k: RegularizedBSpline(lambda_temporal=1.0, lambda_graph=0.5, edges="factored", name="bspline_factored_graph"),
    "bspline_learned_graph":  lambda **k: RegularizedBSpline(edges="learned", name="bspline_learned_graph", **{"lambda_temporal": 1.0, "lambda_graph": 0.5, **k}),
    # data-grounded cross-asset coupling (see results/diagnostics/cross_asset_corr.txt):
    #   market — pull each asset's LEVEL increment toward β·SPY's (single market factor)
    #   pca    — low-rank coupling in the PCA-mode space of the increments (denoised learned)
    "bspline_market_graph":   lambda **k: RegularizedBSpline(edges="market", name="bspline_market_graph", **{"lambda_temporal": 1.0, "lambda_graph": 0.5, **k}),
    "bspline_pca_graph":      lambda **k: RegularizedBSpline(edges="pca", name="bspline_pca_graph", **{"lambda_temporal": 1.0, "lambda_graph": 0.5, **k}),
    # …and the same six WITH the within-asset maturity-smoothness term (λ_maturity > 0),
    # so each can be compared against its no-maturity-smoothing twin.
    "bspline_data_interp":           lambda **k: RegularizedBSpline(lambda_temporal=0.0, lambda_maturity=0.5, name="bspline_data_interp"),
    "bspline_temporal_interp":       lambda **k: RegularizedBSpline(lambda_temporal=1.0, lambda_maturity=0.5, name="bspline_temporal_interp"),
    "bspline_temporal_graph_interp": lambda **k: RegularizedBSpline(lambda_temporal=1.0, lambda_graph=0.5, lambda_maturity=0.5, edges="uniform", name="bspline_temporal_graph_interp"),
    "bspline_tiered_graph_interp":   lambda **k: RegularizedBSpline(lambda_temporal=1.0, lambda_graph=0.5, lambda_maturity=0.5, edges="tiered", name="bspline_tiered_graph_interp"),
    "bspline_factored_graph_interp": lambda **k: RegularizedBSpline(lambda_temporal=1.0, lambda_graph=0.5, lambda_maturity=0.5, edges="factored", name="bspline_factored_graph_interp"),
    "bspline_learned_graph_interp":  lambda **k: RegularizedBSpline(edges="learned", name="bspline_learned_graph_interp", **{"lambda_temporal": 1.0, "lambda_graph": 0.5, "lambda_maturity": 0.5, **k}),
    # regularised SVI / SSVI (scipy)
    # per-day (λ_temporal=0) baselines: same scipy SSVI/SVI fitter as the temporal models,
    # temporal term off → a plain per-surface fit, for an apples-to-apples temporal-on/off
    # comparison (mirrors bspline_data vs bspline_temporal).
    "svi_data":             lambda **k: RegularizedParametric("svi", lambda_temporal=0.0, name="svi_data"),
    "ssvi_data":            lambda **k: RegularizedParametric("ssvi", lambda_temporal=0.0, name="ssvi_data"),
    "svi_temporal":         lambda **k: RegularizedParametric("svi", lambda_temporal=0.5, name="svi_temporal"),
    "svi_temporal_graph":   lambda **k: RegularizedParametric("svi", lambda_temporal=0.5, lambda_graph=0.3, edges="uniform", name="svi_temporal_graph"),
    "ssvi_temporal":        lambda **k: RegularizedParametric("ssvi", lambda_temporal=0.5, name="ssvi_temporal"),
    "ssvi_temporal_graph":  lambda **k: RegularizedParametric("ssvi", lambda_temporal=0.5, lambda_graph=0.3, edges="uniform", name="ssvi_temporal_graph"),
    "ssvi_learned_graph":   lambda **k: RegularizedParametric("ssvi", lambda_temporal=0.5, lambda_graph=0.3, edges="learned", name="ssvi_learned_graph"),
}


# File stems under surfacelab/models/ that people (and the web UI) mistake for model
# names → the registry key they actually correspond to.  Stems with no model behind them
# (registry, edges, factors, base, module, trainer, representations) are NOT models.
ALIASES = {
    "bspline_basis":  "kalman_bspline",     # fixed B-spline basis inside KalmanFactorModel
    "prior_baseline": "prior",
    "model":          "cnp",                # surfacelab/models/cnp/model.py
    "parametric":     "bspline",
}


def resolve(name: str) -> str:
    """Map a user/UI-supplied name to a registry key, with a helpful error otherwise."""
    if name in REGISTRY:
        return name
    if name in ALIASES:
        return ALIASES[name]
    import difflib
    near = difflib.get_close_matches(name, REGISTRY, n=5)
    raise KeyError(f"unknown model '{name}'. Did you mean {near}? Known: {sorted(REGISTRY)}")


def build(name: str, **kwargs):
    return REGISTRY[resolve(name)](**kwargs)
