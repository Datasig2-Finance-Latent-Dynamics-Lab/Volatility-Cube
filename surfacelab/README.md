# surfacelab

Unified volatility-surface modelling and evaluation.

Every method — per-day SVI / SSVI / B-spline / PCA fits, temporally- and
graph-regularised fits, a Kalman factor model, and a Conditional Neural Process — is a
subclass of one `SurfaceModel` with the same contract:

```python
class SurfaceModel:
    def train(self, data, *, saved=False, force=False) -> None: ...   # optional (no-op for per-day fits)
    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction: ...   # required
    # sequential hooks (default = stateless):
    def reset_sequence(self): ...
    def seed_prior(self, quotes: Quotes): ...
    def step(self, context, query) -> SurfacePrediction: ...
```

Because every model shares this contract, the whole cross-method comparison falls out of
**one** evaluation harness and **one** analytics suite.

## The problem

Given *yesterday's* surface per asset plus *a few* of today's quotes, predict today's full
surface, exploiting both temporal persistence and cross-asset structure. The motivating use
case is a desk that wants systematic, arbitrage-free extrapolation into the illiquid region
(wide strikes, long maturities) to support trader judgement.

## Layout

```
surfacelab/
  core/        types (Quotes, QueryPoints, SurfacePrediction), SurfaceModel, arbitrage checks
  data/        Dataset + loaders (load_heston, load_grouptech) + compute_bspline_prior
               + generate_heston (builds the synthetic dataset)
  models/      parametric/{svi,ssvi,bspline,pca} · factors (shared PCA basis) ·
               regularized (closed-form B-spline + scipy SVI/SSVI) · kalman · cnp/ · registry
  eval/        metrics · records (→ records.csv + summary.csv) · harness (run, run_sequential)
  analytics/   plots (reconstruction, RMSE-vs-ctx, decay) · report (HTML) · latent (CNP)
  experiments/ configs · run.py (CLI entry point) · CNP training scripts
  statistics/  cross-asset exploitability analysis (own README)
```

## Running

```bash
# independent ("perfect prior") evaluation across all methods on the Heston DGP
python -m surfacelab.experiments.run --config heston_all_methods

# sequential decay test (carry the model's own prior forward)
python -m surfacelab.experiments.run --config heston_all_methods_sequential

# real market data (group_tech_us.csv, via DuckDB push-down)
python -m surfacelab.experiments.run --config market_all_methods

# fast smoke run on a subset
python -m surfacelab.experiments.run --config heston_all_methods --quick
```

Each run writes `records.csv`, `summary.csv`, and an interactive `report.html` under
`results/surfacelab/<config>/`. Use `.venv/bin/python3`.

## The two harness modes

- **`run` (independent / "perfect prior")** — each eval day is re-seeded with *yesterday's
  full* quotes, then predicts today from a context subset. Sweeps context sizes and adds an
  `extrapolation` split (observe only the liquid region, predict the illiquid region).
- **`run_sequential` (decay test)** — seed once on day 0, then walk forward: tomorrow's
  prior is the model's *own* fit today, so error can compound. Plots RMSE over time.

## Adding a model

Subclass `SurfaceModel`, implement `predict` (and `train`/`seed_prior`/`step` if it learns
or carries state), register it in `models/registry.py`, and add it to an experiment in
`experiments/configs.py`. The harness, metrics, report, and arbitrage diagnostics all work
on it automatically.

## Notes

- Trained CNPs are cached under `../trained_models/` and loaded by `train(saved=True)`.
  Checkpoints are not committed, so the first run of a CNP config trains and caches one
  (slow, GPU recommended); `--retrain` forces a fresh one. A checkpoint that does not match
  the current architecture is discarded with a warning rather than aborting the run.
- The closed-form `RegularizedBSpline` solves the joint data+temporal+graph quadratic in one
  block linear system (the headline efficiency trick). The SVI/SSVI regularised variants use
  scipy and are correspondingly slower in long sequential runs.
- **Cross-asset graph edges** (`models/edges.py`) stay fully general. An edge
  (i→j) carries a `DeltaEdge(precision, matrix)` penalising `(c_i−p_i) − M·(c_j−p_j)`.
  Choose the structure via the regularised models' `edges=` argument:
  `uniform` · `tiered` (SPY-leads-stocks asymmetry) · `factored` (+ cross-maturity decay) ·
  `learned` (OLS coupling matrix M fit on the training Δθ history) · `learned_diag` ·
  `learned_scalar` (|corr| precision). Learned and builder edges are resolved once in
  `train()`.
