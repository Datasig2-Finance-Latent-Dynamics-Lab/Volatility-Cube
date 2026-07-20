# surfacelab.statistics — is cross-asset coupling *exploitable*?

This module answers, empirically, a question the experiments only answered by anecdote:
**the ATM increments of the assets are clearly correlated (ρ̄ ≈ 0.42), so why does no
cross-asset / graph method ever beat the single-asset baselines — even when the prior
fit is poor and today's quotes are few?**

The short answer it quantifies: a high *marginal* correlation is not the same as
*exploitable* information. What can be exploited is only the shared variance that your
**own** observations have not already pinned down. Once you condition on your own quotes,
the extra information a correlated peer can add is tiny — and we can put a hard number
(and an upper bound) on exactly how tiny.

## The one formula

Model ATM increments with a one-factor structure (common factor share ρ, idiosyncratic
share 1−ρ) and let your own observation of today's ATM carry noise-to-signal ratio
`r = ν²/σ²` (large `r` = bad prior fit and/or few quotes). Then the **maximum fraction of
your residual variance that ANY cross-asset information can remove** is

```
        ρ · r
C = ───────────────          (capped at ρ)
    (1 − ρ) + r
```

Derivation (posterior precision is additive). Prior `Var(x)=σ²`. Own obs gives
`V_own = σ² r/(1+r)`. An oracle that knows the common factor exactly leaves only the
idiosyncratic part, `V_floor = σ²(1−ρ)r/((1−ρ)+r)`. The fractional drop is

```
C = 1 − V_floor/V_own = 1 − (1−ρ)(1+r)/((1−ρ)+r) = ρr/((1−ρ)+r).
```

Limits make the intuition exact:
- `r → 0` (own quotes pin the ATM) ⇒ `C → 0`: nothing to add, you already know it.
- `r → ∞` (own quotes useless) ⇒ `C → ρ`: peers recover the *common* part and no more.
- the idiosyncratic share `1−ρ` is a floor cross-asset can **never** cross.

`C` is an **oracle bound**: it assumes peers are observed perfectly and are infinite in
number. Real peers are themselves estimated from few noisy quotes, so the achievable gain
is strictly below `C` (confirmed in `validation.txt`).

## The real answer: it's about liquidity *symmetry*, not ρ

ρ is genuinely high (≈0.42), so this is **not** an idiosyncratic-floor story. Whether the
coupling is *exploitable* turns entirely on **who is sparse**:

- **Symmetric** (every asset equally sparse). This is the regime the evaluation harness
  actually creates — `eval.harness._per_asset_sample` draws the *same* `n_ctx` quotes for
  every asset, SPY included. Here peers are as blind as you: the realizable gain is only
  ~4% RMSE at N=10, and the coupling you could *measure* from N quotes, `ρ̂(N)≈ρ/(1+2r)`,
  collapses to ≈0. Nothing — including the CNP — can or should use it. (`sweep.py`)
- **Asymmetric** (SPY/large names liquid, a single name sparse). This is the *real* market.
  A perfectly-observed SPY removes a **large** chunk of a sparse name's residual variance —
  ≈22% on average, up to ≈42% for AAPL/MSFT at N=10 — through a **beta that is stable and
  learnable from history** (unlike the unmeasurable sparse-quote ρ̂). Here cross-asset
  **should** help. (`asymmetric.py`, on the empirical covariance.)

**Implication for the thesis:** the "cross-asset doesn't help" result is likely an artefact
of sparsifying SPY along with everything else. The decisive experiment is to hold the
liquid names at full quotes and sparsify only the target, then re-test the graph models.

## Why the CNP failing to learn it is *evidence*, not a bug

If the best-case variance reduction is `C`, the RMSE prize is `≈ 1 − √(1−C) ≈ C/2`. A few-
percent `C`, concentrated only in the thin (bad-prior, few-quote) corner, is a sub-percent
RMSE signal — below the run-to-run noise of training. A gradient learner correctly ignores
it. The CNP readily learns the Heston quote→surface map (high-SNR, near-deterministic) but
not cross-asset coupling because, by this measure, **there is almost no there there.**

To make that airtight, run a **positive control** (not yet automated here): same CNP
architecture, a synthetic DGP with `r` cranked up (very sparse own quotes) and high ρ, so
`C` is large. If the CNP *does* exploit cross-asset there, capacity is not the limit and its
abstention elsewhere is a correct inference. That is the cleanest single experiment to add.

## Files (each a small, commented snippet)

| File | What it computes | Why it matters |
|------|------------------|----------------|
| `atm.py` | ATM-IV series and Δ increments (pooled B-spline at lm=0, the project's own fitter) | defines the target `x_i`; reproduces the existing `cross_asset_corr.txt` numbers |
| `correlation.py` | raw ρ, partial ρ given SPY, common-factor variance share | separates marginal co-movement from the idiosyncratic floor `1−ρ` |
| `noise.py` | `r(N)=ν²/σ²` via subsampled fits; observed `ρ̂(N)` | the lever that decides whether ρ is usable; both as functions of quote count |
| `ceiling.py` | `C`, RMSE gain, conditional MI, inverse `required_r`, model-free partial-R² | the headline ceiling + an assumption-light cross-check |
| `validate.py` | Monte-Carlo: `C_formula` vs oracle sim vs realistic finite-peer sim | proves the algebra and that `C` really is an upper bound |
| `sweep.py` | assembles ρ̂(N), r(N), C(N), realizable_sym(N) over a quote-count grid | the "how do ρ and r change with #quotes" plot the analysis asked for |
| `asymmetric.py` | per-target variance reduction with **perfect SPY / perfect peers** vs a sparse target, on the empirical covariance | the realistic regime: shows cross-asset *is* exploitable under asymmetric liquidity |
| `run.py` | runs everything, writes text/csv/png | the argument, executed |

## Running

```bash
.venv/bin/python3 -m surfacelab.statistics.run               # market (default)
.venv/bin/python3 -m surfacelab.statistics.run --dataset heston
```

Outputs → `results/diagnostics/exploitability/`:
`exploitability.txt`, `sweep.csv`, `validation.txt`, `sweep.png`, `ceiling_heatmap.png`,
`asymmetric.png`.

## On quote count as a variable

In the main validation harness the fitting quote count is fixed small by design, so it is
not a useful *stratifier* there. In this module varying it is the whole point: it traces
how own-noise `r` decays and how the *observed* correlation `ρ̂` de-attenuates as
information accrues. `N=10` is the reference (`REF_N` in `run.py`); the sweep shows the
full trajectory so the conclusion does not hinge on one arbitrary budget.
```
