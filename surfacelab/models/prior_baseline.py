"""
Persistence ("prior-only") baseline: predict the previous day's surface, no model on top.

It carries the *exact same* B-spline prior surface the delta-CNP subtracts
(`surfacelab.data.prior._BSplinePrior`: adaptive per-smile knots, ill-conditioning
rejection, clamp-to-data-range), so it is a fair, robust baseline — it has no
representational advantage over the models under test, only a *data* advantage: as a
baseline (not a model), it is allowed to fit on the **full** previous-day surface rather
than a sparse context sample.

(Earlier this wrapped `RegularizedBSpline`'s fixed-knot fitter, which blows up on the
steep short-dated deep-OTM wings of market data — making the baseline look far worse than
honest persistence.  Sharing the delta-CNP's guarded fit fixes that and guarantees the two
priors are identical, so the delta-CNP's gain over this baseline is genuine model skill.)

Concretely: `seed_prior` fits the per-(asset, maturity) smiles on a full day's quotes and
`predict` evaluates them at the query points, ignoring today's context.  The sequential
harness re-seeds it with the true full previous day every step (`reseed_each_step = True`),
so it is a clean "perfect persistence" baseline — it does not free-run on its own fit.
Any skill the delta-CNP / PCA / Kalman show is what they add over carrying yesterday forward.
"""
from __future__ import annotations

import numpy as np

from surfacelab.core.model import SurfaceModel
from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction, query_as_quotes
from surfacelab.data.prior import _BSplinePrior


class PriorModel(SurfaceModel):
    name = "prior"
    #: The sequential harness re-seeds this baseline with the true full previous-day
    #: surface every step (perfect persistence) instead of letting it free-run.
    reseed_each_step = True

    def __init__(self):
        self._prior: _BSplinePrior | None = None

    # ── setup ──────────────────────────────────────────────────────────────────
    def train(self, data, *, saved: bool = False, force: bool = False) -> None:
        # Nothing to fit globally — the prior is an adaptive per-day B-spline.  Record the
        # data tag only (for identity / repr).
        self.data_tag = data.meta.get("dgp", self.data_tag)

    # ── sequential support ─────────────────────────────────────────────────────
    def reset_sequence(self) -> None:
        self._prior = None

    def seed_prior(self, quotes: Quotes) -> None:
        if quotes.n:
            self._prior = _BSplinePrior(quotes)   # same guarded fit the delta-CNP subtracts

    # ── prediction ─────────────────────────────────────────────────────────────
    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        # the seeded (full-surface) prior IS the prediction; today's context is never used
        prior = self._prior
        if prior is None:
            seed = context if context.n else query_as_quotes(query)
            prior = _BSplinePrior(seed)
        iv = prior.eval(query.k, query.T, query.asset_id)
        return SurfacePrediction(iv=np.maximum(iv, 1e-8))

    def step(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        # The harness re-seeds us with the true full previous day each step
        # (reseed_each_step), so there is nothing to roll forward — just predict.
        return self.predict(context, query)
