"""
CNPModel — the SurfaceModel adapter around a trained Conditional Neural Process.

`train(saved=True)` is load-or-train: it loads a cached checkpoint under
TRAINED_MODELS_DIR/{identifier}.pt if present (CNP training is slow), else trains via
`Trainer` and saves.  `predict` converts Quotes/QueryPoints to the network's array API.

Two modes:
  * absolute  (`delta=False`) — predicts IV directly; ignores the prior.
  * increment (`delta=True`)  — predicts IV *deltas* off a B-spline prior surface that
    `seed_prior` fits on a full day's quotes (so the model carries yesterday forward
    exactly like the other temporal models).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from surfacelab.core.model import SurfaceModel, TRAINED_MODELS_DIR
from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction, to_batch_arrays
from surfacelab.models.cnp.module import FittedCNP, FittedDeltaCNP
# The carried prior surfaces live in data.prior so the persistence baseline (PriorModel)
# and the delta-CNP subtract the *same* guarded fit.
from surfacelab.data.prior import _BSplinePrior, _LinearPrior

_IV_FLOOR = 1e-8           # IV positivity floor on predictions


class CNPModel(SurfaceModel):
    def __init__(self, delta: bool = False, config=None, checkpoint: str | None = None,
                 device: str = "cpu", name: str | None = None, prior_fitter: str = "bspline",
                 per_asset: bool = False):
        self.delta = delta
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        # prior_fitter: "bspline" (default) or "linear" — the surface fitter for the prior.
        # Used for BOTH the training prior_targets and the carried inference prior so they
        # match exactly (a delta CNP must see the same prior at train and inference time).
        self.prior_fitter = prior_fitter
        # per_asset: cross-asset ABLATION.  Same trained weights, but at inference each query
        # asset attends ONLY to its own context (peers are withheld point-by-point), so every
        # cross-asset attention pathway is severed.  This is the "8 independent CNPs" control
        # without retraining — the only difference from the joint model is the context it sees.
        self.per_asset = per_asset
        self.name = name or (("cnp_delta" if delta else "cnp") + ("_nox" if per_asset else ""))
        self.fitted = None
        self._prior = None

    def _make_prior(self, quotes: Quotes):
        """Build one day's carried prior with the configured fitter (bspline | linear)."""
        return _LinearPrior(quotes) if self.prior_fitter == "linear" else _BSplinePrior(quotes)

    # ── training / loading ──────────────────────────────────────────────────────
    def train(self, data, *, saved: bool = False, force: bool = False) -> None:
        self.data_tag = data.meta.get("dgp", self.data_tag)
        path = Path(self.checkpoint) if self.checkpoint else self.cache_path(".pt")
        cls = FittedDeltaCNP if self.delta else FittedCNP
        if not force and path.exists():
            self.fitted = cls.load(str(path), device=self.device)
            return
        if self.config is None:
            raise RuntimeError(
                f"No cached CNP at {path} and no training config provided. "
                "Pass config=... to train from scratch.")
        from surfacelab.models.cnp.trainer import Trainer
        # The Trainer infers delta vs absolute purely from `prior_targets is not None`.
        # Force the dataset to match THIS model's declared mode, so an absolute CNP isn't
        # accidentally trained as a delta model just because a delta CNP shares the
        # dataset's prior_targets (and vice-versa).
        if self.delta:
            if data.prior_targets is None:
                if self.prior_fitter == "linear":
                    from surfacelab.data.prior import compute_linear_prior
                    data.prior_targets = compute_linear_prior(data)
                else:
                    from surfacelab.data.prior import compute_bspline_prior
                    data.prior_targets = compute_bspline_prior(data)
            train_data = data
        else:
            import dataclasses
            train_data = (dataclasses.replace(data, prior_targets=None)
                          if data.prior_targets is not None else data)
        self.fitted, _ = Trainer(self.config).train(train_data)
        if saved:
            TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.fitted.save(str(path))

    # ── prior / sequence ──────────────────────────────────────────────────────
    def reset_sequence(self) -> None:
        self._prior = None

    def seed_prior(self, quotes: Quotes) -> None:
        if self.delta and quotes.n:
            self._prior = self._make_prior(quotes)

    # ── prediction ───────────────────────────────────────────────────────────────
    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if self.fitted is None:
            raise RuntimeError("CNPModel.predict before train()/load")
        if self.per_asset:
            return self._predict_per_asset(context, query)
        return self._predict_one(context, query)

    def _predict_per_asset(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        """Cross-asset ablation: predict each asset's queries from ITS OWN context only."""
        iv = np.empty(query.k.shape[0], dtype=float)
        for a in np.unique(query.asset_id):
            qm = query.asset_id == a
            cm = context.asset_id == a
            q_a = QueryPoints(
                query.k[qm], query.T[qm], query.asset_id[qm],
                prior_iv=None if query.prior_iv is None else query.prior_iv[qm])
            if not cm.any():
                # No own context: an absolute CNP has nothing to say (flat fallback); a delta
                # CNP returns its carried/own prior unchanged — i.e. pure persistence, which is
                # exactly what "no cross-asset info" should give an unseen asset.
                if self.delta:
                    prior = self._prior or self._make_prior(context)
                    iv[qm] = (q_a.prior_iv if q_a.prior_iv is not None
                              else prior.eval(q_a.k, q_a.T, q_a.asset_id))
                else:
                    iv[qm] = _IV_FLOOR
                continue
            c_a = Quotes(context.k[cm], context.T[cm], context.iv[cm], context.asset_id[cm])
            iv[qm] = self._predict_one(c_a, q_a).iv
        return SurfacePrediction(iv=np.maximum(iv, _IV_FLOOR))

    def _predict_one(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        of, oa, ot, qf, qa = to_batch_arrays(context, query)
        if not self.delta:
            iv = self.fitted.predict(of, oa, ot, qf, qa)[0]
            return SurfacePrediction(iv=np.maximum(iv, _IV_FLOOR))

        # increment mode: prior at context + query
        prior = self._prior or self._make_prior(context)
        prior_ctx = prior.eval(context.k, context.T, context.asset_id)[None, :]
        prior_qry = (query.prior_iv if query.prior_iv is not None
                     else prior.eval(query.k, query.T, query.asset_id))[None, :]
        iv = self.fitted.predict(of, oa, ot, qf, qa,
                                 obs_prior_iv=prior_ctx, qry_prior_iv=prior_qry)[0]
        return SurfacePrediction(iv=np.maximum(iv, _IV_FLOOR))

    def step(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        if not self.delta:
            return self.predict(context, query)   # absolute CNP carries nothing forward

        # Free-run: predict over context ∪ query using yesterday's carried prior, then roll
        # OUR OWN predicted surface forward as tomorrow's prior — NOT a fresh B-spline of the
        # raw context (that would discard the learned increment and never let error compound,
        # making the delta-CNP collapse onto the persistence baseline).
        uk = np.concatenate([context.k, query.k])
        uT = np.concatenate([context.T, query.T])
        ua = np.concatenate([context.asset_id, query.asset_id])
        own = self.predict(context, QueryPoints(uk, uT, ua)).iv   # one inference over the union
        if uk.size:
            self._prior = self._make_prior(Quotes(uk, uT, own, ua))  # our surface → tomorrow's prior
        return SurfacePrediction(iv=np.maximum(own[context.n:], _IV_FLOOR))
