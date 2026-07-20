"""
The one contract every method implements.

A `SurfaceModel` exposes an optional `train()` and a required `predict(context, query)`.
Temporal behaviour is layered on top via three optional hooks used by the sequential
harness — `reset_sequence`, `seed_prior`, and `step` — which default to a stateless
model (no carried prior).  See `surfacelab.eval.harness` for how the two harness modes
(`run`, `run_sequential`) drive these.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction

if TYPE_CHECKING:  # avoid a hard import cycle with data
    from surfacelab.data.dataset import Dataset


# Where models with a real train() cache themselves, keyed by `identifier`.
TRAINED_MODELS_DIR = Path(__file__).resolve().parents[2] / "trained_models"


class SurfaceModel(ABC):
    """Inference + (optional) training interface shared by every method.

    Subclasses set `name` (a short unique id) and `data_tag` (what the model was
    trained on).  Per-day fitters leave `train` as a no-op; only models with learned
    weights (the CNP) override it.
    """

    name: str = "surface_model"
    data_tag: str = ""
    #: If True, the sequential harness re-seeds this model with the *true* full previous-day
    #: surface before every step (a perfect-persistence baseline), instead of letting it
    #: free-run on its own carried fit.  Real models leave this False so error can compound.
    reseed_each_step: bool = False

    # ── training (optional) ──────────────────────────────────────────────────
    def train(self, data: "Dataset", *, saved: bool = False, force: bool = False) -> None:
        """Fit the model.  Default: no-op (per-day fitters need no global training).

        Models with learned weights override this and honour:
          saved=True  → load a cached model under TRAINED_MODELS_DIR/{identifier} if it
                        exists, else train and save it there.
          force=True  → always retrain, ignoring any cache.
        """
        return None

    # ── prediction (required) ────────────────────────────────────────────────
    @abstractmethod
    def predict(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        """Predict IV at ``query`` given today's observed ``context`` (one day)."""
        ...

    # ── sequential support (optional; default = stateless) ───────────────────
    def reset_sequence(self) -> None:
        """Clear any carried temporal prior/state.  No-op for stateless models."""
        return None

    def seed_prior(self, quotes: Quotes) -> None:
        """Seed the temporal prior from a full set of (yesterday's) quotes.

        Stateless models ignore this.  Regularised fitters fit a 'perfect' prior on
        ``quotes``; the Kalman model warms up its factor state; the delta-CNP fits its
        B-spline prior surface.
        """
        return None

    def step(self, context: Quotes, query: QueryPoints) -> SurfacePrediction:
        """Predict today, then roll the internal prior forward to today's own fit.

        Default = `predict` (truly stateless models carry nothing forward).
        Stateful models override to update their carried prior after predicting.
        """
        return self.predict(context, query)

    # ── identity / caching ───────────────────────────────────────────────────
    @property
    def identifier(self) -> str:
        return f"{self.name}__{self.data_tag}" if self.data_tag else self.name

    def cache_path(self, suffix: str = ".pt") -> Path:
        return TRAINED_MODELS_DIR / f"{self.identifier}{suffix}"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} data_tag={self.data_tag!r}>"
