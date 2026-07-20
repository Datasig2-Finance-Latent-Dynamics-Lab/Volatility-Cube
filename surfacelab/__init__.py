"""
surfacelab — unified volatility-surface modelling.

Every method (per-day SVI/SSVI/B-spline/PCA fit, temporal/graph-regularised fit,
Kalman factor model, Conditional Neural Process) is a `SurfaceModel` subclass with
the same contract: an optional `train()` and a `predict(context, query)`.  One eval
harness (`run`, `run_sequential`) and one analytics suite then work across all of them.
"""
from surfacelab.core.types import Quotes, QueryPoints, SurfacePrediction
from surfacelab.core.model import SurfaceModel

__all__ = ["Quotes", "QueryPoints", "SurfacePrediction", "SurfaceModel"]
