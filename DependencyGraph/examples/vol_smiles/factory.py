"""
Builds Graph objects and ObservationSets from a raw options DataFrame.

Each row of the DataFrame is expected to have columns:
  date, underlying, expiry, dte, T, logmoneyness, iv, weight
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...source.core.graph import Graph
from ...source.core.dependency import Dependencies, StaticDependencies
from ...source.core.observation import Observation, ObservationSet
from .nodes import CurveNode
from .curves.base import CurveState
from .curves.svi import SviRawState, SviJWState


# ---------------------------------------------------------------------------
# Single-curve SVI fit
# ---------------------------------------------------------------------------

def fit_svi(k: np.ndarray, iv: np.ndarray, T: float, weights: np.ndarray | None = None) -> SviRawState:
    """
    Fit SVI (raw parameterisation) to (log-moneyness, implied-vol) observations.
    Minimises weighted MSE in total-variance space.
    """
    w_obs = iv ** 2 * T

    def objective(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        w_fit = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        resid = w_fit - w_obs
        return float(np.average(resid ** 2, weights=weights))

    x0 = np.array([np.mean(w_obs) * 0.8, 0.1, -0.3, 0.0, 0.1])
    bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999), (-0.5, 0.5), (1e-4, None)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    a, b, rho, m, sigma = result.x
    return SviRawState(a=a, b=b, rho=rho, m=m, sigma=sigma)


def fit_svi_jw(k: np.ndarray, iv: np.ndarray, T: float, weights: np.ndarray | None = None) -> SviJWState:
    """
    Fit SVI in Jump-Wing parameterisation to (log-moneyness, implied-vol) observations.
    Minimises weighted MSE in total-variance space with the optimiser working in JW
    coordinates, which have more interpretable bounds:

        v       > 0                 (ATM implied variance)
        p, c    > 0                 (wing slopes, no-crossing condition)
        0 < v_tilde <= v            (minimum variance below ATM)
        |psi|   < (p + c) / 2      (ATM skew bounded by wing slopes, keeps |d| < 1)
    """
    w_obs = iv ** 2 * T
    w_atm_guess = float(np.interp(0.0, k, w_obs)) if k.min() <= 0 <= k.max() else float(np.mean(w_obs))

    def objective(jw_params: np.ndarray) -> float:
        v, psi, p, c, v_tilde = jw_params
        try:
            w_fit = SviJWState(v, psi, p, c, v_tilde, T).total_variance(k)
        except (ValueError, FloatingPointError):
            return 1e10
        resid = w_fit - w_obs
        return float(np.average(resid ** 2, weights=weights))

    v0 = w_atm_guess / T
    x0 = np.array([v0, -0.02, 0.08, 0.12, v0 * 0.8])
    bounds = [(1e-6, None), (-0.5, 0.5), (1e-6, None), (1e-6, None), (1e-6, None)]

    def objective_with_penalty(jw_params: np.ndarray) -> float:
        v, _, _, _, v_tilde = jw_params
        return objective(jw_params) + 1e4 * max(v_tilde - v, 0.0) ** 2

    result = minimize(objective_with_penalty, x0, method="L-BFGS-B", bounds=bounds)
    v, psi, p, c, v_tilde = result.x

    try:
        state = SviJWState(v, psi, p, c, v_tilde, T)
        state.to_raw()  # validate conversion is well-defined
        return state
    except (ValueError, FloatingPointError):
        return fit_svi(k, iv, T, weights).to_jw(T)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

class ObservationFactory:
    """Builds an ObservationSet for a single date from the raw DataFrame."""

    def __init__(self, underlyings: list[str], expiries: list[pd.Timestamp]):
        self.underlyings = underlyings
        self.expiries = expiries

    def build(self, df: pd.DataFrame, date: pd.Timestamp) -> ObservationSet:
        day_df = df[df["date"] == date]
        observations: list[Observation] = []

        for underlying in self.underlyings:
            for expiry in self.expiries:
                mask = (day_df["underlying"] == underlying) & (day_df["expiry"] == expiry)
                slice_df = day_df[mask]
                if slice_df.empty:
                    continue
                T_node = float(slice_df["T"].iloc[0])
                nid = CurveNode(underlying, expiry, T_node)
                for _, row in slice_df.iterrows():
                    observations.append(
                        Observation(nid, (row["logmoneyness"], row["iv"]), float(row["weight"]))
                    )

        return ObservationSet(observations, date)


class GraphFactory:
    """
    Builds a Graph for a single date by fitting SVI to each (underlying, expiry) slice.
    Use this to create a sensible warm-start prior before running the updater.

    Parameters
    ----------
    fit_fn : callable with signature fit_fn(k, iv, T, weights) -> CurveState.
             Defaults to fit_svi (raw parameterisation); pass fit_svi_jw to
             initialise and optimise in JW coordinates.
    """

    def __init__(
        self,
        underlyings: list[str],
        expiries: list[pd.Timestamp],
        dependencies: Dependencies,
        fit_fn=None,
    ):
        self.underlyings = underlyings
        self.expiries = expiries
        self.dependencies = dependencies
        self.fit_fn = fit_fn if fit_fn is not None else fit_svi

    def build(self, df: pd.DataFrame, date: pd.Timestamp) -> Graph:
        day_df = df[df["date"] == date]
        nodes: dict[CurveNode, CurveState] = {}

        for underlying in self.underlyings:
            for expiry in self.expiries:
                mask = (day_df["underlying"] == underlying) & (day_df["expiry"] == expiry)
                slice_df = day_df[mask]
                if slice_df.empty:
                    continue
                T = float(slice_df["T"].iloc[0])
                state = self.fit_fn(
                    slice_df["logmoneyness"].values,
                    slice_df["iv"].values,
                    T,
                    slice_df["weight"].values,
                )
                nodes[CurveNode(underlying, expiry, T)] = state

        return Graph(date, nodes, self.dependencies)


def build_cross_asset_dependencies(
    underlyings: list[str],
    expiries: list[pd.Timestamp],
    df: pd.DataFrame,
    date: pd.Timestamp,
    weight: float = 1.0,
) -> StaticDependencies:
    """
    Connects same-expiry nodes across different underlyings with a uniform weight.
    Edges are symmetric and directed both ways.
    """
    day_df = df[df["date"] == date]
    edges: dict[tuple[CurveNode, CurveNode], float] = {}

    for expiry in expiries:
        nodes_for_expiry: list[CurveNode] = []
        for underlying in underlyings:
            mask = (day_df["underlying"] == underlying) & (day_df["expiry"] == expiry)
            slice_df = day_df[mask]
            if slice_df.empty:
                continue
            T_node = float(slice_df["T"].iloc[0])
            nodes_for_expiry.append(CurveNode(underlying, expiry, T_node))

        for i, n1 in enumerate(nodes_for_expiry):
            for n2 in nodes_for_expiry[i + 1:]:
                edges[(n1, n2)] = weight
                edges[(n2, n1)] = weight

    return StaticDependencies(edges)
