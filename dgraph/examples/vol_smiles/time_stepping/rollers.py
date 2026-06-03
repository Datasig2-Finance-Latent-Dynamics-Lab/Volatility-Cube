import numpy as np
import pandas as pd

from dgraph.source.graph import Graph
from dgraph.time_stepping.roller import Roller, decay_precision
from ..source.curves.bspline import BSplineState
from ..source.curves.svi import SviRawState, SviJWState

# In general not a fan of how rolling is handled, each rolling basically has an if per state type and they get rolled
# differently, at this points it's maybe better to simply have a state method called roll, which would then have 
# a different if based on each roller type xd. I am not too sure how to solve this.


class VolRoller(Roller):

    """
    Evolves state preserving total variance at each log-moneyness level: w(k) = σ²T unchanged,
    so σ scales as √(T_old/T_new) as the expiry approaches.
    """

    def roll(self, graph: Graph, dt: float) -> Graph:
        new_nodes: dict = {}
        for nid, state in graph.nodes.items():
            T_old = state.T
            T_new = T_old - dt

            if isinstance(state, BSplineState):
                scale = np.sqrt(T_old / T_new) if T_new > 0 else 1.0
                new_state = state.with_T(T_new).from_parameters(state.coeffs * scale)
            elif isinstance(state, SviJWState):
                try:
                    new_state = state.to_raw().to_jw(T_new)
                except (ValueError, FloatingPointError):
                    new_state = state.with_T(T_new)
            elif isinstance(state, SviRawState):
                new_state = state.with_T(T_new)
            else:
                new_state = state.copy()

            new_nodes[nid] = decay_precision(new_state, dt)

        return Graph(graph.date + pd.Timedelta(days=round(dt * 365)), new_nodes, graph.edges)


class StickyDeltaRoller(Roller):

    """
    Temporal prior assuming sticky-delta (sticky-moneyness) dynamics: implied
    vol at each log-moneyness level k is unchanged as time passes.

    - SviRawState: scale a and b by T_new / T_old so that w_new(k) / T_new = w_old(k) / T_old.
    - SviJWState:  same transformation applied in raw space, then converted back to JW.
    - BSplineState: coefficients are unchanged.
    """

    def roll(self, graph: Graph, dt: float) -> Graph:
        new_nodes: dict = {}
        for nid, state in graph.nodes.items():
            T_old = state.T
            T_new = T_old - dt
            ratio = T_new / T_old if T_old > 0 else 1.0

            if isinstance(state, BSplineState):
                new_state = state.with_T(T_new)

            elif isinstance(state, SviJWState):
                try:
                    raw = state.to_raw()
                    new_raw = SviRawState(
                        a=raw.a * ratio, b=raw.b * ratio,
                        rho=raw.rho, m=raw.m, sigma=raw.sigma,
                        T=T_new, precision=state.precision,
                    )
                    new_state = new_raw.to_jw(T_new)
                except (ValueError, FloatingPointError):
                    new_state = state.with_T(T_new)

            elif isinstance(state, SviRawState):
                new_state = SviRawState(
                    a=state.a * ratio, b=state.b * ratio,
                    rho=state.rho, m=state.m, sigma=state.sigma,
                    T=T_new, precision=state.precision,
                )
            else:
                new_state = state.copy()

            new_nodes[nid] = decay_precision(new_state, dt)

        return Graph(graph.date + pd.Timedelta(days=round(dt * 365)), new_nodes, graph.edges)
