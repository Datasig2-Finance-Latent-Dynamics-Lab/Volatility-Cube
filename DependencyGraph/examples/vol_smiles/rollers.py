import numpy as np
import pandas as pd

from ...source.core.graph import Graph
from ...source.core.roller import Roller
from .nodes import CurveNode
from .curves.bspline import BSplineState
from .curves.svi import SviRawState, SviJWState

# TODO: The roller class may need to be rethough, including a different way of rolling for each state
# I dont like having 30 different if isinstance here, will be reworked.


class VolRoller(Roller):
    
    """
    Evolves state by simply changing the time to expiry T and nothing else.
    """

    def roll(self, graph: Graph, dt: float) -> Graph:
        dt_years = dt / 365
        new_nodes: dict = {}
        for nid, state in graph.nodes.items():
            new_nid = nid.advance(dt_years) if isinstance(nid, CurveNode) else nid

            if isinstance(state, BSplineState):
                new_nodes[new_nid] = state.with_T(new_nid.T)

            elif isinstance(state, SviJWState):
                new_nodes[new_nid] = state.to_raw().to_jw(new_nid.T)

            else:
                new_nodes[new_nid] = state.copy()
                
        return Graph(graph.date + pd.Timedelta(days=dt), new_nodes, graph.dependencies)


class StickyStrikeRoller(Roller):
    # TODO: I don't think this is correct.
    
    """
    Temporal prior assuming sticky-strike dynamics: implied vol at each absolute
    strike K is unchanged as time passes.

    In log-moneyness space, this means total variance w(k) is
    unchanged, so implied vol changes as √(w/T_new).

    - SviRawState: parameters (a, b, rho, m, sigma) are carried forward unchanged.
    - SviJWState:  total variance preserved; T updated; v and v_tilde scale by T_old/T_new.
    - BSplineState: coefficients scaled by √(T_old / T_new).
    """

    def roll(self, graph: Graph, dt: float) -> Graph:
        dt_years = dt / 365
        new_nodes: dict = {}
        for nid, state in graph.nodes.items():
            new_nid = nid.advance(dt_years) if isinstance(nid, CurveNode) else nid
            if isinstance(state, BSplineState):
                T_old = state.T
                T_new = new_nid.T
                scale = np.sqrt(T_old / T_new) if T_new > 0 else 1.0
                new_state = state.with_T(T_new)
                new_state = new_state.from_parameters(new_state.coeffs * scale)
                new_nodes[new_nid] = new_state

            elif isinstance(state, SviJWState):
                new_nodes[new_nid] = state.to_raw().to_jw(new_nid.T)

            elif isinstance(state, SviRawState):
                new_nodes[new_nid] = state.copy()

            else:
                new_nodes[new_nid] = state.copy()
        return Graph(graph.date + pd.Timedelta(days=dt), new_nodes, graph.dependencies)


class StickyDeltaRoller(Roller):
    
    """
    Temporal prior assuming sticky-delta (sticky-moneyness) dynamics: implied
    vol at each log-moneyness level k is unchanged as time passes.

    - SviRawState: scale a and b by T_new / T_old so that w_new(k) / T_new = w_old(k) / T_old.
    - SviJWState:  same transformation applied in raw space, then converted back to JW.
    - BSplineState: coefficients are unchanged.
    """

    def roll(self, graph: Graph, dt: float) -> Graph:
        dt_years = dt / 365
        new_nodes: dict = {}
        for nid, state in graph.nodes.items():
            new_nid = nid.advance(dt_years) if isinstance(nid, CurveNode) else nid

            if isinstance(state, BSplineState):
                new_nodes[new_nid] = state.with_T(new_nid.T)

            elif isinstance(state, SviJWState):
                T_old = nid.T if isinstance(nid, CurveNode) else 1.0
                T_new = new_nid.T if isinstance(new_nid, CurveNode) else 1.0
                ratio = T_new / T_old if T_old > 0 else 1.0
                raw = state.to_raw()
                new_raw = SviRawState(
                    a=raw.a * ratio, b=raw.b * ratio,
                    rho=raw.rho, m=raw.m, sigma=raw.sigma,
                )
                new_nodes[new_nid] = new_raw.to_jw(T_new)

            elif isinstance(state, SviRawState):
                T_old = nid.T if isinstance(nid, CurveNode) else 1.0
                T_new = new_nid.T if isinstance(new_nid, CurveNode) else 1.0
                ratio = T_new / T_old if T_old > 0 else 1.0
                new_nodes[new_nid] = SviRawState(
                    a=state.a * ratio, b=state.b * ratio,
                    rho=state.rho, m=state.m, sigma=state.sigma,
                )
            else:
                new_nodes[new_nid] = state.copy()
        return Graph(graph.date + pd.Timedelta(days=dt), new_nodes, graph.dependencies)
