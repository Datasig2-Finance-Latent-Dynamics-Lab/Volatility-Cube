"""
Builds Graph objects and ObservationSets from a raw options DataFrame.

Each row of the DataFrame is expected to have columns:
  date, underlying, expiry, dte, T, logmoneyness, iv, weight
"""
import pandas as pd

from DependencyGraph.source.graph import Graph
from DependencyGraph.source.edge import DeltaEdgeState
from DependencyGraph.source.observation import Observation, ObservationSet
from .nodes import CurveNode
from .curves.base import CurveState
from .curves.svi import SviRawState, fit_svi


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
                nid = CurveNode(underlying, expiry)
                for _, row in slice_df.iterrows():
                    observations.append(
                        Observation(nid, (row["logmoneyness"], row["iv"]), float(row["weight"]))
                    )

        return ObservationSet(observations, date)


class GraphFactory:
    """Builds a Graph by fitting one curve state per (underlying, expiry) pair."""

    def __init__(
        self,
        underlyings: list[str],
        expiries: list[pd.Timestamp],
        edges: dict,
        fit_fn=None,
    ):
        self.underlyings = underlyings
        self.expiries = expiries
        self.edges = edges
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
                nodes[CurveNode(underlying, expiry)] = state

        return Graph(date, nodes, self.edges)


def build_cross_asset_edges(
    underlyings: list[str],
    expiries: list[pd.Timestamp],
    df: pd.DataFrame,
    date: pd.Timestamp,
    weight: float = 1.0,
) -> dict:
    """
    Connects same-expiry nodes across different underlyings with a uniform scalar precision.
    Edges are symmetric and directed both ways.
    """
    day_df = df[df["date"] == date]
    edges: dict = {}

    for expiry in expiries:
        nodes_for_expiry: list[CurveNode] = []
        for underlying in underlyings:
            mask = (day_df["underlying"] == underlying) & (day_df["expiry"] == expiry)
            if not day_df[mask].empty:
                nodes_for_expiry.append(CurveNode(underlying, expiry))

        for i, n1 in enumerate(nodes_for_expiry):
            for n2 in nodes_for_expiry[i + 1:]:
                edge = DeltaEdgeState(precision=weight)
                edges[(n1, n2)] = edge
                edges[(n2, n1)] = edge

    return edges
