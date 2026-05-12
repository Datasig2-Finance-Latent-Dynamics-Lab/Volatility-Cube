from __future__ import annotations

import numpy as np
import pandas as pd

from .nodes import CurveNode
from ...source.core.dependency import StaticDependencies


class UniformCrossAssetDependencies(StaticDependencies):
    """
    Connects all nodes with a single weight.
    """

    def __init__(
        self,
        underlyings: list[str],
        expiries: list[pd.Timestamp],
        df: pd.DataFrame,
        date: pd.Timestamp,
        weight: float = 1.0,
    ):
        day_df = df[df["date"] == date]
        edges: dict = {}

        for expiry in expiries:
            nodes_for_expiry: list[CurveNode] = []
            for underlying in underlyings:
                mask = (day_df["underlying"] == underlying) & (day_df["expiry"] == expiry)
                slice_df = day_df[mask]
                if slice_df.empty:
                    continue
                T = float(slice_df["T"].iloc[0])
                nodes_for_expiry.append(CurveNode(underlying, expiry, T))

            for i, n1 in enumerate(nodes_for_expiry):
                for n2 in nodes_for_expiry[i + 1:]:
                    edges[(n1, n2)] = weight
                    edges[(n2, n1)] = weight

        super().__init__(edges)


class CorrelationDependencies(StaticDependencies):
    """
    Connects same-expiry nodes with weights derived from pairwise return
    correlations between underlyings.

    The weight for edge at any expiry is set to the absolute correlation
      computed from a returns df, with some minimum threshold.
    """

    def __init__(
        self,
        underlyings: list[str],
        expiries: list[pd.Timestamp],
        df: pd.DataFrame,
        date: pd.Timestamp,
        returns_df: pd.DataFrame,
        min_corr: float = 0.0,
    ):
        corr = returns_df[underlyings].corr().abs()
        day_df = df[df["date"] == date]
        edges: dict = {}

        for expiry in expiries:
            nodes_for_expiry: dict[str, CurveNode] = {}
            for underlying in underlyings:
                mask = (day_df["underlying"] == underlying) & (day_df["expiry"] == expiry)
                slice_df = day_df[mask]
                if slice_df.empty:
                    continue
                T = float(slice_df["T"].iloc[0])
                nodes_for_expiry[underlying] = CurveNode(underlying, expiry, T)

            underlyings_present = list(nodes_for_expiry.keys())
            for i, u1 in enumerate(underlyings_present):
                for u2 in underlyings_present[i + 1:]:
                    rho = float(corr.loc[u1, u2])
                    if rho < min_corr:
                        continue
                    n1, n2 = nodes_for_expiry[u1], nodes_for_expiry[u2]
                    edges[(n1, n2)] = rho
                    edges[(n2, n1)] = rho

        super().__init__(edges)
