import pandas as pd
import numpy as np

from .nodes import CurveNode
from dgraph.source.edge import DeltaEdgeState

# Factories to build edges.

def build_tiered_smile_edges(
    underlyings: list[str],
    expiries: list[pd.Timestamp],
    spy_to_stock: float = 0.8,
    stock_to_spy: float = 0.2,
    stock_to_stock: float = 0.4,
) -> dict:
    """
    Connect same-expiry nodes across underlyings with tiered DeltaEdgeState precision.
    in this cases DeltEdgeState just stores a scalar.

    Precision weights encode the idea that SPY leads individual stocks:
      SPY to stock : high coupling  (SPY moves drive stock moves)
      stock to SPY : low coupling   (individual stocks have little effect on SPY)
      stock to stock : medium coupling

    """
    edges: dict = {}
    for expiry in expiries:
        nodes = [CurveNode(u, expiry) for u in underlyings]
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                u1, u2 = n1.underlying, n2.underlying
                if u1 == "SPY":
                    w12, w21 = spy_to_stock, stock_to_spy
                elif u2 == "SPY":
                    w12, w21 = stock_to_spy, spy_to_stock
                else:
                    w12 = w21 = stock_to_stock
                edges[(n1, n2)] = DeltaEdgeState(precision=w12)
                edges[(n2, n1)] = DeltaEdgeState(precision=w21)
    return edges


# Factored f * g edges

def build_factored_smile_edges(
    underlyings: list[str],
    expiries: list[pd.Timestamp],
    lambda_f: float = 1.0,
    min_precision: float = 0.2,
    spy_to_stock: float = 0.5,
    stock_to_spy: float = 0.1,
    stock_to_stock: float = 0.2,
) -> dict:
    """
    Attempt for nodes to depend on each other as in tiered edges but also including distance between maturities.
    """
    edges: dict = {}

    for j, expiry1 in enumerate(expiries):
        for expiry2 in expiries[j + 1:]:
            nodes1 = [CurveNode(u, expiry1) for u in underlyings]
            nodes2 = [CurveNode(u, expiry2) for u in underlyings]

            for n1 in nodes1:
                for n2 in nodes2:
                    dt_days = abs((n1.expiry - n2.expiry).days)
                    f = float(np.exp(-lambda_f * (dt_days / 365.0)))

                    if f < min_precision:
                        f = 0.0

                    u1, u2 = n1.underlying, n2.underlying
                    if u1 == "SPY":
                        w12, w21 = spy_to_stock, stock_to_spy
                    elif u2 == "SPY":
                        w12, w21 = stock_to_spy, spy_to_stock
                    else:
                        w12 = w21 = stock_to_stock

                    edges[(n1, n2)] = DeltaEdgeState(precision=w12 * f)
                    edges[(n2, n1)] = DeltaEdgeState(precision=w21 * f)

    return edges