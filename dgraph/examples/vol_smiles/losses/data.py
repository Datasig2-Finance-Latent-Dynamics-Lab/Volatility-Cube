import numpy as np

from dgraph.losses.data import DataLoss
from dgraph.source.observation import ObservationSet
from ..source.nodes import CurveNode


class VolDataLoss(DataLoss):
    """
    Weighted MSE between fitted implied vol and observed implied vol.

    Each Observation.data is expected to be a (k, iv) tuple where
    k is log-moneyness and iv is the observed implied volatility.
    T is read from the CurveNode so it never enters the state vector.
    """

    def _build_node_arrays(self, observations: ObservationSet) -> dict:
        cache: dict[CurveNode, tuple] = {}
        for nid, obs_list in observations._by_node.items():
            if not isinstance(nid, CurveNode):
                continue
            cache[nid] = (
                np.array([o.data[0] for o in obs_list]),   # ks
                np.array([o.data[1] for o in obs_list]),   # iv_obs
                np.array([o.weight  for o in obs_list]),   # weights
            )
        return cache

    def _eval_node(self, state, node_obs: tuple):
        ks, iv_obs, weights = node_obs
        return state.implied_vol(ks), iv_obs, weights
