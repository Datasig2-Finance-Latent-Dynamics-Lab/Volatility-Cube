import numpy as np

from dgraph.losses.data import DataLoss
from dgraph.source.observation import ObservationSet
from ..source.states.base import SurfaceState


class SurfaceDataLoss(DataLoss):
    """
    Weighted MSE between fitted and observed implied vol over the full surface.

    Each Observation.data is expected to be a (k, T, iv) tuple.
    """

    def _build_node_arrays(self, observations: ObservationSet) -> dict:
        cache: dict = {}
        for nid, obs_list in observations._by_node.items():
            cache[nid] = (
                np.array([o.data[0] for o in obs_list]),   # ks
                np.array([o.data[1] for o in obs_list]),   # Ts
                np.array([o.data[2] for o in obs_list]),   # iv_obs
                np.array([o.weight  for o in obs_list]),   # weights
            )
        return cache

    def _eval_node(self, state, node_obs: tuple):
        if not isinstance(state, SurfaceState):
            return None
        ks, Ts, iv_obs, weights = node_obs
        return state.implied_vol(ks, Ts), iv_obs, weights
