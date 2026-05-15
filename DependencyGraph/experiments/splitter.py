from __future__ import annotations

from abc import ABC, abstractmethod
import random

from DependencyGraph.source.node import NodeId
from DependencyGraph.source.observation import Observation, ObservationSet


class Splitter(ABC):
    """
    Abstract data class for splitting observations.
    """
    @abstractmethod
    def split(self, obs_set: ObservationSet) -> tuple[ObservationSet, ObservationSet]:
        """Performs the splitting of observation sets into two different ones."""
        ...


class RandomSplitter(Splitter):
    """
    Splits observations randmoly with no further restrictions.
    """
    def __init__(self, train_frac: float = 0.5, seed: int = 67):
        self.train_frac = train_frac
        self.seed = seed

    def split(self, obs_set: ObservationSet) -> tuple[ObservationSet, ObservationSet]:
        rng = random.Random(self.seed)
        obs = list(obs_set.observations)
        rng.shuffle(obs)
        n_train = max(1, int(len(obs) * self.train_frac))
        train = ObservationSet(obs[:n_train], obs_set.date)
        test = ObservationSet(obs[n_train:], obs_set.date)
        return train, test


class NodeMaskingSplitter(Splitter):
    """
    Splits observations by node first (by maskin each node with a certain probability),
    and then gets random observations on the remaining nodes.
    """

    def __init__(
        self,
        node_mask_prob: float = 0.3,
        train_frac: float = 0.7,
        seed: int = 67,
    ):
        self.node_mask_prob = node_mask_prob
        self.train_frac = train_frac
        self.seed = seed
        self.masked_nodes: set[NodeId] = set()

    def split(self, obs_set: ObservationSet) -> tuple[ObservationSet, ObservationSet]:
        date_seed = self.seed + int(obs_set.date.value // 10**9) # Keeps things reproducible.
        rng = random.Random(date_seed)

        # mask the nodes
        node_ids = list(obs_set._by_node.keys())
        self.masked_nodes = {
            nid for nid in node_ids if rng.random() < self.node_mask_prob
        }

        train_obs: list[Observation] = []
        test_obs:  list[Observation] = list(obs_set.observations)

        for nid, node_obs in obs_set._by_node.items():
            if nid in self.masked_nodes:
                continue
            shuffled = list(node_obs)
            rng.shuffle(shuffled)
            n_train = max(1, int(len(shuffled) * self.train_frac))
            train_obs.extend(shuffled[:n_train])

        return (
            ObservationSet(train_obs, obs_set.date),
            ObservationSet(test_obs,  obs_set.date),
        )
