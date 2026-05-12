from __future__ import annotations

from abc import abstractmethod
import numpy as np

from ....source.core.state import State

# TODO: Fix the mess with the .T, keep it consistent across curve states.

class CurveState(State):
    """General State for a implied volatility curve."""

    @abstractmethod
    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """
        Use total variance instead of implied volatility SVI outputs this.
        """
        ...

    def implied_vol(self, k: float | np.ndarray, T: float) -> float | np.ndarray:
        return np.sqrt(self.total_variance(k) / T)

    def plot(
        self,
        T: float,
        k_min: float = -0.5,
        k_max: float = 0.5,
        n_points: int = 300,
        ax=None,
        pct: bool = True,
        **kwargs,
    ):
        """
        Plots the implied volatility and adds it to an axis, creates one if no axes given.
        """
        import matplotlib.pyplot as plt

        k_grid = np.linspace(k_min, k_max, n_points)
        iv = self.implied_vol(k_grid, T)
        if pct:
            iv = iv * 100

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(k_grid, iv, **kwargs)
        return ax
