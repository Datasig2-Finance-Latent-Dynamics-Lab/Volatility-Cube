from __future__ import annotations

from abc import abstractmethod
import numpy as np

from dgraph.source.state import State


class CurveState(State):
    """General State for a implied volatility curve."""

    @abstractmethod
    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """TODO.

        Args:
            k: TODO.

        Returns:
            TODO.
        """
        ...

    def implied_vol(self, k: float | np.ndarray) -> float | np.ndarray:
        """TODO.

        Args:
            k: TODO.

        Returns:
            TODO.
        """
        return np.sqrt(self.total_variance(k) / self.T)

    def call_price(
        self,
        k: float | np.ndarray,
        forward: float = 1.0,
        discount: float = 1.0,
    ) -> np.ndarray:
        """TODO.

        Args:
            k: TODO.
            forward: TODO.
            discount: TODO.

        Returns:
            TODO.
        """
        from utils.pricing import bs_call_from_iv
        return bs_call_from_iv(self.implied_vol(k), k, self.T, forward, discount)

    def plot(
        self,
        k_min: float = -0.5,
        k_max: float = 0.5,
        n_points: int = 300,
        ax=None,
        pct: bool = True,
        **kwargs,
    ):
        """TODO.

        Args:
            k_min: TODO.
            k_max: TODO.
            n_points: TODO.
            ax: TODO.
            pct: TODO.
            **kwargs: TODO.

        Returns:
            TODO.
        """
        import matplotlib.pyplot as plt

        k_grid = np.linspace(k_min, k_max, n_points)
        iv = self.implied_vol(k_grid)
        if pct:
            iv = iv * 100

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(k_grid, iv, **kwargs)
        return ax
