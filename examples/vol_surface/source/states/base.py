from __future__ import annotations

from abc import abstractmethod
import numpy as np

from DependencyGraph.source.state import State


class SurfaceState(State):
    """
    State for a full implied-volatility surface parameterised in absolute (k, T) coordinates.

    Unlike CurveState, T is a coordinate of the surface, not a field of the state.
    This means SurfaceState parameters are static with respect to calendar time —
    the node identity never changes, which makes cross-asset dependency learning tractable.
    """

    @abstractmethod
    def total_variance(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """Total variance w(k, T) = σ²(k, T) · T."""
        ...

    def implied_vol(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        T_arr = np.asarray(T, dtype=float)
        w = self.total_variance(k, T_arr)
        return np.sqrt(np.maximum(w, 1e-12) / np.maximum(T_arr, 1e-12))

    def plot(
        self,
        T_values: list[float],
        k_min: float = -0.5,
        k_max: float = 0.5,
        n_points: int = 200,
        ax=None,
        pct: bool = True,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        k_grid = np.linspace(k_min, k_max, n_points)
        if ax is None:
            _, ax = plt.subplots()
        for T in T_values:
            iv = self.implied_vol(k_grid, T)
            if pct:
                iv = iv * 100
            ax.plot(k_grid, iv, label=f"T={T:.2f}", **kwargs)
        ax.legend()
        return ax
