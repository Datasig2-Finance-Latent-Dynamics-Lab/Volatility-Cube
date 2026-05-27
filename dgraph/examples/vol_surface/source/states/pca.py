from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .base import SurfaceState


class VolSurfacePCA:
    """
    Functional PCA on total-variance surfaces evaluated on a common (k, T) grid.

    Each surface is represented as a vector of total-variance values w(k_i, T_j)
    on a regular grid.  Standard PCA is applied across those vectors, yielding
    principal components that are piece-wise-linear 'basis surfaces'.

    Storing total variance (w = σ²T) rather than implied vol avoids the 1/√T
    singularity near T = 0.

    Attributes
    ----------
    k_grid : (n_k,)   – log-moneyness grid points (sorted)
    T_grid : (n_T,)   – maturity grid points in years (sorted)
    mean_  : (n_k*n_T,)
    components_ : (n_components, n_k*n_T)
    explained_variance_ratio_ : (n_components,)
    """

    def __init__(
        self,
        k_grid: np.ndarray,
        T_grid: np.ndarray,
        n_components: int = 5,
    ) -> None:
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.n_components = n_components
        self.mean_: np.ndarray = np.zeros(len(k_grid) * len(T_grid))
        self.components_: np.ndarray = np.zeros((n_components, len(k_grid) * len(T_grid)))
        self.explained_variance_ratio_: np.ndarray = np.zeros(n_components)

    def fit(self, X: np.ndarray) -> "VolSurfacePCA":
        """
        Fit PCA on a matrix of surface evaluations.

        Parameters
        ----------
        X : (n_obs, n_k * n_T)
            Each row is a flattened total-variance surface evaluated on the grid.
        """
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        n_comp = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:n_comp]
        self.n_components = n_comp
        var = (s ** 2) / max(len(X) - 1, 1)
        self.explained_variance_ratio_ = var[:n_comp] / var.sum()
        return self

    def transform(self, surface_vec: np.ndarray) -> np.ndarray:
        """Project a flattened surface (n_k*n_T,) to PCA coefficients (n_components,)."""
        return (surface_vec - self.mean_) @ self.components_.T

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct a flattened surface from PCA coefficients."""
        return self.mean_ + coeffs @ self.components_


class PCASurfaceState(SurfaceState):
    """
    Vol surface state parameterised by functional PCA coefficients.

    The shared ``pca`` object holds the basis (mean + principal components).
    A state is fully described by its coefficient vector; ``total_variance``
    reconstructs the surface on the PCA grid and interpolates linearly to
    arbitrary (k, T) query points.

    Coefficients are unbounded — no arbitrage constraints are enforced
    implicitly by the data loss pulling coefficients toward observable data.
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        pca: VolSurfacePCA,
        precision: float | np.ndarray = 1.0,
    ) -> None:
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.pca = pca
        self._precision = precision

    @property
    def precision(self) -> float | np.ndarray:
        return self._precision

    @precision.setter
    def precision(self, value: float | np.ndarray) -> None:
        self._precision = value

    # ------------------------------------------------------------------
    # State interface
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return len(self.coefficients)

    def parameters(self) -> np.ndarray:
        return self.coefficients.copy()

    def from_parameters(self, params: np.ndarray) -> "PCASurfaceState":
        return PCASurfaceState(np.asarray(params, dtype=float), self.pca, self.precision)

    def copy(self) -> "PCASurfaceState":
        return PCASurfaceState(self.coefficients.copy(), self.pca, self.precision)

    def bounds(self):
        return None  # unbounded in PCA space

    # ------------------------------------------------------------------
    # Surface interface
    # ------------------------------------------------------------------

    def total_variance(
        self, k: float | np.ndarray, T: float | np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct total variance w(k, T) = σ²(k,T)·T from PCA coefficients
        via bilinear interpolation on the fitted grid.

        Points outside the grid are assigned the nearest boundary value (flat
        extrapolation).
        """
        k = np.asarray(k, dtype=float)
        T = np.asarray(T, dtype=float)
        out_shape = np.broadcast(k, T).shape
        k_bc = np.broadcast_to(k, out_shape).ravel()
        T_bc = np.broadcast_to(T, out_shape).ravel()

        w_vec = self.pca.inverse_transform(self.coefficients)
        w_grid = w_vec.reshape(len(self.pca.k_grid), len(self.pca.T_grid))

        k_c = np.clip(k_bc, self.pca.k_grid[0], self.pca.k_grid[-1])
        T_c = np.clip(T_bc, self.pca.T_grid[0], self.pca.T_grid[-1])

        interp = RegularGridInterpolator(
            (self.pca.k_grid, self.pca.T_grid),
            w_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        w = interp(np.stack([k_c, T_c], axis=-1))
        w = np.maximum(w, 1e-12)

        if out_shape == ():
            return float(w[0])
        return w.reshape(out_shape)
