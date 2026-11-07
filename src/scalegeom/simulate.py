"""Stationary Gaussian random fields by spectral (circulant embedding) synthesis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CovModel:
    """Isotropic covariance C(h) = sill * rho(h / length) (+ nugget at h = 0).

    model "gaussian":    rho = exp(-h^2 / (2 L^2))   (L = correlation length)
    model "exponential": rho = exp(-h / L)
    model "spherical":   rho = 1 - 1.5 (h/L) + 0.5 (h/L)^3 for h < L, else 0
    """

    model: str = "gaussian"
    sill: float = 1.0
    length: float = 0.1
    nugget: float = 0.0

    def rho(self, h):
        h = np.asarray(h, dtype=float) / self.length
        if self.model == "gaussian":
            return np.exp(-0.5 * h ** 2)
        if self.model == "exponential":
            return np.exp(-h)
        if self.model == "spherical":
            return np.where(h < 1, 1 - 1.5 * h + 0.5 * h ** 3, 0.0)
        raise ValueError(f"unknown covariance model {self.model!r}")

    def cov(self, h):
        h = np.asarray(h, dtype=float)
        return self.sill * self.rho(h) + self.nugget * (h == 0)

    def gamma(self, h):
        h = np.asarray(h, dtype=float)
        return (self.nugget + self.sill) * (h > 0) - self.sill * self.rho(h) * (h > 0)

    def practical_range(self, tau: float = 0.1) -> float:
        """Lag at which the correlation drops to tau."""
        if self.model == "gaussian":
            return self.length * np.sqrt(-2 * np.log(tau))
        if self.model == "exponential":
            return -self.length * np.log(tau)
        hs = np.linspace(0, self.length, 10_001)
        return float(hs[np.argmax(self.rho(hs) <= tau)])


def simulate_grid(model: CovModel, shape, spacing: float, rng=None, pad: int = 2) -> np.ndarray:
    """One realization on a regular grid with the given cell spacing
    (in the same units as model.length), by circulant embedding."""
    rng = np.random.default_rng(rng)
    nr, nc = shape
    mr, mc = pad * nr, pad * nc
    dr = np.minimum(np.arange(mr), mr - np.arange(mr)) * spacing
    dc = np.minimum(np.arange(mc), mc - np.arange(mc)) * spacing
    H = np.hypot(dr[:, None], dc[None, :])
    lam = np.real(np.fft.fft2(model.sill * model.rho(H)))
    lam[lam < 0] = 0.0
    eps = rng.standard_normal((mr, mc)) + 1j * rng.standard_normal((mr, mc))
    f = np.real(np.fft.fft2(np.sqrt(lam / (mr * mc)) * eps))[:nr, :nc]
    if model.nugget > 0:
        f = f + rng.normal(0, np.sqrt(model.nugget), f.shape)
    return f


def simulate_points(model: CovModel, xy: np.ndarray, rng=None, exact_max: int = 3000,
                    grid_max: int = 512) -> np.ndarray:
    """One realization at arbitrary points: exact (Cholesky) for small n,
    otherwise a fine grid covering the points, sampled at the nearest cell."""
    rng = np.random.default_rng(rng)
    n = len(xy)
    if n <= exact_max:
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        K = model.sill * model.rho(d) + 1e-10 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(K)
            L = V * np.sqrt(np.clip(w, 0, None))
        f = L @ rng.standard_normal(n)
    else:
        lo = xy.min(axis=0)
        span = (xy.max(axis=0) - lo).max()
        spacing = max(span / (grid_max - 1), min(model.length / 4.0, span / 64))
        nr = int(np.ceil((xy[:, 1].max() - lo[1]) / spacing)) + 1
        nc = int(np.ceil((xy[:, 0].max() - lo[0]) / spacing)) + 1
        g = simulate_grid(CovModel(model.model, model.sill, model.length, 0.0), (nr, nc), spacing, rng)
        r = np.clip(np.rint((xy[:, 1] - lo[1]) / spacing).astype(int), 0, nr - 1)
        c = np.clip(np.rint((xy[:, 0] - lo[0]) / spacing).astype(int), 0, nc - 1)
        f = g[r, c]
    if model.nugget > 0:
        f = f + rng.normal(0, np.sqrt(model.nugget), n)
    return f


def gaussian_random_field(shape=(160, 160), corr_length: float = 12.0, seed=None,
                          model: str = "gaussian", normalize: str = "minmax") -> np.ndarray:
    """Autocorrelated field on a grid; corr_length in cells.

    Defaults reproduce the setup of Vaz (2026), Section 9.1: a 160 x 160 grid,
    correlation length 12 cells, values rescaled to [0, 1]."""
    f = simulate_grid(CovModel(model, 1.0, float(corr_length)), shape, 1.0, seed)
    return _normalize(f, normalize)


def mixed_scale_field(shape=(160, 160), lengths=(3.0, 30.0), weights=(1.0, 1.0), seed=None,
                      model: str = "gaussian", normalize: str = "minmax") -> np.ndarray:
    """Superposition of independent fields with well-separated correlation
    lengths (Scenario B of Vaz 2026, Section 9.8)."""
    rng = np.random.default_rng(seed)
    f = sum(w * simulate_grid(CovModel(model, 1.0, float(L)), shape, 1.0, rng)
            for L, w in zip(lengths, weights))
    return _normalize(f, normalize)


def _normalize(f, how):
    if how == "minmax":
        return (f - f.min()) / (f.max() - f.min())
    if how == "zscore":
        return (f - f.mean()) / f.std()
    return f
