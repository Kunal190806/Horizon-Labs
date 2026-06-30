"""
backend/preprocessing/outlier_removal.py
Sigma-clipping and IQR-based outlier removal for light curves.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OutlierResult:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None
    mask: np.ndarray          # True where data was KEPT
    num_removed: int
    method: str


def sigma_clip_outliers(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    sigma: float = 4.0,
    maxiters: int = 5,
) -> OutlierResult:
    """
    Remove flux outliers using iterative sigma-clipping (astropy).

    Parameters
    ----------
    time, flux : arrays of equal length
    flux_err   : optional uncertainty array
    sigma      : rejection threshold in standard deviations
    maxiters   : maximum clipping iterations
    """
    good_mask = np.ones_like(flux, dtype=bool)
    for _ in range(maxiters):
        current_flux = flux[good_mask]
        if len(current_flux) == 0:
            break
        mean = np.mean(current_flux)
        std = np.std(current_flux)
        new_mask = np.abs(flux - mean) <= sigma * std
        if np.array_equal(new_mask, good_mask):
            break
        good_mask = new_mask

    num_removed = int(np.sum(~good_mask))

    result = OutlierResult(
        time=time[good_mask],
        flux=flux[good_mask],
        flux_err=flux_err[good_mask] if flux_err is not None else None,
        mask=good_mask,
        num_removed=num_removed,
        method="sigma_clip",
    )
    logger.debug("Sigma-clip: removed %d outliers (sigma=%.1f)", num_removed, sigma)
    return result


def iqr_outliers(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    iqr_factor: float = 5.0,
) -> OutlierResult:
    """
    Remove outliers using the Inter-Quartile Range method.
    Good alternative when distribution is non-Gaussian.
    """
    q25, q75 = np.percentile(flux, [25, 75])
    iqr = q75 - q25
    lower = q25 - iqr_factor * iqr
    upper = q75 + iqr_factor * iqr
    good_mask = (flux >= lower) & (flux <= upper)

    num_removed = int(np.sum(~good_mask))

    result = OutlierResult(
        time=time[good_mask],
        flux=flux[good_mask],
        flux_err=flux_err[good_mask] if flux_err is not None else None,
        mask=good_mask,
        num_removed=num_removed,
        method="iqr",
    )
    logger.debug("IQR-clip: removed %d outliers (factor=%.1f)", num_removed, iqr_factor)
    return result


def remove_nans(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Remove NaN and Inf values from all arrays."""
    mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)
    n_removed = int(np.sum(~mask))
    return time[mask], flux[mask], (flux_err[mask] if flux_err is not None else None), n_removed
