"""
backend/utils/array_utils.py
Shared array manipulation utilities used across pipeline modules.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def phase_fold(
    time: np.ndarray,
    period: float,
    epoch: float,
) -> np.ndarray:
    """
    Phase-fold a time array to the range [-0.5, 0.5].

    Args:
        time:   Array of observation times (BJD or BTJD).
        period: Orbital period in days.
        epoch:  Reference transit epoch (BJD or BTJD).

    Returns:
        phase: Array of phase values in [-0.5, 0.5].
    """
    phase = ((time - epoch) % period) / period
    phase[phase > 0.5] -= 1.0
    return phase


def downsample(
    arr: np.ndarray,
    max_points: int,
) -> np.ndarray:
    """
    Downsample an array to at most *max_points* evenly-spaced indices.
    Returns the original array if len(arr) <= max_points.
    """
    if len(arr) <= max_points:
        return arr
    idx = np.round(np.linspace(0, len(arr) - 1, max_points)).astype(int)
    return arr[idx]


def bin_array(
    time: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 201,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin a light curve into *n_bins* equal-width phase bins.
    Returns (bin_centres, bin_means).
    """
    bins = np.linspace(time.min(), time.max(), n_bins + 1)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (time >= bins[i]) & (time < bins[i + 1])
        if mask.sum() > 0:
            bin_means[i] = float(np.nanmedian(flux[mask]))
    # Fill NaN bins by interpolation
    nans = np.isnan(bin_means)
    if nans.any() and (~nans).sum() >= 2:
        bin_means[nans] = np.interp(
            bin_centres[nans],
            bin_centres[~nans],
            bin_means[~nans],
        )
    return bin_centres, bin_means


def compute_snr(
    flux: np.ndarray,
    depth: float,
    num_transits: int = 1,
) -> float:
    """
    Estimate SNR for a transit signal.
    SNR = depth * sqrt(n_in_transit) / rms_out_of_transit
    """
    rms = float(np.nanstd(flux))
    if rms < 1e-12:
        return 0.0
    return abs(depth) * np.sqrt(max(num_transits, 1)) / rms


def robust_std(arr: np.ndarray, axis: Optional[int] = None) -> float:
    """
    Compute a robust standard deviation using the MAD estimator:
        sigma_robust = 1.4826 * median(|x - median(x)|)
    """
    med = np.nanmedian(arr, axis=axis)
    mad = np.nanmedian(np.abs(arr - med), axis=axis)
    return float(1.4826 * mad)


def fill_nans_linear(
    arr: np.ndarray,
) -> np.ndarray:
    """
    Replace NaN values in a 1-D array using linear interpolation.
    Leading/trailing NaNs are filled by nearest-neighbour extrapolation.
    """
    arr = arr.copy()
    x = np.arange(len(arr))
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    valid_x = x[~nans]
    if len(valid_x) < 2:
        arr[nans] = np.nanmean(arr)
        return arr
    arr[nans] = np.interp(x[nans], valid_x, arr[~nans])
    return arr
