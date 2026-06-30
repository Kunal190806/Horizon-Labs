"""
backend/utils/time_utils.py
Time system conversion utilities for TESS / Kepler data.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# TESS reference epoch (BJD - BTJD offset)
TESS_BTJD_OFFSET = 2_457_000.0

# Kepler reference epoch offset
KEPLER_BKJD_OFFSET = 2_454_833.0


def btjd_to_bjd(btjd: np.ndarray) -> np.ndarray:
    """Convert TESS Barycentric Julian Date (BTJD) to full BJD."""
    return btjd + TESS_BTJD_OFFSET


def bjd_to_btjd(bjd: np.ndarray) -> np.ndarray:
    """Convert full BJD to TESS BTJD."""
    return bjd - TESS_BTJD_OFFSET


def bkjd_to_bjd(bkjd: np.ndarray) -> np.ndarray:
    """Convert Kepler Barycentric Julian Date (BKJD) to full BJD."""
    return bkjd + KEPLER_BKJD_OFFSET


def bjd_to_iso(bjd: float) -> str:
    """
    Convert a BJD value to an approximate ISO-8601 date string.
    Uses astropy if available, falls back to a rough approximation.
    """
    try:
        import importlib
        astropy_time = importlib.import_module("astropy.time")
        t = astropy_time.Time(bjd, format="bjd", scale="tdb")
        return t.iso
    except (ImportError, AttributeError, ModuleNotFoundError):
        # Approximate: BJD 2451545.0 ≈ J2000.0 = 2000-01-01T12:00:00
        j2000_bjd = 2_451_545.0
        days_from_j2000 = bjd - j2000_bjd
        # Very rough: 365.25 days/year
        year = 2000 + days_from_j2000 / 365.25
        return f"~{year:.2f} (approx)"


def estimate_cadence(time: np.ndarray) -> float:
    """
    Estimate the median observation cadence from a time array.
    Returns cadence in days.
    """
    if len(time) < 2:
        return 0.0
    return float(np.nanmedian(np.diff(time)))


def gap_mask(time: np.ndarray, gap_threshold_factor: float = 3.0) -> np.ndarray:
    """
    Return a boolean array marking positions immediately after gaps.
    A gap is defined as a time step > gap_threshold_factor * median_cadence.

    Returns a boolean array of length len(time) - 1.
    """
    dt = np.diff(time)
    median_dt = float(np.nanmedian(dt))
    return dt > gap_threshold_factor * median_dt
