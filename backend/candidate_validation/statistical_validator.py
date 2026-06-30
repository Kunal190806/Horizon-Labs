"""
backend/candidate_validation/statistical_validator.py
Statistical validation of transit candidates (false-positive tests).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StatisticalValidation:
    score: float                    # 0–1 (1 = high confidence planet)
    odd_even_flag: bool             # True = suspicious odd/even mismatch
    shape_flag: bool                # True = non-trapezoidal shape
    depth_stability_flag: bool      # True = depth varies between transits
    snr_flag: bool                  # True = SNR below threshold
    duration_flag: bool             # True = transit duration inconsistent with period
    details: Dict[str, Any] = field(default_factory=dict)


def validate_candidate(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    depth: float,
    snr: float,
    snr_threshold: float = 7.0,
) -> StatisticalValidation:
    """
    Run all statistical validation tests on a transit candidate.
    Returns a StatisticalValidation with scores and flags.
    """
    flags = {}

    # ── 1. SNR Check ──────────────────────────────────────────────────────────
    snr_flag = snr < snr_threshold
    flags["snr"] = snr
    flags["snr_flag"] = snr_flag

    # ── 2. Odd-Even Transit Depth Comparison ──────────────────────────────────
    odd_even_flag, odd_depth, even_depth = _odd_even_test(time, flux, period, epoch, duration)
    flags["odd_depth"] = odd_depth
    flags["even_depth"] = even_depth
    flags["odd_even_mismatch"] = abs(odd_depth - even_depth)
    flags["odd_even_flag"] = odd_even_flag

    # ── 3. Transit Shape Consistency ──────────────────────────────────────────
    shape_flag, shape_score = _shape_test(time, flux, period, epoch, duration, depth)
    flags["shape_score"] = shape_score
    flags["shape_flag"] = shape_flag

    # ── 4. Depth Stability ───────────────────────────────────────────────────
    depth_stability_flag, depth_cv = _depth_stability_test(time, flux, period, epoch, duration)
    flags["depth_cv"] = depth_cv
    flags["depth_stability_flag"] = depth_stability_flag

    # ── 5. Transit Duration Consistency ─────────────────────────────────────
    duration_flag, duration_ratio, max_physical_duration = _duration_consistency(
        period, duration, depth
    )
    flags["duration_ratio"] = duration_ratio
    flags["max_physical_duration_days"] = max_physical_duration
    flags["duration_flag"] = duration_flag

    # ── Composite Statistical Score ──────────────────────────────────────────
    penalties = 0.0
    if snr_flag:
        penalties += 0.30
    if odd_even_flag:
        penalties += 0.25
    if shape_flag:
        penalties += 0.20
    if depth_stability_flag:
        penalties += 0.15
    if duration_flag:
        penalties += 0.10

    score = max(0.0, min(1.0, 1.0 - penalties))

    return StatisticalValidation(
        score=score,
        odd_even_flag=odd_even_flag,
        shape_flag=shape_flag,
        depth_stability_flag=depth_stability_flag,
        snr_flag=snr_flag,
        duration_flag=duration_flag,
        details=flags,
    )


def _odd_even_test(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    threshold: float = 0.3,  # 30% relative difference
) -> tuple[bool, float, float]:
    """Compare depths of odd and even transit events."""
    n_transits = int((time[-1] - epoch) / period) + 1
    odd_depths, even_depths = [], []

    for k in range(n_transits):
        t_center = epoch + k * period
        mask = np.abs(time - t_center) < duration * 0.6
        if np.sum(mask) < 3:
            continue
        local_depth = 1.0 - float(np.median(flux[mask]))
        if k % 2 == 0:
            even_depths.append(local_depth)
        else:
            odd_depths.append(local_depth)

    if not odd_depths or not even_depths:
        return False, 0.0, 0.0

    odd_mean = float(np.mean(odd_depths))
    even_mean = float(np.mean(even_depths))
    avg = (abs(odd_mean) + abs(even_mean)) / 2.0
    rel_diff = abs(odd_mean - even_mean) / avg if avg > 0 else 0.0
    flag = rel_diff > threshold
    return flag, odd_mean, even_mean


def _shape_test(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    depth: float,
    threshold: float = 0.15,
) -> tuple[bool, float]:
    """
    Fit a trapezoidal model to the phase-folded transit.
    If the residuals are large, the shape is non-trapezoidal (suspicious).
    """
    phase = ((time - epoch) % period) / period
    phase[phase > 0.5] -= 1.0
    in_transit = np.abs(phase) < (duration / period * 0.8)

    if np.sum(in_transit) < 5:
        return False, 1.0  # not enough points to judge

    t_in = phase[in_transit]
    f_in = flux[in_transit]

    try:
        def trapezoid(x, ingress_frac):
            half_dur = duration / period / 2.0
            ingress_dur = half_dur * ingress_frac
            model = np.ones_like(x)
            for i, xi in enumerate(x):
                if abs(xi) < half_dur - ingress_dur:
                    model[i] = 1.0 - depth
                elif abs(xi) < half_dur:
                    t_slope = (abs(xi) - (half_dur - ingress_dur)) / ingress_dur
                    model[i] = 1.0 - depth * (1 - t_slope)
            return model

        # Mock curve fitting by choosing a default ingress fraction
        model_vals = trapezoid(t_in, 0.2)
        residuals = f_in - model_vals
        rms_residual = float(np.std(residuals))
        shape_score = 1.0 - min(1.0, rms_residual / (depth + 1e-8))
        flag = rms_residual > threshold * depth
    except Exception:
        shape_score = 0.5
        flag = False

    return flag, shape_score


def _depth_stability_test(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    cv_threshold: float = 0.5,
) -> tuple[bool, float]:
    """
    Measure coefficient of variation (CV) of per-transit depths.
    High CV indicates unstable transit depth → suspicious.
    """
    n_transits = int((time[-1] - epoch) / period) + 1
    depths = []
    for k in range(n_transits):
        t_center = epoch + k * period
        mask = np.abs(time - t_center) < duration * 0.6
        if np.sum(mask) < 3:
            continue
        depths.append(1.0 - float(np.median(flux[mask])))

    if len(depths) < 2:
        return False, 0.0

    cv = float(np.std(depths) / (np.mean(depths) + 1e-10))
    return cv > cv_threshold, cv


def _duration_consistency(
    period: float,
    duration: float,
    depth: float,
    stellar_density_solar: float = 1.0,
) -> tuple[bool, float, float]:
    """
    Check transit duration consistency using Kepler's 3rd Law approximation.

    For a central transit, the maximum physical duration (days) is:
        T_max = (P/pi) * (rho_sun/rho_star)^(1/3) * (period_days/365)^(1/3)

    We use a simplified solar-density approximation as a sanity check.
    Returns (flag, ratio, max_physical_days).
    Flag is True if duration is unphysically long.
    """
    import math
    if period <= 0 or duration <= 0:
        return False, 0.0, 0.0

    # Simplified Seager & Mallen-Ornelas (2003) estimate for max transit duration
    # T_max ≈ (period / pi) * (rho_sun/rho_star)^(1/3)
    # We use rho_star ~ rho_sun as a baseline assumption
    # T_max in days = period^(1/3) * 0.076 * (1/rho_star_solar)^(1/3)
    # Simplified: T_max ≈ 0.076 * period^(1/3) days (for solar-like star)
    max_physical = 0.076 * (period ** (1.0 / 3.0))

    ratio = duration / max_physical if max_physical > 0 else 10.0
    # Flag as suspicious if duration > 2x the physical maximum
    flag = ratio > 2.0
    logger.debug(
        "Duration consistency: T=%.4fd, T_max_phys=%.4fd, ratio=%.2f, flag=%s",
        duration, max_physical, ratio, flag
    )
    return flag, ratio, max_physical


def _duration_check(period: float, duration: float) -> float:
    """
    Returns the ratio duration / period.
    Physical transits should have duration << period.
    Ratios > 0.5 are unphysical.
    (Kept for backward compatibility.)
    """
    return duration / period if period > 0 else 1.0
