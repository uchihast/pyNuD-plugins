"""External conformational-sampling backends for flexible fitting."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class FlexibleFitBackendError(RuntimeError):
    """Raised when an external flexible-fit backend cannot complete."""


class FlexibleFitBackendCanceled(FlexibleFitBackendError):
    """Raised when the user cancels an external backend run."""


@dataclass(frozen=True)
class NolbRunConfig:
    """Configuration for the standalone academic NOLB executable."""

    binary_path: str = "NOLB"
    num_structures: int = 50
    max_rmsd_angstrom: float = 4.0
    cutoff_angstrom: float = 5.0
    minimize: bool = True
    timeout_seconds: float = 600.0


@dataclass(frozen=True)
class NolbRunResult:
    """Files and diagnostics produced by one NOLB ensemble run."""

    ensemble_path: Path
    command: tuple[str, ...]
    output: str
    elapsed_seconds: float


@dataclass(frozen=True)
class NolbProgress:
    """Observable status from a running standalone NOLB process."""

    elapsed_seconds: float
    log_bytes: int
    output_pdb_bytes: int
    last_output_line: str


@dataclass(frozen=True)
class NolbCandidateSafety:
    """Structural checks applied before AFM-scoring a NOLB candidate."""

    accepted: bool
    reason: str
    rms_displacement_nm: float
    peak_displacement_nm: float


@dataclass(frozen=True)
class AfmfitRunConfig:
    """Controls an optional AFMfit installation in a separate Python."""

    python_executable: str
    bridge_path: str
    n_cpu: int = 2
    nmodes: int = 10
    cutoff_angstrom: float = 8.0
    sigma_angstrom: float = 4.0
    angular_distance_deg: float = 10.0
    rigid_angle_limit_deg: float = 25.0
    z_shift_range_angstrom: float = 20.0
    z_shift_points: int = 5
    n_best_views: int = 5
    view_separation_deg: float = 15.0
    iterations: int = 10
    regularization_lambda: float = 25.0
    timeout_seconds: float = 900.0


@dataclass(frozen=True)
class AfmfitProgress:
    """Observable status from the external AFMfit bridge."""

    elapsed_seconds: float
    log_bytes: int
    stage: str
    percent: float
    message: str


@dataclass(frozen=True)
class AfmfitRunResult:
    """Files, metadata, and diagnostics produced by AFMfit."""

    output_pdb_path: Path
    result_path: Path
    command: tuple[str, ...]
    output: str
    elapsed_seconds: float
    metadata: dict


@dataclass(frozen=True)
class NmffRunConfig:
    """Controls the built-in iterative NMFF-AFM search."""

    step_amplitude_nm: float = 0.05
    max_iterations: int = 70
    max_total_rms_nm: float = 1.5
    minimum_cc_gain: float = 1e-4
    convergence_fraction: float = 0.03
    convergence_patience: int = 3
    minimum_iterations: int = 5


@dataclass(frozen=True)
class NmffIteration:
    """One accepted iterative normal-mode deformation."""

    iteration: int
    mode_number: int
    eigenvalue: float
    amplitude_nm: float
    slope: float
    correlation_before: float
    correlation_after: float
    total_rms_nm: float
    nma_method: str


@dataclass(frozen=True)
class NmffRunResult:
    """Coordinates and diagnostics from an iterative NMFF-AFM search."""

    coordinates: np.ndarray
    initial_correlation: float
    final_correlation: float
    evaluations: int
    iterations: tuple[NmffIteration, ...]
    stop_reason: str
    rejected_candidate_count: int = 0


def _normalize_pose_foreground_image(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove ASD row offsets and robustly normalize one AFM image."""
    image = np.asarray(image, dtype=float)
    valid = np.isfinite(image) & (image > -1e8)
    values = image[valid]
    if values.size < 4:
        raise FlexibleFitBackendError(
            "Pose foreground scoring requires at least four valid pixels."
        )
    corrected = image.copy()
    global_baseline = float(np.percentile(values, 20.0))
    for row_index in range(corrected.shape[0]):
        row_valid = valid[row_index]
        row_values = corrected[row_index, row_valid]
        if row_values.size < 2:
            continue
        row_baseline = float(np.percentile(row_values, 20.0))
        corrected[row_index, row_valid] -= row_baseline - global_baseline
    values = corrected[valid]
    low = float(np.percentile(values, 5.0))
    high = float(np.percentile(values, 99.0))
    scale = high - low
    if not np.isfinite(scale) or scale <= 1e-12:
        raise FlexibleFitBackendError(
            "Pose foreground scoring requires non-constant images."
        )
    normalized = np.zeros(image.shape, dtype=float)
    normalized[valid] = np.clip(
        (corrected[valid] - low) / scale,
        0.0,
        1.0,
    )
    return normalized, valid


def prepare_pose_image_correlation_signal(
    image: np.ndarray,
    foreground_threshold: float = 0.18,
) -> np.ndarray:
    """Return an ASD row-corrected molecular image for final correlation."""
    normalized, valid = _normalize_pose_foreground_image(image)
    threshold = float(np.clip(foreground_threshold, 0.05, 0.60))
    signal = np.where(
        valid & (normalized >= threshold),
        normalized - threshold,
        0.0,
    )
    if np.count_nonzero(signal) < 4:
        raise FlexibleFitBackendError(
            "Pose image correlation could not identify molecular foreground."
        )
    return signal


def score_pose_foreground_alignment(
    real_image: np.ndarray,
    simulated_image: np.ndarray,
    foreground_threshold: float = 0.18,
) -> dict[str, float]:
    """Score explicit AFM placement while suppressing flat-background bias.

    Estimate Pose runs before the probe geometry and absolute height scale are
    necessarily known.  Robustly normalizing each image and scoring only the
    union of their molecular foregrounds therefore gives a more useful pose
    objective than whole-frame RMSD, especially when an ASD frame contains
    large flat margins or scan-line background.
    """
    real = np.asarray(real_image, dtype=float)
    simulated = np.asarray(simulated_image, dtype=float)
    if real.ndim != 2 or simulated.shape != real.shape:
        raise FlexibleFitBackendError(
            "Pose foreground scoring requires two equally sized 2D images."
        )

    real_norm, real_valid = _normalize_pose_foreground_image(real)
    sim_norm, sim_valid = _normalize_pose_foreground_image(simulated)
    common = real_valid & sim_valid
    threshold = float(np.clip(foreground_threshold, 0.05, 0.60))
    real_support = common & (real_norm >= threshold)
    sim_support = common & (sim_norm >= threshold)
    union = real_support | sim_support
    union_count = int(np.count_nonzero(union))
    if union_count < 4:
        raise FlexibleFitBackendError(
            "Pose foreground scoring could not identify molecular foreground."
        )

    intersection_count = int(np.count_nonzero(real_support & sim_support))
    support_total = int(np.count_nonzero(real_support)) + int(
        np.count_nonzero(sim_support)
    )
    dice = (
        2.0 * float(intersection_count) / float(support_total)
        if support_total > 0
        else 0.0
    )

    real_values = real_norm[union]
    sim_values = sim_norm[union]
    weights = 0.20 + np.maximum(real_values, sim_values)
    weight_sum = float(np.sum(weights))
    real_mean = float(np.sum(weights * real_values) / weight_sum)
    sim_mean = float(np.sum(weights * sim_values) / weight_sum)
    real_centered = real_values - real_mean
    sim_centered = sim_values - sim_mean
    covariance = float(np.sum(weights * real_centered * sim_centered))
    denominator = math.sqrt(
        float(np.sum(weights * real_centered * real_centered))
        * float(np.sum(weights * sim_centered * sim_centered))
    )
    zncc = covariance / denominator if denominator > 1e-12 else -1.0
    nrmse = math.sqrt(
        float(np.sum(weights * (real_values - sim_values) ** 2) / weight_sum)
    )

    def weighted_center(normalized, support):
        foreground = np.where(support, normalized, 0.0)
        total = float(np.sum(foreground))
        if total <= 1e-12:
            return None
        yy, xx = np.indices(normalized.shape, dtype=float)
        return (
            float(np.sum(xx * foreground) / total),
            float(np.sum(yy * foreground) / total),
        )

    real_center = weighted_center(real_norm, real_support)
    sim_center = weighted_center(sim_norm, sim_support)
    centroid_distance = 1.0
    centroid_dx_px = 0.0
    centroid_dy_px = 0.0
    if real_center is not None and sim_center is not None:
        # Translation that moves the simulated foreground centroid onto the
        # Real-AFM foreground centroid. Positive values mean right/down.
        centroid_dx_px = float(real_center[0] - sim_center[0])
        centroid_dy_px = float(real_center[1] - sim_center[1])
        diagonal = max(
            math.hypot(float(real.shape[1]), float(real.shape[0])),
            1.0,
        )
        centroid_distance = math.hypot(
            real_center[0] - sim_center[0],
            real_center[1] - sim_center[1],
        ) / diagonal

    score = (
        0.55 * float(np.clip(zncc, -1.0, 1.0))
        + 0.35 * float(np.clip(dice, 0.0, 1.0))
        - 0.15 * float(min(nrmse, 2.0))
        - 0.30 * float(min(centroid_distance, 1.0))
    )
    return {
        "score": float(score),
        "foreground_zncc": float(zncc),
        "foreground_dice": float(dice),
        "foreground_nrmse": float(nrmse),
        "centroid_distance_fraction": float(centroid_distance),
        "centroid_dx_px": centroid_dx_px,
        "centroid_dy_px": centroid_dy_px,
        "foreground_fraction": float(union_count / max(np.count_nonzero(common), 1)),
    }


def estimate_phase_correlation_translation(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    apply_hann_window: bool = True,
    max_shift_fraction: Optional[float] = None,
) -> dict[str, float]:
    """Estimate translation with pyNuD's FFT phase-correlation algorithm."""
    reference = np.asarray(reference_image, dtype=float)
    moving = np.asarray(moving_image, dtype=float)
    if reference.ndim != 2 or moving.shape != reference.shape:
        raise FlexibleFitBackendError(
            "Phase correlation requires two equally sized 2D images."
        )
    if min(reference.shape) < 2:
        raise FlexibleFitBackendError(
            "Phase correlation requires images of at least 2x2 pixels."
        )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(moving)):
        raise FlexibleFitBackendError(
            "Phase correlation requires finite image pixels."
        )

    height, width = reference.shape
    if apply_hann_window:
        window = np.outer(np.hanning(height), np.hanning(width))
        reference = reference * window
        moving = moving * window

    reference_fft = np.fft.fft2(reference)
    moving_fft = np.fft.fft2(moving)
    cross_power = reference_fft * np.conj(moving_fft)
    magnitude = np.abs(cross_power)
    usable = magnitude > 1e-12
    if np.count_nonzero(usable) < 4:
        raise FlexibleFitBackendError(
            "Phase correlation has insufficient spectral content."
        )
    cross_power = np.where(
        usable,
        cross_power / np.maximum(magnitude, 1e-12),
        0.0,
    )
    correlation = np.real(np.fft.ifft2(cross_power))

    signed_y = np.arange(height, dtype=float)
    signed_x = np.arange(width, dtype=float)
    signed_y[signed_y > height / 2.0] -= height
    signed_x[signed_x > width / 2.0] -= width
    allowed = np.ones(reference.shape, dtype=bool)
    max_dx = float(np.max(np.abs(signed_x)))
    max_dy = float(np.max(np.abs(signed_y)))
    if max_shift_fraction is not None:
        fraction = float(np.clip(max_shift_fraction, 0.05, 0.49))
        max_dx = float(
            max(1, min(width - 1, int(round(width * fraction))))
        )
        max_dy = float(
            max(1, min(height - 1, int(round(height * fraction))))
        )
        allowed = (
            (np.abs(signed_y)[:, None] <= max_dy)
            & (np.abs(signed_x)[None, :] <= max_dx)
        )

    constrained = np.where(allowed, correlation, -np.inf)
    peak_y, peak_x = np.unravel_index(
        int(np.argmax(constrained)),
        constrained.shape,
    )

    def subpixel_peak(previous, center, following):
        denominator = previous - 2.0 * center + following
        if abs(denominator) <= 1e-12:
            return 0.0
        return float(np.clip(
            0.5 * (previous - following) / denominator,
            -1.0,
            1.0,
        ))

    dy_sub = subpixel_peak(
        correlation[(peak_y - 1) % height, peak_x],
        correlation[peak_y, peak_x],
        correlation[(peak_y + 1) % height, peak_x],
    )
    dx_sub = subpixel_peak(
        correlation[peak_y, (peak_x - 1) % width],
        correlation[peak_y, peak_x],
        correlation[peak_y, (peak_x + 1) % width],
    )
    dy_px = float(np.clip(signed_y[peak_y] + dy_sub, -max_dy, max_dy))
    dx_px = float(np.clip(signed_x[peak_x] + dx_sub, -max_dx, max_dx))
    peak_value = float(correlation[peak_y, peak_x])
    peak_prominence = peak_value - float(np.median(correlation[allowed]))
    return {
        "dx_px": dx_px,
        "dy_px": dy_px,
        "peak": peak_value,
        "peak_prominence": float(peak_prominence),
    }


def estimate_image_correlation_translation(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    max_shift_fraction: float = 0.40,
    min_overlap_fraction: float = 0.35,
) -> dict[str, float]:
    """Find the non-wrapping XY shift with maximum normalized correlation."""
    reference = np.asarray(reference_image, dtype=float)
    moving = np.asarray(moving_image, dtype=float)
    if reference.ndim != 2 or moving.shape != reference.shape:
        raise FlexibleFitBackendError(
            "Image correlation requires two equally sized 2D images."
        )
    if min(reference.shape) < 2:
        raise FlexibleFitBackendError(
            "Image correlation requires images of at least 2x2 pixels."
        )

    reference_valid = np.isfinite(reference)
    moving_valid = np.isfinite(moving)
    reference_count = int(np.count_nonzero(reference_valid))
    moving_count = int(np.count_nonzero(moving_valid))
    if min(reference_count, moving_count) < 4:
        raise FlexibleFitBackendError(
            "Image correlation requires at least four valid pixels per image."
        )
    reference = np.where(reference_valid, reference, 0.0)
    moving = np.where(moving_valid, moving, 0.0)
    reference_mask = reference_valid.astype(float)
    moving_mask = moving_valid.astype(float)

    height, width = reference.shape
    full_shape = (2 * height - 1, 2 * width - 1)

    def correlate(first, second):
        first_fft = np.fft.fft2(first, s=full_shape)
        second_fft = np.fft.fft2(
            np.flip(second, axis=(0, 1)),
            s=full_shape,
        )
        return np.real(np.fft.ifft2(first_fft * second_fft))

    overlap = correlate(reference_mask, moving_mask)
    sum_reference = correlate(reference, moving_mask)
    sum_moving = correlate(reference_mask, moving)
    sum_reference_squared = correlate(
        reference * reference,
        moving_mask,
    )
    sum_moving_squared = correlate(
        reference_mask,
        moving * moving,
    )
    sum_products = correlate(reference, moving)

    safe_overlap = np.maximum(overlap, 1.0)
    covariance = (
        sum_products
        - (sum_reference * sum_moving) / safe_overlap
    )
    reference_variance = np.maximum(
        sum_reference_squared
        - (sum_reference * sum_reference) / safe_overlap,
        0.0,
    )
    moving_variance = np.maximum(
        sum_moving_squared
        - (sum_moving * sum_moving) / safe_overlap,
        0.0,
    )
    denominator = np.sqrt(reference_variance * moving_variance)
    correlation = np.full(full_shape, -np.inf, dtype=float)
    usable = denominator > 1e-12
    correlation[usable] = covariance[usable] / denominator[usable]

    signed_y = np.arange(full_shape[0], dtype=float) - (height - 1)
    signed_x = np.arange(full_shape[1], dtype=float) - (width - 1)
    fraction = float(np.clip(max_shift_fraction, 0.05, 0.49))
    max_dy = max(1, int(round(height * fraction)))
    max_dx = max(1, int(round(width * fraction)))
    minimum_overlap = max(
        4.0,
        float(min(reference_count, moving_count))
        * float(np.clip(min_overlap_fraction, 0.10, 0.95)),
    )
    allowed = (
        (np.abs(signed_y)[:, None] <= max_dy)
        & (np.abs(signed_x)[None, :] <= max_dx)
        & (overlap >= minimum_overlap)
        & np.isfinite(correlation)
    )
    if not np.any(allowed):
        raise FlexibleFitBackendError(
            "Image correlation has no valid translation candidates."
        )
    constrained = np.where(allowed, correlation, -np.inf)
    peak_y, peak_x = np.unravel_index(
        int(np.argmax(constrained)),
        constrained.shape,
    )

    def subpixel_peak(previous, center, following):
        if not (
            np.isfinite(previous)
            and np.isfinite(center)
            and np.isfinite(following)
        ):
            return 0.0
        denominator_1d = previous - 2.0 * center + following
        if abs(denominator_1d) <= 1e-12:
            return 0.0
        return float(np.clip(
            0.5 * (previous - following) / denominator_1d,
            -1.0,
            1.0,
        ))

    dy_sub = 0.0
    dx_sub = 0.0
    if 0 < peak_y < full_shape[0] - 1:
        dy_sub = subpixel_peak(
            correlation[peak_y - 1, peak_x],
            correlation[peak_y, peak_x],
            correlation[peak_y + 1, peak_x],
        )
    if 0 < peak_x < full_shape[1] - 1:
        dx_sub = subpixel_peak(
            correlation[peak_y, peak_x - 1],
            correlation[peak_y, peak_x],
            correlation[peak_y, peak_x + 1],
        )

    zero_y = height - 1
    zero_x = width - 1
    zero_correlation = float(correlation[zero_y, zero_x])
    if not np.isfinite(zero_correlation):
        zero_correlation = -1e9
    peak_overlap = float(overlap[peak_y, peak_x])
    return {
        "dx_px": float(signed_x[peak_x] + dx_sub),
        "dy_px": float(signed_y[peak_y] + dy_sub),
        "correlation": float(correlation[peak_y, peak_x]),
        "zero_shift_correlation": zero_correlation,
        "overlap_fraction": (
            peak_overlap / max(float(min(reference_count, moving_count)), 1.0)
        ),
    }


def estimate_pose_foreground_translation(
    real_image: np.ndarray,
    simulated_image: np.ndarray,
    search_radius_px: int = 2,
    max_shift_fraction: float = 0.40,
) -> dict[str, float]:
    """Find the integer XY shift that directly maximizes foreground alignment.

    Foreground centroids provide only a search center: differing probe width,
    molecular curvature, and partial occlusion can make a centroid-only shift
    worse. Every nearby shift is therefore applied without wraparound and scored
    with the same foreground objective used by Estimate Pose.
    """
    real = np.asarray(real_image, dtype=float)
    simulated = np.asarray(simulated_image, dtype=float)
    if real.ndim != 2 or simulated.shape != real.shape:
        raise FlexibleFitBackendError(
            "Pose translation search requires two equally sized 2D images."
        )
    if min(real.shape) < 2:
        raise FlexibleFitBackendError(
            "Pose translation search requires images of at least 2x2 pixels."
        )

    base = score_pose_foreground_alignment(real, simulated)
    guess_dx = int(round(float(base.get("centroid_dx_px", 0.0))))
    guess_dy = int(round(float(base.get("centroid_dy_px", 0.0))))
    radius = max(1, int(search_radius_px))
    fraction = float(np.clip(max_shift_fraction, 0.05, 0.49))
    max_dx = max(1, min(real.shape[1] - 1, int(round(real.shape[1] * fraction))))
    max_dy = max(1, min(real.shape[0] - 1, int(round(real.shape[0] * fraction))))
    guess_dx = int(np.clip(guess_dx, -max_dx, max_dx))
    guess_dy = int(np.clip(guess_dy, -max_dy, max_dy))
    valid_sim = np.isfinite(simulated) & (simulated > -1e8)
    valid_values = simulated[valid_sim]
    if valid_values.size < 4:
        raise FlexibleFitBackendError(
            "Pose translation search requires valid simulated pixels."
        )
    fill_value = float(np.percentile(valid_values, 5.0))

    def shifted_without_wrap(dx_px: int, dy_px: int) -> np.ndarray:
        shifted = np.full(simulated.shape, fill_value, dtype=float)
        height, width = simulated.shape
        src_x0 = max(0, -dx_px)
        src_x1 = min(width, width - dx_px)
        src_y0 = max(0, -dy_px)
        src_y1 = min(height, height - dy_px)
        dst_x0 = max(0, dx_px)
        dst_x1 = min(width, width + dx_px)
        dst_y0 = max(0, dy_px)
        dst_y1 = min(height, height + dy_px)
        if src_x1 > src_x0 and src_y1 > src_y0:
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = simulated[
                src_y0:src_y1,
                src_x0:src_x1,
            ]
        return shifted

    dx_candidates = set(range(
        max(-max_dx, guess_dx - radius),
        min(max_dx, guess_dx + radius) + 1,
    ))
    dy_candidates = set(range(
        max(-max_dy, guess_dy - radius),
        min(max_dy, guess_dy + radius) + 1,
    ))
    # Always test the current placement and its immediate neighborhood so a
    # biased centroid can never force the model away from a better position.
    dx_candidates.update(value for value in (-1, 0, 1) if abs(value) <= max_dx)
    dy_candidates.update(value for value in (-1, 0, 1) if abs(value) <= max_dy)

    best = dict(base)
    best_dx = 0
    best_dy = 0
    best_score = float(base["score"])
    for dy_px in sorted(dy_candidates):
        for dx_px in sorted(dx_candidates):
            if dx_px == 0 and dy_px == 0:
                continue
            candidate = shifted_without_wrap(dx_px, dy_px)
            try:
                metrics = score_pose_foreground_alignment(real, candidate)
            except FlexibleFitBackendError:
                continue
            score = float(metrics["score"])
            # Prefer the smaller shift when scores are effectively tied.
            better = score > best_score + 1e-9
            tied_and_smaller = (
                abs(score - best_score) <= 1e-9
                and (abs(dx_px) + abs(dy_px)) < (abs(best_dx) + abs(best_dy))
            )
            if better or tied_and_smaller:
                best = dict(metrics)
                best_dx = int(dx_px)
                best_dy = int(dy_px)
                best_score = score

    best.update({
        "dx_px": float(best_dx),
        "dy_px": float(best_dy),
        "unshifted_score": float(base["score"]),
    })
    return best


def is_flexible_fit_score_improvement(
    before_score: float,
    after_score: float,
    minimum_gain: float = 1e-4,
) -> bool:
    """Accept a structurally safe fit when its composite AFM score improves.

    The flexible-fit pose score already combines image correlation and
    normalized height error.  Aligned RMSD remains useful as a reported
    diagnostic, but requiring it to improve as a second hard gate can reject
    a visibly better molecular silhouette by counting height mismatch twice.
    """
    before = float(before_score)
    after = float(after_score)
    gain = max(0.0, float(minimum_gain))
    return bool(
        np.isfinite(before)
        and np.isfinite(after)
        and after > before + gain
    )


def select_flexible_fit_coordinates(
    fit_result: dict,
    structure_mode: str,
    expected_atom_count: int,
) -> np.ndarray:
    """Return a validated copy of the requested pre-fit or best-fit model."""
    mode = str(structure_mode).strip().lower()
    coordinate_key = {
        "original": "original_coords",
        "best_fit": "fitted_coords",
    }.get(mode)
    if coordinate_key is None:
        raise FlexibleFitBackendError(
            f"Unknown active flexible-fit structure: {structure_mode}"
        )
    coordinates = np.asarray(
        (fit_result or {}).get(coordinate_key, []),
        dtype=float,
    )
    expected_shape = (int(expected_atom_count), 3)
    if (
        coordinates.shape != expected_shape
        or not np.all(np.isfinite(coordinates))
    ):
        raise FlexibleFitBackendError(
            f"The {mode} flexible-fit coordinates are unavailable or "
            "incompatible with the loaded atoms."
        )
    return np.array(coordinates, dtype=float, copy=True)


def estimate_nma_fit_parameters(
    node_coordinates_nm: np.ndarray,
    atom_selection: str = "calpha",
    image_shape: Optional[tuple[int, int]] = None,
) -> dict[str, float | int]:
    """Estimate conservative linear-ANM controls from structure geometry.

    These values are intentionally *safe starting limits*, not a claim that a
    PDB alone determines the unique best fit to an AFM image.  The elastic
    network cutoff follows the local node density, while displacement limits
    scale with the molecular dimensions.
    """
    coordinates = np.asarray(node_coordinates_nm, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or coordinates.shape[0] < 4
        or not np.all(np.isfinite(coordinates))
    ):
        raise FlexibleFitBackendError(
            "PDB-safe NMA estimation requires at least four finite XYZ nodes."
        )

    node_count = int(coordinates.shape[0])
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    radius_of_gyration_nm = float(
        np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    )
    extent_nm = float(np.linalg.norm(np.ptp(coordinates, axis=0)))

    # Estimate a robust k-neighbor distance without allocating an N x N
    # matrix for large all-atom structures.
    sample_count = min(node_count, 640)
    sample_indices = np.linspace(
        0, node_count - 1, sample_count, dtype=int
    )
    neighbor_rank = min(
        node_count - 1,
        12 if str(atom_selection) == "all" else 7,
    )
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(coordinates).query(
            coordinates[sample_indices],
            k=neighbor_rank + 1,
        )
        neighbor_distances = np.asarray(distances, dtype=float)[:, -1]
    except (ImportError, ModuleNotFoundError):
        neighbor_distances = []
        for start in range(0, sample_count, 32):
            indices = sample_indices[start:start + 32]
            delta = coordinates[indices, None, :] - coordinates[None, :, :]
            distance_squared = np.sum(delta * delta, axis=2)
            distance_squared[
                np.arange(len(indices), dtype=int),
                indices,
            ] = np.inf
            kth_squared = np.partition(
                distance_squared,
                neighbor_rank - 1,
                axis=1,
            )[:, neighbor_rank - 1]
            neighbor_distances.extend(np.sqrt(kth_squared).tolist())
    local_spacing_nm = float(
        np.percentile(np.asarray(neighbor_distances, dtype=float), 75.0)
    )

    if str(atom_selection) == "all":
        cutoff_nm = float(np.clip(1.20 * local_spacing_nm, 0.50, 0.90))
    else:
        cutoff_nm = float(np.clip(1.20 * local_spacing_nm, 1.20, 1.80))
    cutoff_angstrom = round(cutoff_nm * 20.0) / 2.0

    # A per-mode RMS displacement of roughly 6% of the molecular span is a
    # useful conservative envelope.  Mode-specific clash scanning tightens
    # this estimate again after the ANM basis has been calculated.
    max_amplitude_nm = float(np.clip(0.06 * extent_nm, 0.15, 0.50))
    max_amplitude_nm = round(max_amplitude_nm * 20.0) / 20.0
    max_total_rms_nm = max_amplitude_nm

    if node_count < 40:
        n_modes = 3
    elif node_count < 500:
        n_modes = 5
    elif node_count < 1500:
        n_modes = 6
    else:
        n_modes = 4 if str(atom_selection) == "all" else 7
    maxfev = int(np.clip(40 * n_modes + 40, 160, 360))

    preview_max_px = 96
    if image_shape is not None and len(image_shape) >= 2:
        longest_side = max(int(image_shape[-2]), int(image_shape[-1]))
        if longest_side <= 64:
            preview_max_px = 64
        elif longest_side >= 384 and node_count < 1500:
            preview_max_px = 112

    return {
        "cutoff_angstrom": float(cutoff_angstrom),
        "first_mode": 1,
        "n_modes": int(n_modes),
        "max_amplitude_nm": float(max_amplitude_nm),
        "max_total_rms_nm": float(max_total_rms_nm),
        "preview_max_px": int(preview_max_px),
        "maxfev": int(maxfev),
        "mode_energy_weight": 0.050,
        "backbone_strain_weight": 0.150,
        "clash_weight": 0.500,
        "node_count": int(node_count),
        "extent_nm": float(extent_nm),
        "radius_of_gyration_nm": float(radius_of_gyration_nm),
        "local_spacing_nm": float(local_spacing_nm),
    }


def heavy_atom_indices(
    elements: Optional[np.ndarray],
    atom_count: int,
) -> np.ndarray:
    """Return non-hydrogen indices for steric safety checks."""
    values = np.asarray(
        elements if elements is not None else ["C"] * int(atom_count)
    ).astype(str)
    if values.size != int(atom_count):
        values = np.asarray(["C"] * int(atom_count))
    normalized = np.char.upper(np.char.strip(values))
    return np.where(normalized != "H")[0].astype(int)


def introduces_severe_atomic_clash(
    base_coordinates_nm: np.ndarray,
    candidate_coordinates_nm: np.ndarray,
    selected_atom_indices: np.ndarray,
    clash_distance_nm: float = 0.18,
    original_neighbor_nm: float = 0.28,
) -> bool:
    """Return whether a deformation creates a new severe selected-atom clash."""
    base = np.asarray(base_coordinates_nm, dtype=float)
    candidate = np.asarray(candidate_coordinates_nm, dtype=float)
    indices = np.asarray(selected_atom_indices, dtype=int)
    if (
        base.shape != candidate.shape
        or base.ndim != 2
        or base.shape[1] != 3
        or indices.size < 2
    ):
        return False
    selected = candidate[indices]
    try:
        from scipy.spatial import cKDTree

        pairs = cKDTree(selected).query_pairs(float(clash_distance_nm))
    except (ImportError, ModuleNotFoundError):
        pairs = []
        threshold_squared = float(clash_distance_nm) ** 2
        for left in range(indices.size - 1):
            delta = selected[left + 1:] - selected[left]
            colliding = np.where(
                np.sum(delta * delta, axis=1) < threshold_squared
            )[0]
            pairs.extend(
                (left, left + 1 + int(offset))
                for offset in colliding
            )
    for left, right in pairs:
        atom_left = int(indices[left])
        atom_right = int(indices[right])
        base_distance = float(
            np.linalg.norm(base[atom_right] - base[atom_left])
        )
        if base_distance > float(original_neighbor_nm):
            return True
    return False


def evaluate_nolb_candidate_safety(
    base_coordinates_nm: np.ndarray,
    candidate_coordinates_nm: np.ndarray,
    elements: Optional[np.ndarray],
    maximum_rms_displacement_nm: float,
) -> NolbCandidateSafety:
    """Reject excessive or newly clashing NOLB conformations."""
    base = np.asarray(base_coordinates_nm, dtype=float)
    candidate = np.asarray(candidate_coordinates_nm, dtype=float)
    if (
        base.shape != candidate.shape
        or base.ndim != 2
        or base.shape[1] != 3
        or base.shape[0] < 1
    ):
        raise FlexibleFitBackendError(
            "NOLB safety checks require matching (N, 3) coordinates."
        )
    maximum_rms = float(maximum_rms_displacement_nm)
    if maximum_rms <= 0.0:
        raise FlexibleFitBackendError(
            "NOLB safety RMS-displacement limit must be positive."
        )

    displacement_norms = np.linalg.norm(candidate - base, axis=1)
    rms_displacement = float(
        np.sqrt(np.mean(displacement_norms * displacement_norms))
    )
    peak_displacement = float(np.percentile(displacement_norms, 99.0))
    maximum_peak = max(0.75, 3.0 * maximum_rms)
    if (
        rms_displacement > maximum_rms * 1.05 + 1e-8
        or peak_displacement > maximum_peak + 1e-8
    ):
        return NolbCandidateSafety(
            accepted=False,
            reason="excessive displacement",
            rms_displacement_nm=rms_displacement,
            peak_displacement_nm=peak_displacement,
        )

    selected_heavy_atoms = heavy_atom_indices(elements, base.shape[0])
    if introduces_severe_atomic_clash(
        base,
        candidate,
        selected_heavy_atoms,
        clash_distance_nm=0.20,
        original_neighbor_nm=0.30,
    ):
        return NolbCandidateSafety(
            accepted=False,
            reason="new heavy-atom clash",
            rms_displacement_nm=rms_displacement,
            peak_displacement_nm=peak_displacement,
        )
    return NolbCandidateSafety(
        accepted=True,
        reason="",
        rms_displacement_nm=rms_displacement,
        peak_displacement_nm=peak_displacement,
    )


def estimate_clash_safe_nma_amplitude(
    base_coordinates_nm: np.ndarray,
    mode_basis: np.ndarray,
    selected_atom_indices: np.ndarray,
    requested_amplitude_nm: float,
) -> float:
    """Tighten an automatic amplitude using both signs of every fitted mode."""
    base = np.asarray(base_coordinates_nm, dtype=float)
    modes = np.asarray(mode_basis, dtype=float)
    requested = max(0.01, float(requested_amplitude_nm))
    if modes.ndim != 3 or modes.shape[0] == 0:
        return requested

    trial_levels = np.linspace(requested / 8.0, requested, 8)
    safe_limits = []
    for mode in modes:
        mode_limit = requested
        for sign in (-1.0, 1.0):
            direction_limit = 0.0
            for amplitude in trial_levels:
                candidate = base + sign * float(amplitude) * mode
                if introduces_severe_atomic_clash(
                    base,
                    candidate,
                    selected_atom_indices,
                ):
                    break
                direction_limit = float(amplitude)
            mode_limit = min(mode_limit, direction_limit)
        safe_limits.append(mode_limit)

    safe = min(safe_limits) if safe_limits else requested
    if safe < requested:
        safe *= 0.85
    safe = float(np.clip(safe, 0.01, requested))
    return round(safe * 100.0) / 100.0


def calculate_nmff_correlation_slope(
    amplitudes: np.ndarray,
    correlations: np.ndarray,
) -> float:
    """Fit the NMFF-AFM correlation-versus-amplitude slope."""
    q = np.asarray(amplitudes, dtype=float).reshape(-1)
    cc = np.asarray(correlations, dtype=float).reshape(-1)
    if q.size != cc.size or q.size < 2:
        raise FlexibleFitBackendError(
            "NMFF-AFM slope fitting requires matching amplitude and score arrays."
        )
    valid = np.isfinite(q) & np.isfinite(cc)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    q = q[valid]
    cc = cc[valid]
    centered_q = q - float(np.mean(q))
    denominator = float(np.dot(centered_q, centered_q))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(centered_q, cc - float(np.mean(cc))) / denominator)


def nmff_decay_threshold(
    correlation_gains: np.ndarray,
    fraction: float = 0.03,
) -> Optional[float]:
    """Estimate the paper's exponential-decay stopping threshold.

    Positive per-step correlation gains are fitted as ``A * exp(-k * step)``.
    The returned threshold is ``fraction * A``. ``None`` means that a stable
    decaying fit is not yet available.
    """
    gains = np.asarray(correlation_gains, dtype=float).reshape(-1)
    valid = np.isfinite(gains) & (gains > 1e-12)
    if np.count_nonzero(valid) < 3:
        return None
    x = np.arange(gains.size, dtype=float)[valid]
    y = np.log(gains[valid])
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope >= 0.0:
        return None
    threshold = float(fraction) * float(np.exp(intercept))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return None
    return threshold


def run_iterative_nmff(
    initial_coordinates: np.ndarray,
    calculate_modes: Callable[
        [np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, str],
    ],
    score_coordinates: Callable[[np.ndarray], float],
    config: NmffRunConfig,
    keep_running: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    acceptance_callback: Optional[
        Callable[[np.ndarray, float], None]
    ] = None,
    candidate_validator: Optional[Callable[[np.ndarray], bool]] = None,
    candidate_projector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> NmffRunResult:
    """Iteratively recalculate normal modes and improve AFM correlation.

    ``calculate_modes`` returns modes normalized to 1 nm RMS displacement,
    their eigenvalues, user-facing mode numbers, and the NMA engine name.
    For each mode this function evaluates ``-Q, -Q/2, 0, Q/2, Q`` and follows
    the mode with the largest absolute correlation slope, as in NMFF-AFM.
    """
    coordinates = np.asarray(initial_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise FlexibleFitBackendError(
            "NMFF-AFM requires an (N, 3) coordinate array."
        )
    if float(config.step_amplitude_nm) <= 0.0:
        raise FlexibleFitBackendError(
            "NMFF-AFM step amplitude must be positive."
        )
    if int(config.max_iterations) < 1:
        raise FlexibleFitBackendError(
            "NMFF-AFM maximum iterations must be positive."
        )
    if float(config.max_total_rms_nm) <= 0.0:
        raise FlexibleFitBackendError(
            "NMFF-AFM total RMS displacement limit must be positive."
        )

    initial = np.array(coordinates, dtype=float, copy=True)
    current = np.array(initial, dtype=float, copy=True)
    current_correlation = float(score_coordinates(current))
    evaluations = 1
    if not np.isfinite(current_correlation):
        raise FlexibleFitBackendError(
            "The initial NMFF-AFM correlation score is invalid."
        )

    q_full = float(config.step_amplitude_nm)
    amplitudes = np.asarray(
        [-q_full, -0.5 * q_full, 0.0, 0.5 * q_full, q_full],
        dtype=float,
    )
    accepted: list[NmffIteration] = []
    gains: list[float] = []
    low_gain_count = 0
    rejected_candidate_count = 0
    stop_reason = "maximum iterations reached"

    for iteration_index in range(int(config.max_iterations)):
        if keep_running is not None and not bool(keep_running()):
            raise FlexibleFitBackendCanceled("NMFF-AFM fitting was canceled.")

        mode_basis, eigenvalues, mode_numbers, nma_method = calculate_modes(
            current
        )
        mode_basis = np.asarray(mode_basis, dtype=float)
        eigenvalues = np.asarray(eigenvalues, dtype=float).reshape(-1)
        mode_numbers = np.asarray(mode_numbers, dtype=int).reshape(-1)
        if (
            mode_basis.ndim != 3
            or mode_basis.shape[1:] != current.shape
            or mode_basis.shape[0] == 0
        ):
            stop_reason = "no usable positive normal modes"
            break

        mode_count = mode_basis.shape[0]
        if eigenvalues.size != mode_count or mode_numbers.size != mode_count:
            raise FlexibleFitBackendError(
                "NMFF-AFM mode metadata does not match the mode basis."
            )

        best_choice = None
        for mode_position in range(mode_count):
            correlations = np.full(amplitudes.shape, np.nan, dtype=float)
            correlations[2] = current_correlation
            candidates: dict[float, tuple[np.ndarray, float, float]] = {}
            for amplitude_position, amplitude in enumerate(amplitudes):
                if amplitude_position == 2:
                    continue
                candidate = current + float(amplitude) * mode_basis[mode_position]
                if candidate_projector is not None:
                    candidate = np.asarray(
                        candidate_projector(candidate),
                        dtype=float,
                    )
                    if candidate.shape != current.shape:
                        raise FlexibleFitBackendError(
                            "NMFF-AFM candidate projection changed the "
                            "coordinate-array shape."
                        )
                total_rms = float(
                    np.sqrt(
                        np.mean(np.sum((candidate - initial) ** 2, axis=1))
                    )
                )
                if total_rms > float(config.max_total_rms_nm) + 1e-12:
                    continue
                if (
                    candidate_validator is not None
                    and not bool(candidate_validator(candidate))
                ):
                    rejected_candidate_count += 1
                    continue
                correlation = float(score_coordinates(candidate))
                evaluations += 1
                if progress_callback is not None:
                    progress_callback(
                        iteration_index,
                        evaluations,
                        current_correlation,
                    )
                if keep_running is not None and not bool(keep_running()):
                    raise FlexibleFitBackendCanceled(
                        "NMFF-AFM fitting was canceled."
                    )
                if not np.isfinite(correlation):
                    continue
                correlations[amplitude_position] = correlation
                candidates[float(amplitude)] = (
                    candidate,
                    correlation,
                    total_rms,
                )

            slope = calculate_nmff_correlation_slope(amplitudes, correlations)
            if not np.isfinite(slope) or abs(slope) <= 1e-15:
                continue
            preferred_amplitudes = (
                (q_full, 0.5 * q_full)
                if slope > 0.0
                else (-q_full, -0.5 * q_full)
            )
            selected = None
            selected_amplitude = None
            for amplitude in preferred_amplitudes:
                candidate_for_step = candidates.get(float(amplitude))
                if (
                    candidate_for_step is not None
                    and float(candidate_for_step[1]) - current_correlation
                    >= float(config.minimum_cc_gain)
                ):
                    selected = candidate_for_step
                    selected_amplitude = float(amplitude)
                    break
            if selected is None or selected_amplitude is None:
                continue
            candidate, correlation, total_rms = selected
            choice = (
                abs(float(slope)),
                float(slope),
                mode_position,
                selected_amplitude,
                candidate,
                float(correlation),
                float(total_rms),
            )
            if best_choice is None or choice[0] > best_choice[0]:
                best_choice = choice

        if best_choice is None:
            stop_reason = "all candidate steps exceeded limits or were invalid"
            break

        (
            _,
            selected_slope,
            mode_position,
            selected_amplitude,
            candidate,
            candidate_correlation,
            total_rms,
        ) = best_choice
        gain = candidate_correlation - current_correlation
        if gain < float(config.minimum_cc_gain):
            stop_reason = "no normal-mode step improved correlation"
            break

        correlation_before = current_correlation
        current = np.asarray(candidate, dtype=float)
        current_correlation = float(candidate_correlation)
        gains.append(float(gain))
        accepted.append(
            NmffIteration(
                iteration=iteration_index + 1,
                mode_number=int(mode_numbers[mode_position]),
                eigenvalue=float(eigenvalues[mode_position]),
                amplitude_nm=float(selected_amplitude),
                slope=float(selected_slope),
                correlation_before=float(correlation_before),
                correlation_after=float(current_correlation),
                total_rms_nm=float(total_rms),
                nma_method=str(nma_method),
            )
        )
        if acceptance_callback is not None:
            acceptance_callback(
                np.array(current, dtype=float, copy=True),
                float(current_correlation),
            )

        threshold = nmff_decay_threshold(
            np.asarray(gains, dtype=float),
            fraction=float(config.convergence_fraction),
        )
        if (
            len(accepted) >= int(config.minimum_iterations)
            and threshold is not None
            and gain < threshold
        ):
            low_gain_count += 1
        else:
            low_gain_count = 0
        if low_gain_count >= int(config.convergence_patience):
            stop_reason = "correlation gain reached exponential-decay threshold"
            break

    return NmffRunResult(
        coordinates=np.array(current, dtype=float, copy=True),
        initial_correlation=float(
            accepted[0].correlation_before
            if accepted else current_correlation
        ),
        final_correlation=float(current_correlation),
        evaluations=int(evaluations),
        iterations=tuple(accepted),
        stop_reason=stop_reason,
        rejected_candidate_count=int(rejected_candidate_count),
    )


def align_coordinates_kabsch(
    mobile_coords: np.ndarray,
    reference_coords: np.ndarray,
) -> np.ndarray:
    """Rigidly align mobile coordinates to a reference with the Kabsch method."""
    mobile = np.asarray(mobile_coords, dtype=float)
    reference = np.asarray(reference_coords, dtype=float)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise FlexibleFitBackendError(
            "Kabsch alignment requires matching (N, 3) coordinate arrays."
        )
    if mobile.shape[0] < 3:
        raise FlexibleFitBackendError(
            "Kabsch alignment requires at least three coordinates."
        )

    mobile_center = np.mean(mobile, axis=0)
    reference_center = np.mean(reference, axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    covariance = mobile_centered.T @ reference_centered
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right
    return mobile_centered @ rotation + reference_center


def resolve_executable(executable: str) -> str:
    """Resolve an executable name or explicit path without invoking a shell."""
    value = os.path.expanduser(str(executable).strip())
    if not value:
        raise FlexibleFitBackendError("NOLB executable path is empty.")

    if os.path.dirname(value):
        path = Path(value).resolve()
        if not path.is_file():
            raise FlexibleFitBackendError(
                f"NOLB executable was not found:\n{path}"
            )
        if not os.access(path, os.X_OK):
            raise FlexibleFitBackendError(
                f"NOLB executable is not executable:\n{path}"
            )
        return str(path)

    resolved = shutil.which(value)
    if resolved is None:
        raise FlexibleFitBackendError(
            f"NOLB executable '{value}' was not found on PATH."
        )
    return resolved


def _resolve_afmfit_executable(executable: str) -> str:
    """Resolve the Python executable belonging to an AFMfit environment."""
    value = os.path.expanduser(str(executable).strip())
    if not value:
        raise FlexibleFitBackendError("AFMfit Python executable is empty.")
    if os.path.dirname(value):
        # Preserve the venv launcher path.  Resolving its symlink to the base
        # interpreter bypasses pyvenv.cfg and makes venv-installed AFMfit
        # packages invisible.
        path = Path(os.path.abspath(value))
        if not path.is_file():
            raise FlexibleFitBackendError(
                f"AFMfit Python executable was not found:\n{path}"
            )
        if not os.access(path, os.X_OK):
            raise FlexibleFitBackendError(
                f"AFMfit Python executable is not executable:\n{path}"
            )
        return str(path)
    resolved = shutil.which(value)
    if resolved is None:
        raise FlexibleFitBackendError(
            f"AFMfit Python executable '{value}' was not found on PATH."
        )
    return resolved


def _afmfit_subprocess_environment(
    cache_root: str | Path | None = None,
) -> dict[str, str]:
    """Give AFMfit dependencies a writable, application-owned cache."""
    environment = os.environ.copy()
    root = (
        Path(cache_root)
        if cache_root is not None
        else Path(tempfile.gettempdir()) / "pynud-afmfit-cache"
    )
    numba_cache = root / "numba"
    matplotlib_cache = root / "matplotlib"
    try:
        numba_cache.mkdir(parents=True, exist_ok=True)
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Let the child process report a concrete filesystem error if its
        # inherited cache locations are also unavailable.
        return environment
    environment.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    return environment


def probe_afmfit_environment(
    python_executable: str,
    bridge_path: str | Path,
    timeout_seconds: float = 90.0,
) -> dict:
    """Verify that a separate Python can import the official AFMfit API."""
    python_path = _resolve_afmfit_executable(python_executable)
    bridge = Path(bridge_path).expanduser().resolve()
    if not bridge.is_file():
        raise FlexibleFitBackendError(
            f"pyNuD AFMfit bridge was not found:\n{bridge}"
        )
    try:
        completed = subprocess.run(
            [python_path, str(bridge), "--probe"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
            env=_afmfit_subprocess_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise FlexibleFitBackendError(
            "Timed out while checking the AFMfit Python environment."
        ) from exc
    except OSError as exc:
        raise FlexibleFitBackendError(
            f"Failed to start the AFMfit Python environment:\n{exc}"
        ) from exc
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    )
    if completed.returncode != 0:
        raise FlexibleFitBackendError(
            "The selected Python cannot run AFMfit:\n"
            f"{python_path}\n\n"
            + (output or "(No diagnostic output.)")
        )
    metadata = None
    for line in reversed((completed.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict) and candidate.get("afmfit_version"):
            metadata = candidate
            break
    if metadata is None:
        raise FlexibleFitBackendError(
            "AFMfit environment check succeeded but returned no version metadata."
        )
    metadata["python_executable"] = python_path
    return metadata


def run_afmfit_external(
    input_pdb_path: str | Path,
    input_image_path: str | Path,
    work_directory: str | Path,
    config: AfmfitRunConfig,
    keep_running: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[AfmfitProgress], None]] = None,
) -> AfmfitRunResult:
    """Run official AFMfit through pyNuD's file-based external bridge."""
    input_pdb = Path(input_pdb_path).resolve()
    input_image = Path(input_image_path).resolve()
    work_path = Path(work_directory).resolve()
    bridge = Path(config.bridge_path).expanduser().resolve()
    if not input_pdb.is_file():
        raise FlexibleFitBackendError(
            f"AFMfit input PDB was not found:\n{input_pdb}"
        )
    if not input_image.is_file():
        raise FlexibleFitBackendError(
            f"AFMfit input image was not found:\n{input_image}"
        )
    if not work_path.is_dir():
        raise FlexibleFitBackendError(
            f"AFMfit working directory was not found:\n{work_path}"
        )
    if not bridge.is_file():
        raise FlexibleFitBackendError(
            f"pyNuD AFMfit bridge was not found:\n{bridge}"
        )
    if int(config.n_cpu) < 1:
        raise FlexibleFitBackendError("AFMfit CPU count must be positive.")
    if int(config.nmodes) < 1:
        raise FlexibleFitBackendError("AFMfit mode count must be positive.")
    if float(config.cutoff_angstrom) <= 0.0:
        raise FlexibleFitBackendError("AFMfit NMA cutoff must be positive.")
    if float(config.sigma_angstrom) <= 0.0:
        raise FlexibleFitBackendError("AFMfit sigma must be positive.")
    if int(config.iterations) < 1:
        raise FlexibleFitBackendError("AFMfit iteration count must be positive.")
    if float(config.regularization_lambda) <= 0.0:
        raise FlexibleFitBackendError(
            "AFMfit regularization lambda must be positive."
        )
    if float(config.timeout_seconds) <= 0.0:
        raise FlexibleFitBackendError("AFMfit runtime limit must be positive.")

    python_path = _resolve_afmfit_executable(config.python_executable)
    output_pdb = work_path / "pynud_afmfit_fitted.pdb"
    result_path = work_path / "pynud_afmfit_result.json"
    log_path = work_path / "pynud_afmfit.log"
    command = [
        python_path,
        str(bridge),
        "--input-pdb",
        str(input_pdb),
        "--input-image",
        str(input_image),
        "--output-pdb",
        str(output_pdb),
        "--result-json",
        str(result_path),
        "--n-cpu",
        str(int(config.n_cpu)),
        "--nmodes",
        str(int(config.nmodes)),
        "--cutoff-angstrom",
        f"{float(config.cutoff_angstrom):.8g}",
        "--sigma-angstrom",
        f"{float(config.sigma_angstrom):.8g}",
        "--angular-distance-deg",
        f"{float(config.angular_distance_deg):.8g}",
        "--rigid-angle-limit-deg",
        f"{float(config.rigid_angle_limit_deg):.8g}",
        "--z-shift-range-angstrom",
        f"{float(config.z_shift_range_angstrom):.8g}",
        "--z-shift-points",
        str(int(config.z_shift_points)),
        "--n-best-views",
        str(int(config.n_best_views)),
        "--view-separation-deg",
        f"{float(config.view_separation_deg):.8g}",
        "--iterations",
        str(int(config.iterations)),
        "--regularization-lambda",
        f"{float(config.regularization_lambda):.8g}",
    ]
    started_at = time.monotonic()

    def read_status() -> AfmfitProgress:
        try:
            output = log_path.read_text(encoding="utf-8", errors="replace")
            log_bytes = int(log_path.stat().st_size)
        except OSError:
            output = ""
            log_bytes = 0
        stage = "starting"
        percent = 0.0
        message = ""
        latest_output = ""
        for line in reversed(output.splitlines()):
            if line.startswith("PYNUD_PROGRESS "):
                try:
                    payload = json.loads(line[len("PYNUD_PROGRESS "):])
                    stage = str(payload.get("stage", stage))
                    percent = float(payload.get("percent", percent))
                    message = latest_output or str(
                        payload.get("message", message)
                    )
                    break
                except (TypeError, ValueError):
                    pass
            if not latest_output and line.strip():
                latest_output = line.strip()
        tqdm_match = re.search(r"(\d{1,3})%", message)
        if tqdm_match is not None:
            fraction = float(
                np.clip(float(tqdm_match.group(1)) / 100.0, 0.0, 1.0)
            )
            if "Project Library" in message:
                percent = 30.0 + 20.0 * fraction
            elif "Projection Matching" in message:
                percent = 50.0 + 12.0 * fraction
            elif "Flexible Fitting" in message:
                percent = 62.0 + 30.0 * fraction
        return AfmfitProgress(
            elapsed_seconds=float(time.monotonic() - started_at),
            log_bytes=log_bytes,
            stage=stage,
            percent=float(np.clip(percent, 0.0, 100.0)),
            message=message,
        )

    def stop_process(process: subprocess.Popen) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()

    process = None
    canceled = False
    timed_out = False
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(work_path),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=(os.name == "posix"),
                    env=_afmfit_subprocess_environment(
                        work_path / ".pynud-runtime-cache"
                    ),
                )
            except OSError as exc:
                raise FlexibleFitBackendError(
                    f"Failed to start AFMfit:\n{exc}"
                ) from exc

            while process.poll() is None:
                if keep_running is not None and not keep_running():
                    canceled = True
                    stop_process(process)
                    break
                if (
                    time.monotonic() - started_at
                    > float(config.timeout_seconds)
                ):
                    timed_out = True
                    stop_process(process)
                    break
                if progress_callback is not None:
                    progress_callback(read_status())
                time.sleep(0.10)
            log_file.flush()
    finally:
        if process is not None and process.poll() is None:
            stop_process(process)

    status = read_status()
    if progress_callback is not None:
        progress_callback(status)
    try:
        output = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = ""
    if canceled:
        raise FlexibleFitBackendCanceled("AFMfit fitting was canceled.")
    if timed_out:
        raise FlexibleFitBackendError(
            f"AFMfit exceeded the {float(config.timeout_seconds):.0f}-second "
            "runtime limit.\n\n"
            + (output.strip() or "(AFMfit produced no diagnostic output.)")
        )
    if process is None or process.returncode != 0:
        return_code = process.returncode if process is not None else "unknown"
        raise FlexibleFitBackendError(
            f"AFMfit exited with status {return_code}.\n\n"
            + (output.strip() or "(AFMfit produced no diagnostic output.)")
        )
    if not output_pdb.is_file() or not result_path.is_file():
        raise FlexibleFitBackendError(
            "AFMfit completed but did not produce the expected fitted PDB "
            "and result metadata.\n\n"
            + (output.strip() or "(AFMfit produced no diagnostic output.)")
        )
    try:
        metadata = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FlexibleFitBackendError(
            f"Could not read AFMfit result metadata:\n{exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise FlexibleFitBackendError("AFMfit result metadata is invalid.")
    return AfmfitRunResult(
        output_pdb_path=output_pdb,
        result_path=result_path,
        command=tuple(command),
        output=output,
        elapsed_seconds=float(time.monotonic() - started_at),
        metadata=metadata,
    )


def run_nolb_ensemble(
    input_pdb_path: str | Path,
    work_directory: str | Path,
    config: NolbRunConfig,
    keep_running: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[NolbProgress], None]] = None,
) -> NolbRunResult:
    """Run NOLB and return its generated multi-model PDB ensemble."""
    input_path = Path(input_pdb_path).resolve()
    work_path = Path(work_directory).resolve()
    if not input_path.is_file():
        raise FlexibleFitBackendError(f"NOLB input PDB was not found:\n{input_path}")
    if not work_path.is_dir():
        raise FlexibleFitBackendError(
            f"NOLB working directory was not found:\n{work_path}"
        )
    if int(config.num_structures) < 1:
        raise FlexibleFitBackendError("NOLB structure count must be positive.")
    if float(config.max_rmsd_angstrom) <= 0.0:
        raise FlexibleFitBackendError("NOLB maximum RMSD must be positive.")
    if float(config.cutoff_angstrom) <= 0.0:
        raise FlexibleFitBackendError("NOLB cutoff distance must be positive.")
    if float(config.timeout_seconds) <= 0.0:
        raise FlexibleFitBackendError("NOLB runtime limit must be positive.")

    executable = resolve_executable(config.binary_path)
    output_prefix = "pynud_nolb"
    command = [
        executable,
        input_path.name,
        "-o",
        output_prefix,
    ]
    if config.minimize:
        command.append("-m")
    command.extend([
        "-s",
        str(int(config.num_structures)),
        "--rmsd",
        f"{float(config.max_rmsd_angstrom):.8g}",
        "-c",
        f"{float(config.cutoff_angstrom):.8g}",
    ])

    log_path = work_path / "pynud_nolb.log"
    started_at = time.monotonic()

    def read_status() -> NolbProgress:
        try:
            log_bytes = int(log_path.stat().st_size)
            output = log_path.read_text(encoding="utf-8", errors="replace")
            last_output_line = next(
                (
                    line.strip()
                    for line in reversed(output.splitlines())
                    if line.strip()
                ),
                "",
            )
        except OSError:
            log_bytes = 0
            last_output_line = ""
        output_pdb_bytes = 0
        for path in work_path.glob(f"{output_prefix}*.pdb"):
            try:
                output_pdb_bytes += int(path.stat().st_size)
            except OSError:
                pass
        return NolbProgress(
            elapsed_seconds=float(time.monotonic() - started_at),
            log_bytes=log_bytes,
            output_pdb_bytes=output_pdb_bytes,
            last_output_line=last_output_line,
        )

    def stop_process(process: subprocess.Popen) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=3.0)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        try:
            popen_kwargs = {}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(
                command,
                cwd=str(work_path),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                **popen_kwargs,
            )
        except OSError as exc:
            raise FlexibleFitBackendError(
                f"Failed to start NOLB:\n{exc}"
            ) from exc

        canceled = False
        timed_out = False
        next_progress_update = 0.0
        while process.poll() is None:
            if keep_running is not None and not bool(keep_running()):
                canceled = True
                stop_process(process)
                break
            elapsed = float(time.monotonic() - started_at)
            if elapsed >= float(config.timeout_seconds):
                timed_out = True
                stop_process(process)
                break
            if progress_callback is not None and elapsed >= next_progress_update:
                log_file.flush()
                progress_callback(read_status())
                next_progress_update = elapsed + 0.5
            time.sleep(0.05)
        log_file.flush()

    output = log_path.read_text(encoding="utf-8", errors="replace")
    elapsed_seconds = float(time.monotonic() - started_at)
    if progress_callback is not None:
        progress_callback(read_status())
    if canceled:
        raise FlexibleFitBackendCanceled("NOLB ensemble generation was canceled.")
    if timed_out:
        detail = output.strip() or "(NOLB produced no diagnostic output.)"
        raise FlexibleFitBackendError(
            f"NOLB exceeded the {float(config.timeout_seconds):.0f}-second "
            f"runtime limit and was stopped.\n\n{detail}"
        )
    if process.returncode != 0:
        detail = output.strip() or "(NOLB produced no diagnostic output.)"
        raise FlexibleFitBackendError(
            f"NOLB exited with status {process.returncode}.\n\n{detail}"
        )

    expected_path = work_path / f"{output_prefix}_nlb_decoys.pdb"
    if expected_path.is_file():
        ensemble_path = expected_path
    else:
        candidates = sorted(
            path
            for path in work_path.glob(f"{output_prefix}*.pdb")
            if path.resolve() != input_path
        )
        if not candidates:
            detail = output.strip() or "(NOLB produced no diagnostic output.)"
            raise FlexibleFitBackendError(
                "NOLB completed but no output ensemble PDB was found.\n\n"
                f"{detail}"
            )
        ensemble_path = candidates[0]

    return NolbRunResult(
        ensemble_path=ensemble_path,
        command=tuple(command),
        output=output,
        elapsed_seconds=elapsed_seconds,
    )


def _read_pdb_models_with_keys(
    pdb_path: str | Path,
) -> list[tuple[np.ndarray, list[tuple[str, str, str, str, str]]]]:
    path = Path(pdb_path)
    if not path.is_file():
        raise FlexibleFitBackendError(f"PDB file was not found:\n{path}")

    models: list[
        tuple[np.ndarray, list[tuple[str, str, str, str, str]]]
    ] = []
    current: list[tuple[float, float, float]] = []
    current_keys: list[tuple[str, str, str, str, str]] = []
    saw_model_record = False

    def finish_model() -> None:
        nonlocal current, current_keys
        if not current:
            return
        coords = np.asarray(current, dtype=float) / 10.0
        models.append((coords, list(current_keys)))
        current = []
        current_keys = []

    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = line[:6].strip().upper()
            if record == "MODEL":
                if current:
                    finish_model()
                saw_model_record = True
                continue
            if record == "ENDMDL":
                finish_model()
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            try:
                current.append((
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ))
                current_keys.append((
                    line[12:16].strip(),
                    line[17:20].strip(),
                    line[21:22].strip(),
                    line[22:26].strip(),
                    line[26:27].strip(),
                ))
            except (TypeError, ValueError) as exc:
                raise FlexibleFitBackendError(
                    f"Invalid PDB coordinates at {path}:{line_number}."
                ) from exc

    if current:
        finish_model()
    if not models:
        kind = "multi-model " if saw_model_record else ""
        raise FlexibleFitBackendError(
            f"No coordinates were found in the NOLB {kind}PDB:\n{path}"
        )
    return models


def read_pdb_coordinate_models(
    pdb_path: str | Path,
    expected_atom_count: Optional[int] = None,
    expected_atom_keys: Optional[list[tuple[str, str, str, str, str]]] = None,
) -> list[np.ndarray]:
    """Read ATOM/HETATM coordinates from a single- or multi-model PDB in nm."""
    parsed_models = _read_pdb_models_with_keys(pdb_path)
    models = []
    for coords, atom_keys in parsed_models:
        if expected_atom_count is not None and coords.shape[0] != int(
            expected_atom_count
        ):
            raise FlexibleFitBackendError(
                "NOLB output atom count does not match the loaded structure: "
                f"{coords.shape[0]} != {int(expected_atom_count)}."
            )
        if expected_atom_keys is not None and atom_keys != expected_atom_keys:
            raise FlexibleFitBackendError(
                "NOLB output atom identities/order do not match the loaded "
                "structure. The candidate cannot be applied safely."
            )
        models.append(coords)
    return models


def read_mapped_pdb_coordinate_models(
    pdb_path: str | Path,
    reference_coords: np.ndarray,
    reference_atom_keys: list[tuple[str, str, str, str, str]],
) -> list[np.ndarray]:
    """Map NOLB models onto the loaded atoms, preserving collapsed altlocs."""
    reference = np.asarray(reference_coords, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or reference.shape[0] != len(reference_atom_keys)
    ):
        raise FlexibleFitBackendError(
            "Reference coordinates and atom identities do not match."
        )

    reference_groups: dict[
        tuple[str, str, str, str, str], list[int]
    ] = {}
    for index, key in enumerate(reference_atom_keys):
        reference_groups.setdefault(key, []).append(index)

    mapped_models = []
    for output_coords, output_keys in _read_pdb_models_with_keys(pdb_path):
        output_groups: dict[
            tuple[str, str, str, str, str], list[int]
        ] = {}
        for index, key in enumerate(output_keys):
            output_groups.setdefault(key, []).append(index)

        if set(output_groups) != set(reference_groups):
            missing = len(set(reference_groups) - set(output_groups))
            extra = len(set(output_groups) - set(reference_groups))
            raise FlexibleFitBackendError(
                "NOLB output atom identities do not match the loaded "
                f"structure (missing groups: {missing}, extra groups: {extra})."
            )

        mapped = np.array(reference, dtype=float, copy=True)
        for key, reference_indices in reference_groups.items():
            output_indices = output_groups[key]
            if len(output_indices) == len(reference_indices):
                mapped[reference_indices] = output_coords[output_indices]
                continue
            if len(output_indices) == 1 and len(reference_indices) > 1:
                representative = reference_indices[0]
                displacement = (
                    output_coords[output_indices[0]] - reference[representative]
                )
                mapped[reference_indices] = (
                    reference[reference_indices] + displacement
                )
                continue
            raise FlexibleFitBackendError(
                "NOLB changed the multiplicity of atom group "
                f"{key}: {len(reference_indices)} -> {len(output_indices)}."
            )
        mapped_models.append(mapped)
    return mapped_models
