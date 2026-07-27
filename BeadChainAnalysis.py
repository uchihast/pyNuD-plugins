"""
Bead chain fluctuation analysis plugin for pyNuD.

This plugin targets Cordonin-SAHH-like bead-on-string HS-AFM movies. Users
place anchor points along one chain; the plugin traces the ridge, detects bead
centers along the straightened trace, propagates the trace through frames, and
exports per-bead fluctuation metrics.
"""

from __future__ import annotations

import datetime as dt
import csv
import io
import json
import math
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from skimage import filters as skfilters

import globalvals as gv
from fileio import InitializeAryDataFallback, LoadFrame


PLUGIN_NAME = "Bead Chain Analysis"
ANALYSIS_NAME = "Cordion_Analysis"

DEFAULT_FRANGI_SIGMA = 1.5
DEFAULT_RIDGE_WEIGHT = 0.9
DEFAULT_STRIP_HALF_WIDTH_NM = 10.0
DEFAULT_MIN_BEAD_SPACING_NM = 8.0
DEFAULT_MIN_BEAD_HEIGHT = 0.0
DEFAULT_MIN_BEAD_PROMINENCE = 0.2
DEFAULT_DEVIATION_THRESHOLD_NM = 10.0
ENDPOINT_EXTENSION_FACTOR = 0.75
MAX_PROPAGATION_ANCHORS = 9
DEFAULT_BACKBONE_POLY_DEGREE = 3
BACKBONE_SAMPLES = 200


@dataclass
class BeadObservation:
    bead_id: int
    frame_index: int
    s_nm: float
    x_px: float
    y_px: float
    x_nm: float
    y_nm: float
    height: float
    longitudinal_nm: float = float("nan")
    transverse_nm: float = float("nan")


@dataclass
class ChainFrameResult:
    source_path: str
    image_id: str
    frame_index: int
    anchor_points_xy: List[Tuple[float, float]]
    points_yx: np.ndarray
    length_nm: float
    beads: List[BeadObservation]
    diverged: bool = False
    status: str = "ok"
    mean_deviation_nm: float = float("nan")
    max_deviation_nm: float = float("nan")
    max_bead_shift_nm: float = float("nan")


def create_plugin(main_window):
    """pyNuD plugin entry point."""
    return CordionAnalysisWindow(main_window)


def _finite_values(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=float).ravel()
    return values[np.isfinite(values)]


def _relative_height(frame: np.ndarray) -> np.ndarray:
    """Return heights relative to the finite-value 10th percentile."""
    arr = np.asarray(frame, dtype=float)
    finite = _finite_values(arr)
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    baseline = float(np.percentile(finite, 10.0))
    return arr - baseline


def _arc_lengths_nm(points_yx: np.ndarray, nm_x: float, nm_y: float) -> np.ndarray:
    """Return cumulative physical arc length for an ordered y/x path."""
    points = np.asarray(points_yx)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.array([], dtype=float)
    arc = np.zeros(points.shape[0], dtype=float)
    for index in range(1, points.shape[0]):
        dy = float(points[index, 0] - points[index - 1, 0]) * float(nm_y)
        dx = float(points[index, 1] - points[index - 1, 1]) * float(nm_x)
        arc[index] = arc[index - 1] + math.hypot(dx, dy)
    return arc


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _json_float(value: object) -> Optional[float]:
    value_f = _float_or_nan(value)
    return float(value_f) if np.isfinite(value_f) else None


def _format_float(value: object) -> str:
    value_f = _float_or_nan(value)
    return f"{value_f:.8g}" if np.isfinite(value_f) else ""


def _array_to_json_list(array: np.ndarray) -> List:
    arr = np.asarray(array, dtype=float)
    if arr.size == 0:
        return []
    out = arr.astype(object)
    out[~np.isfinite(arr)] = None
    return out.tolist()


def _array_from_payload(value: object, ndim: int, dtype=float) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=dtype)
    except Exception:
        return np.array([], dtype=dtype)
    if arr.ndim != ndim:
        return np.array([], dtype=dtype)
    return arr


def _normalize01(arr: np.ndarray) -> np.ndarray:
    finite = _finite_values(arr)
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=float)
    out = (np.asarray(arr, dtype=float) - vmin) / (vmax - vmin)
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0)


def _sample_bilinear(arr: np.ndarray, y: float, x: float) -> float:
    h, w = arr.shape
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return float("nan")
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    v00 = arr[y0, x0]
    v01 = arr[y0, x1]
    v10 = arr[y1, x0]
    v11 = arr[y1, x1]
    return float((1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01 + (1 - dx) * dy * v10 + dx * dy * v11)


def _straighten_trace_strip(
    rel: np.ndarray,
    points_yx: np.ndarray,
    nm_x: float,
    nm_y: float,
    half_width_nm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a physical-width strip normal to an ordered trace."""
    points = np.asarray(points_yx, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return np.empty((0, 0), dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    half_width = max(0.1, float(half_width_nm))
    sample_step_nm = max(0.1, min(float(nm_x), float(nm_y)))
    offsets_nm = np.arange(
        -half_width,
        half_width + sample_step_nm * 0.5,
        sample_step_nm,
        dtype=float,
    )
    s_nm = _arc_lengths_nm(points.astype(int), nm_x, nm_y)

    strip = np.full((offsets_nm.size, points.shape[0]), np.nan, dtype=float)
    xs_nm = points[:, 1] * float(nm_x)
    ys_nm = points[:, 0] * float(nm_y)

    for index in range(points.shape[0]):
        if index == 0:
            dx = xs_nm[1] - xs_nm[0]
            dy = ys_nm[1] - ys_nm[0]
        elif index == points.shape[0] - 1:
            dx = xs_nm[-1] - xs_nm[-2]
            dy = ys_nm[-1] - ys_nm[-2]
        else:
            dx = xs_nm[index + 1] - xs_nm[index - 1]
            dy = ys_nm[index + 1] - ys_nm[index - 1]

        norm = math.hypot(dx, dy)
        if norm <= 1e-12:
            continue
        normal_x = -(dy / norm)
        normal_y = dx / norm

        for offset_index, offset in enumerate(offsets_nm):
            sample_x_nm = xs_nm[index] + offset * normal_x
            sample_y_nm = ys_nm[index] + offset * normal_y
            strip[offset_index, index] = _sample_bilinear(
                rel,
                sample_y_nm / float(nm_y),
                sample_x_nm / float(nm_x),
            )

    return strip, s_nm, offsets_nm


def compute_ridge_map(roi_image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Return a Frangi ridge response for bright filament-like structures."""
    image = np.asarray(roi_image, dtype=np.float64)
    if image.size == 0 or image.ndim != 2:
        return np.zeros_like(image)
    try:
        ridge = skfilters.frangi(
            image,
            sigmas=[max(0.5, float(sigma))],
            black_ridges=False,
        )
    except Exception:
        ridge = np.zeros_like(image)
    return np.asarray(ridge, dtype=np.float64)


def _angle_deg_between(ay: float, ax: float, by: float, bx: float) -> float:
    norm_a = math.hypot(ay, ax) + 1e-12
    norm_b = math.hypot(by, bx) + 1e-12
    cosine = (ay * by + ax * bx) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.degrees(math.acos(cosine)))


def path_dijkstra_ridge(
    cost_image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_bending_angle_deg: Optional[float] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Find an eight-neighbor minimum-cost ridge path."""
    import heapq

    height, width = cost_image.shape
    start_y, start_x = int(start[0]), int(start[1])
    end_y, end_x = int(end[0]), int(end[1])
    if not (
        0 <= start_y < height
        and 0 <= start_x < width
        and 0 <= end_y < height
        and 0 <= end_x < width
    ):
        return None

    use_bending = max_bending_angle_deg is not None and max_bending_angle_deg > 0
    bend_penalty = 50.0
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    distance_scale = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]

    distances = np.full((height, width), float("inf"), dtype=np.float64)
    distances[start_y, start_x] = 0.0
    previous: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {
        (start_y, start_x): None
    }
    heap: List[Tuple[float, int, int]] = [(0.0, start_y, start_x)]

    while heap:
        distance, y, x = heapq.heappop(heap)
        if (y, x) == (end_y, end_x):
            break
        if distance > distances[y, x]:
            continue
        previous_point = previous.get((y, x))
        for neighbor_index, (dy, dx) in enumerate(neighbors):
            next_y, next_x = y + dy, x + dx
            if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                continue
            if use_bending and previous_point is not None:
                previous_y, previous_x = previous_point
                angle = _angle_deg_between(
                    float(y - previous_y),
                    float(x - previous_x),
                    float(next_y - y),
                    float(next_x - x),
                )
                if angle > max_bending_angle_deg:
                    continue
                bend_extra = (
                    bend_penalty * (angle / max_bending_angle_deg)
                    if angle > max_bending_angle_deg * 0.6
                    else 0.0
                )
            else:
                bend_extra = 0.0

            pixel_cost = float(cost_image[next_y, next_x])
            if not np.isfinite(pixel_cost):
                pixel_cost = 1e6
            next_distance = (
                distance
                + distance_scale[neighbor_index]
                * (0.5 + 0.5 * max(0.0, min(1.0, pixel_cost)))
                + bend_extra
            )
            if next_distance < distances[next_y, next_x]:
                distances[next_y, next_x] = next_distance
                previous[(next_y, next_x)] = (y, x)
                heapq.heappush(heap, (next_distance, next_y, next_x))

    if not np.isfinite(distances[end_y, end_x]):
        return None
    path: List[Tuple[int, int]] = []
    current: Optional[Tuple[int, int]] = (end_y, end_x)
    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()
    return path


def _trace_ridge_segment(
    rel: np.ndarray,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    frangi_sigma: float,
    ridge_weight: float,
) -> Optional[np.ndarray]:
    """Trace one anchor-to-anchor ridge segment using built-in helpers."""
    if rel is None or rel.ndim != 2 or rel.size == 0:
        return None
    h, w = rel.shape
    img = np.asarray(rel, dtype=float)
    smooth = ndimage.gaussian_filter(np.nan_to_num(img, nan=0.0), sigma=1.0, mode="nearest")
    ridge = compute_ridge_map(smooth, sigma=max(0.5, float(frangi_sigma)))
    ridge_n = _normalize01(ridge)
    intensity = _normalize01(smooth)
    rw = max(0.0, min(1.0, float(ridge_weight)))
    cost = 1.0 - (rw * ridge_n + (1.0 - rw) * intensity)
    cost = np.clip(cost, 1e-6, 1.0)

    sy = max(0, min(h - 1, int(round(start_xy[1]))))
    sx = max(0, min(w - 1, int(round(start_xy[0]))))
    ey = max(0, min(h - 1, int(round(end_xy[1]))))
    ex = max(0, min(w - 1, int(round(end_xy[0]))))
    path = path_dijkstra_ridge(cost, (sy, sx), (ey, ex), max_bending_angle_deg=None)
    if path is None or len(path) < 2:
        return None
    return np.asarray(path, dtype=int)


def _trace_anchor_points_with_params(
    rel: np.ndarray,
    anchor_points_xy: Sequence[Tuple[float, float]],
    frangi_sigma: float,
    ridge_weight: float,
) -> Optional[np.ndarray]:
    """Trace a complete chain through all clicked or propagated anchors."""
    if len(anchor_points_xy) < 2:
        return None
    combined: List[Tuple[int, int]] = []
    anchors: List[Tuple[float, float]] = []
    for x, y in anchor_points_xy:
        point = (float(x), float(y))
        if anchors and math.hypot(point[0] - anchors[-1][0], point[1] - anchors[-1][1]) < 1.0:
            continue
        anchors.append(point)
    if len(anchors) < 2:
        return None
    for idx in range(len(anchors) - 1):
        segment = _trace_ridge_segment(rel, anchors[idx], anchors[idx + 1], frangi_sigma, ridge_weight)
        if segment is None or segment.shape[0] < 2:
            return None
        points = [tuple(map(int, point)) for point in segment]
        if not combined:
            combined.extend(points)
        else:
            combined.extend(points[1:])
    if len(combined) < 2:
        return None
    return np.asarray(combined, dtype=int)


def _resample_path_as_anchors(points_yx: np.ndarray, count: int = MAX_PROPAGATION_ANCHORS) -> List[Tuple[float, float]]:
    """Convert a traced path into a sparse ordered anchor list."""
    points = np.asarray(points_yx, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return []
    n = min(max(2, int(count)), points.shape[0])
    indices = np.linspace(0, points.shape[0] - 1, n)
    anchors: List[Tuple[float, float]] = []
    last: Optional[Tuple[int, int]] = None
    for idx in indices:
        y, x = points[int(round(idx))]
        key = (int(round(y)), int(round(x)))
        if key == last:
            continue
        anchors.append((float(x), float(y)))
        last = key
    return anchors if len(anchors) >= 2 else []


def _snap_anchors_to_local_signal(
    rel: np.ndarray,
    anchors_xy: Sequence[Tuple[float, float]],
    radius_px: int,
) -> List[Tuple[float, float]]:
    """Move propagated anchors to the local bright ridge maximum."""
    if rel is None or rel.ndim != 2:
        return list(anchors_xy)
    h, w = rel.shape
    radius = max(1, int(radius_px))
    signal_img = ndimage.gaussian_filter(np.nan_to_num(rel, nan=0.0), sigma=1.0, mode="nearest")
    snapped: List[Tuple[float, float]] = []
    for x, y in anchors_xy:
        xi = max(0, min(w - 1, int(round(x))))
        yi = max(0, min(h - 1, int(round(y))))
        x0 = max(0, xi - radius)
        x1 = min(w, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1 = min(h, yi + radius + 1)
        crop = signal_img[y0:y1, x0:x1]
        if crop.size == 0 or not np.any(np.isfinite(crop)):
            snapped.append((float(xi), float(yi)))
            continue
        local_index = int(np.nanargmax(crop))
        yy, xx = np.unravel_index(local_index, crop.shape)
        snapped.append((float(x0 + xx), float(y0 + yy)))
    return snapped


def _path_deviation_nm(
    previous_yx: np.ndarray,
    current_yx: np.ndarray,
    nm_x: float,
    nm_y: float,
) -> Tuple[float, float]:
    """Return mean and max nearest-neighbor path deviation in nm."""
    prev = np.asarray(previous_yx, dtype=float)
    cur = np.asarray(current_yx, dtype=float)
    if prev.ndim != 2 or cur.ndim != 2 or prev.shape[0] < 2 or cur.shape[0] < 2:
        return float("nan"), float("nan")
    prev_nm = np.column_stack((prev[:, 1] * nm_x, prev[:, 0] * nm_y))
    cur_nm = np.column_stack((cur[:, 1] * nm_x, cur[:, 0] * nm_y))
    tree = cKDTree(cur_nm)
    distances, _indices = tree.query(prev_nm, k=1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(distances)), float(np.max(distances))


def _path_tangent_at_s(
    points_yx: np.ndarray,
    arc_nm: np.ndarray,
    s_nm: float,
    nm_x: float,
    nm_y: float,
) -> Tuple[float, float]:
    """Return local unit tangent as (tx, ty) in nm coordinates."""
    points = np.asarray(points_yx, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return 1.0, 0.0
    idx = int(np.searchsorted(arc_nm, s_nm))
    idx = max(0, min(points.shape[0] - 1, idx))
    i0 = max(0, idx - 1)
    i1 = min(points.shape[0] - 1, idx + 1)
    if i0 == i1:
        return 1.0, 0.0
    dx = (points[i1, 1] - points[i0, 1]) * nm_x
    dy = (points[i1, 0] - points[i0, 0]) * nm_y
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return 1.0, 0.0
    return float(dx / norm), float(dy / norm)


def _nearest_path_s_nm(
    points_yx: np.ndarray,
    arc_nm: np.ndarray,
    x_nm: float,
    y_nm: float,
    nm_x: float,
    nm_y: float,
) -> float:
    points = np.asarray(points_yx, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or arc_nm.size == 0:
        return float("nan")
    dx = points[:, 1] * nm_x - float(x_nm)
    dy = points[:, 0] * nm_y - float(y_nm)
    idx = int(np.argmin(dx * dx + dy * dy))
    return float(arc_nm[max(0, min(idx, arc_nm.size - 1))])


def _merge_close_beads(
    beads: List["BeadObservation"],
    min_spacing_nm: float,
) -> List["BeadObservation"]:
    """Merge beads that resolve to nearly the same location.

    Endpoint extension can create an extra peak whose s_nm is snapped back to the
    chain endpoint, producing a near-duplicate of the real end bead. Drop such
    duplicates by 2D Euclidean distance, keeping the taller observation.
    """
    if len(beads) < 2:
        return list(beads)
    threshold_nm = max(0.0, float(min_spacing_nm)) * 0.85
    if threshold_nm <= 1e-9:
        return list(beads)
    kept: List["BeadObservation"] = []
    for bead in beads:
        duplicate_idx = -1
        for idx, existing in enumerate(kept):
            if math.hypot(bead.x_nm - existing.x_nm, bead.y_nm - existing.y_nm) < threshold_nm:
                duplicate_idx = idx
                break
        if duplicate_idx < 0:
            kept.append(bead)
        elif bead.height > kept[duplicate_idx].height:
            kept[duplicate_idx] = bead
    return kept


def _dedupe_consecutive_points(points_yx: Sequence[Tuple[int, int]]) -> np.ndarray:
    deduped: List[Tuple[int, int]] = []
    last: Optional[Tuple[int, int]] = None
    for y, x in points_yx:
        point = (int(y), int(x))
        if point == last:
            continue
        deduped.append(point)
        last = point
    return np.asarray(deduped, dtype=int)


def _extend_path_endpoints(
    points_yx: np.ndarray,
    image_shape: Tuple[int, int],
    nm_x: float,
    nm_y: float,
    extension_nm: float,
) -> np.ndarray:
    """Extend a traced path at both ends for endpoint bead peak detection."""
    points = np.asarray(points_yx, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return np.asarray(points_yx, dtype=int)
    extension = max(0.0, float(extension_nm))
    if extension <= 1e-9:
        return np.asarray(points_yx, dtype=int)

    h, w = image_shape
    step_nm = max(0.1, min(float(nm_x), float(nm_y)))
    n_steps = max(1, int(math.ceil(extension / step_nm)))

    points_nm = np.column_stack((points[:, 1] * nm_x, points[:, 0] * nm_y))
    arc = _arc_lengths_nm(np.rint(points).astype(int), nm_x, nm_y)
    lookahead_nm = max(extension * 0.5, step_nm)

    start_idx = int(np.searchsorted(arc, lookahead_nm))
    start_idx = max(1, min(points.shape[0] - 1, start_idx))
    end_target = float(arc[-1] - lookahead_nm) if arc.size else 0.0
    end_idx = int(np.searchsorted(arc, end_target))
    end_idx = max(0, min(points.shape[0] - 2, end_idx))

    start_vec = points_nm[0] - points_nm[start_idx]
    end_vec = points_nm[-1] - points_nm[end_idx]

    def unit(vec: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.hypot(vec[0], vec[1]))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return vec / norm

    start_unit = unit(start_vec)
    end_unit = unit(end_vec)
    if start_unit is None and end_unit is None:
        return np.asarray(points_yx, dtype=int)

    extended: List[Tuple[int, int]] = []
    if start_unit is not None:
        for step in range(n_steps, 0, -1):
            distance = min(extension, step * step_nm)
            x_nm = points_nm[0, 0] + start_unit[0] * distance
            y_nm = points_nm[0, 1] + start_unit[1] * distance
            x = max(0, min(w - 1, int(round(x_nm / max(nm_x, 1e-9)))))
            y = max(0, min(h - 1, int(round(y_nm / max(nm_y, 1e-9)))))
            extended.append((y, x))

    extended.extend((int(round(y)), int(round(x))) for y, x in points)

    if end_unit is not None:
        for step in range(1, n_steps + 1):
            distance = min(extension, step * step_nm)
            x_nm = points_nm[-1, 0] + end_unit[0] * distance
            y_nm = points_nm[-1, 1] + end_unit[1] * distance
            x = max(0, min(w - 1, int(round(x_nm / max(nm_x, 1e-9)))))
            y = max(0, min(h - 1, int(round(y_nm / max(nm_y, 1e-9)))))
            extended.append((y, x))

    out = _dedupe_consecutive_points(extended)
    return out if out.shape[0] >= 2 else np.asarray(points_yx, dtype=int)


def _fit_smooth_backbone(
    points_yx: np.ndarray,
    nm_x: float,
    nm_y: float,
    degree: int,
    samples: int = BACKBONE_SAMPLES,
) -> np.ndarray:
    """Fit a smooth filament backbone through ordered points.

    The points (typically bead centres) are rotated into their PCA main-axis
    frame, a low-order polynomial ``v = poly(u)`` is fit, then sampled densely
    and rotated back. This yields a smooth filament axis spanning all points and
    ignoring individual bead fluctuations. Returns float (y, x) pixel coords.
    """
    pts = np.asarray(points_yx, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts
    x_nm = pts[:, 1] * nm_x
    y_nm = pts[:, 0] * nm_y
    coords = np.column_stack((x_nm, y_nm))
    mean = coords.mean(axis=0)
    centered = coords - mean
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    except Exception:
        axis = np.array([1.0, 0.0])
    norm = float(np.hypot(axis[0], axis[1]))
    if not np.isfinite(norm) or norm <= 1e-12:
        return pts
    axis = axis / norm
    perp = np.array([-axis[1], axis[0]])
    u = centered @ axis
    v = centered @ perp
    order = np.argsort(u)
    u_sorted = u[order]
    v_sorted = v[order]
    if float(u_sorted[-1] - u_sorted[0]) < 1e-6:
        return pts
    deg = max(1, min(int(degree), u_sorted.size - 1))
    try:
        coeffs = np.polyfit(u_sorted, v_sorted, deg)
    except Exception:
        coeffs = np.polyfit(u_sorted, v_sorted, 1)
    u_dense = np.linspace(u_sorted[0], u_sorted[-1], max(2, int(samples)))
    v_dense = np.polyval(coeffs, u_dense)
    bx = mean[0] + u_dense * axis[0] + v_dense * perp[0]
    by = mean[1] + u_dense * axis[1] + v_dense * perp[1]
    return np.column_stack((by / max(nm_y, 1e-9), bx / max(nm_x, 1e-9)))


class CordionAnalysisWindow(QtWidgets.QWidget):
    """Main widget for bead-chain tracing, propagation, and export."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.frame: Optional[np.ndarray] = None
        self.rel: Optional[np.ndarray] = None
        self.anchor_points_xy: List[Tuple[float, float]] = []
        self.current_path_yx: Optional[np.ndarray] = None
        self.preview_beads: List[BeadObservation] = []
        self.preview_diverged = False
        self.results: List[ChainFrameResult] = []
        self.reference_beads: Dict[int, Tuple[float, float]] = {}
        self.reference_bead_order: List[int] = []
        self.last_result: Optional[ChainFrameResult] = None
        self._signal_connected = False
        self._suppress_frame_changed = False
        self._last_propagation_failure_reason = ""
        self._drag_bead: Optional[BeadObservation] = None
        self._drag_context: Optional[Tuple[List[BeadObservation], np.ndarray, Optional[ChainFrameResult]]] = None
        self._preview_lut_rgb: Optional[np.ndarray] = None
        self._preview_lut_frame: int = -1
        self._updating_preview_lut = False

        self.setWindowTitle(PLUGIN_NAME)
        self.setWindowFlags(QtCore.Qt.Window)
        self.resize(1120, 780)
        self._setup_ui()
        self._connect_frame_signal()
        self.refresh_frame()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.source_label = QtWidgets.QLabel("No file loaded.")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        view_panel = QtWidgets.QWidget()
        view_layout = QtWidgets.QVBoxLayout(view_panel)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        view_layout.addWidget(self.toolbar)
        view_layout.addWidget(self.canvas, stretch=1)
        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        splitter.addWidget(view_panel)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 0, 0, 0)
        splitter.addWidget(controls)
        splitter.setSizes([780, 340])

        button_grid = QtWidgets.QGridLayout()
        controls_layout.addLayout(button_grid)
        self.add_anchor_button = QtWidgets.QPushButton("Anchor Add")
        self.add_anchor_button.setCheckable(True)
        self.add_anchor_button.setChecked(True)
        self.add_bead_button = QtWidgets.QPushButton("Add/Fix Bead")
        self.add_bead_button.setCheckable(True)
        self.add_bead_button.setToolTip(
            "When active, click on a missing/failed bead to re-detect its centroid "
            "nearby and add it. Clicking near an existing bead repositions it."
        )
        self.modify_bead_button = QtWidgets.QPushButton("Modify")
        self.modify_bead_button.setCheckable(True)
        self.modify_bead_button.setToolTip(
            "Drag a bead marker to the correct location. "
            "Releasing the mouse searches locally around the drop point and confirms the centroid."
        )
        self.delete_bead_button = QtWidgets.QPushButton("Delete Bead")
        self.delete_bead_button.setCheckable(True)
        self.delete_bead_button.setToolTip(
            "Click a bead marker to remove it from the current frame."
        )
        self.store_button = QtWidgets.QPushButton("Store")
        self.store_button.setToolTip(
            "Save the current bead positions. After Modify, Store keeps the edited positions "
            "instead of re-detecting from scratch."
        )
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.setToolTip(
            "Clear the current frame's stored auto-detection (if any) and preview, "
            "then place two end anchors manually and Store again."
        )
        self.prev_frame_button = QtWidgets.QPushButton("◀ Prev Frame")
        self.next_frame_button = QtWidgets.QPushButton("Next Frame ▶")
        self.propagate_button = QtWidgets.QPushButton("Next Auto")
        self.run_all_button = QtWidgets.QPushButton("Run all")
        self.export_button = QtWidgets.QPushButton("Export")
        self.save_session_button = QtWidgets.QPushButton("Save Session")
        self.load_session_button = QtWidgets.QPushButton("Load Session")

        buttons = [
            self.add_anchor_button,
            self.add_bead_button,
            self.modify_bead_button,
            self.delete_bead_button,
            self.store_button,
            self.clear_button,
            self.prev_frame_button,
            self.next_frame_button,
            self.propagate_button,
            self.run_all_button,
            self.export_button,
            self.save_session_button,
            self.load_session_button,
        ]
        for idx, button in enumerate(buttons):
            button_grid.addWidget(button, idx // 2, idx % 2)

        self.add_anchor_button.toggled.connect(self._on_add_anchor_toggled)
        self.add_bead_button.toggled.connect(self._on_add_bead_toggled)
        self.modify_bead_button.toggled.connect(self._on_modify_bead_toggled)
        self.delete_bead_button.toggled.connect(self._on_delete_bead_toggled)
        self.store_button.clicked.connect(self.store_current_line)
        self.clear_button.clicked.connect(self.clear_current)
        self.prev_frame_button.clicked.connect(lambda: self.goto_frame_delta(-1))
        self.next_frame_button.clicked.connect(lambda: self.goto_frame_delta(1))
        self.propagate_button.clicked.connect(self.propagate_next_frame)
        self.run_all_button.clicked.connect(self.run_all_frames)
        self.export_button.clicked.connect(self.export_results)
        self.save_session_button.clicked.connect(self.save_session)
        self.load_session_button.clicked.connect(self.load_session)

        params_box = QtWidgets.QGroupBox("Parameters")
        params_form = QtWidgets.QFormLayout(params_box)
        controls_layout.addWidget(params_box)

        self.frangi_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.frangi_sigma_spin.setRange(0.5, 10.0)
        self.frangi_sigma_spin.setDecimals(2)
        self.frangi_sigma_spin.setSingleStep(0.25)
        self.frangi_sigma_spin.setValue(DEFAULT_FRANGI_SIGMA)
        params_form.addRow("Frangi sigma (px)", self.frangi_sigma_spin)

        self.ridge_weight_spin = QtWidgets.QDoubleSpinBox()
        self.ridge_weight_spin.setRange(0.0, 1.0)
        self.ridge_weight_spin.setDecimals(2)
        self.ridge_weight_spin.setSingleStep(0.05)
        self.ridge_weight_spin.setValue(DEFAULT_RIDGE_WEIGHT)
        params_form.addRow("Ridge weight", self.ridge_weight_spin)

        self.strip_half_width_spin = QtWidgets.QDoubleSpinBox()
        self.strip_half_width_spin.setRange(1.0, 200.0)
        self.strip_half_width_spin.setDecimals(1)
        self.strip_half_width_spin.setSingleStep(1.0)
        self.strip_half_width_spin.setSuffix(" nm")
        self.strip_half_width_spin.setValue(DEFAULT_STRIP_HALF_WIDTH_NM)
        params_form.addRow("Strip half width", self.strip_half_width_spin)

        self.backbone_degree_spin = QtWidgets.QSpinBox()
        self.backbone_degree_spin.setRange(1, 6)
        self.backbone_degree_spin.setValue(DEFAULT_BACKBONE_POLY_DEGREE)
        self.backbone_degree_spin.setToolTip(
            "Polynomial degree for the smooth filament backbone fit through bead "
            "centres. Lower = smoother overall shape (ignores bead flutter)."
        )
        params_form.addRow("Backbone poly degree", self.backbone_degree_spin)

        self.remove_drift_check = QtWidgets.QCheckBox("Remove common Z drift")
        self.remove_drift_check.setToolTip(
            "XY fluctuations are measured against each frame's own backbone, so "
            "rigid stage drift/rotation is already removed for longitudinal/"
            "transverse. This option additionally subtracts the per-frame common "
            "height offset so whole-chain Z drift is not counted as Z fluctuation."
        )
        params_form.addRow("Drift", self.remove_drift_check)

        self.z_reference_combo = QtWidgets.QComboBox()
        self.z_reference_combo.addItem("Relative height (raw)", "raw")
        self.z_reference_combo.addItem("Minus per-frame bead-height baseline", "frame_baseline")
        self.z_reference_combo.setToolTip(
            "Reference for the bead Z (height) value used in fluctuation analysis. "
            "'Frame baseline' subtracts each frame's mean bead height to suppress "
            "per-frame leveling/background drift."
        )
        params_form.addRow("Z reference", self.z_reference_combo)
        self.remove_drift_check.toggled.connect(self._on_analysis_option_changed)
        self.z_reference_combo.currentIndexChanged.connect(self._on_analysis_option_changed)

        self.min_spacing_spin = QtWidgets.QDoubleSpinBox()
        self.min_spacing_spin.setRange(0.5, 500.0)
        self.min_spacing_spin.setDecimals(1)
        self.min_spacing_spin.setSingleStep(1.0)
        self.min_spacing_spin.setSuffix(" nm")
        self.min_spacing_spin.setValue(DEFAULT_MIN_BEAD_SPACING_NM)
        params_form.addRow("Min bead spacing", self.min_spacing_spin)

        self.min_height_spin = QtWidgets.QDoubleSpinBox()
        self.min_height_spin.setRange(-100000.0, 100000.0)
        self.min_height_spin.setDecimals(3)
        self.min_height_spin.setSingleStep(0.1)
        self.min_height_spin.setValue(DEFAULT_MIN_BEAD_HEIGHT)
        params_form.addRow("Min bead height", self.min_height_spin)

        self.prominence_spin = QtWidgets.QDoubleSpinBox()
        self.prominence_spin.setRange(0.0, 100000.0)
        self.prominence_spin.setDecimals(3)
        self.prominence_spin.setSingleStep(0.1)
        self.prominence_spin.setValue(DEFAULT_MIN_BEAD_PROMINENCE)
        params_form.addRow("Peak prominence", self.prominence_spin)

        self.deviation_spin = QtWidgets.QDoubleSpinBox()
        self.deviation_spin.setRange(0.5, 1000.0)
        self.deviation_spin.setDecimals(1)
        self.deviation_spin.setSingleStep(1.0)
        self.deviation_spin.setSuffix(" nm")
        self.deviation_spin.setValue(DEFAULT_DEVIATION_THRESHOLD_NM)
        params_form.addRow("Deviation threshold", self.deviation_spin)

        movie_box = QtWidgets.QGroupBox("Movie export")
        movie_layout = QtWidgets.QVBoxLayout(movie_box)
        self.export_movie_button = QtWidgets.QPushButton("Export MP4 (bead overlay)")
        self.export_movie_button.setToolTip(
            "Save an MP4 with stored bead traces overlaid on the AFM movie. "
            "Playback speed follows ASD FrameTime (real time). "
            "Timestamp and X scale bar follow the Save tab checkboxes."
        )
        self.export_movie_hint = QtWidgets.QLabel(
            "Real-time playback: FPS = 1000 / FrameTime [ms]. "
            "Timestamp follows Save tab 'Add Time Caption'; "
            "X scale bar on the first frame only when 'Add Scale Bar' is enabled."
        )
        self.export_movie_hint.setWordWrap(True)
        self.export_movie_hint.setStyleSheet("color: #666; font-size: 11px;")
        movie_layout.addWidget(self.export_movie_button)
        movie_layout.addWidget(self.export_movie_hint)
        controls_layout.addWidget(movie_box)
        self.export_movie_button.clicked.connect(self.export_movie)

        self.result_list = QtWidgets.QListWidget()
        self.result_list.setMinimumHeight(160)
        controls_layout.addWidget(QtWidgets.QLabel("Stored frame results"))
        controls_layout.addWidget(self.result_list, stretch=1)

        self.status_label = QtWidgets.QLabel(
            "Click Anchor Add points along one bead chain, then Store."
        )
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

    def _connect_frame_signal(self) -> None:
        if self.main_window is not None and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.connect(self._on_frame_changed)
                self._signal_connected = True
            except Exception:
                self._signal_connected = False

    def cleanup_on_unload(self) -> None:
        """Disconnect pyNuD signals when the plugin is unloaded."""
        if self._signal_connected and self.main_window is not None and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.disconnect(self._on_frame_changed)
            except Exception:
                pass
        self._signal_connected = False

    def closeEvent(self, event) -> None:
        self.cleanup_on_unload()
        event.accept()

    def _on_frame_changed(self, _frame_index: int) -> None:
        if self._suppress_frame_changed:
            return
        self.refresh_frame(clear_working=True)

    def onMainWindowFileChanged(self) -> None:
        """Reset analysis state when pyNuD switches to another file."""
        self.anchor_points_xy = []
        self.current_path_yx = None
        self.preview_beads = []
        self.results = []
        self.reference_beads = {}
        self.reference_bead_order = []
        self.last_result = None
        self._preview_lut_rgb = None
        self._preview_lut_frame = -1
        self.refresh_frame(clear_working=True)
        if self.rel is not None:
            self._try_auto_initial_detection()
            self._redraw()

    def _current_file_path(self) -> str:
        files = getattr(gv, "files", None)
        current = int(getattr(gv, "currentFileNum", 0) or 0)
        if isinstance(files, str):
            return files if files and current == 0 else ""
        if not files or current < 0 or current >= len(files):
            return ""
        return str(files[current])

    def _path_key(self, path: str) -> str:
        return os.path.abspath(path) if path else ""

    def _current_file_key(self) -> str:
        return self._path_key(self._current_file_path())

    def _current_image_id(self) -> str:
        path = self._current_file_path()
        return os.path.splitext(os.path.basename(path))[0] if path else "current_image"

    def _afm_file_stem(self) -> str:
        """AFM data filename without extension."""
        path = self._current_file_path()
        if not path and self.results:
            path = str(self.results[0].source_path)
        return os.path.splitext(os.path.basename(path))[0] if path else "current_image"

    def _export_base_name(self) -> str:
        """Base name for zip/json/png exports: AFM stem + '_BeadsAna'."""
        return f"{self._afm_file_stem()}_BeadsAna"

    def _current_frame_index(self) -> int:
        return int(getattr(gv, "index", 0) or 0)

    def _get_nm_per_pixel(self) -> Tuple[float, float]:
        x_scan = float(getattr(gv, "XScanSize", 0.0) or 0.0)
        y_scan = float(getattr(gv, "YScanSize", 0.0) or 0.0)
        x_pixels = int(getattr(gv, "XPixel", 0) or 0)
        y_pixels = int(getattr(gv, "YPixel", 0) or 0)
        if x_scan > 0 and y_scan > 0 and x_pixels > 0 and y_pixels > 0:
            return x_scan / x_pixels, y_scan / y_pixels
        return 1.0, 1.0

    def _frame_time_s(self, frame_index: int) -> float:
        frame_time_ms = _float_or_nan(getattr(gv, "FrameTime", float("nan")))
        if np.isfinite(frame_time_ms) and frame_time_ms > 0:
            return float(frame_index) * frame_time_ms / 1000.0
        return float("nan")

    def _realtime_movie_fps(self) -> Tuple[float, float]:
        """Return (fps, frame_time_ms) for real-time playback from ASD FrameTime."""
        frame_time_ms = _float_or_nan(getattr(gv, "FrameTime", float("nan")))
        if np.isfinite(frame_time_ms) and frame_time_ms > 0:
            return 1000.0 / float(frame_time_ms), float(frame_time_ms)
        return 10.0, 100.0

    def _frame_from_globals(self, prefer_processed: bool) -> Optional[np.ndarray]:
        data = None
        if prefer_processed and hasattr(gv, "aryData_processed_1ch"):
            data = gv.aryData_processed_1ch
        if data is None:
            data = getattr(gv, "aryData", None)
        if data is None:
            return None
        frame = np.asarray(data, dtype=float)
        if frame.ndim != 2:
            return None
        return frame

    def _load_current_frame(self) -> Optional[np.ndarray]:
        path = self._current_file_path()
        if not path:
            return None
        try:
            LoadFrame(path)
            InitializeAryDataFallback()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load frame:\n{exc}")
            return None
        return self._frame_from_globals(prefer_processed=True)

    def _load_frame_at(self, frame_index: int) -> Optional[np.ndarray]:
        path = self._current_file_path()
        if not path:
            return None
        try:
            gv.index = max(0, int(frame_index))
            LoadFrame(path)
            InitializeAryDataFallback()
        except Exception:
            return None
        return self._frame_from_globals(prefer_processed=False)

    def _snapshot_global_frame_state(self) -> Dict[str, object]:
        ary_data = getattr(gv, "aryData", None)
        processed = getattr(gv, "aryData_processed_1ch", None)
        return {
            "currentFileNum": getattr(gv, "currentFileNum", None),
            "index": getattr(gv, "index", None),
            "path": self._current_file_path(),
            "aryData": ary_data.copy() if isinstance(ary_data, np.ndarray) else ary_data,
            "aryData_processed_1ch": processed.copy() if isinstance(processed, np.ndarray) else processed,
        }

    def _restore_global_frame_state(self, state: Dict[str, object], update_main: bool = True) -> None:
        try:
            if state.get("currentFileNum") is not None:
                gv.currentFileNum = state["currentFileNum"]
            if state.get("index") is not None:
                gv.index = state["index"]
            path = state.get("path")
            if path and os.path.exists(str(path)):
                try:
                    LoadFrame(str(path))
                    InitializeAryDataFallback()
                except Exception:
                    gv.aryData = state.get("aryData")
            else:
                gv.aryData = state.get("aryData")
        finally:
            gv.aryData_processed_1ch = state.get("aryData_processed_1ch")
            if state.get("currentFileNum") is not None:
                gv.currentFileNum = state["currentFileNum"]
            if state.get("index") is not None:
                gv.index = state["index"]
            if update_main:
                self._sync_main_window_frame(int(getattr(gv, "index", 0) or 0), emit=False)

    def _sync_main_window_frame(self, frame_index: int, emit: bool) -> None:
        gv.index = int(frame_index)
        if self.main_window is None:
            return
        previous_suppression = self._suppress_frame_changed
        self._suppress_frame_changed = True
        try:
            slider = getattr(self.main_window, "frameSlider", None)
            if slider is not None:
                try:
                    blocked = slider.blockSignals(True)
                    slider.setValue(int(frame_index))
                    slider.blockSignals(blocked)
                except Exception:
                    pass
            updater = getattr(self.main_window, "updateFrame", None)
            if callable(updater):
                try:
                    updater()
                except Exception:
                    pass
            if emit and hasattr(self.main_window, "frameChanged"):
                try:
                    self.main_window.frameChanged.emit(int(frame_index))
                except Exception:
                    pass
        finally:
            self._suppress_frame_changed = previous_suppression

    def _move_to_frame(self, frame_index: int) -> bool:
        frame = self._load_frame_at(frame_index)
        if frame is None:
            return False
        self.frame = frame
        self.rel = _relative_height(frame)
        self._sync_main_window_frame(frame_index, emit=False)
        return True

    def refresh_frame(self, clear_working: bool = False) -> None:
        self._clear_bead_drag()
        frame = self._load_current_frame()
        self.frame = frame
        self.rel = _relative_height(frame) if frame is not None else None
        if clear_working:
            self.anchor_points_xy = []
            self.current_path_yx = None
            self.preview_beads = []
            self.preview_diverged = False
        self._refresh_source_label()
        self._update_result_list()
        self._update_preview_lut_cache()
        self._redraw()

    def _refresh_source_label(self) -> None:
        path = self._current_file_path()
        if not path:
            self.source_label.setText("No file loaded. Open an AFM movie first.")
            return
        frame_num = int(getattr(gv, "FrameNum", 0) or 0)
        current_frame = self._current_frame_index()
        x_pixels = int(getattr(gv, "XPixel", 0) or 0)
        y_pixels = int(getattr(gv, "YPixel", 0) or 0)
        x_scan = float(getattr(gv, "XScanSize", 0.0) or 0.0)
        y_scan = float(getattr(gv, "YScanSize", 0.0) or 0.0)
        count = len(self._results_for_current_file())
        beads = len(self._result_for_current_frame().beads) if self._result_for_current_frame() else len(self.preview_beads)
        status = "diverged" if self.preview_diverged else "ok"
        self.source_label.setText(
            f"{path}\n"
            f"frame {current_frame + 1}/{max(frame_num, 1)}, "
            f"{x_pixels}x{y_pixels} px, {x_scan:g}x{y_scan:g} nm, "
            f"stored frames: {count}, beads: {beads}, status: {status}"
        )

    def _results_for_current_file(self) -> List[ChainFrameResult]:
        key = self._current_file_key()
        return [row for row in self.results if self._path_key(row.source_path) == key]

    def _result_for_current_frame(self) -> Optional[ChainFrameResult]:
        current = self._current_frame_index()
        for row in self._results_for_current_file():
            if row.frame_index == current:
                return row
        return None

    def _deactivate_other_bead_tools(self, keep: Optional[QtWidgets.QPushButton] = None) -> None:
        for button in (
            self.add_anchor_button,
            self.add_bead_button,
            self.modify_bead_button,
            self.delete_bead_button,
        ):
            if button is not keep and button.isChecked():
                button.setChecked(False)

    def _on_add_anchor_toggled(self, checked: bool) -> None:
        if checked:
            self._deactivate_other_bead_tools(self.add_anchor_button)

    def _on_add_bead_toggled(self, checked: bool) -> None:
        if checked:
            self._deactivate_other_bead_tools(self.add_bead_button)

    def _clear_bead_drag(self) -> None:
        self._drag_bead = None
        self._drag_context = None

    def _on_modify_bead_toggled(self, checked: bool) -> None:
        if not checked:
            self._clear_bead_drag()
            return
        self._deactivate_other_bead_tools(self.modify_bead_button)

    def _on_delete_bead_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self._clear_bead_drag()
        self._deactivate_other_bead_tools(self.delete_bead_button)

    def _clamp_px(self, x_px: float, y_px: float) -> Tuple[float, float]:
        if self.rel is None:
            return x_px, y_px
        h, w = self.rel.shape
        return max(0.0, min(float(w - 1), x_px)), max(0.0, min(float(h - 1), y_px))

    def _bead_pick_radius_px(self) -> float:
        nm_x, nm_y = self._get_nm_per_pixel()
        spacing_px = float(self.min_spacing_spin.value()) / max(min(nm_x, nm_y), 1e-9)
        return max(12.0, spacing_px * 0.45)

    def _editable_bead_context(
        self,
        allow_empty_beads: bool = False,
    ) -> Optional[Tuple[List[BeadObservation], np.ndarray, Optional[ChainFrameResult]]]:
        if (
            self.preview_beads
            and self.current_path_yx is not None
            and self.current_path_yx.shape[0] >= 2
            and (self.preview_diverged or not self._result_for_current_frame())
        ):
            return self.preview_beads, self.current_path_yx, None
        stored = self._result_for_current_frame()
        if stored is not None:
            return stored.beads, stored.points_yx, stored
        if self.current_path_yx is not None and self.current_path_yx.shape[0] >= 2:
            if self.preview_beads or allow_empty_beads:
                return self.preview_beads, self.current_path_yx, None
        return None

    def _pick_bead_index(self, x_px: float, y_px: float, beads: Sequence[BeadObservation]) -> int:
        if not beads:
            return -1
        pick_radius = self._bead_pick_radius_px()
        best_idx = -1
        best_dist = pick_radius
        for idx, bead in enumerate(beads):
            dist = math.hypot(float(bead.x_px) - x_px, float(bead.y_px) - y_px)
            if dist <= best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _local_refine_radius_nm(self) -> float:
        """Small window for the weighted centroid after a 2D peak is found."""
        nm_x, nm_y = self._get_nm_per_pixel()
        min_spacing_nm = float(self.min_spacing_spin.value())
        return max(min(nm_x, nm_y) * 0.8, min_spacing_nm * 0.18)

    def _bead_search_radius_nm(self, *, tracking: bool = False) -> float:
        """Disk radius to search for the 2D intensity peak of one bead."""
        nm_x, nm_y = self._get_nm_per_pixel()
        min_spacing_nm = float(self.min_spacing_spin.value())
        half_width_nm = float(self.strip_half_width_spin.value())
        if tracking:
            return max(min(nm_x, nm_y) * 1.2, min_spacing_nm * 0.28, half_width_nm * 0.55)
        cap = min(min_spacing_nm * 0.4, half_width_nm * 0.85)
        return max(min(nm_x, nm_y) * 1.2, cap)

    def _locate_bead_center_2d(
        self,
        rel: np.ndarray,
        y_px: float,
        x_px: float,
        search_radius_nm: float,
        min_height: float,
        nm_x: float,
        nm_y: float,
    ) -> Tuple[float, float]:
        """Locate a bead by 2D argmax in a disk, then a small weighted centroid.

        The seed point only defines where to search; the final position is the
        centroid of the brightest local blob, not a point on the filament path.
        """
        x_px, y_px = self._clamp_px(x_px, y_px)
        sr_x = max(1, int(math.ceil(search_radius_nm / max(nm_x, 1e-9))))
        sr_y = max(1, int(math.ceil(search_radius_nm / max(nm_y, 1e-9))))
        cx_i = int(round(x_px))
        cy_i = int(round(y_px))
        x0 = max(0, cx_i - sr_x)
        x1 = min(rel.shape[1], cx_i + sr_x + 1)
        y0 = max(0, cy_i - sr_y)
        y1 = min(rel.shape[0], cy_i + sr_y + 1)
        crop = np.asarray(rel[y0:y1, x0:x1], dtype=float)
        if crop.size == 0 or not np.any(np.isfinite(crop)):
            return x_px, y_px
        peak_yy, peak_xx = np.unravel_index(int(np.nanargmax(crop)), crop.shape)
        peak_x = float(x0 + peak_xx)
        peak_y = float(y0 + peak_yy)
        centroid = self._centroid_at_point(
            rel,
            peak_y,
            peak_x,
            self._local_refine_radius_nm(),
            min_height,
            nm_x,
            nm_y,
            recenter=False,
        )
        if centroid is None:
            return peak_x, peak_y
        return centroid

    def _clamp_shift_from_point(
        self,
        x_px: float,
        y_px: float,
        cx: float,
        cy: float,
        max_shift_nm: float,
        nm_x: float,
        nm_y: float,
    ) -> Tuple[float, float]:
        shift_nm = math.hypot((cx - x_px) * nm_x, (cy - y_px) * nm_y)
        if shift_nm > max_shift_nm and shift_nm > 1e-9:
            scale = max_shift_nm / shift_nm
            cx = float(x_px) + (cx - float(x_px)) * scale
            cy = float(y_px) + (cy - float(y_px)) * scale
        return cx, cy

    def _commit_bead_list(self, beads: List[BeadObservation], points_yx: np.ndarray, stored: Optional[ChainFrameResult]) -> None:
        beads.sort(key=lambda b: b.s_nm)
        for bead_id, bead in enumerate(beads, start=1):
            bead.bead_id = bead_id
        nm_x, nm_y = self._get_nm_per_pixel()
        backbone = self._build_backbone_from_beads(beads, points_yx, nm_x, nm_y)
        arc = _arc_lengths_nm(backbone, nm_x, nm_y)
        for bead in beads:
            bead.s_nm = _nearest_path_s_nm(backbone, arc, bead.x_nm, bead.y_nm, nm_x, nm_y)
        beads.sort(key=lambda b: b.s_nm)
        for bead_id, bead in enumerate(beads, start=1):
            bead.bead_id = bead_id
        if stored is not None:
            stored.points_yx = np.asarray(backbone, dtype=float)
            stored.length_nm = float(arc[-1]) if arc.size else stored.length_nm
            self._update_manual_result_status(stored)
            if not self.reference_beads or len(beads) != len(self.reference_bead_order):
                self._set_reference_from_result(stored)
            self._recompute_fluctuations()
            self.last_result = stored
        else:
            self.current_path_yx = np.asarray(backbone, dtype=float)
            self.preview_beads = beads
            self.preview_diverged = not self._preview_spacing_ok(beads)

    def _set_bead_position_from_pixel(
        self,
        bead: BeadObservation,
        x_px: float,
        y_px: float,
        beads: List[BeadObservation],
        points_yx: np.ndarray,
        stored: Optional[ChainFrameResult],
        local_only: bool = False,
        manual_confirm: bool = False,
    ) -> None:
        if self.rel is None:
            return
        x_px, y_px = self._clamp_px(x_px, y_px)
        nm_x, nm_y = self._get_nm_per_pixel()
        min_height = float(self.min_height_spin.value())
        if manual_confirm:
            search_nm = self._bead_search_radius_nm(tracking=False)
            max_shift_nm = search_nm
        elif local_only:
            search_nm = self._local_refine_radius_nm()
            max_shift_nm = search_nm
        else:
            search_nm = self._bead_search_radius_nm(tracking=False)
            max_shift_nm = search_nm
        cx, cy = self._locate_bead_center_2d(self.rel, y_px, x_px, search_nm, min_height, nm_x, nm_y)
        cx, cy = self._clamp_shift_from_point(x_px, y_px, cx, cy, max_shift_nm, nm_x, nm_y)
        bead.x_px, bead.y_px = cx, cy
        bead.x_nm = cx * nm_x
        bead.y_nm = cy * nm_y
        arc = _arc_lengths_nm(points_yx, nm_x, nm_y)
        bead.s_nm = _nearest_path_s_nm(points_yx, arc, bead.x_nm, bead.y_nm, nm_x, nm_y)
        bead.height = _sample_bilinear(self.rel, cy, cx)
        self._commit_bead_list(beads, points_yx, stored)

    def _delete_bead_at(self, x_px: float, y_px: float) -> None:
        ctx = self._editable_bead_context()
        if ctx is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Beads",
                "Detect or store beads on this frame before deleting one.",
            )
            return
        beads, points_yx, stored = ctx
        idx = self._pick_bead_index(x_px, y_px, beads)
        if idx < 0:
            self.status_label.setText("Click directly on a bead marker to delete it.")
            return
        removed_id = int(beads[idx].bead_id)
        beads.pop(idx)
        self._commit_bead_list(beads, points_yx, stored)
        self.status_label.setText(f"Deleted bead {removed_id}. Remaining beads: {len(beads)}.")
        self._refresh_source_label()
        self._update_result_list()
        self._redraw()

    def _on_canvas_press(self, event) -> None:
        if self.rel is None or event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = self._clamp_px(float(event.xdata), float(event.ydata))
        if self.delete_bead_button.isChecked():
            self._delete_bead_at(x, y)
            return
        if self.modify_bead_button.isChecked():
            ctx = self._editable_bead_context()
            if ctx is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Beads",
                    "Detect or store beads on this frame before modifying positions.",
                )
                return
            beads, points_yx, stored = ctx
            idx = self._pick_bead_index(x, y, beads)
            if idx < 0:
                return
            self._drag_bead = beads[idx]
            self._drag_context = (beads, points_yx, stored)
            self._drag_bead.x_px = x
            self._drag_bead.y_px = y
            self._redraw()
            return
        if self.add_bead_button.isChecked():
            self._add_or_fix_bead_at(x, y)
            return
        if not self.add_anchor_button.isChecked():
            return
        self.anchor_points_xy.append((x, y))
        self._update_current_path()
        self._redraw()

    def _on_canvas_motion(self, event) -> None:
        if self._drag_bead is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = self._clamp_px(float(event.xdata), float(event.ydata))
        self._drag_bead.x_px = x
        self._drag_bead.y_px = y
        self._redraw()

    def _on_canvas_release(self, event) -> None:
        if self._drag_bead is None or event.button != 1:
            return
        bead = self._drag_bead
        ctx = self._drag_context
        self._drag_bead = None
        self._drag_context = None
        if ctx is None:
            return
        beads, points_yx, stored = ctx
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            x, y = self._clamp_px(float(event.xdata), float(event.ydata))
        else:
            x, y = float(bead.x_px), float(bead.y_px)
        self._set_bead_position_from_pixel(bead, x, y, beads, points_yx, stored, manual_confirm=True)
        self.status_label.setText(
            f"Bead {bead.bead_id} confirmed near ({bead.x_px:.1f}, {bead.y_px:.1f}) px."
        )
        self._refresh_source_label()
        self._update_result_list()
        self._redraw()

    def _add_or_fix_bead_at(self, x_px: float, y_px: float) -> None:
        """Re-detect a bead centroid near a click and add (or reposition) it.

        Targets the detection shown on the current frame: a stored result if one
        exists, otherwise the live preview. Clicking near an existing bead moves
        that bead; clicking elsewhere adds a new one.
        """
        ctx = self._editable_bead_context(allow_empty_beads=True)
        if ctx is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Chain",
                "Detect or draw a bead chain first, then click a missing bead to add it.",
            )
            return
        beads, points_yx, stored = ctx
        x_px, y_px = self._clamp_px(x_px, y_px)

        replace_idx = -1
        replace_thr_px = self._bead_pick_radius_px()
        for i, bead in enumerate(beads):
            if math.hypot(float(bead.x_px) - x_px, float(bead.y_px) - y_px) < replace_thr_px:
                replace_idx = i
                break
        if replace_idx >= 0:
            bead = beads[replace_idx]
            action = "repositioned"
            local_only = True
        else:
            bead = BeadObservation(
                bead_id=0,
                frame_index=self._current_frame_index(),
                s_nm=0.0,
                x_px=x_px,
                y_px=y_px,
                x_nm=0.0,
                y_nm=0.0,
                height=float("nan"),
            )
            beads.append(bead)
            action = "added"
            local_only = False
        self._set_bead_position_from_pixel(bead, x_px, y_px, beads, points_yx, stored, local_only=local_only)
        self.status_label.setText(
            f"Bead {action} at ({bead.x_px:.1f}, {bead.y_px:.1f}) px. Total beads: {len(beads)}."
        )
        self._refresh_source_label()
        self._update_result_list()
        self._redraw()

    def _update_current_path(self) -> None:
        self.current_path_yx = None
        self.preview_beads = []
        self.preview_diverged = False
        if self.rel is None:
            return
        if len(self.anchor_points_xy) < 2:
            self.status_label.setText(f"{len(self.anchor_points_xy)} anchor point(s). Add at least two.")
            return
        path = _trace_anchor_points_with_params(
            self.rel,
            self.anchor_points_xy,
            self.frangi_sigma_spin.value(),
            self.ridge_weight_spin.value(),
        )
        if path is None or path.shape[0] < 2:
            self.status_label.setText("Trace failed. Add an intermediate anchor along the chain.")
            return
        self.current_path_yx = path
        nm_x, nm_y = self._get_nm_per_pixel()
        arc = _arc_lengths_nm(path, nm_x, nm_y)
        self.preview_beads = self._detect_beads(self.rel, path)
        length_nm = float(arc[-1]) if arc.size else 0.0
        self.status_label.setText(
            f"Current line: {length_nm:.1f} nm, detected beads: {len(self.preview_beads)}. Click Store."
        )
        self._refresh_source_label()

    def _centroid_at_point(
        self,
        rel: np.ndarray,
        y_px: float,
        x_px: float,
        radius_nm: float,
        min_height: float,
        nm_x: float,
        nm_y: float,
        recenter: bool = False,
    ) -> Optional[Tuple[float, float]]:
        """Weighted centroid of the bead blob inside a window around a pixel.

        When ``recenter`` is set, the window is first shifted onto the local
        brightest pixel (bounded by ``radius_nm``) so a bead that drifted between
        frames is still captured, without reaching a neighbouring bead.
        """
        h, w = rel.shape
        rx = max(1, int(math.ceil(radius_nm / max(nm_x, 1e-9))))
        ry = max(1, int(math.ceil(radius_nm / max(nm_y, 1e-9))))
        cx = int(round(x_px))
        cy = int(round(y_px))
        if recenter:
            sx0 = max(0, cx - rx)
            sx1 = min(w, cx + rx + 1)
            sy0 = max(0, cy - ry)
            sy1 = min(h, cy + ry + 1)
            seed = np.asarray(rel[sy0:sy1, sx0:sx1], dtype=float)
            if seed.size and np.any(np.isfinite(seed)):
                syy, sxx = np.unravel_index(int(np.nanargmax(seed)), seed.shape)
                cx = sx0 + sxx
                cy = sy0 + syy
        x0 = max(0, cx - rx)
        x1 = min(w, cx + rx + 1)
        y0 = max(0, cy - ry)
        y1 = min(h, cy + ry + 1)
        crop = np.asarray(rel[y0:y1, x0:x1], dtype=float)
        if crop.size == 0:
            return None
        finite_crop = crop[np.isfinite(crop)]
        if finite_crop.size == 0:
            return None
        local_base = float(np.percentile(finite_crop, 20.0))
        threshold = local_base
        if min_height > 0:
            threshold = max(threshold, min_height)
        weights = np.clip(np.nan_to_num(crop - threshold, nan=0.0), 0.0, None)
        if float(np.sum(weights)) <= 1e-12:
            weights = np.clip(np.nan_to_num(crop - local_base, nan=0.0), 0.0, None)
        if float(np.sum(weights)) <= 1e-12:
            yy, xx = np.unravel_index(int(np.nanargmax(crop)), crop.shape)
            return float(x0 + xx), float(y0 + yy)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        total = float(np.sum(weights))
        return float(np.sum(weights * xx) / total), float(np.sum(weights * yy) / total)

    def _centroid_from_peak(
        self,
        rel: np.ndarray,
        path_yx: np.ndarray,
        peak_idx: int,
        min_height: float,
        nm_x: float,
        nm_y: float,
    ) -> Optional[Tuple[float, float]]:
        """Refine one profile peak to a 2D bead centre (path supplies seed only)."""
        idx = max(0, min(path_yx.shape[0] - 1, int(peak_idx)))
        seed_y = float(path_yx[idx, 0])
        seed_x = float(path_yx[idx, 1])
        search_nm = self._bead_search_radius_nm(tracking=False)
        return self._locate_bead_center_2d(rel, seed_y, seed_x, search_nm, min_height, nm_x, nm_y)

    def _track_beads_from_previous(
        self,
        rel: np.ndarray,
        points_yx: np.ndarray,
        previous_beads: Sequence[BeadObservation],
    ) -> List[BeadObservation]:
        """Track beads with a small local centroid around each previous position."""
        nm_x, nm_y = self._get_nm_per_pixel()
        min_spacing_nm = float(self.min_spacing_spin.value())
        min_height = float(self.min_height_spin.value())
        centroid_radius_nm = max(self._local_refine_radius_nm(), min_spacing_nm * 0.28)
        max_shift_nm = max(min(nm_x, nm_y) * 2.5, min_spacing_nm * 0.45)
        arc = _arc_lengths_nm(points_yx, nm_x, nm_y)
        beads: List[BeadObservation] = []
        for prev in sorted(previous_beads, key=lambda b: b.s_nm):
            x_px0 = float(prev.x_nm) / max(nm_x, 1e-9)
            y_px0 = float(prev.y_nm) / max(nm_y, 1e-9)
            centroid = self._centroid_at_point(
                rel, y_px0, x_px0, centroid_radius_nm, min_height, nm_x, nm_y, recenter=False
            )
            if centroid is None:
                x_px, y_px = x_px0, y_px0
            else:
                x_px, y_px = centroid
            x_px, y_px = self._clamp_shift_from_point(x_px0, y_px0, x_px, y_px, max_shift_nm, nm_x, nm_y)
            x_nm = x_px * nm_x
            y_nm = y_px * nm_y
            s_value = _nearest_path_s_nm(points_yx, arc, x_nm, y_nm, nm_x, nm_y)
            beads.append(
                BeadObservation(
                    bead_id=int(prev.bead_id),
                    frame_index=self._current_frame_index(),
                    s_nm=s_value,
                    x_px=x_px,
                    y_px=y_px,
                    x_nm=x_nm,
                    y_nm=y_nm,
                    height=_sample_bilinear(rel, y_px, x_px),
                )
            )
        return beads

    def _detect_beads(self, rel: np.ndarray, points_yx: np.ndarray) -> List[BeadObservation]:
        """Detect bead centers from the straightened strip peak profile."""
        nm_x, nm_y = self._get_nm_per_pixel()
        half_width_nm = float(self.strip_half_width_spin.value())
        min_spacing_nm = float(self.min_spacing_spin.value())
        endpoint_extension_nm = max(min_spacing_nm * ENDPOINT_EXTENSION_FACTOR, min(nm_x, nm_y) * 2.0)
        detection_path_yx = _extend_path_endpoints(
            points_yx,
            rel.shape,
            nm_x,
            nm_y,
            endpoint_extension_nm,
        )
        strip, s_nm, _offsets_nm = _straighten_trace_strip(rel, detection_path_yx, nm_x, nm_y, half_width_nm)
        if strip.size == 0 or s_nm.size < 2:
            return []
        profile = np.nanmax(strip, axis=0)
        if not np.any(np.isfinite(profile)):
            return []
        finite = profile[np.isfinite(profile)]
        fill_value = float(np.nanmedian(finite)) if finite.size else 0.0
        profile = np.where(np.isfinite(profile), profile, fill_value)
        step_nm = float(np.nanmedian(np.diff(s_nm))) if s_nm.size > 2 else min(nm_x, nm_y)
        if not np.isfinite(step_nm) or step_nm <= 0:
            step_nm = max(0.1, min(nm_x, nm_y))
        smooth_sigma = max(0.5, min(4.0, min_spacing_nm / max(step_nm, 1e-9) / 8.0))
        smooth = ndimage.gaussian_filter1d(profile, sigma=smooth_sigma, mode="nearest")
        distance_px = max(1, int(round(min_spacing_nm / max(step_nm, 1e-9))))
        min_height = float(self.min_height_spin.value())
        min_prominence = float(self.prominence_spin.value())
        height_arg = min_height if min_height > 0 else None
        prominence_arg = min_prominence if min_prominence > 0 else None
        peaks, _props = signal.find_peaks(
            smooth,
            distance=distance_px,
            height=height_arg,
            prominence=prominence_arg,
        )
        if peaks.size == 0 and min_height <= 0 and min_prominence > 0:
            peaks, _props = signal.find_peaks(smooth, distance=distance_px)
        beads: List[BeadObservation] = []
        arc = _arc_lengths_nm(points_yx, nm_x, nm_y)
        peak_order = sorted(peaks, key=lambda idx: float(smooth[int(idx)]), reverse=True)
        for peak in peak_order:
            centroid = self._centroid_from_peak(rel, detection_path_yx, peak, min_height, nm_x, nm_y)
            if centroid is None:
                continue
            x_px, y_px = centroid
            x_nm = x_px * nm_x
            y_nm = y_px * nm_y
            if any(
                math.hypot(x_nm - existing.x_nm, y_nm - existing.y_nm) < min_spacing_nm
                for existing in beads
            ):
                continue
            height = _sample_bilinear(rel, y_px, x_px)
            s_value = _nearest_path_s_nm(points_yx, arc, x_nm, y_nm, nm_x, nm_y)
            beads.append(
                BeadObservation(
                    bead_id=len(beads) + 1,
                    frame_index=self._current_frame_index(),
                    s_nm=s_value,
                    x_px=x_px,
                    y_px=y_px,
                    x_nm=x_nm,
                    y_nm=y_nm,
                    height=height,
                )
            )
        beads.sort(key=lambda bead: bead.s_nm)
        beads = _merge_close_beads(beads, min_spacing_nm)
        for bead_id, bead in enumerate(beads, start=1):
            bead.bead_id = bead_id
        return beads

    def _make_result_from_path(
        self,
        rel: np.ndarray,
        points_yx: np.ndarray,
        anchors_xy: Sequence[Tuple[float, float]],
        frame_index: int,
        template_ids: Optional[Sequence[int]] = None,
        template_beads: Optional[Sequence[BeadObservation]] = None,
    ) -> ChainFrameResult:
        if template_beads:
            beads = self._track_beads_from_previous(rel, points_yx, template_beads)
        else:
            beads = self._detect_beads(rel, points_yx)
        if template_ids is not None and len(template_ids) == len(beads):
            for bead, bead_id in zip(beads, template_ids):
                bead.bead_id = int(bead_id)
        else:
            for bead_id, bead in enumerate(beads, start=1):
                bead.bead_id = bead_id
        for bead in beads:
            bead.frame_index = int(frame_index)
        nm_x, nm_y = self._get_nm_per_pixel()
        backbone = self._build_backbone_from_beads(beads, points_yx, nm_x, nm_y)
        arc = _arc_lengths_nm(backbone, nm_x, nm_y)
        for bead in beads:
            bead.s_nm = _nearest_path_s_nm(backbone, arc, bead.x_nm, bead.y_nm, nm_x, nm_y)
        result = ChainFrameResult(
            source_path=self._current_file_path(),
            image_id=self._current_image_id(),
            frame_index=int(frame_index),
            anchor_points_xy=[(float(x), float(y)) for x, y in anchors_xy],
            points_yx=np.asarray(backbone, dtype=float).copy(),
            length_nm=float(arc[-1]) if arc.size else 0.0,
            beads=beads,
        )
        self._assign_fluctuation_components(result)
        return result

    def _make_result_from_existing_beads(
        self,
        beads: Sequence[BeadObservation],
        path_yx: np.ndarray,
        anchors_xy: Sequence[Tuple[float, float]],
        frame_index: int,
    ) -> ChainFrameResult:
        """Build a stored result from edited bead positions without re-detecting peaks."""
        beads_list = list(beads)
        nm_x, nm_y = self._get_nm_per_pixel()
        backbone = self._build_backbone_from_beads(beads_list, path_yx, nm_x, nm_y)
        arc = _arc_lengths_nm(backbone, nm_x, nm_y)
        for bead in beads_list:
            bead.frame_index = int(frame_index)
            bead.s_nm = _nearest_path_s_nm(backbone, arc, bead.x_nm, bead.y_nm, nm_x, nm_y)
        beads_list.sort(key=lambda bead: bead.s_nm)
        for bead_id, bead in enumerate(beads_list, start=1):
            bead.bead_id = bead_id
        result = ChainFrameResult(
            source_path=self._current_file_path(),
            image_id=self._current_image_id(),
            frame_index=int(frame_index),
            anchor_points_xy=[(float(x), float(y)) for x, y in anchors_xy],
            points_yx=np.asarray(backbone, dtype=float).copy(),
            length_nm=float(arc[-1]) if arc.size else 0.0,
            beads=beads_list,
        )
        self._assign_fluctuation_components(result)
        self._update_manual_result_status(result)
        return result

    def _previous_stored_result(self, frame_index: int) -> Optional[ChainFrameResult]:
        key = self._current_file_key()
        if not key:
            return None
        candidates = [
            row
            for row in self.results
            if self._path_key(row.source_path) == key and row.frame_index < frame_index
        ]
        if candidates:
            return max(candidates, key=lambda row: row.frame_index)
        if (
            self.last_result is not None
            and self._path_key(self.last_result.source_path) == key
            and self.last_result.frame_index < frame_index
        ):
            return self.last_result
        return None

    def _preview_spacing_ok(self, beads: Sequence[BeadObservation]) -> bool:
        min_spacing_nm = float(self.min_spacing_spin.value())
        min_adjacent = self._min_adjacent_bead_spacing_nm(beads)
        close_threshold_nm = min_spacing_nm * 0.85
        return not (np.isfinite(min_adjacent) and min_adjacent < close_threshold_nm)

    def _update_manual_result_status(self, result: ChainFrameResult) -> None:
        """Re-evaluate spacing and propagation checks after manual bead edits."""
        min_spacing_nm = float(self.min_spacing_spin.value())
        min_adjacent = self._min_adjacent_bead_spacing_nm(result.beads)
        close_threshold_nm = min_spacing_nm * 0.85
        reasons: List[str] = []
        previous = self._previous_stored_result(result.frame_index)
        if previous is not None:
            if len(result.beads) != len(previous.beads):
                reasons.append(f"bead count {len(previous.beads)} -> {len(result.beads)}")
            nm_x, nm_y = self._get_nm_per_pixel()
            mean_dev, max_dev = _path_deviation_nm(previous.points_yx, result.points_yx, nm_x, nm_y)
            result.mean_deviation_nm = mean_dev
            result.max_deviation_nm = max_dev
            result.max_bead_shift_nm = self._max_bead_shift(previous, result)
            threshold = float(self.deviation_spin.value())
            if np.isfinite(max_dev) and max_dev > threshold:
                reasons.append(f"max line shift {max_dev:.1f} nm (> {threshold:.1f} nm)")
        if np.isfinite(min_adjacent) and min_adjacent < close_threshold_nm:
            reasons.append(f"min bead spacing {min_adjacent:.1f} nm (< {close_threshold_nm:.1f} nm)")
        if reasons:
            result.diverged = True
            result.status = "; ".join(reasons)
        else:
            result.diverged = False
            result.status = "ok"

    def _build_backbone_from_beads(
        self,
        beads: Sequence[BeadObservation],
        fallback_yx: np.ndarray,
        nm_x: float,
        nm_y: float,
    ) -> np.ndarray:
        """Smooth filament backbone spanning all beads (PCA polynomial fit)."""
        ordered = sorted(beads, key=lambda b: b.s_nm)
        if len(ordered) >= 2:
            bead_pts_yx = np.array([[b.y_px, b.x_px] for b in ordered], dtype=float)
            degree = self._backbone_degree()
            backbone = _fit_smooth_backbone(bead_pts_yx, nm_x, nm_y, degree)
            if backbone is not None and np.asarray(backbone).shape[0] >= 2:
                return np.asarray(backbone, dtype=float)
        return np.asarray(fallback_yx, dtype=float)

    def _backbone_degree(self) -> int:
        spin = getattr(self, "backbone_degree_spin", None)
        if spin is not None:
            return int(spin.value())
        return DEFAULT_BACKBONE_POLY_DEGREE

    def _assign_fluctuation_components(self, result: ChainFrameResult) -> None:
        """Project bead displacement from the reference frame into tangent axes."""
        if not self.reference_beads:
            for bead in result.beads:
                bead.longitudinal_nm = 0.0
                bead.transverse_nm = 0.0
            return
        nm_x, nm_y = self._get_nm_per_pixel()
        arc = _arc_lengths_nm(result.points_yx, nm_x, nm_y)
        for bead in result.beads:
            ref = self.reference_beads.get(bead.bead_id)
            if ref is None:
                bead.longitudinal_nm = float("nan")
                bead.transverse_nm = float("nan")
                continue
            tx, ty = _path_tangent_at_s(result.points_yx, arc, bead.s_nm, nm_x, nm_y)
            nx, ny = -ty, tx
            dx = float(bead.x_nm) - float(ref[0])
            dy = float(bead.y_nm) - float(ref[1])
            bead.longitudinal_nm = float(dx * tx + dy * ty)
            bead.transverse_nm = float(dx * nx + dy * ny)

    def _set_reference_from_result(self, result: ChainFrameResult) -> None:
        self.reference_beads = {bead.bead_id: (float(bead.x_nm), float(bead.y_nm)) for bead in result.beads}
        self.reference_bead_order = [bead.bead_id for bead in sorted(result.beads, key=lambda b: b.s_nm)]
        self._assign_fluctuation_components(result)

    def _drift_removal_enabled(self) -> bool:
        chk = getattr(self, "remove_drift_check", None)
        return bool(chk is not None and chk.isChecked())

    def _z_reference_mode(self) -> str:
        combo = getattr(self, "z_reference_combo", None)
        if combo is not None:
            data = combo.currentData()
            if data:
                return str(data)
        return "raw"

    def _frame_mean_height(self, result: ChainFrameResult) -> float:
        hs = [float(b.height) for b in result.beads if np.isfinite(b.height)]
        return float(np.mean(hs)) if hs else float("nan")

    def _effective_z(self, result: ChainFrameResult, bead: BeadObservation) -> float:
        """Bead Z value for analysis, honoring the selected Z reference and drift."""
        z = float(bead.height)
        if not np.isfinite(z):
            return float("nan")
        if self._z_reference_mode() == "frame_baseline":
            base = self._frame_mean_height(result)
            if np.isfinite(base):
                z = z - base
        if self._drift_removal_enabled():
            common = self._common_z_offset(result)
            if np.isfinite(common):
                z = z - common
        return z

    def _common_z_offset(self, result: ChainFrameResult) -> float:
        """Per-frame common-mode height (mean over beads of the z-reference value)."""
        vals = []
        use_baseline = self._z_reference_mode() == "frame_baseline"
        base = self._frame_mean_height(result) if use_baseline else 0.0
        for bead in result.beads:
            if not np.isfinite(bead.height):
                continue
            v = float(bead.height) - (base if use_baseline and np.isfinite(base) else 0.0)
            vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    def _on_analysis_option_changed(self, *_args) -> None:
        if not getattr(self, "results", None):
            return
        self._recompute_fluctuations()
        self._refresh_source_label()
        self._update_result_list()
        self._redraw()

    def _compute_mean_backbone(self, results: Sequence[ChainFrameResult]) -> Optional[np.ndarray]:
        """Stable reference backbone (pixel y,x) from time-averaged bead positions."""
        nm_x, nm_y = self._get_nm_per_pixel()
        positions: Dict[int, List[Tuple[float, float]]] = {}
        for result in results:
            for bead in result.beads:
                positions.setdefault(bead.bead_id, []).append((float(bead.x_nm), float(bead.y_nm)))
        if not positions:
            return None
        ordered_ids = sorted(positions.keys())
        mean_pts_yx = np.array(
            [
                [
                    float(np.mean([p[1] for p in positions[i]])) / max(nm_y, 1e-9),
                    float(np.mean([p[0] for p in positions[i]])) / max(nm_x, 1e-9),
                ]
                for i in ordered_ids
            ],
            dtype=float,
        )
        if mean_pts_yx.shape[0] >= 2:
            return _fit_smooth_backbone(mean_pts_yx, nm_x, nm_y, self._backbone_degree())
        return mean_pts_yx

    def _recompute_fluctuations(self) -> None:
        """Intrinsic (per-frame backbone) fluctuation decomposition.

        For every frame we use that frame's own smooth filament backbone
        (``result.points_yx``) as a co-moving reference. Two quantities are
        measured per bead:

        - ``transverse_nm``: signed perpendicular offset of the bead from the
          per-frame backbone. Because the backbone moves and rotates with the
          filament, this is invariant to rigid translation/rotation (drift).
        - ``longitudinal_nm``: bead arc-position along the backbone, referenced
          to the chain centroid (mean arc-position over beads in that frame), so
          along-axis translation is removed.

        Both are stored as deviations from each bead's time-mean, so the std of
        the stored values is the intrinsic fluctuation amplitude.
        """
        nm_x, nm_y = self._get_nm_per_pixel()
        by_file: Dict[str, List[ChainFrameResult]] = {}
        for result in self.results:
            by_file.setdefault(self._path_key(result.source_path), []).append(result)
        for results in by_file.values():
            raw_long: Dict[int, List[float]] = {}
            raw_trans: Dict[int, List[float]] = {}
            frame_entries: List[Tuple[ChainFrameResult, Dict[int, Tuple[float, float]]]] = []
            for result in results:
                backbone = np.asarray(result.points_yx, dtype=float)
                if backbone.ndim != 2 or backbone.shape[0] < 2:
                    for bead in result.beads:
                        bead.longitudinal_nm = float("nan")
                        bead.transverse_nm = float("nan")
                    continue
                arc = _arc_lengths_nm(backbone, nm_x, nm_y)
                bx = backbone[:, 1] * nm_x
                by = backbone[:, 0] * nm_y
                s_map: Dict[int, float] = {}
                perp_map: Dict[int, float] = {}
                for bead in result.beads:
                    ddx = bx - float(bead.x_nm)
                    ddy = by - float(bead.y_nm)
                    idx = int(np.argmin(ddx * ddx + ddy * ddy))
                    s = float(arc[max(0, min(idx, arc.size - 1))])
                    tx, ty = _path_tangent_at_s(backbone, arc, s, nm_x, nm_y)
                    nx, ny = -ty, tx
                    perp = (float(bead.x_nm) - float(bx[idx])) * nx + (float(bead.y_nm) - float(by[idx])) * ny
                    s_map[bead.bead_id] = s
                    perp_map[bead.bead_id] = float(perp)
                centroid_s = float(np.mean(list(s_map.values()))) if s_map else 0.0
                entry: Dict[int, Tuple[float, float]] = {}
                for bid in s_map:
                    rel_s = s_map[bid] - centroid_s
                    entry[bid] = (rel_s, perp_map[bid])
                    raw_long.setdefault(bid, []).append(rel_s)
                    raw_trans.setdefault(bid, []).append(perp_map[bid])
                frame_entries.append((result, entry))
            mean_long = {bid: float(np.mean(vals)) for bid, vals in raw_long.items() if vals}
            mean_trans = {bid: float(np.mean(vals)) for bid, vals in raw_trans.items() if vals}
            for result, entry in frame_entries:
                for bead in result.beads:
                    if bead.bead_id in entry:
                        rel_s, perp = entry[bead.bead_id]
                        bead.longitudinal_nm = float(rel_s - mean_long.get(bead.bead_id, 0.0))
                        bead.transverse_nm = float(perp - mean_trans.get(bead.bead_id, 0.0))
                    else:
                        bead.longitudinal_nm = float("nan")
                        bead.transverse_nm = float("nan")

    def store_current_line(self) -> None:
        """Store or overwrite the current frame trace and bead detections."""
        if self.rel is None:
            QtWidgets.QMessageBox.information(self, "No Frame", "Load an AFM frame first.")
            return
        frame_index = self._current_frame_index()
        if (
            self.preview_beads
            and self.current_path_yx is not None
            and self.current_path_yx.shape[0] >= 2
        ):
            result = self._make_result_from_existing_beads(
                self.preview_beads,
                self.current_path_yx,
                self.anchor_points_xy,
                frame_index,
            )
        else:
            stored = self._result_for_current_frame()
            if stored is not None and stored.beads and not self.anchor_points_xy:
                result = self._make_result_from_existing_beads(
                    stored.beads,
                    stored.points_yx,
                    stored.anchor_points_xy,
                    frame_index,
                )
            else:
                if self.current_path_yx is None:
                    self._update_current_path()
                if self.current_path_yx is None or self.current_path_yx.shape[0] < 2:
                    QtWidgets.QMessageBox.information(self, "No Line", "Draw a valid bead-chain line first.")
                    return
                template_ids = (
                    self.reference_bead_order
                    if self.reference_bead_order and len(self.reference_bead_order) == len(self.preview_beads)
                    else None
                )
                result = self._make_result_from_path(
                    self.rel,
                    self.current_path_yx,
                    self.anchor_points_xy,
                    frame_index,
                    template_ids=template_ids,
                )
        if not result.beads:
            QtWidgets.QMessageBox.information(self, "No Beads", "No bead peaks were detected. Adjust height/prominence/spacing.")
            return
        if self.reference_beads and len(result.beads) != len(self.reference_bead_order):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace Reference",
                "Detected bead count differs from the reference. Replace the reference bead set?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer == QtWidgets.QMessageBox.Yes:
                self._set_reference_from_result(result)
        elif not self.reference_beads:
            self._set_reference_from_result(result)
        else:
            self._assign_fluctuation_components(result)
        self._store_result(result)
        self._recompute_fluctuations()
        self.last_result = result
        self.anchor_points_xy = []
        self.current_path_yx = None
        self.preview_beads = []
        self.preview_diverged = False
        if result.diverged:
            self.status_label.setText(
                f"Stored frame {result.frame_index + 1}: {len(result.beads)} beads "
                f"({result.status}). Adjust beads or spacing, then Store again."
            )
        else:
            self.status_label.setText(
                f"Stored frame {result.frame_index + 1}: {len(result.beads)} beads, length {result.length_nm:.1f} nm."
            )
        self._refresh_source_label()
        self._update_result_list()
        self._redraw()

    def _store_result(self, result: ChainFrameResult) -> None:
        key = self._path_key(result.source_path)
        self.results = [
            row
            for row in self.results
            if not (self._path_key(row.source_path) == key and row.frame_index == result.frame_index)
        ]
        self.results.append(result)
        self.results.sort(key=lambda row: (self._path_key(row.source_path), row.frame_index))

    def _remove_result_for_frame(self, frame_index: int) -> bool:
        """Drop the stored detection for one frame on the current file."""
        key = self._current_file_key()
        if not key:
            return False
        before = len(self.results)
        self.results = [
            row
            for row in self.results
            if not (self._path_key(row.source_path) == key and row.frame_index == frame_index)
        ]
        return len(self.results) < before

    def clear_current(self) -> None:
        """Clear the current frame and restart manual end-anchor detection."""
        self._clear_bead_drag()
        frame_index = self._current_frame_index()
        removed = self._remove_result_for_frame(frame_index)
        self.anchor_points_xy = []
        self.current_path_yx = None
        self.preview_beads = []
        self.preview_diverged = False
        if removed:
            self._recompute_fluctuations()
            if (
                self.last_result is not None
                and self.last_result.frame_index == frame_index
                and self._path_key(self.last_result.source_path) == self._current_file_key()
            ):
                file_results = self._results_for_current_file()
                self.last_result = file_results[-1] if file_results else None
        if not self.add_anchor_button.isChecked():
            self.add_anchor_button.setChecked(True)
        else:
            self._deactivate_other_bead_tools(self.add_anchor_button)
        if removed:
            self.status_label.setText(
                f"Frame {frame_index + 1}: auto detection cleared. "
                "Click two end anchors, then Store."
            )
        else:
            self.status_label.setText(
                f"Frame {frame_index + 1}: preview cleared. "
                "Click two end anchors, then Store."
            )
        self._refresh_source_label()
        self._update_result_list()
        self._redraw()

    def _try_auto_initial_detection(self) -> None:
        """Try to seed the first line from the strongest ridge in a new file."""
        if self.rel is None or self.rel.ndim != 2:
            return
        try:
            ridge = compute_ridge_map(self.rel, sigma=self.frangi_sigma_spin.value())
            score = _normalize01(ridge) * 0.7 + _normalize01(self.rel) * 0.3
            finite = score[np.isfinite(score)]
            if finite.size < 10:
                return
            cutoff = float(np.percentile(finite, 97.0))
            ys, xs = np.nonzero(score >= cutoff)
            if xs.size < 2:
                return
            if xs.size > 250:
                order = np.argsort(score[ys, xs])[-250:]
                xs = xs[order]
                ys = ys[order]
            coords = np.column_stack((xs.astype(float), ys.astype(float)))
            centered = coords - np.mean(coords, axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            projection = centered @ axis
            p0 = coords[int(np.argmin(projection))]
            p1 = coords[int(np.argmax(projection))]
            if np.linalg.norm(p1 - p0) < 5:
                return
            anchors = [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))]
            path = _trace_anchor_points_with_params(
                self.rel,
                anchors,
                self.frangi_sigma_spin.value(),
                self.ridge_weight_spin.value(),
            )
            if path is None or path.shape[0] < 2:
                return
            beads = self._detect_beads(self.rel, path)
            if len(beads) < 2:
                return
            self.anchor_points_xy = anchors
            self.current_path_yx = path
            self.preview_beads = beads
            self.status_label.setText(
                f"Auto initial line found ({len(beads)} beads). Review it, or click new anchors to overwrite."
            )
        except Exception:
            self.status_label.setText("Auto initial detection failed. Set the chain manually.")

    def _set_propagation_failure(self, reason: str) -> None:
        self._last_propagation_failure_reason = str(reason)

    def _min_adjacent_bead_spacing_nm(self, beads: Sequence[BeadObservation]) -> float:
        """Minimum centre-to-centre distance between neighbouring beads along the chain."""
        ordered = sorted(beads, key=lambda b: b.s_nm)
        if len(ordered) < 2:
            return float("inf")
        spacings = [
            math.hypot(b.x_nm - a.x_nm, b.y_nm - a.y_nm)
            for a, b in zip(ordered[:-1], ordered[1:])
        ]
        return float(min(spacings)) if spacings else float("inf")

    def _propagate_to_frame(self, frame_index: int, previous: ChainFrameResult) -> Optional[ChainFrameResult]:
        self._last_propagation_failure_reason = ""
        frame = self._load_frame_at(frame_index)
        if frame is None:
            self._set_propagation_failure("frame data could not be loaded")
            return None
        rel = _relative_height(frame)
        nm_x, nm_y = self._get_nm_per_pixel()
        seed = _resample_path_as_anchors(previous.points_yx)
        if len(seed) < 2:
            self._set_propagation_failure("previous backbone has fewer than two usable seed anchors")
            return None
        snap_radius_px = max(2, int(round(float(self.deviation_spin.value()) / max(min(nm_x, nm_y), 1e-9))))
        anchors = _snap_anchors_to_local_signal(rel, seed, snap_radius_px)
        path = _trace_anchor_points_with_params(
            rel,
            anchors,
            self.frangi_sigma_spin.value(),
            self.ridge_weight_spin.value(),
        )
        if path is None or path.shape[0] < 2:
            self._set_propagation_failure(
                "ridge path search failed from propagated anchors; anchors may have snapped to background or collapsed"
            )
            return None
        prev_sorted = sorted(previous.beads, key=lambda b: b.s_nm)
        template_ids = [bead.bead_id for bead in prev_sorted]
        result = self._make_result_from_path(
            rel,
            path,
            anchors,
            frame_index,
            template_ids=template_ids,
            template_beads=prev_sorted,
        )
        mean_dev, max_dev = _path_deviation_nm(previous.points_yx, result.points_yx, nm_x, nm_y)
        result.mean_deviation_nm = mean_dev
        result.max_deviation_nm = max_dev
        result.max_bead_shift_nm = self._max_bead_shift(previous, result)
        threshold = float(self.deviation_spin.value())
        min_spacing_nm = float(self.min_spacing_spin.value())
        min_adjacent = self._min_adjacent_bead_spacing_nm(result.beads)
        close_threshold_nm = min_spacing_nm * 0.85
        reasons = []
        if len(result.beads) != len(previous.beads):
            reasons.append(f"bead count {len(previous.beads)} -> {len(result.beads)}")
        if np.isfinite(min_adjacent) and min_adjacent < close_threshold_nm:
            reasons.append(f"min bead spacing {min_adjacent:.1f} nm (< {close_threshold_nm:.1f} nm)")
        if np.isfinite(mean_dev) and mean_dev > threshold:
            reasons.append(f"mean line shift {mean_dev:.1f} nm")
        if np.isfinite(max_dev) and max_dev > threshold * 2.0:
            reasons.append(f"max line shift {max_dev:.1f} nm")
        if np.isfinite(result.max_bead_shift_nm) and result.max_bead_shift_nm > threshold:
            reasons.append(f"max bead shift {result.max_bead_shift_nm:.1f} nm")
        if reasons:
            result.diverged = True
            result.status = "; ".join(reasons)
        else:
            result.status = "ok"
            result.diverged = False
        return result

    def _max_bead_shift(self, previous: ChainFrameResult, current: ChainFrameResult) -> float:
        prev_by_id = {bead.bead_id: bead for bead in previous.beads}
        shifts = []
        for bead in current.beads:
            prev = prev_by_id.get(bead.bead_id)
            if prev is None:
                continue
            shifts.append(math.hypot(bead.x_nm - prev.x_nm, bead.y_nm - prev.y_nm))
        return float(np.max(shifts)) if shifts else float("nan")

    def _ensure_seed_result(self, action_label: str) -> Optional[ChainFrameResult]:
        """Return a seed result, storing the current preview line when needed."""
        if self.last_result is not None:
            return self.last_result

        current = self._result_for_current_frame()
        if current is not None:
            self.last_result = current
            return current

        if self.current_path_yx is None and self.anchor_points_xy:
            self._update_current_path()
        if self.current_path_yx is not None and self.current_path_yx.shape[0] >= 2:
            self.store_current_line()
            if self.last_result is not None:
                return self.last_result

        message = f"Store a line in the current frame before {action_label}."
        self.status_label.setText(message)
        QtWidgets.QMessageBox.information(self, "No Seed", message)
        return None

    def _show_frame_without_detection(self, frame_index: int, message: str) -> None:
        """Move to a frame and deliberately clear chain/bead overlays."""
        frame = self._load_frame_at(frame_index)
        self.frame = frame
        self.rel = _relative_height(frame) if frame is not None else None
        self.current_path_yx = None
        self.preview_beads = []
        self.preview_diverged = False
        self.anchor_points_xy = []
        self._sync_main_window_frame(frame_index, emit=False)
        self.status_label.setText(message)
        self._refresh_source_label()
        self._redraw()

    def goto_frame_delta(self, delta: int) -> None:
        """Step to a neighbouring frame and show its stored detection for review."""
        frame_num = int(getattr(gv, "FrameNum", 0) or 0)
        if frame_num <= 0:
            self.status_label.setText("No frames available.")
            return
        current = self._current_frame_index()
        target = max(0, min(frame_num - 1, current + int(delta)))
        if target == current and self.rel is not None and not self.preview_beads and not self.anchor_points_xy:
            self.status_label.setText(
                f"Frame {current + 1}/{frame_num}: already at the {'first' if delta < 0 else 'last'} frame."
            )
            return
        if not self._move_to_frame(target):
            self.status_label.setText(f"Frame {target + 1}: could not load.")
            return
        self._clear_bead_drag()
        self.current_path_yx = None
        self.preview_beads = []
        self.preview_diverged = False
        self.anchor_points_xy = []
        stored = self._result_for_current_frame()
        if stored is not None:
            self.status_label.setText(
                f"Frame {target + 1}/{frame_num}: showing stored detection ({len(stored.beads)} beads)."
            )
        else:
            self.status_label.setText(f"Frame {target + 1}/{frame_num}: no stored detection.")
        self._refresh_source_label()
        self._update_result_list()
        self._redraw()

    def propagate_next_frame(self) -> None:
        try:
            seed = self._ensure_seed_result("Next Auto")
            if seed is None:
                return
            frame_num = int(getattr(gv, "FrameNum", 0) or 0)
            next_frame = max(seed.frame_index + 1, self._current_frame_index() + 1)
            if frame_num <= 0 or next_frame >= frame_num:
                message = "No next frame is available."
                self.status_label.setText(message)
                QtWidgets.QMessageBox.information(self, "End", message)
                return
            self.status_label.setText(f"Frame {next_frame + 1}: propagating...")
            QtWidgets.QApplication.processEvents()
            result = self._propagate_to_frame(next_frame, seed)
            if result is None:
                reason = self._last_propagation_failure_reason or "unknown reason"
                message = f"Frame {next_frame + 1}: no detection shown ({reason})."
                self._show_frame_without_detection(next_frame, message)
                return
            self.frame = self._frame_from_globals(prefer_processed=False)
            self.rel = _relative_height(self.frame) if self.frame is not None else None
            self.current_path_yx = result.points_yx
            self.preview_beads = result.beads
            self.preview_diverged = result.diverged
            self.anchor_points_xy = []
            self._sync_main_window_frame(next_frame, emit=False)
            if result.diverged:
                self.status_label.setText(f"Frame {next_frame + 1}: divergence flagged ({result.status}). Reset manually, then Store.")
                self._refresh_source_label()
                self._redraw()
                return
            self._store_result(result)
            self._recompute_fluctuations()
            self.last_result = result
            self.current_path_yx = None
            self.preview_beads = []
            self.status_label.setText(
                f"Frame {next_frame + 1}: propagated and stored. Mean shift {result.mean_deviation_nm:.1f} nm."
            )
            self._refresh_source_label()
            self._update_result_list()
            self._redraw()
        except Exception as exc:
            message = f"Next Auto failed: {exc}"
            self.status_label.setText(message)
            QtWidgets.QMessageBox.critical(self, "Next Auto Error", message)

    def run_all_frames(self) -> None:
        seed = self._ensure_seed_result("Run all")
        if seed is None:
            return
        start = seed.frame_index + 1
        frame_num = int(getattr(gv, "FrameNum", 0) or 0)
        if frame_num <= 0 or start >= frame_num:
            QtWidgets.QMessageBox.information(self, "No Frames", "No later frames are available.")
            return
        saved_state = self._snapshot_global_frame_state()
        previous = seed
        stored = 0
        skipped = 0
        stop_message = ""
        stopped_frame: Optional[int] = None
        progress = QtWidgets.QProgressDialog("Running bead-chain propagation...", "Cancel", start, frame_num, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(300)
        try:
            for frame_index in range(start, frame_num):
                if progress.wasCanceled():
                    stop_message = "Cancelled by user."
                    break
                progress.setValue(frame_index)
                QtWidgets.QApplication.processEvents()
                result = self._propagate_to_frame(frame_index, previous)
                if result is None:
                    skipped += 1
                    continue
                if result.diverged:
                    stop_message = f"Frame {frame_index + 1}: {result.status}."
                    stopped_frame = frame_index
                    self.current_path_yx = result.points_yx
                    self.preview_beads = result.beads
                    self.preview_diverged = True
                    break
                self._store_result(result)
                previous = result
                self.last_result = result
                stored += 1
            progress.setValue(frame_num)
        finally:
            self._recompute_fluctuations()
            if stopped_frame is not None:
                self._move_to_frame(stopped_frame)
                self._refresh_source_label()
                self._update_result_list()
                self._redraw()
            else:
                self._restore_global_frame_state(saved_state, update_main=True)
                self.refresh_frame(clear_working=True)
        if stop_message:
            self.status_label.setText(
                f"Run all stopped after storing {stored} frame(s), skipped {skipped}. {stop_message}"
            )
        else:
            self.status_label.setText(f"Run all complete. Stored {stored} propagated frame(s), skipped {skipped}.")
        self._update_result_list()

    def _main_checkbox_checked(self, attr_name: str, default: bool = False) -> bool:
        try:
            widget = getattr(self.main_window, attr_name, None)
            if widget is not None and hasattr(widget, "isChecked"):
                return bool(widget.isChecked())
        except Exception:
            pass
        return bool(default)

    def _sync_save_overlay_flags(self, *, show_scale: bool) -> Tuple[bool, bool]:
        """Mirror Save tab caption toggles onto gv.showTimeFlag / gv.showScaleFlag."""
        show_time = self._main_checkbox_checked("time_caption_check", True)
        show_scale = bool(show_scale and self._main_checkbox_checked("scale_caption_check", True))
        gv.showTimeFlag = bool(show_time)
        gv.showScaleFlag = bool(show_scale)
        return bool(show_time), bool(show_scale)

    def _render_movie_frame_bgr(
        self,
        frame_index: int,
        result: Optional[ChainFrameResult],
        *,
        is_first_frame: bool = False,
    ) -> Optional[np.ndarray]:
        if self.main_window is None:
            return None
        files = getattr(gv, "files", None)
        file_idx = int(getattr(gv, "currentFileNum", 0) or 0)
        if not files or not (0 <= file_idx < len(files)):
            return None
        old_index = getattr(gv, "index", frame_index)
        old_time = getattr(gv, "showTimeFlag", False)
        old_scale = getattr(gv, "showScaleFlag", False)
        try:
            gv.index = int(frame_index)
            LoadFrame(files[file_idx])
            InitializeAryDataFallback()
            if hasattr(gv, "aryData") and gv.aryData is not None:
                gv.aryData_processed_1ch = np.asarray(gv.aryData).copy()
            if hasattr(self.main_window, "applyImageProcessing"):
                self.main_window.applyImageProcessing(hidden=True)
            if hasattr(self.main_window, "UpdateDisplayImage"):
                self.main_window.UpdateDisplayImage()
            bgr = getattr(gv, "dspimg", None)
            if bgr is None:
                return None
            bgr = np.ascontiguousarray(bgr.copy())
            raw_h, raw_w = self._raw_pixel_shape()
            bgr = self._draw_bead_overlay_bgr(bgr, result, raw_w, raw_h)

            show_time, show_scale = self._sync_save_overlay_flags(show_scale=is_first_frame)
            if show_time and hasattr(self.main_window, "drawTimeCaption"):
                drawn = self.main_window.drawTimeCaption(bgr)
                if drawn is not None:
                    bgr = drawn
            if show_scale and hasattr(self.main_window, "drawScaleCaption"):
                drawn = self.main_window.drawScaleCaption(bgr)
                if drawn is not None:
                    bgr = drawn
            return np.ascontiguousarray(bgr)
        except Exception:
            return None
        finally:
            gv.index = old_index
            gv.showTimeFlag = old_time
            gv.showScaleFlag = old_scale

    def _raw_pixel_shape(self) -> Tuple[int, int]:
        raw_h = int(getattr(gv, "YPixel", 0) or 0)
        raw_w = int(getattr(gv, "XPixel", 0) or 0)
        if self.rel is not None and (raw_h <= 0 or raw_w <= 0):
            raw_h, raw_w = int(self.rel.shape[0]), int(self.rel.shape[1])
        return max(1, raw_h), max(1, raw_w)

    def _raw_px_to_dsp_xy(
        self,
        x_px: float,
        y_px: float,
        raw_w: int,
        raw_h: int,
        dsp_w: int,
        dsp_h: int,
    ) -> Tuple[int, int]:
        sx = float(dsp_w) / max(raw_w - 1, 1)
        sy = float(dsp_h) / max(raw_h - 1, 1)
        x = int(round(float(x_px) * sx))
        y = int(round((raw_h - 1 - float(y_px)) * sy))
        return max(0, min(dsp_w - 1, x)), max(0, min(dsp_h - 1, y))

    def _invalidate_preview_lut_cache(self) -> None:
        self._preview_lut_rgb = None
        self._preview_lut_frame = -1

    def _update_preview_lut_cache(self, force: bool = False) -> None:
        """Refresh cached preview RGB once per frame; avoid heavy work inside _redraw."""
        frame_idx = self._current_frame_index()
        if not force and self._preview_lut_rgb is not None and self._preview_lut_frame == frame_idx:
            return
        self._invalidate_preview_lut_cache()
        if self.main_window is None or self.rel is None or self._updating_preview_lut:
            return
        self._updating_preview_lut = True
        try:
            if hasattr(self.main_window, "UpdateDisplayImage"):
                self.main_window.UpdateDisplayImage()
            rgb = self._main_lut_rgb_from_cvimg()
            if rgb is not None and rgb.shape[:2] == self.rel.shape[:2]:
                self._preview_lut_rgb = rgb
                self._preview_lut_frame = frame_idx
        except Exception:
            pass
        finally:
            self._updating_preview_lut = False

    def _refresh_main_display_cvimg(self, frame_index: Optional[int] = None) -> bool:
        """Run pyNuD's display pipeline so gv.cvimg uses the main-window LUT/contrast."""
        if self.main_window is None:
            return False
        path = self._current_file_path()
        if not path:
            return False
        frame_index = self._current_frame_index() if frame_index is None else int(frame_index)
        try:
            gv.index = frame_index
            LoadFrame(path)
            InitializeAryDataFallback()
            if hasattr(gv, "aryData") and gv.aryData is not None:
                gv.aryData_processed_1ch = np.asarray(gv.aryData).copy()
            if hasattr(self.main_window, "applyImageProcessing"):
                self.main_window.applyImageProcessing(hidden=True)
            if hasattr(self.main_window, "UpdateDisplayImage"):
                self.main_window.UpdateDisplayImage()
            return getattr(gv, "cvimg", None) is not None
        except Exception:
            return False

    def _main_lut_rgb_from_cvimg(self) -> Optional[np.ndarray]:
        cvimg = getattr(gv, "cvimg", None)
        if cvimg is None:
            return None
        try:
            work = np.asarray(cvimg, dtype=np.uint8)
            gamma_lut = getattr(gv, "gamma_lut_1ch", getattr(gv, "gamma_lut", None))
            if gamma_lut is not None:
                work = cv2.LUT(work, gamma_lut)
            color_map = getattr(gv, "color_lut", None)
            if color_map is None:
                return None
            bgr = cv2.applyColorMap(work, color_map)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _render_pynud_display_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Render one frame through pyNuD's main Image View pipeline (LUT + aspect ratio)."""
        if self.main_window is None:
            return None
        path = self._current_file_path()
        if not path:
            return None
        try:
            gv.index = int(frame_index)
            LoadFrame(path)
            InitializeAryDataFallback()
            if hasattr(gv, "aryData") and gv.aryData is not None:
                gv.aryData_processed_1ch = np.asarray(gv.aryData).copy()
            if hasattr(self.main_window, "applyImageProcessing"):
                self.main_window.applyImageProcessing(hidden=True)
            if hasattr(self.main_window, "UpdateDisplayImage"):
                self.main_window.UpdateDisplayImage()
            img = getattr(gv, "dspimg", None)
            if img is None:
                return None
            return np.ascontiguousarray(img.copy())
        except Exception:
            return None

    def _draw_bead_overlay_bgr(
        self,
        bgr: np.ndarray,
        result: Optional[ChainFrameResult],
        raw_w: int,
        raw_h: int,
    ) -> np.ndarray:
        if result is None or bgr is None or bgr.size == 0:
            return bgr
        out = bgr.copy()
        dsp_h, dsp_w = out.shape[:2]
        line_th = max(1, int(round(min(dsp_w, dsp_h) / 180.0)))
        points = np.asarray(result.points_yx, dtype=float)
        if points.ndim == 2 and points.shape[0] >= 2:
            poly = []
            for row in points:
                poly.append(list(self._raw_px_to_dsp_xy(row[1], row[0], raw_w, raw_h, dsp_w, dsp_h)))
            cv2.polylines(
                out,
                [np.asarray(poly, dtype=np.int32)],
                False,
                (255, 255, 0),
                line_th,
                cv2.LINE_AA,
            )
        radius = max(3, int(round(min(dsp_w, dsp_h) / 80.0)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, min(dsp_w, dsp_h) / 500.0)
        for bead in sorted(result.beads, key=lambda row: row.bead_id):
            x, y = self._raw_px_to_dsp_xy(bead.x_px, bead.y_px, raw_w, raw_h, dsp_w, dsp_h)
            cv2.circle(out, (x, y), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), radius, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                out,
                str(bead.bead_id),
                (x + radius + 2, y + max(4, radius // 2)),
                font,
                font_scale,
                (0, 255, 255),
                max(1, line_th),
                cv2.LINE_AA,
            )
        return out

    def export_movie(self) -> None:
        path = self._current_file_path()
        if not path:
            QtWidgets.QMessageBox.information(self, "No File", "Select an AFM file in the main window first.")
            return
        stored = self._results_for_current_file()
        if not stored:
            QtWidgets.QMessageBox.information(
                self,
                "No Stored Results",
                "Store bead detections before exporting a movie.",
            )
            return
        frame_num = int(getattr(gv, "FrameNum", 0) or 0)
        if frame_num <= 0:
            QtWidgets.QMessageBox.information(self, "No Frames", "No frames are available in the current file.")
            return
        default_dir = os.path.dirname(path) or os.getcwd()
        default_name = os.path.join(default_dir, f"{self._export_base_name()}_beads.mp4")
        save_dialog_options = QtWidgets.QFileDialog.Options()
        out_path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Bead Overlay Movie",
            default_name,
            "MP4 Video (*.mp4)",
            options=save_dialog_options,
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"
        stored_by_frame = {row.frame_index: row for row in stored}
        saved_state = self._snapshot_global_frame_state()
        fps, frame_time_ms = self._realtime_movie_fps()
        progress = QtWidgets.QProgressDialog("Exporting bead overlay movie...", "Cancel", 0, frame_num, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(300)
        writer = None
        try:
            for frame_index in range(frame_num):
                if progress.wasCanceled():
                    break
                progress.setValue(frame_index)
                QtWidgets.QApplication.processEvents()
                frame_bgr = self._render_movie_frame_bgr(
                    frame_index,
                    stored_by_frame.get(frame_index),
                    is_first_frame=(frame_index == 0),
                )
                if frame_bgr is None:
                    continue
                if writer is None:
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Could not open the MP4 file for writing.")
                writer.write(frame_bgr)
            progress.setValue(frame_num)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Movie Export Error", str(exc))
            return
        finally:
            if writer is not None:
                writer.release()
            self._restore_global_frame_state(saved_state, update_main=True)
            self.refresh_frame()
        if progress.wasCanceled():
            self.status_label.setText("Movie export cancelled.")
            return
        QtWidgets.QMessageBox.information(
            self,
            "Movie Export Complete",
            f"Saved bead overlay movie to:\n{out_path}\n\n"
            f"Frame time: {frame_time_ms:g} ms\n"
            f"FPS (real time): {fps:g}\n"
            "Timestamp: every frame (Save tab 'Add Time Caption')\n"
            "Scale bar: first frame only (Save tab 'Add Scale Bar')",
        )
        self.status_label.setText(f"Exported bead overlay movie: {out_path}")

    def _display_range(self) -> Tuple[float, float]:
        if self.rel is None:
            return 0.0, 1.0
        finite = _finite_values(self.rel)
        if finite.size == 0:
            return 0.0, 1.0
        vmin = float(np.percentile(finite, 1.0))
        vmax = float(np.percentile(finite, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    def _redraw(self) -> None:
        self.ax.clear()
        if self.rel is None:
            self.ax.text(0.5, 0.5, "Load an AFM frame", ha="center", va="center")
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return
        if self._preview_lut_rgb is None or self._preview_lut_frame != self._current_frame_index():
            self._update_preview_lut_cache()
        rgb = self._preview_lut_rgb
        if rgb is not None:
            self.ax.imshow(rgb, origin="lower", aspect="equal")
        else:
            vmin, vmax = self._display_range()
            self.ax.imshow(self.rel, cmap="afmhot", origin="lower", vmin=vmin, vmax=vmax)
        stored = self._result_for_current_frame()
        drag_id = self._drag_bead.bead_id if self._drag_bead is not None else None
        if stored is not None:
            self._draw_result(stored, line_color="cyan", point_color="lime", label_prefix="", highlight_id=drag_id)
        if self.current_path_yx is not None and self.current_path_yx.shape[0] >= 2:
            line_color = "magenta" if self.preview_diverged else "yellow"
            self.ax.plot(self.current_path_yx[:, 1], self.current_path_yx[:, 0], color=line_color, linewidth=1.8, alpha=0.95)
            if self.preview_beads:
                self._draw_beads(
                    self.preview_beads,
                    color="magenta" if self.preview_diverged else "white",
                    highlight_id=drag_id,
                )
        if self.anchor_points_xy:
            xs = [point[0] for point in self.anchor_points_xy]
            ys = [point[1] for point in self.anchor_points_xy]
            self.ax.scatter(xs, ys, s=44, marker="x", c="cyan", linewidths=1.4, zorder=7)
        self.ax.set_title(f"{ANALYSIS_NAME}: bead chain trace")
        self.ax.set_xlim(0, self.rel.shape[1])
        self.ax.set_ylim(0, self.rel.shape[0])
        self.ax.set_xlabel("x px")
        self.ax.set_ylabel("y px")
        self.canvas.draw_idle()

    def _draw_result(
        self,
        result: ChainFrameResult,
        line_color: str,
        point_color: str,
        label_prefix: str,
        highlight_id: Optional[int] = None,
    ) -> None:
        points = result.points_yx
        if points.shape[0] >= 2:
            self.ax.plot(points[:, 1], points[:, 0], color=line_color, linewidth=1.8, alpha=0.95)
        self._draw_beads(result.beads, color=point_color, label_prefix=label_prefix, highlight_id=highlight_id)

    def _draw_beads(
        self,
        beads: Sequence[BeadObservation],
        color: str,
        label_prefix: str = "",
        highlight_id: Optional[int] = None,
    ) -> None:
        if not beads:
            return
        xs = [bead.x_px for bead in beads]
        ys = [bead.y_px for bead in beads]
        sizes = [58 if bead.bead_id == highlight_id else 42 for bead in beads]
        edge = ["yellow" if bead.bead_id == highlight_id else "black" for bead in beads]
        self.ax.scatter(xs, ys, s=sizes, c=color, edgecolors=edge, linewidths=1.2, zorder=6)
        for bead in beads:
            self.ax.text(
                bead.x_px,
                bead.y_px,
                f"{label_prefix}{bead.bead_id}",
                color="yellow" if bead.bead_id == highlight_id else color,
                fontsize=8,
                zorder=7,
            )

    def _update_result_list(self) -> None:
        if not hasattr(self, "result_list"):
            return
        self.result_list.clear()
        for result in self._results_for_current_file():
            text = (
                f"F{result.frame_index + 1}: {len(result.beads)} beads, "
                f"{result.length_nm:.1f} nm, {result.status}"
            )
            self.result_list.addItem(text)

    def export_results(self) -> None:
        if not self.results:
            QtWidgets.QMessageBox.information(self, "No Results", "No bead-chain results to export.")
            return
        default_dir = os.path.dirname(self._current_file_path()) or os.getcwd()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Export Bead Chain Analysis", default_dir)
        if not out_dir:
            return
        try:
            self._export_all(out_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(exc))
            return
        base = self._export_base_name()
        stem = self._afm_file_stem()
        QtWidgets.QMessageBox.information(
            self,
            "Export Complete",
            f"Results exported to:\n{out_dir}\n\n"
            f"{base}.zip\n"
            f"  ({stem}_metadata.csv, {stem}_per_bead.csv, ...)\n"
            f"{base}.json (session)",
        )

    def _export_all(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self._recompute_fluctuations()
        base = self._export_base_name()
        self._export_csv_zip(os.path.join(out_dir, f"{base}.zip"))
        self._save_session_to_path(os.path.join(out_dir, f"{base}.json"))

    def _iter_bead_rows(self) -> List[Tuple[ChainFrameResult, BeadObservation]]:
        rows: List[Tuple[ChainFrameResult, BeadObservation]] = []
        for result in sorted(self.results, key=lambda row: (self._path_key(row.source_path), row.frame_index)):
            for bead in sorted(result.beads, key=lambda bead: bead.bead_id):
                rows.append((result, bead))
        return rows

    def _collect_metadata_rows(self) -> List[Dict[str, object]]:
        """Acquisition + analysis parameters needed to reproduce/convert the data."""
        nm_x, nm_y = self._get_nm_per_pixel()
        frame_time_ms = _float_or_nan(getattr(gv, "FrameTime", float("nan")))
        n_files = len({self._path_key(r.source_path) for r in self.results})
        n_frames = len({(self._path_key(r.source_path), r.frame_index) for r in self.results})
        items: List[Tuple[str, object]] = [
            ("plugin", PLUGIN_NAME),
            ("analysis_name", ANALYSIS_NAME),
            ("exported_at", dt.datetime.now().isoformat(sep=" ", timespec="seconds")),
            ("source_path", self._current_file_path()),
            ("export_base_name", self._export_base_name()),
            ("n_files", n_files),
            ("n_frames", n_frames),
            ("nm_per_pixel_x", _format_float(nm_x)),
            ("nm_per_pixel_y", _format_float(nm_y)),
            ("x_scan_size_nm", _format_float(getattr(gv, "XScanSize", float("nan")))),
            ("y_scan_size_nm", _format_float(getattr(gv, "YScanSize", float("nan")))),
            ("x_pixels", int(getattr(gv, "XPixel", 0) or 0)),
            ("y_pixels", int(getattr(gv, "YPixel", 0) or 0)),
            ("frame_time_ms", _format_float(frame_time_ms)),
            ("fluctuation_basis", "per_frame_backbone (intrinsic)"),
            ("backbone_poly_degree", self._backbone_degree()),
            ("z_reference", self._z_reference_mode()),
            ("remove_common_z_drift", bool(self._drift_removal_enabled())),
            ("frangi_sigma", _format_float(self.frangi_sigma_spin.value())),
            ("ridge_weight", _format_float(self.ridge_weight_spin.value())),
            ("strip_half_width_nm", _format_float(self.strip_half_width_spin.value())),
            ("min_bead_spacing_nm", _format_float(self.min_spacing_spin.value())),
            ("min_bead_height", _format_float(self.min_height_spin.value())),
            ("peak_prominence", _format_float(self.prominence_spin.value())),
            ("deviation_threshold_nm", _format_float(self.deviation_spin.value())),
        ]
        return [{"key": key, "value": value} for key, value in items]

    def _collect_per_bead_rows(self) -> List[Dict[str, object]]:
        z_ref = self._z_reference_mode()
        rows: List[Dict[str, object]] = []
        for result, bead in self._iter_bead_rows():
            rows.append({
                "file": result.source_path,
                "frame": result.frame_index + 1,
                "time_s": self._frame_time_s(result.frame_index),
                "bead_id": bead.bead_id,
                "s_nm": bead.s_nm,
                "x_nm": bead.x_nm,
                "y_nm": bead.y_nm,
                "height": bead.height,
                "z_nm": self._effective_z(result, bead),
                "z_reference": z_ref,
                "longitudinal_nm": bead.longitudinal_nm,
                "transverse_nm": bead.transverse_nm,
            })
        return rows

    def _collect_spacing_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for result in sorted(self.results, key=lambda row: (self._path_key(row.source_path), row.frame_index)):
            beads = sorted(result.beads, key=lambda bead: bead.s_nm)
            for a, b in zip(beads[:-1], beads[1:]):
                rows.append({
                    "file": result.source_path,
                    "frame": result.frame_index + 1,
                    "time_s": self._frame_time_s(result.frame_index),
                    "bead_id_a": a.bead_id,
                    "bead_id_b": b.bead_id,
                    "distance_nm": math.hypot(b.x_nm - a.x_nm, b.y_nm - a.y_nm),
                })
        return rows

    def _collect_backbone_rows(self) -> List[Dict[str, object]]:
        """Per-frame smooth backbone plus the mean reference backbone."""
        nm_x, nm_y = self._get_nm_per_pixel()
        rows: List[Dict[str, object]] = []
        for result in sorted(self.results, key=lambda row: (self._path_key(row.source_path), row.frame_index)):
            pts = np.asarray(result.points_yx, dtype=float)
            for idx in range(pts.shape[0]):
                rows.append({
                    "file": result.source_path,
                    "frame": result.frame_index + 1,
                    "time_s": self._frame_time_s(result.frame_index),
                    "point_index": idx,
                    "x_nm": float(pts[idx, 1]) * nm_x,
                    "y_nm": float(pts[idx, 0]) * nm_y,
                })
        by_file: Dict[str, List[ChainFrameResult]] = {}
        for result in self.results:
            by_file.setdefault(self._path_key(result.source_path), []).append(result)
        for results in by_file.values():
            backbone = self._compute_mean_backbone(results)
            if backbone is None:
                continue
            src = results[0].source_path
            bb = np.asarray(backbone, dtype=float)
            for idx in range(bb.shape[0]):
                rows.append({
                    "file": src,
                    "frame": "mean",
                    "time_s": float("nan"),
                    "point_index": idx,
                    "x_nm": float(bb[idx, 1]) * nm_x,
                    "y_nm": float(bb[idx, 0]) * nm_y,
                })
        return rows

    def _tabular_sheet_rows(self) -> List[Tuple[str, List[Dict[str, object]]]]:
        return [
            ("metadata", self._collect_metadata_rows()),
            ("per_bead", self._collect_per_bead_rows()),
            ("summary", self._compute_summary_rows()),
            ("spacing", self._collect_spacing_rows()),
            ("backbone", self._collect_backbone_rows()),
        ]

    def _export_csv_zip(self, path: str) -> None:
        """Write one ZIP containing AFM-stem-named CSV files (metadata, per_bead, ...)."""
        stem = self._afm_file_stem()
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for sheet_name, rows in self._tabular_sheet_rows():
                buffer = io.StringIO()
                if rows:
                    fieldnames = list(rows[0].keys())
                    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                zf.writestr(f"{stem}_{sheet_name}.csv", buffer.getvalue())

    def _compute_summary_rows(self) -> List[Dict[str, float]]:
        by_bead: Dict[int, List[Tuple[ChainFrameResult, BeadObservation]]] = {}
        for result, bead in self._iter_bead_rows():
            by_bead.setdefault(bead.bead_id, []).append((result, bead))
        total_frames = len({(self._path_key(r.source_path), r.frame_index) for r in self.results})
        spacing_by_bead: Dict[int, List[float]] = {}
        for result in self.results:
            beads = sorted(result.beads, key=lambda bead: bead.s_nm)
            for a, b in zip(beads[:-1], beads[1:]):
                spacing_by_bead.setdefault(a.bead_id, []).append(math.hypot(b.x_nm - a.x_nm, b.y_nm - a.y_nm))
        rows: List[Dict[str, float]] = []
        for bead_id, items in sorted(by_bead.items()):
            items.sort(key=lambda item: item[0].frame_index)
            longs = np.asarray([item[1].longitudinal_nm for item in items], dtype=float)
            trans = np.asarray([item[1].transverse_nm for item in items], dtype=float)
            xs = np.asarray([item[1].x_nm for item in items], dtype=float)
            ys = np.asarray([item[1].y_nm for item in items], dtype=float)
            zs = np.asarray([self._effective_z(item[0], item[1]) for item in items], dtype=float)
            raw_heights = np.asarray([item[1].height for item in items], dtype=float)
            dx = xs - np.nanmean(xs) if xs.size else xs
            dy = ys - np.nanmean(ys) if ys.size else ys
            xy_sigma = float(np.sqrt(np.nanmean(dx * dx + dy * dy))) if xs.size else float("nan")
            spacings = np.asarray(spacing_by_bead.get(bead_id, []), dtype=float)
            rows.append({
                "bead_id": int(bead_id),
                "n_frames": int(len(items)),
                "frames_present": int(np.sum(np.isfinite(xs))) if xs.size else 0,
                "total_frames": int(total_frames),
                "sigma_long_nm": float(np.nanstd(longs)) if longs.size else float("nan"),
                "sigma_trans_nm": float(np.nanstd(trans)) if trans.size else float("nan"),
                "sigma_xy_nm": xy_sigma,
                "sigma_z_nm": float(np.nanstd(zs)) if zs.size else float("nan"),
                "mean_height_nm": float(np.nanmean(raw_heights)) if raw_heights.size else float("nan"),
                "msd_coeff_nm2_per_s": self._msd_coefficient(items),
                "mean_spacing_to_next_nm": float(np.nanmean(spacings)) if spacings.size else float("nan"),
                "spacing_sigma_to_next_nm": float(np.nanstd(spacings)) if spacings.size else float("nan"),
            })
        return rows

    def _msd_by_lag(self, items: Sequence[Tuple[ChainFrameResult, BeadObservation]]) -> Tuple[np.ndarray, np.ndarray]:
        if len(items) < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        frames = np.asarray([row.frame_index for row, _bead in items], dtype=int)
        xs = np.asarray([bead.x_nm for _row, bead in items], dtype=float)
        ys = np.asarray([bead.y_nm for _row, bead in items], dtype=float)
        unique_lags = np.arange(1, int(np.max(frames) - np.min(frames)) + 1, dtype=int)
        lag_times = []
        msd = []
        frame_time_ms = _float_or_nan(getattr(gv, "FrameTime", float("nan")))
        dt_s = frame_time_ms / 1000.0 if np.isfinite(frame_time_ms) and frame_time_ms > 0 else 1.0
        for lag in unique_lags:
            vals = []
            for i, frame in enumerate(frames):
                targets = np.where(frames == frame + lag)[0]
                if targets.size == 0:
                    continue
                j = int(targets[0])
                vals.append((xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2)
            if vals:
                lag_times.append(float(lag) * dt_s)
                msd.append(float(np.mean(vals)))
        return np.asarray(lag_times, dtype=float), np.asarray(msd, dtype=float)

    def _msd_coefficient(self, items: Sequence[Tuple[ChainFrameResult, BeadObservation]]) -> float:
        lag_times, msd = self._msd_by_lag(items)
        valid = np.isfinite(lag_times) & np.isfinite(msd) & (lag_times > 0)
        lag_times = lag_times[valid]
        msd = msd[valid]
        if lag_times.size < 2:
            return float("nan")
        n_fit = max(2, int(math.ceil(lag_times.size * 0.4)))
        x = lag_times[:n_fit]
        y = msd[:n_fit]
        try:
            slope, _intercept = np.polyfit(x, y, 1)
            return float(slope)
        except Exception:
            return float("nan")

    def save_session(self) -> None:
        if not self.results:
            QtWidgets.QMessageBox.information(self, "No Results", "No bead-chain results to save.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Session",
            self._session_default_path(),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self._save_session_to_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Session Error", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "Session Saved", f"Session saved to:\n{path}")

    def load_session(self) -> None:
        current_path = self._current_file_path()
        if not current_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Load Session",
                "Select an AFM file in the main window file list before loading a session.",
            )
            return
        expected_name = self._expected_session_filename()
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Session",
            self._session_default_path(),
            f"Session ({expected_name});;JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            count = self._load_session_from_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Session Error", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "Session Loaded", f"Loaded {count} frame result(s).")

    def _session_default_path(self) -> str:
        default_dir = os.path.dirname(self._current_file_path()) or os.getcwd()
        return os.path.join(default_dir, self._expected_session_filename())

    def _expected_session_filename(self) -> str:
        return f"{self._export_base_name()}.json"

    def _validate_session_for_current_file(self, path: str, payload: Dict[str, object]) -> None:
        current_path = self._current_file_path()
        if not current_path:
            raise ValueError("No AFM file is selected in the main window.")
        current_key = self._current_file_key()
        expected_name = self._expected_session_filename()
        selected_name = os.path.basename(path)
        if selected_name != expected_name:
            raise ValueError(
                f"Session file name must be '{expected_name}' for the current AFM file:\n{current_path}"
            )
        session_source = self._path_key(str(payload.get("source_path", "") or ""))
        if session_source != current_key:
            raise ValueError(
                "Session file was saved for a different AFM file:\n"
                f"{payload.get('source_path', '')}\n\n"
                f"Current file:\n{current_path}"
            )
        frames = payload.get("frames", [])
        if not isinstance(frames, list):
            return
        for item in frames:
            if not isinstance(item, dict):
                continue
            frame_source = self._path_key(str(item.get("source_path", "") or session_source))
            if frame_source and frame_source != current_key:
                raise ValueError(
                    "Session contains frame results for a different AFM file:\n"
                    f"{item.get('source_path', '')}\n\n"
                    f"Current file:\n{current_path}"
                )

    def _save_session_to_path(self, path: str) -> None:
        payload = {
            "schema": "pynud_bead_chain_analysis_session_v1",
            "plugin": PLUGIN_NAME,
            "analysis_name": ANALYSIS_NAME,
            "saved_at": dt.datetime.now().isoformat(sep=" ", timespec="seconds"),
            "source_path": self._current_file_path(),
            "current_image_id": self._current_image_id(),
            "ui": {
                "frangi_sigma": _json_float(self.frangi_sigma_spin.value()),
                "ridge_weight": _json_float(self.ridge_weight_spin.value()),
                "strip_half_width_nm": _json_float(self.strip_half_width_spin.value()),
                "min_bead_spacing_nm": _json_float(self.min_spacing_spin.value()),
                "min_bead_height": _json_float(self.min_height_spin.value()),
                "peak_prominence": _json_float(self.prominence_spin.value()),
                "deviation_threshold_nm": _json_float(self.deviation_spin.value()),
            },
            "reference_bead_order": [int(v) for v in self.reference_bead_order],
            "reference_beads": {
                str(bead_id): [_json_float(xy[0]), _json_float(xy[1])]
                for bead_id, xy in self.reference_beads.items()
            },
            "frames": [self._result_to_payload(result) for result in self.results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    def _result_to_payload(self, result: ChainFrameResult) -> Dict[str, object]:
        return {
            "source_path": result.source_path,
            "image_id": result.image_id,
            "frame_index": int(result.frame_index),
            "frame": int(result.frame_index) + 1,
            "anchor_points_xy": [[float(x), float(y)] for x, y in result.anchor_points_xy],
            "points_yx": _array_to_json_list(result.points_yx),
            "length_nm": _json_float(result.length_nm),
            "diverged": bool(result.diverged),
            "status": result.status,
            "mean_deviation_nm": _json_float(result.mean_deviation_nm),
            "max_deviation_nm": _json_float(result.max_deviation_nm),
            "max_bead_shift_nm": _json_float(result.max_bead_shift_nm),
            "beads": [self._bead_to_payload(bead) for bead in result.beads],
        }

    def _bead_to_payload(self, bead: BeadObservation) -> Dict[str, object]:
        return {
            "bead_id": int(bead.bead_id),
            "frame_index": int(bead.frame_index),
            "s_nm": _json_float(bead.s_nm),
            "x_px": _json_float(bead.x_px),
            "y_px": _json_float(bead.y_px),
            "x_nm": _json_float(bead.x_nm),
            "y_nm": _json_float(bead.y_nm),
            "height": _json_float(bead.height),
            "longitudinal_nm": _json_float(bead.longitudinal_nm),
            "transverse_nm": _json_float(bead.transverse_nm),
        }

    def _load_session_from_path(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Invalid session file.")
        self._validate_session_for_current_file(path, payload)
        frames = payload.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError("Session file does not contain frame results.")
        self._restore_session_ui(payload.get("ui", {}))
        loaded: List[ChainFrameResult] = []
        for item in frames:
            if not isinstance(item, dict):
                continue
            result = self._result_from_payload(item)
            if result is not None:
                loaded.append(result)
        if not loaded:
            raise ValueError("No valid frame results were found.")
        self.results = loaded
        self.results.sort(key=lambda row: (self._path_key(row.source_path), row.frame_index))
        self.reference_bead_order = [int(v) for v in payload.get("reference_bead_order", []) or []]
        self.reference_beads = {}
        ref = payload.get("reference_beads", {})
        if isinstance(ref, dict):
            for bead_id_text, xy in ref.items():
                try:
                    self.reference_beads[int(bead_id_text)] = (float(xy[0]), float(xy[1]))
                except Exception:
                    continue
        if not self.reference_beads and self.results:
            self._set_reference_from_result(self.results[0])
        self._recompute_fluctuations()
        self.last_result = max(self.results, key=lambda row: row.frame_index)
        self.anchor_points_xy = []
        self.current_path_yx = None
        self.preview_beads = []
        self.preview_diverged = False
        self.status_label.setText(f"Loaded {len(self.results)} frame result(s) from session.")
        self._refresh_source_label()
        self._update_result_list()
        self._redraw()
        return len(self.results)

    def _result_from_payload(self, payload: Dict[str, object]) -> Optional[ChainFrameResult]:
        points = _array_from_payload(payload.get("points_yx"), 2, dtype=float)
        if points.size == 0 or points.shape[1] != 2:
            return None
        beads = []
        for item in payload.get("beads", []) or []:
            if not isinstance(item, dict):
                continue
            beads.append(
                BeadObservation(
                    bead_id=int(item.get("bead_id", len(beads) + 1)),
                    frame_index=int(item.get("frame_index", payload.get("frame_index", 0))),
                    s_nm=_float_or_nan(item.get("s_nm")),
                    x_px=_float_or_nan(item.get("x_px")),
                    y_px=_float_or_nan(item.get("y_px")),
                    x_nm=_float_or_nan(item.get("x_nm")),
                    y_nm=_float_or_nan(item.get("y_nm")),
                    height=_float_or_nan(item.get("height")),
                    longitudinal_nm=_float_or_nan(item.get("longitudinal_nm")),
                    transverse_nm=_float_or_nan(item.get("transverse_nm")),
                )
            )
        anchors = []
        for point in payload.get("anchor_points_xy", []) or []:
            try:
                anchors.append((float(point[0]), float(point[1])))
            except Exception:
                continue
        frame_index = int(payload.get("frame_index", int(payload.get("frame", 1)) - 1))
        return ChainFrameResult(
            source_path=str(payload.get("source_path", "") or ""),
            image_id=str(payload.get("image_id", "") or "loaded_image"),
            frame_index=max(0, frame_index),
            anchor_points_xy=anchors,
            points_yx=np.asarray(points, dtype=float),
            length_nm=_float_or_nan(payload.get("length_nm")),
            beads=beads,
            diverged=bool(payload.get("diverged", False)),
            status=str(payload.get("status", "ok") or "ok"),
            mean_deviation_nm=_float_or_nan(payload.get("mean_deviation_nm")),
            max_deviation_nm=_float_or_nan(payload.get("max_deviation_nm")),
            max_bead_shift_nm=_float_or_nan(payload.get("max_bead_shift_nm")),
        )

    def _restore_session_ui(self, ui: object) -> None:
        if not isinstance(ui, dict):
            return
        mapping = [
            ("frangi_sigma", self.frangi_sigma_spin),
            ("ridge_weight", self.ridge_weight_spin),
            ("strip_half_width_nm", self.strip_half_width_spin),
            ("min_bead_spacing_nm", self.min_spacing_spin),
            ("min_bead_height", self.min_height_spin),
            ("peak_prominence", self.prominence_spin),
            ("deviation_threshold_nm", self.deviation_spin),
        ]
        for key, spin in mapping:
            value = _float_or_nan(ui.get(key))
            if not np.isfinite(value):
                continue
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(value)
            finally:
                spin.blockSignals(blocked)


__all__ = ["PLUGIN_NAME", "create_plugin", "CordionAnalysisWindow"]
