"""
SpotAnalysis
------------
スタンドアロンの輝点数判定モジュール。各フレームに対し、2 つまたは 3 つの
2D ガウス関数の和をフィットし、AIC/BIC でモデル選択する。バンドパス前処理と
重心追跡を行い、3 つ目のピークの S/N を自動評価する。
"""

from __future__ import annotations

import logging
import os
import csv
import glob
import re
import json
import html
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage import feature, filters, morphology, segmentation, measure
from PyQt5 import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse, Circle

import globalvals as gv
from fileio import LoadFrame, InitializeAryDataFallback

logger = logging.getLogger(__name__)

# プラグイン表示名（Pluginメニューに表示される名前）
PLUGIN_NAME = "Spot Analysis"


@dataclass
class PeakStat:
    amplitude: float
    x: float
    y: float
    sigma: float
    snr: float


@dataclass
class ModelSelectionResult:
    n_peaks: int
    popt: np.ndarray
    rss: float
    loglike: float
    aic: float
    bic: float
    residual_std: float
    init_peaks: List[PeakStat]
    peaks: List[PeakStat]


@dataclass
class FrameAnalysis:
    best_n_peaks: int
    criterion: str
    noise_sigma: float
    snr_threshold: float
    models: Dict[int, ModelSelectionResult]
    roi: np.ndarray  # 可視化用のROI画像
    origin: Tuple[int, int]  # 元画像内でのROI起点 (x0, y0)
    roi_mask: Optional[np.ndarray] = None  # ROI内マスク（楕円など）
    seed_spots: Optional[List[Dict[str, float]]] = None  # 検出段階のシード点（絶対座標）。表示用。


class SpotAnalysis:
    """
    2 つ / 3 つガウスモデルの AIC/BIC 比較と S/N 評価を行うクラス。

    使い方例:
        sa = SpotAnalysis(roi_size=48, snr_threshold=2.0)
        result = sa.analyze_frame(frame)  # frame: 2D numpy array
        print(result.best_n_peaks, result.models[2].aic, result.models[3].aic)
    """

    def __init__(
        self,
        roi_size: int = 48,
        bandpass_low_sigma: float = 0.6,
        bandpass_high_sigma: float = 3.0,
        detection_mode: str = "LoG",
        log_sigma: float = 1.6,
        median_enabled: bool = False,
        median_size: int = 3,
        open_enabled: bool = False,
        open_radius: int = 1,
        init_mode: str = "peak",
        blob_min_sigma: float = 3.0,
        blob_max_sigma: float = 20.0,
        blob_num_sigma: int = 10,
        blob_threshold_rel: float = 0.05,
        blob_overlap: float = 0.5,
        blob_doh_min_sigma: float = 3.0,
        blob_doh_max_sigma: float = 20.0,
        blob_doh_num_sigma: int = 10,
        blob_doh_threshold_rel: float = 0.01,
        watershed_h_rel: float = 0.05,
        watershed_adaptive_h: bool = True,
        peak_min_distance: int = 5,
        localmax_threshold_rel: float = 0.05,
        localmax_threshold_snr: float = 0.0,
        precheck_radius_px: int = 2,
        precheck_kmad: float = 1.0,
        multiscale_enabled: bool = False,
        multiscale_sigmas: Optional[Sequence[float]] = None,
        dbscan_enabled: bool = False,
        dbscan_eps: float = 5.0,
        dbscan_min_samples: int = 1,
        snap_enabled: bool = False,
        snap_radius: int = 2,
        snap_refit_enabled: bool = False,
        refit_max_shift_px: int = 3,
        sigma_bounds: Tuple[float, float] = (0.1, 4.0),
        initial_sigma: float = 0.2,
        snr_threshold: float = 1.0,
        max_iterations: int = 8000,
        margin: int = 5,
        min_amplitude: float = 0.0,
        min_sigma_result: float = 0.0,
    ) -> None:
        self.roi_size = max(8, int(roi_size))
        self.bandpass_low_sigma = float(bandpass_low_sigma)
        self.bandpass_high_sigma = float(bandpass_high_sigma)
        self.detection_mode = str(detection_mode).strip().lower()  # "dog" / "log" / "pre"
        self.log_sigma = float(log_sigma)
        self.median_enabled = bool(median_enabled)
        self.median_size = int(median_size)
        self.open_enabled = bool(open_enabled)
        self.open_radius = int(open_radius)
        self.init_mode = str(init_mode).strip().lower()  # "peak" / "peak_subpixel" / "blob" / "blob_doh" / "watershed" / "multiscale"
        self.blob_min_sigma = float(blob_min_sigma)
        self.blob_max_sigma = float(blob_max_sigma)
        self.blob_num_sigma = max(1, int(blob_num_sigma))
        self.blob_threshold_rel = float(blob_threshold_rel)
        self.blob_overlap = float(blob_overlap)
        self.blob_doh_min_sigma = float(blob_doh_min_sigma)
        self.blob_doh_max_sigma = float(blob_doh_max_sigma)
        self.blob_doh_num_sigma = max(1, int(blob_doh_num_sigma))
        self.blob_doh_threshold_rel = float(blob_doh_threshold_rel)
        self.watershed_h_rel = float(watershed_h_rel)
        self.watershed_adaptive_h = bool(watershed_adaptive_h)
        self.peak_min_distance = max(1, int(peak_min_distance))
        self.localmax_threshold_rel = float(localmax_threshold_rel)
        self.localmax_threshold_snr = float(localmax_threshold_snr)
        self.precheck_radius_px = max(0, int(precheck_radius_px))
        self.precheck_kmad = float(precheck_kmad)
        self.multiscale_enabled = bool(multiscale_enabled)
        self.multiscale_sigmas = list(multiscale_sigmas) if multiscale_sigmas else [1.0, 1.6, 2.5, 4.0]
        self.dbscan_enabled = bool(dbscan_enabled)
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = max(1, int(dbscan_min_samples))
        self.snap_enabled = bool(snap_enabled)
        self.snap_radius = max(0, int(snap_radius))
        self.snap_refit_enabled = bool(snap_refit_enabled)
        self.refit_max_shift_px = max(0, int(refit_max_shift_px))
        self.sigma_bounds = (max(1e-3, sigma_bounds[0]), max(sigma_bounds[0] + 1e-3, sigma_bounds[1]))
        self.initial_sigma = float(initial_sigma)
        self.snr_threshold = float(snr_threshold)
        self.max_iterations = max(1000, int(max_iterations))
        self.margin = int(margin)
        self.min_amplitude = float(min_amplitude)
        self.min_sigma_result = float(min_sigma_result)

    def detection_label(self) -> str:
        mode = (self.detection_mode or "dog").lower()
        if mode == "log":
            return f"LoG (σ={self.log_sigma:g})"
        if mode in ("pre", "preprocessed"):
            return "Pre (median/open)"
        return f"DoG (σ={self.bandpass_low_sigma:g}–{self.bandpass_high_sigma:g})"

    def _apply_roi_preprocess(self, img: np.ndarray) -> np.ndarray:
        out = img
        if self.median_enabled and int(self.median_size) > 1:
            k = int(self.median_size)
            if k % 2 == 0:
                k += 1
            # square footprint is intuitive for UI "kernel size"
            out = filters.median(out, footprint=morphology.square(k))
        if self.open_enabled and int(self.open_radius) > 0:
            r = int(self.open_radius)
            out = morphology.opening(out, morphology.disk(r))
        return out

    def _apply_detection(self, img: np.ndarray) -> np.ndarray:
        mode = (self.detection_mode or "dog").lower()
        if mode == "log":
            sigma = max(float(self.log_sigma), 1e-6)
            # Use negative sign so bright blobs become positive peaks
            return -ndimage.gaussian_laplace(img, sigma=sigma)
        if mode in ("pre", "preprocessed"):
            return img
        # DoG
        # NOTE: DoG creates ring-like patterns around spots, which can lead to
        # false peak detection at valleys instead of true peaks.
        # We mitigate this by validating peaks against the preprocessed image.
        if self.bandpass_high_sigma <= self.bandpass_low_sigma:
            return img
        return filters.difference_of_gaussians(img, self.bandpass_low_sigma, self.bandpass_high_sigma)

    def _roi_mask_with_margin(
        self, roi_mask: Optional[np.ndarray], margin_px: int, shape: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        Build an ROI mask that excludes margin from the ROI boundary.

        - If roi_mask is provided (ellipse etc.), it is eroded by margin_px.
        - Independently, we also exclude a rectangular border of width margin_px for ROI crops.
        """
        try:
            m = int(margin_px)
        except Exception:
            m = 0
        m = max(0, m)

        if roi_mask is not None:
            try:
                mask = roi_mask.astype(bool)
            except Exception:
                mask = None
        else:
            mask = None

        if mask is not None and m > 0:
            try:
                mask = morphology.erosion(mask, morphology.disk(m))
            except Exception:
                pass

        if shape is None:
            if mask is None:
                return None
            h, w = mask.shape
        else:
            h, w = shape
            if mask is None:
                try:
                    mask = np.ones((int(h), int(w)), dtype=bool)
                except Exception:
                    return None
            else:
                try:
                    if mask.shape != (int(h), int(w)):
                        # If shape mismatch, ignore mask (caller will fall back to border-only)
                        mask = np.ones((int(h), int(w)), dtype=bool)
                except Exception:
                    mask = np.ones((int(h), int(w)), dtype=bool)

        if m > 0 and h > 2 * m and w > 2 * m:
            try:
                mask = mask.copy()
                mask[:m, :] = False
                mask[-m:, :] = False
                mask[:, :m] = False
                mask[:, -m:] = False
            except Exception:
                pass
        return mask

    def _snap_to_local_max_pixel(
        self,
        img: np.ndarray,
        y: int,
        x: int,
        radius: int,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, bool]:
        """
        Snap (y,x) to the maximum-intensity pixel within a (2r+1)x(2r+1) window.
        Does NOT reject; only moves the coordinate when a better pixel exists.
        Returns (new_y, new_x, moved).
        """
        frame = np.asarray(img, dtype=np.float64)
        if frame.ndim != 2:
            return int(y), int(x), False
        h, w = frame.shape
        iy = int(np.clip(int(y), 0, h - 1))
        ix = int(np.clip(int(x), 0, w - 1))
        try:
            r = max(0, int(radius))
        except Exception:
            r = 0
        if r <= 0:
            return iy, ix, False

        y0 = max(0, iy - r)
        y1 = min(h, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(w, ix + r + 1)
        if y0 >= y1 or x0 >= x1:
            return iy, ix, False

        best_y, best_x = iy, ix
        try:
            best_v = float(frame[iy, ix])
        except Exception:
            best_v = float("-inf")

        for ny in range(y0, y1):
            for nx in range(x0, x1):
                if roi_mask is not None:
                    try:
                        if not bool(roi_mask[ny, nx]):
                            continue
                    except Exception:
                        pass
                try:
                    v = float(frame[ny, nx])
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                if v > best_v:
                    best_v = v
                    best_y, best_x = ny, nx

        moved = (best_y != iy) or (best_x != ix)
        return int(best_y), int(best_x), bool(moved)

    def _pipeline(self, img: np.ndarray) -> np.ndarray:
        return self._apply_detection(self._apply_roi_preprocess(img))

    def _detect_local_maxima(
        self,
        img: np.ndarray,
        n_peaks: int,
        roi_mask: Optional[np.ndarray] = None,
        threshold_rel: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        allow_more: int = 4,
        return_meta: bool = False,
    ) -> np.ndarray:
        """
        Detect local maxima (pixel-grid) on the given image.
        Returns coords as ndarray of shape (k, 2) in (row=y, col=x).

        Notes:
        - This runs on the detection image (LoG/DoG after preprocess), not on the raw image.
        - threshold_rel is relative to max(img); if None, uses self.localmax_threshold_rel.
        """
        frame = np.asarray(img, dtype=np.float64)
        if frame.ndim != 2:
            return np.zeros((0, 2), dtype=np.int64)
        h, w = frame.shape
        if h == 0 or w == 0:
            return np.zeros((0, 2), dtype=np.int64)

        thr_rel = self.localmax_threshold_rel if threshold_rel is None else float(threshold_rel)
        try:
            thr_rel = float(thr_rel)
        except Exception:
            thr_rel = 0.0
        thr_rel = min(max(thr_rel, 0.0), 1.0)
        v_max = float(np.nanmax(frame)) if np.size(frame) else float("nan")
        threshold_abs = None
        if np.isfinite(v_max) and v_max > 0 and thr_rel > 0:
            threshold_abs = thr_rel * v_max
        try:
            snr_thr = float(getattr(self, "localmax_threshold_snr", 0.0))
        except Exception:
            snr_thr = 0.0
        if noise_sigma is not None and snr_thr > 0:
            try:
                snr_abs = snr_thr * float(noise_sigma)
            except Exception:
                snr_abs = 0.0
            if np.isfinite(snr_abs) and snr_abs > 0:
                threshold_abs = snr_abs if threshold_abs is None else max(threshold_abs, snr_abs)

        num_peaks = int(max(1, n_peaks))
        num_peaks = int(max(num_peaks, num_peaks * int(max(1, allow_more))))

        # Determine appropriate border exclusion based on image size
        # Exclude at least 3 pixels from border to avoid DoG/LoG artifacts
        border_exclude = max(3, int(min(h, w) * 0.05))  # 5% of smaller dimension, minimum 3px
        try:
            m0 = max(0, int(getattr(self, "margin", 0)))
        except Exception:
            m0 = 0

        used_mask = None
        used_m = 0
        coords = np.zeros((0, 2), dtype=np.int64)

        # Stepwise margin relaxation: prefer excluding ROI margin, but guarantee we can return peaks.
        for m in range(int(m0), -1, -1):
            use_border = int(max(border_exclude, m))
            used_m = int(m)
            used_mask = self._roi_mask_with_margin(roi_mask, use_border, shape=(h, w))
            labels = None
            if used_mask is not None:
                try:
                    labels = used_mask.astype(np.int32)
                except Exception:
                    labels = None
            try:
                coords_try = feature.peak_local_max(
                    frame,
                    min_distance=int(self.peak_min_distance),
                    num_peaks=num_peaks,
                    exclude_border=use_border,
                    threshold_abs=threshold_abs,
                    labels=labels,
                )
                if coords_try is None:
                    coords_try = np.zeros((0, 2), dtype=np.int64)
            except Exception:
                coords_try = np.zeros((0, 2), dtype=np.int64)
            coords = np.asarray(coords_try, dtype=np.int64)
            if coords.shape[0] >= int(n_peaks):
                break

        if return_meta:
            return coords, used_mask, used_m
        return coords

    def _top_pixels_by_value_nms(
        self,
        img: np.ndarray,
        n_peaks: int,
        roi_mask: Optional[np.ndarray] = None,
        min_distance: Optional[int] = None,
    ) -> np.ndarray:
        """
        Pick peak seed coordinates by descending pixel value with greedy non-maximum suppression.

        This is used for init_mode=peak / peak_subpixel to make the initializer match the
        visualized detection image (LoG/DoG) more directly than peak_local_max.

        Rules:
        - sort pixels by value (descending) on the detection image
        - keep only pixels inside ROI-mask + ROI-margin (and rectangle border exclusion)
        - enforce a minimum separation (min_distance px) between selected seeds (greedy NMS)

        Returns coords as ndarray of shape (k,2) in (row=y, col=x).
        """
        frame = np.asarray(img, dtype=np.float64)
        if frame.ndim != 2:
            return np.zeros((0, 2), dtype=np.int64)
        h, w = frame.shape
        if h == 0 or w == 0:
            return np.zeros((0, 2), dtype=np.int64)

        try:
            n = int(n_peaks)
        except Exception:
            n = 0
        if n <= 0:
            return np.zeros((0, 2), dtype=np.int64)

        try:
            r = int(self.peak_min_distance if min_distance is None else min_distance)
        except Exception:
            r = int(getattr(self, "peak_min_distance", 2) or 0)
        r = max(0, int(r))

        # Determine border exclusion similarly to _detect_local_maxima (avoid edge artifacts)
        border_exclude = max(3, int(min(h, w) * 0.05))
        try:
            m0 = max(0, int(getattr(self, "margin", 0)))
        except Exception:
            m0 = 0

        accepted: List[Tuple[int, int]] = []

        # Like _detect_local_maxima, relax margin stepwise only if we cannot collect enough seeds.
        for m in range(int(m0), -1, -1):
            use_border = int(max(border_exclude, m))
            allowed = self._roi_mask_with_margin(roi_mask, use_border, shape=(h, w))
            if allowed is None:
                allowed = np.ones((int(h), int(w)), dtype=bool)

            # Mask out disallowed/invalid pixels with -inf so they sort to the end.
            masked = np.where(allowed & np.isfinite(frame), frame, -np.inf)
            flat = masked.ravel()
            if flat.size == 0:
                continue

            order = np.argsort(flat)[::-1]
            for idx in order:
                if len(accepted) >= int(n):
                    break
                v = float(flat[int(idx)])
                if not np.isfinite(v) or v == float("-inf"):
                    # Remaining candidates are all invalid/disallowed.
                    break
                y = int(int(idx) // int(w))
                x = int(int(idx) % int(w))

                if r > 0 and accepted:
                    ok = True
                    for ay, ax in accepted:
                        # Window-based NMS: suppress within (2r+1)x(2r+1) neighborhood
                        # (Chebyshev distance <= r).
                        dy = abs(int(ay) - int(y))
                        dx = abs(int(ax) - int(x))
                        if dx <= r and dy <= r:
                            ok = False
                            break
                    if not ok:
                        continue

                accepted.append((y, x))

            if len(accepted) >= int(n):
                break

        if not accepted:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(accepted[: int(n)], dtype=np.int64)

    def _refine_subpixel_quadratic(
        self,
        img: np.ndarray,
        x0: int,
        y0: int,
    ) -> Tuple[float, float]:
        """
        Subpixel refinement around (x0,y0) using 3x3 quadratic approximation.
        Returns (x,y) floats in image coordinates.
        """
        frame = np.asarray(img, dtype=np.float64)
        h, w = frame.shape
        if x0 <= 0 or y0 <= 0 or x0 >= w - 1 or y0 >= h - 1:
            return float(x0), float(y0)

        p = frame[y0 - 1 : y0 + 2, x0 - 1 : x0 + 2]
        if p.shape != (3, 3) or not np.all(np.isfinite(p)):
            return float(x0), float(y0)

        # Central differences for gradient
        gx = 0.5 * (p[1, 2] - p[1, 0])
        gy = 0.5 * (p[2, 1] - p[0, 1])

        # Hessian (second derivatives)
        gxx = p[1, 2] - 2.0 * p[1, 1] + p[1, 0]
        gyy = p[2, 1] - 2.0 * p[1, 1] + p[0, 1]
        gxy = 0.25 * (p[2, 2] - p[2, 0] - p[0, 2] + p[0, 0])

        det = gxx * gyy - gxy * gxy
        if not np.isfinite(det) or abs(det) < 1e-12:
            return float(x0), float(y0)

        # Newton step: offset = -H^{-1} * grad
        dx = (-gyy * gx + gxy * gy) / det
        dy = (gxy * gx - gxx * gy) / det

        if not np.isfinite(dx) or not np.isfinite(dy):
            return float(x0), float(y0)

        # Keep within a reasonable neighborhood to avoid jumping
        dx = float(np.clip(dx, -0.5, 0.5))
        dy = float(np.clip(dy, -0.5, 0.5))
        return float(x0) + dx, float(y0) + dy

    def _snap_peaks_to_local_maxima(
        self,
        peaks: List[PeakStat],
        det_img: np.ndarray,
        origin: Tuple[int, int],
        roi_mask: Optional[np.ndarray] = None,
    ) -> List[PeakStat]:
        """
        Snap peak coordinates to the nearest strong local maximum in det_img within snap_radius.
        - det_img: ROI-local detection image (after preprocess + LoG/DoG).
        - origin: ROI origin in absolute coords.
        """
        if not peaks or self.snap_radius <= 0:
            return peaks
        img = np.asarray(det_img, dtype=np.float64)
        if img.ndim != 2:
            return peaks
        h, w = img.shape
        x0, y0 = int(origin[0]), int(origin[1])

        allowed = None
        if roi_mask is not None:
            try:
                allowed = roi_mask.astype(bool)
                if allowed.shape != (h, w):
                    allowed = None
            except Exception:
                allowed = None

        r = int(self.snap_radius)
        used: set[Tuple[int, int]] = set()
        snapped: List[PeakStat] = []

        # Prefer snapping higher-SNR peaks first to reduce collisions.
        order = sorted(range(len(peaks)), key=lambda i: float(getattr(peaks[i], "snr", 0.0)), reverse=True)
        for i in order:
            pk = peaks[i]
            try:
                lx = float(pk.x) - float(x0)
                ly = float(pk.y) - float(y0)
            except Exception:
                snapped.append(pk)
                continue
            ix = int(round(lx))
            iy = int(round(ly))
            ix = int(np.clip(ix, 0, w - 1))
            iy = int(np.clip(iy, 0, h - 1))

            x_min = max(0, ix - r)
            x_max = min(w - 1, ix + r)
            y_min = max(0, iy - r)
            y_max = min(h - 1, iy + r)
            window = img[y_min : y_max + 1, x_min : x_max + 1]
            if window.size == 0 or not np.any(np.isfinite(window)):
                snapped.append(pk)
                continue

            # Candidate maxima in this window (sorted by response).
            flat = window.ravel()
            finite_mask = np.isfinite(flat)
            if not np.any(finite_mask):
                snapped.append(pk)
                continue
            idxs = np.argsort(flat[finite_mask])[::-1]
            finite_positions = np.flatnonzero(finite_mask)

            chosen = None
            for j in idxs:
                pos = int(finite_positions[int(j)])
                wy, wx = divmod(pos, window.shape[1])
                cx = x_min + int(wx)
                cy = y_min + int(wy)
                if (cx, cy) in used:
                    continue
                if allowed is not None and not bool(allowed[cy, cx]):
                    continue
                chosen = (cx, cy)
                break

            if chosen is None:
                snapped.append(pk)
                continue
            used.add(chosen)

            cx, cy = chosen
            # Optional subpixel refinement around snapped integer maximum.
            rx, ry = self._refine_subpixel_quadratic(img, cx, cy)
            rx = float(np.clip(rx, 0.0, float(w - 1)))
            ry = float(np.clip(ry, 0.0, float(h - 1)))

            snapped.append(
                PeakStat(
                    amplitude=pk.amplitude,
                    x=float(x0) + rx,
                    y=float(y0) + ry,
                    sigma=pk.sigma,
                    snr=pk.snr,
                )
            )

        # Restore original order
        inv = {idx: out for idx, out in zip(order, snapped)}
        return [inv[i] for i in range(len(peaks))]

    def _enforce_min_peak_distance(
        self, peaks: List[PeakStat], return_rejections: bool = False
    ) -> Any:
        """
        Enforce minimum distance between peaks in absolute coordinates.

        Policy:
        - Use self.peak_min_distance (px).
        - Keep higher-SNR peaks first (tie-breaker: higher amplitude).
        - Greedy selection; drops peaks closer than min_distance to any kept peak.

        NOTE: This is a post-process on the *final* peak list. It can reduce the
        number of peaks below the UI min_peaks when the user prioritizes distance.
        """
        rejected_map: Dict[int, Dict[str, Any]] = {}

        def _add_reject(pk: PeakStat, reason: str) -> None:
            if not return_rejections:
                return
            k = id(pk)
            ent = rejected_map.get(k)
            if ent is None:
                ent = {"peak": pk, "reasons": []}
                rejected_map[k] = ent
            ent["reasons"].append(reason)

        try:
            d_min = float(self.peak_min_distance)
        except Exception:
            return (peaks, []) if return_rejections else peaks
        if not peaks or not np.isfinite(d_min) or d_min <= 0:
            return (peaks, []) if return_rejections else peaks

        d2_min = float(d_min) ** 2
        # Sort by (snr desc, amplitude desc)
        order = sorted(
            peaks,
            key=lambda p: (float(getattr(p, "snr", 0.0)), float(getattr(p, "amplitude", 0.0))),
            reverse=True,
        )

        kept: List[PeakStat] = []
        for pk in order:
            try:
                x = float(pk.x)
                y = float(pk.y)
            except Exception:
                continue
            ok = True
            for kp in kept:
                try:
                    dx = x - float(kp.x)
                    dy = y - float(kp.y)
                except Exception:
                    continue
                if dx * dx + dy * dy < d2_min:
                    dist = float(np.hypot(dx, dy))
                    _add_reject(
                        pk,
                        f"ピーク間隔(min={d_min:.1f}px, dist={dist:.2f}px)",
                    )
                    ok = False
                    break
            if ok:
                kept.append(pk)

        # Preserve stable labeling order by x/y (optional) is not desired; keep original relative order.
        kept_set = set(id(p) for p in kept)
        kept_in_original = [p for p in peaks if id(p) in kept_set]
        if return_rejections:
            return kept_in_original, list(rejected_map.values())
        return kept_in_original

    def _filter_peaks(
        self,
        peaks: List[PeakStat],
        roi_origin: Tuple[int, int],
        roi_shape: Tuple[int, int],
        roi_pre: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        return_rejections: bool = False,
    ) -> Any:
        """
        検出されたピークに対してフィルタリングを行う。
        1. 下限値フィルタ (σ, amplitude)
        2. S/Nフィルタ (snr_threshold; 出力用)
        3. 元画像の局所ピーク判定 (roi_pre; 中央値+MAD / 局所最大)
        4. ROI境界フィルタ (margin, roi_mask)
        """
        rejected_map: Dict[int, Dict[str, Any]] = {}

        def _add_reject(pk: PeakStat, reason: str) -> None:
            if not return_rejections:
                return
            k = id(pk)
            ent = rejected_map.get(k)
            if ent is None:
                ent = {"peak": pk, "reasons": []}
                rejected_map[k] = ent
            ent["reasons"].append(reason)

        filtered: List[PeakStat] = []

        # 1. 下限値フィルタ
        for pk in peaks:
            try:
                sigma_v = float(pk.sigma)
            except Exception:
                sigma_v = float("nan")
            try:
                amp_v = float(pk.amplitude)
            except Exception:
                amp_v = float("nan")

            if np.isfinite(sigma_v) and sigma_v < float(self.min_sigma_result):
                _add_reject(pk, f"最小σ: sigma={sigma_v:.3g} < thr={float(self.min_sigma_result):.3g}")
                continue
            if np.isfinite(amp_v) and amp_v < float(self.min_amplitude):
                _add_reject(pk, f"最小振幅: amp={amp_v:.3g} < thr={float(self.min_amplitude):.3g}")
                continue
            filtered.append(pk)

        # 2. S/Nフィルタ（出力用）。全て閾値未満の場合は落とし切らずに保持する。
        if filtered and self.snr_threshold is not None:
            thr = float(self.snr_threshold)
            if thr > 0:
                snr_pass = [pk for pk in filtered if float(pk.snr) >= thr]
                if snr_pass:
                    for pk in filtered:
                        try:
                            snr_v = float(pk.snr)
                        except Exception:
                            snr_v = float("nan")
                        if np.isfinite(snr_v) and snr_v < thr:
                            _add_reject(pk, f"S/N閾値: snr={snr_v:.3g} < thr={thr:.3g}")
                    filtered = snr_pass
                else:
                    # 閾値超えが無い場合は全ピークを残す（見た目の落ち込みを防ぐ）
                    filtered = list(filtered)

        # 3. 元画像（roi_pre）での局所ピーク判定
        if filtered and roi_pre is not None:
            try:
                r = int(self.precheck_radius_px)
            except Exception:
                r = 0
            try:
                kmad = float(self.precheck_kmad)
            except Exception:
                kmad = 0.0

            if r > 0 or kmad > 0:
                h, w = roi_shape
                x0, y0 = roi_origin
                mask = None
                if roi_mask is not None:
                    try:
                        mask = roi_mask.astype(bool)
                        if mask.shape != (h, w):
                            mask = None
                    except Exception:
                        mask = None
                kept_tmp: List[PeakStat] = []
                for pk in filtered:
                    try:
                        lx = float(pk.x) - float(x0)
                        ly = float(pk.y) - float(y0)
                    except Exception:
                        _add_reject(pk, "precheck: 座標変換失敗")
                        continue
                    ix0 = int(round(lx))
                    iy0 = int(round(ly))
                    ix = ix0
                    iy = iy0
                    if ix < 0 or iy < 0 or ix >= w or iy >= h:
                        _add_reject(pk, "precheck: ROI外")
                        continue

                    # Snap to the maximum pixel within precheck radius (instead of rejecting).
                    moved = False
                    if r > 0:
                        # Try strict mask first; if nothing finite, relax mask.
                        roi_mask_for_snap = mask
                        iy_s, ix_s, moved = self._snap_to_local_max_pixel(
                            roi_pre, iy, ix, radius=int(r), roi_mask=roi_mask_for_snap
                        )
                        ix, iy = int(ix_s), int(iy_s)
                        if moved:
                            try:
                                pk.x = float(x0 + ix)
                                pk.y = float(y0 + iy)
                            except Exception:
                                pass

                    x_min = max(0, ix - r)
                    x_max = min(w - 1, ix + r)
                    y_min = max(0, iy - r)
                    y_max = min(h - 1, iy + r)
                    win = roi_pre[y_min : y_max + 1, x_min : x_max + 1]
                    if win.size == 0:
                        _add_reject(pk, "precheck: window空")
                        continue

                    if mask is not None:
                        try:
                            win_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
                            win = np.where(win_mask, win, np.nan)
                        except Exception:
                            pass

                    # If no finite reference, do not reject.
                    if not np.any(np.isfinite(win)):
                        kept_tmp.append(pk)
                        continue

                    try:
                        v0 = float(roi_pre[iy, ix])
                    except Exception:
                        _add_reject(pk, "precheck: v0取得失敗")
                        continue
                    if not np.isfinite(v0):
                        _add_reject(pk, "precheck: v0が非有限")
                        continue

                    ok = True
                    if kmad > 0:
                        med = float(np.nanmedian(win))
                        mad = float(np.nanmedian(np.abs(win - med)))
                        if not np.isfinite(mad) or mad <= 0:
                            mad = float(np.nanstd(win))
                        if not np.isfinite(mad) or mad <= 0:
                            mad = 0.0
                        thr_kmad = med + kmad * mad
                        if v0 < thr_kmad:
                            ok = False
                            snap_str = ""
                            if moved:
                                snap_str = f"precheck snap: ({ix0},{iy0})->({ix},{iy}); "
                            _add_reject(
                                pk,
                                snap_str
                                + f"precheck K(MAD): v0={v0:.3g} < med+K*MAD={thr_kmad:.3g} (med={med:.3g}, MAD={mad:.3g}, K={kmad:.2g})",
                            )

                    if ok:
                        kept_tmp.append(pk)

                filtered = kept_tmp

        # 4. ROI境界フィルタ
        h, w = roi_shape
        x0, y0 = roi_origin

        # If an ROI mask is provided (ellipse etc.), always enforce peaks to be inside it.
        # When margin > 0, additionally erode the mask to exclude boundary-adjacent peaks.
        if roi_mask is not None:
            try:
                base_allowed = roi_mask.astype(bool)
                allowed = base_allowed
                if self.margin > 0:
                    allowed = morphology.erosion(allowed, morphology.disk(int(self.margin)))
            except Exception:
                allowed = None

            if allowed is not None and allowed.shape == (h, w):
                temp: List[PeakStat] = []
                for pk in filtered:
                    lx = float(pk.x) - float(x0)
                    ly = float(pk.y) - float(y0)
                    ix = int(np.floor(lx))
                    iy = int(np.floor(ly))
                    if 0 <= ix < w and 0 <= iy < h and bool(allowed[iy, ix]):
                        temp.append(pk)
                    else:
                        try:
                            if 0 <= ix < w and 0 <= iy < h and "base_allowed" in locals() and base_allowed.shape == (h, w):
                                if not bool(base_allowed[iy, ix]):
                                    _add_reject(pk, "ROI外(マスク)")
                                else:
                                    _add_reject(pk, f"ROIマージン(margin={int(self.margin)}px)")
                            else:
                                _add_reject(pk, "ROI外")
                        except Exception:
                            _add_reject(pk, "ROI外")
                filtered = temp

        # Rectangle ROI: apply simple margin-to-border exclusion.
        elif self.margin > 0:
            min_x = x0 + self.margin
            max_x = x0 + w - self.margin
            min_y = y0 + self.margin
            max_y = y0 + h - self.margin

            temp2: List[PeakStat] = []
            for pk in filtered:
                if min_x <= pk.x < max_x and min_y <= pk.y < max_y:
                    temp2.append(pk)
                else:
                    try:
                        # distance to nearest border in px (rough)
                        dx = min(float(pk.x) - float(x0), float(x0 + w) - float(pk.x))
                        dy = min(float(pk.y) - float(y0), float(y0 + h) - float(pk.y))
                        d_border = min(dx, dy)
                        _add_reject(pk, f"ROIマージン(margin={int(self.margin)}px, border_dist≈{d_border:.2f}px)")
                    except Exception:
                        _add_reject(pk, f"ROIマージン(margin={int(self.margin)}px)")
            filtered = temp2

        # 4. DBSCAN spatial outlier filtering (optional)
        if self.dbscan_enabled and filtered:
            before_ids = set(id(p) for p in filtered)
            filtered2 = self._filter_peaks_dbscan(filtered)
            after_ids = set(id(p) for p in filtered2)
            removed = before_ids - after_ids
            if removed:
                for pk in filtered:
                    if id(pk) in removed:
                        _add_reject(
                            pk,
                            f"DBSCAN外れ値(eps={float(self.dbscan_eps):.3g}, min_samples={int(self.dbscan_min_samples)})",
                        )
            filtered = filtered2

        if return_rejections:
            return filtered, list(rejected_map.values())
        return filtered

    def analyze_frame(
        self,
        frame: np.ndarray,
        prev_center: Optional[Tuple[float, float]] = None,
        criterion: str = "aic",
        center_override: Optional[Tuple[float, float]] = None,
        roi_size_override: Optional[int] = None,
        roi_mask_override: Optional[np.ndarray] = None,
        roi_bounds_override: Optional[Tuple[int, int, int, int]] = None,
        initial_xy: Optional[Sequence[Tuple[float, float]]] = None,
        min_peaks: int = 2,
        max_peaks: int = 3,
    ) -> FrameAnalysis:
        use_roi_size = roi_size_override if roi_size_override is not None else self.roi_size
        frame_f = np.asarray(frame, dtype=np.float64)
        roi_mask = roi_mask_override

        initial_local: Optional[List[Tuple[float, float]]] = None
        seed_spots: Optional[List[Dict[str, float]]] = None
        roi_pre: np.ndarray
        roi_det: np.ndarray
        if roi_bounds_override is not None:
            roi_raw, origin = self._crop_rect(frame_f, roi_bounds_override)
            roi_pre = self._apply_roi_preprocess(roi_raw)
            roi_det = self._apply_detection(roi_pre)
            if initial_xy:
                initial_local = []
                h_roi, w_roi = roi_pre.shape
                for x_abs, y_abs in initial_xy:
                    try:
                        lx = float(x_abs) - float(origin[0])
                        ly = float(y_abs) - float(origin[1])
                    except Exception:
                        continue
                    if 0 <= lx < w_roi and 0 <= ly < h_roi:
                        initial_local.append((lx, ly))
                # Precheck behavior: snap manual seeds to strongest pixel nearby on roi_pre.
                if initial_local:
                    try:
                        r_pre = int(getattr(self, "precheck_radius_px", 0))
                    except Exception:
                        r_pre = 0
                    if r_pre > 0:
                        snapped = []
                        for lx, ly in initial_local:
                            iy = int(round(float(ly)))
                            ix = int(round(float(lx)))
                            iy2, ix2, _moved = self._snap_to_local_max_pixel(
                                roi_pre, iy, ix, radius=int(r_pre), roi_mask=roi_mask
                            )
                            snapped.append((float(ix2), float(iy2)))
                        initial_local = snapped
        else:
            if center_override is None:
                filtered_for_center = self._pipeline(frame_f)
                center = self._estimate_center(filtered_for_center, prev_center)
            else:
                center = center_override
            roi_raw, origin = self._crop_square(frame_f, center, use_roi_size)
            roi_pre = self._apply_roi_preprocess(roi_raw)
            roi_det = self._apply_detection(roi_pre)
            if initial_xy:
                initial_local = []
                h_roi, w_roi = roi_pre.shape
                for x_abs, y_abs in initial_xy:
                    try:
                        lx = float(x_abs) - float(origin[0])
                        ly = float(y_abs) - float(origin[1])
                    except Exception:
                        continue
                    if 0 <= lx < w_roi and 0 <= ly < h_roi:
                        initial_local.append((lx, ly))
                # Precheck behavior: snap manual seeds to strongest pixel nearby on roi_pre.
                if initial_local:
                    try:
                        r_pre = int(getattr(self, "precheck_radius_px", 0))
                    except Exception:
                        r_pre = 0
                    if r_pre > 0:
                        snapped = []
                        for lx, ly in initial_local:
                            iy = int(round(float(ly)))
                            ix = int(round(float(lx)))
                            iy2, ix2, _moved = self._snap_to_local_max_pixel(
                                roi_pre, iy, ix, radius=int(r_pre), roi_mask=roi_mask
                            )
                            snapped.append((float(ix2), float(iy2)))
                        initial_local = snapped

        # Compute detection seeds (independent of fitting sigma) for display.
        # - If manual initial_xy is provided, treat those as the seeds.
        # - Otherwise, for peak-based init_mode, compute seeds from the detection image.
        mode = (self.init_mode or "peak").strip().lower()
        if initial_local:
            seed_spots = [
                {"x": float(origin[0]) + float(lx), "y": float(origin[1]) + float(ly), "snr": 0.0}
                for (lx, ly) in initial_local
            ]
        elif mode.startswith("peak"):
            try:
                max_peaks_i = int(max_peaks)
            except Exception:
                max_peaks_i = 0
            max_peaks_i = max(1, int(max_peaks_i))
            coords_seed = self._top_pixels_by_value_nms(
                roi_det,
                max_peaks_i,
                roi_mask=roi_mask,
                min_distance=int(getattr(self, "peak_min_distance", 2)),
            )
            try:
                initial_local = [(float(x), float(y)) for (y, x) in coords_seed.tolist()]
            except Exception:
                initial_local = [(float(x), float(y)) for (y, x) in coords_seed]
            seed_spots = [
                {"x": float(origin[0]) + float(lx), "y": float(origin[1]) + float(ly), "snr": 0.0}
                for (lx, ly) in (initial_local or [])
            ]

        # Fit/evaluation is performed on the preprocessed ROI (median/open only),
        # while DoG/LoG is used only for peak initialization/snap.
        noise_sigma = self._estimate_noise_sigma(roi_pre, roi_mask)
        det_noise_sigma = self._estimate_noise_sigma(roi_det, roi_mask)
        models = {}
        min_peaks = max(1, int(min_peaks))
        max_peaks = max(min_peaks, int(max_peaks))
        for n_peaks in range(min_peaks, max_peaks + 1):
            models[n_peaks] = self._fit_model(
                roi_pre,
                origin,
                n_peaks,
                noise_sigma,
                roi_mask,
                initial_xy_local=initial_local if mode.startswith("peak") else initial_local,
                init_image=roi_det,
                init_noise_sigma=det_noise_sigma,
            )

        use_bic = criterion.lower() == "bic"
        best = None
        best_score = None
        for n_peaks, model in models.items():
            score = model.bic if use_bic else model.aic
            if best_score is None or score < best_score:
                best_score = score
                best = n_peaks

        # --- Post-processing filter for the best model ---
        if best is not None:
            best_model = models[best]
            # Keep identity map from pre-filter peaks -> original peak index.
            # This avoids any proximity-based matching in the UI.
            try:
                prefilter_peaks = list(best_model.peaks)
                best_model.prefilter_index_by_id = {id(pk): i for i, pk in enumerate(prefilter_peaks)}
            except Exception:
                pass
            filtered_peaks, rejected_infos = self._filter_peaks(
                best_model.peaks,
                origin,
                roi_pre.shape,
                roi_pre=roi_pre,
                roi_mask=roi_mask,
                return_rejections=True,
            )
            # Optionally snap best peaks to local maxima on the detection image (ROI-local).
            if self.snap_enabled and filtered_peaks:
                filtered_peaks = self._snap_peaks_to_local_maxima(filtered_peaks, roi_det, origin, roi_mask=roi_mask)

            # Optional: re-fit once using snapped peak seeds, then filter/snap again.
            # This is OFF by default because it can change AIC/BIC and runtime.
            if self.snap_refit_enabled and best is not None and filtered_peaks:
                try:
                    seeds_local = [(float(pk.x) - float(origin[0]), float(pk.y) - float(origin[1])) for pk in filtered_peaks]
                except Exception:
                    seeds_local = None
                if seeds_local:
                    refit = self._fit_model(
                        roi_pre,
                        origin,
                        best,
                        noise_sigma,
                        roi_mask,
                        initial_xy_local=seeds_local,
                        init_image=roi_det,
                        init_noise_sigma=det_noise_sigma,
                    )
                    models[best] = refit
                    best_model = refit
                    # Rebuild identity map for refit peaks (pre-filter).
                    try:
                        prefilter_peaks = list(best_model.peaks)
                        best_model.prefilter_index_by_id = {id(pk): i for i, pk in enumerate(prefilter_peaks)}
                    except Exception:
                        pass
                    filtered_peaks, rejected_infos = self._filter_peaks(
                        best_model.peaks,
                        origin,
                        roi_pre.shape,
                        roi_pre=roi_pre,
                        roi_mask=roi_mask,
                        return_rejections=True,
                    )
                    if self.snap_enabled and filtered_peaks:
                        filtered_peaks = self._snap_peaks_to_local_maxima(filtered_peaks, roi_det, origin, roi_mask=roi_mask)

            # Enforce minimum peak distance on final peaks (may reduce count).
            if filtered_peaks:
                filtered_peaks, dist_rejected = self._enforce_min_peak_distance(
                    filtered_peaks, return_rejections=True
                )
                try:
                    rejected_infos.extend(list(dist_rejected))
                except Exception:
                    pass

            # Update the peaks in the best model result.
            # Note: We do not update best_n_peaks selection itself here.
            best_model.peaks = filtered_peaks
            try:
                best_model.excluded_infos = rejected_infos
            except Exception:
                pass

        return FrameAnalysis(
            best_n_peaks=best,
            criterion="bic" if use_bic else "aic",
            noise_sigma=noise_sigma,
            snr_threshold=self.snr_threshold,
            models=models,
            roi=roi_pre,
            origin=origin,
            roi_mask=roi_mask,
            seed_spots=seed_spots,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Backward-compat: apply current pipeline to full frame.
        return self._pipeline(frame)

    def compute_detection_image_full(
        self,
        frame: np.ndarray,
        roi_mask_override: Optional[np.ndarray] = None,
        roi_bounds_override: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Full-size detection image used for peak detection.
        If ROI bounds are given, returns an array with NaN outside ROI (and outside ellipse mask if provided).
        """
        frame_f = np.asarray(frame, dtype=np.float64)
        if roi_bounds_override is None:
            return self._pipeline(frame_f)

        roi_raw, origin = self._crop_rect(frame_f, roi_bounds_override)
        roi_det = self._pipeline(roi_raw)
        full = np.full_like(frame_f, np.nan, dtype=np.float64)
        x0, y0 = int(origin[0]), int(origin[1])
        h, w = roi_det.shape
        full[y0 : y0 + h, x0 : x0 + w] = roi_det
        if roi_mask_override is not None:
            mask = roi_mask_override.astype(bool)
            crop = full[y0 : y0 + h, x0 : x0 + w]
            try:
                crop[~mask] = np.nan
            except Exception:
                pass
        return full

    def compute_roi_visual_images(
        self,
        frame: np.ndarray,
        roi_mask_override: Optional[np.ndarray] = None,
        roi_bounds_override: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        ROI表示用の2枚を返す:
        - 左: ROIの前処理後画像（median/openのみ。検出フィルタは未適用）
        - 右: ROIの検出用画像（前処理→LoG/DoG）
        いずれも ROI が無い場合は (None, None)。
        """
        if roi_bounds_override is None:
            return None, None
        frame_f = np.asarray(frame, dtype=np.float64)
        roi_raw, _origin = self._crop_rect(frame_f, roi_bounds_override)
        roi_pre = self._apply_roi_preprocess(roi_raw)
        roi_det = self._apply_detection(roi_pre)
        roi_pre_v = np.asarray(roi_pre, dtype=np.float64).copy()
        roi_det_v = np.asarray(roi_det, dtype=np.float64).copy()
        if roi_mask_override is not None:
            try:
                mask = roi_mask_override.astype(bool)
                roi_pre_v[~mask] = np.nan
                roi_det_v[~mask] = np.nan
            except Exception:
                pass
        return roi_pre_v, roi_det_v

    def _estimate_center(
        self,
        frame: np.ndarray,
        prev_center: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        if prev_center is None:
            cy, cx = ndimage.center_of_mass(np.clip(frame - np.min(frame), 0.0, None))
            if np.isnan(cx) or np.isnan(cy):
                h, w = frame.shape
                return w / 2.0, h / 2.0
            return float(cx), float(cy)

        small_roi, origin = self._crop_square(frame, prev_center, max(8, self.roi_size // 2))
        cy, cx = ndimage.center_of_mass(np.clip(small_roi - np.min(small_roi), 0.0, None))
        if np.isnan(cx) or np.isnan(cy):
            return prev_center
        return float(origin[0] + cx), float(origin[1] + cy)

    def _crop_square(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        size: int,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = frame.shape
        half = size // 2
        cx = int(round(center[0]))
        cy = int(round(center[1]))
        x0 = max(cx - half, 0)
        x1 = min(cx + half + 1, w)
        y0 = max(cy - half, 0)
        y1 = min(cy + half + 1, h)
        return frame[y0:y1, x0:x1], (x0, y0)

    def _crop_rect(
        self,
        frame: np.ndarray,
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        x0, y0, w, h = bounds
        h_img, w_img = frame.shape
        x0_i = max(int(round(x0)), 0)
        y0_i = max(int(round(y0)), 0)
        x1_i = min(int(round(x0 + w)), w_img)
        y1_i = min(int(round(y0 + h)), h_img)
        return frame[y0_i:y1_i, x0_i:x1_i], (x0_i, y0_i)

    def _estimate_noise_sigma(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        data = frame[mask] if mask is not None else frame.ravel()
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        sigma = 1.4826 * mad  # robust estimator
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.std(data))
        return max(sigma, 1e-6)

    def _validate_peak_coords(
        self,
        coords: np.ndarray,
        intensity_image: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        snap_to_local_max: bool = True,
        search_radius: int = 5,
    ) -> np.ndarray:
        """
        Snap/validate detected peak coordinates against an intensity image.

        This is crucial for DoG-based detection, which creates ring patterns and may
        detect valleys instead of true peaks.
        """
        if coords.shape[0] == 0:
            return coords

        h, w = intensity_image.shape
        out = []

        for y, x in coords:
            iy = int(y)
            ix = int(x)

            if not (0 <= iy < h and 0 <= ix < w):
                iy = int(np.clip(iy, 0, h - 1))
                ix = int(np.clip(ix, 0, w - 1))

            if snap_to_local_max:
                try:
                    r = int(search_radius)
                except Exception:
                    r = 0
                iy, ix, _moved = self._snap_to_local_max_pixel(
                    intensity_image, iy, ix, radius=r, roi_mask=roi_mask
                )

            out.append([int(iy), int(ix)])

        return np.asarray(out, dtype=np.int64)

    def _initial_params(
        self,
        frame: np.ndarray,
        n_peaks: int,
        roi_mask: Optional[np.ndarray] = None,
        init_noise_sigma: Optional[float] = None,
        validate_image: Optional[np.ndarray] = None,
    ) -> List[float]:
        try:
            m0 = max(0, int(getattr(self, "margin", 0)))
        except Exception:
            m0 = 0
        try:
            h0, w0 = np.asarray(frame).shape[:2]
            base_border = max(3, int(min(int(h0), int(w0)) * 0.05))
        except Exception:
            base_border = 3

        def _iter_margin_masks(shape_hw: Tuple[int, int]):
            h, w = int(shape_hw[0]), int(shape_hw[1])
            base = max(3, int(min(h, w) * 0.05))
            for m in range(int(m0), -1, -1):
                use_border = int(max(base, m))
                yield use_border, self._roi_mask_with_margin(roi_mask, use_border, shape=(h, w))

        mode = (self.init_mode or "peak").strip().lower()
        if mode == "watershed":
            # When a validation image (typically preprocessed intensity) is provided,
            # prefer running watershed on that image rather than on DoG/LoG to avoid ring artifacts.
            ws_img = validate_image if validate_image is not None else frame
            params_ws = None
            used_mask = roi_mask
            try:
                ws_hw = np.asarray(ws_img).shape[:2]
            except Exception:
                ws_hw = (int(h0), int(w0))
            for use_border, mask_m in _iter_margin_masks((int(ws_hw[0]), int(ws_hw[1]))):
                params_ws = self._initial_params_watershed(
                    ws_img, n_peaks, roi_mask=mask_m, border_exclude=use_border
                )
                if params_ws is not None:
                    used_mask = mask_m
                    break
            if params_ws is not None:
                # If a validation image is provided (typically preprocessed intensity),
                # snap watershed seeds to true intensity maxima to avoid DoG/LoG ring artifacts.
                if validate_image is not None and len(params_ws) >= n_peaks * 4 + 1:
                    try:
                        entries = []
                        for i in range(int(n_peaks)):
                            amp, x, y, sigma0 = params_ws[i * 4 : (i + 1) * 4]
                            entries.append((float(amp), float(x), float(y), float(sigma0)))

                        # Build coords as (row=y, col=x)
                        coords = np.array([[int(round(e[2])), int(round(e[1]))] for e in entries], dtype=np.int64)
                        try:
                            r_pre = int(getattr(self, "precheck_radius_px", 0))
                        except Exception:
                            r_pre = 0
                        coords_v = self._validate_peak_coords(
                            coords,
                            np.asarray(validate_image, dtype=np.float64),
                            roi_mask=used_mask,
                            snap_to_local_max=True,
                            search_radius=max(1, int(r_pre)) if int(r_pre) > 0 else 5,
                        )

                        # Ensure we keep exactly n_peaks coords (pad if validation reduced them)
                        coords_final = []
                        seen = set()
                        for yy, xx in coords_v:
                            t = (int(yy), int(xx))
                            if t not in seen:
                                coords_final.append([int(yy), int(xx)])
                                seen.add(t)
                            if len(coords_final) >= int(n_peaks):
                                break

                        if len(coords_final) < int(n_peaks):
                            # Pad with remaining original coords ordered by validate_image intensity
                            imgv = np.asarray(validate_image, dtype=np.float64)
                            cand = []
                            for yy, xx in coords:
                                t = (int(yy), int(xx))
                                if t in seen:
                                    continue
                                if 0 <= int(yy) < imgv.shape[0] and 0 <= int(xx) < imgv.shape[1]:
                                    cand.append((float(imgv[int(yy), int(xx)]), int(yy), int(xx)))
                            cand.sort(key=lambda t: t[0], reverse=True)
                            for _v, yy, xx in cand:
                                coords_final.append([yy, xx])
                                seen.add((yy, xx))
                                if len(coords_final) >= int(n_peaks):
                                    break

                        # Rebuild params with snapped coords; keep sigma0, refresh amp from init frame.
                        img_init = np.asarray(frame, dtype=np.float64)
                        params: List[float] = []
                        for i in range(int(n_peaks)):
                            sigma0 = float(entries[i][3]) if i < len(entries) else float(self.initial_sigma)
                            yy, xx = coords_final[i] if i < len(coords_final) else coords[i].tolist()
                            yy_i = int(np.clip(int(yy), 0, img_init.shape[0] - 1))
                            xx_i = int(np.clip(int(xx), 0, img_init.shape[1] - 1))
                            amp0 = max(float(img_init[yy_i, xx_i]), 1e-6)
                            params.extend([amp0, float(xx_i), float(yy_i), sigma0])
                        params.append(float(np.nanmedian(img_init)))
                        return params
                    except Exception:
                        return params_ws
                return params_ws
            # fallback order
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_blob = self._initial_params_blob(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_blob is not None:
                    return params_blob
        elif mode == "blob_doh":
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_doh = self._initial_params_blob_doh(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_doh is not None:
                    # If validate_image (typically roi_pre) is provided, snap DoH seeds to
                    # true intensity maxima to avoid LoG/DoG ring artifacts.
                    if validate_image is not None and len(params_doh) >= n_peaks * 4 + 1:
                        try:
                            entries = []
                            for i in range(int(n_peaks)):
                                amp, x, y, sigma0 = params_doh[i * 4 : (i + 1) * 4]
                                entries.append((float(amp), float(x), float(y), float(sigma0)))

                            coords = np.array([[int(round(e[2])), int(round(e[1]))] for e in entries], dtype=np.int64)
                            try:
                                r_pre = int(getattr(self, "precheck_radius_px", 0))
                            except Exception:
                                r_pre = 0
                            coords_v = self._validate_peak_coords(
                                coords,
                                np.asarray(validate_image, dtype=np.float64),
                                roi_mask=mask_m,
                                snap_to_local_max=True,
                                search_radius=max(1, int(r_pre)) if int(r_pre) > 0 else 5,
                            )

                            # Unique + cap to n_peaks
                            coords_final = []
                            seen = set()
                            for yy, xx in coords_v:
                                t = (int(yy), int(xx))
                                if t not in seen:
                                    coords_final.append([int(yy), int(xx)])
                                    seen.add(t)
                                if len(coords_final) >= int(n_peaks):
                                    break

                            # Optional NMS by peak_min_distance on snapped coords (Chebyshev window).
                            try:
                                r_nms = int(getattr(self, "peak_min_distance", 0) or 0)
                            except Exception:
                                r_nms = 0
                            r_nms = max(0, int(r_nms))
                            if r_nms > 0 and coords_final:
                                imgv = np.asarray(validate_image, dtype=np.float64)
                                scored = []
                                for yy, xx in coords_final:
                                    if 0 <= int(yy) < imgv.shape[0] and 0 <= int(xx) < imgv.shape[1]:
                                        scored.append((float(imgv[int(yy), int(xx)]), int(yy), int(xx)))
                                scored.sort(key=lambda t: t[0], reverse=True)
                                kept = []
                                for _v, yy, xx in scored:
                                    ok = True
                                    for ky, kx in kept:
                                        if abs(int(kx) - int(xx)) <= r_nms and abs(int(ky) - int(yy)) <= r_nms:
                                            ok = False
                                            break
                                    if ok:
                                        kept.append((int(yy), int(xx)))
                                    if len(kept) >= int(n_peaks):
                                        break
                                coords_final = [[yy, xx] for (yy, xx) in kept]

                            if len(coords_final) < int(n_peaks):
                                # Pad with original coords ordered by validate_image intensity
                                imgv = np.asarray(validate_image, dtype=np.float64)
                                cand = []
                                for yy, xx in coords:
                                    t = (int(yy), int(xx))
                                    if t in seen:
                                        continue
                                    if 0 <= int(yy) < imgv.shape[0] and 0 <= int(xx) < imgv.shape[1]:
                                        cand.append((float(imgv[int(yy), int(xx)]), int(yy), int(xx)))
                                cand.sort(key=lambda t: t[0], reverse=True)
                                for _v, yy, xx in cand:
                                    coords_final.append([yy, xx])
                                    seen.add((yy, xx))
                                    if len(coords_final) >= int(n_peaks):
                                        break

                            # Rebuild params with snapped coords; keep sigma0, refresh amp from validate_image.
                            imgv = np.asarray(validate_image, dtype=np.float64)
                            params: List[float] = []
                            for i in range(int(n_peaks)):
                                sigma0 = float(entries[i][3]) if i < len(entries) else float(self.initial_sigma)
                                yy, xx = coords_final[i] if i < len(coords_final) else coords[i].tolist()
                                yy_i = int(np.clip(int(yy), 0, imgv.shape[0] - 1))
                                xx_i = int(np.clip(int(xx), 0, imgv.shape[1] - 1))
                                amp0 = max(float(imgv[yy_i, xx_i]), 1e-6)
                                params.extend([amp0, float(xx_i), float(yy_i), sigma0])
                            params.append(float(np.nanmedian(imgv)))
                            return params
                        except Exception:
                            return params_doh
                    return params_doh
            # fallback to blob_log
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_blob = self._initial_params_blob(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_blob is not None:
                    return params_blob
        elif mode == "multiscale":
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_multi = self._initial_params_multiscale(
                    frame, n_peaks, roi_mask=mask_m, border_exclude=use_border
                )
                if params_multi is not None:
                    return params_multi
            # fallback to blob_log
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_blob = self._initial_params_blob(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_blob is not None:
                    return params_blob
        elif mode == "blob":
            for use_border, mask_m in _iter_margin_masks((int(h0), int(w0))):
                params_blob = self._initial_params_blob(frame, n_peaks, roi_mask=mask_m, border_exclude=use_border)
                if params_blob is not None:
                    return params_blob

        # Peak-based initialization: choose by descending pixel value with NMS (min_distance).
        # This makes the initializer deterministic and aligned with the displayed LoG/DoG image.
        if mode.startswith("peak"):
            use_subpixel = "subpixel" in mode
            coords = self._top_pixels_by_value_nms(
                frame,
                n_peaks,
                roi_mask=roi_mask,
                min_distance=int(getattr(self, "peak_min_distance", 2)),
            )

            params: List[float] = []
            for y, x in coords[:n_peaks]:
                iy = int(y)
                ix = int(x)
                amp = max(float(frame[iy, ix]), 1e-6)
                if use_subpixel:
                    rx, ry = self._refine_subpixel_quadratic(frame, ix, iy)
                    params.extend([amp, float(rx), float(ry), float(self.initial_sigma)])
                else:
                    params.extend([amp, float(ix), float(iy), float(self.initial_sigma)])
            while len(params) < n_peaks * 4:
                params.extend([1.0, frame.shape[1] / 2.0, frame.shape[0] / 2.0, float(self.initial_sigma)])
            params.append(float(np.nanmedian(frame)))
            return params

        use_subpixel = "subpixel" in mode

        coords, used_mask, _used_m = self._detect_local_maxima(
            frame,
            n_peaks,
            roi_mask=roi_mask,
            threshold_rel=self.localmax_threshold_rel,
            noise_sigma=init_noise_sigma,
            return_meta=True,
        )

        # Validate/snap peaks against the intensity image if provided
        # This is critical for DoG-based detection which creates ring patterns
        if validate_image is not None and coords.shape[0] > 0:
            try:
                r_pre = int(getattr(self, "precheck_radius_px", 0))
            except Exception:
                r_pre = 0
            coords = self._validate_peak_coords(
                coords,
                validate_image,
                used_mask,
                snap_to_local_max=True,
                search_radius=max(1, int(r_pre)) if int(r_pre) > 0 else 5,
            )

        if coords.shape[0] < n_peaks:
            # Fallback: pick strongest pixels globally, but prefer excluding ROI margin.
            img = np.asarray(frame, dtype=np.float64)
            coords_best = None
            for use_border, mask_m in _iter_margin_masks((int(img.shape[0]), int(img.shape[1]))):
                try:
                    if mask_m is not None and mask_m.shape == img.shape:
                        masked = np.where(mask_m.astype(bool), img, -np.inf)
                    else:
                        masked = img
                        if use_border > 0 and img.shape[0] > 2 * use_border and img.shape[1] > 2 * use_border:
                            masked = masked.copy()
                            masked[:use_border, :] = -np.inf
                            masked[-use_border:, :] = -np.inf
                            masked[:, :use_border] = -np.inf
                            masked[:, -use_border:] = -np.inf
                    flat_idx = np.argsort(masked.ravel())[::-1][: int(n_peaks)]
                    ys, xs = np.unravel_index(flat_idx, img.shape)
                    coords_try = np.stack([ys, xs], axis=1)
                except Exception:
                    continue
                # accept if we have enough finite pixels
                try:
                    ok = 0
                    for yy, xx in coords_try:
                        v = float(img[int(yy), int(xx)])
                        if np.isfinite(v):
                            ok += 1
                    if ok >= int(n_peaks):
                        coords_best = coords_try
                        break
                except Exception:
                    coords_best = coords_try
                    break
            if coords_best is not None:
                coords = np.asarray(coords_best, dtype=np.int64)

        params: List[float] = []
        for y, x in coords[:n_peaks]:
            iy = int(y)
            ix = int(x)
            amp = max(float(frame[iy, ix]), 1e-6)
            if use_subpixel:
                rx, ry = self._refine_subpixel_quadratic(frame, ix, iy)
                params.extend([amp, float(rx), float(ry), float(self.initial_sigma)])
            else:
                params.extend([amp, float(ix), float(iy), float(self.initial_sigma)])
        while len(params) < n_peaks * 4:
            params.extend([1.0, frame.shape[1] / 2.0, frame.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.median(frame)))
        return params

    def _initial_params_watershed(
        self,
        frame: np.ndarray,
        n_peaks: int,
        roi_mask: Optional[np.ndarray] = None,
        border_exclude: int = 0,
    ) -> Optional[List[float]]:
        """
        Watershed-based initializer:
        - h-maxima suppresses shallow maxima (noise)
        - watershed segments basins around markers
        - regionprops yields centroid/size to seed (x,y,sigma)
        Returns None on failure so caller can fall back.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None
        h, w = img.shape
        v_max = float(np.nanmax(img))
        v_min = float(np.nanmin(img))
        if not np.isfinite(v_max) or not np.isfinite(v_min) or v_max <= v_min:
            return None

        h_rel = float(self.watershed_h_rel)
        h_rel = min(max(h_rel, 0.0), 1.0)

        # Adaptive h-maxima: use noise-based threshold if enabled
        if self.watershed_adaptive_h:
            try:
                noise_sigma = self._estimate_noise_sigma(img, roi_mask)
                h_val_adaptive = max(3.0 * noise_sigma, h_rel * (v_max - v_min))
                h_val = h_val_adaptive
            except Exception:
                h_val = h_rel * (v_max - v_min)
        else:
            h_val = h_rel * (v_max - v_min)

        if h_val <= 0:
            return None

        mask = roi_mask.astype(bool) if roi_mask is not None else None

        try:
            maxima = morphology.h_maxima(img, h=h_val)
        except Exception:
            return None
        if maxima is None:
            return None
        maxima = maxima.astype(bool)
        # Exclude border maxima to reduce LoG/DoG edge artifacts.
        try:
            border = max(1, int(self.peak_min_distance), int(border_exclude))
        except Exception:
            border = max(1, int(border_exclude))
        if border > 0 and h > 2 * border and w > 2 * border:
            maxima[:border, :] = False
            maxima[-border:, :] = False
            maxima[:, :border] = False
            maxima[:, -border:] = False
        if mask is not None:
            try:
                maxima &= mask
            except Exception:
                pass
        if not np.any(maxima):
            return None

        # pick strongest maxima as markers (to avoid over-segmentation)
        ys, xs = np.nonzero(maxima)
        if ys.size == 0:
            return None
        resp = img[ys, xs]
        order = np.argsort(resp)[::-1]
        # allow more markers than peaks, but cap
        max_markers = int(max(20, n_peaks * 5))
        order = order[:max_markers]
        markers = np.zeros_like(img, dtype=np.int32)
        m_id = 1
        for idx in order:
            y = int(ys[idx])
            x = int(xs[idx])
            # enforce min_distance roughly by skipping close markers
            # (cheap check against existing markers)
            if self.peak_min_distance > 1:
                r = int(self.peak_min_distance)
                y0 = max(0, y - r)
                y1 = min(h, y + r + 1)
                x0 = max(0, x - r)
                x1 = min(w, x + r + 1)
                if np.any(markers[y0:y1, x0:x1] > 0):
                    continue
            markers[y, x] = m_id
            m_id += 1
        if m_id <= 1:
            return None

        try:
            labels = segmentation.watershed(-img, markers, mask=mask)
        except Exception:
            return None
        if labels is None:
            return None

        # build candidate peaks from regions
        try:
            props = measure.regionprops(labels, intensity_image=img)
        except Exception:
            return None
        if not props:
            return None

        sigma_min, sigma_max = self.sigma_bounds
        candidates = []
        for rp in props:
            if rp.label == 0:
                continue
            # Representative point:
            # Use the brightest pixel inside the watershed region (on roi_pre/intensity image),
            # rather than centroid. Centroid can land on a non-peak area for skewed regions.
            try:
                img_box = np.asarray(getattr(rp, "intensity_image", None), dtype=np.float64)
            except Exception:
                img_box = None
            try:
                reg_mask = np.asarray(getattr(rp, "image", None), dtype=bool)
            except Exception:
                reg_mask = None
            cx = cy = float("nan")
            amp = float(getattr(rp, "max_intensity", 0.0))
            if img_box is not None and reg_mask is not None and img_box.shape == reg_mask.shape and img_box.size > 0:
                try:
                    masked = np.where(reg_mask, img_box, -np.inf)
                    flat_idx = int(np.nanargmax(masked))
                    iy_loc, ix_loc = divmod(flat_idx, int(masked.shape[1]))
                    # bbox: (min_row, min_col, max_row, max_col)
                    b = getattr(rp, "bbox", None)
                    if b is not None and len(b) >= 2:
                        cy = float(int(b[0]) + int(iy_loc))
                        cx = float(int(b[1]) + int(ix_loc))
                    else:
                        cy = float(iy_loc)
                        cx = float(ix_loc)
                    try:
                        amp = float(masked[int(iy_loc), int(ix_loc)])
                    except Exception:
                        amp = float(getattr(rp, "max_intensity", 0.0))
                except Exception:
                    # Fallback: keep centroid if something goes wrong
                    try:
                        cy, cx = rp.centroid  # (row, col)
                    except Exception:
                        pass
            else:
                # Fallback: centroid
                try:
                    cy, cx = rp.centroid  # (row, col)
                except Exception:
                    pass
            eq_d = float(getattr(rp, "equivalent_diameter", 0.0))
            if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(amp):
                continue
            if eq_d <= 0 or not np.isfinite(eq_d):
                sigma0 = float(self.initial_sigma)
            else:
                sigma0 = eq_d / 4.0
            sigma0 = min(max(float(sigma0), float(sigma_min)), float(sigma_max))
            candidates.append((amp, float(cx), float(cy), sigma0))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=True)

        # Enforce peak spacing on the chosen representative points as well.
        # Even after watershed, nearby regions can produce very close maxima; apply a final greedy NMS
        # using the same window notion as Peak initializer (Chebyshev distance <= r).
        try:
            r_nms = int(getattr(self, "peak_min_distance", 0) or 0)
        except Exception:
            r_nms = 0
        r_nms = max(0, int(r_nms))
        chosen = []
        if r_nms > 0:
            for amp, x, y, sigma0 in candidates:
                ok = True
                for _amp2, x2, y2, _s2 in chosen:
                    if abs(float(x2) - float(x)) <= float(r_nms) and abs(float(y2) - float(y)) <= float(r_nms):
                        ok = False
                        break
                if ok:
                    chosen.append((amp, x, y, sigma0))
                if len(chosen) >= int(n_peaks):
                    break
        else:
            chosen = list(candidates[: int(n_peaks)])

        params: List[float] = []
        for amp, x, y, sigma0 in chosen:
            params.extend([max(float(amp), 1e-6), x, y, sigma0])
        while len(params) < n_peaks * 4:
            params.extend([1.0, img.shape[1] / 2.0, img.shape[0] / 2.0, float(self.initial_sigma)])
        params.append(float(np.nanmedian(img)))
        return params

    def _initial_params_blob(
        self, frame: np.ndarray, n_peaks: int, roi_mask: Optional[np.ndarray] = None, border_exclude: int = 0
    ) -> Optional[List[float]]:
        """
        Multiscale blob detection based initializer using `skimage.feature.blob_log`.
        Returns None if blob candidates are not usable and the caller should fall back.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None

        v_max = float(np.nanmax(img))
        if not np.isfinite(v_max) or v_max <= 0:
            return None

        thr_rel = float(self.blob_threshold_rel)
        thr_rel = min(max(thr_rel, 0.0), 1.0)
        threshold = thr_rel * v_max

        min_sigma = max(float(self.blob_min_sigma), 0.1)
        max_sigma = max(float(self.blob_max_sigma), min_sigma + 1e-6)
        num_sigma = max(1, int(self.blob_num_sigma))
        overlap = float(self.blob_overlap)
        overlap = min(max(overlap, 0.0), 0.99)

        try:
            blobs = feature.blob_log(
                img,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=threshold,
                overlap=overlap,
            )
        except Exception:
            return None

        if blobs is None or len(blobs) == 0:
            return None

        h, w = img.shape
        mask = roi_mask.astype(bool) if roi_mask is not None else None
        try:
            b = int(border_exclude)
        except Exception:
            b = 0
        b = max(0, b)
        candidates = []
        for y, x, s in blobs:
            ix = int(round(float(x)))
            iy = int(round(float(y)))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            if b > 0 and (ix < b or iy < b or ix >= w - b or iy >= h - b):
                continue
            if mask is not None:
                try:
                    if not bool(mask[iy, ix]):
                        continue
                except Exception:
                    pass
            resp = float(img[iy, ix])
            candidates.append((resp, float(x), float(y), float(s)))

        if not candidates:
            return None

        # Sort by response descending
        candidates.sort(key=lambda t: t[0], reverse=True)

        sigma_min, sigma_max = self.sigma_bounds
        params: List[float] = []
        for resp, x, y, s in candidates[: int(n_peaks)]:
            amp = max(float(resp), 1e-6)
            sigma0 = float(s)
            if not np.isfinite(sigma0):
                sigma0 = float(self.initial_sigma)
            sigma0 = min(max(sigma0, float(sigma_min)), float(sigma_max))
            params.extend([amp, float(x), float(y), sigma0])

        while len(params) < n_peaks * 4:
            params.extend([1.0, img.shape[1] / 2.0, img.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.nanmedian(img)))
        return params

    def _initial_params_blob_doh(
        self, frame: np.ndarray, n_peaks: int, roi_mask: Optional[np.ndarray] = None, border_exclude: int = 0
    ) -> Optional[List[float]]:
        """
        Determinant of Hessian (DoH) based blob detection initializer.
        DoH is computationally more efficient than LoG and robust for certain structures.
        Returns None if blob candidates are not usable and the caller should fall back.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None

        v_max = float(np.nanmax(img))
        if not np.isfinite(v_max) or v_max <= 0:
            return None

        thr_rel = float(self.blob_doh_threshold_rel)
        thr_rel = min(max(thr_rel, 0.0), 1.0)
        threshold = thr_rel * v_max

        min_sigma = max(float(self.blob_doh_min_sigma), 0.1)
        max_sigma = max(float(self.blob_doh_max_sigma), min_sigma + 1e-6)
        num_sigma = max(1, int(self.blob_doh_num_sigma))

        try:
            blobs = feature.blob_doh(
                img,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                threshold=threshold,
            )
        except Exception:
            return None

        if blobs is None or len(blobs) == 0:
            return None

        h, w = img.shape
        mask = roi_mask.astype(bool) if roi_mask is not None else None
        try:
            b = int(border_exclude)
        except Exception:
            b = 0
        b = max(0, b)
        candidates = []
        for y, x, s in blobs:
            ix = int(round(float(x)))
            iy = int(round(float(y)))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            if b > 0 and (ix < b or iy < b or ix >= w - b or iy >= h - b):
                continue
            if mask is not None:
                try:
                    if not bool(mask[iy, ix]):
                        continue
                except Exception:
                    pass
            resp = float(img[iy, ix])
            candidates.append((resp, float(x), float(y), float(s)))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0], reverse=True)

        sigma_min, sigma_max = self.sigma_bounds
        params: List[float] = []
        for resp, x, y, s in candidates[: int(n_peaks)]:
            amp = max(float(resp), 1e-6)
            sigma0 = float(s)
            if not np.isfinite(sigma0):
                sigma0 = float(self.initial_sigma)
            sigma0 = min(max(sigma0, float(sigma_min)), float(sigma_max))
            params.extend([amp, float(x), float(y), sigma0])

        while len(params) < n_peaks * 4:
            params.extend([1.0, img.shape[1] / 2.0, img.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.nanmedian(img)))
        return params

    def _initial_params_multiscale(
        self, frame: np.ndarray, n_peaks: int, roi_mask: Optional[np.ndarray] = None, border_exclude: int = 0
    ) -> Optional[List[float]]:
        """
        Multiscale LoG-based peak detection that combines results from multiple sigma values.
        This helps detect spots of varying sizes robustly.
        """
        img = np.asarray(frame, dtype=np.float64)
        if img.ndim != 2:
            return None

        sigmas = self.multiscale_sigmas
        if not sigmas:
            sigmas = [1.0, 1.6, 2.5, 4.0]

        all_peaks = []
        for sigma in sigmas:
            try:
                log_img = -ndimage.gaussian_laplace(img, sigma=float(sigma))
                coords = feature.peak_local_max(
                    log_img,
                    min_distance=int(self.peak_min_distance),
                    num_peaks=n_peaks * 3,
                    exclude_border=False,
                    threshold_rel=self.localmax_threshold_rel,
                )
                if coords is not None and len(coords) > 0:
                    for y, x in coords:
                        iy, ix = int(y), int(x)
                        if 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
                            resp = float(img[iy, ix])
                            all_peaks.append((resp, float(ix), float(iy), float(sigma)))
            except Exception:
                continue

        if not all_peaks:
            return None

        # Apply DBSCAN to remove duplicates and spatial outliers
        if len(all_peaks) > 1:
            coords_array = np.array([[p[1], p[2]] for p in all_peaks])
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=self.peak_min_distance * 1.5, min_samples=1).fit(coords_array)
                labels = clustering.labels_

                # For each cluster, keep the peak with highest response
                unique_labels = set(labels)
                clustered_peaks = []
                for label in unique_labels:
                    if label == -1:
                        continue
                    cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
                    best_idx = max(cluster_indices, key=lambda i: all_peaks[i][0])
                    clustered_peaks.append(all_peaks[best_idx])
                all_peaks = clustered_peaks
            except Exception:
                pass

        # Sort by response descending
        all_peaks.sort(key=lambda t: t[0], reverse=True)

        # Filter by ROI mask
        h, w = img.shape
        mask = roi_mask.astype(bool) if roi_mask is not None else None
        try:
            b = int(border_exclude)
        except Exception:
            b = 0
        b = max(0, b)
        filtered_peaks = []
        for resp, x, y, s in all_peaks:
            ix = int(round(float(x)))
            iy = int(round(float(y)))
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                continue
            if b > 0 and (ix < b or iy < b or ix >= w - b or iy >= h - b):
                continue
            if mask is not None:
                try:
                    if not bool(mask[iy, ix]):
                        continue
                except Exception:
                    pass
            filtered_peaks.append((resp, x, y, s))

        if not filtered_peaks:
            return None

        sigma_min, sigma_max = self.sigma_bounds
        params: List[float] = []
        for resp, x, y, s in filtered_peaks[: int(n_peaks)]:
            amp = max(float(resp), 1e-6)
            sigma0 = float(s)
            if not np.isfinite(sigma0):
                sigma0 = float(self.initial_sigma)
            sigma0 = min(max(sigma0, float(sigma_min)), float(sigma_max))
            params.extend([amp, float(x), float(y), sigma0])

        while len(params) < n_peaks * 4:
            params.extend([1.0, img.shape[1] / 2.0, img.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.nanmedian(img)))
        return params

    def _filter_peaks_dbscan(
        self, peaks: List[PeakStat], eps: Optional[float] = None, min_samples: Optional[int] = None
    ) -> List[PeakStat]:
        """
        Apply DBSCAN clustering to filter out spatial outliers (noise-derived peaks).
        Isolated peaks that don't form clusters are considered outliers and removed.
        """
        if not peaks or len(peaks) <= 1:
            return peaks

        eps_val = eps if eps is not None else self.dbscan_eps
        min_samp = min_samples if min_samples is not None else self.dbscan_min_samples

        coords = np.array([[pk.x, pk.y] for pk in peaks])

        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=float(eps_val), min_samples=int(min_samp)).fit(coords)
            labels = clustering.labels_

            # Keep only peaks that are not labeled as noise (-1)
            filtered = [pk for pk, label in zip(peaks, labels) if label != -1]
            return filtered if filtered else peaks
        except Exception:
            return peaks

    def _build_bounds(self, frame: np.ndarray, n_peaks: int) -> Tuple[List[float], List[float]]:
        h, w = frame.shape
        lower: List[float] = []
        upper: List[float] = []
        for _ in range(n_peaks):
            lower.extend([0.0, 0.0, 0.0, self.sigma_bounds[0]])
            upper.extend([np.inf, float(w), float(h), self.sigma_bounds[1]])
        lower.append(-np.inf)  # offset
        upper.append(np.inf)
        return lower, upper

    def _build_bounds_seeded(
        self,
        frame: np.ndarray,
        n_peaks: int,
        seeds_xy_local: Optional[Sequence[Tuple[float, float]]] = None,
        max_shift_px: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Bounds builder that optionally constrains (x,y) around per-peak seeds.
        If seeds are not provided or max_shift_px <= 0, falls back to wide bounds.
        """
        h, w = frame.shape
        r = int(self.refit_max_shift_px if max_shift_px is None else max_shift_px)
        if seeds_xy_local is None or r <= 0:
            return self._build_bounds(frame, n_peaks)

        seeds = list(seeds_xy_local)
        lower: List[float] = []
        upper: List[float] = []
        for i in range(int(n_peaks)):
            # amplitude >= 0
            lower.append(0.0)
            upper.append(np.inf)

            if i < len(seeds):
                sx, sy = seeds[i]
                try:
                    sx = float(sx)
                    sy = float(sy)
                except Exception:
                    sx = float(w) / 2.0
                    sy = float(h) / 2.0
                x_lo = max(0.0, sx - float(r))
                x_hi = min(float(w), sx + float(r))
                y_lo = max(0.0, sy - float(r))
                y_hi = min(float(h), sy + float(r))
                # guard: ensure strict ordering for curve_fit bounds
                if x_hi <= x_lo:
                    x_lo, x_hi = 0.0, float(w)
                if y_hi <= y_lo:
                    y_lo, y_hi = 0.0, float(h)
            else:
                x_lo, x_hi = 0.0, float(w)
                y_lo, y_hi = 0.0, float(h)

            lower.extend([x_lo, y_lo, self.sigma_bounds[0]])
            upper.extend([x_hi, y_hi, self.sigma_bounds[1]])

        lower.append(-np.inf)  # offset
        upper.append(np.inf)
        return lower, upper

    def _multi_gaussian(self, xy_mesh: Tuple[np.ndarray, np.ndarray], *params: float) -> np.ndarray:
        x, y = xy_mesh
        offset = params[-1]
        g = np.full_like(x, offset, dtype=np.float64)
        core = params[:-1]
        for i in range(0, len(core), 4):
            amp, x0, y0, sigma = core[i : i + 4]
            denom = 2.0 * (sigma**2 + 1e-12)
            g += amp * np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / denom))
        return g.ravel()

    def _fit_model(
        self,
        roi: np.ndarray,
        origin: Tuple[int, int],
        n_peaks: int,
        noise_sigma: float,
        roi_mask: Optional[np.ndarray] = None,
        initial_xy_local: Optional[List[Tuple[float, float]]] = None,
        init_image: Optional[np.ndarray] = None,
        init_noise_sigma: Optional[float] = None,
    ) -> ModelSelectionResult:
        h, w = roi.shape
        x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        p0: List[float]
        if initial_xy_local:
            # Prefer provided seed points (x,y in ROI-local coords)
            seeds: List[Tuple[float, float]] = []
            for lx, ly in initial_xy_local[: int(n_peaks)]:
                try:
                    fx = float(lx)
                    fy = float(ly)
                except Exception:
                    continue
                # keep within bounds
                fx = float(np.clip(fx, 0.0, float(w - 1)))
                fy = float(np.clip(fy, 0.0, float(h - 1)))
                seeds.append((fx, fy))

            sigma0 = float(self.initial_sigma)
            p0 = []
            for fx, fy in seeds:
                ix = int(round(fx))
                iy = int(round(fy))
                try:
                    amp0 = float(roi[iy, ix])
                except Exception:
                    amp0 = 1.0
                p0.extend([max(float(amp0), 1e-6), float(fx), float(fy), sigma0])
            while len(p0) < n_peaks * 4:
                p0.extend([1.0, roi.shape[1] / 2.0, roi.shape[0] / 2.0, float(self.initial_sigma)])
            p0.append(float(np.nanmedian(roi)))
        else:
            init_img = init_image if init_image is not None else roi
            # If using a detection image (DoG/LoG), validate peaks against the fit target image
            validate_img = roi if init_image is not None else None
            p0 = self._initial_params(
                init_img,
                n_peaks,
                roi_mask=roi_mask,
                init_noise_sigma=init_noise_sigma,
                validate_image=validate_img,
            )
            # If initialization was performed on detection image (DoG/LoG),
            # adjust amplitude/offset guesses using the fit target image.
            if init_image is not None and len(p0) >= n_peaks * 4 + 1:
                try:
                    for i in range(int(n_peaks)):
                        x0 = float(p0[i * 4 + 1])
                        y0 = float(p0[i * 4 + 2])
                        ix = int(round(x0))
                        iy = int(round(y0))
                        if 0 <= ix < w and 0 <= iy < h:
                            p0[i * 4 + 0] = max(float(roi[iy, ix]), 1e-6)
                    p0[-1] = float(np.nanmedian(roi))
                except Exception:
                    pass
        # Prevent peak collapse by constraining (x,y) around the initializer seeds.
        # If manual seeds are provided, they take precedence.
        seeds_for_bounds: Optional[List[Tuple[float, float]]] = None
        if initial_xy_local:
            seeds_for_bounds = list(initial_xy_local)
        elif len(p0) >= n_peaks * 4:
            try:
                seeds_for_bounds = [(float(p0[i * 4 + 1]), float(p0[i * 4 + 2])) for i in range(int(n_peaks))]
            except Exception:
                seeds_for_bounds = None
        bounds = self._build_bounds_seeded(
            roi,
            n_peaks,
            seeds_xy_local=seeds_for_bounds,
            max_shift_px=self.refit_max_shift_px,
        )

        init_peaks: List[PeakStat] = []
        if len(p0) >= n_peaks * 4:
            for i in range(n_peaks):
                amp0, x0, y0, sigma0 = p0[i * 4 : (i + 1) * 4]
                init_peaks.append(
                    PeakStat(
                        amplitude=float(amp0),
                        x=float(origin[0] + x0),
                        y=float(origin[1] + y0),
                        sigma=float(sigma0),
                        snr=float(amp0) / max(noise_sigma, 1e-6),
                    )
                )

        try:
            if roi_mask is not None:
                mask = roi_mask.astype(bool)
                x_fit = x[mask].ravel()
                y_fit = y[mask].ravel()
                z_fit = roi[mask].ravel()
            else:
                x_fit = x.ravel()
                y_fit = y.ravel()
                z_fit = roi.ravel()
            popt, _ = curve_fit(
                self._multi_gaussian,
                (x_fit, y_fit),
                z_fit,
                p0=p0,
                bounds=bounds,
                maxfev=self.max_iterations,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Gaussian fit failed (n=%s): %s", n_peaks, exc)
            popt = np.asarray(p0, dtype=np.float64)

        model = self._multi_gaussian((x, y), *popt).reshape(roi.shape)
        residuals = roi - model
        if roi_mask is not None:
            mask = roi_mask.astype(bool)
            residuals_eval = residuals[mask]
        else:
            residuals_eval = residuals.ravel()
        rss = float(np.sum(residuals_eval**2))
        residual_std = float(np.std(residuals_eval))

        sigma = max(noise_sigma, 1e-6)
        loglike = -0.5 * np.sum((residuals_eval / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))

        k = len(popt)
        n = int(np.sum(roi_mask)) if roi_mask is not None else roi.size
        aic = -2.0 * loglike + 2.0 * k
        bic = -2.0 * loglike + k * np.log(n)

        peaks: List[PeakStat] = []
        for i in range(n_peaks):
            amp, x0, y0, sigma_i = popt[i * 4 : (i + 1) * 4]
            peaks.append(
                PeakStat(
                    amplitude=float(amp),
                    x=float(origin[0] + x0),
                    y=float(origin[1] + y0),
                    sigma=float(sigma_i),
                    snr=float(amp) / max(noise_sigma, 1e-6),
                )
            )

        return ModelSelectionResult(
            n_peaks=n_peaks,
            popt=popt,
            rss=rss,
            loglike=float(loglike),
            aic=float(aic),
            bic=float(bic),
            residual_std=residual_std,
            init_peaks=init_peaks,
            peaks=peaks,
        )


__all__ = [
    "SpotAnalysis",
    "FrameAnalysis",
    "ModelSelectionResult",
    "PeakStat",
    "SpotAnalysisWindow",
    "PLUGIN_NAME",
    "create_plugin",
]


class SpotVisualizationWindow(QtWidgets.QMainWindow):
    """
    解析結果（ROI画像と検出ピーク）を可視化するウィンドウ。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spot Analysis Visualization")
        self.resize(600, 500)
        
        # メインウィジェット
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Matplotlib Figure
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.cbar = None
        
        # ツールバー（Matplotlib標準）
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

    def update_view(self, roi: np.ndarray, result: FrameAnalysis, origin: Tuple[int, int]):
        """ROI画像とフィッティング結果を描画"""
        self.ax.clear()
        self._safe_remove_colorbar()
        # 背景画像
        im = self.ax.imshow(roi, cmap='viridis', origin='lower')
        self.cbar = self.figure.colorbar(im, ax=self.ax, label='Height (nm)')
        
        # 選択されたモデルのピークをプロット
        best_model = result.models[result.best_n_peaks]
        colors = ["magenta", "white", "orange", "cyan", "yellow", "lime", "red", "deepskyblue"]
        
        for i, pk in enumerate(best_model.peaks):
            # PeakStatの座標は全体座標なので、ROI内相対座標に戻す
            rx = pk.x - origin[0]
            ry = pk.y - origin[1]
            
            label = f"P{i+1}: S/N={pk.snr:.1f}"
            
            self.ax.plot(rx, ry, 'x', color=colors[i % len(colors)], markersize=10, markeredgewidth=2, label=label)
            self.ax.text(rx + 1, ry + 1, f"P{i+1}", color='white', fontsize=10, fontweight='bold')

        # 2ピークモデルと3ピークモデルの両方を比較したい場合のために、
        # 補助的に他のモデルの点も点線などで出すことも検討できますが、
        # まずは「採用されたモデル」を表示します。
        
        self.ax.set_title(f"Model: {result.best_n_peaks} peaks ({result.criterion.upper()})")
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.set_xlabel("X (pixel)")
        self.ax.set_ylabel("Y (pixel)")
        # Z Scale Barは表示しない
        
        self.canvas.draw()

    def _safe_remove_colorbar(self):
        """Colorbarを安全に削除する（MatplotlibのKeyError対策）"""
        try:
            if self.cbar:
                self.cbar.remove()
        except Exception:
            pass
        self.cbar = None


class SpotFullImageWindow(QtWidgets.QMainWindow):
    """
    全画像にピーク位置をオーバーレイ表示するウィンドウ。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spot Analysis: Full Overlay Image")
        self.resize(700, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.figure = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        gs = self.figure.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.35, wspace=0.25)
        self.ax_afm = self.figure.add_subplot(gs[0, :])
        self.ax_roi_pre = self.figure.add_subplot(gs[1, 0])
        self.ax_roi_det = self.figure.add_subplot(gs[1, 1])
        self.cbar = None
        self.selector = None
        self.on_select_callback = None
        self.edit_handler = None
        self._dragging = False
        self._drag_index = None
        self._roi_selector_paused = False

        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def set_edit_handler(self, handler):
        """編集イベントをSpotAnalysisWindowへ渡す"""
        self.edit_handler = handler

    def enable_roi_selector(self, shape: str, callback):
        """ROI選択を有効化し、選択結果をcallback(dict)に通知"""
        from matplotlib.widgets import RectangleSelector, EllipseSelector

        self.on_select_callback = callback

        if self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None

        def _emit_rect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            w = xmax - xmin
            h = ymax - ymin
            if callback:
                callback({
                    "shape": "Rectangle",
                    "x0": xmin,
                    "y0": ymin,
                    "w": w,
                    "h": h,
                })

        def _emit_ellipse(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            rx = (xmax - xmin) / 2.0
            ry = (ymax - ymin) / 2.0
            if callback:
                callback({
                    "shape": "Ellipse",
                    "cx": cx,
                    "cy": cy,
                    "rx": rx,
                    "ry": ry,
                    "x0": xmin,
                    "y0": ymin,
                    "w": xmax - xmin,
                    "h": ymax - ymin,
                })

        if shape == "Ellipse (Circle)":
            self.selector = EllipseSelector(
                self.ax_afm,
                _emit_ellipse,
                useblit=True,
                button=[1],
                minspanx=2,
                minspany=2,
                interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )
        else:
            self.selector = RectangleSelector(
                self.ax_afm,
                _emit_rect,
                useblit=True,
                button=[1],
                minspanx=2,
                minspany=2,
                interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )

    def update_view(
        self,
        frame: np.ndarray,
        result: Optional[FrameAnalysis],
        spots: Optional[List[Dict[str, float]]] = None,
        initial_spots: Optional[List[Dict[str, float]]] = None,
        roi_overlay: Optional[Dict[str, float]] = None,
        spot_radius_px: Optional[float] = None,
        roi_pre_image: Optional[np.ndarray] = None,
        roi_det_image: Optional[np.ndarray] = None,
        det_label: Optional[str] = None,
        show_roi_spots: bool = True,
        show_det_spots: bool = True,
        show_initial_spots: bool = False,
        show_fit_spots_on_roi_pre: bool = False,
        init_mode: Optional[str] = None,
    ) -> None:
        """上段: AFM全画像 / 下段: ROI前処理(左) + ROI検出画像(右) を描画"""
        # NOTE: Avoid UnboundLocalError if 'np' becomes a local binding unexpectedly.
        import numpy as np

        self.ax_afm.clear()
        self.ax_roi_pre.clear()
        self.ax_roi_det.clear()
        self._safe_remove_colorbar()

        # --- AFM (top) ---
        self.ax_afm.imshow(frame, cmap="viridis", origin="lower")

        # --- ROI images (bottom) ---
        pre = roi_pre_image
        det = roi_det_image
        if pre is None:
            pre = np.full((2, 2), np.nan, dtype=np.float64)
        if det is None:
            det = np.full((2, 2), np.nan, dtype=np.float64)

        pre_cmap = plt.get_cmap("viridis").copy()
        det_cmap = plt.get_cmap("magma").copy()
        for cm in (pre_cmap, det_cmap):
            try:
                cm.set_bad(color=(0, 0, 0, 0))
            except Exception:
                pass
        self.ax_roi_pre.imshow(pre, cmap=pre_cmap, origin="lower")
        self.ax_roi_det.imshow(det, cmap=det_cmap, origin="lower")

        # Determine detection mode for conditional overlays.
        # - Preprocessed: det_label starts with "Pre"
        # - DoG/LoG: otherwise (when det_label is missing, assume DoG/LoG-like behavior)
        _dl = (det_label or "").strip().lower()
        is_pre = _dl.startswith("pre")
        is_dog_log = not is_pre
        _im = (init_mode or "").strip().lower()
        is_ws_init = _im.startswith("watershed")

        # Fill defaults from result if caller didn't supply spot lists.
        if spots is None and result is not None:
            spots = [{"x": pk.x, "y": pk.y, "snr": pk.snr} for pk in result.models[result.best_n_peaks].peaks]
        if initial_spots is None and result is not None:
            initial_spots = [{"x": pk.x, "y": pk.y, "snr": pk.snr} for pk in result.models[result.best_n_peaks].init_peaks]

        # Spots
        if spots:
            colors = ["magenta", "white", "orange", "cyan", "yellow", "lime", "red", "deepskyblue"]
            for i, pk in enumerate(spots):
                label = f"P{i+1}: S/N={pk.get('snr', 0.0):.1f}"
                # top: absolute
                self.ax_afm.plot(pk["x"], pk["y"], "x", color=colors[i % len(colors)], markersize=10, markeredgewidth=2, label=label)
                self.ax_afm.text(pk["x"] + 1, pk["y"] + 1, f"P{i+1}", color="white", fontsize=10, fontweight="bold")
                if spot_radius_px is not None and spot_radius_px > 0:
                    try:
                        self.ax_afm.add_patch(
                            Circle(
                                (pk["x"], pk["y"]),
                                radius=float(spot_radius_px),
                                fill=False,
                                linewidth=0.8,
                                edgecolor="white",
                                linestyle="-",
                                alpha=0.9,
                            )
                        )
                    except Exception:
                        pass
            self.ax_afm.legend(loc="upper right", fontsize=8)

        if roi_overlay:
            # top: ROI outline in absolute coords
            if roi_overlay.get("shape") == "Ellipse":
                ellipse = Ellipse(
                    (roi_overlay["cx"], roi_overlay["cy"]),
                    width=roi_overlay["rx"] * 2.0,
                    height=roi_overlay["ry"] * 2.0,
                    linewidth=1.5,
                    edgecolor="white",
                    facecolor="none",
                    linestyle="--",
                )
                self.ax_afm.add_patch(ellipse)
            else:
                rect = Rectangle(
                    (roi_overlay["x0"], roi_overlay["y0"]),
                    roi_overlay["w"],
                    roi_overlay["h"],
                    linewidth=1.5,
                    edgecolor="white",
                    facecolor="none",
                    linestyle="--",
                )
                self.ax_afm.add_patch(rect)

            # bottom: plot spots in ROI-local coords (optional)
            # Rules:
            # - Watershed initializer runs on ROI-pre (intensity), so show init spots on ROI-pre.
            # - Otherwise:
            #   - ROI-pre (left): show init spots when detection label is Preprocessed.
            #   - ROI-det (right): show init spots when detection label is DoG/LoG.
            show_init_on_pre = bool(show_roi_spots and initial_spots and (is_pre or is_ws_init))
            show_init_on_det = bool(show_det_spots and initial_spots and is_dog_log and (not is_ws_init))
            if roi_overlay and (show_init_on_pre or show_init_on_det or (show_fit_spots_on_roi_pre and spots)):
                try:
                    # Use result.origin for coordinate conversion if available (accounts for boundary clipping)
                    if result is not None:
                        x0 = float(result.origin[0])
                        y0 = float(result.origin[1])
                    else:
                        # Fallback: clip roi_overlay coordinates to image boundaries
                        h_img, w_img = frame.shape
                        x0_raw = float(roi_overlay.get("x0", 0.0))
                        y0_raw = float(roi_overlay.get("y0", 0.0))
                        x0 = float(max(int(round(x0_raw)), 0))
                        y0 = float(max(int(round(y0_raw)), 0))

                    w = float(roi_overlay.get("w", 0.0))
                    h = float(roi_overlay.get("h", 0.0))

                    # マージンチェック用の設定（ROIマージンで除外される位置は表示しない）
                    margin = 0
                    roi_mask_check = None
                    try:
                        # SpotAnalysisPluginからmargin値を取得
                        from PyQt5.QtWidgets import QApplication
                        for widget in QApplication.topLevelWidgets():
                            if hasattr(widget, 'margin_spin'):
                                margin = int(widget.margin_spin.value())
                                break
                    except Exception:
                        pass

                    # 楕円ROIの場合はマスクを作成
                    if roi_overlay.get("shape") == "Ellipse":
                        try:
                            cx = float(roi_overlay.get("cx", 0.0))
                            cy = float(roi_overlay.get("cy", 0.0))
                            rx = float(roi_overlay.get("rx", 1.0))
                            ry = float(roi_overlay.get("ry", 1.0))

                            # ROIローカル座標でマスクを作成
                            yy_grid, xx_grid = np.mgrid[0:int(h), 0:int(w)]
                            mask_full = ((xx_grid + x0 - cx) ** 2) / (rx ** 2 + 1e-12) + ((yy_grid + y0 - cy) ** 2) / (ry ** 2 + 1e-12) <= 1.0

                            if margin > 0:
                                roi_mask_check = morphology.erosion(mask_full.astype(bool), morphology.disk(margin))
                            else:
                                roi_mask_check = mask_full.astype(bool)
                        except Exception:
                            pass
                    else:
                        # 矩形ROIの場合
                        if margin > 0:
                            # マージン分の境界チェック用
                            pass  # 後で個別にチェック
                    if w > 0 and h > 0:
                        colors = ["magenta", "white", "orange", "cyan", "yellow", "lime", "red", "deepskyblue"]
                        # Debug overlay: final (fit) peaks on ROI-pre image.
                        # This helps compare initializer vs fit result in the same coordinate system.
                        if show_fit_spots_on_roi_pre and spots:
                            for i, pk in enumerate(spots):
                                sx = float(pk.get("x", 0.0))
                                sy = float(pk.get("y", 0.0))
                                rx = sx - x0
                                ry = sy - y0

                                # マージンチェック（表示ノイズ回避）。ただし、座標不整合の切り分け用途なので
                                # ROI外はそのままスキップ（軸外で見えないため）。
                                if roi_mask_check is not None:
                                    iy, ix = int(round(ry)), int(round(rx))
                                    if 0 <= iy < roi_mask_check.shape[0] and 0 <= ix < roi_mask_check.shape[1]:
                                        if not roi_mask_check[iy, ix]:
                                            continue
                                    else:
                                        continue
                                elif margin > 0:
                                    if rx < margin or ry < margin or rx > w - margin or ry > h - margin:
                                        continue

                                # ROI外は表示しない（見えないため）
                                if rx < 0 or ry < 0 or rx > w or ry > h:
                                    continue

                                self.ax_roi_pre.plot(
                                    rx,
                                    ry,
                                    "o",
                                    markerfacecolor="none",
                                    markeredgecolor=colors[i % len(colors)],
                                    markersize=10,
                                    markeredgewidth=2.0,
                                    linestyle="None",
                                    alpha=0.95,
                                )
                                if spot_radius_px is not None and spot_radius_px > 0:
                                    self.ax_roi_pre.add_patch(
                                        Circle(
                                            (rx, ry),
                                            radius=float(spot_radius_px),
                                            fill=False,
                                            linewidth=0.8,
                                            edgecolor=colors[i % len(colors)],
                                            linestyle="-",
                                            alpha=0.85,
                                        )
                                    )
                        if show_init_on_pre and initial_spots:
                            for i, pk in enumerate(initial_spots):
                                sx = float(pk.get("x", 0.0))
                                sy = float(pk.get("y", 0.0))
                                if sx < x0 or sy < y0 or sx > x0 + w or sy > y0 + h:
                                    continue
                                rx = sx - x0
                                ry = sy - y0

                                # マージンチェック
                                if roi_mask_check is not None:
                                    # 楕円ROI: マスクでチェック
                                    iy, ix = int(round(ry)), int(round(rx))
                                    if 0 <= iy < roi_mask_check.shape[0] and 0 <= ix < roi_mask_check.shape[1]:
                                        if not roi_mask_check[iy, ix]:
                                            continue  # マージン範囲内なのでスキップ
                                    else:
                                        continue
                                elif margin > 0:
                                    # 矩形ROI: 境界からの距離でチェック
                                    if rx < margin or ry < margin or rx > w - margin or ry > h - margin:
                                        continue  # マージン範囲内なのでスキップ
                                self.ax_roi_pre.plot(
                                    rx, ry, "x", color=colors[i % len(colors)], markersize=9, markeredgewidth=2
                                )
                                self.ax_roi_pre.text(
                                    rx + 0.8, ry + 0.8, f"P{i+1}", color="white", fontsize=9, fontweight="bold"
                                )
                                if spot_radius_px is not None and spot_radius_px > 0:
                                    self.ax_roi_pre.add_patch(
                                        Circle(
                                            (rx, ry),
                                            radius=float(spot_radius_px),
                                            fill=False,
                                            linewidth=0.8,
                                            edgecolor="white",
                                            linestyle="-",
                                            alpha=0.9,
                                        )
                                    )
                        if show_init_on_det and initial_spots:
                            for i, pk in enumerate(initial_spots):
                                sx = float(pk.get("x", 0.0))
                                sy = float(pk.get("y", 0.0))
                                if sx < x0 or sy < y0 or sx > x0 + w or sy > y0 + h:
                                    continue
                                rx = sx - x0
                                ry = sy - y0

                                # マージンチェック
                                if roi_mask_check is not None:
                                    # 楕円ROI: マスクでチェック
                                    iy, ix = int(round(ry)), int(round(rx))
                                    if 0 <= iy < roi_mask_check.shape[0] and 0 <= ix < roi_mask_check.shape[1]:
                                        if not roi_mask_check[iy, ix]:
                                            continue  # マージン範囲内なのでスキップ
                                    else:
                                        continue
                                elif margin > 0:
                                    # 矩形ROI: 境界からの距離でチェック
                                    if rx < margin or ry < margin or rx > w - margin or ry > h - margin:
                                        continue  # マージン範囲内なのでスキップ
                                self.ax_roi_det.plot(
                                    rx, ry, "x", color=colors[i % len(colors)], markersize=9, markeredgewidth=2
                                )
                                self.ax_roi_det.text(
                                    rx + 0.8, ry + 0.8, f"P{i+1}", color="white", fontsize=9, fontweight="bold"
                                )
                                if spot_radius_px is not None and spot_radius_px > 0:
                                    self.ax_roi_det.add_patch(
                                        Circle(
                                            (rx, ry),
                                            radius=float(spot_radius_px),
                                            fill=False,
                                            linewidth=0.8,
                                            edgecolor="white",
                                            linestyle="-",
                                            alpha=0.9,
                                        )
                                    )
                except Exception:
                    pass

        if result is not None:
            self.ax_afm.set_title(f"AFM - {result.best_n_peaks} peaks ({result.criterion.upper()})")
        elif spots is not None:
            # Analysis has run but no peaks survived post-filters (strict ROI margin, etc.)
            self.ax_afm.set_title("AFM - 0 peaks")
        else:
            self.ax_afm.set_title("AFM (no analysis yet)")
        self.ax_roi_pre.set_title("ROI (preprocessed)")
        self.ax_roi_det.set_title(det_label or "ROI (LoG/DoG)")

        self.ax_afm.set_xlabel("X (pixel)")
        self.ax_afm.set_ylabel("Y (pixel)")
        for ax in (self.ax_roi_pre, self.ax_roi_det):
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
        self.canvas.draw()

    def _safe_remove_colorbar(self):
        """Colorbarを安全に削除する（MatplotlibのKeyError対策）"""
        try:
            if self.cbar:
                self.cbar.remove()
        except Exception:
            pass
        self.cbar = None

    def _on_press(self, event):
        if self.edit_handler is None:
            return
        # Cmd/Ctrl 押下中はROIセレクタを一時停止して、ドラッグをSpot編集に譲る
        is_mod = False
        try:
            gui_ev = getattr(event, "guiEvent", None)
            if gui_ev is not None and hasattr(gui_ev, "modifiers"):
                mods = gui_ev.modifiers()
                is_mod = bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
        except Exception:
            is_mod = False
        if not is_mod:
            try:
                mods = QtWidgets.QApplication.keyboardModifiers()
                is_mod = bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
            except Exception:
                pass
        if is_mod and self.selector is not None:
            try:
                self.selector.set_active(False)
                self._roi_selector_paused = True
            except Exception:
                pass
        self.edit_handler(event, "press")

    def _on_motion(self, event):
        if self.edit_handler is None:
            return
        self.edit_handler(event, "move")

    def _on_release(self, event):
        if self.edit_handler is None:
            return
        self.edit_handler(event, "release")
        # 一時停止していたROIセレクタを復帰
        if self._roi_selector_paused and self.selector is not None:
            try:
                self.selector.set_active(True)
            except Exception:
                pass
        self._roi_selector_paused = False


class SpotAnalysisWindow(QtWidgets.QWidget):
    """
    PyNuDメインウィンドウからオンデマンドで呼び出す簡易UI。
    FileListで選択されたファイルの現在フレームに対し、SpotAnalysisを実行する。
    """

    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Spot Analysis (AIC/BIC)")
        self.setMinimumWidth(420)
        self.spot_analyzer = SpotAnalysis()
        self.viz_window = None  # ROI可視化ウィンドウ
        self.full_viz_window = None  # 全画像オーバーレイウィンドウ
        self.last_frame = None
        self.last_result = None
        self.manual_roi = None  # dict: {"shape": str, "x0":..., "y0":..., "w":..., "h":..., "cx":..., "cy":..., "rx":..., "ry":...}
        self.roi_by_frame: Dict[int, Dict[str, float]] = {}
        self.spots_by_frame: Dict[int, List[Dict[str, float]]] = {}
        self.initial_spots_by_frame: Dict[int, List[Dict[str, float]]] = {}
        self.export_dir = None
        self._dragging = False
        self._drag_index = None
        self._auto_busy = False
        self._analysis_signature_by_frame: Dict[int, Tuple] = {}
        self._reanalysis_timer = QtCore.QTimer(self)
        self._reanalysis_timer.setSingleShot(True)
        self._reanalysis_timer.timeout.connect(self._reanalyze_current_frame_debounced)
        self._build_ui()
        self._refresh_selection_label()
        self._connect_frame_signal()
        self.show_full_image_view()
        self._update_frame_label()

    def closeEvent(self, event):
        """ウィンドウを閉じるときに可視化ウィンドウも閉じる"""
        if self.viz_window:
            self.viz_window.close()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        outer_layout = QtWidgets.QVBoxLayout(self)

        def _setup_grid(grid: QtWidgets.QGridLayout) -> None:
            # 0: label, 1: input, 2: expanding spacer column
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 0)
            grid.setColumnStretch(2, 1)
            try:
                grid.setHorizontalSpacing(12)
            except Exception:
                pass

        def _add_right_spacer(grid: QtWidgets.QGridLayout, row_span: int) -> None:
            try:
                grid.addItem(
                    QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum),
                    0,
                    2,
                    max(1, int(row_span)),
                    1,
                )
            except Exception:
                pass

        # --- Top buttons (always visible) ---
        top_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("解析を実行")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)

        self.refit_manual_btn = QtWidgets.QPushButton("手動スポットで再フィット")
        self.refit_manual_btn.setToolTip("手動で調整したspot数/座標を初期値にして、±r(px)以内の制約付きで再フィットします（現在フレームのみ）。")
        self.refit_manual_btn.clicked.connect(self.refit_from_manual_spots)
        self.refit_manual_btn.setEnabled(False)

        self.refit_shift_spin = QtWidgets.QSpinBox()
        self.refit_shift_spin.setRange(0, 50)
        self.refit_shift_spin.setValue(3)
        self.refit_shift_spin.setMaximumWidth(70)
        self.refit_shift_spin.setToolTip("手動spotからの許容移動量（±r px）。0なら制約なし。")

        self.run_all_btn = QtWidgets.QPushButton("全フレーム解析")
        self.run_all_btn.clicked.connect(self.run_analysis_all_frames)
        self.run_all_btn.setEnabled(False)

        self.run_all_enable_check = QtWidgets.QCheckBox("全フレーム解析")
        self.run_all_enable_check.setChecked(False)
        self.run_all_enable_check.setToolTip("ONのときのみ「全フレーム解析」ボタンが有効になります。")
        self.run_all_enable_check.toggled.connect(self._sync_run_buttons_enabled)

        self.prev_frame_btn = QtWidgets.QPushButton("◀")
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        self.next_frame_btn = QtWidgets.QPushButton("▶")
        self.next_frame_btn.clicked.connect(self._next_frame)
        self.frame_label = QtWidgets.QLabel("Frame: - / -")

        # --- 1st row: Run button + Frame navigation ---
        top_row.addWidget(self.run_btn)
        top_row.addWidget(self.prev_frame_btn)
        top_row.addWidget(self.next_frame_btn)
        top_row.addWidget(self.frame_label)
        top_row.addStretch(1)
        outer_layout.addLayout(top_row)

        # --- 2nd row: Manual spot reanalysis + tolerance ---
        refit_row = QtWidgets.QHBoxLayout()
        refit_row.addWidget(self.refit_manual_btn)
        refit_row.addWidget(QtWidgets.QLabel("許容移動(px)"))
        refit_row.addWidget(self.refit_shift_spin)
        refit_row.addStretch(1)
        outer_layout.addLayout(refit_row)

        # --- 3rd row: Run all frames + enable ---
        run_all_row = QtWidgets.QHBoxLayout()
        run_all_row.addWidget(self.run_all_btn)
        run_all_row.addWidget(self.run_all_enable_check)
        run_all_row.addStretch(1)
        outer_layout.addLayout(run_all_row)

        # --- Scroll area ---
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer_layout.addWidget(self._scroll)
        content = QtWidgets.QWidget()
        self._scroll.setWidget(content)
        layout = QtWidgets.QHBoxLayout(content)

        # --- Form widget (fixed width, left-aligned) ---
        form_widget = QtWidgets.QWidget()
        form_widget.setMaximumWidth(440)
        form_layout = QtWidgets.QVBoxLayout(form_widget)

        # --- Groups ---
        basic_group = QtWidgets.QGroupBox("ROI / 基本")
        basic_grid = QtWidgets.QGridLayout(basic_group)
        _setup_grid(basic_grid)
        r = 0
        roi_shape_label = QtWidgets.QLabel("ROI形状")
        roi_shape_label.setToolTip(
            "解析領域（ROI）の形状:\n"
            "・Rectangle: 矩形領域\n"
            "・Ellipse (Circle): 楕円/円形領域（境界外を除外）"
        )
        basic_grid.addWidget(roi_shape_label, r, 0)
        self.roi_shape_combo = QtWidgets.QComboBox()
        self.roi_shape_combo.addItems(["Rectangle", "Ellipse (Circle)"])
        self.roi_shape_combo.setToolTip("ROI形状を選択します。")
        self.roi_shape_combo.setMaximumWidth(150)
        self.roi_shape_combo.currentTextChanged.connect(self._on_roi_shape_changed)
        basic_grid.addWidget(self.roi_shape_combo, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        auto_analyze_label = QtWidgets.QLabel("自動解析")
        auto_analyze_label.setToolTip(
            "フレーム切替時に自動的に解析を実行:\n"
            "有効にすると、ROIが設定されているフレームに切り替えた際、\n"
            "自動的にピーク検出とフィッティングを実行します。"
        )
        basic_grid.addWidget(auto_analyze_label, r, 0)
        self.auto_analyze_check = QtWidgets.QCheckBox("フレーム切替時に自動解析")
        self.auto_analyze_check.setChecked(True)  # デフォルトで有効
        self.auto_analyze_check.setToolTip("ROIがあるフレームで自動的に解析を行います。")
        basic_grid.addWidget(self.auto_analyze_check, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        criterion_label = QtWidgets.QLabel("情報量基準")
        criterion_label.setToolTip(
            "最適なピーク数を選択する情報量基準:\n"
            "・AIC (Akaike Information Criterion): 予測精度重視\n"
            "・BIC (Bayesian Information Criterion): モデル簡潔性重視（より保守的）\n"
            "一般的にBICの方がオーバーフィッティングを抑えます。"
        )
        basic_grid.addWidget(criterion_label, r, 0)
        self.criterion_combo = QtWidgets.QComboBox()
        self.criterion_combo.addItems(["AIC", "BIC"])
        self.criterion_combo.setMaximumWidth(100)
        basic_grid.addWidget(self.criterion_combo, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # min/max peaks (paired)
        self.min_peaks_spin = QtWidgets.QSpinBox()
        self.min_peaks_spin.setRange(1, 6)
        self.min_peaks_spin.setValue(1)
        self.min_peaks_spin.setMaximumWidth(70)
        self.max_peaks_spin = QtWidgets.QSpinBox()
        self.max_peaks_spin.setRange(1, 6)
        self.max_peaks_spin.setValue(3)
        self.max_peaks_spin.setMaximumWidth(70)
        peaks_row = QtWidgets.QHBoxLayout()
        peaks_row.addWidget(QtWidgets.QLabel("最小"))
        peaks_row.addWidget(self.min_peaks_spin)
        peaks_row.addSpacing(8)
        peaks_row.addWidget(QtWidgets.QLabel("最大"))
        peaks_row.addWidget(self.max_peaks_spin)
        peaks_label = QtWidgets.QLabel("ピーク数")
        peaks_label.setToolTip(
            "フィッティングで試行するピーク数の範囲。\n"
            "Peak系初期候補（seed）の候補数上限にも使われます。\n"
            "AIC/BICで最適なモデルを選択します。"
        )
        basic_grid.addWidget(peaks_label, r, 0)
        basic_grid.addLayout(peaks_row, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        _add_right_spacer(basic_grid, r)
        form_layout.addWidget(basic_group)

        preprocess_group = QtWidgets.QGroupBox("前処理（ROI内）")
        pre_grid = QtWidgets.QGridLayout(preprocess_group)
        _setup_grid(pre_grid)
        r = 0
        self.median_check = QtWidgets.QCheckBox("Median")
        self.median_check.setChecked(True)  # デフォルトで有効
        self.median_size_spin = QtWidgets.QSpinBox()
        self.median_size_spin.setRange(1, 31)
        self.median_size_spin.setSingleStep(2)
        self.median_size_spin.setValue(int(getattr(self.spot_analyzer, "median_size", 3)))
        self.median_size_spin.setToolTip("Medianカーネルサイズ（奇数推奨）")
        self.median_size_spin.setMaximumWidth(80)
        median_row = QtWidgets.QHBoxLayout()
        median_row.addWidget(self.median_check)
        median_row.addSpacing(8)
        median_k_label = QtWidgets.QLabel("k")
        median_k_label.setToolTip("メディアンフィルタのカーネルサイズ。ノイズを除去しつつスポットを保持します。")
        median_row.addWidget(median_k_label)
        median_row.addWidget(self.median_size_spin)
        pre_grid.addLayout(median_row, r, 0, 1, 3, QtCore.Qt.AlignLeft)
        r += 1

        self.open_check = QtWidgets.QCheckBox("Open")
        self.open_radius_spin = QtWidgets.QSpinBox()
        self.open_radius_spin.setRange(0, 20)
        self.open_radius_spin.setValue(int(getattr(self.spot_analyzer, "open_radius", 1)))
        self.open_radius_spin.setToolTip("Openの半径(px)")
        self.open_radius_spin.setMaximumWidth(80)
        open_row = QtWidgets.QHBoxLayout()
        open_row.addWidget(self.open_check)
        open_row.addSpacing(8)
        open_r_label = QtWidgets.QLabel("r(px)")
        open_r_label.setToolTip("Morphological opening の半径。小さな突起やノイズを除去します。")
        open_row.addWidget(open_r_label)
        open_row.addWidget(self.open_radius_spin)
        pre_grid.addLayout(open_row, r, 0, 1, 3, QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(pre_grid, r)
        form_layout.addWidget(preprocess_group)

        detect_group = QtWidgets.QGroupBox("検出（LoG / DoG / Pre）")
        det_grid = QtWidgets.QGridLayout(detect_group)
        _setup_grid(det_grid)
        r = 0
        detection_mode_label = QtWidgets.QLabel("方式")
        detection_mode_label.setToolTip(
            "ピーク初期位置の検出方法:\n"
            "・DoG: Difference of Gaussians（σ highとσ lowの差分）でエッジ/ブロブ強調\n"
            "・LoG: Laplacian of Gaussian（1つのσ）でブロブ検出\n"
            "・Preprocessed: 前処理画像をそのまま使用（median/morphological opening）"
        )
        det_grid.addWidget(detection_mode_label, r, 0)
        self.detection_mode_combo = QtWidgets.QComboBox()
        self.detection_mode_combo.addItems(["DoG", "LoG", "Preprocessed"])
        self.detection_mode_combo.setCurrentText("LoG")
        self.detection_mode_combo.setMaximumWidth(100)
        self.detection_mode_combo.currentTextChanged.connect(self._update_detection_ui_enabled)
        det_grid.addWidget(self.detection_mode_combo, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # DoG parameters (paired)
        self.bandpass_low_spin = QtWidgets.QDoubleSpinBox()
        self.bandpass_low_spin.setRange(0.1, 20.0)
        self.bandpass_low_spin.setSingleStep(0.1)
        self.bandpass_low_spin.setValue(self.spot_analyzer.bandpass_low_sigma)
        self.bandpass_low_spin.setMaximumWidth(90)
        self.bandpass_high_spin = QtWidgets.QDoubleSpinBox()
        self.bandpass_high_spin.setRange(0.1, 50.0)
        self.bandpass_high_spin.setSingleStep(0.1)
        self.bandpass_high_spin.setValue(self.spot_analyzer.bandpass_high_sigma)
        self.bandpass_high_spin.setMaximumWidth(90)
        dog_row = QtWidgets.QHBoxLayout()
        dog_sigma_low_label = QtWidgets.QLabel("σ low")
        dog_sigma_low_label.setToolTip("DoG低周波側のガウシアンぼかし半径。小さいスポットに対応します。")
        dog_row.addWidget(dog_sigma_low_label)
        dog_row.addWidget(self.bandpass_low_spin)
        dog_row.addSpacing(8)
        dog_sigma_high_label = QtWidgets.QLabel("σ high")
        dog_sigma_high_label.setToolTip("DoG高周波側のガウシアンぼかし半径。大きい構造を除去します。")
        dog_row.addWidget(dog_sigma_high_label)
        dog_row.addWidget(self.bandpass_high_spin)
        det_grid.addLayout(dog_row, r, 0, 1, 3, QtCore.Qt.AlignLeft)
        r += 1

        # LoG parameter
        self.log_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.log_sigma_spin.setRange(0.1, 50.0)
        self.log_sigma_spin.setSingleStep(0.1)
        self.log_sigma_spin.setValue(float(getattr(self.spot_analyzer, "log_sigma", 1.6)))
        self.log_sigma_spin.setMaximumWidth(90)
        log_row = QtWidgets.QHBoxLayout()
        log_sigma_label = QtWidgets.QLabel("σ")
        log_sigma_label.setToolTip("LoGのガウシアンぼかし半径。検出したいスポットのサイズに合わせて調整します。")
        log_row.addWidget(log_sigma_label)
        log_row.addWidget(self.log_sigma_spin)
        det_grid.addLayout(log_row, r, 0, 1, 3, QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(det_grid, r)

        form_layout.addWidget(detect_group)

        # NOTE: This group contains ONLY initializer (seed detection) related controls.
        fit_group = QtWidgets.QGroupBox("初期化（初期位置）")
        fit_grid = QtWidgets.QGridLayout(fit_group)
        _setup_grid(fit_grid)
        r = 0

        # ROI margin (used in initial seed selection AND result filtering)
        self.margin_spin = QtWidgets.QSpinBox()
        self.margin_spin.setRange(0, 100)
        self.margin_spin.setValue(self.spot_analyzer.margin)
        self.margin_spin.setMaximumWidth(70)
        self.margin_spin.setToolTip("ROI境界からの除外距離 (px)")
        margin_label = QtWidgets.QLabel("ROIマージン (px)")
        margin_label.setToolTip(
            "ROI境界からの除外距離（ピクセル）:\n"
            "初期候補の選択と、最終結果の除外の両方に使用されます。\n"
            "ROI境界からこの距離以内にあるピークを除外します。"
        )
        fit_grid.addWidget(margin_label, r, 0)
        fit_grid.addWidget(self.margin_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Peak spacing (used in Peak seed NMS; also a sensible global spacing hint)
        self.peak_min_distance_spin = QtWidgets.QSpinBox()
        self.peak_min_distance_spin.setRange(1, 50)
        self.peak_min_distance_spin.setValue(self.spot_analyzer.peak_min_distance)
        self.peak_min_distance_spin.setMaximumWidth(70)
        peak_min_distance_label = QtWidgets.QLabel("ピーク間隔 (px)")
        peak_min_distance_label.setToolTip(
            "ピーク間の最小距離（ピクセル）:\n"
            "Peak系初期候補では、この距離以内の候補は最も強いもの1つだけが残されます（NMS）。\n"
            "近接したピークの重複検出を防ぎます。"
        )
        fit_grid.addWidget(peak_min_distance_label, r, 0)
        fit_grid.addWidget(self.peak_min_distance_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # initializer mode
        init_mode_label = QtWidgets.QLabel("初期候補")
        init_mode_label.setToolTip(
            "ガウスフィットの初期ピーク候補生成方法:\n"
            "・Watershed: 適応的h-maximaで複雑な画像に強い（推奨）\n"
            "・Blob DoH: 円形スポット検出に最適、計算効率が良い\n"
            "・Peak: シンプルな局所最大値検出"
        )
        fit_grid.addWidget(init_mode_label, r, 0)
        self.init_mode_combo = QtWidgets.QComboBox()
        self.init_mode_combo.addItems([
            "Watershed (推奨)",
            "Blob DoH (高速)",
            "Peak"
        ])
        self.init_mode_combo.setCurrentText("Peak")
        self.init_mode_combo.setMaximumWidth(180)
        self.init_mode_combo.setToolTip(
            "ガウスフィットの初期ピーク候補生成方法:\n\n"
            "• Watershed (推奨)\n"
            "  適応的h-maximaで複雑な画像に強い\n\n"
            "• Blob DoH (高速)\n"
            "  円形スポット検出に最適、計算効率が良い\n\n"
            "• Peak\n"
            "  シンプルな局所最大値検出"
        )
        self.init_mode_combo.currentTextChanged.connect(self._update_init_mode_ui_enabled)
        fit_grid.addWidget(self.init_mode_combo, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Subpixel refinement option
        self.subpixel_check = QtWidgets.QCheckBox("サブピクセル精度")
        self.subpixel_check.setChecked(False)
        self.subpixel_check.setToolTip("Peak モードでサブピクセル精度の位置補正を行います。")
        fit_grid.addWidget(self.subpixel_check, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.watershed_h_rel_spin = QtWidgets.QDoubleSpinBox()
        self.watershed_h_rel_spin.setRange(0.0, 1.0)
        self.watershed_h_rel_spin.setSingleStep(0.01)
        self.watershed_h_rel_spin.setDecimals(3)
        self.watershed_h_rel_spin.setValue(float(getattr(self.spot_analyzer, "watershed_h_rel", 0.05)))
        self.watershed_h_rel_spin.setMaximumWidth(90)
        self.watershed_h_rel_spin.setToolTip("h-maximaの相対高さ。大きいほど過剰分割を抑えます（0〜1）。")
        ws_h_label = QtWidgets.QLabel("WS h(rel)")
        ws_h_label.setToolTip(
            "Watershed h-maximaの相対高さ（0〜1）:\n"
            "画像の強度範囲に対する比率でノイズ抑制の強さを設定します。\n"
            "大きいほど小さな極大値を無視し、過剰分割を抑えます。\n"
            "適応的h-maximaが有効な場合は、ノイズレベルも考慮されます。"
        )
        fit_grid.addWidget(ws_h_label, r, 0)
        fit_grid.addWidget(self.watershed_h_rel_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.watershed_adaptive_h_check = QtWidgets.QCheckBox("適応的h-maxima")
        self.watershed_adaptive_h_check.setChecked(bool(getattr(self.spot_analyzer, "watershed_adaptive_h", True)))
        self.watershed_adaptive_h_check.setToolTip("ノイズレベルに基づいてh値を自動調整します。")
        fit_grid.addWidget(self.watershed_adaptive_h_check, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # blob_doh params
        self.blob_doh_min_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.blob_doh_min_sigma_spin.setRange(0.2, 200.0)
        self.blob_doh_min_sigma_spin.setSingleStep(0.5)
        self.blob_doh_min_sigma_spin.setValue(float(getattr(self.spot_analyzer, "blob_doh_min_sigma", 3.0)))
        self.blob_doh_min_sigma_spin.setMaximumWidth(90)
        self.blob_doh_max_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.blob_doh_max_sigma_spin.setRange(0.2, 400.0)
        self.blob_doh_max_sigma_spin.setSingleStep(0.5)
        self.blob_doh_max_sigma_spin.setValue(float(getattr(self.spot_analyzer, "blob_doh_max_sigma", 20.0)))
        self.blob_doh_max_sigma_spin.setMaximumWidth(90)
        blob_doh_sigma_row = QtWidgets.QHBoxLayout()
        doh_min_sigma_label = QtWidgets.QLabel("minσ")
        doh_min_sigma_label.setToolTip("DoH (Determinant of Hessian) 検出の最小σ")
        blob_doh_sigma_row.addWidget(doh_min_sigma_label)
        blob_doh_sigma_row.addWidget(self.blob_doh_min_sigma_spin)
        blob_doh_sigma_row.addSpacing(8)
        doh_max_sigma_label = QtWidgets.QLabel("maxσ")
        doh_max_sigma_label.setToolTip("DoH (Determinant of Hessian) 検出の最大σ")
        blob_doh_sigma_row.addWidget(doh_max_sigma_label)
        blob_doh_sigma_row.addWidget(self.blob_doh_max_sigma_spin)
        doh_sigma_range_label = QtWidgets.QLabel("DoH σ範囲")
        doh_sigma_range_label.setToolTip(
            "DoH (Determinant of Hessian) blob検出のσ範囲:\n"
            "Hessian行列の行列式を用いた高速blob検出方法です。\n"
            "LoGよりも計算効率が良く、円形スポット検出に適しています。"
        )
        fit_grid.addWidget(doh_sigma_range_label, r, 0)
        fit_grid.addLayout(blob_doh_sigma_row, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.blob_doh_threshold_rel_spin = QtWidgets.QDoubleSpinBox()
        self.blob_doh_threshold_rel_spin.setRange(0.0, 1.0)
        self.blob_doh_threshold_rel_spin.setSingleStep(0.001)
        self.blob_doh_threshold_rel_spin.setDecimals(3)
        self.blob_doh_threshold_rel_spin.setValue(float(getattr(self.spot_analyzer, "blob_doh_threshold_rel", 0.01)))
        self.blob_doh_threshold_rel_spin.setMaximumWidth(90)
        doh_thr_label = QtWidgets.QLabel("DoH thr(rel)")
        doh_thr_label.setToolTip(
            "DoH検出の相対閾値（0〜1）:\n"
            "DoH応答の最大値に対する比率で閾値を設定します。\n"
            "低い値ほど多くのblobを検出します。"
        )
        fit_grid.addWidget(doh_thr_label, r, 0)
        fit_grid.addWidget(self.blob_doh_threshold_rel_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.localmax_thr_spin = QtWidgets.QDoubleSpinBox()
        self.localmax_thr_spin.setRange(0.0, 1.0)
        self.localmax_thr_spin.setSingleStep(0.01)
        self.localmax_thr_spin.setDecimals(3)
        self.localmax_thr_spin.setValue(float(getattr(self.spot_analyzer, "localmax_threshold_rel", 0.05)))
        self.localmax_thr_spin.setMaximumWidth(90)
        self.localmax_thr_spin.setToolTip("peak_local_max の閾値。検出画像（LoG/DoG）最大値に対する相対値（0〜1）。")
        localmax_thr_label = QtWidgets.QLabel("localmax thr(rel)")
        localmax_thr_label.setToolTip(
            "検出画像（LoG/DoG）でのピーク検出相対閾値（0〜1）:\n"
            "検出画像の最大値に対する比率で閾値を設定します。\n"
            "例: 最大値100、thr=0.05 → 閾値5以上のピークを検出"
        )
        # NOTE: Peak初期化は現在NMSベースであり、この閾値は実質使われません。
        # 後方互換（保存/復元・内部参照）のためウィジェットは残しつつUIから非表示にします。
        localmax_thr_label.setVisible(False)
        self.localmax_thr_spin.setVisible(False)
        fit_grid.addWidget(localmax_thr_label, r, 0)
        fit_grid.addWidget(self.localmax_thr_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.localmax_snr_spin = QtWidgets.QDoubleSpinBox()
        self.localmax_snr_spin.setRange(0.0, 50.0)
        self.localmax_snr_spin.setSingleStep(0.1)
        self.localmax_snr_spin.setDecimals(2)
        self.localmax_snr_spin.setValue(float(getattr(self.spot_analyzer, "localmax_threshold_snr", 0.0)))
        self.localmax_snr_spin.setMaximumWidth(90)
        self.localmax_snr_spin.setToolTip("検出画像のノイズ(MAD)基準の閾値。0で無効。")
        localmax_snr_label = QtWidgets.QLabel("localmax thr(SNR)")
        localmax_snr_label.setToolTip(
            "検出画像のノイズ基準SNR閾値:\n"
            "閾値 = thr(SNR) × ノイズレベル(MAD)\n"
            "thr(rel)と併用時は大きい方が採用されます。\n"
            "0で無効化。ノイズに強いピークのみ検出したい場合に設定。"
        )
        localmax_snr_label.setVisible(False)
        self.localmax_snr_spin.setVisible(False)
        fit_grid.addWidget(localmax_snr_label, r, 0)
        fit_grid.addWidget(self.localmax_snr_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(fit_grid, r)

        form_layout.addWidget(fit_group)

        # --- Fit (Gaussian model) ---
        fit_model_group = QtWidgets.QGroupBox("フィット（ガウスモデル）")
        fit_model_grid = QtWidgets.QGridLayout(fit_model_group)
        _setup_grid(fit_model_grid)
        r = 0

        self.initial_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.initial_sigma_spin.setRange(0.1, 20.0)
        self.initial_sigma_spin.setSingleStep(0.1)
        self.initial_sigma_spin.setValue(self.spot_analyzer.initial_sigma)
        self.initial_sigma_spin.setMaximumWidth(90)
        initial_sigma_label = QtWidgets.QLabel("初期σ")
        initial_sigma_label.setToolTip(
            "ガウシアンフィッティングの初期σ値:\n"
            "ピーク候補のσ初期値として使用されます。\n"
            "スポットの典型的なサイズに合わせて設定してください。"
        )
        fit_model_grid.addWidget(initial_sigma_label, r, 0)
        fit_model_grid.addWidget(self.initial_sigma_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # sigma bounds (paired)
        self.sigma_min_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_min_spin.setRange(0.05, 50.0)
        self.sigma_min_spin.setSingleStep(0.05)
        self.sigma_min_spin.setValue(self.spot_analyzer.sigma_bounds[0])
        self.sigma_min_spin.setMaximumWidth(90)
        self.sigma_max_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_max_spin.setRange(0.1, 80.0)
        self.sigma_max_spin.setSingleStep(0.1)
        self.sigma_max_spin.setValue(self.spot_analyzer.sigma_bounds[1])
        self.sigma_max_spin.setMaximumWidth(90)
        sigma_row = QtWidgets.QHBoxLayout()
        sigma_lower_label = QtWidgets.QLabel("下限")
        sigma_lower_label.setToolTip("フィッティング時のσ最小値。これ以下に収束しません。")
        sigma_row.addWidget(sigma_lower_label)
        sigma_row.addWidget(self.sigma_min_spin)
        sigma_row.addSpacing(8)
        sigma_upper_label = QtWidgets.QLabel("上限")
        sigma_upper_label.setToolTip("フィッティング時のσ最大値。これ以上に収束しません。")
        sigma_row.addWidget(sigma_upper_label)
        sigma_row.addWidget(self.sigma_max_spin)
        sigma_bounds_label = QtWidgets.QLabel("σ")
        sigma_bounds_label.setToolTip(
            "ガウシアンフィッティング時のσ制約範囲:\n"
            "フィッティング中にσがこの範囲内に制限されます。\n"
            "スポットサイズの変動範囲に合わせて設定してください。"
        )
        fit_model_grid.addWidget(sigma_bounds_label, r, 0)
        fit_model_grid.addLayout(sigma_row, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(fit_model_grid, r)

        form_layout.addWidget(fit_model_group)

        filter_group = QtWidgets.QGroupBox("結果フィルタ")
        filter_grid = QtWidgets.QGridLayout(filter_group)
        _setup_grid(filter_grid)
        r = 0

        # --- Post-processing / filters ---
        self.snr_spin = QtWidgets.QDoubleSpinBox()
        self.snr_spin.setRange(0.1, 50.0)
        self.snr_spin.setSingleStep(0.1)
        self.snr_spin.setValue(self.spot_analyzer.snr_threshold)
        self.snr_spin.setMaximumWidth(90)
        snr_label = QtWidgets.QLabel("S/N閾値（出力フィルタ）")
        snr_label.setToolTip(
            "最終結果に含めるピークのS/N比（Signal-to-Noise Ratio）最小値:\n"
            "S/N = 振幅 / ノイズレベル\n"
            "この値以下のピークは最終結果から除外されます。\n"
            "ノイズの多い画像では高めに設定してください。"
        )
        filter_grid.addWidget(snr_label, r, 0)
        filter_grid.addWidget(self.snr_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # precheck (fit-result validation on roi_pre)
        self.precheck_radius_spin = QtWidgets.QSpinBox()
        self.precheck_radius_spin.setRange(0, 20)
        self.precheck_radius_spin.setValue(int(getattr(self.spot_analyzer, "precheck_radius_px", 2)))
        self.precheck_radius_spin.setMaximumWidth(70)
        self.precheck_radius_spin.setToolTip("元画像(ROI)の局所ピーク判定半径。0で無効。")
        precheck_r_label = QtWidgets.QLabel("precheck r(px)")
        precheck_r_label.setToolTip(
            "元画像での局所最大値判定半径（ピクセル）:\n"
            "各ピーク候補の周囲r(px)の窓内で最大値かチェック。\n"
            "窓内最大値より10%以上小さい場合、そのピークを除外。\n"
            "0で無効化。検出画像で見つかったが実際には局所最大でないピークを除外。"
        )
        filter_grid.addWidget(precheck_r_label, r, 0)
        filter_grid.addWidget(self.precheck_radius_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.precheck_kmad_spin = QtWidgets.QDoubleSpinBox()
        self.precheck_kmad_spin.setRange(0.0, 10.0)
        self.precheck_kmad_spin.setSingleStep(0.1)
        self.precheck_kmad_spin.setDecimals(2)
        self.precheck_kmad_spin.setValue(float(getattr(self.spot_analyzer, "precheck_kmad", 1.0)))
        self.precheck_kmad_spin.setMaximumWidth(90)
        self.precheck_kmad_spin.setToolTip("元画像(ROI)で中央値+K*MADを下回るピークを除外。0で無効。")
        precheck_kmad_label = QtWidgets.QLabel("precheck K(MAD)")
        precheck_kmad_label.setToolTip(
            "元画像のノイズ基準閾値係数:\n"
            "各ピーク周囲の窓で 中央値 + K×MAD を計算。\n"
            "ピーク強度がこの値を下回る場合、除外します。\n"
            "MAD = Median Absolute Deviation（ロバストな分散推定）\n"
            "0で無効化。ノイズレベル以上の強度を持つピークのみ残したい場合に設定。"
        )
        filter_grid.addWidget(precheck_kmad_label, r, 0)
        filter_grid.addWidget(self.precheck_kmad_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # DBSCAN params (spatial outlier filtering)
        self.dbscan_enabled_check = QtWidgets.QCheckBox("DBSCAN 外れ値除去")
        self.dbscan_enabled_check.setChecked(bool(getattr(self.spot_analyzer, "dbscan_enabled", False)))
        self.dbscan_enabled_check.setToolTip("空間的に孤立したピークをノイズとして除去します。")
        filter_grid.addWidget(self.dbscan_enabled_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.dbscan_eps_spin = QtWidgets.QDoubleSpinBox()
        self.dbscan_eps_spin.setRange(0.5, 50.0)
        self.dbscan_eps_spin.setSingleStep(0.5)
        self.dbscan_eps_spin.setValue(float(getattr(self.spot_analyzer, "dbscan_eps", 5.0)))
        self.dbscan_eps_spin.setMaximumWidth(90)
        self.dbscan_eps_spin.setToolTip("DBSCANの近傍半径（ピクセル）。")
        dbscan_eps_label = QtWidgets.QLabel("DBSCAN eps")
        dbscan_eps_label.setToolTip(
            "DBSCAN クラスタリングの近傍半径（ピクセル）:\n"
            "Multiscale (LoG+DBSCAN) 初期化モードで使用されます。\n"
            "この距離以内の点を同一クラスタとみなし、重複検出を除去します。\n"
            "ピーク間の最小距離程度に設定すると良いでしょう。"
        )
        filter_grid.addWidget(dbscan_eps_label, r, 0)
        filter_grid.addWidget(self.dbscan_eps_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.min_amp_spin = QtWidgets.QDoubleSpinBox()
        self.min_amp_spin.setRange(0.0, 1000.0)
        self.min_amp_spin.setSingleStep(0.1)
        self.min_amp_spin.setValue(self.spot_analyzer.min_amplitude)
        self.min_amp_spin.setMaximumWidth(90)
        min_amp_label = QtWidgets.QLabel("最小振幅")
        min_amp_label.setToolTip(
            "最終結果に含めるピークの最小振幅:\n"
            "フィッティングされたガウス関数の振幅（高さ）がこの値以下の\n"
            "ピークは最終結果から除外されます。"
        )
        filter_grid.addWidget(min_amp_label, r, 0)
        filter_grid.addWidget(self.min_amp_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.min_sigma_result_spin = QtWidgets.QDoubleSpinBox()
        self.min_sigma_result_spin.setRange(0.0, 50.0)
        self.min_sigma_result_spin.setSingleStep(0.1)
        self.min_sigma_result_spin.setValue(self.spot_analyzer.min_sigma_result)
        self.min_sigma_result_spin.setMaximumWidth(90)
        min_sigma_result_label = QtWidgets.QLabel("最小σ (結果)")
        min_sigma_result_label.setToolTip(
            "最終結果に含めるピークの最小σ:\n"
            "フィッティング後のσがこの値以下のピークを除外します。\n"
            "極端に小さい（点状の）ピークをノイズとして除外したい場合に設定します。"
        )
        filter_grid.addWidget(min_sigma_result_label, r, 0)
        filter_grid.addWidget(self.min_sigma_result_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1
        _add_right_spacer(filter_grid, r)

        form_layout.addWidget(filter_group)

        view_group = QtWidgets.QGroupBox("表示 / 記録")
        view_grid = QtWidgets.QGridLayout(view_group)
        _setup_grid(view_grid)
        r = 0
        self.spot_radius_spin = QtWidgets.QSpinBox()
        self.spot_radius_spin.setRange(1, 200)
        self.spot_radius_spin.setValue(4)
        self.spot_radius_spin.setToolTip("スポット中心から半径r(px)円内の平均高さを記録します。")
        self.spot_radius_spin.setMaximumWidth(80)
        self.spot_radius_spin.valueChanged.connect(self._on_spot_radius_changed)
        spot_radius_label = QtWidgets.QLabel("Spot半径 (px)")
        spot_radius_label.setToolTip(
            "スポット測定半径（ピクセル）:\n"
            "検出されたピーク中心から半径r(px)の円内の平均高さを計算し記録します。\n"
            "この値は表示にも使用され、ピークマーカーの円の大きさになります。"
        )
        view_grid.addWidget(spot_radius_label, r, 0)
        view_grid.addWidget(self.spot_radius_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.show_roi_spots_check = QtWidgets.QCheckBox("ROI画像にスポット表示")
        self.show_roi_spots_check.setChecked(True)
        self.show_roi_spots_check.setToolTip("下段のROI画像（前処理ROI/LoG・DoG ROI）にスポットを重ねて表示します。")
        self.show_roi_spots_check.toggled.connect(self._on_any_ui_changed)
        view_grid.addWidget(self.show_roi_spots_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.show_det_spots_check = QtWidgets.QCheckBox("検出画像にスポット表示")
        self.show_det_spots_check.setChecked(True)
        self.show_det_spots_check.setToolTip("下段の検出画像（LoG/DoG）にスポットを重ねて表示します。")
        self.show_det_spots_check.toggled.connect(self._on_any_ui_changed)
        view_grid.addWidget(self.show_det_spots_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Debug: show fit-result (final) peaks on ROI(preprocessed)
        self.show_fit_spots_on_roi_pre_check = QtWidgets.QCheckBox("ROI(pre)にフィット後ピーク表示（デバッグ）")
        self.show_fit_spots_on_roi_pre_check.setChecked(False)
        self.show_fit_spots_on_roi_pre_check.setToolTip(
            "左下のROI(preprocessed)に、フィット後の最終ピーク位置（final peaks）を重ねて表示します。\n"
            "LoG/DoGで得た初期位置とフィット結果の整合チェック用です。"
        )
        self.show_fit_spots_on_roi_pre_check.toggled.connect(self._on_any_ui_changed)
        view_grid.addWidget(self.show_fit_spots_on_roi_pre_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        # Snap-to-local-max (final peak coordinates)
        self.snap_enabled_check = QtWidgets.QCheckBox("最終ピークを局所最大へスナップ")
        self.snap_enabled_check.setChecked(bool(getattr(self.spot_analyzer, "snap_enabled", False)))
        self.snap_enabled_check.setToolTip("最終結果のピーク座標を、検出画像（LoG/DoG）上の局所最大へ寄せます。")
        view_grid.addWidget(self.snap_enabled_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.snap_radius_spin = QtWidgets.QSpinBox()
        self.snap_radius_spin.setRange(0, 50)
        self.snap_radius_spin.setValue(int(getattr(self.spot_analyzer, "snap_radius", 2)))
        self.snap_radius_spin.setMaximumWidth(80)
        self.snap_radius_spin.setToolTip("スナップ探索半径（px）。0でスナップしません。")
        snap_radius_label = QtWidgets.QLabel("スナップ半径 (px)")
        snap_radius_label.setToolTip(
            "スナップ探索半径（ピクセル）:\n"
            "最終ピークを検出画像（LoG/DoG）の局所最大へ移動（スナップ）する際、\n"
            "この半径以内で最大値を探索します。\n"
            "0に設定するとスナップを無効化します。"
        )
        view_grid.addWidget(snap_radius_label, r, 0)
        view_grid.addWidget(self.snap_radius_spin, r, 1, alignment=QtCore.Qt.AlignLeft)
        r += 1

        self.snap_refit_check = QtWidgets.QCheckBox("スナップ後に再フィット(1回)")
        self.snap_refit_check.setChecked(bool(getattr(self.spot_analyzer, "snap_refit_enabled", False)))
        self.snap_refit_check.setToolTip("スナップ位置を初期値にして、同じピーク数で1回だけ再フィットします（遅くなることがあります）。")
        view_grid.addWidget(self.snap_refit_check, r, 0, 1, 3, alignment=QtCore.Qt.AlignLeft)
        r += 1

        _add_right_spacer(view_grid, r)

        form_layout.addWidget(view_group)

        # --- Status / output ---
        self.roi_status_label = QtWidgets.QLabel("ROI未選択")
        form_layout.addWidget(self.roi_status_label)

        self.selection_label = QtWidgets.QLabel("")
        form_layout.addWidget(self.selection_label)

        self.output = QtWidgets.QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        try:
            fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            self.output.setFont(fixed_font)
        except Exception:
            pass
        try:
            self.output.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        except Exception:
            pass
        form_layout.addWidget(self.output)

        self.export_btn = QtWidgets.QPushButton("CSVエクスポート")
        self.export_btn.clicked.connect(self.export_spots_csv)
        self.export_btn.setEnabled(False)
        form_layout.addWidget(self.export_btn)

        self.import_resume_btn = QtWidgets.QPushButton("CSV読み込み→復元")
        self.import_resume_btn.setToolTip("保存したSpot CSVを読み込み、解析途中のスポット表示状態を復元します。")
        self.import_resume_btn.clicked.connect(self.import_csv_restore)
        form_layout.addWidget(self.import_resume_btn)

        self.reset_btn = QtWidgets.QPushButton("解析結果をリセット")
        self.reset_btn.setToolTip("スポット結果・ROI・表示を一括でクリアします（手動）。")
        self.reset_btn.clicked.connect(self._reset_analysis_results)
        form_layout.addWidget(self.reset_btn)

        self.edit_help_label = QtWidgets.QLabel(
            "Spot編集: Ctrl/⌘+ドラッグ=移動, Shift+クリック=追加, Alt(Option)+クリック=削除"
        )
        self.edit_help_label.setStyleSheet("color: #666; font-size: 11px;")
        form_layout.addWidget(self.edit_help_label)

        # --- Add form_widget to content layout (left-aligned) ---
        layout.addWidget(form_widget, alignment=QtCore.Qt.AlignLeft)
        layout.addStretch(1)

        # --- Live updates ---
        # Any UI changes should redraw immediately; analysis-impacting widgets also trigger debounced reanalysis.
        for w in (
            # analysis-impacting controls
            self.refit_shift_spin,
            self.median_check,
            self.open_check,
            self.detection_mode_combo,
            self.bandpass_low_spin,
            self.bandpass_high_spin,
            self.log_sigma_spin,
            self.median_size_spin,
            self.open_radius_spin,
            self.margin_spin,
            self.min_amp_spin,
            self.min_sigma_result_spin,
            self.criterion_combo,
            self.min_peaks_spin,
            self.max_peaks_spin,
            self.init_mode_combo,
            self.subpixel_check,
            self.watershed_h_rel_spin,
            self.watershed_adaptive_h_check,
            self.snr_spin,
            self.peak_min_distance_spin,
            self.localmax_thr_spin,
            self.localmax_snr_spin,
            self.precheck_radius_spin,
            self.precheck_kmad_spin,
            self.blob_doh_min_sigma_spin,
            self.blob_doh_max_sigma_spin,
            self.blob_doh_threshold_rel_spin,
            self.dbscan_enabled_check,
            self.dbscan_eps_spin,
            self.initial_sigma_spin,
            self.sigma_min_spin,
            self.sigma_max_spin,
            self.snap_enabled_check,
            self.snap_radius_spin,
            self.snap_refit_check,
        ):
            try:
                if isinstance(w, QtWidgets.QAbstractButton):
                    w.toggled.connect(self._on_analysis_param_changed)
                elif hasattr(w, "valueChanged"):
                    w.valueChanged.connect(self._on_analysis_param_changed)
                elif hasattr(w, "currentTextChanged"):
                    w.currentTextChanged.connect(self._on_analysis_param_changed)
            except Exception:
                pass

        self._update_detection_ui_enabled()
        self._update_init_mode_ui_enabled()
        self._sync_run_buttons_enabled()

    def _update_detection_ui_enabled(self, *_args) -> None:
        mode = (self.detection_mode_combo.currentText() or "DoG").strip().lower()
        is_log = mode == "log"
        is_pre = mode in ("pre", "preprocessed")
        # DoG widgets
        for w in (self.bandpass_low_spin, self.bandpass_high_spin):
            w.setEnabled(not is_log and not is_pre)
        # LoG widget
        self.log_sigma_spin.setEnabled(is_log and not is_pre)
        # NOTE: redraw/reanalysis is handled by the parameter-change handler

    def _update_init_mode_ui_enabled(self, *_args) -> None:
        mode = (self.init_mode_combo.currentText() or "Watershed (推奨)").strip().lower()
        is_doh = "doh" in mode
        is_ws = "watershed" in mode or "water" in mode
        is_peak = "peak" in mode

        # DoH parameters
        for w in (
            getattr(self, "blob_doh_min_sigma_spin", None),
            getattr(self, "blob_doh_max_sigma_spin", None),
            getattr(self, "blob_doh_threshold_rel_spin", None),
        ):
            try:
                if w is not None:
                    w.setEnabled(bool(is_doh))
            except Exception:
                pass

        # Watershed parameters
        for w in (
            getattr(self, "watershed_h_rel_spin", None),
            getattr(self, "watershed_adaptive_h_check", None),
        ):
            try:
                if w is not None:
                    w.setEnabled(bool(is_ws))
            except Exception:
                pass

        # Peak parameters
        try:
            if getattr(self, "localmax_thr_spin", None) is not None:
                self.localmax_thr_spin.setEnabled(bool(is_peak))
            if getattr(self, "localmax_snr_spin", None) is not None:
                self.localmax_snr_spin.setEnabled(bool(is_peak))
            if getattr(self, "subpixel_check", None) is not None:
                self.subpixel_check.setEnabled(bool(is_peak))
        except Exception:
            pass

    def _sync_run_buttons_enabled(self) -> None:
        has_roi = self.manual_roi is not None
        try:
            self.run_btn.setEnabled(bool(has_roi))
        except Exception:
            pass
        try:
            allow_all = bool(has_roi) and bool(self.run_all_enable_check.isChecked())
            self.run_all_btn.setEnabled(allow_all)
        except Exception:
            pass
        try:
            # manual refit requires existing spots on the current frame
            fi = self._get_current_frame_index()
            has_spots = bool(self.spots_by_frame.get(fi))
            self.refit_manual_btn.setEnabled(bool(has_roi) and has_spots)
        except Exception:
            try:
                self.refit_manual_btn.setEnabled(False)
            except Exception:
                pass

    def _on_any_ui_changed(self, *_args) -> None:
        """UI変更時の即時更新（表示のみ）。"""
        self._refresh_overlay()

    def _on_analysis_param_changed(self, *_args) -> None:
        """
        解析パラメータ変更時:
        - 表示（ROI画像など）は即時更新
        - 解析（spots更新）はデバウンスして実行
        """
        self._refresh_overlay()
        self._schedule_reanalysis()

    # Backward-compat: keep old handler name used by some connections.
    def _on_params_changed(self, *_args) -> None:
        self._on_analysis_param_changed(*_args)

    def _schedule_reanalysis(self) -> None:
        if self.manual_roi is None:
            return
        try:
            # debounce: spinbox連続操作で解析が連打されないようにする
            self._reanalysis_timer.start(250)
        except Exception:
            pass

    def _analysis_signature(self, frame_index: int, roi_info: Optional[Dict[str, float]]) -> Tuple:
        def _rf(v: float) -> float:
            try:
                return round(float(v), 6)
            except Exception:
                return float("nan")

        if roi_info is None:
            roi_sig: Tuple = ("none",)
        else:
            shape = str(roi_info.get("shape", "Rectangle"))
            roi_sig = (
                shape,
                _rf(roi_info.get("x0", 0.0)),
                _rf(roi_info.get("y0", 0.0)),
                _rf(roi_info.get("w", 0.0)),
                _rf(roi_info.get("h", 0.0)),
                _rf(roi_info.get("cx", 0.0)) if shape == "Ellipse" else None,
                _rf(roi_info.get("cy", 0.0)) if shape == "Ellipse" else None,
                _rf(roi_info.get("rx", 0.0)) if shape == "Ellipse" else None,
                _rf(roi_info.get("ry", 0.0)) if shape == "Ellipse" else None,
            )

        # UI -> params signature
        crit = (self.criterion_combo.currentText() or "AIC").strip().lower()
        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())

        mode_ui = (self.detection_mode_combo.currentText() or "DoG").strip().lower()
        if mode_ui in ("pre", "preprocessed"):
            det_mode = "pre"
            det_sig = ("pre",)
        elif mode_ui == "log":
            det_mode = "log"
            det_sig = ("log", _rf(self.log_sigma_spin.value()))
        else:
            det_mode = "dog"
            det_sig = ("dog", _rf(self.bandpass_low_spin.value()), _rf(self.bandpass_high_spin.value()))

        params_sig = (
            crit,
            min_peaks,
            max_peaks,
            det_sig,
            (self.init_mode_combo.currentText() or "blob_log").strip().lower(),
            bool(getattr(self, "subpixel_check", None) is not None and self.subpixel_check.isChecked()),
            _rf(self.watershed_h_rel_spin.value()),
            bool(self.watershed_adaptive_h_check.isChecked()),
            (_rf(self.blob_doh_min_sigma_spin.value()), _rf(self.blob_doh_max_sigma_spin.value())),
            _rf(self.blob_doh_threshold_rel_spin.value()),
            bool(self.dbscan_enabled_check.isChecked()),
            _rf(self.dbscan_eps_spin.value()),
            bool(self.median_check.isChecked()),
            int(self.median_size_spin.value()),
            bool(self.open_check.isChecked()),
            int(self.open_radius_spin.value()),
            int(self.peak_min_distance_spin.value()),
            _rf(self.localmax_thr_spin.value()),
            _rf(self.localmax_snr_spin.value()),
            int(self.precheck_radius_spin.value()),
            _rf(self.precheck_kmad_spin.value()),
            _rf(self.initial_sigma_spin.value()),
            (_rf(self.sigma_min_spin.value()), _rf(self.sigma_max_spin.value())),
            _rf(self.snr_spin.value()),
            # results filters
            int(self.margin_spin.value()),
            _rf(self.min_amp_spin.value()),
            _rf(self.min_sigma_result_spin.value()),
            # final peak snapping
            bool(self.snap_enabled_check.isChecked()),
            int(self.snap_radius_spin.value()),
            bool(self.snap_refit_check.isChecked()),
            # constrained manual refit
            int(getattr(self, "refit_shift_spin", None).value()) if getattr(self, "refit_shift_spin", None) is not None else 3,
        )
        return (int(frame_index), roi_sig, params_sig)

    def _reanalyze_current_frame_debounced(self) -> None:
        if self._auto_busy:
            return
        if self.manual_roi is None:
            return
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            return
        if not self._apply_ui_to_analyzer(show_errors=False):
            return

        frame_index = self._get_current_frame_index()
        roi_info = self._current_roi_overlay()
        sig = self._analysis_signature(frame_index, roi_info)
        if self._analysis_signature_by_frame.get(frame_index) == sig:
            return

        criterion = self.criterion_combo.currentText().lower()
        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())
        if max_peaks < min_peaks:
            return

        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)
        self._auto_busy = True
        try:
            self._analyze_current_frame(
                frame,
                criterion,
                center_override,
                roi_size_override,
                roi_mask,
                roi_bounds,
                min_peaks,
                max_peaks,
                show_errors=False,
            )
        finally:
            self._auto_busy = False

    def _apply_ui_to_analyzer(self, show_errors: bool) -> bool:
        # preprocess
        self.spot_analyzer.median_enabled = bool(self.median_check.isChecked())
        self.spot_analyzer.median_size = int(self.median_size_spin.value())
        self.spot_analyzer.open_enabled = bool(self.open_check.isChecked())
        self.spot_analyzer.open_radius = int(self.open_radius_spin.value())

        # detection mode
        mode = (self.detection_mode_combo.currentText() or "DoG").strip().lower()
        if mode in ("pre", "preprocessed"):
            self.spot_analyzer.detection_mode = "pre"
        elif mode == "log":
            self.spot_analyzer.detection_mode = "log"
        else:
            self.spot_analyzer.detection_mode = "dog"

        if self.spot_analyzer.detection_mode == "dog":
            low_sigma = float(self.bandpass_low_spin.value())
            high_sigma = float(self.bandpass_high_spin.value())
            if high_sigma <= low_sigma:
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "DoG: σ high は σ low より大きくしてください。")
                return False
            self.spot_analyzer.bandpass_low_sigma = low_sigma
            self.spot_analyzer.bandpass_high_sigma = high_sigma
        elif self.spot_analyzer.detection_mode == "log":
            log_sigma = float(self.log_sigma_spin.value())
            if log_sigma <= 0:
                if show_errors:
                    QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "LoG: σ は正の値にしてください。")
                return False
            self.spot_analyzer.log_sigma = log_sigma

        # fit params
        init_ui = (self.init_mode_combo.currentText() or "Watershed (推奨)").strip().lower()
        if "watershed" in init_ui or "water" in init_ui:
            self.spot_analyzer.init_mode = "watershed"
        elif "doh" in init_ui or "blob doh" in init_ui:
            self.spot_analyzer.init_mode = "blob_doh"
        elif "peak" in init_ui:
            # Check subpixel option
            if hasattr(self, "subpixel_check") and self.subpixel_check.isChecked():
                self.spot_analyzer.init_mode = "peak_subpixel"
            else:
                self.spot_analyzer.init_mode = "peak"
        else:
            # Default fallback
            self.spot_analyzer.init_mode = "watershed"
        self.spot_analyzer.watershed_h_rel = float(self.watershed_h_rel_spin.value())
        self.spot_analyzer.watershed_adaptive_h = bool(self.watershed_adaptive_h_check.isChecked())
        self.spot_analyzer.blob_doh_min_sigma = float(self.blob_doh_min_sigma_spin.value())
        self.spot_analyzer.blob_doh_max_sigma = float(self.blob_doh_max_sigma_spin.value())
        self.spot_analyzer.blob_doh_num_sigma = int(getattr(self.spot_analyzer, "blob_doh_num_sigma", 10))
        self.spot_analyzer.blob_doh_threshold_rel = float(self.blob_doh_threshold_rel_spin.value())
        self.spot_analyzer.dbscan_enabled = bool(self.dbscan_enabled_check.isChecked())
        self.spot_analyzer.dbscan_eps = float(self.dbscan_eps_spin.value())

        self.spot_analyzer.peak_min_distance = max(1, int(self.peak_min_distance_spin.value()))
        self.spot_analyzer.localmax_threshold_rel = float(self.localmax_thr_spin.value())
        self.spot_analyzer.localmax_threshold_snr = float(self.localmax_snr_spin.value())
        try:
            self.spot_analyzer.precheck_radius_px = max(0, int(self.precheck_radius_spin.value()))
        except Exception:
            self.spot_analyzer.precheck_radius_px = max(0, int(getattr(self.spot_analyzer, "precheck_radius_px", 2)))
        try:
            self.spot_analyzer.precheck_kmad = float(self.precheck_kmad_spin.value())
        except Exception:
            self.spot_analyzer.precheck_kmad = float(getattr(self.spot_analyzer, "precheck_kmad", 1.0))
        self.spot_analyzer.initial_sigma = float(self.initial_sigma_spin.value())
        sigma_min = float(self.sigma_min_spin.value())
        sigma_max = float(self.sigma_max_spin.value())
        if sigma_max <= sigma_min:
            if show_errors:
                QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "σ上限はσ下限より大きくしてください。")
            return False
        self.spot_analyzer.sigma_bounds = (sigma_min, sigma_max)
        self.spot_analyzer.snr_threshold = float(self.snr_spin.value())

        # filters
        self.spot_analyzer.margin = int(self.margin_spin.value())
        self.spot_analyzer.min_amplitude = float(self.min_amp_spin.value())
        self.spot_analyzer.min_sigma_result = float(self.min_sigma_result_spin.value())

        # final peak snapping (display/output coordinates)
        self.spot_analyzer.snap_enabled = bool(self.snap_enabled_check.isChecked())
        self.spot_analyzer.snap_radius = max(0, int(self.snap_radius_spin.value()))
        self.spot_analyzer.snap_refit_enabled = bool(self.snap_refit_check.isChecked())
        try:
            self.spot_analyzer.refit_max_shift_px = max(0, int(self.refit_shift_spin.value()))
        except Exception:
            self.spot_analyzer.refit_max_shift_px = max(0, int(getattr(self.spot_analyzer, "refit_max_shift_px", 3)))

        return True

    def _refresh_selection_label(self) -> None:
        if not self.main_window or not hasattr(self.main_window, "FileList"):
            self.selection_label.setText("ファイル選択情報を取得できません。")
            return
        selected = self.main_window.FileList.selectedItems()
        if selected:
            names = [item.text() for item in selected]
            self.selection_label.setText(f"選択中: {', '.join(names)}")
        else:
            current = self.main_window.FileList.currentItem()
            if current:
                self.selection_label.setText(f"現在: {current.text()}")
            else:
                self.selection_label.setText("選択なし")

    def _ensure_selection_loaded(self) -> bool:
        if not self.main_window or not hasattr(self.main_window, "FileList"):
            QtWidgets.QMessageBox.warning(self, "No Selection", "FileListが見つかりません。")
            return False
        selected = self.main_window.FileList.selectedIndexes()
        if selected:
            target_row = selected[0].row()
        else:
            target_row = self.main_window.FileList.currentRow()
        if target_row is None or target_row < 0:
            QtWidgets.QMessageBox.information(self, "Select File", "ファイルリストで対象を選択してください。")
            return False

        # MainWindowのロジックに合わせて選択を反映させる
        # setCurrentRowが itemSelectionChanged を発火し、pyNuD側の通常ハンドラ経由で
        # メインウィンドウが最前面化するのを避けるため、シグナルを一時的にブロックする
        try:
            blocker = QtCore.QSignalBlocker(self.main_window.FileList)
            self.main_window.FileList.setCurrentRow(target_row)
        except Exception:
            self.main_window.FileList.setCurrentRow(target_row)
        if hasattr(self.main_window, "ListClickFunction"):
            try:
                # SpotAnalysisからの呼び出しではメインウィンドウを最前面化しない
                try:
                    self.main_window.ListClickFunction(bring_to_front=False)
                except TypeError:
                    # 旧シグネチャ互換
                    self.main_window.ListClickFunction()
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "Load Error", f"ファイル読み込みに失敗しました:\n{exc}")
                return False
        return True

    def _prepare_frame(self) -> Optional[np.ndarray]:
        if not hasattr(gv, "files") or not gv.files:
            QtWidgets.QMessageBox.information(self, "No Files", "ファイルがロードされていません。")
            return None
        if getattr(gv, "currentFileNum", -1) < 0 or gv.currentFileNum >= len(gv.files):
            QtWidgets.QMessageBox.warning(self, "Invalid Selection", "選択中のファイルインデックスが不正です。")
            return None

        # 最新のフレームを取得
        try:
            LoadFrame(gv.files[gv.currentFileNum])
            InitializeAryDataFallback()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Load Error", f"フレーム読み込みに失敗しました:\n{exc}")
            return None

        if not hasattr(gv, "aryData") or gv.aryData is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "画像データが利用できません。")
            return None
        frame = np.asarray(gv.aryData, dtype=np.float64)
        if frame.ndim != 2:
            QtWidgets.QMessageBox.warning(self, "Data Error", "2D画像データのみ解析可能です。")
            return None
        return frame

    def _connect_frame_signal(self) -> None:
        if self.main_window and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.connect(self._on_frame_changed)
            except Exception:
                pass

    def _on_frame_changed(self, frame_index: int) -> None:
        # 現フレームの表示更新
        if self.auto_analyze_check.isChecked():
            # 自動解析時:
            #  - そのフレームに手動ROIがあればそれを優先
            #  - なければ直近過去フレームのROIを引き継ぐ（伝播）
            roi_here = self.roi_by_frame.get(frame_index)
            if roi_here is not None:
                self.manual_roi = roi_here
            else:
                prev_roi = self._get_last_roi_at_or_before(frame_index - 1)
                if prev_roi is not None:
                    # 参照共有による意図しない書き換えを避けるためコピーして保存
                    propagated = dict(prev_roi)
                    self.roi_by_frame[frame_index] = propagated
                    self.manual_roi = propagated
                else:
                    self.manual_roi = None
        else:
            self.manual_roi = self.roi_by_frame.get(frame_index)
        if self.manual_roi is None:
            self.roi_status_label.setText("ROI未選択")
        else:
            self.roi_status_label.setText("ROI選択済み")
        self._sync_run_buttons_enabled()
        frame = self._prepare_frame()
        if frame is None:
            return
        self.last_frame = frame
        # Always pass both fit-spots and init-spots; rendering decides where to show them.
        spots = self.spots_by_frame.get(frame_index)
        initial_spots = self.initial_spots_by_frame.get(frame_index)
        roi_overlay = self._current_roi_overlay()
        if self.full_viz_window and self._is_window_live(self.full_viz_window):
            self._apply_ui_to_analyzer(show_errors=False)
            _co, _rs, roi_mask, roi_bounds = self._roi_overrides(self.last_frame.shape)
            roi_pre, roi_det = self.spot_analyzer.compute_roi_visual_images(
                self.last_frame, roi_mask_override=roi_mask, roi_bounds_override=roi_bounds
            )
            # Use last_result only when it likely corresponds to the current frame.
            disp_result = self.last_result if frame_index == self._get_current_frame_index() else None
            self.full_viz_window.update_view(
                self.last_frame,
                disp_result,
                spots,
                initial_spots,
                roi_overlay,
                self._get_spot_radius_px(),
                roi_pre_image=roi_pre,
                roi_det_image=roi_det,
                det_label=self.spot_analyzer.detection_label(),
                show_det_spots=bool(getattr(self, "show_det_spots_check", None) and self.show_det_spots_check.isChecked()),
                show_initial_spots=False,
                show_fit_spots_on_roi_pre=bool(
                    getattr(self, "show_fit_spots_on_roi_pre_check", None)
                    and self.show_fit_spots_on_roi_pre_check.isChecked()
                ),
                init_mode=str(getattr(self.spot_analyzer, "init_mode", "") or ""),
            )
        self._update_frame_label()
        if self.auto_analyze_check.isChecked():
            self._auto_analyze_if_enabled(frame_index, frame)

    def _get_current_frame_index(self) -> int:
        return int(getattr(gv, "index", 0))

    def _get_last_roi_at_or_before(self, frame_index: int) -> Optional[Dict[str, float]]:
        """
        指定フレーム以前（frame_index を含む）で、最後に設定されたROIを返す。
        存在しない場合は None。
        """
        if frame_index < 0:
            return None
        for idx in range(int(frame_index), -1, -1):
            roi = self.roi_by_frame.get(idx)
            if roi is not None:
                return roi
        return None

    def _update_frame_label(self) -> None:
        total = int(getattr(gv, "FrameNum", 0))
        current = self._get_current_frame_index()
        if total > 0:
            self.frame_label.setText(f"Frame: {current + 1} / {total}")
        else:
            self.frame_label.setText("Frame: - / -")

    def _prev_frame(self) -> None:
        total = int(getattr(gv, "FrameNum", 0))
        if total <= 0:
            return
        new_index = max(0, self._get_current_frame_index() - 1)
        self._set_frame_index(new_index)

    def _next_frame(self) -> None:
        total = int(getattr(gv, "FrameNum", 0))
        if total <= 0:
            return
        new_index = min(total - 1, self._get_current_frame_index() + 1)
        self._set_frame_index(new_index)

    def _set_frame_index(self, index: int) -> None:
        gv.index = int(index)
        if self.main_window and hasattr(self.main_window, "frameSlider"):
            try:
                self.main_window.frameSlider.setValue(gv.index)
            except Exception:
                pass
        if self.main_window and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.emit(gv.index)
            except Exception:
                self._on_frame_changed(gv.index)

    def run_analysis(self) -> None:
        if self.manual_roi is None:
            QtWidgets.QMessageBox.information(self, "ROI Required", "ROIを選択してください。")
            return
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            return

        if not self._apply_ui_to_analyzer(show_errors=True):
            return
        criterion = self.criterion_combo.currentText().lower()

        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())
        if max_peaks < min_peaks:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "最大ピーク数は最小ピーク数以上にしてください。")
            return

        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)

        self._analyze_current_frame(
            frame,
            criterion,
            center_override,
            roi_size_override,
            roi_mask,
            roi_bounds,
            min_peaks,
            max_peaks,
            show_errors=True,
        )

    def refit_from_manual_spots(self) -> None:
        """
        Constrained re-fit using current frame's manually edited spots as seeds.
        - Spot count is fixed to the number of manual spots.
        - Each spot's (x,y) is constrained within ±r(px) from the seed (r from UI).
        """
        if self.manual_roi is None:
            QtWidgets.QMessageBox.information(self, "ROI Required", "ROIを選択してください。")
            return
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            return
        if not self._apply_ui_to_analyzer(show_errors=True):
            return

        frame_index = self._get_current_frame_index()
        spots = list(self.spots_by_frame.get(frame_index) or [])
        if not spots:
            QtWidgets.QMessageBox.information(self, "No Spots", "手動スポットがありません（追加/移動してから再フィットしてください）。")
            return

        # seeds in absolute pixel coords
        seeds_abs: List[Tuple[float, float]] = []
        for pk in spots:
            try:
                x = float(pk.get("x", float("nan")))
                y = float(pk.get("y", float("nan")))
            except Exception:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            seeds_abs.append((x, y))
        if not seeds_abs:
            QtWidgets.QMessageBox.warning(self, "Invalid Spots", "手動スポット座標が不正です。")
            return

        # Apply constrained shift radius to analyzer
        try:
            self.spot_analyzer.refit_max_shift_px = max(0, int(self.refit_shift_spin.value()))
        except Exception:
            self.spot_analyzer.refit_max_shift_px = 3

        # ROI context
        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)
        frame_f = np.asarray(frame, dtype=np.float64)
        if roi_bounds is not None:
            roi_raw, origin = self.spot_analyzer._crop_rect(frame_f, roi_bounds)
        else:
            center = center_override if center_override is not None else (frame_f.shape[1] / 2.0, frame_f.shape[0] / 2.0)
            use_roi_size = roi_size_override if roi_size_override is not None else self.spot_analyzer.roi_size
            roi_raw, origin = self.spot_analyzer._crop_square(frame_f, center, use_roi_size)
        roi_pre = self.spot_analyzer._apply_roi_preprocess(roi_raw)
        roi_det = self.spot_analyzer._apply_detection(roi_pre)
        h_roi, w_roi = roi_pre.shape

        # Validate all seeds are inside ROI bounds/mask (keep spot count fixed)
        initial_local: List[Tuple[float, float]] = []
        for x_abs, y_abs in seeds_abs:
            lx = float(x_abs) - float(origin[0])
            ly = float(y_abs) - float(origin[1])
            if not (0.0 <= lx < float(w_roi) and 0.0 <= ly < float(h_roi)):
                QtWidgets.QMessageBox.warning(self, "Spot Outside ROI", "手動スポットがROIの外にあります。ROI内へ移動してから再フィットしてください。")
                return
            if roi_mask is not None:
                try:
                    ix = int(round(lx))
                    iy = int(round(ly))
                    if ix < 0 or iy < 0 or ix >= w_roi or iy >= h_roi or not bool(roi_mask.astype(bool)[iy, ix]):
                        QtWidgets.QMessageBox.warning(self, "Spot Outside ROI Mask", "手動スポットがROIマスク（楕円など）の外にあります。マスク内へ移動してから再フィットしてください。")
                        return
                except Exception:
                    pass
            initial_local.append((lx, ly))

        n = int(len(initial_local))
        if n <= 0:
            QtWidgets.QMessageBox.warning(self, "No Valid Spots", "再フィットに使えるスポットがありません。")
            return

        # Fit exactly n peaks (fixed count)
        noise_sigma = self.spot_analyzer._estimate_noise_sigma(roi_pre, roi_mask)
        det_noise_sigma = self.spot_analyzer._estimate_noise_sigma(roi_det, roi_mask)
        try:
            model = self.spot_analyzer._fit_model(
                roi_pre,
                origin,
                n,
                noise_sigma,
                roi_mask,
                initial_xy_local=initial_local,
                init_image=roi_det,
                init_noise_sigma=det_noise_sigma,
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Refit Error", f"再フィットに失敗しました:\n{exc}")
            return

        # Optional snap (keeps near local maxima) without dropping peak count.
        if bool(getattr(self.spot_analyzer, "snap_enabled", False)):
            try:
                model.peaks = self.spot_analyzer._snap_peaks_to_local_maxima(model.peaks, roi_det, origin, roi_mask=roi_mask)
            except Exception:
                pass

        criterion = self.criterion_combo.currentText().lower()
        result = FrameAnalysis(
            best_n_peaks=n,
            criterion="bic" if criterion.lower() == "bic" else "aic",
            noise_sigma=noise_sigma,
            snr_threshold=self.spot_analyzer.snr_threshold,
            models={n: model},
            roi=roi_pre,
            origin=origin,
            roi_mask=roi_mask,
        )
        self.last_frame = frame
        self.last_result = result
        self._display_result(result)
        # Manual refit: keep the auto-detected initial peaks for the overlay.
        self._store_spots_for_frame(frame_index, result, frame=frame, update_initial=False)
        self._refresh_overlay()
        self.export_btn.setEnabled(True)

        # Update signature so debounce reanalysis won't immediately overwrite
        try:
            self._analysis_signature_by_frame[frame_index] = self._analysis_signature(frame_index, self._current_roi_overlay())
        except Exception:
            pass

    def _display_result(self, result: FrameAnalysis) -> None:
        def esc(s: str) -> str:
            return html.escape(s, quote=False)

        html_lines: List[str] = []
        html_lines.append(esc(f"判定モデル: {result.best_n_peaks} peaks ({result.criterion.upper()})"))
        for n_peaks in sorted(result.models.keys()):
            model = result.models[n_peaks]
            html_lines.append(
                esc(
                    f"[{n_peaks} peaks] AIC={model.aic:.2f}, BIC={model.bic:.2f}, rss={model.rss:.4g}, loglike={model.loglike:.4g}, residual_std={model.residual_std:.4g}"
                )
            )
            for idx, pk in enumerate(model.peaks, start=1):
                html_lines.append(
                    esc(
                        f"  P{idx}: amp={pk.amplitude:.3g}, sigma={pk.sigma:.3g}, (x,y)=({pk.x:.2f},{pk.y:.2f}), S/N={pk.snr:.2f}"
                    )
                )

        # 除外されたピークの情報を表示（bestモデルのみ）
        if result.best_n_peaks is not None and result.best_n_peaks in result.models:
            best_model = result.models[result.best_n_peaks]
            init_peaks = best_model.init_peaks if hasattr(best_model, 'init_peaks') else []
            final_peaks = best_model.peaks
            excluded_infos = list(getattr(best_model, "excluded_infos", []) or [])
            # 2px近傍マッチを廃止し、解析側で確定収集した excluded_infos（真に落としたピーク）から表示する。
            # P番号はフィルタ前ピークのインデックス（= init_peaks順）をIDマップで引く。
            idx_map = getattr(best_model, "prefilter_index_by_id", None)
            excluded_rows: List[Tuple[int, Dict[str, Any]]] = []
            for ent in excluded_infos:
                try:
                    pk = ent.get("peak")
                except Exception:
                    pk = None
                if pk is None:
                    continue
                idx = None
                try:
                    if isinstance(idx_map, dict):
                        idx = idx_map.get(id(pk))
                except Exception:
                    idx = None
                if idx is None:
                    # No proximity matching here (explicitly disabled)
                    continue
                try:
                    excluded_rows.append((int(idx), ent))
                except Exception:
                    continue

            if excluded_rows:
                excluded_rows.sort(key=lambda t: t[0])
                html_lines.append("")
                html_lines.append(esc("--- 除外されたピーク ---"))
                for idx0, ent in excluded_rows:
                    pk = ent.get("peak")
                    try:
                        reasons = [str(r) for r in list(ent.get("reasons") or []) if str(r).strip()]
                    except Exception:
                        reasons = []
                    if not reasons:
                        reasons = ["理由取得失敗"]
                    reason_str = ", ".join(reasons)
                    prefix = (
                        f"  [除外] P{idx0 + 1}: amp={pk.amplitude:.3g}, sigma={pk.sigma:.3g}, "
                        f"(x,y)=({pk.x:.2f},{pk.y:.2f}), S/N={pk.snr:.2f}  ★理由: "
                    )
                    html_lines.append(
                        esc(prefix)
                        + '<span style="color:#c00; font-weight:600;">'
                        + esc(reason_str)
                        + "</span>"
                    )

        html_lines.append("")
        html_lines.append(esc(f"S/N閾値（出力フィルタ）: {result.snr_threshold:.2f}"))

        # Rich text (HTML) で表示（理由部分のみ赤字）
        body = "\n".join(html_lines)
        html_doc = (
            '<pre style="margin:0; font-family:monospace; font-size:12px; white-space:pre-wrap;">'
            + body
            + "</pre>"
        )
        try:
            self.output.setHtml(html_doc)
        except Exception:
            # Fallback to plain text if needed
            try:
                self.output.setPlainText("\n".join([html.unescape(x) for x in html_lines]))
            except Exception:
                pass

    def _show_roi_view(self, result: FrameAnalysis) -> None:
        """ROI可視化ウィンドウを表示/更新"""
        # ROI可視化ウィンドウは使用しない
        return

    def show_full_image_view(self, result: FrameAnalysis = None) -> None:
        """全画像に検出ピークをオーバーレイ表示（矩形選択も可能）"""
        if result is None:
            result = self.last_result
        # 解析結果がなくても現在のフレームを表示してROI指定を可能にする
        if result is None or self.last_frame is None:
            if not self._ensure_selection_loaded():
                return
            frame = self._prepare_frame()
            if frame is None:
                QtWidgets.QMessageBox.information(self, "No Data", "画像がロードされていません。")
                return
            self.last_frame = frame
            result = None
        win = self._ensure_live_window(self.full_viz_window, SpotFullImageWindow)
        self.full_viz_window = win
        self.full_viz_window.enable_roi_selector(self.roi_shape_combo.currentText(), self._on_full_image_selected)
        self.full_viz_window.set_edit_handler(self._handle_edit_event)
        # Always pass both fit-spots and init-spots; rendering decides where to show them.
        spots = self.spots_by_frame.get(self._get_current_frame_index())
        initial_spots = self.initial_spots_by_frame.get(self._get_current_frame_index())
        self._apply_ui_to_analyzer(show_errors=False)
        _co, _rs, roi_mask, roi_bounds = self._roi_overrides(self.last_frame.shape)
        roi_pre, roi_det = self.spot_analyzer.compute_roi_visual_images(
            self.last_frame, roi_mask_override=roi_mask, roi_bounds_override=roi_bounds
        )
        win.update_view(
            self.last_frame,
            result,
            spots,
            initial_spots,
            self._current_roi_overlay(),
            self._get_spot_radius_px(),
            roi_pre_image=roi_pre,
            roi_det_image=roi_det,
            det_label=self.spot_analyzer.detection_label(),
            show_roi_spots=bool(getattr(self, "show_roi_spots_check", None) and self.show_roi_spots_check.isChecked()),
            show_det_spots=bool(getattr(self, "show_det_spots_check", None) and self.show_det_spots_check.isChecked()),
            show_initial_spots=False,
            show_fit_spots_on_roi_pre=bool(
                getattr(self, "show_fit_spots_on_roi_pre_check", None)
                and self.show_fit_spots_on_roi_pre_check.isChecked()
            ),
            init_mode=str(getattr(self.spot_analyzer, "init_mode", "") or ""),
        )
        win.show()
        win.raise_()
        win.activateWindow()

    def _get_spot_radius_px(self) -> int:
        try:
            return int(self.spot_radius_spin.value())
        except Exception:
            return 4

    def _on_spot_radius_changed(self, _value: int) -> None:
        # 半径が変わったら、現在フレームの高さを再計算して再描画
        frame_index = self._get_current_frame_index()
        self._recompute_spot_heights_for_frame(frame_index)
        self._refresh_overlay()

    def _reset_analysis_results(self) -> None:
        """解析結果（spots/ROI/UI/表示）を一括でクリアする。"""
        # データをクリア
        self.spots_by_frame = {}
        self.initial_spots_by_frame = {}
        self.roi_by_frame = {}
        self.manual_roi = None
        self.last_result = None
        self._analysis_signature_by_frame = {}
        try:
            self._reanalysis_timer.stop()
        except Exception:
            pass
        self._dragging = False
        self._drag_index = None

        # UIを初期状態へ
        try:
            try:
                self.output.setHtml("")
            except Exception:
                self.output.setPlainText("")
        except Exception:
            pass
        try:
            self.roi_status_label.setText("ROI未選択")
        except Exception:
            pass
        for btn in (getattr(self, "run_btn", None), getattr(self, "run_all_btn", None), getattr(self, "export_btn", None)):
            try:
                if btn is not None:
                    btn.setEnabled(False)
            except Exception:
                pass
        self._sync_run_buttons_enabled()

        # 表示を更新（spots/ROIなしで再描画）
        self._refresh_overlay()

    def _on_full_image_selected(self, roi_info: Dict[str, float]) -> None:
        """全画像でのROI選択コールバック"""
        w = roi_info.get("w", 0)
        h = roi_info.get("h", 0)
        if w <= 1 or h <= 1:
            return
        frame_index = self._get_current_frame_index()
        self.roi_by_frame[frame_index] = roi_info
        self.manual_roi = roi_info
        self.roi_status_label.setText("ROI選択済み")
        self._sync_run_buttons_enabled()
        # 自動で再解析（ROI中心・サイズを利用）
        self.run_analysis()

    def _on_roi_shape_changed(self, text: str) -> None:
        if self.full_viz_window and self._is_window_live(self.full_viz_window):
            self.full_viz_window.enable_roi_selector(text, self._on_full_image_selected)

    def _roi_overrides(self, frame_shape: Tuple[int, int]):
        if self.manual_roi is None:
            return None, None, None, None
        h_img, w_img = frame_shape
        if self.manual_roi.get("shape") == "Ellipse":
            x0 = int(round(self.manual_roi["x0"]))
            y0 = int(round(self.manual_roi["y0"]))
            x1 = int(round(self.manual_roi["x0"] + self.manual_roi["w"]))
            y1 = int(round(self.manual_roi["y0"] + self.manual_roi["h"]))
            x0c = max(x0, 0)
            y0c = max(y0, 0)
            x1c = min(x1, w_img)
            y1c = min(y1, h_img)
            bounds = (x0c, y0c, x1c - x0c, y1c - y0c)
            cx = self.manual_roi["cx"]
            cy = self.manual_roi["cy"]
            rx = self.manual_roi["rx"]
            ry = self.manual_roi["ry"]
            roi_w = max(1, bounds[2])
            roi_h = max(1, bounds[3])
            yy, xx = np.mgrid[0:roi_h, 0:roi_w]
            mask = ((xx + bounds[0] - cx) ** 2) / (rx ** 2 + 1e-12) + ((yy + bounds[1] - cy) ** 2) / (ry ** 2 + 1e-12) <= 1.0
            return (cx, cy), None, mask, bounds
        # Rectangle
        x0 = int(round(self.manual_roi["x0"]))
        y0 = int(round(self.manual_roi["y0"]))
        x1 = int(round(self.manual_roi["x0"] + self.manual_roi["w"]))
        y1 = int(round(self.manual_roi["y0"] + self.manual_roi["h"]))
        x0c = max(x0, 0)
        y0c = max(y0, 0)
        x1c = min(x1, w_img)
        y1c = min(y1, h_img)
        bounds = (x0c, y0c, x1c - x0c, y1c - y0c)
        return None, None, None, bounds

    def _current_roi_overlay(self) -> Optional[Dict[str, float]]:
        return self.roi_by_frame.get(self._get_current_frame_index(), self.manual_roi)

    def _store_spots_for_frame(
        self,
        frame_index: int,
        result: FrameAnalysis,
        frame: Optional[np.ndarray] = None,
        update_initial: bool = True,
    ) -> None:
        best = result.models[result.best_n_peaks]
        self.spots_by_frame[frame_index] = [
            {"x": float(pk.x), "y": float(pk.y), "snr": float(pk.snr)}
            for pk in best.peaks
        ]
        if update_initial:
            # Display initial positions as detection seeds (independent of best_n_peaks / initial_sigma)
            # Fallback to legacy init_peaks if seeds are not available.
            if getattr(result, "seed_spots", None):
                self.initial_spots_by_frame[frame_index] = [
                    {"x": float(pk.get("x", 0.0)), "y": float(pk.get("y", 0.0)), "snr": float(pk.get("snr", 0.0))}
                    for pk in (result.seed_spots or [])
                ]
            else:
                self.initial_spots_by_frame[frame_index] = [
                    {"x": float(pk.x), "y": float(pk.y), "snr": float(pk.snr)}
                    for pk in best.init_peaks
                ]
        # 高さ情報を付与
        use_frame = frame if frame is not None else self.last_frame
        if use_frame is not None:
            self._recompute_spot_heights_for_frame(frame_index, frame=use_frame)

    def _refresh_overlay(self) -> None:
        if not self.full_viz_window or not self._is_window_live(self.full_viz_window):
            return
        frame_index = self._get_current_frame_index()
        # Always pass both fit-spots and init-spots; rendering decides where to show them.
        spots = self.spots_by_frame.get(frame_index)
        initial_spots = self.initial_spots_by_frame.get(frame_index)
        if self.last_frame is None:
            return
        self._apply_ui_to_analyzer(show_errors=False)
        _co, _rs, roi_mask, roi_bounds = self._roi_overrides(self.last_frame.shape)
        roi_pre, roi_det = self.spot_analyzer.compute_roi_visual_images(
            self.last_frame, roi_mask_override=roi_mask, roi_bounds_override=roi_bounds
        )
        self.full_viz_window.update_view(
            self.last_frame,
            self.last_result,
            spots,
            initial_spots,
            self._current_roi_overlay(),
            self._get_spot_radius_px(),
            roi_pre_image=roi_pre,
            roi_det_image=roi_det,
            det_label=self.spot_analyzer.detection_label(),
            show_roi_spots=bool(getattr(self, "show_roi_spots_check", None) and self.show_roi_spots_check.isChecked()),
            show_det_spots=bool(getattr(self, "show_det_spots_check", None) and self.show_det_spots_check.isChecked()),
            show_initial_spots=False,
            show_fit_spots_on_roi_pre=bool(
                getattr(self, "show_fit_spots_on_roi_pre_check", None)
                and self.show_fit_spots_on_roi_pre_check.isChecked()
            ),
            init_mode=str(getattr(self.spot_analyzer, "init_mode", "") or ""),
        )

    def _get_roi_info_for_frame(self, frame_index: int) -> Optional[Dict[str, float]]:
        return self.roi_by_frame.get(frame_index, self.manual_roi)

    def _compute_spot_circle_mean_nm(self, frame: np.ndarray, x: float, y: float, r_px: int) -> Optional[float]:
        if r_px <= 0:
            return None
        h, w = frame.shape
        x0 = max(int(np.floor(x - r_px)), 0)
        x1 = min(int(np.ceil(x + r_px)) + 1, w)
        y0 = max(int(np.floor(y - r_px)), 0)
        y1 = min(int(np.ceil(y + r_px)) + 1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= float(r_px) ** 2
        data = frame[y0:y1, x0:x1][mask]
        if data.size == 0:
            return None
        return float(np.mean(data))

    def _compute_background_median_nm(
        self,
        frame: np.ndarray,
        frame_index: int,
        r_px: int,
        spots: Sequence[Dict[str, float]],
    ) -> float:
        """
        背景=ROI内の画素から、spot半径r円を全て除外した残りの中央値。
        画素が空になる場合はROI中央値→フレーム中央値へフォールバック。
        """
        roi_info = self._get_roi_info_for_frame(frame_index)
        h, w = frame.shape

        if roi_info is None:
            roi_x0, roi_y0, roi_w, roi_h = 0, 0, w, h
            roi_mask = np.ones((h, w), dtype=bool)
            roi_crop = frame
            crop_x0, crop_y0 = 0, 0
        else:
            roi_x0 = int(round(float(roi_info.get("x0", 0.0))))
            roi_y0 = int(round(float(roi_info.get("y0", 0.0))))
            roi_w = int(round(float(roi_info.get("w", 0.0))))
            roi_h = int(round(float(roi_info.get("h", 0.0))))
            crop_x0 = max(0, roi_x0)
            crop_y0 = max(0, roi_y0)
            crop_x1 = min(w, roi_x0 + max(0, roi_w))
            crop_y1 = min(h, roi_y0 + max(0, roi_h))
            if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                return float(np.median(frame))
            roi_crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            roi_mask = np.ones_like(roi_crop, dtype=bool)
            if roi_info.get("shape") == "Ellipse":
                cx = float(roi_info["cx"])
                cy = float(roi_info["cy"])
                rx = float(roi_info["rx"])
                ry = float(roi_info["ry"])
                yy, xx = np.mgrid[crop_y0:crop_y1, crop_x0:crop_x1]
                roi_mask = ((xx - cx) ** 2) / (rx ** 2 + 1e-12) + ((yy - cy) ** 2) / (ry ** 2 + 1e-12) <= 1.0

        # spot円の除外マスク（ROIクロップ座標）
        exclude = np.zeros_like(roi_mask, dtype=bool)
        if r_px > 0 and spots:
            yy, xx = np.mgrid[0:roi_crop.shape[0], 0:roi_crop.shape[1]]
            abs_x = xx + crop_x0
            abs_y = yy + crop_y0
            r2 = float(r_px) ** 2
            for pk in spots:
                sx = float(pk.get("x", 0.0))
                sy = float(pk.get("y", 0.0))
                exclude |= (abs_x - sx) ** 2 + (abs_y - sy) ** 2 <= r2

        bg_mask = roi_mask & (~exclude)
        data_bg = roi_crop[bg_mask]
        if data_bg.size > 0:
            return float(np.median(data_bg))
        # fallback: ROI中央値
        data_roi = roi_crop[roi_mask] if roi_mask is not None else roi_crop.ravel()
        if data_roi.size > 0:
            return float(np.median(data_roi))
        return float(np.median(frame))

    def _recompute_spot_heights_for_frame(self, frame_index: int, frame: Optional[np.ndarray] = None) -> None:
        spots = self.spots_by_frame.get(frame_index)
        if not spots:
            return
        use_frame = frame if frame is not None else self.last_frame
        if use_frame is None:
            return
        r_px = self._get_spot_radius_px()
        bg_nm = self._compute_background_median_nm(use_frame, frame_index, r_px, spots)
        for pk in spots:
            mean_nm = self._compute_spot_circle_mean_nm(use_frame, float(pk["x"]), float(pk["y"]), r_px)
            if mean_nm is None:
                continue
            pk["height_mean_nm"] = float(mean_nm)
            pk["height_bg_nm"] = float(bg_nm)
            pk["height_bgsub_nm"] = float(mean_nm - bg_nm)

    def _handle_edit_event(self, event, phase: str) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        frame_index = self._get_current_frame_index()
        spots = self.spots_by_frame.setdefault(frame_index, [])
        key = (event.key or "").lower()
        is_shift = "shift" in key
        is_delete = "alt" in key or "option" in key
        is_ctrl = "control" in key or "ctrl" in key
        is_cmd = "cmd" in key or "command" in key or "meta" in key or "super" in key
        if not key:
            mods = QtWidgets.QApplication.keyboardModifiers()
            is_shift = bool(mods & QtCore.Qt.ShiftModifier)
            is_delete = bool(mods & QtCore.Qt.AltModifier)
            is_ctrl = bool(mods & QtCore.Qt.ControlModifier)
            is_cmd = bool(mods & QtCore.Qt.MetaModifier)
        # Matplotlib(Qt backend)では、修飾キーが event.key / keyboardModifiers に反映されないことがあるため
        # 元のQtイベントからも取得する（macOSのCtrl+クリック/ドラッグ=右クリック扱い等の対策）
        try:
            gui_ev = getattr(event, "guiEvent", None)
            if gui_ev is not None and hasattr(gui_ev, "modifiers"):
                is_ctrl = is_ctrl or bool(gui_ev.modifiers() & QtCore.Qt.ControlModifier)
                is_cmd = is_cmd or bool(gui_ev.modifiers() & QtCore.Qt.MetaModifier)
        except Exception:
            pass

        if phase == "press":
            if is_shift:
                spots.append({"x": float(event.xdata), "y": float(event.ydata), "snr": 0.0})
                self.export_btn.setEnabled(True)
                self._recompute_spot_heights_for_frame(frame_index)
                self._refresh_overlay()
                return
            if is_delete:
                idx = self._find_nearest_spot(spots, event.xdata, event.ydata)
                if idx is not None:
                    spots.pop(idx)
                    self.export_btn.setEnabled(True)
                    self._recompute_spot_heights_for_frame(frame_index)
                    self._refresh_overlay()
                return
            # 移動は Ctrl または Cmd(⌘) 押下時のみ開始（ROI描画のドラッグと衝突しないようにする）
            if not (is_ctrl or is_cmd):
                return
            idx = self._find_nearest_spot(spots, event.xdata, event.ydata)
            if idx is not None:
                self._dragging = True
                self._drag_index = idx
        elif phase == "move":
            if self._dragging and self._drag_index is not None:
                spots[self._drag_index]["x"] = float(event.xdata)
                spots[self._drag_index]["y"] = float(event.ydata)
                self._recompute_spot_heights_for_frame(frame_index)
                self._refresh_overlay()
        elif phase == "release":
            if self._dragging:
                self._dragging = False
                self._drag_index = None
                self.export_btn.setEnabled(True)
                self._recompute_spot_heights_for_frame(frame_index)

    def _find_nearest_spot(self, spots: List[Dict[str, float]], x: float, y: float, threshold: float = 5.0) -> Optional[int]:
        if not spots:
            return None
        best_idx = None
        best_dist = None
        for i, pk in enumerate(spots):
            dx = pk["x"] - x
            dy = pk["y"] - y
            dist = (dx * dx + dy * dy) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist is not None and best_dist <= threshold:
            return best_idx
        return None

    def run_analysis_all_frames(self) -> None:
        if hasattr(self, "run_all_enable_check") and not self.run_all_enable_check.isChecked():
            QtWidgets.QMessageBox.information(self, "All Frames Disabled", "「全フレーム解析を有効化」をONにしてください。")
            return
        if self.manual_roi is None:
            QtWidgets.QMessageBox.information(self, "ROI Required", "ROIを選択してください。")
            return
        if not self._ensure_selection_loaded():
            return
        if not hasattr(gv, "FrameNum") or gv.FrameNum <= 0:
            QtWidgets.QMessageBox.warning(self, "No Frames", "フレーム数が取得できません。")
            return
        if not self._apply_ui_to_analyzer(show_errors=True):
            return
        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())
        if max_peaks < min_peaks:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "最大ピーク数は最小ピーク数以上にしてください。")
            return
        original_index = int(getattr(gv, "index", 0))
        self.spots_by_frame = {}
        self.initial_spots_by_frame = {}
        self._analysis_signature_by_frame = {}
        for idx in range(int(gv.FrameNum)):
            gv.index = idx
            frame = self._prepare_frame()
            if frame is None:
                continue
            self.last_frame = frame
            center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)
            try:
                result = self.spot_analyzer.analyze_frame(
                    frame,
                    prev_center=None,
                    criterion=self.criterion_combo.currentText().lower(),
                    center_override=center_override,
                    roi_size_override=roi_size_override,
                    roi_mask_override=roi_mask,
                    roi_bounds_override=roi_bounds,
                    min_peaks=min_peaks,
                    max_peaks=max_peaks,
                )
            except Exception:
                continue
            self.last_result = result
            self._store_spots_for_frame(idx, result, frame=frame)
            # 署名を記録（同じ条件の再解析をスキップ）
            try:
                roi_info = self.roi_by_frame.get(idx, self.manual_roi)
                self._analysis_signature_by_frame[idx] = self._analysis_signature(idx, roi_info)
            except Exception:
                pass
        gv.index = original_index
        frame = self._prepare_frame()
        if frame is not None:
            self.last_frame = frame
        self.export_btn.setEnabled(True)
        self._refresh_overlay()

    def _analyze_current_frame(
        self,
        frame: np.ndarray,
        criterion: str,
        center_override: Optional[Tuple[float, float]],
        roi_size_override: Optional[int],
        roi_mask: Optional[np.ndarray],
        roi_bounds: Optional[Tuple[int, int, int, int]],
        min_peaks: int,
        max_peaks: int,
        show_errors: bool = False,
    ) -> None:
        try:
            result = self.spot_analyzer.analyze_frame(
                frame,
                prev_center=None,
                criterion=criterion,
                center_override=center_override,
                roi_size_override=roi_size_override,
                roi_mask_override=roi_mask,
                roi_bounds_override=roi_bounds,
                min_peaks=min_peaks,
                max_peaks=max_peaks,
            )
        except Exception as exc:  # noqa: BLE001
            if show_errors:
                QtWidgets.QMessageBox.critical(self, "分析エラー", f"SpotAnalysisの実行に失敗しました:\n{exc}")
            return

        self.last_frame = frame
        self.last_result = result
        self._display_result(result)
        self._store_spots_for_frame(self._get_current_frame_index(), result, frame=frame)
        # 署名を記録（同じ条件なら再解析をスキップできる）
        try:
            fi = self._get_current_frame_index()
            self._analysis_signature_by_frame[fi] = self._analysis_signature(fi, self._current_roi_overlay())
        except Exception:
            pass
        self._refresh_overlay()
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _auto_analyze_if_enabled(self, frame_index: int, frame: np.ndarray) -> None:
        if not self.auto_analyze_check.isChecked():
            return
        if self._auto_busy:
            return
        if self.manual_roi is None:
            return
        if not self._apply_ui_to_analyzer(show_errors=False):
            return

        roi_info = self._current_roi_overlay()
        sig = self._analysis_signature(frame_index, roi_info)
        if self._analysis_signature_by_frame.get(frame_index) == sig:
            return

        min_peaks = int(self.min_peaks_spin.value())
        max_peaks = int(self.max_peaks_spin.value())
        if max_peaks < min_peaks:
            return

        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides(frame.shape)
        self._auto_busy = True
        try:
            self._analyze_current_frame(
                frame,
                self.criterion_combo.currentText().lower(),
                center_override,
                roi_size_override,
                roi_mask,
                roi_bounds,
                min_peaks,
                max_peaks,
                show_errors=False,
            )
        finally:
            self._auto_busy = False

    def export_spots_csv(self) -> None:
        if not self.spots_by_frame:
            QtWidgets.QMessageBox.information(self, "No Data", "保存できるスポットがありません。")
            return
        if not hasattr(gv, "files") or not gv.files or gv.currentFileNum < 0:
            QtWidgets.QMessageBox.warning(self, "No File", "元ファイル情報が取得できません。")
            return
        if not hasattr(gv, "XScanSize") or not hasattr(gv, "YScanSize") or gv.XScanSize == 0 or gv.YScanSize == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Scan Size", "scan_sizeが0です。先に正しいデータを読み込んでください。")
            return
        if not hasattr(gv, "XPixel") or not hasattr(gv, "YPixel") or gv.XPixel == 0 or gv.YPixel == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Pixel Size", "ピクセルサイズ情報が取得できません。")
            return

        nm_per_pixel_x = gv.XScanSize / gv.XPixel
        nm_per_pixel_y = gv.YScanSize / gv.YPixel

        src_path = gv.files[gv.currentFileNum]
        base_dir = os.path.dirname(src_path)
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        if self.export_dir is None:
            self.export_dir = base_dir
        default_path = os.path.join(self.export_dir, f"{base_name}.csv")
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save CSV (base name)",
            default_path,
            "CSV Files (*.csv)",
        )
        if not selected_path:
            return
        export_dir = os.path.dirname(selected_path)
        base_stub = os.path.splitext(os.path.basename(selected_path))[0] or base_name
        self.export_dir = export_dir

        # save meta.json alongside CSVs
        try:
            self._save_spot_meta_json(os.path.join(export_dir, f"{base_stub}_meta.json"), nm_per_pixel_x, nm_per_pixel_y)
        except Exception:
            pass

        max_spots = max(len(v) for v in self.spots_by_frame.values())
        # pre-check conflicts and ask once
        conflict = False
        for p in range(1, int(max_spots) + 1):
            pos_path = os.path.join(export_dir, f"{base_stub}_p{p}.csv")
            h_path = os.path.join(export_dir, f"{base_stub}_p{p}_h.csv")
            if os.path.exists(pos_path) or os.path.exists(h_path):
                conflict = True
                break

        mode = "number"
        if conflict:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("File Exists")
            box.setText("同名のCSVファイルが既に存在します。どうしますか？")
            overwrite_btn = box.addButton("上書き", QtWidgets.QMessageBox.AcceptRole)
            number_btn = box.addButton("連番で回避", QtWidgets.QMessageBox.ActionRole)
            cancel_btn = box.addButton("キャンセル", QtWidgets.QMessageBox.RejectRole)
            box.setDefaultButton(number_btn)
            box.exec_()
            clicked = box.clickedButton()
            if clicked == cancel_btn:
                return
            if clicked == overwrite_btn:
                mode = "overwrite"
            else:
                mode = "number"

        for spot_index in range(max_spots):
            out_pos_path, out_h_path = self._unique_spot_paths(export_dir, base_stub, spot_index + 1, mode=mode)
            if out_pos_path is None or out_h_path is None:
                QtWidgets.QMessageBox.warning(self, "Save Error", "保存先ファイル名の衝突回避に失敗しました。")
                return
            try:
                # 位置CSV
                with open(out_pos_path, "w", encoding="utf-8") as f_pos:
                    f_pos.write("frame_index,x_nm,y_nm\n")
                    for frame_idx in sorted(self.spots_by_frame.keys()):
                        spots = self.spots_by_frame[frame_idx]
                        if spot_index >= len(spots):
                            continue
                        x_nm = spots[spot_index]["x"] * nm_per_pixel_x
                        y_nm = spots[spot_index]["y"] * nm_per_pixel_y

                        f_pos.write(f"{frame_idx},{x_nm:.6f},{y_nm:.6f}\n")

                # 高さCSV
                with open(out_h_path, "w", encoding="utf-8") as f_h:
                    f_h.write("frame_index,height_mean_nm,height_bg_nm,height_bgsub_nm\n")
                    for frame_idx in sorted(self.spots_by_frame.keys()):
                        spots = self.spots_by_frame[frame_idx]
                        if spot_index >= len(spots):
                            continue
                        # 高さが未計算なら、現在のフレームデータで計算（可能な範囲で）
                        if (
                            "height_mean_nm" not in spots[spot_index]
                            or "height_bgsub_nm" not in spots[spot_index]
                            or "height_bg_nm" not in spots[spot_index]
                        ):
                            if self.last_frame is not None and frame_idx == self._get_current_frame_index():
                                self._recompute_spot_heights_for_frame(frame_idx, frame=self.last_frame)
                        h_mean = float(spots[spot_index].get("height_mean_nm", float("nan")))
                        h_bg = float(spots[spot_index].get("height_bg_nm", float("nan")))
                        h_bgsub = float(spots[spot_index].get("height_bgsub_nm", float("nan")))
                        f_h.write(f"{frame_idx},{h_mean:.6f},{h_bg:.6f},{h_bgsub:.6f}\n")
            except Exception:
                continue

    def _unique_spot_paths(
        self, export_dir: str, base_stub: str, p_index: int, mode: str = "number", max_tries: int = 999
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate non-existing paired paths:
          <base>_pN.csv and <base>_pN_h.csv
        If any exists, append _k suffix before extension:
          <base>_pN_1.csv and <base>_pN_h_1.csv
        """
        p = int(p_index)
        if str(mode).strip().lower() == "overwrite":
            pos_path = os.path.join(export_dir, f"{base_stub}_p{p}.csv")
            h_path = os.path.join(export_dir, f"{base_stub}_p{p}_h.csv")
            return pos_path, h_path
        for k in range(int(max_tries) + 1):
            suffix = "" if k == 0 else f"_{k}"
            pos_name = f"{base_stub}_p{p}{suffix}.csv"
            h_name = f"{base_stub}_p{p}_h{suffix}.csv"
            pos_path = os.path.join(export_dir, pos_name)
            h_path = os.path.join(export_dir, h_name)
            if not os.path.exists(pos_path) and not os.path.exists(h_path):
                return pos_path, h_path
        return None, None

    def _save_spot_meta_json(self, path: str, nm_per_pixel_x: float, nm_per_pixel_y: float) -> None:
        ui_params = self._collect_ui_params()
        meta = {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "frame_num": int(getattr(gv, "FrameNum", 0) or 0),
            "nm_per_pixel_x": float(nm_per_pixel_x),
            "nm_per_pixel_y": float(nm_per_pixel_y),
            "manual_roi": self.manual_roi,
            "roi_by_frame": self.roi_by_frame,
            "ui_params": ui_params,
            "analysis_signature_by_frame": {str(k): v for k, v in self._analysis_signature_by_frame.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _collect_ui_params(self) -> Dict[str, object]:
        """Collect UI values needed to reproduce analysis signature/behavior."""
        def _try_get(getter, default=None):
            try:
                return getter()
            except Exception:
                return default

        return {
            "criterion": _try_get(lambda: (self.criterion_combo.currentText() or "AIC")),
            "detection_mode": _try_get(lambda: (self.detection_mode_combo.currentText() or "DoG")),
            "bandpass_low_sigma": _try_get(lambda: float(self.bandpass_low_spin.value()), 0.6),
            "bandpass_high_sigma": _try_get(lambda: float(self.bandpass_high_spin.value()), 3.0),
            "log_sigma": _try_get(lambda: float(self.log_sigma_spin.value()), 1.6),
            "median_enabled": _try_get(lambda: bool(self.median_check.isChecked()), False),
            "median_size": _try_get(lambda: int(self.median_size_spin.value()), 3),
            "open_enabled": _try_get(lambda: bool(self.open_check.isChecked()), False),
            "open_radius": _try_get(lambda: int(self.open_radius_spin.value()), 1),
            "init_mode": _try_get(lambda: (self.init_mode_combo.currentText() or "Watershed (推奨)")),
            "subpixel_refine": _try_get(lambda: bool(self.subpixel_check.isChecked()), False),
            "watershed_h_rel": _try_get(lambda: float(self.watershed_h_rel_spin.value()), 0.05),
            "watershed_adaptive_h": _try_get(lambda: bool(self.watershed_adaptive_h_check.isChecked()), True),
            "blob_doh_min_sigma": _try_get(lambda: float(self.blob_doh_min_sigma_spin.value()), 3.0),
            "blob_doh_max_sigma": _try_get(lambda: float(self.blob_doh_max_sigma_spin.value()), 20.0),
            "blob_doh_threshold_rel": _try_get(lambda: float(self.blob_doh_threshold_rel_spin.value()), 0.01),
            "dbscan_enabled": _try_get(lambda: bool(self.dbscan_enabled_check.isChecked()), False),
            "dbscan_eps": _try_get(lambda: float(self.dbscan_eps_spin.value()), 5.0),
            "min_peaks": _try_get(lambda: int(self.min_peaks_spin.value()), 1),
            "max_peaks": _try_get(lambda: int(self.max_peaks_spin.value()), 3),
            "snr_threshold": _try_get(lambda: float(self.snr_spin.value()), 2.0),
            "peak_min_distance": _try_get(lambda: int(self.peak_min_distance_spin.value()), 2),
            "localmax_threshold_rel": _try_get(lambda: float(self.localmax_thr_spin.value()), 0.05),
            "localmax_threshold_snr": _try_get(lambda: float(self.localmax_snr_spin.value()), 0.0),
            "precheck_radius_px": _try_get(lambda: int(self.precheck_radius_spin.value()), 2),
            "precheck_kmad": _try_get(lambda: float(self.precheck_kmad_spin.value()), 1.0),
            "initial_sigma": _try_get(lambda: float(self.initial_sigma_spin.value()), 2.0),
            "sigma_min": _try_get(lambda: float(self.sigma_min_spin.value()), 0.6),
            "sigma_max": _try_get(lambda: float(self.sigma_max_spin.value()), 8.0),
            "margin": _try_get(lambda: int(self.margin_spin.value()), 0),
            "min_amplitude": _try_get(lambda: float(self.min_amp_spin.value()), 0.0),
            "min_sigma_result": _try_get(lambda: float(self.min_sigma_result_spin.value()), 0.0),
            "snap_enabled": _try_get(lambda: bool(self.snap_enabled_check.isChecked()), False),
            "snap_radius": _try_get(lambda: int(self.snap_radius_spin.value()), 2),
            "snap_refit_enabled": _try_get(lambda: bool(self.snap_refit_check.isChecked()), False),
            "refit_max_shift_px": _try_get(lambda: int(self.refit_shift_spin.value()), 3),
        }

    def _apply_ui_params(self, params: Dict[str, object]) -> None:
        """Best-effort apply UI params without triggering reanalysis repeatedly."""
        blockers = []
        for w in (
            getattr(self, "criterion_combo", None),
            getattr(self, "detection_mode_combo", None),
            getattr(self, "bandpass_low_spin", None),
            getattr(self, "bandpass_high_spin", None),
            getattr(self, "log_sigma_spin", None),
            getattr(self, "median_check", None),
            getattr(self, "median_size_spin", None),
            getattr(self, "open_check", None),
            getattr(self, "open_radius_spin", None),
            getattr(self, "init_mode_combo", None),
            getattr(self, "subpixel_check", None),
            getattr(self, "watershed_h_rel_spin", None),
            getattr(self, "watershed_adaptive_h_check", None),
            getattr(self, "blob_min_sigma_spin", None),
            getattr(self, "blob_max_sigma_spin", None),
            getattr(self, "blob_num_sigma_spin", None),
            getattr(self, "blob_threshold_rel_spin", None),
            getattr(self, "blob_overlap_spin", None),
            getattr(self, "blob_doh_min_sigma_spin", None),
            getattr(self, "blob_doh_max_sigma_spin", None),
            getattr(self, "blob_doh_threshold_rel_spin", None),
            getattr(self, "dbscan_enabled_check", None),
            getattr(self, "dbscan_eps_spin", None),
            getattr(self, "min_peaks_spin", None),
            getattr(self, "max_peaks_spin", None),
            getattr(self, "snr_spin", None),
            getattr(self, "peak_min_distance_spin", None),
            getattr(self, "localmax_thr_spin", None),
            getattr(self, "localmax_snr_spin", None),
            getattr(self, "precheck_radius_spin", None),
            getattr(self, "precheck_kmad_spin", None),
            getattr(self, "initial_sigma_spin", None),
            getattr(self, "sigma_min_spin", None),
            getattr(self, "sigma_max_spin", None),
            getattr(self, "margin_spin", None),
            getattr(self, "min_amp_spin", None),
            getattr(self, "min_sigma_result_spin", None),
            getattr(self, "snap_enabled_check", None),
            getattr(self, "snap_radius_spin", None),
            getattr(self, "snap_refit_check", None),
            getattr(self, "refit_shift_spin", None),
        ):
            try:
                if w is not None:
                    blockers.append(QtCore.QSignalBlocker(w))
            except Exception:
                pass

        try:
            if "criterion" in params:
                self.criterion_combo.setCurrentText(str(params["criterion"]))
            if "detection_mode" in params:
                det = str(params["detection_mode"]).strip().lower()
                if det in ("pre", "preprocessed"):
                    self.detection_mode_combo.setCurrentText("Preprocessed")
                elif det == "log":
                    self.detection_mode_combo.setCurrentText("LoG")
                elif det == "dog":
                    self.detection_mode_combo.setCurrentText("DoG")
                else:
                    self.detection_mode_combo.setCurrentText(str(params["detection_mode"]))
            if "bandpass_low_sigma" in params:
                self.bandpass_low_spin.setValue(float(params["bandpass_low_sigma"]))
            if "bandpass_high_sigma" in params:
                self.bandpass_high_spin.setValue(float(params["bandpass_high_sigma"]))
            if "log_sigma" in params:
                self.log_sigma_spin.setValue(float(params["log_sigma"]))
            if "median_enabled" in params:
                self.median_check.setChecked(bool(params["median_enabled"]))
            if "median_size" in params:
                self.median_size_spin.setValue(int(params["median_size"]))
            if "open_enabled" in params:
                self.open_check.setChecked(bool(params["open_enabled"]))
            if "open_radius" in params:
                self.open_radius_spin.setValue(int(params["open_radius"]))
            if "init_mode" in params:
                # Map old mode names to new UI labels
                old_mode = str(params["init_mode"]).lower()
                if "watershed" in old_mode or "water" in old_mode:
                    self.init_mode_combo.setCurrentText("Watershed (推奨)")
                elif "doh" in old_mode:
                    self.init_mode_combo.setCurrentText("Blob DoH (高速)")
                elif "blob" in old_mode:
                    # Old blob_log maps to blob_doh
                    self.init_mode_combo.setCurrentText("Blob DoH (高速)")
                elif "peak" in old_mode or "multiscale" in old_mode:
                    self.init_mode_combo.setCurrentText("Peak")
                else:
                    # Try to set directly if it matches new format
                    self.init_mode_combo.setCurrentText(str(params["init_mode"]))
            if "subpixel_refine" in params:
                self.subpixel_check.setChecked(bool(params["subpixel_refine"]))
            if "watershed_h_rel" in params:
                self.watershed_h_rel_spin.setValue(float(params["watershed_h_rel"]))
            if "watershed_adaptive_h" in params:
                self.watershed_adaptive_h_check.setChecked(bool(params["watershed_adaptive_h"]))
            if "blob_doh_min_sigma" in params:
                self.blob_doh_min_sigma_spin.setValue(float(params["blob_doh_min_sigma"]))
            if "blob_doh_max_sigma" in params:
                self.blob_doh_max_sigma_spin.setValue(float(params["blob_doh_max_sigma"]))
            if "blob_doh_threshold_rel" in params:
                self.blob_doh_threshold_rel_spin.setValue(float(params["blob_doh_threshold_rel"]))
            if "dbscan_enabled" in params:
                self.dbscan_enabled_check.setChecked(bool(params["dbscan_enabled"]))
            if "dbscan_eps" in params:
                self.dbscan_eps_spin.setValue(float(params["dbscan_eps"]))
            if "min_peaks" in params:
                self.min_peaks_spin.setValue(int(params["min_peaks"]))
            if "max_peaks" in params:
                self.max_peaks_spin.setValue(int(params["max_peaks"]))
            if "snr_threshold" in params:
                self.snr_spin.setValue(float(params["snr_threshold"]))
            if "peak_min_distance" in params:
                self.peak_min_distance_spin.setValue(int(params["peak_min_distance"]))
            if "localmax_threshold_rel" in params:
                self.localmax_thr_spin.setValue(float(params["localmax_threshold_rel"]))
            if "localmax_threshold_snr" in params:
                self.localmax_snr_spin.setValue(float(params["localmax_threshold_snr"]))
            if "precheck_radius_px" in params:
                self.precheck_radius_spin.setValue(int(params["precheck_radius_px"]))
            if "precheck_kmad" in params:
                self.precheck_kmad_spin.setValue(float(params["precheck_kmad"]))
            if "initial_sigma" in params:
                self.initial_sigma_spin.setValue(float(params["initial_sigma"]))
            if "sigma_min" in params:
                self.sigma_min_spin.setValue(float(params["sigma_min"]))
            if "sigma_max" in params:
                self.sigma_max_spin.setValue(float(params["sigma_max"]))
            if "margin" in params:
                self.margin_spin.setValue(int(params["margin"]))
            if "min_amplitude" in params:
                self.min_amp_spin.setValue(float(params["min_amplitude"]))
            if "min_sigma_result" in params:
                self.min_sigma_result_spin.setValue(float(params["min_sigma_result"]))
            if "snap_enabled" in params:
                self.snap_enabled_check.setChecked(bool(params["snap_enabled"]))
            if "snap_radius" in params:
                self.snap_radius_spin.setValue(int(params["snap_radius"]))
            if "snap_refit_enabled" in params:
                self.snap_refit_check.setChecked(bool(params["snap_refit_enabled"]))
            if "refit_max_shift_px" in params:
                self.refit_shift_spin.setValue(int(params["refit_max_shift_px"]))
        finally:
            blockers.clear()

        self._update_detection_ui_enabled()
        self._update_init_mode_ui_enabled()

    def import_csv_restore(self) -> None:
        """
        Load exported spot CSVs and restore the spot state for display.
        If <base>_meta.json exists, restore ROI / UI params / analysis signatures too.
        """
        if not hasattr(gv, "files") or not gv.files or getattr(gv, "currentFileNum", -1) < 0:
            QtWidgets.QMessageBox.warning(self, "No File", "先にデータをロードしてください。")
            return
        if not hasattr(gv, "XScanSize") or not hasattr(gv, "YScanSize") or gv.XScanSize == 0 or gv.YScanSize == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Scan Size", "scan_sizeが0です。先に正しいデータを読み込んでください。")
            return
        if not hasattr(gv, "XPixel") or not hasattr(gv, "YPixel") or gv.XPixel == 0 or gv.YPixel == 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Pixel Size", "ピクセルサイズ情報が取得できません。")
            return

        if self.export_dir is None and hasattr(gv, "files") and gv.files:
            try:
                self.export_dir = os.path.dirname(gv.files[gv.currentFileNum])
            except Exception:
                pass

        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open spot CSV (base or pN)",
            self.export_dir or "",
            "CSV Files (*.csv)",
        )
        if not selected_path:
            return

        export_dir = os.path.dirname(selected_path)
        base_stub = os.path.splitext(os.path.basename(selected_path))[0]
        # allow selecting my_result_p1.csv etc.
        m = re.match(r"^(.*)_p\d+(?:_h)?(?:_\d+)?$", base_stub)
        if m:
            base_stub = m.group(1)
        if not base_stub:
            QtWidgets.QMessageBox.warning(self, "Invalid File", "ベース名が取得できません。")
            return

        nm_per_pixel_x = gv.XScanSize / gv.XPixel
        nm_per_pixel_y = gv.YScanSize / gv.YPixel

        restored = self._load_seed_spots_from_csv(export_dir, base_stub, nm_per_pixel_x, nm_per_pixel_y)
        if not restored:
            QtWidgets.QMessageBox.information(self, "No Data", "読み込めるスポットCSVが見つかりませんでした。")
            return

        # restore in-memory state (display)
        self.spots_by_frame = restored
        self.export_btn.setEnabled(True)
        self.export_dir = export_dir

        # load meta.json if exists
        meta_path = os.path.join(export_dir, f"{base_stub}_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.manual_roi = meta.get("manual_roi") or self.manual_roi
                roi_by_frame = meta.get("roi_by_frame")
                if isinstance(roi_by_frame, dict):
                    # json keys might be strings
                    self.roi_by_frame = {int(k): v for k, v in roi_by_frame.items()}
                ui_params = meta.get("ui_params")
                if isinstance(ui_params, dict):
                    self._apply_ui_params(ui_params)
                sig_map = meta.get("analysis_signature_by_frame")
                if isinstance(sig_map, dict):
                    def _to_tuple(obj):
                        if isinstance(obj, list):
                            return tuple(_to_tuple(v) for v in obj)
                        if isinstance(obj, dict):
                            return {k: _to_tuple(v) for k, v in obj.items()}
                        return obj
                    self._analysis_signature_by_frame = {int(k): _to_tuple(v) for k, v in sig_map.items()}
            except Exception:
                pass

        # ensure signatures exist for restored frames to prevent unwanted auto rerun
        for fi in list(self.spots_by_frame.keys()):
            if fi in self._analysis_signature_by_frame:
                continue
            roi_info = self.roi_by_frame.get(fi, self.manual_roi)
            if roi_info is None:
                continue
            try:
                self._analysis_signature_by_frame[int(fi)] = self._analysis_signature(int(fi), roi_info)
            except Exception:
                pass

        # refresh view
        frame = self._prepare_frame()
        if frame is not None:
            self.last_frame = frame
        try:
            self.roi_status_label.setText("ROI選択済み" if self.manual_roi is not None else "ROI未選択")
        except Exception:
            pass
        self._sync_run_buttons_enabled()
        self.show_full_image_view()
        self.export_btn.setEnabled(True)
        self._refresh_overlay()

    # backward-compatible alias
    def import_csv_and_rerun(self) -> None:
        return self.import_csv_restore()

    def _load_seed_spots_from_csv(
        self, export_dir: str, base_stub: str, nm_per_pixel_x: float, nm_per_pixel_y: float
    ) -> Dict[int, List[Dict[str, float]]]:
        """
        Load <base>_pN.csv and optionally <base>_pN_h.csv.
        Returns frame_index -> list of spot dicts (absolute px).
        """
        # Choose best candidate per pN: prefer no suffix, then lowest suffix
        pos_paths = glob.glob(os.path.join(export_dir, f"{base_stub}_p*.csv"))
        # exclude height files
        pos_paths = [p for p in pos_paths if not re.search(r"_p\d+_h(?:_\d+)?\.csv$", os.path.basename(p))]

        by_p: Dict[int, List[Tuple[int, str]]] = {}
        for path in pos_paths:
            name = os.path.basename(path)
            m = re.match(rf"^{re.escape(base_stub)}_p(\d+)(?:_(\d+))?\.csv$", name)
            if not m:
                continue
            p_idx = int(m.group(1))
            suf = int(m.group(2)) if m.group(2) is not None else 0
            by_p.setdefault(p_idx, []).append((suf, path))
        if not by_p:
            return {}

        best_pos_by_p: Dict[int, str] = {p: sorted(lst, key=lambda t: t[0])[0][1] for p, lst in by_p.items()}

        seed: Dict[int, List[Dict[str, float]]] = {}
        for p_idx, pos_path in best_pos_by_p.items():
            frame_to_spot_idx: Dict[int, int] = {}
            try:
                with open(pos_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            frame_i = int(float(row.get("frame_index", "nan")))
                            x_nm = float(row.get("x_nm", "nan"))
                            y_nm = float(row.get("y_nm", "nan"))
                        except Exception:
                            continue
                        if not np.isfinite(x_nm) or not np.isfinite(y_nm):
                            continue
                        x_px = x_nm / float(nm_per_pixel_x)
                        y_px = y_nm / float(nm_per_pixel_y)
                        lst = seed.setdefault(frame_i, [])
                        lst.append({"x": float(x_px), "y": float(y_px), "snr": 0.0})
                        frame_to_spot_idx[frame_i] = len(lst) - 1
            except Exception:
                continue

            # merge height if exists (prefer no suffix like position)
            base_name = os.path.splitext(os.path.basename(pos_path))[0]
            # base_name is <base>_pN or <base>_pN_k ; derive height file
            h_candidates = []
            for hp in glob.glob(os.path.join(export_dir, f"{base_name}_h*.csv")):
                h_candidates.append(hp)
            # fallback: any base_stub_pN_h*.csv
            if not h_candidates:
                h_candidates = glob.glob(os.path.join(export_dir, f"{base_stub}_p{p_idx}_h*.csv"))
            h_path = None
            if h_candidates:
                # prefer exact match without extra suffix
                h_candidates.sort(key=lambda p: len(os.path.basename(p)))
                h_path = h_candidates[0]
            if h_path:
                try:
                    with open(h_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                frame_i = int(float(row.get("frame_index", "nan")))
                            except Exception:
                                continue
                            # find the p_idx-th spot in this frame if exists
                            spots = seed.get(frame_i)
                            if not spots:
                                continue
                            spot_pos = frame_to_spot_idx.get(frame_i)
                            if spot_pos is None:
                                continue
                            try:
                                h_mean = float(row.get("height_mean_nm", "nan"))
                                h_bg = float(row.get("height_bg_nm", "nan"))
                                h_bgsub = float(row.get("height_bgsub_nm", "nan"))
                            except Exception:
                                continue
                            spots[spot_pos]["height_mean_nm"] = h_mean
                            spots[spot_pos]["height_bg_nm"] = h_bg
                            spots[spot_pos]["height_bgsub_nm"] = h_bgsub
                except Exception:
                    pass

        return seed
    # --- helper ---
    def _ensure_live_window(self, win, cls):
        """Qt側で破棄された場合に再生成する"""
        if win is not None and self._is_window_live(win):
            return win
        return cls(self)

    def _is_window_live(self, win) -> bool:
        try:
            _ = win.winId()  # RuntimeError if deleted
            return True
        except RuntimeError:
            return False

def create_plugin(main_window) -> QtWidgets.QWidget:
    """
    Pluginメニューから呼び出されるファクトリ。
    メインウィンドウを受け取り、SpotAnalysisWindowを返す。
    """
    return SpotAnalysisWindow(main_window)
