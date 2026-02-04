"""
ChugaiAnalysis
--------------
高速AFMの2Dフレーム列に対して、ROI内のみで以下を推定するプラグイン:

- 球状ドメイン（スポット）2個の座標とSNR、距離 d(t)、状態 S(t) ∈ {merged, separated, uncertain}
- IDR（天然変性領域）の“広がり”指標（面積・分位半径・Rg）

既存の SpotAnalysis.py のGUI/ワークフロー（PyQt5 + Matplotlib）を尊重し、
globalvals/fileio の依存（gv.currentFileNum, gv.index, gv.aryData 等）を壊さないように最小侵襲で拡張する。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage import feature, filters
from skimage import measure, morphology
from skimage.segmentation import watershed
from PyQt5 import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse, Circle

import globalvals as gv
from fileio import LoadFrame, InitializeAryDataFallback

logger = logging.getLogger(__name__)

# プラグイン表示名（Pluginメニューに表示される名前）
PLUGIN_NAME = "Chugai Analysis"


def _tt(en: str, jp: str) -> str:
    """
    Build a bilingual tooltip string.

    Format:
      EN: ...

      JP: ...
    """
    en_s = (en or "").strip()
    jp_s = (jp or "").strip()
    return f"EN: {en_s}\n\nJP: {jp_s}"


# ---- SpotAnalysis (copied & reused with minimal changes) ----

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
    peaks: List[PeakStat]


@dataclass
class FrameAnalysis:
    best_n_peaks: int
    criterion: str
    noise_sigma: float
    snr_threshold: float
    models: Dict[int, ModelSelectionResult]
    roi: np.ndarray  # 可視化用のROI画像（SpotAnalysisの前処理後）
    origin: Tuple[int, int]  # 元画像内でのROI起点 (x0, y0)
    roi_mask: Optional[np.ndarray] = None  # ROI内マスク（楕円など）


class SpotAnalysis:
    """
    SpotAnalysis.py と同等の 2D ガウス多峰フィット。
    本プラグインでは最大2スポット推定に流用する。
    """

    def __init__(
        self,
        roi_size: int = 48,
        bandpass_low_sigma: float = 0.6,
        bandpass_high_sigma: float = 3.0,
        peak_min_distance: int = 2,
        sigma_bounds: Tuple[float, float] = (0.6, 8.0),
        initial_sigma: float = 2.0,
        snr_threshold: float = 3.0,
        max_iterations: int = 8000,
    ) -> None:
        self.roi_size = max(8, int(roi_size))
        self.bandpass_low_sigma = float(bandpass_low_sigma)
        self.bandpass_high_sigma = float(bandpass_high_sigma)
        self.peak_min_distance = max(1, int(peak_min_distance))
        self.sigma_bounds = (max(1e-3, sigma_bounds[0]), max(sigma_bounds[0] + 1e-3, sigma_bounds[1]))
        self.initial_sigma = float(initial_sigma)
        self.snr_threshold = float(snr_threshold)
        self.max_iterations = max(1000, int(max_iterations))

    def analyze_frame(
        self,
        frame: np.ndarray,
        prev_center: Optional[Tuple[float, float]] = None,
        criterion: str = "aic",
        center_override: Optional[Tuple[float, float]] = None,
        roi_size_override: Optional[int] = None,
        roi_mask_override: Optional[np.ndarray] = None,
        roi_bounds_override: Optional[Tuple[int, int, int, int]] = None,
        min_peaks: int = 1,
        max_peaks: int = 2,
    ) -> FrameAnalysis:
        frame_f = np.asarray(frame, dtype=np.float64)
        filtered = self._preprocess(frame_f)

        use_roi_size = roi_size_override if roi_size_override is not None else self.roi_size
        center = center_override if center_override is not None else self._estimate_center(filtered, prev_center)
        if roi_bounds_override is not None:
            roi, origin = self._crop_rect(filtered, roi_bounds_override)
        else:
            roi, origin = self._crop_square(filtered, center, use_roi_size)

        roi_mask = roi_mask_override
        noise_sigma = self._estimate_noise_sigma(roi, roi_mask)
        models: Dict[int, ModelSelectionResult] = {}
        min_peaks = max(1, int(min_peaks))
        max_peaks = max(min_peaks, int(max_peaks))
        for n_peaks in range(min_peaks, max_peaks + 1):
            models[n_peaks] = self._fit_model(roi, origin, n_peaks, noise_sigma, roi_mask)

        use_bic = criterion.lower() == "bic"
        best = None
        best_score = None
        for n_peaks, model in models.items():
            score = model.bic if use_bic else model.aic
            if best_score is None or score < best_score:
                best_score = score
                best = n_peaks

        return FrameAnalysis(
            best_n_peaks=int(best),
            criterion="bic" if use_bic else "aic",
            noise_sigma=float(noise_sigma),
            snr_threshold=float(self.snr_threshold),
            models=models,
            roi=roi,
            origin=origin,
            roi_mask=roi_mask,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.bandpass_high_sigma <= self.bandpass_low_sigma:
            return frame
        return filters.difference_of_gaussians(frame, self.bandpass_low_sigma, self.bandpass_high_sigma)

    def _estimate_center(self, frame: np.ndarray, prev_center: Optional[Tuple[float, float]]) -> Tuple[float, float]:
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

    def _crop_square(self, frame: np.ndarray, center: Tuple[float, float], size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = frame.shape
        half = size // 2
        cx = int(round(center[0]))
        cy = int(round(center[1]))
        x0 = max(cx - half, 0)
        x1 = min(cx + half + 1, w)
        y0 = max(cy - half, 0)
        y1 = min(cy + half + 1, h)
        return frame[y0:y1, x0:x1], (x0, y0)

    def _crop_rect(self, frame: np.ndarray, bounds: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
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
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.std(data))
        return max(float(sigma), 1e-6)

    def _initial_params(self, frame: np.ndarray, n_peaks: int) -> List[float]:
        coords = feature.peak_local_max(
            frame,
            min_distance=self.peak_min_distance,
            num_peaks=n_peaks,
            exclude_border=False,
        )
        if coords.shape[0] < n_peaks:
            flat_idx = np.argsort(frame.ravel())[::-1][:n_peaks]
            ys, xs = np.unravel_index(flat_idx, frame.shape)
            coords = np.stack([ys, xs], axis=1)

        params: List[float] = []
        for y, x in coords[:n_peaks]:
            amp = max(float(frame[int(y), int(x)]), 1e-6)
            params.extend([amp, float(x), float(y), float(self.initial_sigma)])
        while len(params) < n_peaks * 4:
            params.extend([1.0, frame.shape[1] / 2.0, frame.shape[0] / 2.0, float(self.initial_sigma)])

        params.append(float(np.median(frame)))
        return params

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
    ) -> ModelSelectionResult:
        h, w = roi.shape
        x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        p0 = self._initial_params(roi, n_peaks)
        bounds = self._build_bounds(roi, n_peaks)

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

        sigma = max(float(noise_sigma), 1e-6)
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
                    snr=float(amp) / max(float(noise_sigma), 1e-6),
                )
            )

        return ModelSelectionResult(
            n_peaks=n_peaks,
            popt=popt,
            rss=rss,
            loglike=float(loglike),
            aic=float(aic),
            bic=float(bic),
            residual_std=float(residual_std),
            peaks=peaks,
        )


# ---- IDR Detector ----

@dataclass
class IDRMetrics:
    idr_area: float
    idr_rg: float
    idr_r80: float
    idr_r90: float


class IDRDetector:
    """
    ROI内でIDRを推定するクラス。DoGとSeeded segmentation（watershed）を実装し統合する。

    前提:
    - 高さは常に正で、明るい=高い（符号反転なし）
    - IDRはドメイン間を橋渡ししない（forbidden corridor で抑制）
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _estimate_noise_sigma(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        data = frame[mask] if mask is not None else frame.ravel()
        if data.size == 0:
            return 1e-6
        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.std(data))
        return max(float(sigma), 1e-6)

    @staticmethod
    def _safe_bool(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        if mask is None:
            return np.ones(shape, dtype=bool)
        return mask.astype(bool)

    @staticmethod
    def _postprocess_mask(mask: np.ndarray, min_area: int = 20, do_close: bool = True) -> np.ndarray:
        m = mask.astype(bool)
        if m.size == 0:
            return m
        try:
            m = morphology.remove_small_objects(m, min_size=int(min_area))
        except Exception:
            pass
        try:
            m = morphology.remove_small_holes(m, area_threshold=int(min_area))
        except Exception:
            pass
        if do_close:
            try:
                se = morphology.disk(1)
                m = morphology.binary_closing(m, se)
            except Exception:
                pass
        return m.astype(bool)

    @staticmethod
    def forbidden_corridor(
        shape: Tuple[int, int],
        spot1_xy: Optional[Tuple[float, float]],
        spot2_xy: Optional[Tuple[float, float]],
        width_px: float,
    ) -> np.ndarray:
        """2点を結ぶ線分への距離が width_px 未満の領域を True（禁止）にする。"""
        h, w = shape
        if spot1_xy is None or spot2_xy is None:
            return np.zeros((h, w), dtype=bool)
        x1, y1 = float(spot1_xy[0]), float(spot1_xy[1])
        x2, y2 = float(spot2_xy[0]), float(spot2_xy[1])
        if not np.isfinite(x1 + y1 + x2 + y2) or width_px <= 0:
            return np.zeros((h, w), dtype=bool)

        yy, xx = np.mgrid[0:h, 0:w]
        # 点Pから線分ABまでの距離（ベクトル投影で計算）
        ax, ay = x1, y1
        bx, by = x2, y2
        abx = bx - ax
        aby = by - ay
        apx = xx - ax
        apy = yy - ay
        denom = abx * abx + aby * aby + 1e-12
        t = (apx * abx + apy * aby) / denom
        t = np.clip(t, 0.0, 1.0)
        projx = ax + t * abx
        projy = ay + t * aby
        dist = np.sqrt((xx - projx) ** 2 + (yy - projy) ** 2)
        return dist < float(width_px)

    def mask_dog(
        self,
        roi: np.ndarray,
        roi_mask: Optional[np.ndarray],
        sigma_bg: float,
        dog_scales: Sequence[float],
        k: float,
        require_positive: bool = True,
        keep_connected_to_domain: bool = False,
        domain_points_xy: Optional[List[Tuple[float, float]]] = None,
        domain_attach_r: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        DoGベースのIDRマスク。返り値: (mask, score_map)
        - score_map は dog_max（正側）
        """
        img = np.asarray(roi, dtype=np.float64)
        roi_m = self._safe_bool(roi_mask, img.shape)

        # 背景推定→減算（正側前提なのでclipはしない）
        if sigma_bg > 0:
            bg = ndimage.gaussian_filter(img, sigma=float(sigma_bg))
        else:
            bg = np.zeros_like(img)
        bgsub = img - bg

        # multi-scale DoG（正側のみを採用）
        dog_max = np.zeros_like(bgsub, dtype=np.float64)
        for s in dog_scales:
            s = float(s)
            if not np.isfinite(s) or s <= 0:
                continue
            # high sigma は低sigmaの比で固定（簡便）
            resp = filters.difference_of_gaussians(bgsub, s, s * 1.6)
            if require_positive:
                resp = np.clip(resp, 0.0, None)
            dog_max = np.maximum(dog_max, resp)
        dog_max[~roi_m] = 0.0

        noise_sigma = self._estimate_noise_sigma(dog_max, roi_m)
        thr = float(k) * float(noise_sigma)
        mask = (dog_max > thr) & roi_m

        mask = self._postprocess_mask(mask, min_area=20, do_close=True)

        # ドメイン連結成分のみ採用（任意）
        if keep_connected_to_domain and domain_points_xy:
            try:
                lab = measure.label(mask, connectivity=2)
                if lab.max() > 0:
                    keep = np.zeros(lab.max() + 1, dtype=bool)
                    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
                    for (cx, cy) in domain_points_xy:
                        rr2 = float(domain_attach_r) ** 2
                        near = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= rr2
                        ids = np.unique(lab[near & (lab > 0)])
                        keep[ids] = True
                    mask = keep[lab]
            except Exception:
                pass
        return mask.astype(bool), dog_max

    def mask_rw(
        self,
        roi: np.ndarray,
        roi_mask: Optional[np.ndarray],
        center_xy: Tuple[float, float],
        ring_r: float,
        ring_w: float,
        bg_dist: float,
        sigma_bg: float,
        require_positive: bool = True,
    ) -> np.ndarray:
        """
        Seeded segmentation（マーカー付きwatershed）でIDRマスクを返す。
        失敗時は例外を投げる（呼び出し側でDoGにフォールバックする）。
        """
        img = np.asarray(roi, dtype=np.float64)
        roi_m = self._safe_bool(roi_mask, img.shape)
        h, w = img.shape
        cx, cy = float(center_xy[0]), float(center_xy[1])
        if not np.isfinite(cx + cy):
            cx, cy = w / 2.0, h / 2.0

        if sigma_bg > 0:
            bg = ndimage.gaussian_filter(img, sigma=float(sigma_bg))
        else:
            bg = np.zeros_like(img)
        bgsub = img - bg
        if require_positive:
            bgsub = np.clip(bgsub, 0.0, None)

        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        r0 = max(0.0, float(ring_r))
        r1 = max(r0 + 1e-6, float(ring_r) + max(0.0, float(ring_w)))
        seed_fg = (rr >= r0) & (rr <= r1) & roi_m

        # BG seed: 外周 or 十分遠方
        border = np.zeros_like(roi_m, dtype=bool)
        border[:2, :] = True
        border[-2:, :] = True
        border[:, :2] = True
        border[:, -2:] = True
        seed_bg = (border | (rr >= float(bg_dist))) & roi_m

        # マーカー生成
        markers = np.zeros_like(img, dtype=np.int32)
        markers[seed_bg] = 2
        markers[seed_fg] = 1
        if np.count_nonzero(markers == 1) < 5 or np.count_nonzero(markers == 2) < 5:
            raise RuntimeError("RW markers are too small")

        # watershed: 高い=IDR側に寄るように、-bgsub を地形として使用
        elevation = ndimage.gaussian_filter(-bgsub, sigma=1.0)
        labels = watershed(elevation, markers=markers, mask=roi_m)
        mask = labels == 1
        mask = self._postprocess_mask(mask, min_area=20, do_close=True)
        return mask.astype(bool)

    def combine(
        self,
        mode: str,
        dog_mask: np.ndarray,
        rw_mask: Optional[np.ndarray],
        dog_score: Optional[np.ndarray],
        center_xy: Tuple[float, float],
        extend_r: float,
        forbidden: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        m = mode.lower().strip()
        if m == "dog":
            out = dog_mask.astype(bool)
        elif m == "rw":
            out = (rw_mask.astype(bool) if rw_mask is not None else dog_mask.astype(bool))
        elif m == "union":
            out = dog_mask.astype(bool) | (rw_mask.astype(bool) if rw_mask is not None else False)
        elif m == "intersection":
            out = dog_mask.astype(bool) & (rw_mask.astype(bool) if rw_mask is not None else False)
        elif m == "rw_plus_dog":
            core = (rw_mask.astype(bool) if rw_mask is not None else dog_mask.astype(bool))
            if dog_score is None:
                out = core
            else:
                h, w = dog_score.shape
                cx, cy = float(center_xy[0]), float(center_xy[1])
                yy, xx = np.mgrid[0:h, 0:w]
                rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                out = core | (dog_mask.astype(bool) & (rr <= float(extend_r)))
        else:
            out = dog_mask.astype(bool)

        if forbidden is not None:
            out = out & (~forbidden.astype(bool))
        return out.astype(bool)

    def compute_metrics(
        self,
        mask: np.ndarray,
        center_xy: Tuple[float, float],
        intensity: Optional[np.ndarray] = None,
        weighted: bool = False,
    ) -> IDRMetrics:
        m = mask.astype(bool)
        if m.size == 0 or np.count_nonzero(m) == 0:
            nan = float("nan")
            return IDRMetrics(idr_area=0.0, idr_rg=nan, idr_r80=nan, idr_r90=nan)
        h, w = m.shape
        cx, cy = float(center_xy[0]), float(center_xy[1])
        if not np.isfinite(cx + cy):
            cx, cy = w / 2.0, h / 2.0
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r = rr[m]
        if r.size == 0:
            nan = float("nan")
            return IDRMetrics(idr_area=0.0, idr_rg=nan, idr_r80=nan, idr_r90=nan)

        idr_area = float(np.count_nonzero(m))
        idr_r80 = float(np.quantile(r, 0.80))
        idr_r90 = float(np.quantile(r, 0.90))

        if weighted and intensity is not None:
            inten = np.asarray(intensity, dtype=np.float64)
            wgt = np.clip(inten[m], 0.0, None)
            s = float(np.sum(wgt))
            if s > 0:
                idr_rg = float(np.sqrt(np.sum(wgt * (r**2)) / s))
            else:
                idr_rg = float(np.sqrt(np.mean(r**2)))
        else:
            idr_rg = float(np.sqrt(np.mean(r**2)))

        return IDRMetrics(idr_area=idr_area, idr_rg=idr_rg, idr_r80=idr_r80, idr_r90=idr_r90)


@dataclass
class ChugaiFrameResult:
    frame_index: int
    x1: float
    y1: float
    snr1: float
    x2: float
    y2: float
    snr2: float
    d: float
    state: str
    idr_area: float
    idr_rg: float
    idr_r80: float
    idr_r90: float
    idr_mode: str
    params_json: str
    idr_mask_full: Optional[np.ndarray] = None
    idr_overlay_full: Optional[np.ndarray] = None


# ---- Visualization window (full image overlay) ----

class ChugaiFullImageWindow(QtWidgets.QMainWindow):
    """
    全画像にspot/ROI/IDRをオーバーレイ表示するウィンドウ。
    SpotAnalysis.py の挙動を踏襲しつつ、IDR描画を追加。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chugai Analysis - Full Image Overlay")
        self.resize(720, 640)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        # Matplotlib canvas がキー入力を受け取れるようにする（IキーでIDR編集モード切替）
        try:
            self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
            self.canvas.setFocus()
        except Exception:
            pass
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        # Reserve a white margin above the axes for instructions text.
        try:
            self.figure.subplots_adjust(top=0.76)
        except Exception:
            pass

        # Help/instructions text shown in the top white margin (above the title).
        self._help_text_artist = None
        self.cbar = None
        self.selector = None
        self.on_select_callback = None
        self.edit_handler = None
        self._dragging = False
        self._drag_index = None
        self._roi_selector_paused = False
        # IDR manual edit (cutline)
        self._idr_edit_mode = False
        self._idr_selector_paused = False
        self._idr_is_drawing = False
        self._idr_points: List[Tuple[float, float]] = []
        self._idr_line_artist = None
        # submode: "cut" or "connect"
        self._idr_edit_action = "cut"
        self._idr_cutline_callback = None
        self._idr_connectline_callback = None

        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _build_help_text(self) -> str:
        # Keep this concise and scannable (shown in a narrow top margin).
        mode = "ON" if self._idr_edit_mode else "OFF"
        action = str(self._idr_edit_action).lower()
        bullet = "•"
        lines = [
            f"{bullet} Move spot: Command+Drag",
            f"{bullet} Add spot: Shift+Click",
            f"{bullet} Delete spot: Option+Click",
            f"{bullet} IDR edit mode: I (currently {mode}, action={action})",
            f"{bullet} Switch IDR action (Cut/Connect): C",
            f"{bullet} Draw IDR edit line: Left-Drag + Release",
            f"{bullet} Cut IDR: split/remove along the line",
            f"{bullet} Connect IDR: bridge regions (endpoints on/near IDR)",
        ]
        return "\n".join(lines)

    def _update_help_text(self) -> None:
        """
        Render operation instructions in the white margin above the axes title.
        """
        txt = self._build_help_text()
        if self._help_text_artist is None:
            try:
                self._help_text_artist = self.figure.text(
                    0.01,
                    0.99,
                    txt,
                    ha="left",
                    va="top",
                    fontsize=9,
                    family="sans-serif",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.95),
                )
            except Exception:
                self._help_text_artist = None
                return
        else:
            try:
                self._help_text_artist.set_text(txt)
            except Exception:
                pass

    def set_edit_handler(self, handler):
        self.edit_handler = handler

    def set_idr_cutline_callback(self, callback):
        """IDRの手描き編集（カットライン）確定時に呼ぶコールバックを設定する。"""
        self._idr_cutline_callback = callback

    def set_idr_connectline_callback(self, callback):
        """IDRの手描き編集（接続ライン）確定時に呼ぶコールバックを設定する。"""
        self._idr_connectline_callback = callback

    # backward compatible name (older plan/impl)
    def set_idr_lasso_callback(self, callback):
        self.set_idr_cutline_callback(callback)

    def enable_roi_selector(self, shape: str, callback):
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
                callback({"shape": "Rectangle", "x0": xmin, "y0": ymin, "w": w, "h": h})

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
                callback(
                    {
                        "shape": "Ellipse",
                        "cx": cx,
                        "cy": cy,
                        "rx": rx,
                        "ry": ry,
                        "x0": xmin,
                        "y0": ymin,
                        "w": xmax - xmin,
                        "h": ymax - ymin,
                    }
                )

        if shape == "Ellipse (Circle)":
            self.selector = EllipseSelector(
                self.ax,
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
                self.ax,
                _emit_rect,
                useblit=True,
                button=[1],
                minspanx=2,
                minspany=2,
                interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )

        # IDR編集モード中にROIセレクタを作り直した場合も、編集優先で無効化しておく
        if self._idr_edit_mode and self.selector is not None:
            try:
                self.selector.set_active(False)
                self._idr_selector_paused = True
            except Exception:
                pass

    def _set_idr_edit_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._idr_edit_mode = enabled

        # ROI selector を一時停止（既存のCtrl/Cmdと同様）
        if enabled:
            if self.selector is not None:
                try:
                    self.selector.set_active(False)
                    self._idr_selector_paused = True
                except Exception:
                    pass
        else:
            if self._idr_selector_paused and self.selector is not None:
                try:
                    self.selector.set_active(True)
                except Exception:
                    pass
            self._idr_selector_paused = False
        # ドラッグ中であれば中断
        if not enabled:
            self._cancel_idr_drawing()

        # 画面上での切替が分かるようにタイトルを軽く更新（表示は任意）
        try:
            if enabled:
                self.ax.set_title(
                    f"Chugai Analysis - Full Image (IDR Edit Mode: ON ({self._idr_edit_action}))",
                    pad=2.0,
                )
            else:
                self.ax.set_title("Chugai Analysis - Full Image", pad=2.0)
            # keep help text in sync
            self._update_help_text()
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_key_press(self, event) -> None:
        try:
            k = (event.key or "").lower()
        except Exception:
            k = ""
        if k == "i":
            self._set_idr_edit_mode(not self._idr_edit_mode)
        elif k == "c":
            # サブモード切替（cut ⇄ connect）
            self._idr_edit_action = "connect" if self._idr_edit_action == "cut" else "cut"
            if self._idr_edit_mode:
                try:
                    self.ax.set_title(
                        f"Chugai Analysis - Full Image (IDR Edit Mode: ON ({self._idr_edit_action}))",
                        pad=2.0,
                    )
                    self._update_help_text()
                    self.canvas.draw_idle()
                except Exception:
                    pass

    def _cancel_idr_drawing(self) -> None:
        self._idr_is_drawing = False
        self._idr_points = []
        if self._idr_line_artist is not None:
            try:
                self._idr_line_artist.remove()
            except Exception:
                pass
        self._idr_line_artist = None
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _start_idr_cutline(self, x: float, y: float) -> None:
        self._cancel_idr_drawing()
        self._idr_is_drawing = True
        self._idr_points = [(float(x), float(y))]
        try:
            (line,) = self.ax.plot([x], [y], color="cyan", linewidth=2.0, alpha=0.9)
            self._idr_line_artist = line
            self.canvas.draw_idle()
        except Exception:
            self._idr_line_artist = None

    def _append_idr_point(self, x: float, y: float, min_step_px: float = 0.75) -> None:
        if not self._idr_is_drawing:
            return
        x = float(x)
        y = float(y)
        if self._idr_points:
            px, py = self._idr_points[-1]
            if (x - px) ** 2 + (y - py) ** 2 < float(min_step_px) ** 2:
                return
        self._idr_points.append((x, y))
        if self._idr_line_artist is not None:
            try:
                xs = [p[0] for p in self._idr_points]
                ys = [p[1] for p in self._idr_points]
                self._idr_line_artist.set_data(xs, ys)
                self.canvas.draw_idle()
            except Exception:
                pass

    def _finish_idr_cutline(self) -> None:
        if not self._idr_is_drawing:
            return
        pts = list(self._idr_points)
        self._cancel_idr_drawing()
        if len(pts) < 2:
            return
        # サブモードに応じてcallbackを振り分け
        if self._idr_edit_action == "connect":
            cb = self._idr_connectline_callback
        else:
            cb = self._idr_cutline_callback
        if cb is None:
            return
        try:
            cb(pts)
        except Exception:
            pass

    def update_view(
        self,
        frame: np.ndarray,
        spots: Optional[List[Dict[str, float]]] = None,
        roi_overlay: Optional[Dict[str, float]] = None,
        spot_radius_px: Optional[float] = None,
        idr_mask_full: Optional[np.ndarray] = None,
        idr_overlay_full: Optional[np.ndarray] = None,
        idr_center: Optional[Tuple[float, float]] = None,
        idr_r80: Optional[float] = None,
        idr_r90: Optional[float] = None,
        idr_vis: Optional[Dict[str, object]] = None,
    ) -> None:
        """全画像とオーバーレイを描画。"""
        self.ax.clear()
        self._safe_remove_colorbar()
        self.ax.imshow(frame, cmap="viridis", origin="lower")

        # IDR overlay（frame → IDR → spots → ROI）
        if idr_mask_full is not None and idr_vis:
            try:
                show_mask = bool(idr_vis.get("show_mask", True))
                show_contour = bool(idr_vis.get("show_contour", False))
                show_r80 = bool(idr_vis.get("show_r80", False))
                show_r90 = bool(idr_vis.get("show_r90", False))
                alpha = float(idr_vis.get("alpha", 0.35))
                color = str(idr_vis.get("color", "magenta"))
                m = np.asarray(idr_mask_full, dtype=bool)
                if show_mask and m.shape == frame.shape:
                    overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
                    if color.lower() in ("red",):
                        overlay[..., 0] = 1.0
                    else:
                        overlay[..., 0] = 1.0
                        overlay[..., 2] = 1.0
                    overlay[..., 3] = alpha * m.astype(np.float32)
                    self.ax.imshow(overlay, origin="lower")
                if show_contour and m.shape == frame.shape:
                    self.ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=1.0, alpha=0.9)
                if idr_center is not None:
                    cx, cy = float(idr_center[0]), float(idr_center[1])
                    if show_r80 and idr_r80 is not None and np.isfinite(float(idr_r80)):
                        self.ax.add_patch(Circle((cx, cy), radius=float(idr_r80), fill=False, edgecolor=color, linewidth=1.0, alpha=0.9))
                    if show_r90 and idr_r90 is not None and np.isfinite(float(idr_r90)):
                        self.ax.add_patch(Circle((cx, cy), radius=float(idr_r90), fill=False, edgecolor=color, linewidth=1.0, alpha=0.9, linestyle="--"))
            except Exception:
                pass
        elif idr_overlay_full is not None and idr_vis:
            # score/overlay map 表示（maskが無い場合の代替）
            try:
                alpha = float(idr_vis.get("alpha", 0.35))
                score = np.asarray(idr_overlay_full, dtype=np.float64)
                if score.shape == frame.shape and np.any(np.isfinite(score)):
                    s = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
                    vmax = float(np.max(s)) if s.size else 0.0
                    if vmax > 0:
                        s = s / vmax
                    rgba = plt.get_cmap("magma")(np.clip(s, 0.0, 1.0))
                    rgba[..., 3] = alpha * np.clip(s, 0.0, 1.0)
                    self.ax.imshow(rgba, origin="lower")
            except Exception:
                pass

        if spots:
            colors = ["magenta", "white", "orange"]
            for i, pk in enumerate(spots):
                label = f"P{i+1}: S/N={pk.get('snr', 0.0):.1f}"
                self.ax.plot(pk["x"], pk["y"], "x", color=colors[i % len(colors)], markersize=10, markeredgewidth=2, label=label)
                self.ax.text(pk["x"] + 1, pk["y"] + 1, f"P{i+1}", color="white", fontsize=10, fontweight="bold")
                if spot_radius_px is not None and spot_radius_px > 0:
                    try:
                        self.ax.add_patch(
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
            self.ax.legend(loc="upper right", fontsize=8)

        if roi_overlay:
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
                self.ax.add_patch(ellipse)
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
                self.ax.add_patch(rect)

        if self._idr_edit_mode:
            self.ax.set_title(
                f"Chugai Analysis - Full Image (IDR Edit Mode: ON ({self._idr_edit_action}))",
                pad=2.0,
            )
        else:
            self.ax.set_title("Chugai Analysis - Full Image", pad=2.0)
        # Instructions text in the top white margin (above the title)
        self._update_help_text()
        self.ax.set_xlabel("X (pixel)")
        self.ax.set_ylabel("Y (pixel)")
        self.canvas.draw()

    def _safe_remove_colorbar(self):
        try:
            if self.cbar:
                self.cbar.remove()
        except Exception:
            pass
        self.cbar = None

    def _on_press(self, event):
        # キー入力フォーカス確保
        try:
            self.canvas.setFocus()
        except Exception:
            pass

        # IDR編集モード中はカットライン描画を優先
        if self._idr_edit_mode:
            try:
                if event.inaxes is None or event.xdata is None or event.ydata is None:
                    return
                if getattr(event, "button", None) != 1:
                    return
                self._start_idr_cutline(event.xdata, event.ydata)
            except Exception:
                return
            return
        if self.edit_handler is None:
            return
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
        if self._idr_edit_mode:
            try:
                if event.inaxes is None or event.xdata is None or event.ydata is None:
                    return
                self._append_idr_point(event.xdata, event.ydata)
            except Exception:
                return
            return
        if self.edit_handler is None:
            return
        self.edit_handler(event, "move")

    def _on_release(self, event):
        if self._idr_edit_mode:
            try:
                self._finish_idr_cutline()
            except Exception:
                pass
            return
        if self.edit_handler is None:
            return
        self.edit_handler(event, "release")
        if self._roi_selector_paused and self.selector is not None:
            try:
                self.selector.set_active(True)
            except Exception:
                pass
        self._roi_selector_paused = False


# ---- Main Window ----

class ChugaiAnalysisWindow(QtWidgets.QWidget):
    """
    SpotAnalysisWindow 相当のUI。既存操作（ROI手動設定、spot編集、フレーム遷移）を維持し、
    Chugai仕様（2スポット状態判定、IDR）を追加する。
    """

    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Chugai Analysis")
        self.setMinimumWidth(460)

        self.spot_analyzer = SpotAnalysis()
        self.idr_detector = IDRDetector()

        # パネルが縦長になりやすいため、スクロール可能にする
        outer_layout = QtWidgets.QVBoxLayout(self)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer_layout.addWidget(self._scroll)
        self._content = QtWidgets.QWidget()
        self._scroll.setWidget(self._content)

        self.full_viz_window = None
        self.last_frame: Optional[np.ndarray] = None
        self.manual_roi: Optional[Dict[str, float]] = None
        self.roi_by_frame: Dict[int, Dict[str, float]] = {}
        self.spots_by_frame: Dict[int, List[Dict[str, float]]] = {}
        self.results_by_frame: Dict[int, ChugaiFrameResult] = {}
        # スポットが手動編集されたフレームを記録（SNR無視のstate判定に切替えるため）
        self._last_edit_was_manual_by_frame: Dict[int, bool] = {}
        self.export_dir = None
        self._dragging = False
        self._drag_index = None
        self._auto_busy = False

        self._build_ui(self._content)
        self._refresh_selection_label()
        self._connect_frame_signal()
        self.show_full_image_view()
        self._update_frame_label()

    # ---- UI ----
    def _build_ui(self, container: QtWidgets.QWidget) -> None:
        layout = QtWidgets.QVBoxLayout(container)
        desc = QtWidgets.QLabel(
            "Estimate 2-spot states (merged/separated) and IDR metrics (area, quantile radii, Rg) within the ROI only."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        form = QtWidgets.QFormLayout()

        self.roi_shape_combo = QtWidgets.QComboBox()
        self.roi_shape_combo.addItems(["Rectangle", "Ellipse (Circle)"])
        self.roi_shape_combo.currentTextChanged.connect(self._on_roi_shape_changed)
        form.addRow("ROI shape", self.roi_shape_combo)
        self.roi_shape_combo.setToolTip(
            _tt(
                "Select the ROI geometry.\n"
                "- Rectangle: use all pixels inside the ROI bounds.\n"
                "- Ellipse (Circle): apply an elliptical mask inside the bounds.\n"
                "Spot fitting and IDR detection use only pixels inside the ROI (and mask).",
                "ROIの形状を選択します。\n"
                "- Rectangle: ROI範囲内の全画素を使用します。\n"
                "- Ellipse (Circle): ROI範囲内に楕円マスクを適用します。\n"
                "スポットフィットとIDR検出は、ROI（およびマスク）内の画素のみを使用します。",
            )
        )

        self.auto_analyze_check = QtWidgets.QCheckBox("Auto-analyze on frame change")
        form.addRow("Auto analyze", self.auto_analyze_check)
        self.auto_analyze_check.setToolTip(
            _tt(
                "If enabled, analysis is automatically executed when the frame changes.\n"
                "This is convenient for browsing, but can slow down playback.\n"
                "When enabled, ROI can be propagated from previous frames if available.",
                "有効にすると、フレームを切り替えたタイミングで自動的に解析を実行します。\n"
                "閲覧が便利になる一方、再生/スクロールが遅くなる可能性があります。\n"
                "有効時は、ROIが存在する場合に前フレームのROIを引き継ぐことがあります。",
            )
        )

        self.criterion_combo = QtWidgets.QComboBox()
        self.criterion_combo.addItems(["AIC", "BIC"])
        form.addRow("Model criterion", self.criterion_combo)
        self.criterion_combo.setToolTip(
            _tt(
                "Model selection criterion for choosing 1-spot vs 2-spot fit.\n"
                "- AIC: tends to select more complex models (more sensitive).\n"
                "- BIC: more conservative (reduces false 2-spot detections).\n"
                "If you often see a spurious second peak, try BIC.",
                "1スポット/2スポットのモデル選択に使う情報量基準です。\n"
                "- AIC: 複雑なモデルを選びやすい（感度高め）。\n"
                "- BIC: より保守的（偽の2スポットを減らしやすい）。\n"
                "2つ目の偽ピークが出やすい場合はBICを試してください。",
            )
        )

        # Spot params（SpotAnalysisと同等）
        self.bandpass_low_spin = QtWidgets.QDoubleSpinBox()
        self.bandpass_low_spin.setRange(0.1, 20.0)
        self.bandpass_low_spin.setSingleStep(0.1)
        self.bandpass_low_spin.setValue(self.spot_analyzer.bandpass_low_sigma)
        form.addRow("Bandpass low σ", self.bandpass_low_spin)
        self.bandpass_low_spin.setToolTip(
            _tt(
                "Bandpass filter (Difference of Gaussians) low σ.\n"
                "Increasing this value suppresses high-frequency noise (small speckles).\n"
                "Must be smaller than the high σ.",
                "バンドパス（DoG: Difference of Gaussians）フィルタの low σ です。\n"
                "値を上げると高周波ノイズ（細かな粒状ノイズ）を抑えやすくなります。\n"
                "High σ より小さくする必要があります。",
            )
        )

        self.bandpass_high_spin = QtWidgets.QDoubleSpinBox()
        self.bandpass_high_spin.setRange(0.1, 50.0)
        self.bandpass_high_spin.setSingleStep(0.1)
        self.bandpass_high_spin.setValue(self.spot_analyzer.bandpass_high_sigma)
        form.addRow("Bandpass high σ", self.bandpass_high_spin)
        self.bandpass_high_spin.setToolTip(
            _tt(
                "Bandpass filter (Difference of Gaussians) high σ.\n"
                "Increasing this value suppresses slowly varying background/gradients.\n"
                "Must be larger than the low σ.",
                "バンドパス（DoG）フィルタの high σ です。\n"
                "値を上げると、ゆっくり変化する背景（勾配やムラ）を抑えやすくなります。\n"
                "Low σ より大きくする必要があります。",
            )
        )

        self.peak_min_distance_spin = QtWidgets.QSpinBox()
        self.peak_min_distance_spin.setRange(1, 50)
        self.peak_min_distance_spin.setValue(self.spot_analyzer.peak_min_distance)
        form.addRow("Peak min distance (px)", self.peak_min_distance_spin)
        self.peak_min_distance_spin.setToolTip(
            _tt(
                "Minimum allowed distance between peak candidates (in pixels).\n"
                "Larger values discourage detecting two very-close peaks inside one blob.\n"
                "If you frequently get a false P2 very near P1, increase this.",
                "ピーク候補同士の最小距離（px）です。\n"
                "値を大きくすると、1つの塊の中で近接した2ピークが立ちにくくなります。\n"
                "P1近傍に偽のP2が出やすい場合は上げてください。",
            )
        )

        self.initial_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.initial_sigma_spin.setRange(0.1, 20.0)
        self.initial_sigma_spin.setSingleStep(0.1)
        self.initial_sigma_spin.setValue(self.spot_analyzer.initial_sigma)
        form.addRow("Initial σ", self.initial_sigma_spin)
        self.initial_sigma_spin.setToolTip(
            _tt(
                "Initial σ (starting guess) for Gaussian peak fitting.\n"
                "This affects convergence speed/stability but not the final bounds.\n"
                "If fitting fails or becomes unstable, try a value closer to the spot size.",
                "ガウスピークフィットの初期値 σ（開始推定値）です。\n"
                "収束の速さや安定性に影響します（最終的にはσの上下限で制約されます）。\n"
                "フィットが不安定/失敗する場合は、スポットサイズに近い値を試してください。",
            )
        )

        self.sigma_min_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_min_spin.setRange(0.05, 50.0)
        self.sigma_min_spin.setSingleStep(0.05)
        self.sigma_min_spin.setValue(self.spot_analyzer.sigma_bounds[0])
        form.addRow("σ min", self.sigma_min_spin)
        self.sigma_min_spin.setToolTip(
            _tt(
                "Lower bound for fitted Gaussian σ.\n"
                "Use this to prevent unrealistically sharp peaks.\n"
                "Too large a value can force poor fits for small spots.",
                "フィットされるガウスσの下限です。\n"
                "過度に鋭い（細すぎる）ピークを防ぐために使います。\n"
                "大きすぎると小さなスポットのフィットが悪化します。",
            )
        )

        self.sigma_max_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_max_spin.setRange(0.1, 80.0)
        self.sigma_max_spin.setSingleStep(0.1)
        self.sigma_max_spin.setValue(self.spot_analyzer.sigma_bounds[1])
        form.addRow("σ max", self.sigma_max_spin)
        self.sigma_max_spin.setToolTip(
            _tt(
                "Upper bound for fitted Gaussian σ.\n"
                "Use this to prevent overly broad peaks absorbing background.\n"
                "Too small a value can prevent fitting large/blurred spots.",
                "フィットされるガウスσの上限です。\n"
                "背景まで吸収するような広すぎるピークを防ぐために使います。\n"
                "小さすぎると大きい/ぼけたスポットのフィットができなくなります。",
            )
        )

        # Spot表示半径（既存と同じ）
        self.spot_radius_spin = QtWidgets.QSpinBox()
        self.spot_radius_spin.setRange(1, 200)
        self.spot_radius_spin.setValue(4)
        self.spot_radius_spin.valueChanged.connect(self._refresh_overlay)
        form.addRow("Spot radius (px)", self.spot_radius_spin)
        self.spot_radius_spin.setToolTip(
            _tt(
                "Visualization only: radius of the circles drawn for P1/P2 (in pixels).\n"
                "This does not affect fitting or IDR computation.",
                "表示のみ: P1/P2の円マーカー半径（px）です。\n"
                "フィット結果やIDR計算には影響しません。",
            )
        )

        # 状態判定
        self.snr_min_spin = QtWidgets.QDoubleSpinBox()
        self.snr_min_spin.setRange(0.0, 50.0)
        self.snr_min_spin.setSingleStep(0.1)
        self.snr_min_spin.setValue(2.0)
        form.addRow("Min SNR (2-spot decision)", self.snr_min_spin)
        self.snr_min_spin.setToolTip(
            _tt(
                "Minimum SNR required to treat a detected second peak as a real second spot.\n"
                "- If the 2nd spot SNR is below this value, it is dropped and treated as 1-spot.\n"
                "- State 'separated' requires both SNRs >= this threshold (and d > separated threshold).\n"
                "Higher values reduce false P2 but may miss a dim real second spot.",
                "2つ目ピークを「実在する2スポット」とみなすための最小SNRです。\n"
                "- 2個目のSNRがこの値未満なら、2スポット→1スポットへ落とします。\n"
                "- 'separated' は両スポットのSNRがこの閾値以上（かつ距離がseparated閾値超）で成立します。\n"
                "値を上げると偽P2は減りますが、暗い実スポットを見落とす可能性があります。",
            )
        )

        self.d_merge_spin = QtWidgets.QDoubleSpinBox()
        self.d_merge_spin.setRange(0.0, 5000.0)
        self.d_merge_spin.setSingleStep(0.5)
        self.d_merge_spin.setValue(3.0)
        form.addRow("Merged threshold d < (px)", self.d_merge_spin)
        self.d_merge_spin.setToolTip(
            _tt(
                "Distance threshold (pixels) to classify the state as 'merged'.\n"
                "If the P1–P2 distance d is smaller than this, the frame is labeled merged.",
                "状態を 'merged'（合体）と判定する距離閾値（px）です。\n"
                "P1–P2距離 d がこの値より小さい場合、そのフレームを merged とします。",
            )
        )

        self.d_sep_spin = QtWidgets.QDoubleSpinBox()
        self.d_sep_spin.setRange(0.0, 5000.0)
        self.d_sep_spin.setSingleStep(0.5)
        self.d_sep_spin.setValue(6.0)
        form.addRow("Separated threshold d > (px)", self.d_sep_spin)
        self.d_sep_spin.setToolTip(
            _tt(
                "Distance threshold (pixels) to classify the state as 'separated'.\n"
                "If d is larger than this AND both SNRs pass the minimum, the frame is labeled separated.\n"
                "Otherwise the state becomes 'uncertain' (unless merged threshold applies).",
                "状態を 'separated'（解離）と判定する距離閾値（px）です。\n"
                "d がこの値より大きく、かつ両スポットSNRが最小SNRを満たす場合に separated とします。\n"
                "それ以外は（merged条件に当てはまらない限り）uncertain になります。",
            )
        )

        # IDR controls
        self.idr_enable_check = QtWidgets.QCheckBox("Enable IDR detection")
        self.idr_enable_check.setChecked(True)
        form.addRow("IDR", self.idr_enable_check)
        self.idr_enable_check.setToolTip(
            _tt(
                "Enable/disable IDR (intrinsically disordered region) detection and metrics.\n"
                "Turning this off can make analysis faster and avoids IDR overlays/CSV fields.",
                "IDR（天然変性領域）検出とメトリクス計算の有効/無効を切り替えます。\n"
                "無効にすると解析が軽くなり、IDRの表示/CSV項目も無効になります。",
            )
        )

        self.idr_mode_combo = QtWidgets.QComboBox()
        self.idr_mode_combo.addItems(["dog", "rw", "union", "intersection", "rw_plus_dog"])
        form.addRow("IDR mode", self.idr_mode_combo)
        self.idr_mode_combo.setToolTip(
            _tt(
                "How to build the IDR mask.\n"
                "- dog: DoG-based segmentation.\n"
                "- rw: seeded segmentation (watershed-like).\n"
                "- union: combine dog OR rw.\n"
                "- intersection: dog AND rw.\n"
                "- rw_plus_dog: start from rw and expand using dog/extension settings.",
                "IDRマスクの作り方を選びます。\n"
                "- dog: DoGベースの分割。\n"
                "- rw: シード付き分割（watershed相当）。\n"
                "- union: dog と rw のOR。\n"
                "- intersection: dog と rw のAND。\n"
                "- rw_plus_dog: rwを基にdog/拡張設定で広げます。",
            )
        )

        self.idr_sigma_bg_spin = QtWidgets.QDoubleSpinBox()
        self.idr_sigma_bg_spin.setRange(0.0, 200.0)
        self.idr_sigma_bg_spin.setSingleStep(0.5)
        self.idr_sigma_bg_spin.setValue(10.0)
        form.addRow("DoG background σ", self.idr_sigma_bg_spin)
        self.idr_sigma_bg_spin.setToolTip(
            _tt(
                "Background smoothing σ used for IDR segmentation (pixels).\n"
                "The background is estimated by Gaussian blur and subtracted before thresholding.\n"
                "Larger σ removes broader background variations but can also reduce signal.",
                "IDR分割で使う背景平滑化σ（px）です。\n"
                "ガウスぼかしで背景を推定して減算し、その後に閾値処理します。\n"
                "σを大きくすると広い背景ムラは取れますが、信号も弱くなる可能性があります。",
            )
        )

        self.idr_k_spin = QtWidgets.QDoubleSpinBox()
        self.idr_k_spin.setRange(0.1, 20.0)
        self.idr_k_spin.setSingleStep(0.1)
        self.idr_k_spin.setValue(3.0)
        form.addRow("DoG threshold k", self.idr_k_spin)
        self.idr_k_spin.setToolTip(
            _tt(
                "DoG threshold multiplier k.\n"
                "Higher k makes the IDR mask smaller/stricter (fewer pixels pass).\n"
                "Lower k makes the mask larger but can include more noise.",
                "DoGの閾値係数 k です。\n"
                "kを上げるとIDRマスクは小さく/厳しくなります（通過画素が減る）。\n"
                "kを下げるとマスクは大きくなりますが、ノイズを含みやすくなります。",
            )
        )

        self.idr_scales_edit = QtWidgets.QLineEdit("0.8,1.2,1.6,2.4")
        self.idr_scales_edit.setToolTip(
            _tt(
                "DoG scales (comma-separated, in pixels).\n"
                "Multiple scales help detect IDR structures of different sizes.\n"
                "Example: 0.8,1.2,1.6,2.4",
                "DoGのスケール一覧（カンマ区切り、単位px）です。\n"
                "複数スケールにすると、サイズの異なるIDR構造を拾いやすくなります。\n"
                "例: 0.8,1.2,1.6,2.4",
            )
        )
        form.addRow("DoG scales", self.idr_scales_edit)

        self.idr_keep_connected_check = QtWidgets.QCheckBox("DoG: keep only components connected near the domain")
        self.idr_keep_connected_check.setChecked(False)
        form.addRow("DoG connectivity", self.idr_keep_connected_check)
        self.idr_keep_connected_check.setToolTip(
            _tt(
                "If enabled, keep only IDR components connected to the domain neighborhood.\n"
                "This helps remove isolated islands far from P1/P2.\n"
                "The neighborhood radius is controlled by the attach radius below.",
                "有効にすると、ドメイン近傍に連結しているIDR成分のみ残します。\n"
                "P1/P2から離れた孤立島（ノイズ）を除去しやすくなります。\n"
                "近傍の定義は下の attach radius で調整します。",
            )
        )

        self.idr_attach_r_spin = QtWidgets.QDoubleSpinBox()
        self.idr_attach_r_spin.setRange(0.0, 500.0)
        self.idr_attach_r_spin.setSingleStep(0.5)
        self.idr_attach_r_spin.setValue(5.0)
        form.addRow("DoG attach radius r (px)", self.idr_attach_r_spin)
        self.idr_attach_r_spin.setToolTip(
            _tt(
                "Radius (pixels) around spot centers considered as the 'domain neighborhood'.\n"
                "Used when DoG connectivity constraint is enabled.\n"
                "Increase if the IDR near the spot is being removed too aggressively.",
                "スポット中心の周囲を「ドメイン近傍」とみなす半径（px）です。\n"
                "DoG connectivity 制約が有効なときに使用されます。\n"
                "スポット近傍のIDRが削られすぎる場合は大きくしてください。",
            )
        )

        self.ring_r_spin = QtWidgets.QDoubleSpinBox()
        self.ring_r_spin.setRange(0.0, 500.0)
        self.ring_r_spin.setSingleStep(0.5)
        self.ring_r_spin.setValue(2.0)
        form.addRow("Seed ring radius", self.ring_r_spin)
        self.ring_r_spin.setToolTip(
            _tt(
                "Seed ring radius (pixels) used by seeded segmentation (rw).\n"
                "A ring around the center is used as a foreground seed.\n"
                "Increase for larger IDR regions; decrease for compact regions.",
                "シード付き分割（rw）で使うシードリング半径（px）です。\n"
                "中心周りのリングを前景シードとして使用します。\n"
                "大きいIDRには大きめ、小さいIDRには小さめが目安です。",
            )
        )

        self.ring_w_spin = QtWidgets.QDoubleSpinBox()
        self.ring_w_spin.setRange(0.1, 500.0)
        self.ring_w_spin.setSingleStep(0.5)
        self.ring_w_spin.setValue(2.0)
        form.addRow("Seed ring width", self.ring_w_spin)
        self.ring_w_spin.setToolTip(
            _tt(
                "Seed ring width (thickness, pixels) for seeded segmentation.\n"
                "Wider rings produce more seed pixels and can stabilize segmentation,\n"
                "but may also leak into background if too wide.",
                "シード付き分割で使うシードリングの幅（太さ、px）です。\n"
                "幅を広げるとシード画素が増えて安定することがありますが、\n"
                "広すぎると背景側に漏れる可能性があります。",
            )
        )

        self.bg_dist_spin = QtWidgets.QDoubleSpinBox()
        self.bg_dist_spin.setRange(1.0, 5000.0)
        self.bg_dist_spin.setSingleStep(1.0)
        self.bg_dist_spin.setValue(15.0)
        form.addRow("BG seed distance", self.bg_dist_spin)
        self.bg_dist_spin.setToolTip(
            _tt(
                "Background seed distance (pixels) for seeded segmentation.\n"
                "Background seeds are placed sufficiently far from the center.\n"
                "Increase if foreground leaks into background; decrease if ROI is small.",
                "シード付き分割で使う背景シード距離（px）です。\n"
                "中心から十分離れた位置を背景シードとして扱います。\n"
                "前景が背景へ漏れる場合は増やし、ROIが小さい場合は減らしてください。",
            )
        )

        self.rw_plus_extend_spin = QtWidgets.QDoubleSpinBox()
        self.rw_plus_extend_spin.setRange(0.0, 5000.0)
        self.rw_plus_extend_spin.setSingleStep(1.0)
        self.rw_plus_extend_spin.setValue(30.0)
        form.addRow("rw+DoG extend distance", self.rw_plus_extend_spin)
        self.rw_plus_extend_spin.setToolTip(
            _tt(
                "Expansion distance (pixels) used in 'rw_plus_dog' mode.\n"
                "After seeded segmentation, the mask can be expanded outward to include fringes.\n"
                "Increase to include more periphery; decrease to keep the mask tight.",
                "'rw_plus_dog' モードで使う拡張距離（px）です。\n"
                "rwで得たマスクを外側へ広げて周辺部を含めることができます。\n"
                "周辺まで含めたいなら増やし、マスクをタイトにしたいなら減らしてください。",
            )
        )

        self.forbidden_check = QtWidgets.QCheckBox("Apply forbidden corridor (suppress bridging)")
        self.forbidden_check.setChecked(False)
        form.addRow("forbidden", self.forbidden_check)
        self.forbidden_check.setToolTip(
            _tt(
                "If enabled, suppress IDR 'bridging' between two separated spots.\n"
                "When the state is separated and both P1/P2 exist, a corridor between them is excluded.\n"
                "This affects IDR mask only (not spot fitting).",
                "有効にすると、2スポットが分離しているときにIDRが2点間を橋渡ししないよう抑制します。\n"
                "stateがseparatedでP1/P2がある場合、2点間のコリドー（帯）を禁止領域にします。\n"
                "IDRマスクにのみ影響し、スポットフィットには影響しません。",
            )
        )

        self.forbidden_w_spin = QtWidgets.QDoubleSpinBox()
        self.forbidden_w_spin.setRange(0.0, 2000.0)
        self.forbidden_w_spin.setSingleStep(0.5)
        self.forbidden_w_spin.setValue(2.5)
        form.addRow("Forbidden width w (px)", self.forbidden_w_spin)
        self.forbidden_w_spin.setToolTip(
            _tt(
                "Forbidden corridor width (pixels).\n"
                "Larger values block a wider band between P1 and P2 (stronger anti-bridging).\n"
                "If set too large, it may remove real IDR pixels between domains.",
                "forbidden corridor の幅（px）です。\n"
                "大きいほどP1–P2間の禁止帯が太くなり、橋渡し抑制が強くなります。\n"
                "大きすぎると実際のIDRが削られる可能性があります。",
            )
        )

        self.idr_weighted_check = QtWidgets.QCheckBox("Compute Rg with intensity weighting (brighter = higher weight)")
        self.idr_weighted_check.setChecked(False)
        form.addRow("IDR weighting", self.idr_weighted_check)
        self.idr_weighted_check.setToolTip(
            _tt(
                "If enabled, compute IDR Rg using intensity-weighted pixels.\n"
                "Bright pixels contribute more to the center-of-mass and Rg.\n"
                "If disabled, each mask pixel contributes equally.",
                "有効にすると、IDRのRgを強度重み付きで計算します。\n"
                "明るい画素ほど重みが大きく、重心やRgへの寄与が増えます。\n"
                "無効の場合、マスク内の各画素は同じ重みで扱われます。",
            )
        )

        # overlay options
        self.ov_mask_check = QtWidgets.QCheckBox("mask")
        self.ov_mask_check.setChecked(True)
        self.ov_contour_check = QtWidgets.QCheckBox("contour")
        self.ov_r80_check = QtWidgets.QCheckBox("r80")
        self.ov_r90_check = QtWidgets.QCheckBox("r90")
        self.ov_mask_check.setToolTip(
            _tt(
                "Overlay the IDR mask as a filled region.",
                "IDRマスクを塗りつぶしで重ねて表示します。",
            )
        )
        self.ov_contour_check.setToolTip(
            _tt(
                "Overlay the IDR mask boundary (contour).",
                "IDRマスクの輪郭（境界線）を重ねて表示します。",
            )
        )
        self.ov_r80_check.setToolTip(
            _tt(
                "Overlay the r80 circle (radius containing 80% of IDR pixels/weight).",
                "r80円（IDRの80%を含む半径）を重ねて表示します。",
            )
        )
        self.ov_r90_check.setToolTip(
            _tt(
                "Overlay the r90 circle (radius containing 90% of IDR pixels/weight).",
                "r90円（IDRの90%を含む半径）を重ねて表示します。",
            )
        )
        ov_row = QtWidgets.QHBoxLayout()
        ov_row.addWidget(self.ov_mask_check)
        ov_row.addWidget(self.ov_contour_check)
        ov_row.addWidget(self.ov_r80_check)
        ov_row.addWidget(self.ov_r90_check)
        form.addRow("Overlay", ov_row)

        self.ov_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.ov_alpha_spin.setRange(0.0, 1.0)
        self.ov_alpha_spin.setSingleStep(0.05)
        self.ov_alpha_spin.setValue(0.35)
        form.addRow("Overlay alpha", self.ov_alpha_spin)
        self.ov_alpha_spin.setToolTip(
            _tt(
                "Opacity of the overlay (0 = transparent, 1 = opaque).\n"
                "This affects only visualization.",
                "オーバーレイの不透明度です（0=透明、1=不透明）。\n"
                "表示にのみ影響します。",
            )
        )

        # overlay UI → redraw immediately
        # NOTE: connect with lambda to ignore signal args (int/bool/float)
        self.ov_mask_check.stateChanged.connect(lambda _=None: self._refresh_overlay())
        self.ov_contour_check.stateChanged.connect(lambda _=None: self._refresh_overlay())
        self.ov_r80_check.stateChanged.connect(lambda _=None: self._refresh_overlay())
        self.ov_r90_check.stateChanged.connect(lambda _=None: self._refresh_overlay())
        self.ov_alpha_spin.valueChanged.connect(lambda _=None: self._refresh_overlay())

        layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        self.run_all_btn = QtWidgets.QPushButton("Analyze all frames")
        self.run_all_btn.clicked.connect(self.run_analysis_all_frames)
        self.run_all_btn.setEnabled(False)
        self.prev_frame_btn = QtWidgets.QPushButton("◀")
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        self.next_frame_btn = QtWidgets.QPushButton("▶")
        self.next_frame_btn.clicked.connect(self._next_frame)
        self.frame_label = QtWidgets.QLabel("Frame: - / -")
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.run_all_btn)
        btn_row.addWidget(self.prev_frame_btn)
        btn_row.addWidget(self.next_frame_btn)
        btn_row.addWidget(self.frame_label)
        layout.addLayout(btn_row)

        self.roi_status_label = QtWidgets.QLabel("ROI: not selected")
        layout.addWidget(self.roi_status_label)

        self.selection_label = QtWidgets.QLabel("")
        layout.addWidget(self.selection_label)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(240)
        layout.addWidget(self.output)

        self.export_btn = QtWidgets.QPushButton("Export CSV (Chugai)")
        self.export_btn.clicked.connect(self.export_chugai_csv)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset analysis results")
        self.reset_btn.clicked.connect(self._reset_analysis_results)
        layout.addWidget(self.reset_btn)

        self.edit_help_label = QtWidgets.QLabel(
            "Spot edit: Ctrl/⌘+drag=move, Shift+click=add, Alt(Option)+click=delete"
        )
        self.edit_help_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.edit_help_label)

    # ---- Core helpers (copied from SpotAnalysisWindow pattern) ----
    def _refresh_selection_label(self) -> None:
        if not self.main_window or not hasattr(self.main_window, "FileList"):
            self.selection_label.setText("Cannot get file selection info.")
            return
        selected = self.main_window.FileList.selectedItems()
        if selected:
            names = [item.text() for item in selected]
            self.selection_label.setText(f"Selected: {', '.join(names)}")
        else:
            current = self.main_window.FileList.currentItem()
            if current:
                self.selection_label.setText(f"Current: {current.text()}")
            else:
                self.selection_label.setText("No selection")

    def _ensure_selection_loaded(self) -> bool:
        if not self.main_window or not hasattr(self.main_window, "FileList"):
            QtWidgets.QMessageBox.warning(self, "No Selection", "FileList was not found.")
            return False
        selected = self.main_window.FileList.selectedIndexes()
        if selected:
            target_row = selected[0].row()
        else:
            target_row = self.main_window.FileList.currentRow()
        if target_row is None or target_row < 0:
            QtWidgets.QMessageBox.information(self, "Select File", "Please select a file in the file list.")
            return False
        try:
            blocker = QtCore.QSignalBlocker(self.main_window.FileList)
            self.main_window.FileList.setCurrentRow(target_row)
        except Exception:
            self.main_window.FileList.setCurrentRow(target_row)
        if hasattr(self.main_window, "ListClickFunction"):
            try:
                try:
                    self.main_window.ListClickFunction(bring_to_front=False)
                except TypeError:
                    self.main_window.ListClickFunction()
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load file:\n{exc}")
                return False
        return True

    def _prepare_frame(self) -> Optional[np.ndarray]:
        if not hasattr(gv, "files") or not gv.files:
            QtWidgets.QMessageBox.information(self, "No Files", "No files are loaded.")
            return None
        if getattr(gv, "currentFileNum", -1) < 0 or gv.currentFileNum >= len(gv.files):
            QtWidgets.QMessageBox.warning(self, "Invalid Selection", "The selected file index is invalid.")
            return None
        try:
            LoadFrame(gv.files[gv.currentFileNum])
            InitializeAryDataFallback()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load frame:\n{exc}")
            return None
        if not hasattr(gv, "aryData") or gv.aryData is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Image data is not available.")
            return None
        frame = np.asarray(gv.aryData, dtype=np.float64)
        if frame.ndim != 2:
            QtWidgets.QMessageBox.warning(self, "Data Error", "Only 2D image data is supported.")
            return None
        return frame

    def _connect_frame_signal(self) -> None:
        if self.main_window and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.connect(self._on_frame_changed)
            except Exception:
                pass

    def _get_current_frame_index(self) -> int:
        return int(getattr(gv, "index", 0))

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

    def _get_last_roi_at_or_before(self, frame_index: int) -> Optional[Dict[str, float]]:
        if frame_index < 0:
            return None
        for idx in range(int(frame_index), -1, -1):
            roi = self.roi_by_frame.get(idx)
            if roi is not None:
                return roi
        return None

    def _roi_overrides_for_roi(self, roi_info: Optional[Dict[str, float]], frame_shape: Tuple[int, int]):
        """SpotAnalysisWindow._roi_overrides を ROI dict 指定版にしたもの。"""
        if roi_info is None:
            return None, None, None, None
        h_img, w_img = frame_shape
        if roi_info.get("shape") == "Ellipse":
            x0 = int(round(roi_info["x0"]))
            y0 = int(round(roi_info["y0"]))
            x1 = int(round(roi_info["x0"] + roi_info["w"]))
            y1 = int(round(roi_info["y0"] + roi_info["h"]))
            x0c = max(x0, 0)
            y0c = max(y0, 0)
            x1c = min(x1, w_img)
            y1c = min(y1, h_img)
            bounds = (x0c, y0c, x1c - x0c, y1c - y0c)
            cx = roi_info["cx"]
            cy = roi_info["cy"]
            rx = roi_info["rx"]
            ry = roi_info["ry"]
            roi_w = max(1, bounds[2])
            roi_h = max(1, bounds[3])
            yy, xx = np.mgrid[0:roi_h, 0:roi_w]
            mask = ((xx + bounds[0] - cx) ** 2) / (rx ** 2 + 1e-12) + ((yy + bounds[1] - cy) ** 2) / (ry ** 2 + 1e-12) <= 1.0
            return (cx, cy), None, mask, bounds
        # Rectangle
        x0 = int(round(roi_info["x0"]))
        y0 = int(round(roi_info["y0"]))
        x1 = int(round(roi_info["x0"] + roi_info["w"]))
        y1 = int(round(roi_info["y0"] + roi_info["h"]))
        x0c = max(x0, 0)
        y0c = max(y0, 0)
        x1c = min(x1, w_img)
        y1c = min(y1, h_img)
        bounds = (x0c, y0c, x1c - x0c, y1c - y0c)
        return None, None, None, bounds

    def _current_roi_overlay(self) -> Optional[Dict[str, float]]:
        return self.roi_by_frame.get(self._get_current_frame_index(), self.manual_roi)

    def _on_frame_changed(self, frame_index: int) -> None:
        # ROI伝播（SpotAnalysisと同じルール）
        if self.auto_analyze_check.isChecked():
            roi_here = self.roi_by_frame.get(frame_index)
            if roi_here is not None:
                self.manual_roi = roi_here
            else:
                prev_roi = self._get_last_roi_at_or_before(frame_index - 1)
                if prev_roi is not None:
                    propagated = dict(prev_roi)
                    self.roi_by_frame[frame_index] = propagated
                    self.manual_roi = propagated
                else:
                    self.manual_roi = None
        else:
            self.manual_roi = self.roi_by_frame.get(frame_index)

        if self.manual_roi is None:
            self.roi_status_label.setText("ROI: not selected")
            self.run_btn.setEnabled(False)
            self.run_all_btn.setEnabled(False)
        else:
            self.roi_status_label.setText("ROI: selected")
            self.run_btn.setEnabled(True)
            self.run_all_btn.setEnabled(True)

        frame = self._prepare_frame()
        if frame is None:
            return
        self.last_frame = frame
        self._refresh_overlay()
        self._update_frame_label()

        if self.auto_analyze_check.isChecked():
            self._auto_analyze_if_enabled(frame_index, frame)

    # ---- Full image view ----
    def _ensure_live_window(self, win, cls):
        if win is not None and self._is_window_live(win):
            return win
        return cls(self)

    def _is_window_live(self, win) -> bool:
        try:
            _ = win.winId()
            return True
        except RuntimeError:
            return False

    def show_full_image_view(self) -> None:
        if self.last_frame is None:
            if not self._ensure_selection_loaded():
                return
            frame = self._prepare_frame()
            if frame is None:
                return
            self.last_frame = frame
        win = self._ensure_live_window(self.full_viz_window, ChugaiFullImageWindow)
        self.full_viz_window = win
        self.full_viz_window.enable_roi_selector(self.roi_shape_combo.currentText(), self._on_full_image_selected)
        self.full_viz_window.set_edit_handler(self._handle_edit_event)
        self.full_viz_window.set_idr_cutline_callback(self._on_idr_cutline_selected)
        self.full_viz_window.set_idr_connectline_callback(self._on_idr_connectline_selected)
        self._refresh_overlay()
        win.show()
        win.raise_()
        win.activateWindow()

    def _on_idr_connectline_selected(self, points) -> None:
        """
        IDR編集（接続ライン）確定時に呼ばれる。
        - 開始点/終了点が既存IDR上（近傍）にある場合のみ、固定太さの線をIDRへ描き足して接続する。
        - 現フレームのみ適用。
        """
        try:
            if points is None or len(points) < 2:
                return
        except Exception:
            return

        frame_index = self._get_current_frame_index()

        # フレームを確保
        if self.last_frame is None:
            if not self._ensure_selection_loaded():
                return
            frame = self._prepare_frame()
            if frame is None:
                return
            self.last_frame = frame
        frame = self.last_frame
        h, w = frame.shape

        fr = self.results_by_frame.get(int(frame_index))
        if fr is None or fr.idr_mask_full is None:
            return

        try:
            base_mask = np.asarray(fr.idr_mask_full, dtype=bool)
        except Exception:
            return
        if base_mask.shape != frame.shape:
            return

        # 端点がIDR上（近傍）にあるか確認（両端必須）
        def _has_idr_near(x: float, y: float, r: int = 10) -> bool:
            if not np.isfinite(float(x) + float(y)):
                return False
            cx = int(round(float(x)))
            cy = int(round(float(y)))
            x0 = max(cx - int(r), 0)
            x1 = min(cx + int(r) + 1, w)
            y0 = max(cy - int(r), 0)
            y1 = min(cy + int(r) + 1, h)
            if x1 <= x0 or y1 <= y0:
                return False
            return bool(np.any(base_mask[y0:y1, x0:x1]))

        try:
            (x0, y0) = points[0]
            (x1, y1) = points[-1]
        except Exception:
            return
        if not (_has_idr_near(x0, y0) and _has_idr_near(x1, y1)):
            return

        # 線をラスタライズ
        line_mask = np.zeros((h, w), dtype=bool)
        try:
            pts = [(float(p[0]), float(p[1])) for p in points]
        except Exception:
            return
        for (ax0, ay0), (ax1, ay1) in zip(pts[:-1], pts[1:]):
            if not np.isfinite(ax0 + ay0 + ax1 + ay1):
                continue
            steps = int(max(abs(ax1 - ax0), abs(ay1 - ay0))) + 1
            steps = max(2, min(steps, 5000))
            xs = np.linspace(ax0, ax1, steps)
            ys = np.linspace(ay0, ay1, steps)
            xi = np.clip(np.rint(xs).astype(int), 0, w - 1)
            yi = np.clip(np.rint(ys).astype(int), 0, h - 1)
            line_mask[yi, xi] = True

        # 固定太さ（例: barrierと同等）
        try:
            painted = ndimage.binary_dilation(line_mask, iterations=2)
        except Exception:
            painted = line_mask

        fr.idr_mask_full = (base_mask | painted.astype(bool)).astype(bool)
        # --- メトリクス再計算 ---
        try:
            if np.isfinite(fr.x1) and np.isfinite(fr.y1) and np.isfinite(fr.x2) and np.isfinite(fr.y2):
                center_xy = ((float(fr.x1) + float(fr.x2)) / 2.0, (float(fr.y1) + float(fr.y2)) / 2.0)
            elif np.isfinite(fr.x1) and np.isfinite(fr.y1):
                center_xy = (float(fr.x1), float(fr.y1))
            else:
                roi = self._current_roi_overlay()
                if roi and roi.get("shape") == "Ellipse":
                    center_xy = (float(roi.get("cx", w / 2.0)), float(roi.get("cy", h / 2.0)))
                elif roi and ("x0" in roi and "y0" in roi and "w" in roi and "h" in roi):
                    center_xy = (float(roi["x0"]) + float(roi["w"]) / 2.0, float(roi["y0"]) + float(roi["h"]) / 2.0)
                else:
                    center_xy = (w / 2.0, h / 2.0)
        except Exception:
            center_xy = (w / 2.0, h / 2.0)

        inten = None
        try:
            sigma_bg = float(self.idr_sigma_bg_spin.value())
            bg = ndimage.gaussian_filter(frame, sigma=sigma_bg) if sigma_bg > 0 else 0.0
            inten = np.clip(np.asarray(frame, dtype=np.float64) - bg, 0.0, None)
        except Exception:
            inten = None

        try:
            metrics = self.idr_detector.compute_metrics(
                fr.idr_mask_full,
                center_xy=center_xy,
                intensity=inten,
                weighted=bool(self.idr_weighted_check.isChecked()),
            )
            fr.idr_area = float(metrics.idr_area)
            fr.idr_rg = float(metrics.idr_rg)
            fr.idr_r80 = float(metrics.idr_r80)
            fr.idr_r90 = float(metrics.idr_r90)
        except Exception:
            pass

        self.results_by_frame[int(frame_index)] = fr
        self._display_frame_result(frame_index)
        self._refresh_overlay()

    def _on_idr_cutline_selected(self, points) -> None:
        """
        IDR編集（カットライン）確定時に呼ばれる。
        カットラインを障壁としてIDRを分断し、P1/P2（粒子）側の成分だけ残す。
        """
        try:
            if points is None or len(points) < 2:
                return
        except Exception:
            return

        frame_index = self._get_current_frame_index()

        # フレームを確保
        if self.last_frame is None:
            if not self._ensure_selection_loaded():
                return
            frame = self._prepare_frame()
            if frame is None:
                return
            self.last_frame = frame
        frame = self.last_frame
        h, w = frame.shape

        fr = self.results_by_frame.get(int(frame_index))
        if fr is None or fr.idr_mask_full is None:
            # まだIDRが無い（未解析/無効）場合は何もしない
            return

        try:
            base_mask = np.asarray(fr.idr_mask_full, dtype=bool)
        except Exception:
            return
        if base_mask.shape != frame.shape:
            return

        # ---- barrier mask（カットライン）を生成 ----
        line_mask = np.zeros((h, w), dtype=bool)
        try:
            pts = [(float(p[0]), float(p[1])) for p in points]
        except Exception:
            return
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if not np.isfinite(x0 + y0 + x1 + y1):
                continue
            steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
            steps = max(2, min(steps, 5000))
            xs = np.linspace(x0, x1, steps)
            ys = np.linspace(y0, y1, steps)
            xi = np.clip(np.rint(xs).astype(int), 0, w - 1)
            yi = np.clip(np.rint(ys).astype(int), 0, h - 1)
            line_mask[yi, xi] = True

        # 太さ（デフォルト: 2px相当）を持たせる
        try:
            barrier = ndimage.binary_dilation(line_mask, iterations=2)
        except Exception:
            barrier = line_mask

        passable = base_mask & (~barrier.astype(bool))
        if np.count_nonzero(passable) == 0:
            return

        # ---- 連結成分から“粒子側”だけ残す ----
        try:
            lab, nlab = ndimage.label(passable)
        except Exception:
            return
        if nlab <= 0:
            return

        def _seed_label_near(x: float, y: float, r: int = 10) -> int:
            if not (np.isfinite(x) and np.isfinite(y)):
                return 0
            cx = int(round(float(x)))
            cy = int(round(float(y)))
            x0 = max(cx - int(r), 0)
            x1 = min(cx + int(r) + 1, w)
            y0 = max(cy - int(r), 0)
            y1 = min(cy + int(r) + 1, h)
            if x1 <= x0 or y1 <= y0:
                return 0
            win = passable[y0:y1, x0:x1]
            if np.count_nonzero(win) == 0:
                return 0
            yy, xx = np.nonzero(win)
            # window coords -> abs coords
            ax = xx + x0
            ay = yy + y0
            d2 = (ax - float(x)) ** 2 + (ay - float(y)) ** 2
            idx = int(np.argmin(d2))
            return int(lab[int(ay[idx]), int(ax[idx])])

        keep_labels = set()
        keep_labels.add(_seed_label_near(fr.x1, fr.y1))
        keep_labels.add(_seed_label_near(fr.x2, fr.y2))
        keep_labels.discard(0)

        if not keep_labels:
            # フォールバック: 最大面積成分
            counts = np.bincount(lab.ravel())
            if counts.size <= 1:
                return
            counts[0] = 0
            keep = int(np.argmax(counts))
            keep_labels = {keep}

        new_mask = np.isin(lab, list(keep_labels))
        fr.idr_mask_full = new_mask.astype(bool)

        # center: 2spotなら中点、1spotならspot1、無ければROI中心/画像中心
        try:
            if np.isfinite(fr.x1) and np.isfinite(fr.y1) and np.isfinite(fr.x2) and np.isfinite(fr.y2):
                center_xy = ((float(fr.x1) + float(fr.x2)) / 2.0, (float(fr.y1) + float(fr.y2)) / 2.0)
            elif np.isfinite(fr.x1) and np.isfinite(fr.y1):
                center_xy = (float(fr.x1), float(fr.y1))
            else:
                roi = self._current_roi_overlay()
                if roi and roi.get("shape") == "Ellipse":
                    center_xy = (float(roi.get("cx", w / 2.0)), float(roi.get("cy", h / 2.0)))
                elif roi and ("x0" in roi and "y0" in roi and "w" in roi and "h" in roi):
                    center_xy = (float(roi["x0"]) + float(roi["w"]) / 2.0, float(roi["y0"]) + float(roi["h"]) / 2.0)
                else:
                    center_xy = (w / 2.0, h / 2.0)
        except Exception:
            center_xy = (w / 2.0, h / 2.0)

        # intensity for weighted Rg
        inten = None
        try:
            sigma_bg = float(self.idr_sigma_bg_spin.value())
            bg = ndimage.gaussian_filter(frame, sigma=sigma_bg) if sigma_bg > 0 else 0.0
            inten = np.clip(np.asarray(frame, dtype=np.float64) - bg, 0.0, None)
        except Exception:
            inten = None

        try:
            metrics = self.idr_detector.compute_metrics(
                fr.idr_mask_full,
                center_xy=center_xy,
                intensity=inten,
                weighted=bool(self.idr_weighted_check.isChecked()),
            )
            fr.idr_area = float(metrics.idr_area)
            fr.idr_rg = float(metrics.idr_rg)
            fr.idr_r80 = float(metrics.idr_r80)
            fr.idr_r90 = float(metrics.idr_r90)
        except Exception:
            pass

        self.results_by_frame[int(frame_index)] = fr
        self._display_frame_result(frame_index)
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        if not self.full_viz_window or not self._is_window_live(self.full_viz_window):
            return
        if self.last_frame is None:
            return
        frame_index = self._get_current_frame_index()
        spots = self.spots_by_frame.get(frame_index)
        roi_overlay = self._current_roi_overlay()
        fr = self.results_by_frame.get(frame_index)
        idr_vis = {
            "show_mask": bool(self.ov_mask_check.isChecked()),
            "show_contour": bool(self.ov_contour_check.isChecked()),
            "show_r80": bool(self.ov_r80_check.isChecked()),
            "show_r90": bool(self.ov_r90_check.isChecked()),
            "alpha": float(self.ov_alpha_spin.value()),
            "color": "magenta",
        }
        idr_center = None
        idr_r80 = None
        idr_r90 = None
        idr_mask_full = None
        # NOTE: score map の全フレーム保持はメモリ負荷が大きくなるため、デフォルトでは保持しない
        idr_overlay_full = None
        if fr is not None:
            idr_mask_full = fr.idr_mask_full
            idr_overlay_full = fr.idr_overlay_full
            if np.isfinite(fr.idr_r80):
                idr_r80 = fr.idr_r80
            if np.isfinite(fr.idr_r90):
                idr_r90 = fr.idr_r90
            # centerは2spotなら中点、1spotならspot1
            if np.isfinite(fr.x1) and np.isfinite(fr.y1) and np.isfinite(fr.x2) and np.isfinite(fr.y2):
                idr_center = ((fr.x1 + fr.x2) / 2.0, (fr.y1 + fr.y2) / 2.0)
            elif np.isfinite(fr.x1) and np.isfinite(fr.y1):
                idr_center = (fr.x1, fr.y1)
        self.full_viz_window.update_view(
            self.last_frame,
            spots=spots,
            roi_overlay=roi_overlay,
            spot_radius_px=float(self.spot_radius_spin.value()),
            idr_mask_full=idr_mask_full,
            idr_overlay_full=idr_overlay_full,
            idr_center=idr_center,
            idr_r80=idr_r80,
            idr_r90=idr_r90,
            idr_vis=idr_vis,
        )

    def _on_full_image_selected(self, roi_info: Dict[str, float]) -> None:
        w = roi_info.get("w", 0)
        h = roi_info.get("h", 0)
        if w <= 1 or h <= 1:
            return
        frame_index = self._get_current_frame_index()
        self.roi_by_frame[frame_index] = roi_info
        self.manual_roi = roi_info
        self.roi_status_label.setText("ROI: selected")
        self.run_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        self.run_analysis()

    def _on_roi_shape_changed(self, text: str) -> None:
        if self.full_viz_window and self._is_window_live(self.full_viz_window):
            self.full_viz_window.enable_roi_selector(text, self._on_full_image_selected)

    # ---- Spot editing (copied behavior) ----
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
                self._last_edit_was_manual_by_frame[int(frame_index)] = True
                self._recompute_after_manual_spot_edit(frame_index)
                return
            if is_delete:
                idx = self._find_nearest_spot(spots, event.xdata, event.ydata)
                if idx is not None:
                    spots.pop(idx)
                    self.export_btn.setEnabled(True)
                    self._last_edit_was_manual_by_frame[int(frame_index)] = True
                    self._recompute_after_manual_spot_edit(frame_index)
                return
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
                self._refresh_overlay()
        elif phase == "release":
            if self._dragging:
                self._dragging = False
                self._drag_index = None
                self.export_btn.setEnabled(True)
                self._last_edit_was_manual_by_frame[int(frame_index)] = True
                # ドラッグ移動はrelease時のみ再計算
                self._recompute_after_manual_spot_edit(frame_index)

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

    # ---- Analysis ----
    def _parse_scales(self) -> List[float]:
        txt = (self.idr_scales_edit.text() or "").strip()
        out: List[float] = []
        for part in txt.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except Exception:
                continue
        if not out:
            out = [0.8, 1.2, 1.6, 2.4]
        return out

    def _extract_spots_two(self, result: FrameAnalysis) -> List[Dict[str, float]]:
        best = result.models[result.best_n_peaks]
        spots = [{"x": float(pk.x), "y": float(pk.y), "snr": float(pk.snr)} for pk in best.peaks]
        # order stable: x then y
        spots.sort(key=lambda d: (d.get("x", 0.0), d.get("y", 0.0)))
        spots = spots[:2]

        # Gate weak second spot when 2-spot was selected.
        # Use existing UI threshold: "SNR最小 (2spot判定)" (self.snr_min_spin).
        if int(getattr(result, "best_n_peaks", 0)) == 2 and len(spots) >= 2:
            snr_min = float(self.snr_min_spin.value())
            kept = [pk for pk in spots if float(pk.get("snr", 0.0)) >= snr_min]
            if len(kept) == 1:
                spots = kept
            elif len(kept) == 2:
                spots = kept
            else:
                # Rare: both spots are below threshold. Keep the higher-SNR one
                # to avoid returning empty spots and breaking downstream display.
                spots = [max(spots, key=lambda d: float(d.get("snr", 0.0)))]

            # Additional gating by SNR ratio to suppress spurious weak peak.
            # If the weaker peak is < r * (stronger peak), drop the weaker one.
            if len(spots) >= 2:
                snr_ratio_r = 0.4
                s0 = float(spots[0].get("snr", 0.0))
                s1 = float(spots[1].get("snr", 0.0))
                high = max(s0, s1)
                low = min(s0, s1)
                if high > 0.0 and low < high * snr_ratio_r:
                    spots = [spots[0]] if s0 >= s1 else [spots[1]]

        return spots

    def _state_from_spots(self, spots: List[Dict[str, float]]) -> Tuple[str, float]:
        snr_min = float(self.snr_min_spin.value())
        d_merge = float(self.d_merge_spin.value())
        d_sep = float(self.d_sep_spin.value())
        if len(spots) >= 2:
            x1, y1 = float(spots[0]["x"]), float(spots[0]["y"])
            x2, y2 = float(spots[1]["x"]), float(spots[1]["y"])
            d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            if float(spots[0].get("snr", 0.0)) >= snr_min and float(spots[1].get("snr", 0.0)) >= snr_min and d > d_sep:
                return "separated", d
            if d < d_merge:
                return "merged", d
            return "uncertain", d
        # 2spot取れない -> merged（仕様）
        return "merged", float("nan")

    def _state_from_spots_manual_override(self, spots: List[Dict[str, float]]) -> Tuple[str, float]:
        """
        手動編集時のstate判定（SNRを無視して距離のみで判定）。
        - separated: d > d_sep_threshold
        - merged: d < d_merge_threshold もしくは 2spot取れない
        - uncertain: それ以外
        """
        d_merge = float(self.d_merge_spin.value())
        d_sep = float(self.d_sep_spin.value())
        if len(spots) >= 2:
            x1, y1 = float(spots[0]["x"]), float(spots[0]["y"])
            x2, y2 = float(spots[1]["x"]), float(spots[1]["y"])
            d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            if d > d_sep:
                return "separated", d
            if d < d_merge:
                return "merged", d
            return "uncertain", d
        return "merged", float("nan")

    def _compute_idr_for_frame(
        self,
        frame_index: int,
        frame: np.ndarray,
        roi_info: Optional[Dict[str, float]],
        spots: List[Dict[str, float]],
        state_for_forbidden: str,
    ) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
        """
        そのフレームのIDRを計算して返す（ROI内のみ）。
        返り値: (idr_area, idr_rg, idr_r80, idr_r90, idr_mask_full)
        """
        nan = float("nan")
        if not self.idr_enable_check.isChecked():
            return nan, nan, nan, nan, None
        if roi_info is None:
            return nan, nan, nan, nan, None

        _center_override, _roi_size_override, roi_mask, roi_bounds = self._roi_overrides_for_roi(roi_info, frame.shape)
        if roi_bounds is None:
            return nan, nan, nan, nan, None

        try:
            x0, y0, w, h = roi_bounds
            roi_raw = frame[y0 : y0 + h, x0 : x0 + w]
            roi_m = roi_mask

            # spot座標をROI座標へ（最大2点）
            spots_in_roi: List[Tuple[float, float]] = []
            abs_spots_xy: List[Tuple[float, float]] = []
            for pk in spots[:2]:
                abs_x = float(pk["x"])
                abs_y = float(pk["y"])
                abs_spots_xy.append((abs_x, abs_y))
                spots_in_roi.append((abs_x - x0, abs_y - y0))

            center_roi = self._idr_center_in_roi(spots_in_roi, roi_raw.shape)
            domain_pts_roi = spots_in_roi if spots_in_roi else None

            dog_mask, dog_score = self.idr_detector.mask_dog(
                roi_raw,
                roi_m,
                sigma_bg=float(self.idr_sigma_bg_spin.value()),
                dog_scales=self._parse_scales(),
                k=float(self.idr_k_spin.value()),
                require_positive=True,
                keep_connected_to_domain=bool(self.idr_keep_connected_check.isChecked()),
                domain_points_xy=domain_pts_roi,
                domain_attach_r=float(self.idr_attach_r_spin.value()),
            )

            rw_mask = None
            try:
                rw_mask = self.idr_detector.mask_rw(
                    roi_raw,
                    roi_m,
                    center_xy=center_roi,
                    ring_r=float(self.ring_r_spin.value()),
                    ring_w=float(self.ring_w_spin.value()),
                    bg_dist=float(self.bg_dist_spin.value()),
                    sigma_bg=float(self.idr_sigma_bg_spin.value()),
                    require_positive=True,
                )
            except Exception as exc_rw:  # noqa: BLE001
                logger.debug("rw segmentation failed, fallback to DoG: %s", exc_rw)
                rw_mask = None

            forbidden = None
            if self.forbidden_check.isChecked() and state_for_forbidden == "separated" and len(abs_spots_xy) >= 2:
                forbidden = IDRDetector.forbidden_corridor(
                    roi_raw.shape,
                    spot1_xy=spots_in_roi[0],
                    spot2_xy=spots_in_roi[1],
                    width_px=float(self.forbidden_w_spin.value()),
                )

            mask = self.idr_detector.combine(
                self.idr_mode_combo.currentText(),
                dog_mask=dog_mask,
                rw_mask=rw_mask,
                dog_score=dog_score,
                center_xy=center_roi,
                extend_r=float(self.rw_plus_extend_spin.value()),
                forbidden=forbidden,
            )

            sigma_bg = float(self.idr_sigma_bg_spin.value())
            bg = ndimage.gaussian_filter(roi_raw, sigma=sigma_bg) if sigma_bg > 0 else 0.0
            inten = np.clip(roi_raw - bg, 0.0, None)
            metrics = self.idr_detector.compute_metrics(
                mask,
                center_xy=center_roi,
                intensity=inten,
                weighted=bool(self.idr_weighted_check.isChecked()),
            )
            idr_mask_full = self._paste_mask_full(mask, origin=(x0, y0), full_shape=frame.shape)
            return metrics.idr_area, metrics.idr_rg, metrics.idr_r80, metrics.idr_r90, idr_mask_full
        except Exception as exc_idr:  # noqa: BLE001
            logger.debug("IDR failed: %s", exc_idr)
            return nan, nan, nan, nan, None

    def _recompute_after_manual_spot_edit(self, frame_index: int) -> None:
        """
        スポットが手動編集された直後に、そのフレームだけ再計算する。
        - spotフィットは再実行しない
        - stateはSNR無視のmanual override
        - IDRはROI内のみ再計算
        """
        try:
            if self.last_frame is None:
                # 念のためフレームを確保
                if not self._ensure_selection_loaded():
                    return
                frame = self._prepare_frame()
                if frame is None:
                    return
                self.last_frame = frame
            frame = self.last_frame
            idx = int(frame_index)
            spots = list(self.spots_by_frame.get(idx, []))
            # order stable
            spots.sort(key=lambda d: (d.get("x", 0.0), d.get("y", 0.0)))
            spots = spots[:2]
            self.spots_by_frame[idx] = spots

            state, d = self._state_from_spots_manual_override(spots)
            roi_info = self._get_roi_for_analysis(idx)
            idr_area, idr_rg, idr_r80, idr_r90, idr_mask_full = self._compute_idr_for_frame(
                idx, frame, roi_info, spots, state_for_forbidden=state
            )

            nan = float("nan")
            x1 = float(spots[0]["x"]) if len(spots) >= 1 else nan
            y1 = float(spots[0]["y"]) if len(spots) >= 1 else nan
            snr1 = float(spots[0].get("snr", 0.0)) if len(spots) >= 1 else nan
            x2 = float(spots[1]["x"]) if len(spots) >= 2 else nan
            y2 = float(spots[1]["y"]) if len(spots) >= 2 else nan
            snr2 = float(spots[1].get("snr", 0.0)) if len(spots) >= 2 else nan

            fr = ChugaiFrameResult(
                frame_index=idx,
                x1=x1,
                y1=y1,
                snr1=snr1,
                x2=x2,
                y2=y2,
                snr2=snr2,
                d=float(d),
                state=str(state),
                idr_area=float(idr_area),
                idr_rg=float(idr_rg),
                idr_r80=float(idr_r80),
                idr_r90=float(idr_r90),
                idr_mode=str(self.idr_mode_combo.currentText()),
                params_json=self._format_params_json(),
                idr_mask_full=idr_mask_full,
                idr_overlay_full=None,
            )
            self.results_by_frame[idx] = fr
            self._display_frame_result(idx)
            self._refresh_overlay()
        except Exception as exc:  # noqa: BLE001
            logger.debug("manual recompute failed: %s", exc)

    def _idr_center_in_roi(self, spots_in_roi: List[Tuple[float, float]], roi_shape: Tuple[int, int]) -> Tuple[float, float]:
        h, w = roi_shape
        if len(spots_in_roi) >= 2:
            return ((spots_in_roi[0][0] + spots_in_roi[1][0]) / 2.0, (spots_in_roi[0][1] + spots_in_roi[1][1]) / 2.0)
        if len(spots_in_roi) == 1:
            return (spots_in_roi[0][0], spots_in_roi[0][1])
        return (w / 2.0, h / 2.0)

    def _paste_mask_full(self, mask_roi: np.ndarray, origin: Tuple[int, int], full_shape: Tuple[int, int]) -> np.ndarray:
        full = np.zeros(full_shape, dtype=bool)
        x0, y0 = int(origin[0]), int(origin[1])
        h, w = mask_roi.shape
        # clip
        x1 = min(x0 + w, full_shape[1])
        y1 = min(y0 + h, full_shape[0])
        if x1 <= x0 or y1 <= y0:
            return full
        roi_x0 = 0
        roi_y0 = 0
        roi_x1 = x1 - x0
        roi_y1 = y1 - y0
        full[y0:y1, x0:x1] = mask_roi[roi_y0:roi_y1, roi_x0:roi_x1]
        return full

    def _paste_map_full(self, map_roi: np.ndarray, origin: Tuple[int, int], full_shape: Tuple[int, int]) -> np.ndarray:
        """ROI座標のスコアマップを全画像へ貼り戻す（はみ出しはクリップ、外側は0）。"""
        full = np.zeros(full_shape, dtype=np.float64)
        x0, y0 = int(origin[0]), int(origin[1])
        h, w = map_roi.shape
        x1 = min(x0 + w, full_shape[1])
        y1 = min(y0 + h, full_shape[0])
        if x1 <= x0 or y1 <= y0:
            return full
        roi_x1 = x1 - x0
        roi_y1 = y1 - y0
        full[y0:y1, x0:x1] = np.asarray(map_roi, dtype=np.float64)[0:roi_y1, 0:roi_x1]
        return full

    def _format_params_json(self) -> str:
        params = {
            "spot": {
                "bandpass_low_sigma": float(self.bandpass_low_spin.value()),
                "bandpass_high_sigma": float(self.bandpass_high_spin.value()),
                "peak_min_distance": int(self.peak_min_distance_spin.value()),
                "initial_sigma": float(self.initial_sigma_spin.value()),
                "sigma_bounds": [float(self.sigma_min_spin.value()), float(self.sigma_max_spin.value())],
            },
            "state": {
                "snr_min": float(self.snr_min_spin.value()),
                "d_merge_threshold": float(self.d_merge_spin.value()),
                "d_sep_threshold": float(self.d_sep_spin.value()),
            },
            "idr": {
                "enabled": bool(self.idr_enable_check.isChecked()),
                "mode": self.idr_mode_combo.currentText(),
                "sigma_bg": float(self.idr_sigma_bg_spin.value()),
                "k": float(self.idr_k_spin.value()),
                "scales": self._parse_scales(),
                "dog_keep_connected": bool(self.idr_keep_connected_check.isChecked()),
                "dog_attach_r": float(self.idr_attach_r_spin.value()),
                "ring_r": float(self.ring_r_spin.value()),
                "ring_w": float(self.ring_w_spin.value()),
                "bg_dist": float(self.bg_dist_spin.value()),
                "rw_plus_extend_r": float(self.rw_plus_extend_spin.value()),
                "forbidden": bool(self.forbidden_check.isChecked()),
                "forbidden_w": float(self.forbidden_w_spin.value()),
                "weighted_rg": bool(self.idr_weighted_check.isChecked()),
            },
        }
        try:
            return json.dumps(params, ensure_ascii=False, sort_keys=True)
        except Exception:
            return "{}"

    def run_analysis(self) -> None:
        if self.manual_roi is None:
            QtWidgets.QMessageBox.information(self, "ROI Required", "Please select an ROI.")
            return
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            return

        # Spot params
        low_sigma = float(self.bandpass_low_spin.value())
        high_sigma = float(self.bandpass_high_spin.value())
        if high_sigma <= low_sigma:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "Bandpass high σ must be greater than low σ.")
            return
        self.spot_analyzer.bandpass_low_sigma = low_sigma
        self.spot_analyzer.bandpass_high_sigma = high_sigma
        self.spot_analyzer.peak_min_distance = max(1, int(self.peak_min_distance_spin.value()))
        self.spot_analyzer.initial_sigma = float(self.initial_sigma_spin.value())
        sigma_min = float(self.sigma_min_spin.value())
        sigma_max = float(self.sigma_max_spin.value())
        if sigma_max <= sigma_min:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameter", "σ max must be greater than σ min.")
            return
        self.spot_analyzer.sigma_bounds = (sigma_min, sigma_max)

        frame_idx = self._get_current_frame_index()
        self._analyze_frame_index(frame_idx, frame, show_errors=True)
        self.export_btn.setEnabled(True)

    def _analyze_frame_index(self, frame_index: int, frame: np.ndarray, show_errors: bool = False) -> None:
        roi_info = self._get_roi_for_analysis(frame_index)
        if roi_info is None:
            # 解析スキップ（NaN埋め）
            self._store_nan_result(frame_index)
            self._display_frame_result(frame_index)
            self._refresh_overlay()
            return
        center_override, roi_size_override, roi_mask, roi_bounds = self._roi_overrides_for_roi(roi_info, frame.shape)
        try:
            result = self.spot_analyzer.analyze_frame(
                frame,
                prev_center=None,
                criterion=self.criterion_combo.currentText().lower(),
                center_override=center_override,
                roi_size_override=roi_size_override,
                roi_mask_override=roi_mask,
                roi_bounds_override=roi_bounds,
                min_peaks=1,
                max_peaks=2,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("spot analyze failed: %s", exc)
            if show_errors:
                QtWidgets.QMessageBox.critical(self, "Analysis Error", f"Spot analysis failed:\n{exc}")
            self._store_nan_result(frame_index)
            return

        spots = self._extract_spots_two(result)
        self.spots_by_frame[frame_index] = spots
        state, d = self._state_from_spots(spots)

        idr_area, idr_rg, idr_r80, idr_r90, idr_mask_full = self._compute_idr_for_frame(
            int(frame_index), frame, roi_info, spots, state_for_forbidden=state
        )

        # store result
        nan = float("nan")
        x1 = float(spots[0]["x"]) if len(spots) >= 1 else nan
        y1 = float(spots[0]["y"]) if len(spots) >= 1 else nan
        snr1 = float(spots[0].get("snr", 0.0)) if len(spots) >= 1 else nan
        x2 = float(spots[1]["x"]) if len(spots) >= 2 else nan
        y2 = float(spots[1]["y"]) if len(spots) >= 2 else nan
        snr2 = float(spots[1].get("snr", 0.0)) if len(spots) >= 2 else nan

        params_json = self._format_params_json()
        fr = ChugaiFrameResult(
            frame_index=int(frame_index),
            x1=x1,
            y1=y1,
            snr1=snr1,
            x2=x2,
            y2=y2,
            snr2=snr2,
            d=float(d),
            state=str(state),
            idr_area=float(idr_area),
            idr_rg=float(idr_rg),
            idr_r80=float(idr_r80),
            idr_r90=float(idr_r90),
            idr_mode=str(self.idr_mode_combo.currentText()),
            params_json=params_json,
            idr_mask_full=idr_mask_full,
            idr_overlay_full=None,
        )
        self.results_by_frame[int(frame_index)] = fr
        self._display_frame_result(frame_index)
        self._refresh_overlay()

    def _get_roi_for_analysis(self, frame_index: int) -> Optional[Dict[str, float]]:
        if self.auto_analyze_check.isChecked():
            roi_here = self.roi_by_frame.get(frame_index)
            if roi_here is not None:
                return roi_here
            prev_roi = self._get_last_roi_at_or_before(frame_index - 1)
            if prev_roi is not None:
                propagated = dict(prev_roi)
                self.roi_by_frame[frame_index] = propagated
                return propagated
            return None
        return self.roi_by_frame.get(frame_index)

    def _store_nan_result(self, frame_index: int) -> None:
        nan = float("nan")
        fr = ChugaiFrameResult(
            frame_index=int(frame_index),
            x1=nan,
            y1=nan,
            snr1=nan,
            x2=nan,
            y2=nan,
            snr2=nan,
            d=nan,
            state="uncertain",
            idr_area=nan,
            idr_rg=nan,
            idr_r80=nan,
            idr_r90=nan,
            idr_mode=str(self.idr_mode_combo.currentText()),
            params_json=self._format_params_json(),
            idr_mask_full=None,
        )
        self.results_by_frame[int(frame_index)] = fr

    def _display_frame_result(self, frame_index: int) -> None:
        fr = self.results_by_frame.get(int(frame_index))
        if fr is None:
            self.output.setPlainText("")
            return
        lines = []
        lines.append(f"frame={fr.frame_index}")
        lines.append(f"spot1: (x,y)=({fr.x1:.2f},{fr.y1:.2f}) snr={fr.snr1:.2f}")
        lines.append(f"spot2: (x,y)=({fr.x2:.2f},{fr.y2:.2f}) snr={fr.snr2:.2f}")
        lines.append(f"d={fr.d:.3g}  state={fr.state}")
        lines.append(f"IDR: area={fr.idr_area:.3g}  rg={fr.idr_rg:.3g}  r80={fr.idr_r80:.3g}  r90={fr.idr_r90:.3g}  mode={fr.idr_mode}")
        self.output.setPlainText("\n".join(lines))

    def _auto_analyze_if_enabled(self, frame_index: int, frame: np.ndarray) -> None:
        if not self.auto_analyze_check.isChecked():
            return
        if self._auto_busy:
            return
        if frame_index in self.results_by_frame:
            return
        if self.manual_roi is None and self.roi_by_frame.get(frame_index) is None:
            # ROIが存在しないフレームはスキップしNaN（仕様）
            self._store_nan_result(frame_index)
            return
        # spot params 反映（簡便）
        low_sigma = float(self.bandpass_low_spin.value())
        high_sigma = float(self.bandpass_high_spin.value())
        if high_sigma <= low_sigma:
            return
        self.spot_analyzer.bandpass_low_sigma = low_sigma
        self.spot_analyzer.bandpass_high_sigma = high_sigma
        self.spot_analyzer.peak_min_distance = max(1, int(self.peak_min_distance_spin.value()))
        self.spot_analyzer.initial_sigma = float(self.initial_sigma_spin.value())
        sigma_min = float(self.sigma_min_spin.value())
        sigma_max = float(self.sigma_max_spin.value())
        if sigma_max <= sigma_min:
            return
        self.spot_analyzer.sigma_bounds = (sigma_min, sigma_max)

        self._auto_busy = True
        try:
            self._analyze_frame_index(frame_index, frame, show_errors=False)
        finally:
            self._auto_busy = False

    def run_analysis_all_frames(self) -> None:
        if not self._ensure_selection_loaded():
            return
        if not hasattr(gv, "FrameNum") or gv.FrameNum <= 0:
            QtWidgets.QMessageBox.warning(self, "No Frames", "Failed to get the number of frames.")
            return
        original_index = int(getattr(gv, "index", 0))
        self.results_by_frame = {}
        self.spots_by_frame = {}
        for idx in range(int(gv.FrameNum)):
            gv.index = idx
            frame = self._prepare_frame()
            if frame is None:
                self._store_nan_result(idx)
                continue
            self.last_frame = frame
            # ROIが無いフレームは解析スキップ（NaN）
            roi_info = self._get_roi_for_analysis(idx)
            if roi_info is None:
                self._store_nan_result(idx)
                continue
            self._analyze_frame_index(idx, frame, show_errors=False)
        gv.index = original_index
        frame = self._prepare_frame()
        if frame is not None:
            self.last_frame = frame
        self.export_btn.setEnabled(True)
        self._refresh_overlay()

    # ---- CSV ----
    def export_chugai_csv(self) -> None:
        if not hasattr(gv, "files") or not gv.files or gv.currentFileNum < 0:
            QtWidgets.QMessageBox.warning(self, "No File", "Failed to get source file information.")
            return
        if not hasattr(gv, "FrameNum") or gv.FrameNum <= 0:
            QtWidgets.QMessageBox.warning(self, "No Frames", "Failed to get the number of frames.")
            return

        src_path = gv.files[gv.currentFileNum]
        base_dir = os.path.dirname(src_path)
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        if self.export_dir is None:
            self.export_dir = base_dir
        export_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder", self.export_dir)
        if export_dir:
            self.export_dir = export_dir
        export_dir = self.export_dir
        out_path = os.path.join(export_dir, f"{base_name}_chugai.csv")

        # 必ず全フレーム出力
        header = [
            "frame_index",
            "x1",
            "y1",
            "snr1",
            "x2",
            "y2",
            "snr2",
            "d",
            "state",
            "idr_area",
            "idr_rg",
            "idr_r80",
            "idr_r90",
            "idr_mode",
            "params",
        ]
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for idx in range(int(gv.FrameNum)):
                    fr = self.results_by_frame.get(idx)
                    if fr is None:
                        # 未解析フレームは仕様通りNaN埋め
                        self._store_nan_result(idx)
                        fr = self.results_by_frame.get(idx)
                    vals = [
                        str(int(idx)),
                        _fmt_float(fr.x1),
                        _fmt_float(fr.y1),
                        _fmt_float(fr.snr1),
                        _fmt_float(fr.x2),
                        _fmt_float(fr.y2),
                        _fmt_float(fr.snr2),
                        _fmt_float(fr.d),
                        str(fr.state),
                        _fmt_float(fr.idr_area),
                        _fmt_float(fr.idr_rg),
                        _fmt_float(fr.idr_r80),
                        _fmt_float(fr.idr_r90),
                        str(fr.idr_mode),
                        _csv_escape(fr.params_json),
                    ]
                    f.write(",".join(vals) + "\n")
            QtWidgets.QMessageBox.information(self, "CSV Export", f"Saved:\n{out_path}")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "CSV Export Error", f"Failed to save CSV:\n{exc}")

    # ---- Reset ----
    def _reset_analysis_results(self) -> None:
        self.spots_by_frame = {}
        self.results_by_frame = {}
        self.roi_by_frame = {}
        self.manual_roi = None
        self._dragging = False
        self._drag_index = None
        try:
            self.output.setPlainText("")
        except Exception:
            pass
        try:
            self.roi_status_label.setText("ROI: not selected")
        except Exception:
            pass
        for btn in (getattr(self, "run_btn", None), getattr(self, "run_all_btn", None), getattr(self, "export_btn", None)):
            try:
                if btn is not None:
                    btn.setEnabled(False)
            except Exception:
                pass
        self._refresh_overlay()


def _fmt_float(x: float) -> str:
    try:
        xf = float(x)
        if np.isnan(xf) or not np.isfinite(xf):
            return "NaN"
        return f"{xf:.6g}"
    except Exception:
        return "NaN"


def _csv_escape(text: str) -> str:
    """
    CSV用に1セルとして安全に出力する。
    - ダブルクオートで囲み、内部の " は "" にエスケープする。
    """
    try:
        s = "" if text is None else str(text)
    except Exception:
        s = ""
    s = s.replace('"', '""')
    return f'"{s}"'


def create_plugin(main_window) -> QtWidgets.QWidget:
    """Pluginメニューから呼び出されるファクトリ。"""
    return ChugaiAnalysisWindow(main_window)

