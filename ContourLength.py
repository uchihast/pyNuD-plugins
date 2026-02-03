"""
ContourLength
-------------
AFM画像上のDNA鎖（紐状構造）の輪郭長を計測するプラグイン。
リッジ検出・経路探索・スプライン補間による弧長積分で高精度に輪郭長を算出する。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate, integrate, ndimage
from skimage import filters as skfilters, morphology, restoration
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse

import globalvals as gv
from fileio import LoadFrame, InitializeAryDataFallback

logger = logging.getLogger(__name__)

PLUGIN_NAME = "Contour Length"

# UI label -> matplotlib marker for start/end/waypoint dots
MARKER_SHAPE_MAP = {"丸": "o", "四角": "s", "三角上": "^", "三角下": "v", "菱形": "D"}


# ---------------------------------------------------------------------------
# ROI preprocessing for measurement (noise, background, contrast)
# ---------------------------------------------------------------------------

def apply_roi_preprocess(
    roi_image: np.ndarray,
    median_k: int = 0,
    deconv_enable: bool = False,
    deconv_sigma: float = 1.0,
    deconv_iterations: int = 10,
    flatten_mode: str = "none",
    flatten_sigma: float = 10.0,
    contrast_mode: str = "none",
    contrast_low: float = 2.0,
    contrast_high: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Apply optional preprocessing to ROI for contour measurement.
    Order: noise reduction -> deconvolution -> background flattening -> contrast.
    """
    out = np.asarray(roi_image, dtype=np.float64).copy()
    if out.size == 0 or out.ndim != 2:
        return out

    # 1. Noise reduction: median filter
    if median_k and int(median_k) >= 2:
        k = int(median_k)
        if k % 2 == 0:
            k += 1
        try:
            out = skfilters.median(out, footprint=morphology.square(k))
        except Exception:
            pass

    # 2. Deconvolution (Richardson-Lucy): sharpen probe blur
    if deconv_enable and float(deconv_iterations) > 0:
        try:
            v_min, v_max = np.nanmin(out), np.nanmax(out)
            if v_max > v_min:
                img = (out - v_min) / (v_max - v_min)
            else:
                img = np.clip(out, 0, 1).astype(np.float64)
            img = np.maximum(img, 1e-8)
            sigma = max(0.3, float(deconv_sigma))
            size = max(3, int(round(sigma * 4)) * 2 + 1)
            ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
            xx, yy = np.meshgrid(ax, ax)
            psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
            psf = psf / np.sum(psf)
            niter = int(deconv_iterations)
            try:
                deconv = restoration.richardson_lucy(img, psf, num_iter=niter)
            except TypeError:
                deconv = restoration.richardson_lucy(img, psf, iterations=niter)
            deconv = np.clip(deconv, 0, None)
            if v_max > v_min:
                out = deconv * (v_max - v_min) + v_min
            else:
                out = deconv.astype(np.float64)
        except Exception:
            pass

    # 3. Background flattening
    flatten_mode = (flatten_mode or "none").strip().lower()
    if flatten_mode == "plane":
        try:
            h, w = out.shape
            yy, xx = np.mgrid[:h, :w]
            X = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
            z = out.ravel()
            ok = np.isfinite(z)
            if np.sum(ok) >= 3:
                coeffs, _, _, _ = np.linalg.lstsq(X[ok], z[ok], rcond=None)
                plane = (X @ coeffs).reshape(h, w)
                out = out - plane
        except Exception:
            pass
    elif flatten_mode == "gaussian" and float(flatten_sigma) > 0:
        try:
            sigma = max(0.5, float(flatten_sigma))
            bg = ndimage.gaussian_filter(out, sigma=sigma, mode="nearest")
            out = out - bg
        except Exception:
            pass

    # 4. Contrast enhancement
    contrast_mode = (contrast_mode or "none").strip().lower()
    if contrast_mode == "percentile":
        try:
            low = max(0, min(100, float(contrast_low)))
            high = max(low, min(100, float(contrast_high)))
            v_min = np.nanpercentile(out, low)
            v_max = np.nanpercentile(out, high)
            if v_max > v_min:
                out = np.clip(out, v_min, v_max)
                out = (out - v_min) / (v_max - v_min)
            else:
                out = np.zeros_like(out)
        except Exception:
            pass
    elif contrast_mode == "gamma" and float(gamma) != 1.0:
        try:
            g = float(gamma)
            v_min, v_max = np.nanmin(out), np.nanmax(out)
            if v_max > v_min:
                out = (out - v_min) / (v_max - v_min)
                out = np.power(np.clip(out, 0, 1), g)
            else:
                out = np.zeros_like(out)
        except Exception:
            pass

    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Algorithm: Ridge detection, path finding, spline arc length
# ---------------------------------------------------------------------------

def compute_ridge_map(roi_image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Frangi filter (Hessian-based ridge detection) for filament center line.
    DNA appears as bright (high) line; ridge map is higher on center line.
    """
    img = np.asarray(roi_image, dtype=np.float64)
    if img.size == 0 or img.ndim != 2:
        return np.zeros_like(img)
    sigmas = [max(0.5, float(sigma))]
    try:
        ridge = skfilters.frangi(img, sigmas=sigmas, black_ridges=False)
    except Exception:
        ridge = np.zeros_like(img)
    return np.asarray(ridge, dtype=np.float64)


def _crop_rect(
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


def _angle_deg_between(ay: float, ax: float, by: float, bx: float) -> float:
    """Angle in degrees between vectors (ay,ax) and (by,bx). 0 = same direction."""
    na = np.sqrt(ay * ay + ax * ax) + 1e-12
    nb = np.sqrt(by * by + bx * bx) + 1e-12
    cos_ab = (ay * by + ax * bx) / (na * nb)
    cos_ab = np.clip(cos_ab, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ab)))


def path_dijkstra_ridge(
    cost_image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_bending_angle_deg: Optional[float] = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    8-neighbor Dijkstra on cost image (lower = prefer).
    max_bending_angle_deg: if set, penalize sharp turns (DNA persistence); None = off.
    """
    import heapq

    h, w = cost_image.shape
    sy, sx = int(start[0]), int(start[1])
    ey, ex = int(end[0]), int(end[1])
    if not (0 <= sy < h and 0 <= sx < w and 0 <= ey < h and 0 <= ex < w):
        return None

    use_bending = max_bending_angle_deg is not None and max_bending_angle_deg > 0
    bend_penalty = 50.0

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    dist_scale = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]

    INF = float("inf")
    dist = np.full((h, w), INF, dtype=np.float64)
    dist[sy, sx] = 0.0
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    prev[(sy, sx)] = None
    heap: List[Tuple[float, int, int]] = [(0.0, sy, sx)]

    while heap:
        d, y, x = heapq.heappop(heap)
        if (y, x) == (ey, ex):
            break
        if d > dist[y, x]:
            continue
        prev_pt = prev.get((y, x))
        for k, (dy, dx) in enumerate(neighbors):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if use_bending and prev_pt is not None:
                py, px = prev_pt
                in_dy, in_dx = y - py, x - px
                out_dy, out_dx = ny - y, nx - x
                ang = _angle_deg_between(float(in_dy), float(in_dx), float(out_dy), float(out_dx))
                if ang > max_bending_angle_deg:
                    continue
                if ang > max_bending_angle_deg * 0.6:
                    bend_extra = bend_penalty * (ang / max_bending_angle_deg)
                else:
                    bend_extra = 0.0
            else:
                bend_extra = 0.0
            c = float(cost_image[ny, nx])
            if not np.isfinite(c):
                c = 1e6
            step = dist_scale[k] * (0.5 + 0.5 * max(0, min(1, c))) + bend_extra
            nd = d + step
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[(ny, nx)] = (y, x)
                heapq.heappush(heap, (nd, ny, nx))

    if not np.isfinite(dist[ey, ex]):
        return None
    path: List[Tuple[int, int]] = []
    cur = (ey, ex)
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def contour_length_spline(path_pixels: List[Tuple[int, int]]) -> float:
    """
    Fit 3rd-order spline to path and compute arc length by integration.
    path_pixels: list of (row, col) = (y, x). We use (x, y) for spline.
    """
    if len(path_pixels) < 2:
        return 0.0
    xs = np.array([p[1] for p in path_pixels], dtype=np.float64)
    ys = np.array([p[0] for p in path_pixels], dtype=np.float64)
    n = len(xs)
    t = np.linspace(0, 1, n)
    try:
        tck, u = interpolate.splprep([xs, ys], s=0.0, k=min(3, n - 1))
    except Exception:
        return float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))
    u_new = np.linspace(0, 1, max(50, n * 2))
    x_new, y_new = interpolate.splev(u_new, tck)

    def speed(u_val: float) -> float:
        der = interpolate.splev(u_val, tck, der=1)
        return float(np.sqrt(der[0] ** 2 + der[1] ** 2))

    length, _ = integrate.quad(speed, 0, 1, limit=100)
    return float(length)


def compute_contour_between_points(
    roi_image: np.ndarray,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    ridge_sigma: float = 1.5,
    ridge_weight: float = 0.9,
    ridge_floor_threshold: float = 0.0,
    max_bending_angle_deg: Optional[float] = None,
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    start_xy, end_xy: (x, y) in ROI image coordinates (col, row).
    ridge_weight: 0–1, weight for ridge in cost (higher = prefer ridge, avoid shortcut).
    ridge_floor_threshold: pixels with ridge_n below this get high cost (0 = off).
    Returns (path_pixels as list of (row,col), contour_length_px).
    If path fails, returns (None, 0.0).
    """
    h, w = roi_image.shape
    ridge = compute_ridge_map(roi_image, sigma=ridge_sigma)
    # Normalize intensity: DNA is bright
    im = np.asarray(roi_image, dtype=np.float64)
    im_min, im_max = np.nanmin(im), np.nanmax(im)
    if im_max > im_min:
        intensity = (im - im_min) / (im_max - im_min)
    else:
        intensity = np.ones_like(im)
    r_min, r_max = np.nanmin(ridge), np.nanmax(ridge)
    if r_max > r_min:
        ridge_n = (ridge - r_min) / (r_max - r_min)
    else:
        ridge_n = np.zeros_like(ridge)
    # Cost: prefer bright + ridge. Higher ridge_weight avoids shortcut through low-ridge interior.
    rw = max(0.0, min(1.0, float(ridge_weight)))
    cost = 1.0 - (rw * ridge_n + (1.0 - rw) * intensity)
    cost = np.clip(cost, 1e-6, 1.0)

    sy, sx = int(round(start_xy[1])), int(round(start_xy[0]))
    ey, ex = int(round(end_xy[1])), int(round(end_xy[0]))
    sy = max(0, min(h - 1, sy))
    sx = max(0, min(w - 1, sx))
    ey = max(0, min(h - 1, ey))
    ex = max(0, min(w - 1, ex))

    # Optional: forbid path through low-ridge pixels (except near start/end for click tolerance)
    if ridge_floor_threshold > 0:
        yy, xx = np.ogrid[:h, :w]
        d_start = np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)
        d_end = np.sqrt((yy - ey) ** 2 + (xx - ex) ** 2)
        exempt = (d_start <= 3) | (d_end <= 3)
        low_ridge = ridge_n < ridge_floor_threshold
        cost = np.where(low_ridge & ~exempt, 10.0, cost)

    path = path_dijkstra_ridge(
        cost, (sy, sx), (ey, ex), max_bending_angle_deg=max_bending_angle_deg
    )
    if path is None or len(path) < 2:
        return None, 0.0
    length_px = contour_length_spline(path)
    return path, length_px


# ---------------------------------------------------------------------------
# Full image window: top = AFM + ROI selector, bottom = ROI crop
# ---------------------------------------------------------------------------

class ContourLengthFullImageWindow(QtWidgets.QMainWindow):
    """全画像でROI選択、下段にROIクロップを1枚表示。ROI上で2点クリックで端点指定→輪郭長計測。"""

    def __init__(
        self,
        parent=None,
        roi_basic_group: Optional[QtWidgets.QWidget] = None,
        contour_length_group: Optional[QtWidgets.QWidget] = None,
        frame_nav_widget: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Contour Length: ROI & Trace")
        self.resize(700, 600)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        h_layout = QtWidgets.QHBoxLayout(central_widget)

        self.figure = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.figure)
        h_layout.addWidget(self.canvas)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.3)
        self.ax_afm = self.figure.add_subplot(gs[0])
        self.ax_roi = self.figure.add_subplot(gs[1])

        self.cbar = None
        self.selector = None
        self.on_select_callback = None
        self._endpoints: List[Tuple[float, float]] = []
        self._endpoint_click_count = 0  # 0=none, 1=start set, 2=end set (normal left-clicks only)
        self._shift_held = False
        self._path_overlay = None
        self._path_length: Optional[float] = None
        self._path_pixels: Optional[List[Tuple[int, int]]] = None
        self._ridge_sigma = 1.5
        self._ridge_weight = 0.9
        self._ridge_floor_threshold = 0.0
        self._max_bending_angle_deg: Optional[float] = None
        self._on_length_computed_callback = None
        self._nm_per_pixel: Optional[float] = None
        self._marker_size = 2
        self._marker_shape = "o"  # matplotlib marker: o, s, ^, v, D, etc.
        self._cmap_name = "viridis"

        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.canvas.mpl_connect("key_release_event", self._on_key_release)

        if roi_basic_group is not None:
            roi_basic_group.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
            )
            wrapper = QtWidgets.QWidget()
            wrap_layout = QtWidgets.QVBoxLayout(wrapper)
            wrap_layout.setContentsMargins(0, 0, 0, 0)
            wrap_layout.setAlignment(QtCore.Qt.AlignTop)
            wrap_layout.addWidget(roi_basic_group)
            if contour_length_group is not None:
                contour_length_group.setSizePolicy(
                    QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
                )
                wrap_layout.addWidget(contour_length_group)
            if frame_nav_widget is not None:
                frame_nav_widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
                )
                wrap_layout.addWidget(frame_nav_widget)
            right_scroll = QtWidgets.QScrollArea()
            right_scroll.setWidget(wrapper)
            right_scroll.setWidgetResizable(True)
            right_scroll.setMinimumWidth(200)
            right_scroll.setMaximumWidth(240)
            right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            h_layout.addWidget(right_scroll)

    def set_ridge_sigma(self, sigma: float) -> None:
        self._ridge_sigma = max(0.5, float(sigma))

    def set_ridge_weight(self, w: float) -> None:
        """0–1: weight for ridge in cost (higher = prefer ridge, avoid U-shortcut)."""
        self._ridge_weight = max(0.0, min(1.0, float(w)))

    def set_ridge_floor_threshold(self, t: float) -> None:
        """0 = off; pixels with ridge_n below this get high cost (except near start/end)."""
        self._ridge_floor_threshold = max(0.0, float(t))

    def set_max_bending_angle_deg(self, deg: Optional[float]) -> None:
        """None or 0 = off; otherwise limit bending angle (DNA persistence)."""
        if deg is None or float(deg) <= 0:
            self._max_bending_angle_deg = None
        else:
            self._max_bending_angle_deg = max(1.0, float(deg))

    def set_marker_size(self, size: float) -> None:
        """Marker size in points for start/end/waypoint dots."""
        self._marker_size = max(2.0, min(60.0, float(size)))

    def set_marker_shape(self, mpl_marker: str) -> None:
        """Matplotlib marker character: o, s, ^, v, D, etc."""
        self._marker_shape = str(mpl_marker) if mpl_marker else "o"

    def set_cmap(self, cmap_name: str) -> None:
        """Set colormap for AFM and ROI images. Falls back to 'viridis' if invalid."""
        if not cmap_name or not cmap_name.strip():
            self._cmap_name = "viridis"
            return
        try:
            import matplotlib.pyplot as plt
            plt.get_cmap(cmap_name.strip())
            self._cmap_name = cmap_name.strip()
        except Exception:
            self._cmap_name = "viridis"

    def set_on_length_computed(self, callback) -> None:
        """Main window sets this so that when 2nd endpoint is set we can report length."""
        self._on_length_computed_callback = callback

    def set_nm_per_pixel(self, nm_per_pixel: Optional[float]) -> None:
        """Set scale (nm/px) for displaying contour length in nm."""
        self._nm_per_pixel = float(nm_per_pixel) if nm_per_pixel is not None and nm_per_pixel > 0 else None

    def enable_roi_selector(self, shape: str, callback) -> None:
        from matplotlib.widgets import RectangleSelector, EllipseSelector
        self.on_select_callback = callback
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

        def _emit_rect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            if callback:
                callback({"shape": "Rectangle", "x0": xmin, "y0": ymin, "w": xmax - xmin, "h": ymax - ymin})

        def _emit_ellipse(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            rx, ry = (xmax - xmin) / 2.0, (ymax - ymin) / 2.0
            if callback:
                callback({"shape": "Ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry, "x0": xmin, "y0": ymin, "w": xmax - xmin, "h": ymax - ymin})

        if shape == "Ellipse (Circle)":
            self.selector = EllipseSelector(
                self.ax_afm, _emit_ellipse, useblit=True, button=[1],
                minspanx=2, minspany=2, interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )
        else:
            self.selector = RectangleSelector(
                self.ax_afm, _emit_rect, useblit=True, button=[1],
                minspanx=2, minspany=2, interactive=False,
                props=dict(edgecolor="lime", facecolor="none", linewidth=1.5, linestyle="--"),
            )

    def _on_key_press(self, event) -> None:
        if getattr(event, "key", None) == "shift":
            self._shift_held = True

    def _on_key_release(self, event) -> None:
        if getattr(event, "key", None) == "shift":
            self._shift_held = False

    def clear_contour(self) -> None:
        """Clear endpoints, path overlay, and length. Redraw ROI."""
        self._endpoints = []
        self._endpoint_click_count = 0
        self._path_overlay = None
        self._path_length = None
        self._path_pixels = None
        self._draw_roi_with_endpoints_and_path()

    def _on_press(self, event):
        if event.inaxes != self.ax_roi or event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        pt = (float(x), float(y))
        # Shift: prefer guiEvent modifier (reliable when clicking); fallback to key_press state
        shift_pressed = self._shift_held
        if hasattr(event, "guiEvent") and event.guiEvent is not None:
            try:
                mod = event.guiEvent.modifiers()
                shift_pressed = bool(mod & QtCore.Qt.ShiftModifier)
            except Exception:
                pass
        if shift_pressed:
            # Shift+Click = waypoint (only when start set, end not set)
            if self._endpoint_click_count == 1:
                self._endpoints.append(pt)
        else:
            # Left click = endpoint: 1st = start, 2nd = end; or "start over" if already 2
            if self._endpoint_click_count == 0:
                self._endpoints.append(pt)
                self._endpoint_click_count = 1
            elif self._endpoint_click_count == 1:
                self._endpoints.append(pt)
                self._endpoint_click_count = 2
            else:
                # Already have start and end: this click = new start (clear and re-specify)
                self._endpoints = [pt]
                self._endpoint_click_count = 1
        self._draw_roi_with_endpoints_and_path()
        if self._endpoint_click_count >= 2 and len(self._endpoints) >= 2 and hasattr(self, "_roi_image") and self._roi_image is not None:
            self.run_trace_and_length(self._roi_image, self._on_length_computed_callback)

    def _draw_roi_with_endpoints_and_path(self) -> None:
        if not hasattr(self, "_roi_image") or self._roi_image is None:
            return
        self.ax_roi.clear()
        self.ax_roi.imshow(self._roi_image, cmap=self._cmap_name, origin="lower")
        n_pt = len(self._endpoints)
        for i, (px, py) in enumerate(self._endpoints):
            if i == 0:
                color = "red"  # start
            elif i == n_pt - 1:
                color = "cyan"  # end
            else:
                color = "yellow"  # waypoint
            self.ax_roi.plot(
                px, py, self._marker_shape, color=color,
                markersize=self._marker_size, markeredgewidth=2,
            )
        if self._path_pixels and len(self._path_pixels) >= 2:
            xs = [p[1] for p in self._path_pixels]
            ys = [p[0] for p in self._path_pixels]
            self.ax_roi.plot(xs, ys, "r-", linewidth=2, alpha=0.9)
        if self._path_length is not None:
            if self._nm_per_pixel is not None and self._nm_per_pixel > 0:
                self.ax_roi.set_title(f"Contour length: {self._path_length * self._nm_per_pixel:.3f} nm")
            else:
                self.ax_roi.set_title(f"Contour length: {self._path_length:.3f} px")
        else:
            self.ax_roi.set_title("ROI: 左クリックで始点・終点。U字などは Shift+Click で通過点を追加")
        self.canvas.draw_idle()

    def run_trace_and_length(self, roi_image: np.ndarray, on_length_computed) -> None:
        """Compute path and length from [start, waypoints..., end]. Callback(length_px) when done."""
        if len(self._endpoints) < 2 or roi_image is None or roi_image.size == 0:
            return
        points = self._endpoints
        if len(points) == 2:
            path, length_px = compute_contour_between_points(
                roi_image,
                points[0],
                points[1],
                ridge_sigma=self._ridge_sigma,
                ridge_weight=self._ridge_weight,
                ridge_floor_threshold=self._ridge_floor_threshold,
                max_bending_angle_deg=self._max_bending_angle_deg,
            )
            self._path_pixels = path
            self._path_length = length_px if path else None
        else:
            # Concatenate segments: start -> wp1 -> ... -> end (avoid duplicate junction pixels)
            combined: List[Tuple[int, int]] = []
            for i in range(len(points) - 1):
                seg, _ = compute_contour_between_points(
                    roi_image,
                    points[i],
                    points[i + 1],
                    ridge_sigma=self._ridge_sigma,
                    ridge_weight=self._ridge_weight,
                    ridge_floor_threshold=self._ridge_floor_threshold,
                    max_bending_angle_deg=self._max_bending_angle_deg,
                )
                if seg is None or len(seg) < 2:
                    self._path_pixels = None
                    self._path_length = None
                    self._roi_image = roi_image
                    self._draw_roi_with_endpoints_and_path()
                    return
                if i == 0:
                    combined.extend(seg)
                else:
                    combined.extend(seg[1:])
            self._path_pixels = combined
            self._path_length = contour_length_spline(combined) if len(combined) >= 2 else 0.0
        self._roi_image = roi_image
        self._draw_roi_with_endpoints_and_path()
        if on_length_computed and self._path_length is not None:
            on_length_computed(self._path_length)

    def update_view(
        self,
        frame: np.ndarray,
        roi_overlay: Optional[Dict[str, Any]] = None,
        roi_image: Optional[np.ndarray] = None,
        recorded_contours: Optional[List[Dict[str, Any]]] = None,
        current_frame_index: Optional[int] = None,
        current_file_id: Optional[str] = None,
    ) -> None:
        self.ax_afm.clear()
        self.ax_roi.clear()
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        self.ax_afm.imshow(frame, cmap=self._cmap_name, origin="lower")
        self.ax_afm.set_title("Full AFM image (drag to set ROI)")

        if recorded_contours and current_frame_index is not None:
            for i, rec in enumerate(recorded_contours):
                if rec.get("frame_index") != current_frame_index:
                    continue
                # Only show contours for the current file (file_id); old records without file_id match any
                rec_fid = rec.get("file_id")
                if rec_fid is not None and current_file_id is not None and rec_fid != current_file_id:
                    continue
                path_xy = rec.get("path_full_xy") or []
                if len(path_xy) < 2:
                    continue
                xs = [p[0] for p in path_xy]
                ys = [p[1] for p in path_xy]
                self.ax_afm.plot(xs, ys, "lime", linewidth=1.5, alpha=0.9)
                self.ax_afm.text(xs[0], ys[0], f" {i + 1}", color="lime", fontsize=9, va="bottom")

        if roi_overlay:
            if roi_overlay.get("shape") == "Ellipse":
                el = Ellipse(
                    (roi_overlay["cx"], roi_overlay["cy"]),
                    width=roi_overlay["rx"] * 2.0,
                    height=roi_overlay["ry"] * 2.0,
                    linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--",
                )
                self.ax_afm.add_patch(el)
            else:
                rect = Rectangle(
                    (roi_overlay["x0"], roi_overlay["y0"]),
                    roi_overlay["w"], roi_overlay["h"],
                    linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--",
                )
                self.ax_afm.add_patch(rect)

        self._roi_image = roi_image
        if roi_image is not None and roi_image.size > 0:
            self.ax_roi.imshow(roi_image, cmap=self._cmap_name, origin="lower")
            if self._path_length is not None:
                if self._nm_per_pixel is not None and self._nm_per_pixel > 0:
                    self.ax_roi.set_title(f"ROI — Contour length: {self._path_length * self._nm_per_pixel:.3f} nm")
                else:
                    self.ax_roi.set_title(f"ROI — Contour length: {self._path_length:.3f} px")
            else:
                self.ax_roi.set_title("ROI: 左クリックで始点・終点。U字などは Shift+Click で通過点を追加")
            n_pt = len(self._endpoints)
            for i, (px, py) in enumerate(self._endpoints):
                color = "red" if i == 0 else ("cyan" if i == n_pt - 1 else "yellow")
                self.ax_roi.plot(
                    px, py, self._marker_shape, color=color,
                    markersize=self._marker_size, markeredgewidth=2,
                )
            if self._path_pixels and len(self._path_pixels) >= 2:
                xs = [p[1] for p in self._path_pixels]
                ys = [p[0] for p in self._path_pixels]
                self.ax_roi.plot(xs, ys, "r-", linewidth=2, alpha=0.9)
        else:
            self.ax_roi.set_title("ROI (select region above)")
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main plugin window
# ---------------------------------------------------------------------------

class ContourLengthWindow(QtWidgets.QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle(PLUGIN_NAME)
        self.setMinimumWidth(380)
        self.full_viz_window: Optional[ContourLengthFullImageWindow] = None
        self.last_frame: Optional[np.ndarray] = None
        self.manual_roi: Optional[Dict[str, Any]] = None
        self.roi_by_frame: Dict[int, Dict[str, Any]] = {}
        self._recorded_contours_list: List[Dict[str, Any]] = []
        self._build_ui()
        self._connect_frame_signal()
        self._show_full_image_view()
        self._update_frame_label()
        self._update_recorded_table()

    def _build_ui(self) -> None:
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        gb = QtWidgets.QGroupBox("ROI / 基本")
        grid = QtWidgets.QGridLayout(gb)
        r = 0
        self.roi_status_label = QtWidgets.QLabel("ROI未選択")
        grid.addWidget(self.roi_status_label, r, 0, 1, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("カラーマップ"), r, 0)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["viridis", "gray", "magma", "inferno", "plasma", "hot", "bone", "coolwarm"])
        self.cmap_combo.setToolTip("AFM画像の表示カラーマップ")
        self.cmap_combo.setMaximumWidth(80)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        grid.addWidget(self.cmap_combo, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("点のサイズ (pt)"), r, 0)
        self.marker_size_spin = QtWidgets.QDoubleSpinBox()
        self.marker_size_spin.setRange(2, 60)
        self.marker_size_spin.setValue(2)
        self.marker_size_spin.setSingleStep(1)
        self.marker_size_spin.setMaximumWidth(80)
        self.marker_size_spin.setToolTip("始点・終点・通過点のマーカーサイズ")
        self.marker_size_spin.valueChanged.connect(self._on_marker_style_changed)
        grid.addWidget(self.marker_size_spin, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("点の形"), r, 0)
        self.marker_shape_combo = QtWidgets.QComboBox()
        self.marker_shape_combo.addItems(["丸", "四角", "三角上", "三角下", "菱形"])
        self.marker_shape_combo.setToolTip("始点・終点・通過点のマーカー形状")
        self.marker_shape_combo.setMaximumWidth(80)
        self.marker_shape_combo.currentIndexChanged.connect(self._on_marker_style_changed)
        grid.addWidget(self.marker_shape_combo, r, 1)
        r += 1
        self.roi_basic_group = gb

        gb_ridge = QtWidgets.QGroupBox("リッジ条件")
        grid_ridge = QtWidgets.QGridLayout(gb_ridge)
        rr = 0
        grid_ridge.addWidget(QtWidgets.QLabel("リッジ σ (px)"), rr, 0)
        self.ridge_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.ridge_sigma_spin.setRange(0.5, 10.0)
        self.ridge_sigma_spin.setValue(1.5)
        self.ridge_sigma_spin.setSingleStep(0.5)
        self.ridge_sigma_spin.valueChanged.connect(self._on_ridge_sigma_changed)
        grid_ridge.addWidget(self.ridge_sigma_spin, rr, 1)
        rr += 1
        grid_ridge.addWidget(QtWidgets.QLabel("最大曲げ角度 (deg)"), rr, 0)
        self.max_bending_spin = QtWidgets.QDoubleSpinBox()
        self.max_bending_spin.setRange(0, 180)
        self.max_bending_spin.setValue(0)
        self.max_bending_spin.setSingleStep(5)
        self.max_bending_spin.setSpecialValueText("オフ")
        self.max_bending_spin.setToolTip("交差点で急角度を禁止。0=オフ。DNAの持続長に合わせて30–90程度を推奨")
        self.max_bending_spin.valueChanged.connect(self._on_max_bending_changed)
        grid_ridge.addWidget(self.max_bending_spin, rr, 1)
        rr += 1
        grid_ridge.addWidget(QtWidgets.QLabel("リッジ重み"), rr, 0)
        self.ridge_weight_spin = QtWidgets.QDoubleSpinBox()
        self.ridge_weight_spin.setRange(0.5, 1.0)
        self.ridge_weight_spin.setValue(0.9)
        self.ridge_weight_spin.setSingleStep(0.05)
        self.ridge_weight_spin.setToolTip("コストでリッジを重視。高めにするとU字の内側を通りにくい")
        self.ridge_weight_spin.valueChanged.connect(self._on_ridge_weight_changed)
        grid_ridge.addWidget(self.ridge_weight_spin, rr, 1)
        rr += 1
        grid_ridge.addWidget(QtWidgets.QLabel("リッジ閾値（通過禁止）"), rr, 0)
        self.ridge_floor_spin = QtWidgets.QDoubleSpinBox()
        self.ridge_floor_spin.setRange(0, 0.5)
        self.ridge_floor_spin.setValue(0)
        self.ridge_floor_spin.setSingleStep(0.05)
        self.ridge_floor_spin.setSpecialValueText("オフ")
        self.ridge_floor_spin.setToolTip("リッジがこの値未満のピクセルを通れなくする。0=オフ。端点付近は除外")
        self.ridge_floor_spin.valueChanged.connect(self._on_ridge_floor_changed)
        grid_ridge.addWidget(self.ridge_floor_spin, rr, 1)
        rr += 1
        layout.addWidget(gb_ridge)

        gb_pre = QtWidgets.QGroupBox("前処理（測長用ROI）")
        grid_pre = QtWidgets.QGridLayout(gb_pre)
        rp = 0
        self.noise_check = QtWidgets.QCheckBox("ノイズ軽減（メディアン）")
        self.noise_check.setToolTip("測長前にメディアンフィルタでノイズを軽減")
        self.noise_check.stateChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.noise_check, rp, 0, 1, 2)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("カーネル k"), rp, 0)
        self.median_spin = QtWidgets.QSpinBox()
        self.median_spin.setRange(0, 15)
        self.median_spin.setValue(3)
        self.median_spin.setSpecialValueText("オフ")
        self.median_spin.setToolTip("0=オフ。奇数推奨（3,5,7）")
        self.median_spin.valueChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.median_spin, rp, 1)
        rp += 1
        self.deconv_check = QtWidgets.QCheckBox("デコンボリューション (Richardson-Lucy)")
        self.deconv_check.setToolTip("探針ボケを軽減し、重なった鎖の境界をシャープに")
        self.deconv_check.stateChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.deconv_check, rp, 0, 1, 2)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("デコンボ PSF σ (px)"), rp, 0)
        self.deconv_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.deconv_sigma_spin.setRange(0.3, 5.0)
        self.deconv_sigma_spin.setValue(1.0)
        self.deconv_sigma_spin.setSingleStep(0.2)
        self.deconv_sigma_spin.valueChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.deconv_sigma_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("デコンボ 反復回数"), rp, 0)
        self.deconv_iter_spin = QtWidgets.QSpinBox()
        self.deconv_iter_spin.setRange(1, 50)
        self.deconv_iter_spin.setValue(10)
        self.deconv_iter_spin.valueChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.deconv_iter_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("背景処理"), rp, 0)
        self.flatten_combo = QtWidgets.QComboBox()
        self.flatten_combo.addItems(["なし", "平面引き", "ガウシアン引き"])
        self.flatten_combo.setToolTip("ROI内の傾き・ドリフトを除去")
        self.flatten_combo.currentIndexChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.flatten_combo, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("背景 σ (px)"), rp, 0)
        self.flatten_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.flatten_sigma_spin.setRange(1.0, 100.0)
        self.flatten_sigma_spin.setValue(10.0)
        self.flatten_sigma_spin.setToolTip("ガウシアン引き時のみ。鎖幅より大きく")
        self.flatten_sigma_spin.valueChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.flatten_sigma_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("コントラスト強調"), rp, 0)
        self.contrast_combo = QtWidgets.QComboBox()
        self.contrast_combo.addItems(["なし", "パーセンタイル", "ガンマ"])
        self.contrast_combo.setToolTip("測長前のコントラスト調整")
        self.contrast_combo.currentIndexChanged.connect(self._refresh_view)
        grid_pre.addWidget(self.contrast_combo, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("パーセンタイル Low"), rp, 0)
        self.contrast_low_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_low_spin.setRange(0, 100)
        self.contrast_low_spin.setValue(2.0)
        grid_pre.addWidget(self.contrast_low_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("パーセンタイル High"), rp, 0)
        self.contrast_high_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_high_spin.setRange(0, 100)
        self.contrast_high_spin.setValue(98.0)
        grid_pre.addWidget(self.contrast_high_spin, rp, 1)
        rp += 1
        grid_pre.addWidget(QtWidgets.QLabel("ガンマ"), rp, 0)
        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.2, 3.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        grid_pre.addWidget(self.gamma_spin, rp, 1)
        rp += 1
        self.contrast_low_spin.valueChanged.connect(self._refresh_view)
        self.contrast_high_spin.valueChanged.connect(self._refresh_view)
        self.gamma_spin.valueChanged.connect(self._refresh_view)
        layout.addWidget(gb_pre)

        gb2 = QtWidgets.QGroupBox("輪郭長")
        v2 = QtWidgets.QVBoxLayout(gb2)
        instr_label = QtWidgets.QLabel("ROI画像上で左クリックで始点・終点。U字などは Shift+Click で通過点を追加")
        instr_label.setWordWrap(True)
        instr_label.setStyleSheet("font-size: 9px;")
        v2.addWidget(instr_label)
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.setMaximumWidth(60)
        self.ok_btn.clicked.connect(self._on_ok_record)
        self.ok_btn.setToolTip("現在の輪郭長を記録し、Full AFM画像に輪郭を描画")
        v2.addWidget(self.ok_btn, 0, QtCore.Qt.AlignLeft)
        self.length_label = QtWidgets.QLabel("輪郭長: — nm")
        self.length_label.setStyleSheet("font-weight: bold;")
        v2.addWidget(self.length_label)
        self.contour_length_group = gb2

        self.frame_label = QtWidgets.QLabel("Frame: —")
        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.prev_btn.clicked.connect(self._prev_frame)
        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.clicked.connect(self._next_frame)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.prev_btn)
        btn_row.addWidget(self.next_btn)
        btn_row.setAlignment(QtCore.Qt.AlignLeft)
        self.frame_nav_widget = QtWidgets.QWidget()
        frame_nav_layout = QtWidgets.QVBoxLayout(self.frame_nav_widget)
        frame_nav_layout.setContentsMargins(6, 0, 0, 0)
        frame_nav_layout.addWidget(self.frame_label)
        frame_nav_layout.addLayout(btn_row)

        gb_table = QtWidgets.QGroupBox("記録した輪郭長")
        table_layout = QtWidgets.QVBoxLayout(gb_table)
        self.recorded_table = QtWidgets.QTableWidget(0, 2)
        self.recorded_table.setHorizontalHeaderLabels(["ID", "輪郭長 (nm)"])
        self.recorded_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.recorded_table.setAlternatingRowColors(True)
        self.recorded_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.recorded_table.customContextMenuRequested.connect(self._on_recorded_table_context_menu)
        copy_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence.Copy, self.recorded_table, self._copy_table_to_clipboard
        )
        copy_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.recorded_table)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(120)
        scroll.setMaximumHeight(220)
        table_layout.addWidget(scroll)
        layout.addWidget(gb_table)

        layout.addStretch()
        data_clear_btn = QtWidgets.QPushButton("Data Clear")
        data_clear_btn.clicked.connect(self._on_data_clear)
        layout.addWidget(data_clear_btn)
        save_session_btn = QtWidgets.QPushButton("Save Session")
        save_session_btn.clicked.connect(self._save_session)
        layout.addWidget(save_session_btn)
        load_session_btn = QtWidgets.QPushButton("Load Session")
        load_session_btn.clicked.connect(self._load_session)
        layout.addWidget(load_session_btn)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

    def _on_ok_record(self) -> None:
        """Record current contour length and path; draw the path on Full AFM image."""
        if self.full_viz_window is None or not self.full_viz_window._path_pixels or len(self.full_viz_window._path_pixels) < 2:
            QtWidgets.QMessageBox.information(self, "記録", "輪郭を計測してからOKを押してください。")
            return
        if self.manual_roi is None:
            return
        path_pixels = self.full_viz_window._path_pixels
        length_px = self.full_viz_window._path_length
        if length_px is None:
            return
        x0 = float(self.manual_roi.get("x0", 0))
        y0 = float(self.manual_roi.get("y0", 0))
        path_full_xy = [(x0 + p[1], y0 + p[0]) for p in path_pixels]
        nm_per_px = self._get_nm_per_pixel()
        length_nm = (length_px * nm_per_px) if (nm_per_px and nm_per_px > 0) else None
        fi = self._get_current_frame_index()
        file_id = self._get_current_file_id()
        rec = {
            "path_full_xy": path_full_xy,
            "length_px": length_px,
            "length_nm": length_nm,
            "frame_index": fi,
            "file_id": file_id,
        }
        self._recorded_contours_list.append(rec)
        self._update_recorded_table()
        self._refresh_view()

    def _on_data_clear(self) -> None:
        """Clear all recorded contour data, table, and AFM overlay."""
        self._recorded_contours_list.clear()
        self._update_recorded_table()
        self._refresh_view()

    def _default_session_path(self) -> str:
        """Default save path: same dir and base name as first measured file, extension .contour_session.json."""
        if not self._recorded_contours_list:
            return os.path.join(os.getcwd(), "contour_session.json")
        first_file_id = self._recorded_contours_list[0].get("file_id") or ""
        if isinstance(first_file_id, str) and (os.sep in first_file_id or "/" in first_file_id):
            base = os.path.splitext(os.path.basename(first_file_id))[0]
            dirpath = os.path.dirname(first_file_id)
            if dirpath:
                return os.path.join(dirpath, base + ".contour_session.json")
        return os.path.join(os.getcwd(), "contour_session.json")

    def _save_session(self) -> None:
        """Save session (recorded contours, ROI, file info) to JSON. Default path from first measured file; overwrite prompt if exists."""
        file_paths = list(getattr(gv, "files", []) or [])
        try:
            file_paths = [str(p) for p in file_paths]
        except Exception:
            file_paths = []
        current_file_index = int(getattr(gv, "currentFileNum", 0))
        if current_file_index < 0 or current_file_index >= len(file_paths):
            current_file_index = 0
        current_frame_index = int(getattr(gv, "index", 0))
        roi_by_frame_str = {str(k): v for k, v in self.roi_by_frame.items()}
        payload = {
            "version": 1,
            "file_paths": file_paths,
            "current_file_index": current_file_index,
            "current_frame_index": current_frame_index,
            "manual_roi": self.manual_roi,
            "roi_by_frame": roi_by_frame_str,
            "recorded_contours": self._recorded_contours_list,
        }
        default_path = self._default_session_path()
        default_dir = os.path.dirname(default_path)
        default_name = os.path.basename(default_path)
        path = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Session",
            default_path,
            "JSON (*.json);;All Files (*)",
        )[0]
        if not path:
            return
        path = path.strip()
        if not path:
            return
        if not path.endswith(".json"):
            path = path + ".json"
        while os.path.exists(path):
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("上書き確認")
            msg.setText(f"ファイル '{os.path.basename(path)}' は既に存在します。上書きしますか？")
            msg.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            msg.setButtonText(QtWidgets.QMessageBox.Yes, "上書き")
            msg.setButtonText(QtWidgets.QMessageBox.No, "ファイル名を変更")
            ret = msg.exec_()
            if ret == QtWidgets.QMessageBox.Yes:
                break
            path = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Session",
                default_path,
                "JSON (*.json);;All Files (*)",
            )[0]
            if not path or not path.strip():
                return
            path = path.strip()
            if not path.endswith(".json"):
                path = path + ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            QtWidgets.QMessageBox.information(self, "Save Session", f"セッションを保存しました:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Session", f"保存に失敗しました:\n{e}")

    def _load_session(self) -> None:
        """Load session from JSON and restore contours, ROI, and optionally file selection."""
        path = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Session",
            "",
            "JSON (*.json);;All Files (*)",
        )[0]
        if not path or not path.strip():
            return
        path = path.strip()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Session", f"読み込みに失敗しました:\n{e}")
            return
        version = data.get("version", 0)
        if version != 1:
            QtWidgets.QMessageBox.warning(self, "Load Session", f"不明なバージョンです: {version}")
            return
        self._recorded_contours_list = data.get("recorded_contours", [])
        self.manual_roi = data.get("manual_roi")
        roi_by_frame_raw = data.get("roi_by_frame", {})
        try:
            self.roi_by_frame = {int(k): v for k, v in roi_by_frame_raw.items()}
        except (ValueError, TypeError):
            self.roi_by_frame = {}
        if self.manual_roi:
            self.roi_status_label.setText("ROI選択済み")
        else:
            self.roi_status_label.setText("ROI未選択")
        file_paths = data.get("file_paths", [])
        current_file_index = int(data.get("current_file_index", 0))
        found_row = -1
        if file_paths and 0 <= current_file_index < len(file_paths) and self.main_window and hasattr(self.main_window, "FileList"):
            target_path = file_paths[current_file_index]
            try:
                target_path = str(target_path)
            except Exception:
                target_path = None
            if target_path:
                fl = self.main_window.FileList
                for row in range(fl.count()):
                    item = fl.item(row)
                    if item is None:
                        continue
                    try:
                        row_path = str(item.text() if hasattr(item, "text") else item.data(QtCore.Qt.UserRole) or "")
                        if row_path == target_path or target_path.endswith(row_path) or row_path.endswith(target_path):
                            found_row = row
                            break
                    except Exception:
                        continue
                if found_row >= 0:
                    try:
                        fl.setCurrentRow(found_row)
                        if hasattr(self.main_window, "ListClickFunction"):
                            try:
                                self.main_window.ListClickFunction(bring_to_front=False)
                            except TypeError:
                                self.main_window.ListClickFunction()
                    except Exception:
                        pass
        self._update_recorded_table()
        self._refresh_view()
        if found_row >= 0:
            QtWidgets.QMessageBox.information(self, "Load Session", "セッションを読み込みました。")
        elif file_paths:
            QtWidgets.QMessageBox.information(
                self, "Load Session", "セッションを読み込みました。同じファイルを開くとオーバーレイが表示されます。"
            )
        else:
            QtWidgets.QMessageBox.information(self, "Load Session", "セッションを読み込みました。")

    def _update_recorded_table(self) -> None:
        """Fill the recorded contour table (all frames, ID continues across frames)."""
        rows = self._recorded_contours_list
        self.recorded_table.setRowCount(len(rows))
        for i, rec in enumerate(rows):
            self.recorded_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            ln = rec.get("length_nm")
            if ln is not None:
                self.recorded_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{ln:.3f}"))
            else:
                self.recorded_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{rec.get('length_px', 0):.3f} px"))
        self.recorded_table.scrollToBottom()

    def _copy_table_to_clipboard(self) -> None:
        """Copy table (selection or full) to clipboard as tab-separated for Excel."""
        table = self.recorded_table
        ranges = table.selectedRanges()
        if ranges:
            r = ranges[0]
            top, bottom = r.topRow(), r.bottomRow()
            left, right = r.leftColumn(), r.rightColumn()
            lines = []
            for row in range(top, bottom + 1):
                cells = []
                for col in range(left, right + 1):
                    it = table.item(row, col)
                    cells.append(it.text() if it is not None else "")
                lines.append("\t".join(cells))
            text = "\n".join(lines)
        else:
            header = "\t".join(
                table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else ""
                for c in range(table.columnCount())
            )
            lines = [header]
            for row in range(table.rowCount()):
                cells = []
                for col in range(table.columnCount()):
                    it = table.item(row, col)
                    cells.append(it.text() if it is not None else "")
                lines.append("\t".join(cells))
            text = "\n".join(lines)
        if text:
            cb = QtWidgets.QApplication.clipboard()
            cb.setText(text)

    def _on_recorded_table_context_menu(self, pos: QtCore.QPoint) -> None:
        """Right-click on table: show delete menu and remove the row and corresponding overlay."""
        index = self.recorded_table.indexAt(pos)
        row = index.row()
        if row < 0:
            return
        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction("この行を削除")
        action = menu.exec_(self.recorded_table.viewport().mapToGlobal(pos))
        if action is delete_action:
            if 0 <= row < len(self._recorded_contours_list):
                self._recorded_contours_list.pop(row)
                self._update_recorded_table()
                self._refresh_view()

    def _on_full_image_selected(self, roi_info: Dict[str, Any]) -> None:
        w = roi_info.get("w", 0)
        h = roi_info.get("h", 0)
        if w <= 1 or h <= 1:
            return
        fi = self._get_current_frame_index()
        self.roi_by_frame[fi] = roi_info
        self.manual_roi = roi_info
        self.roi_status_label.setText("ROI選択済み")
        if self.full_viz_window:
            self.full_viz_window.clear_contour()
        self._refresh_view()

    def _get_current_frame_index(self) -> int:
        return int(getattr(gv, "index", 0))

    def _get_current_file_id(self) -> Optional[str]:
        """Return a string that identifies the current file (path or index). None if no file."""
        if not hasattr(gv, "files") or not gv.files:
            return None
        idx = getattr(gv, "currentFileNum", -1)
        if idx < 0 or idx >= len(gv.files):
            return None
        try:
            return str(gv.files[idx])
        except Exception:
            return str(idx)

    def _get_nm_per_pixel(self) -> Optional[float]:
        """Return nm per pixel (average of X and Y) from gv scan size, or None if unavailable."""
        if not hasattr(gv, "XScanSize") or not hasattr(gv, "YScanSize"):
            return None
        if not hasattr(gv, "XPixel") or not hasattr(gv, "YPixel"):
            return None
        if getattr(gv, "XScanSize", 0) == 0 or getattr(gv, "YScanSize", 0) == 0:
            return None
        if getattr(gv, "XPixel", 0) == 0 or getattr(gv, "YPixel", 0) == 0:
            return None
        nm_x = gv.XScanSize / gv.XPixel
        nm_y = gv.YScanSize / gv.YPixel
        return (float(nm_x) + float(nm_y)) / 2.0

    def _ensure_selection_loaded(self) -> bool:
        if not self.main_window or not hasattr(self.main_window, "FileList"):
            QtWidgets.QMessageBox.warning(self, "No Selection", "FileListが見つかりません。")
            return False
        selected = self.main_window.FileList.selectedIndexes()
        target_row = selected[0].row() if selected else self.main_window.FileList.currentRow()
        if target_row is None or target_row < 0:
            QtWidgets.QMessageBox.information(self, "Select File", "ファイルリストで対象を選択してください。")
            return False
        try:
            blocker = QtCore.QSignalBlocker(self.main_window.FileList)
            self.main_window.FileList.setCurrentRow(target_row)
        except Exception:
            self.main_window.FileList.setCurrentRow(target_row)
        if hasattr(self.main_window, "ListClickFunction"):
            try:
                self.main_window.ListClickFunction(bring_to_front=False)
            except TypeError:
                self.main_window.ListClickFunction()
        return True

    def _prepare_frame(self) -> Optional[np.ndarray]:
        if not hasattr(gv, "files") or not gv.files:
            QtWidgets.QMessageBox.information(self, "No Files", "ファイルがロードされていません。")
            return None
        if getattr(gv, "currentFileNum", -1) < 0 or gv.currentFileNum >= len(gv.files):
            return None
        try:
            LoadFrame(gv.files[gv.currentFileNum])
            InitializeAryDataFallback()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"フレーム読み込みに失敗:\n{exc}")
            return None
        if not hasattr(gv, "aryData") or gv.aryData is None:
            return None
        # メイン画面と同じ: leveling/filter 適用済みデータを優先（aryData_processed_1ch）
        data_for_display = (
            gv.aryData_processed_1ch
            if (hasattr(gv, "aryData_processed_1ch") and gv.aryData_processed_1ch is not None)
            else gv.aryData
        )
        frame = np.asarray(data_for_display, dtype=np.float64)
        if frame.ndim != 2:
            return None
        return frame

    def _roi_bounds(self, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        if self.manual_roi is None:
            return None
        h_img, w_img = frame_shape
        x0 = int(round(self.manual_roi["x0"]))
        y0 = int(round(self.manual_roi["y0"]))
        x1 = int(round(self.manual_roi["x0"] + self.manual_roi["w"]))
        y1 = int(round(self.manual_roi["y0"] + self.manual_roi["h"]))
        x0c = max(x0, 0)
        y0c = max(y0, 0)
        x1c = min(x1, w_img)
        y1c = min(y1, h_img)
        return (x0c, y0c, x1c - x0c, y1c - y0c)

    def _get_roi_crop(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        if self.last_frame is None or self.manual_roi is None:
            return None, None
        bounds = self._roi_bounds(self.last_frame.shape)
        if bounds is None:
            return None, None
        roi_crop, _ = _crop_rect(self.last_frame, bounds)
        return roi_crop, self.manual_roi

    def _apply_preprocess(self, roi: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply UI-selected preprocessing for measurement and display."""
        if roi is None or roi.size == 0:
            return roi
        median_k = 0
        if getattr(self, "noise_check", None) and self.noise_check.isChecked() and getattr(self, "median_spin", None):
            median_k = int(self.median_spin.value())
        flatten_mode = "none"
        flatten_sigma = 10.0
        if getattr(self, "flatten_combo", None):
            t = self.flatten_combo.currentText().strip()
            if "平面" in t:
                flatten_mode = "plane"
            elif "ガウシアン" in t:
                flatten_mode = "gaussian"
                if getattr(self, "flatten_sigma_spin", None):
                    flatten_sigma = float(self.flatten_sigma_spin.value())
        contrast_mode = "none"
        contrast_low, contrast_high = 2.0, 98.0
        gamma = 1.0
        if getattr(self, "contrast_combo", None):
            t = self.contrast_combo.currentText().strip()
            if "パーセンタイル" in t:
                contrast_mode = "percentile"
                if getattr(self, "contrast_low_spin", None):
                    contrast_low = float(self.contrast_low_spin.value())
                if getattr(self, "contrast_high_spin", None):
                    contrast_high = float(self.contrast_high_spin.value())
            elif "ガンマ" in t:
                contrast_mode = "gamma"
                if getattr(self, "gamma_spin", None):
                    gamma = float(self.gamma_spin.value())
        deconv_enable = (
            getattr(self, "deconv_check", None) is not None
            and self.deconv_check.isChecked()
        )
        deconv_sigma = 1.0
        deconv_iterations = 10
        if deconv_enable:
            if getattr(self, "deconv_sigma_spin", None):
                deconv_sigma = float(self.deconv_sigma_spin.value())
            if getattr(self, "deconv_iter_spin", None):
                deconv_iterations = int(self.deconv_iter_spin.value())
        return apply_roi_preprocess(
            roi,
            median_k=median_k,
            deconv_enable=deconv_enable,
            deconv_sigma=deconv_sigma,
            deconv_iterations=deconv_iterations,
            flatten_mode=flatten_mode,
            flatten_sigma=flatten_sigma,
            contrast_mode=contrast_mode,
            contrast_low=contrast_low,
            contrast_high=contrast_high,
            gamma=gamma,
        )

    def _refresh_view(self) -> None:
        if self.full_viz_window is None:
            return
        if self.last_frame is None:
            return
        self.full_viz_window.set_nm_per_pixel(self._get_nm_per_pixel())
        roi_crop, roi_overlay = self._get_roi_crop()
        roi_display = self._apply_preprocess(roi_crop) if roi_crop is not None else None
        self.full_viz_window.update_view(
            self.last_frame,
            roi_overlay=roi_overlay,
            roi_image=roi_display if roi_display is not None else roi_crop,
            recorded_contours=self._recorded_contours_list,
            current_frame_index=self._get_current_frame_index(),
            current_file_id=self._get_current_file_id(),
        )

    def _rerun_trace_if_ready(self) -> None:
        """If ROI exists and full window has 2 endpoints, re-run contour trace and update display."""
        if self.full_viz_window is None or self.last_frame is None:
            return
        roi_crop, _ = self._get_roi_crop()
        if roi_crop is None:
            return
        roi_display = self._apply_preprocess(roi_crop)
        if roi_display is None:
            roi_display = roi_crop
        self.full_viz_window.run_trace_and_length(roi_display, self._on_length_computed)
        self._refresh_view()

    def _on_ridge_sigma_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_ridge_sigma(self.ridge_sigma_spin.value())
        self._rerun_trace_if_ready()

    def _on_max_bending_changed(self) -> None:
        if self.full_viz_window:
            v = self.max_bending_spin.value()
            self.full_viz_window.set_max_bending_angle_deg(v if v > 0 else None)
        self._rerun_trace_if_ready()

    def _on_ridge_weight_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_ridge_weight(self.ridge_weight_spin.value())
        self._rerun_trace_if_ready()

    def _on_ridge_floor_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_ridge_floor_threshold(self.ridge_floor_spin.value())
        self._rerun_trace_if_ready()

    def _on_marker_style_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_marker_size(self.marker_size_spin.value())
            text = self.marker_shape_combo.currentText().strip()
            self.full_viz_window.set_marker_shape(MARKER_SHAPE_MAP.get(text, "o"))
        self._refresh_view()

    def _on_cmap_changed(self) -> None:
        if self.full_viz_window:
            self.full_viz_window.set_cmap(self.cmap_combo.currentText().strip() or "viridis")
        self._refresh_view()

    def _on_length_computed(self, length_px: float) -> None:
        nm_per_px = self._get_nm_per_pixel()
        if nm_per_px is not None and nm_per_px > 0:
            self.length_label.setText(f"輪郭長: {length_px * nm_per_px:.3f} nm")
        else:
            self.length_label.setText(f"輪郭長: {length_px:.3f} px (スケール不明)")

    def _connect_frame_signal(self) -> None:
        if self.main_window and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.connect(self._on_frame_changed)
            except Exception:
                pass

    def _on_frame_changed(self, frame_index: int) -> None:
        self.manual_roi = self.roi_by_frame.get(frame_index)
        if self.manual_roi is None:
            self.roi_status_label.setText("ROI未選択")
        else:
            self.roi_status_label.setText("ROI選択済み")
        frame = self._prepare_frame()
        if frame is None:
            return
        self.last_frame = frame
        self._refresh_view()
        self._update_frame_label()

    def _update_frame_label(self) -> None:
        total = int(getattr(gv, "FrameNum", 0)) or 0
        current = self._get_current_frame_index()
        if total > 0:
            self.frame_label.setText(f"Frame: {current + 1} / {total}")
        else:
            self.frame_label.setText("Frame: —")

    def _prev_frame(self) -> None:
        total = int(getattr(gv, "FrameNum", 0)) or 0
        if total <= 0:
            return
        idx = max(0, self._get_current_frame_index() - 1)
        self._set_frame_index(idx)

    def _next_frame(self) -> None:
        total = int(getattr(gv, "FrameNum", 0)) or 0
        if total <= 0:
            return
        idx = min(total - 1, self._get_current_frame_index() + 1)
        self._set_frame_index(idx)

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
        self._update_frame_label()

    def _show_full_image_view(self) -> None:
        if not self._ensure_selection_loaded():
            return
        frame = self._prepare_frame()
        if frame is None:
            QtWidgets.QMessageBox.information(self, "No Data", "画像がロードされていません。")
            return
        self.last_frame = frame
        if self.full_viz_window is None:
            self.full_viz_window = ContourLengthFullImageWindow(
                self,
                roi_basic_group=self.roi_basic_group,
                contour_length_group=self.contour_length_group,
                frame_nav_widget=self.frame_nav_widget,
            )
        self.full_viz_window.set_on_length_computed(self._on_length_computed)
        self.full_viz_window.set_ridge_sigma(self.ridge_sigma_spin.value())
        self.full_viz_window.set_ridge_weight(self.ridge_weight_spin.value())
        self.full_viz_window.set_ridge_floor_threshold(self.ridge_floor_spin.value())
        v = self.max_bending_spin.value()
        self.full_viz_window.set_max_bending_angle_deg(v if v > 0 else None)
        self.full_viz_window.set_marker_size(self.marker_size_spin.value())
        text = self.marker_shape_combo.currentText().strip()
        self.full_viz_window.set_marker_shape(MARKER_SHAPE_MAP.get(text, "o"))
        self.full_viz_window.set_cmap(self.cmap_combo.currentText().strip() or "viridis")
        self.full_viz_window.set_nm_per_pixel(self._get_nm_per_pixel())
        self.full_viz_window.enable_roi_selector("Rectangle", self._on_full_image_selected)
        roi_crop, roi_overlay = self._get_roi_crop()
        roi_display = self._apply_preprocess(roi_crop) if roi_crop is not None else None
        self.full_viz_window.update_view(
            frame,
            roi_overlay=roi_overlay,
            roi_image=roi_display if roi_display is not None else roi_crop,
            recorded_contours=self._recorded_contours_list,
            current_frame_index=self._get_current_frame_index(),
            current_file_id=self._get_current_file_id(),
        )
        self.full_viz_window.show()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.full_viz_window is None:
            self._show_full_image_view()

    def closeEvent(self, event) -> None:
        if self.full_viz_window:
            self.full_viz_window.close()
            self.full_viz_window = None
        super().closeEvent(event)


def create_plugin(main_window):
    return ContourLengthWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin"]
