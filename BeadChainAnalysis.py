"""
Bead chain fluctuation analysis plugin for pyNuD.

This plugin targets Cordonin-SAHH-like bead-on-string HS-AFM movies. Users
place anchor points along one chain; the plugin traces the ridge, detects bead
centers along the straightened trace, propagates the trace through frames, and
exports per-bead fluctuation metrics.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy import ndimage, signal
from scipy.spatial import cKDTree

import globalvals as gv
from fileio import InitializeAryDataFallback, LoadFrame

try:
    from ScleroglucanDectinAnalysis import (
        _arc_lengths_nm,
        _relative_height,
        _straighten_trace_strip,
    )
except ImportError:
    from plugins.ScleroglucanDectinAnalysis import (
        _arc_lengths_nm,
        _relative_height,
        _straighten_trace_strip,
    )

try:
    from FilamentAnalysis import compute_ridge_map, path_dijkstra_ridge
except ImportError:
    from plugins.FilamentAnalysis import compute_ridge_map, path_dijkstra_ridge


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


def _trace_ridge_segment(
    rel: np.ndarray,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    frangi_sigma: float,
    ridge_weight: float,
) -> Optional[np.ndarray]:
    """Trace one anchor-to-anchor ridge segment using FilamentAnalysis helpers."""
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
        self.store_button = QtWidgets.QPushButton("Store")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.propagate_button = QtWidgets.QPushButton("Next Auto")
        self.run_all_button = QtWidgets.QPushButton("Run all")
        self.export_button = QtWidgets.QPushButton("Export")
        self.save_session_button = QtWidgets.QPushButton("Save Session")
        self.load_session_button = QtWidgets.QPushButton("Load Session")

        buttons = [
            self.add_anchor_button,
            self.store_button,
            self.clear_button,
            self.propagate_button,
            self.run_all_button,
            self.export_button,
            self.save_session_button,
            self.load_session_button,
        ]
        for idx, button in enumerate(buttons):
            button_grid.addWidget(button, idx // 2, idx % 2)

        self.store_button.clicked.connect(self.store_current_line)
        self.clear_button.clicked.connect(self.clear_current)
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

    def _on_canvas_press(self, event) -> None:
        if not self.add_anchor_button.isChecked():
            return
        if self.rel is None or event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        h, w = self.rel.shape
        x = max(0.0, min(float(w - 1), float(event.xdata)))
        y = max(0.0, min(float(h - 1), float(event.ydata)))
        self.anchor_points_xy.append((x, y))
        self._update_current_path()
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
        radius_nm = max(half_width_nm, min_spacing_nm * 0.45, min(nm_x, nm_y) * 2.0)
        h, w = rel.shape
        for peak in peaks:
            idx = max(0, min(detection_path_yx.shape[0] - 1, int(peak)))
            y0_px = float(detection_path_yx[idx, 0])
            x0_px = float(detection_path_yx[idx, 1])
            rx = max(1, int(math.ceil(radius_nm / max(nm_x, 1e-9))))
            ry = max(1, int(math.ceil(radius_nm / max(nm_y, 1e-9))))
            x0 = max(0, int(round(x0_px)) - rx)
            x1 = min(w, int(round(x0_px)) + rx + 1)
            y0 = max(0, int(round(y0_px)) - ry)
            y1 = min(h, int(round(y0_px)) + ry + 1)
            crop = np.asarray(rel[y0:y1, x0:x1], dtype=float)
            if crop.size == 0:
                continue
            finite_crop = crop[np.isfinite(crop)]
            if finite_crop.size == 0:
                continue
            local_base = float(np.percentile(finite_crop, 20.0))
            threshold = local_base
            if min_height > 0:
                threshold = max(threshold, min_height)
            weights = np.clip(np.nan_to_num(crop - threshold, nan=0.0), 0.0, None)
            if float(np.sum(weights)) <= 1e-12:
                weights = np.clip(np.nan_to_num(crop - local_base, nan=0.0), 0.0, None)
            if float(np.sum(weights)) <= 1e-12:
                yy, xx = np.unravel_index(int(np.nanargmax(crop)), crop.shape)
                x_px = float(x0 + xx)
                y_px = float(y0 + yy)
            else:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                total = float(np.sum(weights))
                x_px = float(np.sum(weights * xx) / total)
                y_px = float(np.sum(weights * yy) / total)
            x_nm = x_px * nm_x
            y_nm = y_px * nm_y
            s_value = _nearest_path_s_nm(points_yx, arc, x_nm, y_nm, nm_x, nm_y)
            height = _sample_bilinear(rel, y_px, x_px)
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
    ) -> ChainFrameResult:
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
        arc = _arc_lengths_nm(points_yx, nm_x, nm_y)
        result = ChainFrameResult(
            source_path=self._current_file_path(),
            image_id=self._current_image_id(),
            frame_index=int(frame_index),
            anchor_points_xy=[(float(x), float(y)) for x, y in anchors_xy],
            points_yx=np.asarray(points_yx, dtype=int).copy(),
            length_nm=float(arc[-1]) if arc.size else 0.0,
            beads=beads,
        )
        self._assign_fluctuation_components(result)
        return result

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

    def store_current_line(self) -> None:
        """Store or overwrite the current frame trace and bead detections."""
        if self.rel is None:
            QtWidgets.QMessageBox.information(self, "No Frame", "Load an AFM frame first.")
            return
        if self.current_path_yx is None:
            self._update_current_path()
        if self.current_path_yx is None or self.current_path_yx.shape[0] < 2:
            QtWidgets.QMessageBox.information(self, "No Line", "Draw a valid bead-chain line first.")
            return
        template_ids = self.reference_bead_order if self.reference_bead_order and len(self.reference_bead_order) == len(self.preview_beads) else None
        result = self._make_result_from_path(
            self.rel,
            self.current_path_yx,
            self.anchor_points_xy,
            self._current_frame_index(),
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
        self.last_result = result
        self.anchor_points_xy = []
        self.current_path_yx = None
        self.preview_beads = []
        self.preview_diverged = False
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

    def clear_current(self) -> None:
        self.anchor_points_xy = []
        self.current_path_yx = None
        self.preview_beads = []
        self.preview_diverged = False
        self.status_label.setText("Current anchors and preview line cleared.")
        self._refresh_source_label()
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
        template_ids = [bead.bead_id for bead in sorted(previous.beads, key=lambda b: b.s_nm)]
        result = self._make_result_from_path(rel, path, anchors, frame_index, template_ids=template_ids)
        mean_dev, max_dev = _path_deviation_nm(previous.points_yx, result.points_yx, nm_x, nm_y)
        result.mean_deviation_nm = mean_dev
        result.max_deviation_nm = max_dev
        result.max_bead_shift_nm = self._max_bead_shift(previous, result)
        threshold = float(self.deviation_spin.value())
        reasons = []
        if len(result.beads) != len(previous.beads):
            reasons.append(f"bead count {len(previous.beads)} -> {len(result.beads)}")
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
                    stop_message = f"Frame {frame_index + 1}: divergence flagged ({result.status})."
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
            self._restore_global_frame_state(saved_state, update_main=True)
            self.refresh_frame(clear_working=True)
        if stop_message:
            self.status_label.setText(
                f"Run all stopped after storing {stored} frame(s), skipped {skipped}. {stop_message}"
            )
        else:
            self.status_label.setText(f"Run all complete. Stored {stored} propagated frame(s), skipped {skipped}.")
        self._update_result_list()

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
        vmin, vmax = self._display_range()
        self.ax.imshow(self.rel, cmap="afmhot", origin="lower", vmin=vmin, vmax=vmax)
        stored = self._result_for_current_frame()
        if stored is not None:
            self._draw_result(stored, line_color="cyan", point_color="lime", label_prefix="")
        if self.current_path_yx is not None and self.current_path_yx.shape[0] >= 2:
            line_color = "magenta" if self.preview_diverged else "yellow"
            self.ax.plot(self.current_path_yx[:, 1], self.current_path_yx[:, 0], color=line_color, linewidth=1.8, alpha=0.95)
            if self.preview_beads:
                self._draw_beads(self.preview_beads, color="magenta" if self.preview_diverged else "white")
        if self.anchor_points_xy:
            xs = [point[0] for point in self.anchor_points_xy]
            ys = [point[1] for point in self.anchor_points_xy]
            self.ax.scatter(xs, ys, s=34, c="white", edgecolors="black", linewidths=0.8, zorder=5)
            for idx, (x, y) in enumerate(self.anchor_points_xy, start=1):
                self.ax.text(x, y, str(idx), color="white", fontsize=8, zorder=6)
        self.ax.set_title(f"{ANALYSIS_NAME}: bead chain trace")
        self.ax.set_xlim(0, self.rel.shape[1])
        self.ax.set_ylim(0, self.rel.shape[0])
        self.ax.set_xlabel("x px")
        self.ax.set_ylabel("y px")
        self.canvas.draw_idle()

    def _draw_result(self, result: ChainFrameResult, line_color: str, point_color: str, label_prefix: str) -> None:
        points = result.points_yx
        if points.shape[0] >= 2:
            self.ax.plot(points[:, 1], points[:, 0], color=line_color, linewidth=1.8, alpha=0.95)
        self._draw_beads(result.beads, color=point_color, label_prefix=label_prefix)

    def _draw_beads(self, beads: Sequence[BeadObservation], color: str, label_prefix: str = "") -> None:
        if not beads:
            return
        xs = [bead.x_px for bead in beads]
        ys = [bead.y_px for bead in beads]
        self.ax.scatter(xs, ys, s=42, c=color, edgecolors="black", linewidths=0.8, zorder=6)
        for bead in beads:
            self.ax.text(
                bead.x_px,
                bead.y_px,
                f"{label_prefix}{bead.bead_id}",
                color=color,
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
        QtWidgets.QMessageBox.information(self, "Export Complete", f"Results exported to:\n{out_dir}")

    def _export_all(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        prefix = self._current_image_id()
        self._export_per_bead_csv(os.path.join(out_dir, f"{prefix}_bead_chain_per_bead.csv"))
        self._export_summary_csv(os.path.join(out_dir, f"{prefix}_bead_chain_summary.csv"))
        self._export_spacing_csv(os.path.join(out_dir, f"{prefix}_bead_chain_spacing.csv"))
        self.figure.savefig(os.path.join(out_dir, f"{prefix}_bead_chain_overlay.png"), dpi=220)
        self._save_trajectory_plot(os.path.join(out_dir, f"{prefix}_bead_chain_trajectories.png"))
        self._save_msd_plot(os.path.join(out_dir, f"{prefix}_bead_chain_msd.png"))
        self._save_fluctuation_histogram(os.path.join(out_dir, f"{prefix}_bead_chain_fluctuation_histogram.png"))

    def _iter_bead_rows(self) -> List[Tuple[ChainFrameResult, BeadObservation]]:
        rows: List[Tuple[ChainFrameResult, BeadObservation]] = []
        for result in sorted(self.results, key=lambda row: (self._path_key(row.source_path), row.frame_index)):
            for bead in sorted(result.beads, key=lambda bead: bead.bead_id):
                rows.append((result, bead))
        return rows

    def _export_per_bead_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file",
                "frame",
                "time_s",
                "bead_id",
                "s_nm",
                "x_nm",
                "y_nm",
                "height",
                "longitudinal_nm",
                "transverse_nm",
            ])
            for result, bead in self._iter_bead_rows():
                writer.writerow([
                    result.source_path,
                    result.frame_index + 1,
                    _format_float(self._frame_time_s(result.frame_index)),
                    bead.bead_id,
                    _format_float(bead.s_nm),
                    _format_float(bead.x_nm),
                    _format_float(bead.y_nm),
                    _format_float(bead.height),
                    _format_float(bead.longitudinal_nm),
                    _format_float(bead.transverse_nm),
                ])

    def _export_summary_csv(self, path: str) -> None:
        summary = self._compute_summary_rows()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "bead_id",
                "n_frames",
                "sigma_long_nm",
                "sigma_trans_nm",
                "sigma_xy_nm",
                "msd_coeff_nm2_per_s",
                "mean_spacing_to_next_nm",
                "spacing_sigma_to_next_nm",
            ])
            for row in summary:
                writer.writerow([
                    row["bead_id"],
                    row["n_frames"],
                    _format_float(row["sigma_long_nm"]),
                    _format_float(row["sigma_trans_nm"]),
                    _format_float(row["sigma_xy_nm"]),
                    _format_float(row["msd_coeff_nm2_per_s"]),
                    _format_float(row["mean_spacing_to_next_nm"]),
                    _format_float(row["spacing_sigma_to_next_nm"]),
                ])

    def _export_spacing_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "frame", "time_s", "bead_id_a", "bead_id_b", "distance_nm"])
            for result in sorted(self.results, key=lambda row: (self._path_key(row.source_path), row.frame_index)):
                beads = sorted(result.beads, key=lambda bead: bead.s_nm)
                for a, b in zip(beads[:-1], beads[1:]):
                    distance = math.hypot(b.x_nm - a.x_nm, b.y_nm - a.y_nm)
                    writer.writerow([
                        result.source_path,
                        result.frame_index + 1,
                        _format_float(self._frame_time_s(result.frame_index)),
                        a.bead_id,
                        b.bead_id,
                        _format_float(distance),
                    ])

    def _compute_summary_rows(self) -> List[Dict[str, float]]:
        by_bead: Dict[int, List[Tuple[ChainFrameResult, BeadObservation]]] = {}
        for result, bead in self._iter_bead_rows():
            by_bead.setdefault(bead.bead_id, []).append((result, bead))
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
            dx = xs - np.nanmean(xs) if xs.size else xs
            dy = ys - np.nanmean(ys) if ys.size else ys
            xy_sigma = float(np.sqrt(np.nanmean(dx * dx + dy * dy))) if xs.size else float("nan")
            spacings = np.asarray(spacing_by_bead.get(bead_id, []), dtype=float)
            rows.append({
                "bead_id": int(bead_id),
                "n_frames": int(len(items)),
                "sigma_long_nm": float(np.nanstd(longs)) if longs.size else float("nan"),
                "sigma_trans_nm": float(np.nanstd(trans)) if trans.size else float("nan"),
                "sigma_xy_nm": xy_sigma,
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

    def _save_trajectory_plot(self, path: str) -> None:
        fig = Figure(figsize=(6.4, 5.2), tight_layout=True)
        ax = fig.add_subplot(111)
        by_bead: Dict[int, List[BeadObservation]] = {}
        for _result, bead in self._iter_bead_rows():
            by_bead.setdefault(bead.bead_id, []).append(bead)
        for bead_id, beads in sorted(by_bead.items()):
            beads.sort(key=lambda bead: bead.frame_index)
            ax.plot([b.x_nm for b in beads], [b.y_nm for b in beads], marker="o", linewidth=1.0, markersize=3, label=str(bead_id))
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title("Bead centroid trajectories")
        if by_bead:
            ax.legend(title="bead", fontsize=7, ncol=2)
        fig.savefig(path, dpi=220)

    def _save_msd_plot(self, path: str) -> None:
        fig = Figure(figsize=(6.4, 5.2), tight_layout=True)
        ax = fig.add_subplot(111)
        by_bead: Dict[int, List[Tuple[ChainFrameResult, BeadObservation]]] = {}
        for result, bead in self._iter_bead_rows():
            by_bead.setdefault(bead.bead_id, []).append((result, bead))
        for bead_id, items in sorted(by_bead.items()):
            lag_times, msd = self._msd_by_lag(items)
            if lag_times.size:
                ax.plot(lag_times, msd, marker="o", linewidth=1.0, markersize=3, label=str(bead_id))
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("MSD (nm^2)")
        ax.set_title("Bead MSD")
        if by_bead:
            ax.legend(title="bead", fontsize=7, ncol=2)
        fig.savefig(path, dpi=220)

    def _save_fluctuation_histogram(self, path: str) -> None:
        longs = []
        trans = []
        for _result, bead in self._iter_bead_rows():
            if np.isfinite(bead.longitudinal_nm):
                longs.append(bead.longitudinal_nm)
            if np.isfinite(bead.transverse_nm):
                trans.append(bead.transverse_nm)
        fig = Figure(figsize=(6.4, 5.2), tight_layout=True)
        ax = fig.add_subplot(111)
        if longs:
            ax.hist(longs, bins=30, alpha=0.55, label="longitudinal")
        if trans:
            ax.hist(trans, bins=30, alpha=0.55, label="transverse")
        ax.set_xlabel("fluctuation (nm)")
        ax.set_ylabel("count")
        ax.set_title("Longitudinal/transverse fluctuation")
        ax.legend()
        fig.savefig(path, dpi=220)

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
        path, _selected = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Session",
            self._session_default_path(),
            "JSON files (*.json);;All files (*)",
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
        return os.path.join(default_dir, f"{self._current_image_id()}_bead_chain_session.json")

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
        for result in self.results:
            self._assign_fluctuation_components(result)
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
            points_yx=np.rint(points).astype(int),
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
