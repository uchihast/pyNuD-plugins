"""
Reference-based object cropper plugin for pyNuD.

Workflow:
1. Draw an ROI around one structure in the current frame.
2. Click "Set Reference" to memorize the ROI and derive an object mask.
3. Change frames manually in pyNuD.
4. Click "Crop Current" to find the best-matching object and extract it.
5. Review extracted crops in the side panel (right-click thumbnail → set ROI center on click) and save each as its own ASD (fileio.SaveASD, same as Save tab); filenames use _f{frame}_c{crop_id}.
"""

from __future__ import annotations

# Accept direct .py selection by a host still using the single-file loader.
if not __package__:
    try:
        from plugin_bundle import adopt_legacy_entry
    except ImportError as exc:
        raise RuntimeError(
            "This plugin requires updated pyNuD bundle support. "
            "Update pyNuD and restart it, then select the plugin folder."
        ) from exc
    adopt_legacy_entry(__name__, __file__)


import os
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

try:
    import globalvals as gv
except ImportError:
    gv = None

try:
    from fileio import InitializeAryDataFallback, LoadFrame, SaveASD
except ImportError:
    InitializeAryDataFallback = None
    LoadFrame = None
    SaveASD = None


PLUGIN_NAME = "Reference Object Cropper"

# Stored in CropRecord.match_score for user-added crops (thumbnail shows "manual").
MANUAL_CROP_SCORE = -1.0

# QListWidgetItem data: UserRole + 1 = True if this crop is from the frame/channel shown in the main viewer.
THUMB_USERROLE_IS_CURRENT = QtCore.Qt.UserRole + 1


class ExtractedCropThumbnailDelegate(QtWidgets.QStyledItemDelegate):
    """Paints a strong border/background so current-frame crops are obvious in Icon mode."""

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index) -> None:
        rect = option.rect
        is_cur = index.data(THUMB_USERROLE_IS_CURRENT)
        is_cur = bool(is_cur) if is_cur is not None else False

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if is_cur:
            bg = QtGui.QColor("#065f46")
            border = QtGui.QColor("#34d399")
            pen_w = 4
        else:
            bg = QtGui.QColor("#171717")
            border = QtGui.QColor("#404040")
            pen_w = 1
        painter.setPen(QtGui.QPen(border, pen_w))
        painter.setBrush(QtGui.QBrush(bg))
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 10, 10)
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.setPen(QtGui.QPen(QtGui.QColor("#fbbf24"), 3))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 11, 11)
        painter.restore()

        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.backgroundBrush = QtGui.QBrush(QtCore.Qt.transparent)
        pal = opt.palette
        pal.setBrush(QtGui.QPalette.Base, QtGui.QBrush(QtCore.Qt.transparent))
        pal.setBrush(QtGui.QPalette.AlternateBase, QtGui.QBrush(QtCore.Qt.transparent))
        fg = QtGui.QColor("#ecfdf5") if is_cur else QtGui.QColor("#b0b0b0")
        pal.setBrush(QtGui.QPalette.Text, QtGui.QBrush(fg))
        opt.palette = pal
        super().paint(painter, opt, index)


@dataclass
class CropRecord:
    frame_index: int
    object_id: int
    channel: str
    match_score: float
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    crop: np.ndarray
    source_path: str = ""
    source_fingerprint: tuple | None = None
    pixel_size_nm: tuple | None = None
    conversion_type: str = "Other"
    conversion_factor: float = 1.0
    signal_kind: int | None = None


def _normalize_float(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-12:
        out = np.zeros(arr.shape, dtype=np.float32)
        out[finite] = 0.5
        return out
    out = np.zeros(arr.shape, dtype=np.float32)
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return out


def _normalize_to_u8(array: np.ndarray) -> np.ndarray:
    norm = _normalize_float(array)
    return np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)


def _otsu_threshold(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.5
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo < 1e-12:
        return lo
    hist, bin_edges = np.histogram(vals, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_total = mu[-1]
    sigma_b = (mu_total * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    idx = int(np.nanargmax(sigma_b))
    return float(bin_edges[min(idx + 1, len(bin_edges) - 1)])


def _component_mask(
    mask: np.ndarray,
    center_seed: tuple[int, int],
    peak_seed: tuple[int, int],
    fill_holes: bool = True,
) -> np.ndarray:
    if not np.any(mask):
        return mask
    if ndimage is None:
        return mask

    labeled, count = ndimage.label(mask)
    if count <= 0:
        return mask

    chosen = 0
    cy, cx = center_seed
    py, px = peak_seed
    if 0 <= cy < labeled.shape[0] and 0 <= cx < labeled.shape[1]:
        chosen = int(labeled[cy, cx])
    if chosen == 0 and 0 <= py < labeled.shape[0] and 0 <= px < labeled.shape[1]:
        chosen = int(labeled[py, px])
    if chosen == 0:
        labels, counts = np.unique(labeled[labeled > 0], return_counts=True)
        if labels.size:
            best_score = -np.inf
            center_y = float(cy)
            center_x = float(cx)
            peak_y = float(py)
            peak_x = float(px)
            for label_id, area in zip(labels, counts):
                ys, xs = np.nonzero(labeled == label_id)
                if ys.size == 0:
                    continue
                comp_cy = float(np.mean(ys))
                comp_cx = float(np.mean(xs))
                dist_center = float(np.hypot(comp_cy - center_y, comp_cx - center_x))
                dist_peak = float(np.hypot(comp_cy - peak_y, comp_cx - peak_x))
                score = (
                    1.8 * np.log1p(float(area))
                    - 1.25 * dist_center
                    - 0.55 * dist_peak
                )
                if score > best_score:
                    best_score = score
                    chosen = int(label_id)
    if chosen == 0:
        return mask

    refined = labeled == chosen
    if fill_holes:
        refined = ndimage.binary_fill_holes(refined)
    if min(refined.shape) >= 3:
        refined = ndimage.binary_opening(refined, structure=np.ones((3, 3), dtype=bool))
        refined = ndimage.binary_closing(refined, structure=np.ones((3, 3), dtype=bool))
    return refined.astype(bool)


def _infer_object_polarity(patch: np.ndarray) -> bool:
    norm = _normalize_float(patch)
    h, w = norm.shape
    cy0 = max(0, h // 2 - max(1, h // 6))
    cy1 = min(h, h // 2 + max(1, h // 6) + 1)
    cx0 = max(0, w // 2 - max(1, w // 6))
    cx1 = min(w, w // 2 + max(1, w // 6) + 1)
    center_region = norm[cy0:cy1, cx0:cx1]

    global_mean = float(np.mean(norm))
    center_mean = float(np.mean(center_region)) if center_region.size else global_mean
    return center_mean >= global_mean


def _build_object_masks(
    patch: np.ndarray,
    bright_object: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    norm = _normalize_float(patch)
    h, w = norm.shape
    bright = _infer_object_polarity(patch) if bright_object is None else bool(bright_object)
    threshold = _otsu_threshold(norm.ravel())
    mask = norm >= threshold if bright else norm <= threshold

    if np.count_nonzero(mask) < max(4, patch.size // 40):
        percentile = 70 if bright else 30
        pivot = float(np.percentile(norm, percentile))
        mask = norm >= pivot if bright else norm <= pivot

    if bright:
        peak_seed = tuple(int(v) for v in np.unravel_index(int(np.nanargmax(norm)), norm.shape))
    else:
        peak_seed = tuple(int(v) for v in np.unravel_index(int(np.nanargmin(norm)), norm.shape))
    center_seed = (h // 2, w // 2)
    band_mask = _component_mask(
        mask.astype(bool),
        center_seed,
        peak_seed,
        fill_holes=False,
    )
    filled_mask = _component_mask(
        mask.astype(bool),
        center_seed,
        peak_seed,
        fill_holes=True,
    )

    if np.count_nonzero(filled_mask) < max(4, patch.size // 60):
        filled_mask = np.ones_like(mask, dtype=bool)
        band_mask = filled_mask.copy()
    return filled_mask, band_mask, bright


def _build_object_mask(
    patch: np.ndarray,
    bright_object: bool | None = None,
) -> tuple[np.ndarray, bool]:
    filled_mask, _band_mask, bright = _build_object_masks(
        patch,
        bright_object=bright_object,
    )
    return filled_mask, bright


def _build_reference_mask(patch: np.ndarray) -> tuple[np.ndarray, bool]:
    return _build_object_mask(patch, bright_object=None)


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        h, w = mask.shape
        return h / 2.0, w / 2.0
    return float(np.mean(ys)), float(np.mean(xs))


def _weighted_intensity_centroid(
    patch: np.ndarray,
    mask: np.ndarray,
    bright_object: bool,
) -> tuple[float, float]:
    """Intensity-weighted centroid on ``mask`` (I²): favors the dominant bright/dark blob.

    When two objects share one Otsu foreground, the geometric mask centroid sits between them;
    weighting by squared normalized intensity pulls the center toward the main peak.
    """
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return _mask_center(m)
    norm = _normalize_float(patch)
    if bright_object:
        wmap = np.square(np.clip(norm, 1e-6, 1.0))
    else:
        wmap = np.square(np.clip(1.0 - norm, 1e-6, 1.0))
    w = np.where(m, wmap, 0.0)
    s = float(np.sum(w))
    if s < 1e-18:
        return _mask_center(m)
    h, w0 = m.shape
    ys, xs = np.indices((h, w0), dtype=np.float64)
    cy = float(np.sum(ys * w) / s)
    cx = float(np.sum(xs * w) / s)
    return cy, cx


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        h, w = mask.shape
        return 0, 0, w, h
    y0 = int(np.min(ys))
    y1 = int(np.max(ys))
    x0 = int(np.min(xs))
    x1 = int(np.max(xs))
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _center_signature_masks(
    mask: np.ndarray,
    center: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    core_mask = np.zeros_like(mask, dtype=bool)
    halo_mask = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size < 8:
        return core_mask, halo_mask

    cy, cx = float(center[0]), float(center[1])
    dist = np.hypot(ys.astype(np.float32) - cy, xs.astype(np.float32) - cx)
    if not np.isfinite(dist).any():
        return core_mask, halo_mask

    core_q = float(np.percentile(dist, 35.0))
    halo_q0 = float(np.percentile(dist, 52.0))
    halo_q1 = float(np.percentile(dist, 86.0))

    core_sel = dist <= core_q
    halo_sel = (dist >= halo_q0) & (dist <= halo_q1)
    if np.count_nonzero(core_sel) < 6:
        core_sel = dist <= float(np.percentile(dist, 45.0))
    if np.count_nonzero(halo_sel) < 6:
        halo_sel = dist >= float(np.percentile(dist, 60.0))

    if np.count_nonzero(core_sel) >= 4:
        core_mask[ys[core_sel], xs[core_sel]] = True
    if np.count_nonzero(halo_sel) >= 4:
        halo_mask[ys[halo_sel], xs[halo_sel]] = True
    return core_mask, halo_mask


def _rectangle_iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU for axis-aligned rectangles (x, y, w, h) in pixel coordinates."""
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(a_x2, b_x2)
    iy1 = min(a_y2, b_y2)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return float(inter / max(ua, 1e-12))


def _radial_profile(image: np.ndarray, center: tuple[float, float], bins: int = 12) -> np.ndarray:
    img = _normalize_float(image)
    h, w = img.shape
    cy, cx = center
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(max(1.0, np.percentile(rr, 95)))
    edges = np.linspace(0.0, rmax, bins + 1)
    profile = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        sel = (rr >= edges[i]) & (rr < edges[i + 1])
        if np.any(sel):
            profile[i] = float(np.mean(img[sel]))
        else:
            profile[i] = profile[i - 1] if i > 0 else 0.0
    return profile


def _array_to_pixmap(array: np.ndarray) -> QtGui.QPixmap:
    if array is None or not isinstance(array, np.ndarray) or array.size == 0:
        return QtGui.QPixmap()

    finite = np.isfinite(array)
    base = np.zeros(array.shape, dtype=np.uint8)
    if np.any(finite):
        base = _normalize_to_u8(np.where(finite, array, np.nanmin(array[finite])))

    if cv2 is not None:
        bgr = cv2.applyColorMap(base, cv2.COLORMAP_HOT)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
        qimg = QtGui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QtGui.QImage.Format_RGBA8888,
        ).copy()
        return QtGui.QPixmap.fromImage(qimg)

    gray = base.copy()
    qimg = QtGui.QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QtGui.QImage.Format_Grayscale8).copy()
    return QtGui.QPixmap.fromImage(qimg)


class RoiImageViewer(QtWidgets.QWidget):
    roi_changed = QtCore.pyqtSignal()
    manual_crop_requested = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #101010; border: 1px solid #444;")

        self._pixmap = QtGui.QPixmap()
        self._image_shape: tuple[int, int] | None = None
        self._display_rect = QtCore.QRectF()
        self._roi_rect: QtCore.QRect | None = None
        self._match_rects: list[QtCore.QRect] = []
        self._dragging = False
        self._interaction_mode: str | None = None
        self._active_handle: str | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_end: tuple[int, int] | None = None
        self._roi_start_rect: QtCore.QRect | None = None

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        pt = self._widget_to_image(event.pos())
        if pt is None:
            return
        img_x, img_y = int(pt[0]), int(pt[1])
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu {"
            " background-color: #161616;"
            " color: #f3f4f6;"
            " border: 1px solid #4b5563;"
            " }"
            "QMenu::item {"
            " background-color: transparent;"
            " color: #f3f4f6;"
            " padding: 6px 24px 6px 12px;"
            " }"
            "QMenu::item:selected {"
            " background-color: #2563eb;"
            " color: #ffffff;"
            " }"
        )
        act_crop = menu.addAction("この位置を中心に手動で Crop を追加")
        act_clear = menu.addAction("Reference ROI をクリア")
        chosen = menu.exec_(event.globalPos())
        if chosen == act_crop:
            self.manual_crop_requested.emit(img_x, img_y)
        elif chosen == act_clear:
            self.clear_roi()

    def set_image(self, array: np.ndarray | None) -> None:
        self._image_shape = None if array is None else tuple(int(v) for v in array.shape[:2])
        self._pixmap = _array_to_pixmap(array) if array is not None else QtGui.QPixmap()
        self.update()

    def set_roi(self, rect: tuple[int, int, int, int] | None) -> None:
        if rect is None:
            self._roi_rect = None
        else:
            x, y, w, h = [int(v) for v in rect]
            self._roi_rect = QtCore.QRect(x, y, max(1, w), max(1, h))
        self.roi_changed.emit()
        self.update()

    def get_roi(self) -> tuple[int, int, int, int] | None:
        if self._roi_rect is None:
            return None
        return (
            int(self._roi_rect.x()),
            int(self._roi_rect.y()),
            int(self._roi_rect.width()),
            int(self._roi_rect.height()),
        )

    def clear_roi(self) -> None:
        self.set_roi(None)

    def set_match_rects(self, rects: list[tuple[int, int, int, int]] | None) -> None:
        self._match_rects = []
        if rects:
            for rect in rects:
                x, y, w, h = [int(v) for v in rect]
                self._match_rects.append(QtCore.QRect(x, y, max(1, w), max(1, h)))
        self.update()

    def _fit_rect(self) -> QtCore.QRectF:
        if self._pixmap.isNull():
            return QtCore.QRectF()
        pw = float(self._pixmap.width())
        ph = float(self._pixmap.height())
        if pw <= 0 or ph <= 0:
            return QtCore.QRectF()
        scale = min(self.width() / pw, self.height() / ph)
        dw = pw * scale
        dh = ph * scale
        x = (self.width() - dw) / 2.0
        y = (self.height() - dh) / 2.0
        return QtCore.QRectF(x, y, dw, dh)

    def _widget_to_image(self, pos: QtCore.QPoint) -> tuple[int, int] | None:
        if self._image_shape is None:
            return None
        rect = self._display_rect
        if rect.isNull() or not rect.contains(QtCore.QPointF(pos)):
            return None
        x_ratio = (pos.x() - rect.left()) / max(rect.width(), 1.0)
        y_ratio = (pos.y() - rect.top()) / max(rect.height(), 1.0)
        h, w = self._image_shape
        x = int(np.clip(round(x_ratio * (w - 1)), 0, max(w - 1, 0)))
        y = int(np.clip(round(y_ratio * (h - 1)), 0, max(h - 1, 0)))
        return x, y

    def _widget_to_image_clamped(self, pos: QtCore.QPoint) -> tuple[int, int] | None:
        if self._image_shape is None or self._display_rect.isNull():
            return None
        x = int(np.clip(pos.x(), int(self._display_rect.left()), int(self._display_rect.right())))
        y = int(np.clip(pos.y(), int(self._display_rect.top()), int(self._display_rect.bottom())))
        return self._widget_to_image(QtCore.QPoint(x, y))

    def _image_rect_to_widget(self, rect: QtCore.QRect | None) -> QtCore.QRectF:
        if rect is None or self._image_shape is None or self._display_rect.isNull():
            return QtCore.QRectF()
        h, w = self._image_shape
        sx = self._display_rect.width() / max(w, 1)
        sy = self._display_rect.height() / max(h, 1)
        return QtCore.QRectF(
            self._display_rect.left() + rect.x() * sx,
            self._display_rect.top() + rect.y() * sy,
            rect.width() * sx,
            rect.height() * sy,
        )

    def _handle_rects(self) -> dict[str, QtCore.QRectF]:
        wr = self._image_rect_to_widget(self._roi_rect)
        if wr.isNull():
            return {}
        hs = 10.0
        cx = wr.center().x()
        cy = wr.center().y()
        pts = {
            "nw": QtCore.QPointF(wr.left(), wr.top()),
            "n": QtCore.QPointF(cx, wr.top()),
            "ne": QtCore.QPointF(wr.right(), wr.top()),
            "e": QtCore.QPointF(wr.right(), cy),
            "se": QtCore.QPointF(wr.right(), wr.bottom()),
            "s": QtCore.QPointF(cx, wr.bottom()),
            "sw": QtCore.QPointF(wr.left(), wr.bottom()),
            "w": QtCore.QPointF(wr.left(), cy),
        }
        return {
            key: QtCore.QRectF(point.x() - hs / 2.0, point.y() - hs / 2.0, hs, hs)
            for key, point in pts.items()
        }

    def _hit_test_roi(self, pos: QtCore.QPoint) -> tuple[str | None, str | None]:
        if self._roi_rect is None:
            return None, None
        for handle, rect in self._handle_rects().items():
            if rect.contains(QtCore.QPointF(pos)):
                return "resize", handle
        wr = self._image_rect_to_widget(self._roi_rect)
        if not wr.isNull() and wr.contains(QtCore.QPointF(pos)):
            return "move", None
        return None, None

    def _clamp_roi(self, rect: QtCore.QRect) -> QtCore.QRect:
        if self._image_shape is None:
            return rect
        h, w = self._image_shape
        width = int(np.clip(rect.width(), 2, max(2, w)))
        height = int(np.clip(rect.height(), 2, max(2, h)))
        x = int(np.clip(rect.x(), 0, max(0, w - width)))
        y = int(np.clip(rect.y(), 0, max(0, h - height)))
        return QtCore.QRect(x, y, width, height)

    def _rect_from_edges(self, left: int, top: int, right: int, bottom: int) -> QtCore.QRect:
        x0, x1 = sorted((left, right))
        y0, y1 = sorted((top, bottom))
        return self._clamp_roi(QtCore.QRect(x0, y0, max(2, x1 - x0 + 1), max(2, y1 - y0 + 1)))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#101010"))

        self._display_rect = self._fit_rect()
        if not self._pixmap.isNull() and not self._display_rect.isNull():
            painter.drawPixmap(self._display_rect, self._pixmap, QtCore.QRectF(self._pixmap.rect()))

        for idx, match_rect in enumerate(self._match_rects):
            wr = self._image_rect_to_widget(match_rect)
            if not wr.isNull():
                pen = QtGui.QPen(QtGui.QColor("#00d7ff"), 2)
                painter.setPen(pen)
                painter.drawRect(wr)
                painter.setPen(QtGui.QColor("#e8f7ff"))
                painter.drawText(wr.topLeft() + QtCore.QPointF(4, 14), str(idx + 1))

        if self._roi_rect is not None:
            wr = self._image_rect_to_widget(self._roi_rect)
            if not wr.isNull():
                pen = QtGui.QPen(QtGui.QColor("#00ff66"), 2, QtCore.Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(wr)
                painter.setBrush(QtGui.QColor("#00ff66"))
                painter.setPen(QtGui.QPen(QtGui.QColor("#0a0a0a"), 1))
                for handle_rect in self._handle_rects().values():
                    painter.drawRect(handle_rect)

        if self._dragging and self._drag_start and self._drag_end:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_end
            rx = min(x0, x1)
            ry = min(y0, y1)
            rw = abs(x1 - x0) + 1
            rh = abs(y1 - y0) + 1
            wr = self._image_rect_to_widget(QtCore.QRect(rx, ry, rw, rh))
            pen = QtGui.QPen(QtGui.QColor("#ffd400"), 2, QtCore.Qt.DotLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(wr)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        mode, handle = self._hit_test_roi(event.pos())
        point = self._widget_to_image_clamped(event.pos())
        if point is None:
            return
        self._dragging = True
        self._drag_start = point
        self._drag_end = point
        self._interaction_mode = mode or "draw"
        self._active_handle = handle
        self._roi_start_rect = QtCore.QRect(self._roi_rect) if self._roi_rect is not None else None
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._dragging:
            return
        point = self._widget_to_image_clamped(event.pos())
        if point is None:
            return
        self._drag_end = point
        if self._interaction_mode == "draw":
            x0, y0 = self._drag_start or point
            x1, y1 = point
            self.set_roi((min(x0, x1), min(y0, y1), abs(x1 - x0) + 1, abs(y1 - y0) + 1))
            return
        if self._roi_start_rect is None:
            return
        if self._interaction_mode == "move":
            dx = point[0] - self._drag_start[0]
            dy = point[1] - self._drag_start[1]
            moved = QtCore.QRect(self._roi_start_rect)
            moved.translate(dx, dy)
            moved = self._clamp_roi(moved)
            self.set_roi((moved.x(), moved.y(), moved.width(), moved.height()))
            return
        if self._interaction_mode == "resize":
            r = self._roi_start_rect
            left = r.left()
            right = r.right()
            top = r.top()
            bottom = r.bottom()
            x, y = point
            if self._active_handle in {"nw", "w", "sw"}:
                left = x
            if self._active_handle in {"ne", "e", "se"}:
                right = x
            if self._active_handle in {"nw", "n", "ne"}:
                top = y
            if self._active_handle in {"sw", "s", "se"}:
                bottom = y
            rect = self._rect_from_edges(left, top, right, bottom)
            self.set_roi((rect.x(), rect.y(), rect.width(), rect.height()))
            return
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton or not self._dragging:
            return
        self._dragging = False
        point = self._widget_to_image_clamped(event.pos())
        if point is not None:
            self._drag_end = point
        if self._interaction_mode == "draw" and self._drag_start and self._drag_end:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_end
            rect = (
                min(x0, x1),
                min(y0, y1),
                abs(x1 - x0) + 1,
                abs(y1 - y0) + 1,
            )
            self.set_roi(rect)
        self._drag_start = None
        self._drag_end = None
        self._interaction_mode = None
        self._active_handle = None
        self._roi_start_rect = None
        self.update()


class ReferenceObjectCropperWindow(QtWidgets.QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle(PLUGIN_NAME)
        self.resize(1100, 700)
        self.setMinimumSize(860, 560)

        self.current_frame_index = int(getattr(gv, "index", 0)) if gv is not None else 0
        self.current_channel = self._get_active_channel()
        self.current_image: np.ndarray | None = None

        self.reference_patch: np.ndarray | None = None
        self.reference_mask: np.ndarray | None = None
        self.reference_band_mask: np.ndarray | None = None
        self.reference_roi: tuple[int, int, int, int] | None = None
        self.reference_frame_index: int | None = None
        self.reference_bright_object: bool = True
        self.reference_mask_center: tuple[float, float] = (0.0, 0.0)
        self.reference_detection_center: tuple[float, float] = (0.0, 0.0)
        self.reference_mask_area: int = 0
        self.reference_bbox_aspect: float = 1.0
        self.reference_bbox_fill: float = 0.0
        self.reference_profile: np.ndarray | None = None
        self.reference_core_mask: np.ndarray | None = None
        self.reference_halo_mask: np.ndarray | None = None
        self.reference_center_signature: float = 0.0
        self.reference_inside_mean: float = 0.0
        self.reference_outside_mean: float = 0.0
        self.reference_intensity_median: float | None = None
        self.reference_intensity_scale: float = 1.0
        self.records: list[CropRecord] = []
        self._opencv_install_notice_shown = False

        self._build_ui()
        self._connect_signals()
        self._update_filter_ui_state()
        self.refresh_from_main_window()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        usage = QtWidgets.QLabel(
            "1) ROIで対象を囲んで Set Reference  2) 本体でフレームを手動変更  "
            "3) Crop Current  4) 画像を右クリック→手動 Crop  "
            "5) Save で各クロップを別ASD保存（ファイル名にフレーム番号_f・クロップ番号_c）"
        )
        usage.setStyleSheet("color: #dddddd;")
        usage.setWordWrap(True)
        root.addWidget(usage)

        controls = QtWidgets.QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)

        def field_row(label_text: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            label = QtWidgets.QLabel(label_text)
            layout.addWidget(label)
            layout.addWidget(widget)
            return row

        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        self.frame_info_label = QtWidgets.QLabel("Frame: - | Channel: -")
        self.frame_info_label.setMinimumWidth(160)
        top_row.addWidget(self.frame_info_label)
        top_row.addSpacing(6)

        self.reference_button = QtWidgets.QPushButton("Set Reference")
        self.crop_button = QtWidgets.QPushButton("Crop Current")
        self.remove_button = QtWidgets.QPushButton("Remove Selected")
        self.clear_button = QtWidgets.QPushButton("Clear All")
        self.save_button = QtWidgets.QPushButton("Save")
        for btn in (
            self.reference_button,
            self.crop_button,
            self.remove_button,
            self.clear_button,
            self.save_button,
        ):
            btn.setMinimumHeight(34)
            top_row.addWidget(btn)
        top_row.addStretch(1)
        controls.addLayout(top_row)

        mid_row = QtWidgets.QHBoxLayout()
        mid_row.setContentsMargins(0, 0, 0, 0)
        mid_row.setSpacing(8)

        self.score_spin = QtWidgets.QDoubleSpinBox()
        self.score_spin.setRange(0.0, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setDecimals(2)
        self.score_spin.setValue(0.45)
        self.score_spin.setFixedWidth(90)
        mid_row.addWidget(field_row("Min score", self.score_spin))

        self.final_filter_combo = QtWidgets.QComboBox()
        self.final_filter_combo.addItems(["Off", "Median Match"])
        self.final_filter_combo.setFixedWidth(150)
        self.final_filter_combo.setToolTip(
            "Off: shape/appearance criteria only\n"
            "Median Match: additionally keep only objects whose ROI median intensity "
            "matches the reference ROI median within the specified tolerance."
        )
        mid_row.addWidget(field_row("Final filter", self.final_filter_combo))

        self.median_tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.median_tolerance_spin.setRange(0.1, 100.0)
        self.median_tolerance_spin.setSingleStep(1.0)
        self.median_tolerance_spin.setDecimals(1)
        self.median_tolerance_spin.setValue(10.0)
        self.median_tolerance_spin.setFixedWidth(90)
        self.median_tolerance_spin.setToolTip(
            "Allowed deviation from the reference ROI median intensity in percent."
        )
        mid_row.addWidget(field_row("Median tol (%)", self.median_tolerance_spin))
        mid_row.addStretch(1)
        controls.addLayout(mid_row)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(8)

        self.roi_width_nm_spin = QtWidgets.QDoubleSpinBox()
        self.roi_width_nm_spin.setRange(0.1, 1_000_000.0)
        self.roi_width_nm_spin.setDecimals(2)
        self.roi_width_nm_spin.setSingleStep(1.0)
        self.roi_width_nm_spin.setValue(50.0)
        self.roi_width_nm_spin.setFixedWidth(100)
        self.roi_width_nm_spin.setToolTip("ROI width in nm")
        bottom_row.addWidget(field_row("ROI W (nm)", self.roi_width_nm_spin))

        self.roi_height_nm_spin = QtWidgets.QDoubleSpinBox()
        self.roi_height_nm_spin.setRange(0.1, 1_000_000.0)
        self.roi_height_nm_spin.setDecimals(2)
        self.roi_height_nm_spin.setSingleStep(1.0)
        self.roi_height_nm_spin.setValue(50.0)
        self.roi_height_nm_spin.setFixedWidth(100)
        self.roi_height_nm_spin.setToolTip("ROI height in nm")
        bottom_row.addWidget(field_row("ROI H (nm)", self.roi_height_nm_spin))

        self.apply_roi_nm_button = QtWidgets.QPushButton("Apply ROI nm")
        self.apply_roi_nm_button.setToolTip("Resize current ROI to the specified nm size while keeping its center.")
        bottom_row.addWidget(self.apply_roi_nm_button)
        bottom_row.addStretch(1)
        controls.addLayout(bottom_row)

        root.addLayout(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)
        splitter.setChildrenCollapsible(False)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        self.image_viewer = RoiImageViewer()
        left_layout.addWidget(self.image_viewer, 1)

        self.roi_info_label = QtWidgets.QLabel("ROI: not set")
        self.roi_info_label.setStyleSheet("color: #bbbbbb;")
        left_layout.addWidget(self.roi_info_label)
        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        ref_group = QtWidgets.QGroupBox("Reference")
        ref_layout = QtWidgets.QVBoxLayout(ref_group)
        self.reference_preview = QtWidgets.QLabel("Reference preview")
        self.reference_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.reference_preview.setMinimumSize(180, 180)
        self.reference_preview.setStyleSheet("background: #141414; border: 1px solid #444; color: #888;")
        ref_layout.addWidget(self.reference_preview)
        self.reference_meta_label = QtWidgets.QLabel("Reference: not recorded")
        self.reference_meta_label.setWordWrap(True)
        ref_layout.addWidget(self.reference_meta_label)
        right_layout.addWidget(ref_group)

        crops_group = QtWidgets.QGroupBox("Extracted Crops")
        crops_layout = QtWidgets.QVBoxLayout(crops_group)
        self.crops_hint_label = QtWidgets.QLabel(
            "明るい緑の枠 = 現在のフレームの Crop / Ctrl・Shift で複数選択 / 黄枠 = 選択中 / "
            "サムネを右クリック→「この位置をROI中心に」で位置調整"
        )
        self.crops_hint_label.setStyleSheet("color: #6ee7b7; font-size: 11px;")
        self.crops_hint_label.setWordWrap(True)
        crops_layout.addWidget(self.crops_hint_label)
        self.thumbnail_list = QtWidgets.QListWidget()
        self.thumbnail_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.thumbnail_list.setItemDelegate(ExtractedCropThumbnailDelegate(self.thumbnail_list))
        self.thumbnail_list.setViewMode(QtWidgets.QListView.IconMode)
        self.thumbnail_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.thumbnail_list.setMovement(QtWidgets.QListView.Static)
        self.thumbnail_list.setWrapping(True)
        self.thumbnail_list.setSpacing(12)
        self.thumbnail_list.setIconSize(QtCore.QSize(128, 128))
        self.thumbnail_list.setMinimumWidth(240)
        self.thumbnail_list.setStyleSheet(
            "QListWidget { background: #0a0a0a; border: none; }"
            "QListWidget::item { padding: 4px; }"
            "QListWidget::item:selected { background: transparent; }"
        )
        self.thumbnail_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        crops_layout.addWidget(self.thumbnail_list, 1)
        right_layout.addWidget(crops_group, 1)
        splitter.addWidget(right_panel)

        splitter.setSizes([720, 320])

        self.status_label = QtWidgets.QLabel("Reference ROIを選択してください。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #dddddd;")
        root.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        self.reference_button.clicked.connect(self.set_reference_from_roi)
        self.crop_button.clicked.connect(self.crop_current_frame)
        self.remove_button.clicked.connect(self.remove_selected_record)
        self.clear_button.clicked.connect(self.clear_all_records)
        self.save_button.clicked.connect(self.save_records)
        self.apply_roi_nm_button.clicked.connect(self.apply_roi_size_from_nm)
        self.image_viewer.roi_changed.connect(self._update_roi_label)
        self.image_viewer.manual_crop_requested.connect(self._add_manual_crop_at_image_position)
        self.final_filter_combo.currentIndexChanged.connect(self._update_filter_ui_state)
        self.thumbnail_list.customContextMenuRequested.connect(self._on_thumbnail_context_menu)
        if hasattr(self.main_window, "frameChanged"):
            self.main_window.frameChanged.connect(self._on_main_frame_changed)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            if hasattr(self.main_window, "frameChanged"):
                self.main_window.frameChanged.disconnect(self._on_main_frame_changed)
        except Exception:
            pass
        super().closeEvent(event)

    def _notify_opencv_missing_for_search(self) -> None:
        """OpenCV is required for template-matching (NCC) object search; prompt install once per session."""
        if cv2 is not None or self._opencv_install_notice_shown:
            return
        self._opencv_install_notice_shown = True
        QtWidgets.QMessageBox.information(
            self,
            "OpenCV が見つかりません",
            "Object 検出の高精度モード（テンプレートマッチング）には OpenCV（opencv-python）が必要です。\n\n"
            "次のコマンドで pyNuD の環境にインストールしてください:\n"
            "  ./.venv/bin/pip install opencv-python\n\n"
            "OpenCV が無い場合は従来の候補検出のみが使われます。",
        )

    def _set_status(self, text: str, error: bool = False) -> None:
        color = "#ff7777" if error else "#dddddd"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)

    def _update_filter_ui_state(self) -> None:
        median_enabled = self.final_filter_combo.currentText() == "Median Match"
        self.median_tolerance_spin.setEnabled(median_enabled)

    def _median_match_enabled(self) -> bool:
        return self.final_filter_combo.currentText() == "Median Match"

    def _object_center_from_patch(
        self,
        patch: np.ndarray,
        filled_mask: np.ndarray,
        band_mask: np.ndarray | None = None,
        bright_object: bool | None = None,
    ) -> tuple[float, float]:
        bright = self.reference_bright_object if bright_object is None else bool(bright_object)
        return _weighted_intensity_centroid(patch, filled_mask, bright)

    def _detection_center_from_patch(
        self,
        patch: np.ndarray,
        filled_mask: np.ndarray,
        band_mask: np.ndarray | None = None,
        bright_object: bool | None = None,
    ) -> tuple[float, float]:
        bright = self.reference_bright_object if bright_object is None else bool(bright_object)
        return _weighted_intensity_centroid(patch, filled_mask, bright)

    def _recompute_reference_center_metrics(self) -> None:
        if self.reference_patch is None or self.reference_mask is None:
            return
        if self.reference_band_mask is None:
            _filled, band_mask, _bright = _build_object_masks(
                self.reference_patch,
                bright_object=self.reference_bright_object,
            )
            self.reference_band_mask = band_mask
        self.reference_detection_center = self._detection_center_from_patch(
            self.reference_patch,
            self.reference_mask,
            self.reference_band_mask,
            self.reference_bright_object,
        )
        self.reference_mask_center = self._object_center_from_patch(
            self.reference_patch,
            self.reference_mask,
            self.reference_band_mask,
            self.reference_bright_object,
        )
        self.reference_profile = _radial_profile(self.reference_patch, self.reference_detection_center)
        self.reference_core_mask, self.reference_halo_mask = _center_signature_masks(
            self.reference_mask,
            self.reference_detection_center,
        )
        self.reference_center_signature = self._center_signature_value(
            self.reference_patch,
            self.reference_core_mask,
            self.reference_halo_mask,
        )
    def _center_signature_value(
        self,
        patch: np.ndarray,
        core_mask: np.ndarray | None,
        halo_mask: np.ndarray | None,
    ) -> float:
        if core_mask is None or halo_mask is None:
            return 0.0
        core_mask = np.asarray(core_mask, dtype=bool)
        halo_mask = np.asarray(halo_mask, dtype=bool)
        if np.count_nonzero(core_mask) < 4 or np.count_nonzero(halo_mask) < 4:
            return 0.0
        norm = _normalize_float(patch)
        core_mean = float(np.mean(norm[core_mask]))
        halo_mean = float(np.mean(norm[halo_mask]))
        return core_mean - halo_mean

    def _rebuild_reference_object_metrics(self) -> None:
        if self.reference_patch is None:
            return
        mask, band_mask, bright_object = _build_object_masks(
            self.reference_patch,
            bright_object=self.reference_bright_object,
        )
        self.reference_mask = mask
        self.reference_band_mask = band_mask
        self.reference_bright_object = bright_object
        self.reference_mask_area = int(np.count_nonzero(mask))
        ref_bbox_x, ref_bbox_y, ref_bbox_w, ref_bbox_h = _mask_bbox(mask)
        self.reference_bbox_aspect = float(ref_bbox_w / max(ref_bbox_h, 1))
        self.reference_bbox_fill = float(self.reference_mask_area / max(ref_bbox_w * ref_bbox_h, 1))
        norm_patch = _normalize_float(self.reference_patch)
        self.reference_inside_mean = float(np.mean(norm_patch[mask])) if np.any(mask) else 0.0
        outside = norm_patch[~mask]
        self.reference_outside_mean = float(np.mean(outside)) if outside.size else 0.0
        self._recompute_reference_center_metrics()

    def _compute_reference_median_stats(self, patch: np.ndarray) -> tuple[float | None, float]:
        values = np.asarray(patch, dtype=np.float32).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None, 1.0
        median = float(np.median(values))
        q05, q95 = np.percentile(values, [5.0, 95.0])
        robust_span = float(max(abs(q95 - q05), 1e-9))
        scale = max(abs(median), robust_span * 0.1, 1e-9)
        return median, scale

    def _passes_median_match(self, candidate_patch: np.ndarray) -> tuple[bool, float | None, float | None]:
        if not self._median_match_enabled():
            return True, None, None
        if self.reference_intensity_median is None:
            return False, None, None
        values = np.asarray(candidate_patch, dtype=np.float32).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return False, None, None
        cand_median = float(np.median(values))
        tol_pct = float(self.median_tolerance_spin.value())
        tol_abs = abs(self.reference_intensity_scale) * (tol_pct / 100.0)
        diff_abs = abs(cand_median - self.reference_intensity_median)
        return diff_abs <= tol_abs, cand_median, tol_abs

    def _get_active_channel(self) -> str:
        if gv is None:
            return "1ch"
        return "2ch" if getattr(gv, "active_display_channel", "1ch") == "2ch" else "1ch"

    def _nm_per_pixel(self) -> tuple[float, float] | None:
        if gv is None:
            return None
        x_px = float(getattr(gv, "XPixel", 0) or 0)
        y_px = float(getattr(gv, "YPixel", 0) or 0)
        x_nm = float(getattr(gv, "XScanSize", 0) or 0)
        y_nm = float(getattr(gv, "YScanSize", 0) or 0)
        if x_px <= 0 or y_px <= 0 or x_nm <= 0 or y_nm <= 0:
            return None
        return x_nm / x_px, y_nm / y_px

    def _crop_record_metadata(self):
        path = os.path.realpath(gv.files[gv.currentFileNum])
        stat = os.stat(path)
        second = self.current_channel == '2ch'
        kind = (20564 if getattr(gv, 'SourceFileFormat', '') == 'Bruker SPM'
                and getattr(gv, 'DataType2ch', 0) == 20564 else 18512) if second else gv.DataType1ch
        return dict(source_path=path, source_fingerprint=(stat.st_ino, stat.st_size, stat.st_mtime_ns),
                    pixel_size_nm=self._nm_per_pixel(), signal_kind=kind,
                    conversion_type=getattr(gv, 'ch2_data_type', 'Other'),
                    conversion_factor=float(getattr(gv, 'ch2_conversion_factor', 1.)))

    def _roi_size_nm_from_px(self, width_px: int, height_px: int) -> tuple[float, float] | None:
        nmpp = self._nm_per_pixel()
        if nmpp is None:
            return None
        return float(width_px) * nmpp[0], float(height_px) * nmpp[1]

    def _roi_size_px_from_nm(self, width_nm: float, height_nm: float) -> tuple[int, int] | None:
        nmpp = self._nm_per_pixel()
        if nmpp is None:
            return None
        width_px = max(2, int(round(float(width_nm) / max(nmpp[0], 1e-12))))
        height_px = max(2, int(round(float(height_nm) / max(nmpp[1], 1e-12))))
        return width_px, height_px

    @staticmethod
    def _vertical_orientation_for_plugin_display(image: np.ndarray | None) -> np.ndarray | None:
        """本体のメイン画像（dspimg は cv2.flip(...,0)）と同じ向きに揃えるため aryData を縦反転する。"""
        if image is None:
            return None
        arr = np.array(image, dtype=np.float32, copy=True)
        if arr.ndim != 2:
            return arr
        return np.ascontiguousarray(np.flipud(arr))

    def _get_current_image_from_globals(self) -> np.ndarray | None:
        if gv is None:
            return None
        channel = self._get_active_channel()
        if channel == "2ch":
            image = getattr(gv, "aryData_processed_2ch", None)
            if image is None:
                image = getattr(gv, "aryData2ch", None)
            raw = None if image is None else np.array(image, dtype=np.float32, copy=True)
            return self._vertical_orientation_for_plugin_display(raw)

        image = getattr(gv, "aryData_processed_1ch", None)
        if image is None:
            image = getattr(gv, "aryData", None)
        raw = None if image is None else np.array(image, dtype=np.float32, copy=True)
        return self._vertical_orientation_for_plugin_display(raw)

    def _get_image_for_frame_channel(self, frame_index: int, channel: str) -> np.ndarray | None:
        """指定フレーム・チャンネルの画像（MainWindow.getImageDataForFrame があれば使用）。"""
        mw = self.main_window
        if mw is not None and hasattr(mw, "getImageDataForFrame"):
            try:
                img = mw.getImageDataForFrame(int(frame_index), str(channel))
            except Exception:
                img = None
            if img is not None:
                return self._vertical_orientation_for_plugin_display(np.array(img, dtype=np.float32, copy=True))
        if int(frame_index) == int(self.current_frame_index) and str(channel) == str(self.current_channel):
            return self.current_image
        return None

    def _find_record(self, frame_index: int, object_id: int, channel: str) -> CropRecord | None:
        for rec in self.records:
            if (
                int(rec.frame_index) == int(frame_index)
                and int(rec.object_id) == int(object_id)
                and str(rec.channel) == str(channel)
            ):
                return rec
        return None

    def _thumbnail_icon_rect_viewport(self, item: QtWidgets.QListWidgetItem) -> QtCore.QRect:
        """Viewport 座標系でサムネイル画像が描画されている矩形。

        QStyle.subElementRect はカスタムデリゲートと一致しないことがあり、
        クリックが常に crop 中央にマッピングされる不具合の原因になる。
        _refresh_thumbnail_list と同じ QIcon.pixmap(iconSize()) の寸法でレイアウトする。
        """
        idx = self.thumbnail_list.indexFromItem(item)
        vr = self.thumbnail_list.visualRect(idx)
        pm = item.icon().pixmap(self.thumbnail_list.iconSize())
        if pm.isNull():
            return QtCore.QRect()
        pw, ph = pm.width(), pm.height()
        x = vr.x() + max(0, (vr.width() - pw) // 2)
        y = vr.y()
        return QtCore.QRect(x, y, pw, ph)

    def _viewport_pos_to_crop_column_row(
        self,
        item: QtWidgets.QListWidgetItem,
        pos_viewport: QtCore.QPoint,
    ) -> tuple[int, int] | None:
        """サムネイル表示上の座標を crop 配列の列・行インデックスに変換。範囲外は None。"""
        data = item.data(QtCore.Qt.UserRole)
        if not data:
            return None
        fi, oid, ch = data
        rec = self._find_record(int(fi), int(oid), str(ch))
        if rec is None:
            return None
        crop_h, crop_w = int(rec.crop.shape[0]), int(rec.crop.shape[1])
        idx = self.thumbnail_list.indexFromItem(item)
        vr = self.thumbnail_list.visualRect(idx)
        if not vr.contains(pos_viewport):
            return None
        icon_rect = self._thumbnail_icon_rect_viewport(item)
        if icon_rect.width() <= 0 or icon_rect.height() <= 0:
            return None
        # ラベル文字列（F# / score）の領域は除外（アイコンより下）
        # ラベル行（アイコン直下〜）を除外（アイコン最下行は y == y0 + height - 1 まで許可）
        if pos_viewport.y() >= icon_rect.y() + icon_rect.height():
            return None
        lx = float(pos_viewport.x() - icon_rect.x())
        ly = float(pos_viewport.y() - icon_rect.y())
        lx = float(np.clip(lx, 0.0, max(icon_rect.width() - 1e-6, 0.0)))
        ly = float(np.clip(ly, 0.0, max(icon_rect.height() - 1e-6, 0.0)))
        ix = int(np.clip(np.round(lx / icon_rect.width() * max(crop_w - 1, 0)), 0, max(crop_w - 1, 0)))
        iy = int(np.clip(np.round(ly / icon_rect.height() * max(crop_h - 1, 0)), 0, max(crop_h - 1, 0)))
        return ix, iy

    def _on_thumbnail_context_menu(self, pos_viewport: QtCore.QPoint) -> None:
        item = self.thumbnail_list.itemAt(pos_viewport)
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu {"
            " background-color: #161616;"
            " color: #f3f4f6;"
            " border: 1px solid #4b5563;"
            " }"
            "QMenu::item {"
            " background-color: transparent;"
            " color: #f3f4f6;"
            " padding: 6px 24px 6px 12px;"
            " }"
            "QMenu::item:selected {"
            " background-color: #2563eb;"
            " color: #ffffff;"
            " }"
        )
        act = menu.addAction("この位置をROI中心に")
        chosen = menu.exec_(self.thumbnail_list.viewport().mapToGlobal(pos_viewport))
        if chosen != act:
            return
        self._recenter_crop_from_thumbnail_click(item, pos_viewport)

    def _recenter_crop_from_thumbnail_click(
        self,
        item: QtWidgets.QListWidgetItem,
        pos_viewport: QtCore.QPoint,
    ) -> None:
        if self.reference_patch is None:
            self._set_status("先に Set Reference で基準ROIを登録してください。", error=True)
            return
        col_row = self._viewport_pos_to_crop_column_row(item, pos_viewport)
        if col_row is None:
            self._set_status("サムネイルの画像部分を右クリックしてください。", error=True)
            return
        ix, iy = col_row
        data = item.data(QtCore.Qt.UserRole)
        if not data:
            return
        fi, oid, ch = int(data[0]), int(data[1]), str(data[2])
        rec = self._find_record(fi, oid, ch)
        if rec is None:
            self._set_status("対象の Crop レコードが見つかりません。", error=True)
            return
        image = self._get_image_for_frame_channel(fi, ch)
        if image is None:
            self._set_status(
                "そのフレームの画像を取得できません（本体で該当フレームを表示してから試すか、"
                "MainWindow.getImageDataForFrame が利用可能か確認してください）。",
                error=True,
            )
            return
        rw, rh = int(rec.roi_width), int(rec.roi_height)
        H, W = int(image.shape[0]), int(image.shape[1])
        if W < rw or H < rh:
            self._set_status("画像サイズが ROI より小さいため再配置できません。", error=True)
            return
        old_nx, old_ny = int(rec.roi_x), int(rec.roi_y)
        # クリックした画素の中心を ROI 中心に合わせる（画素は左上座標 ix,iy なので +0.5）
        fx = float(rec.roi_x + ix + 0.5)
        fy = float(rec.roi_y + iy + 0.5)
        nx = int(np.round(fx - rw / 2.0))
        ny = int(np.round(fy - rh / 2.0))
        nx = int(np.clip(nx, 0, W - rw))
        ny = int(np.clip(ny, 0, H - rh))
        if nx == old_nx and ny == old_ny:
            self._set_status(
                f"F{fi} #{oid}: ROI 左上はすでに ({nx},{ny}) です。"
                " サムネの端ではなく対象のずれた位置をクリックしてください。"
            )
            return
        patch = self._extract_roi_patch(image, (nx, ny, rw, rh))
        if patch is None:
            self._set_status("パッチを切り出せませんでした。", error=True)
            return
        if patch.shape[0] != rh or patch.shape[1] != rw:
            self._set_status("ROI 境界付近で切り出しサイズが一致しませんでした。", error=True)
            return
        rec.roi_x = nx
        rec.roi_y = ny
        rec.crop = patch.copy()
        self._refresh_thumbnail_list()
        self._sync_match_rects_overlay()
        overlay_note = ""
        if int(fi) != int(self.current_frame_index) or str(ch) != str(self.current_channel):
            overlay_note = (
                " 左の緑枠は「現在表示中のフレーム・チャンネル」の検出のみ表示します。"
                " 本体で該当フレームに切り替えると枠が追従します。"
            )
        self._set_status(
            f"F{fi} #{oid}: ROI 左上 ({old_nx},{old_ny}) → ({nx},{ny})、"
            f"クリック対応 crop 内 (x={ix}, y={iy})。{overlay_note}"
        )

    def _default_save_path(self) -> str:
        if gv is not None and hasattr(gv, "files") and gv.files:
            current_file = gv.files[getattr(gv, "currentFileNum", 0)]
            base = os.path.splitext(os.path.basename(current_file))[0]
            return os.path.join(os.path.dirname(current_file), f"{base}_reference_crops.asd")
        return os.path.join(os.getcwd(), "reference_crops.asd")

    def _update_roi_label(self) -> None:
        roi = self.image_viewer.get_roi()
        if roi is None:
            self.roi_info_label.setText("ROI: not set")
            return
        x, y, w, h = roi
        size_nm = self._roi_size_nm_from_px(w, h)
        if size_nm is not None:
            self.roi_width_nm_spin.blockSignals(True)
            self.roi_height_nm_spin.blockSignals(True)
            self.roi_width_nm_spin.setValue(size_nm[0])
            self.roi_height_nm_spin.setValue(size_nm[1])
            self.roi_width_nm_spin.blockSignals(False)
            self.roi_height_nm_spin.blockSignals(False)
            self.roi_info_label.setText(
                f"ROI: x={x}, y={y}, w={w}, h={h} | {size_nm[0]:.2f} nm x {size_nm[1]:.2f} nm"
            )
        else:
            self.roi_info_label.setText(f"ROI: x={x}, y={y}, w={w}, h={h}")

    def apply_roi_size_from_nm(self) -> None:
        roi = self.image_viewer.get_roi()
        if roi is None:
            self._set_status("先にROIを設定してから nm サイズを適用してください。", error=True)
            return
        dims = self._roi_size_px_from_nm(self.roi_width_nm_spin.value(), self.roi_height_nm_spin.value())
        if dims is None:
            self._set_status("nm と pixel の変換情報がありません。X/Y scan size を確認してください。", error=True)
            return
        x, y, w, h = roi
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_w, new_h = dims
        new_x = int(round(cx - new_w / 2.0))
        new_y = int(round(cy - new_h / 2.0))
        image = self.current_image
        if image is None:
            self._set_status("現在画像が無いため ROI サイズを更新できません。", error=True)
            return
        max_w = int(image.shape[1])
        max_h = int(image.shape[0])
        new_w = min(new_w, max_w)
        new_h = min(new_h, max_h)
        new_x = int(np.clip(new_x, 0, max(0, max_w - new_w)))
        new_y = int(np.clip(new_y, 0, max(0, max_h - new_h)))
        self.image_viewer.set_roi((new_x, new_y, new_w, new_h))
        self._set_status(
            f"ROI サイズを {self.roi_width_nm_spin.value():.2f} nm x {self.roi_height_nm_spin.value():.2f} nm に更新しました。"
        )

    def _refresh_reference_preview(self) -> None:
        if self.reference_patch is None or self.reference_mask is None:
            self.reference_preview.setPixmap(QtGui.QPixmap())
            self.reference_preview.setText("Reference preview")
            self.reference_meta_label.setText("Reference: not recorded")
            return

        pixmap = _array_to_pixmap(self.reference_patch)
        scaled = pixmap.scaled(
            self.reference_preview.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.reference_preview.setPixmap(scaled)
        self.reference_preview.setText("")
        x, y, w, h = self.reference_roi or (0, 0, 0, 0)
        median_text = ""
        if self.reference_intensity_median is not None:
            median_text = f" | median={self.reference_intensity_median:.4g}"
        self.reference_meta_label.setText(
            f"Frame {self.reference_frame_index} | ROI x={x}, y={y}, w={w}, h={h} | "
            f"Mask pixels={int(np.count_nonzero(self.reference_mask))} | center=I²-weighted{median_text}"
        )

    def _sync_match_rects_overlay(self) -> None:
        """Redraw detection boxes from saved records for the current frame and channel."""
        rects = [
            (int(rec.roi_x), int(rec.roi_y), int(rec.roi_width), int(rec.roi_height))
            for rec in sorted(
                (
                    r
                    for r in self.records
                    if r.frame_index == self.current_frame_index and r.channel == self.current_channel
                ),
                key=lambda r: (r.object_id, r.roi_y, r.roi_x),
            )
        ]
        self.image_viewer.set_match_rects(rects if rects else None)

    def _apply_thumbnail_item_frame_style(
        self,
        item: QtWidgets.QListWidgetItem,
        frame_index: int,
        channel: str,
    ) -> None:
        is_current = int(frame_index) == int(self.current_frame_index) and str(channel) == str(
            self.current_channel
        )
        item.setData(THUMB_USERROLE_IS_CURRENT, is_current)
        if is_current:
            item.setToolTip("現在メイン表示のフレーム・チャンネルで Crop した結果です。")
        else:
            item.setToolTip(f"Frame {int(frame_index)}（{channel}）の Crop です。")

    def _update_thumbnail_frame_highlight(self) -> None:
        """Update only background/text for current frame without rebuilding icons."""
        for i in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(i)
            data = item.data(QtCore.Qt.UserRole)
            if not data:
                continue
            fi, _oid, ch = data
            self._apply_thumbnail_item_frame_style(item, int(fi), str(ch))
        self.thumbnail_list.viewport().update()

    def _refresh_thumbnail_list(self) -> None:
        self.thumbnail_list.clear()
        for record in sorted(self.records, key=lambda rec: (rec.frame_index, rec.object_id)):
            pixmap = _array_to_pixmap(record.crop)
            icon = QtGui.QIcon(
                pixmap.scaled(128, 128, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )
            score_line = (
                "manual"
                if record.match_score == MANUAL_CROP_SCORE
                else f"score {record.match_score:.3f}"
            )
            item = QtWidgets.QListWidgetItem(
                icon,
                f"F{record.frame_index} #{record.object_id}\n{score_line}",
            )
            item.setData(QtCore.Qt.UserRole, (record.frame_index, record.object_id, record.channel))
            self._apply_thumbnail_item_frame_style(item, record.frame_index, record.channel)
            self.thumbnail_list.addItem(item)

    def refresh_from_main_window(self, frame_index_override: int | None = None) -> None:
        if gv is not None:
            if frame_index_override is not None:
                self.current_frame_index = int(frame_index_override)
            else:
                self.current_frame_index = int(getattr(gv, "index", 0))
        else:
            self.current_frame_index = int(frame_index_override or 0)
        self.current_channel = self._get_active_channel()
        self.current_image = self._get_current_image_from_globals()
        self.frame_info_label.setText(f"Frame: {self.current_frame_index} | Channel: {self.current_channel}")
        self.image_viewer.set_image(self.current_image)
        self._update_roi_label()
        self._sync_match_rects_overlay()
        self._update_thumbnail_frame_highlight()

    def _on_main_frame_changed(self, frame_index: int) -> None:
        self.refresh_from_main_window(frame_index_override=int(frame_index))

    def _extract_roi_patch(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray | None:
        if image is None:
            return None
        x, y, w, h = [int(v) for v in roi]
        if w <= 1 or h <= 1:
            return None
        y0 = max(0, y)
        x0 = max(0, x)
        y1 = min(image.shape[0], y0 + h)
        x1 = min(image.shape[1], x0 + w)
        if y1 - y0 < 2 or x1 - x0 < 2:
            return None
        return np.array(image[y0:y1, x0:x1], dtype=np.float32, copy=True)

    def set_reference_from_roi(self) -> None:
        if self.current_image is None:
            self._set_status("現在フレームの画像を取得できません。", error=True)
            return
        roi = self.image_viewer.get_roi()
        if roi is None:
            self._set_status(
                "先にROIをドラッグで設定してください。右クリックメニューから Reference ROI をクリアできます。",
                error=True,
            )
            return

        patch = self._extract_roi_patch(self.current_image, roi)
        if patch is None or patch.size == 0:
            self._set_status("ROIが小さすぎるか画像範囲外です。", error=True)
            return

        self.reference_patch = patch
        self.reference_roi = roi
        self.reference_frame_index = self.current_frame_index
        _mask, _band_mask, bright_object = _build_object_masks(patch, bright_object=None)
        self.reference_bright_object = bright_object
        self._rebuild_reference_object_metrics()
        self.reference_intensity_median, self.reference_intensity_scale = self._compute_reference_median_stats(patch)
        self.image_viewer.set_match_rects(None)
        self._refresh_reference_preview()
        median_note = ""
        if self.reference_intensity_median is not None:
            median_note = f" median={self.reference_intensity_median:.4g}"
        center_note = "強度²重み重心（メイン斑点寄り）"
        self._set_status(
            f"Referenceを記録しました。Frame {self.reference_frame_index} のROIを基準に、"
            f"以後の Crop Current で同形状を抽出します。 center={center_note}.{median_note}"
        )

    def _find_peak_locations(
        self,
        match_map: np.ndarray,
        min_score: float,
        tpl_w: int,
        tpl_h: int,
    ) -> list[tuple[int, int, float]]:
        if match_map is None or match_map.size == 0:
            return []

        score_mask = match_map >= min_score
        if ndimage is not None:
            window = int(max(3, round(min(tpl_w, tpl_h) * 0.45)))
            local_max = ndimage.maximum_filter(match_map, size=window, mode="nearest")
            score_mask &= match_map >= (local_max - 1e-6)

        candidates = np.argwhere(score_mask)
        if candidates.size == 0:
            return []

        sorted_candidates = sorted(
            ((int(y), int(x), float(match_map[y, x])) for y, x in candidates),
            key=lambda item: item[2],
            reverse=True,
        )

        min_dist = float(max(4, min(tpl_w, tpl_h) * 0.60))
        accepted: list[tuple[int, int, float]] = []
        for y, x, score in sorted_candidates:
            cx = x + tpl_w / 2.0
            cy = y + tpl_h / 2.0
            keep = True
            for ay, ax, _ in accepted:
                acx = ax + tpl_w / 2.0
                acy = ay + tpl_h / 2.0
                if float(np.hypot(cx - acx, cy - acy)) < min_dist:
                    keep = False
                    break
            if not keep:
                continue
            accepted.append((y, x, score))
            if len(accepted) >= 256:
                break

        accepted.sort(key=lambda item: (item[0], item[1]))
        return accepted

    def _extract_centered_patch(
        self,
        frame: np.ndarray,
        center_y: float,
        center_x: float,
        patch_h: int,
        patch_w: int,
    ) -> tuple[np.ndarray, int, int] | None:
        y = int(round(center_y - self.reference_detection_center[0]))
        x = int(round(center_x - self.reference_detection_center[1]))
        y = int(np.clip(y, 0, max(0, frame.shape[0] - patch_h)))
        x = int(np.clip(x, 0, max(0, frame.shape[1] - patch_w)))
        patch = frame[y:y + patch_h, x:x + patch_w]
        if patch.shape != (patch_h, patch_w):
            return None
        return np.array(patch, dtype=np.float32, copy=True), x, y

    def _shift_roi_to_object_center(
        self,
        frame: np.ndarray,
        x0: int,
        y0: int,
        tpl_h: int,
        tpl_w: int,
        _mask: np.ndarray,
        max_iter: int = 8,
    ) -> tuple[np.ndarray, int, int, np.ndarray]:
        """Move the tpl_w×tpl_h window so its center tracks the main blob (I²-weighted centroid).

        Repeats shift until the integer top-left (x, y) stabilizes or ``max_iter`` is reached,
        so manual crops and automatic matches both settle consistently.
        ``_mask`` is kept for call-site compatibility; center uses intensity-weighted centroid
        from the mask built at the current window position.
        """
        fh, fw = int(frame.shape[0]), int(frame.shape[1])
        x_curr = int(x0)
        y_curr = int(y0)
        for _ in range(max(1, int(max_iter))):
            patch = np.array(
                frame[y_curr : y_curr + tpl_h, x_curr : x_curr + tpl_w],
                dtype=np.float32,
                copy=True,
            )
            m, band_m, _ = _build_object_masks(
                patch,
                bright_object=self.reference_bright_object,
            )
            if not np.any(m):
                return patch, x_curr, y_curr, m
            cy, cx = self._object_center_from_patch(
                patch,
                m,
                band_m,
                self.reference_bright_object,
            )
            x1 = int(round(x_curr + cx - tpl_w / 2.0))
            y1 = int(round(y_curr + cy - tpl_h / 2.0))
            x1 = int(np.clip(x1, 0, max(0, fw - tpl_w)))
            y1 = int(np.clip(y1, 0, max(0, fh - tpl_h)))
            if x1 == x_curr and y1 == y_curr:
                return patch, x_curr, y_curr, m
            x_curr, y_curr = x1, y1
        patch = np.array(
            frame[y_curr : y_curr + tpl_h, x_curr : x_curr + tpl_w],
            dtype=np.float32,
            copy=True,
        )
        m, _band_m, _ = _build_object_masks(
            patch,
            bright_object=self.reference_bright_object,
        )
        return patch, x_curr, y_curr, m

    def _frame_norm_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """Normalize and lightly smooth (recall-friendly; user can delete bad boxes manually)."""
        norm = _normalize_float(frame)
        if ndimage is not None:
            norm = ndimage.gaussian_filter(np.asarray(norm, dtype=np.float32), sigma=1.15)
            norm = ndimage.median_filter(norm, size=3)
        return norm

    def _find_ncc_template_candidates(
        self,
        frame: np.ndarray,
        min_score: float = 0.0,
    ) -> list[dict[str, float]]:
        """Primary search: NCC peaks above the user-specified minimum score."""
        if self.reference_patch is None or cv2 is None:
            return []
        tpl_h, tpl_w = self.reference_patch.shape
        if tpl_h < 3 or tpl_w < 3:
            return []
        if frame.shape[0] < tpl_h or frame.shape[1] < tpl_w:
            return []

        norm_f = self._frame_norm_for_detection(frame)
        norm_t = _normalize_float(self.reference_patch)
        if ndimage is not None:
            norm_t = ndimage.gaussian_filter(np.asarray(norm_t, dtype=np.float32), sigma=0.85)

        img_u8 = np.clip(norm_f * 255.0, 0, 255).astype(np.uint8)
        tpl_u8 = np.clip(norm_t * 255.0, 0, 255).astype(np.uint8)
        res = cv2.matchTemplate(img_u8, tpl_u8, cv2.TM_CCOEFF_NORMED)
        if res is None or res.size == 0:
            return []
        if not np.isfinite(float(np.max(res))):
            return []
        ncc_floor = float(np.clip(min_score, -1.0, 1.0))

        if ndimage is not None:
            window = int(max(5, round(min(tpl_w, tpl_h) * 0.52)))
            local_max = ndimage.maximum_filter(res, size=window, mode="nearest")
            peak_mask = (res >= local_max - 1e-6) & (res >= ncc_floor)
        else:
            peak_mask = res >= ncc_floor

        ys, xs = np.nonzero(peak_mask)
        peaks = sorted(
            ((int(y), int(x), float(res[y, x])) for y, x in zip(ys, xs)),
            key=lambda item: item[2],
            reverse=True,
        )

        min_dist = float(max(6.0, min(tpl_w, tpl_h) * 0.62))
        accepted: list[dict[str, float]] = []
        for y, x, ncc in peaks:
            cx = float(x) + float(tpl_w) * 0.5
            cy = float(y) + float(tpl_h) * 0.5
            keep = True
            for cand in accepted:
                if float(np.hypot(cx - cand["cx"], cy - cand["cy"])) < min_dist:
                    keep = False
                    break
            if not keep:
                continue
            accepted.append(
                {
                    "cy": cy,
                    "cx": cx,
                    "area": float(tpl_w * tpl_h),
                    "fill": 0.0,
                    "aspect": float(tpl_w / max(tpl_h, 1)),
                    "ncc": float(ncc),
                    "src": "ncc",
                }
            )
            if len(accepted) >= 384:
                break
        return accepted

    def _find_component_candidates(self, frame: np.ndarray) -> list[dict[str, float]]:
        if self.reference_patch is None:
            return []

        norm = self._frame_norm_for_detection(frame)
        threshold = _otsu_threshold(norm.ravel())
        binary = norm >= threshold if self.reference_bright_object else norm <= threshold

        if ndimage is not None:
            binary = ndimage.binary_opening(binary, structure=np.ones((3, 3), dtype=bool))
            binary = ndimage.binary_closing(binary, structure=np.ones((3, 3), dtype=bool))
            labeled, count = ndimage.label(binary)
            if count <= 0:
                return []
            objects = ndimage.find_objects(labeled)
            candidates: list[dict[str, float]] = []
            min_area = max(4.0, self.reference_mask_area * 0.26)
            max_area = max(min_area + 1.0, self.reference_mask_area * 2.55)
            for label_id, slc in enumerate(objects, start=1):
                if slc is None:
                    continue
                comp = labeled[slc] == label_id
                area = int(np.count_nonzero(comp))
                if area < min_area or area > max_area:
                    continue
                ys, xs = np.nonzero(comp)
                if ys.size == 0:
                    continue
                y0 = int(slc[0].start)
                x0 = int(slc[1].start)
                cy = y0 + float(np.mean(ys))
                cx = x0 + float(np.mean(xs))
                bbox_w = int(slc[1].stop - slc[1].start)
                bbox_h = int(slc[0].stop - slc[0].start)
                fill = float(area / max(bbox_w * bbox_h, 1))
                aspect = float(bbox_w / max(bbox_h, 1))
                candidates.append(
                    {
                        "cy": cy,
                        "cx": cx,
                        "area": float(area),
                        "fill": fill,
                        "aspect": aspect,
                        "src": "component",
                    }
                )
            return candidates

        # Fallback: no connected-component analysis available.
        peaks = self._find_peak_locations(norm, float(np.percentile(norm, 80)), self.reference_patch.shape[1], self.reference_patch.shape[0])
        return [{"cy": float(y), "cx": float(x), "area": 0.0, "fill": 0.0, "aspect": 1.0} for y, x, _ in peaks]

    def _find_peak_center_candidates(self, frame: np.ndarray) -> list[dict[str, float]]:
        if self.reference_patch is None:
            return []

        norm = self._frame_norm_for_detection(frame)
        if ndimage is not None:
            smooth = ndimage.gaussian_filter(norm, sigma=1.05)
            size = int(max(5, round(min(self.reference_patch.shape) * 0.48)))
            local = ndimage.maximum_filter(smooth, size=size, mode="nearest")
            threshold = float(np.percentile(smooth, 76))
            mask = smooth >= threshold
            peak_mask = mask & (smooth >= (local - 1e-6))
            ys, xs = np.nonzero(peak_mask)
            peaks = sorted(
                ((int(y), int(x), float(smooth[y, x])) for y, x in zip(ys, xs)),
                key=lambda item: item[2],
                reverse=True,
            )
        else:
            peaks = self._find_peak_locations(norm, float(np.percentile(norm, 75)), self.reference_patch.shape[1], self.reference_patch.shape[0])

        min_dist = float(max(6.0, min(self.reference_patch.shape) * 0.64))
        accepted: list[dict[str, float]] = []
        for y, x, strength in peaks:
            keep = True
            for cand in accepted:
                if float(np.hypot(float(y) - cand["cy"], float(x) - cand["cx"])) < min_dist:
                    keep = False
                    break
            if not keep:
                continue
            accepted.append(
                {
                    "cy": float(y),
                    "cx": float(x),
                    "area": 0.0,
                    "fill": 0.0,
                    "aspect": 1.0,
                    "strength": float(strength),
                    "src": "peak",
                }
            )
            if len(accepted) >= 128:
                break
        return accepted

    def _collect_candidates(self, frame: np.ndarray) -> list[dict[str, float]]:
        """Merge NCC (best), then components, then peaks; suppress duplicates by center distance."""
        if self.reference_patch is None:
            return []
        tpl_min = float(min(self.reference_patch.shape))
        min_sep = float(max(6.0, tpl_min * 0.66))

        combined: list[dict[str, float]] = []
        for cand in (
            self._find_ncc_template_candidates(frame)
            + self._find_component_candidates(frame)
            + self._find_peak_center_candidates(frame)
        ):
            keep = True
            for existing in combined:
                if float(np.hypot(cand["cy"] - existing["cy"], cand["cx"] - existing["cx"])) < min_sep:
                    keep = False
                    break
            if keep:
                combined.append(cand)
        return combined

    def _match_reference(
        self,
        frame: np.ndarray,
        min_score: float,
    ) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
        if self.reference_patch is None:
            return []
        if frame is None or frame.ndim != 2:
            return []

        tpl_h, tpl_w = self.reference_patch.shape
        if frame.shape[0] < tpl_h or frame.shape[1] < tpl_w:
            return []
        matches: list[tuple[tuple[int, int, int, int], float, np.ndarray]] = []
        seen_rects: set[tuple[int, int, int, int]] = set()
        candidates = self._find_ncc_template_candidates(frame, min_score)

        for candidate in candidates:
            ncc_boost = float(candidate.get("ncc", 0.0) or 0.0)
            cand_y = float(candidate["cy"])
            cand_x = float(candidate["cx"])
            extracted = self._extract_centered_patch(frame, cand_y, cand_x, tpl_h, tpl_w)
            if extracted is None:
                continue
            matched_patch, x, y = extracted
            median_ok, cand_median, median_tol_abs = self._passes_median_match(matched_patch)
            if not median_ok:
                continue
            score = float(ncc_boost)

            candidate_mask, _candidate_band_mask, _ = _build_object_masks(
                matched_patch,
                bright_object=self.reference_bright_object,
            )
            patch_out, xr, yr, _mask_out = self._shift_roi_to_object_center(
                frame, int(x), int(y), tpl_h, tpl_w, candidate_mask
            )
            rect_final = (xr, yr, tpl_w, tpl_h)
            if rect_final in seen_rects:
                continue
            seen_rects.add(rect_final)
            matches.append((rect_final, float(score), patch_out.copy()))

        matches.sort(key=lambda item: item[1], reverse=True)
        return self._nms_match_rects(matches, iou_threshold=0.38)

    def _nms_match_rects(
        self,
        matches: list[tuple[tuple[int, int, int, int], float, np.ndarray]],
        iou_threshold: float = 0.38,
    ) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
        """Drop overlapping detections (same physical object) keeping the higher score."""
        if not matches:
            return []
        kept: list[tuple[tuple[int, int, int, int], float, np.ndarray]] = []
        for rect, score, crop in matches:
            dup = False
            for r2, s2, _ in kept:
                if _rectangle_iou_xywh(rect, r2) >= iou_threshold:
                    dup = True
                    break
            if not dup:
                kept.append((rect, score, crop))
        return kept

    def crop_current_frame(self) -> None:
        if self.reference_patch is None or self.reference_mask is None:
            self._set_status("先に Set Reference で基準ROIを登録してください。", error=True)
            return
        if self.current_image is None:
            self._set_status("現在フレームの画像データを取得できません。", error=True)
            return
        self._notify_opencv_missing_for_search()

        min_score = float(self.score_spin.value())
        matches = self._match_reference(self.current_image, min_score)
        if not matches:
            self.records = [
                rec for rec in self.records
                if not (rec.frame_index == self.current_frame_index and rec.channel == self.current_channel)
            ]
            self.image_viewer.set_match_rects(None)
            self._refresh_thumbnail_list()
            self._set_status(
                f"閾値 {min_score:.2f} 以上の類似 object は見つかりませんでした。"
                " ROIを見直すか閾値を下げてください。",
                error=True,
            )
            return

        replaced_count = sum(
            1 for rec in self.records
            if rec.frame_index == self.current_frame_index and rec.channel == self.current_channel
        )
        self.records = [
            rec for rec in self.records
            if not (rec.frame_index == self.current_frame_index and rec.channel == self.current_channel)
        ]

        for object_id, (rect, score, crop) in enumerate(matches, start=1):
            self.records.append(
                CropRecord(
                    frame_index=self.current_frame_index,
                    object_id=object_id,
                    channel=self.current_channel,
                    match_score=score,
                    roi_x=rect[0],
                    roi_y=rect[1],
                    roi_width=rect[2],
                    roi_height=rect[3],
                    crop=crop,
                    **self._crop_record_metadata(),
                )
            )

        self._sync_match_rects_overlay()
        self._refresh_thumbnail_list()
        action = "更新" if replaced_count else "追加"
        self._set_status(
            f"Frame {self.current_frame_index} を{action}しました。"
            f" {len(matches)} 個の object を抽出しました。"
        )

    def _next_object_id_for_current_frame(self) -> int:
        ids = [
            rec.object_id
            for rec in self.records
            if rec.frame_index == self.current_frame_index and rec.channel == self.current_channel
        ]
        return max(ids, default=0) + 1

    def _add_manual_crop_at_image_position(self, img_x: int, img_y: int) -> None:
        if self.reference_patch is None:
            self._set_status("先に Set Reference で基準ROIを登録してください。", error=True)
            return
        if self.current_image is None:
            self._set_status("現在フレームの画像データを取得できません。", error=True)
            return
        tpl_h, tpl_w = self.reference_patch.shape
        extracted = self._extract_centered_patch(
            self.current_image, float(img_y), float(img_x), tpl_h, tpl_w
        )
        if extracted is None:
            self._set_status("クリック位置からパッチを切り出せません（画像が小さすぎる可能性があります）。", error=True)
            return
        matched_patch, x0, y0 = extracted
        candidate_mask, _candidate_band_mask, _ = _build_object_masks(
            matched_patch,
            bright_object=self.reference_bright_object,
        )
        patch_out, xr, yr, mask_out = self._shift_roi_to_object_center(
            self.current_image, int(x0), int(y0), tpl_h, tpl_w, candidate_mask
        )
        rect_final = (xr, yr, tpl_w, tpl_h)
        for rec in self.records:
            if (
                rec.frame_index == self.current_frame_index
                and rec.channel == self.current_channel
                and int(rec.roi_x) == int(rect_final[0])
                and int(rec.roi_y) == int(rect_final[1])
                and int(rec.roi_width) == int(rect_final[2])
                and int(rec.roi_height) == int(rect_final[3])
            ):
                self._set_status("同じ位置の Crop が既に存在します。", error=True)
                return
        oid = self._next_object_id_for_current_frame()
        self.records.append(
            CropRecord(
                frame_index=self.current_frame_index,
                object_id=oid,
                channel=self.current_channel,
                match_score=MANUAL_CROP_SCORE,
                roi_x=rect_final[0],
                roi_y=rect_final[1],
                roi_width=rect_final[2],
                roi_height=rect_final[3],
                crop=patch_out.copy(),
                **self._crop_record_metadata(),
            )
        )
        self._sync_match_rects_overlay()
        self._refresh_thumbnail_list()
        self._set_status(
            f"手動で Crop を追加しました（Frame {self.current_frame_index}、object #{oid}）。"
        )

    def remove_selected_record(self) -> None:
        items = self.thumbnail_list.selectedItems()
        if not items:
            self._set_status("削除するサムネイルを選択してください。", error=True)
            return
        remove_keys: set[tuple[int, int, str]] = set()
        for item in items:
            data = item.data(QtCore.Qt.UserRole)
            if not data:
                continue
            fi, oid, ch = data
            remove_keys.add((int(fi), int(oid), str(ch)))
        if not remove_keys:
            self._set_status("選択中の結果を削除できませんでした。", error=True)
            return
        before = len(self.records)
        self.records = [
            rec for rec in self.records
            if (rec.frame_index, rec.object_id, rec.channel) not in remove_keys
        ]
        removed = before - len(self.records)
        if removed > 0:
            self._refresh_thumbnail_list()
            self._sync_match_rects_overlay()
            self._set_status(f"{removed} 件の Crop を削除しました。")
        else:
            self._set_status("選択中の結果を削除できませんでした。", error=True)

    def clear_all_records(self) -> None:
        self.records.clear()
        self.thumbnail_list.clear()
        self.image_viewer.set_match_rects(None)
        self._set_status("抽出済みデータをクリアしました。")

    def save_records(self) -> None:
        if not self.records:
            self._set_status("保存する抽出結果がありません。", error=True)
            return
        if self.reference_patch is None:
            self._set_status("Referenceが未設定です。", error=True)
            return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "クロップを別ファイルで ASD 保存（ベース名を指定）",
            self._default_save_path(),
            "ASD files (*.asd)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".asd"):
            save_path = save_path + ".asd"

        base_dir = os.path.dirname(save_path)
        base_stem = os.path.splitext(os.path.basename(save_path))[0]
        self._save_records_asd_per_crop(base_dir, base_stem)

    def _save_records_asd_per_crop(self, base_dir: str, base_stem: str) -> None:
        """各クロップを個別の ASD に保存。ファイル名: {base_stem}_f{frame}_c{object_id}.asd"""
        if SaveASD is None:
            self._set_status("fileio.SaveASD を読み込めません。pyNuD 本体の fileio を確認してください。", error=True)
            return
        if gv is None or not getattr(gv, "files", None):
            self._set_status("ASD 保存には読み込み済みファイル（gv.files）が必要です。", error=True)
            return

        ref_h, ref_w = self.reference_patch.shape
        sorted_recs = sorted(self.records, key=lambda rec: (rec.frame_index, rec.object_id))
        for i, rec in enumerate(sorted_recs):
            ch, cw = int(rec.crop.shape[0]), int(rec.crop.shape[1])
            if (ch, cw) != (ref_h, ref_w):
                self._set_status(
                    f"全クロップのサイズが一致している必要があります（期待 {ref_h}x{ref_w}、"
                    f"レコード #{i} は {ch}x{cw}）。",
                    error=True,
                )
                return

        channels = {str(rec.channel) for rec in sorted_recs}
        if len(channels) > 1:
            self._set_status(
                "1ch と 2ch の Crop が混在しています。どちらか一方だけの結果にしてから ASD 保存してください。",
                error=True,
            )
            return

        base_comment = getattr(gv, "Comment", "Saved by pyNuD")
        ch_note = next(iter(channels)) if channels else "1ch"
        original_index = int(getattr(gv, "index", 0))
        original_state = dict(vars(gv))

        try:
            for rec in sorted_recs:
                fi = int(rec.frame_index)
                oid = int(rec.object_id)
                out_name = f"{base_stem}_f{fi}_c{oid}.asd"
                out_path = os.path.join(base_dir, out_name)

                current_path = os.path.realpath(gv.files[gv.currentFileNum])
                if rec.source_path and rec.source_path != current_path:
                    self._set_status("Cropの取得元と現在のファイルが一致しません。", error=True)
                    return
                stat = os.stat(current_path)
                if rec.source_fingerprint and rec.source_fingerprint != (stat.st_ino, stat.st_size, stat.st_mtime_ns):
                    self._set_status("Crop取得後に元ファイルが変更されています。", error=True)
                    return

                try:
                    gv.index = fi
                    LoadFrame(gv.files[gv.currentFileNum])
                    if InitializeAryDataFallback is not None:
                        InitializeAryDataFallback()
                    frame_header_info_list = [
                        {
                            "xoffset": getattr(gv, "XOffset", 0),
                            "yoffset": getattr(gv, "YOffset", 0),
                            "xtilt": getattr(gv, "XTilt", 0.0),
                            "ytilt": getattr(gv, "YTilt", 0.0),
                            "laser_flag": getattr(gv, "LaserFlag", 0),
                            "temp_flag": getattr(gv, "TempFlag", 0),
                        }
                    ]
                except Exception as exc:
                    print(f"[WARNING] ReferenceObjectCropper ASD header frame {fi}: {exc}")
                    frame_header_info_list = [{}]

                # CropRecord is stored in display orientation; ASD rows use the
                # original acquisition orientation.
                data = np.array(np.flipud(rec.crop), dtype=np.float64, copy=True)
                kind = rec.signal_kind or (18512 if rec.channel == '2ch' else gv.DataType1ch)
                voltage_offset = None
                if rec.channel == '2ch' and kind == 18512:
                    if rec.conversion_type in ('Phase', 'Amplitude', 'AmplitudeError'):
                        if not np.isfinite(rec.conversion_factor) or rec.conversion_factor == 0:
                            self._set_status("2chの変換係数が無効です。", error=True)
                            return
                        data /= rec.conversion_factor
                    voltage_offset = float((data.min() + data.max()) / 2.)
                    data -= voltage_offset
                processed_data_array = [data]
                pixel_size = rec.pixel_size_nm or self._nm_per_pixel()
                if pixel_size is None:
                    self._set_status("Cropの物理ピクセルサイズが不明です。", error=True)
                    return
                comment = (
                    f"{base_comment}\n[Reference Object Cropper] frame={fi} object={oid} "
                    f"channel={ch_note} ref_frame={self.reference_frame_index}"
                )
                if voltage_offset is not None:
                    comment += f"\n[Original 2ch voltage = saved 1ch voltage + {voltage_offset:.17g} V]"

                try:
                    ok = SaveASD(
                        save_path=out_path,
                        comment=comment,
                        processed_data_array=processed_data_array,
                        processed_data_array_2ch=None,
                        progress_callback=None,
                        frame_header_info_list=frame_header_info_list,
                        save_scan_size_nm=(rec.roi_width * pixel_size[0], rec.roi_height * pixel_size[1]),
                        data_type_1ch=kind,
                    )
                except Exception as exc:
                    self._set_status(f"ASD 保存に失敗しました ({out_name}): {exc}", error=True)
                    return

                if not ok:
                    self._set_status(
                        f"ASD 保存が拒否されました（fileio.SaveASD が False）: {out_name}",
                        error=True,
                    )
                    return
        finally:
            for key in set(vars(gv)) - set(original_state):
                delattr(gv, key)
            vars(gv).update(original_state)

        self._set_status(
            f"{len(sorted_recs)} ファイルを保存しました（{base_stem}_f<フレーム>_c<番号>.asd、Save タブと同じ形式）。"
            f" 先頭: {base_stem}_f{int(sorted_recs[0].frame_index)}_c{int(sorted_recs[0].object_id)}.asd"
        )


def create_plugin(main_window):
    return ReferenceObjectCropperWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "ReferenceObjectCropperWindow"]
