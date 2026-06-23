"""
AFM Movie Editor plugin for pyNuD.

This plugin provides a compact timeline editor for AFM frame stacks:
source-frame thumbnails, source/title clips, text overlays, simple
transitions, and MP4/AVI export.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import textwrap
import traceback
import zipfile
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import cv2
except Exception:
    cv2 = None

try:
    import globalvals as gv
except Exception:
    gv = None

try:
    import colorbar_menu
except Exception:
    colorbar_menu = None

try:
    from fileio import InitializeAryDataFallback, LoadFrame
except Exception:
    InitializeAryDataFallback = None
    LoadFrame = None


PLUGIN_NAME = "AFM Movie Editor"


TRANSITIONS = ["Cut", "Cross Dissolve", "Fade Black", "Wipe Left"]
TRIMMED_OUTPUT_MODES = ["Skip Trimmed", "Show Trimmed"]
TEXT_ALIGNMENTS = ["Left", "Center", "Right", "Justify"]
OVERLAY_TEXT_ALIGNMENTS = ["Auto"] + TEXT_ALIGNMENTS
SHAPE_TYPES = ["Circle", "Ellipse", "Rectangle", "Triangle", "Line", "Arrow", "Arrowhead"]
SHAPE_LINE_STYLES = ["Solid", "Dash", "Dot", "Wave"]
POSITIONS = [
    "Top Left",
    "Top",
    "Top Right",
    "Left",
    "Center",
    "Right",
    "Bottom Left",
    "Bottom",
    "Bottom Right",
]
COLOR_MAPS = {
    "Gray": None,
    "Hot": "COLORMAP_HOT",
    "Viridis": "COLORMAP_VIRIDIS",
    "Turbo": "COLORMAP_TURBO",
    "Jet": "COLORMAP_JET",
    "Bone": "COLORMAP_BONE",
}

SESSION_GV_KEYS = [
    "active_display_channel",
    "currentColorMapName",
    "color_lut",
    "gamma_lut",
    "gamma_lut_1ch",
    "gamma_lut_2ch",
    "tone_state_1ch",
    "tone_state_2ch",
    "contrastMode",
    "contrastMin",
    "contrastMax",
    "contrastRange",
    "manualContrastMin",
    "manualContrastMax",
    "autoContrastMin",
    "autoContrastMax",
    "colorbar_dock_vertical_x",
    "colorbar_dock_vertical_y",
    "colorbar_dock_vertical_w",
    "colorbar_dock_vertical_h",
    "colorbar_dock_horizontal_x",
    "colorbar_dock_horizontal_y",
    "colorbar_dock_horizontal_w",
    "colorbar_dock_horizontal_h",
    "movie_zscale_padding",
    "movie_zscale_margin",
    "movie_zscale_offset_x_px",
    "movie_zscale_offset_y_px",
    "movie_zscale_grad_w",
    "movie_zscale_grad_h_ratio",
    "movie_zscale_tick_fs",
    "movie_zscale_precision",
    "movie_zscale_num_offset_x_px",
    "movie_zscale_num_offset_y_px",
    "movie_zscale_unit_fs",
    "movie_zscale_unit_offset_x_px",
    "movie_zscale_unit_offset_y_px",
    "timeFontSize",
    "timeFontFamily",
    "timeFontStyle",
    "timeFontPath",
    "timeFontColor",
    "timeBgBox",
    "timeBgBoxColor",
    "timeXPosRatio",
    "timeYPosRatio",
    "scaleFontSize",
    "scaleFontFamily",
    "scaleFontStyle",
    "scaleFontPath",
    "scaleFontColor",
    "scaleBgBox",
    "scaleBgBoxColor",
    "scaleXPosRatio",
    "scaleYPosRatio",
]


@dataclass
class ClipSpec:
    kind: str = "source"
    start: int = 0
    end: int = 0
    title: str = ""
    title_position: str = "Center"
    title_font_size: int = 48
    title_color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(255, 255, 255))
    title_background: bool = False
    title_font_family: str = "Arial"
    title_font_style: str = "Bold"
    title_align: str = "Center"
    title_background_color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(32, 34, 38))
    title_use_first_frame_background: bool = False
    title_background_frame_opacity: int = 35
    title_custom_position: bool = False
    title_x_ratio: float = 0.5
    title_y_ratio: float = 0.5
    duration: int = 30
    transition_in: str = "Cut"
    transition_in_frames: int = 0
    transition_in_hold_frames: int = 0
    transition_out: str = "Cut"
    transition_out_frames: int = 0
    transition_out_hold_frames: int = 0
    transition: str = "Cut"
    transition_frames: int = 0
    transition_hold_frames: int = 0
    trimmed: bool = False


@dataclass
class TextOverlaySpec:
    text: str
    start: int
    end: int
    position: str
    font_size: int
    color: QtGui.QColor
    background: bool = True
    font_family: str = "Arial"
    font_style: str = "Normal"
    text_align: str = "Auto"
    custom_position: bool = False
    x_ratio: float = 0.5
    y_ratio: float = 0.5


@dataclass
class ShapeOverlaySpec:
    shape: str
    start: int
    end: int
    x_ratio: float = 0.35
    y_ratio: float = 0.35
    x2_ratio: float = 0.65
    y2_ratio: float = 0.35
    w_ratio: float = 0.30
    h_ratio: float = 0.20
    line_width: int = 3
    line_style: str = "Solid"
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(255, 255, 0))
    line_opacity: int = 100
    fill: bool = False
    fill_color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(255, 255, 0))
    fill_opacity: int = 35
    arrow_head_percent: int = 20
    rotation_degrees: float = 0.0


COLOR_BAR_VALUE_POSITIONS_VERTICAL = ["Left", "Right"]
COLOR_BAR_VALUE_POSITIONS_HORIZONTAL = ["Above", "Below"]


class PluginMovieScaleWidget(QtWidgets.QWidget):
    """Movie Editor preview color-bar scale labels (plugin-local typography)."""

    def __init__(self, movie_editor: "AFMMovieEditorWindow", parent_window, orientation="Vertical", mode="Docking", parent=None):
        super().__init__(parent)
        self.movie_editor = movie_editor
        self.parent_window = parent_window
        self.orientation = orientation
        self.mode = mode
        self.setStyleSheet("background-color: transparent; border: none;")
        self.setMinimumSize(10, 10)

    def _values_style(self) -> Dict[str, Any]:
        return self.movie_editor._color_bar_text_style.get("values", {})

    def _value_font(self) -> QtGui.QFont:
        spec = self._values_style()
        font = QtGui.QFont(str(spec.get("font_family", "Arial")))
        base_size = max(6, int(spec.get("font_size", 8)))
        font.setPixelSize(self.movie_editor._scaled_color_bar_font_size(base_size, self.orientation))
        style = self.movie_editor._normalize_font_style(str(spec.get("font_style", "Bold")))
        font.setBold("Bold" in style)
        font.setItalic("Italic" in style)
        return font

    def _value_color(self) -> QtGui.QColor:
        color = self._values_style().get("color", QtGui.QColor(255, 255, 255))
        if isinstance(color, QtGui.QColor):
            return color
        return QtGui.QColor(255, 255, 255)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        text_color = self._value_color()
        painter.setPen(QtGui.QPen(text_color, 1))
        painter.setBrush(QtGui.QBrush(text_color))
        painter.setFont(self._value_font())

        z_min, z_max = self.movie_editor._plugin_colorbar_z_range()
        try:
            if self.parent_window is not None and hasattr(self.parent_window, "_get_z_range"):
                pw_min, pw_max = self.parent_window._get_z_range()
                if abs(float(pw_max) - float(pw_min)) > 1e-9:
                    z_min, z_max = pw_min, pw_max
        except Exception:
            pass

        if abs(z_max - z_min) < 1e-9:
            text = f"{z_min:.1f}"
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(text)
            x = (self.width() - text_rect.width()) / 2
            y = (self.height() - text_rect.height()) / 2 + metrics.ascent()
            painter.drawText(QtCore.QPointF(x, y), text)
            painter.end()
            return

        rect = self.rect()
        font_metrics = painter.fontMetrics()
        position = str(self._values_style().get("position", "Left" if self.orientation == "Vertical" else "Below"))
        x_offset = int(self._values_style().get("x_offset", 0))
        y_offset = int(self._values_style().get("y_offset", 0))

        if self.orientation == "Vertical":
            text_margin = font_metrics.height() // 2
            axis_start_y = rect.top() + text_margin
            axis_end_y = rect.bottom() - text_margin
            axis_height = axis_end_y - axis_start_y
            max_ticks = 5
            min_tick_spacing_px = font_metrics.height() * 1.5
            num_ticks = min(max_ticks, max(2, int(axis_height / max(1, min_tick_spacing_px))))
            if num_ticks <= 1 or axis_height <= 0:
                painter.end()
                return
            for i in range(num_ticks):
                ratio = i / (num_ticks - 1)
                y_pos = axis_start_y + (ratio * axis_height)
                value = z_max - (z_max - z_min) * ratio
                text = f"{value:.1f}"
                tick_x = rect.right()
                painter.drawLine(tick_x - 5, int(y_pos), tick_x, int(y_pos))
                text_width = font_metrics.horizontalAdvance(text) + 5
                text_height = font_metrics.height()
                if position == "Right":
                    text_x = max(0, rect.left() + 2 + x_offset)
                    text_rect = QtCore.QRect(
                        text_x,
                        int(y_pos - (text_height // 2)) + y_offset,
                        max(text_width, rect.width() - text_x),
                        text_height,
                    )
                    align = QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft
                else:
                    text_x = rect.right() - 10 - text_width + x_offset
                    if text_x < rect.left():
                        text_x = rect.left() + 2
                        text_width = max(1, rect.right() - 10 - text_x)
                    text_rect = QtCore.QRect(
                        text_x,
                        int(y_pos - (text_height // 2)) + y_offset,
                        text_width,
                        text_height,
                    )
                    align = QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight
                painter.drawText(text_rect, align, text)
        else:
            max_ticks = 5
            min_tick_spacing_px = font_metrics.horizontalAdvance("000.00")
            max_text_width = 0
            for i in range(max_ticks):
                ratio = i / (max_ticks - 1) if max_ticks > 1 else 0
                value = z_min + (z_max - z_min) * ratio
                text_width = font_metrics.horizontalAdvance(f"{value:.1f}")
                max_text_width = max(max_text_width, text_width)
            margin = max_text_width // 2
            drawable_width = rect.width() - (2 * margin)
            num_ticks = min(max_ticks, max(2, int(drawable_width / max(1, min_tick_spacing_px * 1.2))))
            if num_ticks <= 1 or drawable_width <= 0:
                painter.end()
                return
            tick_height = 5
            for i in range(num_ticks):
                ratio = i / (num_ticks - 1)
                value = z_min + (z_max - z_min) * ratio
                text = f"{value:.1f}"
                x_pos_tick = rect.left() + margin + int(drawable_width * ratio)
                text_width = font_metrics.horizontalAdvance(text)
                text_x_center = x_pos_tick - (text_width // 2) + x_offset
                if position == "Above":
                    text_y = rect.top() + font_metrics.ascent() + y_offset
                    painter.drawLine(x_pos_tick, rect.top() + font_metrics.height() + 2, x_pos_tick, rect.top() + font_metrics.height() + 2 + tick_height)
                else:
                    text_y = rect.top() + tick_height + 2 + font_metrics.ascent() + y_offset
                    painter.drawLine(x_pos_tick, rect.top(), x_pos_tick, rect.top() + tick_height)
                painter.drawText(QtCore.QPoint(int(text_x_center), int(text_y)), text)
        painter.end()


class EditablePreviewLabel(QtWidgets.QLabel):
    def __init__(self, owner: "AFMMovieEditorWindow"):
        super().__init__("Preview")
        self.owner = owner
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.owner._preview_mouse_press(event):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.owner._preview_mouse_double_click(event):
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.owner._preview_mouse_move(event):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.owner._preview_mouse_release(event):
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.owner._preview_wheel(event):
            event.accept()
            return
        super().wheelEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        event.accept()


class SourceFrameListWidget(QtWidgets.QListWidget):
    thumbnailZoomChanged = QtCore.pyqtSignal(int)
    thumbnailZoomFinished = QtCore.pyqtSignal()
    sourceFrameRightDoubleClicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._right_drag_active = False
        self._right_drag_start_x = 0
        self._right_drag_start_size = 72
        self._right_drag_moved = False
        self.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton:
            self._right_drag_active = True
            self._right_drag_start_x = int(event.pos().x())
            try:
                self._right_drag_start_size = max(1, int(self.property("thumbnailImageSize")))
            except Exception:
                self._right_drag_start_size = max(1, int(self.iconSize().width()))
            self._right_drag_moved = False
            self.setCursor(QtCore.Qt.SizeHorCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._right_drag_active:
            dx = int(event.pos().x()) - self._right_drag_start_x
            if abs(dx) >= 2:
                self._right_drag_moved = True
                new_size = self._right_drag_start_size + dx
                self.thumbnailZoomChanged.emit(new_size)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton:
            self._right_drag_active = False
            self._right_drag_moved = False
            self.unsetCursor()
            item = self.itemAt(event.pos())
            if item is not None:
                try:
                    frame_index = int(item.data(QtCore.Qt.UserRole))
                    self.sourceFrameRightDoubleClicked.emit(frame_index)
                except Exception:
                    pass
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton and self._right_drag_active:
            self._right_drag_active = False
            self.unsetCursor()
            if self._right_drag_moved:
                self.thumbnailZoomFinished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        event.accept()
        self._right_drag_moved = False


class SourceFrameItemDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        widget = option.widget
        try:
            thumb_size = int(widget.property("thumbnailImageSize")) if widget is not None else 72
        except Exception:
            thumb_size = 72
        thumb_size = max(1, thumb_size)
        rect = option.rect
        selected = bool(option.state & QtWidgets.QStyle.State_Selected)

        painter.save()
        icon_rect = QtCore.QRect(rect.x(), rect.y(), rect.width(), min(thumb_size, rect.height()))
        painter.fillRect(icon_rect, QtGui.QColor(0, 0, 0))

        icon = index.data(QtCore.Qt.DecorationRole)
        if isinstance(icon, QtGui.QIcon) and not icon.isNull():
            pix = icon.pixmap(QtCore.QSize(rect.width(), thumb_size))
            if not pix.isNull():
                painter.drawPixmap(rect.x(), rect.y(), pix)

        text_rect = QtCore.QRect(rect.x(), rect.y() + thumb_size, rect.width(), max(0, rect.height() - thumb_size))
        if text_rect.height() > 0:
            if selected:
                painter.fillRect(text_rect, option.palette.highlight())
                painter.setPen(option.palette.highlightedText().color())
            else:
                background = index.data(QtCore.Qt.BackgroundRole)
                if isinstance(background, QtGui.QBrush):
                    painter.fillRect(text_rect, background)
                else:
                    painter.fillRect(text_rect, option.palette.base())
                foreground = index.data(QtCore.Qt.ForegroundRole)
                if isinstance(foreground, QtGui.QBrush):
                    painter.setPen(foreground.color())
                else:
                    painter.setPen(option.palette.text().color())
            painter.drawText(text_rect, QtCore.Qt.AlignCenter, str(index.data(QtCore.Qt.DisplayRole) or ""))
        painter.restore()

    def sizeHint(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtCore.QSize:
        widget = option.widget
        try:
            thumb_size = int(widget.property("thumbnailImageSize")) if widget is not None else 72
        except Exception:
            thumb_size = 72
        return QtCore.QSize(thumb_size + 2, max(thumb_size + 24, 42))


class OutputFrameSlider(QtWidgets.QSlider):
    rightClicked = QtCore.pyqtSignal(int)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton:
            self.rightClicked.emit(int(self.value()))
            event.accept()
            return
        super().mousePressEvent(event)


class AFMMovieEditorWindow(QtWidgets.QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.total_frames = self._detect_total_frames()
        self.clips: List[ClipSpec] = []
        self.overlays: List[TextOverlaySpec] = []
        self.shape_overlays: List[ShapeOverlaySpec] = []
        self._last_split_output_index: Optional[int] = None
        self._settings = QtCore.QSettings("pyNuD", PLUGIN_NAME)
        self._last_preview_index = 0
        self._timeline_cache: Optional[List[Dict[str, Any]]] = None
        self._thumbnail_timer: Optional[QtCore.QTimer] = None
        self._movie_preview_timer: Optional[QtCore.QTimer] = None
        self._thumbnail_cursor = 0
        self._thumbnail_restore_index = 0
        self._last_preview_bgr: Optional[np.ndarray] = None
        self._contrast_limits: Optional[Tuple[float, float]] = None
        self._ui_updating = False
        self._clip_ui_updating = False
        self._overlay_ui_updating = False
        self._shape_ui_updating = False
        self._title_ui_updating = False
        self._preview_drag_target: Optional[str] = None
        self._preview_drag_mode: Optional[str] = None
        self._preview_last_img_pos: Optional[QtCore.QPoint] = None
        self._preview_drag_offset: Optional[QtCore.QPoint] = None
        self._plugin_colorbar_render_rect: Optional[Tuple[int, int, int, int]] = None
        self._plugin_colorbar_render_frame_size: Optional[Tuple[int, int]] = None
        self._plugin_colorbar_render_orientation: Optional[str] = None
        self._plugin_colorbar_widget_cache: Dict[Any, Any] = {}
        self._color_bar_ref_size = {
            "Vertical": (80, 200),
            "Horizontal": (180, 80),
        }
        self._color_bar_text_style = {
            "unit": {
                "font_family": "Arial",
                "font_style": "Bold",
                "font_size": 8,
                "color": QtGui.QColor(255, 255, 255),
                "position": "Top",
                "align": "Center",
            },
            "values": {
                "font_family": "Arial",
                "font_style": "Bold",
                "font_size": 8,
                "color": QtGui.QColor(255, 255, 255),
                "position": "Left",
                "x_offset": 0,
                "y_offset": 0,
            },
        }
        self._sync_color_bar_text_style_from_gv()
        self._shape_rotate_start_angle: float = 0.0
        self._shape_rotate_start_mouse_angle: float = 0.0
        self._frame_num = {
            "font_family": "Arial",
            "font_style": "Normal",
            "font_size": 18,
            "color": QtGui.QColor(255, 255, 255),
            "background": True,
            "position": "Top Right",
            "custom": False,
            "x_ratio": 0.75,
            "y_ratio": 0.04,
        }
        self._init_frame_number_defaults()

        self.setWindowTitle(PLUGIN_NAME)
        self.resize(1280, 820)
        self._setup_ui()
        self._restore_ui_settings()
        self._refresh_source_frames(reset_clips=True)
        QtCore.QTimer.singleShot(0, self._update_preview)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    @staticmethod
    def _set_tip(widget: QtWidgets.QWidget, text: str) -> None:
        plain_text = str(text).strip()
        wrapped_text = "\n".join(
            textwrap.fill(
                line.strip(),
                width=72,
                break_long_words=False,
                break_on_hyphens=False,
            )
            if line.strip()
            else ""
            for line in plain_text.splitlines()
        )
        widget.setToolTip(wrapped_text)
        widget.setStatusTip(plain_text)
        widget.setWhatsThis(wrapped_text)

    def _setup_ui(self) -> None:
        tip = self._set_tip
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(self.main_splitter, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        self.main_splitter.addWidget(left)

        self.preview_scroll_area = QtWidgets.QScrollArea()
        self.preview_scroll_area.setWidgetResizable(False)
        self.preview_scroll_area.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.preview_scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.preview_label = EditablePreviewLabel(self)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(160, 120)
        self.preview_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.preview_label.setStyleSheet("QLabel { background: #16181c; color: #d8dce3; border: 1px solid #3b3f46; }")
        tip(
            self.preview_label,
            "Preview of the exported movie frame. Drag text, shapes, and the Color bar here; double-click text to edit it; right-click supported items for style settings.",
        )
        self.preview_scroll_area.setWidget(self.preview_label)
        left_layout.addWidget(self.preview_scroll_area, stretch=1)

        preview_controls = QtWidgets.QGridLayout()
        preview_controls.setContentsMargins(0, 0, 0, 0)
        preview_controls.setHorizontalSpacing(8)
        preview_controls.setVerticalSpacing(4)
        self.preview_slider = OutputFrameSlider(QtCore.Qt.Horizontal)
        self.preview_slider.setRange(0, 0)
        self.preview_slider.setMinimumWidth(90)
        self.preview_slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.preview_slider.valueChanged.connect(self._update_preview)
        self.preview_slider.rightClicked.connect(self._on_preview_slider_right_clicked)
        tip(
            self.preview_slider,
            "Select the output movie frame shown in Preview. Right-click the slider at the current position to split the active source clip.",
        )
        self.preview_frame_label = QtWidgets.QLabel("0 / 0")
        self.preview_frame_label.setMinimumWidth(64)
        self.split_status_label = QtWidgets.QLabel("Right-click slider to split clip")
        self.split_status_label.setMinimumWidth(80)
        self.trimmed_output_combo = QtWidgets.QComboBox()
        self.trimmed_output_combo.addItems(TRIMMED_OUTPUT_MODES)
        self.trimmed_output_combo.currentTextChanged.connect(self._trimmed_output_mode_changed)
        tip(
            self.trimmed_output_combo,
            "Choose whether trimmed clips are skipped from output or still shown in the preview timeline.",
        )
        preview_controls.addWidget(QtWidgets.QLabel("Output frame:"), 0, 0)
        preview_controls.addWidget(self.preview_slider, 0, 1)
        preview_controls.addWidget(self.preview_frame_label, 0, 2)
        preview_controls.addWidget(QtWidgets.QLabel("Trimmed clips:"), 1, 0)
        preview_controls.addWidget(self.trimmed_output_combo, 1, 1)
        preview_controls.addWidget(self.split_status_label, 1, 2)
        preview_controls.setColumnStretch(1, 1)
        left_layout.addLayout(preview_controls)

        source_group = QtWidgets.QGroupBox("Source Frames")
        source_layout = QtWidgets.QVBoxLayout(source_group)
        self.source_list = SourceFrameListWidget()
        self.source_list.setViewMode(QtWidgets.QListView.IconMode)
        self.source_list.setFlow(QtWidgets.QListView.LeftToRight)
        self.source_list.setWrapping(True)
        self.source_list.setMovement(QtWidgets.QListView.Static)
        self.source_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.source_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.source_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.source_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.source_list.setStyleSheet("QListWidget::item { margin: 0px; padding: 0px; }")
        self.source_list.setItemDelegate(SourceFrameItemDelegate(self.source_list))
        self._source_thumb_size = 72
        self._source_thumb_min = 28
        self._source_thumb_max = 180
        self._apply_source_thumbnail_size(self._source_thumb_size)
        self.source_list.setMinimumHeight(56)
        self.source_list.itemSelectionChanged.connect(self._on_source_selection_changed)
        self.source_list.thumbnailZoomChanged.connect(self._apply_source_thumbnail_size)
        self.source_list.thumbnailZoomFinished.connect(self._finish_source_thumbnail_resize)
        self.source_list.sourceFrameRightDoubleClicked.connect(self._on_source_frame_right_double_clicked)
        tip(
            self.source_list,
            "Loaded AFM source frames. Click a frame to preview it, right-drag to resize thumbnails, or right-double-click inside a source clip to split there.",
        )
        source_layout.addWidget(self.source_list)
        source_buttons = QtWidgets.QHBoxLayout()
        refresh_button = QtWidgets.QPushButton("Refresh Frames")
        refresh_button.clicked.connect(lambda: self._refresh_source_frames(reset_project=True))
        tip(refresh_button, "Reload thumbnails from the currently opened AFM data and reset the Movie Editor project.")
        self.thumb_status_label = QtWidgets.QLabel("")
        source_buttons.addWidget(refresh_button)
        source_buttons.addWidget(self.thumb_status_label, stretch=1)
        source_layout.addLayout(source_buttons)
        left_layout.addWidget(source_group)

        right_scroll = QtWidgets.QScrollArea()
        self.right_scroll = right_scroll
        right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        right_scroll.setWidgetResizable(False)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        right_scroll.setMinimumSize(180, 120)

        right = QtWidgets.QWidget()
        self.right_panel = right
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 6, 0)
        right_scroll.setWidget(right)
        self.main_splitter.addWidget(right_scroll)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)

        clip_group = QtWidgets.QGroupBox("Clips")
        clip_layout = QtWidgets.QVBoxLayout(clip_group)

        transition_layout = QtWidgets.QGridLayout()
        self.transition_out_combo = QtWidgets.QComboBox()
        self.transition_out_combo.addItems(TRANSITIONS)
        self.transition_out_combo.setFixedWidth(140)
        tip(self.transition_out_combo, "Transition effect applied from the selected clip to the next clip.")
        self.transition_out_frames_spin = QtWidgets.QSpinBox()
        self.transition_out_frames_spin.setRange(0, 300)
        self.transition_out_frames_spin.setValue(0)
        self.transition_out_frames_spin.setFixedWidth(60)
        tip(self.transition_out_frames_spin, "Number of output frames used for the selected transition.")
        self.transition_out_hold_frames_spin = QtWidgets.QSpinBox()
        self.transition_out_hold_frames_spin.setRange(0, 300)
        self.transition_out_hold_frames_spin.setValue(0)
        self.transition_out_hold_frames_spin.setFixedWidth(60)
        tip(self.transition_out_hold_frames_spin, "Number of black hold frames added around Fade Black transitions.")
        self.transition_out_preview_button = QtWidgets.QPushButton("Preview")
        self.transition_out_preview_button.clicked.connect(lambda: self._preview_selected_transition("out"))
        tip(self.transition_out_preview_button, "Jump the preview to the selected clip boundary to check the transition.")
        for widget in (self.transition_out_combo, self.transition_out_frames_spin, self.transition_out_hold_frames_spin):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._on_clip_transition_changed)
            else:
                widget.valueChanged.connect(self._on_clip_transition_changed)
        transition_layout.addWidget(QtWidgets.QLabel("Transition to next:"), 0, 0)
        transition_layout.addWidget(self.transition_out_combo, 0, 1)
        transition_layout.addWidget(self.transition_out_preview_button, 0, 2)
        transition_frames_row = QtWidgets.QHBoxLayout()
        transition_frames_row.setContentsMargins(0, 0, 0, 0)
        transition_frames_row.setSpacing(4)
        transition_frames_row.addWidget(QtWidgets.QLabel("frames:"))
        transition_frames_row.addWidget(self.transition_out_frames_spin)
        transition_frames_row.addSpacing(12)
        transition_frames_row.addWidget(QtWidgets.QLabel("black:"))
        transition_frames_row.addWidget(self.transition_out_hold_frames_spin)
        transition_frames_row.addStretch(1)
        transition_layout.addLayout(transition_frames_row, 1, 0, 1, 5)
        transition_layout.addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum),
            0,
            4,
        )
        transition_layout.setColumnStretch(1, 0)
        transition_layout.setColumnStretch(4, 1)
        clip_layout.addLayout(transition_layout)

        clip_buttons = QtWidgets.QGridLayout()
        self.add_all_button = QtWidgets.QPushButton("Reset Clips")
        self.add_source_button = QtWidgets.QPushButton("Add Source Clip")
        self.trim_clip_button = QtWidgets.QPushButton("Trim")
        self.move_up_button = QtWidgets.QPushButton("Move Up")
        self.move_down_button = QtWidgets.QPushButton("Move Down")
        self.add_all_button.clicked.connect(self._reset_source_clips)
        self.add_source_button.clicked.connect(self._add_source_clip)
        self.trim_clip_button.clicked.connect(self._trim_selected_clip)
        self.move_up_button.clicked.connect(lambda: self._move_selected_clip(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected_clip(1))
        tip(self.add_all_button, "Restore one source clip covering all source frames. Existing title and overlay clips are kept.")
        tip(self.add_source_button, "Add a source clip from the selected Source Frames range.")
        tip(self.trim_clip_button, "Mark the selected clip as trimmed so it is removed from normal output.")
        tip(self.move_up_button, "Move the selected clip earlier in the clip list.")
        tip(self.move_down_button, "Move the selected clip later in the clip list.")
        buttons = [
            self.add_all_button,
            self.add_source_button,
            self.trim_clip_button,
            self.move_up_button,
            self.move_down_button,
        ]
        for idx, button in enumerate(buttons):
            button.setFixedWidth(140)
            clip_buttons.addWidget(button, idx // 2, idx % 2, alignment=QtCore.Qt.AlignLeft)
        clip_buttons.addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum),
            0,
            2,
        )
        clip_buttons.setColumnStretch(0, 0)
        clip_buttons.setColumnStretch(1, 0)
        clip_buttons.setColumnStretch(2, 1)
        clip_layout.addLayout(clip_buttons)

        self.clip_table = QtWidgets.QTableWidget()
        self.clip_table.setColumnCount(5)
        self.clip_table.setHorizontalHeaderLabels(["#", "Type", "Content", "Duration", "Transition to next"])
        self.clip_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.clip_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.clip_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.clip_table.verticalHeader().setVisible(False)
        self.clip_table.verticalHeader().setDefaultSectionSize(24)
        self.clip_table.verticalHeader().setMinimumSectionSize(20)
        self.clip_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.clip_table.horizontalHeader().setStretchLastSection(False)
        self.clip_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.clip_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.clip_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.clip_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.clip_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.clip_table.itemSelectionChanged.connect(self._on_clip_selection_changed)
        tip(
            self.clip_table,
            "Timeline clips used to build the movie. Select a row to edit its transition, trim state, or ordering.",
        )
        self._set_compact_table_height(self.clip_table, rows=1, max_rows=5)
        clip_layout.addWidget(self.clip_table, 0, QtCore.Qt.AlignLeft)
        right_layout.addWidget(clip_group)

        title_group = QtWidgets.QGroupBox("Title")
        title_layout = QtWidgets.QVBoxLayout(title_group)
        title_form = QtWidgets.QVBoxLayout()
        title_form.setSpacing(8)

        self.title_text_edit = QtWidgets.QPlainTextEdit()
        self.title_text_edit.setPlaceholderText("Title text")
        self.title_text_edit.setFixedHeight(54)
        self.title_text_edit.setFixedWidth(260)
        self.title_text_edit.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        tip(self.title_text_edit, "Text used when adding a title clip. Select an existing title clip to edit its text.")
        self.title_duration_spin = QtWidgets.QSpinBox()
        self.title_duration_spin.setRange(1, 5000)
        self.title_duration_spin.setValue(max(1, int(round(self._default_fps()))))
        self.title_duration_seconds_label = QtWidgets.QLabel("")
        self.title_duration_spin.valueChanged.connect(self._update_title_duration_seconds_label)
        tip(self.title_duration_spin, "Length of the title clip in output frames.")
        self.title_font_size_spin = QtWidgets.QSpinBox()
        self.title_font_size_spin.setRange(8, 160)
        _title_default_w, title_default_h = self._default_output_size()
        self.title_font_size_spin.setValue(max(18, min(96, int(title_default_h * 0.07))))
        tip(self.title_font_size_spin, "Default title text size in pixels for newly added title clips.")
        self.title_font_combo = QtWidgets.QComboBox()
        self.title_font_combo.addItems(self._available_font_families())
        current_family = str(self._frame_num.get("font_family", "Arial"))
        if current_family:
            self.title_font_combo.setCurrentText(current_family)
        tip(self.title_font_combo, "Default title font family for newly added title clips.")
        self.title_style_combo = QtWidgets.QComboBox()
        self.title_style_combo.addItems(["Normal", "Bold", "Italic", "Bold Italic"])
        self.title_style_combo.setCurrentText("Bold")
        tip(self.title_style_combo, "Default title font style for newly added title clips.")
        self.title_align_combo = QtWidgets.QComboBox()
        self.title_align_combo.addItems(TEXT_ALIGNMENTS)
        self.title_align_combo.setCurrentText("Center")
        tip(self.title_align_combo, "Default alignment for multiline title text.")
        self.title_text_color = QtGui.QColor(255, 255, 255)
        self.title_bg_color = QtGui.QColor(32, 34, 38)
        self.title_color_button = QtWidgets.QPushButton("Text Color")
        self.title_color_button.clicked.connect(self._choose_title_text_color)
        tip(self.title_color_button, "Choose the title text color.")
        self.title_bg_color_button = QtWidgets.QPushButton("Background Color")
        self.title_bg_color_button.clicked.connect(self._choose_title_background_color)
        tip(self.title_bg_color_button, "Choose the solid background color behind title text.")
        self.title_use_first_frame_bg_check = QtWidgets.QCheckBox("Use first frame as background")
        tip(self.title_use_first_frame_bg_check, "Use the first AFM frame as the title background instead of a flat color.")
        self.title_bg_opacity_spin = QtWidgets.QSpinBox()
        self.title_bg_opacity_spin.setRange(0, 100)
        self.title_bg_opacity_spin.setSuffix("%")
        self.title_bg_opacity_spin.setValue(35)
        tip(self.title_bg_opacity_spin, "Opacity of the first-frame background when it is used for the title clip.")

        title_label_texts = ("Text:", "Duration:")
        title_label_width = max(QtWidgets.QLabel(text).sizeHint().width() for text in title_label_texts)
        self.title_duration_spin.setFixedWidth(60)
        self.title_font_size_spin.setFixedWidth(60)
        self.title_font_combo.setFixedWidth(40)
        self.title_align_combo.setFixedWidth(70)
        self.title_style_combo.setFixedWidth(70)
        self.title_bg_opacity_spin.setFixedWidth(45)

        def title_label(text: str, alignment=QtCore.Qt.AlignLeft) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setFixedWidth(title_label_width)
            label.setAlignment(alignment | QtCore.Qt.AlignVCenter)
            return label

        def inline_title_label(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return label

        def title_row(label_text: str, compact_label: bool = False) -> QtWidgets.QHBoxLayout:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            alignment = QtCore.Qt.AlignRight if compact_label else QtCore.Qt.AlignLeft
            row.addWidget(title_label(label_text, alignment))
            return row

        title_text_row = title_row("Text:")
        title_text_row.addWidget(self.title_text_edit)
        title_text_row.addStretch(1)
        title_form.addLayout(title_text_row)

        title_duration_row = title_row("Duration:", compact_label=True)
        title_duration_row.addWidget(self.title_duration_spin)
        title_duration_row.addWidget(inline_title_label("frames"))
        title_duration_row.addSpacing(8)
        title_duration_row.addWidget(self.title_duration_seconds_label)
        title_duration_row.addStretch(1)
        title_form.addLayout(title_duration_row)

        title_color_row = QtWidgets.QHBoxLayout()
        title_color_row.setContentsMargins(0, 0, 0, 0)
        title_color_row.setSpacing(8)
        title_color_row.addWidget(self.title_bg_color_button)
        title_color_row.addStretch(1)
        title_form.addLayout(title_color_row)

        title_bg_row = QtWidgets.QHBoxLayout()
        title_bg_row.setContentsMargins(0, 0, 0, 0)
        title_bg_row.setSpacing(8)
        title_bg_row.addWidget(self.title_use_first_frame_bg_check)
        title_bg_row.addWidget(self.title_bg_opacity_spin)
        title_bg_row.addStretch(1)
        title_form.addLayout(title_bg_row)
        title_layout.addLayout(title_form)

        title_buttons = QtWidgets.QHBoxLayout()
        self.add_title_button = QtWidgets.QPushButton("Add Title at Start")
        self.add_title_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.add_title_button.clicked.connect(self._add_title_clip)
        tip(self.add_title_button, "Insert a title clip at the start of the movie using the current title settings.")
        title_buttons.addWidget(self.add_title_button, 0, QtCore.Qt.AlignLeft)
        title_buttons.addStretch(1)
        title_layout.addLayout(title_buttons)
        self._set_title_color_buttons()
        self._update_title_duration_seconds_label()
        self._connect_title_controls()
        right_layout.addWidget(title_group)

        overlay_group = QtWidgets.QGroupBox("Text Overlays")
        overlay_layout = QtWidgets.QVBoxLayout(overlay_group)
        overlay_form = QtWidgets.QVBoxLayout()
        overlay_form.setSpacing(8)
        self.overlay_text_edit = QtWidgets.QPlainTextEdit()
        self.overlay_text_edit.setPlaceholderText("Text overlay")
        self.overlay_text_edit.setFixedHeight(54)
        self.overlay_text_edit.setFixedWidth(260)
        self.overlay_text_edit.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        tip(self.overlay_text_edit, "Text for a movie overlay. Select an overlay row to edit its text.")
        self.overlay_start_spin = QtWidgets.QSpinBox()
        self.overlay_end_spin = QtWidgets.QSpinBox()
        for spin in (self.overlay_start_spin, self.overlay_end_spin):
            spin.setRange(0, 0)
            spin.setFixedWidth(60)
        tip(self.overlay_start_spin, "First output frame where the selected text overlay is visible.")
        tip(self.overlay_end_spin, "Last output frame where the selected text overlay is visible.")
        self.overlay_pos_combo = QtWidgets.QComboBox()
        self.overlay_pos_combo.addItems(POSITIONS)
        self.overlay_pos_combo.setCurrentText("Bottom")
        self.overlay_pos_combo.setFixedWidth(70)
        tip(self.overlay_pos_combo, "Preset position for newly added text overlays before they are dragged in Preview.")
        self.overlay_size_spin = QtWidgets.QSpinBox()
        self.overlay_size_spin.setRange(8, 160)
        self.overlay_size_spin.setValue(28)
        self.overlay_size_spin.setFixedWidth(60)
        tip(self.overlay_size_spin, "Text overlay font size in pixels.")
        self.overlay_bg_check = QtWidgets.QCheckBox("Background")
        self.overlay_bg_check.setChecked(True)
        tip(self.overlay_bg_check, "Draw a dark background box behind the text overlay for contrast.")
        self.overlay_color = QtGui.QColor(255, 255, 255)
        self.color_button = QtWidgets.QPushButton("Color")
        self.color_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.color_button.clicked.connect(self._choose_overlay_color)
        tip(self.color_button, "Choose the selected text overlay color.")

        overlay_label_texts = ("Text:", "Start:")
        overlay_label_width = max(QtWidgets.QLabel(text).sizeHint().width() for text in overlay_label_texts)

        def overlay_label(text: str, alignment=QtCore.Qt.AlignLeft) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setFixedWidth(overlay_label_width)
            label.setAlignment(alignment | QtCore.Qt.AlignVCenter)
            return label

        def overlay_inline_label(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return label

        def overlay_row(label_text: str, compact_label: bool = False) -> QtWidgets.QHBoxLayout:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            alignment = QtCore.Qt.AlignRight if compact_label else QtCore.Qt.AlignLeft
            row.addWidget(overlay_label(label_text, alignment))
            return row

        overlay_text_row = overlay_row("Text:")
        overlay_text_row.addWidget(self.overlay_text_edit)
        overlay_text_row.addStretch(1)
        overlay_form.addLayout(overlay_text_row)

        overlay_range_row = overlay_row("Start:", compact_label=True)
        overlay_range_row.addWidget(self.overlay_start_spin)
        overlay_range_row.addSpacing(8)
        overlay_range_row.addWidget(overlay_inline_label("End:"))
        overlay_range_row.addWidget(self.overlay_end_spin)
        overlay_range_row.addStretch(1)
        overlay_form.addLayout(overlay_range_row)

        overlay_layout.addLayout(overlay_form)

        overlay_buttons = QtWidgets.QHBoxLayout()
        add_overlay_button = QtWidgets.QPushButton("Add Overlay")
        delete_overlay_button = QtWidgets.QPushButton("Delete Overlay")
        for button in (add_overlay_button, delete_overlay_button):
            button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        add_overlay_button.clicked.connect(self._add_overlay)
        delete_overlay_button.clicked.connect(self._delete_overlay)
        tip(add_overlay_button, "Add a text overlay using the current text and output-frame range.")
        tip(delete_overlay_button, "Delete the selected text overlay.")
        overlay_buttons.addWidget(add_overlay_button, 0, QtCore.Qt.AlignLeft)
        overlay_buttons.addWidget(delete_overlay_button, 0, QtCore.Qt.AlignLeft)
        overlay_buttons.addStretch(1)
        overlay_layout.addLayout(overlay_buttons)

        self.overlay_table = QtWidgets.QTableWidget()
        self.overlay_table.setColumnCount(5)
        self.overlay_table.setHorizontalHeaderLabels(["Text", "Start", "End", "Position", "Size"])
        self.overlay_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.overlay_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.overlay_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.overlay_table.verticalHeader().setVisible(False)
        self.overlay_table.verticalHeader().setDefaultSectionSize(24)
        self.overlay_table.verticalHeader().setMinimumSectionSize(20)
        self.overlay_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        overlay_header = self.overlay_table.horizontalHeader()
        overlay_header.setStretchLastSection(False)
        overlay_header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        for col in (1, 2, 3):
            overlay_header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        overlay_header.setSectionResizeMode(4, QtWidgets.QHeaderView.Fixed)
        self.overlay_table.setColumnWidth(0, 180)
        self.overlay_table.setColumnWidth(4, 70)
        self.overlay_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.overlay_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.overlay_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.overlay_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.overlay_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.overlay_table.itemSelectionChanged.connect(self._on_overlay_selection_changed)
        tip(
            self.overlay_table,
            "Text overlays drawn on top of the movie. Select a row to edit text and frame range; drag the overlay in Preview to reposition it.",
        )
        self._set_compact_table_height(self.overlay_table, rows=1, max_rows=4)
        self.overlay_table.resizeColumnsToContents()
        self.overlay_table.setColumnWidth(0, 180)
        self.overlay_table.setColumnWidth(4, 70)
        self._fit_table_width_to_columns(self.overlay_table)
        overlay_layout.addWidget(self.overlay_table, 0, QtCore.Qt.AlignLeft)
        right_layout.addWidget(overlay_group)

        self.overlay_start_spin.valueChanged.connect(self._on_overlay_controls_changed)
        self.overlay_end_spin.valueChanged.connect(self._on_overlay_controls_changed)
        self.overlay_text_edit.textChanged.connect(self._on_overlay_controls_changed)

        shape_group = QtWidgets.QGroupBox("Shape Overlays")
        shape_layout = QtWidgets.QVBoxLayout(shape_group)
        shape_form = QtWidgets.QVBoxLayout()
        shape_form.setSpacing(8)
        self.shape_type_combo = QtWidgets.QComboBox()
        self.shape_type_combo.addItems(SHAPE_TYPES)
        self.shape_type_combo.setFixedWidth(100)
        tip(self.shape_type_combo, "Shape type to add or apply to the selected shape overlay.")
        self.shape_start_spin = QtWidgets.QSpinBox()
        self.shape_end_spin = QtWidgets.QSpinBox()
        for spin in (self.shape_start_spin, self.shape_end_spin):
            spin.setRange(0, 0)
            spin.setFixedWidth(60)
        tip(self.shape_start_spin, "First output frame where the selected shape overlay is visible.")
        tip(self.shape_end_spin, "Last output frame where the selected shape overlay is visible.")

        shape_label_texts = ("Shape:", "Start:")
        shape_label_width = max(QtWidgets.QLabel(text).sizeHint().width() for text in shape_label_texts)

        def shape_label(text: str, alignment=QtCore.Qt.AlignLeft) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setFixedWidth(shape_label_width)
            label.setAlignment(alignment | QtCore.Qt.AlignVCenter)
            return label

        def shape_inline_label(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return label

        def shape_row(label_text: str, compact_label: bool = False) -> QtWidgets.QHBoxLayout:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            alignment = QtCore.Qt.AlignRight if compact_label else QtCore.Qt.AlignLeft
            row.addWidget(shape_label(label_text, alignment))
            return row

        shape_type_row = shape_row("Shape:")
        shape_type_row.addWidget(self.shape_type_combo)
        shape_type_row.addStretch(1)
        shape_form.addLayout(shape_type_row)

        shape_range_row = shape_row("Start:", compact_label=True)
        shape_range_row.addWidget(self.shape_start_spin)
        shape_range_row.addSpacing(8)
        shape_range_row.addWidget(shape_inline_label("End:"))
        shape_range_row.addWidget(self.shape_end_spin)
        shape_range_row.addStretch(1)
        shape_form.addLayout(shape_range_row)
        shape_layout.addLayout(shape_form)

        shape_buttons = QtWidgets.QHBoxLayout()
        add_shape_button = QtWidgets.QPushButton("Add Shape")
        delete_shape_button = QtWidgets.QPushButton("Delete Shape")
        for button in (add_shape_button, delete_shape_button):
            button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        add_shape_button.clicked.connect(self._add_shape_overlay)
        delete_shape_button.clicked.connect(self._delete_shape_overlay)
        tip(add_shape_button, "Add a shape overlay using the current type and output-frame range.")
        tip(delete_shape_button, "Delete the selected shape overlay.")
        shape_buttons.addWidget(add_shape_button, 0, QtCore.Qt.AlignLeft)
        shape_buttons.addWidget(delete_shape_button, 0, QtCore.Qt.AlignLeft)
        shape_buttons.addStretch(1)
        shape_layout.addLayout(shape_buttons)

        self.shape_table = QtWidgets.QTableWidget()
        self.shape_table.setColumnCount(5)
        self.shape_table.setHorizontalHeaderLabels(["Shape", "Start", "End", "Line", "Fill"])
        self.shape_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.shape_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.shape_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.shape_table.verticalHeader().setVisible(False)
        self.shape_table.verticalHeader().setDefaultSectionSize(24)
        self.shape_table.verticalHeader().setMinimumSectionSize(20)
        self.shape_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        shape_header = self.shape_table.horizontalHeader()
        shape_header.setStretchLastSection(False)
        for col in range(5):
            shape_header.setSectionResizeMode(col, QtWidgets.QHeaderView.Fixed)
        self.shape_table.setColumnWidth(0, 90)
        self.shape_table.setColumnWidth(1, 54)
        self.shape_table.setColumnWidth(2, 54)
        self.shape_table.setColumnWidth(3, 118)
        self.shape_table.setColumnWidth(4, 64)
        self.shape_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.shape_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.shape_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.shape_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.shape_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.shape_table.itemSelectionChanged.connect(self._on_shape_overlay_selection_changed)
        tip(
            self.shape_table,
            "Shape overlays drawn on top of the movie. Select a row to edit its type and range; drag or resize the shape in Preview.",
        )
        self._set_compact_table_height(self.shape_table, rows=1, max_rows=4)
        self._fit_table_width_to_columns(self.shape_table)
        shape_layout.addWidget(self.shape_table, 0, QtCore.Qt.AlignLeft)
        right_layout.addWidget(shape_group)

        self.shape_type_combo.currentTextChanged.connect(self._on_shape_overlay_controls_changed)
        self.shape_start_spin.valueChanged.connect(self._on_shape_overlay_controls_changed)
        self.shape_end_spin.valueChanged.connect(self._on_shape_overlay_controls_changed)

        export_group = QtWidgets.QGroupBox("Export")
        export_layout = QtWidgets.QVBoxLayout(export_group)
        export_layout.setSpacing(8)
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 240.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setValue(self._default_fps())
        self.fps_spin.valueChanged.connect(self._update_title_duration_seconds_label)
        tip(self.fps_spin, "Movie playback and export frame rate in frames per second.")
        self.width_spin = QtWidgets.QSpinBox()
        self.height_spin = QtWidgets.QSpinBox()
        for spin in (self.width_spin, self.height_spin):
            spin.setRange(2, 4096)
            spin.setSingleStep(2)
        default_w, default_h = self._default_output_size()
        self.width_spin.setValue(default_w)
        self.height_spin.setValue(default_h)
        tip(self.width_spin, "Exported movie width in pixels.")
        tip(self.height_spin, "Exported movie height in pixels.")
        self.quality_spin = QtWidgets.QSpinBox()
        self.quality_spin.setRange(50, 100)
        self.quality_spin.setValue(85)
        tip(self.quality_spin, "Export quality setting. Higher values preserve more detail and usually create larger files.")
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["mp4", "avi"])
        tip(self.format_combo, "Output movie file format.")
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(["1ch", "2ch"])
        if not self._has_2ch_data():
            self.channel_combo.setEnabled(False)
        tip(self.channel_combo, "AFM data channel to render. 2ch is available only when the loaded data contains a second channel.")
        self.display_source_label = QtWidgets.QLabel("Color/Tone: pyNuD main display")
        self.time_check = QtWidgets.QCheckBox("Time stamp")
        self.time_check.setChecked(self._main_checkbox_checked("time_caption_check", True))
        tip(self.time_check, "Overlay the pyNuD time stamp on exported movie frames.")
        self.frame_no_check = QtWidgets.QCheckBox("Frame number")
        self.frame_no_check.setChecked(False)
        tip(self.frame_no_check, "Overlay the output frame number on exported movie frames.")
        self.scale_bar_check = QtWidgets.QCheckBox("X scale bar")
        self.scale_bar_check.setChecked(self._main_checkbox_checked("scale_caption_check", True))
        tip(self.scale_bar_check, "Overlay the horizontal X scale bar using the pyNuD save/display scale-bar settings.")
        self.z_scale_bar_check = QtWidgets.QCheckBox("Color bar")
        self.z_scale_bar_check.setChecked(self._main_checkbox_checked("zscale_check", False))
        tip(self.z_scale_bar_check, "Overlay the height Color bar using the current pyNuD color and tone settings.")
        self.scale_bar_frames_spin = QtWidgets.QSpinBox()
        self.scale_bar_frames_spin.setRange(1, 1000000)
        self.scale_bar_frames_spin.setValue(10)
        self.scale_bar_frames_spin.setEnabled(self.scale_bar_check.isChecked())
        self.scale_bar_check.toggled.connect(self.scale_bar_frames_spin.setEnabled)
        tip(self.scale_bar_frames_spin, "Number of initial output frames where the X scale bar is shown.")
        self.movie_preview_play_button = QtWidgets.QPushButton("Play")
        self.movie_preview_stop_button = QtWidgets.QPushButton("Stop")
        self.movie_preview_stop_button.setEnabled(False)
        self.movie_preview_play_button.clicked.connect(self._start_movie_preview_playback)
        self.movie_preview_stop_button.clicked.connect(self._stop_movie_preview_playback)
        tip(self.movie_preview_play_button, "Play the preview from the current output frame using the selected FPS.")
        tip(self.movie_preview_stop_button, "Stop preview playback and keep the current output frame selected.")
        self.movie_preview_loop_check = QtWidgets.QCheckBox("Loop")
        self.movie_preview_loop_check.setChecked(False)
        tip(self.movie_preview_loop_check, "Repeat preview playback within the selected loop range.")
        self.loop_start_spin = QtWidgets.QSpinBox()
        self.loop_end_spin = QtWidgets.QSpinBox()
        for spin in (self.loop_start_spin, self.loop_end_spin):
            spin.setRange(0, 0)
            spin.setMinimumWidth(64)
            spin.valueChanged.connect(self._loop_range_changed)
        tip(self.loop_start_spin, "First output frame used when Loop preview playback is enabled.")
        tip(self.loop_end_spin, "Last output frame used when Loop preview playback is enabled.")
        self.export_button = QtWidgets.QPushButton("Export Movie")
        self.export_button.clicked.connect(self._export_movie)
        tip(self.export_button, "Render all output frames with the current overlays and save them as a movie file.")

        for spin in (
            self.fps_spin,
            self.width_spin,
            self.height_spin,
            self.quality_spin,
            self.scale_bar_frames_spin,
        ):
            spin.setFixedWidth(60)
        for spin in (self.loop_start_spin, self.loop_end_spin):
            spin.setFixedWidth(44)
        self.format_combo.setFixedWidth(80)
        self.channel_combo.setFixedWidth(80)
        for button in (
            self.movie_preview_play_button,
            self.movie_preview_stop_button,
            self.export_button,
        ):
            button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        def export_label(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            return label

        def export_inline_label(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return label

        def export_row(label_text: str) -> QtWidgets.QHBoxLayout:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(export_label(label_text))
            return row

        preview_row = export_row("Preview:")
        preview_row.addWidget(self.movie_preview_play_button)
        preview_row.addWidget(self.movie_preview_stop_button)
        preview_row.addWidget(self.movie_preview_loop_check)
        preview_row.addStretch(1)
        export_layout.addLayout(preview_row)

        loop_range_row = export_row("Loop range:")
        loop_range_row.addWidget(export_inline_label("Start:"))
        loop_range_row.addWidget(self.loop_start_spin)
        loop_range_row.addSpacing(8)
        loop_range_row.addWidget(export_inline_label("End:"))
        loop_range_row.addWidget(self.loop_end_spin)
        loop_range_row.addStretch(1)
        export_layout.addLayout(loop_range_row)

        fps_row = export_row("FPS:")
        fps_row.addWidget(self.fps_spin)
        fps_row.addSpacing(8)
        fps_row.addWidget(export_inline_label("Format:"))
        fps_row.addWidget(self.format_combo)
        fps_row.addStretch(1)
        export_layout.addLayout(fps_row)

        size_row = export_row("Width:")
        size_row.addWidget(self.width_spin)
        size_row.addSpacing(8)
        size_row.addWidget(export_inline_label("Height:"))
        size_row.addWidget(self.height_spin)
        size_row.addStretch(1)
        export_layout.addLayout(size_row)

        quality_row = export_row("Quality:")
        quality_row.addWidget(self.quality_spin)
        quality_row.addSpacing(8)
        quality_row.addWidget(export_inline_label("Channel:"))
        quality_row.addWidget(self.channel_combo)
        quality_row.addStretch(1)
        export_layout.addLayout(quality_row)

        export_layout.addWidget(self.display_source_label, 0, QtCore.Qt.AlignLeft)

        option_row = QtWidgets.QHBoxLayout()
        option_row.setContentsMargins(0, 0, 0, 0)
        option_row.setSpacing(16)
        option_row.addWidget(self.time_check)
        option_row.addWidget(self.frame_no_check)
        option_row.addStretch(1)
        export_layout.addLayout(option_row)

        scale_option_row = QtWidgets.QHBoxLayout()
        scale_option_row.setContentsMargins(0, 0, 0, 0)
        scale_option_row.setSpacing(16)
        scale_option_row.addWidget(self.scale_bar_check)
        scale_option_row.addWidget(self.z_scale_bar_check)
        scale_option_row.addStretch(1)
        export_layout.addLayout(scale_option_row)

        scale_bar_row = export_row("X scale bar frames:")
        scale_bar_row.addWidget(self.scale_bar_frames_spin)
        scale_bar_row.addStretch(1)
        export_layout.addLayout(scale_bar_row)

        export_button_row = QtWidgets.QHBoxLayout()
        export_button_row.setContentsMargins(0, 0, 0, 0)
        export_button_row.setSpacing(6)
        export_button_row.addWidget(self.export_button, 0, QtCore.Qt.AlignLeft)
        export_button_row.addStretch(1)
        export_layout.addLayout(export_button_row)
        right_layout.addWidget(export_group)

        session_group = QtWidgets.QGroupBox("Session")
        session_layout = QtWidgets.QHBoxLayout(session_group)
        session_layout.setContentsMargins(8, 8, 8, 8)
        session_layout.setSpacing(8)
        self.save_session_button = QtWidgets.QPushButton("Save Session")
        self.load_session_button = QtWidgets.QPushButton("Load Session")
        self.save_session_button.clicked.connect(self._save_session)
        self.load_session_button.clicked.connect(self._load_session)
        tip(self.save_session_button, "Save clips, overlays, transitions, preview/export settings, and optional source data to an .afmmoviesession file.")
        tip(self.load_session_button, "Load an .afmmoviesession file and restore the saved Movie Editor state.")
        for button in (self.save_session_button, self.load_session_button):
            button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        session_layout.addWidget(self.save_session_button, 0, QtCore.Qt.AlignLeft)
        session_layout.addWidget(self.load_session_button, 0, QtCore.Qt.AlignLeft)
        session_layout.addStretch(1)
        right_layout.addWidget(session_group)

        for widget in (
            self.fps_spin,
            self.width_spin,
            self.height_spin,
            self.channel_combo,
            self.time_check,
            self.frame_no_check,
            self.scale_bar_check,
            self.z_scale_bar_check,
            self.scale_bar_frames_spin,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._preview_settings_changed)
            elif hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._preview_settings_changed)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._preview_settings_changed)

        right_layout.addStretch(1)
        right.adjustSize()
        right.setMinimumSize(right.sizeHint())

    # ------------------------------------------------------------------
    # Source frames and thumbnails
    # ------------------------------------------------------------------

    def _detect_total_frames(self) -> int:
        if gv is None:
            return 0
        try:
            return max(0, int(getattr(gv, "FrameNum", 0) or 0))
        except Exception:
            return 0

    def _current_file_path(self) -> str:
        if gv is None:
            return ""
        try:
            files = getattr(gv, "files", None)
            idx = int(getattr(gv, "currentFileNum", 0) or 0)
            if files and 0 <= idx < len(files):
                return os.path.abspath(files[idx])
        except Exception:
            pass
        return ""

    def _has_2ch_data(self) -> bool:
        if gv is None:
            return False
        return bool(getattr(gv, "DataType2ch", 0) not in (0, None))

    def _main_checkbox_checked(self, attr_name: str, default: bool = False) -> bool:
        try:
            widget = getattr(self.main_window, attr_name, None)
            if widget is not None and hasattr(widget, "isChecked"):
                return bool(widget.isChecked())
        except Exception:
            pass
        return bool(default)

    def _init_frame_number_defaults(self) -> None:
        try:
            widget = getattr(self.main_window, "time_font_size_spin", None)
            if widget is not None and hasattr(widget, "value"):
                self._frame_num["font_size"] = int(widget.value())
        except Exception:
            pass
        try:
            if self.main_window is not None and hasattr(self.main_window, "_getSaveOverlayFontFamily"):
                family = self.main_window._getSaveOverlayFontFamily("time")
                if family:
                    self._frame_num["font_family"] = str(family)
        except Exception:
            pass
        if gv is not None:
            try:
                self._frame_num["font_style"] = str(getattr(gv, "timeFontStyle", "Normal") or "Normal")
            except Exception:
                pass
            try:
                bgr = getattr(gv, "timeFontColor", (255, 255, 255))
                self._frame_num["color"] = QtGui.QColor(int(bgr[2]), int(bgr[1]), int(bgr[0]))
            except Exception:
                pass

    def _set_compact_table_height(self, table: QtWidgets.QTableWidget, rows: int, max_rows: int = 5) -> None:
        header = table.horizontalHeader()
        vertical_header = table.verticalHeader()
        header_h = header.height() if header is not None else 24
        if header is not None:
            header_h = max(header_h, header.sizeHint().height())
        row_h = vertical_header.defaultSectionSize() if vertical_header is not None else 24
        row_h = max(24, int(row_h))
        row_count = max(1, int(rows))
        visible_rows = max(1, min(max_rows, row_count))
        for row in range(table.rowCount()):
            table.setRowHeight(row, row_h)
        frame_extra = 2 * table.frameWidth() + 14
        height = int(header_h + visible_rows * row_h + frame_extra)
        table.setMinimumHeight(height)
        table.setMaximumHeight(height)
        table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn if row_count > max_rows else QtCore.Qt.ScrollBarAsNeeded)
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.verticalScrollBar().setSingleStep(max(1, row_h // 2))
        table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def _fit_table_width_to_columns(self, table: QtWidgets.QTableWidget) -> None:
        header = table.verticalHeader()
        width = 2 * table.frameWidth() + 4
        if header is not None and header.isVisible():
            width += header.width()
        for col in range(table.columnCount()):
            width += table.columnWidth(col)
        if table.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOn:
            width += table.verticalScrollBar().sizeHint().width()
        table.setMinimumWidth(width)
        table.setMaximumWidth(width)

    def _update_right_panel_size(self) -> None:
        panel = getattr(self, "right_panel", None)
        if panel is None:
            return
        panel.updateGeometry()
        hint = panel.sizeHint()
        if hint.isValid():
            panel.setMinimumSize(hint)
            panel.resize(hint)
        scroll = getattr(self, "right_scroll", None)
        if scroll is not None:
            scroll.updateGeometry()

    def _settings_bytearray(self, key: str) -> Optional[QtCore.QByteArray]:
        try:
            value = self._settings.value(key)
            if isinstance(value, QtCore.QByteArray):
                return value
            if isinstance(value, bytes):
                return QtCore.QByteArray(value)
            if isinstance(value, bytearray):
                return QtCore.QByteArray(bytes(value))
        except Exception:
            return None
        return None

    def _restore_ui_settings(self) -> None:
        geometry = self._settings_bytearray("windowGeometry")
        if geometry is not None and not geometry.isEmpty():
            try:
                self.restoreGeometry(geometry)
            except Exception:
                pass

        splitter_state = self._settings_bytearray("mainSplitterState")
        if splitter_state is not None and not splitter_state.isEmpty():
            try:
                self.main_splitter.restoreState(splitter_state)
            except Exception:
                pass

        try:
            thumb_size = self._settings.value("sourceThumbSize", self._source_thumb_size, type=int)
            self._apply_source_thumbnail_size(int(thumb_size))
        except Exception:
            pass
        try:
            trimmed_mode = str(self._settings.value("trimmedOutputMode", TRIMMED_OUTPUT_MODES[0]) or TRIMMED_OUTPUT_MODES[0])
            if trimmed_mode in TRIMMED_OUTPUT_MODES:
                self.trimmed_output_combo.setCurrentText(trimmed_mode)
        except Exception:
            pass

    def _save_ui_settings(self) -> None:
        try:
            self._settings.setValue("windowGeometry", self.saveGeometry())
            self._settings.setValue("mainSplitterState", self.main_splitter.saveState())
            self._settings.setValue("sourceThumbSize", int(self._source_thumb_size))
            self._settings.setValue("trimmedOutputMode", self.trimmed_output_combo.currentText())
            self._settings.sync()
        except Exception as exc:
            print(f"[WARNING] Failed to save AFM Movie Editor settings: {exc}")

    def _session_filter(self) -> str:
        return "AFM Movie Session (*.afmmoviesession);;All Files (*)"

    def _current_data_file_path(self) -> str:
        try:
            files = getattr(gv, "files", None) if gv is not None else None
            file_idx = int(getattr(gv, "currentFileNum", 0) or 0) if gv is not None else 0
            if files and 0 <= file_idx < len(files):
                return str(files[file_idx])
        except Exception:
            pass
        return ""

    def _default_session_path(self) -> str:
        source_path = self._current_data_file_path()
        if source_path:
            folder = os.path.dirname(source_path)
            base = os.path.splitext(os.path.basename(source_path))[0]
        else:
            folder = getattr(gv, "movieSaveDir", "") if gv is not None else ""
            if not folder or not os.path.isdir(folder):
                folder = os.path.expanduser("~")
            base = "afm_movie"
        return os.path.join(folder, f"{base}.afmmoviesession")

    def _to_session_value(self, value: Any) -> Any:
        if isinstance(value, QtGui.QColor):
            return {"__qcolor__": [int(value.red()), int(value.green()), int(value.blue()), int(value.alpha())]}
        if isinstance(value, QtCore.QSize):
            return {"__qsize__": [int(value.width()), int(value.height())]}
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
        if isinstance(value, tuple):
            return [self._to_session_value(v) for v in value]
        if isinstance(value, list):
            return [self._to_session_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._to_session_value(v) for k, v in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _from_session_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if "__qcolor__" in value:
                rgba = list(value.get("__qcolor__", [255, 255, 255, 255]))
                while len(rgba) < 4:
                    rgba.append(255)
                return QtGui.QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))
            if "__qsize__" in value:
                size = list(value.get("__qsize__", [0, 0]))
                return QtCore.QSize(int(size[0]), int(size[1]))
            if "__ndarray__" in value:
                data = value.get("__ndarray__", [])
                dtype = value.get("dtype", None)
                try:
                    return np.asarray(data, dtype=dtype)
                except Exception:
                    return np.asarray(data)
            return {str(k): self._from_session_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._from_session_value(v) for v in value]
        return value

    def _dataclass_to_session(self, obj: Any) -> Dict[str, Any]:
        return {field_info.name: self._to_session_value(getattr(obj, field_info.name)) for field_info in fields(obj)}

    def _dataclass_from_session(self, cls, payload: Any):
        if not isinstance(payload, dict):
            return None
        valid_names = {field_info.name for field_info in fields(cls)}
        kwargs = {
            key: self._from_session_value(value)
            for key, value in payload.items()
            if key in valid_names
        }
        try:
            return cls(**kwargs)
        except Exception as exc:
            print(f"[WARNING] Failed to restore {cls.__name__}: {exc}")
            return None

    def _session_gv_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if gv is None:
            return state
        for key in SESSION_GV_KEYS:
            if hasattr(gv, key):
                state[key] = self._to_session_value(getattr(gv, key))
        try:
            combo = getattr(self.main_window, "zscale_orient_combo", None)
            if combo is not None and hasattr(combo, "currentText"):
                state["zScaleOrientation"] = str(combo.currentText())
        except Exception:
            pass
        return state

    def _restore_session_gv_state(self, state: Any) -> None:
        if gv is None or not isinstance(state, dict):
            return
        for key, value in state.items():
            if key == "zScaleOrientation":
                continue
            try:
                setattr(gv, key, self._from_session_value(value))
            except Exception:
                pass
        try:
            if "zScaleOrientation" in state:
                combo = getattr(self.main_window, "zscale_orient_combo", None)
                if combo is not None and hasattr(combo, "setCurrentText"):
                    combo.setCurrentText(str(state["zScaleOrientation"]))
                gv.zScaleOrientation = str(state["zScaleOrientation"])
        except Exception:
            pass
        for cache_name in ("timeFont", "scaleFont", "tempFont", "filenameFont"):
            try:
                if hasattr(gv, cache_name):
                    delattr(gv, cache_name)
            except Exception:
                pass

    def _build_session_payload(self) -> Dict[str, Any]:
        source_path = self._current_data_file_path()
        export_state = {
            "fps": float(self.fps_spin.value()),
            "width": int(self.width_spin.value()),
            "height": int(self.height_spin.value()),
            "quality": int(self.quality_spin.value()),
            "format": str(self.format_combo.currentText()),
            "channel": str(self.channel_combo.currentText()),
            "time_stamp": bool(self.time_check.isChecked()),
            "frame_number": bool(self.frame_no_check.isChecked()),
            "x_scale_bar": bool(self.scale_bar_check.isChecked()),
            "color_bar": bool(self.z_scale_bar_check.isChecked()),
            "x_scale_bar_frames": int(self.scale_bar_frames_spin.value()),
            "preview_loop": bool(self.movie_preview_loop_check.isChecked()),
            "loop_start": int(self.loop_start_spin.value()),
            "loop_end": int(self.loop_end_spin.value()),
        }
        title_defaults = {
            "text": self._title_text_value(),
            "duration": int(self.title_duration_spin.value()),
            "background_color": self._to_session_value(self.title_bg_color),
            "use_first_frame_background": bool(self.title_use_first_frame_bg_check.isChecked()),
            "background_frame_opacity": int(self.title_bg_opacity_spin.value()),
        }
        overlay_defaults = {
            "text": self._overlay_text_value() if hasattr(self, "overlay_text_edit") else "",
            "start": int(self.overlay_start_spin.value()) if hasattr(self, "overlay_start_spin") else 0,
            "end": int(self.overlay_end_spin.value()) if hasattr(self, "overlay_end_spin") else 0,
            "position": str(self.overlay_pos_combo.currentText()) if hasattr(self, "overlay_pos_combo") else "Bottom",
            "font_size": int(self.overlay_size_spin.value()) if hasattr(self, "overlay_size_spin") else 28,
            "background": bool(self.overlay_bg_check.isChecked()) if hasattr(self, "overlay_bg_check") else True,
            "color": self._to_session_value(getattr(self, "overlay_color", QtGui.QColor(255, 255, 255))),
        }
        shape_defaults = {
            "shape": str(self.shape_type_combo.currentText()) if hasattr(self, "shape_type_combo") else "Rectangle",
            "start": int(self.shape_start_spin.value()) if hasattr(self, "shape_start_spin") else 0,
            "end": int(self.shape_end_spin.value()) if hasattr(self, "shape_end_spin") else 0,
        }
        transition_defaults = {
            "effect": str(self.transition_out_combo.currentText()),
            "frames": int(self.transition_out_frames_spin.value()),
            "black_frames": int(self.transition_out_hold_frames_spin.value()),
        }
        return {
            "schema": "pyNuD.afm_movie_editor.session",
            "version": 1,
            "plugin": PLUGIN_NAME,
            "source": {
                "original_path": source_path,
                "basename": os.path.basename(source_path) if source_path else "",
                "embedded_member": f"data/{os.path.basename(source_path)}" if source_path and os.path.isfile(source_path) else "",
                "total_frames": int(self.total_frames),
            },
            "editor": {
                "clips": [self._dataclass_to_session(clip) for clip in self.clips],
                "text_overlays": [self._dataclass_to_session(overlay) for overlay in self.overlays],
                "shape_overlays": [self._dataclass_to_session(overlay) for overlay in self.shape_overlays],
                "frame_number_style": self._to_session_value(self._frame_num),
                "color_bar_text_style": self._to_session_value(self._color_bar_text_style),
                "color_bar_ref_size": self._to_session_value(self._color_bar_ref_size),
                "source_thumbnail_size": int(self._source_thumb_size),
                "trimmed_output_mode": str(self.trimmed_output_combo.currentText()),
                "last_split_output_index": self._last_split_output_index,
                "preview_output_index": int(self.preview_slider.value()) if hasattr(self, "preview_slider") else int(self._last_preview_index),
                "title_defaults": title_defaults,
                "overlay_defaults": overlay_defaults,
                "shape_defaults": shape_defaults,
                "transition_defaults": transition_defaults,
                "export": export_state,
                "gv_state": self._session_gv_state(),
            },
        }

    def _write_session_archive(self, session_path: str, payload: Dict[str, Any]) -> bool:
        source_path = str(payload.get("source", {}).get("original_path", "") or "")
        embedded_member = str(payload.get("source", {}).get("embedded_member", "") or "")
        try:
            with zipfile.ZipFile(session_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                zf.writestr("session.json", json.dumps(payload, ensure_ascii=False, indent=2))
                if embedded_member and source_path and os.path.isfile(source_path):
                    zf.write(source_path, embedded_member)
            return True
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Session", f"Failed to save session:\n{exc}")
            return False

    def _save_session(self) -> None:
        self._stop_movie_preview_playback(update_status=False)
        default_path = self._default_session_path()
        session_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save AFM Movie Session",
            default_path,
            self._session_filter(),
        )
        if not session_path:
            return
        if not session_path.lower().endswith(".afmmoviesession"):
            session_path += ".afmmoviesession"
        payload = self._build_session_payload()
        source_path = str(payload.get("source", {}).get("original_path", "") or "")
        if not source_path or not os.path.isfile(source_path):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Save Session",
                "No source AFM data file was found. Save editor settings only?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        if not self._write_session_archive(session_path, payload):
            return
        QtWidgets.QMessageBox.information(
            self,
            "Save Session",
            f"Session saved:\n{os.path.basename(session_path)}",
        )

    def _read_session_archive(self, session_path: str) -> Dict[str, Any]:
        if zipfile.is_zipfile(session_path):
            with zipfile.ZipFile(session_path, "r") as zf:
                with zf.open("session.json", "r") as f:
                    return json.loads(f.read().decode("utf-8"))
        with open(session_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _session_cache_dir(self, session_path: str) -> str:
        try:
            stat = os.stat(session_path)
            signature = f"{os.path.abspath(session_path)}:{stat.st_mtime_ns}:{stat.st_size}"
        except Exception:
            signature = os.path.abspath(session_path)
        digest = hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return os.path.join(tempfile.gettempdir(), "pynud_afm_movie_sessions", digest)

    def _extract_session_data_file(self, session_path: str, payload: Dict[str, Any]) -> str:
        source = payload.get("source", {}) if isinstance(payload, dict) else {}
        embedded_member = str(source.get("embedded_member", "") or "")
        if embedded_member and zipfile.is_zipfile(session_path):
            member_name = embedded_member.replace("\\", "/")
            if not member_name.startswith("data/") or member_name.endswith("/"):
                return ""
            basename = os.path.basename(member_name)
            if not basename:
                return ""
            cache_dir = self._session_cache_dir(session_path)
            os.makedirs(cache_dir, exist_ok=True)
            out_path = os.path.join(cache_dir, basename)
            with zipfile.ZipFile(session_path, "r") as zf:
                with zf.open(member_name, "r") as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            return out_path
        original_path = str(source.get("original_path", "") or "")
        if original_path and os.path.isfile(original_path):
            return original_path
        return ""

    def _load_data_file_for_session(self, data_path: str) -> bool:
        if not data_path or not os.path.isfile(data_path):
            return False
        try:
            if self.main_window is not None and hasattr(self.main_window, "loadFileSystemPaths"):
                loaded = bool(self.main_window.loadFileSystemPaths([data_path]))
                if loaded:
                    file_list = getattr(self.main_window, "FileList", None)
                    if file_list is not None and hasattr(file_list, "setCurrentRow"):
                        try:
                            file_list.setCurrentRow(0)
                        except Exception:
                            pass
                    if hasattr(self.main_window, "ListClickFunction"):
                        try:
                            self.main_window.ListClickFunction(bring_to_front=False)
                        except TypeError:
                            self.main_window.ListClickFunction()
                    return True
        except Exception as exc:
            print(f"[WARNING] Failed to load session data through pyNuD: {exc}")
        try:
            from fileio import LoadASD

            if gv is not None:
                gv.files = [data_path]
                gv.currentFileNum = 0
                gv.n_items = 1
            LoadASD(data_path)
            return True
        except Exception as exc:
            print(f"[WARNING] Failed to load session data directly: {exc}")
            return False

    def _clamp_loaded_clips(self) -> None:
        if self.total_frames <= 0:
            return
        last_frame = self.total_frames - 1
        for clip in self.clips:
            if clip.kind != "source":
                continue
            clip.start = max(0, min(last_frame, int(getattr(clip, "start", 0))))
            clip.end = max(0, min(last_frame, int(getattr(clip, "end", clip.start))))
            if clip.start > clip.end:
                clip.start, clip.end = clip.end, clip.start

    def _restore_session_payload(self, payload: Dict[str, Any], session_path: str) -> None:
        if not isinstance(payload, dict) or payload.get("schema") != "pyNuD.afm_movie_editor.session":
            raise ValueError("This is not an AFM Movie Editor session file.")
        self._stop_movie_preview_playback(update_status=False)

        data_path = self._extract_session_data_file(session_path, payload)
        data_loaded = self._load_data_file_for_session(data_path) if data_path else False
        self.total_frames = self._detect_total_frames()
        if self.total_frames > 0:
            self._refresh_source_frames(reset_clips=True)

        editor = payload.get("editor", {})
        if not isinstance(editor, dict):
            raise ValueError("Session file is missing editor settings.")
        self._restore_session_gv_state(editor.get("gv_state", {}))

        clips = [self._dataclass_from_session(ClipSpec, item) for item in editor.get("clips", [])]
        overlays = [self._dataclass_from_session(TextOverlaySpec, item) for item in editor.get("text_overlays", [])]
        shapes = [self._dataclass_from_session(ShapeOverlaySpec, item) for item in editor.get("shape_overlays", [])]
        self.clips = [clip for clip in clips if clip is not None]
        self.overlays = [overlay for overlay in overlays if overlay is not None]
        self.shape_overlays = [shape for shape in shapes if shape is not None]
        if not self.clips and self.total_frames > 0:
            self.clips = [ClipSpec(kind="source", start=0, end=self.total_frames - 1)]
        self._clamp_loaded_clips()

        frame_num = self._from_session_value(editor.get("frame_number_style", {}))
        if isinstance(frame_num, dict):
            self._frame_num.update(frame_num)
        color_bar_style = self._from_session_value(editor.get("color_bar_text_style", {}))
        if isinstance(color_bar_style, dict):
            self._color_bar_text_style.update(color_bar_style)
        color_bar_ref_size = self._from_session_value(editor.get("color_bar_ref_size", {}))
        if isinstance(color_bar_ref_size, dict):
            self._color_bar_ref_size.update(color_bar_ref_size)
        try:
            self._source_thumb_size = max(self._source_thumb_min, min(self._source_thumb_max, int(editor.get("source_thumbnail_size", self._source_thumb_size))))
            self._apply_source_thumbnail_size(self._source_thumb_size)
        except Exception:
            pass
        trimmed_mode = str(editor.get("trimmed_output_mode", TRIMMED_OUTPUT_MODES[0]) or TRIMMED_OUTPUT_MODES[0])
        if trimmed_mode in TRIMMED_OUTPUT_MODES:
            self.trimmed_output_combo.setCurrentText(trimmed_mode)
        self._last_split_output_index = editor.get("last_split_output_index", None)

        export = editor.get("export", {})
        if isinstance(export, dict):
            self.fps_spin.setValue(float(export.get("fps", self.fps_spin.value())))
            self.width_spin.setValue(int(export.get("width", self.width_spin.value())))
            self.height_spin.setValue(int(export.get("height", self.height_spin.value())))
            self.quality_spin.setValue(int(export.get("quality", self.quality_spin.value())))
            if str(export.get("format", "")) in [self.format_combo.itemText(i) for i in range(self.format_combo.count())]:
                self.format_combo.setCurrentText(str(export.get("format")))
            if str(export.get("channel", "")) in [self.channel_combo.itemText(i) for i in range(self.channel_combo.count())]:
                self.channel_combo.setCurrentText(str(export.get("channel")))
            self.time_check.setChecked(bool(export.get("time_stamp", self.time_check.isChecked())))
            self.frame_no_check.setChecked(bool(export.get("frame_number", self.frame_no_check.isChecked())))
            self.scale_bar_check.setChecked(bool(export.get("x_scale_bar", self.scale_bar_check.isChecked())))
            self.z_scale_bar_check.setChecked(bool(export.get("color_bar", self.z_scale_bar_check.isChecked())))
            self.scale_bar_frames_spin.setValue(max(0, int(export.get("x_scale_bar_frames", self.scale_bar_frames_spin.value()))))
            self.movie_preview_loop_check.setChecked(bool(export.get("preview_loop", self.movie_preview_loop_check.isChecked())))

        transition_defaults = editor.get("transition_defaults", {})
        if isinstance(transition_defaults, dict):
            effect = str(transition_defaults.get("effect", self.transition_out_combo.currentText()))
            if effect in TRANSITIONS:
                self.transition_out_combo.setCurrentText(effect)
            self.transition_out_frames_spin.setValue(max(0, int(transition_defaults.get("frames", self.transition_out_frames_spin.value()))))
            self.transition_out_hold_frames_spin.setValue(max(0, int(transition_defaults.get("black_frames", self.transition_out_hold_frames_spin.value()))))

        title_defaults = editor.get("title_defaults", {})
        if isinstance(title_defaults, dict):
            self._title_ui_updating = True
            try:
                self.title_text_edit.setPlainText(str(title_defaults.get("text", "")))
                self.title_duration_spin.setValue(max(1, int(title_defaults.get("duration", self.title_duration_spin.value()))))
                color = self._from_session_value(title_defaults.get("background_color", self._to_session_value(self.title_bg_color)))
                if isinstance(color, QtGui.QColor):
                    self.title_bg_color = color
                self.title_use_first_frame_bg_check.setChecked(bool(title_defaults.get("use_first_frame_background", self.title_use_first_frame_bg_check.isChecked())))
                self.title_bg_opacity_spin.setValue(max(0, min(100, int(title_defaults.get("background_frame_opacity", self.title_bg_opacity_spin.value())))))
                self._set_title_color_buttons()
                self._update_title_duration_seconds_label()
            finally:
                self._title_ui_updating = False

        overlay_defaults = editor.get("overlay_defaults", {})
        if isinstance(overlay_defaults, dict):
            self._overlay_ui_updating = True
            try:
                self._set_overlay_text_value(str(overlay_defaults.get("text", "")))
                self.overlay_start_spin.setValue(max(0, int(overlay_defaults.get("start", self.overlay_start_spin.value()))))
                self.overlay_end_spin.setValue(max(0, int(overlay_defaults.get("end", self.overlay_end_spin.value()))))
                position = str(overlay_defaults.get("position", self.overlay_pos_combo.currentText()))
                if position in POSITIONS:
                    self.overlay_pos_combo.setCurrentText(position)
                self.overlay_size_spin.setValue(max(8, min(160, int(overlay_defaults.get("font_size", self.overlay_size_spin.value())))))
                self.overlay_bg_check.setChecked(bool(overlay_defaults.get("background", self.overlay_bg_check.isChecked())))
                color = self._from_session_value(overlay_defaults.get("color", self._to_session_value(self.overlay_color)))
                if isinstance(color, QtGui.QColor):
                    self.overlay_color = color
                    self.color_button.setStyleSheet(self._button_color_stylesheet(self.overlay_color))
            finally:
                self._overlay_ui_updating = False

        shape_defaults = editor.get("shape_defaults", {})
        if isinstance(shape_defaults, dict):
            self._shape_ui_updating = True
            try:
                shape = str(shape_defaults.get("shape", self.shape_type_combo.currentText()))
                if shape in SHAPE_TYPES:
                    self.shape_type_combo.setCurrentText(shape)
                self.shape_start_spin.setValue(max(0, int(shape_defaults.get("start", self.shape_start_spin.value()))))
                self.shape_end_spin.setValue(max(0, int(shape_defaults.get("end", self.shape_end_spin.value()))))
            finally:
                self._shape_ui_updating = False

        self._plugin_colorbar_widget_cache = {}
        self._plugin_colorbar_render_rect = None
        self._plugin_colorbar_render_frame_size = None
        self._plugin_colorbar_render_orientation = None
        self._refresh_clip_table(select_row=0 if self.clips else None)
        self._refresh_overlay_table()
        self._refresh_shape_overlay_table()
        self._invalidate_timeline()
        self._update_preview_range()
        if isinstance(export, dict) and len(self._get_timeline()) > 0:
            self._set_loop_range_values(
                max(0, int(export.get("loop_start", 1)) - 1),
                max(0, int(export.get("loop_end", len(self._get_timeline()))) - 1),
            )
        duration = len(self._get_timeline())
        if duration > 0:
            preview_index = max(0, min(duration - 1, int(editor.get("preview_output_index", 0))))
            self.preview_slider.setValue(preview_index)
        self._refresh_source_frame_visuals()
        self._rebuild_source_thumbnails()
        self._update_preview()
        if not data_loaded and data_path:
            QtWidgets.QMessageBox.warning(self, "Load Session", "Session settings were loaded, but the AFM data file could not be loaded.")

    def _load_session(self) -> None:
        self._stop_movie_preview_playback(update_status=False)
        session_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load AFM Movie Session",
            os.path.dirname(self._default_session_path()),
            self._session_filter(),
        )
        if not session_path:
            return
        try:
            payload = self._read_session_archive(session_path)
            self._restore_session_payload(payload, session_path)
        except Exception as exc:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Load Session", f"Failed to load session:\n{exc}")

    def _source_split_boundaries(self) -> set:
        boundaries = set()
        for left, right in zip(self.clips, self.clips[1:]):
            if bool(getattr(left, "trimmed", False)) or bool(getattr(right, "trimmed", False)):
                continue
            if left.kind != "source" or right.kind != "source":
                continue
            if int(left.end) + 1 == int(right.start):
                boundaries.add(int(left.end))
        return boundaries

    def _source_split_start_frames(self) -> set:
        starts = set()
        for left, right in zip(self.clips, self.clips[1:]):
            if bool(getattr(left, "trimmed", False)) or bool(getattr(right, "trimmed", False)):
                continue
            if left.kind != "source" or right.kind != "source":
                continue
            if int(left.end) + 1 == int(right.start):
                starts.add(int(right.start))
        return starts

    def _source_frame_active(self, frame_index: int) -> bool:
        frame_index = int(frame_index)
        for clip in self.clips:
            if bool(getattr(clip, "trimmed", False)):
                continue
            if clip.kind != "source":
                continue
            start = max(0, min(self.total_frames - 1, int(clip.start)))
            end = max(0, min(self.total_frames - 1, int(clip.end)))
            if start > end:
                start, end = end, start
            if start <= frame_index <= end:
                return True
        return False

    def _source_separator_color(self, frame_index: int) -> QtGui.QColor:
        if int(frame_index) in self._source_split_boundaries():
            return QtGui.QColor(220, 0, 0)
        return QtGui.QColor(0, 0, 0)

    def _source_left_separator_color(self, frame_index: int) -> QtGui.QColor:
        if int(frame_index) in self._source_split_start_frames():
            return QtGui.QColor(220, 0, 0)
        return QtGui.QColor(0, 0, 0)

    def _style_source_frame_item(self, item: QtWidgets.QListWidgetItem, frame_index: int) -> None:
        if self._source_frame_active(frame_index):
            item.setBackground(QtGui.QBrush())
            item.setForeground(QtGui.QBrush())
            return
        item.setBackground(QtGui.QBrush(QtGui.QColor(205, 205, 205)))
        item.setForeground(QtGui.QBrush(QtGui.QColor(90, 90, 90)))

    def _source_icon_from_base(self, frame_index: int, base_pix: QtGui.QPixmap) -> QtGui.QIcon:
        size = max(self._source_thumb_min, min(self._source_thumb_max, int(self._source_thumb_size)))
        source = base_pix.scaled(size, size, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        canvas = QtGui.QPixmap(size + 2, size)
        canvas.fill(QtGui.QColor(0, 0, 0))
        painter = QtGui.QPainter(canvas)
        painter.drawPixmap(1, 0, source)
        if not self._source_frame_active(frame_index):
            painter.fillRect(1, 0, size, size, QtGui.QColor(190, 190, 190, 150))
        painter.fillRect(0, 0, 1, size, self._source_left_separator_color(frame_index))
        painter.fillRect(size + 1, 0, 1, size, self._source_separator_color(frame_index))
        painter.end()
        return QtGui.QIcon(canvas)

    def _refresh_source_frame_visuals(self) -> None:
        if not hasattr(self, "source_list"):
            return
        icon_size = self.source_list.iconSize()
        for row in range(self.source_list.count()):
            item = self.source_list.item(row)
            if item is None:
                continue
            frame_index = int(item.data(QtCore.Qt.UserRole))
            self._style_source_frame_item(item, frame_index)
            base_pix = item.data(QtCore.Qt.UserRole + 1)
            if isinstance(base_pix, QtGui.QPixmap) and not base_pix.isNull():
                item.setIcon(self._source_icon_from_base(frame_index, base_pix))
                continue
            icon = item.icon()
            if icon.isNull():
                continue
            pix = icon.pixmap(icon.actualSize(icon_size))
            if pix.isNull():
                pix = icon.pixmap(icon_size)
            if pix.isNull():
                continue
            pix = pix.copy()
            painter = QtGui.QPainter(pix)
            if not self._source_frame_active(frame_index):
                painter.fillRect(0, 0, max(0, pix.width() - 1), pix.height(), QtGui.QColor(190, 190, 190, 150))
            painter.fillRect(0, 0, 1, pix.height(), self._source_left_separator_color(frame_index))
            painter.fillRect(pix.width() - 1, 0, 1, pix.height(), self._source_separator_color(frame_index))
            painter.end()
            item.setIcon(QtGui.QIcon(pix))
        self.source_list.viewport().update()

    def _apply_source_thumbnail_size(self, requested_size: int) -> None:
        size = max(self._source_thumb_min, min(self._source_thumb_max, int(requested_size)))
        self._source_thumb_size = size
        separator_w = 2
        label_h = 20
        icon_w = size + separator_w
        grid_w = icon_w
        grid_h = max(size + label_h + 4, 42)
        self.source_list.setProperty("thumbnailImageSize", size)
        self.source_list.setIconSize(QtCore.QSize(icon_w, size))
        self.source_list.setGridSize(QtCore.QSize(grid_w, grid_h))
        self.source_list.setSpacing(0)
        self._rescale_existing_source_icons()
        self.source_list.doItemsLayout()
        self.source_list.viewport().update()
        if hasattr(self, "thumb_status_label"):
            self.thumb_status_label.setText(f"{self.total_frames} frames  |  thumbnail {size}px")

    def _rescale_existing_source_icons(self) -> None:
        if not hasattr(self, "source_list"):
            return
        size = max(self._source_thumb_min, min(self._source_thumb_max, int(self._source_thumb_size)))
        icon_size = QtCore.QSize(size + 2, size)
        for row in range(self.source_list.count()):
            item = self.source_list.item(row)
            if item is None:
                continue
            frame_index = int(item.data(QtCore.Qt.UserRole))
            base_pix = item.data(QtCore.Qt.UserRole + 1)
            if isinstance(base_pix, QtGui.QPixmap) and not base_pix.isNull():
                source = base_pix.scaled(
                    size,
                    size,
                    QtCore.Qt.IgnoreAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            else:
                icon = item.icon()
                if icon.isNull():
                    self._style_source_frame_item(item, frame_index)
                    continue
                pix = icon.pixmap(icon.actualSize(icon_size))
                if pix.isNull():
                    pix = icon.pixmap(icon_size)
                if pix.isNull():
                    self._style_source_frame_item(item, frame_index)
                    continue
                source_w = max(1, pix.width() - 2)
                source = pix.copy(1 if pix.width() > 2 else 0, 0, source_w, pix.height()).scaled(
                    size,
                    size,
                    QtCore.Qt.IgnoreAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            item.setIcon(self._source_icon_from_base(frame_index, source))
            self._style_source_frame_item(item, frame_index)

    def _rebuild_source_thumbnails(self) -> None:
        if not hasattr(self, "source_list") or self.total_frames <= 0:
            return
        self._stop_thumbnail_timer()
        self._start_thumbnail_timer()

    def _finish_source_thumbnail_resize(self) -> None:
        if hasattr(self, "thumb_status_label"):
            self.thumb_status_label.setText(f"{self.total_frames} frames  |  thumbnail {self._source_thumb_size}px")

    def _update_split_status(self, message: Optional[str] = None) -> None:
        if not hasattr(self, "split_status_label"):
            return
        if message:
            self.split_status_label.setText(message)
            return
        if self._last_split_output_index is not None:
            self.split_status_label.setText(f"Last split: output frame {int(self._last_split_output_index) + 1}")
            return
        self.split_status_label.setText("Right-click slider / right-double-click source frame to split")

    def _show_trimmed_output(self) -> bool:
        if not hasattr(self, "trimmed_output_combo"):
            return False
        return self.trimmed_output_combo.currentText() == "Show Trimmed"

    def _trimmed_output_mode_changed(self, *_args) -> None:
        self._invalidate_timeline()
        self._update_preview_range()
        self._update_preview()

    def _source_frame_from_timeline_ref(self, ref: Dict[str, Any]) -> Optional[int]:
        try:
            if ref["kind"] == "clip":
                clip = self.clips[int(ref["clip"])]
                if clip.kind != "source":
                    return None
                return max(0, min(self.total_frames - 1, clip.start + int(ref["local"])))

            for clip_key, local_key in (("b_clip", "b_local"), ("a_clip", "a_local")):
                clip_idx = int(ref.get(clip_key, -1))
                if not (0 <= clip_idx < len(self.clips)):
                    continue
                clip = self.clips[clip_idx]
                if clip.kind == "source":
                    local = int(ref.get(local_key, 0))
                    return max(0, min(self.total_frames - 1, clip.start + local))
        except Exception:
            return None
        return None

    def _timeline_ref_source_frames(self, ref: Dict[str, Any]) -> List[int]:
        frames: List[int] = []
        try:
            if ref["kind"] == "clip":
                frame = self._source_frame_from_timeline_ref(ref)
                return [] if frame is None else [frame]
            for clip_key, local_key in (("a_clip", "a_local"), ("b_clip", "b_local")):
                clip_idx = int(ref.get(clip_key, -1))
                if not (0 <= clip_idx < len(self.clips)):
                    continue
                clip = self.clips[clip_idx]
                if clip.kind != "source":
                    continue
                local = int(ref.get(local_key, 0))
                frame = max(0, min(self.total_frames - 1, int(clip.start) + local))
                frames.append(frame)
        except Exception:
            return []
        return frames

    def _output_index_for_source_frame(self, frame_index: int) -> Optional[int]:
        timeline = self._get_timeline()
        if not timeline:
            return None
        frame_index = int(frame_index)
        matches = [
            output_index
            for output_index, ref in enumerate(timeline)
            if frame_index in self._timeline_ref_source_frames(ref)
        ]
        if not matches:
            return None
        current = max(0, min(len(timeline) - 1, int(self.preview_slider.value())))
        return min(matches, key=lambda output_index: abs(output_index - current))

    def _timeline_ref_contains_clip(self, ref: Dict[str, Any], clip_idx: int) -> bool:
        try:
            if ref.get("kind") == "clip":
                return int(ref.get("clip", -1)) == int(clip_idx)
            return int(ref.get("a_clip", -1)) == int(clip_idx) or int(ref.get("b_clip", -1)) == int(clip_idx)
        except Exception:
            return False

    def _output_index_for_clip_start(self, clip_idx: int) -> Optional[int]:
        timeline = self._get_timeline()
        if not timeline:
            return None
        clip_idx = int(clip_idx)
        for output_index, ref in enumerate(timeline):
            if self._timeline_ref_contains_clip(ref, clip_idx):
                return output_index
        return None

    def _output_index_for_clip_local(self, clip_idx: int, local: int) -> Optional[int]:
        timeline = self._get_timeline()
        if not timeline:
            return None
        clip_idx = int(clip_idx)
        local = int(local)
        matches: List[int] = []
        for output_index, ref in enumerate(timeline):
            try:
                if ref.get("kind") == "clip":
                    if int(ref.get("clip", -1)) == clip_idx and int(ref.get("local", -1)) == local:
                        matches.append(output_index)
                    continue
                for clip_key, local_key in (("a_clip", "a_local"), ("b_clip", "b_local")):
                    if int(ref.get(clip_key, -1)) == clip_idx and int(ref.get(local_key, -1)) == local:
                        matches.append(output_index)
                        break
            except Exception:
                continue
        if not matches:
            return None
        current = max(0, min(len(timeline) - 1, int(self.preview_slider.value())))
        return min(matches, key=lambda output_index: abs(output_index - current))

    def _primary_clip_local_from_timeline_ref(self, ref: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        try:
            if ref["kind"] == "clip":
                return int(ref["clip"]), int(ref["local"])
            alpha = float(ref.get("alpha", 0.5))
            if alpha >= 0.5:
                return int(ref["b_clip"]), int(ref["b_local"])
            return int(ref["a_clip"]), int(ref["a_local"])
        except Exception:
            return None

    def _split_clip_at_local(self, clip_idx: int, split_local: int, output_index: Optional[int] = None) -> bool:
        if not (0 <= clip_idx < len(self.clips)):
            return False
        clip = self.clips[clip_idx]
        if bool(getattr(clip, "trimmed", False)):
            return False
        duration = self._clip_duration(clip)
        split_local = int(split_local)
        if not (0 < split_local < duration):
            return False

        first = replace(clip)
        second = replace(clip)
        if clip.kind == "source":
            first.end = int(clip.start) + split_local - 1
            second.start = int(clip.start) + split_local
        else:
            first.duration = split_local
            second.duration = duration - split_local

        self._set_clip_transition_out(first, "Cut", 0)
        self._set_clip_transition_in(second, "Cut", 0)
        self.clips[clip_idx:clip_idx + 1] = [first, second]
        self._last_split_output_index = output_index
        self._refresh_clip_table(select_row=clip_idx + 1)
        self._update_split_status()
        self._update_preview()
        return True

    def _on_preview_slider_right_clicked(self, output_index: int) -> None:
        timeline = self._get_timeline()
        if not timeline:
            return
        output_index = max(0, min(len(timeline) - 1, int(output_index)))
        mapped = self._primary_clip_local_from_timeline_ref(timeline[output_index])
        if mapped is None:
            self._update_split_status("Cannot split here")
            return
        clip_idx, local = mapped
        if 0 <= clip_idx < len(self.clips) and bool(getattr(self.clips[clip_idx], "trimmed", False)):
            self._update_split_status("Cannot split trimmed clip")
            return
        if not self._split_clip_at_local(clip_idx, local, output_index):
            self._update_split_status("Already at clip boundary")

    def _on_source_frame_right_double_clicked(self, frame_index: int) -> None:
        if self.total_frames <= 0:
            return
        frame_index = max(0, min(self.total_frames - 1, int(frame_index)))
        current_output = int(self.preview_slider.value()) if hasattr(self, "preview_slider") else 0
        selected_clip = self._selected_clip_index()
        candidates: List[Tuple[int, int, int, int, Optional[int]]] = []

        for clip_idx, clip in enumerate(self.clips):
            if clip.kind != "source" or bool(getattr(clip, "trimmed", False)):
                continue
            start = max(0, min(self.total_frames - 1, int(clip.start)))
            end = max(0, min(self.total_frames - 1, int(clip.end)))
            if start > end:
                start, end = end, start
            if not (start <= frame_index < end):
                continue
            split_local = frame_index - start + 1
            if not (0 < split_local < self._clip_duration(clip)):
                continue
            clicked_output = self._output_index_for_clip_local(clip_idx, split_local - 1)
            distance = abs((clicked_output if clicked_output is not None else current_output) - current_output)
            selected_rank = 0 if selected_clip == clip_idx else 1
            candidates.append((selected_rank, distance, clip_idx, split_local, clicked_output))

        if not candidates:
            self._preview_source_frame(frame_index)
            if self._source_frame_active(frame_index):
                self._update_split_status("Already at clip boundary")
            else:
                self._update_split_status("Source frame is not in output")
            return

        _selected_rank, _distance, clip_idx, split_local, clicked_output = min(candidates, key=lambda item: item[:4])
        if clicked_output is not None and hasattr(self, "preview_slider"):
            self.preview_slider.setValue(clicked_output)
        if self._split_clip_at_local(clip_idx, split_local, clicked_output):
            self._update_split_status(f"Last split: source frame {frame_index + 1}")
        else:
            self._update_split_status("Already at clip boundary")

    def _reset_title_controls(self) -> None:
        if not hasattr(self, "title_text_edit"):
            return
        self._title_ui_updating = True
        try:
            self.title_text_edit.clear()
            self.title_duration_spin.setValue(max(1, int(round(self._default_fps()))))
            _title_default_w, title_default_h = self._default_output_size()
            self.title_font_size_spin.setValue(max(18, min(96, int(title_default_h * 0.07))))
            current_family = str(self._frame_num.get("font_family", "Arial"))
            if current_family:
                self.title_font_combo.setCurrentText(current_family)
            self.title_style_combo.setCurrentText("Bold")
            self.title_align_combo.setCurrentText("Center")
            self.title_text_color = QtGui.QColor(255, 255, 255)
            self.title_bg_color = QtGui.QColor(32, 34, 38)
            self.title_use_first_frame_bg_check.setChecked(False)
            self.title_bg_opacity_spin.setValue(35)
            self._set_title_color_buttons()
            self._update_title_duration_seconds_label()
        finally:
            self._title_ui_updating = False

    def _reset_overlay_controls(self) -> None:
        if not hasattr(self, "overlay_text_edit"):
            return
        self._overlay_ui_updating = True
        try:
            self.overlay_text_edit.clear()
            self.overlay_start_spin.setValue(0)
            self.overlay_end_spin.setValue(0)
            self.overlay_pos_combo.setCurrentText("Bottom")
            self.overlay_size_spin.setValue(28)
            self.overlay_bg_check.setChecked(True)
            self.overlay_color = QtGui.QColor(255, 255, 255)
            self.color_button.setStyleSheet(self._button_color_stylesheet(self.overlay_color))
        finally:
            self._overlay_ui_updating = False

    def _reset_shape_controls(self) -> None:
        if not hasattr(self, "shape_type_combo"):
            return
        self._shape_ui_updating = True
        try:
            self.shape_type_combo.setCurrentText("Rectangle")
            self.shape_start_spin.setValue(0)
            self.shape_end_spin.setValue(0)
        finally:
            self._shape_ui_updating = False

    def _reset_editor_state(self) -> None:
        self.clips = []
        self.overlays = []
        self.shape_overlays = []
        self._last_split_output_index = None
        self._preview_drag_target = None
        self._preview_drag_mode = None
        self._preview_drag_offset = None
        self._preview_last_img_pos = None
        self._shape_rotate_start_angle = 0.0
        self._shape_rotate_start_mouse_angle = 0.0
        self._reset_title_controls()
        self._reset_overlay_controls()
        self._reset_shape_controls()
        if hasattr(self, "overlay_table"):
            self._refresh_overlay_table()
        if hasattr(self, "shape_table"):
            self._refresh_shape_overlay_table()

    def _refresh_source_frames(self, reset_clips: bool = False, reset_project: bool = False) -> None:
        self._stop_movie_preview_playback(update_status=False)
        self.total_frames = self._detect_total_frames()
        if self.total_frames <= 0:
            QtWidgets.QMessageBox.warning(self, "No Frames", "No AFM frame stack is currently loaded.")
            return

        if reset_project:
            self._reset_editor_state()

        if reset_clips or reset_project:
            self._last_split_output_index = None

        self._stop_thumbnail_timer()
        self.source_list.clear()
        for i in range(self.total_frames):
            item = QtWidgets.QListWidgetItem(f"{i + 1}")
            item.setData(QtCore.Qt.UserRole, i)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.source_list.addItem(item)

        self.channel_combo.setEnabled(self._has_2ch_data())
        self.thumb_status_label.setText(f"{self.total_frames} frames")

        if reset_clips or not self.clips:
            self.clips = [ClipSpec(kind="source", start=0, end=self.total_frames - 1)]
            self._refresh_clip_table(select_row=0)
        else:
            self._refresh_clip_table()
        self._invalidate_timeline()
        self._start_thumbnail_timer()
        self._update_preview_range()
        self._update_split_status()
        self._update_preview()

    def _start_thumbnail_timer(self) -> None:
        self._thumbnail_cursor = 0
        self._thumbnail_restore_index = int(getattr(gv, "index", 0) or 0) if gv is not None else 0
        self._thumbnail_timer = QtCore.QTimer(self)
        self._thumbnail_timer.timeout.connect(self._build_thumbnail_batch)
        self._thumbnail_timer.start(10)

    def _stop_thumbnail_timer(self) -> None:
        if self._thumbnail_timer is not None:
            self._thumbnail_timer.stop()
            self._thumbnail_timer.deleteLater()
            self._thumbnail_timer = None

    def _build_thumbnail_batch(self) -> None:
        if self.total_frames <= 0 or cv2 is None:
            self._stop_thumbnail_timer()
            return

        batch_count = 3
        built = 0
        while self._thumbnail_cursor < self.total_frames and built < batch_count:
            item = self.source_list.item(self._thumbnail_cursor)
            frame_index = int(item.data(QtCore.Qt.UserRole))
            base_pix = self._make_frame_pixmap(frame_index)
            if base_pix is not None:
                item.setData(QtCore.Qt.UserRole + 1, base_pix)
                item.setIcon(self._source_icon_from_base(frame_index, base_pix))
            self._style_source_frame_item(item, frame_index)
            self._thumbnail_cursor += 1
            built += 1

        self.thumb_status_label.setText(f"Thumbnails {min(self._thumbnail_cursor, self.total_frames)} / {self.total_frames}")
        if self._thumbnail_cursor >= self.total_frames:
            self._stop_thumbnail_timer()
            self.thumb_status_label.setText(f"{self.total_frames} frames ready")
            self._restore_main_frame(self._thumbnail_restore_index)

    def _make_frame_pixmap(self, frame_index: int) -> Optional[QtGui.QPixmap]:
        try:
            size = max(self._source_thumb_min, min(self._source_thumb_max, int(self._source_thumb_max)))
            bgr = self._render_pynud_display_frame(frame_index)
            if bgr is None:
                array = self._get_frame_array(frame_index)
                if array is None:
                    return None
                bgr = self._array_to_bgr(array, size, size, for_thumbnail=True)
            else:
                bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
            bgr = np.ascontiguousarray(bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
            return QtGui.QPixmap.fromImage(qimg.copy())
        except Exception:
            return None

    def _on_source_selection_changed(self) -> None:
        selected = self.source_list.selectedItems()
        if not selected:
            return
        frames = sorted(int(item.data(QtCore.Qt.UserRole)) for item in selected)
        frame_index = frames[0]
        output_index = self._output_index_for_source_frame(frame_index)
        if output_index is None:
            self._preview_source_frame(frame_index)
            self._update_split_status("Source frame is not in output")
            return
        if self.preview_slider.value() == output_index:
            self._update_preview()
        else:
            self.preview_slider.setValue(output_index)

    # ------------------------------------------------------------------
    # Clips
    # ------------------------------------------------------------------

    def _clip_duration(self, clip: ClipSpec) -> int:
        if clip.kind == "title":
            return max(1, int(clip.duration))
        return max(1, int(clip.end) - int(clip.start) + 1)

    def _normalize_transition_value(self, effect: Any, frames: Any) -> Tuple[str, int]:
        effect_text = str(effect or "Cut")
        if effect_text not in TRANSITIONS:
            effect_text = "Cut"
        try:
            frame_count = max(0, int(frames))
        except Exception:
            frame_count = 0
        if effect_text == "Cut":
            frame_count = 0
        return effect_text, frame_count

    def _normalize_transition_hold_frames(self, effect: Any, hold_frames: Any) -> int:
        effect_text = str(effect or "Cut")
        try:
            frame_count = max(0, int(hold_frames))
        except Exception:
            frame_count = 0
        if effect_text != "Fade Black":
            frame_count = 0
        return frame_count

    def _clip_transition_in(self, clip: ClipSpec) -> Tuple[str, int]:
        return self._normalize_transition_value(
            getattr(clip, "transition_in", "Cut"),
            getattr(clip, "transition_in_frames", 0),
        )

    def _clip_transition_out(self, clip: ClipSpec) -> Tuple[str, int]:
        effect = getattr(clip, "transition_out", "Cut")
        frames = getattr(clip, "transition_out_frames", 0)
        legacy_effect = getattr(clip, "transition", "Cut")
        legacy_frames = getattr(clip, "transition_frames", 0)
        try:
            frame_count = int(frames or 0)
        except Exception:
            frame_count = 0
        try:
            legacy_frame_count = int(legacy_frames or 0)
        except Exception:
            legacy_frame_count = 0
        if (effect in (None, "", "Cut")) and frame_count == 0 and (
            legacy_effect not in (None, "", "Cut") or legacy_frame_count > 0
        ):
            effect = legacy_effect
            frames = legacy_frames
        return self._normalize_transition_value(effect, frames)

    def _clip_transition_in_hold(self, clip: ClipSpec) -> int:
        effect, _frames = self._clip_transition_in(clip)
        return self._normalize_transition_hold_frames(
            effect,
            getattr(clip, "transition_in_hold_frames", 0),
        )

    def _clip_transition_out_hold(self, clip: ClipSpec) -> int:
        effect, _frames = self._clip_transition_out(clip)
        hold_frames = getattr(clip, "transition_out_hold_frames", 0)
        legacy_hold_frames = getattr(clip, "transition_hold_frames", 0)
        try:
            frame_count = int(hold_frames or 0)
        except Exception:
            frame_count = 0
        try:
            legacy_frame_count = int(legacy_hold_frames or 0)
        except Exception:
            legacy_frame_count = 0
        if frame_count == 0 and legacy_frame_count > 0:
            hold_frames = legacy_hold_frames
        return self._normalize_transition_hold_frames(effect, hold_frames)

    def _set_clip_transition_in(self, clip: ClipSpec, effect: Any, frames: Any, hold_frames: Any = 0) -> None:
        effect_text, frame_count = self._normalize_transition_value(effect, frames)
        hold_frame_count = self._normalize_transition_hold_frames(effect_text, hold_frames)
        clip.transition_in = effect_text
        clip.transition_in_frames = frame_count
        clip.transition_in_hold_frames = hold_frame_count

    def _set_clip_transition_out(self, clip: ClipSpec, effect: Any, frames: Any, hold_frames: Any = 0) -> None:
        effect_text, frame_count = self._normalize_transition_value(effect, frames)
        hold_frame_count = self._normalize_transition_hold_frames(effect_text, hold_frames)
        clip.transition_out = effect_text
        clip.transition_out_frames = frame_count
        clip.transition_out_hold_frames = hold_frame_count
        clip.transition = effect_text
        clip.transition_frames = frame_count
        clip.transition_hold_frames = hold_frame_count

    def _sync_transition_ins_from_outs(self) -> None:
        if not self.clips:
            return
        self._set_clip_transition_in(self.clips[0], "Cut", 0)
        for idx in range(1, len(self.clips)):
            effect, frames = self._clip_transition_out(self.clips[idx - 1])
            hold_frames = self._clip_transition_out_hold(self.clips[idx - 1])
            self._set_clip_transition_in(self.clips[idx], effect, frames, hold_frames)

    def _set_clip_transition_to_next(self, clip_idx: int, effect: Any, frames: Any, hold_frames: Any = 0) -> None:
        if not (0 <= int(clip_idx) < len(self.clips)):
            return
        self._set_clip_transition_out(self.clips[int(clip_idx)], effect, frames, hold_frames)
        self._sync_transition_ins_from_outs()

    def _transition_summary(self, clip: ClipSpec, direction: str) -> str:
        if direction == "in":
            effect, frames = self._clip_transition_in(clip)
            hold_frames = self._clip_transition_in_hold(clip)
        else:
            effect, frames = self._clip_transition_out(clip)
            hold_frames = self._clip_transition_out_hold(clip)
        if effect == "Fade Black" and hold_frames > 0:
            return f"{effect} ({frames}+{hold_frames}+{frames})"
        return f"{effect} ({frames})"

    def _clip_content_text(self, clip: ClipSpec) -> str:
        if clip.kind == "title":
            title = clip.title or "(untitled)"
            return title.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " / ")
        return f"{clip.start + 1} - {clip.end + 1}"

    def _refresh_clip_table(self, select_row: Optional[int] = None) -> None:
        self._ui_updating = True
        try:
            self._sync_transition_ins_from_outs()
            self.clip_table.setRowCount(len(self.clips))
            self._set_compact_table_height(self.clip_table, rows=len(self.clips), max_rows=5)
            for row, clip in enumerate(self.clips):
                is_trimmed = bool(getattr(clip, "trimmed", False))
                type_text = "Title" if clip.kind == "title" else "Source"
                if is_trimmed:
                    type_text = f"Trimmed {type_text}"
                values = [
                    str(row + 1),
                    type_text,
                    self._clip_content_text(clip),
                    str(self._clip_duration(clip)),
                    self._transition_summary(clip, "out"),
                ]
                for col, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setTextAlignment(QtCore.Qt.AlignCenter if col in (0, 1, 3) else QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                    if is_trimmed:
                        item.setBackground(QtGui.QColor(220, 220, 220))
                        item.setForeground(QtGui.QColor(90, 90, 90))
                    self.clip_table.setItem(row, col, item)
            self.clip_table.resizeColumnsToContents()
            self._fit_table_width_to_columns(self.clip_table)
            if select_row is not None and 0 <= select_row < len(self.clips):
                self.clip_table.selectRow(select_row)
                item = self.clip_table.item(select_row, 0)
                if item is not None:
                    self.clip_table.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
            self._update_right_panel_size()
        finally:
            self._ui_updating = False
        if select_row is not None and 0 <= select_row < len(self.clips):
            self._on_clip_selection_changed()
        self._invalidate_timeline()
        self._update_preview_range()
        self._refresh_source_frame_visuals()

    def _selected_clip_index(self) -> Optional[int]:
        rows = self.clip_table.selectionModel().selectedRows() if self.clip_table.selectionModel() else []
        if not rows:
            return None
        row = rows[0].row()
        if 0 <= row < len(self.clips):
            return row
        return None

    def _on_clip_selection_changed(self) -> None:
        if self._ui_updating:
            return
        idx = self._selected_clip_index()
        if idx is None:
            return
        clip = self.clips[idx]
        self._clip_ui_updating = True
        try:
            out_effect, out_frames = self._clip_transition_out(clip)
            out_hold_frames = self._clip_transition_out_hold(clip)
            self.transition_out_combo.setCurrentText(out_effect)
            self.transition_out_frames_spin.setValue(out_frames)
            self.transition_out_hold_frames_spin.setValue(out_hold_frames)
        finally:
            self._clip_ui_updating = False
        if clip.kind == "source":
            self._select_source_range(clip.start, clip.end)
        else:
            self._set_title_controls_from_clip(clip)
        self._sync_output_slider_to_clip(idx)

    def _sync_output_slider_to_clip(self, clip_idx: int) -> None:
        output_index = self._output_index_for_clip_start(clip_idx)
        if output_index is None:
            if 0 <= int(clip_idx) < len(self.clips) and bool(getattr(self.clips[int(clip_idx)], "trimmed", False)):
                self._update_split_status("Trimmed clip is skipped")
            return
        if self.preview_slider.value() == output_index:
            self._update_preview()
        else:
            self.preview_slider.setValue(output_index)

    def _on_clip_transition_changed(self, *_args) -> None:
        if self._ui_updating or self._clip_ui_updating:
            return
        idx = self._selected_clip_index()
        if idx is None or not (0 <= idx < len(self.clips)):
            return
        self._set_clip_transition_to_next(
            idx,
            self.transition_out_combo.currentText(),
            self.transition_out_frames_spin.value(),
            self.transition_out_hold_frames_spin.value(),
        )
        self._refresh_clip_table(select_row=idx)
        self._update_preview()

    def _transition_range_for_clip(self, clip_idx: int, direction: str) -> Optional[Tuple[int, int]]:
        timeline = self._get_timeline()
        indices = [
            output_index
            for output_index, ref in enumerate(timeline)
            if (
                ref.get("kind") == "transition"
                and int(ref.get("transition_owner", -1)) == int(clip_idx)
                and str(ref.get("transition_side", "")) == direction
            )
        ]
        if not indices:
            return None
        return min(indices), max(indices)

    def _visible_clip_indices(self) -> List[int]:
        include_trimmed = self._show_trimmed_output()
        return [
            idx for idx, clip in enumerate(self.clips)
            if include_trimmed or not bool(getattr(clip, "trimmed", False))
        ]

    def _visible_neighbor_missing(self, clip_idx: int, direction: str) -> bool:
        visible = self._visible_clip_indices()
        if clip_idx not in visible:
            return True
        pos = visible.index(clip_idx)
        if direction == "in":
            return pos == 0
        return pos >= len(visible) - 1

    def _boundary_transition(self, prev_idx: int, curr_idx: int) -> Tuple[str, int, int, int, str]:
        prev_effect, prev_frames = self._clip_transition_out(self.clips[prev_idx])
        prev_hold_frames = self._clip_transition_out_hold(self.clips[prev_idx])
        if prev_effect != "Cut" and (prev_frames > 0 or prev_hold_frames > 0):
            return prev_effect, prev_frames, prev_hold_frames, prev_idx, "out"
        return "Cut", 0, 0, prev_idx, "out"

    def _transition_status_for_output(self, output_index: int) -> Optional[str]:
        timeline = self._get_timeline()
        if not (0 <= int(output_index) < len(timeline)):
            return None
        ref = timeline[int(output_index)]
        if ref.get("kind") != "transition":
            return None
        key = (
            int(ref.get("a_clip", -1)),
            int(ref.get("b_clip", -1)),
            str(ref.get("effect", "")),
            int(ref.get("transition_owner", -1)),
            str(ref.get("transition_side", "")),
        )
        start = int(output_index)
        end = int(output_index)
        while start > 0:
            prev = timeline[start - 1]
            prev_key = (
                int(prev.get("a_clip", -1)),
                int(prev.get("b_clip", -1)),
                str(prev.get("effect", "")),
                int(prev.get("transition_owner", -1)),
                str(prev.get("transition_side", "")),
            )
            if prev.get("kind") != "transition" or prev_key != key:
                break
            start -= 1
        while end + 1 < len(timeline):
            nxt = timeline[end + 1]
            next_key = (
                int(nxt.get("a_clip", -1)),
                int(nxt.get("b_clip", -1)),
                str(nxt.get("effect", "")),
                int(nxt.get("transition_owner", -1)),
                str(nxt.get("transition_side", "")),
            )
            if nxt.get("kind") != "transition" or next_key != key:
                break
            end += 1
        return f"Transition: {key[2]} {int(output_index) - start + 1}/{end - start + 1}"

    def _preview_selected_transition(self, direction: str) -> None:
        idx = self._selected_clip_index()
        if idx is None or not (0 <= idx < len(self.clips)):
            self._update_split_status("Select a clip first")
            return
        clip = self.clips[idx]
        direction = "out"
        effect, frames = self._clip_transition_out(clip)
        hold_frames = self._clip_transition_out_hold(clip)
        no_neighbor_message = "No next clip to transition"
        no_frames_message = "No transition frames"
        no_active_message = "No active transition"
        if effect == "Cut" or (frames <= 0 and hold_frames <= 0):
            self._update_split_status(no_frames_message)
            return
        transition_range = self._transition_range_for_clip(idx, direction)
        if transition_range is None:
            self._update_split_status(no_neighbor_message if self._visible_neighbor_missing(idx, direction) else no_active_message)
            return
        start, _end = transition_range
        if self.preview_slider.value() == start:
            self._update_preview()
        else:
            self.preview_slider.setValue(start)

    def _select_source_range(self, start: int, end: int) -> None:
        self.source_list.blockSignals(True)
        try:
            self.source_list.clearSelection()
            for frame in range(max(0, start), min(self.total_frames - 1, end) + 1):
                item = self.source_list.item(frame)
                if item is not None:
                    item.setSelected(True)
            item = self.source_list.item(max(0, min(self.total_frames - 1, start)))
            if item is not None:
                self.source_list.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        finally:
            self.source_list.blockSignals(False)

    def _selected_source_range(self) -> Tuple[int, int]:
        selected = self.source_list.selectedItems()
        if selected:
            frames = sorted(int(item.data(QtCore.Qt.UserRole)) for item in selected)
            return frames[0], frames[-1]
        current = self.source_list.currentItem() if hasattr(self, "source_list") else None
        if current is not None:
            frame = int(current.data(QtCore.Qt.UserRole))
            return frame, frame
        return 0, max(0, self.total_frames - 1)

    def _title_text_value(self) -> str:
        if hasattr(self.title_text_edit, "toPlainText"):
            return self.title_text_edit.toPlainText().strip()
        return self.title_text_edit.text().strip()

    def _overlay_text_value(self) -> str:
        if hasattr(self.overlay_text_edit, "toPlainText"):
            return self.overlay_text_edit.toPlainText().strip()
        return self.overlay_text_edit.text().strip()

    def _set_overlay_text_value(self, text: str) -> None:
        if hasattr(self.overlay_text_edit, "setPlainText"):
            self.overlay_text_edit.setPlainText(text)
        else:
            self.overlay_text_edit.setText(text)

    def _button_color_stylesheet(self, color: QtGui.QColor) -> str:
        qcolor = QtGui.QColor(color)
        luminance = 0.299 * qcolor.red() + 0.587 * qcolor.green() + 0.114 * qcolor.blue()
        text_color = "#ffffff" if luminance < 128 else "#000000"
        return f"background-color: {qcolor.name()}; color: {text_color};"

    def _set_title_color_buttons(self) -> None:
        if hasattr(self, "title_color_button"):
            self.title_color_button.setStyleSheet(self._button_color_stylesheet(self.title_text_color))
        if hasattr(self, "title_bg_color_button"):
            self.title_bg_color_button.setStyleSheet(self._button_color_stylesheet(self.title_bg_color))

    def _update_title_duration_seconds_label(self, *_args) -> None:
        if not hasattr(self, "title_duration_seconds_label"):
            return
        try:
            fps = float(self.fps_spin.value()) if hasattr(self, "fps_spin") else float(self._default_fps())
        except Exception:
            fps = 20.0
        frames = int(self.title_duration_spin.value()) if hasattr(self, "title_duration_spin") else 0
        seconds = frames / max(0.1, fps)
        self.title_duration_seconds_label.setText(f"{seconds:.2f} s")

    def _connect_title_controls(self) -> None:
        self.title_text_edit.textChanged.connect(self._on_title_controls_changed)
        self.title_duration_spin.valueChanged.connect(self._on_title_controls_changed)
        self.title_use_first_frame_bg_check.toggled.connect(self._on_title_controls_changed)
        self.title_bg_opacity_spin.valueChanged.connect(self._on_title_controls_changed)

    def _on_title_controls_changed(self, *_args) -> None:
        self._update_title_duration_seconds_label()
        if self._title_ui_updating:
            return
        idx = self._selected_clip_index()
        if idx is None or not (0 <= idx < len(self.clips)) or self.clips[idx].kind != "title":
            return
        self._apply_title_controls_to_clip(self.clips[idx])
        self._refresh_clip_table(select_row=idx)
        self._update_preview()

    def _choose_title_text_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(self.title_text_color, self, "Title Text Color")
        if color.isValid():
            self.title_text_color = color
            self._set_title_color_buttons()
            self._on_title_controls_changed()

    def _choose_title_background_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(self.title_bg_color, self, "Title Background Color")
        if color.isValid():
            self.title_bg_color = color
            self._set_title_color_buttons()
            self._on_title_controls_changed()

    def _set_title_controls_from_clip(self, clip: ClipSpec) -> None:
        if clip.kind != "title":
            return
        self._title_ui_updating = True
        try:
            if hasattr(self.title_text_edit, "setPlainText"):
                self.title_text_edit.setPlainText(clip.title)
            else:
                self.title_text_edit.setText(clip.title)
            self.title_duration_spin.setValue(self._clip_duration(clip))
            self.title_font_size_spin.setValue(max(8, int(getattr(clip, "title_font_size", self._default_title_font_size()))))
            self.title_font_combo.setCurrentText(str(getattr(clip, "title_font_family", "Arial") or "Arial"))
            self.title_style_combo.setCurrentText(self._normalize_font_style(str(getattr(clip, "title_font_style", "Bold") or "Bold")))
            self.title_align_combo.setCurrentText(str(getattr(clip, "title_align", "Center") or "Center"))
            self.title_text_color = QtGui.QColor(getattr(clip, "title_color", QtGui.QColor(255, 255, 255)))
            self.title_bg_color = QtGui.QColor(getattr(clip, "title_background_color", QtGui.QColor(32, 34, 38)))
            self.title_use_first_frame_bg_check.setChecked(bool(getattr(clip, "title_use_first_frame_background", False)))
            self.title_bg_opacity_spin.setValue(max(0, min(100, int(getattr(clip, "title_background_frame_opacity", 35)))))
            self._set_title_color_buttons()
            self._update_title_duration_seconds_label()
        finally:
            self._title_ui_updating = False

    def _apply_title_controls_to_clip(self, clip: ClipSpec) -> None:
        clip.kind = "title"
        clip.title = self._title_text_value() or clip.title or "Title"
        clip.duration = max(1, int(self.title_duration_spin.value()))
        clip.title_background_color = QtGui.QColor(self.title_bg_color)
        clip.title_use_first_frame_background = bool(self.title_use_first_frame_bg_check.isChecked())
        clip.title_background_frame_opacity = max(0, min(100, int(self.title_bg_opacity_spin.value())))

    def _reset_source_clips(self) -> None:
        if self.total_frames <= 0:
            return
        source_indices = [idx for idx, clip in enumerate(self.clips) if clip.kind == "source"]
        insert_at = source_indices[0] if source_indices else len(self.clips)
        transition_source = None
        for idx in range(insert_at, len(self.clips)):
            if self.clips[idx].kind == "source":
                continue
            for prev_idx in range(idx - 1, -1, -1):
                if self.clips[prev_idx].kind == "source":
                    transition_source = self.clips[prev_idx]
                    break
            break
        if transition_source is None and source_indices:
            transition_source = self.clips[source_indices[-1]]
        reset_clip = ClipSpec(kind="source", start=0, end=self.total_frames - 1)
        if transition_source is not None:
            out_effect, out_frames = self._clip_transition_out(transition_source)
            out_hold_frames = self._clip_transition_out_hold(transition_source)
            self._set_clip_transition_out(reset_clip, out_effect, out_frames, out_hold_frames)

        kept_clips = [clip for clip in self.clips if clip.kind != "source"]
        insert_at = max(0, min(insert_at, len(kept_clips)))
        kept_clips.insert(insert_at, reset_clip)
        self.clips = kept_clips
        self._last_split_output_index = None
        self._refresh_clip_table(select_row=insert_at)
        self._update_split_status("Source clips reset")
        self._update_preview()

    def _add_source_clip(self) -> None:
        if self.total_frames <= 0:
            return
        start, end = self._selected_source_range()
        clip = ClipSpec(
            kind="source",
            start=max(0, min(self.total_frames - 1, start)),
            end=max(0, min(self.total_frames - 1, end)),
            transition_in="Cut",
            transition_in_frames=0,
            transition_in_hold_frames=0,
            transition_out=self.transition_out_combo.currentText(),
            transition_out_frames=int(self.transition_out_frames_spin.value()),
            transition_out_hold_frames=self._normalize_transition_hold_frames(
                self.transition_out_combo.currentText(),
                self.transition_out_hold_frames_spin.value(),
            ),
            transition=self.transition_out_combo.currentText(),
            transition_frames=int(self.transition_out_frames_spin.value()),
            transition_hold_frames=self._normalize_transition_hold_frames(
                self.transition_out_combo.currentText(),
                self.transition_out_hold_frames_spin.value(),
            ),
        )
        if clip.start > clip.end:
            clip.start, clip.end = clip.end, clip.start
        self.clips.append(clip)
        self._refresh_clip_table(select_row=len(self.clips) - 1)
        self._update_preview()

    def _add_title_clip(self) -> None:
        clip = ClipSpec(
            kind="title",
            title_font_size=self._default_title_font_size(),
            title_font_family=str(self._frame_num.get("font_family", "Arial")),
            title_font_style="Bold",
            transition_in="Cut",
            transition_in_frames=0,
            transition_in_hold_frames=0,
            transition_out=self.transition_out_combo.currentText(),
            transition_out_frames=int(self.transition_out_frames_spin.value()),
            transition_out_hold_frames=self._normalize_transition_hold_frames(
                self.transition_out_combo.currentText(),
                self.transition_out_hold_frames_spin.value(),
            ),
            transition=self.transition_out_combo.currentText(),
            transition_frames=int(self.transition_out_frames_spin.value()),
            transition_hold_frames=self._normalize_transition_hold_frames(
                self.transition_out_combo.currentText(),
                self.transition_out_hold_frames_spin.value(),
            ),
        )
        self._apply_title_controls_to_clip(clip)
        self.clips.insert(0, clip)
        select_row = 0
        self._refresh_clip_table(select_row=select_row)
        self._update_preview()

    def _trim_selected_clip(self) -> None:
        idx = self._selected_clip_index()
        if idx is None:
            return
        clip = self.clips[idx]
        if bool(getattr(clip, "trimmed", False)):
            self._update_split_status("Clip is already trimmed")
            return
        clip.trimmed = True
        self._refresh_clip_table(select_row=idx)
        self._update_split_status("Clip trimmed")
        self._update_preview()

    def _move_selected_clip(self, delta: int) -> None:
        idx = self._selected_clip_index()
        if idx is None:
            return
        target = idx + delta
        if not (0 <= target < len(self.clips)):
            return
        self.clips[idx], self.clips[target] = self.clips[target], self.clips[idx]
        self._refresh_clip_table(select_row=target)
        self._update_preview()

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _choose_overlay_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(self.overlay_color, self, "Text Color")
        if color.isValid():
            self.overlay_color = color
            self.color_button.setStyleSheet(self._button_color_stylesheet(color))
            idx = self._selected_overlay_index()
            if idx is not None:
                self.overlays[idx].color = QtGui.QColor(color)
                self._update_preview()

    def _add_overlay(self) -> None:
        text = self._overlay_text_value()
        if not text:
            return
        start = int(self.overlay_start_spin.value())
        end = int(self.overlay_end_spin.value())
        if start > end:
            start, end = end, start
        overlay = TextOverlaySpec(
            text=text,
            start=start,
            end=end,
            position="Bottom",
            font_size=28,
            color=QtGui.QColor(255, 255, 255),
            background=True,
            font_family=str(self._frame_num.get("font_family", "Arial")),
            font_style=str(self._frame_num.get("font_style", "Normal")),
        )
        self.overlays.append(overlay)
        self._refresh_overlay_table(select_row=len(self.overlays) - 1)
        self._update_preview()

    def _selected_overlay_index(self) -> Optional[int]:
        rows = self.overlay_table.selectionModel().selectedRows() if self.overlay_table.selectionModel() else []
        if not rows:
            return None
        row = rows[0].row()
        if 0 <= row < len(self.overlays):
            return row
        return None

    def _on_overlay_selection_changed(self) -> None:
        idx = self._selected_overlay_index()
        if idx is None:
            return
        overlay = self.overlays[idx]
        self._overlay_ui_updating = True
        try:
            self._set_overlay_text_value(overlay.text)
            self.overlay_start_spin.setValue(int(overlay.start))
            self.overlay_end_spin.setValue(int(overlay.end))
            if not getattr(overlay, "custom_position", False):
                self.overlay_pos_combo.setCurrentText(overlay.position)
            self.overlay_size_spin.setValue(int(overlay.font_size))
            self.overlay_bg_check.setChecked(bool(overlay.background))
            self.overlay_color = QtGui.QColor(overlay.color)
            self.color_button.setStyleSheet(self._button_color_stylesheet(self.overlay_color))
        finally:
            self._overlay_ui_updating = False

    def _on_overlay_controls_changed(self, *_args) -> None:
        if self._overlay_ui_updating:
            return
        idx = self._selected_overlay_index()
        if idx is None:
            return
        overlay = self.overlays[idx]
        start = int(self.overlay_start_spin.value())
        end = int(self.overlay_end_spin.value())
        if start > end:
            start, end = end, start
        overlay.start = start
        overlay.end = end
        overlay.text = self._overlay_text_value()
        if self.sender() is self.overlay_text_edit:
            item = self.overlay_table.item(idx, 0)
            if item is not None:
                item.setText(str(overlay.text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " / "))
                self.overlay_table.setColumnWidth(0, 180)
                self.overlay_table.setColumnWidth(4, 70)
                self._fit_table_width_to_columns(self.overlay_table)
        else:
            self._refresh_overlay_table(select_row=idx)
        self._update_preview()

    def _delete_overlay(self) -> None:
        rows = self.overlay_table.selectionModel().selectedRows() if self.overlay_table.selectionModel() else []
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self.overlays):
            del self.overlays[row]
        self._refresh_overlay_table()
        self._update_preview()

    def _refresh_overlay_table(self, select_row: Optional[int] = None) -> None:
        self.overlay_table.setRowCount(len(self.overlays))
        self._set_compact_table_height(self.overlay_table, rows=len(self.overlays), max_rows=4)
        for row, overlay in enumerate(self.overlays):
            position_text = "Custom" if getattr(overlay, "custom_position", False) else overlay.position
            values = [
                str(overlay.text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " / "),
                str(overlay.start),
                str(overlay.end),
                position_text,
                str(overlay.font_size),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col in (1, 2, 4):
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.overlay_table.setItem(row, col, item)
        self.overlay_table.resizeColumnsToContents()
        self.overlay_table.setColumnWidth(0, 180)
        self.overlay_table.setColumnWidth(4, 70)
        self._fit_table_width_to_columns(self.overlay_table)
        if select_row is not None and 0 <= select_row < len(self.overlays):
            self.overlay_table.selectRow(select_row)
            item = self.overlay_table.item(select_row, 0)
            if item is not None:
                self.overlay_table.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        self._update_right_panel_size()

    def _shape_default_size(self, shape: str) -> Tuple[float, float]:
        if shape == "Circle":
            return 0.22, 0.22
        if shape in ("Line", "Arrow"):
            return 0.34, 0.14
        if shape == "Arrowhead":
            return 0.18, 0.18
        if shape == "Triangle":
            return 0.24, 0.24
        return 0.30, 0.20

    def _add_shape_overlay(self) -> None:
        start = int(self.shape_start_spin.value())
        end = int(self.shape_end_spin.value())
        if start > end:
            start, end = end, start
        shape_name = self.shape_type_combo.currentText() or "Rectangle"
        w_ratio, h_ratio = self._shape_default_size(shape_name)
        if shape_name in ("Line", "Arrow"):
            overlay = ShapeOverlaySpec(
                shape=shape_name,
                start=start,
                end=end,
                x_ratio=0.33,
                y_ratio=0.50,
                x2_ratio=0.67,
                y2_ratio=0.50,
                w_ratio=w_ratio,
                h_ratio=h_ratio,
            )
        else:
            overlay = ShapeOverlaySpec(
                shape=shape_name,
                start=start,
                end=end,
                w_ratio=w_ratio,
                h_ratio=h_ratio,
            )
        self.shape_overlays.append(overlay)
        self._refresh_shape_overlay_table(select_row=len(self.shape_overlays) - 1)
        self._update_preview()

    def _selected_shape_overlay_index(self) -> Optional[int]:
        rows = self.shape_table.selectionModel().selectedRows() if self.shape_table.selectionModel() else []
        if not rows:
            return None
        row = rows[0].row()
        if 0 <= row < len(self.shape_overlays):
            return row
        return None

    def _on_shape_overlay_selection_changed(self) -> None:
        idx = self._selected_shape_overlay_index()
        if idx is None:
            return
        overlay = self.shape_overlays[idx]
        self._shape_ui_updating = True
        try:
            self.shape_type_combo.setCurrentText(str(overlay.shape or "Rectangle"))
            self.shape_start_spin.setValue(int(overlay.start))
            self.shape_end_spin.setValue(int(overlay.end))
        finally:
            self._shape_ui_updating = False

    def _on_shape_overlay_controls_changed(self, *_args) -> None:
        if self._shape_ui_updating:
            return
        idx = self._selected_shape_overlay_index()
        if idx is None:
            return
        overlay = self.shape_overlays[idx]
        start = int(self.shape_start_spin.value())
        end = int(self.shape_end_spin.value())
        if start > end:
            start, end = end, start
        new_shape = self.shape_type_combo.currentText() or overlay.shape or "Rectangle"
        if new_shape != overlay.shape:
            self._convert_shape_overlay_type(overlay, new_shape)
        overlay.shape = new_shape
        overlay.start = start
        overlay.end = end
        self._refresh_shape_overlay_table(select_row=idx)
        self._update_preview()

    def _shape_uses_endpoints(self, overlay_or_shape: Any) -> bool:
        shape = overlay_or_shape if isinstance(overlay_or_shape, str) else getattr(overlay_or_shape, "shape", "")
        return str(shape or "") in ("Line", "Arrow")

    def _convert_shape_overlay_type(self, overlay: ShapeOverlaySpec, new_shape: str) -> None:
        old_uses_endpoints = self._shape_uses_endpoints(overlay)
        new_uses_endpoints = self._shape_uses_endpoints(new_shape)
        if old_uses_endpoints == new_uses_endpoints:
            return
        if new_uses_endpoints:
            x = max(0.0, min(1.0, float(getattr(overlay, "x_ratio", 0.35))))
            y = max(0.0, min(1.0, float(getattr(overlay, "y_ratio", 0.35))))
            w = max(0.02, min(1.0, float(getattr(overlay, "w_ratio", 0.30))))
            h = max(0.02, min(1.0, float(getattr(overlay, "h_ratio", 0.20))))
            center_y = max(0.0, min(1.0, y + h * 0.5))
            overlay.x_ratio = x
            overlay.y_ratio = center_y
            overlay.x2_ratio = max(0.0, min(1.0, x + w))
            overlay.y2_ratio = center_y
            return

        x1 = max(0.0, min(1.0, float(getattr(overlay, "x_ratio", 0.33))))
        y1 = max(0.0, min(1.0, float(getattr(overlay, "y_ratio", 0.50))))
        x2 = max(0.0, min(1.0, float(getattr(overlay, "x2_ratio", 0.67))))
        y2 = max(0.0, min(1.0, float(getattr(overlay, "y2_ratio", 0.50))))
        overlay.x_ratio = min(x1, x2)
        overlay.y_ratio = min(y1, y2)
        overlay.w_ratio = max(0.02, abs(x2 - x1))
        overlay.h_ratio = max(0.02, abs(y2 - y1))

    def _delete_shape_overlay(self) -> None:
        rows = self.shape_table.selectionModel().selectedRows() if self.shape_table.selectionModel() else []
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self.shape_overlays):
            del self.shape_overlays[row]
        self._refresh_shape_overlay_table()
        self._update_preview()

    def _refresh_shape_overlay_table(self, select_row: Optional[int] = None) -> None:
        self.shape_table.setRowCount(len(self.shape_overlays))
        self._set_compact_table_height(self.shape_table, rows=len(self.shape_overlays), max_rows=4)
        for row, overlay in enumerate(self.shape_overlays):
            fill_text = f"On {int(getattr(overlay, 'fill_opacity', 35))}%" if bool(getattr(overlay, "fill", False)) else "Off"
            line_text = f"{getattr(overlay, 'line_style', 'Solid')} {int(getattr(overlay, 'line_width', 3))}"
            if str(getattr(overlay, "shape", "")) == "Arrow":
                line_text = f"{line_text} H{int(getattr(overlay, 'arrow_head_percent', 20))}%"
            if self._shape_rotation_enabled(overlay):
                angle = self._shape_rotation_degrees(overlay)
                if abs(angle) > 0.001:
                    line_text = f"{line_text} R{angle:g}deg"
            values = [
                str(getattr(overlay, "shape", "Rectangle")),
                str(int(getattr(overlay, "start", 0))),
                str(int(getattr(overlay, "end", 0))),
                line_text,
                fill_text,
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col in (1, 2, 3, 4):
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.shape_table.setItem(row, col, item)
        self.shape_table.setColumnWidth(0, 90)
        self.shape_table.setColumnWidth(1, 54)
        self.shape_table.setColumnWidth(2, 54)
        self.shape_table.setColumnWidth(3, 118)
        self.shape_table.setColumnWidth(4, 64)
        self._fit_table_width_to_columns(self.shape_table)
        if select_row is not None and 0 <= select_row < len(self.shape_overlays):
            self.shape_table.selectRow(select_row)
            item = self.shape_table.item(select_row, 0)
            if item is not None:
                self.shape_table.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        self._update_right_panel_size()

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _invalidate_timeline(self) -> None:
        self._timeline_cache = None
        self._contrast_limits = None

    def _get_timeline(self) -> List[Dict[str, Any]]:
        if self._timeline_cache is None:
            self._timeline_cache = self._build_timeline()
        return self._timeline_cache

    def _build_timeline(self) -> List[Dict[str, Any]]:
        clip_indices = self._visible_clip_indices()
        if not clip_indices:
            return []

        timeline: List[Dict[str, Any]] = []
        first_clip_idx = clip_indices[0]
        first_duration = self._clip_duration(self.clips[first_clip_idx])
        for local in range(first_duration):
            timeline.append({"kind": "clip", "clip": first_clip_idx, "local": local})

        for visible_idx in range(1, len(clip_indices)):
            prev_idx = clip_indices[visible_idx - 1]
            clip_idx = clip_indices[visible_idx]
            prev_clip = self.clips[prev_idx]
            curr_clip = self.clips[clip_idx]
            prev_duration = self._clip_duration(prev_clip)
            curr_duration = self._clip_duration(curr_clip)
            trans, trans_frames, hold_frames, transition_owner, transition_side = self._boundary_transition(prev_idx, clip_idx)
            trans_frames = max(0, min(trans_frames, prev_duration, curr_duration, len(timeline)))
            hold_frames = max(0, int(hold_frames))

            if trans == "Fade Black" and (trans_frames > 0 or hold_frames > 0):
                if trans_frames > 0:
                    del timeline[-trans_frames:]
                    for step in range(trans_frames):
                        timeline.append(
                            {
                                "kind": "transition",
                                "effect": trans,
                                "phase": "fade_out",
                                "a_clip": prev_idx,
                                "a_local": prev_duration - trans_frames + step,
                                "b_clip": clip_idx,
                                "b_local": 0,
                                "alpha": (step + 1) / float(trans_frames + 1),
                                "transition_owner": transition_owner,
                                "transition_side": transition_side,
                            }
                        )
                for step in range(hold_frames):
                    timeline.append(
                        {
                            "kind": "transition",
                            "effect": trans,
                            "phase": "hold_black",
                            "a_clip": prev_idx,
                            "a_local": max(0, prev_duration - 1),
                            "b_clip": clip_idx,
                            "b_local": 0,
                            "alpha": 1.0,
                            "transition_owner": transition_owner,
                            "transition_side": transition_side,
                        }
                    )
                if trans_frames > 0:
                    for step in range(trans_frames):
                        timeline.append(
                            {
                                "kind": "transition",
                                "effect": trans,
                                "phase": "fade_in",
                                "a_clip": prev_idx,
                                "a_local": max(0, prev_duration - 1),
                                "b_clip": clip_idx,
                                "b_local": step,
                                "alpha": (step + 1) / float(trans_frames + 1),
                                "transition_owner": transition_owner,
                                "transition_side": transition_side,
                            }
                        )
                start_local = trans_frames
            elif trans_frames > 0:
                del timeline[-trans_frames:]
                for step in range(trans_frames):
                    timeline.append(
                        {
                            "kind": "transition",
                            "effect": trans,
                            "a_clip": prev_idx,
                            "a_local": prev_duration - trans_frames + step,
                            "b_clip": clip_idx,
                            "b_local": step,
                            "alpha": (step + 1) / float(trans_frames + 1),
                            "transition_owner": transition_owner,
                            "transition_side": transition_side,
                        }
                    )
                start_local = trans_frames
            else:
                start_local = 0

            for local in range(start_local, curr_duration):
                timeline.append({"kind": "clip", "clip": clip_idx, "local": local})

        return timeline

    def _update_preview_range(self) -> None:
        duration = len(self._get_timeline())
        self.preview_slider.blockSignals(True)
        try:
            self.preview_slider.setRange(0, max(0, duration - 1))
            if self.preview_slider.value() >= duration:
                self.preview_slider.setValue(max(0, duration - 1))
        finally:
            self.preview_slider.blockSignals(False)
        for spin in (self.overlay_start_spin, self.overlay_end_spin):
            spin.setRange(0, max(0, duration - 1))
        self.overlay_end_spin.setValue(max(0, duration - 1))
        if hasattr(self, "shape_start_spin") and hasattr(self, "shape_end_spin"):
            for spin in (self.shape_start_spin, self.shape_end_spin):
                spin.setRange(0, max(0, duration - 1))
            self.shape_end_spin.setValue(max(0, duration - 1))
        self._set_loop_range_limits(duration)
        self._set_scale_bar_frame_limit(duration)
        self.preview_frame_label.setText(f"{self.preview_slider.value() + 1 if duration else 0} / {duration}")
        self._update_split_status()

    def _set_scale_bar_frame_limit(self, duration: int) -> None:
        if not hasattr(self, "scale_bar_frames_spin"):
            return
        duration = max(0, int(duration))
        current = int(self.scale_bar_frames_spin.value())
        self.scale_bar_frames_spin.blockSignals(True)
        try:
            if duration <= 0:
                self.scale_bar_frames_spin.setRange(0, 0)
                self.scale_bar_frames_spin.setValue(0)
            else:
                self.scale_bar_frames_spin.setRange(1, duration)
                self.scale_bar_frames_spin.setValue(max(1, min(duration, current if current > 0 else min(10, duration))))
        finally:
            self.scale_bar_frames_spin.blockSignals(False)

    def _scale_bar_visible_for_output(self, output_index: int) -> bool:
        if not hasattr(self, "scale_bar_check") or not self.scale_bar_check.isChecked():
            return False
        if not hasattr(self, "scale_bar_frames_spin"):
            return True
        try:
            frame_count = int(self.scale_bar_frames_spin.value())
        except Exception:
            frame_count = 0
        return frame_count > 0 and int(output_index) < frame_count

    def _set_loop_range_limits(self, duration: int) -> None:
        if not hasattr(self, "loop_start_spin") or not hasattr(self, "loop_end_spin"):
            return
        duration = max(0, int(duration))
        old_start = int(self.loop_start_spin.value())
        old_end = int(self.loop_end_spin.value())
        self.loop_start_spin.blockSignals(True)
        self.loop_end_spin.blockSignals(True)
        try:
            if duration <= 0:
                self.loop_start_spin.setRange(0, 0)
                self.loop_end_spin.setRange(0, 0)
                self.loop_start_spin.setValue(0)
                self.loop_end_spin.setValue(0)
                return

            self.loop_start_spin.setRange(1, duration)
            self.loop_end_spin.setRange(1, duration)
            start = old_start if old_start > 0 else 1
            end = old_end if old_end > 0 else duration
            start = max(1, min(duration, start))
            end = max(1, min(duration, end))
            if start > end:
                end = start
            self.loop_start_spin.setValue(start)
            self.loop_end_spin.setValue(end)
        finally:
            self.loop_start_spin.blockSignals(False)
            self.loop_end_spin.blockSignals(False)

    def _loop_range_changed(self, *_args) -> None:
        if not hasattr(self, "loop_start_spin") or not hasattr(self, "loop_end_spin"):
            return
        duration = len(self._get_timeline())
        if duration <= 0:
            return
        start = int(self.loop_start_spin.value())
        end = int(self.loop_end_spin.value())
        if start <= end:
            return
        sender = self.sender()
        self.loop_start_spin.blockSignals(True)
        self.loop_end_spin.blockSignals(True)
        try:
            if sender is self.loop_start_spin:
                self.loop_end_spin.setValue(start)
            else:
                self.loop_start_spin.setValue(end)
        finally:
            self.loop_start_spin.blockSignals(False)
            self.loop_end_spin.blockSignals(False)

    def _loop_range_indices(self) -> Tuple[int, int]:
        duration = len(self._get_timeline())
        if duration <= 0 or not hasattr(self, "loop_start_spin") or not hasattr(self, "loop_end_spin"):
            return 0, 0
        start = max(0, min(duration - 1, int(self.loop_start_spin.value()) - 1))
        end = max(0, min(duration - 1, int(self.loop_end_spin.value()) - 1))
        if start > end:
            start, end = end, start
        return start, end

    def _preview_playback_bounds(self) -> Tuple[int, int]:
        duration = len(self._get_timeline())
        if duration <= 0:
            return 0, 0
        if hasattr(self, "movie_preview_loop_check") and self.movie_preview_loop_check.isChecked():
            return self._loop_range_indices()
        return 0, duration - 1

    def _set_loop_range_values(self, start_index: int, end_index: int) -> None:
        duration = len(self._get_timeline())
        if duration <= 0 or not hasattr(self, "loop_start_spin") or not hasattr(self, "loop_end_spin"):
            return
        start = max(0, min(duration - 1, int(start_index)))
        end = max(0, min(duration - 1, int(end_index)))
        if start > end:
            start, end = end, start
        self.loop_start_spin.blockSignals(True)
        self.loop_end_spin.blockSignals(True)
        try:
            self.loop_start_spin.setValue(start + 1)
            self.loop_end_spin.setValue(end + 1)
        finally:
            self.loop_start_spin.blockSignals(False)
            self.loop_end_spin.blockSignals(False)

    def _set_loop_start_to_current(self) -> None:
        duration = len(self._get_timeline())
        if duration <= 0:
            return
        current = max(0, min(duration - 1, int(self.preview_slider.value())))
        _start, end = self._loop_range_indices()
        self._set_loop_range_values(current, max(current, end))
        self.movie_preview_loop_check.setChecked(True)

    def _set_loop_end_to_current(self) -> None:
        duration = len(self._get_timeline())
        if duration <= 0:
            return
        current = max(0, min(duration - 1, int(self.preview_slider.value())))
        start, _end = self._loop_range_indices()
        self._set_loop_range_values(min(start, current), current)
        self.movie_preview_loop_check.setChecked(True)

    def _set_loop_range_to_selected_transition(self) -> None:
        idx = self._selected_clip_index()
        if idx is None or not (0 <= idx < len(self.clips)):
            self._update_split_status("Select a clip first")
            return
        transition_range = self._transition_range_for_clip(idx, "out")
        if transition_range is None:
            self._update_split_status("No active transition")
            return
        duration = len(self._get_timeline())
        if duration <= 0:
            return
        start, end = transition_range
        pad = 5
        start = max(0, int(start) - pad)
        end = min(duration - 1, int(end) + pad)
        self._set_loop_range_values(start, end)
        self.movie_preview_loop_check.setChecked(True)
        if self.preview_slider.value() == start:
            self._update_preview()
        else:
            self.preview_slider.setValue(start)
        self._update_split_status(f"Loop range: {start + 1} - {end + 1}")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _default_fps(self) -> float:
        if gv is not None:
            try:
                fps = float(getattr(gv, "FrameRate", 0) or 0)
                if fps > 0:
                    return fps
            except Exception:
                pass
            try:
                frame_time = float(getattr(gv, "FrameTime", 0) or 0)
                if frame_time > 0:
                    return max(0.1, min(240.0, 1000.0 / frame_time))
            except Exception:
                pass
        return 20.0

    def _default_output_size(self) -> Tuple[int, int]:
        width = 512
        height = 512
        if gv is not None:
            try:
                dspsize = getattr(gv, "dspsize", None)
                if dspsize and len(dspsize) >= 2:
                    width = int(dspsize[0])
                    height = int(dspsize[1])
                    return max(2, min(4096, self._make_even(width))), max(2, min(4096, self._make_even(height)))
            except Exception:
                pass
        if gv is not None:
            try:
                width = int(getattr(gv, "XPixel", width) or width)
                height = int(getattr(gv, "YPixel", height) or height)
            except Exception:
                pass
        try:
            if self.main_window is not None and hasattr(self.main_window, "_calculate_display_dimensions"):
                display_w, display_h, _scale = self.main_window._calculate_display_dimensions(width, height)
                width = int(display_w)
                height = int(display_h)
        except Exception:
            pass
        width = max(2, min(4096, self._make_even(width)))
        height = max(2, min(4096, self._make_even(height)))
        return width, height

    def _output_size(self) -> Tuple[int, int]:
        width = self._make_even(int(self.width_spin.value()))
        height = self._make_even(int(self.height_spin.value()))
        return max(2, width), max(2, height)

    def _make_even(self, value: int) -> int:
        value = int(value)
        return value if value % 2 == 0 else value + 1

    def _get_frame_array(self, frame_index: int) -> Optional[np.ndarray]:
        channel = self.channel_combo.currentText() if hasattr(self, "channel_combo") else "1ch"
        if self.main_window is not None and hasattr(self.main_window, "getImageDataForFrame"):
            getter = getattr(self.main_window, "getImageDataForFrame")
            try:
                return getter(int(frame_index), channel=channel)
            except TypeError:
                try:
                    return getter(int(frame_index))
                except Exception:
                    pass
            except Exception:
                pass

        if gv is None or LoadFrame is None:
            return None
        try:
            files = getattr(gv, "files", None)
            file_idx = int(getattr(gv, "currentFileNum", 0) or 0)
            original_index = int(getattr(gv, "index", 0) or 0)
            if not files or not (0 <= file_idx < len(files)):
                return None
            gv.index = int(frame_index)
            LoadFrame(files[file_idx])
            if InitializeAryDataFallback is not None:
                InitializeAryDataFallback()
            if channel == "2ch" and getattr(gv, "aryData2ch", None) is not None:
                array = np.asarray(gv.aryData2ch).copy()
            else:
                array = np.asarray(gv.aryData).copy()
            gv.index = original_index
            return array
        except Exception:
            return None

    def _render_pynud_display_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Render one source frame through pyNuD's main display pipeline.

        This keeps Color/Tone, gamma, LUT, display aspect ratio, and active
        preprocessing consistent with the main Image View / Movie output.
        """
        if cv2 is None or gv is None or LoadFrame is None or self.main_window is None:
            return None
        try:
            files = getattr(gv, "files", None)
            file_idx = int(getattr(gv, "currentFileNum", 0) or 0)
            if not files or not (0 <= file_idx < len(files)):
                return None

            gv.index = int(frame_index)
            LoadFrame(files[file_idx])
            if InitializeAryDataFallback is not None:
                InitializeAryDataFallback()

            if hasattr(gv, "aryData") and gv.aryData is not None:
                gv.aryData_processed_1ch = gv.aryData.copy()
            else:
                gv.aryData_processed_1ch = None
            if hasattr(gv, "aryData2ch") and gv.aryData2ch is not None:
                gv.aryData_processed_2ch = gv.aryData2ch.copy()
            else:
                gv.aryData_processed_2ch = None

            if hasattr(self.main_window, "applyImageProcessing"):
                self.main_window.applyImageProcessing(hidden=True)

            channel = self.channel_combo.currentText() if hasattr(self, "channel_combo") else "1ch"
            if channel == "2ch":
                if hasattr(self.main_window, "UpdateDisplayImage2ch"):
                    if getattr(gv, "dspsize", None) is None and hasattr(self.main_window, "UpdateDisplayImage"):
                        self.main_window.UpdateDisplayImage()
                    self.main_window.UpdateDisplayImage2ch()
                img = getattr(gv, "dspimg2ch", None)
            else:
                if hasattr(self.main_window, "UpdateDisplayImage"):
                    self.main_window.UpdateDisplayImage()
                img = getattr(gv, "dspimg", None)

            if img is None:
                return None
            return np.ascontiguousarray(img.copy())
        except Exception as exc:
            print(f"[WARNING] Failed to render pyNuD display frame {frame_index}: {exc}")
            return None

    def _restore_main_frame(self, frame_index: Optional[int] = None) -> None:
        if gv is None:
            return
        try:
            files = getattr(gv, "files", None)
            file_idx = int(getattr(gv, "currentFileNum", 0) or 0)
            if frame_index is not None:
                total = max(1, int(getattr(gv, "FrameNum", 1) or 1))
                gv.index = max(0, min(total - 1, int(frame_index)))
            if LoadFrame is not None and files and 0 <= file_idx < len(files):
                LoadFrame(files[file_idx])
                if InitializeAryDataFallback is not None:
                    InitializeAryDataFallback()
            if self.main_window is not None and hasattr(self.main_window, "updateFrame"):
                self.main_window.updateFrame()
        except Exception:
            pass

    def _array_to_bgr(self, array: Any, width: int, height: int, for_thumbnail: bool = False) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is not available.")

        arr = np.asarray(array)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.dtype != np.uint8:
                arr = self._normalize_to_uint8(arr[:, :, :3])
            else:
                arr = arr[:, :, :3].copy()
            if arr.shape[2] == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                bgr = arr
        else:
            gray = self._normalize_to_uint8(arr, for_thumbnail=for_thumbnail)
            if gv is not None and getattr(gv, "color_lut", None) is not None:
                bgr = cv2.applyColorMap(gray, gv.color_lut)
            else:
                cmap_name = self.colormap_combo.currentText() if hasattr(self, "colormap_combo") else "Hot"
                cmap_attr = COLOR_MAPS.get(cmap_name)
                if cmap_attr is None:
                    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    cmap = getattr(cv2, cmap_attr, cv2.COLORMAP_HOT)
                    bgr = cv2.applyColorMap(gray, cmap)

        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(bgr)

    def _normalize_to_uint8(self, array: Any, for_thumbnail: bool = False) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim == 3:
            finite = arr[np.isfinite(arr)]
        else:
            finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(arr.shape[:2], dtype=np.uint8)

        lock_contrast = hasattr(self, "lock_contrast_check") and self.lock_contrast_check.isChecked()
        if not for_thumbnail and lock_contrast:
            lo, hi = self._fixed_contrast_limits()
        else:
            lo, hi = self._percentile_limits(arr)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
        if hi <= lo:
            return np.zeros(arr.shape[:2], dtype=np.uint8)
        scaled = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255)
        if scaled.ndim == 3 and scaled.shape[2] >= 3:
            return scaled[:, :, :3].astype(np.uint8)
        return scaled.astype(np.uint8)

    def _percentile_limits(self, array: Any) -> Tuple[float, float]:
        arr = np.asarray(array, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0, 1.0
        if finite.size < 20:
            return float(np.nanmin(finite)), float(np.nanmax(finite))
        return float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))

    def _fixed_contrast_limits(self) -> Tuple[float, float]:
        if self._contrast_limits is not None:
            return self._contrast_limits
        first_array = None
        for clip in self.clips:
            if clip.kind == "source":
                first_array = self._get_frame_array(clip.start)
                break
        if first_array is None:
            self._contrast_limits = (0.0, 1.0)
        else:
            self._contrast_limits = self._percentile_limits(first_array)
        return self._contrast_limits

    def _default_title_font_size(self) -> int:
        _width, height = self._output_size()
        return max(18, min(96, int(height * 0.07)))

    def _first_source_frame_index(self) -> Optional[int]:
        for clip in self.clips:
            if clip.kind == "source":
                return max(0, min(self.total_frames - 1, int(clip.start)))
        if self.total_frames > 0:
            return 0
        return None

    def _title_background_bgr(self, clip: ClipSpec, width: int, height: int) -> np.ndarray:
        color = QtGui.QColor(getattr(clip, "title_background_color", QtGui.QColor(32, 34, 38)))
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (color.blue(), color.green(), color.red())
        if not bool(getattr(clip, "title_use_first_frame_background", False)) or cv2 is None:
            return img

        frame_idx = self._first_source_frame_index()
        if frame_idx is None:
            return img
        frame = self._render_pynud_display_frame(frame_idx)
        if frame is None:
            array = self._get_frame_array(frame_idx)
            if array is not None:
                frame = self._array_to_bgr(array, width, height)
        if frame is None:
            return img
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        alpha = max(0.0, min(1.0, float(getattr(clip, "title_background_frame_opacity", 35)) / 100.0))
        return cv2.addWeighted(np.ascontiguousarray(img), 1.0 - alpha, np.ascontiguousarray(frame), alpha, 0)

    def _title_overlay_spec(self, clip: ClipSpec) -> TextOverlaySpec:
        default_size = self._default_title_font_size()
        try:
            font_size = int(getattr(clip, "title_font_size", default_size) or default_size)
        except Exception:
            font_size = default_size
        try:
            x_ratio = float(getattr(clip, "title_x_ratio", 0.5))
        except Exception:
            x_ratio = 0.5
        try:
            y_ratio = float(getattr(clip, "title_y_ratio", 0.5))
        except Exception:
            y_ratio = 0.5
        return TextOverlaySpec(
            text=str(getattr(clip, "title", "") or "Title"),
            start=0,
            end=0,
            position=str(getattr(clip, "title_position", "Center") or "Center"),
            font_size=max(8, font_size),
            color=QtGui.QColor(getattr(clip, "title_color", QtGui.QColor(255, 255, 255))),
            background=bool(getattr(clip, "title_background", False)),
            font_family=str(getattr(clip, "title_font_family", "Arial") or "Arial"),
            font_style=self._normalize_font_style(str(getattr(clip, "title_font_style", "Bold") or "Bold")),
            text_align=str(getattr(clip, "title_align", "Center") or "Center"),
            custom_position=bool(getattr(clip, "title_custom_position", False)),
            x_ratio=x_ratio,
            y_ratio=y_ratio,
        )

    def _draw_title_text(self, bgr: np.ndarray, clip: ClipSpec) -> np.ndarray:
        return self._draw_text_overlay(bgr, self._title_overlay_spec(clip))

    def _draw_trimmed_clip_overlay(self, bgr: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            gray = np.full_like(np.ascontiguousarray(bgr), 150)
            bgr = cv2.addWeighted(np.ascontiguousarray(bgr), 0.35, gray, 0.65, 0)
        h, w = bgr.shape[:2]
        font_size = max(18, int(min(w, h) * 0.12))
        return self._draw_text_qt(
            bgr,
            "TRIMMED",
            "Center",
            font_size,
            QtGui.QColor(255, 255, 255),
            background=True,
        )

    def _render_clip_frame(self, clip_idx: int, local_index: int, draw_scale_bar: bool = True) -> Tuple[np.ndarray, Optional[int]]:
        width, height = self._output_size()
        clip = self.clips[clip_idx]
        if clip.kind == "title":
            img = self._title_background_bgr(clip, width, height)
            img = self._draw_title_text(img, clip)
            if bool(getattr(clip, "trimmed", False)):
                img = self._draw_trimmed_clip_overlay(img)
            return img, None

        frame_idx = max(0, min(self.total_frames - 1, clip.start + local_index))
        img = self._render_pynud_display_frame(frame_idx)
        if img is None:
            array = self._get_frame_array(frame_idx)
            if array is not None:
                img = self._array_to_bgr(array, width, height)

        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img = self._draw_text_qt(
                img,
                f"Frame {frame_idx + 1} unavailable",
                "Center",
                24,
                QtGui.QColor(255, 255, 255),
                background=True,
            )
        else:
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = self._draw_standard_overlays(img, frame_idx, draw_scale_bar=draw_scale_bar)
        if bool(getattr(clip, "trimmed", False)):
            img = self._draw_trimmed_clip_overlay(img)
        return img, frame_idx

    def _render_timeline_frame(self, output_index: int) -> np.ndarray:
        timeline = self._get_timeline()
        width, height = self._output_size()
        if not timeline:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            return self._draw_text_qt(img, "No clips", "Center", 28, QtGui.QColor(255, 255, 255), True)

        output_index = max(0, min(len(timeline) - 1, int(output_index)))
        ref = timeline[output_index]
        draw_scale_bar = self._scale_bar_visible_for_output(output_index)
        skip_post_overlays = False
        if ref["kind"] == "clip":
            img, _frame_idx = self._render_clip_frame(int(ref["clip"]), int(ref["local"]), draw_scale_bar=draw_scale_bar)
        else:
            effect = str(ref["effect"])
            phase = str(ref.get("phase", ""))
            alpha = float(ref["alpha"])
            if effect == "Fade Black" and phase == "hold_black":
                out_w, out_h = self._output_size()
                img = np.zeros((max(1, int(out_h)), max(1, int(out_w)), 3), dtype=np.uint8)
                skip_post_overlays = True
            elif effect == "Fade Black" and phase == "fade_out":
                img_a, _frame_a = self._render_clip_frame(int(ref["a_clip"]), int(ref["a_local"]), draw_scale_bar=draw_scale_bar)
                img = self._fade_image_to_black(img_a, alpha)
            elif effect == "Fade Black" and phase == "fade_in":
                img_b, _frame_b = self._render_clip_frame(int(ref["b_clip"]), int(ref["b_local"]), draw_scale_bar=draw_scale_bar)
                img = self._fade_image_from_black(img_b, alpha)
            else:
                img_a, _frame_a = self._render_clip_frame(int(ref["a_clip"]), int(ref["a_local"]), draw_scale_bar=draw_scale_bar)
                img_b, _frame_b = self._render_clip_frame(int(ref["b_clip"]), int(ref["b_local"]), draw_scale_bar=draw_scale_bar)
                img = self._apply_transition(img_a, img_b, effect, alpha)

        if not skip_post_overlays:
            img = self._draw_z_scale_bar(img)
            img = self._draw_user_overlays(img, output_index)
        return np.ascontiguousarray(img)

    def _fade_image_to_black(self, img: np.ndarray, alpha: float) -> np.ndarray:
        alpha = max(0.0, min(1.0, float(alpha)))
        return np.clip(img.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)

    def _fade_image_from_black(self, img: np.ndarray, alpha: float) -> np.ndarray:
        alpha = max(0.0, min(1.0, float(alpha)))
        return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def _apply_transition(self, img_a: np.ndarray, img_b: np.ndarray, effect: str, alpha: float) -> np.ndarray:
        if cv2 is None:
            return img_b
        alpha = max(0.0, min(1.0, float(alpha)))
        if effect == "Wipe Left":
            out = img_a.copy()
            split = int(round(out.shape[1] * alpha))
            if split > 0:
                out[:, :split] = img_b[:, :split]
            return out
        if effect == "Fade Black":
            if alpha < 0.5:
                a = 1.0 - alpha * 2.0
                return np.clip(img_a.astype(np.float32) * a, 0, 255).astype(np.uint8)
            b = (alpha - 0.5) * 2.0
            return np.clip(img_b.astype(np.float32) * b, 0, 255).astype(np.uint8)
        return cv2.addWeighted(img_a, 1.0 - alpha, img_b, alpha, 0)

    def _draw_standard_overlays(self, img: np.ndarray, frame_idx: int, draw_scale_bar: bool = True) -> np.ndarray:
        if gv is not None:
            old_index = getattr(gv, "index", 0)
            old_time = getattr(gv, "showTimeFlag", False)
            old_scale = getattr(gv, "showScaleFlag", False)
            try:
                gv.index = int(frame_idx)
                gv.showTimeFlag = bool(self.time_check.isChecked())
                gv.showScaleFlag = bool(draw_scale_bar and self.scale_bar_check.isChecked())
                if self.main_window is not None:
                    if hasattr(self.main_window, "drawTimeCaption"):
                        img = self.main_window.drawTimeCaption(img)
                    if hasattr(self.main_window, "drawScaleCaption"):
                        img = self.main_window.drawScaleCaption(img)
            finally:
                gv.index = old_index
                gv.showTimeFlag = old_time
                gv.showScaleFlag = old_scale

        if self.frame_no_check.isChecked():
            img = self._draw_frame_number(img, frame_idx)
        return img

    def _clear_plugin_colorbar_render_meta(self) -> None:
        self._plugin_colorbar_render_rect = None
        self._plugin_colorbar_render_frame_size = None
        self._plugin_colorbar_render_orientation = None

    def _set_plugin_colorbar_render_meta(
        self,
        buf_w: int,
        buf_h: int,
        cb_x: int,
        cb_y: int,
        cb_w: int,
        cb_h: int,
        area_w: int,
        area_h: int,
        orientation: str,
    ) -> None:
        if orientation == "Vertical":
            self._plugin_colorbar_render_rect = (int(buf_w + cb_x), int(cb_y), int(cb_w), int(cb_h))
            self._plugin_colorbar_render_frame_size = (int(buf_w + area_w), int(buf_h))
        else:
            self._plugin_colorbar_render_rect = (int(cb_x), int(buf_h + cb_y), int(cb_w), int(cb_h))
            self._plugin_colorbar_render_frame_size = (int(buf_w), int(buf_h + area_h))
        self._plugin_colorbar_render_orientation = orientation

    def _plugin_colorbar_target_window(self):
        if self.main_window is None:
            return None
        if self.channel_combo.currentText() == "2ch":
            return getattr(self.main_window, "image_window_2ch", None)
        return getattr(self.main_window, "image_window", None)

    def _plugin_colorbar_target_size(self, orientation: str) -> Tuple[int, int]:
        if orientation == "Vertical":
            return (
                max(20, int(getattr(gv, "colorbar_dock_vertical_w", 80))),
                max(40, int(getattr(gv, "colorbar_dock_vertical_h", 200))),
            )
        return (
            max(60, int(getattr(gv, "colorbar_dock_horizontal_w", 180))),
            max(20, int(getattr(gv, "colorbar_dock_horizontal_h", 80))),
        )

    def _color_bar_render_font_scale(self, orientation: str) -> float:
        target_w, target_h = self._plugin_colorbar_target_size(orientation)
        ref_w, ref_h = self._color_bar_ref_size.get(orientation, (80, 200))
        scale_w = target_w / float(max(1, ref_w))
        scale_h = target_h / float(max(1, ref_h))
        scale = (scale_w + scale_h) / 2.0
        return max(0.35, min(2.5, scale))

    def _scaled_color_bar_font_size(self, base_size: int, orientation: str) -> int:
        return max(6, int(round(max(6, int(base_size)) * self._color_bar_render_font_scale(orientation))))

    def _plugin_colorbar_scale_min_size(self, orientation: str, target_w: int, target_h: int) -> Tuple[int, int]:
        values_spec = self._color_bar_text_style.get("values", {})
        font = self._color_bar_font(values_spec, orientation)
        font_metrics = QtGui.QFontMetrics(font)
        text_w = font_metrics.horizontalAdvance("-00.0") + 18
        target_w = max(1, int(target_w))
        target_h = max(1, int(target_h))
        if orientation == "Vertical":
            colorbar_min_w = 12
            scale_min_w = max(24, text_w)
            if scale_min_w + colorbar_min_w + 4 > target_w:
                scale_min_w = max(16, target_w - colorbar_min_w - 4)
            scale_min_h = max(30, min(target_h - 20, 40))
            return scale_min_w, scale_min_h
        scale_min_w = max(60, min(target_w - 20, text_w + 40))
        scale_min_h = max(20, text_w // 2 + 8)
        if scale_min_h + 16 > target_h:
            scale_min_h = max(16, target_h - 16)
        return scale_min_w, scale_min_h

    def _apply_plugin_colorbar_layout_mins(self, cbw, orientation: str, target_w: int, target_h: int) -> None:
        if cbw is None:
            return
        scale = getattr(cbw, "scale_widget", None)
        colorbar_label = getattr(cbw, "colorbar_label", None)
        if scale is None:
            return
        scale_min_w, scale_min_h = self._plugin_colorbar_scale_min_size(orientation, target_w, target_h)
        scale.setMinimumSize(scale_min_w, scale_min_h)
        scale.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        if colorbar_label is not None:
            if orientation == "Vertical":
                colorbar_label.setMinimumSize(12, scale_min_h)
            else:
                colorbar_label.setMinimumSize(max(60, min(target_w, 120)), 12)
        bar_container = scale.parent()
        bar_layout = bar_container.layout() if bar_container is not None else None
        if bar_layout is not None:
            bar_layout.setStretch(0, 2)
            bar_layout.setStretch(1, 1)
            bar_layout.activate()
        if orientation == "Vertical":
            cbw.setMinimumSize(max(36, scale_min_w + 16), max(40, target_h))
        else:
            cbw.setMinimumSize(max(80, target_w), max(32, scale_min_h + 20))

    def _pixmap_to_bgr(self, pixmap: QtGui.QPixmap) -> Optional[np.ndarray]:
        if cv2 is None or pixmap is None or pixmap.isNull():
            return None
        try:
            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QIODevice.ReadWrite)
            if not pixmap.save(buffer, "PNG"):
                return None
            data = bytes(buffer.data())
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return None
            return np.ascontiguousarray(arr)
        except Exception:
            return None

    def _render_colorbar_widget_to_bgr(self, widget, target_w: int, target_h: int, orientation: str = "Vertical") -> Optional[np.ndarray]:
        if cv2 is None or widget is None:
            return None
        try:
            target_w = max(1, int(target_w))
            target_h = max(1, int(target_h))
            self._apply_plugin_colorbar_layout_mins(widget, orientation, target_w, target_h)
            widget.setFixedSize(target_w, target_h)
            widget.updateGeometry()
            layout = widget.layout()
            if layout is not None:
                layout.activate()
            scale = getattr(widget, "scale_widget", None)
            bar_container = scale.parent() if scale is not None else None
            bar_layout = bar_container.layout() if bar_container is not None else None
            if bar_layout is not None:
                bar_layout.activate()
            if scale is not None:
                scale.updateGeometry()
                scale.repaint()
            widget.repaint()
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents(QtCore.QEventLoop.AllEvents, 30)
            pixmap = QtGui.QPixmap(target_w, target_h)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            widget.render(painter)
            painter.end()
            widget.setMinimumSize(0, 0)
            widget.setMaximumSize(16777215, 16777215)
            widget.resize(target_w, target_h)
            return self._pixmap_to_bgr(pixmap)
        except Exception:
            return None

    def _plugin_colorbar_widget(self, target_window, orientation: str):
        if target_window is None or not hasattr(target_window, "_create_colorbar_widget"):
            return None
        cache_key = (id(target_window), orientation)
        cbw = self._plugin_colorbar_widget_cache.get(cache_key)
        if cbw is not None:
            return cbw
        orig_orient = getattr(target_window, "colorbar_orientation", "Vertical")
        try:
            target_window.colorbar_orientation = orientation
            cbw = target_window._create_colorbar_widget(mode="Docking")
            if orientation == "Vertical":
                cbw.setMinimumSize(32, 80)
                cbw.setMaximumSize(16777215, 16777215)
                cbw.resize(
                    int(getattr(gv, "colorbar_dock_vertical_w", 80)),
                    int(getattr(gv, "colorbar_dock_vertical_h", 200)),
                )
            else:
                cbw.setMinimumSize(80, 32)
                cbw.setMaximumSize(16777215, 16777215)
                cbw.resize(
                    int(getattr(gv, "colorbar_dock_horizontal_w", 180)),
                    int(getattr(gv, "colorbar_dock_horizontal_h", 80)),
                )
        finally:
            target_window.colorbar_orientation = orig_orient
        self._plugin_colorbar_widget_cache[cache_key] = cbw
        self._customize_plugin_colorbar_widget(cbw, target_window, orientation)
        return cbw

    def _ensure_plugin_colorbar_scale_widget(self, cbw, target_window, orientation: str) -> None:
        if cbw is None or target_window is None:
            return
        scale = getattr(cbw, "scale_widget", None)
        if isinstance(scale, PluginMovieScaleWidget):
            scale.orientation = orientation
            scale.movie_editor = self
            scale.parent_window = target_window
            return
        old_scale = scale
        if old_scale is None:
            return
        bar_container = old_scale.parent()
        bar_layout = bar_container.layout() if bar_container is not None else None
        if bar_layout is None:
            return
        idx = bar_layout.indexOf(old_scale)
        scale_min_w, scale_min_h = self._plugin_colorbar_scale_min_size(
            orientation,
            *self._plugin_colorbar_target_size(orientation),
        )
        new_scale = PluginMovieScaleWidget(self, target_window, orientation, "Docking", bar_container)
        new_scale.setMinimumSize(scale_min_w, scale_min_h)
        new_scale.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        bar_layout.insertWidget(max(0, idx), new_scale)
        bar_layout.removeWidget(old_scale)
        old_scale.deleteLater()
        cbw.scale_widget = new_scale

    def _sync_plugin_colorbar_text_styles(self, cbw, target_window, orientation: str) -> None:
        if cbw is None or target_window is None:
            return
        self._sync_color_bar_text_style_from_gv()
        self._ensure_plugin_colorbar_scale_widget(cbw, target_window, orientation)
        self._apply_plugin_colorbar_unit_style(cbw, orientation)
        scale = getattr(cbw, "scale_widget", None)
        if scale is not None:
            scale.update()
            scale.repaint()
        cbw.update()

    def _refresh_plugin_colorbar_text_styles_for_view(self, part: str = "") -> None:
        target_window = self._plugin_colorbar_target_window()
        if target_window is None:
            return
        orientation = self._color_bar_orientation()
        cbw = self._plugin_colorbar_widget(target_window, orientation)
        if cbw is None:
            return
        if part == "unit":
            cbw._plugin_unit_position = None
        self._sync_plugin_colorbar_text_styles(cbw, target_window, orientation)

    def _opencv_zscale_style_overrides(self, orientation: str) -> Dict[str, Any]:
        unit_spec = self._color_bar_text_style.get("unit", {})
        values_spec = self._color_bar_text_style.get("values", {})
        unit_px = self._scaled_color_bar_font_size(int(unit_spec.get("font_size", 8)), orientation)
        tick_px = self._scaled_color_bar_font_size(int(values_spec.get("font_size", 8)), orientation)
        return {
            "movie_zscale_unit_fs": max(0.2, unit_px / 22.86),
            "movie_zscale_tick_fs": max(0.2, tick_px / 26.67),
        }

    def _apply_gv_style_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        saved: Dict[str, Any] = {}
        if gv is None:
            return saved
        for key, value in overrides.items():
            saved[key] = getattr(gv, key, None)
            setattr(gv, key, value)
        return saved

    def _restore_gv_style_overrides(self, saved: Dict[str, Any]) -> None:
        if gv is None:
            return
        for key, value in saved.items():
            setattr(gv, key, value)

    def _color_bar_font(self, spec: Dict[str, Any], orientation: Optional[str] = None) -> QtGui.QFont:
        font = QtGui.QFont(str(spec.get("font_family", "Arial")))
        base_size = max(6, int(spec.get("font_size", 8)))
        if orientation:
            base_size = self._scaled_color_bar_font_size(base_size, orientation)
        font.setPixelSize(base_size)
        style = self._normalize_font_style(str(spec.get("font_style", "Normal")))
        font.setBold("Bold" in style)
        font.setItalic("Italic" in style)
        return font

    def _color_bar_align_for_position(self, align: str) -> QtCore.Qt.Alignment:
        value = str(align or "Center")
        if value == "Left":
            return QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        if value == "Right":
            return QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        return QtCore.Qt.AlignCenter

    def _apply_plugin_colorbar_unit_style(self, cbw, orientation: str) -> None:
        unit_label = getattr(cbw, "unit_label", None)
        if unit_label is None:
            return
        spec = self._color_bar_text_style.get("unit", {})
        unit_label.setFont(self._color_bar_font(spec, orientation))
        color = spec.get("color", QtGui.QColor(255, 255, 255))
        if isinstance(color, QtGui.QColor):
            unit_label.setStyleSheet(f"color: rgb({color.red()}, {color.green()}, {color.blue()});")
        unit_label.setAlignment(self._color_bar_align_for_position(str(spec.get("align", "Center"))))
        position = str(spec.get("position", "Top"))
        if getattr(cbw, "_plugin_unit_position", None) == position:
            return
        bar_container = cbw.scale_widget.parent() if getattr(cbw, "scale_widget", None) is not None else None
        main_layout = cbw.layout()
        if main_layout is None or bar_container is None:
            cbw._plugin_unit_position = position
            return
        main_layout.removeWidget(unit_label)
        main_layout.removeWidget(bar_container)
        if position == "Bottom":
            main_layout.addWidget(bar_container)
            main_layout.addWidget(unit_label)
        else:
            main_layout.addWidget(unit_label)
            main_layout.addWidget(bar_container)
        cbw._plugin_unit_position = position

    def _customize_plugin_colorbar_widget(self, cbw, target_window, orientation: str) -> None:
        if cbw is None or target_window is None:
            return
        self._sync_plugin_colorbar_text_styles(cbw, target_window, orientation)
        target_w, target_h = self._plugin_colorbar_target_size(orientation)
        cbw._plugin_orientation = orientation
        self._apply_plugin_colorbar_layout_mins(cbw, orientation, target_w, target_h)

    def _prepare_plugin_colorbar_for_render(self, cbw, orientation: str, target_w: int, target_h: int) -> None:
        if cbw is None:
            return
        try:
            target_w = max(1, int(target_w))
            target_h = max(1, int(target_h))
            target_window = self._plugin_colorbar_target_window()
            if target_window is not None:
                self._sync_plugin_colorbar_text_styles(cbw, target_window, orientation)
            cbw._plugin_orientation = orientation
            self._apply_plugin_colorbar_layout_mins(cbw, orientation, target_w, target_h)
            cbw.resize(target_w, target_h)
            cbw.updateGeometry()
            layout = cbw.layout()
            if layout is not None:
                layout.activate()
            scale = getattr(cbw, "scale_widget", None)
            bar_container = scale.parent() if scale is not None else None
            bar_layout = bar_container.layout() if bar_container is not None else None
            if bar_layout is not None:
                bar_layout.activate()
            if scale is not None:
                scale.updateGeometry()
                scale.repaint()
        except Exception:
            pass

    def _sync_color_bar_text_style_from_gv(self) -> None:
        if gv is None:
            return
        shared = getattr(gv, "colorbar_text_style", None)
        if not isinstance(shared, dict):
            return
        for part in ("unit", "values"):
            part_spec = shared.get(part)
            if isinstance(part_spec, dict):
                self._color_bar_text_style.setdefault(part, {}).update(part_spec)

    def _sync_color_bar_text_style_to_gv(self) -> None:
        if gv is None:
            return
        if not isinstance(getattr(gv, "colorbar_text_style", None), dict):
            gv.colorbar_text_style = {}
        for part in ("unit", "values"):
            spec = self._color_bar_text_style.get(part, {})
            gv.colorbar_text_style[part] = dict(spec)

    def _plugin_colorbar_z_range(self) -> Tuple[float, float]:
        channel = self.channel_combo.currentText() if hasattr(self, "channel_combo") else "1ch"
        if self.main_window is not None and hasattr(self.main_window, "_get_z_range_for_movie"):
            try:
                return self.main_window._get_z_range_for_movie(channel)
            except Exception:
                pass
        return 0.0, 100.0

    def _invalidate_plugin_colorbar_widgets(self) -> None:
        self._plugin_colorbar_widget_cache.clear()

    def _color_bar_value_positions(self) -> List[str]:
        if self._color_bar_orientation() == "Horizontal":
            return list(COLOR_BAR_VALUE_POSITIONS_HORIZONTAL)
        return list(COLOR_BAR_VALUE_POSITIONS_VERTICAL)

    def _default_color_bar_value_position(self) -> str:
        if self._color_bar_orientation() == "Horizontal":
            return "Below"
        return "Left"

    def _refresh_plugin_colorbar_widget(self, target_window, cbw, orientation: str, channel: str) -> None:
        if cbw is None or target_window is None:
            return
        if orientation == "Vertical":
            cbw.setMinimumSize(32, 80)
            cbw.setMaximumSize(16777215, 16777215)
            cbw.resize(
                int(getattr(gv, "colorbar_dock_vertical_w", max(1, cbw.width()))),
                int(getattr(gv, "colorbar_dock_vertical_h", max(1, cbw.height()))),
            )
        else:
            cbw.setMinimumSize(80, 32)
            cbw.setMaximumSize(16777215, 16777215)
            cbw.resize(
                int(getattr(gv, "colorbar_dock_horizontal_w", max(1, cbw.width()))),
                int(getattr(gv, "colorbar_dock_horizontal_h", max(1, cbw.height()))),
            )
        try:
            cbw.updateGeometry()
            layout = cbw.layout()
            if layout is not None:
                layout.activate()
        except Exception:
            pass
        self._customize_plugin_colorbar_widget(cbw, target_window, orientation)
        try:
            if hasattr(target_window, "update_colorbar") and getattr(target_window, "colorbar_widget", None) is cbw:
                target_window.update_colorbar()
            else:
                if hasattr(cbw, "unit_label") and hasattr(target_window, "_get_z_unit_for_window"):
                    cbw.unit_label.setText(target_window._get_z_unit_for_window())
                if hasattr(cbw, "colorbar_label") and hasattr(target_window, "_generate_colorbar_image"):
                    colorbar_image = target_window._generate_colorbar_image()
                    if colorbar_image is not None:
                        h, w, ch = colorbar_image.shape
                        q_img = QtGui.QImage(colorbar_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888).rgbSwapped()
                        cbw.colorbar_label.setPixmap(QtGui.QPixmap.fromImage(q_img))
                if hasattr(cbw, "scale_widget"):
                    cbw.scale_widget.repaint()
                cbw.repaint()
        except Exception:
            pass

    def _append_plugin_colorbar_from_widget(self, buf: np.ndarray, channel: str, orientation: str) -> Optional[np.ndarray]:
        if cv2 is None or gv is None:
            return None
        target_window = self._plugin_colorbar_target_window()
        if target_window is None:
            return None
        cbw = self._plugin_colorbar_widget(target_window, orientation)
        if cbw is None:
            return None
        target_w, target_h = self._plugin_colorbar_target_size(orientation)
        self._refresh_plugin_colorbar_widget(target_window, cbw, orientation, channel)
        self._prepare_plugin_colorbar_for_render(cbw, orientation, target_w, target_h)
        cb_img = self._render_colorbar_widget_to_bgr(cbw, target_w, target_h, orientation)
        if cb_img is None:
            return None
        if cb_img.shape[0] != target_h or cb_img.shape[1] != target_w:
            cb_img = cv2.resize(cb_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        buf_h, buf_w = buf.shape[:2]
        cb_h, cb_w = cb_img.shape[:2]
        default_x = 10
        if orientation == "Vertical":
            area_h = buf_h
            default_y = 10
            saved_x = int(getattr(gv, "colorbar_dock_vertical_x", default_x))
            saved_y = int(getattr(gv, "colorbar_dock_vertical_y", default_y))
            if cb_h > max(1, area_h):
                ratio = max(1, area_h) / float(max(1, cb_h))
                cb_h = area_h
                cb_w = max(16, int(round(cb_w * ratio)))
                cb_img = cv2.resize(cb_img, (cb_w, cb_h), interpolation=cv2.INTER_AREA)
            area_w = max(100, cb_w + max(default_x, saved_x) + default_x)
        else:
            area_w = buf_w
            default_y = max((max(100, cb_h + 20) - cb_h) // 2, 0)
            saved_x = int(getattr(gv, "colorbar_dock_horizontal_x", default_x))
            saved_y = int(getattr(gv, "colorbar_dock_horizontal_y", default_y))
            if cb_w > max(1, area_w):
                ratio = max(1, area_w) / float(max(1, cb_w))
                cb_w = area_w
                cb_h = max(20, int(round(cb_h * ratio)))
                cb_img = cv2.resize(cb_img, (cb_w, cb_h), interpolation=cv2.INTER_AREA)
            area_h = max(100, cb_h + max(default_y, saved_y) + default_y)

        cb_x = max(0, min(saved_x, max(0, area_w - cb_w)))
        cb_y = max(0, min(saved_y, max(0, area_h - cb_h)))
        area_img = np.zeros((max(1, area_h), max(1, area_w), 3), dtype=buf.dtype)
        if cb_img.dtype != buf.dtype:
            cb_img = cb_img.astype(buf.dtype, copy=False)
        area_img[cb_y:cb_y + cb_h, cb_x:cb_x + cb_w] = cb_img
        area_h, area_w = area_img.shape[:2]
        self._set_plugin_colorbar_render_meta(buf_w, buf_h, cb_x, cb_y, cb_w, cb_h, area_w, area_h, orientation)

        if orientation == "Vertical":
            out = np.zeros((buf_h, buf_w + area_w, 3), dtype=buf.dtype)
            out[:, :buf_w] = buf
            out[:, buf_w:buf_w + area_w] = area_img
            return out

        out = np.zeros((buf_h + area_h, buf_w, 3), dtype=buf.dtype)
        out[:buf_h, :buf_w] = buf
        out[buf_h:buf_h + area_h, :area_w] = area_img
        return out

    def _append_plugin_colorbar_opencv(self, buf: np.ndarray, channel: str, orientation: str) -> Optional[np.ndarray]:
        if cv2 is None or gv is None or self.main_window is None:
            return None
        build = getattr(self.main_window, "_build_zscale_block", None)
        if build is None:
            return None
        try:
            z_min, z_max = self.main_window._get_z_range_for_movie(channel)
        except Exception:
            z_min, z_max = 0.0, 100.0
        gv_saved = self._apply_gv_style_overrides(self._opencv_zscale_style_overrides(orientation))
        try:
            block_info = build(buf.shape, orientation, "Docking", z_min, z_max, channel)
        finally:
            self._restore_gv_style_overrides(gv_saved)
        if not block_info:
            return None
        block, background_color = block_info
        buf_h, buf_w = buf.shape[:2]
        block = np.ascontiguousarray(block)
        if block.dtype != buf.dtype:
            block = block.astype(buf.dtype, copy=False)

        default_x = 10
        if orientation == "Vertical":
            target_w = max(16, int(getattr(gv, "colorbar_dock_vertical_w", 80)))
            target_h = max(40, int(getattr(gv, "colorbar_dock_vertical_h", 200)))
            block = cv2.resize(block, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cb_h, cb_w = block.shape[:2]
            area_h = buf_h
            default_y = 10
            saved_x = int(getattr(gv, "colorbar_dock_vertical_x", default_x))
            saved_y = int(getattr(gv, "colorbar_dock_vertical_y", default_y))
            if cb_h > max(1, area_h):
                ratio = max(1, area_h) / float(max(1, cb_h))
                cb_h = area_h
                cb_w = max(16, int(round(cb_w * ratio)))
                block = cv2.resize(block, (cb_w, cb_h), interpolation=cv2.INTER_AREA)
            area_w = max(100, cb_w + max(default_x, saved_x) + default_x)
        else:
            target_w = max(60, int(getattr(gv, "colorbar_dock_horizontal_w", 180)))
            target_h = max(20, int(getattr(gv, "colorbar_dock_horizontal_h", 80)))
            block = cv2.resize(block, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cb_h, cb_w = block.shape[:2]
            area_w = buf_w
            default_y = max((max(100, cb_h + 20) - cb_h) // 2, 0)
            saved_x = int(getattr(gv, "colorbar_dock_horizontal_x", default_x))
            saved_y = int(getattr(gv, "colorbar_dock_horizontal_y", default_y))
            if cb_w > max(1, area_w):
                ratio = max(1, area_w) / float(max(1, cb_w))
                cb_w = area_w
                cb_h = max(20, int(round(cb_h * ratio)))
                block = cv2.resize(block, (cb_w, cb_h), interpolation=cv2.INTER_AREA)
            area_h = max(100, cb_h + max(default_y, saved_y) + default_y)

        cb_x = max(0, min(saved_x, max(0, area_w - cb_w)))
        cb_y = max(0, min(saved_y, max(0, area_h - cb_h)))
        area_img = np.full((max(1, area_h), max(1, area_w), 3), background_color, dtype=buf.dtype)
        area_img[cb_y:cb_y + cb_h, cb_x:cb_x + cb_w] = block
        self._set_plugin_colorbar_render_meta(buf_w, buf_h, cb_x, cb_y, cb_w, cb_h, area_w, area_h, orientation)

        if orientation == "Vertical":
            out = np.zeros((buf_h, buf_w + area_w, 3), dtype=buf.dtype)
            out[:, :buf_w] = buf
            out[:, buf_w:buf_w + area_w] = area_img
            return out

        out = np.zeros((buf_h + area_h, buf_w, 3), dtype=buf.dtype)
        out[:buf_h, :buf_w] = buf
        out[buf_h:buf_h + area_h, :area_w] = area_img
        return out

    def _draw_z_scale_bar(self, img: np.ndarray) -> np.ndarray:
        if not hasattr(self, "z_scale_bar_check") or not self.z_scale_bar_check.isChecked():
            self._clear_plugin_colorbar_render_meta()
            return img
        if cv2 is None or gv is None:
            return img

        self._apply_color_bar_geometry_to_main()
        channel = self.channel_combo.currentText() if hasattr(self, "channel_combo") else "1ch"
        orientation = self._color_bar_orientation()
        try:
            rendered = self._append_plugin_colorbar_from_widget(img, channel, orientation)
            if rendered is None:
                rendered = self._append_plugin_colorbar_opencv(img, channel, orientation)
            if rendered is not None:
                return rendered
        except Exception as exc:
            print(f"[WARNING] Failed to draw Color Bar in AFM Movie Editor: {exc}")
        self._clear_plugin_colorbar_render_meta()
        return img

    def _normalize_font_style(self, style: str) -> str:
        value = str(style or "").strip().lower()
        if value in ("bold italic", "italic bold"):
            return "Bold Italic"
        if value == "bold":
            return "Bold"
        if value == "italic":
            return "Italic"
        return "Normal"

    def _frame_number_font(self) -> QtGui.QFont:
        font = QtGui.QFont(str(self._frame_num.get("font_family", "Arial")))
        font.setPixelSize(max(8, int(self._frame_num.get("font_size", 18))))
        style = self._normalize_font_style(str(self._frame_num.get("font_style", "Normal")))
        font.setBold("Bold" in style)
        font.setItalic("Italic" in style)
        return font

    def _frame_number_text(self, frame_idx: int) -> str:
        return f"Frame {int(frame_idx) + 1}"

    def _frame_number_layout(self, frame_w: int, frame_h: int, frame_idx: int) -> Optional[Dict[str, Any]]:
        if not self.frame_no_check.isChecked():
            return None
        text = self._frame_number_text(frame_idx)
        font = self._frame_number_font()
        metrics = QtGui.QFontMetrics(font)
        font_size = max(8, int(self._frame_num.get("font_size", 18)))
        text_rect_raw = metrics.boundingRect(text)
        text_w = max(1, int(text_rect_raw.width()))
        text_h = max(1, int(text_rect_raw.height()))
        pad_x = max(6, int(font_size * 0.45))
        pad_y = max(4, int(font_size * 0.30))
        box_w = min(max(1, frame_w), text_w + pad_x * 2)
        box_h = min(max(1, frame_h), text_h + pad_y * 2)
        margin = max(4, int(min(frame_w, frame_h) * 0.035))

        if bool(self._frame_num.get("custom", False)):
            x = int(float(frame_w) * float(self._frame_num.get("x_ratio", 0.75)))
            y = int(float(frame_h) * float(self._frame_num.get("y_ratio", 0.04)))
        else:
            position = str(self._frame_num.get("position", "Top Right"))
            if position == "Top Left":
                x, y = margin, margin
            elif position == "Top":
                x, y = (frame_w - box_w) // 2, margin
            elif position == "Left":
                x, y = margin, (frame_h - box_h) // 2
            elif position == "Center":
                x, y = (frame_w - box_w) // 2, (frame_h - box_h) // 2
            elif position == "Right":
                x, y = frame_w - box_w - margin, (frame_h - box_h) // 2
            elif position == "Bottom Left":
                x, y = margin, frame_h - box_h - margin
            elif position == "Bottom":
                x, y = (frame_w - box_w) // 2, frame_h - box_h - margin
            elif position == "Bottom Right":
                x, y = frame_w - box_w - margin, frame_h - box_h - margin
            else:
                x, y = frame_w - box_w - margin, margin

        x = max(0, min(max(0, frame_w - box_w), int(x)))
        y = max(0, min(max(0, frame_h - box_h), int(y)))
        rect = QtCore.QRect(x, y, box_w, box_h)
        text_rect = rect.adjusted(pad_x, pad_y, -pad_x, -pad_y)
        return {"text": text, "font": font, "rect": rect, "text_rect": text_rect}

    def _draw_frame_number(self, bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        if cv2 is None:
            return bgr
        rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        layout = self._frame_number_layout(w, h, frame_idx)
        if layout is None:
            return bgr

        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        painter = QtGui.QPainter(qimg)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        if bool(self._frame_num.get("background", True)):
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 150))
            painter.drawRoundedRect(layout["rect"], 4, 4)
        painter.setFont(layout["font"])
        painter.setPen(QtGui.QPen(self._frame_num.get("color", QtGui.QColor(255, 255, 255))))
        painter.drawText(layout["text_rect"], QtCore.Qt.AlignCenter, layout["text"])
        painter.end()
        return self._qimage_to_bgr(qimg)

    def _text_overlay_font(self, overlay: TextOverlaySpec) -> QtGui.QFont:
        font = QtGui.QFont(str(getattr(overlay, "font_family", "Arial") or "Arial"))
        font.setPixelSize(max(8, int(overlay.font_size)))
        style = self._normalize_font_style(str(getattr(overlay, "font_style", "Normal") or "Normal"))
        font.setBold("Bold" in style)
        font.setItalic("Italic" in style)
        return font

    def _text_align_for_position(self, position: str) -> QtCore.Qt.Alignment:
        if position in ("Top Left", "Left", "Bottom Left"):
            return QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        if position in ("Top Right", "Right", "Bottom Right"):
            return QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        return QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap

    def _text_align_for_overlay(self, overlay: TextOverlaySpec) -> QtCore.Qt.Alignment:
        align = str(getattr(overlay, "text_align", "Auto") or "Auto")
        if align == "Left":
            return QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        if align == "Right":
            return QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        if align == "Justify":
            return QtCore.Qt.AlignJustify | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        if align == "Center":
            return QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap
        return self._text_align_for_position(str(overlay.position or "Bottom"))

    def _text_overlay_layout(self, overlay: TextOverlaySpec, frame_w: int, frame_h: int) -> Optional[Dict[str, Any]]:
        text = str(overlay.text or "")
        if not text:
            return None
        font = self._text_overlay_font(overlay)
        metrics = QtGui.QFontMetrics(font)
        font_size = max(8, int(overlay.font_size))
        max_text_w = max(40, int(frame_w * 0.82))
        bounding = metrics.boundingRect(QtCore.QRect(0, 0, max_text_w, frame_h), QtCore.Qt.TextWordWrap, text)
        pad_x = max(6, int(font_size * 0.45))
        pad_y = max(4, int(font_size * 0.30))
        box_w = min(max(1, frame_w), max(1, bounding.width()) + pad_x * 2)
        box_h = min(max(1, frame_h), max(1, bounding.height()) + pad_y * 2)
        margin = max(4, int(min(frame_w, frame_h) * 0.035))
        position = str(overlay.position or "Bottom")

        if bool(getattr(overlay, "custom_position", False)):
            x = int(float(frame_w) * float(getattr(overlay, "x_ratio", 0.5)))
            y = int(float(frame_h) * float(getattr(overlay, "y_ratio", 0.5)))
        elif position == "Top Left":
            x, y = margin, margin
        elif position == "Top":
            x, y = (frame_w - box_w) // 2, margin
        elif position == "Top Right":
            x, y = frame_w - box_w - margin, margin
        elif position == "Left":
            x, y = margin, (frame_h - box_h) // 2
        elif position == "Center":
            x, y = (frame_w - box_w) // 2, (frame_h - box_h) // 2
        elif position == "Right":
            x, y = frame_w - box_w - margin, (frame_h - box_h) // 2
        elif position == "Bottom Left":
            x, y = margin, frame_h - box_h - margin
        elif position == "Bottom Right":
            x, y = frame_w - box_w - margin, frame_h - box_h - margin
        else:
            x, y = (frame_w - box_w) // 2, frame_h - box_h - margin

        x = max(0, min(max(0, frame_w - box_w), int(x)))
        y = max(0, min(max(0, frame_h - box_h), int(y)))
        rect = QtCore.QRect(x, y, box_w, box_h)
        text_rect = rect.adjusted(pad_x, pad_y, -pad_x, -pad_y)
        return {
            "text": text,
            "font": font,
            "rect": rect,
            "text_rect": text_rect,
            "align": self._text_align_for_overlay(overlay),
        }

    def _draw_text_overlay(self, bgr: np.ndarray, overlay: TextOverlaySpec) -> np.ndarray:
        if cv2 is None:
            return bgr
        rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        layout = self._text_overlay_layout(overlay, w, h)
        if layout is None:
            return bgr
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        painter = QtGui.QPainter(qimg)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        if overlay.background:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 150))
            painter.drawRoundedRect(layout["rect"], 4, 4)
        painter.setFont(layout["font"])
        painter.setPen(QtGui.QPen(overlay.color))
        painter.drawText(layout["text_rect"], layout["align"], layout["text"])
        painter.end()
        return self._qimage_to_bgr(qimg)

    def _shape_pen_style(self, style: str) -> QtCore.Qt.PenStyle:
        value = str(style or "Solid")
        if value == "Dash":
            return QtCore.Qt.DashLine
        if value == "Dot":
            return QtCore.Qt.DotLine
        if value == "Wave":
            return QtCore.Qt.DashDotLine
        return QtCore.Qt.SolidLine

    def _shape_overlay_endpoints(self, overlay: ShapeOverlaySpec, frame_w: int, frame_h: int) -> Tuple[QtCore.QPointF, QtCore.QPointF]:
        x1 = max(0.0, min(1.0, float(getattr(overlay, "x_ratio", 0.33))))
        y1 = max(0.0, min(1.0, float(getattr(overlay, "y_ratio", 0.50))))
        x2 = max(0.0, min(1.0, float(getattr(overlay, "x2_ratio", 0.67))))
        y2 = max(0.0, min(1.0, float(getattr(overlay, "y2_ratio", 0.50))))
        return QtCore.QPointF(x1 * frame_w, y1 * frame_h), QtCore.QPointF(x2 * frame_w, y2 * frame_h)

    def _shape_overlay_layout(self, overlay: ShapeOverlaySpec, frame_w: int, frame_h: int) -> Dict[str, Any]:
        pad = max(4, int(getattr(overlay, "line_width", 3)) + 4)
        if self._shape_uses_endpoints(overlay):
            start, end = self._shape_overlay_endpoints(overlay, frame_w, frame_h)
            left = int(np.floor(min(start.x(), end.x())))
            top = int(np.floor(min(start.y(), end.y())))
            right = int(np.ceil(max(start.x(), end.x())))
            bottom = int(np.ceil(max(start.y(), end.y())))
            rect = QtCore.QRect(left, top, max(1, right - left), max(1, bottom - top))
            return {"rect": rect, "hit_rect": rect.adjusted(-pad, -pad, pad, pad), "start": start, "end": end}

        try:
            w_ratio = float(getattr(overlay, "w_ratio", 0.30))
            h_ratio = float(getattr(overlay, "h_ratio", 0.20))
            x_ratio = float(getattr(overlay, "x_ratio", 0.35))
            y_ratio = float(getattr(overlay, "y_ratio", 0.35))
        except Exception:
            w_ratio, h_ratio, x_ratio, y_ratio = 0.30, 0.20, 0.35, 0.35
        rect_w = max(6, min(frame_w, int(round(frame_w * max(0.02, min(1.0, w_ratio))))))
        rect_h = max(6, min(frame_h, int(round(frame_h * max(0.02, min(1.0, h_ratio))))))
        x = max(0, min(max(0, frame_w - rect_w), int(round(frame_w * x_ratio))))
        y = max(0, min(max(0, frame_h - rect_h), int(round(frame_h * y_ratio))))
        rect = QtCore.QRect(x, y, rect_w, rect_h)
        return {"rect": rect, "hit_rect": rect.adjusted(-pad, -pad, pad, pad)}

    def _distance_point_to_segment(self, point: QtCore.QPointF, start: QtCore.QPointF, end: QtCore.QPointF) -> float:
        vx = float(end.x() - start.x())
        vy = float(end.y() - start.y())
        wx = float(point.x() - start.x())
        wy = float(point.y() - start.y())
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-9:
            return float(((point.x() - start.x()) ** 2 + (point.y() - start.y()) ** 2) ** 0.5)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
        proj_x = start.x() + t * vx
        proj_y = start.y() + t * vy
        return float(((point.x() - proj_x) ** 2 + (point.y() - proj_y) ** 2) ** 0.5)

    def _shape_rotation_enabled(self, overlay_or_shape: Any) -> bool:
        shape = overlay_or_shape if isinstance(overlay_or_shape, str) else getattr(overlay_or_shape, "shape", "")
        return str(shape or "") in ("Ellipse", "Rectangle", "Triangle")

    def _shape_rotation_degrees(self, overlay: ShapeOverlaySpec) -> float:
        try:
            angle = float(getattr(overlay, "rotation_degrees", 0.0) or 0.0)
        except Exception:
            angle = 0.0
        return ((angle + 180.0) % 360.0) - 180.0

    def _rect_center_point(self, rect: QtCore.QRect) -> QtCore.QPointF:
        return QtCore.QPointF(rect.x() + rect.width() / 2.0, rect.y() + rect.height() / 2.0)

    def _rotate_point(self, point: QtCore.QPointF, center: QtCore.QPointF, degrees: float) -> QtCore.QPointF:
        radians = np.deg2rad(float(degrees))
        cos_v = float(np.cos(radians))
        sin_v = float(np.sin(radians))
        dx = float(point.x() - center.x())
        dy = float(point.y() - center.y())
        return QtCore.QPointF(center.x() + dx * cos_v - dy * sin_v, center.y() + dx * sin_v + dy * cos_v)

    def _unrotate_point(self, point: QtCore.QPointF, center: QtCore.QPointF, degrees: float) -> QtCore.QPointF:
        return self._rotate_point(point, center, -float(degrees))

    def _shape_rotation_handle_pos(self, overlay: ShapeOverlaySpec, frame_w: int, frame_h: int) -> Optional[QtCore.QPointF]:
        if not self._shape_rotation_enabled(overlay):
            return None
        layout = self._shape_overlay_layout(overlay, frame_w, frame_h)
        rect = layout["rect"]
        center = self._rect_center_point(rect)
        offset = max(12.0, min(32.0, min(frame_w, frame_h) * 0.055 + float(getattr(overlay, "line_width", 3))))
        handle = QtCore.QPointF(rect.x() + rect.width() + offset, rect.y() - offset)
        return self._rotate_point(handle, center, self._shape_rotation_degrees(overlay))

    def _shape_mouse_angle(self, overlay: ShapeOverlaySpec, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> float:
        rect = self._shape_overlay_layout(overlay, frame_w, frame_h)["rect"]
        center = self._rect_center_point(rect)
        return float(np.degrees(np.arctan2(float(image_pos[1]) - center.y(), float(image_pos[0]) - center.x())))

    def _draw_wavy_line(self, painter: QtGui.QPainter, start: QtCore.QPointF, end: QtCore.QPointF, line_width: int) -> None:
        dx = float(end.x() - start.x())
        dy = float(end.y() - start.y())
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        amp = max(2.0, float(line_width) * 1.25)
        wave = max(8.0, float(line_width) * 5.0)
        px = -dy / length
        py = dx / length
        steps = max(12, int(length / 3.0))
        path = QtGui.QPainterPath(start)
        for step in range(1, steps + 1):
            t = step / float(steps)
            offset = np.sin(t * length / wave * 2.0 * np.pi) * amp
            x = start.x() + dx * t + px * offset
            y = start.y() + dy * t + py * offset
            path.lineTo(QtCore.QPointF(x, y))
        painter.drawPath(path)

    def _draw_wavy_polyline(self, painter: QtGui.QPainter, points: List[QtCore.QPointF], closed: bool, line_width: int) -> None:
        if len(points) < 2:
            return
        count = len(points)
        for idx in range(count - 1):
            self._draw_wavy_line(painter, points[idx], points[idx + 1], line_width)
        if closed:
            self._draw_wavy_line(painter, points[-1], points[0], line_width)

    def _draw_wavy_ellipse(self, painter: QtGui.QPainter, rect: QtCore.QRect, line_width: int) -> None:
        cx = rect.center().x()
        cy = rect.center().y()
        rx = max(2.0, rect.width() / 2.0)
        ry = max(2.0, rect.height() / 2.0)
        amp = min(rx * 0.18, ry * 0.18, max(2.0, float(line_width) * 1.10))
        steps = max(48, int((rect.width() + rect.height()) / 2))
        cycles = max(8, int((rect.width() + rect.height()) / max(16, int(line_width) * 6)))
        path = QtGui.QPainterPath()
        for step in range(steps + 1):
            t = 2.0 * np.pi * step / float(steps)
            offset = np.sin(t * cycles) * amp
            x = cx + (rx + offset) * np.cos(t)
            y = cy + (ry + offset) * np.sin(t)
            point = QtCore.QPointF(x, y)
            if step == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.drawPath(path)

    def _draw_shape_overlay(self, bgr: np.ndarray, overlay: ShapeOverlaySpec) -> np.ndarray:
        if cv2 is None:
            return bgr
        rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        layout = self._shape_overlay_layout(overlay, w, h)
        rect = QtCore.QRect(layout["rect"])
        shape = str(getattr(overlay, "shape", "Rectangle") or "Rectangle")
        if shape == "Circle":
            side = min(rect.width(), rect.height())
            rect = QtCore.QRect(rect.x(), rect.y(), side, side)

        line_width = max(1, min(80, int(getattr(overlay, "line_width", 3))))
        line_color = QtGui.QColor(getattr(overlay, "color", QtGui.QColor(255, 255, 0)))
        line_alpha = max(0, min(255, int(round(255 * int(getattr(overlay, "line_opacity", 100)) / 100.0))))
        line_color.setAlpha(line_alpha)
        fill_color = QtGui.QColor(getattr(overlay, "fill_color", line_color))
        fill_alpha = max(0, min(255, int(round(255 * int(getattr(overlay, "fill_opacity", 35)) / 100.0))))
        fill_color.setAlpha(fill_alpha)

        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        painter = QtGui.QPainter(qimg)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        line_style = str(getattr(overlay, "line_style", "Solid") or "Solid")
        is_wave = line_style == "Wave"
        pen_style = QtCore.Qt.SolidLine if is_wave else self._shape_pen_style(line_style)
        pen = QtGui.QPen(line_color, line_width, pen_style, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        fill_brush = QtGui.QBrush(fill_color) if bool(getattr(overlay, "fill", False)) else QtGui.QBrush(QtCore.Qt.NoBrush)
        no_brush = QtGui.QBrush(QtCore.Qt.NoBrush)
        painter.setPen(pen)
        painter.setBrush(fill_brush)
        rotated_shape = self._shape_rotation_enabled(shape)
        rotation_degrees = self._shape_rotation_degrees(overlay) if rotated_shape else 0.0
        shape_rect = QtCore.QRectF(rect)
        if rotated_shape and abs(rotation_degrees) > 0.001:
            center = self._rect_center_point(rect)
            painter.save()
            painter.translate(center)
            painter.rotate(rotation_degrees)
            shape_rect = QtCore.QRectF(-rect.width() / 2.0, -rect.height() / 2.0, rect.width(), rect.height())

        if shape in ("Line", "Arrow"):
            start = layout.get("start", QtCore.QPointF(rect.left(), rect.center().y()))
            end = layout.get("end", QtCore.QPointF(rect.right(), rect.center().y()))
            if not isinstance(start, QtCore.QPointF):
                start = QtCore.QPointF(start)
            if not isinstance(end, QtCore.QPointF):
                end = QtCore.QPointF(end)
            dx = float(end.x() - start.x())
            dy = float(end.y() - start.y())
            length = max(1.0, (dx * dx + dy * dy) ** 0.5)
            ux = dx / length
            uy = dy / length
            nx = -uy
            ny = ux

            if shape == "Arrow":
                head_percent = max(5, min(80, int(getattr(overlay, "arrow_head_percent", 20))))
                head_len = max(float(line_width) * 3.0, min(length * 0.80, length * head_percent / 100.0))
                head_half = max(float(line_width) * 1.75, head_len * 0.42)
                shaft_end = QtCore.QPointF(end.x() - ux * head_len, end.y() - uy * head_len)
            else:
                shaft_end = end
                head_len = 0.0
                head_half = 0.0

            if is_wave:
                self._draw_wavy_line(painter, start, shaft_end, line_width)
            else:
                painter.drawLine(start, shaft_end)

            if shape == "Arrow":
                painter.setPen(pen)
                arrow_brush = QtGui.QBrush(fill_color if bool(getattr(overlay, "fill", False)) else line_color)
                painter.setBrush(arrow_brush)
                painter.drawPolygon(QtGui.QPolygonF([
                    end,
                    QtCore.QPointF(shaft_end.x() + nx * head_half, shaft_end.y() + ny * head_half),
                    QtCore.QPointF(shaft_end.x() - nx * head_half, shaft_end.y() - ny * head_half),
                ]))
        elif shape in ("Circle", "Ellipse"):
            target_rect = shape_rect if shape == "Ellipse" else QtCore.QRectF(rect)
            if is_wave:
                if bool(getattr(overlay, "fill", False)):
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.setBrush(fill_brush)
                    painter.drawEllipse(target_rect)
                painter.setPen(pen)
                painter.setBrush(no_brush)
                self._draw_wavy_ellipse(painter, target_rect, line_width)
            else:
                painter.drawEllipse(target_rect)
        elif shape == "Triangle":
            points_list = [
                QtCore.QPointF(shape_rect.center().x(), shape_rect.top()),
                QtCore.QPointF(shape_rect.left(), shape_rect.bottom()),
                QtCore.QPointF(shape_rect.right(), shape_rect.bottom()),
            ]
            points = QtGui.QPolygonF(points_list)
            if is_wave:
                if bool(getattr(overlay, "fill", False)):
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.setBrush(fill_brush)
                    painter.drawPolygon(points)
                painter.setPen(pen)
                painter.setBrush(no_brush)
                self._draw_wavy_polyline(painter, points_list, True, line_width)
            else:
                painter.drawPolygon(points)
        elif shape == "Arrowhead":
            points_list = [
                QtCore.QPointF(rect.right(), rect.center().y()),
                QtCore.QPointF(rect.left(), rect.top()),
                QtCore.QPointF(rect.left(), rect.bottom()),
            ]
            points = QtGui.QPolygonF(points_list)
            if is_wave:
                if bool(getattr(overlay, "fill", False)):
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.setBrush(fill_brush)
                    painter.drawPolygon(points)
                painter.setPen(pen)
                painter.setBrush(no_brush)
                self._draw_wavy_polyline(painter, points_list, True, line_width)
            else:
                painter.drawPolygon(points)
        else:
            if is_wave:
                if bool(getattr(overlay, "fill", False)):
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.setBrush(fill_brush)
                    painter.drawRect(shape_rect)
                painter.setPen(pen)
                painter.setBrush(no_brush)
                points_list = [
                    QtCore.QPointF(shape_rect.left(), shape_rect.top()),
                    QtCore.QPointF(shape_rect.right(), shape_rect.top()),
                    QtCore.QPointF(shape_rect.right(), shape_rect.bottom()),
                    QtCore.QPointF(shape_rect.left(), shape_rect.bottom()),
                ]
                self._draw_wavy_polyline(painter, points_list, True, line_width)
            else:
                painter.drawRect(shape_rect)
        if rotated_shape and abs(rotation_degrees) > 0.001:
            painter.restore()
        painter.end()
        return self._qimage_to_bgr(qimg)

    def _draw_user_overlays(self, img: np.ndarray, output_index: int) -> np.ndarray:
        out = img
        for overlay in self.shape_overlays:
            if overlay.start <= output_index <= overlay.end:
                out = self._draw_shape_overlay(out, overlay)
        for overlay in self.overlays:
            if overlay.start <= output_index <= overlay.end:
                out = self._draw_text_overlay(out, overlay)
        return out

    def _draw_text_qt(
        self,
        bgr: np.ndarray,
        text: str,
        position: str,
        font_size: int,
        color: QtGui.QColor,
        background: bool = True,
        offset_y: int = 0,
    ) -> np.ndarray:
        if cv2 is None:
            return bgr
        rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()

        painter = QtGui.QPainter(qimg)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        font = QtGui.QFont("Arial")
        font.setPixelSize(max(8, int(font_size)))
        font.setBold(position == "Center")
        painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)

        max_text_w = max(40, int(w * 0.82))
        bounding = metrics.boundingRect(QtCore.QRect(0, 0, max_text_w, h), QtCore.Qt.TextWordWrap, text)
        pad_x = max(6, int(font_size * 0.45))
        pad_y = max(4, int(font_size * 0.30))
        box_w = min(w - 8, bounding.width() + pad_x * 2)
        box_h = min(h - 8, bounding.height() + pad_y * 2)
        margin = max(8, int(min(w, h) * 0.035))

        if position == "Top Left":
            x = margin
            y = margin + offset_y
            align = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        elif position == "Top":
            x = (w - box_w) // 2
            y = margin + offset_y
            align = QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap
        elif position == "Top Right":
            x = w - box_w - margin
            y = margin + offset_y
            align = QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        elif position == "Left":
            x = margin
            y = (h - box_h) // 2 + offset_y
            align = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        elif position == "Center":
            x = (w - box_w) // 2
            y = (h - box_h) // 2 + offset_y
            align = QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap
        elif position == "Right":
            x = w - box_w - margin
            y = (h - box_h) // 2 + offset_y
            align = QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        elif position == "Bottom Left":
            x = margin
            y = h - box_h - margin - offset_y
            align = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        elif position == "Bottom Right":
            x = w - box_w - margin
            y = h - box_h - margin - offset_y
            align = QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
        else:
            x = (w - box_w) // 2
            y = h - box_h - margin - offset_y
            align = QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap

        x = max(4, min(w - box_w - 4, x))
        y = max(4, min(h - box_h - 4, y))
        rect = QtCore.QRect(x, y, box_w, box_h)
        text_rect = rect.adjusted(pad_x, pad_y, -pad_x, -pad_y)

        if background:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 150))
            painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(QtGui.QPen(color))
        painter.drawText(text_rect, align, text)
        painter.end()
        return self._qimage_to_bgr(qimg)

    def _qimage_to_bgr(self, qimg: QtGui.QImage) -> np.ndarray:
        img = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
        w = img.width()
        h = img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, img.bytesPerLine()))
        rgb = arr[:, : w * 3].reshape((h, w, 3)).copy()
        if cv2 is None:
            return rgb
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Interactive preview overlay editing
    # ------------------------------------------------------------------

    def _preview_image_rect(self) -> Optional[QtCore.QRect]:
        pixmap = self.preview_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        label_w = max(1, self.preview_label.width())
        label_h = max(1, self.preview_label.height())
        pix_w = max(1, pixmap.width())
        pix_h = max(1, pixmap.height())
        return QtCore.QRect((label_w - pix_w) // 2, (label_h - pix_h) // 2, pix_w, pix_h)

    def _preview_pos_to_image_pos(self, pos: QtCore.QPoint) -> Optional[Tuple[int, int]]:
        if self._last_preview_bgr is None:
            return None
        rect = self._preview_image_rect()
        if rect is None or not rect.contains(pos):
            return None
        frame_h, frame_w = self._last_preview_bgr.shape[:2]
        x = int(round((pos.x() - rect.x()) * frame_w / max(1, rect.width())))
        y = int(round((pos.y() - rect.y()) * frame_h / max(1, rect.height())))
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        return x, y

    def _source_frame_for_output_index(self, output_index: Optional[int] = None) -> Optional[int]:
        timeline = self._get_timeline()
        if not timeline:
            return None
        if output_index is None:
            output_index = self._current_output_index()
        if output_index is None:
            return None
        output_index = max(0, min(len(timeline) - 1, int(output_index)))
        ref = timeline[output_index]
        try:
            return self._source_frame_from_timeline_ref(ref)
        except Exception:
            return None

    def _title_clip_for_output_index(self, output_index: Optional[int] = None) -> Optional[int]:
        timeline = self._get_timeline()
        if not timeline:
            return None
        if output_index is None:
            output_index = self._current_output_index()
        if output_index is None:
            return None
        output_index = max(0, min(len(timeline) - 1, int(output_index)))
        ref = timeline[output_index]
        try:
            if ref["kind"] == "clip":
                clip_idx = int(ref["clip"])
                if 0 <= clip_idx < len(self.clips) and self.clips[clip_idx].kind == "title":
                    return clip_idx
                return None

            a_idx = int(ref.get("a_clip", -1))
            b_idx = int(ref.get("b_clip", -1))
            alpha = float(ref.get("alpha", 0.5))
            a_is_title = 0 <= a_idx < len(self.clips) and self.clips[a_idx].kind == "title"
            b_is_title = 0 <= b_idx < len(self.clips) and self.clips[b_idx].kind == "title"
            if a_is_title and b_is_title:
                return b_idx if alpha >= 0.5 else a_idx
            if b_is_title:
                return b_idx
            if a_is_title:
                return a_idx
        except Exception:
            return None
        return None

    def _with_main_overlay_state(self, source_frame: Optional[int], callback):
        if gv is None:
            return callback()
        old_index = getattr(gv, "index", 0)
        old_time = getattr(gv, "showTimeFlag", False)
        old_scale = getattr(gv, "showScaleFlag", False)
        try:
            if source_frame is not None:
                gv.index = int(source_frame)
            gv.showTimeFlag = bool(self.time_check.isChecked())
            gv.showScaleFlag = bool(self.scale_bar_check.isChecked())
            return callback()
        finally:
            gv.index = old_index
            gv.showTimeFlag = old_time
            gv.showScaleFlag = old_scale

    def _pre_colorbar_hit_size(self, frame_w: int, frame_h: int) -> Tuple[int, int]:
        """Overlays drawn before the color bar use the base output frame size."""
        base_w, base_h = self._output_size()
        if hasattr(self, "z_scale_bar_check") and self.z_scale_bar_check.isChecked():
            return max(1, base_w), max(1, base_h)
        return max(1, frame_w), max(1, frame_h)

    def _hit_main_overlay(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int, source_frame: Optional[int]) -> Optional[str]:
        if self.main_window is None or not hasattr(self.main_window, "hitTestSaveOverlay"):
            return None
        if not (self.time_check.isChecked() or self.scale_bar_check.isChecked()):
            return None
        hit_w, hit_h = self._pre_colorbar_hit_size(frame_w, frame_h)

        def _hit():
            return self.main_window.hitTestSaveOverlay(
                image_pos[0],
                image_pos[1],
                hit_w,
                hit_h,
                bool(self.time_check.isChecked()),
                bool(self.scale_bar_check.isChecked()),
                False,
                False,
            )

        return self._with_main_overlay_state(source_frame, _hit)

    def _hit_frame_number(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int, source_frame: Optional[int]) -> Optional[str]:
        if source_frame is None or not self.frame_no_check.isChecked():
            return None
        hit_w, hit_h = self._pre_colorbar_hit_size(frame_w, frame_h)
        layout = self._frame_number_layout(hit_w, hit_h, source_frame)
        if layout is None:
            return None
        if layout["rect"].contains(QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))):
            return "frame_num"
        return None

    def _current_output_index(self) -> int:
        timeline = self._get_timeline()
        if not timeline:
            return 0
        return max(0, min(len(timeline) - 1, int(self.preview_slider.value())))

    def _text_overlay_target(self, index: int) -> str:
        return f"text_overlay:{int(index)}"

    def _shape_overlay_target(self, index: int) -> str:
        return f"shape_overlay:{int(index)}"

    def _title_overlay_target(self, index: int) -> str:
        return f"title_clip:{int(index)}"

    def _parse_text_overlay_target(self, target: Optional[str]) -> Optional[int]:
        if not isinstance(target, str) or not target.startswith("text_overlay:"):
            return None
        try:
            idx = int(target.split(":", 1)[1])
        except Exception:
            return None
        if 0 <= idx < len(self.overlays):
            return idx
        return None

    def _parse_shape_overlay_target(self, target: Optional[str]) -> Optional[int]:
        if not isinstance(target, str) or not target.startswith("shape_overlay:"):
            return None
        try:
            idx = int(target.split(":", 1)[1])
        except Exception:
            return None
        if 0 <= idx < len(self.shape_overlays):
            return idx
        return None

    def _parse_title_overlay_target(self, target: Optional[str]) -> Optional[int]:
        if not isinstance(target, str) or not target.startswith("title_clip:"):
            return None
        try:
            idx = int(target.split(":", 1)[1])
        except Exception:
            return None
        if 0 <= idx < len(self.clips) and self.clips[idx].kind == "title":
            return idx
        return None

    def _color_bar_orientation(self) -> str:
        combo = getattr(self.main_window, "zscale_orient_combo", None) if self.main_window is not None else None
        if combo is not None and hasattr(combo, "currentText"):
            try:
                value = str(combo.currentText())
                if value in ("Vertical", "Horizontal"):
                    return value
            except Exception:
                pass
        return "Vertical"

    def _color_bar_rect(self, frame_w: int, frame_h: int) -> Optional[QtCore.QRect]:
        if gv is None or not hasattr(self, "z_scale_bar_check") or not self.z_scale_bar_check.isChecked():
            return None
        orientation = self._color_bar_orientation()
        rendered_rect = self._plugin_colorbar_render_rect
        rendered_frame_size = self._plugin_colorbar_render_frame_size
        rendered_orientation = self._plugin_colorbar_render_orientation
        try:
            if rendered_rect is not None and len(rendered_rect) == 4:
                if rendered_orientation in (None, orientation):
                    if (
                        rendered_frame_size is None
                        or (
                            len(rendered_frame_size) == 2
                            and abs(int(rendered_frame_size[0]) - int(frame_w)) <= 2
                            and abs(int(rendered_frame_size[1]) - int(frame_h)) <= 2
                        )
                    ):
                        x, y, w, h = (int(v) for v in rendered_rect)
                        rect = QtCore.QRect(x, y, max(1, w), max(1, h))
                        frame_rect = QtCore.QRect(0, 0, max(1, frame_w), max(1, frame_h))
                        if rect.intersects(frame_rect):
                            return rect.intersected(frame_rect)
        except Exception:
            pass
        base_w, base_h = self._output_size()
        if orientation == "Vertical":
            if frame_w <= base_w:
                return None
            area_w = max(1, frame_w - base_w)
            area_h = min(max(1, base_h), max(1, frame_h))
            cb_w = max(16, int(getattr(gv, "colorbar_dock_vertical_w", 80)))
            cb_h = max(40, int(getattr(gv, "colorbar_dock_vertical_h", 200)))
            if cb_h > area_h:
                ratio = area_h / float(max(1, cb_h))
                cb_h = area_h
                cb_w = max(16, int(round(cb_w * ratio)))
            cb_w = min(cb_w, area_w)
            x = base_w + max(0, min(int(getattr(gv, "colorbar_dock_vertical_x", 10)), max(0, area_w - cb_w)))
            y = max(0, min(int(getattr(gv, "colorbar_dock_vertical_y", 10)), max(0, area_h - cb_h)))
            return QtCore.QRect(x, y, cb_w, cb_h)
        if frame_h <= base_h:
            return None
        area_w = min(max(1, base_w), max(1, frame_w))
        area_h = max(1, frame_h - base_h)
        cb_w = max(60, int(getattr(gv, "colorbar_dock_horizontal_w", 180)))
        cb_h = max(20, int(getattr(gv, "colorbar_dock_horizontal_h", 80)))
        if cb_w > area_w:
            ratio = area_w / float(max(1, cb_w))
            cb_w = area_w
            cb_h = max(20, int(round(cb_h * ratio)))
        cb_h = min(cb_h, area_h)
        x = max(0, min(int(getattr(gv, "colorbar_dock_horizontal_x", 10)), max(0, area_w - cb_w)))
        y = base_h + max(0, min(int(getattr(gv, "colorbar_dock_horizontal_y", max((area_h - cb_h) // 2, 0))), max(0, area_h - cb_h)))
        return QtCore.QRect(x, y, cb_w, cb_h)

    def _color_bar_handle_pixels(self, frame_w: int, frame_h: int) -> int:
        handle = max(14, min(28, int(min(frame_w, frame_h) * 0.045)))
        try:
            pixmap = self.preview_label.pixmap()
            if pixmap is not None and not pixmap.isNull() and frame_w > 0 and frame_h > 0:
                scale_x = pixmap.width() / float(frame_w)
                scale_y = pixmap.height() / float(frame_h)
                scale = max(0.01, min(scale_x, scale_y))
                handle = max(handle, int(round(18.0 / scale)))
        except Exception:
            pass
        return int(handle)

    def _color_bar_resize_zone(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> Optional[str]:
        rect = self._color_bar_rect(frame_w, frame_h)
        if rect is None:
            return None
        point = QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))
        handle = self._color_bar_handle_pixels(frame_w, frame_h)
        if not rect.adjusted(-handle, -handle, handle, handle).contains(point):
            return None
        x_handle = min(handle, max(6, rect.width() // 3))
        y_handle = min(handle, max(6, rect.height() // 3))
        near_left = abs(point.x() - rect.left()) <= x_handle
        near_right = abs(point.x() - (rect.x() + rect.width())) <= x_handle
        near_top = abs(point.y() - rect.top()) <= y_handle
        near_bottom = abs(point.y() - (rect.y() + rect.height())) <= y_handle
        horizontal = "left" if near_left else "right" if near_right else ""
        vertical = "top" if near_top else "bottom" if near_bottom else ""
        if horizontal and vertical:
            return f"{vertical}-{horizontal}"
        return horizontal or vertical or None

    def _hit_color_bar(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> Optional[str]:
        rect = self._color_bar_rect(frame_w, frame_h)
        if rect is None:
            return None
        handle = self._color_bar_handle_pixels(frame_w, frame_h)
        if rect.adjusted(-handle, -handle, handle, handle).contains(QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))):
            return "color_bar"
        return None

    def _active_text_overlay_indices(self, output_index: Optional[int] = None) -> List[int]:
        if output_index is None:
            output_index = self._current_output_index()
        active = []
        for idx, overlay in enumerate(self.overlays):
            if overlay.start <= output_index <= overlay.end:
                active.append(idx)
        return active

    def _active_shape_overlay_indices(self, output_index: Optional[int] = None) -> List[int]:
        if output_index is None:
            output_index = self._current_output_index()
        active = []
        for idx, overlay in enumerate(self.shape_overlays):
            if overlay.start <= output_index <= overlay.end:
                active.append(idx)
        return active

    def _hit_text_overlay(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int, output_index: Optional[int] = None) -> Optional[str]:
        for idx in reversed(self._active_text_overlay_indices(output_index)):
            layout = self._text_overlay_layout(self.overlays[idx], frame_w, frame_h)
            if layout is not None and layout["rect"].contains(QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))):
                return self._text_overlay_target(idx)
        return None

    def _hit_shape_overlay(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int, output_index: Optional[int] = None) -> Optional[str]:
        point = QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))
        point_f = QtCore.QPointF(point)
        for idx in reversed(self._active_shape_overlay_indices(output_index)):
            overlay = self.shape_overlays[idx]
            layout = self._shape_overlay_layout(overlay, frame_w, frame_h)
            if self._shape_uses_endpoints(overlay):
                handle = max(8, min(24, int(min(frame_w, frame_h) * 0.035) + int(getattr(overlay, "line_width", 3))))
                start = layout["start"]
                end = layout["end"]
                start_dist = float(((point_f.x() - start.x()) ** 2 + (point_f.y() - start.y()) ** 2) ** 0.5)
                end_dist = float(((point_f.x() - end.x()) ** 2 + (point_f.y() - end.y()) ** 2) ** 0.5)
                line_dist = self._distance_point_to_segment(point_f, start, end)
                if start_dist <= handle or end_dist <= handle or line_dist <= handle:
                    return self._shape_overlay_target(idx)
                continue

            if self._shape_rotation_enabled(overlay):
                handle_pos = self._shape_rotation_handle_pos(overlay, frame_w, frame_h)
                handle_radius = max(9, min(24, int(min(frame_w, frame_h) * 0.04) + int(getattr(overlay, "line_width", 3))))
                if handle_pos is not None:
                    handle_dist = float(((point_f.x() - handle_pos.x()) ** 2 + (point_f.y() - handle_pos.y()) ** 2) ** 0.5)
                    if handle_dist <= handle_radius:
                        return self._shape_overlay_target(idx)
                rect = layout["rect"]
                center = self._rect_center_point(rect)
                test_point = self._unrotate_point(point_f, center, self._shape_rotation_degrees(overlay))
                if layout["hit_rect"].contains(QtCore.QPoint(int(test_point.x()), int(test_point.y()))):
                    return self._shape_overlay_target(idx)
                continue

            if layout["hit_rect"].contains(point):
                return self._shape_overlay_target(idx)
        return None

    def _hit_title_overlay(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int, output_index: Optional[int] = None) -> Optional[str]:
        clip_idx = self._title_clip_for_output_index(output_index)
        if clip_idx is None:
            return None
        layout = self._text_overlay_layout(self._title_overlay_spec(self.clips[clip_idx]), frame_w, frame_h)
        if layout is not None and layout["rect"].contains(QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))):
            return self._title_overlay_target(clip_idx)
        return None

    def _hit_preview_overlay(self, image_pos: Tuple[int, int]) -> Tuple[Optional[str], Optional[int], int, int]:
        if self._last_preview_bgr is None:
            return None, None, 0, 0
        frame_h, frame_w = self._last_preview_bgr.shape[:2]
        source_frame = self._source_frame_for_output_index()
        output_index = self._current_output_index()
        target = self._hit_text_overlay(image_pos, frame_w, frame_h, output_index)
        if target is None:
            target = self._hit_title_overlay(image_pos, frame_w, frame_h, output_index)
        if target is None:
            target = self._hit_shape_overlay(image_pos, frame_w, frame_h, output_index)
        if target is None:
            target = self._hit_color_bar(image_pos, frame_w, frame_h)
        if target is None:
            target = self._hit_frame_number(image_pos, frame_w, frame_h, source_frame)
        if target is None:
            target = self._hit_main_overlay(image_pos, frame_w, frame_h, source_frame)
        return target, source_frame, frame_w, frame_h

    def _move_text_overlay_to(self, index: int, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        if self._preview_drag_offset is None or not (0 <= index < len(self.overlays)):
            return False
        overlay = self.overlays[index]
        layout = self._text_overlay_layout(overlay, frame_w, frame_h)
        if layout is None:
            return False
        desired_x = int(image_pos[0]) - int(self._preview_drag_offset.x())
        desired_y = int(image_pos[1]) - int(self._preview_drag_offset.y())
        desired_x = max(0, min(max(0, frame_w - layout["rect"].width()), desired_x))
        desired_y = max(0, min(max(0, frame_h - layout["rect"].height()), desired_y))
        overlay.custom_position = True
        overlay.x_ratio = desired_x / float(max(1, frame_w))
        overlay.y_ratio = desired_y / float(max(1, frame_h))
        return True

    def _move_title_overlay_to(self, index: int, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        if self._preview_drag_offset is None or not (0 <= index < len(self.clips)):
            return False
        clip = self.clips[index]
        if clip.kind != "title":
            return False
        layout = self._text_overlay_layout(self._title_overlay_spec(clip), frame_w, frame_h)
        if layout is None:
            return False
        desired_x = int(image_pos[0]) - int(self._preview_drag_offset.x())
        desired_y = int(image_pos[1]) - int(self._preview_drag_offset.y())
        desired_x = max(0, min(max(0, frame_w - layout["rect"].width()), desired_x))
        desired_y = max(0, min(max(0, frame_h - layout["rect"].height()), desired_y))
        clip.title_custom_position = True
        clip.title_x_ratio = desired_x / float(max(1, frame_w))
        clip.title_y_ratio = desired_y / float(max(1, frame_h))
        return True

    def _move_shape_overlay_to(self, index: int, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        if not (0 <= index < len(self.shape_overlays)):
            return False
        overlay = self.shape_overlays[index]
        if self._shape_uses_endpoints(overlay):
            if self._preview_last_img_pos is None:
                return False
            dx = int(image_pos[0]) - int(self._preview_last_img_pos.x())
            dy = int(image_pos[1]) - int(self._preview_last_img_pos.y())
            start, end = self._shape_overlay_endpoints(overlay, frame_w, frame_h)
            min_x = min(start.x(), end.x())
            max_x = max(start.x(), end.x())
            min_y = min(start.y(), end.y())
            max_y = max(start.y(), end.y())
            dx = int(max(-min_x, min(frame_w - max_x, dx)))
            dy = int(max(-min_y, min(frame_h - max_y, dy)))
            overlay.x_ratio = (start.x() + dx) / float(max(1, frame_w))
            overlay.y_ratio = (start.y() + dy) / float(max(1, frame_h))
            overlay.x2_ratio = (end.x() + dx) / float(max(1, frame_w))
            overlay.y2_ratio = (end.y() + dy) / float(max(1, frame_h))
            return dx != 0 or dy != 0

        if self._preview_drag_offset is None:
            return False
        layout = self._shape_overlay_layout(overlay, frame_w, frame_h)
        rect = layout["rect"]
        desired_x = int(image_pos[0]) - int(self._preview_drag_offset.x())
        desired_y = int(image_pos[1]) - int(self._preview_drag_offset.y())
        desired_x = max(0, min(max(0, frame_w - rect.width()), desired_x))
        desired_y = max(0, min(max(0, frame_h - rect.height()), desired_y))
        overlay.x_ratio = desired_x / float(max(1, frame_w))
        overlay.y_ratio = desired_y / float(max(1, frame_h))
        return True

    def _shape_resize_zone(self, overlay: ShapeOverlaySpec, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> Optional[str]:
        layout = self._shape_overlay_layout(overlay, frame_w, frame_h)
        point = QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))
        point_f = QtCore.QPointF(point)
        if self._shape_uses_endpoints(overlay):
            handle = max(8, min(24, int(min(frame_w, frame_h) * 0.035) + int(getattr(overlay, "line_width", 3))))
            start = layout["start"]
            end = layout["end"]
            start_dist = float(((point_f.x() - start.x()) ** 2 + (point_f.y() - start.y()) ** 2) ** 0.5)
            end_dist = float(((point_f.x() - end.x()) ** 2 + (point_f.y() - end.y()) ** 2) ** 0.5)
            if start_dist <= handle and start_dist <= end_dist:
                return "start"
            if end_dist <= handle:
                return "end"
            return None

        rect = layout["rect"]
        if self._shape_rotation_enabled(overlay):
            handle_pos = self._shape_rotation_handle_pos(overlay, frame_w, frame_h)
            handle_radius = max(9, min(24, int(min(frame_w, frame_h) * 0.04) + int(getattr(overlay, "line_width", 3))))
            if handle_pos is not None:
                handle_dist = float(((point_f.x() - handle_pos.x()) ** 2 + (point_f.y() - handle_pos.y()) ** 2) ** 0.5)
                if handle_dist <= handle_radius:
                    return "rotate"
            point_f = self._unrotate_point(point_f, self._rect_center_point(rect), self._shape_rotation_degrees(overlay))
            point = QtCore.QPoint(int(point_f.x()), int(point_f.y()))

        if not layout["hit_rect"].contains(point):
            return None
        handle = max(8, min(24, int(min(frame_w, frame_h) * 0.035) + int(getattr(overlay, "line_width", 3))))
        near_left = abs(point.x() - rect.left()) <= handle
        near_right = abs(point.x() - (rect.x() + rect.width())) <= handle
        near_top = abs(point.y() - rect.top()) <= handle
        near_bottom = abs(point.y() - (rect.y() + rect.height())) <= handle
        horizontal = "left" if near_left else "right" if near_right else ""
        vertical = "top" if near_top else "bottom" if near_bottom else ""
        if horizontal and vertical:
            return f"{vertical}-{horizontal}"
        return horizontal or vertical or None

    def _shape_resize_cursor(self, zone: Optional[str]) -> QtCore.Qt.CursorShape:
        if zone == "rotate":
            return QtCore.Qt.SizeAllCursor
        if zone in ("start", "end"):
            return QtCore.Qt.CrossCursor
        if zone in ("left", "right"):
            return QtCore.Qt.SizeHorCursor
        if zone in ("top", "bottom"):
            return QtCore.Qt.SizeVerCursor
        if zone in ("top-left", "bottom-right"):
            return QtCore.Qt.SizeFDiagCursor
        if zone in ("top-right", "bottom-left"):
            return QtCore.Qt.SizeBDiagCursor
        return QtCore.Qt.OpenHandCursor

    def _resize_shape_overlay_to(self, index: int, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        if not (0 <= index < len(self.shape_overlays)):
            return False
        if not self._preview_drag_mode or not self._preview_drag_mode.startswith("shape_resize:"):
            return False
        if self._preview_last_img_pos is None:
            return False

        overlay = self.shape_overlays[index]
        zone = self._preview_drag_mode.split(":", 1)[1]
        if self._shape_uses_endpoints(overlay):
            x = max(0, min(frame_w, int(image_pos[0])))
            y = max(0, min(frame_h, int(image_pos[1])))
            if zone == "start":
                overlay.x_ratio = x / float(max(1, frame_w))
                overlay.y_ratio = y / float(max(1, frame_h))
                return True
            if zone == "end":
                overlay.x2_ratio = x / float(max(1, frame_w))
                overlay.y2_ratio = y / float(max(1, frame_h))
                return True
            return False

        if zone == "rotate" and self._shape_rotation_enabled(overlay):
            current_mouse_angle = self._shape_mouse_angle(overlay, image_pos, frame_w, frame_h)
            delta = current_mouse_angle - float(self._shape_rotate_start_mouse_angle)
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0
            overlay.rotation_degrees = ((float(self._shape_rotate_start_angle) + delta + 180.0) % 360.0) - 180.0
            return True

        layout = self._shape_overlay_layout(overlay, frame_w, frame_h)
        rect = layout["rect"]
        left = int(rect.x())
        top = int(rect.y())
        right = int(rect.x() + rect.width())
        bottom = int(rect.y() + rect.height())
        dx = int(image_pos[0]) - int(self._preview_last_img_pos.x())
        dy = int(image_pos[1]) - int(self._preview_last_img_pos.y())
        if self._shape_rotation_enabled(overlay) and abs(self._shape_rotation_degrees(overlay)) > 0.001:
            center = self._rect_center_point(rect)
            prev_point = self._unrotate_point(QtCore.QPointF(self._preview_last_img_pos), center, self._shape_rotation_degrees(overlay))
            curr_point = self._unrotate_point(QtCore.QPointF(float(image_pos[0]), float(image_pos[1])), center, self._shape_rotation_degrees(overlay))
            dx = int(round(curr_point.x() - prev_point.x()))
            dy = int(round(curr_point.y() - prev_point.y()))
        min_w = max(6, int(frame_w * 0.02))
        min_h = max(6, int(frame_h * 0.02))

        if "left" in zone:
            left = max(0, min(right - min_w, left + dx))
        if "right" in zone:
            right = min(frame_w, max(left + min_w, right + dx))
        if "top" in zone:
            top = max(0, min(bottom - min_h, top + dy))
        if "bottom" in zone:
            bottom = min(frame_h, max(top + min_h, bottom + dy))

        new_w = max(min_w, right - left)
        new_h = max(min_h, bottom - top)
        overlay.x_ratio = left / float(max(1, frame_w))
        overlay.y_ratio = top / float(max(1, frame_h))
        overlay.w_ratio = new_w / float(max(1, frame_w))
        overlay.h_ratio = new_h / float(max(1, frame_h))
        return True

    def _move_frame_number_to(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int, source_frame: Optional[int]) -> bool:
        if source_frame is None or self._preview_drag_offset is None:
            return False
        hit_w, hit_h = self._pre_colorbar_hit_size(frame_w, frame_h)
        layout = self._frame_number_layout(hit_w, hit_h, source_frame)
        if layout is None:
            return False
        desired_x = int(image_pos[0]) - int(self._preview_drag_offset.x())
        desired_y = int(image_pos[1]) - int(self._preview_drag_offset.y())
        desired_x = max(0, min(max(0, hit_w - layout["rect"].width()), desired_x))
        desired_y = max(0, min(max(0, hit_h - layout["rect"].height()), desired_y))
        self._frame_num["custom"] = True
        self._frame_num["x_ratio"] = desired_x / float(max(1, hit_w))
        self._frame_num["y_ratio"] = desired_y / float(max(1, hit_h))
        return True

    def _apply_color_bar_geometry_to_main(self) -> None:
        if gv is None or self.main_window is None:
            return
        try:
            target_window = getattr(self.main_window, "image_window", None)
            if self.channel_combo.currentText() == "2ch":
                target_window = getattr(self.main_window, "image_window_2ch", target_window)
            cbw = getattr(target_window, "colorbar_widget", None) if target_window is not None else None
            if cbw is None:
                return
            orientation = self._color_bar_orientation()
            if orientation == "Vertical":
                cbw.setMinimumSize(32, 80)
                cbw.setMaximumSize(16777215, 16777215)
                cbw.resize(int(getattr(gv, "colorbar_dock_vertical_w", cbw.width())), int(getattr(gv, "colorbar_dock_vertical_h", cbw.height())))
                cbw.move(int(getattr(gv, "colorbar_dock_vertical_x", cbw.x())), int(getattr(gv, "colorbar_dock_vertical_y", cbw.y())))
            else:
                cbw.setMinimumSize(80, 32)
                cbw.setMaximumSize(16777215, 16777215)
                cbw.resize(int(getattr(gv, "colorbar_dock_horizontal_w", cbw.width())), int(getattr(gv, "colorbar_dock_horizontal_h", cbw.height())))
                cbw.move(int(getattr(gv, "colorbar_dock_horizontal_x", cbw.x())), int(getattr(gv, "colorbar_dock_horizontal_y", cbw.y())))
            cbw.updateGeometry()
            layout = cbw.layout()
            if layout is not None:
                layout.activate()
        except Exception:
            pass

    def _move_color_bar_to(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        if gv is None or self._preview_drag_offset is None:
            return False
        rect = self._color_bar_rect(frame_w, frame_h)
        if rect is None:
            return False
        base_w, base_h = self._output_size()
        orientation = self._color_bar_orientation()
        desired_x = int(image_pos[0]) - int(self._preview_drag_offset.x())
        desired_y = int(image_pos[1]) - int(self._preview_drag_offset.y())
        if orientation == "Vertical":
            area_w = max(1, frame_w - base_w)
            local_x = max(0, min(max(0, area_w - rect.width()), desired_x - base_w))
            local_y = max(0, min(max(0, base_h - rect.height()), desired_y))
            changed = (
                int(getattr(gv, "colorbar_dock_vertical_x", 10)) != local_x
                or int(getattr(gv, "colorbar_dock_vertical_y", 10)) != local_y
            )
            gv.colorbar_dock_vertical_x = local_x
            gv.colorbar_dock_vertical_y = local_y
        else:
            local_x = max(0, min(max(0, base_w - rect.width()), desired_x))
            area_h = max(1, frame_h - base_h)
            local_y = max(0, min(max(0, area_h - rect.height()), desired_y - base_h))
            changed = (
                int(getattr(gv, "colorbar_dock_horizontal_x", 10)) != local_x
                or int(getattr(gv, "colorbar_dock_horizontal_y", 10)) != local_y
            )
            gv.colorbar_dock_horizontal_x = local_x
            gv.colorbar_dock_horizontal_y = local_y
        self._apply_color_bar_geometry_to_main()
        return changed

    def _resize_color_bar_to(self, image_pos: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        if gv is None or not self._preview_drag_mode or not self._preview_drag_mode.startswith("color_bar_resize:"):
            return False
        rect = self._color_bar_rect(frame_w, frame_h)
        if rect is None or self._preview_last_img_pos is None:
            return False
        base_w, base_h = self._output_size()
        zone = self._preview_drag_mode.split(":", 1)[1]
        left = int(rect.x())
        top = int(rect.y())
        right = int(rect.x() + rect.width())
        bottom = int(rect.y() + rect.height())
        dx = int(image_pos[0]) - int(self._preview_last_img_pos.x())
        dy = int(image_pos[1]) - int(self._preview_last_img_pos.y())
        min_w = 20
        min_h = 40
        if self._color_bar_orientation() == "Horizontal":
            min_w = 60
            min_h = 20

        if "left" in zone:
            left = min(right - min_w, left + dx)
        if "right" in zone:
            right = max(left + min_w, right + dx)
        if "top" in zone:
            top = min(bottom - min_h, top + dy)
        if "bottom" in zone:
            bottom = max(top + min_h, bottom + dy)

        orientation = self._color_bar_orientation()
        if orientation == "Vertical":
            left = max(base_w, left)
            top = max(0, min(max(0, base_h - min_h), top))
            bottom = max(top + min_h, min(base_h, bottom))
            new_w = max(min_w, right - left)
            new_h = max(min_h, bottom - top)
            gv.colorbar_dock_vertical_x = max(0, left - base_w)
            gv.colorbar_dock_vertical_y = max(0, min(max(0, base_h - new_h), top))
            gv.colorbar_dock_vertical_w = max(min_w, new_w)
            gv.colorbar_dock_vertical_h = max(min_h, min(base_h, new_h))
        else:
            left = max(0, min(max(0, base_w - min_w), left))
            right = max(left + min_w, min(base_w, right))
            top = max(base_h, top)
            new_w = max(min_w, right - left)
            new_h = max(min_h, bottom - top)
            gv.colorbar_dock_horizontal_x = max(0, min(max(0, base_w - new_w), left))
            gv.colorbar_dock_horizontal_y = max(0, top - base_h)
            gv.colorbar_dock_horizontal_w = max(min_w, min(base_w, new_w))
            gv.colorbar_dock_horizontal_h = max(min_h, new_h)
        self._apply_color_bar_geometry_to_main()
        return True

    def _scale_color_bar_size(self, steps: int, thickness_only: bool = False) -> bool:
        if gv is None:
            return False
        factor = max(0.10, 1.0 + int(steps) * 0.06)
        orientation = self._color_bar_orientation()
        if orientation == "Vertical":
            w = int(getattr(gv, "colorbar_dock_vertical_w", 80))
            h = int(getattr(gv, "colorbar_dock_vertical_h", 200))
            gv.colorbar_dock_vertical_w = max(20, min(260, int(round(w * factor))))
            if not thickness_only:
                base_h = self._output_size()[1]
                gv.colorbar_dock_vertical_h = max(40, min(max(40, base_h), int(round(h * factor))))
        else:
            w = int(getattr(gv, "colorbar_dock_horizontal_w", 180))
            h = int(getattr(gv, "colorbar_dock_horizontal_h", 80))
            base_w = self._output_size()[0]
            if not thickness_only:
                gv.colorbar_dock_horizontal_w = max(60, min(max(60, base_w), int(round(w * factor))))
            gv.colorbar_dock_horizontal_h = max(20, min(260, int(round(h * factor))))
        self._apply_color_bar_geometry_to_main()
        return True

    def _on_colorbar_text_style_changed(self, part: str) -> None:
        self._sync_color_bar_text_style_to_gv()
        self._refresh_plugin_colorbar_text_styles_for_view(part)
        self._update_preview()

    def _colorbar_menu_set_size(self) -> None:
        if gv is None:
            return
        orientation = self._color_bar_orientation()
        if orientation == "Vertical":
            current_w = int(getattr(gv, "colorbar_dock_vertical_w", 80))
            current_h = int(getattr(gv, "colorbar_dock_vertical_h", 200))
        else:
            current_w = int(getattr(gv, "colorbar_dock_horizontal_w", 180))
            current_h = int(getattr(gv, "colorbar_dock_horizontal_h", 80))
        if colorbar_menu is None:
            return
        width, ok = colorbar_menu.input_dialog_int("Color Bar", "Width:", current_w, 16, 1000)
        if not ok:
            return
        height, ok = colorbar_menu.input_dialog_int("Color Bar", "Height:", current_h, 20, 1000)
        if not ok:
            return
        if orientation == "Vertical":
            gv.colorbar_dock_vertical_w = int(width)
            gv.colorbar_dock_vertical_h = int(height)
        else:
            gv.colorbar_dock_horizontal_w = int(width)
            gv.colorbar_dock_horizontal_h = int(height)
        self._apply_color_bar_geometry_to_main()
        self._update_preview()

    def _colorbar_menu_reset_position(self) -> None:
        if gv is None:
            return
        orientation = self._color_bar_orientation()
        if orientation == "Vertical":
            gv.colorbar_dock_vertical_x = 10
            gv.colorbar_dock_vertical_y = 10
        else:
            gv.colorbar_dock_horizontal_x = 10
            gv.colorbar_dock_horizontal_y = 10
        self._apply_color_bar_geometry_to_main()
        self._update_preview()

    def _colorbar_menu_reset_size(self) -> None:
        if gv is None:
            return
        orientation = self._color_bar_orientation()
        if orientation == "Vertical":
            gv.colorbar_dock_vertical_w = 80
            gv.colorbar_dock_vertical_h = 200
        else:
            gv.colorbar_dock_horizontal_w = 180
            gv.colorbar_dock_horizontal_h = 80
        self._apply_color_bar_geometry_to_main()
        self._update_preview()

    def _default_colorbar_part_style(self, part: str) -> dict:
        if colorbar_menu is None:
            return {}
        style = colorbar_menu.default_colorbar_text_style()[part].copy()
        if part == "values":
            style["position"] = self._default_color_bar_value_position()
        return style

    def _show_color_bar_text_style_menu(self, global_pos: QtCore.QPoint, part: str) -> None:
        if colorbar_menu is None:
            return
        self._sync_color_bar_text_style_from_gv()
        colorbar_menu.show_colorbar_text_style_menu(
            global_pos,
            part,
            self._color_bar_text_style,
            normalize_font_style=self._normalize_font_style,
            value_positions=self._color_bar_value_positions,
            default_value_position=self._default_color_bar_value_position,
            font_families=self._available_font_families,
            on_changed=self._on_colorbar_text_style_changed,
            get_default_part_style=self._default_colorbar_part_style,
        )

    def _show_color_bar_context_menu(self, global_pos: QtCore.QPoint) -> None:
        if gv is None or colorbar_menu is None:
            return
        colorbar_menu.show_colorbar_context_menu(
            global_pos,
            on_unit=lambda: self._show_color_bar_text_style_menu(global_pos, "unit"),
            on_values=lambda: self._show_color_bar_text_style_menu(global_pos, "values"),
            on_size=self._colorbar_menu_set_size,
            on_reset_position=self._colorbar_menu_reset_position,
            on_reset_size=self._colorbar_menu_reset_size,
        )

    def _wheel_steps(self, event: QtGui.QWheelEvent) -> int:
        delta_y = int(event.angleDelta().y())
        delta_x = int(event.angleDelta().x())
        raw_delta = delta_y if delta_y != 0 else delta_x
        steps = int(raw_delta / 120)
        if steps == 0 and raw_delta != 0:
            steps = 1 if raw_delta > 0 else -1
        return steps

    def _preview_mouse_press(self, event: QtGui.QMouseEvent) -> bool:
        if event.modifiers() != QtCore.Qt.NoModifier:
            return False
        image_pos = self._preview_pos_to_image_pos(event.pos())
        if image_pos is None:
            return False
        target, source_frame, frame_w, frame_h = self._hit_preview_overlay(image_pos)
        if target is None:
            return False

        if event.button() == QtCore.Qt.RightButton:
            text_idx = self._parse_text_overlay_target(target)
            title_idx = self._parse_title_overlay_target(target)
            shape_idx = self._parse_shape_overlay_target(target)
            if text_idx is not None:
                self._show_text_overlay_context_menu(text_idx, event.globalPos())
            elif title_idx is not None:
                self._show_title_overlay_context_menu(title_idx, event.globalPos())
            elif shape_idx is not None:
                self._show_shape_overlay_context_menu(shape_idx, event.globalPos())
            elif target == "color_bar":
                self._show_color_bar_context_menu(event.globalPos())
            elif target == "frame_num":
                self._show_frame_number_context_menu(event.globalPos())
            elif self.main_window is not None and hasattr(self.main_window, "showSaveOverlayContextMenu"):
                self._with_main_overlay_state(
                    source_frame,
                    lambda: self.main_window.showSaveOverlayContextMenu(target, event.globalPos()),
                )
            self._update_preview()
            return True

        if event.button() == QtCore.Qt.LeftButton:
            self._preview_drag_target = target
            self._preview_drag_mode = "move"
            self._preview_last_img_pos = QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))
            self._preview_drag_offset = None
            text_idx = self._parse_text_overlay_target(target)
            title_idx = self._parse_title_overlay_target(target)
            shape_idx = self._parse_shape_overlay_target(target)
            if text_idx is not None:
                layout = self._text_overlay_layout(self.overlays[text_idx], frame_w, frame_h)
                if layout is not None:
                    rect = layout["rect"]
                    self._preview_drag_offset = QtCore.QPoint(int(image_pos[0]) - rect.x(), int(image_pos[1]) - rect.y())
                    self.overlay_table.selectRow(text_idx)
            elif title_idx is not None:
                layout = self._text_overlay_layout(self._title_overlay_spec(self.clips[title_idx]), frame_w, frame_h)
                if layout is not None:
                    rect = layout["rect"]
                    self._preview_drag_offset = QtCore.QPoint(int(image_pos[0]) - rect.x(), int(image_pos[1]) - rect.y())
                    self.clip_table.selectRow(title_idx)
            elif shape_idx is not None:
                layout = self._shape_overlay_layout(self.shape_overlays[shape_idx], frame_w, frame_h)
                rect = layout["rect"]
                resize_zone = self._shape_resize_zone(self.shape_overlays[shape_idx], image_pos, frame_w, frame_h)
                if resize_zone is not None:
                    self._preview_drag_mode = f"shape_resize:{resize_zone}"
                    if resize_zone == "rotate":
                        overlay = self.shape_overlays[shape_idx]
                        self._shape_rotate_start_angle = self._shape_rotation_degrees(overlay)
                        self._shape_rotate_start_mouse_angle = self._shape_mouse_angle(overlay, image_pos, frame_w, frame_h)
                else:
                    self._preview_drag_offset = QtCore.QPoint(int(image_pos[0]) - rect.x(), int(image_pos[1]) - rect.y())
                self.shape_table.selectRow(shape_idx)
            elif target == "color_bar":
                rect = self._color_bar_rect(frame_w, frame_h)
                resize_zone = self._color_bar_resize_zone(image_pos, frame_w, frame_h)
                if resize_zone is not None:
                    self._preview_drag_mode = f"color_bar_resize:{resize_zone}"
                elif rect is not None:
                    self._preview_drag_offset = QtCore.QPoint(int(image_pos[0]) - rect.x(), int(image_pos[1]) - rect.y())
            elif target == "frame_num" and source_frame is not None:
                hit_w, hit_h = self._pre_colorbar_hit_size(frame_w, frame_h)
                layout = self._frame_number_layout(hit_w, hit_h, source_frame)
                if layout is not None:
                    rect = layout["rect"]
                    self._preview_drag_offset = QtCore.QPoint(int(image_pos[0]) - rect.x(), int(image_pos[1]) - rect.y())
            elif target == "time" and self.main_window is not None and hasattr(self.main_window, "_getSaveTimeOverlayLayout"):
                hit_w, hit_h = self._pre_colorbar_hit_size(frame_w, frame_h)

                def _time_offset():
                    layout = self.main_window._getSaveTimeOverlayLayout(hit_w, hit_h)
                    if layout is not None and layout.get("text_rect") is not None:
                        tx, ty, _tw, _th = layout["text_rect"]
                        self._preview_drag_offset = QtCore.QPoint(int(image_pos[0]) - int(tx), int(image_pos[1]) - int(ty))
                self._with_main_overlay_state(source_frame, _time_offset)
            if self._preview_drag_mode and self._preview_drag_mode.startswith("color_bar_resize:"):
                self.preview_label.setCursor(self._shape_resize_cursor(self._preview_drag_mode.split(":", 1)[1]))
            elif shape_idx is not None and self._preview_drag_mode and self._preview_drag_mode.startswith("shape_resize:"):
                self.preview_label.setCursor(self._shape_resize_cursor(self._preview_drag_mode.split(":", 1)[1]))
            else:
                self.preview_label.setCursor(QtCore.Qt.ClosedHandCursor)
            return True
        return False

    def _preview_mouse_double_click(self, event: QtGui.QMouseEvent) -> bool:
        if event.button() != QtCore.Qt.LeftButton or event.modifiers() != QtCore.Qt.NoModifier:
            return False
        image_pos = self._preview_pos_to_image_pos(event.pos())
        if image_pos is None:
            return False
        target, _source_frame, _frame_w, _frame_h = self._hit_preview_overlay(image_pos)
        text_idx = self._parse_text_overlay_target(target)
        if text_idx is not None:
            self._preview_drag_target = None
            self._preview_drag_mode = None
            self._preview_last_img_pos = None
            self._preview_drag_offset = None
            self.overlay_table.selectRow(text_idx)
            self._edit_text_overlay_text(text_idx, refresh=True)
            self.preview_label.unsetCursor()
            return True

        title_idx = self._parse_title_overlay_target(target)
        if title_idx is None:
            return False

        self._preview_drag_target = None
        self._preview_drag_mode = None
        self._preview_last_img_pos = None
        self._preview_drag_offset = None
        self.clip_table.selectRow(title_idx)
        self._edit_title_overlay_text(title_idx, refresh=True)
        self.preview_label.unsetCursor()
        return True

    def _preview_mouse_move(self, event: QtGui.QMouseEvent) -> bool:
        image_pos = self._preview_pos_to_image_pos(event.pos())
        if self._last_preview_bgr is None:
            return False
        frame_h, frame_w = self._last_preview_bgr.shape[:2]
        source_frame = self._source_frame_for_output_index()

        if self._preview_drag_target is not None and self._preview_last_img_pos is not None:
            if image_pos is None:
                return True
            moved = False
            text_idx = self._parse_text_overlay_target(self._preview_drag_target)
            title_idx = self._parse_title_overlay_target(self._preview_drag_target)
            shape_idx = self._parse_shape_overlay_target(self._preview_drag_target)
            if text_idx is not None:
                moved = self._move_text_overlay_to(text_idx, image_pos, frame_w, frame_h)
            elif title_idx is not None:
                moved = self._move_title_overlay_to(title_idx, image_pos, frame_w, frame_h)
            elif shape_idx is not None:
                if self._preview_drag_mode and self._preview_drag_mode.startswith("shape_resize:"):
                    moved = self._resize_shape_overlay_to(shape_idx, image_pos, frame_w, frame_h)
                else:
                    moved = self._move_shape_overlay_to(shape_idx, image_pos, frame_w, frame_h)
            elif self._preview_drag_target == "color_bar":
                if self._preview_drag_mode and self._preview_drag_mode.startswith("color_bar_resize:"):
                    moved = self._resize_color_bar_to(image_pos, frame_w, frame_h)
                else:
                    moved = self._move_color_bar_to(image_pos, frame_w, frame_h)
            elif self._preview_drag_target == "frame_num":
                moved = self._move_frame_number_to(image_pos, frame_w, frame_h, source_frame)
            elif self.main_window is not None and hasattr(self.main_window, "moveSaveOverlay"):
                dx = int(image_pos[0]) - int(self._preview_last_img_pos.x())
                dy = int(image_pos[1]) - int(self._preview_last_img_pos.y())
                hit_w, hit_h = self._pre_colorbar_hit_size(frame_w, frame_h)
                if (
                    self._preview_drag_target == "time"
                    and self._preview_drag_offset is not None
                    and hasattr(self.main_window, "_getSaveTimeOverlayLayout")
                ):
                    def _time_delta():
                        layout = self.main_window._getSaveTimeOverlayLayout(hit_w, hit_h)
                        if layout is not None and layout.get("text_rect") is not None:
                            tx, ty, _tw, _th = layout["text_rect"]
                            return int(image_pos[0]) - int(self._preview_drag_offset.x()) - int(tx), int(image_pos[1]) - int(self._preview_drag_offset.y()) - int(ty)
                        return dx, dy
                    dx, dy = self._with_main_overlay_state(source_frame, _time_delta)
                if dx != 0 or dy != 0:
                    moved = bool(self._with_main_overlay_state(
                        source_frame,
                        lambda: self.main_window.moveSaveOverlay(self._preview_drag_target, dx, dy, hit_w, hit_h),
                    ))
            if moved:
                self._preview_last_img_pos = QtCore.QPoint(int(image_pos[0]), int(image_pos[1]))
                if text_idx is not None:
                    self._refresh_overlay_table(select_row=text_idx)
                elif shape_idx is not None:
                    self._refresh_shape_overlay_table(select_row=shape_idx)
                self._update_preview()
            if self._preview_drag_mode and self._preview_drag_mode.startswith("color_bar_resize:"):
                self.preview_label.setCursor(self._shape_resize_cursor(self._preview_drag_mode.split(":", 1)[1]))
            elif shape_idx is not None and self._preview_drag_mode and self._preview_drag_mode.startswith("shape_resize:"):
                self.preview_label.setCursor(self._shape_resize_cursor(self._preview_drag_mode.split(":", 1)[1]))
            else:
                self.preview_label.setCursor(QtCore.Qt.ClosedHandCursor)
            return True

        if image_pos is None:
            self.preview_label.unsetCursor()
            return False
        target, _source_frame, _frame_w, _frame_h = self._hit_preview_overlay(image_pos)
        if target is not None:
            shape_idx = self._parse_shape_overlay_target(target)
            if shape_idx is not None:
                zone = self._shape_resize_zone(self.shape_overlays[shape_idx], image_pos, frame_w, frame_h)
                self.preview_label.setCursor(self._shape_resize_cursor(zone))
            elif target == "color_bar":
                zone = self._color_bar_resize_zone(image_pos, frame_w, frame_h)
                self.preview_label.setCursor(self._shape_resize_cursor(zone))
            else:
                self.preview_label.setCursor(QtCore.Qt.OpenHandCursor)
            return True
        self.preview_label.unsetCursor()
        return False

    def _preview_mouse_release(self, event: QtGui.QMouseEvent) -> bool:
        if event.button() != QtCore.Qt.LeftButton or self._preview_drag_target is None:
            return False
        self._preview_drag_target = None
        self._preview_drag_mode = None
        self._preview_last_img_pos = None
        self._preview_drag_offset = None
        image_pos = self._preview_pos_to_image_pos(event.pos())
        if image_pos is not None:
            target, _source_frame, _frame_w, _frame_h = self._hit_preview_overlay(image_pos)
            if target is not None:
                self.preview_label.setCursor(QtCore.Qt.OpenHandCursor)
                return True
        self.preview_label.unsetCursor()
        return True

    def _preview_wheel(self, event: QtGui.QWheelEvent) -> bool:
        modifiers = event.modifiers()
        shift_pressed = bool(modifiers & QtCore.Qt.ShiftModifier)
        other_mods = modifiers & ~QtCore.Qt.ShiftModifier
        if other_mods != QtCore.Qt.NoModifier:
            return False
        image_pos = self._preview_pos_to_image_pos(event.pos())
        if image_pos is None:
            return False
        target, source_frame, frame_w, frame_h = self._hit_preview_overlay(image_pos)
        if target is None:
            return False
        steps = self._wheel_steps(event)
        if steps == 0:
            return False
        if target == "frame_num":
            self._frame_num["font_size"] = max(8, min(160, int(self._frame_num.get("font_size", 18)) + steps))
        elif self._parse_text_overlay_target(target) is not None:
            idx = self._parse_text_overlay_target(target)
            if idx is None:
                return False
            self.overlays[idx].font_size = max(8, min(160, int(self.overlays[idx].font_size) + steps))
            self._refresh_overlay_table(select_row=idx)
        elif self._parse_title_overlay_target(target) is not None:
            idx = self._parse_title_overlay_target(target)
            if idx is None:
                return False
            clip = self.clips[idx]
            clip.title_font_size = max(8, min(160, int(getattr(clip, "title_font_size", self._default_title_font_size())) + steps))
            self.clip_table.selectRow(idx)
        elif self._parse_shape_overlay_target(target) is not None:
            idx = self._parse_shape_overlay_target(target)
            if idx is None:
                return False
            overlay = self.shape_overlays[idx]
            if shift_pressed:
                overlay.line_width = max(1, min(80, int(getattr(overlay, "line_width", 3)) + steps))
            else:
                factor = max(0.10, 1.0 + steps * 0.06)
                overlay.w_ratio = max(0.02, min(1.0, float(getattr(overlay, "w_ratio", 0.30)) * factor))
                overlay.h_ratio = max(0.02, min(1.0, float(getattr(overlay, "h_ratio", 0.20)) * factor))
            self._refresh_shape_overlay_table(select_row=idx)
        elif target == "color_bar":
            self._scale_color_bar_size(steps, thickness_only=shift_pressed)
        elif self.main_window is not None and hasattr(self.main_window, "resizeSaveOverlay"):
            wheel_target = target
            if target == "scale":
                wheel_target = "scale_thickness" if shift_pressed else "scale"
            elif shift_pressed:
                return False
            self._with_main_overlay_state(
                source_frame,
                lambda: self.main_window.resizeSaveOverlay(wheel_target, steps),
            )
        else:
            return False
        self._update_preview()
        self.preview_label.setCursor(QtCore.Qt.OpenHandCursor)
        return True

    def _available_font_families(self) -> List[str]:
        families = []
        try:
            if self.main_window is not None and hasattr(self.main_window, "_getSaveOverlayFontOptions"):
                families = list(self.main_window._getSaveOverlayFontOptions())
        except Exception:
            families = []
        if not families:
            families = QtGui.QFontDatabase().families()
        return [str(f) for f in families if str(f).strip()]

    def _set_frame_number_custom_to_current_layout(self) -> None:
        if self._last_preview_bgr is None:
            self._frame_num["custom"] = True
            return
        source_frame = self._source_frame_for_output_index()
        if source_frame is None:
            self._frame_num["custom"] = True
            return
        frame_h, frame_w = self._last_preview_bgr.shape[:2]
        hit_w, hit_h = self._pre_colorbar_hit_size(frame_w, frame_h)
        layout = self._frame_number_layout(hit_w, hit_h, source_frame)
        if layout is not None:
            rect = layout["rect"]
            self._frame_num["x_ratio"] = rect.x() / float(max(1, hit_w))
            self._frame_num["y_ratio"] = rect.y() / float(max(1, hit_h))
        self._frame_num["custom"] = True

    def _show_frame_number_context_menu(self, global_pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        font_size_action = menu.addAction("Font Size...")
        menu.addSeparator()

        position_menu = menu.addMenu("Position")
        position_actions = {}
        for position in POSITIONS:
            action = position_menu.addAction(position)
            action.setCheckable(True)
            action.setChecked(not self._frame_num.get("custom", False) and self._frame_num.get("position") == position)
            position_actions[action] = position
        custom_action = position_menu.addAction("Use Current Custom Position")
        custom_action.setCheckable(True)
        custom_action.setChecked(bool(self._frame_num.get("custom", False)))

        font_menu = menu.addMenu("Font")
        font_actions = {}
        current_font = str(self._frame_num.get("font_family", "Arial"))
        for family in self._available_font_families():
            action = font_menu.addAction(family)
            action.setCheckable(True)
            action.setChecked(family == current_font)
            font_actions[action] = family

        style_menu = menu.addMenu("Style")
        style_actions = {}
        current_style = self._normalize_font_style(str(self._frame_num.get("font_style", "Normal")))
        for style in ["Normal", "Bold", "Italic", "Bold Italic"]:
            action = style_menu.addAction(style)
            action.setCheckable(True)
            action.setChecked(style == current_style)
            style_actions[action] = style

        color_action = menu.addAction("Color...")
        bg_action = menu.addAction("Background")
        bg_action.setCheckable(True)
        bg_action.setChecked(bool(self._frame_num.get("background", True)))

        selected = menu.exec_(global_pos)
        if selected is None:
            return
        if selected == font_size_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Frame Number",
                "Font Size:",
                int(self._frame_num.get("font_size", 18)),
                8,
                160,
            )
            if ok:
                self._frame_num["font_size"] = int(value)
            return
        if selected in position_actions:
            self._frame_num["position"] = position_actions[selected]
            self._frame_num["custom"] = False
            return
        if selected == custom_action:
            self._set_frame_number_custom_to_current_layout()
            return
        if selected in font_actions:
            self._frame_num["font_family"] = font_actions[selected]
            return
        if selected in style_actions:
            self._frame_num["font_style"] = style_actions[selected]
            return
        if selected == color_action:
            color = QtWidgets.QColorDialog.getColor(self._frame_num.get("color", QtGui.QColor(255, 255, 255)), self, "Frame Number Color")
            if color.isValid():
                self._frame_num["color"] = color
            return
        if selected == bg_action:
            self._frame_num["background"] = bg_action.isChecked()

    def _set_title_overlay_custom_to_current_layout(self, index: int) -> None:
        if not (0 <= index < len(self.clips)) or self.clips[index].kind != "title":
            return
        clip = self.clips[index]
        if self._last_preview_bgr is None:
            clip.title_custom_position = True
            return
        frame_h, frame_w = self._last_preview_bgr.shape[:2]
        layout = self._text_overlay_layout(self._title_overlay_spec(clip), frame_w, frame_h)
        if layout is not None:
            rect = layout["rect"]
            clip.title_x_ratio = rect.x() / float(max(1, frame_w))
            clip.title_y_ratio = rect.y() / float(max(1, frame_h))
        clip.title_custom_position = True

    def _edit_title_overlay_text(self, index: int, refresh: bool = True) -> bool:
        if not (0 <= index < len(self.clips)) or self.clips[index].kind != "title":
            return False
        clip = self.clips[index]
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Title Text", "Text:", clip.title or "Title")
        if not ok:
            return True
        clip.title = text.strip() or clip.title or "Title"
        if self._selected_clip_index() == index:
            self._set_title_controls_from_clip(clip)
        if refresh:
            self._refresh_clip_table(select_row=index)
            self._update_preview()
        return True

    def _show_title_overlay_context_menu(self, index: int, global_pos: QtCore.QPoint) -> None:
        if not (0 <= index < len(self.clips)) or self.clips[index].kind != "title":
            return
        clip = self.clips[index]
        menu = QtWidgets.QMenu(self)
        edit_action = menu.addAction("Edit Title...")
        font_size_action = menu.addAction("Font Size...")
        menu.addSeparator()

        position_menu = menu.addMenu("Position")
        position_actions = {}
        current_position = str(getattr(clip, "title_position", "Center") or "Center")
        for position in POSITIONS:
            action = position_menu.addAction(position)
            action.setCheckable(True)
            action.setChecked(not getattr(clip, "title_custom_position", False) and current_position == position)
            position_actions[action] = position
        custom_action = position_menu.addAction("Use Current Custom Position")
        custom_action.setCheckable(True)
        custom_action.setChecked(bool(getattr(clip, "title_custom_position", False)))

        font_menu = menu.addMenu("Font")
        font_actions = {}
        current_font = str(getattr(clip, "title_font_family", "Arial") or "Arial")
        for family in self._available_font_families():
            action = font_menu.addAction(family)
            action.setCheckable(True)
            action.setChecked(family == current_font)
            font_actions[action] = family

        style_menu = menu.addMenu("Style")
        style_actions = {}
        current_style = self._normalize_font_style(str(getattr(clip, "title_font_style", "Bold") or "Bold"))
        for style in ["Normal", "Bold", "Italic", "Bold Italic"]:
            action = style_menu.addAction(style)
            action.setCheckable(True)
            action.setChecked(style == current_style)
            style_actions[action] = style

        align_menu = menu.addMenu("Text Align")
        align_actions = {}
        current_align = str(getattr(clip, "title_align", "Center") or "Center")
        for align in TEXT_ALIGNMENTS:
            action = align_menu.addAction(align)
            action.setCheckable(True)
            action.setChecked(align == current_align)
            align_actions[action] = align

        color_action = menu.addAction("Color...")
        bg_action = menu.addAction("Background")
        bg_action.setCheckable(True)
        bg_action.setChecked(bool(getattr(clip, "title_background", False)))

        selected = menu.exec_(global_pos)
        if selected is None:
            return
        if selected == edit_action:
            self._edit_title_overlay_text(index, refresh=False)
        elif selected == font_size_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Title Text",
                "Font Size:",
                int(getattr(clip, "title_font_size", self._default_title_font_size())),
                8,
                160,
            )
            if ok:
                clip.title_font_size = int(value)
        elif selected in position_actions:
            clip.title_position = position_actions[selected]
            clip.title_custom_position = False
        elif selected == custom_action:
            self._set_title_overlay_custom_to_current_layout(index)
        elif selected in font_actions:
            clip.title_font_family = font_actions[selected]
        elif selected in style_actions:
            clip.title_font_style = style_actions[selected]
        elif selected in align_actions:
            clip.title_align = align_actions[selected]
        elif selected == color_action:
            color = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(getattr(clip, "title_color", QtGui.QColor(255, 255, 255))),
                self,
                "Title Text Color",
            )
            if color.isValid():
                clip.title_color = color
        elif selected == bg_action:
            clip.title_background = bg_action.isChecked()

        if self._selected_clip_index() == index:
            self._set_title_controls_from_clip(clip)
        self._refresh_clip_table(select_row=index)

    def _set_text_overlay_custom_to_current_layout(self, index: int) -> None:
        if not (0 <= index < len(self.overlays)):
            return
        overlay = self.overlays[index]
        if self._last_preview_bgr is None:
            overlay.custom_position = True
            return
        frame_h, frame_w = self._last_preview_bgr.shape[:2]
        layout = self._text_overlay_layout(overlay, frame_w, frame_h)
        if layout is not None:
            rect = layout["rect"]
            overlay.x_ratio = rect.x() / float(max(1, frame_w))
            overlay.y_ratio = rect.y() / float(max(1, frame_h))
        overlay.custom_position = True

    def _edit_text_overlay_text(self, index: int, refresh: bool = True) -> bool:
        if not (0 <= index < len(self.overlays)):
            return False
        overlay = self.overlays[index]
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Text Overlay", "Text:", overlay.text)
        if not ok:
            return True
        overlay.text = text.strip() or overlay.text
        if refresh:
            self._refresh_overlay_table(select_row=index)
            self._update_preview()
        self._overlay_ui_updating = True
        try:
            self._set_overlay_text_value(overlay.text)
        finally:
            self._overlay_ui_updating = False
        return True

    def _show_text_overlay_context_menu(self, index: int, global_pos: QtCore.QPoint) -> None:
        if not (0 <= index < len(self.overlays)):
            return
        overlay = self.overlays[index]
        menu = QtWidgets.QMenu(self)
        edit_action = menu.addAction("Edit Text...")
        font_size_action = menu.addAction("Font Size...")
        menu.addSeparator()

        position_menu = menu.addMenu("Position")
        position_actions = {}
        for position in POSITIONS:
            action = position_menu.addAction(position)
            action.setCheckable(True)
            action.setChecked(not overlay.custom_position and overlay.position == position)
            position_actions[action] = position
        custom_action = position_menu.addAction("Use Current Custom Position")
        custom_action.setCheckable(True)
        custom_action.setChecked(bool(overlay.custom_position))

        font_menu = menu.addMenu("Font")
        font_actions = {}
        current_font = str(getattr(overlay, "font_family", "Arial") or "Arial")
        for family in self._available_font_families():
            action = font_menu.addAction(family)
            action.setCheckable(True)
            action.setChecked(family == current_font)
            font_actions[action] = family

        style_menu = menu.addMenu("Style")
        style_actions = {}
        current_style = self._normalize_font_style(getattr(overlay, "font_style", "Normal"))
        for style in ["Normal", "Bold", "Italic", "Bold Italic"]:
            action = style_menu.addAction(style)
            action.setCheckable(True)
            action.setChecked(style == current_style)
            style_actions[action] = style

        align_menu = menu.addMenu("Text Align")
        align_actions = {}
        current_align = str(getattr(overlay, "text_align", "Auto") or "Auto")
        for align in OVERLAY_TEXT_ALIGNMENTS:
            action = align_menu.addAction(align)
            action.setCheckable(True)
            action.setChecked(align == current_align)
            align_actions[action] = align

        color_action = menu.addAction("Color...")
        bg_action = menu.addAction("Background")
        bg_action.setCheckable(True)
        bg_action.setChecked(bool(overlay.background))

        selected = menu.exec_(global_pos)
        if selected is None:
            return
        if selected == edit_action:
            self._edit_text_overlay_text(index, refresh=False)
        elif selected == font_size_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Text Overlay",
                "Font Size:",
                int(overlay.font_size),
                8,
                160,
            )
            if ok:
                overlay.font_size = int(value)
        elif selected in position_actions:
            overlay.position = position_actions[selected]
            overlay.custom_position = False
        elif selected == custom_action:
            self._set_text_overlay_custom_to_current_layout(index)
        elif selected in font_actions:
            overlay.font_family = font_actions[selected]
        elif selected in style_actions:
            overlay.font_style = style_actions[selected]
        elif selected in align_actions:
            overlay.text_align = align_actions[selected]
        elif selected == color_action:
            color = QtWidgets.QColorDialog.getColor(overlay.color, self, "Text Overlay Color")
            if color.isValid():
                overlay.color = color
        elif selected == bg_action:
            overlay.background = bg_action.isChecked()

        self._refresh_overlay_table(select_row=index)

    def _show_shape_overlay_context_menu(self, index: int, global_pos: QtCore.QPoint) -> None:
        if not (0 <= index < len(self.shape_overlays)):
            return
        overlay = self.shape_overlays[index]
        menu = QtWidgets.QMenu(self)

        shape_menu = menu.addMenu("Shape")
        shape_actions = {}
        current_shape = str(getattr(overlay, "shape", "Rectangle") or "Rectangle")
        for shape_name in SHAPE_TYPES:
            action = shape_menu.addAction(shape_name)
            action.setCheckable(True)
            action.setChecked(shape_name == current_shape)
            shape_actions[action] = shape_name

        width_action = menu.addAction("Width...")
        height_action = menu.addAction("Height...")
        rotation_action = menu.addAction("Rotation...")
        rotation_action.setEnabled(self._shape_rotation_enabled(current_shape))
        line_width_action = menu.addAction("Line Width...")
        arrow_head_action = menu.addAction("Arrow Head Size...")
        arrow_head_action.setEnabled(current_shape == "Arrow")
        if current_shape in ("Line", "Arrow"):
            width_action.setEnabled(False)
            height_action.setEnabled(False)

        line_style_menu = menu.addMenu("Line Style")
        line_style_actions = {}
        current_style = str(getattr(overlay, "line_style", "Solid") or "Solid")
        for style_name in SHAPE_LINE_STYLES:
            action = line_style_menu.addAction(style_name)
            action.setCheckable(True)
            action.setChecked(style_name == current_style)
            line_style_actions[action] = style_name

        color_action = menu.addAction("Line Color...")
        line_opacity_action = menu.addAction("Line Opacity...")
        fill_action = menu.addAction("Fill")
        fill_action.setCheckable(True)
        fill_action.setChecked(bool(getattr(overlay, "fill", False)))
        fill_color_action = menu.addAction("Fill Color...")
        fill_opacity_action = menu.addAction("Fill Opacity...")

        selected = menu.exec_(global_pos)
        if selected is None:
            return
        if selected in shape_actions:
            new_shape = shape_actions[selected]
            if new_shape != overlay.shape:
                self._convert_shape_overlay_type(overlay, new_shape)
            overlay.shape = new_shape
        elif selected == width_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Shape Width",
                "Width (% of frame):",
                max(2, min(100, int(round(float(getattr(overlay, "w_ratio", 0.30)) * 100)))),
                2,
                100,
            )
            if ok:
                overlay.w_ratio = value / 100.0
        elif selected == height_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Shape Height",
                "Height (% of frame):",
                max(2, min(100, int(round(float(getattr(overlay, "h_ratio", 0.20)) * 100)))),
                2,
                100,
            )
            if ok:
                overlay.h_ratio = value / 100.0
        elif selected == rotation_action:
            value, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Shape Rotation",
                "Rotation angle (degrees):",
                self._shape_rotation_degrees(overlay),
                -180.0,
                180.0,
                1,
            )
            if ok:
                overlay.rotation_degrees = float(value)
        elif selected == line_width_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Shape Line",
                "Line Width:",
                int(getattr(overlay, "line_width", 3)),
                1,
                80,
            )
            if ok:
                overlay.line_width = int(value)
        elif selected == arrow_head_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Arrow",
                "Arrow head size (% of line length):",
                int(getattr(overlay, "arrow_head_percent", 20)),
                5,
                80,
            )
            if ok:
                overlay.arrow_head_percent = int(value)
        elif selected in line_style_actions:
            overlay.line_style = line_style_actions[selected]
        elif selected == color_action:
            color = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(getattr(overlay, "color", QtGui.QColor(255, 255, 0))),
                self,
                "Shape Line Color",
            )
            if color.isValid():
                overlay.color = color
        elif selected == line_opacity_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Shape Line",
                "Line Opacity (%):",
                int(getattr(overlay, "line_opacity", 100)),
                0,
                100,
            )
            if ok:
                overlay.line_opacity = int(value)
        elif selected == fill_action:
            overlay.fill = fill_action.isChecked()
        elif selected == fill_color_action:
            color = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(getattr(overlay, "fill_color", QtGui.QColor(255, 255, 0))),
                self,
                "Shape Fill Color",
            )
            if color.isValid():
                overlay.fill_color = color
                overlay.fill = True
        elif selected == fill_opacity_action:
            value, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Shape Fill",
                "Fill Opacity (%):",
                int(getattr(overlay, "fill_opacity", 35)),
                0,
                100,
            )
            if ok:
                overlay.fill_opacity = int(value)
                overlay.fill = True

        self._shape_ui_updating = True
        try:
            self.shape_type_combo.setCurrentText(str(getattr(overlay, "shape", "Rectangle") or "Rectangle"))
            self.shape_start_spin.setValue(int(getattr(overlay, "start", 0)))
            self.shape_end_spin.setValue(int(getattr(overlay, "end", 0)))
        finally:
            self._shape_ui_updating = False
        self._refresh_shape_overlay_table(select_row=index)
        self._update_preview()

    # ------------------------------------------------------------------
    # Preview and export
    # ------------------------------------------------------------------

    def _movie_preview_interval_ms(self) -> int:
        try:
            fps = float(self.fps_spin.value())
        except Exception:
            fps = self._default_fps()
        return max(1, int(round(1000.0 / max(0.1, fps))))

    def _set_movie_preview_buttons(self, playing: bool) -> None:
        if hasattr(self, "movie_preview_play_button"):
            self.movie_preview_play_button.setEnabled(not playing)
        if hasattr(self, "movie_preview_stop_button"):
            self.movie_preview_stop_button.setEnabled(playing)

    def _start_movie_preview_playback(self) -> None:
        if cv2 is None:
            self.preview_label.setText("OpenCV is not available.")
            return
        if not self._get_timeline():
            self._update_split_status("No output frames")
            return
        if self._movie_preview_timer is None:
            self._movie_preview_timer = QtCore.QTimer(self)
            self._movie_preview_timer.timeout.connect(self._advance_movie_preview_playback)
        start, end = self._preview_playback_bounds()
        current = int(self.preview_slider.value())
        if current < start or current > end:
            self.preview_slider.setValue(start)
        self._movie_preview_timer.setInterval(self._movie_preview_interval_ms())
        self._movie_preview_timer.start()
        self._set_movie_preview_buttons(True)
        self._update_preview()

    def _stop_movie_preview_playback(self, update_status: bool = True) -> None:
        if self._movie_preview_timer is not None:
            self._movie_preview_timer.stop()
        self._set_movie_preview_buttons(False)
        if update_status:
            self._update_split_status("Preview stopped")

    def _advance_movie_preview_playback(self) -> None:
        timeline = self._get_timeline()
        if not timeline:
            self._stop_movie_preview_playback(update_status=False)
            self._update_split_status("No output frames")
            return
        start, end = self._preview_playback_bounds()
        current = max(start, min(end, int(self.preview_slider.value())))
        next_index = current + 1
        if next_index > end:
            if hasattr(self, "movie_preview_loop_check") and self.movie_preview_loop_check.isChecked():
                next_index = start
            else:
                self.preview_slider.setValue(end)
                self._stop_movie_preview_playback(update_status=False)
                self._update_split_status("Preview finished")
                return
        if self.preview_slider.value() == next_index:
            self._update_preview()
        else:
            self.preview_slider.setValue(next_index)

    def _preview_source_frame(self, frame_index: int) -> None:
        original_index = int(getattr(gv, "index", 0) or 0) if gv is not None else None
        width, height = self._output_size()
        try:
            img = self._render_pynud_display_frame(frame_index)
            if img is None:
                array = self._get_frame_array(frame_index)
                if array is None:
                    return
                img = self._array_to_bgr(array, width, height)
            elif img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            img = self._draw_standard_overlays(img, frame_index)
            img = self._draw_z_scale_bar(img)
            self._set_preview_image(img)
        except Exception:
            pass
        finally:
            if original_index is not None:
                self._restore_main_frame(original_index)

    def _focus_source_frame_for_output(self, output_index: int) -> None:
        frame_index = self._source_frame_for_output_index(output_index)
        if frame_index is None:
            return
        if not (0 <= frame_index < self.source_list.count()):
            return
        item = self.source_list.item(frame_index)
        if item is None:
            return
        self.source_list.blockSignals(True)
        try:
            flags = QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Current
            self.source_list.setCurrentItem(item, flags)
            self.source_list.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        finally:
            self.source_list.blockSignals(False)

    def _update_preview(self, *_args) -> None:
        if cv2 is None:
            self.preview_label.setText("OpenCV is not available.")
            return
        duration = len(self._get_timeline())
        if duration <= 0:
            self.preview_frame_label.setText("0 / 0")
            self._last_preview_bgr = None
            self.preview_label.clear()
            self.preview_label.setText("No output frames.")
            return
        idx = max(0, min(duration - 1, int(self.preview_slider.value())))
        self.preview_frame_label.setText(f"{idx + 1} / {duration}")
        original_index = int(getattr(gv, "index", 0) or 0) if gv is not None else None
        try:
            img = self._render_timeline_frame(idx)
            self._set_preview_image(img)
        except Exception as exc:
            self.preview_label.setText(f"Preview failed:\n{exc}")
        finally:
            if original_index is not None:
                self._restore_main_frame(original_index)
        transition_status = self._transition_status_for_output(idx)
        if transition_status:
            self._update_split_status(transition_status)
        else:
            self._update_split_status()
        self._focus_source_frame_for_output(idx)
        self._last_preview_index = idx

    def _preview_settings_changed(self, *_args) -> None:
        self._contrast_limits = None
        if self._movie_preview_timer is not None and self._movie_preview_timer.isActive():
            self._movie_preview_timer.setInterval(self._movie_preview_interval_ms())
        self._update_preview()

    def _set_preview_image(self, bgr: np.ndarray) -> None:
        self._last_preview_bgr = np.ascontiguousarray(bgr)
        if cv2 is None:
            return
        rgb = cv2.cvtColor(self._last_preview_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg.copy())
        if hasattr(self, "preview_scroll_area"):
            viewport_size = self.preview_scroll_area.viewport().size()
        else:
            viewport_size = self.preview_label.size()
        frame_h, frame_w = self._last_preview_bgr.shape[:2]
        base_w, base_h = self._output_size()
        if viewport_size.width() < 80 or viewport_size.height() < 80:
            fallback_size = self.preview_label.size()
            if fallback_size.width() >= 80 and fallback_size.height() >= 80:
                viewport_size = fallback_size
            else:
                viewport_size = QtCore.QSize(max(160, int(base_w)), max(120, int(base_h)))
        scale_ref_w = max(1, base_w)
        scale_ref_h = max(1, base_h)
        if not (hasattr(self, "z_scale_bar_check") and self.z_scale_bar_check.isChecked()):
            scale_ref_w = max(1, frame_w)
            scale_ref_h = max(1, frame_h)
        scale = min(
            max(1, viewport_size.width()) / scale_ref_w,
            max(1, viewport_size.height()) / scale_ref_h,
        )
        scaled_w = max(1, int(round(frame_w * scale)))
        scaled_h = max(1, int(round(frame_h * scale)))
        pix = pix.scaled(scaled_w, scaled_h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview_label.resize(pix.size())
        self.preview_label.setPixmap(pix)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._last_preview_bgr is not None:
            self._set_preview_image(self._last_preview_bgr)

    def _create_video_writer(self, movie_path: str, fmt: str, fps: float, size: Tuple[int, int]):
        if cv2 is None:
            return None, None
        width, height = size
        codecs = ["mp4v", "avc1", "H264"] if fmt == "mp4" else ["XVID", "MJPG"]
        for codec in codecs:
            writer = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if writer.isOpened():
                return writer, codec
            writer.release()
        return None, None

    def _default_movie_path(self, fmt: str) -> str:
        current_file = self._current_file_path()
        folder = os.path.dirname(current_file) if current_file else ""
        if not folder or not os.path.isdir(folder):
            folder = getattr(gv, "movieSaveDir", "") if gv is not None else ""
        if not folder or not os.path.isdir(folder):
            folder = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation)
        base = os.path.splitext(os.path.basename(current_file))[0] if current_file else "afm_movie"
        return os.path.join(folder, f"{base}_edited.{fmt}")

    def _export_movie(self) -> None:
        self._stop_movie_preview_playback(update_status=False)
        if cv2 is None:
            QtWidgets.QMessageBox.critical(self, "OpenCV Missing", "OpenCV is required to export movies.")
            return
        timeline = self._get_timeline()
        if not timeline:
            QtWidgets.QMessageBox.warning(self, "No Clips", "No clips are available for export.")
            return
        render_indices = list(range(len(timeline)))

        fmt = self.format_combo.currentText()
        default_path = self._default_movie_path(fmt)
        options = QtWidgets.QFileDialog.Options()
        if sys.platform != "darwin":
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
        movie_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Edited AFM Movie As",
            default_path,
            f"{fmt.upper()} files (*.{fmt})",
            options=options,
        )
        if not movie_path:
            return
        if not movie_path.lower().endswith(f".{fmt}"):
            movie_path += f".{fmt}"
        if os.path.exists(movie_path):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Overwrite Movie",
                f"File '{os.path.basename(movie_path)}' already exists.\nOverwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        original_index = int(getattr(gv, "index", 0) or 0) if gv is not None else None
        fps = float(self.fps_spin.value())
        try:
            first_frame = self._render_timeline_frame(render_indices[0])
        except Exception as exc:
            if original_index is not None:
                self._restore_main_frame(original_index)
            QtWidgets.QMessageBox.critical(self, "Export Failed", f"Failed to render first frame:\n{exc}")
            traceback.print_exc()
            return

        size = (self._make_even(first_frame.shape[1]), self._make_even(first_frame.shape[0]))
        if first_frame.shape[1] != size[0] or first_frame.shape[0] != size[1]:
            first_frame = cv2.resize(first_frame, size, interpolation=cv2.INTER_AREA)

        writer, codec = self._create_video_writer(movie_path, fmt, fps, size)
        if writer is None:
            if original_index is not None:
                self._restore_main_frame(original_index)
            QtWidgets.QMessageBox.critical(self, "Export Failed", "Failed to initialize VideoWriter.")
            return

        progress = QtWidgets.QProgressDialog("Exporting edited AFM movie...", "Cancel", 0, len(render_indices), self)
        progress.setWindowTitle("Export AFM Movie")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        self.export_button.setEnabled(False)
        self._contrast_limits = None

        cancelled = False
        try:
            for render_pos, out_idx in enumerate(render_indices):
                if progress.wasCanceled():
                    cancelled = True
                    break
                progress.setValue(render_pos)
                progress.setLabelText(f"Rendering frame {render_pos + 1} / {len(render_indices)}")
                QtWidgets.QApplication.processEvents()
                frame = first_frame if render_pos == 0 else self._render_timeline_frame(out_idx)
                if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                writer.write(np.ascontiguousarray(frame))
            progress.setValue(len(render_indices))
        except Exception as exc:
            writer.release()
            if os.path.exists(movie_path):
                try:
                    os.remove(movie_path)
                except Exception:
                    pass
            if original_index is not None:
                self._restore_main_frame(original_index)
            QtWidgets.QMessageBox.critical(self, "Export Failed", f"Failed to create movie:\n{exc}")
            traceback.print_exc()
            return
        finally:
            writer.release()
            self.export_button.setEnabled(True)

        if cancelled:
            if os.path.exists(movie_path):
                try:
                    os.remove(movie_path)
                except Exception:
                    pass
            if original_index is not None:
                self._restore_main_frame(original_index)
            QtWidgets.QMessageBox.information(self, "Cancelled", "Movie export was cancelled.")
            return

        self._run_optional_reencode(movie_path, fmt, fps)
        if original_index is not None:
            self._restore_main_frame(original_index)
        QtWidgets.QMessageBox.information(
            self,
            "Movie Created",
            f"Edited AFM movie created:\n{os.path.basename(movie_path)}\n\n"
            f"Frames: {len(render_indices)}\n"
            f"FPS: {fps:g}\n"
            f"Codec: {codec}\n"
            f"Save location: {os.path.dirname(movie_path)}",
        )

    def _run_optional_reencode(self, movie_path: str, fmt: str, fps: float) -> None:
        quality = int(self.quality_spin.value())
        try:
            if fmt == "mp4" and self.main_window is not None and hasattr(self.main_window, "fix_mp4_for_quicktime"):
                self.main_window.fix_mp4_for_quicktime(movie_path, quality, fps)
            elif fmt == "avi" and self.main_window is not None and hasattr(self.main_window, "fix_avi_for_quality"):
                self.main_window.fix_avi_for_quality(movie_path, quality, fps)
        except Exception as exc:
            print(f"[WARNING] Movie re-encode failed: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_ui_settings()
        self._stop_movie_preview_playback(update_status=False)
        self._stop_thumbnail_timer()
        super().closeEvent(event)


def create_plugin(main_window):
    return AFMMovieEditorWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "AFMMovieEditorWindow"]
