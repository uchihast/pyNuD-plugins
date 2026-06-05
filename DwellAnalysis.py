import sys
import os
import random
import math
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
import globalvals as gv

PLUGIN_NAME = "Dwell Analysis"

# ─────────────────────────────────────────────
# Substrate palette (BGR for OpenCV, RGB for Qt)
# ─────────────────────────────────────────────
SUBSTRATE_PALETTE_BGR = [
    (170, 170, 0),    # Teal
    (244, 133, 66),   # Blue
    (176, 39, 156),   # Purple
    (90, 180, 0),     # Green
    (47, 50, 220),    # Crimson
    (220, 150, 0),    # Azure
]
SUBSTRATE_PALETTE_RGB = [
    (0, 170, 170),
    (66, 133, 244),
    (156, 39, 176),
    (0, 180, 90),
    (220, 50, 47),
    (0, 150, 220),
]


def get_substrate_color_index(substrate):
    """Return palette index for a substrate dict."""
    palette_len = max(1, len(SUBSTRATE_PALETTE_BGR))
    return int(substrate.get('color_id', substrate.get('id', 0))) % palette_len

HELP_HTML_EN = """
<h1>Dwell Analysis</h1>
<h2>Overview</h2>
<p>Dwell Analysis quantifies how long a molecule stays at a location across frames.
Place marks on the main image, save per frame (Memorize), then run Finalization to
link marks across frames and compute dwell times.</p>
<h2>Tabs</h2>
<ul>
  <li><strong>Dwell Analysis</strong> – Mark molecules and run Finalization.</li>
    <li><strong>Restore</strong> – On frame N+1, restore all frame N memorized marks as unmemorized temporary marks.</li>
        <li><strong>Show ID</strong> – Display molecule IDs next to markers in the image overlay (default off).</li>
  <li><strong>Substrate</strong> – Draw polyline masks for linear substrates (e.g. DNA).</li>
  <li><strong>Proximity Analysis</strong> – After Finalization, compute substrate assignment
      and per-frame neighbor counts. Export enriched CSV.</li>
</ul>
<h2>Substrate Editor</h2>
<ul>
  <li><strong>New Substrate</strong>: Enter draw mode → left-click to add nodes → right-click or
      Finish Substrate to confirm.</li>
  <li><strong>Undo Last Point</strong>: Remove the last node while drawing.</li>
  <li><strong>Edit Substrate</strong>: Drag nodes to move, right-click a node to delete.</li>
  <li><strong>Delete Selected</strong>: Remove the selected substrate.</li>
  <li><strong>Clear All</strong>: Remove all substrates.</li>
  <li><strong>Substrate Width (nm)</strong>: Maximum perpendicular distance from centerline for
      a molecule to be assigned to that substrate.</li>
</ul>
<h2>Proximity Analysis</h2>
<ul>
  <li>Set <strong>Proximity Radius R (nm)</strong> and click <strong>Run Proximity Analysis</strong>.</li>
  <li>The table shows every molecule×frame event with substrate ID, neighbor count, and neighbor
      molecule indices.</li>
  <li>Use <strong>Export CSV</strong> to save the enriched table.</li>
</ul>
"""

HELP_HTML_JA = """
<h1>ドウェル解析</h1>
<h2>概要</h2>
<p>ドウェル解析は、分子が同じ位置に何フレーム存在したかを定量する機能です。
メイン画像にマークを付け、フレームごとに Memorize して保存し、Finalization でリンク・計算します。</p>
<h2>タブ</h2>
<ul>
  <li><strong>Dwell Analysis</strong> – 分子のマークと Finalization。</li>
    <li><strong>Restore</strong> – フレーム N+1 で、フレーム N の Memorize 済みマークを未保存マークとして復元。</li>
        <li><strong>Show ID</strong> – 画像オーバーレイでマーカー横に分子IDを表示（初期値はOFF）。</li>
  <li><strong>Substrate</strong> – 線状基質（DNA 等）の折線マスクを描画。</li>
  <li><strong>Proximity Analysis</strong> – Finalization 後、基質の割り当てとフレームごとの
      隣接分子数を計算。拡張 CSV をエクスポート。</li>
</ul>
<h2>基質エディタ</h2>
<ul>
  <li><strong>New Substrate</strong>: 描画モードに入り → 左クリックでノード追加 →
      右クリックまたは Finish Substrate で確定。</li>
  <li><strong>Undo Last Point</strong>: 描画中に最後のノードを取り消し。</li>
  <li><strong>Edit Substrate</strong>: ノードのドラッグで移動、右クリックで削除。</li>
  <li><strong>Delete Selected</strong>: 選択した基質を削除。</li>
  <li><strong>Clear All</strong>: 全基質を削除。</li>
  <li><strong>Substrate Width (nm)</strong>: 分子が基質に割り当てられる最大垂直距離。</li>
</ul>
<h2>近接解析</h2>
<ul>
  <li><strong>Proximity Radius R (nm)</strong> を設定して <strong>Run Proximity Analysis</strong> をクリック。</li>
  <li>テーブルに分子×フレームごとの基質ID・隣接数・隣接IDを表示。</li>
  <li><strong>Export CSV</strong> で拡張テーブルを保存。</li>
</ul>
"""


# ══════════════════════════════════════════════════════════════════
#  Helper: point-to-polyline distance (nm or pixel)
# ══════════════════════════════════════════════════════════════════
def _point_to_polyline_dist(px, py, points):
    """Return the minimum distance from point (px,py) to polyline segments."""
    if len(points) == 0:
        return float('inf')
    if len(points) == 1:
        return np.hypot(px - points[0][0], py - points[0][1])
    min_dist = float('inf')
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 == 0:
            d = np.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
            cx, cy = ax + t * abx, ay + t * aby
            d = np.hypot(px - cx, py - cy)
        min_dist = min(min_dist, d)
    return min_dist


def _numpy_to_pixmap(image_array):
    """Convert a grayscale or BGR OpenCV image into a QPixmap."""
    if image_array is None or not isinstance(image_array, np.ndarray) or image_array.size == 0:
        return QtGui.QPixmap()

    if image_array.ndim == 2:
        height, width = image_array.shape[:2]
        q_image = QtGui.QImage(image_array.data, width, height, width, QtGui.QImage.Format_Grayscale8)
        return QtGui.QPixmap.fromImage(q_image.copy())

    if image_array.ndim == 3 and image_array.shape[2] >= 3:
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        q_image = QtGui.QImage(rgb_image.data, width, height, rgb_image.strides[0], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_image.copy())

    return QtGui.QPixmap()


def _compose_dwell_overlay(target_pixmap):
    """Paint dwell overlays onto a target pixmap and return the result."""
    try:
        if target_pixmap is None or target_pixmap.isNull():
            return target_pixmap
        if not bool(getattr(gv, 'dwell_show_overlay', True)):
            return target_pixmap

        label_items = []
        gv.dwell_id_label_items = []
        show_marker_id = bool(getattr(gv, 'dwell_show_id', False))
        is_dwell_active = bool(getattr(gv, 'dwell_analysis_active', False))
        is_draw_active = bool(getattr(gv, 'dwell_substrate_draw_active', False))
        is_edit_active = bool(getattr(gv, 'dwell_substrate_edit_mode', False))
        substrates = list(getattr(gv, 'dwell_substrates', []))
        draw_pts = list(getattr(gv, 'dwell_substrate_draw_points', []))
        temp_marks = list(getattr(gv, 'dwell_temp_marks', []))
        molecules = list(getattr(gv, 'dwell_molecules', []))
        current_frame = int(getattr(gv, 'index', 0))
        marker_records = dict(getattr(gv, 'dwell_marker_records', {}))

        if not (is_dwell_active or is_draw_active or is_edit_active or substrates):
            return target_pixmap

        painter = QtGui.QPainter(target_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

        def _scaled_point(x, y):
            return QtCore.QPointF(float(x), float(y))

        substrate_linewidth_px = max(1, int(getattr(gv, 'dwell_substrate_linewidth_px', 2)))
        marker_size_px = max(1, int(getattr(gv, 'dwell_marker_size_px', 8)))
        marker_half_px = marker_size_px / 2.0

        # Confirmed substrates
        for substrate in substrates:
            color_idx = get_substrate_color_index(substrate)
            rgb = SUBSTRATE_PALETTE_RGB[color_idx % len(SUBSTRATE_PALETTE_RGB)]
            painter.setPen(QtGui.QPen(QtGui.QColor(*rgb), substrate_linewidth_px))
            painter.setBrush(QtCore.Qt.NoBrush)
            points = substrate.get('points', [])
            for idx in range(len(points) - 1):
                p1 = points[idx]
                p2 = points[idx + 1]
                painter.drawLine(_scaled_point(p1[0], p1[1]), _scaled_point(p2[0], p2[1]))
            for node_x, node_y in points:
                painter.drawEllipse(QtCore.QRectF(node_x - 2, node_y - 2, 4, 4))

        # In-progress substrate polyline
        if is_draw_active and len(draw_pts) >= 1:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), substrate_linewidth_px))
            for idx in range(len(draw_pts) - 1):
                p1 = draw_pts[idx]
                p2 = draw_pts[idx + 1]
                painter.drawLine(_scaled_point(p1[0], p1[1]), _scaled_point(p2[0], p2[1]))
            for node_x, node_y in draw_pts:
                painter.drawEllipse(QtCore.QRectF(node_x - 3, node_y - 3, 6, 6))

        if is_dwell_active:
            memorized_lookup = {
                (int(round(px)), int(round(py)))
                for (px, py) in getattr(gv, 'dwell_marks', {}).get(current_frame, [])
            }

            active_record_positions = []
            for rec_id, record in marker_records.items():
                start_frame = int(record.get('start_frame', 0))
                stop_frame = int(record.get('stop_frame', -1))
                if current_frame < start_frame:
                    continue
                if stop_frame >= 0 and current_frame > stop_frame:
                    continue
                loc_map = record.get('location_by_frame', {})
                if current_frame in loc_map:
                    rx, ry = loc_map[current_frame]
                else:
                    previous_frames = [frame for frame in loc_map.keys() if frame <= current_frame]
                    if not previous_frames:
                        continue
                    rx, ry = loc_map[max(previous_frames)]
                active_record_positions.append((int(rec_id), float(rx), float(ry)))

            dx_nm = getattr(gv, 'XScanSize', 0) / max(1, getattr(gv, 'XPixel', 1))
            dy_nm = getattr(gv, 'YScanSize', 0) / max(1, getattr(gv, 'YPixel', 1))
            radius_nm = float(getattr(gv, 'dwell_mol_radius', 5.0))

            def _nearest_record_id(mx, my):
                nearest_id = -1
                nearest_dist = float('inf')
                for rec_id, rx, ry in active_record_positions:
                    dist_nm = float(np.hypot((mx - rx) * dx_nm, (my - ry) * dy_nm))
                    if dist_nm <= radius_nm and dist_nm < nearest_dist:
                        nearest_dist = dist_nm
                        nearest_id = rec_id
                return nearest_id

            molecule_neighbors = {}
            for mol in molecules:
                if current_frame in mol.get('frames', []):
                    frame_idx = mol['frames'].index(current_frame)
                    pos = mol['positions'][frame_idx]
                    pos_key = (int(round(pos[0])), int(round(pos[1])))
                    nc_list = mol.get('neighbor_count', [])
                    molecule_neighbors[pos_key] = nc_list[frame_idx] if frame_idx < len(nc_list) else 0

            for mark_x, mark_y in temp_marks:
                mark_x_i = int(mark_x)
                mark_y_i = int(mark_y)
                mark_key = (int(round(mark_x)), int(round(mark_y)))
                is_memorized = mark_key in memorized_lookup

                marker_color = QtGui.QColor(255, 0, 0)
                if substrates:
                    mark_x_nm = mark_x * dx_nm
                    mark_y_nm = mark_y * dy_nm
                    best_color_idx = -1
                    best_dist = float('inf')
                    for substrate in substrates:
                        pts_nm = [(px * dx_nm, py * dy_nm) for (px, py) in substrate.get('points', [])]
                        dist = _point_to_polyline_dist(mark_x_nm, mark_y_nm, pts_nm)
                        if dist < best_dist:
                            best_dist = dist
                            best_color_idx = get_substrate_color_index(substrate)
                    if best_color_idx >= 0:
                        marker_rgb = SUBSTRATE_PALETTE_RGB[best_color_idx % len(SUBSTRATE_PALETTE_RGB)]
                        marker_color = QtGui.QColor(*marker_rgb)

                painter.setPen(QtGui.QPen(marker_color, 1))
                painter.setBrush(marker_color if is_memorized else QtCore.Qt.NoBrush)
                painter.drawRect(QtCore.QRectF(mark_x_i - marker_half_px, mark_y_i - marker_half_px, marker_size_px, marker_size_px))

                n_count = molecule_neighbors.get(mark_key, 0)
                if n_count > 0:
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255), 1))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawText(QtCore.QPointF(mark_x_i + 8, mark_y_i - 8), str(n_count))

                if show_marker_id:
                    marker_id = _nearest_record_id(mark_x, mark_y)
                    if marker_id >= 0:
                        label_items.append({
                            'x': float(mark_x_i + 8),
                            'y': float(mark_y_i + 12),
                            'text': f"ID:{marker_id}",
                        })

        if show_marker_id and label_items:
            font = QtGui.QFont(getattr(gv, 'qt_font_family', 'Segoe UI'))
            font.setPointSizeF(float(getattr(gv, 'dwell_id_font_pt', 8.0)))
            font.setStyleStrategy(QtGui.QFont.PreferAntialias)
            painter.setFont(font)

            outline_pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 180))
            outline_pen.setWidth(1)
            text_pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
            text_pen.setWidth(1)

            for item in label_items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get('text', ''))
                if not text:
                    continue
                x_img = float(item.get('x', 0.0))
                y_img = float(item.get('y', 0.0))
                base_pos = QtCore.QPointF(x_img, y_img)
                painter.setPen(outline_pen)
                for off in (
                    QtCore.QPointF(-1, 0), QtCore.QPointF(1, 0),
                    QtCore.QPointF(0, -1), QtCore.QPointF(0, 1),
                ):
                    painter.drawText(base_pos + off, text)
                painter.setPen(text_pen)
                painter.drawText(base_pos, text)

        painter.end()
        gv.dwell_id_label_items = label_items
        return target_pixmap
    except Exception as e:
        print(f"[WARNING] _compose_dwell_overlay: {e}")
        return target_pixmap


class DwellDisplayWindow(QtWidgets.QWidget):
    """Dedicated dwell overlay panel that mirrors the main AFM display."""

    closed = QtCore.pyqtSignal()

    def __init__(self, main_window, parent=None, embedded=False):
        super().__init__(parent)
        self.parent_win = main_window
        self.embedded = bool(embedded)
        self.original_pixmap = None
        self._dwell_marker_drag_active = False
        self._dwell_marker_drag_rec_id = -1
        self._dwell_edit_drag_sub = -1
        self._dwell_edit_drag_node = -1
        self._dwell_marker_drag_win = None
        self._dwell_edit_drag_win = None

        if not self.embedded:
            try:
                from window_manager import register_pyNuD_window
                register_pyNuD_window(self, "sub")
            except ImportError:
                pass

            self.setWindowFlags(
                QtCore.Qt.Window |
                QtCore.Qt.WindowCloseButtonHint |
                QtCore.Qt.WindowMinMaxButtonsHint
            )
            self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Dwell View")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: none;")
        self.image_label.setMinimumSize(0, 0)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.image_label, 1)

        self.info_bar = QtWidgets.QLabel("Dwell overlay follows the main AFM frame.")
        self.info_bar.setStyleSheet("background-color: white; color: black; padding: 4px 8px; font-size: 11px;")
        self.info_bar.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(self.info_bar, 0)

        self._bind_parent_updates()
        if not self.embedded:
            self._restore_window()
        QtCore.QTimer.singleShot(0, self.refresh_display)
        self.refresh_display()

    def _bind_parent_updates(self):
        if self.parent_win and hasattr(self.parent_win, 'frameChanged'):
            try:
                self.parent_win.frameChanged.connect(self.refresh_display)
            except Exception:
                pass

    def _restore_window(self):
        from helperFunctions import restore_window_geometry
        restore_window_geometry(self, 'DwellDisplayWindow', 1040, 240, 720, 640)

    def refresh_display(self):
        try:
            source_pixmap = _numpy_to_pixmap(getattr(gv, 'dspimg', None))
            if source_pixmap.isNull():
                self.image_label.clear()
                return

            composed_pixmap = _compose_dwell_overlay(source_pixmap.copy())
            self.original_pixmap = composed_pixmap
            target_size = self.image_label.size()
            if target_size.width() <= 0 or target_size.height() <= 0:
                target_size = self.original_pixmap.size()
            scaled_pixmap = self.original_pixmap.scaled(
                target_size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        except Exception as e:
            print(f"[WARNING] DwellDisplayWindow.refresh_display: {e}")

    def map_widget_to_image_coords(self, widget_pos):
        if self.image_label.pixmap() is None or self.image_label.pixmap().isNull():
            return None

        scaled_pixmap = self.image_label.pixmap()
        label_rect = self.image_label.rect()
        pixmap_rect = scaled_pixmap.rect()

        px_offset = (label_rect.width() - pixmap_rect.width()) / 2
        py_offset = (label_rect.height() - pixmap_rect.height()) / 2

        x_on_scaled_pixmap = widget_pos.x() - px_offset
        y_on_scaled_pixmap = widget_pos.y() - py_offset

        if not (0 <= x_on_scaled_pixmap < pixmap_rect.width() and 0 <= y_on_scaled_pixmap < pixmap_rect.height()):
            return None

        original_size = self.original_pixmap.size() if self.original_pixmap is not None else None
        scaled_size = scaled_pixmap.size()
        if original_size is None or scaled_size.width() == 0 or scaled_size.height() == 0:
            return None

        x_ratio = original_size.width() / scaled_size.width()
        y_ratio = original_size.height() / scaled_size.height()
        final_x = x_on_scaled_pixmap * x_ratio
        final_y = y_on_scaled_pixmap * y_ratio
        return int(final_x), int(final_y)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        label_pos = self.image_label.mapFrom(self, event.pos())
        controller = getattr(self.parent_win, 'dwell_analysis_window', None)

        if controller is None:
            super().mousePressEvent(event)
            return

        if getattr(gv, 'dwell_analysis_active', False) and event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.ControlModifier:
            coords = self.map_widget_to_image_coords(label_pos)
            if coords:
                rec_id = controller.marker_begin_drag(coords[0], coords[1], getattr(gv, 'index', 0))
                if rec_id is not None and int(rec_id) >= 0:
                    self._dwell_marker_drag_active = True
                    self._dwell_marker_drag_rec_id = int(rec_id)
                    self._dwell_marker_drag_win = controller
                    event.accept()
                    return

        if getattr(gv, 'dwell_substrate_draw_active', False):
            coords = self.map_widget_to_image_coords(label_pos)
            if coords:
                if event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.NoModifier:
                    controller.substrate_add_draw_point(coords[0], coords[1])
                    event.accept()
                    return
                if event.button() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.NoModifier:
                    controller.substrate_finish_draw_on_rightclick()
                    event.accept()
                    return

        if getattr(gv, 'dwell_substrate_edit_mode', False):
            coords = self.map_widget_to_image_coords(label_pos)
            if coords:
                cx, cy = coords
                hit_radius = 8
                sub_hit, node_hit = -1, -1
                for sub_idx, substrate in enumerate(getattr(gv, 'dwell_substrates', [])):
                    for node_idx, (node_x, node_y) in enumerate(substrate.get('points', [])):
                        if abs(cx - node_x) <= hit_radius and abs(cy - node_y) <= hit_radius:
                            sub_hit, node_hit = sub_idx, node_idx
                            break
                    if sub_hit >= 0:
                        break
                if event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.NoModifier and sub_hit >= 0:
                    self._dwell_edit_drag_sub = sub_hit
                    self._dwell_edit_drag_node = node_hit
                    self._dwell_edit_drag_win = controller
                    event.accept()
                    return
                if event.button() == QtCore.Qt.RightButton and event.modifiers() == QtCore.Qt.NoModifier and sub_hit >= 0:
                    controller.substrate_edit_delete_node(sub_hit, node_hit)
                    event.accept()
                    return

        if getattr(gv, 'dwell_analysis_active', False) and event.button() == QtCore.Qt.LeftButton and event.modifiers() == QtCore.Qt.NoModifier:
            coords = self.map_widget_to_image_coords(label_pos)
            if coords:
                controller.add_mark(coords[0], coords[1])
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        label_pos = self.image_label.mapFrom(self, event.pos())
        controller = getattr(self.parent_win, 'dwell_analysis_window', None)

        if controller is None:
            super().mouseMoveEvent(event)
            return

        if self._dwell_marker_drag_active and (event.buttons() & QtCore.Qt.LeftButton):
            coords = self.map_widget_to_image_coords(label_pos)
            rec_id = self._dwell_marker_drag_rec_id
            if coords and rec_id >= 0:
                controller.marker_drag_to(rec_id, coords[0], coords[1], getattr(gv, 'index', 0))
            event.accept()
            return

        if self._dwell_edit_drag_sub >= 0 and (event.buttons() & QtCore.Qt.LeftButton):
            coords = self.map_widget_to_image_coords(label_pos)
            if coords:
                controller.substrate_edit_drag_node(self._dwell_edit_drag_sub, self._dwell_edit_drag_node, coords[0], coords[1])
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        label_pos = self.image_label.mapFrom(self, event.pos())
        controller = getattr(self.parent_win, 'dwell_analysis_window', None)

        if controller is None:
            super().mouseReleaseEvent(event)
            return

        if event.button() == QtCore.Qt.LeftButton and self._dwell_marker_drag_active:
            coords = self.map_widget_to_image_coords(label_pos)
            if coords and self._dwell_marker_drag_rec_id >= 0:
                controller.marker_end_drag(self._dwell_marker_drag_rec_id, coords[0], coords[1], getattr(gv, 'index', 0))
            self._dwell_marker_drag_active = False
            self._dwell_marker_drag_rec_id = -1
            self._dwell_marker_drag_win = None
            event.accept()
            return

        if event.button() == QtCore.Qt.LeftButton and self._dwell_edit_drag_sub >= 0:
            self._dwell_edit_drag_sub = -1
            self._dwell_edit_drag_node = -1
            self._dwell_edit_drag_win = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_display()

    def closeEvent(self, event):
        if self.embedded:
            event.accept()
            return

        try:
            geo = self.geometry()
            ws = getattr(gv, 'windowSettings', {})
            ws['DwellDisplayWindow'] = {
                'width': geo.width(), 'height': geo.height(),
                'x': geo.x(), 'y': geo.y(),
                'visible': False, 'title': self.windowTitle(),
                'class_name': 'DwellDisplayWindow'
            }
            gv.windowSettings = ws
            if self.parent_win and hasattr(self.parent_win, 'saveAllInitialParams'):
                self.parent_win.saveAllInitialParams()
        except Exception as e:
            print(f"[WARNING] DwellDisplayWindow save settings: {e}")

        try:
            self.closed.emit()
        except Exception:
            pass

        super().closeEvent(event)

    def cleanup_on_unload(self):
        self._dwell_marker_drag_active = False
        self._dwell_marker_drag_rec_id = -1
        self._dwell_edit_drag_sub = -1
        self._dwell_edit_drag_node = -1
        self._dwell_marker_drag_win = None
        self._dwell_edit_drag_win = None
        self.original_pixmap = None
        try:
            self.image_label.clear()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════
class DwellAnalysisWindow(QtWidgets.QMainWindow):
    """
    Dwell Analysis Window with Substrate Mask and Proximity Analysis support.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.parent_win = main_window
        self.dwell_display_window = None
        self._window_settings_restored = False
        if self.parent_win is not None:
            try:
                setattr(self.parent_win, 'dwell_analysis_window', self)
            except Exception:
                pass

        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self, "sub")
        except ImportError:
            pass

        self.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.WindowCloseButtonHint |
            QtCore.Qt.WindowMinMaxButtonsHint
        )
        self.setWindowTitle("Dwell Analysis")

        # ── gv initialisation ──────────────────────────────────────
        defaults = {
            'dwell_marks': {},
            'dwell_temp_marks': [],
            'dwell_frame_status': {},
            'dwell_marker_records': {},
            'dwell_next_marker_id': 0,
            'dwell_restore_source_frame': -1,
            'dwell_restore_backup': [],
            'dwell_restored_from_previous': False,
            'dwell_analysis_active': False,
            'dwell_show_overlay': True,
            'dwell_show_id': False,
            'dwell_mol_radius': 5.0,
            'dwell_marker_size_px': 8,
            'dwell_substrate_linewidth_px': 2,
            'dwell_molecules': [],
            'dwell_last_image_index': -1,
            # Substrate
            'dwell_substrates': [],
            'dwell_substrate_draw_active': False,
            'dwell_substrate_draw_points': [],  # WIP polyline
            'dwell_substrate_edit_mode': False,
            'dwell_substrate_selected': -1,     # index in gv.dwell_substrates
            # Proximity
            'dwell_proximity_radius': 20.0,
            'dwell_substrate_tolerance': 10.0,
            'dwell_proximity_results': [],
            'dwell_hist_bin_width_s': 0.45,
        }
        for k, v in defaults.items():
            if not hasattr(gv, k):
                setattr(gv, k, v)

        self._setup_ui()

        if self.parent_win and hasattr(self.parent_win, 'frameChanged'):
            self.parent_win.frameChanged.connect(self._on_frame_changed)

        self._on_frame_changed(getattr(gv, 'index', 0))
        self._update_info()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._window_settings_restored:
            self.restoreWindowSettings()
            self._window_settings_restored = True
        QtCore.QTimer.singleShot(0, self._restore_splitter_settings)
        self._refresh_display()

    # ─────────────────────────────────────────────
    #  UI Setup
    # ─────────────────────────────────────────────
    def _setup_ui(self):
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        root = QtWidgets.QVBoxLayout(self.central_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Menu bar
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        is_ja = QtCore.QLocale().language() == QtCore.QLocale.Japanese
        help_menu = menu_bar.addMenu("ヘルプ" if is_ja else "Help")
        manual_act = help_menu.addAction("マニュアル" if is_ja else "Manual")
        manual_act.triggered.connect(self._show_help)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self.central_widget)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)

        self.dwell_display_window = DwellDisplayWindow(self.parent_win, parent=self.splitter, embedded=True)
        self.splitter.addWidget(self.dwell_display_window)

        # Tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([420, 260])

        self.tabs.addTab(self._build_dwell_tab(),     "Dwell Analysis")
        self.tabs.addTab(self._build_substrate_tab(), "Substrate")
        self.tabs.addTab(self._build_proximity_tab(), "Proximity Analysis")
        self.tabs.addTab(self._build_histogram_tab(), "Histogram")

        self.shortcut_frame_prev = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+A"), self)
        self.shortcut_frame_prev.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcut_frame_prev.activated.connect(self._frame_prev)

        self.shortcut_frame_next = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+D"), self)
        self.shortcut_frame_next.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcut_frame_next.activated.connect(self._frame_next)

    # ── Tab 1: Dwell Analysis ─────────────────────────────────────
    def _build_dwell_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(4)

        # Row 0: Start / Restore / Reset
        self.start_marking_btn = QtWidgets.QPushButton("Start Marking")
        self.start_marking_btn.setCheckable(True)
        self.start_marking_btn.clicked.connect(self._toggle_marking)
        btn_grid.addWidget(self.start_marking_btn, 0, 0)

        self.restore_btn = QtWidgets.QPushButton("Restore")
        self.restore_btn.clicked.connect(self._restore_from_previous_memorized)
        self.restore_btn.setEnabled(False)
        btn_grid.addWidget(self.restore_btn, 0, 1)

        self.open_dwell_view_btn = QtWidgets.QPushButton("Focus View")
        self.open_dwell_view_btn.clicked.connect(self._focus_display_panel)
        btn_grid.addWidget(self.open_dwell_view_btn, 0, 3)

        self.reset_btn = QtWidgets.QPushButton("All Reset")
        self.reset_btn.clicked.connect(self._all_reset)
        btn_grid.addWidget(self.reset_btn, 0, 2)

        # Row 1: Frame navigation  (below Start / Restore / Reset)
        self.frame_prev_btn = QtWidgets.QPushButton("◀  Frame -")
        self.frame_prev_btn.clicked.connect(self._frame_prev)
        btn_grid.addWidget(self.frame_prev_btn, 1, 0)

        self.frame_next_btn = QtWidgets.QPushButton("Frame +  ▶")
        self.frame_next_btn.clicked.connect(self._frame_next)
        btn_grid.addWidget(self.frame_next_btn, 1, 2)

        # Frame indicator (spans all columns)
        self.frame_label = QtWidgets.QLabel("Frame: – / –")
        self.frame_label.setAlignment(QtCore.Qt.AlignCenter)
        self.frame_label.setStyleSheet("color: #555; font-size: 11px;")
        btn_grid.addWidget(self.frame_label, 2, 0, 1, 3)

        # Row 3: Memorize / Finalization
        self.memorize_btn = QtWidgets.QPushButton("Memorize")
        self.memorize_btn.clicked.connect(self._memorize)
        btn_grid.addWidget(self.memorize_btn, 3, 0)

        self.finalization_btn = QtWidgets.QPushButton("Finalization")
        self.finalization_btn.clicked.connect(self._finalize)
        btn_grid.addWidget(self.finalization_btn, 3, 2)

        # Row 4: Export binary CSV (full width)
        self.export_btn = QtWidgets.QPushButton("Export (Binary CSV)")
        self.export_btn.clicked.connect(self._export_binary_csv)
        btn_grid.addWidget(self.export_btn, 4, 0, 1, 3)

        # Row 5: Save / Load session
        self.save_session_btn = QtWidgets.QPushButton("Save Session")
        self.save_session_btn.clicked.connect(self._save_session)
        btn_grid.addWidget(self.save_session_btn, 5, 0)

        self.load_session_btn = QtWidgets.QPushButton("Load Session")
        self.load_session_btn.clicked.connect(self._load_session)
        btn_grid.addWidget(self.load_session_btn, 5, 2)

        lay.addLayout(btn_grid)

        # ── Parameters ──────────────────────────────────────────
        params = QtWidgets.QFormLayout()
        params.setSpacing(3)

        self.mol_radius_sb = QtWidgets.QDoubleSpinBox()
        self.mol_radius_sb.setRange(0.1, 100.0)
        self.mol_radius_sb.setValue(gv.dwell_mol_radius)
        self.mol_radius_sb.setSuffix(" nm")
        self.mol_radius_sb.valueChanged.connect(lambda v: setattr(gv, 'dwell_mol_radius', v))
        params.addRow("Mol Radius:", self.mol_radius_sb)

        self.show_id_cb = QtWidgets.QCheckBox("Show ID")
        self.show_id_cb.setChecked(bool(getattr(gv, 'dwell_show_id', False)))
        self.show_id_cb.toggled.connect(self._toggle_show_id)
        params.addRow("", self.show_id_cb)

        self.marker_size_sb = QtWidgets.QSpinBox()
        self.marker_size_sb.setRange(1, 100)
        self.marker_size_sb.setValue(int(getattr(gv, 'dwell_marker_size_px', 8)))
        self.marker_size_sb.valueChanged.connect(lambda v: setattr(gv, 'dwell_marker_size_px', int(v)))
        params.addRow("Marker Size (px):", self.marker_size_sb)

        self.mark_num_label = QtWidgets.QLabel("0")
        params.addRow("Mark Number:", self.mark_num_label)

        self.total_num_label = QtWidgets.QLabel("0")
        params.addRow("Total Number:", self.total_num_label)

        lay.addLayout(params)
        lay.addStretch()
        return w

    # ── Tab 2: Substrate Editor ────────────────────────────────────
    def _build_substrate_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # Row 1 buttons
        row1 = QtWidgets.QHBoxLayout()
        self.new_sub_btn = QtWidgets.QPushButton("New Substrate")
        self.new_sub_btn.setCheckable(True)
        self.new_sub_btn.clicked.connect(self._toggle_draw_substrate)
        row1.addWidget(self.new_sub_btn)

        self.finish_sub_btn = QtWidgets.QPushButton("Finish Substrate")
        self.finish_sub_btn.clicked.connect(self._finish_substrate)
        self.finish_sub_btn.setEnabled(False)
        row1.addWidget(self.finish_sub_btn)

        self.undo_sub_btn = QtWidgets.QPushButton("Undo Last Point")
        self.undo_sub_btn.clicked.connect(self._undo_last_point)
        self.undo_sub_btn.setEnabled(False)
        row1.addWidget(self.undo_sub_btn)
        lay.addLayout(row1)

        # Row 2 buttons
        row2 = QtWidgets.QHBoxLayout()
        self.edit_sub_btn = QtWidgets.QPushButton("Edit Substrate")
        self.edit_sub_btn.setCheckable(True)
        self.edit_sub_btn.clicked.connect(self._toggle_edit_substrate)
        row2.addWidget(self.edit_sub_btn)

        self.del_sub_btn = QtWidgets.QPushButton("Delete Selected")
        self.del_sub_btn.clicked.connect(self._delete_selected_substrate)
        row2.addWidget(self.del_sub_btn)

        self.clear_sub_btn = QtWidgets.QPushButton("Clear All")
        self.clear_sub_btn.clicked.connect(self._clear_all_substrates)
        row2.addWidget(self.clear_sub_btn)

        self.shuffle_sub_color_btn = QtWidgets.QPushButton("Shuffle Colors")
        self.shuffle_sub_color_btn.clicked.connect(self._shuffle_substrate_colors)
        row2.addWidget(self.shuffle_sub_color_btn)
        lay.addLayout(row2)

        # Substrate Width parameter
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("Substrate Width:"))
        self.sub_width_sb = QtWidgets.QDoubleSpinBox()
        self.sub_width_sb.setRange(0.1, 500.0)
        self.sub_width_sb.setValue(gv.dwell_substrate_tolerance)
        self.sub_width_sb.setSuffix(" nm")
        self.sub_width_sb.valueChanged.connect(lambda v: setattr(gv, 'dwell_substrate_tolerance', v))
        param_row.addWidget(self.sub_width_sb)

        self.sub_linewidth_sb = QtWidgets.QSpinBox()
        self.sub_linewidth_sb.setRange(1, 20)
        self.sub_linewidth_sb.setValue(int(getattr(gv, 'dwell_substrate_linewidth_px', 2)))
        self.sub_linewidth_sb.valueChanged.connect(lambda v: setattr(gv, 'dwell_substrate_linewidth_px', int(v)))
        param_row.addWidget(QtWidgets.QLabel("Substrate Linewidth (px):"))
        param_row.addWidget(self.sub_linewidth_sb)

        self.show_overlay_cb = QtWidgets.QCheckBox("Show Substrates && Markers")
        self.show_overlay_cb.setChecked(getattr(gv, 'dwell_show_overlay', True))
        self.show_overlay_cb.toggled.connect(self._toggle_overlay_visibility)
        param_row.addWidget(self.show_overlay_cb)

        param_row.addStretch()
        lay.addLayout(param_row)

        # Substrate list
        lay.addWidget(QtWidgets.QLabel("Substrates:"))
        self.substrate_list = QtWidgets.QListWidget()
        self.substrate_list.setMaximumHeight(120)
        self.substrate_list.currentRowChanged.connect(self._on_substrate_selected)
        self.substrate_list.itemDoubleClicked.connect(self._rename_substrate)
        lay.addWidget(self.substrate_list)

        # Status label
        self.sub_status_label = QtWidgets.QLabel("Ready")
        self.sub_status_label.setStyleSheet("color: gray; font-style: italic;")
        lay.addWidget(self.sub_status_label)
        lay.addStretch()
        return w

    # ── Tab 3: Proximity Analysis ──────────────────────────────────
    def _build_proximity_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("Proximity Radius R:"))
        self.prox_radius_sb = QtWidgets.QDoubleSpinBox()
        self.prox_radius_sb.setRange(0.1, 1000.0)
        self.prox_radius_sb.setValue(gv.dwell_proximity_radius)
        self.prox_radius_sb.setSuffix(" nm")
        self.prox_radius_sb.valueChanged.connect(lambda v: setattr(gv, 'dwell_proximity_radius', v))
        param_row.addWidget(self.prox_radius_sb)
        param_row.addStretch()
        lay.addLayout(param_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.run_prox_btn = QtWidgets.QPushButton("Run Proximity Analysis")
        self.run_prox_btn.clicked.connect(self._run_proximity_analysis)
        btn_row.addWidget(self.run_prox_btn)

        self.export_prox_btn = QtWidgets.QPushButton("Export CSV")
        self.export_prox_btn.clicked.connect(self._export_proximity_csv)
        btn_row.addWidget(self.export_prox_btn)
        lay.addLayout(btn_row)

        # Results table
        self.prox_table = QtWidgets.QTableWidget(0, 7)
        self.prox_table.setHorizontalHeaderLabels([
            "Mol ID", "Frame", "X (nm)", "Y (nm)",
            "Substrate ID", "Neighbor Count", "Neighbor IDs"
        ])
        self.prox_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.prox_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.prox_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        lay.addWidget(self.prox_table, 1)

        self.prox_status_label = QtWidgets.QLabel("Run Finalization first, then Run Proximity Analysis.")
        self.prox_status_label.setStyleSheet("color: gray; font-style: italic;")
        lay.addWidget(self.prox_status_label)
        return w

    # ── Tab 4: Histogram ─────────────────────────────────────────
    def _build_histogram_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("Bin Width:"))

        self.hist_bin_width_sb = QtWidgets.QDoubleSpinBox()
        self.hist_bin_width_sb.setRange(0.01, 60.0)
        self.hist_bin_width_sb.setDecimals(2)
        self.hist_bin_width_sb.setSingleStep(0.05)
        self.hist_bin_width_sb.setSuffix(" s")
        self.hist_bin_width_sb.setValue(float(getattr(gv, 'dwell_hist_bin_width_s', 0.45)))
        self.hist_bin_width_sb.valueChanged.connect(
            lambda v: setattr(gv, 'dwell_hist_bin_width_s', float(v))
        )
        param_row.addWidget(self.hist_bin_width_sb)
        param_row.addStretch()
        lay.addLayout(param_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.replot_hist_btn = QtWidgets.QPushButton("Replot Histogram")
        self.replot_hist_btn.clicked.connect(self._replot_histogram)
        btn_row.addWidget(self.replot_hist_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.hist_status_label = QtWidgets.QLabel("Adjust bin width and click Replot Histogram.")
        self.hist_status_label.setStyleSheet("color: gray; font-style: italic;")
        lay.addWidget(self.hist_status_label)
        lay.addStretch()
        return w

    # ─────────────────────────────────────────────
    #  Window lifecycle
    # ─────────────────────────────────────────────
    def restoreWindowSettings(self):
        from helperFunctions import restore_window_geometry
        restore_window_geometry(self, 'DwellAnalysisWindow', 300, 300, 320, 480)

    def _restore_window(self):
        self.restoreWindowSettings()

    def _restore_splitter_settings(self):
        try:
            window_settings = getattr(gv, 'windowSettings', {}) or {}
            splitter_sizes = window_settings.get('DwellAnalysisWindow_splitter_sizes', None)
            if splitter_sizes and hasattr(self, 'splitter'):
                self.splitter.setSizes([int(size) for size in splitter_sizes])
        except Exception as e:
            print(f"[WARNING] DwellAnalysisWindow restore splitter: {e}")

    def saveWindowSettings(self):
        try:
            geo = self.geometry()
            ws = getattr(gv, 'windowSettings', {})
            ws['DwellAnalysisWindow'] = {
                'width': geo.width(), 'height': geo.height(),
                'x': geo.x(), 'y': geo.y(),
                'visible': False, 'title': self.windowTitle(),
                'class_name': 'DwellAnalysisWindow'
            }
            if hasattr(self, 'splitter'):
                ws['DwellAnalysisWindow_splitter_sizes'] = [int(size) for size in self.splitter.sizes()]
            gv.windowSettings = ws
            if self.parent_win and hasattr(self.parent_win, 'saveAllInitialParams'):
                self.parent_win.saveAllInitialParams()
        except Exception as e:
            print(f"[WARNING] DwellAnalysisWindow save settings: {e}")

    def closeEvent(self, event):
        gv.dwell_analysis_active = False
        gv.dwell_substrate_draw_active = False
        gv.dwell_substrate_edit_mode = False
        if self.start_marking_btn.isChecked():
            self.start_marking_btn.setChecked(False)
        if self.new_sub_btn.isChecked():
            self.new_sub_btn.setChecked(False)
        if self.edit_sub_btn.isChecked():
            self.edit_sub_btn.setChecked(False)

        self.saveWindowSettings()

        # Clear plugin highlight
        try:
            if self.parent_win and hasattr(self.parent_win, 'setActionHighlight'):
                act = getattr(self.parent_win, 'plugin_actions', {}).get("Dwell Analysis")
                if act:
                    self.parent_win.setActionHighlight(act, False)
        except Exception:
            pass

        self._refresh_display()
        event.accept()

    def cleanup_on_unload(self):
        """Reset runtime analysis state before the plugin is unloaded."""
        runtime_defaults = {
            'dwell_marks': {},
            'dwell_temp_marks': [],
            'dwell_frame_status': {},
            'dwell_marker_records': {},
            'dwell_next_marker_id': 0,
            'dwell_restore_source_frame': -1,
            'dwell_restore_backup': [],
            'dwell_restored_from_previous': False,
            'dwell_analysis_active': False,
            'dwell_molecules': [],
            'dwell_last_image_index': -1,
            'dwell_substrates': [],
            'dwell_substrate_draw_active': False,
            'dwell_substrate_draw_points': [],
            'dwell_substrate_edit_mode': False,
            'dwell_substrate_selected': -1,
            'dwell_proximity_results': [],
            'dwell_id_label_items': [],
        }
        for key, value in runtime_defaults.items():
            setattr(gv, key, value)

        if self.parent_win is not None:
            try:
                if getattr(self.parent_win, 'dwell_analysis_window', None) is self:
                    setattr(self.parent_win, 'dwell_analysis_window', None)
            except Exception:
                pass

        if getattr(self, 'dwell_display_window', None) is not None:
            try:
                self.dwell_display_window.cleanup_on_unload()
            except Exception:
                pass
            self.dwell_display_window = None

        try:
            if hasattr(self, 'start_marking_btn'):
                self.start_marking_btn.setChecked(False)
            if hasattr(self, 'new_sub_btn'):
                self.new_sub_btn.setChecked(False)
            if hasattr(self, 'edit_sub_btn'):
                self.edit_sub_btn.setChecked(False)
        except Exception:
            pass

        try:
            self._refresh_substrate_list()
        except Exception:
            pass
        try:
            self._update_info()
        except Exception:
            pass

    # ─────────────────────────────────────────────
    #  Dwell Analysis Tab – methods
    # ─────────────────────────────────────────────
    def _frame_prev(self):
        """Navigate one frame backwards, syncing the main window."""
        new_idx = max(0, getattr(gv, 'index', 0) - 1)
        if self.parent_win and hasattr(self.parent_win, 'onCurrFrameValueChanged'):
            self.parent_win.onCurrFrameValueChanged(new_idx)

    def _frame_next(self):
        """Navigate one frame forwards, syncing the main window."""
        max_frame = max(0, getattr(gv, 'FrameNum', 1) - 1)
        new_idx   = min(max_frame, getattr(gv, 'index', 0) + 1)
        if self.parent_win and hasattr(self.parent_win, 'onCurrFrameValueChanged'):
            self.parent_win.onCurrFrameValueChanged(new_idx)

    def _update_frame_label(self, frame_index):
        """Keep the Frame: X / N label current."""
        if not hasattr(self, 'frame_label'):
            return
        total = max(0, getattr(gv, 'FrameNum', 1) - 1)
        self.frame_label.setText(f"Frame: {frame_index} / {total}")

    def _get_record_position_for_frame(self, record, frame_index):
        loc_map = record.get('location_by_frame', {})
        if frame_index in loc_map:
            x_px, y_px = loc_map[frame_index]
            return (float(x_px), float(y_px))
        previous_frames = [f for f in loc_map.keys() if f <= frame_index]
        if not previous_frames:
            return None
        nearest_frame = max(previous_frames)
        x_px, y_px = loc_map[nearest_frame]
        return (float(x_px), float(y_px))

    def _get_nm_per_display_pixel(self):
        """Return (nm/px_x, nm/px_y) in the same coordinate space used by Dwell marks (display image pixels)."""
        display_w = 0
        display_h = 0

        dspimg = getattr(gv, 'dspimg', None)
        if isinstance(dspimg, np.ndarray) and dspimg.ndim >= 2 and dspimg.size > 0:
            display_h, display_w = dspimg.shape[:2]
        elif hasattr(gv, 'dspsize') and gv.dspsize is not None:
            display_w, display_h = gv.dspsize
        else:
            display_w = getattr(gv, 'XPixel', 1)
            display_h = getattr(gv, 'YPixel', 1)

        x_scan_nm = float(getattr(gv, 'XScanSize', 0) or 0.0)
        y_scan_nm = float(getattr(gv, 'YScanSize', 0) or 0.0)
        dx_nm = x_scan_nm / max(1.0, float(display_w))
        dy_nm = y_scan_nm / max(1.0, float(display_h))
        return dx_nm, dy_nm

    def _get_active_records_with_positions(self, frame_index):
        active = []
        records = getattr(gv, 'dwell_marker_records', {})
        for rec_id, record in records.items():
            start_frame = int(record.get('start_frame', 0))
            stop_frame = int(record.get('stop_frame', -1))
            if frame_index < start_frame:
                continue
            if stop_frame >= 0 and frame_index > stop_frame:
                continue
            pos = self._get_record_position_for_frame(record, frame_index)
            if pos is None:
                continue
            active.append((int(rec_id), record, pos))
        return active

    def _build_frame_marks_from_records(self, frame_index):
        return [pos for _, _, pos in self._get_active_records_with_positions(frame_index)]

    def _create_marker_record(self, frame_index, px, py):
        rec_id = int(getattr(gv, 'dwell_next_marker_id', 0))
        records = getattr(gv, 'dwell_marker_records', {})
        records[rec_id] = {
            'molecule_id': rec_id,
            'substrate_id': -1,
            'start_frame': int(frame_index),
            'stop_frame': -1,
            'location_by_frame': {int(frame_index): (float(px), float(py))},
            'adjacent_ids_by_frame': {},
        }
        gv.dwell_marker_records = records
        gv.dwell_next_marker_id = rec_id + 1
        return rec_id

    def _infer_substrate_id_for_point(self, px, py):
        substrates = getattr(gv, 'dwell_substrates', [])
        if not substrates:
            return -1

        dx_nm, dy_nm = self._get_nm_per_display_pixel()
        px_nm, py_nm = px * dx_nm, py * dy_nm
        threshold = float(getattr(gv, 'dwell_substrate_tolerance', 10.0))

        best_sid = -1
        best_dist = float('inf')
        for s in substrates:
            pts_nm = [(sx * dx_nm, sy * dy_nm) for (sx, sy) in s.get('points', [])]
            if len(pts_nm) < 2:
                continue
            dist = _point_to_polyline_dist(px_nm, py_nm, pts_nm)
            if dist < best_dist:
                best_dist = dist
                best_sid = int(s.get('id', -1))

        if best_sid >= 0 and best_dist <= threshold:
            return best_sid
        return -1

    def _compute_adjacent_ids_for_records(self, rec_positions):
        adjacency = {rec_id: [] for rec_id in rec_positions.keys()}
        rec_ids = list(rec_positions.keys())
        if len(rec_ids) <= 1:
            return adjacency

        dx_nm, dy_nm = self._get_nm_per_display_pixel()
        radius_nm = float(getattr(gv, 'dwell_proximity_radius', 20.0))

        for i, rec_a in enumerate(rec_ids):
            ax, ay = rec_positions[rec_a]
            for rec_b in rec_ids[i + 1:]:
                bx, by = rec_positions[rec_b]
                dist_nm = float(np.hypot((ax - bx) * dx_nm, (ay - by) * dy_nm))
                if dist_nm <= radius_nm:
                    adjacency[rec_a].append(int(rec_b))
                    adjacency[rec_b].append(int(rec_a))
        return adjacency

    def _distance_nm_between_points(self, x1, y1, x2, y2):
        dx_nm, dy_nm = self._get_nm_per_display_pixel()
        if dx_nm > 0 and dy_nm > 0:
            return float(np.hypot((x1 - x2) * dx_nm, (y1 - y2) * dy_nm))
        return float(np.hypot(x1 - x2, y1 - y2))

    def _find_nearest_point_within_radius_nm(self, points, px, py, radius_nm):
        nearest_idx = -1
        nearest_dist = float('inf')
        for idx, (mx, my) in enumerate(points):
            dist_nm = self._distance_nm_between_points(mx, my, px, py)
            if dist_nm <= radius_nm and dist_nm < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist_nm
        return nearest_idx

    def _find_nearest_record_for_click(self, frame_index, px, py, radius_nm):
        nearest_rec_id = -1
        nearest_pos = None
        nearest_dist = float('inf')
        for rec_id, _, rec_pos in self._get_active_records_with_positions(frame_index):
            dist_nm = self._distance_nm_between_points(rec_pos[0], rec_pos[1], px, py)
            if dist_nm <= radius_nm and dist_nm < nearest_dist:
                nearest_dist = dist_nm
                nearest_rec_id = int(rec_id)
                nearest_pos = (float(rec_pos[0]), float(rec_pos[1]))
        return nearest_rec_id, nearest_pos

    def marker_begin_drag(self, px, py, frame_index):
        if not getattr(gv, 'dwell_marker_records', {}):
            self._migrate_records_from_dwell_marks()

        radius_nm = float(getattr(gv, 'dwell_mol_radius', 5.0))
        rec_id, _ = self._find_nearest_record_for_click(int(frame_index), float(px), float(py), radius_nm)
        return rec_id

    def marker_drag_to(self, rec_id, px, py, frame_index):
        records = getattr(gv, 'dwell_marker_records', {})
        if rec_id not in records:
            return False

        record = records[rec_id]
        current_frame = int(frame_index)
        old_pos = self._get_record_position_for_frame(record, current_frame)
        if old_pos is None:
            old_pos = (float(px), float(py))

        record['location_by_frame'][current_frame] = (float(px), float(py))

        radius_nm = float(getattr(gv, 'dwell_mol_radius', 5.0))
        temp_points = list(getattr(gv, 'dwell_temp_marks', []))
        hit_idx = self._find_nearest_point_within_radius_nm(temp_points, old_pos[0], old_pos[1], max(0.1, radius_nm))
        if hit_idx >= 0:
            gv.dwell_temp_marks[hit_idx] = (float(px), float(py))
        else:
            gv.dwell_temp_marks.append((float(px), float(py)))

        self._update_info()
        self._refresh_display()
        return True

    def marker_end_drag(self, rec_id, px, py, frame_index):
        return self.marker_drag_to(rec_id, px, py, frame_index)

    def _migrate_records_from_dwell_marks(self):
        records = {}
        next_id = 0
        sorted_frames = sorted(getattr(gv, 'dwell_marks', {}).keys())
        for frame_index in sorted_frames:
            marks = [tuple(p) for p in getattr(gv, 'dwell_marks', {}).get(frame_index, [])]
            matched_ids = set()
            for px, py in marks:
                chosen_id = None
                best_dist = float('inf')
                for rec_id, record in records.items():
                    if rec_id in matched_ids:
                        continue
                    stop_frame = int(record.get('stop_frame', -1))
                    if stop_frame >= 0 and stop_frame < frame_index - 1:
                        continue
                    prev_pos = self._get_record_position_for_frame(record, frame_index)
                    if prev_pos is None:
                        continue
                    dist = float(np.hypot(prev_pos[0] - px, prev_pos[1] - py))
                    if dist < 6.0 and dist < best_dist:
                        chosen_id = rec_id
                        best_dist = dist

                if chosen_id is None:
                    chosen_id = next_id
                    next_id += 1
                    records[chosen_id] = {
                        'molecule_id': chosen_id,
                        'substrate_id': -1,
                        'start_frame': int(frame_index),
                        'stop_frame': -1,
                        'location_by_frame': {},
                        'adjacent_ids_by_frame': {},
                    }

                records[chosen_id]['location_by_frame'][int(frame_index)] = (float(px), float(py))
                records[chosen_id]['stop_frame'] = -1
                matched_ids.add(chosen_id)

            for rec_id, record in records.items():
                if rec_id in matched_ids:
                    continue
                if int(record.get('stop_frame', -1)) == -1 and int(record.get('start_frame', frame_index)) <= frame_index:
                    record['stop_frame'] = int(frame_index - 1)

        gv.dwell_marker_records = records
        gv.dwell_next_marker_id = next_id

    def _toggle_marking(self, checked):
        gv.dwell_analysis_active = checked
        if checked:
            self.start_marking_btn.setText("Stop Marking")
            if not hasattr(gv, 'aryData') or gv.aryData is None:
                QtWidgets.QMessageBox.warning(self, "No Data", "Please load image data first.")
                self.start_marking_btn.setChecked(False)
                gv.dwell_analysis_active = False
                self.start_marking_btn.setText("Start Marking")
                return
        else:
            self.start_marking_btn.setText("Start Marking")
        self._refresh_display()

    def _on_frame_changed(self, frame_index):
        if not getattr(gv, 'dwell_marker_records', {}):
            self._migrate_records_from_dwell_marks()

        frame_marks = self._build_frame_marks_from_records(frame_index)
        gv.dwell_temp_marks = list(frame_marks)

        previous_marks = self._build_frame_marks_from_records(frame_index - 1) if frame_index > 0 else []
        if frame_index > 0 and previous_marks:
            gv.dwell_restore_source_frame = frame_index - 1
            gv.dwell_restore_backup = list(previous_marks)
        else:
            gv.dwell_restore_source_frame = -1
            gv.dwell_restore_backup = []
        gv.dwell_restored_from_previous = False
        gv.dwell_last_image_index = frame_index
        self._update_frame_label(frame_index)
        self._update_info()
        self._update_restore_button_state()
        self._refresh_display()

    def _restore_from_previous_memorized(self):
        current_frame = getattr(gv, 'index', 0)
        if current_frame <= 0:
            QtWidgets.QMessageBox.information(
                self, "Restore", "No previous-frame memorized marks to restore.")
            self._update_restore_button_state()
            return

        backup = list(self._build_frame_marks_from_records(current_frame - 1))
        if not backup:
            QtWidgets.QMessageBox.information(
                self, "Restore", "Frame N-1 has no memorized marks. Nothing to replot.")
            self._update_restore_button_state()
            return

        gv.dwell_temp_marks = backup
        gv.dwell_restore_source_frame = current_frame - 1
        gv.dwell_restore_backup = list(backup)
        gv.dwell_restored_from_previous = True
        self._update_info()
        self._update_restore_button_state()
        self._refresh_display()

    def _memorize(self):
        current_frame = getattr(gv, 'index', 0)
        if not getattr(gv, 'dwell_marker_records', {}):
            self._migrate_records_from_dwell_marks()

        active = self._get_active_records_with_positions(current_frame)
        active_candidates_map = {
            int(rec_id): (float(pos[0]), float(pos[1]))
            for rec_id, _, pos in active
        }

        if bool(getattr(gv, 'dwell_restored_from_previous', False)):
            restore_source = int(getattr(gv, 'dwell_restore_source_frame', -1))
            if restore_source >= 0:
                for rec_id, _, pos in self._get_active_records_with_positions(restore_source):
                    active_candidates_map.setdefault(
                        int(rec_id), (float(pos[0]), float(pos[1]))
                    )

        active_candidates = list(active_candidates_map.items())
        matched_record_ids = set()
        used_indices = set()
        temp_marks = list(getattr(gv, 'dwell_temp_marks', []))
        rec_positions_now = {}

        for px, py in temp_marks:
            best_idx = -1
            best_dist = float('inf')
            for idx, (rec_id, pos) in enumerate(active_candidates):
                if idx in used_indices:
                    continue
                dist = float(np.hypot(pos[0] - px, pos[1] - py))
                if dist < 6.0 and dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0:
                rec_id, _ = active_candidates[best_idx]
                used_indices.add(best_idx)
                matched_record_ids.add(rec_id)
                record = gv.dwell_marker_records[rec_id]
                record['location_by_frame'][int(current_frame)] = (float(px), float(py))
                record['stop_frame'] = -1
                record['substrate_id'] = int(self._infer_substrate_id_for_point(px, py))
                adj_map = record.setdefault('adjacent_ids_by_frame', {})
                adj_map[int(current_frame)] = list(adj_map.get(int(current_frame), []))
                rec_positions_now[int(rec_id)] = (float(px), float(py))
            else:
                rec_id = self._create_marker_record(current_frame, px, py)
                gv.dwell_marker_records[rec_id]['substrate_id'] = int(self._infer_substrate_id_for_point(px, py))
                matched_record_ids.add(rec_id)
                rec_positions_now[int(rec_id)] = (float(px), float(py))

        for rec_id, _, _ in active:
            if rec_id in matched_record_ids:
                continue
            record = gv.dwell_marker_records.get(rec_id)
            if not record:
                continue
            stop_frame = int(record.get('stop_frame', -1))
            if stop_frame == -1 or stop_frame >= current_frame:
                record['stop_frame'] = int(current_frame - 1)

        adjacency_now = self._compute_adjacent_ids_for_records(rec_positions_now)
        for rec_id, adjacent_ids in adjacency_now.items():
            record = gv.dwell_marker_records.get(rec_id)
            if not record:
                continue
            adj_map = record.setdefault('adjacent_ids_by_frame', {})
            adj_map[int(current_frame)] = [int(x) for x in adjacent_ids]

        gv.dwell_marks[current_frame] = list(gv.dwell_temp_marks)
        frame_status = getattr(gv, 'dwell_frame_status', {})
        frame_status[current_frame] = {
            'visited': True,
            'state': 'memorized',
            'mark_count': len(gv.dwell_temp_marks),
        }
        gv.dwell_frame_status = frame_status
        gv.dwell_restored_from_previous = False
        self._update_info()
        self._refresh_display()

    def _all_reset(self):
        reply = QtWidgets.QMessageBox.question(
            self, 'Reset', "Clear all marks and results?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            gv.dwell_marks = {}
            gv.dwell_temp_marks = []
            gv.dwell_frame_status = {}
            gv.dwell_marker_records = {}
            gv.dwell_next_marker_id = 0
            gv.dwell_restore_source_frame = -1
            gv.dwell_restore_backup = []
            gv.dwell_restored_from_previous = False
            gv.dwell_molecules = []
            gv.dwell_proximity_results = []
            self._update_info()
            self._update_restore_button_state()
            self._refresh_display()

    def _finalize(self):
        if not gv.dwell_marks:
            QtWidgets.QMessageBox.warning(self, "No Data", "No marks to process.")
            return

        sorted_frames = sorted(gv.dwell_marks.keys())
        radius       = gv.dwell_mol_radius
        dx_nm, dy_nm = self._get_nm_per_display_pixel()

        molecules = []
        for frame in sorted_frames:
            current_marks = gv.dwell_marks[frame]
            if not molecules:
                for m in current_marks:
                    molecules.append({'frames': [frame], 'positions': [m]})
                continue

            matched = [False] * len(current_marks)
            for mol in molecules:
                if mol['frames'][-1] == frame - 1:
                    lx_px, ly_px = mol['positions'][-1]
                    lx_nm, ly_nm = lx_px * dx_nm, ly_px * dy_nm
                    for i, (cx_px, cy_px) in enumerate(current_marks):
                        if not matched[i]:
                            d = np.sqrt((cx_px * dx_nm - lx_nm)**2 + (cy_px * dy_nm - ly_nm)**2)
                            if d <= radius:
                                mol['frames'].append(frame)
                                mol['positions'].append((cx_px, cy_px))
                                matched[i] = True
                                break
            for i, ok in enumerate(matched):
                if not ok:
                    molecules.append({'frames': [frame], 'positions': [current_marks[i]]})

        # Enrich with nm coordinates only (substrate/neighbor filled on Run Proximity)
        for mol in molecules:
            mol['positions_nm'] = [
                (px * dx_nm, py * dy_nm) for (px, py) in mol['positions']
            ]
            mol['substrate_id']   = -1
            mol['neighbor_count'] = [0] * len(mol['frames'])
            mol['neighbor_ids']   = [[] for _ in mol['frames']]

        gv.dwell_molecules = molecules
        self._update_info()

        # Histogram
        frame_time = getattr(gv, 'FrameTime', 100) or 100
        dwell_times = [len(m['frames']) * frame_time / 1000.0 for m in molecules]
        self._show_histogram(dwell_times)
        QtWidgets.QMessageBox.information(
            self, "Finalization", f"Processed {len(molecules)} molecules.")

    def _show_histogram(self, data):
        if not data:
            return
        import matplotlib.pyplot as plt
        bin_width_s = float(getattr(gv, 'dwell_hist_bin_width_s', 0.45))
        if bin_width_s <= 0:
            bin_width_s = 0.45
        values = np.asarray(data, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        min_v = float(np.min(values))
        max_v = float(np.max(values))
        if max_v <= min_v:
            bins = [min_v - bin_width_s / 2.0, min_v + bin_width_s / 2.0]
        else:
            start = np.floor(min_v / bin_width_s) * bin_width_s
            stop = np.ceil(max_v / bin_width_s) * bin_width_s + bin_width_s
            bins = np.arange(start, stop + 1e-12, bin_width_s)

        plt.figure("Dwell Time Histogram")
        plt.hist(values, bins=bins, color='skyblue', edgecolor='black')
        plt.title("Dwell Time Distribution")
        plt.xlabel("Dwell Time (s)")
        plt.ylabel("Counts")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _replot_histogram(self):
        mols = getattr(gv, 'dwell_molecules', [])
        if not mols:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please run Finalization first.")
            return

        frame_time = getattr(gv, 'FrameTime', 100) or 100
        dwell_times = [len(m.get('frames', [])) * frame_time / 1000.0 for m in mols]
        self._show_histogram(dwell_times)
        if hasattr(self, 'hist_status_label'):
            bw = float(getattr(gv, 'dwell_hist_bin_width_s', 0.45))
            self.hist_status_label.setText(f"Replotted with bin width: {bw:.2f} s")

    def _export_binary_csv(self):
        if not getattr(gv, 'dwell_molecules', []):
            QtWidgets.QMessageBox.warning(self, "No Data", "Please run Finalization first.")
            return
        num_frames = (gv.aryData.shape[0] if hasattr(gv, 'aryData') and gv.aryData is not None
                      else (max(gv.dwell_marks.keys()) + 1 if gv.dwell_marks else 1))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Binary CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import csv
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                ft = getattr(gv, 'FrameTime', 0)
                w.writerow(["Total particle:", len(gv.dwell_molecules)] + [""] * (num_frames - 2))
                w.writerow(["Frame rate:", ft / 1000.0, "Unit:", "second"] + [""] * (num_frames - 4))
                for mol in gv.dwell_molecules:
                    row = [0] * num_frames
                    for fi in mol['frames']:
                        if 0 <= fi < num_frames:
                            row[fi] = 1
                    w.writerow(row)
            QtWidgets.QMessageBox.information(self, "Export", f"Saved to {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    # ─────────────────────────────────────────────
    #  Session Save / Load
    # ─────────────────────────────────────────────
    def _save_session(self):
        """Save complete dwell session to a JSON file."""
        import json, datetime
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Dwell Session", "", "Dwell Session (*.dwell.json);;All Files (*)")
        if not path:
            return
        try:
            marker_records_payload = []
            for rec_id, rec in getattr(gv, 'dwell_marker_records', {}).items():
                loc_map = rec.get('location_by_frame', {})
                resolved_sid = int(rec.get('substrate_id', -1))
                if isinstance(loc_map, dict) and loc_map:
                    latest_frame = max(int(f) for f in loc_map.keys())
                    latest_pos = loc_map.get(latest_frame)
                    if latest_pos is None:
                        latest_pos = loc_map.get(str(latest_frame))
                    if isinstance(latest_pos, (list, tuple)) and len(latest_pos) >= 2:
                        resolved_sid = int(self._infer_substrate_id_for_point(float(latest_pos[0]), float(latest_pos[1])))

                marker_records_payload.append({
                    "molecule_id": int(rec.get('molecule_id', rec_id)),
                    "substrate_belonging": resolved_sid,
                    "start_frame": int(rec.get('start_frame', 0)),
                    "stop_frame": int(rec.get('stop_frame', -1)),
                    "marker_location": {
                        str(f): [float(p[0]), float(p[1])]
                        for f, p in rec.get('location_by_frame', {}).items()
                    },
                    "current_adjacent_molecule_ids": {
                        str(f): [int(x) for x in ids]
                        for f, ids in rec.get('adjacent_ids_by_frame', {}).items()
                    },
                })

            payload = {
                "version": 2,
                "metadata": {
                    "saved_at": datetime.datetime.now().isoformat(timespec='seconds'),
                    "XScanSize":  getattr(gv, 'XScanSize', 0),
                    "YScanSize":  getattr(gv, 'YScanSize', 0),
                    "XPixel":     getattr(gv, 'XPixel', 1),
                    "YPixel":     getattr(gv, 'YPixel', 1),
                    "FrameTime":  getattr(gv, 'FrameTime', 0),
                    "FrameNum":   getattr(gv, 'FrameNum', 0),
                },
                "parameters": {
                    "mol_radius_nm":          getattr(gv, 'dwell_mol_radius', 5.0),
                    "proximity_radius_nm":    getattr(gv, 'dwell_proximity_radius', 20.0),
                    "substrate_tolerance_nm": getattr(gv, 'dwell_substrate_tolerance', 10.0),
                },
                # dwell_marks: keys must be strings for JSON
                "dwell_marks": {
                    str(k): [list(p) for p in v]
                    for k, v in getattr(gv, 'dwell_marks', {}).items()
                },
                "frame_status": {
                    str(k): {
                        "visited": bool(v.get('visited', False)),
                        "state": v.get('state', 'memorized'),
                        "mark_count": int(v.get('mark_count', 0)),
                    }
                    for k, v in getattr(gv, 'dwell_frame_status', {}).items()
                },
                "marker_records": marker_records_payload,
                "substrates": [
                    {
                        "id":     s['id'],
                        "label":  s['label'],
                        "color_id": s.get('color_id', s['id']),
                        "points": [list(p) for p in s['points']],
                    }
                    for s in getattr(gv, 'dwell_substrates', [])
                ],
                "molecules": [
                    {
                        "frames":         m['frames'],
                        "positions":      [list(p) for p in m['positions']],
                        "substrate_id":   m.get('substrate_id', -1),
                        "neighbor_count": m.get('neighbor_count', []),
                        "neighbor_ids":   m.get('neighbor_ids', []),
                    }
                    for m in getattr(gv, 'dwell_molecules', [])
                ],
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            QtWidgets.QMessageBox.information(
                self, "Save Session",
                f"Session saved:\n{os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def _load_session(self):
        """Load a previously saved dwell session from a JSON file."""
        import json
        # Warn before overwriting current work
        if getattr(gv, 'dwell_marks', {}):
            reply = QtWidgets.QMessageBox.question(
                self, "Load Session",
                "Current marks will be overwritten. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Dwell Session", "", "Dwell Session (*.dwell.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get('version', 0) < 2:
                QtWidgets.QMessageBox.warning(
                    self, "Load Session",
                    "Unsupported session file version. Please use a file saved by this version of pyNuD.")
                return

            # Scan-size compatibility check
            meta = data.get('metadata', {})
            cur_xs = getattr(gv, 'XScanSize', None)
            cur_xp = getattr(gv, 'XPixel', None)
            if (cur_xs is not None and meta.get('XScanSize') and
                    abs(float(meta['XScanSize']) - float(cur_xs)) > 1e-3) or \
               (cur_xp is not None and meta.get('XPixel') and
                    int(meta['XPixel']) != int(cur_xp)):
                QtWidgets.QMessageBox.warning(
                    self, "Load Session",
                    "Warning: The session was saved with a different scan size.\n"
                    "Pixel positions may not align with the currently loaded image.")

            # ── Restore gv state ──────────────────────────────
            params = data.get('parameters', {})
            gv.dwell_mol_radius          = float(params.get('mol_radius_nm', 5.0))
            gv.dwell_proximity_radius    = float(params.get('proximity_radius_nm', 20.0))
            gv.dwell_substrate_tolerance = float(params.get('substrate_tolerance_nm', 10.0))
            gv.dwell_show_overlay        = True

            gv.dwell_marks = {
                int(k): [tuple(p) for p in v]
                for k, v in data.get('dwell_marks', {}).items()
            }

            loaded_records = data.get('marker_records', None)
            if isinstance(loaded_records, list) and loaded_records:
                parsed_records = {}
                max_id = -1
                for rec in loaded_records:
                    rec_id = int(rec.get('molecule_id', -1))
                    if rec_id < 0:
                        continue
                    loc_map_raw = rec.get('marker_location', {})
                    adj_raw = rec.get('current_adjacent_molecule_ids', {})
                    parsed_records[rec_id] = {
                        'molecule_id': rec_id,
                        'substrate_id': int(rec.get('substrate_belonging', rec.get('substrate_id', -1))),
                        'start_frame': int(rec.get('start_frame', 0)),
                        'stop_frame': int(rec.get('stop_frame', -1)),
                        'location_by_frame': {
                            int(f): (float(p[0]), float(p[1]))
                            for f, p in loc_map_raw.items()
                            if isinstance(p, (list, tuple)) and len(p) >= 2
                        },
                        'adjacent_ids_by_frame': {
                            int(f): [int(x) for x in ids]
                            for f, ids in adj_raw.items()
                            if isinstance(ids, list)
                        },
                    }
                    if rec_id > max_id:
                        max_id = rec_id
                gv.dwell_marker_records = parsed_records
                gv.dwell_next_marker_id = max_id + 1 if max_id >= 0 else 0
            else:
                self._migrate_records_from_dwell_marks()

            loaded_frame_status = data.get('frame_status', None)
            if isinstance(loaded_frame_status, dict):
                gv.dwell_frame_status = {
                    int(k): {
                        'visited': bool(v.get('visited', False)),
                        'state': v.get('state', 'memorized'),
                        'mark_count': int(v.get('mark_count', 0)),
                    }
                    for k, v in loaded_frame_status.items()
                }
            else:
                gv.dwell_frame_status = {
                    int(k): {
                        'visited': True,
                        'state': 'memorized',
                        'mark_count': len(v),
                    }
                    for k, v in gv.dwell_marks.items()
                }
            gv.dwell_substrates = [
                {
                    'id':     s['id'],
                    'label':  s['label'],
                    'color_id': s.get('color_id', s['id']),
                    'points': [tuple(p) for p in s['points']],
                }
                for s in data.get('substrates', [])
            ]

            def _restore_mol(m):
                dx_nm, dy_nm = self._get_nm_per_display_pixel()
                positions = [tuple(p) for p in m.get('positions', [])]
                return {
                    'frames':         m.get('frames', []),
                    'positions':      positions,
                    'positions_nm':   [(px * dx_nm, py * dy_nm) for (px, py) in positions],
                    'substrate_id':   m.get('substrate_id', -1),
                    'neighbor_count': m.get('neighbor_count', []),
                    'neighbor_ids':   [list(x) for x in m.get('neighbor_ids', [])],
                }
            gv.dwell_molecules = [_restore_mol(m) for m in data.get('molecules', [])]

            # Restore current-frame temp marks
            cur = getattr(gv, 'index', 0)

            # Sync SpinBox UI
            self.mol_radius_sb.setValue(gv.dwell_mol_radius)
            if hasattr(self, 'prox_radius_sb'):
                self.prox_radius_sb.setValue(gv.dwell_proximity_radius)
            if hasattr(self, 'sub_width_sb'):
                self.sub_width_sb.setValue(gv.dwell_substrate_tolerance)
            if hasattr(self, 'show_overlay_cb'):
                self.show_overlay_cb.setChecked(True)
            gv.dwell_show_id = False
            if hasattr(self, 'show_id_cb'):
                self.show_id_cb.setChecked(False)

            self._refresh_substrate_list()
            self._on_frame_changed(cur)

            QtWidgets.QMessageBox.information(
                self, "Load Session",
                f"Session loaded from:\n{os.path.basename(path)}\n"
                f"Molecules: {len(gv.dwell_molecules)}  "
                f"Substrates: {len(gv.dwell_substrates)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))

    # Public API called from pyNuD.py
    def add_mark(self, px, py):
        current_frame = int(getattr(gv, 'index', 0))
        radius_nm = float(getattr(gv, 'dwell_mol_radius', 5.0))

        # 1) K1 priority: any current-frame marker in temp state
        k1_points = list(getattr(gv, 'dwell_temp_marks', []))
        k1_idx = self._find_nearest_point_within_radius_nm(k1_points, px, py, radius_nm)
        if k1_idx >= 0:
            gv.dwell_temp_marks.pop(k1_idx)
            self._update_info()
            self._refresh_display()
            return

        # 2) K0 fallback: memorized marker from frame N or frame N+1
        memorized_points = []
        memorized_points.extend(list(getattr(gv, 'dwell_marks', {}).get(current_frame, [])))
        if current_frame > 0:
            memorized_points.extend(list(getattr(gv, 'dwell_marks', {}).get(current_frame - 1, [])))

        k0_idx = self._find_nearest_point_within_radius_nm(memorized_points, px, py, radius_nm)
        if k0_idx >= 0:
            snap_x, snap_y = memorized_points[k0_idx]
            gv.dwell_temp_marks.append((float(snap_x), float(snap_y)))
        else:
            gv.dwell_temp_marks.append((float(px), float(py)))

        self._update_info()
        self._refresh_display()

    def get_marks(self, frame_index):
        return gv.dwell_marks.get(frame_index, [])

    # ─────────────────────────────────────────────
    #  Substrate Tab – methods
    # ─────────────────────────────────────────────
    def _toggle_draw_substrate(self, checked):
        if checked:
            # Exit edit mode if active
            self.edit_sub_btn.setChecked(False)
            gv.dwell_substrate_edit_mode = False

            gv.dwell_substrate_draw_active = True
            gv.dwell_substrate_draw_points = []
            self.new_sub_btn.setText("Drawing…")
            self.finish_sub_btn.setEnabled(True)
            self.undo_sub_btn.setEnabled(True)
            self.sub_status_label.setText(
                "Left-click on the image to add nodes. Right-click or Finish to confirm.")
        else:
            gv.dwell_substrate_draw_active = False
            gv.dwell_substrate_draw_points = []
            self.new_sub_btn.setText("New Substrate")
            self.finish_sub_btn.setEnabled(False)
            self.undo_sub_btn.setEnabled(False)
            self.sub_status_label.setText("Ready")
        self._refresh_display()

    def _finish_substrate(self):
        pts = getattr(gv, 'dwell_substrate_draw_points', [])
        if len(pts) < 2:
            QtWidgets.QMessageBox.information(
                self, "Substrate", "Need at least 2 points to define a substrate.")
            return
        sid = len(gv.dwell_substrates)
        gv.dwell_substrates.append({
            'id': sid,
            'points': list(pts),
            'label': f"DNA-{sid}",
            'color_id': sid % max(1, len(SUBSTRATE_PALETTE_BGR)),
        })
        gv.dwell_substrate_draw_points = []
        gv.dwell_substrate_draw_active = False
        self.new_sub_btn.setChecked(False)
        self.new_sub_btn.setText("New Substrate")
        self.finish_sub_btn.setEnabled(False)
        self.undo_sub_btn.setEnabled(False)
        self.sub_status_label.setText(
            f"Substrate #{sid} confirmed ({len(pts)} nodes).")
        self._refresh_substrate_list()
        self._refresh_display()

    def _undo_last_point(self):
        pts = getattr(gv, 'dwell_substrate_draw_points', [])
        if pts:
            pts.pop()
            self.sub_status_label.setText(f"{len(pts)} node(s) placed.")
        self._refresh_display()

    def _toggle_edit_substrate(self, checked):
        if checked:
            self.new_sub_btn.setChecked(False)
            gv.dwell_substrate_draw_active = False
            gv.dwell_substrate_draw_points = []
            gv.dwell_substrate_edit_mode = True
            self.sub_status_label.setText(
                "Drag nodes to move. Right-click a node to delete.")
        else:
            gv.dwell_substrate_edit_mode = False
            self.sub_status_label.setText("Ready")
        self._refresh_display()

    def _delete_selected_substrate(self):
        idx = gv.dwell_substrate_selected
        if 0 <= idx < len(gv.dwell_substrates):
            del gv.dwell_substrates[idx]
            # Re-number ids
            for i, s in enumerate(gv.dwell_substrates):
                s['id'] = i
            gv.dwell_substrate_selected = -1
            self._refresh_substrate_list()
            self._refresh_display()

    def _clear_all_substrates(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Clear", "Remove all substrates?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            gv.dwell_substrates = []
            gv.dwell_substrate_draw_points = []
            gv.dwell_substrate_selected = -1
            self._refresh_substrate_list()
            self._refresh_display()

    def _shuffle_substrate_colors(self):
        substrates = getattr(gv, 'dwell_substrates', [])
        if not substrates:
            self.sub_status_label.setText("No substrate to shuffle.")
            return

        palette_len = len(SUBSTRATE_PALETTE_BGR)
        if palette_len <= 1:
            self.sub_status_label.setText("Palette has only one color.")
            return

        steps = [s for s in range(1, palette_len) if math.gcd(s, palette_len) == 1]
        step = random.choice(steps) if steps else 1
        offset = random.randrange(palette_len)

        color_cycle = [((offset + i * step) % palette_len) for i in range(len(substrates))]
        sub_order = list(range(len(substrates)))
        random.shuffle(sub_order)

        for rank, sub_idx in enumerate(sub_order):
            substrates[sub_idx]['color_id'] = color_cycle[rank]

        self.sub_status_label.setText("Substrate colors shuffled.")
        self._refresh_substrate_list()
        self._refresh_display()

    def _on_substrate_selected(self, row):
        gv.dwell_substrate_selected = row
        self._refresh_display()

    def _rename_substrate(self, item):
        row = self.substrate_list.row(item)
        if 0 <= row < len(gv.dwell_substrates):
            new_name, ok = QtWidgets.QInputDialog.getText(
                self, "Rename Substrate", "New name:",
                text=gv.dwell_substrates[row]['label'])
            if ok and new_name.strip():
                gv.dwell_substrates[row]['label'] = new_name.strip()
                self._refresh_substrate_list()

    def _refresh_substrate_list(self):
        self.substrate_list.clear()
        for s in gv.dwell_substrates:
            color = SUBSTRATE_PALETTE_RGB[get_substrate_color_index(s) % len(SUBSTRATE_PALETTE_RGB)]
            pix = QtGui.QPixmap(16, 16)
            pix.fill(QtGui.QColor(*color))
            icon = QtGui.QIcon(pix)
            item = QtWidgets.QListWidgetItem(icon, f"#{s['id']}  {s['label']}  ({len(s['points'])} pts)")
            self.substrate_list.addItem(item)

    # Substrate node editing – called from pyNuD.py image click handlers
    def substrate_add_draw_point(self, px, py):
        """Called when user left-clicks image in draw mode."""
        if not gv.dwell_substrate_draw_active:
            return
        gv.dwell_substrate_draw_points.append((px, py))
        n = len(gv.dwell_substrate_draw_points)
        self.sub_status_label.setText(f"{n} node(s) placed. Right-click to finish.")
        self._refresh_display()

    def substrate_finish_draw_on_rightclick(self):
        """Called on right-click in draw mode to finish current polyline."""
        if gv.dwell_substrate_draw_active:
            self._finish_substrate()

    def substrate_edit_drag_node(self, sub_idx, node_idx, new_px, new_py):
        """Called during drag in edit mode."""
        if 0 <= sub_idx < len(gv.dwell_substrates):
            pts = gv.dwell_substrates[sub_idx]['points']
            if 0 <= node_idx < len(pts):
                pts[node_idx] = (new_px, new_py)
                self._refresh_display()

    def substrate_edit_delete_node(self, sub_idx, node_idx):
        """Right-click a node during edit mode."""
        if 0 <= sub_idx < len(gv.dwell_substrates):
            pts = gv.dwell_substrates[sub_idx]['points']
            if 0 <= node_idx < len(pts):
                pts.pop(node_idx)
                if len(pts) == 0:
                    del gv.dwell_substrates[sub_idx]
                    for i, s in enumerate(gv.dwell_substrates):
                        s['id'] = i
                    self._refresh_substrate_list()
                self._refresh_display()

    # ─────────────────────────────────────────────
    #  Proximity Analysis Tab – methods
    # ─────────────────────────────────────────────
    def _run_proximity_analysis(self):
        mols = getattr(gv, 'dwell_molecules', [])
        if not mols:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Please run Finalization first.")
            return

        dx_nm, dy_nm = self._get_nm_per_display_pixel()
        R           = gv.dwell_proximity_radius
        W           = gv.dwell_substrate_tolerance
        substrates  = getattr(gv, 'dwell_substrates', [])

        # ── Step 1: Assign substrate_id ─────────────────────────
        for mol in mols:
            # Use mean position for substrate assignment
            xs = [p[0] * dx_nm for p in mol['positions']]
            ys = [p[1] * dy_nm for p in mol['positions']]
            mx, my = np.mean(xs), np.mean(ys)

            best_sid  = -1
            best_dist = float('inf')
            for s in substrates:
                # Convert substrate points to nm
                pts_nm = [(px * dx_nm, py * dy_nm) for (px, py) in s['points']]
                d = _point_to_polyline_dist(mx, my, pts_nm)
                if d < best_dist:
                    best_dist = d
                    best_sid  = s['id']

            if best_sid >= 0 and best_dist <= W:
                mol['substrate_id'] = best_sid
            else:
                mol['substrate_id'] = -1  # no assigned substrate

        # ── Step 2: Compute per-frame neighbor_count / neighbor_ids ─
        # Build lookup: mol_idx -> {frame -> position_nm}
        frame_pos = []
        for mol in mols:
            fp = {}
            for f, (px, py) in zip(mol['frames'], mol['positions']):
                fp[f] = (px * dx_nm, py * dy_nm)
            frame_pos.append(fp)

        for idx_a, mol_a in enumerate(mols):
            nc = []
            ni = []
            for f, (px_a, py_a) in zip(mol_a['frames'], mol_a['positions']):
                xa_nm = px_a * dx_nm
                ya_nm = py_a * dy_nm
                nbrs = []
                for idx_b, mol_b in enumerate(mols):
                    if idx_b == idx_a:
                        continue
                    # Same substrate (or both unassigned → skip)
                    if mol_a['substrate_id'] != mol_b['substrate_id']:
                        continue
                    if mol_a['substrate_id'] == -1:
                        continue
                    if f not in frame_pos[idx_b]:
                        continue
                    xb_nm, yb_nm = frame_pos[idx_b][f]
                    dist = np.hypot(xa_nm - xb_nm, ya_nm - yb_nm)
                    if dist <= R:
                        nbrs.append(idx_b)
                nc.append(len(nbrs))
                ni.append(nbrs)
            mol_a['neighbor_count'] = nc
            mol_a['neighbor_ids']   = ni

        gv.dwell_proximity_results = mols  # same list, now enriched

        # ── Step 3: Populate table ──────────────────────────────
        rows = []
        for mol_idx, mol in enumerate(mols):
            for fi, (frame, (px, py)) in enumerate(zip(mol['frames'], mol['positions'])):
                rows.append({
                    'mol_id':    mol_idx,
                    'frame':     frame,
                    'x_nm':      px * dx_nm,
                    'y_nm':      py * dy_nm,
                    'sub_id':    mol.get('substrate_id', -1),
                    'n_count':   mol['neighbor_count'][fi],
                    'n_ids':     ",".join(str(x) for x in mol['neighbor_ids'][fi]),
                })

        self.prox_table.setRowCount(0)
        self.prox_table.setRowCount(len(rows))
        for r, d in enumerate(rows):
            vals = [
                str(d['mol_id']), str(d['frame']),
                f"{d['x_nm']:.2f}", f"{d['y_nm']:.2f}",
                str(d['sub_id']), str(d['n_count']), d['n_ids']
            ]
            for c, v in enumerate(vals):
                self.prox_table.setItem(r, c, QtWidgets.QTableWidgetItem(v))

        n_events = sum(sum(m['neighbor_count']) for m in mols)
        self.prox_status_label.setText(
            f"Done. {len(mols)} molecules, {len(rows)} frame-events, "
            f"{n_events} co-proximity events.")
        self._refresh_display()

    def _export_proximity_csv(self):
        mols = getattr(gv, 'dwell_proximity_results', [])
        if not mols:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Run Proximity Analysis first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Proximity CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import csv
            dx_nm, dy_nm = self._get_nm_per_display_pixel()
            frame_time = getattr(gv, 'FrameTime', 0) or 0  # ms
            substrates = getattr(gv, 'dwell_substrates', [])
            sub_labels = {s['id']: s['label'] for s in substrates}

            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    "mol_id", "frame", "time_s",
                    "x_nm", "y_nm",
                    "substrate_id", "substrate_label",
                    "dwell_time_s",
                    "neighbor_count", "neighbor_ids"
                ])
                for mol_idx, mol in enumerate(mols):
                    dwell_s = len(mol['frames']) * frame_time / 1000.0
                    sub_id  = mol.get('substrate_id', -1)
                    sub_lbl = sub_labels.get(sub_id, "") if sub_id >= 0 else ""
                    for fi, (frame, (px, py)) in enumerate(
                            zip(mol['frames'], mol['positions'])):
                        time_s = frame * frame_time / 1000.0
                        w.writerow([
                            mol_idx,
                            frame,
                            f"{time_s:.4f}",
                            f"{px * dx_nm:.4f}",
                            f"{py * dy_nm:.4f}",
                            sub_id,
                            sub_lbl,
                            f"{dwell_s:.4f}",
                            mol['neighbor_count'][fi],
                            ";".join(str(x) for x in mol['neighbor_ids'][fi])
                        ])
            QtWidgets.QMessageBox.information(
                self, "Export", f"Saved to {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    # ─────────────────────────────────────────────
    #  Shared helpers
    # ─────────────────────────────────────────────
    def _update_info(self):
        self.mark_num_label.setText(str(len(gv.dwell_temp_marks)))
        self.total_num_label.setText(str(len(getattr(gv, 'dwell_molecules', []))))

    def _focus_display_panel(self):
        try:
            if hasattr(self, 'tabs'):
                self.tabs.setCurrentIndex(0)
            if getattr(self, 'dwell_display_window', None) is not None:
                self.dwell_display_window.refresh_display()
                self.dwell_display_window.setFocus(QtCore.Qt.OtherFocusReason)
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

    def _refresh_display(self):
        if self.parent_win and hasattr(self.parent_win, 'UpdateDisplayImage'):
            self.parent_win.UpdateDisplayImage()
            self.parent_win.showDisplayImage()
        if getattr(self, 'dwell_display_window', None) is not None:
            self.dwell_display_window.refresh_display()

    def _toggle_overlay_visibility(self, checked):
        gv.dwell_show_overlay = checked
        self._refresh_display()

    def _toggle_show_id(self, checked):
        gv.dwell_show_id = bool(checked)
        self._refresh_display()

    def _update_restore_button_state(self):
        if not hasattr(self, 'restore_btn'):
            return
        current_frame = getattr(gv, 'index', 0)
        can_restore = current_frame > 0
        self.restore_btn.setEnabled(can_restore)

    def _show_help(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setMinimumSize(520, 540)
        dialog.resize(620, 660)
        ll = QtWidgets.QVBoxLayout(dialog)
        lang_row = QtWidgets.QHBoxLayout()
        lang_row.addWidget(QtWidgets.QLabel("Language / 言語:"))
        btn_ja = QtWidgets.QPushButton("日本語", dialog)
        btn_en = QtWidgets.QPushButton("English", dialog)
        btn_ja.setCheckable(True); btn_en.setCheckable(True)
        grp = QtWidgets.QButtonGroup(dialog)
        grp.addButton(btn_ja); grp.addButton(btn_en)
        grp.setExclusive(True)
        sel = "QPushButton { background-color: #007aff; color: white; font-weight: bold; }"
        nrm = "QPushButton { background-color: #e5e5e5; color: black; }"
        lang_row.addWidget(btn_ja); lang_row.addWidget(btn_en)
        lang_row.addStretch()
        ll.addLayout(lang_row)
        browser = QtWidgets.QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        css = ("body{font-size:14px;line-height:1.6}"
               "h1{font-size:20px;color:#2c3e50;border-bottom:2px solid #3498db}"
               "h2{font-size:16px;color:#2c3e50;margin-top:14px}"
               "ul{padding-left:20px}")
        browser.document().setDefaultStyleSheet(css)
        close_btn = QtWidgets.QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)

        def set_lang(ja):
            btn_ja.setChecked(ja); btn_en.setChecked(not ja)
            btn_ja.setStyleSheet(sel if ja else nrm)
            btn_en.setStyleSheet(sel if not ja else nrm)
            html = HELP_HTML_JA if ja else HELP_HTML_EN
            browser.setHtml(f"<html><body>{html.strip()}</body></html>")
            dialog.setWindowTitle("ドウェル解析 - マニュアル" if ja else "Dwell Analysis - Manual")
            close_btn.setText("閉じる" if ja else "Close")

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        ll.addWidget(browser); ll.addWidget(close_btn)
        set_lang(False)
        dialog.exec_()


# ══════════════════════════════════════════════════════════════════
#  Plugin entry point
# ══════════════════════════════════════════════════════════════════
def create_plugin(main_window):
    return DwellAnalysisWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "DwellAnalysisWindow"]
