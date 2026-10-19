"""
Interactive Drift Compensation Plugin for pyNuD.
インタラクティブ ドリフト補正プラグイン。

User manually aligns key frames by translating the tracked view,
then correlation tracking computes drift for all frames.
"""

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

from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
import csv
import scipy.ndimage
from skimage.registration import phase_cross_correlation

try:
    import globalvals as gv
except ImportError:
    gv = None

PLUGIN_NAME = "Interactive Drift Compensation"


def _source_identity():
    """Bind an applied snapshot to its source movie and pixel calibration."""
    try:
        return (os.path.abspath(gv.files[gv.currentFileNum]), int(gv.FrameNum),
                int(gv.XPixel), int(gv.YPixel), float(gv.XScanSize), float(gv.YScanSize))
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def corrected_movie(frames, shift_x, shift_y, *, expanded=False):
    """Translate by viewer correction offsets in array coordinates.

    Apply retains the original field; Save retains the union of shifted fields.
    Both use linear interpolation and the frame minimum for uncovered pixels.
    """
    movie = np.asarray(frames, dtype=float)
    sx, sy = np.asarray(shift_x, dtype=float), np.asarray(shift_y, dtype=float)
    if movie.ndim != 3 or not movie.size or sx.shape != (len(movie),) or sy.shape != sx.shape:
        raise ValueError("A correction offset is required for every movie frame.")
    if not np.isfinite(movie).all() or not np.isfinite(sx).all() or not np.isfinite(sy).all():
        raise ValueError("Movie data and correction offsets must be finite.")
    h, w = movie.shape[1:]
    x0 = min(0, int(np.floor(sx.min()))) if expanded else 0
    y0 = min(0, int(np.floor(sy.min()))) if expanded else 0
    x1 = max(w, int(np.ceil(sx.max() + w))) if expanded else w
    y1 = max(h, int(np.ceil(sy.max() + h))) if expanded else h
    result = np.empty((len(movie), y1 - y0, x1 - x0), dtype=float)
    for i, frame in enumerate(movie):
        # One resampling step makes Apply and Save identical within their overlap.
        result[i] = scipy.ndimage.affine_transform(
            frame, np.eye(2), offset=(y0 - sy[i], x0 - sx[i]),
            output_shape=result.shape[1:], order=1, mode='constant', cval=float(frame.min()))
    return result

# ---------------------------------------------------------------------------
# Help content (EN / JA)
# ---------------------------------------------------------------------------

HELP_HTML_EN = """
<h1>Interactive Drift Compensation</h1>
<p>Apply and Save preserve both captured channels and apply the same shifts to each. Save verifies the physical values after reading the output ASD. Keep the original source selected while applying or saving.</p>
<h2>Overview</h2>
<p>This plugin corrects sample drift in AFM image stacks through an
interactive, human-guided workflow.  Because AFM molecule morphology is
often too ambiguous for fully automatic template matching, the user
manually aligns a handful of <em>keyframes</em>; the plugin then
propagates the correction to every frame in between using phase
cross-correlation.</p>

<h2>Workflow (step by step)</h2>
<ol>
  <li><strong>Load data</strong> — open an ASD file in pyNuD as usual, then
      launch this plugin from the Plugin menu.  The current image stack is
      loaded automatically.</li>
  <li><strong>Navigate to a keyframe</strong> — use the Frame slider or the
      keyboard shortcuts (Ctrl+A / Ctrl+D) to move to a frame where drift
      is clearly visible.</li>
  <li><strong>Align the Tracked View</strong> — drag the <em>Tracked View</em>
      canvas (left mouse button) or use the arrow buttons (▲▼◀▶) to
      translate the image until it is visually aligned with the
      <em>Original View</em>.  The fixed grid assists alignment.</li>
  <li><strong>Record the offset</strong> — click <strong>Record Offset</strong>
      (or press Ctrl+R).  The keyframe is added to the table with its
      measured (dX, dY) in pixels and nanometres.</li>
  <li><strong>Repeat</strong> for additional frames that show significant drift.</li>
  <li><strong>Track</strong> — click <strong>Track</strong>.  The plugin runs
      segment-wise phase cross-correlation between consecutive keyframes and
      blends the results so that every keyframe endpoint is exactly honoured.
      The drift plot updates when tracking completes.</li>
  <li><strong>Apply</strong> — shows the corrected 1ch movie in the main view at
      the original image size. Analysis tools that read the main view use these frames.
      Turn off <strong>Use Applied Correction</strong> to return to the current processing
      settings. To apply further filters, save the corrected ASD and open it.</li>
  <li><strong>Save .asd</strong> — exports the corrected stack as an ASD file.
      The image canvas is automatically <em>expanded</em> to encompass the
      full drift range: pixel count and scan range are recalculated from the
      bounding box of all shifted frame positions.</li>
  <li><strong>Export CSV</strong> — saves the per-frame drift values
      (shift_x, shift_y in px and nm) together with keyframe flags.</li>
  <li><strong>Load CSV</strong> — restores a previously exported CSV to
      reconstruct the keyframe list without re-doing the manual alignment.</li>
</ol>

<h2>Look Back</h2>
<p>Hold the <strong>◁ Look Back</strong> button to temporarily show the
most recent keyframe's image at its recorded offset.  This lets you compare
the current alignment against the last confirmed position.  Release to
return to the current frame.</p>

<h2>Grid</h2>
<p>The reference grid is fixed to the <em>viewport</em> and does not move
with the image.  Use it as a stable ruler to judge translational drift between
frames.  Toggle visibility with the <strong>Show Grid</strong> checkbox.</p>

<h2>Keyboard Shortcuts</h2>
<table>
  <tr><td><strong>Ctrl+A</strong></td><td>Previous frame</td></tr>
  <tr><td><strong>Ctrl+D</strong></td><td>Next frame</td></tr>
  <tr><td><strong>Ctrl+R</strong></td><td>Record current offset as keyframe</td></tr>
</table>

<h2>Drift Plot</h2>
<p>After tracking, the bottom plot shows shift X (blue) and shift Y (orange)
curves across all frames.  Keyframe positions are marked with circles.  A
shaded uncertainty band reflects the tracking confidence (narrower near
keyframes, wider in extrapolated segments).</p>

<h2>Notes</h2>
<ul>
  <li>At least <strong>one keyframe</strong> is required to run tracking.
      Two or more keyframes are recommended for segment-wise interpolation.</li>
  <li>Phase cross-correlation uses <code>upsample_factor=10</code>, giving
      sub-pixel accuracy of ±0.1 px.</li>
  <li>The plugin does <em>not</em> modify the original ASD file;
      use <strong>Save .asd</strong> to persist the corrected result.</li>
</ul>
"""

HELP_HTML_JA = """
<p>ApplyとSaveは取得した両チャンネルに同じシフトを適用します。保存後のASDを読み直して数値を検証します。Apply・Save時は取得元ファイルを選択してください。</p>
<h1>インタラクティブ ドリフト補正</h1>
<h2>概要</h2>
<p>このプラグインは、人間が少数の<em>キーフレーム</em>を手動で位置合わせし、
その情報をもとにフェーズ相互相関でフレーム間の補正量を計算することで、
AFM 画像スタックのサンプルドリフトを補正します。</p>

<h2>操作手順</h2>
<ol>
  <li><strong>データの読み込み</strong> — pyNuD で ASD ファイルを開き、
      Plugin メニューからこのプラグインを起動します。
      現在の画像スタックは自動で読み込まれます。</li>
  <li><strong>キーフレームへ移動</strong> — Frame スライダーまたはキーボード
      ショートカット（Ctrl+A / Ctrl+D）でドリフトが目立つフレームに移動します。</li>
  <li><strong>Tracked View を整列</strong> — <em>Tracked View</em> キャンバスを
      左ドラッグ、または矢印ボタン（▲▼◀▶）で平行移動して
      <em>Original View</em> と目視で位置合わせします。固定グリッドが目安になります。</li>
  <li><strong>オフセットを記録</strong> — <strong>Record Offset</strong>（または Ctrl+R）
      をクリックします。ピクセル・nm 単位の (dX, dY) がテーブルに追加されます。</li>
  <li><strong>繰り返し</strong> — ドリフトが顕著なフレームに対して手順 2〜4 を繰り返します。</li>
  <li><strong>Track</strong> — <strong>Track</strong> ボタンをクリックします。
      連続するキーフレーム間でセグメントごとにフェーズ相互相関を実行し、
      各キーフレーム端点を厳密に通過するようにブレンドします。
      完了後にドリフトプロットが更新されます。</li>
  <li><strong>Apply</strong> — 補正済みの1ch動画を元の画像サイズでメイン画面へ反映します。
      メイン画面を読む解析機能はこの画像を使います。<strong>Use Applied Correction</strong>
      をオフにすると現在の処理設定に戻ります。補正後にフィルタを追加する場合は
      ASDを保存して開いてください。</li>
  <li><strong>Save .asd</strong> — 補正済みスタックを ASD ファイルとして保存します。
      全シフト量のバウンディングボックスからキャンバスを<em>自動拡張</em>し、
      ピクセル数とスキャン範囲を再計算します。</li>
  <li><strong>Export CSV</strong> — フレームごとのドリフト値
      （shift_x・shift_y のピクセル／nm）とキーフレームフラグを CSV 保存します。</li>
  <li><strong>Load CSV</strong> — 以前エクスポートした CSV を読み込み、
      手動位置合わせをやり直さずにキーフレームリストを復元します。</li>
</ol>

<h2>Look Back</h2>
<p><strong>◁ Look Back</strong> ボタンを押し続けると、直近のキーフレームの
画像をその記録オフセットで一時表示します。
現在の整列状態と最後に確定した位置を比較できます。
ボタンを離すと現在フレームに戻ります。</p>

<h2>グリッド</h2>
<p>参照グリッドは<em>ビューポート固定</em>で、画像移動に追従しません。
フレーム間の並進ドリフトを判断する安定した目盛りとして活用してください。
<strong>Show Grid</strong> チェックボックスで表示切り替えができます。</p>

<h2>キーボードショートカット</h2>
<table>
  <tr><td><strong>Ctrl+A</strong></td><td>前のフレームへ</td></tr>
  <tr><td><strong>Ctrl+D</strong></td><td>次のフレームへ</td></tr>
  <tr><td><strong>Ctrl+R</strong></td><td>現在のオフセットをキーフレームとして記録</td></tr>
</table>

<h2>ドリフトプロット</h2>
<p>Track 完了後、下部のプロットに全フレームの shift X（青）と shift Y（橙）が表示されます。
キーフレーム位置は丸印でマークされます。
シェード帯は追跡信頼度を示し、キーフレーム近傍で狭く、外挿区間で広くなります。</p>

<h2>注意事項</h2>
<ul>
  <li>Track を実行するには<strong>少なくとも 1 つ</strong>のキーフレームが必要です。
      セグメント補間には 2 つ以上を推奨します。</li>
  <li>フェーズ相互相関は <code>upsample_factor=10</code> を使用し、
      ±0.1 px のサブピクセル精度を実現します。</li>
  <li>このプラグインは元の ASD ファイルを変更しません。
      補正結果を保持するには <strong>Save .asd</strong> を使用してください。</li>
</ul>
"""

# ---------------------------------------------------------------------------
# Package 1: FixedGridImageViewer
# ---------------------------------------------------------------------------

class FixedGridImageViewer(QtWidgets.QWidget):
    """
    Image viewer with a viewport-fixed reference grid.
    Grid covers the entire widget area and does NOT move with the image.

    ビューポート固定グリッド付き画像ビューア。
    グリッドは画像の移動に追従しない。
    """
    offset_changed = QtCore.pyqtSignal(float, float)

    def __init__(self, interactive: bool = False, parent=None):
        super().__init__(parent)
        self.interactive = interactive
        self.setMinimumSize(250, 250)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setStyleSheet("background-color: black; border: 1px solid #555;")

        self._source_pixmap: QtGui.QPixmap | None = None
        self._offset_x: float = 0.0  # image translation in source-pixel units
        self._offset_y: float = 0.0
        self._drag_active: bool = False
        self._drag_start: QtCore.QPoint | None = None
        self._drag_offset_start: tuple[float, float] = (0.0, 0.0)

        # Cached geometry of the drawn image (widget coords)
        self._img_rect = QtCore.QRectF()
        self._scale_factor: float = 1.0  # widget-pixels per source-pixel

        # Grid appearance
        self._grid_cols = 16
        self._grid_rows = 9
        self._show_grid = True

    # --- public API ---

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        self._source_pixmap = pixmap
        self.update()

    def set_offset(self, dx: float, dy: float) -> None:
        """Set offset in **source-pixel** units."""
        if abs(dx - self._offset_x) > 1e-6 or abs(dy - self._offset_y) > 1e-6:
            self._offset_x = dx
            self._offset_y = dy
            self.update()
            self.offset_changed.emit(self._offset_x, self._offset_y)

    def get_offset(self) -> tuple[float, float]:
        return (self._offset_x, self._offset_y)

    def reset_offset(self) -> None:
        self.set_offset(0.0, 0.0)

    def set_grid_visible(self, visible: bool) -> None:
        self._show_grid = visible
        self.update()

    # --- painting ---

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # 1) Draw the image (80% of viewport, centred + offset)
        if self._source_pixmap and not self._source_pixmap.isNull():
            target_w = int(w * 0.80)
            target_h = int(h * 0.80)
            scaled = self._source_pixmap.scaled(
                target_w, target_h,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self._scale_factor = scaled.width() / max(1, self._source_pixmap.width())

            # Centre position + user offset (converted to widget pixels)
            cx = (w - scaled.width()) / 2.0 + self._offset_x * self._scale_factor
            cy = (h - scaled.height()) / 2.0 + self._offset_y * self._scale_factor
            self._img_rect = QtCore.QRectF(cx, cy, scaled.width(), scaled.height())
            painter.drawPixmap(QtCore.QPointF(cx, cy), scaled)
        else:
            self._img_rect = QtCore.QRectF()
            self._scale_factor = 1.0

        # 2) Draw viewport-fixed grid (covers full widget)
        if self._show_grid:
            pen = QtGui.QPen(QtGui.QColor(200, 200, 200, 120))
            pen.setWidthF(0.8)
            pen.setStyle(QtCore.Qt.DashLine)
            painter.setPen(pen)

            for i in range(1, self._grid_cols):
                px = i * (w / self._grid_cols)
                painter.drawLine(QtCore.QPointF(px, 0), QtCore.QPointF(px, h))

            for j in range(1, self._grid_rows):
                py = j * (h / self._grid_rows)
                painter.drawLine(QtCore.QPointF(0, py), QtCore.QPointF(w, py))

        # 3) Border highlight for interactive viewer
        if self.interactive:
            pen = QtGui.QPen(QtGui.QColor(100, 180, 255, 200))
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.drawRect(1, 1, w - 2, h - 2)

        painter.end()

    # --- mouse interaction (Package 2) ---

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.interactive and event.button() == QtCore.Qt.LeftButton:
            self._drag_active = True
            self._drag_start = event.pos()
            self._drag_offset_start = (self._offset_x, self._offset_y)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_active and self._drag_start is not None:
            delta = event.pos() - self._drag_start
            # Convert widget-pixel delta to source-pixel delta
            sf = self._scale_factor if self._scale_factor > 0 else 1.0
            new_dx = self._drag_offset_start[0] + delta.x() / sf
            new_dy = self._drag_offset_start[1] + delta.y() / sf
            self.set_offset(new_dx, new_dy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_active:
            self._drag_active = False
            self.setCursor(QtCore.Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not self.interactive:
            super().keyPressEvent(event)
            return

        step = 0.1 if event.modifiers() & QtCore.Qt.ShiftModifier else 1.0
        dx, dy = self._offset_x, self._offset_y
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            dx -= step
        elif key == QtCore.Qt.Key_Right:
            dx += step
        elif key == QtCore.Qt.Key_Up:
            dy -= step
        elif key == QtCore.Qt.Key_Down:
            dy += step
        else:
            super().keyPressEvent(event)
            return
        self.set_offset(dx, dy)

# ---------------------------------------------------------------------------
# Package 4: SegmentCorrelationTracker
# ---------------------------------------------------------------------------

class SegmentCorrelationTracker:
    """
    Track drift between keyframes using phase correlation.
    Runs frame-by-frame correlation within each keyframe segment,
    then applies affine blending to match keyframe endpoints.

    キーフレーム間の位相相関によるドリフト追跡。
    各セグメント内で逐フレーム相関を実行し、
    アフィンブレンディングでキーフレーム端点に一致させる。
    """
    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.shift_x = np.zeros(total_frames)
        self.shift_y = np.zeros(total_frames)
        self.uncertainty = np.ones(total_frames)
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def track(self, keyframes: list[dict], frames: list[np.ndarray],
              progress_callback=None) -> str:
        """
        Track drift using phase correlation between keyframe segments.

        Parameters
        ----------
        keyframes : list[dict]
            Each dict: {'frame_index': int, 'dx_px': float, 'dy_px': float}
        frames : list[np.ndarray]
            Image frames (float64 AFM data).
        progress_callback : callable, optional
            Called with (current_step, total_steps).
        """
        self.shift_x[:] = 0.0
        self.shift_y[:] = 0.0
        self.uncertainty[:] = 1.0
        self._fitted = False

        if not keyframes:
            return "No keyframes provided."

        kf_sorted = sorted(keyframes, key=lambda k: k['frame_index'])
        total_corr_steps = self.total_frames - 1  # max correlation steps
        step_counter = [0]

        def _correlate_pair(j: int) -> tuple[float, float]:
            """Run phase correlation between frame j-1 and j."""
            shift_vec, _, _ = phase_cross_correlation(
                frames[j - 1], frames[j], upsample_factor=10)
            step_counter[0] += 1
            if progress_callback:
                progress_callback(step_counter[0], total_corr_steps)
            return float(shift_vec[1]), float(shift_vec[0])  # dx, dy

        def _track_segment_constrained(frame_a: int, frame_b: int,
                                       base_dx: float, base_dy: float,
                                       target_dx: float, target_dy: float):
            """Constrained segment: correlation + affine blending."""
            seg_len = frame_b - frame_a
            if seg_len <= 0:
                return

            # Raw cumulative correlation relative to frame_a
            raw_dx, raw_dy = 0.0, 0.0
            raw_cum_dx = np.zeros(seg_len + 1)
            raw_cum_dy = np.zeros(seg_len + 1)

            for j in range(frame_a + 1, frame_b + 1):
                ddx, ddy = _correlate_pair(j)
                raw_dx += ddx
                raw_dy += ddy
                raw_cum_dx[j - frame_a] = raw_dx
                raw_cum_dy[j - frame_a] = raw_dy

            # Endpoint error
            expected_dx = target_dx - base_dx
            expected_dy = target_dy - base_dy
            error_dx = expected_dx - raw_cum_dx[seg_len]
            error_dy = expected_dy - raw_cum_dy[seg_len]

            # Affine blending + base offset
            for k in range(seg_len + 1):
                t = k / seg_len if seg_len > 0 else 0.0
                self.shift_x[frame_a + k] = base_dx + raw_cum_dx[k] + t * error_dx
                self.shift_y[frame_a + k] = base_dy + raw_cum_dy[k] + t * error_dy
                # Uncertainty: magnitude of linear correction at this point
                correction_mag = t * np.sqrt(error_dx**2 + error_dy**2)
                self.uncertainty[frame_a + k] = correction_mag

        def _track_segment_free(frame_a: int, frame_b: int,
                                base_dx: float, base_dy: float,
                                forward: bool = True):
            """Extrapolation segment: raw correlation only, no endpoint constraint."""
            if forward:
                cum_dx, cum_dy = 0.0, 0.0
                self.shift_x[frame_a] = base_dx
                self.shift_y[frame_a] = base_dy
                self.uncertainty[frame_a] = 0.5
                for j in range(frame_a + 1, frame_b + 1):
                    ddx, ddy = _correlate_pair(j)
                    cum_dx += ddx
                    cum_dy += ddy
                    self.shift_x[j] = base_dx + cum_dx
                    self.shift_y[j] = base_dy + cum_dy
                    self.uncertainty[j] = 0.5
            else:
                # Backward: correlate from frame_b down to frame_a
                cum_dx, cum_dy = 0.0, 0.0
                self.shift_x[frame_b] = base_dx
                self.shift_y[frame_b] = base_dy
                self.uncertainty[frame_b] = 0.5
                for j in range(frame_b - 1, frame_a - 1, -1):
                    # Correlate j+1 vs j (reversed)
                    shift_vec, _, _ = phase_cross_correlation(
                        frames[j + 1], frames[j], upsample_factor=10)
                    step_counter[0] += 1
                    if progress_callback:
                        progress_callback(step_counter[0], total_corr_steps)
                    cum_dx += float(shift_vec[1])
                    cum_dy += float(shift_vec[0])
                    self.shift_x[j] = base_dx + cum_dx
                    self.shift_y[j] = base_dy + cum_dy
                    self.uncertainty[j] = 0.5

        # --- Single keyframe: forward + backward from that point ---
        if len(kf_sorted) == 1:
            kf = kf_sorted[0]
            fi = kf['frame_index']
            dx0, dy0 = kf['dx_px'], kf['dy_px']
            self.shift_x[fi] = dx0
            self.shift_y[fi] = dy0
            self.uncertainty[fi] = 0.0

            if fi > 0:
                _track_segment_free(0, fi, dx0, dy0, forward=False)
            if fi < self.total_frames - 1:
                _track_segment_free(fi, self.total_frames - 1, dx0, dy0, forward=True)

            self._fitted = True
            return f"Tracking complete (1 keyframe, {step_counter[0]} correlations)."

        # --- Multiple keyframes ---
        # 1. Extrapolation before first keyframe
        first_kf = kf_sorted[0]
        if first_kf['frame_index'] > 0:
            _track_segment_free(0, first_kf['frame_index'],
                                first_kf['dx_px'], first_kf['dy_px'],
                                forward=False)

        # 2. Constrained segments between consecutive keyframes
        for i in range(len(kf_sorted) - 1):
            kf_a = kf_sorted[i]
            kf_b = kf_sorted[i + 1]
            _track_segment_constrained(
                kf_a['frame_index'], kf_b['frame_index'],
                kf_a['dx_px'], kf_a['dy_px'],
                kf_b['dx_px'], kf_b['dy_px'])

        # 3. Extrapolation after last keyframe
        last_kf = kf_sorted[-1]
        if last_kf['frame_index'] < self.total_frames - 1:
            _track_segment_free(last_kf['frame_index'], self.total_frames - 1,
                                last_kf['dx_px'], last_kf['dy_px'],
                                forward=True)

        # Mark keyframe uncertainty as 0
        for kf in kf_sorted:
            idx = kf['frame_index']
            if 0 <= idx < self.total_frames:
                self.uncertainty[idx] = 0.0

        self._fitted = True
        return (f"Tracking complete ({len(kf_sorted)} keyframes, "
                f"{step_counter[0]} correlations).")


class CorrelationWorker(QtCore.QThread):
    """
    Background worker for segment correlation tracking.
    セグメント相関追跡のバックグラウンドワーカー。
    """
    progress = QtCore.pyqtSignal(int, int)    # (current, total)
    finished = QtCore.pyqtSignal(str)         # status message
    error = QtCore.pyqtSignal(str)            # error message

    def __init__(self, tracker: SegmentCorrelationTracker,
                 keyframes: list[dict], frames: list[np.ndarray]):
        super().__init__()
        self.tracker = tracker
        self.keyframes = keyframes
        self.frames = frames

    def run(self):
        try:
            msg = self.tracker.track(
                self.keyframes, self.frames,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot))
            self.finished.emit(msg)
        except Exception as e:
            self.error.emit(str(e))

# ---------------------------------------------------------------------------
# Package 3 + 5 + 6 + 7: Main Plugin Window
# ---------------------------------------------------------------------------

class InteractiveDriftWindow(QtWidgets.QWidget):
    """
    Main plugin window for interactive drift compensation.

    インタラクティブ ドリフト補正のメインウィンドウ。
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.current_frame = 0
        self.source_identity = _source_identity()

        # Load frames from pyNuD
        self.frames: list[np.ndarray] = []
        self.data_type_2ch = int(getattr(gv, 'DataType2ch', 0))
        self.frames_2ch = [] if self.data_type_2ch else None
        self.corrected_frames_2ch_cache = None
        total_frames = getattr(gv, "FrameNum", 1) if gv else 1
        for i in range(total_frames):
            try:
                frame_data = self.main_window.getImageDataForFrame(i)
                if frame_data is None or np.asarray(frame_data).ndim != 2:
                    raise ValueError(f"Could not read frame {i + 1}.")
                self.frames.append(np.array(frame_data, copy=True))
                if self.frames_2ch is not None:
                    from fileio import asd_channel2_to_volts
                    second = self.main_window.getImageDataForFrame(i, channel='2ch')
                    if second is None or np.shape(second) != np.shape(frame_data):
                        raise ValueError(f"Could not read second-channel frame {i + 1}.")
                    self.frames_2ch.append(asd_channel2_to_volts(second).copy())
            except Exception as exc:
                # Never skip a frame: that would change all subsequent keyframe indices.
                raise ValueError(f"Cannot load the complete movie: {exc}") from exc

        self.total_frames = len(self.frames) if self.frames else 1

        # Physical units
        self.scan_x_nm = getattr(gv, 'XScanSize', 100.0) if gv else 100.0
        self.scan_y_nm = getattr(gv, 'YScanSize', 100.0) if gv else 100.0
        x_pixel = max(1, getattr(gv, 'XPixel', 1) if gv else 1)
        y_pixel = max(1, getattr(gv, 'YPixel', 1) if gv else 1)
        self.nm_per_px_x = self.scan_x_nm / x_pixel
        self.nm_per_px_y = self.scan_y_nm / y_pixel

        # Keyframes and tracking
        self.keyframes: list[dict] = []  # {'frame_index', 'dx_px', 'dy_px'}
        self.interpolator: SegmentCorrelationTracker | None = None
        self.corrected_frames_cache: list[np.ndarray] | None = None
        self._lookback_saved: dict | None = None  # saved state during look-back hold

        self.setWindowTitle(PLUGIN_NAME)
        self.resize(1200, 800)
        self._setup_ui()
        self._setup_shortcuts()
        self._refresh_main_frame()

    # ---- UI Setup ----

    def _setup_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        # Top area: viewers + controls
        top_layout = QtWidgets.QHBoxLayout()

        # -- Left: Original viewer --
        orig_box = QtWidgets.QVBoxLayout()
        orig_box.addWidget(QtWidgets.QLabel("Original View (read-only)"))
        self.original_viewer = FixedGridImageViewer(interactive=False)
        orig_box.addWidget(self.original_viewer, stretch=1)
        top_layout.addLayout(orig_box, stretch=1)

        # -- Centre: Tracked viewer --
        tracked_box = QtWidgets.QVBoxLayout()
        self.offset_label = QtWidgets.QLabel("Tracked View — offset: (0.0, 0.0) px")
        tracked_box.addWidget(self.offset_label)
        self.tracked_viewer = FixedGridImageViewer(interactive=True)
        self.tracked_viewer.offset_changed.connect(self._on_offset_changed)
        tracked_box.addWidget(self.tracked_viewer, stretch=1)
        top_layout.addLayout(tracked_box, stretch=1)

        # -- Right: Controls --
        ctrl_layout = QtWidgets.QVBoxLayout()

        # Frame slider
        frame_group = QtWidgets.QGroupBox("Frame")
        frame_gl = QtWidgets.QVBoxLayout(frame_group)
        self.frame_label = QtWidgets.QLabel("Frame: 0")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        frame_gl.addWidget(self.frame_label)
        frame_gl.addWidget(self.frame_slider)
        ctrl_layout.addWidget(frame_group)

        # Navigation buttons
        nav_group = QtWidgets.QGroupBox("Navigation")
        nav_gl = QtWidgets.QGridLayout(nav_group)

        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(0.1, 50.0)
        self.step_spin.setValue(1.0)
        self.step_spin.setSingleStep(0.1)
        self.step_spin.setSuffix(" px")
        nav_gl.addWidget(QtWidgets.QLabel("Step:"), 0, 0)
        nav_gl.addWidget(self.step_spin, 0, 1, 1, 2)

        btn_up = QtWidgets.QPushButton("▲")
        btn_up.setToolTip('Move the displayed image up by Step pixels.')
        btn_down = QtWidgets.QPushButton("▼")
        btn_down.setToolTip('Move the displayed image down by Step pixels.')
        btn_left = QtWidgets.QPushButton("◀")
        btn_left.setToolTip('Move the displayed image left by Step pixels.')
        btn_right = QtWidgets.QPushButton("▶")
        btn_right.setToolTip('Move the displayed image right by Step pixels.')
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.setToolTip('Reset the displayed alignment offset to zero.')
        for btn in (btn_up, btn_down, btn_left, btn_right, btn_reset):
            btn.setFixedHeight(32)

        btn_up.clicked.connect(lambda: self._nudge(0, -1))
        btn_down.clicked.connect(lambda: self._nudge(0, 1))
        btn_left.clicked.connect(lambda: self._nudge(-1, 0))
        btn_right.clicked.connect(lambda: self._nudge(1, 0))
        btn_reset.clicked.connect(self.tracked_viewer.reset_offset)

        self.btn_lookback = QtWidgets.QPushButton("◁ Look Back")
        self.btn_lookback.setFixedHeight(32)
        self.btn_lookback.setToolTip(
            "Hold to preview previous recorded keyframe's shifted image")
        self.btn_lookback.pressed.connect(self._on_lookback_pressed)
        self.btn_lookback.released.connect(self._on_lookback_released)

        nav_gl.addWidget(btn_up, 1, 1)
        nav_gl.addWidget(btn_left, 2, 0)
        nav_gl.addWidget(btn_reset, 2, 1)
        nav_gl.addWidget(btn_right, 2, 2)
        nav_gl.addWidget(btn_down, 3, 1)
        nav_gl.addWidget(self.btn_lookback, 4, 0, 1, 3)
        ctrl_layout.addWidget(nav_group)

        # Actions
        action_group = QtWidgets.QGroupBox("Actions")
        action_layout = QtWidgets.QVBoxLayout(action_group)

        self.btn_record = QtWidgets.QPushButton("Record Offset")
        self.btn_record.setMinimumHeight(36)
        self.btn_record.clicked.connect(self._on_record_clicked)

        self.btn_track = QtWidgets.QPushButton("Track")
        self.btn_track.setMinimumHeight(36)
        self.btn_track.clicked.connect(self._on_track_clicked)

        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setMinimumHeight(36)
        self.btn_apply.clicked.connect(self._on_apply_clicked)

        self.btn_export = QtWidgets.QPushButton("Export CSV")
        self.btn_export.setMinimumHeight(36)
        self.btn_export.clicked.connect(self._on_export_clicked)

        self.btn_load_csv = QtWidgets.QPushButton("Load CSV")
        self.btn_load_csv.setMinimumHeight(36)
        self.btn_load_csv.clicked.connect(self._on_load_csv_clicked)

        self.btn_save_asd = QtWidgets.QPushButton("Save .asd")
        self.btn_save_asd.setMinimumHeight(36)
        self.btn_save_asd.clicked.connect(self._on_save_asd_clicked)

        for btn in (self.btn_record, self.btn_track, self.btn_apply,
                    self.btn_export, self.btn_load_csv, self.btn_save_asd):
            action_layout.addWidget(btn)
        ctrl_layout.addWidget(action_group)

        channels = '1ch + 2ch' if self.frames_2ch is not None else '1ch'
        self.use_applied_check = QtWidgets.QCheckBox(f"Use Applied Correction ({channels})")
        self.use_applied_check.setEnabled(False)
        self.use_applied_check.setToolTip(
            "Show the movie captured by Apply in the main view. Turn off to return to the current processing settings. "
            f"The same correction is applied to {channels}. Changes to tracking require another Apply. "
            "The source file is preserved.")
        self.use_applied_check.toggled.connect(self._refresh_main_frame)
        ctrl_layout.addWidget(self.use_applied_check)

        # Grid toggle
        self.grid_check = QtWidgets.QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self._on_grid_toggled)
        ctrl_layout.addWidget(self.grid_check)

        # Progress bar (hidden by default, shown during tracking)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        ctrl_layout.addWidget(self.progress_bar)

        # Help button
        self.btn_help = QtWidgets.QPushButton("? Help")
        self.btn_help.setFixedHeight(28)
        self.btn_help.clicked.connect(self._show_help)
        ctrl_layout.addWidget(self.btn_help)

        # Status
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: green;")
        ctrl_layout.addWidget(self.status_label)
        ctrl_layout.addStretch()

        top_layout.addLayout(ctrl_layout)
        root.addLayout(top_layout, stretch=3)

        # Bottom area: keyframe table + drift plot
        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Keyframe table
        table_box = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(QtWidgets.QLabel("Keyframes"))

        self.kf_table = QtWidgets.QTableWidget()
        self.kf_table.setColumnCount(6)
        self.kf_table.setHorizontalHeaderLabels(
            ["Frame", "dX (px)", "dY (px)", "dX (nm)", "dY (nm)", ""]
        )
        self.kf_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.kf_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table_layout.addWidget(self.kf_table)

        btn_del_kf = QtWidgets.QPushButton("Delete Selected Keyframe")
        btn_del_kf.setToolTip('Delete the selected recorded keyframe from the manual drift constraints.')
        btn_del_kf.clicked.connect(self._on_delete_keyframe)
        table_layout.addWidget(btn_del_kf)
        bottom_splitter.addWidget(table_box)

        # Drift plot
        plot_box = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_canvas = FigureCanvas(Figure(figsize=(5, 2.5)))
        self.plot_ax = self.plot_canvas.figure.subplots()
        self.plot_ax.set_title("Drift over Frames", fontsize=9)
        plot_layout.addWidget(self.plot_canvas)
        bottom_splitter.addWidget(plot_box)

        root.addWidget(bottom_splitter, stretch=1)

        # Connect to main window frame changes
        if hasattr(self.main_window, 'frameChanged'):
            try:
                self.main_window.frameChanged.connect(self._on_main_frame_changed)
            except Exception:
                pass

        # Initial display
        if self.frames:
            self._on_main_frame_changed(getattr(gv, 'index', 0) if gv else 0)

    # ---- Shortcuts ----

    def _setup_shortcuts(self) -> None:
        """Register keyboard shortcuts for frame navigation and actions."""
        ctx = QtCore.Qt.WidgetWithChildrenShortcut

        # Ctrl+A: previous frame
        sc_prev = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+A"), self)
        sc_prev.setContext(ctx)
        sc_prev.activated.connect(self._shortcut_prev_frame)

        # Ctrl+D: next frame
        sc_next = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+D"), self)
        sc_next.setContext(ctx)
        sc_next.activated.connect(self._shortcut_next_frame)

        # Ctrl+R: record offset
        sc_record = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        sc_record.setContext(ctx)
        sc_record.activated.connect(self._on_record_clicked)

    def _shortcut_prev_frame(self) -> None:
        new_frame = max(0, self.current_frame - 1)
        self.frame_slider.setValue(new_frame)

    def _shortcut_next_frame(self) -> None:
        new_frame = min(self.total_frames - 1, self.current_frame + 1)
        self.frame_slider.setValue(new_frame)

    # ---- Helpers ----

    def _render_frame_to_pixmap(self, arr2d: np.ndarray) -> QtGui.QPixmap:
        import cv2
        norm = cv2.normalize(arr2d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        q_img = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img.copy())

    def _nudge(self, dir_x: int, dir_y: int) -> None:
        step = self.step_spin.value()
        dx, dy = self.tracked_viewer.get_offset()
        self.tracked_viewer.set_offset(dx + dir_x * step, dy + dir_y * step)

    def _keyframe_for_frame(self, frame_index: int) -> dict | None:
        for kf in self.keyframes:
            if kf['frame_index'] == frame_index:
                return kf
        return None

    def _prev_keyframe(self, frame_index: int) -> dict | None:
        """Return the latest keyframe strictly before frame_index, or None.
        前フレームの直近キーフレームを返す。keyframes は常にソート済みを前提とする。"""
        result = None
        for kf in self.keyframes:
            if kf['frame_index'] < frame_index:
                result = kf
            else:
                break  # sorted — can exit early
        return result

    # ---- Slots ----

    def _on_offset_changed(self, dx: float, dy: float) -> None:
        dx_nm = dx * self.nm_per_px_x
        dy_nm = dy * self.nm_per_px_y
        self.offset_label.setText(
            f"Tracked View — offset: ({dx:.2f}, {dy:.2f}) px  |  ({dx_nm:.2f}, {dy_nm:.2f}) nm"
        )

    def _on_frame_slider_changed(self, value: int) -> None:
        # Sync to main window
        if hasattr(self.main_window, 'frameSlider'):
            self.main_window.frameSlider.setValue(value)
        elif hasattr(self.main_window, 'setFrame'):
            self.main_window.setFrame(value)
        else:
            self._on_main_frame_changed(value)

    def _on_main_frame_changed(self, frame_index: int) -> None:
        self.current_frame = frame_index
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"Frame: {frame_index}")

        if not self.frames or frame_index >= len(self.frames):
            return

        pixmap = self._render_frame_to_pixmap(self.frames[frame_index])
        self.original_viewer.set_image(pixmap)
        self.tracked_viewer.set_image(pixmap)

        # Load offset: keyframe > tracked > zero
        kf = self._keyframe_for_frame(frame_index)
        if kf is not None:
            self.tracked_viewer.set_offset(kf['dx_px'], kf['dy_px'])
            self.status_label.setText(f"Keyframe (frame {frame_index})")
            self.status_label.setStyleSheet("color: #00aa00;")
        elif self.interpolator is not None and self.interpolator.is_fitted:
            dx = float(self.interpolator.shift_x[frame_index])
            dy = float(self.interpolator.shift_y[frame_index])
            self.tracked_viewer.set_offset(dx, dy)
            unc = self.interpolator.uncertainty[frame_index]
            self.status_label.setText(f"Tracked (unc={unc:.2f})")
            self.status_label.setStyleSheet("color: #888;")
        else:
            prev_kf = self._prev_keyframe(frame_index)
            if prev_kf is not None:
                self.tracked_viewer.set_offset(prev_kf['dx_px'], prev_kf['dy_px'])
                self.status_label.setText(
                    f"Carry-forward from frame {prev_kf['frame_index']} "
                    f"({prev_kf['dx_px']:.2f}, {prev_kf['dy_px']:.2f}) px")
                self.status_label.setStyleSheet("color: #888;")
            else:
                self.tracked_viewer.set_offset(0.0, 0.0)
                self.status_label.setText("No data")
                self.status_label.setStyleSheet("color: #888;")

    def _on_grid_toggled(self, checked: bool) -> None:
        self.original_viewer.set_grid_visible(checked)
        self.tracked_viewer.set_grid_visible(checked)

    # ---- Look Back ----

    def _on_lookback_pressed(self) -> None:
        """Show previous recorded keyframe's shifted image while held."""
        if not self.frames or not self.keyframes:
            return

        # Find the most recent keyframe strictly before current_frame
        prev_kf = None
        for kf in reversed(self.keyframes):
            if kf['frame_index'] < self.current_frame:
                prev_kf = kf
                break

        if prev_kf is None:
            self.status_label.setText("No previous keyframe to look back to.")
            return

        # Save current state for restoration on release
        self._lookback_saved = {
            'offset': self.tracked_viewer.get_offset(),
            'frame': self.current_frame,
        }

        # Show the previous keyframe's original image with its recorded offset
        fi = prev_kf['frame_index']
        dx, dy = prev_kf['dx_px'], prev_kf['dy_px']

        if fi < len(self.frames):
            pixmap = self._render_frame_to_pixmap(self.frames[fi])
            self.tracked_viewer.set_image(pixmap)
            self.tracked_viewer.set_offset(dx, dy)
            self.offset_label.setText(
                f"◁ Look Back — frame {fi} (dx={dx:.2f}, dy={dy:.2f}) px")

    def _on_lookback_released(self) -> None:
        """Restore current frame state after look-back release."""
        if self._lookback_saved is None:
            return

        saved = self._lookback_saved
        self._lookback_saved = None

        # Restore current frame's image
        fi = saved['frame']
        if fi < len(self.frames):
            pixmap = self._render_frame_to_pixmap(self.frames[fi])
            self.tracked_viewer.set_image(pixmap)

        # Restore cached offset
        dx, dy = saved['offset']
        self.tracked_viewer.set_offset(dx, dy)

    # ---- Package 3: Keyframe Recording ----

    def _on_record_clicked(self) -> None:
        dx, dy = self.tracked_viewer.get_offset()
        fi = self.current_frame

        # Update existing or add new
        existing = self._keyframe_for_frame(fi)
        if existing is not None:
            existing['dx_px'] = dx
            existing['dy_px'] = dy
        else:
            self.keyframes.append({'frame_index': fi, 'dx_px': dx, 'dy_px': dy})

        self.keyframes.sort(key=lambda k: k['frame_index'])
        self.interpolator = None
        self._refresh_keyframe_table()
        self._update_drift_plot()
        self.status_label.setText(f"Recorded offset at frame {fi}: ({dx:.2f}, {dy:.2f}) px")
        self.status_label.setStyleSheet("color: #00aa00;")

    def _on_delete_keyframe(self) -> None:
        row = self.kf_table.currentRow()
        if row < 0 or row >= len(self.keyframes):
            return
        removed = self.keyframes.pop(row)
        self.interpolator = None
        self._refresh_keyframe_table()
        self._update_drift_plot()
        self.status_label.setText(f"Deleted keyframe at frame {removed['frame_index']}")

    def _refresh_keyframe_table(self) -> None:
        self.kf_table.setRowCount(len(self.keyframes))
        for i, kf in enumerate(self.keyframes):
            self.kf_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(kf['frame_index'])))
            self.kf_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{kf['dx_px']:.2f}"))
            self.kf_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{kf['dy_px']:.2f}"))
            self.kf_table.setItem(i, 3, QtWidgets.QTableWidgetItem(
                f"{kf['dx_px'] * self.nm_per_px_x:.2f}"))
            self.kf_table.setItem(i, 4, QtWidgets.QTableWidgetItem(
                f"{kf['dy_px'] * self.nm_per_px_y:.2f}"))
            # Indicator column
            self.kf_table.setItem(i, 5, QtWidgets.QTableWidgetItem("●"))

    # ---- Package 4: Correlation Tracking ----

    def _on_track_clicked(self) -> None:
        """Start segment-wise correlation tracking in a background thread."""
        if len(self.keyframes) < 1:
            self.status_label.setText("Need at least 1 keyframe to track.")
            self.status_label.setStyleSheet("color: red;")
            return
        if not self.frames:
            self.status_label.setText("No frames loaded.")
            self.status_label.setStyleSheet("color: red;")
            return

        self.btn_track.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Tracking...")
        self.status_label.setStyleSheet("color: #888;")

        self.interpolator = SegmentCorrelationTracker(self.total_frames)
        self._worker = CorrelationWorker(
            self.interpolator, self.keyframes, self.frames)
        self._worker.progress.connect(self._on_track_progress)
        self._worker.finished.connect(self._on_track_finished)
        self._worker.error.connect(self._on_track_error)
        self._worker.start()

    def _on_track_progress(self, current: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_track_finished(self, msg: str) -> None:
        self.btn_track.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #00aa00;")
        self._update_drift_plot()
        self._on_main_frame_changed(self.current_frame)

    def _on_track_error(self, msg: str) -> None:
        self.btn_track.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Tracking error: {msg}")
        self.status_label.setStyleSheet("color: red;")

    # ---- Load CSV ----

    def _on_load_csv_clicked(self) -> None:
        """Load keyframes from a previously exported CSV file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Keyframe CSV", "", "CSV (*.csv);;All Files (*)")
        if not path:
            return

        try:
            loaded_keyframes = []
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row.get('is_keyframe', 0)) == 1:
                        fi = int(row['frame_index'])
                        dx = float(row['shift_x_px'])
                        dy = float(row['shift_y_px'])
                        if 0 <= fi < self.total_frames:
                            loaded_keyframes.append({
                                'frame_index': fi, 'dx_px': dx, 'dy_px': dy
                            })

            if not loaded_keyframes:
                self.status_label.setText("No keyframes found in CSV.")
                self.status_label.setStyleSheet("color: red;")
                return

            self.keyframes = sorted(loaded_keyframes, key=lambda k: k['frame_index'])
            self.interpolator = None
            self._refresh_keyframe_table()
            self._update_drift_plot()
            self.status_label.setText(
                f"Loaded {len(self.keyframes)} keyframes from {os.path.basename(path)}")
            self.status_label.setStyleSheet("color: #00aa00;")
            self._on_main_frame_changed(self.current_frame)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load CSV:\n{e}")
            self.status_label.setText(f"Load error: {e}")
            self.status_label.setStyleSheet("color: red;")

    # ---- Package 5: Drift Plot ----

    def _update_drift_plot(self) -> None:
        ax = self.plot_ax
        ax.clear()
        ax.set_title("Drift over Frames", fontsize=9)
        ax.set_xlabel("Frame index", fontsize=8)
        ax.set_ylabel("Drift (px)", fontsize=8)

        frames_arr = np.arange(self.total_frames)

        # Plot tracked curves
        if self.interpolator is not None and self.interpolator.is_fitted:
            ax.plot(frames_arr, self.interpolator.shift_x,
                    label='Shift X', color='#1f77b4', linewidth=1.2)
            ax.plot(frames_arr, self.interpolator.shift_y,
                    label='Shift Y', color='#ff7f0e', linewidth=1.2)
            # Uncertainty band (direct values, not ×3)
            unc = self.interpolator.uncertainty
            ax.fill_between(frames_arr,
                            self.interpolator.shift_x - unc,
                            self.interpolator.shift_x + unc,
                            alpha=0.1, color='#1f77b4')
            ax.fill_between(frames_arr,
                            self.interpolator.shift_y - unc,
                            self.interpolator.shift_y + unc,
                            alpha=0.1, color='#ff7f0e')

        # Keyframe segment boundary markers
        if self.keyframes:
            for kf in self.keyframes:
                ax.axvline(kf['frame_index'], color='green', linestyle=':',
                           alpha=0.4, linewidth=0.7)

        # Plot keyframe markers
        if self.keyframes:
            kf_frames = [kf['frame_index'] for kf in self.keyframes]
            kf_dx = [kf['dx_px'] for kf in self.keyframes]
            kf_dy = [kf['dy_px'] for kf in self.keyframes]
            ax.scatter(kf_frames, kf_dx, marker='o', s=40, c='#1f77b4',
                       zorder=5, edgecolors='white', linewidths=0.8)
            ax.scatter(kf_frames, kf_dy, marker='o', s=40, c='#ff7f0e',
                       zorder=5, edgecolors='white', linewidths=0.8)

        # Current frame indicator
        ax.axvline(self.current_frame, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        ax.legend(fontsize=7, loc='upper left')
        ax.tick_params(labelsize=7)
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

    # ---- Package 7: Apply & Export ----

    def _on_apply_clicked(self) -> None:
        if self.interpolator is None or not self.interpolator.is_fitted:
            self.status_label.setText("Run Track first.")
            self.status_label.setStyleSheet("color: red;")
            return

        self.status_label.setText("Applying drift correction...")
        self.status_label.setStyleSheet("color: #888;")
        QtWidgets.QApplication.processEvents()

        if self.source_identity is None or _source_identity() != self.source_identity:
            self.status_label.setText("Select the original source movie before applying its correction.")
            return
        try:
            self.corrected_frames_cache = corrected_movie(
                self.frames, self.interpolator.shift_x, self.interpolator.shift_y)
            if self.frames_2ch is not None:
                self.corrected_frames_2ch_cache = corrected_movie(
                    self.frames_2ch, self.interpolator.shift_x, self.interpolator.shift_y)
        except (ValueError, MemoryError) as exc:
            self.status_label.setText(f"Could not apply drift correction: {exc}")
            self.status_label.setStyleSheet("color: red;")
            return
        self.use_applied_check.setEnabled(True)
        with QtCore.QSignalBlocker(self.use_applied_check):
            self.use_applied_check.setChecked(True)
        self._refresh_main_frame()
        channels = '1ch + 2ch' if self.frames_2ch is not None else '1ch'
        self.status_label.setText(f"Drift applied to the main view ({channels}, original image size).")
        self.status_label.setStyleSheet("color: #00aa00;")

    def apply_to_current_frame(self):
        """Main processing hook: publish one frame, never a 3-D stack to gv.aryData."""
        if (self.corrected_frames_cache is None or not self.use_applied_check.isChecked()
                or self.source_identity is None or _source_identity() != self.source_identity):
            return
        index = int(getattr(gv, 'index', -1))
        if 0 <= index < len(self.corrected_frames_cache):
            gv.aryData_processed_1ch = self.corrected_frames_cache[index].copy()
            second = getattr(self, 'corrected_frames_2ch_cache', None)
            if second is not None:
                data = second[index].copy()
                if getattr(gv, 'ch2_data_type', 'Other') in ('Phase', 'Amplitude', 'AmplitudeError'):
                    data *= float(getattr(gv, 'ch2_conversion_factor', 1.))
                gv.aryData_processed_2ch = data

    def _refresh_main_frame(self, *_):
        # Reload first so toggling off cannot reprocess an already corrected frame.
        refresh = getattr(self.main_window, 'updateFrame', None)
        if callable(refresh):
            refresh()
            return
        if hasattr(self.main_window, 'getImageDataForFrame'):
            self.main_window.getImageDataForFrame(int(getattr(gv, 'index', 0)))
        if hasattr(self.main_window, 'UpdateDisplayImage'):
            self.main_window.UpdateDisplayImage()

    def _on_export_clicked(self) -> None:
        if self.interpolator is None or not self.interpolator.is_fitted:
            self.status_label.setText("Run Track first.")
            self.status_label.setStyleSheet("color: red;")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Drift CSV", "interactive_drift.csv", "CSV (*.csv)")
        if not path:
            return

        kf_set = {kf['frame_index'] for kf in self.keyframes}
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "shift_x_px", "shift_y_px",
                        "shift_x_nm", "shift_y_nm", "uncertainty", "is_keyframe"])
            for i in range(self.total_frames):
                w.writerow([
                    i,
                    f"{self.interpolator.shift_x[i]:.4f}",
                    f"{self.interpolator.shift_y[i]:.4f}",
                    f"{self.interpolator.shift_x[i] * self.nm_per_px_x:.4f}",
                    f"{self.interpolator.shift_y[i] * self.nm_per_px_y:.4f}",
                    f"{self.interpolator.uncertainty[i]:.4f}",
                    1 if i in kf_set else 0,
                ])

        QtWidgets.QMessageBox.information(
            self, "Export", f"CSV exported:\n{path}")
        self.status_label.setText(f"CSV exported: {os.path.basename(path)}")

    def _on_save_asd_clicked(self) -> None:
        """
        Save drift-compensated ASD with expanded image scope.
        拡張キャンバスにシフト適用済みフレームを配置してASD保存。

        The output pixel dimensions and scan range are enlarged to
        accommodate the full drift range (bounding box of all shifted
        frame positions), following the same algorithm used in
        ``drift_compensation.py:run_make_shifted_image``.
        """
        if self.interpolator is None or not self.interpolator.is_fitted:
            self.status_label.setText("Run Track first.")
            self.status_label.setStyleSheet("color: red;")
            return

        if not self.frames:
            self.status_label.setText("No frames loaded.")
            self.status_label.setStyleSheet("color: red;")
            return

        default_save = "DriftCompensated.asd"
        try:
            if getattr(gv, "files", None) and len(gv.files) > getattr(gv, "currentFileNum", -1):
                base, _ = os.path.splitext(gv.files[gv.currentFileNum])
                default_save = f"{base}_DriftComp.asd"
        except Exception:
            pass

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Compensated ASD", default_save, "ASD Files (*.asd)")
        if not path:
            return

        try:
            from fileio import SaveASD

            if self.source_identity is None or _source_identity() != self.source_identity:
                raise ValueError("Select the original source movie before saving its correction.")
            expanded_stack = corrected_movie(
                self.frames, self.interpolator.shift_x, self.interpolator.shift_y, expanded=True)
            second_frames = getattr(self, 'frames_2ch', None)
            if getattr(gv, 'DataType2ch', 0) and second_frames is None:
                raise ValueError("Reload the plugin to capture both channels before saving.")
            expanded_second = (corrected_movie(second_frames, self.interpolator.shift_x,
                self.interpolator.shift_y, expanded=True) if second_frames is not None else None)
            h_orig, w_orig = self.frames[0].shape
            new_height, new_width = expanded_stack.shape[1:]
            saved = SaveASD(path, "Interactive Drift Compensation", expanded_stack,
                           save_scan_size_nm=(new_width * self.nm_per_px_x, new_height * self.nm_per_px_y),
                           processed_data_array_2ch=expanded_second,
                           data_type_2ch=getattr(self, 'data_type_2ch', getattr(gv, 'DataType2ch', 0)))
            if not saved:
                raise RuntimeError("The ASD writer did not complete the save.")

            QtWidgets.QMessageBox.information(
                self, "Save", f"ASD saved:\n{path}\n"
                f"Expanded: {w_orig}×{h_orig} → {new_width}×{new_height} px")
            self.status_label.setText(f"ASD saved: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save:\n{e}")
            self.status_label.setText(f"Save error: {e}")

    # ---- Help ----

    def _show_help(self) -> None:
        """Show bilingual help dialog.  バイリンガルヘルプダイアログを表示。"""
        dialog = QtWidgets.QDialog(self)
        dialog.setMinimumSize(560, 580)
        dialog.resize(660, 700)
        ll = QtWidgets.QVBoxLayout(dialog)

        # Language toggle row
        lang_row = QtWidgets.QHBoxLayout()
        lang_row.addWidget(QtWidgets.QLabel("Language / 言語:"))
        btn_ja = QtWidgets.QPushButton("日本語", dialog)
        btn_en = QtWidgets.QPushButton("English", dialog)
        btn_ja.setCheckable(True)
        btn_en.setCheckable(True)
        grp = QtWidgets.QButtonGroup(dialog)
        grp.addButton(btn_ja)
        grp.addButton(btn_en)
        grp.setExclusive(True)
        sel = "QPushButton { background-color: #007aff; color: white; font-weight: bold; }"
        nrm = "QPushButton { background-color: #e5e5e5; color: black; }"
        lang_row.addWidget(btn_ja)
        lang_row.addWidget(btn_en)
        lang_row.addStretch()
        ll.addLayout(lang_row)

        # Content browser
        browser = QtWidgets.QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        css = (
            "body { font-size: 14px; line-height: 1.6; }"
            "h1 { font-size: 20px; color: #2c3e50;"
            "     border-bottom: 2px solid #3498db; padding-bottom: 4px; }"
            "h2 { font-size: 16px; color: #2c3e50; margin-top: 14px; }"
            "ul, ol { padding-left: 20px; }"
            "table { border-collapse: collapse; margin-top: 6px; }"
            "td { padding: 2px 12px 2px 0; }"
        )
        browser.document().setDefaultStyleSheet(css)
        ll.addWidget(browser, stretch=1)

        close_btn = QtWidgets.QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        ll.addWidget(close_btn)

        def set_lang(ja: bool) -> None:
            btn_ja.setChecked(ja)
            btn_en.setChecked(not ja)
            btn_ja.setStyleSheet(sel if ja else nrm)
            btn_en.setStyleSheet(sel if not ja else nrm)
            html = HELP_HTML_JA if ja else HELP_HTML_EN
            browser.setHtml(f"<html><body>{html.strip()}</body></html>")
            dialog.setWindowTitle(
                "インタラクティブ ドリフト補正 - マニュアル" if ja
                else "Interactive Drift Compensation - Manual")
            close_btn.setText("閉じる" if ja else "Close")

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))

        # Default language: English
        set_lang(False)

        dialog.exec_()

    # ---- Lifecycle ----

    def closeEvent(self, event) -> None:
        if hasattr(self.main_window, 'frameChanged'):
            try:
                self.main_window.frameChanged.disconnect(self._on_main_frame_changed)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Plugin Entry Point
# ---------------------------------------------------------------------------

def create_plugin(main_window):
    return InteractiveDriftWindow(main_window)

__all__ = ["PLUGIN_NAME", "create_plugin"]
