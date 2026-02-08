#!/usr/bin/env python3
# type: ignore
"""
pyNuD Simulator - Standalone application for AFM image simulation from PDB/CIF/MRC.
Uses simplified VTK rendering for speed and reliability.

Version: 1.0.0
"""

__version__ = "1.0.0"

import sys
import numpy as np
import os
import json 
import html
import re
import struct  # ★★★ 追加 ★★★
import datetime # ★★★ 追加 ★★★
import time
import tempfile
import math
from pathlib import Path
from types import SimpleNamespace
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                            QSlider, QComboBox, QSpinBox, QDoubleSpinBox,
                            QGroupBox, QFileDialog, QMessageBox, QTextEdit,
                            QSplitter, QFrame, QCheckBox, QScrollArea,
                            QColorDialog, QTabWidget, QProgressBar, QInputDialog, QAction,
                            QTreeWidget, QTextBrowser, QTreeWidgetItem, QSpacerItem, QSizePolicy, QLineEdit, QDialog, QProgressDialog,
                            QStackedLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QTime, QSettings, QEventLoop, QEvent, QObject, QRect
from PyQt5.QtGui import QFont, QColor, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

# Support plugin launch: use globalvals when run from pyNuD, else minimal stub.
try:
    import globalvals as gv
except ModuleNotFoundError:
    class _GlobalValsStub:
        standardFont = "Helvetica"
        main_window = None
    gv = _GlobalValsStub()

try:
    from vtkmodules import (  # type: ignore
        vtkCommonCore,
        vtkCommonDataModel,
        vtkCommonMath,
        vtkCommonTransforms,
        vtkFiltersCore,
        vtkFiltersGeneral,
        vtkFiltersGeometry,
        vtkFiltersSources,
        vtkInteractionStyle,
        vtkInteractionWidgets,
        vtkRenderingAnnotation,
        vtkRenderingCore,
        # Required for rendering backend registration (prevents blank/black renders on some builds)
        vtkRenderingOpenGL2,
        vtkIOImage,
    )
    vtk = SimpleNamespace(  # type: ignore
        VTK_UNSIGNED_CHAR=vtkCommonCore.VTK_UNSIGNED_CHAR,
        vtkObject=vtkCommonCore.vtkObject,
        vtkPoints=vtkCommonCore.vtkPoints,
        vtkUnsignedCharArray=vtkCommonCore.vtkUnsignedCharArray,
        vtkMatrix4x4=vtkCommonMath.vtkMatrix4x4,
        vtkTransform=vtkCommonTransforms.vtkTransform,
        vtkPolyData=vtkCommonDataModel.vtkPolyData,
        vtkCellArray=vtkCommonDataModel.vtkCellArray,
        vtkLine=vtkCommonDataModel.vtkLine,
        vtkTriangle=vtkCommonDataModel.vtkTriangle,
        vtkBoundingBox=vtkCommonDataModel.vtkBoundingBox,
        vtkImageData=vtkCommonDataModel.vtkImageData,
        vtkGlyph3D=vtkFiltersCore.vtkGlyph3D,
        vtkAppendPolyData=vtkFiltersCore.vtkAppendPolyData,
        vtkVertexGlyphFilter=vtkFiltersGeneral.vtkVertexGlyphFilter,
        vtkTubeFilter=vtkFiltersCore.vtkTubeFilter,
        vtkDelaunay2D=vtkFiltersCore.vtkDelaunay2D,
        vtkDelaunay3D=vtkFiltersCore.vtkDelaunay3D,
        vtkMarchingCubes=vtkFiltersCore.vtkMarchingCubes,
        vtkCenterOfMass=vtkFiltersCore.vtkCenterOfMass,
        vtkSmoothPolyDataFilter=vtkFiltersCore.vtkSmoothPolyDataFilter,
        vtkDataSetSurfaceFilter=vtkFiltersGeometry.vtkDataSetSurfaceFilter,
        vtkSphereSource=vtkFiltersSources.vtkSphereSource,
        vtkTransformPolyDataFilter=vtkFiltersGeneral.vtkTransformPolyDataFilter,
        vtkActor=vtkRenderingCore.vtkActor,
        vtkRenderer=vtkRenderingCore.vtkRenderer,
        vtkPolyDataMapper=vtkRenderingCore.vtkPolyDataMapper,
        vtkLight=vtkRenderingCore.vtkLight,
        vtkWindowToImageFilter=vtkRenderingCore.vtkWindowToImageFilter,
        vtkAxesActor=vtkRenderingAnnotation.vtkAxesActor,
        vtkOrientationMarkerWidget=vtkInteractionWidgets.vtkOrientationMarkerWidget,
        vtkInteractorStyleTrackballCamera=vtkInteractionStyle.vtkInteractorStyleTrackballCamera,
        vtkPNGWriter=vtkIOImage.vtkPNGWriter,
        vtkTIFFWriter=vtkIOImage.vtkTIFFWriter,
    )
except Exception as e:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    is_conda = bool(conda_prefix)
    lines = [
        "Error: Failed to import VTK modules (`vtkmodules`).",
        "",
        f"Python: {sys.executable}",
        f"CONDA_PREFIX: {conda_prefix or '(not set)'}",
        "",
        f"ImportError: {e}",
        "",
        "This typically happens when VTK binaries are mixed (e.g., pip-installed `vtk` + conda-installed `vtk`).",
    ]
    if is_conda:
        lines += [
            "",
            "Fix (recommended, conda-forge):",
            "  conda activate <your-env>",
            "  # If `pip uninstall vtk` fails (e.g. 'no RECORD file'), remove via conda:",
            "  conda remove -y vtk vtk-base",
            "  # Then reinstall a single VTK from conda-forge:",
            "  conda install -c conda-forge --force-reinstall vtk pyqt -y",
            "",
            "If you previously ran `pip install -r requirements.txt` inside conda, it likely pulled in pip's VTK.",
            "Use `requirements-conda.txt` for pip installs in conda environments.",
        ]
    else:
        lines += [
            "",
            "Fix (pip / venv):",
            "  python -m pip install --force-reinstall --no-cache-dir vtk",
        ]

    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(1) from e

STANDARD_FONT = "Helvetica"

# VTK 9.x compatibility: Try different import methods for Qt integration
try:
    # Try VTK 9.x import method
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # type: ignore
except ImportError:
    try:
        # Alternative VTK 9.x import method
        from vtkmodules.vtkRenderingQt import QVTKRenderWindowInteractor  # type: ignore
    except ImportError:
        # Fallback: show a QWidget placeholder so the UI still comes up.
        # Interactive VTK embedding requires a VTK build with Qt support.
        print("Warning: VTK Qt integration not available. VTK view will be disabled.")
        class QVTKRenderWindowInteractor(QWidget):  # type: ignore
            def __init__(self, parent=None):
                super().__init__(parent)
                layout = QVBoxLayout(self)
                layout.setContentsMargins(10, 10, 10, 10)
                msg = QLabel(
                    "VTK Qt integration is not available.\n\n"
                    "Please install VTK with Qt support (e.g., conda-forge: `vtk` + `pyqt`).\n"
                    "If you are in a conda env, avoid mixing pip VTK and conda VTK."
                )
                msg.setAlignment(Qt.AlignCenter)
                msg.setStyleSheet("color: #b00; background: #fff; border: 1px dashed #c99;")
                msg.setWordWrap(True)
                layout.addWidget(msg, 1)

            def GetRenderWindow(self):  # pragma: no cover
                return None
# Numbaをインポートして計算を高速化（オプション）

import scipy.ndimage

from scipy.fft import fft2, ifft2, fftshift, ifftshift # ★★★ この行を追加 ★★★

# Numbaをインポートして計算を高速化（オプションですが強く推奨します）
try:
    from numba import jit
except ImportError:
    # numbaがインストールされていない場合、何もしないダミーのデコレータを作成
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator


class _EspGradientBar(QWidget):
    """Small horizontal red-white-blue gradient bar for ESP legend."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(10)
        self.setMaximumHeight(10)

    def paintEvent(self, event):  # type: ignore[override]
        from PyQt5.QtGui import QPainter, QLinearGradient, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect()
        grad = QLinearGradient(rect.left(), rect.center().y(), rect.right(), rect.center().y())
        # Typical ESP legend: negative (red) -> neutral (white) -> positive (blue)
        grad.setColorAt(0.0, QColor(220, 40, 40))
        grad.setColorAt(0.5, QColor(245, 245, 245))
        grad.setColorAt(1.0, QColor(40, 80, 220))
        painter.fillRect(rect, grad)
        painter.setPen(QPen(QColor(120, 120, 120)))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

class _AspectPixmapView(QWidget):
    """
    Lightweight pixmap viewer that does not change its sizeHint when frames update.

    This avoids layout "jitter" (one-frame resize flicker) during frequent updates
    such as Interactive Update while rotating the structure.

    Behavior:
    - If display_aspect_ratio is set, we letterbox to that ratio, then draw the
      pixmap stretched into that box (IgnoreAspectRatio) to reflect physical pixel
      aspect (ScanX/ScanY).
    - Otherwise, we draw the pixmap with KeepAspectRatio into the widget rect.
    """

    roiSelected = pyqtSignal(int, int, int, int)
    roiResetRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap = None
        self._display_aspect_ratio = None
        self._roi_enabled = False
        self._roi_dragging = False
        self._roi_start_pos = None
        self._roi_end_pos = None
        self._roi_last_rect = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(1, 1)

    def setSourcePixmap(self, pixmap):
        self._source_pixmap = pixmap
        self.update()

    def setDisplayAspectRatio(self, width_over_height):
        try:
            v = float(width_over_height)
        except Exception:
            v = None
        self._display_aspect_ratio = v if (v is not None and v > 0) else None
        self.update()

    def setRoiSelectionEnabled(self, enabled):
        self._roi_enabled = bool(enabled)
        try:
            self.setCursor(Qt.CrossCursor if self._roi_enabled else Qt.ArrowCursor)
        except Exception:
            pass
        if not self._roi_enabled:
            self._roi_dragging = False
            self._roi_start_pos = None
            self._roi_end_pos = None
        self.update()

    def clearRoiOverlay(self):
        self._roi_last_rect = None
        self.update()

    def _target_rect(self):
        from PyQt5.QtCore import QRect
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return QRect()

        rect = self.rect()
        avail_w = int(rect.width())
        avail_h = int(rect.height())
        if avail_w <= 0 or avail_h <= 0:
            return QRect()

        if self._display_aspect_ratio is None:
            pix_w = max(1, int(self._source_pixmap.width()))
            pix_h = max(1, int(self._source_pixmap.height()))
            pix_ar = pix_w / float(pix_h)
            avail_ar = avail_w / float(avail_h)
            if avail_ar >= pix_ar:
                box_h = avail_h
                box_w = max(1, int(round(box_h * pix_ar)))
            else:
                box_w = avail_w
                box_h = max(1, int(round(box_w / pix_ar)))
        else:
            target_ar = float(self._display_aspect_ratio)
            avail_ar = avail_w / float(avail_h)
            if avail_ar >= target_ar:
                box_h = avail_h
                box_w = max(1, int(round(box_h * target_ar)))
            else:
                box_w = avail_w
                box_h = max(1, int(round(box_w / target_ar)))

        x0 = rect.x() + (avail_w - box_w) // 2
        y0 = rect.y() + (avail_h - box_h) // 2
        return QRect(x0, y0, box_w, box_h)

    def _map_widget_pos_to_image_px(self, pos):
        target_rect = self._target_rect()
        if target_rect.isNull():
            return None
        if not target_rect.contains(pos):
            return None

        pix_w = max(1, int(self._source_pixmap.width()))
        pix_h = max(1, int(self._source_pixmap.height()))
        u = (float(pos.x()) - float(target_rect.x())) / float(max(1, target_rect.width()))
        v = (float(pos.y()) - float(target_rect.y())) / float(max(1, target_rect.height()))
        u = min(max(u, 0.0), 1.0 - 1e-12)
        v = min(max(v, 0.0), 1.0 - 1e-12)
        px = int(u * pix_w)
        py = int(v * pix_h)
        px = min(max(px, 0), pix_w - 1)
        py = min(max(py, 0), pix_h - 1)
        return px, py

    def paintEvent(self, event):  # type: ignore[override]
        from PyQt5.QtGui import QPainter, QPen
        from PyQt5.QtCore import QRect

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self._source_pixmap is None or self._source_pixmap.isNull():
            return

        rect = self.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        if self._display_aspect_ratio is None:
            # Keep pixmap aspect ratio
            scaled = self._source_pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x0 = rect.x() + (rect.width() - scaled.width()) // 2
            y0 = rect.y() + (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x0, y0, scaled)
        else:
            # Letterbox to target ratio, then stretch content into that box.
            target_rect = self._target_rect()
            painter.drawPixmap(target_rect, self._source_pixmap)

        # ROI overlay
        roi_rect = None
        if self._roi_dragging and self._roi_start_pos is not None and self._roi_end_pos is not None:
            roi_rect = QRect(self._roi_start_pos, self._roi_end_pos).normalized()
        elif self._roi_last_rect is not None:
            roi_rect = self._roi_last_rect
        if roi_rect is not None and not roi_rect.isNull():
            painter.setRenderHint(QPainter.Antialiasing, False)
            pen = QPen(QColor(80, 180, 255), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(roi_rect)
            painter.fillRect(roi_rect, QColor(80, 180, 255, 40))

    def mousePressEvent(self, event):  # type: ignore[override]
        if not self._roi_enabled or event.button() != Qt.LeftButton:
            if self._roi_enabled and event.button() == Qt.RightButton:
                self._roi_last_rect = None
                self.roiResetRequested.emit()
                self.update()
                event.accept()
                return
            return super().mousePressEvent(event)
        self._roi_dragging = True
        self._roi_start_pos = event.pos()
        self._roi_end_pos = event.pos()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if not self._roi_enabled or not self._roi_dragging:
            return super().mouseMoveEvent(event)
        self._roi_end_pos = event.pos()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if not self._roi_enabled or not self._roi_dragging or event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)

        self._roi_dragging = False
        self._roi_end_pos = event.pos()
        sel_rect = QRect(self._roi_start_pos, self._roi_end_pos).normalized()
        self._roi_last_rect = sel_rect
        self.update()

        p0 = self._map_widget_pos_to_image_px(sel_rect.topLeft())
        p1 = self._map_widget_pos_to_image_px(sel_rect.bottomRight())
        if p0 is None or p1 is None:
            event.accept()
            return
        x0 = min(p0[0], p1[0])
        y0 = min(p0[1], p1[1])
        x1 = max(p0[0], p1[0]) + 1
        y1 = max(p0[1], p1[1]) + 1
        if (x1 - x0) >= 8 and (y1 - y0) >= 8:
            self.roiSelected.emit(int(x0), int(y0), int(x1), int(y1))
        event.accept()


class _ASDDropFilter(QObject):
    def __init__(self, simulator):
        super().__init__()
        self._simulator = simulator

    def eventFilter(self, obj, event):
        try:
            et = event.type()
            if et in (QEvent.DragEnter, QEvent.DragMove):
                md = event.mimeData()
                if md is not None and md.hasUrls():
                    for url in md.urls():
                        try:
                            path = url.toLocalFile()
                        except Exception:
                            path = ""
                        if path and path.lower().endswith(".asd"):
                            event.acceptProposedAction()
                            return True
                event.ignore()
                return True

            if et == QEvent.Drop:
                md = event.mimeData()
                if md is None or not md.hasUrls():
                    event.ignore()
                    return True
                for url in md.urls():
                    try:
                        path = url.toLocalFile()
                    except Exception:
                        path = ""
                    if path and path.lower().endswith(".asd"):
                        self._simulator.load_real_asd_file(path, sync=True)
                        event.acceptProposedAction()
                        return True
                event.ignore()
                return True
        except Exception:
            return False
        return False


APP_NAME = "pyNuD Simulator"
PLUGIN_NAME = "AFM Simulator"

HELP_HTML_EN = """
<h1>pyNuD Simulator</h1>
<h2>Overview</h2>
<p>pyNuD Simulator is a standalone application that generates simulated AFM images from molecular structure files. It is useful for comparing experimental AFM data with structural models.</p>
<h2>Importing Structure Files</h2>
<div class="feature-box">
    <h3>Supported Formats</h3>
    <ul>
        <li><strong>PDB:</strong> Standard PDB files (<code>.pdb</code>) are supported.</li>
        <li><strong>mmCIF:</strong> mmCIF format files (<code>.cif</code>, <code>.mmcif</code>) are supported.</li>
        <li><strong>MRC:</strong> MRC (Medical Research Council) volume data files (<code>.mrc</code>) are supported.</li>
    </ul>
</div>
<h2>File Import</h2>
<div class="step"><strong>Step 1:</strong> Click <strong>Import File...</strong> button.</div>
<div class="step"><strong>Step 2:</strong> Select a structure file (<code>.pdb</code>, <code>.cif</code>, <code>.mmcif</code>, or <code>.mrc</code>).</div>
<div class="step"><strong>Step 3:</strong> Confirm that the loaded file name is displayed in the simulator window.</div>
<div class="step">You can also drag and drop a file onto the file name line below the Import File button.</div>
<h2>Display style: Ribbon and secondary structure</h2>
<p>The AFM Simulator supports PyMOL-style ribbon visualization (Catmull-Rom spline interpolation) based on secondary structure detection. Select <strong>Ribbon (PyMOL-style)</strong> in the display style to show the protein backbone as a ribbon. You can also change the display style from the context menu by right-clicking on the molecule view.</p>
"""

HELP_HTML_JA = """
<h1>pyNuD Simulator</h1>
<h2>概要</h2>
<p>pyNuD Simulator はスタンドアロンアプリで、分子構造ファイルからシミュレートAFM像を生成します。実験AFMデータと構造モデルの比較に利用できます。</p>
<h2>構造ファイルのインポート</h2>
<div class="feature-box">
    <h3>対応形式</h3>
    <ul>
        <li><strong>PDB:</strong> 標準のPDBファイル（<code>.pdb</code>）に対応しています。</li>
        <li><strong>mmCIF:</strong> mmCIF形式ファイル（<code>.cif</code>、<code>.mmcif</code>）に対応しています。</li>
        <li><strong>MRC:</strong> MRC（Medical Research Council）ボリュームデータファイル（<code>.mrc</code>）に対応しています。</li>
    </ul>
</div>
<h2>ファイルインポート</h2>
<div class="step"><strong>Step 1:</strong> <strong>Import File...</strong> ボタンをクリック。</div>
<div class="step"><strong>Step 2:</strong> 構造ファイル（<code>.pdb</code>、<code>.cif</code>、<code>.mmcif</code>、または <code>.mrc</code>）を選択。</div>
<div class="step"><strong>Step 3:</strong> シミュレータウィンドウに読み込んだファイル名が表示されることを確認。</div>
<div class="step">Import File ボタン下のファイル名の行にドラッグ＆ドロップすることもできます。</div>
<h2>表示スタイル: リボンと二次構造</h2>
<p>AFMシミュレータでは二次構造の検出に基づき、PyMOL風のリボン可視化（Catmull-Romスプライン補間）が利用できます。表示スタイルで <strong>Ribbon (PyMOL-style)</strong> を選択すると、タンパク質の主鎖がリボンとして表示されます。分子表示上で右クリックするコンテキストメニューからも表示スタイルを変更できます。</p>
"""


def create_frequency_grid(image_shape, scan_x_nm, scan_y_nm):
    """
    実際のスキャンサイズを考慮した周波数グリッドを作成 (cycles/nm)
    """
    ny, nx = image_shape
    freq_x = np.fft.fftfreq(nx, d=scan_x_nm/nx)
    freq_y = np.fft.fftfreq(ny, d=scan_y_nm/ny)
    fx, fy = np.meshgrid(freq_x, freq_y)
    return np.sqrt(fx**2 + fy**2)

def apply_low_pass_filter(image, scan_x_nm, scan_y_nm, cutoff_wl_nm):
    """
    バターワース・ローパスフィルターを画像に適用する
    """
    if cutoff_wl_nm <= 0:
        return image
    
    # 周波数グリッド (cycles/nm)
    f_grid = create_frequency_grid(image.shape, scan_x_nm, scan_y_nm)
    
    # カットオフ周波数 (1/nm)
    f_cutoff = 1.0 / cutoff_wl_nm
    
    # バターワースフィルター (n=2)
    # H(f) = 1 / sqrt(1 + (f/f_cutoff)^(2n))
    h_f = 1.0 / np.sqrt(1.0 + (f_grid / f_cutoff)**4)
    
    # FFT
    img_fft = fft2(image)
    filtered_fft = img_fft * h_f
    
    # 逆FFT
    filtered_image = np.real(ifft2(filtered_fft))
    return filtered_image
    
@jit(nopython=True)
def _calculate_dilation_row(r_out, sample_surface, tip_footprint):
    """
    形態学的ダイレーションの1行分だけを計算するNumba高速化関数。
    """
    s_rows, s_cols = sample_surface.shape
    t_rows, t_cols = tip_footprint.shape
    t_center_r, t_center_c = t_rows // 2, t_cols // 2

    output_row = np.full(s_cols, -1e9, dtype=np.float64)

    for c_out in range(s_cols):
        max_h = -1e9
        for r_tip in range(t_rows):
            for c_tip in range(t_cols):
                s_r = r_out + r_tip - t_center_r
                s_c = c_out + c_tip - t_center_c

                if 0 <= s_r < s_rows and 0 <= s_c < s_cols:
                    h = sample_surface[s_r, s_c] - tip_footprint[r_tip, c_tip]
                    if h > max_h:
                        max_h = h
        output_row[c_out] = max_h

    return output_row


#@jit(nopython=True)
def _create_vdw_surface_loop(nx, ny, pixel_x, pixel_y, x_start, y_start, min_z, atom_coords, atom_radii):
    """
    原子のファンデルワールス半径を考慮して表面マップを生成するNumba高速化関数。
    """
    surface_map = np.full((ny, nx), min_z - 5.0, dtype=np.float64)
    px_coords = x_start + (np.arange(nx) + 0.5) * pixel_x
    py_coords = y_start + (np.arange(ny) + 0.5) * pixel_y
    
    for i in range(len(atom_coords)):
        ax, ay, az = atom_coords[i]
        az -= min_z
        r = atom_radii[i]
        r_sq = r**2

        ix_min = int(np.floor((ax - r - x_start) / pixel_x))
        ix_max = int(np.ceil((ax + r - x_start) / pixel_x))
        iy_min = int(np.floor((ay - r - y_start) / pixel_y))
        iy_max = int(np.ceil((ay + r - y_start) / pixel_y))
        
        ix_min, ix_max = max(0, ix_min), min(nx, ix_max)
        iy_min, iy_max = max(0, iy_min), min(ny, iy_max)

        for iy in range(iy_min, iy_max):
            py = py_coords[iy]
            dy_sq = (py - ay)**2
            if dy_sq > r_sq: continue
            
            for ix in range(ix_min, ix_max):
                px = px_coords[ix]
                d_sq = (px - ax)**2 + dy_sq
                
                if d_sq <= r_sq:
                    h = az + np.sqrt(r_sq - d_sq)
                    if h > surface_map[iy, ix]:
                        surface_map[iy, ix] = h
                        
    surface_map[surface_map < min_z - 4.0] = 0.0
    return surface_map

DETAILED_MANUAL_MD_JA = """
# pyNuD Simulator Manual (Detailed)

このマニュアルは `pyNuD_simulator.py` の現行実装に基づき、主要機能・パラメータ・操作手順をできるだけ詳細にまとめたものです。  
UI表示名は実際のラベル（英語）をそのまま記載しています。

## 1. アプリの目的 {#ja-sec1}

`pyNuD Simulator` は、PDB/CIF/MRC 構造から AFM 像をシミュレーションし、実験の ASD データ（Real AFM）と比較・整合するための GUI アプリです。

主な用途:
- 3D構造の表示（VTK / PyMOL）
- AFM探針モデルを使った高さマップ生成
- XY / YZ / ZX 像の比較
- 実AFMデータとのサイズ同期・姿勢推定（Estimate Pose）
- 実AFMに見た目を寄せる外観最適化（Auto-fit Appearance）

---

## 2. 画面構成 {#ja-sec2}

### 2.1 左パネル（Control Panel） {#ja-sec2-1}
- File Import
- Density Threshold（MRC時）
- Display Settings
- Color & Lighting Settings
- AFM Tip Settings
- Tip Position Control
- AFM Simulation
- View Control
- AFM Appearance（独立ウィンドウ）

### 2.2 右パネル（Structure + AFM） {#ja-sec2-2}
- 上段: 構造表示（PyMOLビュー / VTKビュー）
- 中段: Structure & View Control（回転、Find Initial Plane など）
- 下段: Simulated AFM Images（XY/YZ/ZX）

### 2.3 独立ウィンドウ {#ja-sec2-3}
- `Real AFM (ASD) / Sim Aligned`
- `AFM Appearance`
- `Help`

メインウィンドウを閉じると、上記サブウィンドウも連動して閉じます。

---

## 3. メニュー {#ja-sec3}

- `AFM Appearance`: AFM外観パラメータ（ノイズ・低域フィルタ）を開く
- `Real AFM image`: Real AFM / Sim Aligned 比較ウィンドウを開く
- `Help` → `View Help...` (`F1`): ヘルプウィンドウを開く

`Help` メニューは `View Help...` の1項目に統合されています（`Manual` は廃止）。

---

## 4. File Import {#ja-sec4}

### 4.1 読み込み対応 {#ja-sec4-1}
- PDB: `.pdb`
- mmCIF: `.cif`, `.mmcif`
- MRC: `.mrc`
- Real AFM: `.asd`（専用ウィンドウ側で読み込み・ドラッグ&ドロップ）

### 4.2 読み込み後の基本挙動 {#ja-sec4-2}
- 回転UIが有効化
- 構造表示更新
- Interactive Update がONなら、回転や条件変更に応じて自動再計算

---

## 5. 左パネル詳細 {#ja-sec5}

### 5.1 Density Threshold（MRC専用） {#ja-sec5-1}
- `Value`（スライダ）: 等値面しきい値
- `Flip Z-axis`（デフォルト ON）: MRCのZ向きを反転して表示を合わせる

### 5.2 Display Settings {#ja-sec5-2}
- `Style`: Ball & Stick / Stick Only / Spheres / Points / Wireframe / Simple Cartoon / Ribbon
- `Color`: By Element / By Chain / Single Color / By B-Factor
- `Show`: All Atoms / Heavy Atoms / Backbone / C / N / O
- `Size`: 10–200（表示スケール）
- `Opacity`: 10–100%
- `Quality`: Fast / Good / High
- `Renderer`: PyMOL (image) / VTK (interactive)（環境依存で利用可否あり）
- `Electrostatics (ESP)`: 表面電荷可視化（対象構造・レンダラ条件あり）

### 5.3 Color & Lighting Settings {#ja-sec5-3}
- `Background`: 背景色
- `Brightness`: 20–200%（デフォルト 100）
- `Single Color`: 単色モード色
- `Ambient`: 0–50%（デフォルト 10）
- `Specular`: 0–100%（デフォルト 60）
- `PyMOL Style`: 見た目プリセット
- `Dark Theme`: ダーク背景系プリセット

### 5.4 AFM Tip Settings {#ja-sec5-4}
- `Shape`: Cone / Sphere / Paraboloid
- `Radius (nm)`: 0.5–30.0（デフォルト 0.5）
- `Radius of Minitip (nm)`: 0.1–10.0（Sphere使用時）
- `Angle (deg)`: 5–35（Cone/Sphere向け）
- `Tip Info`: 現在の形状情報表示

### 5.5 Tip Position Control {#ja-sec5-5}
- `X (nm)`: -50〜50（スライダ）
- `Y (nm)`: -50〜50
- `Z (nm)`: 10〜100（内部表示は5.0nm相当初期）

### 5.6 AFM Simulation {#ja-sec5-6}
- `Scan Size X (nm)`: 1.0–500.0（デフォルト 20.0）
- `Scan Size Y (nm)`: 1.0–500.0（デフォルト 20.0）
- `Pixels X (Nx)`: 8–2048（デフォルト 64）
- `Pixels Y (Ny)`: 8–2048（デフォルト 64）
- `Quick Res`: 32x32 / 64x64 / 128x128 / 256x256
- `Rctangle (force Y = X)`（デフォルト ON）:
  - ON時は `Scan Y = Scan X`, `Ny = Nx` を強制
  - OFFで長方形スキャンを許可
- `Sync Sim Params to Real AFM`: Real AFMのサイズ・解像度を反映
- `Interactive Update`（デフォルト ON）:
  - 回転中は軽量更新（低解像度）
  - 回転停止後に高解像度更新
  - YZ/ZXが有効なら停止後にそれらも更新
- `Consider atom size (vdW)`: 原子を点ではなく半径付き球で扱う
- `Run AFM Simulation`: 現在条件でシミュレーション実行

### 5.7 AFM Appearance（独立ウィンドウ） {#ja-sec5-7}

### Low-pass Filter
- `Apply Low-pass Filter`
- `Cutoff Wavelength (nm)`（0.1–20.0、デフォルト 2.0）

### Physical Noise / Artifacts
- `Enable Physical Noise`
- `Use fixed seed`
- `Seed`
- `Height Noise` / `sigma (nm)`
- `Line Noise` / `sigma_line (nm)` / `mode` (`offset` / `rw`)
- `Drift` / `vx (nm/line)` / `vy (nm/line)` / `jitter (nm/line)`
- `Feedback Lag`
  - `mode`: `linear_lag` / `tapping_parachute`
  - `Scan Direction`: Left -> Right（`L2R`）/ Right -> Left（`R2L`）
  - `tau (lines)`
  - `tap drop (nm)`
  - `tap tau_track (lines)`
  - `tap tau_para (lines)`
  - `tap release (nm)`

---

## 6. 右パネル詳細（Structure & View Control） {#ja-sec6}

### 6.1 Rotation {#ja-sec6-1}
- `Rotation X/Y/Z`:
  - Spin: -180.0〜180.0°
  - Slider: -1800〜1800（0.1°刻み）
- `CTRL + Drag`（構造回転）
- `Shift + Drag`（パン）

### 6.2 ボタン {#ja-sec6-2}
- `Reset All`: 回転・Tip位置・カメラを初期化
- `Save 3D View...`: 3Dビュー画像保存
- `Find Initial Plane`: 初期支持面推定
- `Save Params...` / `Load Params...`: 設定JSONの保存/読み込み
- `XY` / `YZ` / `ZX`: 標準視点へ切替

### 6.3 Atom Statistics {#ja-sec6-3}
- Total, C, O, N, H, Other を表示

### 6.4 Find Initial Plane Params {#ja-sec6-4}
幾何・静電・探索速度のバランスを制御します。主要項目:
| パラメータ | 意味 |
| --- | --- |
| `Use electrostatics (residue charge)` | 静電項を使うか |
| `pH` | 残基の電荷推定 |
| `substrate` | `MICA` / `APTES_MICA` |
| `salt`, `salt [mM]` | Debye遮蔽関連 |
| `alpha` | 静電重み |
| `r_pseudo [A]` | 擬似原子半径 |
| `delta_elec [A]` | 電荷評価殻厚 |
| `K_geom` | 幾何上位候補数 |
| `N_dir`, `K` | 球面探索密度・局所候補数 |
| `delta [A]`, `lambda` | 接触評価の閾値・重み |
| `theta1/2/3 [deg]` | 局所探索角ステップ |
| `local grid` | 局所格子半径 |
| `surf r [A]`, `surf n` | 表面原子抽出条件 |

---

## 7. Simulated AFM Images パネル {#ja-sec7}

- 表示チェック: `XY`, `YZ`, `ZX`
- 少なくとも1つは常時ON
- `Save as ASD...`: シミュレーション像をASD保存
- `Save Image...`: PNG/TIFF保存（選択ビューや回転増分指定を含む）

補足:
- XY表示が選択されているとき、VTK上のTipは透明度が下がります（見やすさ向上）。

---

## 8. Real AFM (ASD) / Sim Aligned ウィンドウ {#ja-sec8}

### 8.1 起動 {#ja-sec8-1}
- メニュー `Real AFM image` から開く

### 8.2 レイアウト {#ja-sec8-2}
- 左: `Real AFM (ASD)`
  - `Frame: i / N` とフレームスライダー
  - ASD多フレームを切替
- 右: `Sim Aligned`
  - `Scan X/Y` と `Pixel X/Y` の情報表示

左右の画像描画領域は同サイズになるように揃えられます。

### 8.3 ASD読み込み {#ja-sec8-3}
- Real AFM側パネルへ `.asd` をドラッグ&ドロップ
- または内部ロード処理から選択
- 読み込み時:
  - フレーム情報更新
  - メタ情報（Scan size / pixel）更新
  - 必要に応じシミュレーション条件同期

### 8.4 ROI（Real AFM切り出し） {#ja-sec8-4}
- 左ドラッグ: ROI選択
- 右クリック: ROI解除（全体に戻す）
- ROI適用時:
  - Real AFMの有効領域が切り替わる
  - scan size / resolution メタ情報もROI基準に再計算
  - 既存の整合像（Sim Aligned）はクリアされ、再推定が必要

### 8.5 ボタン {#ja-sec8-5}
- `Get Simulated image`
  - Real AFMの `Scan X/Y` と `Nx/Ny` に合わせて、PDBから再シミュレーション
  - 結果を `Sim Aligned` に表示
  - PDB未ロード時は警告
  - `Rctangle` がONでも、ASDが長方形なら自動でOFFに切替
- `Estimate Pose`
  - Rotation XYZを変えながら反復シミュレーションで最適方位探索
  - 推定後、Rotation XYZへ反映
  - Sim Aligned とメイン `Simulated AFM Images` のXY像が一致する運用
- `Auto-fit Appearance`
  - ノイズ/アーティファクト条件を探索してReal AFMへ見た目を寄せる
  - 反映結果は Sim Aligned 側にも適用

---

## 9. Estimate Pose の仕様 {#ja-sec9}

### 9.1 前提 {#ja-sec9-1}
- Real AFM 読み込み済み
- PDB/CIF 構造読み込み済み
- 他のシミュレーションワーカーが競合していないこと

### 9.2 精度選択 {#ja-sec9-2}
`Estimate Pose` 押下時に精度を選択:
- `Low`
- `Medium`
- `High`

精度が高いほど探索点が増え、時間は長くなります。

### 9.3 探索の流れ（実装） {#ja-sec9-3}
1. 粗探索（seed姿勢 + Z方向グリッド）
2. XYZ方向の座標降下で段階的に細密化
3. 最良姿勢で再シミュレーション
4. 残差平行移動（Dx, Dy）とスコア算出

### 9.4 進捗表示 {#ja-sec9-4}
- 推定中は `QProgressDialog` を表示
- `Cancel` 可能
- 評価回数・ベストスコアを更新表示

---

## 10. Auto-fit Appearance の仕様 {#ja-sec10}

目的: 実AFM像に近い外観ノイズ条件を自動探索。

主な特徴:
- Real AFM と Simulated AFM が必要
- 低域フィルタの `Cutoff Wavelength` は固定（Auto-fitで変更しない）
- `Scan Direction` は内部的に `L2R` 固定で評価
- 探索候補:
  - Height noise
  - Line noise
  - Drift
  - Feedback mode（linear_lag / tapping_parachute 系）
- 最良候補を AFM Appearance UI に反映して再描画

---

## 11. Interactive Update の仕様 {#ja-sec11}

ON時:
- 回転・条件変更中:
  - 低解像度（64x64）で高速更新（主にXY）
- 操作停止後:
  - 1秒後に高解像度再計算
  - YZ/ZXチェック時はそれらも更新対象

OFF時:
- `Run AFM Simulation` など明示操作で更新

---

## 12. 保存機能 {#ja-sec12}

- `Save as ASD...`:
  - AFMシミュレーション結果をASDとして保存
- `Save Image...`:
  - AFM画像をPNG/TIFF保存
  - ビュー選択・回転増分適用付きのエクスポートダイアログあり
- `Save 3D View...`:
  - 構造表示領域を画像保存
- `Save Params...` / `Load Params...`:
  - 多数の設定をJSONで保存/復元

---

## 13. 操作一覧（マウス/キー） {#ja-sec13}

- 構造ビュー:
  - `CTRL + 左ドラッグ`: 回転
  - `Shift + 左ドラッグ`: パン
- Real AFMビュー:
  - 左ドラッグ: ROI選択
  - 右クリック: ROI解除
- ドラッグ&ドロップ:
  - 構造表示エリア: PDB/CIF/MRC
  - Real AFMパネル: ASD

---

## 14. 推奨ワークフロー（実AFM整合） {#ja-sec14}

1. PDB/CIFを読み込む  
2. `Real AFM image` ウィンドウを開く  
3. ASDを読み込む（必要ならフレーム選択）  
4. 必要に応じROIで対象分子を中心化  
5. `Get Simulated image` でスキャン条件一致シミュレーション  
6. `Estimate Pose` で方位推定（精度選択）  
7. `Auto-fit Appearance` でノイズ外観を調整  
8. メイン下段 `Simulated AFM Images` と `Sim Aligned` が一致することを確認  
9. ASD/画像を保存

---

## 15. トラブルシューティング {#ja-sec15}

- `PDB is not loaded`:
  - 先に構造ファイルを読み込み
- `Real AFM metadata is incomplete`:
  - ASDのメタ情報不足。別フレーム/別ファイルで確認
- `Another simulation is running. Please wait.`:
  - 既存ワーカー完了待ち。連打を避ける
- Estimate Pose結果が悪い:
  - ROIで対象を中心化
  - フレームを変える
  - `High` 精度を選ぶ
  - `Get Simulated image` を先に実行して条件一致を確認
- 画像が縦横で合わない:
  - `Rctangle` の状態と Real AFM の実サイズ比を確認

---

## 16. 依存モジュールに関する注意 {#ja-sec16}

- ASD読み込み/表示に `cv2` は必須ではありません。
- 実装は `asd_io.py` のローダーを利用しており、ASD読み込みはこれで完結します。

---

## 17. 変更履歴メモ（現行実装で重要な挙動） {#ja-sec17}

- `Help` メニューは `View Help...` のみ
- Real AFM と Sim Aligned は専用ウィンドウで管理
- Estimate Pose は画像回転比較だけでなく、Rotation XYZを変えて再シミュレーションする方式
- Auto-fit Appearance は Sim Aligned 側にも反映
- Interactive Update はデフォルト ON
- `Rctangle` はデフォルト ON（必要に応じ自動解除）
"""

DETAILED_MANUAL_MD_EN = """
# pyNuD Simulator Detailed Manual

This manual describes the current implementation in `pyNuD_simulator.py` in detail, including
features, parameter meanings, and operational workflow.  
UI labels are written as displayed in the application.

## 1. Purpose {#en-sec1}

`pyNuD Simulator` is a GUI application to simulate AFM images from PDB/CIF/MRC structures and
compare/align them with experimental ASD data (Real AFM).

Main use cases:
- 3D structure visualization (VTK / PyMOL)
- AFM tip-based height map simulation
- XY / YZ / ZX simulated view comparison
- Real AFM size synchronization and pose estimation
- Appearance fitting to real AFM data

---

## 2. Layout {#en-sec2}

### 2.1 Left Panel (Control Panel) {#en-sec2-1}
- File Import
- Density Threshold (MRC)
- Display Settings
- Color & Lighting Settings
- AFM Tip Settings
- Tip Position Control
- AFM Simulation
- View Control
- AFM Appearance (separate window)

### 2.2 Right Panel (Structure + AFM) {#en-sec2-2}
- Top: structure view (PyMOL / VTK)
- Middle: Structure & View Control (rotation, Find Initial Plane, etc.)
- Bottom: Simulated AFM Images (XY/YZ/ZX)

### 2.3 Separate Windows {#en-sec2-3}
- `Real AFM (ASD) / Sim Aligned`
- `AFM Appearance`
- `Help`

When the main window closes, these child windows close automatically.

---

## 3. Menu {#en-sec3}

- `AFM Appearance`: open appearance controls (filter/noise/artifacts)
- `Real AFM image`: open Real AFM / Sim Aligned window
- `Help` → `View Help...` (`F1`): open built-in detailed manual

`Help` is unified into one entry (`View Help...`).

---

## 4. File Import {#en-sec4}

### 4.1 Supported Files {#en-sec4-1}
- PDB: `.pdb`
- mmCIF: `.cif`, `.mmcif`
- MRC: `.mrc`
- Real AFM: `.asd` (loaded in Real AFM window via dialog or drag-and-drop)

### 4.2 Behavior After Load {#en-sec4-2}
- Rotation controls become active
- Structure view updates
- If Interactive Update is ON, simulation updates automatically as parameters change

---

## 5. Left Panel Details {#en-sec5}

### 5.1 Density Threshold (MRC only) {#en-sec5-1}
- `Value` slider: isosurface threshold
- `Flip Z-axis` (default ON): flips MRC Z orientation for expected display

### 5.2 Display Settings {#en-sec5-2}
- `Style`: Ball & Stick / Stick Only / Spheres / Points / Wireframe / Simple Cartoon / Ribbon
- `Color`: By Element / By Chain / Single Color / By B-Factor
- `Show`: All Atoms / Heavy Atoms / Backbone / C / N / O
- `Size`: 10-200 (display scale)
- `Opacity`: 10-100%
- `Quality`: Fast / Good / High
- `Renderer`: PyMOL (image) / VTK (interactive)
- `Electrostatics (ESP)`: enable electrostatic surface rendering

### 5.3 Color & Lighting Settings {#en-sec5-3}
- `Background`
- `Brightness`: 20-200% (default 100)
- `Single Color`
- `Ambient`: 0-50% (default 10)
- `Specular`: 0-100% (default 60)
- `PyMOL Style` preset
- `Dark Theme` preset

### 5.4 AFM Tip Settings {#en-sec5-4}
- `Shape`: Cone / Sphere / Paraboloid
- `Radius (nm)`: 0.5-30.0 (default 0.5)
- `Radius of Minitip (nm)`: 0.1-10.0 (Sphere mode)
- `Angle (deg)`: 5-35 (Cone/Sphere)
- `Tip Info`: current tip geometry summary

### 5.5 Tip Position Control {#en-sec5-5}
- `X (nm)`: -50 to 50
- `Y (nm)`: -50 to 50
- `Z (nm)`: 10 to 100

### 5.6 AFM Simulation {#en-sec5-6}
- `Scan Size X (nm)`: 1.0-500.0 (default 20.0)
- `Scan Size Y (nm)`: 1.0-500.0 (default 20.0)
- `Pixels X (Nx)`: 8-2048 (default 64)
- `Pixels Y (Ny)`: 8-2048 (default 64)
- `Quick Res`: 32x32 / 64x64 / 128x128 / 256x256
- `Rctangle (force Y = X)` (default ON):
  - Forces `Scan Y = Scan X` and `Ny = Nx`
  - Disable for rectangular scan settings
- `Sync Sim Params to Real AFM`: copy size/resolution from current Real AFM metadata
- `Interactive Update` (default ON):
  - low-resolution fast updates while interacting
  - high-resolution follow-up after interaction stops
  - YZ/ZX also refresh if enabled
- `Consider atom size (vdW)`
- `Run AFM Simulation`

### 5.7 AFM Appearance (Separate Window) {#en-sec5-7}

### Low-pass Filter
- `Apply Low-pass Filter`
- `Cutoff Wavelength (nm)` (0.1-20.0, default 2.0)

### Physical Noise / Artifacts
- `Enable Physical Noise`
- `Use fixed seed`
- `Seed`
- `Height Noise` / `sigma (nm)`
- `Line Noise` / `sigma_line (nm)` / `mode` (`offset` / `rw`)
- `Drift` / `vx (nm/line)` / `vy (nm/line)` / `jitter (nm/line)`
- `Feedback Lag`
  - `mode`: `linear_lag` / `tapping_parachute`
  - `Scan Direction`: `L2R` / `R2L`
  - `tau (lines)`
  - `tap drop (nm)`
  - `tap tau_track (lines)`
  - `tap tau_para (lines)`
  - `tap release (nm)`

---

## 6. Structure & View Control {#en-sec6}

### 6.1 Rotation {#en-sec6-1}
- `Rotation X/Y/Z`:
  - spin: -180.0 to 180.0 deg
  - slider: 0.1 deg step mapped to -1800 to 1800
- `CTRL + Drag`: rotate structure
- `Shift + Drag`: pan

### 6.2 Buttons {#en-sec6-2}
- `Reset All`
- `Save 3D View...`
- `Find Initial Plane`
- `Save Params...` / `Load Params...`
- `XY` / `YZ` / `ZX` standard view shortcuts

### 6.3 Atom Statistics {#en-sec6-3}
- Displays: Total, C, O, N, H, Other

### 6.4 Find Initial Plane Params (key meaning) {#en-sec6-4}
| Parameter | Meaning |
| --- | --- |
| `Use electrostatics (residue charge)` | Enable/disable electrostatic term |
| `pH` | Residue charge estimation condition |
| `substrate` | `MICA` / `APTES_MICA` |
| `salt`, `salt [mM]` | Debye screening condition |
| `alpha` | Electrostatics weight |
| `r_pseudo [A]` | Pseudo-atom radius |
| `delta_elec [A]` | Electrostatic shell thickness |
| `K_geom` | Number of top geometry candidates |
| `N_dir`, `K` | Sphere sampling density / local candidate count |
| `delta [A]`, `lambda` | Contact threshold / spread weight |
| `theta1/2/3 [deg]` | Local search angle steps |
| `local grid` | Local neighborhood radius |
| `surf r [A]`, `surf n` | Surface atom extraction condition |

---

## 7. Simulated AFM Images Panel {#en-sec7}

- View toggles: `XY`, `YZ`, `ZX` (at least one must stay enabled)
- `Save as ASD...`: export simulated AFM as ASD
- `Save Image...`: export PNG/TIFF

Note:
- In XY view, AFM tip is rendered with lower opacity for visibility.

---

## 8. Real AFM (ASD) / Sim Aligned Window {#en-sec8}

### 8.1 How to Open {#en-sec8-1}
- From menu: `Real AFM image`

### 8.2 Layout {#en-sec8-2}
- Left: `Real AFM (ASD)` with frame label/slider
- Right: `Sim Aligned` with scan/pixel metadata row

Image drawing areas are kept aligned in size.

### 8.3 ASD Loading {#en-sec8-3}
- Drag-and-drop `.asd` onto Real AFM panel, or load via dialog
- On load:
  - frame controls update
  - scan/pixel metadata updates
  - optional simulation parameter sync

### 8.4 ROI on Real AFM {#en-sec8-4}
- Left drag: select ROI
- Right click: clear ROI
- When ROI is applied:
  - active Real AFM area changes
  - scan size and resolution metadata are recomputed from ROI
  - previous alignment preview is cleared

### 8.5 Buttons {#en-sec8-5}
- `Get Simulated image`
  - simulates XY from current structure using Real AFM scan size/resolution
  - displays result on `Sim Aligned`
  - warns if PDB is missing
  - auto-disables rectangle lock when ASD X/Y differ
- `Estimate Pose`
  - performs iterative rotation search by re-simulating (Rotation XYZ optimization)
  - applies estimated rotation to controls
  - keeps Sim Aligned and main XY simulation consistent
- `Auto-fit Appearance`
  - searches noise/artifact parameters to visually match Real AFM
  - applies result to aligned preview as well

---

## 9. Estimate Pose Behavior {#en-sec9}

### 9.1 Prerequisites {#en-sec9-1}
- Real AFM loaded
- PDB/CIF structure loaded
- no conflicting running simulation worker

### 9.2 Precision {#en-sec9-2}
- `Low`
- `Medium`
- `High`

Higher precision increases search evaluations and runtime.

### 9.3 Search Flow {#en-sec9-3}
1. coarse seeding + Z scan
2. coordinate-descent refinement in XYZ
3. final simulation at best orientation
4. residual translation and final score calculation

### 9.4 Progress UI {#en-sec9-4}
- Uses `QProgressDialog`
- Supports cancel
- Displays evaluation count and best score during search

---

## 10. Auto-fit Appearance Behavior {#en-sec10}

Purpose: fit simulated appearance to Real AFM look.

Key points:
- requires Real AFM and simulated XY image
- keeps `Cutoff Wavelength` fixed (no auto-change of low-pass cutoff)
- uses fixed `Scan Direction = L2R` during search
- explores candidates over:
  - height noise
  - line noise
  - drift
  - feedback mode parameters
- writes best candidate back to AFM Appearance controls

---

## 11. Interactive Update Behavior {#en-sec11}

When ON:
- during interaction: low-resolution fast updates (mainly XY)
- after interaction stops: delayed high-resolution update
- YZ/ZX are also updated if enabled

When OFF:
- updates occur only on explicit simulation actions

---

## 12. Save/Export {#en-sec12}

- `Save as ASD...`: simulated data export
- `Save Image...`: image export with view selection/incremental rotation options
- `Save 3D View...`: structure view snapshot
- `Save Params...` / `Load Params...`: JSON parameter persistence

---

## 13. Mouse/Key Operations {#en-sec13}

- Structure view:
  - `CTRL + Left Drag`: rotate
  - `Shift + Left Drag`: pan
- Real AFM view:
  - left drag: ROI select
  - right click: ROI reset
- Drag-and-drop:
  - structure area: PDB/CIF/MRC
  - Real AFM panel: ASD

---

## 14. Recommended Real-AFM Alignment Workflow {#en-sec14}

1. Load PDB/CIF  
2. Open `Real AFM image` window  
3. Load ASD and select frame  
4. Optionally define ROI around target molecule  
5. Run `Get Simulated image`  
6. Run `Estimate Pose` with selected precision  
7. Run `Auto-fit Appearance`  
8. Confirm consistency between main XY and Sim Aligned  
9. Save ASD/image outputs

---

## 15. Troubleshooting {#en-sec15}

- `PDB is not loaded`:
  - load structure first
- `Real AFM metadata is incomplete`:
  - check ASD file/frame metadata
- `Another simulation is running. Please wait.`:
  - wait for current worker completion
- poor Estimate Pose result:
  - center target with ROI
  - try another frame
  - use `High` precision
  - run `Get Simulated image` first to ensure matching scan settings
- aspect mismatch:
  - verify rectangle lock and Real AFM X/Y ratio

---

## 16. Dependency Note {#en-sec16}

- `cv2` is not required for ASD loading/display.
- ASD loading is handled through `asd_io.py`.

---

## 17. Current Important Behavior Summary {#en-sec17}

- `Help` menu has only `View Help...`
- Real AFM and aligned preview are managed in a separate window
- Estimate Pose optimizes Rotation XYZ with re-simulation
- Auto-fit Appearance is applied to aligned preview as well
- Interactive Update default is ON
- Rectangle lock default is ON and can auto-disable when needed
"""

class HelpContentManager:
    """
    Manages all help content, supporting multiple languages.
    """
    def __init__(self):
        self._initialize_content()

    def set_language(self, lang_code):
        if lang_code in self.content:
            self.current_language = lang_code

    def get_toc_structure(self):
        return self.content[self.current_language]['toc_structure']

    def get_content(self, page_id):
        if page_id != "detailed_manual":
            page_id = "detailed_manual"
        return self._wrap_content(self._embedded_detailed_manual_page())
            
    def get_ui_text(self, key):
        return self.content[self.current_language]['ui_text'].get(key, '')

    def _embedded_detailed_manual_page(self):
        md_text = DETAILED_MANUAL_MD_JA if self.current_language == 'ja' else DETAILED_MANUAL_MD_EN
        body_html = self._markdown_to_html(md_text)
        if self.current_language == 'ja':
            return f"""
            <h1>pyNuD Simulator 詳細マニュアル</h1>
            {body_html}
            """
        return f"<h1>pyNuD Simulator Detailed Manual</h1>{body_html}"

    def _build_detailed_manual_toc_items(self, lang_code):
        """
        Build a left-side TOC from the embedded Markdown headings (## / ###).
        Anchor ids must match those generated by _markdown_to_html().
        """
        md_text = DETAILED_MANUAL_MD_JA if lang_code == 'ja' else DETAILED_MANUAL_MD_EN
        used_ids = {}
        items_h2 = []
        current_h2 = None

        for raw_line in md_text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("### ") and current_h2 is not None:
                heading_text, anchor_id = self._extract_heading_text_and_id(stripped[4:], used_ids)
                current_h2["children"].append((heading_text, f"detailed_manual#{anchor_id}"))
                continue
            if stripped.startswith("## "):
                heading_text, anchor_id = self._extract_heading_text_and_id(stripped[3:], used_ids)
                current_h2 = {"name": heading_text, "id": f"detailed_manual#{anchor_id}", "children": []}
                items_h2.append(current_h2)
                continue
            if stripped.startswith("# "):
                # Keep id generation consistent with _markdown_to_html().
                self._extract_heading_text_and_id(stripped[2:], used_ids)
                continue

        out = []
        for item in items_h2:
            if item["children"]:
                out.append((item["name"], item["id"], item["children"]))
            else:
                out.append((item["name"], item["id"]))
        return out

    def _format_inline_text_fragment(self, text):
        esc = html.escape(text)
        esc = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
            esc,
        )
        bold_parts = esc.split('**')
        if len(bold_parts) <= 1:
            return esc
        out = []
        for idx, part in enumerate(bold_parts):
            if idx % 2 == 1:
                out.append(f"<strong>{part}</strong>")
            else:
                out.append(part)
        return ''.join(out)

    def _format_inline_markdown(self, text):
        chunks = text.split('`')
        formatted = []
        for idx, part in enumerate(chunks):
            if idx % 2 == 1:
                formatted.append(f"<code>{html.escape(part)}</code>")
            else:
                formatted.append(self._format_inline_text_fragment(part))
        return ''.join(formatted)

    def _extract_heading_text_and_id(self, heading_text, used_ids):
        match = re.search(r'\s\{#([A-Za-z0-9_-]+)\}\s*$', heading_text)
        if match:
            return heading_text[:match.start()].rstrip(), match.group(1)

        base = re.sub(r'[`*]', '', heading_text)
        base = re.sub(r'[^\w\- ]+', '', base, flags=re.UNICODE).strip().lower()
        base = re.sub(r'\s+', '-', base)
        if not base:
            base = f"sec-{len(used_ids) + 1}"
        count = used_ids.get(base, 0) + 1
        used_ids[base] = count
        if count > 1:
            base = f"{base}-{count}"
        return heading_text, base

    def _markdown_to_html(self, markdown_text):
        lines = markdown_text.splitlines()
        html_lines = []
        in_list = False
        used_ids = {}

        def close_list():
            nonlocal in_list
            if in_list:
                html_lines.append("</ul>")
                in_list = False

        def _split_table_row(row_text):
            return [cell.strip() for cell in row_text.strip().strip('|').split('|')]

        def _is_table_separator(row_text):
            s = row_text.strip()
            if not s or '|' not in s:
                return False
            parts = _split_table_row(s)
            if not parts:
                return False
            return all(re.fullmatch(r':?-{3,}:?', part) for part in parts)

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            stripped = line.strip()

            if not stripped:
                close_list()
                i += 1
                continue
            if stripped == "---":
                close_list()
                html_lines.append("<hr>")
                i += 1
                continue
            if stripped.startswith("### "):
                close_list()
                heading_text, anchor_id = self._extract_heading_text_and_id(stripped[4:], used_ids)
                html_lines.append(f'<h3 id="{html.escape(anchor_id, quote=True)}">{self._format_inline_markdown(heading_text)}</h3>')
                i += 1
                continue
            if stripped.startswith("## "):
                close_list()
                heading_text, anchor_id = self._extract_heading_text_and_id(stripped[3:], used_ids)
                html_lines.append(f'<h2 id="{html.escape(anchor_id, quote=True)}">{self._format_inline_markdown(heading_text)}</h2>')
                i += 1
                continue
            if stripped.startswith("# "):
                close_list()
                heading_text, anchor_id = self._extract_heading_text_and_id(stripped[2:], used_ids)
                html_lines.append(f'<h1 id="{html.escape(anchor_id, quote=True)}">{self._format_inline_markdown(heading_text)}</h1>')
                i += 1
                continue
            if stripped.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{self._format_inline_markdown(stripped[2:])}</li>")
                i += 1
                continue

            if '|' in stripped and (i + 1) < len(lines):
                next_line = lines[i + 1].strip()
                if _is_table_separator(next_line):
                    close_list()
                    headers = _split_table_row(stripped)
                    html_lines.append('<table class="doc-table">')
                    html_lines.append('<thead><tr>')
                    for head in headers:
                        html_lines.append(f"<th>{self._format_inline_markdown(head)}</th>")
                    html_lines.append('</tr></thead>')
                    html_lines.append('<tbody>')
                    i += 2
                    while i < len(lines):
                        row = lines[i].strip()
                        if (not row or row.startswith('#') or row.startswith('- ') or row == '---' or '|' not in row):
                            break
                        cells = _split_table_row(row)
                        if len(cells) < len(headers):
                            cells.extend([''] * (len(headers) - len(cells)))
                        elif len(cells) > len(headers):
                            cells = cells[:len(headers)]
                        html_lines.append('<tr>')
                        for cell in cells:
                            html_lines.append(f"<td>{self._format_inline_markdown(cell)}</td>")
                        html_lines.append('</tr>')
                        i += 1
                    html_lines.append('</tbody></table>')
                    continue

            close_list()
            html_lines.append(f"<p>{self._format_inline_markdown(stripped)}</p>")
            i += 1

        close_list()
        return "\n".join(html_lines)

    def _wrap_content(self, content):
        return f"<html><head>{self.STYLES}</head><body>{content}</body></html>"

    STYLES = """
    <style>
        :root {
            --bg: #f7f9fc;
            --panel: #ffffff;
            --text: #1f2937;
            --muted: #4b5563;
            --h1: #0f172a;
            --h2: #1d4ed8;
            --h3: #0f766e;
            --accent: #2563eb;
            --accent-soft: #dbeafe;
            --line: #d1d5db;
            --code-bg: #eef2ff;
            --code-fg: #3730a3;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            margin: 16px;
            padding: 16px 18px;
            line-height: 1.72;
            color: var(--text);
            background: var(--bg);
        }

        h1, h2, h3 {
            margin-top: 1.1em;
            margin-bottom: 0.5em;
            font-weight: 800;
            line-height: 1.35;
            scroll-margin-top: 12px;
        }

        h1 {
            color: var(--h1);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 10px;
            margin-top: 0.2em;
            font-size: 25px;
            letter-spacing: 0.2px;
        }

        h2 {
            color: var(--h2);
            font-size: 20px;
            border-left: 5px solid var(--accent);
            padding: 6px 10px;
            background: linear-gradient(90deg, var(--accent-soft), rgba(219, 234, 254, 0));
            border-radius: 6px;
        }

        h3 {
            color: var(--h3);
            font-size: 17px;
            border-bottom: 1px dashed var(--line);
            padding-bottom: 4px;
        }

        p {
            margin: 0.5em 0 0.85em;
            color: var(--text);
        }

        ul {
            margin: 0.35em 0 1.05em 1.35em;
            padding-left: 0.45em;
        }

        li {
            margin: 0.36em 0;
            color: var(--text);
        }

        li::marker {
            color: var(--accent);
        }

        strong {
            color: #0b1220;
            font-weight: 800;
            background: linear-gradient(transparent 65%, #fef08a 65%);
            padding: 0 1px;
        }

        code {
            background: var(--code-bg);
            color: var(--code-fg);
            border: 1px solid #c7d2fe;
            border-radius: 5px;
            padding: 1px 6px;
            font-size: 0.94em;
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }

        a {
            color: var(--accent);
            text-decoration: underline;
            text-underline-offset: 2px;
            font-weight: 700;
        }

        a:hover {
            color: #1e40af;
        }

        hr {
            border: none;
            border-top: 1px solid var(--line);
            margin: 18px 0;
        }

        .doc-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 16px;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        .doc-table thead th {
            text-align: left;
            font-weight: 800;
            color: #0b1220;
            background: #dbeafe;
            border-bottom: 1px solid #c7d2fe;
            padding: 8px 10px;
        }

        .doc-table tbody td {
            padding: 7px 10px;
            border-top: 1px solid #e5e7eb;
            vertical-align: top;
        }

        .doc-table tbody tr:nth-child(even) td {
            background: #f8fafc;
        }
    </style>
    """
    
    def _initialize_content(self):
        self.current_language = 'en'
        
        # --- 英語コンテンツ ---
        toc_structure_en = [("Manual", self._build_detailed_manual_toc_items('en'))]
        pages_en = {
            "home": """
            <h1>pyNuD Simulator Help</h1>
            <p>This is a help guide for the parameters used in the AFM Simulator.</p>
            <p>Select a topic from the table of contents on the left to view detailed explanations.</p>
            """,
            "display": """
            <h2>Display Settings</h2>
            <ul>
                <li><strong>Style:</strong> Selects the display style for the molecule (e.g., Ball & Stick, Spheres).</li>
                <li><strong>Color:</strong> Selects the coloring scheme (e.g., By Element, By Chain).</li>
                <li><strong>Show:</strong> Filters which atoms are displayed (e.g., All Atoms, Heavy Atoms).</li>
                <li><strong>Size / Opacity:</strong> Adjusts the size and opacity of atoms and bonds.</li>
                <li><strong>Quality:</strong> Quality of the 3D rendering. 'Fast' is quick, while 'High' is smoother.</li>
            </ul>
            """,
            "tip": """
            <h2>AFM Tip Settings</h2>
            <ul>
                <li><strong>Shape:</strong> Selects the overall shape of the tip.</li>
                <li><strong>Radius:</strong> Radius of curvature of the tip apex in nm. Smaller is sharper.</li>
                <li><strong>Angle:</strong> Half-angle of the cone part in degrees. Smaller is sharper.</li>
                <li><strong>Minitip Radius:</strong> Only for 'Sphere' shape. The radius of the sphere attached to the very end of the tip.</li>
            </ul>
            """,
            "simulation": """
            <h2>AFM Simulation Settings</h2>
            <ul>
                <li><strong>Scan Size (nm):</strong> The side length of the square area to be simulated, in nm.</li>
                <li><strong>Resolution:</strong> The number of pixels in the simulated image.</li>
                <li><strong>Consider atom size (vdW):</strong> If checked, treats atoms as spheres with van der Waals radii instead of points, calculating a more physically accurate surface.</li>
                <li><strong>Apply Low-pass Filter:</strong> If checked, applies an FFT low-pass filter to the result to match the resolution of real experimental data.</li>
                <li><strong>Cutoff Wavelength (nm):</strong> The cutoff wavelength for the filter. Empirically, a value around 2 nm often produces results that correspond well with real high-speed AFM images.</li>
                <li><strong>Interactive Update:</strong> If checked, automatically updates the simulation at low resolution (64x64) when PDB rotation, tip, or scan parameters are changed.</li>
            </ul>
            """,
            "file_loading": """
            <h2>File Loading</h2>
            <h3>File Import</h3>
            <ul>
                <li><strong>Import File:</strong> Loads structure data from PDB (<code>.pdb</code>), mmCIF (<code>.cif</code>, <code>.mmcif</code>), or MRC (<code>.mrc</code>) format files.</li>
                <li><strong>Automatic Tip Positioning:</strong> The tip is automatically positioned 2nm above the highest point of the loaded structure.</li>
                <li><strong>Rotation Controls:</strong> X, Y, Z rotation controls are automatically enabled after loading.</li>
            </ul>
            <h3>MRC Files</h3>
            <ul>
                <li><strong>MRC Format:</strong> MRC (Medical Research Council) format files (<code>.mrc</code>) are supported for volume data.</li>
                <li><strong>Density Threshold:</strong> Adjusts the isosurface threshold for volume rendering.</li>
                <li><strong>Flip Z-axis:</strong> Automatically flips the Z-axis orientation by default for proper display.</li>
                <li><strong>Voxel Size:</strong> Displays the physical size of each voxel in the volume data.</li>
            </ul>
            """,
            "structure_manipulation": """
            <h2>Structure Manipulation</h2>
            <h3>Rotation Controls</h3>
            <ul>
                <li><strong>Rotation X, Y, Z:</strong> Numeric input fields and sliders to rotate the structure around each axis.</li>
                <li><strong>CTRL+Drag:</strong> Hold CTRL and drag with the mouse to interactively rotate the structure in 3D space.</li>
                <li><strong>Reset Rotation:</strong> Use the "Reset Rotation" button to return all rotations to zero.</li>
            </ul>
            <h3>Find Initial Plane</h3>
            <ul>
                <li><strong>Purpose:</strong> Automatically orients the structure to its optimal viewing angle.</li>
                <li><strong>PDB Files:</strong> Uses Principal Component Analysis (PCA) to find the best orientation based on atom distribution.</li>
                <li><strong>MRC Files:</strong> Uses surface coordinate analysis to find the optimal orientation for volume data.</li>
                <li><strong>Usage:</strong> Click the button to automatically rotate the structure to its most stable orientation.</li>
            </ul>
            <h3>MRC-Specific Features</h3>
            <ul>
                <li><strong>Z-axis Flip:</strong> Toggle checkbox to flip the Z-axis orientation of MRC volume data.</li>
                <li><strong>Surface Rendering:</strong> Volume data is rendered as an isosurface based on the density threshold.</li>
                <li><strong>Interactive Rotation:</strong> MRC structures support the same rotation controls as PDB structures.</li>
            </ul>
            """
        }
        ui_text_en = {
            'window_title': "pyNuD Simulator Help", 'toc_header': "Contents",
            'home_tooltip': "Go to help home page"
        }

        # --- 日本語コンテンツ ---
        toc_structure_ja = [("マニュアル", self._build_detailed_manual_toc_items('ja'))]
        pages_ja = {
            "home": """
            <h1>pyNuD Simulator ヘルプ</h1>
            <p>pyNuD Simulatorで使われるパラメータの解説ガイドです。</p>
            <p>左の目次から項目を選択して、詳細な解説をご覧ください。</p>
            """,
            "display": """
            <h2>Display Settings / 表示設定</h2>
            <ul>
                <li><strong>Style:</strong> 分子の表示形式（例: Ball & Stick, Spheres）を選択します。</li>
                <li><strong>Color:</strong> 色付け方法（例: By Element, By Chain）を選択します。</li>
                <li><strong>Show:</strong> 表示する原子の種類（例: All Atoms, Heavy Atoms）をフィルタリングします。</li>
                <li><strong>Size / Opacity:</strong> 原子や結合のサイズ・不透明度を調整します。</li>
                <li><strong>Quality:</strong> 3D表示の品質。Fastは高速ですが、Highはより滑らかです。</li>
            </ul>
            """,
            "tip": """
            <h2>AFM Tip Settings / 探針条件</h2>
            <ul>
                <li><strong>Shape:</strong> 探針の全体的な形状を選択します。</li>
                <li><strong>Radius:</strong> 探針先端の曲率半径 (nm)。小さいほどシャープです。</li>
                <li><strong>Angle:</strong> 円錐部分の半頂角 (deg)。小さいほどシャープです。</li>
                <li><strong>Minitip Radius:</strong> 'Sphere'形状の時のみ有効。探針の最先端に取り付けられた球の半径です。</li>
            </ul>
            """,
            "simulation": """
            <h2>AFM Simulation / シミュレーション設定</h2>
            <ul>
                <li><strong>Scan Size (nm):</strong> シミュレーションを行う正方形領域の一辺の長さ (nm)。</li>
                <li><strong>Resolution:</strong> シミュレーション画像のピクセル数。</li>
                <li><strong>Consider atom size (vdW):</strong> チェックすると、原子を点ではなくファンデルワールス半径を持つ球として扱い、より物理的に正確な表面を計算します。</li>
                <li><strong>Apply Low-pass Filter:</strong> シミュレーション画像は実際の高速AFMデータより空間分解能が高いため、チェックするとFFTローパスフィルターで分解能を近づけます。</li>
                <li><strong>Cutoff Wavelength (nm):</strong> ローパスフィルターのカットオフ波長。経験的に2nm程度の値で実際の高速AFM画像とよく一致します。</li>
                <li><strong>Interactive Update:</strong> チェックすると、PDB回転や探針・スキャン条件の変更時に、低解像度(64x64)でシミュレーションを自動更新します。</li>
            </ul>
            """,
            "file_loading": """
            <h2>File Loading / ファイル読み込み</h2>
            <h3>File Import / ファイルインポート</h3>
            <ul>
                <li><strong>Import File:</strong> Loads structure data from PDB (<code>.pdb</code>), mmCIF (<code>.cif</code>, <code>.mmcif</code>), or MRC (<code>.mrc</code>) format files.</li>
                <li><strong>Import File / ファイルインポート:</strong> PDB（<code>.pdb</code>）、mmCIF（<code>.cif</code>、<code>.mmcif</code>）、またはMRC（<code>.mrc</code>）形式ファイルから構造データを読み込みます。</li>
                <li><strong>Automatic Tip Positioning:</strong> Automatically positions the tip 2nm above the highest point of the loaded structure.</li>
                <li><strong>Automatic Tip Positioning / 自動探針配置:</strong> 読み込んだ構造の最高点から2nm上に探針を自動配置します。</li>
                <li><strong>Rotation Controls:</strong> Rotation controls (X, Y, Z) are automatically enabled after loading.</li>
                <li><strong>Rotation Controls / 回転コントロール:</strong> 読み込み後にX、Y、Z回転コントロールが自動的に有効になります。</li>
            </ul>
            <h3>MRC Files / MRCファイル</h3>
            <ul>
                <li><strong>MRC Format:</strong> MRC (Medical Research Council) format files (<code>.mrc</code>) are supported for volume data.</li>
                <li><strong>MRC形式:</strong> MRC（Medical Research Council）形式ファイル（<code>.mrc</code>）がボリュームデータとしてサポートされています。</li>
                <li><strong>Density Threshold:</strong> ボリュームレンダリングの等値面閾値を調整します。</li>
                <li><strong>Flip Z-axis:</strong> デフォルトでZ軸の向きを自動的にフリップして正しい表示にします。</li>
                <li><strong>Voxel Size:</strong> ボリュームデータの各ボクセルの物理サイズを表示します。</li>
            </ul>
            """,
            "structure_manipulation": """
            <h2>Structure Manipulation / 構造操作</h2>
            <h3>Rotation Controls / 回転コントロール</h3>
            <ul>
                <li><strong>Rotation X, Y, Z:</strong> 各軸周りの構造回転用の数値入力フィールドとスライダーです。</li>
                <li><strong>CTRL+Drag:</strong> CTRLキーを押しながらマウスドラッグで3D空間内で構造をインタラクティブに回転できます。</li>
                <li><strong>Reset Rotation:</strong> 「Reset Rotation」ボタンで全ての回転をゼロに戻します。</li>
            </ul>
            <h3>Find Initial Plane / 初期平面検出</h3>
            <ul>
                <li><strong>Purpose:</strong> 構造を最適な視角に自動的に向けます。</li>
                <li><strong>PDB Files:</strong> 主成分分析（PCA）を使用して原子分布に基づく最適な向きを見つけます。</li>
                <li><strong>MRC Files:</strong> 表面座標解析を使用してボリュームデータの最適な向きを見つけます。</li>
                <li><strong>Usage:</strong> ボタンをクリックして構造を最も安定した向きに自動回転します。</li>
            </ul>
            <h3>MRC-Specific Features / MRC専用機能</h3>
            <ul>
                <li><strong>Z-axis Flip:</strong> チェックボックスでMRCボリュームデータのZ軸向きを切り替えます。</li>
                <li><strong>Surface Rendering:</strong> ボリュームデータは密度閾値に基づいて等値面としてレンダリングされます。</li>
                <li><strong>Interactive Rotation:</strong> MRC構造もPDB構造と同じ回転コントロールをサポートします。</li>
            </ul>
            """
        }
        ui_text_ja = {
            'window_title': "pyNuD Simulator ヘルプ", 'toc_header': "目次",
            'home_tooltip': "ヘルプのホームページに戻る"
        }

        self.content = {
            'en': {'toc_structure': toc_structure_en, 'pages': pages_en, 'ui_text': ui_text_en},
            'ja': {'toc_structure': toc_structure_ja, 'pages': pages_ja, 'ui_text': ui_text_ja},
        }

class HelpWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content_manager = HelpContentManager()
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(self.content_manager.get_ui_text('window_title'))
        self.resize(800, 600)
        self.setupUI()
        self.switch_language('ja') # デフォルトを日本語に

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar.setFixedHeight(40) # 明示的に高さを設定

        self.home_action = QPushButton("🏠 Home")
        self.home_action.clicked.connect(self.showHomePage)
        toolbar_layout.addWidget(self.home_action)

        # 中央にスペーサーを追加して左右のボタンを分離
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        toolbar_layout.addItem(spacer)

        self.lang_en_action = QPushButton("🇬🇧 English")
        self.lang_en_action.clicked.connect(lambda: self.switch_language('en'))
        toolbar_layout.addWidget(self.lang_en_action)

        self.lang_ja_action = QPushButton("🇯🇵 日本語")
        self.lang_ja_action.clicked.connect(lambda: self.switch_language('ja'))
        toolbar_layout.addWidget(self.lang_ja_action)

        layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.toc_tree = QTreeWidget()
        self.toc_tree.setHeaderHidden(True)
        self.toc_tree.setFixedWidth(220)
        self.toc_tree.setIndentation(26)
        self.toc_tree.setStyleSheet("""
            QTreeWidget {
                border: 1px solid #d1d5db;
                background: #f8fafc;
                font-size: 14px;
            }
            QTreeWidget::item {
                padding: 4px 6px;
            }
            QTreeWidget::item:selected {
                background: #dbeafe;
                color: #0f172a;
            }
        """)
        self.toc_tree.itemClicked.connect(self.onTocItemClicked)

        self.help_viewer = QTextBrowser()
        self.help_viewer.setOpenExternalLinks(True)

        splitter.addWidget(self.toc_tree)
        splitter.addWidget(self.help_viewer)
        splitter.setSizes([220, 580])
        layout.addWidget(splitter)
    
    def switch_language(self, lang_code):
        self.content_manager.set_language(lang_code)
        self.setWindowTitle(self.content_manager.get_ui_text('window_title'))
        self.home_action.setToolTip(self.content_manager.get_ui_text('home_tooltip'))
        self.loadTocContent()
        self.showHomePage()

    def loadTocContent(self):
        self.toc_tree.clear()
        toc_structure = self.content_manager.get_toc_structure()
        def add_items(parent_item, items_list, depth):
            for item_data in items_list:
                if len(item_data) == 2:
                    name, item_id = item_data
                    children = None
                else:
                    name, item_id, children = item_data
                child_item = QTreeWidgetItem([name])
                child_item.setData(0, Qt.UserRole, item_id)
                font = child_item.font(0)
                if depth == 1:
                    font.setBold(True)
                    font.setPointSize(14)
                    child_item.setForeground(0, QColor("#1d4ed8"))
                elif depth == 2:
                    font.setBold(True)
                    font.setPointSize(13)
                    child_item.setForeground(0, QColor("#0f766e"))
                else:
                    font.setBold(False)
                    font.setPointSize(12)
                    child_item.setForeground(0, QColor("#374151"))
                child_item.setFont(0, font)
                parent_item.addChild(child_item)
                if children:
                    add_items(child_item, children, depth + 1)
        for category_name, items in toc_structure:
            category_item = QTreeWidgetItem([category_name])
            category_font = category_item.font(0)
            category_font.setBold(True)
            category_font.setPointSize(14)
            category_item.setFont(0, category_font)
            self.toc_tree.addTopLevelItem(category_item)
            add_items(category_item, items, 1)
        self.toc_tree.expandAll()
    
    def onTocItemClicked(self, item, column):
        item_id = item.data(0, Qt.UserRole)
        if item_id: self.showHelpPage(item_id)
    
    def showHelpPage(self, page_id):
        base_id, anchor = (page_id.split('#', 1) + [None])[:2] if '#' in page_id else (page_id, None)
        self.help_viewer.setHtml(self.content_manager.get_content(base_id))
        if anchor:
            QTimer.singleShot(0, lambda: self.help_viewer.scrollToAnchor(anchor))
    
    def showHomePage(self):
        self.showHelpPage('detailed_manual')
class AFMSimulationWorker(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(object)
    status_update = pyqtSignal(str)
    task_done = pyqtSignal(object, QFrame)

    def __init__(self, parent, sim_params, tasks, element_symbols=None, vdw_radii=None, silent_mode=False):
        # 親を持たせて寿命をGUI側に寄せ、GCタイミング依存を減らす
        super().__init__(parent)
        self.parent = parent
        self.sim_params = sim_params
        self.tasks = tasks
        self.element_symbols = element_symbols
        self.vdw_radii = vdw_radii
        self._is_cancelled = False
        self.silent_mode = silent_mode  # ★★★ 軽量モードフラグ ★★★

    def cancel(self):
        self._is_cancelled = True
    
    def __del__(self):
        """
        デストラクタではwait/terminateしない。
        ここで同期停止すると、GC/破棄タイミング次第で「自分自身をwait」してデッドロックし得る。
        停止はpyNuD_simulator側の明示的なクリーンアップで行う。
        """
        try:
            self.cancel()
        except Exception:
            pass

    def run(self):
        try:
            total_tasks = len(self.tasks)
            if total_tasks == 0:
                self.done.emit(None)
                return

            for i, task in enumerate(self.tasks):
                start_progress = int((i / total_tasks) * 100)
                end_progress = int(((i + 1) / total_tasks) * 100)
                
                task_name = task["name"]
                scan_coords = task["coords"]
                target_panel = task["panel"]

                # ★★★ 軽量モードではプログレス更新を減らす ★★★
                if not self.silent_mode:
                    self.progress.emit(start_progress)
                if self._is_cancelled: break
                
                self.rotated_atom_coords = scan_coords
                if self._is_cancelled: break
                
                if self.sim_params.get('use_vdw', False) and self.element_symbols is not None:
                    sample_surface = self.create_vdw_surface()
                else:
                    sample_surface = self.create_atom_center_surface()
                
                if not self.silent_mode:
                    self.progress.emit(start_progress + int((end_progress - start_progress) * 0.1))
                if self._is_cancelled: break
                
                nx = self.sim_params.get('nx', self.sim_params.get('resolution', 64))
                ny = self.sim_params.get('ny', self.sim_params.get('resolution', 64))
                scan_x = self.sim_params.get('scan_x_nm', self.sim_params.get('scan_size', 20.0))
                scan_y = self.sim_params.get('scan_y_nm', self.sim_params.get('scan_size', 20.0))
                
                dx = scan_x / nx
                dy = scan_y / ny
                
                z_coords = scan_coords[:, 2]
                mol_depth = np.max(z_coords) - np.min(z_coords) if z_coords.size > 0 else 0
                tip_footprint = self.create_igor_style_tip(dx, dy, mol_depth)

                if not self.silent_mode:
                    self.progress.emit(start_progress + int((end_progress - start_progress) * 0.2))
                if self._is_cancelled: break

                afm_image = np.zeros((ny, nx), dtype=np.float64)
                
                last_emitted_progress = -1

                for r in range(ny):
                    if self._is_cancelled: break
                    afm_image[r, :] = _calculate_dilation_row(r, sample_surface, tip_footprint)
                    
                    if not self.silent_mode:
                        task_progress_fraction = 0.2 + (((r + 1) / ny) * 0.8)
                        current_overall_progress = start_progress + int(task_progress_fraction * (end_progress - start_progress))
                        
                        if current_overall_progress > last_emitted_progress:
                            self.progress.emit(current_overall_progress)
                            last_emitted_progress = current_overall_progress

                if self._is_cancelled: break
                self.task_done.emit(afm_image, target_panel)

            if self._is_cancelled:
                # ★★★ 削除：ステータス表示を無効化 ★★★
                # self.status_update.emit("Calculation cancelled.")
                pass
            else:
                # ★★★ 削除：ステータス表示を無効化 ★★★
                # self.status_update.emit("All tasks completed!")
                pass
            
            if not self.silent_mode:
                self.progress.emit(100)
            self.done.emit(None)

        except Exception as e:
            print(f"An error occurred during the AFM simulation: {e}")
            self.done.emit(None)
    
    def create_vdw_surface(self):
        """ファンデルワールス半径を考慮した表面マップを作成する。"""
        nx = self.sim_params.get('nx', self.sim_params.get('resolution', 64))
        ny = self.sim_params.get('ny', self.sim_params.get('resolution', 64))
        scan_x = self.sim_params.get('scan_x_nm', self.sim_params.get('scan_size', 20.0))
        scan_y = self.sim_params.get('scan_y_nm', self.sim_params.get('scan_size', 20.0))
        
        center_x = self.sim_params['center_x']
        center_y = self.sim_params['center_y']
        
        pixel_x = scan_x / nx
        pixel_y = scan_y / ny
        
        x_start = center_x - scan_x / 2.0
        y_start = center_y - scan_y / 2.0
        
        min_z = np.min(self.rotated_atom_coords[:, 2]) if self.rotated_atom_coords.size > 0 else 0
        
        if self.rotated_atom_coords.size == 0:
            return np.zeros((ny, nx), dtype=np.float64)

        atom_radii = np.array([self.vdw_radii.get(e, self.vdw_radii['other']) for e in self.element_symbols], dtype=np.float64)
        
        surface_map = _create_vdw_surface_loop(
            nx, ny, pixel_x, pixel_y, x_start, y_start, min_z,
            self.rotated_atom_coords, atom_radii
        )
        
        return surface_map


    def create_atom_center_surface(self):
        """UIで指定されたスキャンサイズと中心座標に基づいて、原子中心のZ座標から表面マップを作成"""
        nx = self.sim_params.get('nx', self.sim_params.get('resolution', 64))
        ny = self.sim_params.get('ny', self.sim_params.get('resolution', 64))
        scan_x = self.sim_params.get('scan_x_nm', self.sim_params.get('scan_size', 20.0))
        scan_y = self.sim_params.get('scan_y_nm', self.sim_params.get('scan_size', 20.0))
        
        center_x = self.sim_params['center_x']
        center_y = self.sim_params['center_y']
        
        pixel_x = scan_x / nx
        pixel_y = scan_y / ny
        
        x_start = center_x - scan_x / 2.0
        y_start = center_y - scan_y / 2.0
        
        min_z = np.min(self.rotated_atom_coords[:, 2]) if self.rotated_atom_coords.size > 0 else 0
        surface_map = np.full((ny, nx), min_z - 5.0, dtype=np.float64)

        if self.rotated_atom_coords.size == 0:
            return surface_map

        atom_x, atom_y, atom_z = self.rotated_atom_coords.T
        atom_z -= min_z
        ix = np.floor((atom_x - x_start) / pixel_x).astype(np.int32)
        iy = np.floor((atom_y - y_start) / pixel_y).astype(np.int32)

        mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        if np.any(mask):
            np.maximum.at(surface_map, (iy[mask], ix[mask]), atom_z[mask])
        
        surface_map[surface_map < min_z - 4.0] = 0.0 # 原子がないピクセルは高さ0とする

        return surface_map

    def create_igor_style_tip(self, dx, dy, mol_z_range):
        """UI基準のピクセルサイズ(dx,dy)と分子の高さ(mol_z_range)から探針を作成"""
        R = self.sim_params['tip_radius']
        miniR = self.sim_params['minitip_radius']
        alpha_deg = self.sim_params['tip_angle']
        tip_shape = self.sim_params['tip_shape']
        alpha_rad = np.radians(alpha_deg)

        if ((tip_shape == 'cone') or (tip_shape == 'sphere')):
            r_crit = R * np.cos(alpha_rad)
            z_offset = (R / np.sin(alpha_rad)) - R
            z_crit_related = R - r_crit / np.tan(alpha_rad)
            if z_crit_related > mol_z_range:
                max_tip_radius_nm = np.sqrt(max(0, R**2 - (R - mol_z_range)**2))
            else:
                max_tip_radius_nm = (mol_z_range + z_offset) * np.tan(alpha_rad)
        else: # Paraboloid
            max_tip_radius_nm = np.sqrt(max(0, 2 * R * mol_z_range))

        tip_px = int(np.ceil(max_tip_radius_nm / dx))
        tip_py = int(np.ceil(max_tip_radius_nm / dy))
        
        tip_wave = np.zeros((2 * tip_py + 1, 2 * tip_px + 1), dtype=np.float64)
        y_idx, x_idx = np.indices(tip_wave.shape)
        
        r_i = np.sqrt(((x_idx - tip_px) * dx)**2 + ((y_idx - tip_py) * dy)**2)
        if tip_shape == 'cone':
            r_crit = R * np.cos(alpha_rad)
            z_offset = (R / np.sin(alpha_rad)) - R
            sphere_mask = r_i <= r_crit
            cone_mask = r_i > r_crit
            tip_wave[sphere_mask] = R - np.sqrt(R**2 - r_i[sphere_mask]**2)
            tip_wave[cone_mask] = (r_i[cone_mask] / np.tan(alpha_rad)) - z_offset
        elif tip_shape == 'sphere':
            r_crit = R * np.cos(alpha_rad)
            z_offset = (R / np.sin(alpha_rad)) - R
            sphere_mask = r_i <= r_crit
            cone_mask = r_i > r_crit
            miniSphere_mask = r_i < miniR
            tip_wave[sphere_mask] = 2*miniR + R - np.sqrt(R**2 - r_i[sphere_mask]**2)
            tip_wave[cone_mask] = (r_i[cone_mask] / np.tan(alpha_rad)) - z_offset + 2*miniR
            tip_wave[miniSphere_mask] = miniR - np.sqrt(miniR**2 - r_i[miniSphere_mask]**2)
        else: # Paraboloid
            tip_wave = (r_i**2) / (2 * R)

        if np.any(tip_wave):
            tip_wave -= np.min(tip_wave)
        return tip_wave
    
    def simulate_views_blocking(self, desired_keys):
        """
        Run simulation only for desired view keys (['XY_Frame','YZ_Frame','ZX_Frame'])
        blocking this method until finished.
        """
        # Map internal keys to checkboxes
        key_to_check = {
            "XY_Frame": self.afm_x_check,
            "YZ_Frame": self.afm_y_check,
            "ZX_Frame": self.afm_z_check
        }
        # Save original states
        original = {k: key_to_check[k].isChecked() for k in key_to_check}
        try:
            # Apply new checkbox states
            for k, cb in key_to_check.items():
                cb.blockSignals(True)
                cb.setChecked(k in desired_keys)
                cb.blockSignals(False)
            # Kick simulation
            self.run_simulation()
            loop = QEventLoop()
            def _quit_once(_):
                if loop.isRunning():
                    loop.quit()
            self.simulation_done.connect(_quit_once)
            loop.exec_()
        finally:
            # Restore original states
            for k, cb in key_to_check.items():
                cb.blockSignals(True)
                cb.setChecked(original[k])
                cb.blockSignals(False)
            # Restore display layout
            self.update_afm_display()

    def handle_save_image(self):
        """Export one or more simulated AFM images (PNG) with optional incremental rotation."""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Data", "No simulation data available to save.")
            return
        
        # Build available (only those already simulated)
        available_keys = list(self.simulation_results.keys())
        display_names = {"XY_Frame": "XY View", "YZ_Frame": "YZ View", "ZX_Frame": "ZX View"}
        
        dlg = SaveAFMImageDialog(available_keys, display_names, self.get_active_dataset_id(), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        result = dlg.get_result()
        selected_view_keys = result['selected_views']
        rot_inc = result['drot']
        base_name = result['base_name']
        
        if not selected_view_keys:
            QMessageBox.warning(self, "No Selection", "No views selected.")
            return
        
        # Map for filename friendly
        def key_to_short(k):
            return {
                "XY_Frame": "XY",
                "YZ_Frame": "YZ",
                "ZX_Frame": "ZX"
            }.get(k, k.replace("_Frame", ""))
        
        # Prepare directory & ensure last_import_dir is valid
        directory = ""
        if self.last_import_dir and os.path.isdir(self.last_import_dir):
            directory = self.last_import_dir
        if not directory:
            directory = os.getcwd()
        
        # Save original rotation
        orig_rx = self.rotation_widgets['X']['spin'].value()
        orig_ry = self.rotation_widgets['Y']['spin'].value()
        orig_rz = self.rotation_widgets['Z']['spin'].value()
        
        apply_rotation = any(abs(v) > 1e-6 for v in rot_inc.values())
        
        try:
            if apply_rotation:
                # Apply incremental rotation (add to current)
                self.rotation_widgets['X']['spin'].setValue(self.normalize_angle(orig_rx + rot_inc['x']))
                self.rotation_widgets['Y']['spin'].setValue(self.normalize_angle(orig_ry + rot_inc['y']))
                self.rotation_widgets['Z']['spin'].setValue(self.normalize_angle(orig_rz + rot_inc['z']))
                # Force apply transform & run simulation for required views
                self.apply_structure_rotation()
                self.simulate_views_blocking(selected_view_keys)
            
            # Export each selected view
            export_count = 0
            for key in selected_view_keys:
                if key not in self.simulation_results:
                    continue
                data = self.simulation_results[key]
                # Normalize to 8-bit grayscale
                mn, mx = float(np.min(data)), float(np.max(data))
                if mx <= mn:
                    norm = np.zeros_like(data, dtype=np.uint8)
                else:
                    norm = ((data - mn) / (mx - mn) * 255).astype(np.uint8)
                
                # Resize to 512x512
                try:
                    from PIL import Image
                except ImportError:
                    QMessageBox.critical(self, "Missing Pillow", "Install Pillow to export images (pip install Pillow).")
                    return
                img = Image.fromarray(norm, mode='L')
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS
                img = img.resize((512, 512), resample=resample_filter)
                
                fname = f"{base_name}_{key_to_short(key)}_dx{rot_inc['x']:+.0f}_dy{rot_inc['y']:+.0f}_dz{rot_inc['z']:+.0f}.png"
                save_path = os.path.join(directory, fname)
                try:
                    img.save(save_path)
                    export_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to save {save_path}: {e}")
            
            if export_count:
                QMessageBox.information(self, "Export Complete", f"Exported {export_count} image(s) to:\n{directory}")
            else:
                QMessageBox.warning(self, "No Export", "No images were exported.")
        
        finally:
            # Restore original rotation if we changed it
            if apply_rotation:
                self.rotation_widgets['X']['spin'].setValue(orig_rx)
                self.rotation_widgets['Y']['spin'].setValue(orig_ry)
                self.rotation_widgets['Z']['spin'].setValue(orig_rz)
                self.apply_structure_rotation()
                # (Optionally regenerate original visible views if needed)
                # self.simulate_views_blocking(available_keys)

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, window_instance):
        super().__init__()
        self.window = window_instance
        self.panning = False
        self.actor_rotating = False
        self.pan_anchor_point = None
        self.pan_anchor_z = None

    def OnLeftButtonDown(self):
        rwi = self.GetInteractor()
        
        # macOSのCommandキーにも対応するため、GetCommandKey()のチェックを追加
        is_ctrl_or_cmd_pressed = rwi.GetControlKey() or rwi.GetCommandKey()
        
        # Ctrl(またはCmd)キーが押されているか最初にチェック
        if is_ctrl_or_cmd_pressed and not rwi.GetShiftKey():
            self.actor_rotating = True
            self.StartRotate()
        # Shiftキーが押されているかチェック
        elif rwi.GetShiftKey() and not is_ctrl_or_cmd_pressed:
            self.panning = True
            self.StartPan()
            renderer = self.GetCurrentRenderer()
            if renderer is None: return
            x, y = rwi.GetEventPosition()
            self.pan_anchor_z = renderer.GetZ(x, y)
            self.pan_anchor_point = self.get_world_point(renderer, x, y, self.pan_anchor_z)
        else:
            # 何も押されていなければ、通常のカメラ回転
            super().OnLeftButtonDown()

    def OnLeftButtonUp(self):
        if self.actor_rotating:
            self.actor_rotating = False
            self.EndRotate()
            
            # ドラッグ終了時の高解像度シミュレーション
            if hasattr(self.window, 'interactive_update_check') and self.window.interactive_update_check.isChecked():
                if hasattr(self.window, 'schedule_high_res_simulation'):
                    self.window.schedule_high_res_simulation()
                    
        elif self.panning:
            self.panning = False
            self.EndPan()
        else:
            super().OnLeftButtonUp()

    def OnMouseMove(self):
        if self.actor_rotating:
            self.RotateActor()
        elif self.panning:
            rwi = self.GetInteractor()
            renderer = self.GetCurrentRenderer()
            if renderer is None: return
            camera = renderer.GetActiveCamera()
            x, y = rwi.GetEventPosition()
            new_point = self.get_world_point(renderer, x, y, self.pan_anchor_z)
            motion_vector = [new_point[i] - self.pan_anchor_point[i] for i in range(3)]
            cam_pos = list(camera.GetPosition())
            cam_fp = list(camera.GetFocalPoint())
            camera.SetPosition([cam_pos[i] - motion_vector[i] for i in range(3)])
            camera.SetFocalPoint([cam_fp[i] - motion_vector[i] for i in range(3)])
            rwi.Render()
        else:
            super().OnMouseMove()

    def RotateActor(self):
        """カメラビューに応じた構造回転を実行（Qt側の回転処理へ委譲）"""
        rwi = self.GetInteractor()
        renderer = self.GetCurrentRenderer()
        if renderer is None:
            return

        # マウスの移動量を取得
        dx = rwi.GetEventPosition()[0] - rwi.GetLastEventPosition()[0]
        dy = rwi.GetEventPosition()[1] - rwi.GetLastEventPosition()[1]
        try:
            if hasattr(self.window, "update_rotation_from_drag_view_dependent"):
                self.window.update_rotation_from_drag_view_dependent(dx, dy)
        except Exception:
            pass
    
    def normalize_angle(self, angle):
        """角度を-180〜180の範囲に正規化"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def get_world_point(self, renderer, x, y, z):
        renderer.SetDisplayPoint(float(x), float(y), float(z))
        renderer.DisplayToWorld()
        world_point = renderer.GetWorldPoint()
        return [world_point[0] / world_point[3], 
                world_point[1] / world_point[3], 
                world_point[2] / world_point[3]]
    
class pyNuD_simulator(QMainWindow):

    simulation_done = pyqtSignal(object)
    simulation_progress = pyqtSignal(int)

    def normalize_angle(self, angle):
        """Normalize angle to [-180, 180]."""
        value = float(angle)
        while value > 180.0:
            value -= 360.0
        while value < -180.0:
            value += 360.0
        return value

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        
        # Windows固有の設定
        #if sys.platform.startswith('win'):
            # Windowsでの安定性向上のための設定
        #    self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        #    self.setAttribute(Qt.WA_NoSystemBackground, True)
        
        # スタンドアロンアプリケーションなのでwindow_managerは使用しない
        
        # ウィンドウの位置とサイズを復元
        self.settings = QSettings("pyNuD", "pyNuD_Simulator")
        self.restore_geometry()
        
        # 設定が保存されていない場合はデフォルトサイズを使用
        if not self.settings.contains("geometry"):
            # ウィンドウサイズ設定
            from PyQt5.QtWidgets import QDesktopWidget
            desktop = QDesktopWidget()
            screen_geometry = desktop.screenGeometry()
            
            width = int(screen_geometry.width() * 0.6)
            height = int(screen_geometry.height() * 0.6)
            
            # ★★★ 変更点: 最小サイズを小さく設定 ★★★
            self.setMinimumSize(600, 450)
            self.resize(width, height)
        self.center_on_screen()
        
        # データ格納
        self.atoms_data = None
        self.pdb_name = ""
        self.pdb_id = ""
        # 二次構造情報を格納（(chain_id, residue_id) -> 'H'/'E'/'C'）
        self.secondary_structure = {}
        # ★★★ MRC関連の変数を追加 ★★★
        self.mrc_data = None
        self.mrc_data_original = None  # 元のMRCデータ（フリップ前）
        self.mrc_voxel_size_nm = 1.0 / 10.0
        self.mrc_threshold = 0.3
        self.mrc_z_flip = True  # Z軸フリップ状態（デフォルトでTrue）
        # ★★★ ここまで ★★★
        self.tip_actor = None
        self.sample_actor = None
        self.bonds_actor = None
        self.simulation_results = {} 
        self.raw_simulation_results = {}

        self.help_window = None

        # --- Real AFM (ASD) state ---
        self.real_afm_nm = None
        self.real_afm_nm_full = None
        self.real_meta = None
        self.real_meta_full = None
        self.real_asd_path = None
        self.real_afm_roi_px = None  # (x0, y0, x1, y1) in full-image raw coordinates
        self.real_asd_frame_num = 0
        self.real_asd_frame_index = 0
        self.pose = {'theta_deg': 0.0, 'dx_px': 0.0, 'dy_px': 0.0, 'mirror_mode': 'none', 'score': None}
        self.sim_aligned_nm = None
        self.real_afm_window = None
        self.real_afm_window_real_frame = None
        self.real_afm_window_aligned_frame = None
        self.real_afm_frame_slider = None
        self.real_afm_frame_label = None
        self.real_afm_sim_info_label = None
        self._real_afm_pending_frame_index = None
        self._real_afm_frame_load_timer = None
        self.afm_appearance_window = None
        self.afm_appearance_group = None

        # --- PyMOL rendering state ---
        self.render_backend = "vtk"
        # User preference for protein structures (pdb/cif): "dual" | "pymol" | "vtk"
        # MRC is always forced to VTK-only.
        self.user_render_backend_preference = "dual"
        self.pymol_cmd = None
        self.pymol_widget = None
        self.pymol_image_label = None
        self.pymol_image_mode = False
        self.pymol_force_image_mode = False
        self.pymol_available = False
        self.dual_view_mode = True
        self._vtk_camera_observer_id = None
        self._vtk_camera_sync_timer = None
        self.pymol_object_name = "pynud_molecule"
        self.pymol_loaded_path = None
        self.current_structure_path = None
        self.current_structure_type = None
        self.pymol_esp_object = None
        self._color_scheme_before_esp = None
        self.display_widget = None
        self.vtk_initialized = False

        # --- PyMOL image-mode performance throttling ---
        self._pymol_render_timer = None
        self._pymol_interaction_clear_timer = None
        self._pymol_interacting = False
        self._pymol_last_render_ts = 0.0
        self._pymol_min_render_interval_s = 0.08  # ~12.5 FPS max in image mode
        self._pymol_mouse_dragging = False
        self._pymol_mouse_panning = False
        self._pymol_mouse_last_pos = None
        self._pymol_mouse_mode = None
        
        # 変換を二段に分離
        self.base_transform = vtk.vtkTransform()
        self.base_transform.Identity()
        
        self.local_transform = vtk.vtkTransform()
        self.local_transform.Identity()
        self.local_transform.PostMultiply()  # ローカル回転を右に積む（オブジェクト座標で回す）
        
        self.combined_transform = vtk.vtkTransform()
        self.combined_transform.Identity()
        self.combined_transform.PostMultiply()
        
        # 後方互換性のため残す
        self.molecule_transform = vtk.vtkTransform()
        self.last_import_dir = ""
        
        # スライダ差分適用用の前回値
        self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # インタラクティブモード用にユーザーの解像度選択を記憶する変数 ★★★
        self.user_selected_resolution = ""

        # マウスイベント用の状態変数
        self.actor_rotating = False
        self.panning = False

        # UI操作フラグ（PyMOL/VTKどちらでも必要）
        self.tip_slider_pressed = False
        self.scan_size_keyboard_input = False
        self.tip_radius_keyboard_input = False
        self.minitip_radius_keyboard_input = False
        self.tip_angle_keyboard_input = False
        self.scan_size_debounce_timer = None
        self.tip_radius_debounce_timer = None
        self.minitip_radius_debounce_timer = None
        self.tip_angle_debounce_timer = None
        
        # カラー・ライティング設定
        self.current_bg_color = (0.05, 0.05, 0.05)
        self.current_single_color = (0.5, 0.7, 0.9)
        self.brightness_factor = 1.0
        
        # AFM像表示用の参照
        self.afm_x_widget = None
        self.afm_y_widget = None
        self.afm_z_widget = None
        
        # 簡単で確実なカラーマップ
        self.element_colors = {
            'C': (0.3, 0.3, 0.3), 'O': (1.0, 0.3, 0.3), 'N': (0.3, 0.3, 1.0),
            'H': (0.9, 0.9, 0.9), 'S': (1.0, 1.0, 0.3), 'P': (1.0, 0.5, 0.0),
            'other': (0.7, 0.7, 0.7)
        }
        
        # チェーンカラー
        self.chain_colors = [
            (0.2, 0.8, 0.2), (0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.2),
            (0.8, 0.2, 0.8), (0.2, 0.8, 0.8), (1.0, 0.5, 0.0), (0.5, 0.0, 0.8),
        ]
        
         # ★★★ ここから追加 ★★★
        # 一般的なファンデルワールス半径 (nm)
        self.vdw_radii = {
            'H': 0.120, 'C': 0.170, 'N': 0.155, 'O': 0.152,
            'P': 0.180, 'S': 0.180, 'other': 0.170
        }
        
        # バックグラウンド処理からのシグナル
        #self.simulation_done = pyqtSignal(object)
        #self.simulation_progress = pyqtSignal(int)
        # ★★★ ここまで追加 ★★★

        # AFMパラメータ
        self.afm_params = {
            'tip_radius': 2.0, 'tip_shape': 'cone', 'tip_angle': 15.0,
            'tip_x': 0.0, 'tip_y': 0.0, 'tip_z': 5.0,
        }
        
        
        # メニューバーを作成（Helpはここへ移動）
        self.create_menu_bar()

        # ★★★ 修正点: 呼び出し順序を変更 ★★★
        self.setup_ui()    # UIウィジェットを全て作成
        self.setup_pymol()  # PyMOL環境を初期化（失敗時はVTKへフォールバック）

         # シミュレーション結果が一つでもあれば、各種保存ボタンを有効化する
        self.simulation_done.connect(self.on_simulation_finished)

        # PyInstaller環境を検出して適切なファイルパスを決定
        if getattr(sys, 'frozen', False):
            # PyInstallerで作成されたアプリの場合
            # ユーザーのホームディレクトリ内に設定ファイルを作成
            home_dir = Path.home()
            config_dir = home_dir / "pyNuD_config"
            config_dir.mkdir(exist_ok=True)
            self.settings_file = str(config_dir / "simulator_config.json")
        else:
            # 開発環境の場合
            self.settings_file = "config.json"

        # ★★★ 追加: 全ての準備が完了した後に、UIの初期状態を設定 ★★★
        self.update_tip_ui(self.tip_shape_combo.currentText())

        self.load_settings()

    def setup_pymol(self):
        """PyMOL環境のセットアップ（Qt埋め込み）"""
        self.render_backend = "pymol"
        self.pymol_available = False
        try:
            import pymol
            # Some PyMOL builds expect argv to be set to avoid empty-list parse errors
            is_macos = sys.platform.startswith('darwin')
            if is_macos:
                # On macOS, GUI launch from a thread is not supported.
                # Use no-GUI mode and render to images.
                self.pymol_force_image_mode = True
                try:
                    pymol.pymol_argv = ['pymol', '-cq']
                except Exception:
                    pass
                pymol.finish_launching(['pymol', '-cq'])
            else:
                try:
                    pymol.pymol_argv = ['pymol', '-q']
                except Exception:
                    pass
                try:
                    pymol.finish_launching()
                except Exception:
                    pymol.finish_launching(['pymol', '-q'])
            from pymol import cmd
            self.pymol_cmd = cmd
            self.pymol_available = True
        except Exception as e:
            print(f"PyMOL setup error: {e}")
            self.render_backend = "vtk"
            if hasattr(self, 'esp_check'):
                self.esp_check.setEnabled(False)
            self._setup_vtk_legacy()
            return

        # Qtウィジェットの初期化
        self._init_pymol_widget()
        self._apply_pymol_defaults()
        # Always initialize VTK in dual-view mode to avoid missing renderer
        self._ensure_vtk_initialized()
        # Default: show both when possible
        if getattr(self, 'current_structure_type', None) == "mrc":
            self._set_render_backend("vtk")
        else:
            self._set_render_backend("dual")
        if hasattr(self, 'esp_check'):
            self.esp_check.setEnabled(self.render_backend in ("pymol", "dual") and self.pymol_available)

    def _init_pymol_widget(self):
        """PyMOLの描画ウィジェットを作成してビューへ追加"""
        if not self.pymol_available:
            return

        if self.pymol_force_image_mode:
            # 強制イメージモード（macOS no-GUI）
            self.pymol_image_mode = True
            self.pymol_image_label = QLabel("PyMOL view (image mode)")
            self.pymol_image_label.setAlignment(Qt.AlignCenter)
            self.pymol_image_label.setToolTip(
                "PyMOL image interaction:\n"
                "Left drag: view rotate\n"
                "Cmd/Ctrl+Left drag: object rotate\n"
                "Shift+Left or Right drag: pan\n"
                "Mouse wheel: zoom"
            )
            self.pymol_widget = self.pymol_image_label
            if hasattr(self.pymol_widget, 'setAcceptDrops'):
                self.pymol_widget.setAcceptDrops(True)
            if hasattr(self.pymol_widget, 'installEventFilter'):
                self.pymol_widget.installEventFilter(self)
            if hasattr(self, 'pymol_widget_layout') and self.pymol_widget_layout is not None:
                while self.pymol_widget_layout.count():
                    item = self.pymol_widget_layout.takeAt(0)
                    if item.widget():
                        item.widget().setParent(None)
                self.pymol_widget_layout.addWidget(self.pymol_widget)
            self._update_esp_colorbar_visibility()
            return

        pymol_widget = None
        try:
            from pymol import Qt as pymol_qt
            get_widget = None
            if hasattr(pymol_qt, "get_widget"):
                get_widget = pymol_qt.get_widget
            elif hasattr(pymol_qt, "utils") and hasattr(pymol_qt.utils, "get_pymol_qt_widget"):
                get_widget = pymol_qt.utils.get_pymol_qt_widget
            elif hasattr(pymol_qt, "get_pymol_qt_widget"):
                get_widget = pymol_qt.get_pymol_qt_widget
            if get_widget:
                pymol_widget = get_widget()
        except Exception:
            pymol_widget = None

        if pymol_widget is None:
            # フォールバック: PyMOLの描画を画像として表示
            self.pymol_image_mode = True
            self.pymol_image_label = QLabel("PyMOL view (image mode)")
            self.pymol_image_label.setAlignment(Qt.AlignCenter)
            self.pymol_image_label.setToolTip(
                "PyMOL image interaction:\n"
                "Left drag: view rotate\n"
                "Cmd/Ctrl+Left drag: object rotate\n"
                "Shift+Left or Right drag: pan\n"
                "Mouse wheel: zoom"
            )
            pymol_widget = self.pymol_image_label
        else:
            self.pymol_image_mode = False

        self.pymol_widget = pymol_widget
        if hasattr(self.pymol_widget, 'setAcceptDrops'):
            self.pymol_widget.setAcceptDrops(True)
        if hasattr(self.pymol_widget, 'installEventFilter'):
            self.pymol_widget.installEventFilter(self)

        # 既存のPyMOLビューコンテナに追加
        if hasattr(self, 'pymol_widget_layout') and self.pymol_widget_layout is not None:
            # プレースホルダを削除
            while self.pymol_widget_layout.count():
                item = self.pymol_widget_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            self.pymol_widget_layout.addWidget(self.pymol_widget)
        self._update_esp_colorbar_visibility()

    def _apply_pymol_defaults(self):
        """PyMOLの表示設定を初期化"""
        if not self.pymol_available or self.pymol_cmd is None:
            return
        try:
            # 背景色を現在の設定に合わせる
            self._pymol_set_background(self.current_bg_color)
            # Cartoonの品質設定
            self.pymol_cmd.set("cartoon_fancy_helices", 1)
            self.pymol_cmd.set("cartoon_smooth_loops", 1)
            self.pymol_cmd.set("cartoon_sampling", 10)
        except Exception:
            pass

    def _setup_vtk_legacy(self):
        """VTK環境のセットアップ（フォールバック用）"""
        # VTKウィジェットが存在することを確認
        if not hasattr(self, 'vtk_widget') or self.vtk_widget is None:
            print("Error: VTK widget not found")
            return
            
        try:
            # レンダラー作成
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(*self.current_bg_color)
            
            # スライダー操作フラグの初期化
            self.tip_slider_pressed = False
            
            # スピンボックスの入力方法フラグ（True=キー入力中, False=マウス/ボタン操作）
            self.scan_size_keyboard_input = False
            self.tip_radius_keyboard_input = False
            self.minitip_radius_keyboard_input = False
            self.tip_angle_keyboard_input = False
            
            # デバウンス用のタイマー
            self.scan_size_debounce_timer = None
            self.tip_radius_debounce_timer = None
            self.minitip_radius_debounce_timer = None
            self.tip_angle_debounce_timer = None
            
            # アンチエイリアシング
            render_window = self.vtk_widget.GetRenderWindow()
            render_window.AddRenderer(self.renderer)
            render_window.SetMultiSamples(4)
            
            # インタラクター設定
            self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

            # CustomInteractorStyleにメインウィンドウ(self)への参照を渡す
            style = CustomInteractorStyle(self)
            self.interactor.SetInteractorStyle(style)
            
            # macOSでのイベントハンドラー問題を回避するため、直接イベントを監視
            # 元のイベントハンドラーを保存
            self.original_mouse_press = self.vtk_widget.mousePressEvent
            self.original_mouse_move = self.vtk_widget.mouseMoveEvent
            self.original_mouse_release = self.vtk_widget.mouseReleaseEvent
            
            self.vtk_widget.mousePressEvent = self.on_mouse_press
            self.vtk_widget.mouseMoveEvent = self.on_mouse_move
            self.vtk_widget.mouseReleaseEvent = self.on_mouse_release
            
            # ライティング改善
            self.setup_lighting()
            
            # 座標軸追加
            self.add_axes()
            
            # 初期カメラ設定
            self.reset_camera()

            # カメラ変更の同期
            self._attach_vtk_camera_observer()
            
            # レンダリング開始
            self.interactor.Initialize()
            self.vtk_initialized = True
            
        except Exception as e:
            print(f"VTK setup error: {e}")

    def _ensure_vtk_initialized(self):
        """必要な場合にVTKを初期化する"""
        if not getattr(self, 'vtk_initialized', False):
            self._setup_vtk_legacy()
        self._attach_vtk_camera_observer()

    def _attach_vtk_camera_observer(self):
        if not hasattr(self, 'renderer') or self.renderer is None:
            return
        try:
            camera = self.renderer.GetActiveCamera()
            if camera is None:
                return
            if self._vtk_camera_observer_id is None:
                self._vtk_camera_observer_id = camera.AddObserver(
                    "ModifiedEvent", self._on_vtk_camera_modified
                )
        except Exception:
            pass

    def _is_dual_mode(self):
        if not self.pymol_available:
            return False
        if getattr(self, 'current_structure_type', None) == "mrc":
            return False
        return self.render_backend == "dual"

    def _is_pymol_active(self):
        return bool(self.pymol_available and self.pymol_cmd is not None and self.render_backend in ("pymol", "dual"))

    def _is_pymol_only(self):
        return bool(self._is_pymol_active() and not self._is_dual_mode())

    def _set_render_backend(self, backend):
        """表示バックエンドを切り替える"""
        self.render_backend = backend
        if hasattr(self, 'esp_check'):
            self.esp_check.setEnabled(backend in ("pymol", "dual") and self.pymol_available)
        # Stop pending PyMOL renders when leaving PyMOL modes
        if backend == "vtk":
            try:
                if self._pymol_render_timer is not None:
                    self._pymol_render_timer.stop()
            except Exception:
                pass
        if self._is_dual_mode():
            self._apply_view_visibility()
            if hasattr(self, 'vtk_widget') and self.vtk_widget is not None:
                self.display_widget = self.vtk_widget
        elif backend == "pymol" and self.pymol_widget is not None:
            self._apply_view_visibility()
            self.display_widget = self.pymol_widget
        else:
            self._ensure_vtk_initialized()
            self._apply_view_visibility()
            self.display_widget = self.vtk_widget
        self._apply_esp_color_lock()
        self._update_esp_colorbar_visibility()
        self._update_renderer_combo()

    def _apply_view_visibility(self):
        """2ペイン/単一表示の可視性を設定"""
        if not hasattr(self, 'pymol_view_container') or not hasattr(self, 'vtk_view_container'):
            return
        if self._is_dual_mode():
            self.pymol_view_container.setVisible(True)
            self.vtk_view_container.setVisible(True)
        else:
            if self.render_backend == "pymol" and self.pymol_available:
                self.pymol_view_container.setVisible(True)
                self.vtk_view_container.setVisible(False)
            else:
                self.pymol_view_container.setVisible(False)
                self.vtk_view_container.setVisible(True)

    def _update_renderer_combo(self):
        if not hasattr(self, 'renderer_combo'):
            return
        try:
            self.renderer_combo.blockSignals(True)
            self.renderer_combo.clear()
            if self.pymol_available and getattr(self, 'current_structure_type', None) != "mrc":
                pymol_label = "PyMOL (image)" if self.pymol_force_image_mode else "PyMOL (embedded)"
                self.renderer_combo.addItems(["Dual (PyMOL + VTK)", pymol_label, "VTK (interactive)"])
                if self.render_backend == "dual":
                    self.renderer_combo.setCurrentText("Dual (PyMOL + VTK)")
                elif self.render_backend == "vtk":
                    self.renderer_combo.setCurrentText("VTK (interactive)")
                else:
                    self.renderer_combo.setCurrentText(pymol_label)
                self.renderer_combo.setEnabled(True)
            else:
                self.renderer_combo.addItems(["VTK (interactive)"])
                self.renderer_combo.setCurrentIndex(0)
                self.renderer_combo.setEnabled(False)
        finally:
            self.renderer_combo.blockSignals(False)

    def on_renderer_changed(self, text):
        """Rendererコンボの変更を反映"""
        if text.startswith("Dual") and self.pymol_available and getattr(self, 'current_structure_type', None) != "mrc":
            self.user_render_backend_preference = "dual"
            self._set_render_backend("dual")
            if self.atoms_data is not None:
                self.display_molecule()
            self.request_render()
            return

        if "PyMOL" in text and self.pymol_available:
            self.user_render_backend_preference = "pymol"
            self._set_render_backend("pymol")
            if self.atoms_data is not None:
                self.display_molecule()
            if hasattr(self, 'show_tip_check') and self.show_tip_check.isChecked():
                QMessageBox.information(
                    self, "Tip Display",
                    "Tip rendering is only available in VTK mode.\n"
                    "PyMOL (image) mode disables interactive tip display."
                )
        else:
            self.user_render_backend_preference = "vtk"
            self._set_render_backend("vtk")
            if self.atoms_data is not None:
                self.display_molecule()
            if hasattr(self, 'show_tip_check') and self.show_tip_check.isChecked():
                self.create_tip()
        self.request_render()

    def _pymol_set_background(self, rgb):
        # Background should be kept consistent even if the current renderer is VTK-only.
        if not self.pymol_available or self.pymol_cmd is None:
            return
        try:
            # Ensure PNG output uses an opaque background; otherwise the QLabel's default
            # (often white) will show through.
            try:
                self.pymol_cmd.set("opaque_background", 1)
            except Exception:
                pass
            try:
                self.pymol_cmd.set("ray_opaque_background", 1)
            except Exception:
                pass

            name = "pynud_bg"
            self.pymol_cmd.set_color(name, [rgb[0], rgb[1], rgb[2]])
            self.pymol_cmd.bg_color(name)
            # Fallback for some PyMOL builds/settings
            try:
                self.pymol_cmd.set("bg_rgb", [rgb[0], rgb[1], rgb[2]])
            except Exception:
                pass

            # If we are in image mode, also paint the QLabel background to match in case
            # the generated PNG still has alpha for any reason.
            if getattr(self, "pymol_image_label", None) is not None:
                r255 = int(max(0.0, min(1.0, float(rgb[0]))) * 255)
                g255 = int(max(0.0, min(1.0, float(rgb[1]))) * 255)
                b255 = int(max(0.0, min(1.0, float(rgb[2]))) * 255)
                self.pymol_image_label.setStyleSheet(f"background-color: rgb({r255}, {g255}, {b255});")
        except Exception:
            pass

    def _pymol_set_standard_view(self, view_plane):
        """PyMOLの標準視点に切り替え"""
        if not self._is_pymol_active():
            return
        try:
            view = list(self.pymol_cmd.get_view())
            if len(view) < 18:
                return
            if view_plane == 'xy':
                rot = [1, 0, 0,
                       0, 1, 0,
                       0, 0, 1]
            elif view_plane == 'yz':
                rot = [0, 1, 0,
                       0, 0, 1,
                       1, 0, 0]
            elif view_plane == 'zx':
                rot = [1, 0, 0,
                       0, 0, 1,
                       0, -1, 0]
            else:
                return
            view[:9] = rot
            self.pymol_cmd.set_view(view)
        except Exception:
            pass
        self.request_render()

    def request_render(self):
        """現在のバックエンドに応じて再描画を要求"""
        if self._is_pymol_active():
            if self.pymol_image_mode and self.pymol_image_label is not None:
                self._schedule_pymol_render()
            else:
                try:
                    self.pymol_cmd.refresh()
                except Exception:
                    pass
        if (not self._is_pymol_only()) and hasattr(self, 'vtk_widget') and self.vtk_widget is not None:
            try:
                self.vtk_widget.GetRenderWindow().Render()
            except Exception:
                pass

    def _render_pymol_image(self):
        """PyMOL描画を画像として取得し、QLabelに反映"""
        if not self._is_pymol_active() or self.pymol_image_label is None:
            return
        # Throttle renders in image mode (png file IO is expensive)
        now = time.monotonic()
        if self._pymol_interacting and (now - self._pymol_last_render_ts) < self._pymol_min_render_interval_s:
            self._schedule_pymol_render()
            return
        self._pymol_last_render_ts = now
        try:
            scale = 0.6 if self._pymol_interacting else 1.0
            width = max(200, int(self.pymol_image_label.width() * scale))
            height = max(200, int(self.pymol_image_label.height() * scale))
            dpi = 72 if self._pymol_interacting else 120
            tmp_path = os.path.join(tempfile.gettempdir(), "pynud_pymol_view.png")
            self.pymol_cmd.png(tmp_path, width, height, dpi=dpi, ray=0, quiet=1)
            pixmap = QPixmap(tmp_path)
            self.pymol_image_label.setPixmap(pixmap)
        except Exception:
            pass

    def _mark_pymol_interaction(self):
        """ユーザー操作中としてマーク（PyMOL image-mode を低負荷で更新）"""
        self._pymol_interacting = True
        if self._pymol_interaction_clear_timer is None:
            self._pymol_interaction_clear_timer = QTimer(self)
            self._pymol_interaction_clear_timer.setSingleShot(True)

            def _clear():
                self._pymol_interacting = False
                # 操作終了後に高品質で1回描画
                self._schedule_pymol_render(force=True)

            self._pymol_interaction_clear_timer.timeout.connect(_clear)
        # 最後の操作から200msで「操作終了」とみなす
        self._pymol_interaction_clear_timer.start(200)

    def _schedule_pymol_render(self, force=False):
        """PyMOL image-mode 描画をデバウンスしてスケジュール"""
        if not self._is_pymol_active() or not getattr(self, "pymol_image_mode", False):
            return
        if self.pymol_image_label is None:
            return
        if self._pymol_render_timer is None:
            self._pymol_render_timer = QTimer(self)
            self._pymol_render_timer.setSingleShot(True)
            self._pymol_render_timer.timeout.connect(self._render_pymol_image)
        delay_ms = 80 if self._pymol_interacting else 20
        if force:
            delay_ms = 0
        self._pymol_render_timer.start(delay_ms)

    def _apply_pymol_lighting(self):
        """PyMOLのライティング設定を反映"""
        if not self._is_pymol_active():
            return
        try:
            ambient = self.ambient_slider.value() / 100.0
            specular = self.specular_slider.value() / 100.0
            ambient = min(1.0, ambient * self.brightness_factor)
            self.pymol_cmd.set("ambient", ambient)
            self.pymol_cmd.set("specular", min(1.0, specular))
        except Exception:
            pass

    def _on_vtk_camera_modified(self, obj, event):
        """VTKカメラ変更をPyMOLに同期（デバウンス）"""
        if not self._is_dual_mode() or not self._is_pymol_active():
            return
        if self._vtk_camera_sync_timer is None:
            self._vtk_camera_sync_timer = QTimer(self)
            self._vtk_camera_sync_timer.setSingleShot(True)
            self._vtk_camera_sync_timer.timeout.connect(self._sync_pymol_view_from_vtk)
        # 連続イベントをまとめる
        self._mark_pymol_interaction()
        self._vtk_camera_sync_timer.start(60)

    def _sync_pymol_view_from_vtk(self):
        """VTKカメラの視点をPyMOLへ反映"""
        if not self._is_dual_mode() or not self._is_pymol_active():
            return
        if not hasattr(self, 'renderer') or self.renderer is None:
            return
        try:
            camera = self.renderer.GetActiveCamera()
            if camera is None:
                return
            pos = np.array(camera.GetPosition(), dtype=float)
            focal = np.array(camera.GetFocalPoint(), dtype=float)
            up = np.array(camera.GetViewUp(), dtype=float)
            forward = focal - pos
            f_norm = np.linalg.norm(forward)
            if f_norm < 1e-9:
                return
            forward = forward / f_norm
            right = np.cross(forward, up)
            r_norm = np.linalg.norm(right)
            if r_norm < 1e-9:
                return
            right = right / r_norm
            true_up = np.cross(right, forward)

            # PyMOL `set_view` expects the 3x3 rotation in a different layout than VTK's
            # camera-axis rows. Empirically, providing the camera axes as *columns*
            # matches the on-screen orientation (avoids a 90° mismatch).
            rot = [
                right[0], true_up[0], -forward[0],
                right[1], true_up[1], -forward[1],
                right[2], true_up[2], -forward[2],
            ]

            try:
                view = list(self.pymol_cmd.get_view())
            except Exception:
                view = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        10.0, -10.0, 1.0]
            if len(view) < 18:
                view = (view + [0.0] * 18)[:18]
            view[0:9] = rot

            # Sync zoom/pan to match VTK wheel operations.
            # VTK in this app uses nm units (PDB Å -> nm conversion), while PyMOL uses Å.
            # Convert VTK camera parameters nm -> Å (x10).
            dist_nm = float(np.linalg.norm(pos - focal))
            dist_a = dist_nm * 10.0

            # Preserve PyMOL's clip thickness relative to distance
            try:
                old_pos = np.array(view[9:12], dtype=float)
                old_dist_a = float(np.linalg.norm(old_pos))
                if old_dist_a < 1e-6:
                    old_dist_a = float(abs(float(view[11])))
                d_front = float(view[15]) - old_dist_a
                d_back = float(view[16]) - old_dist_a
            except Exception:
                d_front, d_back = -3.0, 3.0

            # In PyMOL view representation, camera position is typically along -Z in camera coords.
            # Only adjust zoom (distance) to follow VTK wheel zoom, keep current pan/origin.
            view[11] = -dist_a
            view[15] = dist_a + d_front
            view[16] = dist_a + d_back

            # Keep VTK field-of-view when possible (PyMOL stores it in view[17] with sign)
            try:
                fov = float(camera.GetViewAngle())
                sign = -1.0 if float(view[17]) < 0.0 else 1.0
                view[17] = sign * fov
            except Exception:
                pass
            self.pymol_cmd.set_view(view)
            if self.pymol_image_mode:
                self._schedule_pymol_render()
            else:
                try:
                    self.pymol_cmd.refresh()
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_pymol_object_ttt_from_vtk(self):
        """VTKの分子変換（base+local）をPyMOLのオブジェクト変換(TTT)へ反映"""
        if not self._is_dual_mode() or not self._is_pymol_active():
            return
        if not self.pymol_available or self.pymol_cmd is None:
            return
        if not self.current_structure_path or self.pymol_loaded_path != self.current_structure_path:
            return
        if not hasattr(self, "combined_transform") or self.combined_transform is None:
            return

        obj = self.pymol_object_name
        try:
            vtk_matrix = self.combined_transform.GetMatrix()
            # VTKはnm、PyMOLはÅ。回転はそのまま、平行移動のみ×10する。
            ttt = []
            for i in range(4):
                for j in range(4):
                    val = float(vtk_matrix.GetElement(i, j))
                    if j == 3 and i < 3:
                        val *= 10.0
                    ttt.append(val)

            # 変更がない場合は更新しない（画像更新コストを抑える）
            if getattr(self, "_pymol_last_ttt", None) == tuple(ttt):
                return
            self._pymol_last_ttt = tuple(ttt)

            try:
                self.pymol_cmd.matrix_reset(obj)
            except Exception:
                pass
            # homogenous=0: 入力をそのまま使用（homogenous=1 は転置されるため注意）
            self.pymol_cmd.set_object_ttt(obj, ttt, homogenous=0)
            if self.pymol_image_mode:
                self._schedule_pymol_render()
            else:
                try:
                    self.pymol_cmd.refresh()
                except Exception:
                    pass
        except Exception:
            pass
        
    def center_on_screen(self):
        """ウィンドウを画面中央に配置"""
        from PyQt5.QtWidgets import QDesktopWidget
        frame_geometry = self.frameGeometry()
        desktop = QDesktopWidget()
        center_point = desktop.availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())
    
    def restore_geometry(self):
        """ウィンドウの位置とサイズを復元"""
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except Exception:
            pass  # 復元に失敗した場合は無視
    
    def save_geometry(self):
        """ウィンドウの位置とサイズを保存"""
        try:
            geometry = self.saveGeometry()
            self.settings.setValue("geometry", geometry)
        except Exception:
            pass  # 保存に失敗した場合は無視
        
    def setup_ui(self):
        """UIセットアップ"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # ★★★ 修正: progress_containerの作成をメソッドの先頭に移動 ★★★
        # 呼び出し先の create_vtk_panel で使用されるため、先に定義する必要があります。
        self.progress_container = QWidget()
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(5, 3, 5, 5)
        progress_layout.setSpacing(3)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 12px; color: #1E8449; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #aaa; border-radius: 5px; text-align: center; font-weight: bold; height: 18px; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 4px; }
        """)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        self.progress_container.setVisible(False)
        # ★★★ 修正ここまで ★★★

        main_layout = QHBoxLayout(central_widget)
        
        # --- メインのスプリッター ---
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # --- 左右パネルの作成とスプリッターへの追加 ---
        left_scroll_area = QScrollArea()
        left_panel = self.create_control_panel()
        left_scroll_area.setWidget(left_panel)
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll_area.setMinimumWidth(280)
        self.main_splitter.addWidget(left_scroll_area)
        
        right_scroll_area = QScrollArea()
        right_scroll_area.setWidgetResizable(True)
        right_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_panel = self.create_vtk_panel()
        right_scroll_area.setWidget(right_panel)
        self.main_splitter.addWidget(right_scroll_area)
        
        self.main_splitter.setSizes([280, 1020])
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)

    def create_menu_bar(self):
        """アプリケーションのメニューバーを作成する"""
        # ヘルプウィンドウの参照を初期化
        self.help_window = None
        
        # QMainWindow標準のメニューバーを取得
        menu_bar = self.menuBar()
        # ウィンドウ内に表示（macOSでも常に表示）
        try:
            menu_bar.setNativeMenuBar(False)
        except Exception:
            pass

        # 「AFM Appearance」アクション（Real AFM image の左に配置）
        afm_appearance_action = QAction("AFM Appearance", self)
        afm_appearance_action.setToolTip("Open AFM appearance controls window\nAFM Appearance設定ウィンドウを開く")
        afm_appearance_action.triggered.connect(self.open_afm_appearance_window)
        menu_bar.addAction(afm_appearance_action)

        # 「Real AFM image」アクション（Help の左に配置）
        load_real_action = QAction("Real AFM image", self)
        load_real_action.setToolTip("Open Real AFM / Aligned image window\nReal AFM / Aligned の表示ウィンドウを開く")
        load_real_action.triggered.connect(self.open_real_afm_image_window)
        menu_bar.addAction(load_real_action)
        
        # 「Help」メニューを作成
        help_menu = menu_bar.addMenu("&Help")
        
        # 「View Help」アクションを作成し、クリックされたらshow_help_windowを呼び出す
        show_help_action = QAction("View Help...", self)
        show_help_action.setShortcut("F1")
        show_help_action.triggered.connect(self.show_help_window)
        help_menu.addAction(show_help_action)

    def show_help_window(self):
        """ヘルプウィンドウを作成して表示する"""
        # ウィンドウが既に開いている場合は、新しく作らずに最前面に表示
        if self.help_window is None or not self.help_window.isVisible():
            self.help_window = HelpWindow(parent=None)
            #self.help_window = HelpWindow(self)
            self.help_window.show()
        else:
            self.help_window.activateWindow()
            self.help_window.raise_()

    def _ensure_afm_appearance_window(self, show=False):
        """Create (if needed) and optionally show AFM Appearance window."""
        if self.afm_appearance_group is None:
            return

        if self.afm_appearance_window is None:
            win = QWidget(None)
            win.setWindowTitle("AFM Appearance")
            win.resize(420, 740)

            outer = QVBoxLayout(win)
            outer.setContentsMargins(6, 6, 6, 6)
            outer.setSpacing(6)

            scroll = QScrollArea(win)
            scroll.setWidgetResizable(True)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setWidget(self.afm_appearance_group)
            outer.addWidget(scroll)

            self.afm_appearance_window = win

        if show and self.afm_appearance_window is not None:
            self.afm_appearance_window.show()
            self.afm_appearance_window.raise_()
            self.afm_appearance_window.activateWindow()

    def open_afm_appearance_window(self):
        """Open AFM appearance controls window."""
        if self.afm_appearance_group is None:
            QMessageBox.information(self, "AFM Appearance", "AFM Appearance controls are not initialized yet.")
            return
        self._ensure_afm_appearance_window(show=True)

    def create_control_panel(self):
        """左側のコントロールパネル作成"""
        panel = QWidget()
        panel.setMinimumWidth(270)
        layout = QVBoxLayout(panel)
        layout.setSpacing(6) # 8から変更
        layout.setContentsMargins(8, 8, 8, 8) # 10から変更

        # ▼▼▼ 全体のフォントサイズを小さくするスタイルシートを追加 ▼▼▼
        panel.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
            }
            QLabel, QCheckBox, QPushButton, QComboBox, QDoubleSpinBox {
                font-size: 11px;
            }
        """)
        
        # File Import (統合: PDB/CIF/MRC)
        file_import_group = QGroupBox("File Import")
        file_import_layout = QVBoxLayout(file_import_group)
        
        self.import_btn = QPushButton("Import File...")
        self.import_btn.setMinimumHeight(35)
        self.import_btn.setToolTip("Load structure file (PDB/CIF/MRC) for AFM simulation\nAFMシミュレーション用の構造ファイル（PDB/CIF/MRC）を読み込み")
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.import_btn.clicked.connect(self.import_file)
        file_import_layout.addWidget(self.import_btn)

        # インポートされたファイル名の表示のみ（ドロップは PDB Structure 領域で受付）
        self.file_label = QLabel("File Name: (none)")
        self.file_label.setStyleSheet("color: #666; font-size: 12px;")
        file_import_layout.addWidget(self.file_label)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        file_import_layout.addWidget(self.progress_bar)

        layout.addWidget(file_import_group)
        
        # ★★★ Density Thresholdセクションを追加 ★★★
        self.mrc_group = QGroupBox("Density Threshold")
        mrc_layout = QGridLayout(self.mrc_group)

        self.mrc_threshold_label = QLabel(f"Value: {self.mrc_threshold:.2f}")
        mrc_layout.addWidget(self.mrc_threshold_label, 0, 0, 1, 2)

        self.mrc_threshold_slider = QSlider(Qt.Horizontal)
        self.mrc_threshold_slider.setRange(0, 100)
        self.mrc_threshold_slider.setValue(int(self.mrc_threshold * 100))
        # スライダーを動かしている最中はラベル更新のみ
        self.mrc_threshold_slider.valueChanged.connect(self.on_mrc_threshold_changed)
        # スライダーを離したときに再描画
        self.mrc_threshold_slider.sliderReleased.connect(self.on_mrc_threshold_released)
        mrc_layout.addWidget(self.mrc_threshold_slider, 1, 0, 1, 2)

        # Z軸フリップ用のチェックボックスを追加（デフォルトで有効）
        self.mrc_z_flip_check = QCheckBox("Flip Z-axis")
        self.mrc_z_flip_check.setChecked(True)  # デフォルトで有効
        self.mrc_z_flip_check.stateChanged.connect(self.on_mrc_z_flip_changed)
        self.mrc_z_flip_check.setToolTip("Toggle Z-axis flip for MRC data (default: enabled)")
        mrc_layout.addWidget(self.mrc_z_flip_check, 2, 0, 1, 2)

        self.mrc_group.setEnabled(False) # 最初は無効
        layout.addWidget(self.mrc_group)
        # ★★★ ここまで ★★★

        # 表示設定
        display_group = QGroupBox("Display Settings")
        display_layout = QGridLayout(display_group)
        
        # 表示スタイル
        display_layout.addWidget(QLabel("Style:"), 0, 0)
        self.style_combo = QComboBox()
        self.style_combo.addItems([
            "Ball & Stick", "Stick Only", "Spheres", "Points", "Wireframe", "Simple Cartoon", "Ribbon"
        ])
        self.style_combo.currentTextChanged.connect(self.update_display)
        display_layout.addWidget(self.style_combo, 0, 1)
        
        # カラーリング
        display_layout.addWidget(QLabel("Color:"), 1, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            "By Element", "By Chain", "Single Color", "By B-Factor"
        ])
        self.color_combo.currentTextChanged.connect(self.on_color_scheme_changed)
        display_layout.addWidget(self.color_combo, 1, 1)
        
        # 原子選択
        display_layout.addWidget(QLabel("Show:"), 2, 0)
        self.atom_combo = QComboBox()
        self.atom_combo.addItems(["All Atoms", "Heavy Atoms", "Backbone", "C", "N", "O"])
        self.atom_combo.currentTextChanged.connect(self.update_display)
        display_layout.addWidget(self.atom_combo, 2, 1)
        
        # サイズ
        display_layout.addWidget(QLabel("Size:"), 3, 0)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(10, 200)
        self.size_slider.setValue(100)
        self.size_slider.valueChanged.connect(self.update_display)
        display_layout.addWidget(self.size_slider, 3, 1)
        
        # 透明度
        display_layout.addWidget(QLabel("Opacity:"), 4, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.update_display)
        display_layout.addWidget(self.opacity_slider, 4, 1)
        
        # 品質設定
        display_layout.addWidget(QLabel("Quality:"), 5, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Fast", "Good", "High"])
        self.quality_combo.setCurrentText("Good")
        self.quality_combo.currentTextChanged.connect(self.update_display)
        display_layout.addWidget(self.quality_combo, 5, 1)

        # レンダラー選択（PyMOL/VTK）
        display_layout.addWidget(QLabel("Renderer:"), 6, 0)
        self.renderer_combo = QComboBox()
        # 初期は両方用意（後でsetup_pymolで更新）
        self.renderer_combo.addItems(["PyMOL (image)", "VTK (interactive)"])
        self.renderer_combo.currentTextChanged.connect(self.on_renderer_changed)
        display_layout.addWidget(self.renderer_combo, 6, 1)

        # 表面電荷（ESP）表示
        self.esp_check = QCheckBox("Electrostatics (ESP)")
        self.esp_check.setToolTip("Show electrostatic surface (Red-White-Blue)\n表面電荷（ESP）を表示")
        self.esp_check.toggled.connect(self.display_electrostatics)
        display_layout.addWidget(self.esp_check, 7, 0, 1, 2)
        
        layout.addWidget(display_group)
        
        # カラー・ライティング設定
        color_group = QGroupBox("Color & Lighting Settings")
        color_layout = QGridLayout(color_group)
        
        # 背景色設定
        color_layout.addWidget(QLabel("Background:"), 0, 0)
        self.bg_color_btn = QPushButton("Choose Color")
        self.bg_color_btn.setMinimumHeight(30)
        self.bg_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #191919;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                border-color: #777;
            }
        """)
        self.bg_color_btn.clicked.connect(self.choose_background_color)
        color_layout.addWidget(self.bg_color_btn, 0, 1)
        
        # 明るさ調整
        color_layout.addWidget(QLabel("Brightness:"), 1, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(20, 200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        color_layout.addWidget(self.brightness_slider, 1, 1)
        
        self.brightness_label = QLabel("100%")
        self.brightness_label.setMinimumWidth(40)
        color_layout.addWidget(self.brightness_label, 1, 2)
        
        # 単色モード用カラー選択
        color_layout.addWidget(QLabel("Single Color:"), 2, 0)
        self.single_color_btn = QPushButton("Choose Color")
        self.single_color_btn.setMinimumHeight(30)
        self.single_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #7FB3D3;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                border-color: #777;
            }
        """)
        self.single_color_btn.clicked.connect(self.choose_single_color)
        color_layout.addWidget(self.single_color_btn, 2, 1)
        
        # 環境光設定
        color_layout.addWidget(QLabel("Ambient:"), 3, 0)
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 50)
        self.ambient_slider.setValue(10)
        self.ambient_slider.valueChanged.connect(self.update_lighting)
        color_layout.addWidget(self.ambient_slider, 3, 1)
        
        self.ambient_label = QLabel("10%")
        self.ambient_label.setMinimumWidth(40)
        color_layout.addWidget(self.ambient_label, 3, 2)
        
        # スペキュラ設定
        color_layout.addWidget(QLabel("Specular:"), 4, 0)
        self.specular_slider = QSlider(Qt.Horizontal)
        self.specular_slider.setRange(0, 100)
        self.specular_slider.setValue(60)
        self.specular_slider.valueChanged.connect(self.update_material)
        color_layout.addWidget(self.specular_slider, 4, 1)
        
        self.specular_label = QLabel("60%")
        self.specular_label.setMinimumWidth(40)
        color_layout.addWidget(self.specular_label, 4, 2)
        
        # プリセットボタン
        preset_layout = QHBoxLayout()
        
        pymol_btn = QPushButton("PyMOL Style")
        pymol_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        pymol_btn.clicked.connect(self.apply_pymol_style)
        preset_layout.addWidget(pymol_btn)
        
        dark_btn = QPushButton("Dark Theme")
        dark_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        dark_btn.clicked.connect(self.apply_dark_theme)
        preset_layout.addWidget(dark_btn)
        
        color_layout.addLayout(preset_layout, 5, 0, 1, 3)
        
        layout.addWidget(color_group)


        # AFM探針設定
        tip_group = QGroupBox("AFM Tip Settings")
        tip_layout = QGridLayout(tip_group)

        # Row 0: Shape
        tip_layout.addWidget(QLabel("Shape:"), 0, 0)
        self.tip_shape_combo = QComboBox()
        self.tip_shape_combo.addItems(["Cone", "Sphere", "Paraboloid"])
        self.tip_shape_combo.setToolTip("AFM tip shape\nAFM探針の形状")
        self.tip_shape_combo.currentTextChanged.connect(self.update_tip_ui)
        tip_layout.addWidget(self.tip_shape_combo, 0, 1)

        # Row 1: Radius (of cone part)
        tip_layout.addWidget(QLabel("Radius (nm):"), 1, 0)
        self.tip_radius_spin = QDoubleSpinBox()
        self.tip_radius_spin.setRange(0.5, 30.0)
        self.tip_radius_spin.setValue(0.5)
        self.tip_radius_spin.setSingleStep(0.1)
        self.tip_radius_spin.setDecimals(1)
        self.tip_radius_spin.setToolTip("AFM tip radius in nanometers\nAFM探針の半径（ナノメートル）")
        self.tip_radius_spin.valueChanged.connect(self.tip_radius_value_changed)
        self.tip_radius_spin.editingFinished.connect(self.tip_radius_editing_finished)
        self.tip_radius_spin.keyPressEvent = self.tip_radius_key_press_event
        tip_layout.addWidget(self.tip_radius_spin, 1, 1)

        # Row 2: Radius of Minitip (for Sphere shape)
        self.minitip_label = QLabel("Radius of Minitip (nm):")
        tip_layout.addWidget(self.minitip_label, 2, 0)
        self.minitip_radius_spin = QDoubleSpinBox()
        self.minitip_radius_spin.setRange(0.1, 10.0)
        self.minitip_radius_spin.setValue(0.1)
        self.minitip_radius_spin.setSingleStep(0.1)
        self.minitip_radius_spin.setToolTip("Radius of minitip in nanometers\nミニチップの半径（ナノメートル）")
        self.minitip_radius_spin.setDecimals(1)
        self.minitip_radius_spin.valueChanged.connect(self.minitip_radius_value_changed)
        self.minitip_radius_spin.editingFinished.connect(self.minitip_radius_editing_finished)
        self.minitip_radius_spin.keyPressEvent = self.minitip_radius_key_press_event
        tip_layout.addWidget(self.minitip_radius_spin, 2, 1)

        # Row 3: Angle (for Cone/Sphere)
        self.tip_angle_label = QLabel("Angle (deg):")
        tip_layout.addWidget(self.tip_angle_label, 3, 0)
        self.tip_angle_spin = QDoubleSpinBox()
        self.tip_angle_spin.setRange(5.0, 35.0)
        self.tip_angle_spin.setValue(5)
        self.tip_angle_spin.setSingleStep(1.0)
        self.tip_angle_spin.valueChanged.connect(self.tip_angle_value_changed)
        self.tip_angle_spin.editingFinished.connect(self.tip_angle_editing_finished)
        self.tip_angle_spin.keyPressEvent = self.tip_angle_key_press_event
        tip_layout.addWidget(self.tip_angle_spin, 3, 1)
        
        # Row 4: Tip Info
        self.tip_info_label = QLabel("Tip Info: -")
        self.tip_info_label.setStyleSheet("""
            QLabel {
                font-size: 9px; color: #666; background-color: #f9f9f9;
                border: 1px solid #ddd; border-radius: 3px; padding: 3px;
            }
        """)
        self.tip_info_label.setWordWrap(True)
        tip_layout.addWidget(self.tip_info_label, 4, 0, 1, 2)

        layout.addWidget(tip_group)
        
        # 探針位置制御
        pos_group = QGroupBox("Tip Position Control")
        pos_layout = QGridLayout(pos_group)
        
        # X位置
        pos_layout.addWidget(QLabel("X (nm):"), 0, 0)
        self.tip_x_slider = QSlider(Qt.Horizontal)
        self.tip_x_slider.setRange(-50, 50)
        self.tip_x_slider.setValue(0)
        self.tip_x_slider.setToolTip("AFM tip X position in nanometers\nAFM探針のX位置（ナノメートル）")
        self.tip_x_slider.valueChanged.connect(self.update_tip_position)
        self.tip_x_slider.sliderPressed.connect(self.on_tip_slider_pressed)
        self.tip_x_slider.sliderReleased.connect(self.on_tip_slider_released)
        pos_layout.addWidget(self.tip_x_slider, 0, 1)
        self.tip_x_label = QLabel("0.0")
        self.tip_x_label.setMinimumWidth(30)
        pos_layout.addWidget(self.tip_x_label, 0, 2)
        
        # Y位置
        pos_layout.addWidget(QLabel("Y (nm):"), 1, 0)
        self.tip_y_slider = QSlider(Qt.Horizontal)
        self.tip_y_slider.setRange(-50, 50)
        self.tip_y_slider.setValue(0)
        self.tip_y_slider.setToolTip("AFM tip Y position in nanometers\nAFM探針のY位置（ナノメートル）")
        self.tip_y_slider.valueChanged.connect(self.update_tip_position)
        self.tip_y_slider.sliderPressed.connect(self.on_tip_slider_pressed)
        self.tip_y_slider.sliderReleased.connect(self.on_tip_slider_released)
        pos_layout.addWidget(self.tip_y_slider, 1, 1)
        self.tip_y_label = QLabel("0.0")
        self.tip_y_label.setMinimumWidth(30)
        pos_layout.addWidget(self.tip_y_label, 1, 2)
        
        # Z位置
        pos_layout.addWidget(QLabel("Z (nm):"), 2, 0)
        self.tip_z_slider = QSlider(Qt.Horizontal)
        self.tip_z_slider.setRange(10, 100)
        self.tip_z_slider.setValue(25)
        self.tip_z_slider.setToolTip("AFM tip Z position (height) in nanometers\nAFM探針のZ位置（高さ）（ナノメートル）")
        self.tip_z_slider.valueChanged.connect(self.update_tip_position)
        self.tip_z_slider.sliderPressed.connect(self.on_tip_slider_pressed)
        self.tip_z_slider.sliderReleased.connect(self.on_tip_slider_released)
        pos_layout.addWidget(self.tip_z_slider, 2, 1)
        self.tip_z_label = QLabel("5.0")
        self.tip_z_label.setMinimumWidth(30)
        pos_layout.addWidget(self.tip_z_label, 2, 2)
        
        layout.addWidget(pos_group)
        
        # シミュレーション設定
        sim_group = QGroupBox("AFM Simulation")
        sim_layout = QGridLayout(sim_group)
        
        # スキャンサイズ (X/Y)
        sim_layout.addWidget(QLabel("Scan Size X (nm):"), 0, 0)
        self.spinScanXNm = QDoubleSpinBox()
        self.spinScanXNm.setRange(1.0, 500.0)
        self.spinScanXNm.setValue(20.0)
        self.spinScanXNm.setDecimals(1)
        self.spinScanXNm.setToolTip("Scan area size X in nanometers\nスキャン領域のXサイズ（ナノメートル）")
        self.spinScanXNm.valueChanged.connect(self.scan_size_value_changed)
        self.spinScanXNm.valueChanged.connect(self._enforce_rectangle_lock_from_scan)
        self.spinScanXNm.editingFinished.connect(self.scan_size_editing_finished)
        self.spinScanXNm.keyPressEvent = lambda event: self.scan_size_key_press_event(self.spinScanXNm, event)
        sim_layout.addWidget(self.spinScanXNm, 0, 1)

        sim_layout.addWidget(QLabel("Scan Size Y (nm):"), 1, 0)
        self.spinScanYNm = QDoubleSpinBox()
        self.spinScanYNm.setRange(1.0, 500.0)
        self.spinScanYNm.setValue(20.0)
        self.spinScanYNm.setDecimals(1)
        self.spinScanYNm.setToolTip("Scan area size Y in nanometers\nスキャン領域のYサイズ（ナノメートル）")
        self.spinScanYNm.valueChanged.connect(self.scan_size_value_changed)
        self.spinScanYNm.valueChanged.connect(self._enforce_rectangle_lock_from_scan)
        self.spinScanYNm.editingFinished.connect(self.scan_size_editing_finished)
        self.spinScanYNm.keyPressEvent = lambda event: self.scan_size_key_press_event(self.spinScanYNm, event)
        sim_layout.addWidget(self.spinScanYNm, 1, 1)

        # 解像度 (Nx/Ny)
        sim_layout.addWidget(QLabel("Pixels X (Nx):"), 2, 0)
        self.spinNx = QSpinBox()
        self.spinNx.setRange(8, 2048)
        self.spinNx.setValue(64)
        self.spinNx.setToolTip("Number of pixels in X direction\nX方向のピクセル数")
        self.spinNx.valueChanged.connect(self.on_resolution_changed)
        self.spinNx.valueChanged.connect(self._enforce_rectangle_lock_from_resolution)
        sim_layout.addWidget(self.spinNx, 2, 1)

        sim_layout.addWidget(QLabel("Pixels Y (Ny):"), 3, 0)
        self.spinNy = QSpinBox()
        self.spinNy.setRange(8, 2048)
        self.spinNy.setValue(64)
        self.spinNy.setToolTip("Number of pixels in Y direction\nY方向のピクセル数")
        self.spinNy.valueChanged.connect(self.on_resolution_changed)
        self.spinNy.valueChanged.connect(self._enforce_rectangle_lock_from_resolution)
        sim_layout.addWidget(self.spinNy, 3, 1)

        # 互換用：resolution_combo (Nx=Nyを一括設定するショートカット)
        sim_layout.addWidget(QLabel("Quick Res:"), 4, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["32x32", "64x64", "128x128", "256x256"])
        self.resolution_combo.setCurrentText("64x64")
        self.resolution_combo.setToolTip("Quick setting for symmetric resolution\n対称解像度の一括設定")
        self.resolution_combo.currentTextChanged.connect(self.on_resolution_combo_changed)
        sim_layout.addWidget(self.resolution_combo, 4, 1)

        # Rectangle lock: force Y values to match X (default ON)
        self.rectangle_check = QCheckBox("Rctangle (force Y = X)")
        self.rectangle_check.setToolTip(
            "When enabled, forces Scan Size Y = Scan Size X and Ny = Nx.\n"
            "Disable to allow rectangular scans.\n"
            "有効時、Scan Size Y と Ny を X 側と同一に固定します。"
        )
        self.rectangle_check.toggled.connect(self.on_rectangle_lock_toggled)
        try:
            self.rectangle_check.blockSignals(True)
            self.rectangle_check.setChecked(True)
        finally:
            self.rectangle_check.blockSignals(False)
        sim_layout.addWidget(self.rectangle_check, 5, 0, 1, 2)
        # Apply default lock state now that widgets exist
        self._apply_rectangle_lock(enforce_values=True)

        # Sync to Real button
        self.btnSyncToReal = QPushButton("Sync Sim Params to Real AFM")
        self.btnSyncToReal.setToolTip("Synchronize simulator scan parameters to the loaded Real AFM metadata")
        self.btnSyncToReal.clicked.connect(self.sync_sim_params_to_real)
        sim_layout.addWidget(self.btnSyncToReal, 6, 0, 1, 2)

        self.interactive_update_check = QCheckBox("Interactive Update")
        self.interactive_update_check.setToolTip("Automatically update simulation when parameters change\nパラメータ変更時に自動でシミュレーションを実行")
        self.interactive_update_check.toggled.connect(self.handle_interactive_update_toggle)
        # Default ON (historical default behavior): enable without running simulation yet.
        try:
            self.interactive_update_check.blockSignals(True)
            self.interactive_update_check.setChecked(True)
        finally:
            self.interactive_update_check.blockSignals(False)
        sim_layout.addWidget(self.interactive_update_check, 7, 0, 1, 2)
        
        # Consider VDW check
        self.use_vdw_check = QCheckBox("Consider atom size (vdW)")
        self.use_vdw_check.setToolTip("Treat atoms as spheres with van der Waals radii instead of points\n原子を点ではなくファンデルワールス半径を持つ球として扱う")
        self.use_vdw_check.toggled.connect(self.trigger_interactive_simulation)
        sim_layout.addWidget(self.use_vdw_check, 8, 0, 1, 2)


        # シミュレーション実行
        self.simulate_btn = QPushButton("Run AFM Simulation")
        self.simulate_btn.setMinimumHeight(40)
        self.simulate_btn.setToolTip("Run AFM simulation with current settings\n現在の設定でAFMシミュレーションを実行")
        self.simulate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.simulate_btn.clicked.connect(self.run_simulation)
        self.simulate_btn.setEnabled(False)
        sim_layout.addWidget(self.simulate_btn, 9, 0, 1, 2)
        
        layout.addWidget(sim_group)

        # AFM Appearance (filters + noise/artifacts)
        appearance_group = QGroupBox("AFM Appearance")
        appearance_layout = QVBoxLayout(appearance_group)
        appearance_layout.setSpacing(6)

        # (A) Low-pass filter (moved from AFM Simulation)
        lowpass_layout = QGridLayout()
        self.apply_filter_check = QCheckBox("Apply Low-pass Filter")
        self.apply_filter_check.setToolTip(
            "Apply FFT low-pass filter to match experimental resolution\n"
            "FFTローパスフィルターを適用して実験解像度に合わせる"
        )
        lowpass_layout.addWidget(self.apply_filter_check, 0, 0, 1, 2)

        lowpass_layout.addWidget(QLabel("Cutoff Wavelength (nm):"), 1, 0)
        self.filter_cutoff_spin = QDoubleSpinBox()
        self.filter_cutoff_spin.setRange(0.1, 20.0)
        self.filter_cutoff_spin.setValue(2.0)
        self.filter_cutoff_spin.setDecimals(1)
        self.filter_cutoff_spin.setSingleStep(0.1)
        self.filter_cutoff_spin.setToolTip("Cutoff wavelength for low-pass filter\nローパスフィルターのカットオフ波長")
        lowpass_layout.addWidget(self.filter_cutoff_spin, 1, 1)

        self.apply_filter_check.toggled.connect(self.filter_cutoff_spin.setEnabled)
        self.apply_filter_check.toggled.connect(self.process_and_display_all_images)
        self.filter_cutoff_spin.valueChanged.connect(self.start_filter_update_timer)
        self.filter_cutoff_spin.setEnabled(False)

        appearance_layout.addLayout(lowpass_layout)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        appearance_layout.addWidget(sep1)

        # (B) Physical Noise / Artifacts
        noise_header = QLabel("Physical Noise / Artifacts")
        noise_header.setStyleSheet("font-weight: bold;")
        appearance_layout.addWidget(noise_header)

        noise_layout = QGridLayout()
        row = 0

        self.chkNoiseEnable = QCheckBox("Enable Physical Noise")
        noise_layout.addWidget(self.chkNoiseEnable, row, 0, 1, 2)
        row += 1

        self.chkUseNoiseSeed = QCheckBox("Use fixed seed")
        noise_layout.addWidget(self.chkUseNoiseSeed, row, 0, 1, 2)
        row += 1

        noise_layout.addWidget(QLabel("Seed:"), row, 0)
        self.spinNoiseSeed = QSpinBox()
        self.spinNoiseSeed.setRange(0, 2_147_483_647)
        self.spinNoiseSeed.setValue(42)
        noise_layout.addWidget(self.spinNoiseSeed, row, 1)
        row += 1

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        noise_layout.addWidget(sep2, row, 0, 1, 2)
        row += 1

        # Height Noise
        self.chkHeightNoise = QCheckBox("Height Noise")
        noise_layout.addWidget(self.chkHeightNoise, row, 0, 1, 2)
        row += 1

        noise_layout.addWidget(QLabel("sigma (nm):"), row, 0)
        self.spinHeightNoiseSigmaNm = QDoubleSpinBox()
        self.spinHeightNoiseSigmaNm.setRange(0.0, 5.0)
        self.spinHeightNoiseSigmaNm.setValue(0.1)
        self.spinHeightNoiseSigmaNm.setDecimals(3)
        self.spinHeightNoiseSigmaNm.setSingleStep(0.01)
        noise_layout.addWidget(self.spinHeightNoiseSigmaNm, row, 1)
        row += 1

        # Line Noise
        self.chkLineNoise = QCheckBox("Line Noise")
        noise_layout.addWidget(self.chkLineNoise, row, 0, 1, 2)
        row += 1

        noise_layout.addWidget(QLabel("sigma_line (nm):"), row, 0)
        self.spinLineNoiseSigmaNm = QDoubleSpinBox()
        self.spinLineNoiseSigmaNm.setRange(0.0, 5.0)
        self.spinLineNoiseSigmaNm.setValue(0.05)
        self.spinLineNoiseSigmaNm.setDecimals(3)
        self.spinLineNoiseSigmaNm.setSingleStep(0.01)
        noise_layout.addWidget(self.spinLineNoiseSigmaNm, row, 1)
        row += 1

        noise_layout.addWidget(QLabel("mode:"), row, 0)
        self.comboLineNoiseMode = QComboBox()
        self.comboLineNoiseMode.addItems(["offset", "rw"])
        noise_layout.addWidget(self.comboLineNoiseMode, row, 1)
        row += 1

        # Drift
        self.chkDrift = QCheckBox("Drift")
        noise_layout.addWidget(self.chkDrift, row, 0, 1, 2)
        row += 1

        noise_layout.addWidget(QLabel("vx (nm/line):"), row, 0)
        self.spinDriftVxNmPerLine = QDoubleSpinBox()
        self.spinDriftVxNmPerLine.setRange(-5.0, 5.0)
        self.spinDriftVxNmPerLine.setValue(0.0)
        self.spinDriftVxNmPerLine.setDecimals(3)
        self.spinDriftVxNmPerLine.setSingleStep(0.01)
        noise_layout.addWidget(self.spinDriftVxNmPerLine, row, 1)
        row += 1

        noise_layout.addWidget(QLabel("vy (nm/line):"), row, 0)
        self.spinDriftVyNmPerLine = QDoubleSpinBox()
        self.spinDriftVyNmPerLine.setRange(-5.0, 5.0)
        self.spinDriftVyNmPerLine.setValue(0.0)
        self.spinDriftVyNmPerLine.setDecimals(3)
        self.spinDriftVyNmPerLine.setSingleStep(0.01)
        noise_layout.addWidget(self.spinDriftVyNmPerLine, row, 1)
        row += 1

        noise_layout.addWidget(QLabel("jitter (nm/line):"), row, 0)
        self.spinDriftJitterNmPerLine = QDoubleSpinBox()
        self.spinDriftJitterNmPerLine.setRange(0.0, 5.0)
        self.spinDriftJitterNmPerLine.setValue(0.0)
        self.spinDriftJitterNmPerLine.setDecimals(3)
        self.spinDriftJitterNmPerLine.setSingleStep(0.01)
        noise_layout.addWidget(self.spinDriftJitterNmPerLine, row, 1)
        row += 1

        # Feedback/Scan artifacts
        self.chkFeedbackLag = QCheckBox("Feedback Lag")
        noise_layout.addWidget(self.chkFeedbackLag, row, 0, 1, 2)
        row += 1

        noise_layout.addWidget(QLabel("mode:"), row, 0)
        self.comboFeedbackMode = QComboBox()
        self.comboFeedbackMode.addItems(["linear_lag", "tapping_parachute"])
        noise_layout.addWidget(self.comboFeedbackMode, row, 1)
        row += 1

        noise_layout.addWidget(QLabel("Scan Direction:"), row, 0)
        self.comboScanDirection = QComboBox()
        self.comboScanDirection.addItem("Left -> Right", "L2R")
        self.comboScanDirection.addItem("Right -> Left", "R2L")
        self.comboScanDirection.setCurrentIndex(0)
        noise_layout.addWidget(self.comboScanDirection, row, 1)
        row += 1

        noise_layout.addWidget(QLabel("tau (lines):"), row, 0)
        self.spinLagTauLines = QDoubleSpinBox()
        self.spinLagTauLines.setRange(0.1, 100.0)
        self.spinLagTauLines.setValue(2.0)
        self.spinLagTauLines.setDecimals(2)
        self.spinLagTauLines.setSingleStep(0.1)
        noise_layout.addWidget(self.spinLagTauLines, row, 1)
        row += 1

        # Tapping parachute parameters (mode-specific)
        noise_layout.addWidget(QLabel("tap drop (nm):"), row, 0)
        self.spinTapDropThresholdNm = QDoubleSpinBox()
        self.spinTapDropThresholdNm.setRange(0.0, 10.0)
        self.spinTapDropThresholdNm.setValue(1.0)
        self.spinTapDropThresholdNm.setDecimals(2)
        self.spinTapDropThresholdNm.setSingleStep(0.1)
        noise_layout.addWidget(self.spinTapDropThresholdNm, row, 1)
        row += 1

        noise_layout.addWidget(QLabel("tap tau_track (lines):"), row, 0)
        self.spinTapTauTrackLines = QDoubleSpinBox()
        self.spinTapTauTrackLines.setRange(0.1, 100.0)
        self.spinTapTauTrackLines.setValue(2.0)
        self.spinTapTauTrackLines.setDecimals(2)
        self.spinTapTauTrackLines.setSingleStep(0.1)
        noise_layout.addWidget(self.spinTapTauTrackLines, row, 1)
        row += 1

        noise_layout.addWidget(QLabel("tap tau_para (lines):"), row, 0)
        self.spinTapTauParachuteLines = QDoubleSpinBox()
        self.spinTapTauParachuteLines.setRange(0.1, 200.0)
        self.spinTapTauParachuteLines.setValue(15.0)
        self.spinTapTauParachuteLines.setDecimals(2)
        self.spinTapTauParachuteLines.setSingleStep(0.5)
        noise_layout.addWidget(self.spinTapTauParachuteLines, row, 1)
        row += 1

        noise_layout.addWidget(QLabel("tap release (nm):"), row, 0)
        self.spinTapReleaseThresholdNm = QDoubleSpinBox()
        self.spinTapReleaseThresholdNm.setRange(0.0, 10.0)
        self.spinTapReleaseThresholdNm.setValue(0.3)
        self.spinTapReleaseThresholdNm.setDecimals(2)
        self.spinTapReleaseThresholdNm.setSingleStep(0.05)
        noise_layout.addWidget(self.spinTapReleaseThresholdNm, row, 1)
        row += 1

        appearance_layout.addLayout(noise_layout)
        # AFM Appearance is hosted in a dedicated window opened from the menu.
        self.afm_appearance_group = appearance_group

        # Noise/appearance UI state + updates
        self.chkNoiseEnable.toggled.connect(self._update_noise_ui_states)
        self.chkUseNoiseSeed.toggled.connect(self._update_noise_ui_states)
        self.chkHeightNoise.toggled.connect(self._update_noise_ui_states)
        self.chkLineNoise.toggled.connect(self._update_noise_ui_states)
        self.chkDrift.toggled.connect(self._update_noise_ui_states)
        self.chkFeedbackLag.toggled.connect(self._update_noise_ui_states)
        self.comboFeedbackMode.currentTextChanged.connect(self._update_noise_ui_states)

        for w in (
            self.chkNoiseEnable, self.chkUseNoiseSeed, self.spinNoiseSeed,
            self.chkHeightNoise, self.spinHeightNoiseSigmaNm,
            self.chkLineNoise, self.spinLineNoiseSigmaNm, self.comboLineNoiseMode,
            self.chkDrift, self.spinDriftVxNmPerLine, self.spinDriftVyNmPerLine, self.spinDriftJitterNmPerLine,
            self.chkFeedbackLag, self.comboFeedbackMode, self.comboScanDirection, self.spinLagTauLines,
            self.spinTapDropThresholdNm, self.spinTapTauTrackLines, self.spinTapTauParachuteLines,
            self.spinTapReleaseThresholdNm,
        ):
            try:
                if isinstance(w, QComboBox):
                    w.currentTextChanged.connect(self.process_and_display_all_images)
                else:
                    w.toggled.connect(self.process_and_display_all_images) if isinstance(w, QCheckBox) else w.valueChanged.connect(self.process_and_display_all_images)
            except Exception:
                pass

        self._update_noise_ui_states()
        
        # 表示制御
        view_group = QGroupBox("View Control")
        view_layout = QVBoxLayout(view_group)
        
        self.show_molecule_check = QCheckBox("Show Molecule")
        self.show_molecule_check.setChecked(True)
        self.show_molecule_check.toggled.connect(self.toggle_molecule_visibility)
        view_layout.addWidget(self.show_molecule_check)
        
        self.show_tip_check = QCheckBox("Show AFM Tip")
        self.show_tip_check.setChecked(True)
        self.show_tip_check.toggled.connect(self.toggle_tip_visibility)
        view_layout.addWidget(self.show_tip_check)
        
        self.show_bonds_check = QCheckBox("Show Bonds")
        self.show_bonds_check.setChecked(True)
        self.show_bonds_check.toggled.connect(self.toggle_bonds_visibility)
        view_layout.addWidget(self.show_bonds_check)
        
        reset_view_btn = QPushButton("Reset View")
        reset_view_btn.setToolTip("Reset camera to default view\nカメラをデフォルトビューにリセット")
        reset_view_btn.clicked.connect(self.reset_camera)
        view_layout.addWidget(reset_view_btn)
        
        layout.addWidget(view_group)

        #self.update_tip_ui(self.tip_shape_combo.currentText())
        
        return panel
        
    
    def update_tip_ui(self, shape):
        """探針設定UIの表示を、選択された形状に応じて更新する"""
        shape = shape.lower()
        
        is_sphere = (shape == "sphere")
        is_cone = (shape == "cone")
        
        # Minitip Radius widgets visibility
        self.minitip_label.setVisible(is_sphere)
        self.minitip_radius_spin.setVisible(is_sphere)
        
        # Angle widgets visibility/enabled state
        angle_is_relevant = is_cone or is_sphere
        self.tip_angle_label.setEnabled(angle_is_relevant)
        self.tip_angle_spin.setEnabled(angle_is_relevant)
        
        # Trigger a tip redraw
        self.update_tip()

    def _update_noise_ui_states(self):
        """Enable/disable noise/appearance sub-controls based on master toggles."""
        if not hasattr(self, 'chkNoiseEnable'):
            return
        noise_on = self.chkNoiseEnable.isChecked()

        # Seed control
        if hasattr(self, 'chkUseNoiseSeed'):
            self.chkUseNoiseSeed.setEnabled(noise_on)
        if hasattr(self, 'spinNoiseSeed') and hasattr(self, 'chkUseNoiseSeed'):
            self.spinNoiseSeed.setEnabled(noise_on and self.chkUseNoiseSeed.isChecked())

        # Height noise
        if hasattr(self, 'chkHeightNoise'):
            self.chkHeightNoise.setEnabled(noise_on)
        if hasattr(self, 'spinHeightNoiseSigmaNm') and hasattr(self, 'chkHeightNoise'):
            self.spinHeightNoiseSigmaNm.setEnabled(noise_on and self.chkHeightNoise.isChecked())

        # Line noise
        if hasattr(self, 'chkLineNoise'):
            self.chkLineNoise.setEnabled(noise_on)
        if hasattr(self, 'spinLineNoiseSigmaNm') and hasattr(self, 'chkLineNoise'):
            self.spinLineNoiseSigmaNm.setEnabled(noise_on and self.chkLineNoise.isChecked())
        if hasattr(self, 'comboLineNoiseMode') and hasattr(self, 'chkLineNoise'):
            self.comboLineNoiseMode.setEnabled(noise_on and self.chkLineNoise.isChecked())

        # Drift
        if hasattr(self, 'chkDrift'):
            self.chkDrift.setEnabled(noise_on)
        drift_on = noise_on and hasattr(self, 'chkDrift') and self.chkDrift.isChecked()
        if hasattr(self, 'spinDriftVxNmPerLine'):
            self.spinDriftVxNmPerLine.setEnabled(drift_on)
        if hasattr(self, 'spinDriftVyNmPerLine'):
            self.spinDriftVyNmPerLine.setEnabled(drift_on)
        if hasattr(self, 'spinDriftJitterNmPerLine'):
            self.spinDriftJitterNmPerLine.setEnabled(drift_on)

        # Feedback artifacts (single scan direction)
        if hasattr(self, 'chkFeedbackLag'):
            self.chkFeedbackLag.setEnabled(noise_on)
        lag_on = noise_on and hasattr(self, 'chkFeedbackLag') and self.chkFeedbackLag.isChecked()

        mode = "linear_lag"
        if hasattr(self, 'comboFeedbackMode'):
            self.comboFeedbackMode.setEnabled(lag_on)
            if lag_on:
                mode = self.comboFeedbackMode.currentText()

        if hasattr(self, 'comboScanDirection'):
            self.comboScanDirection.setEnabled(lag_on)
        if hasattr(self, 'spinLagTauLines'):
            self.spinLagTauLines.setEnabled(lag_on)

        # Tapping parachute params
        tap_on = lag_on and (mode == "tapping_parachute")
        if hasattr(self, 'spinTapDropThresholdNm'):
            self.spinTapDropThresholdNm.setEnabled(tap_on)
        if hasattr(self, 'spinTapTauTrackLines'):
            self.spinTapTauTrackLines.setEnabled(tap_on)
        if hasattr(self, 'spinTapTauParachuteLines'):
            self.spinTapTauParachuteLines.setEnabled(tap_on)
        if hasattr(self, 'spinTapReleaseThresholdNm'):
            self.spinTapReleaseThresholdNm.setEnabled(tap_on)

    def apply_noise_artifacts(self, height_nm, pixel_x_nm, pixel_y_nm):
        """Apply physical noise/artifacts to height map in nm."""
        if not hasattr(self, 'chkNoiseEnable') or not self.chkNoiseEnable.isChecked():
            return height_nm

        if height_nm is None:
            return height_nm

        height = np.array(height_nm, dtype=float, copy=True)
        ny, nx = height.shape

        seed = None
        if hasattr(self, 'chkUseNoiseSeed') and self.chkUseNoiseSeed.isChecked():
            seed = int(self.spinNoiseSeed.value()) if hasattr(self, 'spinNoiseSeed') else 0
        rng = np.random.default_rng(seed)

        # Height noise
        if hasattr(self, 'chkHeightNoise') and self.chkHeightNoise.isChecked():
            sigma = float(self.spinHeightNoiseSigmaNm.value()) if hasattr(self, 'spinHeightNoiseSigmaNm') else 0.0
            if sigma > 0:
                height += rng.normal(0.0, sigma, size=height.shape)

        # Line noise
        if hasattr(self, 'chkLineNoise') and self.chkLineNoise.isChecked():
            sigma = float(self.spinLineNoiseSigmaNm.value()) if hasattr(self, 'spinLineNoiseSigmaNm') else 0.0
            if sigma > 0:
                mode = self.comboLineNoiseMode.currentText() if hasattr(self, 'comboLineNoiseMode') else "offset"
                if mode == "rw":
                    b = np.zeros(ny, dtype=float)
                    b[0] = rng.normal(0.0, sigma)
                    for y in range(1, ny):
                        b[y] = b[y - 1] + rng.normal(0.0, sigma * 0.2)
                else:
                    b = rng.normal(0.0, sigma, size=ny)
                height += b[:, None]

        # Drift (integer pixel shift per line)
        if hasattr(self, 'chkDrift') and self.chkDrift.isChecked():
            vx = float(self.spinDriftVxNmPerLine.value()) if hasattr(self, 'spinDriftVxNmPerLine') else 0.0
            vy = float(self.spinDriftVyNmPerLine.value()) if hasattr(self, 'spinDriftVyNmPerLine') else 0.0
            jitter = float(self.spinDriftJitterNmPerLine.value()) if hasattr(self, 'spinDriftJitterNmPerLine') else 0.0
            
            shifted = np.zeros_like(height)
            for y in range(ny):
                dx_nm = vx * y + rng.normal(0.0, jitter)
                dy_nm = vy * y + rng.normal(0.0, jitter)
                dx_px = int(round(dx_nm / pixel_x_nm)) if pixel_x_nm > 0 else 0
                dy_px = int(round(dy_nm / pixel_y_nm)) if pixel_y_nm > 0 else 0
                y_src = min(max(y + dy_px, 0), ny - 1)
                line = height[y_src]
                if dx_px != 0:
                    line = np.roll(line, dx_px)
                height[y] = line
                shifted[y] = line
            height = shifted

        # Feedback artifacts (single scan direction)
        if hasattr(self, 'chkFeedbackLag') and self.chkFeedbackLag.isChecked():
            mode = self.comboFeedbackMode.currentText() if hasattr(self, 'comboFeedbackMode') else "linear_lag"
            scan_dir = "L2R"
            if hasattr(self, 'comboScanDirection'):
                scan_dir = self.comboScanDirection.currentData() or self.comboScanDirection.currentText()
            reverse = (scan_dir == "R2L")

            def apply_iir_1d(line, alpha, reverse=False):
                if reverse:
                    line = line[::-1]
                out = np.empty_like(line, dtype=float)
                out[0] = line[0]
                for i in range(1, line.size):
                    out[i] = alpha * out[i - 1] + (1.0 - alpha) * line[i]
                if reverse:
                    out = out[::-1]
                return out

            def apply_tapping_parachute_1d(line, drop_th, tau_track, tau_para, rel_th, reverse=False):
                if reverse:
                    line = line[::-1]
                out = np.empty_like(line, dtype=float)
                out[0] = line[0]
                state = "TRACK"
                for i in range(1, line.size):
                    drop = line[i] - line[i - 1]
                    if drop < -drop_th:
                        state = "PARACHUTE"
                    tau = tau_track if state == "TRACK" else tau_para
                    alpha = math.exp(-1.0 / max(tau, 1e-6))
                    out[i] = alpha * out[i - 1] + (1.0 - alpha) * line[i]
                    if state == "PARACHUTE" and (out[i] - line[i]) < rel_th:
                        state = "TRACK"
                if reverse:
                    out = out[::-1]
                return out

            if mode == "tapping_parachute":
                drop_th = float(self.spinTapDropThresholdNm.value()) if hasattr(self, 'spinTapDropThresholdNm') else 1.0
                tau_track = float(self.spinTapTauTrackLines.value()) if hasattr(self, 'spinTapTauTrackLines') else 2.0
                tau_para = float(self.spinTapTauParachuteLines.value()) if hasattr(self, 'spinTapTauParachuteLines') else 15.0
                rel_th = float(self.spinTapReleaseThresholdNm.value()) if hasattr(self, 'spinTapReleaseThresholdNm') else 0.3
                for y in range(ny):
                    height[y] = apply_tapping_parachute_1d(
                        height[y], drop_th, tau_track, tau_para, rel_th, reverse=reverse
                    )
            else:
                # linear_lag (default)
                tau = float(self.spinLagTauLines.value()) if hasattr(self, 'spinLagTauLines') else 1.0
                alpha = math.exp(-1.0 / max(tau, 1e-6))
                for y in range(ny):
                    height[y] = apply_iir_1d(height[y], alpha, reverse=reverse)

        return height

    def _map_asd_scan_direction(self, val):
        """Map ASD scan direction to L2R/R2L."""
        try:
            v = int(val)
        except Exception:
            v = 0
        return "R2L" if v == 1 else "L2R"

    def _rebuild_real_afm_active_from_full(self):
        """Build active real_afm_nm/meta from full image and optional ROI."""
        full = getattr(self, 'real_afm_nm_full', None)
        meta_full = getattr(self, 'real_meta_full', None)
        if full is None or not meta_full:
            return False

        full_arr = np.asarray(full)
        if full_arr.ndim != 2:
            return False
        full_h, full_w = full_arr.shape
        if full_h <= 0 or full_w <= 0:
            return False

        roi = getattr(self, 'real_afm_roi_px', None)
        if roi is None:
            self.real_afm_nm = full_arr
            self.real_meta = dict(meta_full)
            return True

        try:
            x0, y0, x1, y1 = [int(v) for v in roi]
        except Exception:
            self.real_afm_roi_px = None
            self.real_afm_nm = full_arr
            self.real_meta = dict(meta_full)
            return True

        x0 = max(0, min(x0, full_w))
        x1 = max(0, min(x1, full_w))
        y0 = max(0, min(y0, full_h))
        y1 = max(0, min(y1, full_h))
        if x1 <= x0 or y1 <= y0 or (x1 - x0) < 8 or (y1 - y0) < 8:
            self.real_afm_roi_px = None
            self.real_afm_nm = full_arr
            self.real_meta = dict(meta_full)
            return True

        crop = full_arr[y0:y1, x0:x1]
        if crop.size == 0:
            self.real_afm_roi_px = None
            self.real_afm_nm = full_arr
            self.real_meta = dict(meta_full)
            return True

        nx_full = int(meta_full.get('nx', full_w) or full_w)
        ny_full = int(meta_full.get('ny', full_h) or full_h)
        sx_full = float(meta_full.get('scan_x_nm', 0.0) or 0.0)
        sy_full = float(meta_full.get('scan_y_nm', 0.0) or 0.0)
        w = int(x1 - x0)
        h = int(y1 - y0)

        # Convert ROI pixel window to physical scan size.
        sx = (sx_full * (w / float(max(nx_full, 1)))) if sx_full > 0 else 0.0
        sy = (sy_full * (h / float(max(ny_full, 1)))) if sy_full > 0 else 0.0
        if sx <= 0 and sy > 0:
            sx = sy
        if sy <= 0 and sx > 0:
            sy = sx

        meta = dict(meta_full)
        meta['nx'] = w
        meta['ny'] = h
        meta['scan_x_nm'] = sx
        meta['scan_y_nm'] = sy
        meta['nm_per_pixel_x'] = (sx / float(max(w, 1))) if sx > 0 else 0.0
        meta['nm_per_pixel_y'] = (sy / float(max(h, 1))) if sy > 0 else 0.0

        self.real_afm_nm = crop
        self.real_meta = meta
        return True

    def on_real_afm_roi_selected(self, x0, y0, x1, y1):
        """Apply ROI selected on displayed Real AFM image (left-drag)."""
        full = getattr(self, 'real_afm_nm_full', None)
        if full is None:
            return
        full_arr = np.asarray(full)
        if full_arr.ndim != 2:
            return
        h, w = full_arr.shape
        if h <= 0 or w <= 0:
            return

        # Displayed image is vertically flipped, so convert back to raw coordinates.
        x0 = max(0, min(int(x0), w))
        x1 = max(0, min(int(x1), w))
        y0 = max(0, min(int(y0), h))
        y1 = max(0, min(int(y1), h))
        rx0, rx1 = min(x0, x1), max(x0, x1)
        ry0 = max(0, min(h, h - max(y0, y1)))
        ry1 = max(0, min(h, h - min(y0, y1)))
        if rx1 <= rx0 or ry1 <= ry0:
            return

        self.real_afm_roi_px = (int(rx0), int(ry0), int(rx1), int(ry1))
        if not self._rebuild_real_afm_active_from_full():
            return

        # ROI change invalidates current alignment until re-simulated.
        self.sim_aligned_nm = None
        self.show_real_afm()
        self._update_real_afm_frame_controls()
        self.sync_sim_params_to_real()

    def _clear_real_afm_roi_overlay(self):
        frame = getattr(self, 'real_afm_window_real_frame', None)
        if frame is None:
            return
        try:
            view = frame.findChild(_AspectPixmapView, "afm_image_view")
            if view is not None:
                view.clearRoiOverlay()
        except Exception:
            pass

    def clear_real_afm_roi(self):
        """Clear ROI and restore full Real AFM view."""
        if getattr(self, 'real_afm_nm_full', None) is None:
            return
        self.real_afm_roi_px = None
        self._clear_real_afm_roi_overlay()
        if not self._rebuild_real_afm_active_from_full():
            return
        self.sim_aligned_nm = None
        self.show_real_afm()
        self._update_real_afm_frame_controls()
        self.sync_sim_params_to_real()

    def on_load_real_asd(self):
        """Load Real AFM ASD via file dialog."""
        initial_dir = self.last_import_dir if hasattr(self, 'last_import_dir') and self.last_import_dir else ''
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Real AFM (ASD)", initial_dir,
            "ASD files (*.asd);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog
        )
        if not path:
            return
        self.load_real_asd_file(path, sync=True)

    def open_real_afm_image_window(self):
        """Open the Real AFM / Aligned image window (no file dialog)."""
        self._ensure_real_afm_window(show=True)

    def _update_real_afm_frame_controls(self):
        """Update frame slider/label based on current ASD state."""
        slider = getattr(self, 'real_afm_frame_slider', None)
        label = getattr(self, 'real_afm_frame_label', None)
        if slider is None or label is None:
            self._update_sim_aligned_info_label()
            return

        total = int(getattr(self, 'real_asd_frame_num', 0) or 0)
        idx = int(getattr(self, 'real_asd_frame_index', 0) or 0)
        if total <= 0:
            slider.setEnabled(False)
            slider.setRange(0, 0)
            slider.setValue(0)
            label.setText("Frame: -")
            self._real_afm_pending_frame_index = None
            self._update_sim_aligned_info_label()
            return

        idx = max(0, min(idx, total - 1))
        try:
            slider.blockSignals(True)
            slider.setEnabled(True)
            slider.setRange(0, total - 1)
            slider.setValue(idx)
        finally:
            slider.blockSignals(False)
        label.setText(f"Frame: {idx + 1} / {total}")
        self._real_afm_pending_frame_index = idx
        self._update_sim_aligned_info_label()

    def _update_sim_aligned_info_label(self):
        """Update Sim Aligned metadata row (scan size/pixel size)."""
        info_label = getattr(self, 'real_afm_sim_info_label', None)
        if info_label is None:
            return

        meta = getattr(self, 'real_meta', None) or {}
        try:
            sx = float(meta.get('scan_x_nm', 0.0))
            sy = float(meta.get('scan_y_nm', 0.0))
            nx = int(meta.get('nx', 0))
            ny = int(meta.get('ny', 0))
        except Exception:
            sx, sy, nx, ny = 0.0, 0.0, 0, 0

        if sx > 0 and sy > 0 and nx > 0 and ny > 0:
            px = sx / float(nx)
            py = sy / float(ny)
            info_label.setText(
                f"Scan X/Y: {sx:.1f} / {sy:.1f} nm    Pixel X/Y: {px:.3f} / {py:.3f} nm/px"
            )
        else:
            info_label.setText("Scan X/Y: - / - nm    Pixel X/Y: - / - nm/px")

    def _schedule_real_afm_frame_load(self, idx: int, *, immediate: bool = False):
        """Debounced frame loading for slider moves."""
        if not getattr(self, 'real_asd_path', None):
            return
        total = int(getattr(self, 'real_asd_frame_num', 0) or 0)
        if total <= 0:
            return
        idx = max(0, min(int(idx), total - 1))
        self._real_afm_pending_frame_index = idx

        if self._real_afm_frame_load_timer is None:
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(self._perform_scheduled_real_afm_frame_load)
            self._real_afm_frame_load_timer = t

        delay_ms = 0 if immediate else 80
        try:
            self._real_afm_frame_load_timer.start(delay_ms)
        except Exception:
            pass

    def _perform_scheduled_real_afm_frame_load(self):
        idx = getattr(self, "_real_afm_pending_frame_index", None)
        if idx is None:
            return
        try:
            idx = int(idx)
        except Exception:
            return
        # If already on this frame, just refresh controls/label.
        if idx == int(getattr(self, 'real_asd_frame_index', 0) or 0):
            self._update_real_afm_frame_controls()
            return
        self.load_real_asd_frame(idx)

    def on_real_afm_frame_changed(self, value: int):
        """Load a different frame from the currently-loaded ASD."""
        try:
            idx = int(value)
        except Exception:
            return

        if not getattr(self, 'real_asd_path', None):
            return
        total = int(getattr(self, 'real_asd_frame_num', 0) or 0)
        if total <= 0:
            return
        idx = max(0, min(idx, total - 1))
        # Update label immediately for responsive UI while dragging.
        try:
            label = getattr(self, 'real_afm_frame_label', None)
            if label is not None:
                label.setText(f"Frame: {idx + 1} / {total}")
        except Exception:
            pass
        self._schedule_real_afm_frame_load(idx, immediate=False)

    def load_real_asd_frame(self, frame_index: int):
        """Load a specific frame from the currently selected ASD file."""
        path = getattr(self, 'real_asd_path', None)
        if not path:
            return
        try:
            from asd_io import read_asd_frame
            real, _real2, meta_in = read_asd_frame(path, frame_index=int(frame_index))
        except Exception as e:
            QMessageBox.critical(self, "ASD Load Error", f"Failed to load ASD frame:\n{e}")
            return

        self.real_afm_nm_full = real
        self.real_asd_frame_num = int(meta_in.get('frame_num', 0) or 0)
        self.real_asd_frame_index = int(meta_in.get('frame_index', frame_index) or 0)
        # Full-frame meta is kept from file load; if unavailable, rebuild a minimal one.
        if getattr(self, 'real_meta_full', None):
            full_meta = dict(self.real_meta_full)
            full_meta['nx'] = int(meta_in.get('x_pixel', real.shape[1]))
            full_meta['ny'] = int(meta_in.get('y_pixel', real.shape[0]))
            self.real_meta_full = full_meta
        else:
            nx = int(meta_in.get('x_pixel', real.shape[1]))
            ny = int(meta_in.get('y_pixel', real.shape[0]))
            sx = float(meta_in.get('x_scan_size', 0.0))
            sy = float(meta_in.get('y_scan_size', 0.0))
            if sx <= 0 or sy <= 0:
                sx = float(sx) if sx > 0 else float(sy)
                sy = float(sy) if sy > 0 else float(sx)
            self.real_meta_full = {
                'nx': nx,
                'ny': ny,
                'scan_x_nm': sx,
                'scan_y_nm': sy,
                'nm_per_pixel_x': (sx / nx) if nx else 0.0,
                'nm_per_pixel_y': (sy / ny) if ny else 0.0,
                'scan_direction': self._map_asd_scan_direction(meta_in.get('scan_direction', 0)),
            }
        self._rebuild_real_afm_active_from_full()
        # Changing frames invalidates alignment preview until re-estimated.
        self.sim_aligned_nm = None

        self._ensure_real_afm_window(show=True)
        self.show_real_afm()
        self._update_real_afm_frame_controls()

    def load_real_asd_file(self, filepath, sync=True):
        """Load ASD with pyNuD loader and store Real AFM data."""
        try:
            from asd_io import read_asd_frame
            real, _real2, meta_in = read_asd_frame(filepath, frame_index=0)
        except Exception as e:
            QMessageBox.critical(self, "ASD Load Error", f"Failed to load ASD:\n{e}")
            return

        nx = int(meta_in.get('x_pixel', real.shape[1]))
        ny = int(meta_in.get('y_pixel', real.shape[0]))
        scan_x_nm = float(meta_in.get('x_scan_size', 0.0))
        scan_y_nm = float(meta_in.get('y_scan_size', 0.0))
        if scan_x_nm <= 0 or scan_y_nm <= 0:
            scan_x_nm = float(scan_x_nm) if scan_x_nm > 0 else float(scan_y_nm)
            scan_y_nm = float(scan_y_nm) if scan_y_nm > 0 else float(scan_x_nm)

        meta = {
            'nx': nx,
            'ny': ny,
            'scan_x_nm': scan_x_nm,
            'scan_y_nm': scan_y_nm,
            'nm_per_pixel_x': (scan_x_nm / nx) if nx else 0.0,
            'nm_per_pixel_y': (scan_y_nm / ny) if ny else 0.0,
            'scan_direction': self._map_asd_scan_direction(meta_in.get('scan_direction', 0)),
        }

        self.real_afm_nm_full = real
        self.real_meta_full = dict(meta)
        # New file load resets ROI by default.
        self.real_afm_roi_px = None
        self._clear_real_afm_roi_overlay()
        self._rebuild_real_afm_active_from_full()
        self.real_asd_path = filepath
        self.real_asd_frame_num = int(meta_in.get('frame_num', 0) or 0)
        self.real_asd_frame_index = int(meta_in.get('frame_index', 0) or 0)
        self.last_import_dir = os.path.dirname(filepath)
        # New Real AFM loaded: clear any previous alignment preview until re-estimated.
        self.sim_aligned_nm = None

        self._ensure_real_afm_window(show=True)
        self.show_real_afm()
        self._update_real_afm_frame_controls()
        if sync:
            self.sync_sim_params_to_real()

    def show_real_afm(self):
        """Display Real AFM (ASD) image in the dedicated window."""
        if self.real_afm_nm is None:
            return
        self._ensure_real_afm_window(show=False)
        target = getattr(self, 'real_afm_window_real_frame', None)
        if target is None:
            return
        self.display_afm_image(self.real_afm_nm, target)
        # Clear aligned panel until pose is estimated
        if getattr(self, 'sim_aligned_nm', None) is None:
            self._clear_afm_panel(getattr(self, 'real_afm_window_aligned_frame', None))
        try:
            self.update_afm_display()
        except Exception:
            pass

    def sync_sim_params_to_real(self):
        """Sync simulator scan params to Real AFM metadata."""
        if not self.real_meta:
            return
        nx = int(self.real_meta.get('nx', 0))
        ny = int(self.real_meta.get('ny', 0))
        scan_x_nm = float(self.real_meta.get('scan_x_nm', 0.0))
        scan_y_nm = float(self.real_meta.get('scan_y_nm', 0.0))

        if hasattr(self, 'spinScanXNm') and scan_x_nm > 0:
            self.spinScanXNm.setValue(scan_x_nm)
        if hasattr(self, 'spinScanYNm') and scan_y_nm > 0:
            self.spinScanYNm.setValue(scan_y_nm)

        if hasattr(self, 'spinNx') and nx > 0:
            self.spinNx.setValue(nx)
        if hasattr(self, 'spinNy') and ny > 0:
            self.spinNy.setValue(ny)

        if hasattr(self, 'comboScanDirection') and 'scan_direction' in self.real_meta:
            val = self.real_meta['scan_direction']
            idx = self.comboScanDirection.findData(val)
            if idx >= 0:
                self.comboScanDirection.setCurrentIndex(idx)
            else:
                self.comboScanDirection.setCurrentText(str(val))

        try:
            self.trigger_interactive_simulation()
        except Exception:
            try:
                self.process_and_display_all_images()
            except Exception:
                pass

    def score_zncc(self, a, b):
        """Zero-mean normalized cross correlation."""
        if a is None or b is None:
            return -1e9
        if a.shape != b.shape:
            return -1e9
        mask = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(mask) < 2:
            return -1e9
        aa = a[mask].astype(np.float64)
        bb = b[mask].astype(np.float64)
        aa -= aa.mean()
        bb -= bb.mean()
        denom = np.linalg.norm(aa) * np.linalg.norm(bb)
        if denom <= 1e-12:
            return -1e9
        return float(np.dot(aa, bb) / denom)

    def preprocess_pose_image(self, img):
        """Robust normalization + high-pass filtering for pose estimation."""
        if img is None:
            return None
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim != 2:
            return None
        mask = np.isfinite(arr)
        if np.count_nonzero(mask) < 4:
            return None
        vals = arr[mask]
        p1 = float(np.percentile(vals, 1.0))
        p99 = float(np.percentile(vals, 99.0))
        if p99 <= p1:
            p1 = float(np.min(vals))
            p99 = float(np.max(vals))
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / max(p99 - p1, 1e-12)
        arr -= float(np.mean(arr))
        # Remove low-frequency background to reduce bias from global gradients/noise.
        low = scipy.ndimage.gaussian_filter(arr, sigma=2.0)
        arr_hp = arr - low
        arr_hp -= float(np.mean(arr_hp))
        return arr_hp

    def apply_mirror_mode(self, img, mirror_mode):
        """Apply mirror mode for alignment search/display."""
        if img is None:
            return None
        mode = str(mirror_mode or "none")
        if mode == "flip_lr":
            return np.fliplr(img)
        if mode == "flip_ud":
            return np.flipud(img)
        if mode == "flip_both":
            return np.flipud(np.fliplr(img))
        return img

    def apply_pose_to_image(self, img, pose):
        """Apply stored pose (mirror + rotation + translation) to an image."""
        if img is None:
            return None
        if not isinstance(pose, dict):
            return img
        mirror_mode = pose.get('mirror_mode', 'none')
        theta = float(pose.get('theta_deg', 0.0))
        dx = float(pose.get('dx_px', 0.0))
        dy = float(pose.get('dy_px', 0.0))
        base = self.apply_mirror_mode(img, mirror_mode)
        return self.transform_image(base, theta, dx, dy)

    def apply_estimated_pose_to_rotation_controls(self, theta_deg):
        """Reflect estimated in-plane pose rotation to Rotation XYZ controls (Z-axis update)."""
        if not hasattr(self, 'rotation_widgets'):
            return
        try:
            current_rx = float(self.rotation_widgets['X']['spin'].value())
            current_ry = float(self.rotation_widgets['Y']['spin'].value())
            current_rz = float(self.rotation_widgets['Z']['spin'].value())
        except Exception:
            return

        new_rx = self.normalize_angle(current_rx)
        new_ry = self.normalize_angle(current_ry)
        new_rz = self.normalize_angle(current_rz + float(theta_deg))
        self.set_rotation_controls_xyz(
            new_rx, new_ry, new_rz,
            apply_transform=True,
            trigger_simulation=True,
        )

    def set_rotation_controls_xyz(self, rx, ry, rz, *, apply_transform=True, trigger_simulation=True):
        """Set rotation controls safely and optionally apply transform."""
        if not hasattr(self, 'rotation_widgets'):
            return False

        new_rx = self.normalize_angle(float(rx))
        new_ry = self.normalize_angle(float(ry))
        new_rz = self.normalize_angle(float(rz))

        for axis in ['X', 'Y', 'Z']:
            try:
                self.rotation_widgets[axis]['spin'].blockSignals(True)
                self.rotation_widgets[axis]['slider'].blockSignals(True)
            except Exception:
                pass
        try:
            self.rotation_widgets['X']['spin'].setValue(new_rx)
            self.rotation_widgets['Y']['spin'].setValue(new_ry)
            self.rotation_widgets['Z']['spin'].setValue(new_rz)
            self.rotation_widgets['X']['slider'].setValue(int(round(new_rx * 10.0)))
            self.rotation_widgets['Y']['slider'].setValue(int(round(new_ry * 10.0)))
            self.rotation_widgets['Z']['slider'].setValue(int(round(new_rz * 10.0)))
        finally:
            for axis in ['X', 'Y', 'Z']:
                try:
                    self.rotation_widgets[axis]['spin'].blockSignals(False)
                    self.rotation_widgets[axis]['slider'].blockSignals(False)
                except Exception:
                    pass

        if apply_transform:
            self.apply_structure_rotation(trigger_simulation=trigger_simulation)
        return True

    def transform_image(self, img, theta_deg, dx_px, dy_px):
        """Rotate + shift with edge padding (no wrap)."""
        if img is None:
            return None
        rotated = scipy.ndimage.rotate(img, theta_deg, reshape=False, order=1, mode='nearest')
        shifted = scipy.ndimage.shift(rotated, shift=(dy_px, dx_px), order=1, mode='nearest')
        return shifted

    def estimate_translation_phase_corr(self, real, sim):
        """Phase correlation translation estimate."""
        if real is None or sim is None or real.shape != sim.shape:
            return 0.0, 0.0
        ny, nx = real.shape
        win_y = np.hanning(ny)
        win_x = np.hanning(nx)
        if win_y.size == 0 or win_x.size == 0:
            return 0.0, 0.0
        window = np.outer(win_y, win_x)
        A = fft2(real * window)
        B = fft2(sim * window)
        R = A * np.conj(B)
        R /= (np.abs(R) + 1e-12)
        r = np.real(ifft2(R))
        py, px = np.unravel_index(np.argmax(r), r.shape)

        # Subpixel peak refinement (parabolic fit around correlation peak).
        def _subpixel_1d(c_prev, c0, c_next):
            denom = (c_prev - 2.0 * c0 + c_next)
            if abs(denom) < 1e-12:
                return 0.0
            delta = 0.5 * (c_prev - c_next) / denom
            return float(np.clip(delta, -1.0, 1.0))

        py_m1 = (py - 1) % ny
        py_p1 = (py + 1) % ny
        px_m1 = (px - 1) % nx
        px_p1 = (px + 1) % nx
        dy_sub = _subpixel_1d(r[py_m1, px], r[py, px], r[py_p1, px])
        dx_sub = _subpixel_1d(r[py, px_m1], r[py, px], r[py, px_p1])

        shifts = np.array([py + dy_sub, px + dx_sub], dtype=float)
        if shifts[0] > ny / 2:
            shifts[0] -= ny
        if shifts[1] > nx / 2:
            shifts[1] -= nx
        dy, dx = shifts
        return float(dx), float(dy)

    def estimate_pose(self, real, sim0, theta_range_deg=180.0, coarse_step=5.0, fine_step=0.2, return_details=False, allow_mirror=True):
        """Estimate pose (mirror + theta + dx + dy) with robust coarse->fine search."""
        if real is None or sim0 is None:
            if return_details:
                return {'theta': 0.0, 'dx': 0.0, 'dy': 0.0, 'score': -1e9, 'mirror_mode': 'none'}
            return 0.0, 0.0, 0.0, -1e9
        if real.shape != sim0.shape:
            zoom_y = real.shape[0] / sim0.shape[0]
            zoom_x = real.shape[1] / sim0.shape[1]
            sim0 = scipy.ndimage.zoom(sim0, (zoom_y, zoom_x), order=1)

        real_p = self.preprocess_pose_image(real)
        sim_p0 = self.preprocess_pose_image(sim0)
        if real_p is None or sim_p0 is None:
            if return_details:
                return {'theta': 0.0, 'dx': 0.0, 'dy': 0.0, 'score': -1e9, 'mirror_mode': 'none'}
            return 0.0, 0.0, 0.0, -1e9

        best = {'theta': 0.0, 'dx': 0.0, 'dy': 0.0, 'score': -1e9, 'mirror_mode': 'none'}
        mirror_modes = ['none', 'flip_lr', 'flip_ud', 'flip_both'] if allow_mirror else ['none']

        def eval_theta(base_img, theta):
            sim_rot = self.transform_image(base_img, theta, 0, 0)
            dx, dy = self.estimate_translation_phase_corr(real_p, sim_rot)
            sim_aligned = self.transform_image(sim_rot, 0, dx, dy)
            score = self.score_zncc(real_p, sim_aligned)
            return dx, dy, score

        # Stage 1: global coarse search.
        coarse_angles = np.arange(-theta_range_deg, theta_range_deg + 1e-9, coarse_step)
        for mirror_mode in mirror_modes:
            base_img = self.apply_mirror_mode(sim_p0, mirror_mode)
            for theta in coarse_angles:
                dx, dy, score = eval_theta(base_img, theta)
                if score > best['score']:
                    best.update({'theta': float(theta), 'dx': float(dx), 'dy': float(dy), 'score': float(score), 'mirror_mode': mirror_mode})

        # Stage 2: medium search around best (1.0 deg).
        medium_step = 1.0
        medium_half = max(coarse_step, 4.0)
        base_img = self.apply_mirror_mode(sim_p0, best['mirror_mode'])
        medium_angles = np.arange(best['theta'] - medium_half, best['theta'] + medium_half + 1e-9, medium_step)
        for theta in medium_angles:
            dx, dy, score = eval_theta(base_img, theta)
            if score > best['score']:
                best.update({'theta': float(theta), 'dx': float(dx), 'dy': float(dy), 'score': float(score)})

        # Stage 3: fine search around best.
        fine_half = 1.5
        fine_angles = np.arange(best['theta'] - fine_half, best['theta'] + fine_half + 1e-9, fine_step)
        for theta in fine_angles:
            dx, dy, score = eval_theta(base_img, theta)
            if score > best['score']:
                best.update({'theta': float(theta), 'dx': float(dx), 'dy': float(dy), 'score': float(score)})

        # Normalize angle to [-180, 180)
        theta_norm = ((best['theta'] + 180.0) % 360.0) - 180.0
        best['theta'] = float(theta_norm)

        if return_details:
            return best
        return best['theta'], best['dx'], best['dy'], best['score']

    def _get_simulated_xy_image(self, use_processed=True):
        key = "XY_Frame"
        if use_processed and key in self.simulation_results:
            return self.simulation_results.get(key)
        if key in self.raw_simulation_results:
            return self.raw_simulation_results.get(key)
        return None

    def _get_real_afm_simulation_meta(self):
        """Return (scan_x_nm, scan_y_nm, nx, ny) from loaded ASD metadata."""
        if self.real_afm_nm is None or not getattr(self, "real_meta", None):
            return None
        try:
            scan_x_nm = float(self.real_meta.get('scan_x_nm', 0.0))
            scan_y_nm = float(self.real_meta.get('scan_y_nm', 0.0))
            nx = int(self.real_meta.get('nx', 0))
            ny = int(self.real_meta.get('ny', 0))
        except Exception:
            return None
        if scan_x_nm <= 0 or scan_y_nm <= 0 or nx <= 0 or ny <= 0:
            return None
        return scan_x_nm, scan_y_nm, nx, ny

    def _apply_real_afm_scan_to_controls(self, scan_x_nm, scan_y_nm, nx, ny):
        """Apply ASD scan size/resolution to simulation controls safely."""
        for widget in (self.spinScanXNm, self.spinScanYNm, self.spinNx, self.spinNy):
            try:
                widget.blockSignals(True)
            except Exception:
                pass
        try:
            self.spinScanXNm.setValue(float(scan_x_nm))
            self.spinScanYNm.setValue(float(scan_y_nm))
            self.spinNx.setValue(int(nx))
            self.spinNy.setValue(int(ny))
        finally:
            for widget in (self.spinScanXNm, self.spinScanYNm, self.spinNx, self.spinNy):
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass

    def _run_xy_simulation_blocking(self, coords, scan_x_nm, scan_y_nm, nx, ny):
        """Run one XY AFM simulation synchronously and return raw height map."""
        if coords is None:
            return None

        panel = getattr(self, 'afm_x_frame', None) or getattr(self, 'real_afm_window_aligned_frame', None)
        if panel is None:
            panel = QFrame(self)
            panel.setObjectName("XY_Frame")

        sim_params = {
            'scan_x_nm': float(scan_x_nm),
            'scan_y_nm': float(scan_y_nm),
            'nx': int(nx),
            'ny': int(ny),
            'center_x': self.tip_x_slider.value() / 5.0,
            'center_y': self.tip_y_slider.value() / 5.0,
            'tip_radius': self.tip_radius_spin.value(),
            'minitip_radius': self.minitip_radius_spin.value(),
            'tip_angle': self.tip_angle_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'use_vdw': self.use_vdw_check.isChecked(),
        }
        sim_params['scan_size'] = sim_params['scan_x_nm']
        sim_params['resolution'] = sim_params['nx']

        tasks = [{
            "name": "XY",
            "panel": panel,
            "coords": coords,
        }]

        worker = AFMSimulationWorker(
            self, sim_params, tasks,
            self.atoms_data['element'] if sim_params['use_vdw'] and self.atoms_data is not None else None,
            self.vdw_radii if sim_params['use_vdw'] and hasattr(self, 'vdw_radii') else None,
            silent_mode=True
        )
        self._connect_worker_delete_later(worker)

        result_holder = {"raw": None}
        loop = QEventLoop(self)

        def _on_task_done(z_map, _panel):
            result_holder["raw"] = z_map

        worker.task_done.connect(_on_task_done)
        worker.done.connect(loop.quit)
        worker.start()
        loop.exec_()
        return result_holder.get("raw", None)

    def _postprocess_xy_result(self, raw_xy, scan_x_nm, scan_y_nm, nx, ny):
        """Apply current filter/noise settings to simulated XY map."""
        if raw_xy is None:
            return None
        processed_xy = raw_xy
        if self.apply_filter_check.isChecked():
            processed_xy = apply_low_pass_filter(processed_xy, scan_x_nm, scan_y_nm, self.filter_cutoff_spin.value())
        try:
            pixel_x_nm = float(scan_x_nm) / max(float(nx), 1e-12)
            pixel_y_nm = float(scan_y_nm) / max(float(ny), 1e-12)
            processed_xy = self.apply_noise_artifacts(processed_xy, pixel_x_nm, pixel_y_nm)
        except Exception:
            pass
        return processed_xy

    def _score_real_vs_sim(self, real_preprocessed, sim_img):
        """Translation-invariant similarity score for pose search."""
        if real_preprocessed is None or sim_img is None:
            return -1e9, 0.0, 0.0

        sim_cmp = sim_img
        if real_preprocessed.shape != sim_cmp.shape:
            zoom_y = real_preprocessed.shape[0] / sim_cmp.shape[0]
            zoom_x = real_preprocessed.shape[1] / sim_cmp.shape[1]
            sim_cmp = scipy.ndimage.zoom(sim_cmp, (zoom_y, zoom_x), order=1)

        sim_p = self.preprocess_pose_image(sim_cmp)
        if sim_p is None:
            return -1e9, 0.0, 0.0

        dx, dy = self.estimate_translation_phase_corr(real_preprocessed, sim_p)
        sim_aligned = self.transform_image(sim_p, 0.0, dx, dy)
        score = self.score_zncc(real_preprocessed, sim_aligned)
        return float(score), float(dx), float(dy)

    def _simulate_xy_for_real_afm(self, *, update_panels=True, store_results=True, check_busy=True, show_messages=True):
        """Simulate XY using ASD scan metadata and return {'raw','processed','meta'} or None."""
        if self.real_afm_nm is None or not getattr(self, "real_meta", None):
            if show_messages:
                QMessageBox.information(self, "Get Simulated image", "Real AFM is not loaded.")
            return None

        if self.atoms_data is None:
            if show_messages:
                QMessageBox.warning(self, "Get Simulated image", "PDB is not loaded.")
            return None

        if check_busy:
            if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker') or \
               self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent') or \
               self.is_worker_running(getattr(self, 'sim_worker_high_res', None), attr_name='sim_worker_high_res'):
                if show_messages:
                    QMessageBox.information(self, "Get Simulated image", "Another simulation is running. Please wait.")
                return None

        meta = self._get_real_afm_simulation_meta()
        if meta is None:
            if show_messages:
                QMessageBox.warning(self, "Get Simulated image", "Real AFM metadata is incomplete.")
            return None
        scan_x_nm, scan_y_nm, nx, ny = meta

        try:
            if self._is_rectangle_lock_enabled() and (abs(scan_x_nm - scan_y_nm) > 1e-9 or int(nx) != int(ny)):
                self.rectangle_check.blockSignals(True)
                self.rectangle_check.setChecked(False)
                self.rectangle_check.blockSignals(False)
                self._apply_rectangle_lock(enforce_values=False)
        except Exception:
            try:
                self.rectangle_check.blockSignals(False)
            except Exception:
                pass

        self._apply_real_afm_scan_to_controls(scan_x_nm, scan_y_nm, nx, ny)

        coords = self.get_rotated_atom_coords()
        if coords is None:
            if show_messages:
                QMessageBox.warning(self, "Get Simulated image", "Failed to get rotated PDB coordinates.")
            return None

        raw_xy = self._run_xy_simulation_blocking(coords, scan_x_nm, scan_y_nm, nx, ny)
        if raw_xy is None:
            if show_messages:
                QMessageBox.warning(self, "Get Simulated image", "Simulation failed.")
            return None

        processed_xy = self._postprocess_xy_result(raw_xy, scan_x_nm, scan_y_nm, nx, ny)
        if processed_xy is None:
            if show_messages:
                QMessageBox.warning(self, "Get Simulated image", "Post-processing failed.")
            return None

        if store_results:
            self.raw_simulation_results["XY_Frame"] = raw_xy
            self.simulation_results["XY_Frame"] = processed_xy
            self.sim_aligned_nm = processed_xy

        if update_panels:
            try:
                if hasattr(self, 'afm_x_frame') and self.afm_x_frame is not None:
                    self.display_afm_image(processed_xy, self.afm_x_frame)
            except Exception:
                pass
            target_panel = getattr(self, 'real_afm_window_aligned_frame', None)
            if target_panel is not None:
                self.display_afm_image(processed_xy, target_panel)

        return {
            "raw": raw_xy,
            "processed": processed_xy,
            "meta": {
                "scan_x_nm": scan_x_nm,
                "scan_y_nm": scan_y_nm,
                "nx": nx,
                "ny": ny,
            },
        }

    def estimate_pose_from_real(self):
        """Estimate Rotation XYZ by iteratively re-simulating AFM and maximizing similarity."""
        if self.real_afm_nm is None:
            QMessageBox.information(self, "Pose", "Real AFM is not loaded.")
            return
        if self.atoms_data is None:
            QMessageBox.warning(self, "Pose", "PDB is not loaded.")
            return

        # Keep explicit simulation from conflicting with active workers.
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker') or \
           self.is_worker_running(getattr(self, 'sim_worker_high_res', None), attr_name='sim_worker_high_res'):
            QMessageBox.information(self, "Pose", "Another simulation is running. Please wait.")
            return
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            self.stop_worker(self.sim_worker_silent, timeout_ms=300, allow_terminate=True, worker_name="sim_worker_silent")
            if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
                QMessageBox.information(self, "Pose", "Another simulation is running. Please wait.")
                return

        try:
            orig_rx = float(self.rotation_widgets['X']['spin'].value())
            orig_ry = float(self.rotation_widgets['Y']['spin'].value())
            orig_rz = float(self.rotation_widgets['Z']['spin'].value())
        except Exception:
            QMessageBox.warning(self, "Pose", "Rotation controls are unavailable.")
            return

        real_p = self.preprocess_pose_image(self.real_afm_nm)
        if real_p is None:
            QMessageBox.warning(self, "Pose", "Real AFM image is invalid for pose estimation.")
            return

        # Ask precision level before running iterative search.
        precision_levels = ["Medium", "Low", "High"]
        selected_level, ok = QInputDialog.getItem(
            self,
            "Estimate Pose",
            "Precision:",
            precision_levels,
            0,
            False,
        )
        if not ok:
            return

        profiles = {
            "Low": {
                "seed_offsets": [(0.0, 0.0), (30.0, 0.0), (0.0, 30.0)],
                "z_step": 45.0,
                "refine_steps": (20.0, 10.0, 5.0, 2.0),
                "max_refine_iter": 8,
            },
            "Medium": {
                "seed_offsets": [(0.0, 0.0), (45.0, 0.0), (-45.0, 0.0), (0.0, 45.0), (0.0, -45.0)],
                "z_step": 30.0,
                "refine_steps": (15.0, 8.0, 4.0, 2.0, 1.0),
                "max_refine_iter": 16,
            },
            "High": {
                "seed_offsets": [
                    (0.0, 0.0), (60.0, 0.0), (-60.0, 0.0),
                    (0.0, 60.0), (0.0, -60.0),
                    (30.0, 30.0), (-30.0, 30.0), (30.0, -30.0), (-30.0, -30.0),
                ],
                "z_step": 20.0,
                "refine_steps": (12.0, 6.0, 3.0, 1.5, 0.75, 0.4),
                "max_refine_iter": 20,
            },
        }
        cfg = profiles.get(str(selected_level), profiles["Medium"])

        eval_cache = {}
        eval_count = 0
        best = {'rx': orig_rx, 'ry': orig_ry, 'rz': orig_rz, 'score': -1e9, 'dx': 0.0, 'dy': 0.0}
        cancel_requested = False
        z_coarse = np.arange(-180.0, 180.0 + 1e-9, float(cfg["z_step"]))
        max_eval_est = max(
            1,
            len(cfg["seed_offsets"]) * len(z_coarse)
            + len(cfg["refine_steps"]) * int(cfg["max_refine_iter"]) * 6
            + 2,
        )
        progress = QProgressDialog("Estimating pose...", "Cancel", 0, int(max_eval_est), self)
        progress.setWindowTitle("Estimate Pose")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        self._pose_estimation_running = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            def evaluate(rx, ry, rz):
                nonlocal eval_count, cancel_requested
                if cancel_requested:
                    return None

                nrx = self.normalize_angle(rx)
                nry = self.normalize_angle(ry)
                nrz = self.normalize_angle(rz)
                key = (round(nrx, 3), round(nry, 3), round(nrz, 3))
                if key in eval_cache:
                    return eval_cache[key]

                ok = self.set_rotation_controls_xyz(
                    nrx, nry, nrz,
                    apply_transform=True,
                    trigger_simulation=False,
                )
                if not ok:
                    result = {'rx': nrx, 'ry': nry, 'rz': nrz, 'score': -1e9, 'dx': 0.0, 'dy': 0.0}
                    eval_cache[key] = result
                    return result

                sim_pack = self._simulate_xy_for_real_afm(
                    update_panels=False,
                    store_results=False,
                    check_busy=False,
                    show_messages=False,
                )
                if sim_pack is None:
                    result = {'rx': nrx, 'ry': nry, 'rz': nrz, 'score': -1e9, 'dx': 0.0, 'dy': 0.0}
                else:
                    score, dx, dy = self._score_real_vs_sim(real_p, sim_pack.get("processed"))
                    result = {'rx': nrx, 'ry': nry, 'rz': nrz, 'score': score, 'dx': dx, 'dy': dy}
                eval_cache[key] = result
                eval_count += 1
                progress.setValue(min(eval_count, int(max_eval_est)))
                progress.setLabelText(
                    f"Estimating pose ({selected_level})...\n"
                    f"Evaluations: {eval_count}\n"
                    f"Best score: {best['score']:.4f}"
                )
                QApplication.processEvents()
                if progress.wasCanceled():
                    cancel_requested = True
                return result

            # Stage 1: coarse seeding (global Z scan from several X/Y starts).
            for off_x, off_y in cfg["seed_offsets"]:
                if cancel_requested:
                    break
                sx = self.normalize_angle(orig_rx + off_x)
                sy = self.normalize_angle(orig_ry + off_y)
                for zc in z_coarse:
                    if cancel_requested:
                        break
                    cand = evaluate(sx, sy, zc)
                    if cand is None:
                        break
                    if cand['score'] > best['score']:
                        best = dict(cand)

            # Stage 2: coordinate-descent refinement over XYZ.
            for step in cfg["refine_steps"]:
                if cancel_requested:
                    break
                for _ in range(int(cfg["max_refine_iter"])):
                    if cancel_requested:
                        break
                    center = best
                    candidates = [
                        (center['rx'] + step, center['ry'], center['rz']),
                        (center['rx'] - step, center['ry'], center['rz']),
                        (center['rx'], center['ry'] + step, center['rz']),
                        (center['rx'], center['ry'] - step, center['rz']),
                        (center['rx'], center['ry'], center['rz'] + step),
                        (center['rx'], center['ry'], center['rz'] - step),
                    ]
                    improved = False
                    for crx, cry, crz in candidates:
                        if cancel_requested:
                            break
                        cand = evaluate(crx, cry, crz)
                        if cand is None:
                            break
                        if cand['score'] > (best['score'] + 1e-6):
                            best = dict(cand)
                            improved = True
                    if not improved:
                        break

            if cancel_requested:
                self.set_rotation_controls_xyz(orig_rx, orig_ry, orig_rz, apply_transform=True, trigger_simulation=False)
                QMessageBox.information(self, "Pose", "Pose estimation canceled.")
                return

            if best['score'] <= -1e8:
                self.set_rotation_controls_xyz(orig_rx, orig_ry, orig_rz, apply_transform=True, trigger_simulation=False)
                QMessageBox.warning(self, "Pose", "Pose estimation failed.")
                return

            self.set_rotation_controls_xyz(
                best['rx'], best['ry'], best['rz'],
                apply_transform=True,
                trigger_simulation=False,
            )
            final_pack = self._simulate_xy_for_real_afm(
                update_panels=True,
                store_results=True,
                check_busy=False,
                show_messages=False,
            )
            if final_pack is None:
                self.set_rotation_controls_xyz(orig_rx, orig_ry, orig_rz, apply_transform=True, trigger_simulation=False)
                QMessageBox.warning(self, "Pose", "Failed to generate final simulated image.")
                return

            final_score, final_dx, final_dy = self._score_real_vs_sim(real_p, final_pack.get("processed"))
            self.pose = {
                'theta_deg': 0.0,
                'dx_px': float(final_dx),
                'dy_px': float(final_dy),
                'score': float(final_score),
                'mirror_mode': 'none',
                'rot_x_deg': float(best['rx']),
                'rot_y_deg': float(best['ry']),
                'rot_z_deg': float(best['rz']),
            }

            QMessageBox.information(
                self, "Pose Estimated",
                f"RotX: {best['rx']:.2f} deg\n"
                f"RotY: {best['ry']:.2f} deg\n"
                f"RotZ: {best['rz']:.2f} deg\n"
                f"Residual Dx: {final_dx:.2f} px\n"
                f"Residual Dy: {final_dy:.2f} px\n"
                f"Score: {final_score:.4f}\n"
                f"Evaluations: {eval_count}\n"
                f"Precision: {selected_level}"
            )
        finally:
            try:
                progress.setValue(int(max_eval_est))
                progress.close()
            except Exception:
                pass
            self._pose_estimation_running = False
            QApplication.restoreOverrideCursor()

    def get_simulated_image_for_real_afm(self):
        """Generate simulated XY image matching Real AFM scan size/resolution."""
        result = self._simulate_xy_for_real_afm(
            update_panels=True,
            store_results=True,
            check_busy=True,
            show_messages=True,
        )
        if result is None:
            return

    def _clear_afm_panel(self, target_panel):
        """Show placeholder and clear the image view for a panel created by create_afm_image_panel()."""
        if target_panel is None:
            return
        try:
            container = target_panel.findChild(QWidget, "afm_image_container")
            if container is None:
                return
            layout = container.layout()
            placeholder = container.findChild(QLabel, "afm_placeholder")
            view = container.findChild(_AspectPixmapView, "afm_image_view")
            if view is not None:
                view.setSourcePixmap(None)
            if layout is not None and hasattr(layout, "setCurrentWidget") and placeholder is not None:
                layout.setCurrentWidget(placeholder)
        except Exception:
            pass

    def _ensure_real_afm_window(self, show=False):
        """Create (if needed) and optionally show the Real AFM / Sim Aligned window."""
        # If already exists and not deleted, reuse it.
        w = getattr(self, 'real_afm_window', None)
        if w is not None:
            try:
                _ = w.isVisible()
                if show:
                    w.show()
                    w.raise_()
                    w.activateWindow()
                return
            except RuntimeError:
                self.real_afm_window = None
                self.real_afm_window_real_frame = None
                self.real_afm_window_aligned_frame = None

        win = QWidget(None)
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        win.setWindowTitle("Real AFM (ASD) / Sim Aligned")
        win.resize(900, 480)

        outer = QVBoxLayout(win)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Optional action row
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Real AFM and alignment preview")
        label.setStyleSheet("font-weight: bold; color: #333;")
        row.addWidget(label)
        row.addStretch(1)
        btn_get_sim = QPushButton("Get Simulated image")
        btn_get_sim.clicked.connect(self.get_simulated_image_for_real_afm)
        row.addWidget(btn_get_sim)
        btn_pose = QPushButton("🧭 Estimate Pose")
        btn_pose.clicked.connect(self.estimate_pose_from_real)
        row.addWidget(btn_pose)
        btn_auto_fit = QPushButton("Auto-fit Appearance")
        btn_auto_fit.clicked.connect(self.auto_fit_appearance)
        row.addWidget(btn_auto_fit)
        outer.addLayout(row)

        splitter = QSplitter(Qt.Horizontal, win)
        splitter.setHandleWidth(6)

        real_frame = self.create_afm_image_panel("Real AFM (ASD)")
        real_frame.setObjectName("REAL_AFM_Frame")
        aligned_frame = self.create_afm_image_panel("Sim Aligned")
        aligned_frame.setObjectName("SIM_ALIGNED_Frame")

        # Frame slider (ASD can have multiple frames)
        control_row_height = 28
        try:
            slider_row = QWidget(real_frame)
            slider_row.setObjectName("REAL_AFM_FrameSliderRow")
            slider_layout = QHBoxLayout(slider_row)
            slider_layout.setContentsMargins(4, 2, 4, 2)
            slider_layout.setSpacing(6)

            frame_label = QLabel("Frame: -", slider_row)
            frame_label.setMinimumWidth(90)
            frame_label.setStyleSheet("color: #444; font-size: 9px;")
            slider_layout.addWidget(frame_label)

            frame_slider = QSlider(Qt.Horizontal, slider_row)
            frame_slider.setObjectName("REAL_AFM_FrameSlider")
            frame_slider.setRange(0, 0)
            frame_slider.setValue(0)
            frame_slider.setTracking(True)  # debounced loader handles rapid updates
            frame_slider.setEnabled(False)
            frame_slider.valueChanged.connect(self.on_real_afm_frame_changed)
            try:
                frame_slider.sliderReleased.connect(lambda: self._schedule_real_afm_frame_load(frame_slider.value(), immediate=True))
            except Exception:
                pass
            slider_layout.addWidget(frame_slider, 1)

            self.real_afm_frame_label = frame_label
            self.real_afm_frame_slider = frame_slider

            # Insert row below the title label, above the image container
            lf = real_frame.layout()
            if lf is not None:
                lf.insertWidget(1, slider_row)
            try:
                control_row_height = max(int(slider_row.sizeHint().height()), 1)
            except Exception:
                control_row_height = 28
        except Exception:
            self.real_afm_frame_label = None
            self.real_afm_frame_slider = None

        # Keep the Sim Aligned panel image area identical to Real AFM and use the row for metadata display.
        try:
            aligned_spacer_row = QWidget(aligned_frame)
            aligned_spacer_row.setObjectName("SIM_ALIGNED_FrameControlSpacer")
            aligned_spacer_row.setFixedHeight(control_row_height)
            aligned_spacer_layout = QHBoxLayout(aligned_spacer_row)
            aligned_spacer_layout.setContentsMargins(4, 2, 4, 2)
            aligned_spacer_layout.setSpacing(6)

            info_label = QLabel("Scan X/Y: - / - nm    Pixel X/Y: - / - nm/px", aligned_spacer_row)
            info_label.setObjectName("SIM_ALIGNED_InfoLabel")
            info_label.setStyleSheet("color: #444; font-size: 9px;")
            info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            aligned_spacer_layout.addWidget(info_label, 1)
            self.real_afm_sim_info_label = info_label

            af = aligned_frame.layout()
            if af is not None:
                af.insertWidget(1, aligned_spacer_row)
        except Exception:
            self.real_afm_sim_info_label = None

        # Enable drag&drop of ASD file onto the Real AFM panel
        try:
            self._asd_drop_filter = _ASDDropFilter(self)
            container = real_frame.findChild(QWidget, "afm_image_container")
            placeholder = real_frame.findChild(QLabel, "afm_placeholder")
            view = real_frame.findChild(_AspectPixmapView, "afm_image_view")
            if view is not None:
                try:
                    view.setRoiSelectionEnabled(True)
                    view.setToolTip("Left-drag: select ROI for pose/simulation\nRight-click: reset ROI")
                    view.roiSelected.connect(self.on_real_afm_roi_selected)
                    view.roiResetRequested.connect(self.clear_real_afm_roi)
                except Exception:
                    pass
            for wdg in (real_frame, container, placeholder, view):
                if wdg is None:
                    continue
                wdg.setAcceptDrops(True)
                wdg.installEventFilter(self._asd_drop_filter)
        except Exception:
            pass

        splitter.addWidget(real_frame)
        splitter.addWidget(aligned_frame)
        splitter.setSizes([450, 450])
        outer.addWidget(splitter, 1)

        def _on_destroyed(*_):
            self.real_afm_window = None
            self.real_afm_window_real_frame = None
            self.real_afm_window_aligned_frame = None
            self.real_afm_sim_info_label = None

        try:
            win.destroyed.connect(_on_destroyed)
        except Exception:
            pass

        self.real_afm_window = win
        self.real_afm_window_real_frame = real_frame
        self.real_afm_window_aligned_frame = aligned_frame

        # Populate initial content if available
        if getattr(self, 'real_afm_nm', None) is not None:
            self.display_afm_image(self.real_afm_nm, real_frame)
        else:
            self._clear_afm_panel(real_frame)
        if getattr(self, 'sim_aligned_nm', None) is not None:
            self.display_afm_image(self.sim_aligned_nm, aligned_frame)
        else:
            self._clear_afm_panel(aligned_frame)

        try:
            self._update_real_afm_frame_controls()
        except Exception:
            pass
        try:
            self._update_sim_aligned_info_label()
        except Exception:
            pass

        if show:
            win.show()
            win.raise_()
            win.activateWindow()

    def apply_noise_artifacts_with_params(self, height_nm, pixel_x_nm, pixel_y_nm, params):
        """Apply noise/artifacts using explicit parameters (for auto-fit)."""
        height = np.array(height_nm, dtype=float, copy=True)
        rng = np.random.default_rng(params.get('seed', 0))

        sigma_h = params.get('height_sigma_nm', 0.0)
        if sigma_h > 0:
            height += rng.normal(0.0, sigma_h, size=height.shape)

        sigma_line = params.get('line_sigma_nm', 0.0)
        if sigma_line > 0:
            ny = height.shape[0]
            b = rng.normal(0.0, sigma_line, size=ny)
            height += b[:, None]

        drift_vx = params.get('drift_vx_nm_per_line', 0.0)
        if drift_vx != 0.0:
            ny, nx = height.shape
            shifted = np.zeros_like(height)
            for y in range(ny):
                dx_px = int(round((drift_vx * y) / max(pixel_x_nm, 1e-6)))
                line = height[y]
                if dx_px != 0:
                    line = np.roll(line, dx_px)
                shifted[y] = line
            height = shifted

        scan_dir = str(params.get('scan_direction', 'L2R'))
        reverse = (scan_dir == "R2L" or scan_dir == "Right -> Left")

        def apply_iir(line, alpha, reverse=False):
            if reverse:
                line = line[::-1]
            out = np.empty_like(line, dtype=float)
            out[0] = line[0]
            for i in range(1, line.size):
                out[i] = alpha * out[i - 1] + (1.0 - alpha) * line[i]
            if reverse:
                out = out[::-1]
            return out

        if params.get('feedback_mode') == "tapping_parachute":
            drop_th = params.get('tap_drop_threshold_nm', 1.0)
            tau_track = params.get('tap_tau_track_lines', 2.0)
            tau_para = params.get('tap_tau_parachute_lines', 15.0)
            rel_th = params.get('tap_release_threshold_nm', 0.3)

            def apply_tapping(line, reverse=False):
                if reverse:
                    line = line[::-1]
                out = np.empty_like(line, dtype=float)
                out[0] = line[0]
                state = "TRACK"
                for i in range(1, line.size):
                    drop = line[i] - line[i - 1]
                    if drop < -drop_th:
                        state = "PARACHUTE"
                    tau = tau_track if state == "TRACK" else tau_para
                    alpha = math.exp(-1.0 / max(tau, 1e-6))
                    out[i] = alpha * out[i - 1] + (1.0 - alpha) * line[i]
                    if state == "PARACHUTE" and (out[i] - line[i]) < rel_th:
                        state = "TRACK"
                if reverse:
                    out = out[::-1]
                return out

            for y in range(height.shape[0]):
                height[y] = apply_tapping(height[y], reverse=reverse)
        else:
            tau = float(params.get('lag_tau_lines', 2.0))
            alpha = math.exp(-1.0 / max(tau, 1e-6))
            for y in range(height.shape[0]):
                height[y] = apply_iir(height[y], alpha, reverse=reverse)

        return height

    def auto_fit_appearance(self):
        """Auto-fit AFM appearance parameters to Real AFM (coarse grid)."""
        if self.real_afm_nm is None:
            QMessageBox.information(self, "Auto-fit", "Real AFM is not loaded.")
            return
        sim_clean = self._get_simulated_xy_image(use_processed=False)
        if sim_clean is None:
            QMessageBox.information(self, "Auto-fit", "Simulated AFM not available.")
            return

        # Apply pose if available for comparison
        pose = getattr(self, 'pose', None) or {'theta_deg': 0.0, 'dx_px': 0.0, 'dy_px': 0.0}
        
        scan_x = self.spinScanXNm.value()
        scan_y = self.spinScanYNm.value()
        ny, nx = sim_clean.shape
        pixel_x_nm = scan_x / nx
        pixel_y_nm = scan_y / ny

        best = {'score': -1e9, 'params': None}
        fixed_scan_direction = "L2R"

        # Keep Low-pass filter settings fixed to current UI state (do not auto-fit cutoff).
        if self.apply_filter_check.isChecked():
            base = apply_low_pass_filter(sim_clean, scan_x, scan_y, self.filter_cutoff_spin.value())
        else:
            base = sim_clean

        height_sigma = [0.0, 0.05, 0.1, 0.2]
        line_sigma = [0.0, 0.05, 0.1]
        drift_vx = [0.0, 0.05, 0.1]
        tap_drop = [0.0, 0.6, 1.0]
        tap_tau_para = [5.0, 10.0, 20.0]

        for hs in height_sigma:
            for ls in line_sigma:
                for dvx in drift_vx:
                    for td in tap_drop:
                        for tp in tap_tau_para:
                            params = {
                                'seed': 0,
                                'height_sigma_nm': hs,
                                'line_sigma_nm': ls,
                                'drift_vx_nm_per_line': dvx,
                                'feedback_mode': "tapping_parachute" if td > 0 else "linear_lag",
                                'scan_direction': fixed_scan_direction,
                                'lag_tau_lines': float(self.spinLagTauLines.value()) if hasattr(self, 'spinLagTauLines') else 2.0,
                                'tap_drop_threshold_nm': max(td, 0.3),
                                'tap_tau_track_lines': 2.0,
                                'tap_tau_parachute_lines': tp,
                                'tap_release_threshold_nm': 0.3,
                            }
                            sim_candidate = self.apply_noise_artifacts_with_params(base, pixel_x_nm, pixel_y_nm, params)
                            sim_candidate = self.apply_pose_to_image(sim_candidate, pose)
                            score = self.score_zncc(self.real_afm_nm, sim_candidate)
                            if score > best['score']:
                                best = {'score': score, 'params': params}

        if best['params'] is None:
            QMessageBox.information(self, "Auto-fit", "No valid candidates.")
            return

        params = best['params']

        self.chkNoiseEnable.setChecked(True)
        self.chkHeightNoise.setChecked(params['height_sigma_nm'] > 0)
        self.spinHeightNoiseSigmaNm.setValue(float(params['height_sigma_nm']))
        self.chkLineNoise.setChecked(params['line_sigma_nm'] > 0)
        self.spinLineNoiseSigmaNm.setValue(float(params['line_sigma_nm']))
        self.chkDrift.setChecked(params['drift_vx_nm_per_line'] > 0)
        self.spinDriftVxNmPerLine.setValue(float(params['drift_vx_nm_per_line']))

        if params['feedback_mode'] == "tapping_parachute":
            self.chkFeedbackLag.setChecked(True)
            self.comboFeedbackMode.setCurrentText("tapping_parachute")
            self.spinTapDropThresholdNm.setValue(float(params['tap_drop_threshold_nm']))
            self.spinTapTauParachuteLines.setValue(float(params['tap_tau_parachute_lines']))
        else:
            self.chkFeedbackLag.setChecked(False)
        if hasattr(self, 'comboScanDirection'):
            idx = self.comboScanDirection.findData(fixed_scan_direction)
            if idx >= 0:
                self.comboScanDirection.setCurrentIndex(idx)

        try:
            self._update_noise_ui_states()
        except Exception:
            pass

        self.process_and_display_all_images()

    # 既存の create_vtk_panel メソッドを、以下の完全なコードで置き換えてください。

    def create_vtk_panel(self):
        """右側のVTK表示パネル作成（上下可変分割 + 下部3分割）"""
        panel = QWidget()
        panel.setMinimumSize(550, 600)
        
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 上下のメインスプリッター
        self.afm_splitter = QSplitter(Qt.Vertical)
        self.afm_splitter.setHandleWidth(8)
        self.afm_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #ddd;
                border: 1px solid #ccc;
            }
            QSplitter::handle:hover {
                background-color: #bbb;
            }
        """)
        
        # --- 上部：PDB構造表示エリア ---
        structure_frame = QFrame()
        structure_frame.setFrameStyle(QFrame.StyledPanel)
        structure_frame.setLineWidth(1)
        structure_layout = QVBoxLayout(structure_frame)
        structure_layout.setContentsMargins(2, 2, 2, 2)
        structure_layout.setSpacing(2)
        # if 
        structure_label = QLabel("Drop PDB, CIF, MRC files here")
        structure_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #333;
                padding: 3px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        structure_label.setAlignment(Qt.AlignCenter)
        structure_label.setMaximumHeight(25)
        structure_layout.addWidget(structure_label)

        structure_layout.addWidget(self.progress_container)
        
        # VTKウィンドウとコントロールパネルを配置するための垂直スプリッター
        self.view_control_splitter = QSplitter(Qt.Vertical)
        self.view_control_splitter.setHandleWidth(6)
        self.view_control_splitter.setStyleSheet("""
            QSplitter::handle:vertical {
                height: 6px;
                background-color: #e0e0e0;
                border-top: 1px solid #c0c0c0;
                border-bottom: 1px solid #c0c0c0;
            }
            QSplitter::handle:vertical:hover {
                background-color: #cccccc;
            }
        """)
        
        # 分子ビュー用の2ペイン（PyMOL + VTK）
        self.structure_view_splitter = QSplitter(Qt.Horizontal)
        self.structure_view_splitter.setHandleWidth(6)
        self.structure_view_splitter.setStyleSheet("""
            QSplitter::handle:horizontal {
                width: 6px;
                background-color: #e0e0e0;
                border-left: 1px solid #c0c0c0;
                border-right: 1px solid #c0c0c0;
            }
            QSplitter::handle:horizontal:hover {
                background-color: #cccccc;
            }
        """)

        # PyMOLビュー（画像 or 埋め込み）
        self.pymol_view_container = QWidget(self.structure_view_splitter)
        self.pymol_view_layout = QVBoxLayout(self.pymol_view_container)
        self.pymol_view_layout.setContentsMargins(0, 0, 0, 0)
        self.pymol_view_layout.setSpacing(0)
        self.pymol_widget_container = QWidget(self.pymol_view_container)
        self.pymol_widget_layout = QVBoxLayout(self.pymol_widget_container)
        self.pymol_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.pymol_widget_layout.setSpacing(0)
        self.pymol_placeholder = QLabel("PyMOL view (initializing...)")
        self.pymol_placeholder.setAlignment(Qt.AlignCenter)
        self.pymol_widget_layout.addWidget(self.pymol_placeholder)
        self.pymol_view_layout.addWidget(self.pymol_widget_container, 1)

        # ESP legend bar (shown only when ESP is enabled in PyMOL)
        self.esp_colorbar_widget = QWidget(self.pymol_view_container)
        esp_bar_layout = QHBoxLayout(self.esp_colorbar_widget)
        esp_bar_layout.setContentsMargins(8, 4, 8, 6)
        esp_bar_layout.setSpacing(6)
        neg = QLabel("−")
        neg.setToolTip("Negative")
        neg.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        pos = QLabel("+")
        pos.setToolTip("Positive")
        pos.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        for lbl in (neg, pos):
            lbl.setMinimumWidth(14)
            lbl.setStyleSheet("font-size: 10px; color: #444; font-weight: bold;")
        bar = _EspGradientBar(self.esp_colorbar_widget)
        esp_bar_layout.addWidget(neg)
        esp_bar_layout.addWidget(bar, 1)
        esp_bar_layout.addWidget(pos)
        self.esp_colorbar_widget.setVisible(False)
        self.pymol_view_layout.addWidget(self.esp_colorbar_widget, 0)

        # VTKビュー（インタラクティブ）
        self.vtk_view_container = QWidget(self.structure_view_splitter)
        vtk_layout = QVBoxLayout(self.vtk_view_container)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        vtk_layout.setSpacing(0)
        self.vtk_widget = QVTKRenderWindowInteractor(self.vtk_view_container)
        self.vtk_widget.setAcceptDrops(True)
        self.vtk_widget.installEventFilter(self)
        vtk_layout.addWidget(self.vtk_widget)
        self.display_widget = self.vtk_widget

        self.structure_view_splitter.addWidget(self.pymol_view_container)
        self.structure_view_splitter.addWidget(self.vtk_view_container)
        self.structure_view_splitter.setSizes([600, 600])
        # ドロップを受け付ける
        for w in (self.pymol_view_container, self.vtk_view_container, self.structure_view_splitter):
            try:
                w.setAcceptDrops(True)
                w.installEventFilter(self)
            except Exception:
                pass

        self.view_control_splitter.addWidget(self.structure_view_splitter)

        rotation_controls = self.create_rotation_controls()
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setWidget(rotation_controls)
        controls_scroll.setMinimumHeight(80)
        self.view_control_splitter.addWidget(controls_scroll)
        
        self.view_control_splitter.setSizes([560, 120])
        self.view_control_splitter.setCollapsible(0, False)
        self.view_control_splitter.setCollapsible(1, True)
        self.view_control_splitter.setStretchFactor(0, 5)
        self.view_control_splitter.setStretchFactor(1, 1)

        structure_layout.addWidget(self.view_control_splitter)

        # --- 下部：AFM像表示エリア --- (省略されていた部分を復元)
        afm_frame = QFrame()
        afm_frame.setFrameStyle(QFrame.StyledPanel)
        afm_frame.setLineWidth(1)
        afm_frame.setMinimumHeight(120)
        afm_layout = QVBoxLayout(afm_frame)
        afm_layout.setContentsMargins(2, 2, 2, 2)
        afm_layout.setSpacing(2)
        
        afm_header_layout = QHBoxLayout()
        afm_header_layout.setContentsMargins(3, 3, 3, 3)
        afm_header_layout.setSpacing(0)
        
        afm_label = QLabel("Simulated AFM Images")
        afm_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #333;
                padding: 3px;
                background-color: #f0f0f0;
                border-radius: 3px;
                margin-right: 0px;
            }
        """)
        afm_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        afm_header_layout.addWidget(afm_label)
        
        afm_header_layout.addSpacing(10)

        
    
        
        self.afm_x_check = QCheckBox("XY")
        self.afm_y_check = QCheckBox("YZ")
        self.afm_z_check = QCheckBox("ZX")
        
        self.afm_x_check.setChecked(True)
        self.afm_y_check.setChecked(False)
        self.afm_z_check.setChecked(False)
        
        checkbox_style = """
            QCheckBox {
                font-size: 10px; font-weight: bold; color: #555;
                spacing: 3px; margin-right: 2px;
            }
            QCheckBox::indicator { width: 14px; height: 14px; border-radius: 2px; }
            QCheckBox::indicator:checked { background-color: #4CAF50; border: 2px solid #45a049; }
            QCheckBox::indicator:unchecked { background-color: white; border: 2px solid #ccc; }
            QCheckBox::indicator:hover { border-color: #888; }
        """
        
        self.afm_x_check.setStyleSheet(checkbox_style)
        self.afm_y_check.setStyleSheet(checkbox_style)
        self.afm_z_check.setStyleSheet(checkbox_style)        
        
        self.afm_x_check.toggled.connect(self.update_afm_display)
        self.afm_y_check.toggled.connect(self.update_afm_display)
        self.afm_z_check.toggled.connect(self.update_afm_display)

        # 新しい接続（チェックがONになったらシミュレーションを自動実行する）
        self.afm_x_check.toggled.connect(self.run_simulation_on_view_change)
        self.afm_y_check.toggled.connect(self.run_simulation_on_view_change)
        self.afm_z_check.toggled.connect(self.run_simulation_on_view_change)
 
        
        afm_header_layout.addWidget(self.afm_x_check)
        afm_header_layout.addSpacing(12)
        afm_header_layout.addWidget(self.afm_y_check)
        afm_header_layout.addSpacing(12)
        afm_header_layout.addWidget(self.afm_z_check)

        afm_header_layout.addStretch(1)

        self.save_asd_button = QPushButton("💾 Save as ASD...")
        self.save_asd_button.setToolTip("Save AFM simulation data as ASD file\nAFMシミュレーションデータをASDファイルとして保存")
        self.save_asd_button.setStyleSheet("""
            QPushButton { padding: 3px 8px; font-size: 10px; background-color: #17a2b8; color: white; border-radius: 3px; }
            QPushButton:hover { background-color: #117a8b; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.save_asd_button.clicked.connect(self.handle_save_asd)
        self.save_asd_button.setEnabled(False) # 初期状態は無効
        afm_header_layout.addWidget(self.save_asd_button)

        self.save_image_button = QPushButton("🖼️ Save Image...") # アイコンを少し変更
        self.save_image_button.setToolTip("Save AFM simulation image as PNG/TIFF file\nAFMシミュレーション画像をPNG/TIFFファイルとして保存")
        self.save_image_button.setStyleSheet("""
            QPushButton { padding: 3px 8px; font-size: 10px; background-color: #007bff; color: white; border-radius: 3px; }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.save_image_button.clicked.connect(self.handle_save_image)
        self.save_image_button.setEnabled(False)
        afm_header_layout.addWidget(self.save_image_button)
        
        afm_header_widget = QWidget()
        afm_header_widget.setLayout(afm_header_layout)
        afm_header_widget.setMaximumHeight(30)
        afm_header_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        afm_layout.addWidget(afm_header_widget)
        
        self.afm_images_layout = QHBoxLayout()
        self.afm_images_layout.setSpacing(3)
        self.afm_images_layout.setContentsMargins(0, 0, 0, 0)
        
        # 画像パネルのタイトルを XY View, YZ View, ZX View に変更
        self.afm_x_frame = self.create_afm_image_panel("XY View")
        self.afm_x_frame.setObjectName("XY_Frame") # 追加
        self.afm_y_frame = self.create_afm_image_panel("YZ View")
        self.afm_y_frame.setObjectName("YZ_Frame") # 追加
        self.afm_z_frame = self.create_afm_image_panel("ZX View")
        self.afm_z_frame.setObjectName("ZX_Frame") # 追加
        
        self.afm_images_layout.addWidget(self.afm_x_frame, 1)
        self.afm_images_layout.addWidget(self.afm_y_frame, 1)
        self.afm_images_layout.addWidget(self.afm_z_frame, 1)
        
        afm_layout.addLayout(self.afm_images_layout)
        
        # メインスプリッターにウィジェットを追加
        self.afm_splitter.addWidget(structure_frame)
        self.afm_splitter.addWidget(afm_frame)
        
        self.afm_splitter.setSizes([560, 300])
        self.afm_splitter.setCollapsible(0, False)
        self.afm_splitter.setCollapsible(1, False)
        self.afm_splitter.setStretchFactor(0, 2)
        self.afm_splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(self.afm_splitter)

        self.update_afm_display()
        # Ensure Interactive Update default state is applied now that AFM view checkboxes exist.
        try:
            if hasattr(self, 'interactive_update_check'):
                self._apply_interactive_update_mode(
                    bool(self.interactive_update_check.isChecked()),
                    show_message=False,
                    run_initial=False
                )
        except Exception:
            pass
        
        return panel
    

    def create_rotation_controls(self):
        """PDB構造回転用コントロールと視点コントロールを作成"""
        group = QGroupBox("Structure & View Control (CTRL+Drag can rotate the PDB structure)")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        # メインの水平レイアウト
        main_layout = QHBoxLayout(group)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(8, 12, 8, 8)

        # --- 左側: Atom Stats + 回転コントロール ---
        left_widget = QWidget()
        left_container = QVBoxLayout(left_widget)
        left_container.setSpacing(6)
        left_container.setContentsMargins(0, 0, 0, 0)

        stats_group = QGroupBox("Atom Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_labels = {}
        for atom_type in ['Total', 'C', 'O', 'N', 'H', 'Other']:
            label = QLabel(f"{atom_type}: 0")
            label.setFont(QFont(STANDARD_FONT, 9))
            stats_layout.addWidget(label)
            self.stats_labels[atom_type] = label
        left_container.addWidget(stats_group)

        rotation_widget = QWidget()
        left_layout = QGridLayout(rotation_widget)
        left_layout.setSpacing(2)
        left_layout.setContentsMargins(0, 6, 0, 0)

        self.rotation_widgets = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            label = QLabel(f"Rotation {axis}:")
            spin_box = QDoubleSpinBox()
            spin_box.setRange(-180.0, 180.0)
            spin_box.setDecimals(1)
            spin_box.setSingleStep(1.0)
            spin_box.setSuffix(" °")
            spin_box.setToolTip(f"Rotation {axis} angle in degrees\n{axis}軸の回転角度（度）")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-1800, 1800)
            slider.setToolTip(f"Rotation {axis} slider\n{axis}軸回転スライダー")

            left_layout.addWidget(label, i, 0)
            left_layout.addWidget(spin_box, i, 1)
            left_layout.addWidget(slider, i, 2)
            left_layout.setColumnStretch(2, 1)

            self.rotation_widgets[axis] = {'spin': spin_box, 'slider': slider}
            # ★★★ ここからが修正箇所 ★★★
            # 1. 値が「変化している最中」は、UIの同期のみを行う
            slider.valueChanged.connect(self.sync_rotation_widgets)
            spin_box.valueChanged.connect(self.sync_rotation_widgets)
            
            # 2. 操作が「完了した時」にのみ、3Dモデルの回転とシミュレーションのトリガーを実行
            slider.sliderReleased.connect(self.apply_rotation_and_trigger_simulation)
            spin_box.valueChanged.connect(self.start_rotation_update_timer)
            #spin_box.editingFinished.connect(self.apply_rotation_and_trigger_simulation)
            # ★★★ 修正箇所ここまで ★★★

        left_container.addWidget(rotation_widget)

        # --- 右側: 視点コントロール ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(5, 0, 0, 0)
        
         # 1. ボタンを格納する水平レイアウトを作成
        top_button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset All")
        reset_btn.setToolTip("Reset molecule rotation, tip position, and camera view to initial state\n分子の回転、探針の位置、カメラの視点を初期状態に戻します")
        reset_btn.clicked.connect(self.handle_reset_button_clicked)
        top_button_layout.addWidget(reset_btn) # 水平レイアウトに追加

        # 2. 新しい保存ボタンを作成
        save_view_btn = QPushButton("📷 Save 3D View...")
        save_view_btn.setToolTip("Save the current 3D view as a PNG or TIFF image\n現在の3DビューをPNGまたはTIFF画像として保存")
        save_view_btn.clicked.connect(self.handle_save_3d_view) # 新しいメソッドに接続
        top_button_layout.addWidget(save_view_btn) # 水平レイアウトに追加   

        # Find Initial Plane（XY接触最大）ボタン
        find_plane_btn = QPushButton("Find Initial Plane")
        find_plane_btn.setToolTip("Rotate molecule to maximize XY-plane contact\n分子を回転してXY平面接触を最大化")
        find_plane_btn.clicked.connect(self.handle_find_initial_plane)
        top_button_layout.addWidget(find_plane_btn)

         # 3. 水平レイアウトを垂直レイアウトに追加
        right_layout.addLayout(top_button_layout)

        # Find Initial Plane parameters
        params_group = QGroupBox("Find Initial Plane Params")
        params_group_layout = QVBoxLayout(params_group)
        params_group_layout.setContentsMargins(8, 10, 8, 8)
        params_group_layout.setSpacing(6)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        params_scroll.setFrameShape(QFrame.NoFrame)

        params_content = QWidget()
        params_layout = QGridLayout(params_content)
        params_layout.setContentsMargins(6, 6, 6, 6)
        params_layout.setHorizontalSpacing(8)
        params_layout.setVerticalSpacing(8)
        params_group.setToolTip(
            "Controls how the initial support plane is estimated from a PDB.\n"
            "These parameters affect geometry, electrostatics, and search speed.\n"
            "\n"
            "PDBから支持面を推定する際の探索パラメータです。\n"
            "幾何・静電・速度のバランスに影響します。"
        )

        self.find_plane_params = {}

        use_elec_check = QCheckBox("Use electrostatics (residue charge)")
        use_elec_check.setChecked(False)
        use_elec_check.setToolTip(
            "Enable electrostatic term based on residue charges.\n"
            "When off, only geometric contact and spread are used.\n"
            "\n"
            "残基電荷に基づく静電項を有効にします。\n"
            "OFFの場合は幾何（接触＋広がり）のみで評価します。"
        )
        params_layout.addWidget(use_elec_check, 0, 0, 1, 2)

        base_row = 1

        def _add_param(row, label_text, widget, tooltip_text):
            label = QLabel(label_text)
            label.setToolTip(tooltip_text)
            params_layout.addWidget(label, base_row + row, 0)
            params_layout.addWidget(widget, base_row + row, 1)

        pH_spin = QDoubleSpinBox()
        pH_spin.setToolTip("")
        pH_spin.setRange(0.0, 14.0)
        pH_spin.setDecimals(2)
        pH_spin.setSingleStep(0.1)
        pH_spin.setValue(7.0)
        pH_tip = (
            "Solution pH used for residue charge estimation.\n"
            "Henderson–Hasselbalch average charges.\n"
            "\n"
            "残基の平均電荷計算に使うpH。\n"
            "Henderson–Hasselbalchで平均電荷を計算します。"
        )
        _add_param(0, "pH:", pH_spin, pH_tip)

        substrate_combo = QComboBox()
        substrate_combo.setToolTip("")
        substrate_combo.addItems(["MICA", "APTES_MICA"])
        substrate_tip = (
            "Substrate type affects electrostatic sign/strength.\n"
            "MICA: negative surface; APTES: pH-dependent positive.\n"
            "\n"
            "基板の種類で静電の符号・強度が変わります。\n"
            "MICA: 負電荷、APTES: pH依存の正電荷。"
        )
        _add_param(1, "substrate:", substrate_combo, substrate_tip)

        salt_combo = QComboBox()
        salt_combo.setToolTip("")
        salt_combo.addItems(["NaCl", "KCl"])
        salt_tip = (
            "Salt type (currently same screening model).\n"
            "Affects Debye length via concentration.\n"
            "\n"
            "塩の種類（遮蔽モデルは現状同じ）。\n"
            "濃度からDebye長を計算します。"
        )
        _add_param(2, "salt:", salt_combo, salt_tip)

        salt_mM_spin = QDoubleSpinBox()
        salt_mM_spin.setToolTip("")
        salt_mM_spin.setRange(0.0, 1000.0)
        salt_mM_spin.setDecimals(1)
        salt_mM_spin.setSingleStep(10.0)
        salt_mM_spin.setValue(50.0)
        salt_mM_tip = (
            "Salt concentration [mM] for Debye screening.\n"
            "Higher concentration -> shorter screening length.\n"
            "\n"
            "塩濃度 [mM]（Debye遮蔽に使用）。\n"
            "高濃度ほど遮蔽が強く、作用範囲が短くなります。"
        )
        _add_param(3, "salt [mM]:", salt_mM_spin, salt_mM_tip)

        alpha_spin = QDoubleSpinBox()
        alpha_spin.setToolTip("")
        alpha_spin.setRange(0.0, 5.0)
        alpha_spin.setDecimals(3)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(0.3)
        alpha_tip = (
            "Weight for electrostatic term S_elec in the score.\n"
            "Larger alpha increases electrostatic influence.\n"
            "\n"
            "静電項 S_elec の重み。\n"
            "大きいほど静電の影響を強めます。"
        )
        _add_param(4, "alpha:", alpha_spin, alpha_tip)

        rpseudo_spin = QDoubleSpinBox()
        rpseudo_spin.setToolTip("")
        rpseudo_spin.setRange(0.5, 6.0)
        rpseudo_spin.setDecimals(2)
        rpseudo_spin.setSingleStep(0.1)
        rpseudo_spin.setValue(2.0)
        rpseudo_tip = (
            "Pseudo-atom radius for each residue [Å].\n"
            "Larger values make contact easier for CA-based centers.\n"
            "Example: 2.0 Å (default)\n"
            "\n"
            "残基擬似原子の半径 [Å]。\n"
            "CA中心の場合は大きめにすると接触しやすくなります。\n"
            "例: 2.0 Å（デフォルト）"
        )
        _add_param(5, "r_pseudo [A]:", rpseudo_spin, rpseudo_tip)

        delta_elec_spin = QDoubleSpinBox()
        delta_elec_spin.setToolTip("")
        delta_elec_spin.setRange(0.5, 20.0)
        delta_elec_spin.setDecimals(1)
        delta_elec_spin.setSingleStep(0.5)
        delta_elec_spin.setValue(8.0)
        delta_elec_tip = (
            "Electrostatic shell thickness from the support plane [Å].\n"
            "Residues within 0..delta_elec contribute to Qshell.\n"
            "Example: 8.0 Å (default)\n"
            "\n"
            "支持面からの電荷評価殻の厚み [Å]。\n"
            "0..delta_elec の範囲にある残基が電荷に寄与します。\n"
            "例: 8.0 Å（デフォルト）"
        )
        _add_param(6, "delta_elec [A]:", delta_elec_spin, delta_elec_tip)

        k_geom_spin = QSpinBox()
        k_geom_spin.setToolTip("")
        k_geom_spin.setRange(1, 100)
        k_geom_spin.setSingleStep(1)
        k_geom_spin.setValue(10)
        k_geom_tip = (
            "Number of top geometric candidates to pass to electrostatics.\n"
            "Larger values allow more electrostatic re-ranking.\n"
            "Example: 10 (default)\n"
            "\n"
            "幾何スコア上位の候補数（電荷で再評価する数）。\n"
            "大きいほど電荷での再評価の余地が増えます。\n"
            "例: 10（デフォルト）"
        )
        _add_param(7, "K_geom:", k_geom_spin, k_geom_tip)

        n_dir_spin = QSpinBox()
        n_dir_spin.setToolTip("")
        n_dir_spin.setRange(200, 50000)
        n_dir_spin.setSingleStep(500)
        n_dir_spin.setValue(10000)
        n_dir_tip = (
            "Number of directions on the sphere (Fibonacci sampling).\n"
            "Larger = more global coverage but slower.\n"
            "Example: 10000 (default)\n"
            "\n"
            "球面上の探索方向数（フィボナッチサンプリング）。\n"
            "大きいほど網羅的だが遅くなります。\n"
            "例: 10000（デフォルト）"
        )
        _add_param(8, "N_dir:", n_dir_spin, n_dir_tip)

        k_spin = QSpinBox()
        k_spin.setToolTip("")
        k_spin.setRange(1, 500)
        k_spin.setSingleStep(10)
        k_spin.setValue(50)
        k_tip = (
            "Number of top directions kept for local refinement.\n"
            "Larger = better chance to escape local optima, but slower.\n"
            "Example: 50 (default)\n"
            "\n"
            "局所探索に回す上位候補の数。\n"
            "大きいほど精度が上がりやすいが遅くなります。\n"
            "例: 50（デフォルト）"
        )
        _add_param(9, "K:", k_spin, k_tip)

        delta_spin = QDoubleSpinBox()
        delta_spin.setToolTip("")
        delta_spin.setRange(0.1, 5.0)
        delta_spin.setDecimals(2)
        delta_spin.setSingleStep(0.1)
        delta_spin.setValue(1.5)
        delta_tip = (
            "Contact tolerance delta [Å].\n"
            "Atoms with gap g <= delta contribute; g = 0 means touching.\n"
            "Larger delta makes the contact patch thicker/softer.\n"
            "Example: 1.5 Å (default)\n"
            "\n"
            "接触許容幅 delta [Å]。\n"
            "ギャップ g <= delta の原子が接触に寄与（g=0 が接触）。\n"
            "大きいほど接触パッチが“厚く・甘く”なります。\n"
            "例: 1.5 Å（デフォルト）"
        )
        _add_param(10, "delta [A]:", delta_spin, delta_tip)

        lambda_spin = QDoubleSpinBox()
        lambda_spin.setToolTip("")
        lambda_spin.setRange(0.0, 1.0)
        lambda_spin.setDecimals(3)
        lambda_spin.setSingleStep(0.01)
        lambda_spin.setValue(0.02)
        lambda_tip = (
            "Weight for patch spread (variance) in the score.\n"
            "Score_geo = S_contact + lambda * Var_norm - 0.30 * (Height/delta).\n"
            "Larger lambda favors wide, stable contact patches.\n"
            "Example: 0.02 (default)\n"
            "\n"
            "パッチ広がり（分散）の重み。\n"
            "Score_geo = S_contact + lambda * Var_norm - 0.30 * (Height/delta)。\n"
            "大きいほど“広く安定に当たる面”を優先します。\n"
            "例: 0.02（デフォルト）"
        )
        _add_param(11, "lambda:", lambda_spin, lambda_tip)

        step1_spin = QDoubleSpinBox()
        step1_spin.setToolTip("")
        step1_spin.setRange(0.1, 10.0)
        step1_spin.setDecimals(1)
        step1_spin.setSingleStep(0.1)
        step1_spin.setValue(3.0)
        step1_tip = (
            "Local refinement step 1 [deg].\n"
            "Coarse angle for hill-climb around top directions.\n"
            "Example: 3.0 deg (default)\n"
            "\n"
            "局所探索ステップ1 [度]。\n"
            "上位方向の周りを粗く探索する角度。\n"
            "例: 3.0 度（デフォルト）"
        )
        _add_param(12, "theta1 [deg]:", step1_spin, step1_tip)

        step2_spin = QDoubleSpinBox()
        step2_spin.setToolTip("")
        step2_spin.setRange(0.1, 10.0)
        step2_spin.setDecimals(1)
        step2_spin.setSingleStep(0.1)
        step2_spin.setValue(1.0)
        step2_tip = (
            "Local refinement step 2 [deg].\n"
            "Medium angle after step 1.\n"
            "Example: 1.0 deg (default)\n"
            "\n"
            "局所探索ステップ2 [度]。\n"
            "ステップ1の後に中間スケールで詰める角度。\n"
            "例: 1.0 度（デフォルト）"
        )
        _add_param(13, "theta2 [deg]:", step2_spin, step2_tip)

        step3_spin = QDoubleSpinBox()
        step3_spin.setToolTip("")
        step3_spin.setRange(0.1, 10.0)
        step3_spin.setDecimals(1)
        step3_spin.setSingleStep(0.1)
        step3_spin.setValue(0.3)
        step3_tip = (
            "Local refinement step 3 [deg].\n"
            "Fine angle for the final polish.\n"
            "Example: 0.3 deg (default)\n"
            "\n"
            "局所探索ステップ3 [度]。\n"
            "最後に微調整するための細かい角度。\n"
            "例: 0.3 度（デフォルト）"
        )
        _add_param(14, "theta3 [deg]:", step3_spin, step3_tip)

        grid_spin = QSpinBox()
        grid_spin.setToolTip("")
        grid_spin.setRange(1, 4)
        grid_spin.setSingleStep(1)
        grid_spin.setValue(2)
        grid_tip = (
            "Local grid radius (in steps) around the current direction.\n"
            "Evaluations per step ≈ (2*grid+1)^2 - 1.\n"
            "Example: 2 -> 24 directions per step.\n"
            "\n"
            "局所探索の格子半径（ステップ数）。\n"
            "1ステップあたりの評価数 ≈ (2*grid+1)^2 - 1。\n"
            "例: 2 -> 24 方向/ステップ"
        )
        _add_param(15, "local grid:", grid_spin, grid_tip)

        neigh_r_spin = QDoubleSpinBox()
        neigh_r_spin.setToolTip("")
        neigh_r_spin.setRange(2.0, 12.0)
        neigh_r_spin.setDecimals(1)
        neigh_r_spin.setSingleStep(0.5)
        neigh_r_spin.setValue(6.0)
        neigh_r_tip = (
            "Surface filter radius [Å] for neighbor counting.\n"
            "Smaller radius keeps more atoms as 'surface'.\n"
            "Example: 6.0 Å (default)\n"
            "\n"
            "表面抽出に使う近傍半径 [Å]。\n"
            "小さいほど“表面”として残る原子が増えます。\n"
            "例: 6.0 Å（デフォルト）"
        )
        _add_param(16, "surf r [A]:", neigh_r_spin, neigh_r_tip)

        neigh_n_spin = QSpinBox()
        neigh_n_spin.setToolTip("")
        neigh_n_spin.setRange(1, 200)
        neigh_n_spin.setSingleStep(5)
        neigh_n_spin.setValue(25)
        neigh_n_tip = (
            "Surface filter neighbor threshold (count within radius).\n"
            "Atoms with neighbors <= threshold are kept as surface.\n"
            "Smaller threshold = fewer atoms, faster but rougher.\n"
            "Example: 25 (default)\n"
            "\n"
            "表面抽出の近傍数しきい値（半径内の原子数）。\n"
            "近傍数が閾値以下の原子を表面として残します。\n"
            "小さいほど原子数が減り高速だが粗くなります。\n"
            "例: 25（デフォルト）"
        )
        _add_param(17, "surf n:", neigh_n_spin, neigh_n_tip)

        def _toggle_elec_widgets(enabled):
            pH_spin.setEnabled(enabled)
            substrate_combo.setEnabled(enabled)
            salt_combo.setEnabled(enabled)
            salt_mM_spin.setEnabled(enabled)
            alpha_spin.setEnabled(enabled)
            rpseudo_spin.setEnabled(enabled)
            delta_elec_spin.setEnabled(enabled)
            k_geom_spin.setEnabled(enabled)

        use_elec_check.toggled.connect(_toggle_elec_widgets)
        _toggle_elec_widgets(use_elec_check.isChecked())

        self.find_plane_params["pH"] = pH_spin
        self.find_plane_params["substrate"] = substrate_combo
        self.find_plane_params["salt"] = salt_combo
        self.find_plane_params["salt_mM"] = salt_mM_spin
        self.find_plane_params["alpha_elec"] = alpha_spin
        self.find_plane_params["r_pseudo"] = rpseudo_spin
        self.find_plane_params["delta_elec"] = delta_elec_spin
        self.find_plane_params["K_geom"] = k_geom_spin
        self.find_plane_params["use_elec"] = use_elec_check
        self.find_plane_params["N_dir"] = n_dir_spin
        self.find_plane_params["K"] = k_spin
        self.find_plane_params["delta"] = delta_spin
        self.find_plane_params["lambda"] = lambda_spin
        self.find_plane_params["theta1"] = step1_spin
        self.find_plane_params["theta2"] = step2_spin
        self.find_plane_params["theta3"] = step3_spin
        self.find_plane_params["grid"] = grid_spin
        self.find_plane_params["surf_r"] = neigh_r_spin
        self.find_plane_params["surf_n"] = neigh_n_spin

        params_scroll.setWidget(params_content)
        params_group_layout.addWidget(params_scroll)
        right_layout.addWidget(params_group)

        params_io_layout = QHBoxLayout()
        self.save_params_button = QPushButton("Save Params...")
        self.save_params_button.setToolTip(
            "Save all simulator parameters to a JSON file.\n"
            "全パラメータをJSONとして保存します。"
        )
        self.save_params_button.clicked.connect(self.handle_save_params_json)
        params_io_layout.addWidget(self.save_params_button)

        self.load_params_button = QPushButton("Load Params...")
        self.load_params_button.setToolTip(
            "Load parameters from a JSON file and apply them.\n"
            "JSONからパラメータを読み込み反映します。"
        )
        self.load_params_button.clicked.connect(self.handle_load_params_json)
        params_io_layout.addWidget(self.load_params_button)

        right_layout.addLayout(params_io_layout)
        
        # 標準視点ボタンを水平に配置
        view_btn_layout = QHBoxLayout()
        xy_btn = QPushButton("XY")
        yz_btn = QPushButton("YZ")
        zx_btn = QPushButton("ZX")

        xy_btn.setToolTip("XY平面が画面に平行になるように視点を変更します (Z軸視点)")
        yz_btn.setToolTip("YZ平面が画面に平行になるように視点を変更します (X軸視点)")
        zx_btn.setToolTip("ZX平面が画面に平行になるように視点を変更します (Y軸視点)")

        xy_btn.clicked.connect(lambda: self.set_standard_view('xy'))
        yz_btn.clicked.connect(lambda: self.set_standard_view('yz'))
        zx_btn.clicked.connect(lambda: self.set_standard_view('zx'))

        view_btn_layout.addWidget(xy_btn)
        view_btn_layout.addWidget(yz_btn)
        view_btn_layout.addWidget(zx_btn)

        right_layout.addWidget(reset_btn)
        right_layout.addLayout(view_btn_layout)
        #right_layout.addStretch() # ボタンを上部に寄せる

        # 左右のウィジェットをメインレイアウトに追加
        main_layout.addWidget(left_widget, stretch=3) # 回転コントロールに多くのスペースを割り当てる
        main_layout.addWidget(right_widget, stretch=1)
                
        return group

    def handle_reset_button_clicked(self):
        """Resetボタンが押されたときの処理（回転、探針位置、カメラをリセット）"""
        self.reset_structure_rotation()
        self.reset_tip_position()
        self.reset_camera()

    def reset_tip_position(self):
        """探針の位置をUIのデフォルト値にリセットする"""
        if hasattr(self, 'tip_x_slider'):
            self.tip_x_slider.setValue(0)
            self.tip_y_slider.setValue(0)
            self.tip_z_slider.setValue(25) # UI定義時の初期値

    def set_standard_view(self, view_plane):
        """XY, YZ, ZXの標準視点にカメラをセットする（現在の距離を保持）"""
        if self._is_pymol_active():
            self._pymol_set_standard_view(view_plane)
            if self._is_pymol_only():
                return
        if not hasattr(self, 'renderer') or (not self.sample_actor and not (hasattr(self, 'mrc_actor') and self.mrc_actor is not None)):
            return

        camera = self.renderer.GetActiveCamera()
        
        # 現在のカメラの状態を保存
        current_position = camera.GetPosition()
        current_focal_point = camera.GetFocalPoint()
        current_view_up = camera.GetViewUp()
        
        # 現在のカメラと焦点の距離を計算
        distance = np.sqrt(sum((current_position[i] - current_focal_point[i]) ** 2 for i in range(3)))
        
        # 分子の中心を計算
        bbox = vtk.vtkBoundingBox()
        if self.sample_actor and self.show_molecule_check.isChecked():
            bbox.AddBounds(self.sample_actor.GetBounds())
        if self.bonds_actor and self.show_bonds_check.isChecked():
            bbox.AddBounds(self.bonds_actor.GetBounds())
        # MRCサーフェス
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            bbox.AddBounds(self.mrc_actor.GetBounds())
        
        if not bbox.IsValid():
            # 分子が表示されていない場合は、現在の焦点を中心とする
            molecule_center = current_focal_point
        else:
            # バウンディングボックスの中心を計算
            molecule_center = [0.0, 0.0, 0.0]
            bbox.GetCenter(molecule_center)

        # --- カメラの向きを設定（距離は保持） ---
        if view_plane == 'xy':
            # Z軸の上から見る (Y軸が画面の上方向)
            direction = np.array([0, 0, 1])
            new_position = np.array(molecule_center) + direction * distance
            camera.SetPosition(new_position[0], new_position[1], new_position[2])
            camera.SetFocalPoint(molecule_center)
            camera.SetViewUp(0, 1, 0)
        elif view_plane == 'yz':
            # X軸の正方向から見る (Z軸が画面の上方向)
            direction = np.array([1, 0, 0])
            new_position = np.array(molecule_center) + direction * distance
            camera.SetPosition(new_position[0], new_position[1], new_position[2])
            camera.SetFocalPoint(molecule_center)
            camera.SetViewUp(0, 0, 1)
        elif view_plane == 'zx':
            # Y軸の負方向から見る (Z軸が画面の上方向)
            direction = np.array([0, -1, 0])
            new_position = np.array(molecule_center) + direction * distance
            camera.SetPosition(new_position[0], new_position[1], new_position[2])
            camera.SetFocalPoint(molecule_center)
            camera.SetViewUp(0, 0, 1)
        
        # PDB分子の回転適用後、MRCアクターにも同じ回転を適用
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            self.mrc_actor.SetUserTransform(self.molecule_transform)
        
        # Tipの表示/透明度制御
        self._update_tip_visual_state(
            self.show_tip_check.isChecked() if hasattr(self, 'show_tip_check') else True
        )
        
        self.request_render()
        if self._is_dual_mode():
            self._sync_pymol_view_from_vtk()

    def _pymol_selection_for_atoms(self):
        """UIの原子選択をPyMOL selectionに変換"""
        obj = self.pymol_object_name
        atom_filter = self.atom_combo.currentText()
        if atom_filter == "All Atoms":
            return obj
        if atom_filter == "Heavy Atoms":
            return f"{obj} and not elem H"
        if atom_filter == "Backbone":
            return f"{obj} and name N+CA+C+O"
        if atom_filter in ["C", "N", "O"]:
            return f"{obj} and elem {atom_filter}"
        return obj

    def _pymol_apply_quality(self, quality):
        if not self._is_pymol_active():
            return
        try:
            if quality == "Fast":
                self.pymol_cmd.set("cartoon_sampling", 6)
                self.pymol_cmd.set("stick_quality", 5)
                self.pymol_cmd.set("sphere_quality", 0)
            elif quality == "Good":
                self.pymol_cmd.set("cartoon_sampling", 10)
                self.pymol_cmd.set("stick_quality", 8)
                self.pymol_cmd.set("sphere_quality", 1)
            else:  # High
                self.pymol_cmd.set("cartoon_sampling", 16)
                self.pymol_cmd.set("stick_quality", 12)
                self.pymol_cmd.set("sphere_quality", 2)
        except Exception:
            pass

    def _pymol_apply_opacity(self, opacity):
        if not self._is_pymol_active():
            return
        try:
            transparency = max(0.0, min(1.0, 1.0 - opacity))
            self.pymol_cmd.set("stick_transparency", transparency, self.pymol_object_name)
            self.pymol_cmd.set("sphere_transparency", transparency, self.pymol_object_name)
            self.pymol_cmd.set("cartoon_transparency", transparency, self.pymol_object_name)
            self.pymol_cmd.set("surface_transparency", transparency, self.pymol_object_name)
        except Exception:
            pass

    def _pymol_apply_color_scheme(self, selection):
        if not self._is_pymol_active():
            return
        color_scheme = self.color_combo.currentText()
        try:
            if color_scheme == "By Element":
                self.pymol_cmd.color("atomic", selection)
            elif color_scheme == "By Chain":
                try:
                    from pymol import util as pymol_util
                    pymol_util.cbc(selection)
                except Exception:
                    self.pymol_cmd.color("chainbow", selection)
            elif color_scheme == "Single Color":
                name = "pynud_single_color"
                self.pymol_cmd.set_color(name, [self.current_single_color[0],
                                                self.current_single_color[1],
                                                self.current_single_color[2]])
                self.pymol_cmd.color(name, selection)
            elif color_scheme == "By B-Factor":
                # B-factor coloring (also used as an ESP colormap when ESP is enabled)
                if hasattr(self, "esp_check") and self.esp_check.isChecked():
                    primary = "red_white_blue"
                    fallback = "blue_white_red"
                else:
                    primary = "blue_white_red"
                    fallback = "red_white_blue"
                try:
                    self.pymol_cmd.spectrum("b", primary, selection)
                except Exception:
                    self.pymol_cmd.spectrum("b", fallback, selection)
            else:
                self.pymol_cmd.color("atomic", selection)
        except Exception:
            pass

    def _display_molecule_pymol(self):
        """PyMOLで分子を表示"""
        if not self._is_pymol_active():
            return
        if not self.current_structure_path:
            return

        obj = self.pymol_object_name
        try:
            if self.pymol_loaded_path != self.current_structure_path:
                # 既存オブジェクトを削除して再ロード
                try:
                    self.pymol_cmd.delete(obj)
                except Exception:
                    pass
                self.pymol_cmd.load(self.current_structure_path, obj)
                self.pymol_loaded_path = self.current_structure_path
                self._pymol_last_ttt = None
                self.pymol_cmd.hide("everything", obj)
                # VTK側は読み込み時に (max+min)/2 で座標中心を原点へ移動しているため、
                # PyMOL側も同じ中心化を行って座標系を合わせる。
                try:
                    ext = self.pymol_cmd.get_extent(obj)
                    if ext and len(ext) == 2:
                        mn, mx = ext
                        center = [(mn[i] + mx[i]) * 0.5 for i in range(3)]
                        self.pymol_cmd.translate([-center[0], -center[1], -center[2]], obj, camera=0)
                except Exception:
                    pass
                try:
                    self.pymol_cmd.matrix_reset(obj)
                except Exception:
                    pass
                self.pymol_cmd.orient(obj)
        except Exception as e:
            print(f"[WARNING] PyMOL load failed: {e}")
            return

        style = self.style_combo.currentText()
        size_factor = self.size_slider.value() / 100.0
        opacity = self.opacity_slider.value() / 100.0
        quality = self.quality_combo.currentText()

        selection = self._pymol_selection_for_atoms()

        try:
            self.pymol_cmd.hide("everything", obj)
        except Exception:
            pass

        # スタイルに応じた表示
        try:
            if style == "Ball & Stick":
                self.pymol_cmd.show("sticks", selection)
                self.pymol_cmd.show("spheres", selection)
                self.pymol_cmd.set("stick_radius", 0.15 * size_factor, obj)
                self.pymol_cmd.set("sphere_scale", 0.25 * size_factor, obj)
            elif style == "Stick Only":
                self.pymol_cmd.show("sticks", selection)
                self.pymol_cmd.set("stick_radius", 0.15 * size_factor, obj)
            elif style == "Spheres":
                self.pymol_cmd.show("spheres", selection)
                self.pymol_cmd.set("sphere_scale", 0.30 * size_factor, obj)
            elif style == "Points":
                self.pymol_cmd.show("dots", selection)
                self.pymol_cmd.set("dot_radius", 0.20 * size_factor, obj)
            elif style == "Wireframe":
                self.pymol_cmd.show("lines", selection)
            elif style == "Simple Cartoon":
                self.pymol_cmd.show("cartoon", selection)
                self.pymol_cmd.set("cartoon_fancy_helices", 0)
                self.pymol_cmd.set("cartoon_smooth_loops", 0)
            elif style == "Ribbon":
                self.pymol_cmd.show("cartoon", selection)
                self.pymol_cmd.set("cartoon_fancy_helices", 1)
                self.pymol_cmd.set("cartoon_smooth_loops", 1)
        except Exception:
            pass

        # 表示品質・色・透明度
        self._pymol_apply_quality(quality)
        self._pymol_apply_color_scheme(selection)
        self._pymol_apply_opacity(opacity)

        # Electrostatics表示が有効なら再適用
        if hasattr(self, 'esp_check') and self.esp_check.isChecked():
            self.display_electrostatics(True)

        self.request_render()

    def display_electrostatics(self, enabled):
        """表面電荷（ESP）表示を切り替える"""
        if not self._is_pymol_active():
            return
        self._apply_esp_color_lock()
        obj = self.pymol_object_name
        if not self.current_structure_path:
            return
        try:
            if enabled:
                # 既存のESPオブジェクトを掃除
                if self.pymol_esp_object:
                    try:
                        self.pymol_cmd.delete(self.pymol_esp_object)
                    except Exception:
                        pass
                # PyMOL標準の簡易ESP
                esp_obj = f"{obj}_esp"
                try:
                    if hasattr(self.pymol_cmd, "protein_vacuum_electrostatics"):
                        self.pymol_cmd.protein_vacuum_electrostatics(obj, esp_obj)
                    else:
                        from pymol import util as pymol_util
                        if hasattr(pymol_util, "protein_vacuum_electrostatics"):
                            pymol_util.protein_vacuum_electrostatics(obj, esp_obj)
                except Exception:
                    # フォールバック: 表面＋B-factorスペクトラム
                    self.pymol_cmd.show("surface", obj)
                    try:
                        self.pymol_cmd.spectrum("b", "red_white_blue", obj)
                    except Exception:
                        self.pymol_cmd.spectrum("b", "blue_white_red", obj)
                    esp_obj = None
                self.pymol_esp_object = esp_obj
                self.pymol_cmd.show("surface", obj)
                # Keep ESP colormap stable even if the user previously chose element/chain colors.
                try:
                    self.pymol_cmd.spectrum("b", "red_white_blue", obj)
                except Exception:
                    try:
                        self.pymol_cmd.spectrum("b", "blue_white_red", obj)
                    except Exception:
                        pass
            else:
                if self.pymol_esp_object:
                    try:
                        self.pymol_cmd.delete(self.pymol_esp_object)
                    except Exception:
                        pass
                    self.pymol_esp_object = None
                self.pymol_cmd.hide("surface", obj)
        except Exception:
            pass

        self._update_esp_colorbar_visibility()
        self.request_render()
        if not enabled:
            # Restore non-ESP coloring immediately when turning ESP off.
            try:
                self.update_display()
            except Exception:
                pass

    def _apply_esp_color_lock(self):
        """When ESP is enabled, force coloring to 'By B-Factor' to avoid mixed colors."""
        try:
            if not hasattr(self, "color_combo") or self.color_combo is None:
                return
            esp_effective = bool(
                self.pymol_available
                and self.pymol_cmd is not None
                and self.render_backend in ("pymol", "dual")
                and hasattr(self, "esp_check")
                and self.esp_check.isChecked()
                and getattr(self, "current_structure_type", None) != "mrc"
            )
            if esp_effective:
                if self._color_scheme_before_esp is None:
                    self._color_scheme_before_esp = self.color_combo.currentText()
                if self.color_combo.currentText() != "By B-Factor":
                    self.color_combo.blockSignals(True)
                    self.color_combo.setCurrentText("By B-Factor")
                    self.color_combo.blockSignals(False)
                self.color_combo.setEnabled(False)
                self.color_combo.setToolTip("ESP is enabled: color is forced to 'By B-Factor' (red-white-blue).")
            else:
                if self._color_scheme_before_esp is not None:
                    prev = self._color_scheme_before_esp
                    self._color_scheme_before_esp = None
                    self.color_combo.setEnabled(True)
                    self.color_combo.blockSignals(True)
                    self.color_combo.setCurrentText(prev)
                    self.color_combo.blockSignals(False)
                else:
                    self.color_combo.setEnabled(True)
        except Exception:
            pass

    def _update_esp_colorbar_visibility(self):
        """ESP表示時にPyMOLビュー下へカラーバーを表示"""
        try:
            if not hasattr(self, "esp_colorbar_widget") or self.esp_colorbar_widget is None:
                return
            show = bool(
                self.pymol_available
                and hasattr(self, "esp_check")
                and self.esp_check.isChecked()
                and self.render_backend in ("pymol", "dual")
                and getattr(self, "current_structure_type", None) != "mrc"
            )
            self.esp_colorbar_widget.setVisible(show)
        except Exception:
            pass

    def on_xy_checked(self, checked):
        if checked:
            # 他のチェックボックスの信号を一時的にブロック
            self.afm_y_check.blockSignals(True)
            self.afm_z_check.blockSignals(True)
            # 他をオフにする
            self.afm_y_check.setChecked(False)
            self.afm_z_check.setChecked(False)
            # ブロックを解除
            self.afm_y_check.blockSignals(False)
            self.afm_z_check.blockSignals(False)
        self.update_afm_display()

    def on_yz_checked(self, checked):
        if checked:
            self.afm_x_check.blockSignals(True)
            self.afm_z_check.blockSignals(True)
            self.afm_x_check.setChecked(False)
            self.afm_z_check.setChecked(False)
            self.afm_x_check.blockSignals(False)
            self.afm_z_check.blockSignals(False)
        self.update_afm_display()

    def on_zx_checked(self, checked):
        if checked:
            self.afm_x_check.blockSignals(True)
            self.afm_y_check.blockSignals(True)
            self.afm_x_check.setChecked(False)
            self.afm_y_check.setChecked(False)
            self.afm_x_check.blockSignals(False)
            self.afm_y_check.blockSignals(False)
        self.update_afm_display()
    
    def sync_rotation_widgets(self):
        """スライダーとスピンボックスの値を同期させ、Interactive Updateが有効な場合はリアルタイム更新も実行"""
        sender = self.sender()
        changed_axis = None
        for axis, widgets in self.rotation_widgets.items():
            if sender is widgets['slider'] or sender is widgets['spin']:
                changed_axis = axis
                break
        if not changed_axis: return

        widgets = self.rotation_widgets[changed_axis]
        spin_box = widgets['spin']
        slider = widgets['slider']

        # 無限ループを防ぐため、シグナルをブロックしながら値を設定
        if isinstance(sender, QSlider):
            new_val = sender.value() / 10.0
            spin_box.blockSignals(True)
            spin_box.setValue(new_val)
            spin_box.blockSignals(False)
        elif isinstance(sender, QDoubleSpinBox):
            new_val = sender.value()
            slider.blockSignals(True)
            slider.setValue(int(new_val * 10))
            slider.blockSignals(False)
        
        # 構造回転を適用
        self.apply_structure_rotation()
        
        # Interactive Updateが有効で、スライダーからの変更の場合はリアルタイム更新
        if (hasattr(self, 'interactive_update_check') and 
            self.interactive_update_check.isChecked() and 
            isinstance(sender, QSlider)):
            self.run_simulation_immediate_controlled()
    
    def start_rotation_update_timer(self):
        """
        スピンボックスからの回転更新を遅延させるためのタイマーを開始/リセットする。
        これにより、連続クリック中に不要な更新が走るのを防ぐ。
        """
        # タイマーがまだ存在しなければ作成する
        if not hasattr(self, 'rotation_update_timer'):
            self.rotation_update_timer = QTimer(self)  # 親ウィンドウを設定
            self.rotation_update_timer.setSingleShot(True)
            self.rotation_update_timer.timeout.connect(self.apply_rotation_and_trigger_simulation)
        
        # 500ミリ秒後に更新を実行するようにタイマーを開始（またはリセット）
        self.rotation_update_timer.start(500)

    def apply_rotation_and_trigger_simulation(self):
        """UIの操作完了後に、3Dモデルの回転を適用し、シミュレーションをトリガーする"""
        #print("Rotation change finished. Applying transform and triggering simulation if interactive.")
        self.apply_structure_rotation()
        
        # Interactive Updateが有効な場合は高解像度シミュレーションをスケジュール
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            if hasattr(self, 'schedule_high_res_simulation'):
                self.schedule_high_res_simulation()

    def update_rotation(self):
        """回転コントロールの変更を検知し、UIを同期して回転を適用"""
        sender = self.sender()

        # どの軸のウィジェットが変更されたか特定
        changed_axis = None
        for axis, widgets in self.rotation_widgets.items():
            if sender is widgets['slider'] or sender is widgets['spin']:
                changed_axis = axis
                break
        
        if not changed_axis:
            return

        widgets = self.rotation_widgets[changed_axis]
        spin_box = widgets['spin']
        slider = widgets['slider']

        # senderに応じて値を同期（無限ループを防ぐためシグナルをブロック）
        if isinstance(sender, QSlider):
            new_val = sender.value() / 10.0
            spin_box.blockSignals(True)
            spin_box.setValue(new_val)
            spin_box.blockSignals(False)
        elif isinstance(sender, QDoubleSpinBox):
            new_val = sender.value()
            slider.blockSignals(True)
            slider.setValue(int(new_val * 10))
            slider.blockSignals(False)
        else:
            return

        # 実際の回転を適用
        self.apply_structure_rotation()

    def update_actor_transform(self):
        """base_transformとlocal_transformを組み合わせてアクターに適用"""
        if self._is_pymol_only():
            return
        try:
            # 変換行列を安全に初期化
            self.combined_transform.Identity()
            
            # 変換行列の妥当性をチェック
            if self.base_transform is not None:
                base_matrix = self.base_transform.GetMatrix()
                if self._is_transform_matrix_valid(base_matrix):
                    self.combined_transform.Concatenate(self.base_transform)
                else:
                    print("[WARNING] Invalid base_transform, using identity")
            
            if self.local_transform is not None:
                local_matrix = self.local_transform.GetMatrix()
                if self._is_transform_matrix_valid(local_matrix):
                    self.combined_transform.Concatenate(self.local_transform)
                else:
                    print("[WARNING] Invalid local_transform, using identity")
            
            # 最終的な変換行列の妥当性をチェック
            final_matrix = self.combined_transform.GetMatrix()
            if not self._is_transform_matrix_valid(final_matrix):
                print("[WARNING] Invalid combined_transform, resetting to identity")
                self.combined_transform.Identity()
            
            # アクターに適用
            if self.sample_actor:
                self.sample_actor.SetUserTransform(self.combined_transform)
            if self.bonds_actor:
                self.bonds_actor.SetUserTransform(self.combined_transform)
            if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
                self.mrc_actor.SetUserTransform(self.combined_transform)
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
            if self._is_dual_mode() and self._is_pymol_active():
                self._mark_pymol_interaction()
                self._sync_pymol_object_ttt_from_vtk()
                
        except Exception as e:
            print(f"[WARNING] Error in update_actor_transform: {e}")
            # エラーが発生した場合は単位行列にリセット
            self.combined_transform.Identity()
    
    def _is_transform_matrix_valid(self, vtk_matrix):
        """VTK変換行列が妥当かどうかをチェック"""
        try:
            for i in range(4):
                for j in range(4):
                    element = vtk_matrix.GetElement(i, j)
                    if not np.isfinite(element) or abs(element) > 1e6:
                        return False
            return True
        except Exception:
            return False

    def apply_structure_rotation(self, trigger_simulation=True):
        """スライダー（絶対角）→ 差分回転をlocal_transformに適用"""
        if not hasattr(self, 'rotation_widgets'):
            return
        
        # PDBデータまたはMRCデータのどちらかが読み込まれているかチェック
        if (getattr(self, 'atoms_data', None) is None and 
            not (hasattr(self, 'mrc_data') and self.mrc_data is not None)):
            return

        # 実行中ワーカーのガード（元コードと同様）
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            return

        rx = float(self.rotation_widgets['X']['spin'].value())
        ry = float(self.rotation_widgets['Y']['spin'].value())
        rz = float(self.rotation_widgets['Z']['spin'].value())

        dx = rx - self.prev_rot['x']
        dy = ry - self.prev_rot['y']
        dz = rz - self.prev_rot['z']

        if self._is_pymol_active():
            # Dual(P yMOL + VTK) では、PyMOL側は VTK の「カメラ回転×オブジェクト回転」を
            # `_sync_pymol_view_from_vtk()` で反映する。ここで PyMOL を個別に turn/rotate すると
            # VTKと見た目がズレる（特に Cmd+ドラッグの構造回転）。
            if not self._is_dual_mode():
                try:
                    # PyMOL-only の場合はビュー回転として適用
                    self.pymol_cmd.turn("x", dx)
                    self.pymol_cmd.turn("y", dy)
                    self.pymol_cmd.turn("z", dz)
                except Exception:
                    try:
                        # turnが使えない場合はcamera回転を試す
                        self.pymol_cmd.rotate("x", dx, self.pymol_object_name, camera=1)
                        self.pymol_cmd.rotate("y", dy, self.pymol_object_name, camera=1)
                        self.pymol_cmd.rotate("z", dz, self.pymol_object_name, camera=1)
                    except Exception:
                        pass
                self.request_render()

        if not self._is_pymol_only():
            # VTK側は絶対角から毎回local_transformを再構築（ドラッグ操作でも安定させる）
            self.local_transform.Identity()
            self.local_transform.PostMultiply()
            self.local_transform.RotateX(rx)
            self.local_transform.RotateY(ry)
            self.local_transform.RotateZ(rz)
            self.update_actor_transform()
            if self._is_dual_mode() and self._is_pymol_active():
                # カメラ同期はModifiedEventで行うが、保険としても呼ぶ
                self._sync_pymol_view_from_vtk()

        self.prev_rot['x'], self.prev_rot['y'], self.prev_rot['z'] = rx, ry, rz
        if trigger_simulation:
            self.trigger_interactive_simulation()

    def _vdw_radius_angstrom(self, element):
        """Return vdW radius in Å (Bondi-like)."""
        if element is None:
            return 1.7
        s = str(element).strip()
        if not s:
            return 1.7
        elem = (s[0].upper() + s[1:].lower()) if len(s) >= 2 else s.upper()
        table = {
            "H": 1.20,
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "F": 1.47,
            "P": 1.80,
            "S": 1.80,
            "Cl": 1.75,
            "Br": 1.85,
            "I": 1.98,
            "Zn": 1.39,
            "Mg": 1.73,
            "Na": 2.27,
            "K": 2.75,
            "Ca": 2.31,
            "Fe": 1.80,
        }
        return table.get(elem, 1.70)

    def _frac_deprot(self, pH, pKa):
        return 1.0 / (1.0 + (10.0 ** (pKa - pH)))

    def _frac_prot(self, pH, pKa):
        return 1.0 / (1.0 + (10.0 ** (pH - pKa)))

    def _residue_charge(self, res_name, pH):
        """Return average residue charge at given pH (Henderson–Hasselbalch)."""
        rn = str(res_name).upper().strip()
        if rn == "ASP":
            return -self._frac_deprot(pH, 3.9)
        if rn == "GLU":
            return -self._frac_deprot(pH, 4.3)
        if rn == "HIS":
            return +self._frac_prot(pH, 6.0)
        if rn == "LYS":
            return +self._frac_prot(pH, 10.5)
        if rn == "ARG":
            return +self._frac_prot(pH, 12.5)
        return 0.0

    def _build_pseudo_atoms_from_residues(self, pH, r_pseudo=2.0):
        """Build residue-level pseudo atoms from current atoms_data (Å)."""
        if self.atoms_data is None:
            return None

        standard_res = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }

        x = self.atoms_data.get('x', None)
        y = self.atoms_data.get('y', None)
        z = self.atoms_data.get('z', None)
        res_name = self.atoms_data.get('residue_name', None)
        atom_name = self.atoms_data.get('atom_name', None)
        chain_id = self.atoms_data.get('chain_id', None)
        res_id = self.atoms_data.get('residue_id', None)
        icode = self.atoms_data.get('icode', None)
        if x is None or y is None or z is None or res_name is None or chain_id is None or res_id is None:
            return None

        n = len(x)
        if icode is None or len(icode) != n:
            icode = np.array([""] * n)

        coords_nm = np.column_stack([x, y, z]).astype(float)
        coords_ang = coords_nm * 10.0

        groups = {}
        chains = {}
        for i in range(n):
            rn = str(res_name[i]).upper().strip()
            if rn not in standard_res:
                continue
            ch = str(chain_id[i]) if chain_id[i] is not None else " "
            ch = ch if ch != "" else " "
            rid = int(res_id[i])
            ic = str(icode[i]) if icode is not None else ""
            key = (ch, rid, ic)
            g = groups.get(key)
            if g is None:
                g = {
                    "res_name": rn,
                    "coords": [],
                    "ca": None,
                }
                groups[key] = g
            g["coords"].append(coords_ang[i])
            an = str(atom_name[i]).strip() if atom_name is not None else ""
            if an == "CA":
                g["ca"] = coords_ang[i]
            chains.setdefault(ch, []).append((rid, ic, key))

        if not groups:
            return None

        # Add termini charges per chain
        n_term_pKa = 8.0
        c_term_pKa = 3.1
        for ch, items in chains.items():
            if not items:
                continue
            items_sorted = sorted(items, key=lambda t: (t[0], t[1]))
            n_key = items_sorted[0][2]
            c_key = items_sorted[-1][2]
            groups[n_key]["n_term"] = True
            groups[c_key]["c_term"] = True

        pseudo = []
        for key, g in groups.items():
            if g["ca"] is not None:
                pos = g["ca"]
            else:
                coords = np.array(g["coords"], dtype=float)
                pos = coords.mean(axis=0)
            q = self._residue_charge(g["res_name"], pH)
            if g.get("n_term", False):
                q += self._frac_prot(pH, n_term_pKa)
            if g.get("c_term", False):
                q -= self._frac_deprot(pH, c_term_pKa)
            pseudo.append((pos[0], pos[1], pos[2], float(r_pseudo), float(q)))

        if not pseudo:
            return None

        coords = np.array([[p[0], p[1], p[2]] for p in pseudo], dtype=float)
        radii = np.array([p[3] for p in pseudo], dtype=float)
        charges = np.array([p[4] for p in pseudo], dtype=float)
        return coords, radii, charges

    def _select_surface_atoms(self, coords_ang, neighbor_radius=6.0, neighbor_threshold=25):
        """
        Rough surface filter using neighbor counts within radius.
        coords_ang: (N,3) in Å.
        """
        n = int(coords_ang.shape[0])
        if n < 200:
            return np.ones(n, dtype=bool)
        try:
            from scipy.spatial import cKDTree  # local import to avoid hard dependency
        except Exception:
            return np.ones(n, dtype=bool)

        tree = cKDTree(coords_ang)
        try:
            counts = tree.query_ball_point(coords_ang, r=neighbor_radius, return_length=True)
        except TypeError:
            # Fallback for older SciPy without return_length
            counts = np.array([len(lst) for lst in tree.query_ball_point(coords_ang, r=neighbor_radius)], dtype=int)
        return counts <= neighbor_threshold

    def _fibonacci_sphere_directions(self, n_dir):
        """Generate roughly uniform directions on a sphere."""
        n_dir = int(max(1, n_dir))
        i = np.arange(n_dir, dtype=float)
        golden = (1.0 + 5.0 ** 0.5) / 2.0
        z = 1.0 - (2.0 * i + 1.0) / n_dir
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        phi = (2.0 * np.pi * i) / golden
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.column_stack([x, y, z])

    def _make_perp_unit(self, n):
        """Create a unit vector perpendicular to n."""
        n = np.asarray(n, dtype=float)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(ref, n)
        norm = float(np.linalg.norm(e1))
        if norm < 1e-12:
            ref = np.array([1.0, 0.0, 0.0])
            e1 = np.cross(ref, n)
            norm = float(np.linalg.norm(e1))
        return e1 / max(1e-12, norm)

    def _eval_support_score(self, coords_ang, radii_ang, n, delta, lambda_var,
                            charges=None, lambdaD_A=None, alpha_elec=0.0,
                            s_sub=0.0, A_sub=0.0):
        """Evaluate Score(n) and related metrics (geometry + electrostatics)."""
        n = np.asarray(n, dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            return None
        n = n / n_norm

        proj = coords_ang @ n
        t = proj - radii_ang
        d = float(np.max(t))
        g = d - t
        s = delta - g
        mask = s > 0.0
        if not np.any(mask):
            return {
                "n": n,
                "d": d,
                "score": -1e300,
                "S_contact": 0.0,
                "var": 0.0,
                "Qpatch": 0.0,
                "S_elec": 0.0,
                "mask": mask,
            }
        w = s[mask]
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            return {
                "n": n,
                "d": d,
                "score": -1e300,
                "S_contact": 0.0,
                "var": 0.0,
                "Qpatch": 0.0,
                "S_elec": 0.0,
                "mask": mask,
            }

        e1 = self._make_perp_unit(n)
        e2 = np.cross(n, e1)

        u = coords_ang @ e1
        v = coords_ang @ e2
        um = u[mask]
        vm = v[mask]
        U = float(np.sum(w * um) / wsum)
        V = float(np.sum(w * vm) / wsum)
        du = um - U
        dv = vm - V
        var = float(np.sum(w * (du * du + dv * dv)) / wsum)

        Qpatch = 0.0
        S_elec = 0.0
        if charges is not None:
            g_m = g[mask]
            if lambdaD_A is None or lambdaD_A <= 0.0:
                w_e = w
            else:
                w_e = w * np.exp(-g_m / float(lambdaD_A))
            Qpatch = float(np.sum(charges[mask] * w_e))
            S_elec = -float(A_sub) * float(s_sub) * Qpatch

        score = float(wsum + lambda_var * var + alpha_elec * S_elec)
        return {
            "n": n,
            "d": d,
            "score": score,
            "S_contact": wsum,
            "var": var,
            "Qpatch": Qpatch,
            "S_elec": S_elec,
            "mask": mask,
        }

    def _eval_geo(self, coordsA, radiiA, n, delta_contact_A, lambda_var):
        """Evaluate geometric score using atom-based coordinates."""
        n = np.asarray(n, dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            return None
        n = n / n_norm

        proj = coordsA @ n
        t = proj - radiiA
        d = float(np.max(t))
        g = d - t
        height_A = float(np.max(g)) if g.size > 0 else 0.0
        delta_safe = max(float(delta_contact_A), 1e-6)
        height_norm = height_A / delta_safe
        # Encourage flatter poses against the support plane.
        height_weight = 0.30
        s = delta_contact_A - g
        mask = s > 0.0
        if not np.any(mask):
            return {
                "n": n,
                "d": d,
                "score_geo": -1e300,
                "S_contact": 0.0,
                "var": 0.0,
                "height_A": height_A,
                "height_norm": height_norm,
                "Nc": 0,
                "mask": mask,
            }
        w = s[mask]
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            return {
                "n": n,
                "d": d,
                "score_geo": -1e300,
                "S_contact": 0.0,
                "var": 0.0,
                "height_A": height_A,
                "height_norm": height_norm,
                "Nc": int(np.count_nonzero(mask)),
                "mask": mask,
            }

        e1 = self._make_perp_unit(n)
        e2 = np.cross(n, e1)
        u = coordsA[mask] @ e1
        v = coordsA[mask] @ e2
        U = float(np.sum(w * u) / wsum)
        V = float(np.sum(w * v) / wsum)
        du = u - U
        dv = v - V
        var = float(np.sum(w * (du * du + dv * dv)) / wsum)
        # Normalize Var by a global geometry scale to reduce size bias
        geom_scale = self._geom_var_scale if hasattr(self, "_geom_var_scale") else None
        if geom_scale is None or geom_scale <= 0.0:
            var_norm = var
        else:
            var_norm = var / geom_scale
        score_geo = float(wsum + lambda_var * var_norm - height_weight * height_norm)
        return {
            "n": n,
            "d": d,
            "score_geo": score_geo,
            "S_contact": wsum,
            "var": var,
            "var_norm": var_norm,
            "height_A": height_A,
            "height_norm": height_norm,
            "Nc": int(np.count_nonzero(mask)),
            "mask": mask,
        }

    def _eval_elec(self, coordsR, qR, n, d, delta_elec_A, lambdaD_A, A_sub, s_sub):
        """Evaluate electrostatic term using residue pseudo charges."""
        if coordsR is None or qR is None or len(coordsR) == 0:
            return {"Qshell": 0.0, "S_elec": 0.0, "Nr": 0}
        n = np.asarray(n, dtype=float)
        proj = coordsR @ n
        gR = d - proj
        maskR = (gR >= 0.0) & (gR < float(delta_elec_A))
        if not np.any(maskR):
            return {"Qshell": 0.0, "S_elec": 0.0, "Nr": 0}
        if lambdaD_A is None or lambdaD_A <= 0.0:
            wR = np.ones(np.count_nonzero(maskR), dtype=float)
        else:
            wR = np.exp(-gR[maskR] / float(lambdaD_A))
        Qshell = float(np.sum(qR[maskR] * wR))
        S_elec = -float(A_sub) * float(s_sub) * Qshell
        return {"Qshell": Qshell, "S_elec": S_elec, "Nr": int(np.count_nonzero(maskR))}

    def _compute_geom_var_scale(self, coordsA):
        """Compute normalization scale for Var (Å^2) from bounding box size."""
        try:
            mins = np.min(coordsA, axis=0)
            maxs = np.max(coordsA, axis=0)
            span = maxs - mins
            diag = float(np.linalg.norm(span))
            scale = (0.5 * diag) ** 2
            return max(scale, 1e-6)
        except Exception:
            return 1.0

    def _apply_initial_plane_rotation(self, R, center):
        """Apply rotation matrix R around center to the structure."""
        R = np.asarray(R, dtype=float)
        c = np.asarray(center, dtype=float)
        t = c - R @ c

        def matrix_to_euler_zyx(Rm):
            sy = np.hypot(Rm[0, 0], Rm[1, 0])
            singular = sy < 1e-8
            if not singular:
                z = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))
                y = np.degrees(np.arctan2(-Rm[2, 0], sy))
                x = np.degrees(np.arctan2(Rm[2, 1], Rm[2, 2]))
            else:
                z = 0.0
                y = np.degrees(np.arctan2(-Rm[2, 0], sy))
                x = np.degrees(np.arctan2(-Rm[1, 2], Rm[1, 1]))
            def _wrap(a):
                return (a + 180) % 360 - 180
            return _wrap(x), _wrap(y), _wrap(z)

        if self._is_pymol_active():
            rot_x, rot_y, rot_z = matrix_to_euler_zyx(R)
            if hasattr(self, 'rotation_widgets'):
                for ax, val in zip(('X', 'Y', 'Z'), (rot_x, rot_y, rot_z)):
                    self.rotation_widgets[ax]['spin'].blockSignals(True)
                    self.rotation_widgets[ax]['slider'].blockSignals(True)
                    self.rotation_widgets[ax]['spin'].setValue(val)
                    self.rotation_widgets[ax]['slider'].setValue(int(val * 10))
                    self.rotation_widgets[ax]['spin'].blockSignals(False)
                    self.rotation_widgets[ax]['slider'].blockSignals(False)
            self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self.apply_structure_rotation()
            if hasattr(self, 'trigger_interactive_simulation'):
                self.trigger_interactive_simulation()
            return

        M = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                M.SetElement(i, j, float(R[i, j]))
        M.SetElement(0, 3, float(t[0]))
        M.SetElement(1, 3, float(t[1]))
        M.SetElement(2, 3, float(t[2]))
        M.SetElement(3, 0, 0.0)
        M.SetElement(3, 1, 0.0)
        M.SetElement(3, 2, 0.0)
        M.SetElement(3, 3, 1.0)

        self.base_transform.Identity()
        self.base_transform.SetMatrix(M)
        self.local_transform.Identity()
        self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        if hasattr(self, 'rotation_widgets'):
            for ax in ('X', 'Y', 'Z'):
                self.rotation_widgets[ax]['spin'].blockSignals(True)
                self.rotation_widgets[ax]['slider'].blockSignals(True)
                self.rotation_widgets[ax]['spin'].setValue(0)
                self.rotation_widgets[ax]['slider'].setValue(0)
                self.rotation_widgets[ax]['spin'].blockSignals(False)
                self.rotation_widgets[ax]['slider'].blockSignals(False)

        self.molecule_transform.Identity()
        self.molecule_transform.SetMatrix(M)

        self.update_actor_transform()
        if hasattr(self, 'set_standard_view'):
            self.set_standard_view('yz')
        if hasattr(self, 'trigger_interactive_simulation'):
            self.trigger_interactive_simulation()

    def handle_find_initial_plane(self):
        """
        XY平面への"寝かせ"を自動化：
        - PDB: 支持面の接触パッチを広く・安定に当てる法線探索
        - MRC: 既存の厚み最小化ロジックを使用
        """
        # PDBデータまたはMRCデータのどちらかが読み込まれているかチェック
        if getattr(self, 'atoms_data', None) is None and not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            QMessageBox.warning(self, "Warning", "PDBまたはMRCファイルが読み込まれていません。")
            return
        
        # --- PDB: contact-patch based search ---
        if getattr(self, 'atoms_data', None) is not None:
            # parameters (Å) + electrostatics
            if hasattr(self, 'find_plane_params'):
                N_dir = int(self.find_plane_params["N_dir"].value())
                K_local = int(self.find_plane_params["K"].value())
                K_geom = int(self.find_plane_params["K_geom"].value())
                delta_contact = float(self.find_plane_params["delta"].value())
                delta_elec_A = float(self.find_plane_params["delta_elec"].value())
                lambda_var = float(self.find_plane_params["lambda"].value())
                use_elec = bool(self.find_plane_params["use_elec"].isChecked())
                alpha_elec = float(self.find_plane_params["alpha_elec"].value()) if use_elec else 0.0
                r_pseudo = float(self.find_plane_params["r_pseudo"].value())
                pH = float(self.find_plane_params["pH"].value())
                substrate_type = str(self.find_plane_params["substrate"].currentText())
                salt_type = str(self.find_plane_params["salt"].currentText())
                salt_mM = float(self.find_plane_params["salt_mM"].value())
                step_deg_list = [
                    float(self.find_plane_params["theta1"].value()),
                    float(self.find_plane_params["theta2"].value()),
                    float(self.find_plane_params["theta3"].value()),
                ]
                local_grid = int(self.find_plane_params["grid"].value())
                neigh_r = float(self.find_plane_params["surf_r"].value())
                neigh_n = int(self.find_plane_params["surf_n"].value())
            else:
                N_dir = 10000
                K_local = 50
                K_geom = 10
                delta_contact = 1.5
                delta_elec_A = 8.0
                lambda_var = 0.02
                use_elec = True
                alpha_elec = 0.3
                r_pseudo = 2.0
                pH = 7.0
                substrate_type = "MICA"
                salt_type = "NaCl"
                salt_mM = 50.0
                step_deg_list = [3.0, 1.0, 0.3]
                local_grid = 2
                neigh_r = 6.0
                neigh_n = 25

            N_dir = max(1, N_dir)
            K_local = max(1, min(K_local, N_dir))
            K_geom = max(1, K_geom)
            local_grid = max(1, min(local_grid, 6))
            step_deg_list = [s for s in step_deg_list if s > 0.0]
            if not step_deg_list:
                step_deg_list = [3.0, 1.0, 0.3]

            # atom-based geometry (always)
            coords_nm = np.column_stack([
                self.atoms_data['x'],
                self.atoms_data['y'],
                self.atoms_data['z'],
            ]).astype(float)
            coords_atom_ang = coords_nm * 10.0
            elements = self.atoms_data.get('element', None)
            if elements is None or len(elements) != coords_nm.shape[0]:
                elements = np.array(["C"] * coords_nm.shape[0])
            radii_atom_ang = np.array([self._vdw_radius_angstrom(e) for e in elements], dtype=float)

            surface_mask = self._select_surface_atoms(coords_atom_ang, neighbor_radius=neigh_r, neighbor_threshold=neigh_n)
            if np.count_nonzero(surface_mask) >= 30:
                coordsA = coords_atom_ang[surface_mask]
                radiiA = radii_atom_ang[surface_mask]
                index_map = np.where(surface_mask)[0]
            else:
                coordsA = coords_atom_ang
                radiiA = radii_atom_ang
                index_map = np.arange(coords_atom_ang.shape[0])

            # normalization scale for Var (geometry)
            self._geom_var_scale = self._compute_geom_var_scale(coordsA)

            # residue-based charges for electrostatics
            coordsR = None
            qR = None
            lambdaD_A = None
            s_sub = 0.0
            A_sub = 0.0
            if use_elec:
                pseudo = self._build_pseudo_atoms_from_residues(pH, r_pseudo=r_pseudo)
                if pseudo is None:
                    QMessageBox.warning(self, "Warning", "残基の擬似原子を作成できませんでした。")
                    return
                coordsR = pseudo[0]
                qR = pseudo[2]

                # Debye length (Å)
                c_M = max(float(salt_mM) / 1000.0, 1e-6)
                ionic_strength = max(c_M, 1e-6)
                lambdaD_nm = 0.304 / np.sqrt(ionic_strength)
                lambdaD_A = 10.0 * lambdaD_nm

                # substrate parameters
                pKa_aptes = 9.5
                if substrate_type.upper().startswith("MICA"):
                    s_sub = -1.0
                    A_sub = 1.0
                else:
                    s_sub = 1.0
                    A_sub = 1.0 / (1.0 + (10.0 ** (pH - pKa_aptes)))

            directions = self._fibonacci_sphere_directions(N_dir)
            heap = []
            import heapq
            heap_counter = 0

            def _geo_key(res):
                return (float(res["score_geo"]), -float(res.get("height_norm", np.inf)))

            def _geo_better(a, b):
                if b is None:
                    return True
                return _geo_key(a) > _geo_key(b)

            best_geo = None
            for n in directions:
                res_geo = self._eval_geo(coordsA, radiiA, n, delta_contact, lambda_var)
                if res_geo is None:
                    continue
                key_geo = _geo_key(res_geo)
                if _geo_better(res_geo, best_geo):
                    best_geo = res_geo
                if len(heap) < K_local:
                    heapq.heappush(heap, (key_geo, heap_counter, res_geo))
                    heap_counter += 1
                else:
                    if key_geo > heap[0][0]:
                        heapq.heapreplace(heap, (key_geo, heap_counter, res_geo))
                        heap_counter += 1

            # Local refinement (geo only)
            if not heap:
                QMessageBox.warning(self, "Warning", "接触パッチを評価できませんでした。")
                return

            def refine_direction_geo(n0):
                best = self._eval_geo(coordsA, radiiA, n0, delta_contact, lambda_var)
                if best is None:
                    return None
                n = best["n"]
                for step_deg in step_deg_list:
                    step = np.deg2rad(step_deg)
                    e1 = self._make_perp_unit(n)
                    e2 = np.cross(n, e1)
                    grid = range(-local_grid, local_grid + 1)
                    for a in grid:
                        for b in grid:
                            if a == 0 and b == 0:
                                continue
                            n2 = n + (step * (a * e1 + b * e2))
                            n2 = n2 / max(1e-12, float(np.linalg.norm(n2)))
                            res2 = self._eval_geo(coordsA, radiiA, n2, delta_contact, lambda_var)
                            if res2 is None:
                                continue
                            if _geo_better(res2, best):
                                best = res2
                                n = best["n"]
                return best

            refined_results = []
            for _, _, res in heap:
                refined = refine_direction_geo(res["n"])
                if refined is not None:
                    refined_results.append(refined)
            if not refined_results and best_geo is not None:
                refined_results = [best_geo]

            if not refined_results:
                QMessageBox.warning(self, "Warning", "最適法線の探索に失敗しました。")
                return

            refined_results.sort(key=lambda r: (float(r["score_geo"]), -float(r.get("height_norm", np.inf))), reverse=True)
            K_geom = min(K_geom, len(refined_results))
            candidates = refined_results[:K_geom]
            best_geo_candidate = candidates[0]

            # Stage 2: electrostatics re-ranking (optional)
            best_final = best_geo_candidate
            best_final_score = best_geo_candidate["score_geo"]
            best_S_contact = max(c["S_contact"] for c in candidates) if candidates else 0.0
            for c in candidates:
                c["Qshell"] = 0.0
                c["S_elec"] = 0.0
                c["Nr"] = 0
                c["score_final"] = c["score_geo"]
                if use_elec:
                    elec = self._eval_elec(coordsR, qR, c["n"], c["d"], delta_elec_A, lambdaD_A, A_sub, s_sub)
                    if best_S_contact > 0.0 and c["S_contact"] < 0.8 * best_S_contact:
                        elec = {"Qshell": 0.0, "S_elec": 0.0, "Nr": 0}
                    c["Qshell"] = elec["Qshell"]
                    c["S_elec"] = elec["S_elec"]
                    c["Nr"] = elec["Nr"]
                    c["score_final"] = c["score_geo"] + alpha_elec * c["S_elec"]
                if (
                    c["score_final"] > best_final_score
                    or (
                        abs(c["score_final"] - best_final_score) <= 1e-12
                        and float(c.get("height_norm", np.inf)) < float(best_final.get("height_norm", np.inf))
                    )
                ):
                    best_final_score = c["score_final"]
                    best_final = c

            # store result for reference (final)
            patch_indices = index_map[np.where(best_final["mask"])[0]] if best_final.get("mask") is not None else np.array([], dtype=int)
            self.last_initial_plane_result = {
                "n_best": best_final["n"],
                "d_best": best_final["d"],
                "score_geo": best_final["score_geo"],
                "score_final": best_final["score_final"],
                "S_contact": best_final["S_contact"],
                "var": best_final["var"],
                "height_A": best_final.get("height_A", 0.0),
                "height_norm": best_final.get("height_norm", 0.0),
                "Nc": best_final.get("Nc", 0),
                "S_elec": best_final.get("S_elec", 0.0),
                "Qshell": best_final.get("Qshell", 0.0),
                "Nr": best_final.get("Nr", 0),
                "patch_indices": patch_indices,
            }

            def _qsign(val):
                return "positive" if val > 0 else ("negative" if val < 0 else "neutral")

            print(
                "[FindInitialPlane][best-geo]",
                "Score_geo:", f"{best_geo_candidate['score_geo']:.4f}",
                "S_contact:", f"{best_geo_candidate['S_contact']:.4f}",
                "Var:", f"{best_geo_candidate['var']:.4f}",
                "Var_norm:", f"{best_geo_candidate.get('var_norm', 0.0):.4f}",
                "Height[A]:", f"{best_geo_candidate.get('height_A', 0.0):.4f}",
                "Height_norm:", f"{best_geo_candidate.get('height_norm', 0.0):.4f}",
                "Nc:", best_geo_candidate.get("Nc", 0),
            )
            print(
                "[FindInitialPlane][best-final]",
                "Score_final:", f"{best_final['score_final']:.4f}",
                "Score_geo:", f"{best_final['score_geo']:.4f}",
                "S_contact:", f"{best_final['S_contact']:.4f}",
                "Var:", f"{best_final['var']:.4f}",
                "Var_norm:", f"{best_final.get('var_norm', 0.0):.4f}",
                "Height[A]:", f"{best_final.get('height_A', 0.0):.4f}",
                "Height_norm:", f"{best_final.get('height_norm', 0.0):.4f}",
                "Nc:", best_final.get("Nc", 0),
                "Qshell:", f"{best_final.get('Qshell', 0.0):.4f} ({_qsign(best_final.get('Qshell', 0.0))})",
                "S_elec:", f"{best_final.get('S_elec', 0.0):.4f}",
                "Nr:", best_final.get("Nr", 0),
                "salt:", f"{salt_type} {salt_mM} mM" if use_elec else "(off)",
                "pH:", f"{pH:.2f}" if use_elec else "(off)",
                "substrate:", substrate_type if use_elec else "(off)",
            )

            # rotate so that n_best aligns with -Z (support side is -Z)
            n_best = best_final["n"]
            target = np.array([0.0, 0.0, -1.0], dtype=float)
            v = np.cross(n_best, target)
            c = float(np.dot(n_best, target))
            s = float(np.linalg.norm(v))
            if s < 1e-8:
                if c > 0:
                    R = np.eye(3)
                else:
                    axis = self._make_perp_unit(n_best)
                    ax = axis / max(1e-12, float(np.linalg.norm(axis)))
                    x, y, z = ax
                    R = np.array([
                        [-1 + 2*x*x, 2*x*y, 2*x*z],
                        [2*x*y, -1 + 2*y*y, 2*y*z],
                        [2*x*z, 2*y*z, -1 + 2*z*z],
                    ], dtype=float)
            else:
                vx = np.array([
                    [0.0, -v[2], v[1]],
                    [v[2], 0.0, -v[0]],
                    [-v[1], v[0], 0.0],
                ], dtype=float)
                R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))

            center_nm = (coords_atom_ang.mean(axis=0) / 10.0) if coords_atom_ang is not None else np.array([0.0, 0.0, 0.0])
            self._apply_initial_plane_rotation(R, center_nm)
            return

        # --- MRC: fallback to thickness minimization ---
        coords = self._get_mrc_surface_coords()
        if coords is None:
            QMessageBox.warning(self, "Warning", "MRCデータから座標を取得できませんでした。")
            return

        # ---- 元座標と重心 ----
        c = coords.mean(axis=0)
        X = coords - c  # 重心回り

        # ---- PCAで基準姿勢（PC3→Z, PC1→X）----
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        Rr0 = VT.T                                  # 列: PC1,PC2,PC3
        if np.linalg.det(Rr0) < 0:                  # 右手系を担保
            Rr0[:, 0] *= -1
        if Rr0[2, 2] < 0:                           # Z(PC3)は+Zを向くよう反転
            Rr0[:, 2] *= -1
            Rr0[:, 1] *= -1  # 右手系維持

        # ---- 評価関数（厚み＋接触数）----
        eps_nm = 0.20        # 接触しきい値（必要に応じて 0.2–0.5nm）
        thick_tie_tol = 1e-4 # 厚みの同点判定 [nm]（=0.0001nm ≒ 0.001Å）

        def Rx(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        def Ry(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])

        def evaluate(Rr):
            """(厚みh, 接触原子数cnt) を返す。厚みは最小化、cntは最大化。"""
            try:
                # 入力データの妥当性チェック
                if X is None or len(X) == 0:
                    return float('inf'), 0
                
                if Rr is None or len(Rr) == 0:
                    return float('inf'), 0
                
                # 数値の安全性を確保（より厳格な範囲制限）
                X_safe = np.clip(X, -1000, 1000)  # より狭い範囲
                Rr_safe = np.clip(Rr, -100, 100)  # 回転行列は小さい値
                
                # 行列の形状をチェック
                if X_safe.shape[1] != Rr_safe.shape[0]:
                    return float('inf'), 0
                
                # ゼロ除算を防ぐためのチェック
                if np.any(np.abs(Rr_safe) < 1e-10):
                    return float('inf'), 0
                
                # 行列積を安全に実行
                try:
                    with np.errstate(all='ignore'):  # 警告を無視
                        z = (X_safe @ Rr_safe)[:, 2]
                except (OverflowError, RuntimeWarning, ValueError):
                    return float('inf'), 0
                
                # NaNやInfをチェック
                if not np.all(np.isfinite(z)):
                    return float('inf'), 0
                
                # 結果の妥当性チェック
                zmin = z.min()
                zmax = z.max()
                h = zmax - zmin
                
                # 厚みが異常に大きい場合は無効
                if h > 1000 or not np.isfinite(h) or h < 0:
                    return float('inf'), 0
                
                # 接触原子数の計算
                try:
                    cnt = int(np.count_nonzero(z - zmin <= eps_nm))
                    if cnt < 0 or cnt > len(z):
                        return float('inf'), 0
                except (OverflowError, ValueError):
                    return float('inf'), 0
                
                return h, cnt
                
            except Exception as e:
                # 全ての例外をキャッチ
                return float('inf'), 0

        # 初期値
        best_Rr = Rr0
        best_h, best_cnt = evaluate(best_Rr)

        # ---- 近傍粗探索（±8°）----
        grid = np.deg2rad(np.array([-8,-6,-4,-2,0,2,4,6,8], dtype=float))
        for ax in grid:          # X tilt
            for ay in grid:      # Y tilt
                Rr = Rr0 @ (Ry(ay) @ Rx(ax))
                h, cnt = evaluate(Rr)
                if (h < best_h - thick_tie_tol) or (abs(h - best_h) <= thick_tie_tol and cnt > best_cnt):
                    best_Rr, best_h, best_cnt = Rr, h, cnt

        # ---- （任意）微細探索：±2°でもう一段詰める ----
        fine = np.deg2rad(np.array([-2,-1,0,1,2], dtype=float))
        base = best_Rr
        for ax in fine:
            for ay in fine:
                Rr = base @ (Ry(ay) @ Rx(ax))
                h, cnt = evaluate(Rr)
                if (h < best_h - thick_tie_tol) or (abs(h - best_h) <= thick_tie_tol and cnt > best_cnt):
                    best_Rr, best_h, best_cnt = Rr, h, cnt

        # ---- VTK（列ベクトル系）へ適用： p' = R p + t,  R = best_Rr.T,  t = c - R c ----
        R = best_Rr.T
        t = c - R @ c

        # ---- 回転行列からEuler角を抽出してスライダーに反映 ----
        def matrix_to_euler_zyx(R):
            """回転行列からEuler角（ZYX順）を抽出"""
            sy = np.hypot(R[0,0], R[1,0])
            singular = sy < 1e-8
            if not singular:
                z = np.degrees(np.arctan2(R[1,0], R[0,0]))         # yaw
                y = np.degrees(np.arctan2(-R[2,0], sy))            # pitch
                x = np.degrees(np.arctan2(R[2,1], R[2,2]))         # roll
            else:
                # gimbal lock: z は意味を持ちにくいので0に、xで帳尻
                z = 0.0
                y = np.degrees(np.arctan2(-R[2,0], sy))
                x = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            # -180〜180に正規化
            def _wrap(a): 
                return (a + 180) % 360 - 180
            return _wrap(x), _wrap(y), _wrap(z)

        # 回転行列からEuler角を取得
        rot_x, rot_y, rot_z = matrix_to_euler_zyx(R)

        # PyMOLバックエンドの場合は、回転ウィジェットに反映して適用する
        if self._is_pymol_active():
            if hasattr(self, 'rotation_widgets'):
                for ax, val in zip(('X', 'Y', 'Z'), (rot_x, rot_y, rot_z)):
                    self.rotation_widgets[ax]['spin'].blockSignals(True)
                    self.rotation_widgets[ax]['slider'].blockSignals(True)
                    self.rotation_widgets[ax]['spin'].setValue(val)
                    self.rotation_widgets[ax]['slider'].setValue(int(val * 10))
                    self.rotation_widgets[ax]['spin'].blockSignals(False)
                    self.rotation_widgets[ax]['slider'].blockSignals(False)
            self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self.apply_structure_rotation()
            if hasattr(self, 'trigger_interactive_simulation'):
                self.trigger_interactive_simulation()
            return

        # Find Initial Plane 内：回転行列 R と平行移動 t を作った後
        M = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                M.SetElement(i, j, float(R[i, j]))
        M.SetElement(0, 3, float(t[0]))
        M.SetElement(1, 3, float(t[1]))
        M.SetElement(2, 3, float(t[2]))
        M.SetElement(3, 0, 0.0); M.SetElement(3, 1, 0.0); M.SetElement(3, 2, 0.0); M.SetElement(3, 3, 1.0)

        # 計算した回転行列M（ワールド基準）は base_transform にだけ入れる
        self.base_transform.Identity()
        self.base_transform.SetMatrix(M)

        # ローカル操作は一旦ゼロから（＝整列後もローカル軸で自由に回せる）
        self.local_transform.Identity()
        self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}  # スライダ絶対値→差分適用用

        # スライダUIも 0° にリセット（任意）
        if hasattr(self, 'rotation_widgets'):
            for ax in ('X', 'Y', 'Z'):
                self.rotation_widgets[ax]['spin'].blockSignals(True)
                self.rotation_widgets[ax]['slider'].blockSignals(True)
                self.rotation_widgets[ax]['spin'].setValue(0)
                self.rotation_widgets[ax]['slider'].setValue(0)
                self.rotation_widgets[ax]['spin'].blockSignals(False)
                self.rotation_widgets[ax]['slider'].blockSignals(False)

        # 後方互換性のため molecule_transform も更新
        self.molecule_transform.Identity()
        self.molecule_transform.SetMatrix(M)

        self.update_actor_transform()
        
        if hasattr(self, 'set_standard_view'):
            self.set_standard_view('yz')
        if hasattr(self, 'trigger_interactive_simulation'):
            self.trigger_interactive_simulation()

    def on_mouse_press(self, event):
        """直接的なマウスプレスイベントハンドラー"""
        
        if event.button() == Qt.LeftButton:
            # キーの状態をチェック
            modifiers = event.modifiers()
            ctrl_pressed = bool(modifiers & Qt.ControlModifier)
            shift_pressed = bool(modifiers & Qt.ShiftModifier)
            
            if ctrl_pressed and not shift_pressed:
                self.actor_rotating = True
                self.drag_start_pos = event.pos()
                event.accept()
                return
            elif shift_pressed and not ctrl_pressed:
                self.panning = True
                self.pan_start_pos = event.pos()
                event.accept()
                return
        
        # 通常のマウスイベントをVTKウィジェットの元のハンドラーに渡す
        if hasattr(self, 'original_mouse_press'):
            self.original_mouse_press(event)

    def on_mouse_move(self, event):
        """直接的なマウスムーブイベントハンドラー"""
        if self.actor_rotating:
            if hasattr(self, 'drag_start_pos'):
                dx = event.pos().x() - self.drag_start_pos.x()
                dy = event.pos().y() - self.drag_start_pos.y()
                
                # 視点に応じた回転軸マッピング
                self.update_rotation_from_drag_view_dependent(dx, dy)
                
                self.drag_start_pos = event.pos()
            event.accept()
            return
        elif self.panning:
            # パニング処理は後で実装
            event.accept()
            return
        
        # 通常のマウスイベントをVTKウィジェットの元のハンドラーに渡す
        if hasattr(self, 'original_mouse_move'):
            self.original_mouse_move(event)

    def on_mouse_release(self, event):
        """直接的なマウスリリースイベントハンドラー"""
        
        if event.button() == Qt.LeftButton:
            if self.actor_rotating:
                self.actor_rotating = False
                
                # ★★★ 追加：ドラッグ終了時の高解像度シミュレーション ★★★
                if self.interactive_update_check.isChecked():
                    self.schedule_high_res_simulation()
                
                event.accept()
                return
            elif self.panning:
                self.panning = False
                event.accept()
                return
        
        # 通常のマウスイベントをVTKウィジェットの元のハンドラーに渡す
        if hasattr(self, 'original_mouse_release'):
            self.original_mouse_release(event)

    def reset_structure_rotation(self):
        """分子の回転をリセット（PDB/MRC読み込み時の状態に戻す）"""
        if not hasattr(self, 'rotation_widgets'):
            return
        
        # 回転ウィジェットを0にリセット
        for axis in ['X', 'Y', 'Z']:
            self.rotation_widgets[axis]['spin'].blockSignals(True)
            self.rotation_widgets[axis]['slider'].blockSignals(True)
            self.rotation_widgets[axis]['spin'].setValue(0.0)
            self.rotation_widgets[axis]['slider'].setValue(0)
            self.rotation_widgets[axis]['spin'].blockSignals(False)
            self.rotation_widgets[axis]['slider'].blockSignals(False)
        
        # prev_rotをリセット
        self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        if self._is_pymol_only():
            try:
                self.pymol_cmd.reset()
            except Exception:
                pass
            self.request_render()
            return

        # 回転変換をリセット（VTK）
        self.base_transform.Identity()
        self.local_transform.Identity()
        self.combined_transform.Identity()
        self.molecule_transform.Identity()
        
        # アクターの変換を更新
        self.update_actor_transform()
        if self._is_dual_mode() and self._is_pymol_active():
            self._sync_pymol_view_from_vtk()

    def get_current_view_orientation(self):
        """現在のカメラの向きから視点方向を判定"""
        if not hasattr(self, 'renderer') or not self.renderer:
            return 'free'
        
        camera = self.renderer.GetActiveCamera()
        pos = camera.GetPosition()
        focal = camera.GetFocalPoint()
        
        # カメラから焦点への方向ベクトル
        view_dir = [focal[i] - pos[i] for i in range(3)]
        # 正規化
        length = (sum(d*d for d in view_dir)) ** 0.5
        if length < 1e-10:
            return 'free'
        view_dir = [d/length for d in view_dir]
        
        # 各軸方向との内積で判定（閾値0.8）
        if abs(view_dir[2]) > 0.8:  # Z方向
            return 'xy'  # XY面を見ている
        elif abs(view_dir[0]) > 0.8:  # X方向
            return 'yz'  # YZ面を見ている
        elif abs(view_dir[1]) > 0.8:  # Y方向
            return 'zx'  # ZX面を見ている
        else:
            return 'free'  # 斜め視点

    def update_rotation_from_drag_view_dependent(self, dx, dy):
        """カメラのスクリーン座標に沿って構造を回転（常にドラッグ方向と見た目が一致するようにする）"""
        if not hasattr(self, 'rotation_widgets'):
            return
        # VTKが有効でない場合は従来のオイラー加算にフォールバック
        if not hasattr(self, 'renderer') or self.renderer is None or self._is_pymol_only():
            sensitivity = 0.5
            self.update_rotation_from_drag(
                angle_x_delta=dy * sensitivity,
                angle_y_delta=dx * sensitivity,
                angle_z_delta=0,
            )
            return

        camera = self.renderer.GetActiveCamera()
        if camera is None:
            return

        # --- カメラのスクリーン座標系（right, up）を構成 ---
        cam_pos = np.array(camera.GetPosition(), dtype=float)
        focal = np.array(camera.GetFocalPoint(), dtype=float)
        view_up = np.array(camera.GetViewUp(), dtype=float)
        view_dir = focal - cam_pos
        n = float(np.linalg.norm(view_dir))
        if n < 1e-9:
            return
        view_dir = view_dir / n
        up_dir = view_up / max(1e-9, float(np.linalg.norm(view_up)))
        right_dir = np.cross(view_dir, up_dir)
        rn = float(np.linalg.norm(right_dir))
        if rn < 1e-9:
            return
        right_dir = right_dir / rn
        true_up = np.cross(right_dir, view_dir)
        un = float(np.linalg.norm(true_up))
        if un < 1e-9:
            return
        true_up = true_up / un

        # --- 現在のlocal回転（UI値）を行列へ ---
        rx = float(self.rotation_widgets['X']['spin'].value())
        ry = float(self.rotation_widgets['Y']['spin'].value())
        rz = float(self.rotation_widgets['Z']['spin'].value())

        def _deg2rad(a):
            return a * (np.pi / 180.0)

        def _rot_x(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[1, 0, 0],
                             [0, ca, -sa],
                             [0, sa, ca]], dtype=float)

        def _rot_y(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[ca, 0, sa],
                             [0, 1, 0],
                             [-sa, 0, ca]], dtype=float)

        def _rot_z(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[ca, -sa, 0],
                             [sa, ca, 0],
                             [0, 0, 1]], dtype=float)

        R_current = _rot_x(_deg2rad(rx)) @ _rot_y(_deg2rad(ry)) @ _rot_z(_deg2rad(rz))

        def _axis_angle(axis, angle_deg):
            a = _deg2rad(angle_deg)
            ax = np.array(axis, dtype=float)
            axn = float(np.linalg.norm(ax))
            if axn < 1e-9:
                return np.eye(3)
            ax = ax / axn
            x, y, z = ax
            c = float(np.cos(a))
            s = float(np.sin(a))
            C = 1.0 - c
            return np.array([
                [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
            ], dtype=float)

        sensitivity = 0.5
        angle_up = dx * sensitivity
        angle_right = -dy * sensitivity

        R_h = _axis_angle(true_up, angle_up)
        R_v = _axis_angle(right_dir, angle_right)
        R_new = R_v @ (R_h @ R_current)

        # --- R_new を Rx*Ry*Rz に戻してUIへ（-180..180） ---
        def _rad2deg(a):
            return a * (180.0 / np.pi)

        sy = float(np.clip(R_new[0, 2], -1.0, 1.0))
        new_ry = float(np.arcsin(sy))
        cy = float(np.cos(new_ry))
        if abs(cy) < 1e-6:
            # gimbal: 近似的にZ=0としてXを復元
            new_rz = 0.0
            new_rx = float(np.arctan2(R_new[2, 1], R_new[1, 1]))
        else:
            new_rz = float(np.arctan2(-R_new[0, 1], R_new[0, 0]))
            new_rx = float(np.arctan2(-R_new[1, 2], R_new[2, 2]))

        new_rx = (_rad2deg(new_rx) + 180.0) % 360.0 - 180.0
        new_ry = (_rad2deg(new_ry) + 180.0) % 360.0 - 180.0
        new_rz = (_rad2deg(new_rz) + 180.0) % 360.0 - 180.0

        for axis in ['X', 'Y', 'Z']:
            self.rotation_widgets[axis]['spin'].blockSignals(True)
            self.rotation_widgets[axis]['slider'].blockSignals(True)
        self.rotation_widgets['X']['spin'].setValue(new_rx)
        self.rotation_widgets['Y']['spin'].setValue(new_ry)
        self.rotation_widgets['Z']['spin'].setValue(new_rz)
        self.rotation_widgets['X']['slider'].setValue(int(new_rx * 10))
        self.rotation_widgets['Y']['slider'].setValue(int(new_ry * 10))
        self.rotation_widgets['Z']['slider'].setValue(int(new_rz * 10))
        for axis in ['X', 'Y', 'Z']:
            self.rotation_widgets[axis]['spin'].blockSignals(False)
            self.rotation_widgets[axis]['slider'].blockSignals(False)

        self.apply_structure_rotation()

        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            if hasattr(self, 'actor_rotating') and self.actor_rotating:
                self.run_simulation_immediate_controlled()

    def update_rotation_from_drag(self, angle_x_delta=0, angle_y_delta=0, angle_z_delta=0):
        """マウスドラッグに応じてPDB/MRC構造の回転角度を更新する（改良版）"""
        if not hasattr(self, 'rotation_widgets'):
            return

        current_rot_x = self.rotation_widgets['X']['spin'].value()
        current_rot_y = self.rotation_widgets['Y']['spin'].value()
        current_rot_z = self.rotation_widgets['Z']['spin'].value()

        # ドラッグによる移動量を加算
        raw_x = current_rot_x + angle_x_delta
        raw_y = current_rot_y + angle_y_delta
        raw_z = current_rot_z + angle_z_delta
        
        # 角度を-180から+180の範囲に正規化する
        new_rot_x = (raw_x + 180) % 360 - 180
        new_rot_y = (raw_y + 180) % 360 - 180
        new_rot_z = (raw_z + 180) % 360 - 180

        # スライダーの値変更を一時的に無効化してから設定
        for axis in ['X', 'Y', 'Z']:
            self.rotation_widgets[axis]['spin'].blockSignals(True)
            self.rotation_widgets[axis]['slider'].blockSignals(True)
        
        self.rotation_widgets['X']['spin'].setValue(new_rot_x)
        self.rotation_widgets['Y']['spin'].setValue(new_rot_y)
        self.rotation_widgets['Z']['spin'].setValue(new_rot_z)
        
        # スライダーも同期
        self.rotation_widgets['X']['slider'].setValue(int(new_rot_x * 10))
        self.rotation_widgets['Y']['slider'].setValue(int(new_rot_y * 10))
        self.rotation_widgets['Z']['slider'].setValue(int(new_rot_z * 10))
        
        # シグナルを再有効化
        for axis in ['X', 'Y', 'Z']:
            self.rotation_widgets[axis]['spin'].blockSignals(False)
            self.rotation_widgets[axis]['slider'].blockSignals(False)
        
        # スライダー値を変更した後、回転を適用
        self.apply_structure_rotation()
        
        # ★★★ 修正: ドラッグ中の制御されたリアルタイム更新 ★★★
        if self.interactive_update_check.isChecked():
            # ドラッグ中は制御付きで更新（頻度制限あり）
            if hasattr(self, 'actor_rotating') and self.actor_rotating:
                self.run_simulation_immediate_controlled()

    def update_afm_display(self):
        """AFM画像表示の更新（チェックボックスに基づく）"""
        # 現在チェックされている数を確認
        checked_count = sum([
            self.afm_x_check.isChecked(),
            self.afm_y_check.isChecked(),
            self.afm_z_check.isChecked()
        ])
        
        # 最低1つはチェックされている必要がある
        if checked_count == 0:
            # どのチェックボックスが最後に変更されたかを確認して元に戻す
            sender = self.sender()
            if sender:
                sender.blockSignals(True)  # 再帰呼び出しを防ぐ
                sender.setChecked(True)
                sender.blockSignals(False)
                
            QMessageBox.warning(self, "Warning", 
                            "At least one AFM view must be selected!")
            return
        
        # 各パネルの表示/非表示を設定
        self.afm_x_frame.setVisible(self.afm_x_check.isChecked())
        self.afm_y_frame.setVisible(self.afm_y_check.isChecked())
        self.afm_z_frame.setVisible(self.afm_z_check.isChecked())
        if hasattr(self, 'afm_aligned_frame'):
            has_aligned = getattr(self, 'sim_aligned_nm', None) is not None
            self.afm_aligned_frame.setVisible(bool(has_aligned))
        if hasattr(self, 'afm_real_frame'):
            has_real = getattr(self, 'real_afm_nm', None) is not None
            self.afm_real_frame.setVisible(bool(has_real))
        
        # デバッグ情報
        visible_views = []
        if self.afm_x_check.isChecked():
            visible_views.append("X")
        if self.afm_y_check.isChecked():
            visible_views.append("Y")
        if self.afm_z_check.isChecked():
            visible_views.append("Z")
        
        #print(f"AFM views visible: {', '.join(visible_views)}")

    def create_afm_image_panel(self, title):
        """個別のAFM像表示パネル作成（表示制御対応）"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        frame.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        
        # タイトルラベル
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 9px;
                color: #555;
                padding: 2px;
                background-color: #e8e8e8;
                border-radius: 2px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMaximumHeight(18)
        layout.addWidget(title_label)
        
        # 画像コンテナ
        image_container = QWidget()
        image_container.setObjectName("afm_image_container")
        image_layout = QStackedLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        # プレースホルダー
        placeholder_text = "AFM Image\n(Not Simulated)"
        if title.startswith("Real AFM"):
            placeholder_text = "Real AFM (ASD)\nDrop .asd here"
        if title.startswith("Sim Aligned"):
            placeholder_text = "Sim Aligned\n(Estimate Pose)"

        placeholder = QLabel(placeholder_text)
        placeholder.setObjectName("afm_placeholder")
        placeholder.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 8px;
                background-color: white;
                border: 1px dashed #ccc;
                border-radius: 2px;
            }
        """)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setMinimumHeight(80)
        image_layout.addWidget(placeholder)

        view = _AspectPixmapView(image_container)
        view.setObjectName("afm_image_view")
        image_layout.addWidget(view)
        image_layout.setCurrentWidget(placeholder)

        layout.addWidget(image_container)
        
        return frame
    
    def reset_camera(self):
        """カメラのリセット（デフォルトでYZ平面視点）"""
        if self._is_pymol_active():
            try:
                self.pymol_cmd.reset()
            except Exception:
                pass
            self.request_render()
            if self._is_pymol_only():
                return
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        
        # デフォルトでYZ平面視点に設定
        camera.SetViewUp(0, 0, 1)  # Z軸が上方向
        camera.SetPosition(15, 0, 0)  # X軸の正方向から見る
        camera.SetFocalPoint(0, 0, 0)  # 原点を焦点に
        
        self.renderer.ResetCameraClippingRange()
        self.request_render()
        if self._is_dual_mode():
            self._sync_pymol_view_from_vtk()

    def fit_view_to_contents(self):
        """現在の構造が画面内に収まるように自動調整（PDB/CIFロード直後用）"""
        # PyMOL: 画面内に収まるように zoom/orient
        if self._is_pymol_active() and not self._is_dual_mode():
            try:
                self.pymol_cmd.zoom(self.pymol_object_name, complete=1)
            except Exception:
                pass

        # VTK: バウンディングボックスに合わせてカメラをフィット
        if not self._is_pymol_only():
            self._ensure_vtk_initialized()
            if hasattr(self, 'renderer') and self.renderer is not None:
                try:
                    bounds = self._get_structure_bounds_for_camera_fit()
                    if bounds is not None:
                        # ResetCamera(bounds) ensures we don't accidentally include AFM tip bounds.
                        self.renderer.ResetCamera(bounds)
                    else:
                        self.renderer.ResetCamera()
                    self._vtk_focus_on_structure_center(preserve_distance=True)
                    self.renderer.ResetCameraClippingRange()
                except Exception:
                    pass

        self.request_render()
        if self._is_dual_mode():
            self._sync_pymol_view_from_vtk()

    def _get_structure_bounds_for_camera_fit(self):
        """Return bounds (xmin,xmax,ymin,ymax,zmin,zmax) for current structure, excluding the AFM tip."""
        try:
            bbox = vtk.vtkBoundingBox()
            # Prefer actor visibility when available
            if getattr(self, "sample_actor", None) is not None:
                try:
                    if self.sample_actor.GetVisibility():
                        bbox.AddBounds(self.sample_actor.GetBounds())
                except Exception:
                    bbox.AddBounds(self.sample_actor.GetBounds())
            if getattr(self, "bonds_actor", None) is not None:
                try:
                    if self.bonds_actor.GetVisibility():
                        bbox.AddBounds(self.bonds_actor.GetBounds())
                except Exception:
                    bbox.AddBounds(self.bonds_actor.GetBounds())
            if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
                try:
                    if self.mrc_actor.GetVisibility():
                        bbox.AddBounds(self.mrc_actor.GetBounds())
                except Exception:
                    bbox.AddBounds(self.mrc_actor.GetBounds())

            if not bbox.IsValid():
                return None
            b = [0.0] * 6
            bbox.GetBounds(b)
            return tuple(float(x) for x in b)
        except Exception:
            return None

    def _vtk_focus_on_structure_center(self, preserve_distance=True):
        """Set VTK camera focal point to the structure center (excluding tip)."""
        if not hasattr(self, 'renderer') or self.renderer is None:
            return
        bounds = self._get_structure_bounds_for_camera_fit()
        if bounds is None:
            return
        center = np.array([(bounds[0] + bounds[1]) * 0.5,
                           (bounds[2] + bounds[3]) * 0.5,
                           (bounds[4] + bounds[5]) * 0.5], dtype=float)
        try:
            cam = self.renderer.GetActiveCamera()
            if cam is None:
                return
            if preserve_distance:
                pos = np.array(cam.GetPosition(), dtype=float)
                fp = np.array(cam.GetFocalPoint(), dtype=float)
                delta = pos - fp
                if float(np.linalg.norm(delta)) < 1e-9:
                    delta = np.array([1.0, 0.0, 0.0], dtype=float)
                cam.SetFocalPoint(center[0], center[1], center[2])
                cam.SetPosition(center[0] + delta[0], center[1] + delta[1], center[2] + delta[2])
            else:
                cam.SetFocalPoint(center[0], center[1], center[2])
        except Exception:
            pass
    
    def setup_lighting(self):
        """ライティング設定"""
        # メインライト
        light1 = vtk.vtkLight()
        light1.SetPosition(10, 10, 10)
        light1.SetIntensity(0.8)
        light1.SetColor(1.0, 1.0, 1.0)
        self.renderer.AddLight(light1)
        
        # フィルライト
        light2 = vtk.vtkLight()
        light2.SetPosition(-5, -5, 5)
        light2.SetIntensity(0.4)
        light2.SetColor(0.9, 0.9, 1.0)
        self.renderer.AddLight(light2)
        
    def add_axes(self):
        """大きな座標軸を画面左下隅に追加"""
        # 座標軸アクターを作成
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(4.5, 4.5, 4.5)  # ★★★ 長さは大きく維持 ★★★
        axes.SetCylinderRadius(0.05)        # ★★★ 線を細く（0.24→0.05） ★★★
        axes.SetShaftType(0)                # シンプルな軸
        axes.SetAxisLabels(1)               # ラベル表示
        
        # ★★★ ラベルのフォントサイズは大きく維持 ★★★
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(54)  # 大きく維持
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(54)  # 大きく維持
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(54)  # 大きく維持
        
        # 軸ラベルの色設定（より鮮明に）
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 0.1, 0.1)  # より鮮明な赤
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.1, 1, 0.1)  # より鮮明な緑
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.1, 0.1, 1)  # より鮮明な青
        
        # オリエンテーションマーカーウィジェットを作成
        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(self.interactor)
        
        # ★★★ 位置とサイズを設定（左下隅、より小さく配置） ★★★
        self.orientation_widget.SetViewport(0.0, 0.0, 0.3, 0.3)  # 左下の30%×30%（60%→30%）
        self.orientation_widget.SetEnabled(True)
        self.orientation_widget.InteractiveOff()  # 相互作用を無効（邪魔にならない）
    
    def debug_molecule_info(self):
        """分子情報のデバッグ表示"""
        if self.atoms_data is None:
            print("No molecule data available")
            QMessageBox.warning(self, "Debug", "No molecule data loaded!")
            return
        
        atom_x = self.atoms_data['x']
        atom_y = self.atoms_data['y'] 
        atom_z = self.atoms_data['z']
        
        #print("\n" + "="*50)
        #print("MOLECULE DEBUG INFO")
        #print("="*50)
        
        # 基本統計
        #print(f"Total atoms: {len(atom_x)}")
        #print(f"X range: {np.min(atom_x):.2f} to {np.max(atom_x):.2f}nm (size: {np.max(atom_x)-np.min(atom_x):.2f}nm)")
        #print(f"Y range: {np.min(atom_y):.2f} to {np.max(atom_y):.2f}nm (size: {np.max(atom_y)-np.min(atom_y):.2f}nm)")
        #print(f"Z range: {np.min(atom_z):.2f} to {np.max(atom_z):.2f}nm (size: {np.max(atom_z)-np.min(atom_z):.2f}nm)")
        
        # 中心位置
        center_x = np.mean(atom_x)
        center_y = np.mean(atom_y)
        center_z = np.mean(atom_z)
        print(f"Center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})nm")
        
        # 推奨設定
        mol_size = max(np.max(atom_x)-np.min(atom_x), np.max(atom_y)-np.min(atom_y))
        recommended_scan = mol_size * 1.5
        recommended_tip_z = np.max(atom_z) + 2.0
        
        #print(f"\nRECOMMENDED SETTINGS:")
        #print(f"Scan size: {recommended_scan:.1f}nm (current X: {self.spinScanXNm.value():.1f}nm)")
       # print(f"Tip Z position: {recommended_tip_z:.1f}nm (current: {self.afm_params['tip_z']:.1f}nm)")
        
        # 探針位置チェック
        tip_x = self.afm_params['tip_x']
        tip_y = self.afm_params['tip_y']
        tip_z = self.afm_params['tip_z']
        
        #print(f"\nTIP POSITION CHECK:")
        #print(f"Current tip: ({tip_x:.2f}, {tip_y:.2f}, {tip_z:.2f})nm")
        
        # 分子との重なりチェック
        if (np.min(atom_x) <= tip_x <= np.max(atom_x) and 
            np.min(atom_y) <= tip_y <= np.max(atom_y)):
            #print("✓ Tip is positioned over the molecule")
            pass
        else:
            #print("⚠ WARNING: Tip is NOT over the molecule!")
            pass
        
        if tip_z > np.max(atom_z) + 1.0:
            #print("✓ Tip Z position is safe")
            pass
        else:
            #print("⚠ WARNING: Tip Z position may be too low!")
            pass
        
        #print("="*50)
        
        # UIに推奨設定を表示
        msg = f"""Debug Information:
        
Molecule size: {mol_size:.1f}nm
Current scan size X: {self.spinScanXNm.value():.1f}nm
Current scan size Y: {self.spinScanYNm.value():.1f}nm
Recommended scan size: {recommended_scan:.1f}nm

Current tip Z: {tip_z:.1f}nm  
Recommended tip Z: {recommended_tip_z:.1f}nm

Tip over molecule: {np.min(atom_x) <= tip_x <= np.max(atom_x) and np.min(atom_y) <= tip_y <= np.max(atom_y)}

Check console for detailed information."""
        
        QMessageBox.information(self, "Debug Info", msg)

    def quick_collision_test(self):
        """特定の点での衝突テスト"""
        if self.atoms_data is None:
            print("No molecule data available")
            return
        
        atom_x = self.atoms_data['x']
        atom_y = self.atoms_data['y']
        atom_z = self.atoms_data['z']
        atom_elem = self.atoms_data['element']
        atom_radii = np.array([self.vdw_radii.get(e, self.vdw_radii['other']) for e in atom_elem])
        
        # 分子の中心での衝突テスト
        center_x = np.mean(atom_x)
        center_y = np.mean(atom_y)
        test_z = np.max(atom_z) + 3.0
        
        #print(f"\nQUICK COLLISION TEST:")
        #print(f"Test point: ({center_x:.2f}, {center_y:.2f}, {test_z:.2f})nm")
        
        try:
            height = self.find_collision_height(center_x, center_y, atom_x, atom_y, atom_z, atom_radii)
            #print(f"Calculated collision height: {height:.3f}nm")
            
            # 妥当性チェック
            if height > np.max(atom_z):
                #print("✓ Result seems reasonable (above molecule)")
                result_msg = f"✓ Collision test PASSED\n\nTest point: ({center_x:.2f}, {center_y:.2f})\nCalculated height: {height:.3f}nm\nMolecule top: {np.max(atom_z):.3f}nm"
            else:
                #print("⚠ WARNING: Result may be too low")
                result_msg = f"⚠ Collision test FAILED\n\nTest point: ({center_x:.2f}, {center_y:.2f})\nCalculated height: {height:.3f}nm\nMolecule top: {np.max(atom_z):.3f}nm\n\nHeight is too low!"
                
            QMessageBox.information(self, "Collision Test", result_msg)
            
        except Exception as e:
            print(f"ERROR in collision calculation: {e}")
            QMessageBox.critical(self, "Error", f"Collision test failed:\n{str(e)}")

    def apply_recommended_settings(self):
        """推奨設定を自動適用"""
        if self.atoms_data is None:
            QMessageBox.warning(self, "Warning", "No molecule data loaded!")
            return
        
        atom_x = self.atoms_data['x']
        atom_y = self.atoms_data['y'] 
        atom_z = self.atoms_data['z']
        
        # 推奨設定を計算
        mol_size = max(np.max(atom_x)-np.min(atom_x), np.max(atom_y)-np.min(atom_y))
        recommended_scan = mol_size * 1.5
        recommended_tip_z = np.max(atom_z) + 2.0
        
        # UIに設定を適用
        self.spinScanXNm.setValue(recommended_scan)
        self.spinScanYNm.setValue(recommended_scan)
        
        # 探針Z位置を設定（スライダー値に変換）
        slider_value = int(recommended_tip_z * 5.0)  # z = value / 5.0 の逆算
        slider_value = max(self.tip_z_slider.minimum(), 
                          min(self.tip_z_slider.maximum(), slider_value))
        self.tip_z_slider.setValue(slider_value)
        
        # 探針を分子中心に移動
        center_x = np.mean(atom_x)
        center_y = np.mean(atom_y)
        
        self.tip_x_slider.setValue(int(center_x * 5.0))  # x = value / 5.0 の逆算
        self.tip_y_slider.setValue(int(center_y * 5.0))  # y = value / 5.0 の逆算
        
        print(f"Applied recommended settings:")
        print(f"- Scan size: {recommended_scan:.1f}nm")
        print(f"- Tip position: ({center_x:.1f}, {center_y:.1f}, {recommended_tip_z:.1f})nm")
        
        QMessageBox.information(self, "Settings Applied", 
                               f"Recommended settings applied:\n\n"
                               f"Scan size: {recommended_scan:.1f}nm\n"
                               f"Tip position: ({center_x:.1f}, {center_y:.1f}, {recommended_tip_z:.1f})nm")

        
    def import_file(self):
        """統合ファイルインポート（PDB/CIF/MRC）"""
        initial_dir = self.last_import_dir if hasattr(self, 'last_import_dir') and self.last_import_dir else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Structure File", initial_dir,
            "Structure Files (*.pdb *.cif *.mmcif *.mrc);;PDB files (*.pdb);;mmCIF files (*.cif *.mmcif);;MRC Files (*.mrc);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog)
        
        if not file_path:
            return
        
        self.last_import_dir = os.path.dirname(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdb':
            self._import_pdb_internal(file_path)
        elif ext in ['.cif', '.mmcif']:
            self._import_cif_internal(file_path)
        elif ext == '.mrc':
            self._import_mrc_internal(file_path)
        else:
            QMessageBox.warning(self, "Unsupported Format", 
                              f"File format '{ext}' is not supported.\nSupported formats: .pdb, .cif, .mmcif, .mrc")

    def eventFilter(self, obj, event):
        """Filter events for vtk_widget: accept drag & drop of PDB/CIF/MRC files on PDB Structure area."""
        if self._handle_pymol_mouse_interaction(obj, event):
            return True

        target = False
        # 互換性: vtk/pymol/コンテナいずれにもドロップを許可
        if hasattr(self, 'vtk_widget') and obj is self.vtk_widget:
            target = True
        if hasattr(self, 'pymol_widget') and obj is self.pymol_widget:
            target = True
        if hasattr(self, 'pymol_image_label') and obj is self.pymol_image_label:
            target = True
        if hasattr(self, 'pymol_view_container') and obj is self.pymol_view_container:
            target = True
        if hasattr(self, 'vtk_view_container') and obj is self.vtk_view_container:
            target = True
        if hasattr(self, 'structure_view_splitter') and obj is self.structure_view_splitter:
            target = True
        if target:
            if event.type() == QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    urls = event.mimeData().urls()
                    allowed = ('.pdb', '.cif', '.mmcif', '.mrc')
                    if urls and urls[0].isLocalFile():
                        path = urls[0].toLocalFile()
                        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in allowed:
                            event.acceptProposedAction()
                            return True
            elif event.type() == QEvent.Drop:
                urls = event.mimeData().urls()
                if urls and urls[0].isLocalFile():
                    path = urls[0].toLocalFile()
                    if os.path.isfile(path):
                        self.last_import_dir = os.path.dirname(path)
                        ext = os.path.splitext(path)[1].lower()
                        if ext == '.pdb':
                            self._import_pdb_internal(path)
                        elif ext in ['.cif', '.mmcif']:
                            self._import_cif_internal(path)
                        elif ext == '.mrc':
                            self._import_mrc_internal(path)
                        event.acceptProposedAction()
                        return True
        return super().eventFilter(obj, event)

    def _is_pymol_mouse_target(self, obj):
        if getattr(self, 'pymol_widget', None) is not None and obj is self.pymol_widget:
            return True
        if getattr(self, 'pymol_image_label', None) is not None and obj is self.pymol_image_label:
            return True
        return False

    def _handle_pymol_mouse_interaction(self, obj, event):
        """
        Enable basic mouse interaction for PyMOL image mode.
        - Left drag: rotate
        - Shift+Left drag or Right drag: pan
        - Wheel: zoom
        """
        if not self._is_pymol_active() or not getattr(self, "pymol_image_mode", False):
            return False
        if not self._is_pymol_mouse_target(obj):
            return False

        etype = event.type()
        if etype == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                modifiers = event.modifiers()
                is_ctrl_or_cmd = bool(modifiers & Qt.ControlModifier) or bool(modifiers & Qt.MetaModifier)
                is_shift = bool(modifiers & Qt.ShiftModifier)
                self._pymol_mouse_dragging = True
                self._pymol_mouse_last_pos = event.pos()
                if is_ctrl_or_cmd and not is_shift:
                    self._pymol_mouse_mode = "object_rotate"
                    self._pymol_mouse_panning = False
                    self.actor_rotating = True
                elif is_shift and not is_ctrl_or_cmd:
                    self._pymol_mouse_mode = "pan"
                    self._pymol_mouse_panning = True
                    self.actor_rotating = False
                else:
                    self._pymol_mouse_mode = "view_rotate"
                    self._pymol_mouse_panning = False
                    self.actor_rotating = False
                self._mark_pymol_interaction()
                event.accept()
                return True
            if event.button() == Qt.RightButton:
                self._pymol_mouse_dragging = True
                self._pymol_mouse_mode = "pan"
                self._pymol_mouse_panning = True
                self._pymol_mouse_last_pos = event.pos()
                self.actor_rotating = False
                self._mark_pymol_interaction()
                event.accept()
                return True
            return False

        if etype == QEvent.MouseMove:
            if not self._pymol_mouse_dragging or self._pymol_mouse_last_pos is None:
                return False
            dx = event.pos().x() - self._pymol_mouse_last_pos.x()
            dy = event.pos().y() - self._pymol_mouse_last_pos.y()
            self._pymol_mouse_last_pos = event.pos()
            if dx == 0 and dy == 0:
                return True
            if self._pymol_mouse_mode == "pan":
                self._pymol_pan_from_drag(dx, dy)
            elif self._pymol_mouse_mode == "object_rotate":
                sensitivity = 0.5
                self.update_rotation_from_drag(
                    angle_x_delta=dy * sensitivity,
                    angle_y_delta=dx * sensitivity,
                    angle_z_delta=0.0,
                )
            else:
                self._pymol_view_rotate_from_drag(dx, dy)
            self._mark_pymol_interaction()
            event.accept()
            return True

        if etype == QEvent.MouseButtonRelease:
            if not self._pymol_mouse_dragging:
                return False
            if event.button() not in (Qt.LeftButton, Qt.RightButton):
                return False
            self._pymol_mouse_dragging = False
            self._pymol_mouse_panning = False
            self._pymol_mouse_last_pos = None
            mode = self._pymol_mouse_mode
            self._pymol_mouse_mode = None
            self.actor_rotating = False
            if (
                mode == "object_rotate"
                and hasattr(self, 'interactive_update_check')
                and self.interactive_update_check.isChecked()
            ):
                self.schedule_high_res_simulation()
            self._mark_pymol_interaction()
            event.accept()
            return True

        if etype == QEvent.Wheel:
            try:
                delta_y = int(event.angleDelta().y())
            except Exception:
                delta_y = 0
            if delta_y == 0:
                try:
                    delta_y = int(event.pixelDelta().y())
                except Exception:
                    delta_y = 0
            if delta_y == 0:
                return False
            self._pymol_zoom_from_wheel(delta_y)
            self._mark_pymol_interaction()
            event.accept()
            return True

        if etype == QEvent.Leave:
            if self._pymol_mouse_dragging:
                self._pymol_mouse_dragging = False
                self._pymol_mouse_panning = False
                self._pymol_mouse_last_pos = None
                self._pymol_mouse_mode = None
                self.actor_rotating = False
            return False

        return False

    def _pymol_zoom_from_wheel(self, delta_y):
        """Zoom PyMOL camera in image mode using wheel delta."""
        if not self._is_pymol_active():
            return
        try:
            view = list(self.pymol_cmd.get_view())
            if len(view) < 18:
                return
            old_dist = abs(float(view[11]))
            if old_dist < 1e-6:
                old_dist = max(1e-3, abs(float(view[16] - view[15])) * 0.5)
            zoom_factor = 0.90 if delta_y > 0 else (1.0 / 0.90)
            new_dist = max(1e-3, min(1e6, old_dist * zoom_factor))
            dolly_factor = old_dist / max(new_dist, 1e-9)
            front_offset = float(view[15]) - old_dist
            back_offset = float(view[16]) - old_dist
            view[11] = -new_dist if float(view[11]) < 0.0 else new_dist
            view[15] = new_dist + front_offset
            view[16] = new_dist + back_offset
            self.pymol_cmd.set_view(view)
            self._sync_vtk_zoom_from_pymol(dolly_factor)
            self.request_render()
        except Exception:
            pass

    def _sync_vtk_zoom_from_pymol(self, dolly_factor):
        """Mirror PyMOL wheel zoom to VTK camera in dual mode."""
        if not self._is_dual_mode():
            return
        if not hasattr(self, 'renderer') or self.renderer is None:
            return
        if not hasattr(self, 'vtk_widget') or self.vtk_widget is None:
            return
        try:
            camera = self.renderer.GetActiveCamera()
            if camera is None:
                return
            f = float(dolly_factor)
            if not np.isfinite(f) or f <= 0.0:
                return
            camera.Dolly(f)
            self.renderer.ResetCameraClippingRange()
        except Exception:
            pass

    def _pymol_view_rotate_from_drag(self, dx, dy):
        """Rotate camera/view (not object) to mimic VTK left-drag behavior."""
        if self._is_dual_mode():
            if not hasattr(self, 'renderer') or self.renderer is None:
                return
            try:
                camera = self.renderer.GetActiveCamera()
                if camera is None:
                    return
                sensitivity = 0.4
                camera.Azimuth(float(-dx) * sensitivity)
                camera.Elevation(float(dy) * sensitivity)
                camera.OrthogonalizeViewUp()
                self.renderer.ResetCameraClippingRange()
                if hasattr(self, 'vtk_widget') and self.vtk_widget is not None:
                    self.vtk_widget.GetRenderWindow().Render()
                # In dual mode, ModifiedEvent path syncs VTK camera to PyMOL.
                self._sync_pymol_view_from_vtk()
            except Exception:
                pass
            return

        # PyMOL-only: rotate PyMOL view directly.
        if not self._is_pymol_active():
            return
        try:
            sensitivity = 0.5
            self.pymol_cmd.turn("y", float(dx) * sensitivity)
            self.pymol_cmd.turn("x", float(dy) * sensitivity)
            self.request_render()
        except Exception:
            pass

    def _pymol_pan_from_drag(self, dx, dy):
        """Pan PyMOL view in image mode by changing view origin in camera XY."""
        if not self._is_pymol_active():
            return
        try:
            view = list(self.pymol_cmd.get_view())
            if len(view) < 18:
                return
            dist = abs(float(view[11]))
            if dist < 1e-6:
                dist = 10.0
            pan_scale = max(1e-4, dist * 0.0015)
            view[12] -= float(dx) * pan_scale
            view[13] += float(dy) * pan_scale
            self.pymol_cmd.set_view(view)
            self.request_render()
        except Exception:
            pass

    def _import_pdb_internal(self, file_path):
        """PDBファイルの読み込み（内部メソッド）"""
            
        try:
            # MRCデータをクリア（PDBファイルimport時）
            self.clear_mrc_data()
            # CIF情報をリセット（PDB読み込み時）
            if hasattr(self, 'cif_name'):
                self.cif_name = None
                self.cif_id = ""

            if hasattr(self, 'rotation_widgets'):
                self.reset_structure_rotation()

            # 現在の構造ファイル情報を保持
            self.current_structure_path = file_path
            self.current_structure_type = "pdb"
            # Apply renderer preference (protein structures)
            if not self.pymol_available:
                self._set_render_backend("vtk")
            else:
                pref = getattr(self, "user_render_backend_preference", "dual")
                if pref not in ("dual", "pymol", "vtk"):
                    pref = "dual"
                self._set_render_backend(pref)
            if hasattr(self, 'esp_check'):
                self.esp_check.setChecked(False)
            self.pymol_esp_object = None

            # プログレスバー表示
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()
            
            self.read_pdb_file(file_path)
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            self.update_statistics()
            self.progress_bar.setValue(70)
            QApplication.processEvents()
            
            self.display_molecule()
            self.progress_bar.setValue(90)
            QApplication.processEvents()

            # ロード直後にモデルが画面内に収まるように調整
            self.fit_view_to_contents()
            
            self.create_tip()
             # ★★★ ここから追加 ★★★
            # PDB構造の最高点から2nm上に探針の初期位置を設定
            if self.atoms_data is not None:
                z_max = self.atoms_data['z'].max()
                initial_tip_z = z_max + 2.0
                
                # Z位置スライダーの物理値と表示値を更新
                # スライダー値は物理値の5倍 (z = value / 5.0 の逆算)
                slider_value = int(initial_tip_z * 5.0)
                
                # スライダーが設定可能な範囲内に収まるように調整
                min_val, max_val = self.tip_z_slider.minimum(), self.tip_z_slider.maximum()
                slider_value = max(min_val, min(max_val, slider_value))
                
                # スライダーの値を設定 (これによりupdate_tip_positionが自動で呼ばれる)
                self.tip_z_slider.setValue(slider_value)
            # ★★★ ここまで追加 ★★★

            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            # ファイル名表示
            self.pdb_name = os.path.basename(file_path) 
            self.pdb_id = os.path.splitext(self.pdb_name)[0]
            self.file_label.setText(f"File Name: {self.pdb_name} (PDB)")
            
            # シミュレーションボタンを有効化
            self.simulate_btn.setEnabled(True)
            
            # 回転ウィジェットも有効化
            if hasattr(self, 'rotation_widgets'):
                for axis in ['X', 'Y', 'Z']:
                    self.rotation_widgets[axis]['spin'].setEnabled(True)
                    self.rotation_widgets[axis]['slider'].setEnabled(True)
            
            # プログレスバー非表示
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))

            # Default behavior: auto-run a first simulation when Interactive Update is enabled.
            if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
                QTimer.singleShot(0, self.trigger_interactive_simulation)
            
            QMessageBox.information(self, "Success", 
                                f"Successfully loaded {self.pdb_name}\n"
                                f"Atoms: {len(self.atoms_data['x'])}")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", 
                            f"Failed to load PDB file:\n{str(e)}")

    def _import_cif_internal(self, file_path):
        """mmCIFファイルの読み込み（内部メソッド）"""
        try:
            # MRCデータをクリア（CIFファイルimport時）
            self.clear_mrc_data()

            # PDB情報をリセット（CIF読み込み時）
            if hasattr(self, 'pdb_name'):
                self.pdb_name = None
                self.pdb_id = ""

            if hasattr(self, 'rotation_widgets'):
                self.reset_structure_rotation()

            # 現在の構造ファイル情報を保持
            self.current_structure_path = file_path
            self.current_structure_type = "cif"
            # Apply renderer preference (protein structures)
            if not self.pymol_available:
                self._set_render_backend("vtk")
            else:
                pref = getattr(self, "user_render_backend_preference", "dual")
                if pref not in ("dual", "pymol", "vtk"):
                    pref = "dual"
                self._set_render_backend(pref)
            if hasattr(self, 'esp_check'):
                self.esp_check.setChecked(False)
            self.pymol_esp_object = None

            # プログレスバー表示
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()

            self.read_cif_file(file_path)
            self.progress_bar.setValue(50)
            QApplication.processEvents()

            self.update_statistics()
            self.progress_bar.setValue(70)
            QApplication.processEvents()

            self.display_molecule()
            self.progress_bar.setValue(90)
            QApplication.processEvents()

            # ロード直後にモデルが画面内に収まるように調整
            self.fit_view_to_contents()

            self.create_tip()
            # 分子の最高点から2nm上に探針の初期位置を設定
            if self.atoms_data is not None:
                z_max = self.atoms_data['z'].max()
                initial_tip_z = z_max + 2.0
                slider_value = int(initial_tip_z * 5.0)  # z = value / 5.0 の逆算
                min_val, max_val = self.tip_z_slider.minimum(), self.tip_z_slider.maximum()
                slider_value = max(min_val, min(max_val, slider_value))
                self.tip_z_slider.setValue(slider_value)

            self.progress_bar.setValue(100)
            QApplication.processEvents()

            # ファイル名表示
            self.cif_name = os.path.basename(file_path)
            self.cif_id = os.path.splitext(self.cif_name)[0]
            self.file_label.setText(f"File Name: {self.cif_name} (CIF)")

            # シミュレーションボタンを有効化
            self.simulate_btn.setEnabled(True)

            # 回転ウィジェットも有効化
            if hasattr(self, 'rotation_widgets'):
                for axis in ['X', 'Y', 'Z']:
                    self.rotation_widgets[axis]['spin'].setEnabled(True)
                    self.rotation_widgets[axis]['slider'].setEnabled(True)

            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))

            # Default behavior: auto-run a first simulation when Interactive Update is enabled.
            if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
                QTimer.singleShot(0, self.trigger_interactive_simulation)

            QMessageBox.information(
                self, "Success",
                f"Successfully loaded {self.cif_name}\n"
                f"Atoms: {len(self.atoms_data['x'])}"
            )

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load mmCIF file:\n{str(e)}")
            
    def read_pdb_file(self, file_path):
        """PDBファイルの解析"""
        atoms = []
        helices = []  # (chain_id, start_residue, end_residue)
        sheets = []   # (chain_id, start_residue, end_residue)
        
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                # HELIXレコードの解析
                if line.startswith('HELIX'):
                    try:
                        chain_id = line[19:20].strip()
                        start_residue = int(line[21:25].strip())
                        end_residue = int(line[33:37].strip())
                        helices.append((chain_id, start_residue, end_residue))
                    except (ValueError, IndexError):
                        pass
                
                # SHEETレコードの解析
                elif line.startswith('SHEET'):
                    try:
                        chain_id = line[21:22].strip()
                        start_residue = int(line[22:26].strip())
                        end_residue = int(line[33:37].strip())
                        sheets.append((chain_id, start_residue, end_residue))
                    except (ValueError, IndexError):
                        pass
                
                # ATOM/HETATMレコードの解析
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        # PDBフォーマット解析
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain_id = line[21:22].strip()
                        residue_id = int(line[22:26].strip())
                        icode = line[26:27].strip()
                        
                        x = float(line[30:38]) / 10.0  # Åからnmに変換
                        y = float(line[38:46]) / 10.0
                        z = float(line[46:54]) / 10.0
                        
                        # 元素名取得
                        element = line[76:78].strip()
                        if not element:
                            element = atom_name[0]
                        
                        # B-factor取得
                        try:
                            b_factor = float(line[60:66])
                        except:
                            b_factor = 20.0
                        
                        atoms.append({
                            'name': atom_name,
                            'x': x, 'y': y, 'z': z,
                            'element': element,
                            'residue_name': residue_name,
                            'chain_id': chain_id,
                            'residue_id': residue_id,
                            'icode': icode,
                            'b_factor': b_factor
                        })
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        
        if not atoms:
            raise ValueError("No valid atoms found in PDB file")
        
        # numpy配列に変換
        self.atoms_data = {
            'x': np.array([atom['x'] for atom in atoms]),
            'y': np.array([atom['y'] for atom in atoms]),
            'z': np.array([atom['z'] for atom in atoms]),
            'element': np.array([atom['element'] for atom in atoms]),
            'atom_name': np.array([atom['name'] for atom in atoms]),
            'residue_name': np.array([atom['residue_name'] for atom in atoms]),
            'chain_id': np.array([atom['chain_id'] for atom in atoms]),
            'residue_id': np.array([atom['residue_id'] for atom in atoms]),
            'icode': np.array([atom['icode'] for atom in atoms]),
            'b_factor': np.array([atom['b_factor'] for atom in atoms])
        }
        
        # 二次構造情報を辞書に格納
        self.secondary_structure = {}
        
        # ヘリックスを登録
        for chain_id, start_res, end_res in helices:
            for res_id in range(start_res, end_res + 1):
                key = (chain_id, res_id)
                self.secondary_structure[key] = 'H'
        
        # シートを登録
        for chain_id, start_res, end_res in sheets:
            for res_id in range(start_res, end_res + 1):
                key = (chain_id, res_id)
                self.secondary_structure[key] = 'E'
        
        # 座標を中心化
        self.center_coordinates()
        
        print(f"Loaded {len(atoms)} atoms")
        if helices:
            print(f"Found {len(helices)} helix regions (from PDB)")
        if sheets:
            print(f"Found {len(sheets)} sheet regions (from PDB)")
        
        # HELIX/SHEETレコードがない、または少ない場合は幾何学的検出を実行
        if len(helices) + len(sheets) < 3:
            print("Running geometric secondary structure detection...")
            self.detect_secondary_structure_geometric()

    def read_cif_file(self, file_path):
        """mmCIFファイルの解析（_atom_site loop_ から原子座標を抽出）"""
        tags = []
        atoms = []

        def _as_int(value, default_int):
            try:
                return int(value)
            except Exception:
                return default_int

        def _as_float(value):
            if value in ('.', '?', None):
                return None
            try:
                return float(value)
            except Exception:
                return None

        def _norm_str(value):
            if value in ('.', '?', None):
                return ""
            return str(value)

        def _infer_element(atom_name):
            # mmCIF/PDB互換: 先頭の英字を拾い、2文字元素も最低限対応
            if not atom_name:
                return "C"
            s = str(atom_name).strip()
            if not s:
                return "C"
            # 例: "CA" はカルシウムではなくCαであることが多いが、
            # type_symbolが無いケースのフォールバックなので単純推定に留まる
            s2 = "".join([ch for ch in s if ch.isalpha()])
            if not s2:
                return s[0].upper()
            if len(s2) >= 2 and s2[0].isalpha() and s2[1].islower():
                return (s2[0] + s2[1]).capitalize()
            return s2[0].upper()

        def _tokenize_cif_row(text):
            """
            Tokenize one mmCIF data row defensively.
            Unlike shlex.split, this tolerates unclosed quotes and keeps parsing.
            """
            tokens = []
            current = []
            quote = None
            for ch in text:
                if quote is None:
                    if ch.isspace():
                        if current:
                            tokens.append(''.join(current))
                            current = []
                    elif ch in ("'", '"'):
                        quote = ch
                    else:
                        current.append(ch)
                else:
                    if ch == quote:
                        quote = None
                    else:
                        current.append(ch)
            if current or quote is not None:
                tokens.append(''.join(current))
            return tokens

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "loop_":
                # collect tags
                tags = []
                j = i + 1
                while j < len(lines):
                    t = lines[j].strip()
                    if not t:
                        j += 1
                        continue
                    if t.startswith('_'):
                        tags.append(t.split()[0])
                        j += 1
                        continue
                    break

                is_atom_site_loop = bool(tags) and all(tag.startswith("_atom_site.") for tag in tags)
                if not is_atom_site_loop:
                    i = j
                    continue

                tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}

                # indices with fallbacks
                def _idx(*candidates):
                    for c in candidates:
                        if c in tag_to_idx:
                            return tag_to_idx[c]
                    return None

                ix = _idx("_atom_site.Cartn_x")
                iy = _idx("_atom_site.Cartn_y")
                iz = _idx("_atom_site.Cartn_z")
                if ix is None or iy is None or iz is None:
                    raise ValueError("mmCIF _atom_site loop_ does not contain Cartn_x/Cartn_y/Cartn_z")

                itype = _idx("_atom_site.type_symbol")
                i_atom = _idx("_atom_site.label_atom_id", "_atom_site.auth_atom_id")
                i_comp = _idx("_atom_site.label_comp_id", "_atom_site.auth_comp_id")
                i_asym = _idx("_atom_site.label_asym_id", "_atom_site.auth_asym_id")
                i_seq = _idx("_atom_site.label_seq_id", "_atom_site.auth_seq_id")
                i_b = _idx("_atom_site.B_iso_or_equiv")

                # parse data rows
                k = j
                seq_fallback = 1
                while k < len(lines):
                    raw = lines[k].rstrip("\n")
                    s = raw.strip()

                    if not s:
                        k += 1
                        continue
                    if s.startswith('#'):
                        k += 1
                        break
                    if s == "loop_" or s.startswith("data_") or s.startswith("_"):
                        break
                    if s.startswith(';'):
                        # Multiline values are not expected for _atom_site. Skip block defensively.
                        k += 1
                        while k < len(lines) and not lines[k].startswith(';'):
                            k += 1
                        k += 1
                        continue

                    tokens = _tokenize_cif_row(s)
                    # mmCIF may wrap a row across lines; accumulate tokens until enough
                    while len(tokens) < len(tags) and (k + 1) < len(lines):
                        nxt = lines[k + 1].strip()
                        if not nxt or nxt.startswith('#') or nxt == "loop_" or nxt.startswith("data_") or nxt.startswith("_"):
                            break
                        k += 1
                        tokens.extend(_tokenize_cif_row(lines[k].strip()))

                    if len(tokens) < len(tags):
                        k += 1
                        continue

                    x = _as_float(tokens[ix])
                    y = _as_float(tokens[iy])
                    z = _as_float(tokens[iz])
                    if x is None or y is None or z is None:
                        k += 1
                        continue

                    # Å -> nm
                    x /= 10.0
                    y /= 10.0
                    z /= 10.0

                    atom_name = _norm_str(tokens[i_atom]) if i_atom is not None else ""
                    residue_name = _norm_str(tokens[i_comp]) if i_comp is not None else ""
                    chain_id = _norm_str(tokens[i_asym]) if i_asym is not None else ""

                    residue_id = None
                    if i_seq is not None:
                        residue_id = _as_int(tokens[i_seq], seq_fallback)
                    else:
                        residue_id = seq_fallback
                    seq_fallback += 1

                    element = _norm_str(tokens[itype]) if itype is not None else ""
                    if not element:
                        element = _infer_element(atom_name)

                    b_factor = 20.0
                    if i_b is not None:
                        bf = _as_float(tokens[i_b])
                        if bf is not None:
                            b_factor = float(bf)

                    atoms.append({
                        'name': atom_name,
                        'x': x, 'y': y, 'z': z,
                        'element': element,
                        'residue_name': residue_name,
                        'chain_id': chain_id,
                        'residue_id': int(residue_id) if residue_id is not None else 0,
                        'icode': "",
                        'b_factor': float(b_factor)
                    })

                    k += 1

                i = k
                continue

            i += 1

        if not atoms:
            raise ValueError("No valid atoms found in mmCIF file (_atom_site)")

        self.atoms_data = {
            'x': np.array([atom['x'] for atom in atoms]),
            'y': np.array([atom['y'] for atom in atoms]),
            'z': np.array([atom['z'] for atom in atoms]),
            'element': np.array([atom['element'] for atom in atoms]),
            'atom_name': np.array([atom['name'] for atom in atoms]),
            'residue_name': np.array([atom['residue_name'] for atom in atoms]),
            'chain_id': np.array([atom['chain_id'] for atom in atoms]),
            'residue_id': np.array([atom['residue_id'] for atom in atoms]),
            'icode': np.array([atom['icode'] for atom in atoms]),
            'b_factor': np.array([atom['b_factor'] for atom in atoms])
        }

        self.center_coordinates()
        print(f"Loaded {len(atoms)} atoms from mmCIF")
    
    def detect_secondary_structure_geometric(self):
        """
        幾何学的な二次構造検出（PyMOL風）
        CA原子間の距離パターンからヘリックスとシートを推定
        """
        if self.atoms_data is None:
            return
        
        # Cα原子のみを抽出
        mask = (self.atoms_data['atom_name'] == 'CA')
        if not np.any(mask):
            return
        
        ca_x = self.atoms_data['x'][mask]
        ca_y = self.atoms_data['y'][mask]
        ca_z = self.atoms_data['z'][mask]
        chain_ids = self.atoms_data['chain_id'][mask]
        residue_ids = self.atoms_data['residue_id'][mask]
        
        unique_chains = np.unique(chain_ids)
        
        helix_count = 0
        sheet_count = 0
        
        for chain in unique_chains:
            # チェーン内のCα原子を抽出
            chain_mask = (chain_ids == chain)
            c_x = ca_x[chain_mask]
            c_y = ca_y[chain_mask]
            c_z = ca_z[chain_mask]
            c_res_id = residue_ids[chain_mask]
            
            # 残基ID順にソート
            sort_idx = np.argsort(c_res_id)
            c_x = c_x[sort_idx]
            c_y = c_y[sort_idx]
            c_z = c_z[sort_idx]
            c_res_id_sorted = c_res_id[sort_idx]
            
            if len(c_x) < 5:
                continue
            
            # 各残基について二次構造を判定
            for i in range(len(c_x)):
                res_id = c_res_id_sorted[i]
                key = (chain, res_id)
                
                # 既に二次構造が割り当てられている場合はスキップ
                if key in self.secondary_structure:
                    continue
                
                # ヘリックス検出: i, i+3, i+4 の距離パターン
                is_helix = False
                if i + 4 < len(c_x):
                    # 隣接CA間の距離
                    d1 = np.sqrt((c_x[i+1] - c_x[i])**2 + 
                                 (c_y[i+1] - c_y[i])**2 + 
                                 (c_z[i+1] - c_z[i])**2)
                    
                    # i と i+3 の距離（ヘリックスの特徴）
                    d3 = np.sqrt((c_x[i+3] - c_x[i])**2 + 
                                 (c_y[i+3] - c_y[i])**2 + 
                                 (c_z[i+3] - c_z[i])**2)
                    
                    # i と i+4 の距離（ヘリックスの特徴）
                    d4 = np.sqrt((c_x[i+4] - c_x[i])**2 + 
                                 (c_y[i+4] - c_y[i])**2 + 
                                 (c_z[i+4] - c_z[i])**2)
                    
                    # ヘリックスの判定基準
                    # - 隣接CA距離: 約3.6-4.0Å (0.36-0.40 nm)
                    # - i→i+3距離: 約5.0-5.5Å (0.50-0.55 nm)
                    # - i→i+4距離: 約5.8-6.5Å (0.58-0.65 nm)
                    if (0.34 < d1 < 0.42 and 
                        0.48 < d3 < 0.58 and 
                        0.56 < d4 < 0.68):
                        is_helix = True
                
                if is_helix:
                    self.secondary_structure[key] = 'H'
                    helix_count += 1
                else:
                    # シート検出（簡易版）: 連続で平らな構造
                    is_sheet = False
                    if i + 2 < len(c_x) and i > 0:
                        # 隣接CA間の距離が約3.3-3.5Å (シートの特徴)
                        d1 = np.sqrt((c_x[i+1] - c_x[i])**2 + 
                                     (c_y[i+1] - c_y[i])**2 + 
                                     (c_z[i+1] - c_z[i])**2)
                        
                        d_prev = np.sqrt((c_x[i] - c_x[i-1])**2 + 
                                        (c_y[i] - c_y[i-1])**2 + 
                                        (c_z[i] - c_z[i-1])**2)
                        
                        # シートの判定基準
                        # - CA間距離: 約3.2-3.5Å (0.32-0.35 nm)
                        # - 比較的伸びた構造
                        if 0.31 < d1 < 0.36 and 0.31 < d_prev < 0.36:
                            # 前後の点を含めて判定
                            if i + 2 < len(c_x):
                                # 3つの連続したCAがほぼ直線状かチェック
                                vec1 = np.array([c_x[i] - c_x[i-1], 
                                                c_y[i] - c_y[i-1], 
                                                c_z[i] - c_z[i-1]])
                                vec2 = np.array([c_x[i+1] - c_x[i], 
                                                c_y[i+1] - c_y[i], 
                                                c_z[i+1] - c_z[i]])
                                
                                # ベクトルを正規化
                                vec1_norm = np.linalg.norm(vec1)
                                vec2_norm = np.linalg.norm(vec2)
                                
                                if vec1_norm > 1e-6 and vec2_norm > 1e-6:
                                    vec1 = vec1 / vec1_norm
                                    vec2 = vec2 / vec2_norm
                                    
                                    # 内積が大きい（ほぼ同じ方向）ならシート
                                    dot_product = np.dot(vec1, vec2)
                                    if dot_product > 0.85:  # 約30度以内
                                        is_sheet = True
                    
                    if is_sheet:
                        self.secondary_structure[key] = 'E'
                        sheet_count += 1
                    else:
                        # デフォルトはコイル
                        self.secondary_structure[key] = 'C'
        
        print(f"Geometric detection: {helix_count} helix, {sheet_count} sheet residues")
        
    def center_coordinates(self):
        """座標を中心に移動"""
        for coord in ['x', 'y', 'z']:
            center = (self.atoms_data[coord].max() + self.atoms_data[coord].min()) / 2
            self.atoms_data[coord] -= center
            
    def update_statistics(self):
        """原子統計の更新"""
        if self.atoms_data is None:
            return
            
        total = len(self.atoms_data['x'])
        self.stats_labels['Total'].setText(f"Total: {total}")
        
        for atom_type in ['C', 'O', 'N', 'H']:
            count = np.sum(self.atoms_data['element'] == atom_type)
            self.stats_labels[atom_type].setText(f"{atom_type}: {count}")
        
        # その他の原子
        known_types = ['C', 'O', 'N', 'H']
        other_count = np.sum(~np.isin(self.atoms_data['element'], known_types))
        self.stats_labels['Other'].setText(f"Other: {other_count}")
        
    def get_filtered_atoms(self):
        """表示フィルターに基づいて原子を選択"""
        if self.atoms_data is None:
            return None, None, None, None, None, None, None
            
        atom_filter = self.atom_combo.currentText()
        
        if atom_filter == "All Atoms":
            mask = np.ones(len(self.atoms_data['x']), dtype=bool)
        elif atom_filter == "Heavy Atoms":
            mask = self.atoms_data['element'] != 'H'
        elif atom_filter == "Backbone":
            mask = np.isin(self.atoms_data['atom_name'], ['N', 'CA', 'C', 'O'])
        elif atom_filter in ['C', 'N', 'O']:
            mask = self.atoms_data['element'] == atom_filter
        else:
            mask = np.ones(len(self.atoms_data['x']), dtype=bool)
        
        if not np.any(mask):
            return None, None, None, None, None, None, None
            
        return (self.atoms_data['x'][mask], 
                self.atoms_data['y'][mask],
                self.atoms_data['z'][mask],
                self.atoms_data['element'][mask],
                self.atoms_data['chain_id'][mask],
                self.atoms_data['b_factor'][mask],
                mask)
        
    def get_atom_color(self, element, chain_id, b_factor):
        """原子の色を取得"""
        color_scheme = self.color_combo.currentText()
        
        if color_scheme == "By Element":
            base_color = self.element_colors.get(element, self.element_colors['other'])
        elif color_scheme == "By Chain":
            chain_hash = hash(chain_id) % len(self.chain_colors)
            base_color = self.chain_colors[chain_hash]
        elif color_scheme == "Single Color":
            # Single Colorの場合は選択された色を直接返す
            base_color = self.current_single_color  
            #print(f"Using single color / 単色を使用: {base_color}")  # デバッグ用
        elif color_scheme == "By B-Factor":
            # B-factorを0-1に正規化（0-50の範囲を想定）
            norm_b = np.clip(b_factor / 50.0, 0, 1)
            # 青→緑→黄→赤のグラデーション
            if norm_b < 0.33:
                t = norm_b * 3
                base_color = (0, 0.5 + 0.5*t, 1 - t)
            elif norm_b < 0.66:
                t = (norm_b - 0.33) * 3
                base_color = (t, 1, 0)
            else:
                t = (norm_b - 0.66) * 3
                base_color = (1, 1 - 0.5*t, 0)
        else:
            base_color = self.element_colors.get(element, self.element_colors['other'])
        
        # 明るさファクターを適用
        adjusted_color = tuple(min(1.0, c * self.brightness_factor) for c in base_color)
        return adjusted_color
        
    def display_molecule(self):
        """分子の表示"""
        if self.pymol_available:
            self._apply_view_visibility()
            self._update_renderer_combo()
        if self._is_pymol_active():
            self._display_molecule_pymol()
            if self._is_pymol_only():
                return

        # VTK側を使う場合は初期化を保証
        self._ensure_vtk_initialized()
        if not hasattr(self, 'renderer') or self.renderer is None:
            print("[WARNING] VTK renderer not initialized.")
            return

        # 既存のアクターを削除
        if self.sample_actor:
            self.renderer.RemoveActor(self.sample_actor)
        if self.bonds_actor:
            self.renderer.RemoveActor(self.bonds_actor)
            
        x, y, z, elements, chain_ids, b_factors, mask = self.get_filtered_atoms()
        if x is None:
            return
            
        style = self.style_combo.currentText()
        size_factor = self.size_slider.value() / 100.0
        opacity = self.opacity_slider.value() / 100.0
        quality = self.quality_combo.currentText()
        
        # 品質設定
        if quality == "Fast":
            resolution = 8
            max_atoms = 5000
        elif quality == "Good":
            resolution = 12
            max_atoms = 10000
        else:  # High
            resolution = 16
            max_atoms = 20000
        
        # サンプリング処理
        sampled_indices = None
        if len(x) > max_atoms:
            sampled_indices = np.random.choice(len(x), max_atoms, replace=False)
            x, y, z = x[sampled_indices], y[sampled_indices], z[sampled_indices]
            elements = elements[sampled_indices]
            chain_ids = chain_ids[sampled_indices]
            b_factors = b_factors[sampled_indices]
        
        # スタイルに応じた表示
        if style == "Ball & Stick":
            self.sample_actor = self.create_ball_stick_display(
                x, y, z, elements, chain_ids, b_factors, size_factor, resolution)
        elif style == "Stick Only":
            self.sample_actor = self.create_stick_display(
                x, y, z, elements, chain_ids, b_factors, size_factor, resolution)
        elif style == "Spheres":
            self.sample_actor = self.create_sphere_display(
                x, y, z, elements, chain_ids, b_factors, size_factor, resolution)
        elif style == "Points":
            self.sample_actor = self.create_point_display(
                x, y, z, elements, chain_ids, b_factors, size_factor)
        elif style == "Wireframe":
            self.sample_actor = self.create_wireframe_display(x, y, z)
        elif style == "Simple Cartoon":
            # Cartoon表示は元のデータを使用（サンプリングなし）
            self.sample_actor = self.create_simple_cartoon_display_safe()
        elif style == "Ribbon":
            # Ribbon表示はCα原子を使用（サンプリングなし）
            self.sample_actor = self.create_ribbon_display(size_factor)
        
        # 透明度設定
        if self.sample_actor and hasattr(self.sample_actor, 'GetProperty'):
            self.sample_actor.GetProperty().SetOpacity(opacity)
            
        # アクターを追加
        if self.sample_actor:
            self.renderer.AddActor(self.sample_actor)
            
        # 結合表示（Stick系の場合）
        if style in ["Ball & Stick", "Stick Only"]:
            self.create_bonds_display(x, y, z, elements, chain_ids, b_factors, 
                                    size_factor * 0.3, resolution)
        
        # 現在の回転設定をアクターに適用
        self.apply_structure_rotation()
        
        # 初期回転角度を保存（Reset Allで使用）
        if hasattr(self, 'rotation_widgets'):
            self.initial_rotation_angles = {
                'X': self.rotation_widgets['X']['spin'].value(),
                'Y': self.rotation_widgets['Y']['spin'].value(),
                'Z': self.rotation_widgets['Z']['spin'].value()
            }
            
        self.request_render()
        
    def create_sphere_display(self, x, y, z, elements, chain_ids, b_factors, size_factor, resolution):
        """球体表示"""
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # 全ての点と色を設定（Single Colorでも個別に設定）
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
            
            # 色を取得（Single Colorでも get_atom_color を通す）
            color = self.get_atom_color(elements[i], chain_ids[i], b_factors[i])
            colors.InsertNextTuple3(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(colors)
        polydata.Modified()
        
        # 球体ソース
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.15 * size_factor)
        sphere.SetPhiResolution(resolution)
        sphere.SetThetaResolution(resolution)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetColorModeToColorByScalar()  # 重要：色をスカラーで制御
        glyph.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOn()  # 常にOn
        mapper.SetScalarModeToUsePointData()  # ポイントデータを使用
        mapper.Update()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(20)
        
        return actor
        
    def create_point_display(self, x, y, z, elements, chain_ids, b_factors, size_factor):
        """点表示"""
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
            color = self.get_atom_color(elements[i], chain_ids[i], b_factors[i])
            colors.InsertNextTuple3(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(colors)
        polydata.Modified()  # 追加
        
        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()  # 追加
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())
        mapper.ScalarVisibilityOn()  # 追加
        mapper.Update()  # 追加
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(max(1, size_factor * 5))
        
        return actor
        
    def create_wireframe_display(self, x, y, z):
        """ワイヤーフレーム表示"""
        points = vtk.vtkPoints()
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Delaunay 3D
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(polydata)
        
        # 表面抽出
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputConnection(delaunay.GetOutputPort())
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surface_filter.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.GetProperty().SetLineWidth(1.5)
        
        return actor
        
    def create_simple_cartoon_display_safe(self):
        """
        簡易的なCartoon表示を作成（スプライン補間などを行わない安全な実装）
        """
        # Cα原子のみを抽出
        mask = (self.atoms_data['atom_name'] == 'CA')
        if not np.any(mask):
            return None
            
        ca_x = self.atoms_data['x'][mask]
        ca_y = self.atoms_data['y'][mask]
        ca_z = self.atoms_data['z'][mask]
        chain_ids = self.atoms_data['chain_id'][mask]
        residue_ids = self.atoms_data['residue_id'][mask]
        
        # チェーンごとにソート
        unique_chains = np.unique(chain_ids)
        
        append_poly = vtk.vtkAppendPolyData()
        
        for chain in unique_chains:
            chain_mask = (chain_ids == chain)
            c_x = ca_x[chain_mask]
            c_y = ca_y[chain_mask]
            c_z = ca_z[chain_mask]
            c_res_id = residue_ids[chain_mask]
            
            # 残基ID順にソート
            sort_idx = np.argsort(c_res_id)
            c_x = c_x[sort_idx]
            c_y = c_y[sort_idx]
            c_z = c_z[sort_idx]
            
            if len(c_x) < 2:
                continue
                
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            lines.InsertNextCell(len(c_x))
            
            for i in range(len(c_x)):
                points.InsertNextPoint(c_x[i], c_y[i], c_z[i])
                lines.InsertCellPoint(i)
                
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetLines(lines)
            
            # チューブフィルターで太さを持たせる
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(poly)
            tube.SetRadius(0.15 * (self.size_slider.value() / 100.0)) # 太さは固定
            tube.SetNumberOfSides(8)
            tube.CappingOn()
            tube.Update()
            
            append_poly.AddInputData(tube.GetOutput())
            
        append_poly.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append_poly.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # 色は一律（またはチェーンごとに変えるなど改善の余地あり）
        # 这里ではAtomごとの色を取得して適用
        # 簡易実装では単色などにするが、既存動作に合わせる
        
        return actor

    def create_ribbon_display(self, size_factor):
        """
        PyMOL風のリボン表示を作成（二次構造対応版）
        ヘリックス、シート、ループで異なる形状を生成
        """
        # Cα原子のみを抽出
        if self.atoms_data is None:
            return None

        mask = (self.atoms_data['atom_name'] == 'CA')
        if not np.any(mask):
            mask = (self.atoms_data['atom_name'] == 'P')
            if not np.any(mask):
                return None
            
        ca_x = self.atoms_data['x'][mask]
        ca_y = self.atoms_data['y'][mask]
        ca_z = self.atoms_data['z'][mask]
        elements = self.atoms_data['element'][mask]
        chain_ids = self.atoms_data['chain_id'][mask]
        residue_ids = self.atoms_data['residue_id'][mask]
        b_factors = self.atoms_data['b_factor'][mask]
        
        unique_chains = np.unique(chain_ids)
        
        append_poly = vtk.vtkAppendPolyData()
        
        for chain in unique_chains:
            # チェーン内の原子を抽出
            chain_mask = (chain_ids == chain)
            c_x = ca_x[chain_mask]
            c_y = ca_y[chain_mask]
            c_z = ca_z[chain_mask]
            c_res_id = residue_ids[chain_mask]
            c_elements = elements[chain_mask]
            c_b_factors = b_factors[chain_mask]
            
            # 残基ID順にソート
            sort_idx = np.argsort(c_res_id)
            c_x = c_x[sort_idx]
            c_y = c_y[sort_idx]
            c_z = c_z[sort_idx]
            c_res_id_sorted = c_res_id[sort_idx]
            c_elements = c_elements[sort_idx]
            c_b_factors = c_b_factors[sort_idx]
            
            if len(c_x) < 4:  # スプライン補間のため最低4点必要
                continue
            
            # 各残基の二次構造タイプを取得
            ss_types = []
            for res_id in c_res_id_sorted:
                key = (chain, res_id)
                ss_type = self.secondary_structure.get(key, 'C')  # デフォルトはコイル
                ss_types.append(ss_type)
            
            # Catmull-Romスプラインで滑らかに補間
            num_points = len(c_x)
            subdivisions = 10  # 各セグメント間の分割数
            
            interpolated_points = []
            interpolated_colors = []
            interpolated_ss = []  # 二次構造タイプも補間点に関連付け
            
            for i in range(num_points - 1):
                # Catmull-Romスプライン用の4点を取得
                p0_idx = max(0, i - 1)
                p1_idx = i
                p2_idx = i + 1
                p3_idx = min(num_points - 1, i + 2)
                
                p0 = np.array([c_x[p0_idx], c_y[p0_idx], c_z[p0_idx]])
                p1 = np.array([c_x[p1_idx], c_y[p1_idx], c_z[p1_idx]])
                p2 = np.array([c_x[p2_idx], c_y[p2_idx], c_z[p2_idx]])
                p3 = np.array([c_x[p3_idx], c_y[p3_idx], c_z[p3_idx]])
                
                # 色（p1とp2の間を補間）
                color1 = self.get_atom_color(c_elements[p1_idx], chain, c_b_factors[p1_idx])
                color2 = self.get_atom_color(c_elements[p2_idx], chain, c_b_factors[p2_idx])
                
                # 二次構造タイプ（p1を使用）
                ss_type = ss_types[p1_idx]
                
                for j in range(subdivisions):
                    t = j / subdivisions
                    
                    # Catmull-Romスプライン補間
                    point = 0.5 * (
                        (2 * p1) +
                        (-p0 + p2) * t +
                        (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
                        (-p0 + 3*p1 - 3*p2 + p3) * t**3
                    )
                    
                    interpolated_points.append(point)
                    
                    # 色を線形補間
                    interp_color = tuple(
                        color1[k] * (1 - t) + color2[k] * t
                        for k in range(3)
                    )
                    interpolated_colors.append(interp_color)
                    interpolated_ss.append(ss_type)
            
            # 最後の点を追加
            interpolated_points.append(np.array([c_x[-1], c_y[-1], c_z[-1]]))
            color_last = self.get_atom_color(c_elements[-1], chain, c_b_factors[-1])
            interpolated_colors.append(color_last)
            interpolated_ss.append(ss_types[-1])
            
            # NumPy配列に変換
            interpolated_points = np.array(interpolated_points)
            n_interp = len(interpolated_points)
            
            if n_interp < 3:
                continue
            
            # リボンメッシュを構築（二次構造に応じて幅を変える）
            points = vtk.vtkPoints()
            triangles = vtk.vtkCellArray()
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            
            # 各補間点でリボンの左右の点を生成
            for i in range(n_interp):
                # 二次構造に応じた幅を決定
                ss_type = interpolated_ss[i]
                if ss_type == 'H':  # ヘリックス
                    ribbon_width = 0.6 * size_factor
                elif ss_type == 'E':  # シート
                    ribbon_width = 0.8 * size_factor
                else:  # コイル
                    ribbon_width = 0.2 * size_factor
                
                # 接線ベクトル（進行方向）
                if i == 0:
                    tangent = interpolated_points[1] - interpolated_points[0]
                elif i == n_interp - 1:
                    tangent = interpolated_points[i] - interpolated_points[i-1]
                else:
                    tangent = interpolated_points[i+1] - interpolated_points[i-1]
                
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-6:
                    tangent = tangent / tangent_norm
                else:
                    tangent = np.array([1.0, 0.0, 0.0])
                
                # リボンの幅方向を計算
                up = np.array([0.0, 0.0, 1.0])
                
                # 接線がZ軸と平行な場合は別の軸を使用
                if abs(np.dot(tangent, up)) > 0.99:
                    up = np.array([1.0, 0.0, 0.0])
                
                # リボンの幅方向
                width_dir = np.cross(tangent, up)
                width_norm = np.linalg.norm(width_dir)
                if width_norm > 1e-6:
                    width_dir = width_dir / width_norm
                else:
                    width_dir = np.array([0.0, 1.0, 0.0])
                
                # 前の点との一貫性を保つため、必要に応じて方向を反転
                if i > 0:
                    if np.dot(width_dir, prev_width_dir) < 0:
                        width_dir = -width_dir
                
                prev_width_dir = width_dir.copy()
                
                # リボンの左右の点
                half_width = ribbon_width / 2.0
                center = interpolated_points[i]
                left_point = center - width_dir * half_width
                right_point = center + width_dir * half_width
                
                # 点を追加
                points.InsertNextPoint(left_point[0], left_point[1], left_point[2])
                points.InsertNextPoint(right_point[0], right_point[1], right_point[2])
                
                # 色を設定
                color = interpolated_colors[i]
                color_tuple = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                colors.InsertNextTuple3(*color_tuple)
                colors.InsertNextTuple3(*color_tuple)
                
                # 三角形メッシュを構築
                if i > 0:
                    prev_left = (i - 1) * 2
                    prev_right = (i - 1) * 2 + 1
                    curr_left = i * 2
                    curr_right = i * 2 + 1
                    
                    # 三角形1
                    triangle1 = vtk.vtkTriangle()
                    triangle1.GetPointIds().SetId(0, prev_left)
                    triangle1.GetPointIds().SetId(1, curr_left)
                    triangle1.GetPointIds().SetId(2, prev_right)
                    triangles.InsertNextCell(triangle1)
                    
                    # 三角形2
                    triangle2 = vtk.vtkTriangle()
                    triangle2.GetPointIds().SetId(0, curr_left)
                    triangle2.GetPointIds().SetId(1, curr_right)
                    triangle2.GetPointIds().SetId(2, prev_right)
                    triangles.InsertNextCell(triangle2)
            
            # PolyDataを作成
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetPolys(triangles)
            poly.GetPointData().SetScalars(colors)
            
            append_poly.AddInputData(poly)
            
        append_poly.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append_poly.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointData()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # マテリアル設定
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(40)
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.7)
        
        return actor
    
    def create_simple_ca_points(self, ca_x, ca_y, ca_z, ca_chains):
        """CAアトムの点表示（フォールバック用）"""
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        for i in range(len(ca_x)):
            points.InsertNextPoint(ca_x[i], ca_y[i], ca_z[i])
            
            # チェーン色
            chain_hash = hash(ca_chains[i]) % len(self.chain_colors)
            color = self.chain_colors[chain_hash]
            colors.InsertNextTuple3(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(colors)
        
        # 球体で表示
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.3)
        sphere.SetPhiResolution(12)
        sphere.SetThetaResolution(12)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(20)
        
        return actor
        
    def create_ball_stick_display(self, x, y, z, elements, chain_ids, b_factors, 
                                size_factor, resolution):
        """ボール&スティック表示"""
        return self.create_sphere_display(x, y, z, elements, chain_ids, b_factors, 
                                        size_factor * 0.7, resolution)
        
    def create_stick_display(self, x, y, z, elements, chain_ids, b_factors, 
                           size_factor, resolution):
        """スティック表示"""
        return self.create_sphere_display(x, y, z, elements, chain_ids, b_factors, 
                                        size_factor * 0.3, resolution)
        
    def create_bonds_display(self, x, y, z, elements, chain_ids, b_factors, 
                           bond_radius, resolution):
        """結合の表示"""
        if self.bonds_actor:
            self.renderer.RemoveActor(self.bonds_actor)
            
        # 簡単な距離ベース結合判定
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # 全ての点を追加
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
        
        # 近接原子間で結合を作成（効率化のため制限）
        max_bonds = 10000
        bond_count = 0
        
        for i in range(len(x)):
            if bond_count >= max_bonds:
                break
                
            for j in range(i + 1, min(i + 20, len(x))):  # 近くの原子のみチェック
                if bond_count >= max_bonds:
                    break
                    
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)
                
                # 結合距離判定
                if dist < 0.18:  # 1.8 Å
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i)
                    line.GetPointIds().SetId(1, j)
                    lines.InsertNextCell(line)
                    
                    # 結合の色（平均色）
                    color1 = self.get_atom_color(elements[i], chain_ids[i], b_factors[i])
                    color2 = self.get_atom_color(elements[j], chain_ids[j], b_factors[j])
                    avg_color = [(color1[k] + color2[k])/2 for k in range(3)]
                    colors.InsertNextTuple3(
                        int(avg_color[0]*255), 
                        int(avg_color[1]*255), 
                        int(avg_color[2]*255)
                    )
                    
                    bond_count += 1
        
        if bond_count > 0:
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            polydata.GetCellData().SetScalars(colors)
            
            # チューブフィルター
            tube_filter = vtk.vtkTubeFilter()
            tube_filter.SetInputData(polydata)
            tube_filter.SetRadius(bond_radius)
            tube_filter.SetNumberOfSides(max(4, resolution // 2))
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tube_filter.GetOutputPort())
            
            self.bonds_actor = vtk.vtkActor()
            self.bonds_actor.SetMapper(mapper)
            self.bonds_actor.GetProperty().SetSpecular(0.3)
            self.bonds_actor.GetProperty().SetSpecularPower(20)
            
            self.renderer.AddActor(self.bonds_actor)
        
    def create_tip(self):
        """AFM探針の作成（実際のパラメーターに基づく）"""
        if self._is_pymol_only():
            return
        self._ensure_vtk_initialized()
        if not hasattr(self, 'renderer') or self.renderer is None:
            print("[WARNING] VTK renderer not initialized for tip.")
            return
        if self.tip_actor:
            self.renderer.RemoveActor(self.tip_actor)
            
        tip_shape = self.tip_shape_combo.currentText().lower()
        radius = self.tip_radius_spin.value()
        angle = self.tip_angle_spin.value()
        # ★★★ 追加: 新しいUIからminitipの半径を取得 ★★★
        minitip_radius = self.minitip_radius_spin.value()
        
        #print(f"Creating tip: {tip_shape}, radius={radius}nm, angle={angle}°, minitip_radius={minitip_radius}nm")
        
        if tip_shape == "cone":
            self.tip_actor = self.create_cone_tip(radius, angle)
        elif tip_shape == "sphere":
            # ★★★ 変更点: minitip_radiusを引数として渡す ★★★
            self.tip_actor = self.create_sphere_tip(radius, angle, minitip_radius)
        else:  # paraboloid
            self.tip_actor = self.create_paraboloid_tip(radius)
        
        if self.tip_actor:
            self.update_tip_position()
            self.renderer.AddActor(self.tip_actor)
            self._update_tip_visual_state(
                self.show_tip_check.isChecked() if hasattr(self, 'show_tip_check') else True
            )

    # +++ この関数で既存のcreate_cone_tipを置き換えてください +++
    def create_cone_tip(self, tip_radius, half_angle):
        """
        Igor Proの数式に基づいて高さマップを生成し、そこから探針形状を作成します。
        この方法は非常に安定しており、環境に依存する問題を回避します。
        先端は-Z方向を向き、長さも調整されています。
        """
        if self.tip_actor:
            self.renderer.RemoveActor(self.tip_actor)

        # --- Igor Proのロジックに基づいたパラメータ計算 ---
        if half_angle < 1.0: half_angle = 1.0
        if half_angle >= 89.0: half_angle = 89.0
        half_angle_rad = np.radians(float(half_angle))
        
        # 形状が球から円錐に切り替わる臨界半径
        r_crit = tip_radius * np.cos(half_angle_rad)
        # 円錐部分が滑らかに接続するためのZオフセット
        z_offset = (tip_radius / np.sin(half_angle_rad)) - tip_radius

        # --- 点群グリッドの生成 ---
        resolution = 101  # グリッドの解像度 (奇数にすると中心点ができます)
        
        # ★★★ 変更点1: コーンを長くするため、高さを大きく設定 ★★★
        max_height = tip_radius * 50.0  # 以前は 25.0 でした
        
        max_radius = (max_height + z_offset) * np.tan(half_angle_rad)
        
        points = vtk.vtkPoints()
        
        # グリッド上の各点の3D座標を計算
        for i in range(resolution):
            for j in range(resolution):
                # グリッド座標(i, j)を物理座標(x, y)に変換
                x = (j - (resolution - 1) / 2.0) * (2 * max_radius / (resolution - 1))
                y = (i - (resolution - 1) / 2.0) * (2 * max_radius / (resolution - 1))
                
                # 中心からの距離rを計算
                r = np.sqrt(x**2 + y**2)
                
                # Igorの数式を使ってz座標(高さ)を計算
                if r <= r_crit:
                    # 球状部分の計算式
                    sqrt_arg = tip_radius**2 - r**2
                    z = tip_radius - np.sqrt(max(0, sqrt_arg))
                else:
                    # 円錐状部分の計算式
                    z = (r / np.tan(half_angle_rad)) - z_offset
                
                # ★★★ 変更点2: 先端が-Z方向を向くように、Z座標を反転 ★★★
                points.InsertNextPoint(x, y, z)

        # --- 点群からサーフェスメッシュを生成 ---
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Delaunay2Dアルゴリズムで点群から三角形メッシュを生成
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.Update() # 念のためUpdateを呼び出します

        # --- ★★★ 変更点3: Z反転を直接行ったため、後処理が不要に ★★★
        # 以前のtransformやnormalsの処理は不要になり、コードがシンプルになりました。
        
        # --- アクターの作成 ---
        mapper = vtk.vtkPolyDataMapper()
        # Delaunayの結果を直接マッパーに接続します
        mapper.SetInputConnection(delaunay.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # 材質を設定
        actor.GetProperty().SetColor(1.0, 0.84, 0.0)  # ゴールドのRGB値
        actor.GetProperty().SetSpecular(0.9)         # 高い鏡面反射で金属感を強調
        actor.GetProperty().SetSpecularPower(100)    # 光沢を強くする
        actor.GetProperty().SetDiffuse(0.6)          # 拡散反射
        actor.GetProperty().SetAmbient(0.3)    

        #print(f"SUCCESS: Flipped and elongated cone tip created: radius={tip_radius:.1f}nm, angle={half_angle}°")
        
        return actor
    
    # +++ この関数で既存のcreate_sphere_tipを置き換えてください +++
    # +++ この関数で既存のcreate_sphere_tipを置き換えてください +++
    def create_sphere_tip(self, tip_radius, half_angle, minitip_radius):
        """
        Cone形状の上に、指定された半径(minitip_radius)の球を接着した形状を生成します。
        """
        if self.tip_actor:
            self.renderer.RemoveActor(self.tip_actor)

        # --- 部品1: 先端に突き出る球を作成 ---
        # ★★★ 変更点: 引数で渡されたminitip_radiusを使用 ★★★
        sphere_radius = minitip_radius
        
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(sphere_radius)
        sphere_source.SetPhiResolution(50)
        sphere_source.SetThetaResolution(50)
        
        sphere_transform = vtk.vtkTransform()
        sphere_transform.Translate(0, 0, sphere_radius)
        
        sphere_filter = vtk.vtkTransformPolyDataFilter()
        sphere_filter.SetInputConnection(sphere_source.GetOutputPort())
        sphere_filter.SetTransform(sphere_transform)
        sphere_filter.Update()

        # --- 部品2: Cone部分を作成し、球の上部に移動 ---
        if half_angle < 1.0: half_angle = 1.0
        if half_angle >= 89.0: half_angle = 89.0
        half_angle_rad = np.radians(float(half_angle))
        
        r_crit_cone = tip_radius * np.cos(half_angle_rad)
        z_offset_cone = (tip_radius / np.sin(half_angle_rad)) - tip_radius
        
        resolution = 101
        max_height_cone = tip_radius * 50.0
        max_radius_cone = (max_height_cone + z_offset_cone) * np.tan(half_angle_rad)
        
        cone_points = vtk.vtkPoints()
        for i in range(resolution):
            for j in range(resolution):
                x = (j - (resolution - 1) / 2.0) * (2 * max_radius_cone / (resolution - 1))
                y = (i - (resolution - 1) / 2.0) * (2 * max_radius_cone / (resolution - 1))
                r = np.sqrt(x**2 + y**2)
                
                if r <= r_crit_cone:
                    z = tip_radius - np.sqrt(max(0, tip_radius**2 - r**2))
                else:
                    z = (r / np.tan(half_angle_rad)) - z_offset_cone
                cone_points.InsertNextPoint(x, y, z)
        
        cone_polydata = vtk.vtkPolyData()
        cone_polydata.SetPoints(cone_points)
        cone_delaunay = vtk.vtkDelaunay2D()
        cone_delaunay.SetInputData(cone_polydata)
        
        cone_transform = vtk.vtkTransform()
        cone_transform.Translate(0, 0, 2 * sphere_radius)
        
        cone_filter = vtk.vtkTransformPolyDataFilter()
        cone_filter.SetInputConnection(cone_delaunay.GetOutputPort())
        cone_filter.SetTransform(cone_transform)
        cone_filter.Update()

        # --- 2つの部品を結合 ---
        append_filter = vtk.vtkAppendPolyData()
        append_filter.AddInputData(sphere_filter.GetOutput())
        append_filter.AddInputData(cone_filter.GetOutput())
        append_filter.Update()

        # --- アクターを作成 ---
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(1.0, 0.84, 0.0)
        prop.SetSpecular(0.9)
        prop.SetSpecularPower(100)
        prop.SetDiffuse(0.6)
        prop.SetAmbient(0.3)
        prop.SetOpacity(0.95)

        print(f"SUCCESS: Composite 'Sphere' created. Cone R={tip_radius:.1f}, Minitip R={minitip_radius:.1f}")
        return actor
    
    def create_paraboloid_tip(self, tip_radius):
        """
        Igor Proの数式に基づき、先端が下(-Z)を向く放物面探針を生成します。
        """
        if self.tip_actor:
            self.renderer.RemoveActor(self.tip_actor)

        # --- グリッドと点群の準備 ---
        resolution = 101
        display_height = 20.0 
        max_radius = np.sqrt(2 * tip_radius * display_height)
        points = vtk.vtkPoints()
        
        for i in range(resolution):
            for j in range(resolution):
                x = (j - (resolution - 1) / 2.0) * (2 * max_radius / (resolution - 1))
                y = (i - (resolution - 1) / 2.0) * (2 * max_radius / (resolution - 1))
                
                # Igorの数式 z = (x^2 + y^2) / (2 * R)
                z = (x**2 + y**2) / (2 * tip_radius)
                
                # ★★★ 修正点: 先端が下(-Z)を向くようにZ座標を反転 ★★★
                points.InsertNextPoint(x, y, z)

        # --- メッシュ生成とアクター作成 ---
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(delaunay.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # 材質をゴールドに設定
        prop = actor.GetProperty()
        prop.SetColor(1.0, 0.84, 0.0)
        prop.SetSpecular(0.9)
        prop.SetSpecularPower(100)
        prop.SetDiffuse(0.6)
        prop.SetAmbient(0.3)
        prop.SetOpacity(0.95)

        print(f"SUCCESS: Paraboloid tip created (pointing down): R={tip_radius:.1f}nm")
        return actor
    
        
    def update_display(self):
        """表示の更新"""
        if self.atoms_data is not None:
            current_scheme = self.color_combo.currentText()
            #print(f"Updating display with color scheme: {current_scheme}")
            ##if current_scheme == "Single Color":
                #print(f"Single color value: {self.current_single_color}")
            
            self.display_molecule()
            
            # レンダリングを強制実行
            self.request_render()
    
    def update_tip_info(self):
        """探針情報の更新"""
        shape = self.tip_shape_combo.currentText()
        radius = self.tip_radius_spin.value()
        angle = self.tip_angle_spin.value()
        
        if shape == "Cone":
            height = radius * 3
            base_radius = radius + height * np.tan(np.radians(angle))
            info = f"Tip: {radius}nm radius\nCone: {height:.1f}nm height\nBase: {base_radius:.1f}nm radius"
        elif shape == "Sphere":
            info = f"Sphere: {radius}nm radius"
        else:
            info = f"Paraboloid: {radius}nm radius\nAngle: {angle}°"
        
        self.tip_info_label.setText(info)
            
    def update_tip(self):
        """探針の更新（パラメーター変更時）"""
        #print("Tip parameters changed - updating display...")
        self.create_tip()
        self.update_tip_info()  # 追加
        
        # AFMパラメーターも更新
        self.afm_params.update({
            'tip_radius': self.tip_radius_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'tip_angle': self.tip_angle_spin.value(),
        })

        # スレッドの安全性をチェックしてからシミュレーションを実行
        self.trigger_interactive_simulation()
    
    def trigger_interactive_simulation(self):
        """インタラクティブモードがONの場合にシミュレーションを実行する汎用トリガー"""
        if getattr(self, '_pose_estimation_running', False):
            return

        # スライダー操作中は実行しない
        if hasattr(self, 'tip_slider_pressed') and self.tip_slider_pressed:
            return
        
        # 既にシミュレーションが実行中の場合は実行しない
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            return
            
        if self.interactive_update_check.isChecked():
            # ★★★ 通常のInteractive Updateでも軽量版を使用 ★★★
            self.run_simulation_silent()
        
    def on_tip_slider_pressed(self):
        """Tip positionスライダーが押された時の処理"""
        # スライダー操作中のフラグを設定
        self.tip_slider_pressed = True
        
    def on_tip_slider_released(self):
        """Tip positionスライダーが離された時の処理"""
        # スライダー操作完了のフラグを設定
        self.tip_slider_pressed = False
        
        # スライダー操作完了後にシミュレーションを実行（Interactive UpdateがONの場合のみ）
        if self.interactive_update_check.isChecked():
            # 遅延実行でシミュレーションをトリガー
            QTimer.singleShot(100, self.trigger_interactive_simulation)
    
    # Scan Size関連のイベントハンドラー
    def scan_size_value_changed(self, value):
        """Scan Size値変更時の処理（マウス/ボタン操作時は即時更新）"""
        if not self.scan_size_keyboard_input:
            # デバウンス処理：既存のタイマーを停止して新しいタイマーを設定
            if self.scan_size_debounce_timer:
                self.scan_size_debounce_timer.stop()
            self.scan_size_debounce_timer = QTimer(self)
            self.scan_size_debounce_timer.setSingleShot(True)
            self.scan_size_debounce_timer.timeout.connect(self.trigger_interactive_simulation)
            self.scan_size_debounce_timer.start(100)  # 100ms後に実行
    
    def scan_size_editing_finished(self):
        """Scan Size編集完了時の処理（キー入力時はリターンで更新）"""
        self.scan_size_keyboard_input = False
        self.trigger_interactive_simulation()
    
    # Tip Radius関連のイベントハンドラー
    def tip_radius_value_changed(self, value):
        """Tip Radius値変更時の処理（マウス/ボタン操作時は即時更新）"""
        if not self.tip_radius_keyboard_input:
            # デバウンス処理
            if self.tip_radius_debounce_timer:
                self.tip_radius_debounce_timer.stop()
            self.tip_radius_debounce_timer = QTimer(self)
            self.tip_radius_debounce_timer.setSingleShot(True)
            self.tip_radius_debounce_timer.timeout.connect(self.update_tip)
            self.tip_radius_debounce_timer.start(100)
    
    def tip_radius_editing_finished(self):
        """Tip Radius編集完了時の処理（キー入力時はリターンで更新）"""
        self.tip_radius_keyboard_input = False
        self.update_tip()
    
    # Minitip Radius関連のイベントハンドラー
    def minitip_radius_value_changed(self, value):
        """Minitip Radius値変更時の処理（マウス/ボタン操作時は即時更新）"""
        if not self.minitip_radius_keyboard_input:
            # デバウンス処理
            if self.minitip_radius_debounce_timer:
                self.minitip_radius_debounce_timer.stop()
            self.minitip_radius_debounce_timer = QTimer(self)
            self.minitip_radius_debounce_timer.setSingleShot(True)
            self.minitip_radius_debounce_timer.timeout.connect(self.update_tip)
            self.minitip_radius_debounce_timer.start(100)
    
    def minitip_radius_editing_finished(self):
        """Minitip Radius編集完了時の処理（キー入力時はリターンで更新）"""
        self.minitip_radius_keyboard_input = False
        self.update_tip()
    
    # Tip Angle関連のイベントハンドラー
    def tip_angle_value_changed(self, value):
        """Tip Angle値変更時の処理（マウス/ボタン操作時は即時更新）"""
        if not self.tip_angle_keyboard_input:
            # デバウンス処理
            if self.tip_angle_debounce_timer:
                self.tip_angle_debounce_timer.stop()
            self.tip_angle_debounce_timer = QTimer(self)
            self.tip_angle_debounce_timer.setSingleShot(True)
            self.tip_angle_debounce_timer.timeout.connect(self.update_tip)
            self.tip_angle_debounce_timer.start(100)
    
    def tip_angle_editing_finished(self):
        """Tip Angle編集完了時の処理（キー入力時はリターンで更新）"""
        self.tip_angle_keyboard_input = False
        self.update_tip()
    
    # キープレスイベントハンドラー
    def scan_size_key_press_event(self, widget, event):
        """Scan Sizeキー入力時の処理"""
        # 数字キーや編集キーが押された場合はキーボード入力フラグを設定
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                          Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
                          Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right]:
            self.scan_size_keyboard_input = True
        QDoubleSpinBox.keyPressEvent(widget, event)
    
    def tip_radius_key_press_event(self, event):
        """Tip Radiusキー入力時の処理"""
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                          Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
                          Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right]:
            self.tip_radius_keyboard_input = True
        QDoubleSpinBox.keyPressEvent(self.tip_radius_spin, event)
    
    def minitip_radius_key_press_event(self, event):
        """Minitip Radiusキー入力時の処理"""
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                          Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
                          Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right]:
            self.minitip_radius_keyboard_input = True
        QDoubleSpinBox.keyPressEvent(self.minitip_radius_spin, event)
    
    def tip_angle_key_press_event(self, event):
        """Tip Angleキー入力時の処理"""
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                          Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
                          Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right]:
            self.tip_angle_keyboard_input = True
        QDoubleSpinBox.keyPressEvent(self.tip_angle_spin, event)
    
    def update_tip_position(self):
        """探針位置の更新（適切な範囲）"""
        if not self.tip_actor:
            return
            
        # スライダー値をnm単位に変換（範囲を調整）
        x = self.tip_x_slider.value() / 5.0  # -10 to +10 nm
        y = self.tip_y_slider.value() / 5.0  # -10 to +10 nm
        z = self.tip_z_slider.value() / 5.0  # 2 to 20 nm
        
        self.tip_actor.SetPosition(x, y, z)
        
        self.tip_x_label.setText(f"{x:.1f}")
        self.tip_y_label.setText(f"{y:.1f}")
        self.tip_z_label.setText(f"{z:.1f}")
        
        # AFMパラメーターも更新
        self.afm_params.update({
            'tip_x': x,
            'tip_y': y,
            'tip_z': z,
        })
        
        self.request_render()
        
        # スライダー操作中はシミュレーションを実行しない
        if hasattr(self, 'tip_slider_pressed') and self.tip_slider_pressed:
            return
        
    def toggle_molecule_visibility(self, visible):
        """分子表示の切り替え"""
        if self._is_pymol_active():
            try:
                if visible:
                    self.pymol_cmd.show("everything", self.pymol_object_name)
                else:
                    self.pymol_cmd.hide("everything", self.pymol_object_name)
            except Exception:
                pass
            self.request_render()
            if self._is_pymol_only():
                return
        if self.sample_actor:
            self.sample_actor.SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()
            
    def _update_tip_visual_state(self, visible=None):
        """Apply tip visibility and opacity based on checkbox and current view."""
        if self._is_pymol_only():
            return
        if not self.tip_actor:
            return

        if visible is None:
            if hasattr(self, 'show_tip_check'):
                visible = bool(self.show_tip_check.isChecked())
            else:
                visible = True
        visible = bool(visible)

        self.tip_actor.SetVisibility(visible)
        if visible:
            current_view = self.get_current_view_orientation()
            # XY view: keep tip visible but transparent.
            opacity = 0.20 if current_view == 'xy' else 0.95
            try:
                prop = self.tip_actor.GetProperty()
                if prop is not None:
                    prop.SetOpacity(opacity)
            except Exception:
                pass

        try:
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def toggle_tip_visibility(self, visible):
        """探針表示の切り替え"""
        self._update_tip_visual_state(visible)
            
    def toggle_bonds_visibility(self, visible):
        """結合表示の切り替え"""
        if self._is_pymol_only():
            return
        if self.bonds_actor:
            self.bonds_actor.SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()
            
   
    def get_rotated_atom_coords(self):
        """Applies the current rotation transform to the base atom coordinates."""
        if self.atoms_data is None:
            return None

        # Get original coordinates
        x = self.atoms_data['x']
        y = self.atoms_data['y']
        z = self.atoms_data['z']
        num_atoms = len(x)

        # PyMOL-only の場合は PyMOLビュー回転を優先して反映（VTK transformが未使用のため）
        if self._is_pymol_only():
            coords = np.column_stack([x, y, z])
            try:
                view = self.pymol_cmd.get_view()
                if view and len(view) >= 18:
                    rot = np.array(view[:9], dtype=float).reshape((3, 3))
                    origin = np.array(view[12:15], dtype=float)
                    # 原点を中心に回転
                    centered = coords - origin
                    rotated = centered @ rot.T
                    rotated = rotated + origin
                    if np.all(np.isfinite(rotated)):
                        return rotated
            except Exception as e:
                print(f"[WARNING] PyMOL view rotation failed: {e}, using fallback transform")

        # 変換行列が存在しない場合は元の座標を返す
        if not hasattr(self, 'combined_transform') or self.combined_transform is None:
            return np.column_stack([x, y, z])

        try:
            # Get the 4x4 transformation matrix from the combined_transform (base + local)
            vtk_matrix = self.combined_transform.GetMatrix()
            
            # 変換行列の値を安全に取得
            transform_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    element = vtk_matrix.GetElement(i, j)
                    # 異常な値をチェック
                    if not np.isfinite(element) or abs(element) > 1e6:
                        print(f"[WARNING] Invalid transform matrix element [{i},{j}]: {element}")
                        return np.column_stack([x, y, z])
                    transform_matrix[i, j] = element
            
            # 変換行列の妥当性をチェック（単位行列に近いかどうか）
            identity = np.eye(4)
            if np.allclose(transform_matrix, identity, atol=1e-6):
                # 変換がない場合は元の座標を返す
                return np.column_stack([x, y, z])
            
            # 座標を同次座標に変換
            original_coords = np.vstack([x, y, z, np.ones(num_atoms)])
            
            # 変換を適用
            with np.errstate(all='ignore'):  # 警告を無視
                rotated_coords_homogeneous = transform_matrix @ original_coords
            
            # NaNやInfをチェック
            if not np.all(np.isfinite(rotated_coords_homogeneous)):
                print("[WARNING] Non-finite values in rotation calculation, using original coordinates")
                return np.column_stack([x, y, z])
            
            # 3D座標に変換
            rotated_coords = rotated_coords_homogeneous[:3, :].T
            
            # 結果の妥当性をチェック
            if not np.all(np.isfinite(rotated_coords)):
                print("[WARNING] Non-finite values in rotated coordinates, using original coordinates")
                return np.column_stack([x, y, z])
            
            # 座標が異常に大きくなっていないかチェック
            max_coord = np.max(np.abs(rotated_coords))
            if max_coord > 1e6:
                print(f"[WARNING] Rotated coordinates too large (max: {max_coord}), using original coordinates")
                return np.column_stack([x, y, z])
            
            return rotated_coords
            
        except Exception as e:
            print(f"[WARNING] Error in rotation calculation: {e}, using original coordinates")
            return np.column_stack([x, y, z])
        
    
    def _connect_worker_delete_later(self, worker):
        """ワーカー終了時にdeleteLaterで安全に破棄する（重複接続は避ける）"""
        if worker is None:
            return
        try:
            worker.finished.connect(worker.deleteLater, type=Qt.UniqueConnection)  # type: ignore[arg-type]
        except Exception:
            # 既に接続済み/環境差異などは黙って無視
            try:
                worker.finished.connect(worker.deleteLater)
            except Exception:
                pass

    def _clear_worker_ref(self, attr_name, worker):
        """self.<attr_name> が worker を指している場合のみ None にする"""
        try:
            if attr_name and hasattr(self, attr_name) and getattr(self, attr_name) is worker:
                setattr(self, attr_name, None)
        except Exception:
            pass

    def is_worker_running(self, worker, attr_name=None):
        """
        deleteLater等で破棄済みのQObjectを考慮した isRunning 判定。
        - RuntimeError（wrapped C/C++ object ... has been deleted）を握りつぶし、
          可能なら参照をクリアして False を返す。
        """
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            if attr_name:
                self._clear_worker_ref(attr_name, worker)
            return False
        except Exception:
            return False

    def _track_worker_ref(self, attr_name, worker):
        """finished/destroyedで参照を確実にクリアするための接続を追加"""
        if worker is None:
            return
        try:
            # finished時に参照をクリア（worker変数は参照比較にしか使わないので安全）
            worker.finished.connect(lambda _=None, w=worker: self._clear_worker_ref(attr_name, w))
        except Exception:
            pass
        try:
            # destroyed時も参照をクリア（Qt側が先に消えるケース対策）
            worker.destroyed.connect(lambda _=None, w=worker: self._clear_worker_ref(attr_name, w))
        except Exception:
            pass

    def stop_worker(self, worker, timeout_ms=100, allow_terminate=False, worker_name="worker"):
        """
        QThreadを安全に停止する。
        - 自己wait（QThread::wait: Thread tried to wait on itself）を防ぐため、
          currentThread == worker の場合はwaitしない。
        - finished→deleteLater を接続してGCタイミング依存を減らす。

        Returns:
            bool: 停止済み（=実行中でない）ならTrue
        """
        if worker is None:
            return True

        try:
            # 自分自身のスレッドからwaitしない（Qt警告＆デッドロック回避）
            if QThread.currentThread() == worker:
                try:
                    if hasattr(worker, "cancel"):
                        worker.cancel()
                except Exception:
                    pass
                try:
                    worker.requestInterruption()
                except Exception:
                    pass
                return False

            self._connect_worker_delete_later(worker)

            # 協調的停止
            try:
                if hasattr(worker, "cancel"):
                    worker.cancel()
            except Exception:
                pass
            try:
                worker.requestInterruption()
            except Exception:
                pass

            if self.is_worker_running(worker):
                if worker.wait(int(timeout_ms)):
                    return True
                if allow_terminate:
                    print(f"Force terminating {worker_name}...")
                    worker.terminate()
                    # terminate後は待機しない（デッドロック/自己wait回避）
                    return not self.is_worker_running(worker)
                return False

            return True
        except Exception as e:
            print(f"[WARNING] stop_worker failed for {worker_name}: {e}")
            return False

    def run_simulation(self):
        coords, mode = self.get_simulation_coords()
        if coords is None:
            QMessageBox.warning(self, "Error", "PDBまたはMRCデータがありません。")
            return
        # 以降、coordsを使ってシミュレーション
        # mode == 'mrc' ならMRC、'pdb' ならPDB
        # 既存のrun_simulationの処理のうち、self.get_rotated_atom_coords()の代わりにcoordsを使うように修正
        self.simulate_btn.setText("Cancel")
        try:
            self.simulate_btn.clicked.disconnect(self.run_simulation)
        except TypeError:
            pass
        self.simulate_btn.clicked.connect(self.cancel_simulation)
        self.progress_container.setVisible(True)

        base_coords = coords
        if base_coords is None:
            QMessageBox.critical(self, "Error", "Could not get atom coordinates.")
            self.on_simulation_finished(None)
            return

        # UIから共通パラメータを取得
        sim_params = {
            'scan_x_nm': self.spinScanXNm.value(),
            'scan_y_nm': self.spinScanYNm.value(),
            'nx': self.spinNx.value(),
            'ny': self.spinNy.value(),
            'center_x': self.tip_x_slider.value() / 5.0,
            'center_y': self.tip_y_slider.value() / 5.0,
            'tip_radius': self.tip_radius_spin.value(),
            'minitip_radius': self.minitip_radius_spin.value(),
            'tip_angle': self.tip_angle_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'use_vdw': self.use_vdw_check.isChecked()
        }
        # Backward compatibility for worker (if not yet updated)
        sim_params['scan_size'] = sim_params['scan_x_nm']
        sim_params['resolution'] = sim_params['nx']

        # --- チェックされた全ての面の計算タスクを作成 ---
        tasks = []
        if self.afm_x_check.isChecked():
            tasks.append({
                "name": "XY",
                "panel": self.afm_x_frame,
                "coords": base_coords
            })
        if self.afm_y_check.isChecked():
            x_scan = base_coords[:, 1]
            y_scan = base_coords[:, 2]
            z_scan = -base_coords[:, 0]
            tasks.append({
                "name": "YZ",
                "panel": self.afm_y_frame,
                "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)
            })
        if self.afm_z_check.isChecked():
            x_scan, y_scan, z_scan = base_coords[:, 0], base_coords[:, 2], -base_coords[:, 1]
            tasks.append({
                "name": "ZX",
                "panel": self.afm_z_frame,
                "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)
            })

        if not tasks:
            self.on_simulation_finished(None)
            return

        # 既存のワーカーを停止
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker'):
            stopped = self.stop_worker(self.sim_worker, timeout_ms=300, allow_terminate=False, worker_name="sim_worker")
            # 停止できない場合は、実行中スレッドの寿命を切らないよう新規起動を見送る
            if not stopped:
                print("[INFO] sim_worker still running; skipping new simulation start.")
                return
        
        self.sim_worker = AFMSimulationWorker(
            self, sim_params, tasks,
            self.atoms_data['element'] if sim_params['use_vdw'] and self.atoms_data is not None else None,
            self.vdw_radii if sim_params['use_vdw'] and hasattr(self, 'vdw_radii') else None
        )
        self._connect_worker_delete_later(self.sim_worker)
        self._track_worker_ref('sim_worker', self.sim_worker)

        self.simulation_results.clear()
        self.save_image_button.setEnabled(False)
        self.save_asd_button.setEnabled(False)

        # ★★★ 削除：ステータス更新接続を無効化 ★★★
        # self.sim_worker.status_update.connect(self.status_label.setText)
        self.sim_worker.progress.connect(self.progress_bar.setValue)
        self.sim_worker.task_done.connect(self.on_task_finished)
        self.sim_worker.done.connect(self.on_simulation_finished)
        self.sim_worker.start()

    def cancel_simulation(self):
        """シミュレーションのキャンセルを要求する"""
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker'):
            print("Cancel request sent.")
            self.status_label.setText("Cancelling...")
            self.sim_worker.cancel()
    
    def show_afm_result(self, z_map):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        if np.all(np.isnan(z_map)):
            QMessageBox.warning(self, "AFM Result", "No collisions detected.")
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(z_map, cmap='viridis', origin='lower', 
                    interpolation='nearest',
                    extent=[-0.5, 0.5, -0.5, 0.5])  # 正規化不要なら適宜修正
        ax.set_title("Simulated AFM Topography")
        plt.colorbar(im, ax=ax, label="Height [nm]")
        plt.tight_layout()
        plt.show()
        
    def _simulation_worker(self):
        """シミュレーションワーカー（デバッグ用レガシーコード）"""
        # UIからパラメータを取得
        scan_x = self.spinScanXNm.value()
        scan_y = self.spinScanYNm.value()
        nx = self.spinNx.value()
        ny = self.spinNy.value()
        
        # スキャン範囲を計算
        x_coords = np.linspace(-scan_x/2.0, scan_x/2.0, nx)
        y_coords = np.linspace(-scan_y/2.0, scan_y/2.0, ny)
        
        height_map = np.zeros((ny, nx))

        # 衝突判定用の原子データを準備
        atom_x = self.atoms_data['x']
        atom_y = self.atoms_data['y']
        atom_z = self.atoms_data['z']
        atom_elem = self.atoms_data['element']
        atom_radii = np.array([self.vdw_radii.get(e, self.vdw_radii['other']) for e in atom_elem])

        total_steps = resolution * resolution
        current_step = 0
        
        # ★追加: 分子の統計情報を表示
        mol_center_x = np.mean(atom_x)
        mol_center_y = np.mean(atom_y)
        mol_center_z = np.mean(atom_z)
        mol_size_x = np.max(atom_x) - np.min(atom_x)
        mol_size_y = np.max(atom_y) - np.min(atom_y)
        mol_size_z = np.max(atom_z) - np.min(atom_z)
        
        print(f"=== AFM Simulation Started (FIXED v2) ===")
        print(f"Scan size: {scan_size}nm, Resolution: {resolution}x{resolution}")
        print(f"Total atoms: {len(atom_x)}")
        print(f"Molecule center: ({mol_center_x:.2f}, {mol_center_y:.2f}, {mol_center_z:.2f})nm")
        print(f"Molecule size: {mol_size_x:.2f} x {mol_size_y:.2f} x {mol_size_z:.2f}nm")
        print(f"Z range: {np.min(atom_z):.2f} to {np.max(atom_z):.2f}nm")
        print(f"Tip: {self.tip_shape_combo.currentText()}, R={self.tip_radius_spin.value()}nm")
        print(f"Scan range: {-half_size:.1f} to {+half_size:.1f}nm")

        # スキャンループ
        debug_count = 0
        for iy, y in enumerate(y_coords):
            for ix, x in enumerate(x_coords):
                if self.progress_dialog.wasCanceled():
                    print("Simulation canceled by user.")
                    self.simulation_done.emit(None)
                    return

                # 衝突高さ計算
                z_height = self.find_collision_height(x, y, atom_x, atom_y, atom_z, atom_radii)
                height_map[iy, ix] = z_height
                
                # ★改良: より多様な位置でデバッグ出力
                if debug_count < 10:  # 最初の10点
                    print(f"Point ({x:6.2f}, {y:6.2f}) -> Z={z_height:8.3f}nm")
                    debug_count += 1
                elif (iy == resolution//2 and ix == resolution//2):  # 中心点
                    print(f"Center ({x:6.2f}, {y:6.2f}) -> Z={z_height:8.3f}nm")
                elif (iy == resolution-1 and ix == resolution-1):  # 最後の点
                    print(f"End    ({x:6.2f}, {y:6.2f}) -> Z={z_height:8.3f}nm")

                current_step += 1
                progress = int((current_step / total_steps) * 100)
                self.simulation_progress.emit(progress)

        # ★追加: 詳細な統計情報
        valid_heights = height_map[height_map > mol_center_z - 10]  # 明らかに低すぎる値を除外
        
        print(f"=== Simulation Completed ===")
        print(f"Height range: {np.min(height_map):.3f} to {np.max(height_map):.3f}nm")
        print(f"Valid heights: {np.min(valid_heights):.3f} to {np.max(valid_heights):.3f}nm")
        print(f"Mean height: {np.mean(valid_heights):.3f}nm")
        print(f"Height std: {np.std(valid_heights):.3f}nm")
        
        # 完了シグナルを送信
        self.simulation_done.emit(height_map)

    def check_tip_position_and_molecule_overlap(self):
        """探針位置と分子の位置関係を確認するデバッグメソッド"""
        if self.atoms_data is None:
            print("No molecule loaded")
            return
        
        # 現在の探針位置を取得
        tip_x = self.afm_params['tip_x']
        tip_y = self.afm_params['tip_y'] 
        tip_z = self.afm_params['tip_z']
        
        # 分子の統計
        mol_x_range = (np.min(self.atoms_data['x']), np.max(self.atoms_data['x']))
        mol_y_range = (np.min(self.atoms_data['y']), np.max(self.atoms_data['y']))
        mol_z_range = (np.min(self.atoms_data['z']), np.max(self.atoms_data['z']))
        
        print(f"\n=== Position Check ===")
        print(f"Tip position: ({tip_x:.2f}, {tip_y:.2f}, {tip_z:.2f})nm")
        print(f"Molecule X range: {mol_x_range[0]:.2f} to {mol_x_range[1]:.2f}nm")
        print(f"Molecule Y range: {mol_y_range[0]:.2f} to {mol_y_range[1]:.2f}nm") 
        print(f"Molecule Z range: {mol_z_range[0]:.2f} to {mol_z_range[1]:.2f}nm")
        
        # 探針が分子の上にあるかチェック
        tip_over_molecule = (mol_x_range[0] <= tip_x <= mol_x_range[1] and 
                            mol_y_range[0] <= tip_y <= mol_y_range[1])
        
        print(f"Tip over molecule: {tip_over_molecule}")
        
        if tip_z <= mol_z_range[1]:
            print(f"WARNING: Tip Z position ({tip_z:.2f}) is too low! Molecule top is at {mol_z_range[1]:.2f}nm")


    def create_tip_footprint(self, R, alpha_deg, pixel_size):
        """Dilation演算に使うための、探針の2Dフットプリントを生成する"""
        # 探針の影響範囲をピクセル単位で計算
        tip_pixel_radius = int(np.ceil(R * 3 / pixel_size))
        size = 2 * tip_pixel_radius + 1
        footprint = np.zeros((size, size))
        
        center = tip_pixel_radius
        alpha = np.radians(alpha_deg)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        for iy in range(size):
            for ix in range(size):
                # ピクセル中心からの物理的な距離
                r_2d = np.sqrt(((ix - center) * pixel_size)**2 + ((iy - center) * pixel_size)**2)
                
                # 探針の高さを計算 (反転させた形状)
                r_crit = R * ca
                if r_2d <= r_crit:
                    z = R - np.sqrt(R**2 - r_2d**2)
                else:
                    z = (r_2d * sa + R * (1 - ca)) / ca # 修正された円錐式

                footprint[iy, ix] = -z # Dilationでは反転した探針を使う
        
        return footprint
    
    def on_task_finished(self, z_map, target_panel):
        """個別の計算タスクが完了した際に呼び出されるスロット"""
        if z_map is not None and target_panel is not None:
            image_key = target_panel.objectName()
            
            # ★★★ 修正箇所: 生データを保存し、表示更新関数を呼び出す ★★★
            # 1. フィルターをかける前の「生」データを保存
            self.raw_simulation_results[image_key] = z_map
            
            # 2. フィルター適用と表示更新を行う関数を呼び出す
            self.process_and_display_single_image(image_key)
    
    def process_and_display_single_image(self, image_key):
        """指定されたキーの画像を処理して表示する"""
        if image_key not in self.raw_simulation_results:
            return

        raw_data = self.raw_simulation_results[image_key]
        ny, nx = raw_data.shape
        
        scan_x = self.spinScanXNm.value()
        scan_y = self.spinScanYNm.value()
        
        # フィルターが有効かチェック
        if self.apply_filter_check.isChecked():
            cutoff_wl = self.filter_cutoff_spin.value()
            processed_data = apply_low_pass_filter(raw_data, scan_x, scan_y, cutoff_wl)
        else:
            processed_data = raw_data

        # 物理ノイズ/アーティファクト
        try:
            pixel_x_nm = scan_x / nx
            pixel_y_nm = scan_y / ny
            processed_data = self.apply_noise_artifacts(processed_data, pixel_x_nm, pixel_y_nm)
        except Exception as e:
            # print(f"Error applying noise: {e}")
            pass

        # 表示用と保存用のデータを更新
        self.simulation_results[image_key] = processed_data
        
        # 対応するパネルを見つけて表示を更新
        target_panel = self.findChild(QFrame, image_key)
        if target_panel:
            self.display_afm_image(processed_data, target_panel)

        # Keep Real AFM window "Sim Aligned" panel identical to current XY simulated image.
        if image_key == "XY_Frame":
            self.sim_aligned_nm = processed_data
            aligned_panel = getattr(self, 'real_afm_window_aligned_frame', None)
            if aligned_panel is not None:
                self.display_afm_image(processed_data, aligned_panel)

    
    def process_and_display_all_images(self):
        """現在表示されている全ての画像を再処理・再表示する"""
        #print("Filter settings changed, updating all views...")
        for image_key in self.raw_simulation_results.keys():
            self.process_and_display_single_image(image_key)

    def start_filter_update_timer(self):
        """フィルターのカットオフ値変更時にタイマーで更新を遅延させる"""
        if not self.apply_filter_check.isChecked():
            return # フィルターがOFFの時は何もしない
            
        if not hasattr(self, 'filter_update_timer'):
            self.filter_update_timer = QTimer(self)  # 親ウィンドウを設定
            self.filter_update_timer.setSingleShot(True)
            self.filter_update_timer.timeout.connect(self.process_and_display_all_images)
        
        self.filter_update_timer.start(500) # 500ミリ秒後に更新

    def on_simulation_finished(self, result):
        """
        シミュレーションの完了・失敗・キャンセル後の全てのクリーンアップ処理を担当します。
        このメソッドは、バックグラウンドのスレッドが終了した際に一度だけ呼び出されます。
        """
        # 1. ボタンを「Run」状態に戻し、再度クリックできるようにする
        self.simulate_btn.setText("Run AFM Simulation")
        try:
            self.simulate_btn.clicked.disconnect(self.cancel_simulation)
        except TypeError:
            pass  # すでに接続が解除されている場合は何もしない
        self.simulate_btn.clicked.connect(self.run_simulation)
        self.simulate_btn.setEnabled(True)

        # 2. プログレス表示用のコンテナを非表示にする
        self.progress_container.setVisible(False)

        # 3. シミュレーション結果が一つでもあれば、各種保存ボタンを有効化する
        if self.simulation_results:
            #print("Simulation finished. Enabling save buttons.")
            self.save_image_button.setEnabled(True)
            self.save_asd_button.setEnabled(True)
        else:
            #print("Simulation finished, but no results were generated (or it was cancelled).")
            pass

        


    
    def display_afm_image(self, height_map, target_panel):
        """
        計算された高さマップをグレイスケールでUIに表示します。
        """
        if target_panel is None or height_map is None: return
        
        import matplotlib.cm as cm
        from PyQt5.QtGui import QImage, QPixmap
        
        # --- 正規化処理 ---
        valid_pixels = height_map[height_map > -1e8]
        if valid_pixels.size < 2:
            image_data = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8)
        else:
            min_h, max_h = np.min(valid_pixels), np.max(valid_pixels)
            if max_h <= min_h:
                image_data = np.full((height_map.shape[0], height_map.shape[1], 3), 128, dtype=np.uint8)
            else:
                clipped_map = np.clip(height_map, min_h, max_h)
                norm_map = (clipped_map - min_h) / (max_h - min_h)
                image_data = (cm.gray(norm_map)[:, :, :3] * 255).astype(np.uint8)

        # ★★★ ここからが修正箇所 ★★★
        # 3Dビューの上下方向 (Y軸が上) と2D画像の表示 (Y軸が下) を合わせるため、
        # 画像データを上下反転させます。
        # NOTE: np.flipud returns a view with negative strides (non-contiguous).
        # QImage needs a contiguous buffer (and PyQt5 on some Python/numpy combos
        # doesn't accept a memoryview directly), so we force contiguity below.
        image_data_flipped = np.ascontiguousarray(np.flipud(image_data))
        # ★★★ 修正箇所ここまで ★★★

        height, width, channel = image_data_flipped.shape
        bytes_per_line = int(image_data_flipped.strides[0])

        # IMPORTANT:
        # - PyQt5's QImage constructors can reject numpy's memoryview on some versions.
        # - QImage(pointer, ...) does not own the buffer, so we .copy() to detach.
        try:
            import sip  # type: ignore
        except Exception:
            from PyQt5 import sip  # type: ignore
        ptr = sip.voidptr(int(image_data_flipped.ctypes.data))
        qimg = QImage(ptr, width, height, bytes_per_line, QImage.Format_RGB888).copy()

        # Update existing view widget (avoid re-creating widgets to prevent layout jitter).
        container = target_panel.findChild(QWidget, "afm_image_container")
        if container is None:
            container = target_panel

        view = container.findChild(_AspectPixmapView, "afm_image_view")
        placeholder = container.findChild(QLabel, "afm_placeholder")
        layout = container.layout()

        if view is None:
            # Backward-compatible fallback for panels created before this viewer existed.
            view = _AspectPixmapView(container)
            view.setObjectName("afm_image_view")
            if layout is None:
                layout = QStackedLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
            try:
                layout.addWidget(view)
            except Exception:
                pass

        pixmap = QPixmap.fromImage(qimg)
        view.setSourcePixmap(pixmap)
        # Match the display box to the physical scan aspect so molecular shapes are not distorted.
        try:
            panel_name = target_panel.objectName() if hasattr(target_panel, "objectName") else ""
        except Exception:
            panel_name = ""
        aspect = None
        try:
            if panel_name in ("REAL_AFM_Frame", "SIM_ALIGNED_Frame") and getattr(self, "real_meta", None):
                sx = float(self.real_meta.get("scan_x_nm", 0.0))
                sy = float(self.real_meta.get("scan_y_nm", 0.0))
            else:
                sx = float(self.spinScanXNm.value()) if hasattr(self, "spinScanXNm") else 0.0
                sy = float(self.spinScanYNm.value()) if hasattr(self, "spinScanYNm") else 0.0
            if sx > 0 and sy > 0:
                aspect = sx / sy
        except Exception:
            aspect = None
        if aspect is not None:
            view.setDisplayAspectRatio(aspect)
        else:
            view.setDisplayAspectRatio(None)

        # Switch from placeholder -> view if using a stacked layout
        if layout is not None and hasattr(layout, "setCurrentWidget"):
            try:
                layout.setCurrentWidget(view)
            except Exception:
                pass
        else:
            # If not stacked, just hide placeholder when we have an image.
            if placeholder is not None:
                try:
                    placeholder.setVisible(False)
                except Exception:
                    pass
            try:
                view.setVisible(True)
            except Exception:
                pass

   

    def find_collision_height(self, x, y, atom_x, atom_y, atom_z, atom_radii):
        """VTKで作成されたtip_actorと分子との衝突Z高さを返す"""

        # tip_actor から vtkPolyData を取得
        polydata = self.tip_actor.GetMapper().GetInput()
        if polydata is None:
            print("[WARNING] tip geometry is not defined.")
            return None

        points = polydata.GetPoints()
        n_points = points.GetNumberOfPoints()
        if n_points == 0:
            print("[WARNING] tip geometry has no points.")
            return None

        # tip の座標を (x, y) に移動（tip作成時は原点を中心と仮定）
        transformed_tip_points = []
        for i in range(n_points):
            px, py, pz = points.GetPoint(i)
            transformed_tip_points.append([px + x, py + y, pz])

        transformed_tip_points = np.array(transformed_tip_points)

        # 各原子とtip点群の最近接距離を計算（高速化のためBallTreeなどを使うのが理想だがここでは総当り）
        min_collision_z = None
        for i in range(len(atom_x)):
            ax, ay, az = atom_x[i], atom_y[i], atom_z[i]
            ar = atom_radii[i]

            for tp in transformed_tip_points:
                dx = tp[0] - ax
                dy = tp[1] - ay
                dz = tp[2] - az
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                if dist <= ar:
                    if (min_collision_z is None) or (tp[2] < min_collision_z):
                        min_collision_z = tp[2]

        return min_collision_z

    def is_colliding(self, tip_x, tip_y, tip_z, atom_x, atom_y, atom_z, atom_radii):
        """探針と原子群の衝突判定（修正版）"""
        tip_shape = self.tip_shape_combo.currentText().lower()
        tip_radius = self.tip_radius_spin.value()
        tip_angle = self.tip_angle_spin.value()
        minitip_radius = self.minitip_radius_spin.value()

        # 各原子について衝突をチェック
        for i in range(len(atom_x)):
            atom_pos = (atom_x[i], atom_y[i], atom_z[i])
            tip_apex = (tip_x, tip_y, tip_z)
            
            # 探針表面から原子中心までの距離を計算
            if tip_shape == "cone":
                dist_surface = self.dist_point_to_cone_tip(
                    atom_pos, tip_apex, tip_radius, tip_angle)
            elif tip_shape == "sphere":
                dist_surface = self.dist_point_to_sphere_tip(
                    atom_pos, tip_apex, tip_radius, tip_angle, minitip_radius)
            else:  # Paraboloid
                dist_surface = self.dist_point_to_paraboloid_tip(
                    atom_pos, tip_apex, tip_radius)
            
            # 衝突判定：探針表面から原子中心までの距離が原子半径以下なら衝突
            if dist_surface <= atom_radii[i]:
                return True
                
        return False

    def dist_point_to_cone_tip(self, p, tip_apex, R, alpha_deg):
        """点pと円錐探針表面との最短距離（修正版）"""
        alpha = np.radians(alpha_deg)
        px, py, pz = p
        tx, ty, tz = tip_apex
        
        # 探針の先端（apex）を原点とした相対座標
        dx, dy, dz = px - tx, py - ty, pz - tz
        r_2d = np.sqrt(dx**2 + dy**2)
        
        # 修正1: 円錐の幾何学を正確に計算
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # 球状先端部分の中心位置を修正
        sphere_center_z = R  # 球の中心は先端からR上方
        
        # 球との境界半径を正確に計算
        r_crit = R * sin_alpha  # 球と円錐の接続部の半径
        
        # 修正2: 距離計算を改善
        if r_2d <= r_crit and dz <= sphere_center_z:
            # 球状部分との距離
            dist_to_sphere_center = np.sqrt(r_2d**2 + (dz - sphere_center_z)**2)
            dist_surface = dist_to_sphere_center - R
        else:
            # 円錐部分との距離を正確に計算
            # 円錐の母線方向の単位ベクトル：(sin_alpha, 0, cos_alpha)
            # 点から円錐軸（Z軸）への垂直距離：r_2d
            # 点のZ座標から適切な円錐面までの距離を計算
            
            # 円錐面上の対応点のZ座標
            z_on_cone = sphere_center_z + (r_2d - r_crit) / np.tan(alpha)
            
            # 修正3: 符号付き距離を正確に計算
            # 円錐面の法線ベクトル：(-sin_alpha, 0, cos_alpha)
            # 点から円錐面への符号付き距離
            dist_surface = (r_2d - r_crit) * cos_alpha + (dz - z_on_cone) * sin_alpha
            
        return dist_surface

    def dist_point_to_sphere_tip(self, p, tip_apex, R_cone, alpha_deg, R_sphere):
        """点pと球+円錐の複合探針表面との最短距離"""
        # この実装では、先端球が支配的として簡易計算
        return self.dist_point_to_cone_tip(p, tip_apex, R_sphere, 90)

    def dist_point_to_paraboloid_tip(self, p, tip_apex, R):
        """点pと放物面探針表面との最短距離"""
        px, py, pz = p
        tx, ty, tz = tip_apex
        # 座標変換
        dx, dy, dz = px - tx, py - ty, pz - tz
        r_sq = dx**2 + dy**2
        # 放物面上の対応する高さ
        z_parabola = r_sq / (2 * R)
        return dz - z_parabola




    def choose_background_color(self):
        """背景色選択ダイアログ"""
        # 現在の背景色を取得
        current_color = QColor()
        current_color.setRgbF(self.current_bg_color[0], 
                             self.current_bg_color[1], 
                             self.current_bg_color[2])
        
        color = QColorDialog.getColor(current_color, self, "Choose Background Color")
        if color.isValid():
            # RGB値を0-1範囲に変換
            self.current_bg_color = (color.redF(), color.greenF(), color.blueF())
            self._apply_current_background_color()
    
    def clear_mrc_data(self):
        """MRCデータとアクターをクリア"""
        # MRCアクターをレンダラーから削除
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            self.renderer.RemoveActor(self.mrc_actor)
            self.mrc_actor = None
        
        # MRCデータをクリア
        if hasattr(self, 'mrc_data'):
            self.mrc_data = None
        if hasattr(self, 'mrc_data_original'):
            self.mrc_data_original = None
        if hasattr(self, 'mrc_metadata'):
            self.mrc_metadata = None
        if hasattr(self, 'mrc_name'):
            self.mrc_name = None
            self.mrc_id = ""
        if hasattr(self, 'mrc_surface_coords'):
            self.mrc_surface_coords = None
        if self.current_structure_type == "mrc":
            self.current_structure_path = None
            self.current_structure_type = None
        
        # MRCラベルをリセット
        if hasattr(self, 'file_label'):
            self.file_label.setText("File Name: (none)")
        
        # MRCグループを無効化
        if hasattr(self, 'mrc_group'):
            self.mrc_group.setEnabled(False)
        
        # 回転ウィジェットも無効化（PDBデータがない場合）
        if not hasattr(self, 'atoms_data') or self.atoms_data is None:
            if hasattr(self, 'rotation_widgets'):
                for axis in ['X', 'Y', 'Z']:
                    self.rotation_widgets[axis]['spin'].setEnabled(False)
                    self.rotation_widgets[axis]['slider'].setEnabled(False)
        
        # レンダリング更新
        self.request_render()
    
    def clear_pdb_data(self):
        """PDBデータとアクターをクリア"""
        # PDBアクターをレンダラーから削除
        if hasattr(self, 'sample_actor') and self.sample_actor is not None:
            self.renderer.RemoveActor(self.sample_actor)
            self.sample_actor = None
        if hasattr(self, 'bonds_actor') and self.bonds_actor is not None:
            self.renderer.RemoveActor(self.bonds_actor)
            self.bonds_actor = None
        
        # PDBデータをクリア
        if hasattr(self, 'atoms_data'):
            self.atoms_data = None
        if hasattr(self, 'pdb_name'):
            self.pdb_name = None
            self.pdb_id = ""
        if hasattr(self, 'cif_name'):
            self.cif_name = None
            self.cif_id = ""
        self.current_structure_path = None
        self.current_structure_type = None
        self.pymol_esp_object = None
        
        # PDBラベルをリセット
        if hasattr(self, 'file_label'):
            self.file_label.setText("File Name: (none)")
        
        # 統計情報をリセット
        if hasattr(self, 'stats_label'):
            self.stats_label.setText("No data loaded")
        
        # 回転ウィジェットも無効化（MRCデータがない場合）
        if not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            if hasattr(self, 'rotation_widgets'):
                for axis in ['X', 'Y', 'Z']:
                    self.rotation_widgets[axis]['spin'].setEnabled(False)
                    self.rotation_widgets[axis]['slider'].setEnabled(False)
        
        # PyMOLオブジェクトを削除
        if self._is_pymol_active():
            try:
                self.pymol_cmd.delete(self.pymol_object_name)
            except Exception:
                pass
            self.pymol_loaded_path = None

        # レンダリング更新
        self.request_render()
    
    def update_mrc_actor_color(self):
        """既存のMRCアクターの色を更新"""
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            # マッパーのスカラー可視性を無効にする
            mapper = self.mrc_actor.GetMapper()
            if mapper:
                mapper.ScalarVisibilityOff()
            
            prop = self.mrc_actor.GetProperty()
            # MRCは常に選択された色を使用（カラースキームは関係ない）
           
            prop.SetColor(self.current_single_color[0], self.current_single_color[1], self.current_single_color[2])
            self.request_render()
    
    def on_color_scheme_changed(self):
        """カラースキーム変更時の処理"""
        print(f"Color scheme changed to: {self.color_combo.currentText()}")
        if self.atoms_data is not None:
            self.update_display()
        # MRCデータの場合はカラースキームは関係ないので何もしない
    
    def choose_single_color(self):
        """単色モード用カラー選択"""
        # 現在の単色を取得
        current_color = QColor()
        current_color.setRgbF(self.current_single_color[0],
                            self.current_single_color[1],
                            self.current_single_color[2])
        
        color = QColorDialog.getColor(current_color, self, "Choose Single Color")
        if color.isValid():
            # RGB値を0-1範囲に変換
            old_color = self.current_single_color
            self.current_single_color = (color.redF(), color.greenF(), color.blueF())
            
           
            # ボタンの色を更新
            self.single_color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                    color: {'black' if sum([color.red(), color.green(), color.blue()]) > 400 else 'white'};
                    border: 2px solid #555;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    border-color: #777;
                }}
            """)
            
            # 表示を更新
            if self.atoms_data is not None:                
                self.update_display()
            elif hasattr(self, 'mrc_data') and self.mrc_data is not None:             # MRCデータの場合も色を更新
                self.update_mrc_actor_color()
    
    def update_brightness(self):
        """明るさ調整"""
        brightness = self.brightness_slider.value()
        self.brightness_factor = brightness / 100.0
        self.brightness_label.setText(f"{brightness}%")
        
        # ライティングを更新
        if self._is_pymol_active():
            self._apply_pymol_lighting()
        if not self._is_pymol_only():
            self.update_lighting_intensity()
        self.request_render()
    
    def update_lighting(self):
        """環境光設定の更新"""
        ambient = self.ambient_slider.value()
        self.ambient_label.setText(f"{ambient}%")
        
        # レンダラーの環境光を設定
        if self._is_pymol_active():
            self._apply_pymol_lighting()
        if not self._is_pymol_only():
            ambient_factor = ambient / 100.0
            self.renderer.SetAmbient(ambient_factor, ambient_factor, ambient_factor)
        self.request_render()
    
    def update_material(self):
        """マテリアル設定の更新"""
        specular = self.specular_slider.value()
        self.specular_label.setText(f"{specular}%")
        
        # 全てのアクターのスペキュラを更新
        if self._is_pymol_active():
            self._apply_pymol_lighting()
        if not self._is_pymol_only():
            self.update_actor_materials()
        self.request_render()
    
    def update_lighting_intensity(self):
        """ライトの強度を明るさファクターで調整"""
        lights = self.renderer.GetLights()
        lights.InitTraversal()
        
        light = lights.GetNextItem()
        while light:
            # 元の強度に明るさファクターを適用
            if hasattr(light, '_original_intensity'):
                light.SetIntensity(light._original_intensity * self.brightness_factor)
            else:
                # 初回は現在の強度を保存
                light._original_intensity = light.GetIntensity()
                light.SetIntensity(light._original_intensity * self.brightness_factor)
            
            light = lights.GetNextItem()
    
    def update_actor_materials(self):
        """全アクターのマテリアル特性を更新"""
        specular_factor = self.specular_slider.value() / 100.0
        
        # 分子アクター
        if self.sample_actor and hasattr(self.sample_actor, 'GetProperty'):
            self.sample_actor.GetProperty().SetSpecular(specular_factor)
            self.sample_actor.GetProperty().SetSpecularPower(50)
        
        # 結合アクター
        if self.bonds_actor and hasattr(self.bonds_actor, 'GetProperty'):
            self.bonds_actor.GetProperty().SetSpecular(specular_factor * 0.5)
        
        # 探針アクター
        if self.tip_actor and hasattr(self.tip_actor, 'GetProperty'):
            self.tip_actor.GetProperty().SetSpecular(min(0.9, specular_factor * 1.5))
    
    def apply_pymol_style(self):
        """PyMOLスタイルプリセット適用"""
        # 背景を黒に
        self.current_bg_color = (0.0, 0.0, 0.0)
        if self.pymol_available and self.pymol_cmd is not None:
            self._pymol_set_background(self.current_bg_color)
        if not self._is_pymol_only():
            self.renderer.SetBackground(*self.current_bg_color)
        
        # ボタンの色を更新
        self.bg_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #000000;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                border-color: #777;
            }
        """)
        
        # 明るさを120%に
        self.brightness_slider.setValue(120)
        self.brightness_factor = 1.2
        self.brightness_label.setText("120%")
        
        # 環境光を5%に
        self.ambient_slider.setValue(5)
        self.ambient_label.setText("5%")
        if not self._is_pymol_only():
            self.renderer.SetAmbient(0.05, 0.05, 0.05)
        
        # スペキュラを80%に
        self.specular_slider.setValue(80)
        self.specular_label.setText("80%")
        
        # 設定を適用
        if self._is_pymol_active():
            self._apply_pymol_lighting()
        if not self._is_pymol_only():
            self.update_lighting_intensity()
            self.update_actor_materials()
        
        # PyMOLライクな元素カラーに変更
        self.element_colors.update({
            'C': (0.565, 0.565, 0.565),  # PyMOLのカーボングレー
            'O': (1.0, 0.051, 0.051),    # 鮮やかな赤
            'N': (0.188, 0.314, 0.973),  # 鮮やかな青
            'H': (0.9, 0.9, 0.9),        # 白
            'S': (1.0, 1.0, 0.188),      # 鮮やかな黄色
            'P': (1.0, 0.502, 0.0),      # オレンジ
        })
        
        # 表示を更新
        if self.atoms_data is not None:
            self.update_display()

        self.request_render()
        
        QMessageBox.information(self, "Style Applied", "PyMOL style applied successfully!")
    
    def apply_dark_theme(self):
        """ダークテーマプリセット適用"""
        # 背景をダークグレーに
        self.current_bg_color = (0.1, 0.1, 0.15)
        if self.pymol_available and self.pymol_cmd is not None:
            self._pymol_set_background(self.current_bg_color)
        if not self._is_pymol_only():
            self.renderer.SetBackground(*self.current_bg_color)
        
        # ボタンの色を更新
        self.bg_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #191926;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                border-color: #777;
            }
        """)
        
        # 明るさを100%に
        self.brightness_slider.setValue(100)
        self.brightness_factor = 1.0
        self.brightness_label.setText("100%")
        
        # 環境光を15%に
        self.ambient_slider.setValue(15)
        self.ambient_label.setText("15%")
        if not self._is_pymol_only():
            self.renderer.SetAmbient(0.15, 0.15, 0.15)
        
        # スペキュラを60%に
        self.specular_slider.setValue(60)
        self.specular_label.setText("60%")
        
        # 設定を適用
        if self._is_pymol_active():
            self._apply_pymol_lighting()
        if not self._is_pymol_only():
            self.update_lighting_intensity()
            self.update_actor_materials()
        
        # 表示を更新
        if self.atoms_data is not None:
            self.update_display()

        self.request_render()
        
        QMessageBox.information(self, "Style Applied", "Dark theme applied successfully!")
    
    def load_settings(self):
        """起動時にウィンドウの位置、サイズ、スプリッターの状態を復元する"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                
                # ウィンドウのジオメトリ（位置とサイズ）を復元
                if 'geometry' in settings:
                    self.setGeometry(*settings['geometry'])
                
                # 各スプリッターの状態を復元
                if 'main_splitter' in settings:
                    self.main_splitter.setSizes(settings['main_splitter'])
                if 'afm_splitter' in settings:
                    self.afm_splitter.setSizes(settings['afm_splitter'])
                if 'view_control_splitter' in settings:
                    self.view_control_splitter.setSizes(settings['view_control_splitter'])

                if 'last_import_dir' in settings:
                    self.last_import_dir = settings['last_import_dir']
                
                # MRCのZ軸フリップ状態を復元（デフォルトはTrue）
                if 'mrc_z_flip' in settings:
                    self.mrc_z_flip = settings['mrc_z_flip']
                else:
                    self.mrc_z_flip = True  # デフォルトで有効

                # 背景色を復元
                if 'bg_color' in settings and isinstance(settings['bg_color'], (list, tuple)) and len(settings['bg_color']) == 3:
                    try:
                        r, g, b = settings['bg_color']
                        self.current_bg_color = (float(r), float(g), float(b))
                    except Exception:
                        pass
                self._apply_current_background_color()
                
                # チェックボックスの状態を確実に設定
                if hasattr(self, 'mrc_z_flip_check'):
                    self.mrc_z_flip_check.blockSignals(True)  # シグナルを一時的にブロック
                    self.mrc_z_flip_check.setChecked(self.mrc_z_flip)
                    self.mrc_z_flip_check.blockSignals(False)  # シグナルを再有効化
                
                #print("Settings loaded successfully.")

        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Could not load settings: {e}")
            # エラー発生時はデフォルトで起動

    def save_settings(self):
        """ウィンドウの位置、サイズ、スプリッターの状態を保存する"""
        settings = {
            'geometry': self.geometry().getRect(),
            'main_splitter': self.main_splitter.sizes(),
            'afm_splitter': self.afm_splitter.sizes(),
            'view_control_splitter': self.view_control_splitter.sizes(),
            'last_import_dir': self.last_import_dir,
            'mrc_z_flip': self.mrc_z_flip,
            'bg_color': list(self.current_bg_color),
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            #print("Settings saved successfully.")
        except IOError as e:
            print(f"Could not save settings: {e}")

    def collect_sim_params(self):
        """Collect all simulation parameters from UI into a dict."""
        params = {}

        # Tip settings
        tip = {}
        if hasattr(self, 'tip_shape_combo'):
            tip['shape'] = self.tip_shape_combo.currentText()
        if hasattr(self, 'tip_radius_spin'):
            tip['radius_nm'] = float(self.tip_radius_spin.value())
        if hasattr(self, 'minitip_radius_spin'):
            tip['minitip_radius_nm'] = float(self.minitip_radius_spin.value())
        if hasattr(self, 'tip_angle_spin'):
            tip['angle_deg'] = float(self.tip_angle_spin.value())
        if tip:
            params['tip'] = tip

        # Tip position
        tip_pos = {}
        if hasattr(self, 'tip_x_slider'):
            tip_pos['x_nm'] = float(self.tip_x_slider.value()) / 5.0
        if hasattr(self, 'tip_y_slider'):
            tip_pos['y_nm'] = float(self.tip_y_slider.value()) / 5.0
        if hasattr(self, 'tip_z_slider'):
            tip_pos['z_nm'] = float(self.tip_z_slider.value()) / 5.0
        if tip_pos:
            params['tip_position'] = tip_pos

        # Scan / image
        scan = {}
        if hasattr(self, 'spinScanXNm'):
            scan['scan_x_nm'] = float(self.spinScanXNm.value())
        if hasattr(self, 'spinScanYNm'):
            scan['scan_y_nm'] = float(self.spinScanYNm.value())
        if hasattr(self, 'spinNx'):
            scan['nx'] = int(self.spinNx.value())
        if hasattr(self, 'spinNy'):
            scan['ny'] = int(self.spinNy.value())
            
        # Optional backward compatibility in saved JSON
        if 'scan_x_nm' in scan:
            scan['size_nm'] = scan['scan_x_nm']
        if 'nx' in scan:
            scan['resolution'] = f"{scan['nx']}x{scan['nx']}"
            
        if scan:
            params['scan'] = scan

        # Simulation toggles
        if hasattr(self, 'use_vdw_check'):
            params['use_vdw'] = bool(self.use_vdw_check.isChecked())
        if hasattr(self, 'interactive_update_check'):
            params['interactive_update'] = bool(self.interactive_update_check.isChecked())
        if hasattr(self, 'rectangle_check'):
            params['lock_square'] = bool(self.rectangle_check.isChecked())
        if hasattr(self, 'apply_filter_check'):
            params['apply_filter'] = bool(self.apply_filter_check.isChecked())
            params['lowpass_enable'] = bool(self.apply_filter_check.isChecked())
        if hasattr(self, 'filter_cutoff_spin'):
            params['filter_cutoff_nm'] = float(self.filter_cutoff_spin.value())
            params['lowpass_cutoff_nm'] = float(self.filter_cutoff_spin.value())

        # Physical noise / artifacts
        if hasattr(self, 'chkNoiseEnable'):
            params['noise_enable'] = bool(self.chkNoiseEnable.isChecked())
        if hasattr(self, 'chkUseNoiseSeed'):
            params['noise_use_seed'] = bool(self.chkUseNoiseSeed.isChecked())
        if hasattr(self, 'spinNoiseSeed'):
            params['noise_seed'] = int(self.spinNoiseSeed.value())
        if hasattr(self, 'chkHeightNoise'):
            params['height_noise_enable'] = bool(self.chkHeightNoise.isChecked())
        if hasattr(self, 'spinHeightNoiseSigmaNm'):
            params['height_noise_sigma_nm'] = float(self.spinHeightNoiseSigmaNm.value())
        if hasattr(self, 'chkLineNoise'):
            params['line_noise_enable'] = bool(self.chkLineNoise.isChecked())
        if hasattr(self, 'spinLineNoiseSigmaNm'):
            params['line_noise_sigma_nm'] = float(self.spinLineNoiseSigmaNm.value())
        if hasattr(self, 'comboLineNoiseMode'):
            params['line_noise_mode'] = self.comboLineNoiseMode.currentText()
        if hasattr(self, 'chkDrift'):
            params['drift_enable'] = bool(self.chkDrift.isChecked())
        if hasattr(self, 'spinDriftVxNmPerLine'):
            params['drift_vx_nm_per_line'] = float(self.spinDriftVxNmPerLine.value())
        if hasattr(self, 'spinDriftVyNmPerLine'):
            params['drift_vy_nm_per_line'] = float(self.spinDriftVyNmPerLine.value())
        if hasattr(self, 'spinDriftJitterNmPerLine'):
            params['drift_jitter_nm_per_line'] = float(self.spinDriftJitterNmPerLine.value())
        if hasattr(self, 'chkFeedbackLag'):
            params['feedback_lag_enable'] = bool(self.chkFeedbackLag.isChecked())
        if hasattr(self, 'comboFeedbackMode'):
            params['feedback_mode'] = self.comboFeedbackMode.currentText()
        if hasattr(self, 'comboScanDirection'):
            params['scan_direction'] = self.comboScanDirection.currentData() or self.comboScanDirection.currentText()
        if hasattr(self, 'spinLagTauLines'):
            params['lag_tau_lines'] = float(self.spinLagTauLines.value())
        if hasattr(self, 'spinTapDropThresholdNm'):
            params['tap_drop_threshold_nm'] = float(self.spinTapDropThresholdNm.value())
        if hasattr(self, 'spinTapTauTrackLines'):
            params['tap_tau_track_lines'] = float(self.spinTapTauTrackLines.value())
        if hasattr(self, 'spinTapTauParachuteLines'):
            params['tap_tau_parachute_lines'] = float(self.spinTapTauParachuteLines.value())
        if hasattr(self, 'spinTapReleaseThresholdNm'):
            params['tap_release_threshold_nm'] = float(self.spinTapReleaseThresholdNm.value())

        # Display settings
        display = {}
        if hasattr(self, 'style_combo'):
            display['style'] = self.style_combo.currentText()
        if hasattr(self, 'color_combo'):
            display['color'] = self.color_combo.currentText()
        if hasattr(self, 'atom_combo'):
            display['atom_filter'] = self.atom_combo.currentText()
        if hasattr(self, 'size_slider'):
            display['size'] = int(self.size_slider.value())
        if hasattr(self, 'opacity_slider'):
            display['opacity'] = int(self.opacity_slider.value())
        if hasattr(self, 'quality_combo'):
            display['quality'] = self.quality_combo.currentText()
        if hasattr(self, 'renderer_combo'):
            display['renderer'] = self.renderer_combo.currentText()
        if hasattr(self, 'esp_check'):
            display['esp'] = bool(self.esp_check.isChecked())
        if display:
            params['display'] = display

        # View controls
        view = {}
        if hasattr(self, 'show_molecule_check'):
            view['show_molecule'] = bool(self.show_molecule_check.isChecked())
        if hasattr(self, 'show_tip_check'):
            view['show_tip'] = bool(self.show_tip_check.isChecked())
        if hasattr(self, 'show_bonds_check'):
            view['show_bonds'] = bool(self.show_bonds_check.isChecked())
        if view:
            params['view'] = view

        # AFM view selection
        afm_views = {}
        if hasattr(self, 'afm_x_check'):
            afm_views['xy'] = bool(self.afm_x_check.isChecked())
        if hasattr(self, 'afm_y_check'):
            afm_views['yz'] = bool(self.afm_y_check.isChecked())
        if hasattr(self, 'afm_z_check'):
            afm_views['zx'] = bool(self.afm_z_check.isChecked())
        if afm_views:
            params['afm_views'] = afm_views

        # Lighting / colors
        lighting = {}
        if hasattr(self, 'brightness_slider'):
            lighting['brightness'] = int(self.brightness_slider.value())
        if hasattr(self, 'ambient_slider'):
            lighting['ambient'] = int(self.ambient_slider.value())
        if hasattr(self, 'specular_slider'):
            lighting['specular'] = int(self.specular_slider.value())
        if lighting:
            params['lighting'] = lighting

        colors = {}
        if hasattr(self, 'current_bg_color'):
            colors['bg_color'] = list(self.current_bg_color)
        if hasattr(self, 'current_single_color'):
            colors['single_color'] = list(self.current_single_color)
        if colors:
            params['colors'] = colors

        # Rotation
        rot = {}
        if hasattr(self, 'rotation_widgets'):
            try:
                rot['x'] = float(self.rotation_widgets['X']['spin'].value())
                rot['y'] = float(self.rotation_widgets['Y']['spin'].value())
                rot['z'] = float(self.rotation_widgets['Z']['spin'].value())
            except Exception:
                pass
        if rot:
            params['rotation'] = rot

        # Find initial plane params
        fip = {}
        if hasattr(self, 'find_plane_params'):
            fp = self.find_plane_params
            if 'use_elec' in fp:
                fip['use_elec'] = bool(fp['use_elec'].isChecked())
            for key in ('pH', 'alpha_elec', 'r_pseudo', 'delta_elec'):
                if key in fp:
                    fip[key] = float(fp[key].value())
            for key in ('N_dir', 'K', 'K_geom', 'grid', 'surf_n'):
                if key in fp:
                    fip[key] = int(fp[key].value())
            for key in ('delta', 'lambda', 'theta1', 'theta2', 'theta3', 'surf_r'):
                if key in fp:
                    fip[key] = float(fp[key].value())
            if 'substrate' in fp:
                fip['substrate'] = fp['substrate'].currentText()
            if 'salt' in fp:
                fip['salt'] = fp['salt'].currentText()
            if 'salt_mM' in fp:
                fip['salt_mM'] = float(fp['salt_mM'].value())
        if fip:
            params['find_initial_plane'] = fip

        # MRC settings
        mrc = {}
        if hasattr(self, 'mrc_threshold_slider'):
            mrc['threshold'] = float(self.mrc_threshold_slider.value()) / 100.0
        elif hasattr(self, 'mrc_threshold'):
            mrc['threshold'] = float(self.mrc_threshold)
        if hasattr(self, 'mrc_z_flip'):
            mrc['z_flip'] = bool(self.mrc_z_flip)
        if mrc:
            params['mrc'] = mrc

        return params

    def apply_sim_params(self, params):
        """Apply parameters dict to UI and internal state."""
        if not isinstance(params, dict):
            return

        def _set_spin(widget, value):
            try:
                widget.blockSignals(True)
                widget.setValue(value)
            finally:
                widget.blockSignals(False)

        def _set_slider(widget, value):
            try:
                widget.blockSignals(True)
                widget.setValue(value)
            finally:
                widget.blockSignals(False)

        def _set_check(widget, value):
            try:
                widget.blockSignals(True)
                widget.setChecked(bool(value))
            finally:
                widget.blockSignals(False)

        def _set_combo(widget, text):
            try:
                idx = widget.findData(text)
                if idx < 0:
                    idx = widget.findText(str(text))
                if idx >= 0:
                    widget.blockSignals(True)
                    widget.setCurrentIndex(idx)
                    widget.blockSignals(False)
            except Exception:
                pass

        # Tip
        tip = params.get('tip', {})
        if tip and hasattr(self, 'tip_shape_combo'):
            _set_combo(self.tip_shape_combo, tip.get('shape', self.tip_shape_combo.currentText()))
        if tip and hasattr(self, 'tip_radius_spin') and 'radius_nm' in tip:
            _set_spin(self.tip_radius_spin, float(tip['radius_nm']))
        if tip and hasattr(self, 'minitip_radius_spin') and 'minitip_radius_nm' in tip:
            _set_spin(self.minitip_radius_spin, float(tip['minitip_radius_nm']))
        if tip and hasattr(self, 'tip_angle_spin') and 'angle_deg' in tip:
            _set_spin(self.tip_angle_spin, float(tip['angle_deg']))

        # Tip position
        tip_pos = params.get('tip_position', {})
        if tip_pos:
            if hasattr(self, 'tip_x_slider') and 'x_nm' in tip_pos:
                _set_slider(self.tip_x_slider, int(round(float(tip_pos['x_nm']) * 5.0)))
            if hasattr(self, 'tip_y_slider') and 'y_nm' in tip_pos:
                _set_slider(self.tip_y_slider, int(round(float(tip_pos['y_nm']) * 5.0)))
            if hasattr(self, 'tip_z_slider') and 'z_nm' in tip_pos:
                _set_slider(self.tip_z_slider, int(round(float(tip_pos['z_nm']) * 5.0)))

        # Scan
        scan = params.get('scan', {})
        if scan:
            # New rectangular parameters
            if hasattr(self, 'spinScanXNm') and 'scan_x_nm' in scan:
                _set_spin(self.spinScanXNm, float(scan['scan_x_nm']))
            elif hasattr(self, 'spinScanXNm') and 'size_nm' in scan: # Migration
                _set_spin(self.spinScanXNm, float(scan['size_nm']))
                
            if hasattr(self, 'spinScanYNm') and 'scan_y_nm' in scan:
                _set_spin(self.spinScanYNm, float(scan['scan_y_nm']))
            elif hasattr(self, 'spinScanYNm') and 'size_nm' in scan: # Migration
                _set_spin(self.spinScanYNm, float(scan['size_nm']))
                
            if hasattr(self, 'spinNx') and 'nx' in scan:
                _set_spin(self.spinNx, int(scan['nx']))
            elif hasattr(self, 'spinNx') and 'resolution' in scan: # Migration
                try:
                    res = int(str(scan['resolution']).split('x')[0])
                    _set_spin(self.spinNx, res)
                except: pass
                
            if hasattr(self, 'spinNy') and 'ny' in scan:
                _set_spin(self.spinNy, int(scan['ny']))
            elif hasattr(self, 'spinNy') and 'resolution' in scan: # Migration
                try:
                    res = int(str(scan['resolution']).split('x')[0])
                    _set_spin(self.spinNy, res)
                except: pass
                
            if hasattr(self, 'resolution_combo') and 'resolution' in scan:
                _set_combo(self.resolution_combo, scan['resolution'])

        # Toggles
        if 'use_vdw' in params and hasattr(self, 'use_vdw_check'):
            _set_check(self.use_vdw_check, params.get('use_vdw'))
        if 'interactive_update' in params and hasattr(self, 'interactive_update_check'):
            _set_check(self.interactive_update_check, params.get('interactive_update'))
        if 'lock_square' in params and hasattr(self, 'rectangle_check'):
            _set_check(self.rectangle_check, params.get('lock_square'))
        if hasattr(self, 'apply_filter_check'):
            if 'lowpass_enable' in params:
                _set_check(self.apply_filter_check, params.get('lowpass_enable'))
            elif 'apply_filter' in params:
                _set_check(self.apply_filter_check, params.get('apply_filter'))
        if hasattr(self, 'filter_cutoff_spin'):
            if 'lowpass_cutoff_nm' in params:
                _set_spin(self.filter_cutoff_spin, float(params.get('lowpass_cutoff_nm', 2.0)))
            elif 'filter_cutoff_nm' in params:
                _set_spin(self.filter_cutoff_spin, float(params.get('filter_cutoff_nm', 2.0)))

        # Physical noise / artifacts
        if 'noise_enable' in params and hasattr(self, 'chkNoiseEnable'):
            _set_check(self.chkNoiseEnable, params.get('noise_enable'))
        if 'noise_use_seed' in params and hasattr(self, 'chkUseNoiseSeed'):
            _set_check(self.chkUseNoiseSeed, params.get('noise_use_seed'))
        if 'noise_seed' in params and hasattr(self, 'spinNoiseSeed'):
            _set_spin(self.spinNoiseSeed, int(params.get('noise_seed', 42)))
        if 'height_noise_enable' in params and hasattr(self, 'chkHeightNoise'):
            _set_check(self.chkHeightNoise, params.get('height_noise_enable'))
        if 'height_noise_sigma_nm' in params and hasattr(self, 'spinHeightNoiseSigmaNm'):
            _set_spin(self.spinHeightNoiseSigmaNm, float(params.get('height_noise_sigma_nm', 0.1)))
        if 'line_noise_enable' in params and hasattr(self, 'chkLineNoise'):
            _set_check(self.chkLineNoise, params.get('line_noise_enable'))
        if 'line_noise_sigma_nm' in params and hasattr(self, 'spinLineNoiseSigmaNm'):
            _set_spin(self.spinLineNoiseSigmaNm, float(params.get('line_noise_sigma_nm', 0.05)))
        if 'line_noise_mode' in params and hasattr(self, 'comboLineNoiseMode'):
            _set_combo(self.comboLineNoiseMode, params.get('line_noise_mode', 'offset'))
        if 'drift_enable' in params and hasattr(self, 'chkDrift'):
            _set_check(self.chkDrift, params.get('drift_enable'))
        if 'drift_vx_nm_per_line' in params and hasattr(self, 'spinDriftVxNmPerLine'):
            _set_spin(self.spinDriftVxNmPerLine, float(params.get('drift_vx_nm_per_line', 0.0)))
        if 'drift_vy_nm_per_line' in params and hasattr(self, 'spinDriftVyNmPerLine'):
            _set_spin(self.spinDriftVyNmPerLine, float(params.get('drift_vy_nm_per_line', 0.0)))
        if 'drift_jitter_nm_per_line' in params and hasattr(self, 'spinDriftJitterNmPerLine'):
            _set_spin(self.spinDriftJitterNmPerLine, float(params.get('drift_jitter_nm_per_line', 0.0)))
        if 'feedback_lag_enable' in params and hasattr(self, 'chkFeedbackLag'):
            _set_check(self.chkFeedbackLag, params.get('feedback_lag_enable'))
        if 'feedback_mode' in params and hasattr(self, 'comboFeedbackMode'):
            _set_combo(self.comboFeedbackMode, params.get('feedback_mode', 'linear_lag'))
        if 'scan_direction' in params and hasattr(self, 'comboScanDirection'):
            _set_combo(self.comboScanDirection, params.get('scan_direction', 'L2R'))
        if 'lag_tau_lines' in params and hasattr(self, 'spinLagTauLines'):
            _set_spin(self.spinLagTauLines, float(params.get('lag_tau_lines', 2.0)))
        elif 'lag_tau_trace_lines' in params and hasattr(self, 'spinLagTauLines'):
            _set_spin(self.spinLagTauLines, float(params.get('lag_tau_trace_lines', 2.0)))
        if 'tap_drop_threshold_nm' in params and hasattr(self, 'spinTapDropThresholdNm'):
            _set_spin(self.spinTapDropThresholdNm, float(params.get('tap_drop_threshold_nm', 1.0)))
        if 'tap_tau_track_lines' in params and hasattr(self, 'spinTapTauTrackLines'):
            _set_spin(self.spinTapTauTrackLines, float(params.get('tap_tau_track_lines', 2.0)))
        if 'tap_tau_parachute_lines' in params and hasattr(self, 'spinTapTauParachuteLines'):
            _set_spin(self.spinTapTauParachuteLines, float(params.get('tap_tau_parachute_lines', 15.0)))
        if 'tap_release_threshold_nm' in params and hasattr(self, 'spinTapReleaseThresholdNm'):
            _set_spin(self.spinTapReleaseThresholdNm, float(params.get('tap_release_threshold_nm', 0.3)))

        esp_state = None

        # Display
        display = params.get('display', {})
        if display:
            if hasattr(self, 'style_combo') and 'style' in display:
                _set_combo(self.style_combo, display['style'])
            if hasattr(self, 'color_combo') and 'color' in display:
                _set_combo(self.color_combo, display['color'])
            if hasattr(self, 'atom_combo') and 'atom_filter' in display:
                _set_combo(self.atom_combo, display['atom_filter'])
            if hasattr(self, 'size_slider') and 'size' in display:
                _set_slider(self.size_slider, int(display['size']))
            if hasattr(self, 'opacity_slider') and 'opacity' in display:
                _set_slider(self.opacity_slider, int(display['opacity']))
            if hasattr(self, 'quality_combo') and 'quality' in display:
                _set_combo(self.quality_combo, display['quality'])
            if hasattr(self, 'renderer_combo') and 'renderer' in display:
                _set_combo(self.renderer_combo, display['renderer'])
            if hasattr(self, 'esp_check') and 'esp' in display:
                esp_state = bool(display['esp'])
                _set_check(self.esp_check, esp_state)

        # View
        view = params.get('view', {})
        if view:
            if hasattr(self, 'show_molecule_check') and 'show_molecule' in view:
                _set_check(self.show_molecule_check, view['show_molecule'])
            if hasattr(self, 'show_tip_check') and 'show_tip' in view:
                _set_check(self.show_tip_check, view['show_tip'])
            if hasattr(self, 'show_bonds_check') and 'show_bonds' in view:
                _set_check(self.show_bonds_check, view['show_bonds'])

        # AFM view selection
        afm_views = params.get('afm_views', {})
        if afm_views:
            if hasattr(self, 'afm_x_check') and 'xy' in afm_views:
                _set_check(self.afm_x_check, afm_views['xy'])
            if hasattr(self, 'afm_y_check') and 'yz' in afm_views:
                _set_check(self.afm_y_check, afm_views['yz'])
            if hasattr(self, 'afm_z_check') and 'zx' in afm_views:
                _set_check(self.afm_z_check, afm_views['zx'])

        # Lighting / colors
        lighting = params.get('lighting', {})
        if lighting:
            if hasattr(self, 'brightness_slider') and 'brightness' in lighting:
                _set_slider(self.brightness_slider, int(lighting['brightness']))
            if hasattr(self, 'ambient_slider') and 'ambient' in lighting:
                _set_slider(self.ambient_slider, int(lighting['ambient']))
            if hasattr(self, 'specular_slider') and 'specular' in lighting:
                _set_slider(self.specular_slider, int(lighting['specular']))

        colors = params.get('colors', {})
        if colors:
            if 'bg_color' in colors:
                try:
                    r, g, b = colors['bg_color']
                    self.current_bg_color = (float(r), float(g), float(b))
                except Exception:
                    pass
            if 'single_color' in colors:
                try:
                    r, g, b = colors['single_color']
                    self.current_single_color = (float(r), float(g), float(b))
                except Exception:
                    pass

        # Rotation
        rot = params.get('rotation', {})
        if rot and hasattr(self, 'rotation_widgets'):
            try:
                self.rotation_widgets['X']['spin'].blockSignals(True)
                self.rotation_widgets['Y']['spin'].blockSignals(True)
                self.rotation_widgets['Z']['spin'].blockSignals(True)
                self.rotation_widgets['X']['slider'].blockSignals(True)
                self.rotation_widgets['Y']['slider'].blockSignals(True)
                self.rotation_widgets['Z']['slider'].blockSignals(True)
                if 'x' in rot:
                    self.rotation_widgets['X']['spin'].setValue(float(rot['x']))
                    self.rotation_widgets['X']['slider'].setValue(int(float(rot['x']) * 10))
                if 'y' in rot:
                    self.rotation_widgets['Y']['spin'].setValue(float(rot['y']))
                    self.rotation_widgets['Y']['slider'].setValue(int(float(rot['y']) * 10))
                if 'z' in rot:
                    self.rotation_widgets['Z']['spin'].setValue(float(rot['z']))
                    self.rotation_widgets['Z']['slider'].setValue(int(float(rot['z']) * 10))
            finally:
                self.rotation_widgets['X']['spin'].blockSignals(False)
                self.rotation_widgets['Y']['spin'].blockSignals(False)
                self.rotation_widgets['Z']['spin'].blockSignals(False)
                self.rotation_widgets['X']['slider'].blockSignals(False)
                self.rotation_widgets['Y']['slider'].blockSignals(False)
                self.rotation_widgets['Z']['slider'].blockSignals(False)

        # Find initial plane
        fip = params.get('find_initial_plane', {})
        if fip and hasattr(self, 'find_plane_params'):
            fp = self.find_plane_params
            if 'use_elec' in fip and 'use_elec' in fp:
                _set_check(fp['use_elec'], fip['use_elec'])
            for key in ('pH', 'alpha_elec', 'r_pseudo', 'delta_elec'):
                if key in fip and key in fp:
                    _set_spin(fp[key], float(fip[key]))
            for key in ('N_dir', 'K', 'K_geom', 'grid', 'surf_n'):
                if key in fip and key in fp:
                    _set_spin(fp[key], int(fip[key]))
            for key in ('delta', 'lambda', 'theta1', 'theta2', 'theta3', 'surf_r'):
                if key in fip and key in fp:
                    _set_spin(fp[key], float(fip[key]))
            if 'substrate' in fip and 'substrate' in fp:
                _set_combo(fp['substrate'], fip['substrate'])
            if 'salt' in fip and 'salt' in fp:
                _set_combo(fp['salt'], fip['salt'])
            if 'salt_mM' in fip and 'salt_mM' in fp:
                _set_spin(fp['salt_mM'], float(fip['salt_mM']))

        # MRC
        mrc = params.get('mrc', {})
        if mrc:
            if 'threshold' in mrc and hasattr(self, 'mrc_threshold_slider'):
                try:
                    self.mrc_threshold = float(mrc['threshold'])
                except Exception:
                    pass
                _set_slider(self.mrc_threshold_slider, int(float(mrc['threshold']) * 100))
            if 'z_flip' in mrc and hasattr(self, 'mrc_z_flip_check'):
                try:
                    self.mrc_z_flip = bool(mrc['z_flip'])
                except Exception:
                    pass
                _set_check(self.mrc_z_flip_check, bool(mrc['z_flip']))

        # Apply updates
        try:
            if hasattr(self, 'apply_filter_check') and hasattr(self, 'filter_cutoff_spin'):
                self.filter_cutoff_spin.setEnabled(self.apply_filter_check.isChecked())
        except Exception:
            pass

        try:
            self._update_noise_ui_states()
        except Exception:
            pass

        try:
            if hasattr(self, 'find_plane_params'):
                fp = self.find_plane_params
                if 'use_elec' in fp:
                    enabled = fp['use_elec'].isChecked()
                    for key in ('pH', 'substrate', 'salt', 'salt_mM', 'alpha_elec', 'r_pseudo', 'delta_elec', 'K_geom'):
                        if key in fp:
                            fp[key].setEnabled(enabled)
        except Exception:
            pass

        try:
            if hasattr(self, 'tip_shape_combo'):
                self.update_tip_ui(self.tip_shape_combo.currentText())
        except Exception:
            pass
        try:
            if hasattr(self, 'renderer_combo'):
                self.on_renderer_changed(self.renderer_combo.currentText())
        except Exception:
            pass
        try:
            self.update_tip()
        except Exception:
            pass
        try:
            self.update_tip_position()
        except Exception:
            pass
        try:
            self.update_display()
        except Exception:
            pass
        try:
            if hasattr(self, 'show_molecule_check'):
                self.toggle_molecule_visibility(self.show_molecule_check.isChecked())
            if hasattr(self, 'show_tip_check'):
                self.toggle_tip_visibility(self.show_tip_check.isChecked())
            if hasattr(self, 'show_bonds_check'):
                self.toggle_bonds_visibility(self.show_bonds_check.isChecked())
        except Exception:
            pass
        try:
            if hasattr(self, 'update_afm_display'):
                self.update_afm_display()
        except Exception:
            pass
        try:
            if hasattr(self, 'process_and_display_all_images'):
                self.process_and_display_all_images()
        except Exception:
            pass
        try:
            if hasattr(self, '_apply_rectangle_lock'):
                self._apply_rectangle_lock(enforce_values=True)
        except Exception:
            pass
        try:
            if esp_state is not None and hasattr(self, 'display_electrostatics'):
                self.display_electrostatics(bool(self.esp_check.isChecked()))
        except Exception:
            pass
        try:
            self._apply_current_background_color()
        except Exception:
            pass
        try:
            if hasattr(self, 'rotation_widgets'):
                self.apply_structure_rotation()
        except Exception:
            pass
        try:
            self.request_render()
        except Exception:
            pass

    def get_current_structure_info(self):
        """Return current structure file info or None."""
        path = getattr(self, 'current_structure_path', None)
        if not path:
            return None
        abs_path = os.path.abspath(path)
        fmt = getattr(self, 'current_structure_type', None)
        if not fmt:
            ext = os.path.splitext(abs_path)[1].lower().lstrip('.')
            fmt = ext
        return {
            'path': abs_path,
            'basename': os.path.basename(abs_path),
            'format': fmt,
        }

    def save_session_json(self, json_path):
        """Save current session (params + structure) to JSON."""
        doc = {
            'schema_version': 2,
            'app': 'pyNuD_simulator',
            'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'structure': self.get_current_structure_info(),
            'params': self.collect_sim_params(),
            'real_asd_path': self.real_asd_path,
            'pose': {
                'theta_deg': float(self.pose.get('theta_deg', 0.0)) if isinstance(self.pose, dict) else 0.0,
                'dx_px': float(self.pose.get('dx_px', 0.0)) if isinstance(self.pose, dict) else 0.0,
                'dy_px': float(self.pose.get('dy_px', 0.0)) if isinstance(self.pose, dict) else 0.0,
                'mirror_mode': str(self.pose.get('mirror_mode', 'none')) if isinstance(self.pose, dict) else 'none',
            },
        }
        with open(json_path, 'w') as f:
            json.dump(doc, f, indent=2)

    def _load_structure_file(self, path):
        """Load structure using existing import functions. Returns True if loaded."""
        if not path or not os.path.exists(path):
            return False
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.pdb':
                self._import_pdb_internal(path)
            elif ext in ('.cif', '.mmcif'):
                self._import_cif_internal(path)
            elif ext == '.mrc':
                self._import_mrc_internal(path)
            else:
                return False
            self.last_import_dir = os.path.dirname(path)
            return True
        except Exception:
            return False

    def load_session_json(self, json_path):
        """Load session JSON and apply parameters (schema v1/v2)."""
        try:
            with open(json_path, 'r') as f:
                doc = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', f'Failed to load JSON:\n{e}')
            return

        schema = int(doc.get('schema_version', 1) or 1)
        params = doc.get('params', {}) if isinstance(doc.get('params', {}), dict) else {}

        loaded = False
        if schema >= 2:
            structure = doc.get('structure', None)
            if isinstance(structure, dict):
                path = structure.get('path', None)
                basename = structure.get('basename', None)
                if path and os.path.exists(path):
                    loaded = self._load_structure_file(path)
                if not loaded and basename:
                    candidate = os.path.join(os.path.dirname(json_path), basename)
                    if os.path.exists(candidate):
                        loaded = self._load_structure_file(candidate)
                if not loaded:
                    # prompt user
                    initial_dir = os.path.dirname(json_path)
                    file_path, _ = QFileDialog.getOpenFileName(
                        self, 'Select Structure File', initial_dir,
                        'Structure Files (*.pdb *.cif *.mmcif);;All Files (*)',
                        options=QFileDialog.DontUseNativeDialog
                    )
                    if file_path:
                        loaded = self._load_structure_file(file_path)
        # schema v1: no structure

        # Real ASD (optional)
        real_asd_path = doc.get('real_asd_path', None)
        if isinstance(real_asd_path, str) and real_asd_path:
            if os.path.exists(real_asd_path):
                self.load_real_asd_file(real_asd_path, sync=True)

        pose_doc = doc.get('pose', None)
        if isinstance(pose_doc, dict):
            try:
                self.pose = {
                    'theta_deg': float(pose_doc.get('theta_deg', 0.0)),
                    'dx_px': float(pose_doc.get('dx_px', 0.0)),
                    'dy_px': float(pose_doc.get('dy_px', 0.0)),
                    'mirror_mode': str(pose_doc.get('mirror_mode', 'none')),
                    'score': None,
                }
            except Exception:
                pass

        # Apply params regardless of structure load
        self.apply_sim_params(params)

    def handle_save_params_json(self):
        """Save params to JSON via dialog."""
        initial_dir = self.last_import_dir if hasattr(self, 'last_import_dir') and self.last_import_dir else ''
        default_id = self.get_active_dataset_id() if hasattr(self, 'get_active_dataset_id') else 'session'
        default_name = f"{default_id}.json"
        default_path = os.path.join(initial_dir, default_name) if initial_dir else default_name
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Params as JSON', default_path,
            'JSON files (*.json);;All Files (*)',
            options=QFileDialog.DontUseNativeDialog
        )
        if not save_path:
            return
        try:
            self.save_session_json(save_path)
            QMessageBox.information(self, 'Saved', f'Session saved to:\n{save_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', f'Failed to save JSON:\n{e}')

    def handle_load_params_json(self):
        """Load params from JSON via dialog."""
        initial_dir = self.last_import_dir if hasattr(self, 'last_import_dir') and self.last_import_dir else ''
        load_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Params from JSON', initial_dir,
            'JSON files (*.json);;All Files (*)',
            options=QFileDialog.DontUseNativeDialog
        )
        if not load_path:
            return
        self.load_session_json(load_path)

    def _apply_current_background_color(self):
        """現在の背景色設定をVTK/PyMOL/ボタンへ反映"""
        try:
            r, g, b = self.current_bg_color
            r255 = int(max(0.0, min(1.0, r)) * 255)
            g255 = int(max(0.0, min(1.0, g)) * 255)
            b255 = int(max(0.0, min(1.0, b)) * 255)
            if hasattr(self, 'bg_color_btn') and self.bg_color_btn is not None:
                text_is_dark = (r255 + g255 + b255) > 400
                self.bg_color_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgb({r255}, {g255}, {b255});
                        color: {'black' if text_is_dark else 'white'};
                        border: 2px solid #555;
                        border-radius: 5px;
                    }}
                    QPushButton:hover {{
                        border-color: #777;
                    }}
                """)
        except Exception:
            pass

        try:
            if self.pymol_available and self.pymol_cmd is not None:
                self._pymol_set_background(self.current_bg_color)
                # Force one immediate re-render in image mode so the user sees the change right away.
                if getattr(self, "pymol_image_mode", False) and getattr(self, "pymol_image_label", None) is not None:
                    try:
                        self._pymol_last_render_ts = 0.0
                        self._schedule_pymol_render(force=True)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if hasattr(self, 'renderer') and self.renderer is not None and not self._is_pymol_only():
                self.renderer.SetBackground(*self.current_bg_color)
        except Exception:
            pass

        try:
            self.request_render()
        except Exception:
            pass
    
    def handle_save_asd(self):
        """「Save as ASD...」ボタンが押されたときの処理"""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Data", "No simulation data available to save.")
            return

         # 保存可能なデータの名前（キー）を取得
        available_keys = list(self.simulation_results.keys())
        
        # ユーザーに表示するための分かりやすい名前の辞書
        display_names = {
            "XY_Frame": "XY View",
            "YZ_Frame": "YZ View",
            "ZX_Frame": "ZX View"
        }
        
        # 選択肢リストを作成
        choices = [display_names.get(key, key) for key in available_keys]
        
        selected_key = None
        if len(available_keys) > 1:
            # データが複数ある場合、ダイアログで選択させる
            choice, ok = QInputDialog.getItem(self, "Select Data to Save", "保存するデータを選択してください:", choices, 0, False)
            if not ok or not choice:
                return # キャンセルされた場合
            # 選択された表示名から内部キーを逆引き
            for key, name in display_names.items():
                if name == choice:
                    selected_key = key
                    break
        elif len(available_keys) == 1:
            # データが1つだけなら、それを自動で選択
            selected_key = available_keys[0]
        else:
            # データがない場合は何もしない
            return

        if selected_key is None:
            return
                
        data_to_save = self.simulation_results[selected_key]
        image_key_name = display_names.get(selected_key, selected_key).replace(" ", "") # ファイル名用
        default_id = self.get_active_dataset_id()
        default_filename = f"{default_id}_{image_key_name}.asd"

        directory = ""
        # 最後に使用したディレクトリが存在し、アクセス可能かチェック
        if self.last_import_dir and os.path.isdir(self.last_import_dir):
            directory = self.last_import_dir
        
        # ファイル名と安全なディレクトリを結合して、最終的なデフォルトパスを作成
        default_save_path = os.path.join(directory, default_filename)

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation as ASD", default_save_path, "ASD files (*.asd)",
            options=QFileDialog.DontUseNativeDialog
        )

        if not save_path:
            return

        try:            
            # --- シミュレーション条件を収集 ---
            rot_x = self.rotation_widgets['X']['spin'].value()
            rot_y = self.rotation_widgets['Y']['spin'].value()
            rot_z = self.rotation_widgets['Z']['spin'].value()
            
            tip_shape = self.tip_shape_combo.currentText()
            tip_radius = self.tip_radius_spin.value()
            tip_angle = self.tip_angle_spin.value()
            
            scan_x = self.spinScanXNm.value()
            scan_y = self.spinScanYNm.value()
            nx = self.spinNx.value()
            ny = self.spinNy.value()
            center_x = self.tip_x_slider.value() / 5.0
            center_y = self.tip_y_slider.value() / 5.0
            
            use_vdw = "Yes" if self.use_vdw_check.isChecked() else "No"
            sim_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # --- コメント文字列を生成 ---
            pdb_file = self.pdb_name if getattr(self, 'pdb_id', '') else "(none)"
            mrc_file = self.mrc_name if getattr(self, 'mrc_id', '') else "(none)"

            comment_lines = [
                f"pyNuD_simulator Log",
                f"Date: {sim_date}",
                f"--------------------",
                f"[File Info]",
                f"PDB File: {pdb_file}",
                f"MRC File: {mrc_file}",
                f"",
                f"[View Settings]",
                f"Rotation X: {rot_x:.1f} deg",
                f"Rotation Y: {rot_y:.1f} deg",
                f"Rotation Z: {rot_z:.1f} deg",
                f"",
                f"[Tip Conditions]",
                f"Shape: {tip_shape}",
                f"Radius: {tip_radius:.2f} nm",
            ]
            
            if tip_shape == "Cone":
                comment_lines.append(f"Angle: {tip_angle:.1f} deg")
            elif tip_shape == "Sphere":
                minitip_radius = self.minitip_radius_spin.value()
                comment_lines.append(f"Angle: {tip_angle:.1f} deg")
                comment_lines.append(f"Minitip Radius: {minitip_radius:.2f} nm")

            comment_lines.extend([
                f"",
                f"[Scan Parameters]",
                f"Scan Size X: {scan_x:.1f} nm",
                f"Scan Size Y: {scan_y:.1f} nm",
                f"Resolution: {nx}x{ny}",
                f"Center: ({center_x:.2f}, {center_y:.2f}) nm",
                f"",
                f"[Calculation Method]",
                f"Consider vdW: {use_vdw}",
            ])
            
            comment = "\n".join(comment_lines)
            
            # # save_simulation_as_asd を呼び出す
            success = self.save_simulation_as_asd(save_path, comment, data_to_save)
            if success:
                try:
                    json_path = os.path.splitext(save_path)[0] + ".json"
                    self.save_session_json(json_path)
                except Exception as e:
                    print(f"[Save Params] Failed to write JSON: {e}")
                QMessageBox.information(self, "Save Successful", f"Data successfully saved to:\n{save_path}")
            else:
                QMessageBox.critical(self, "Save Error", "Failed to save ASD file. Check console for details.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the ASD file:\n{e}")



    def save_simulation_as_asd(self, save_path, comment_string, height_map):
        """
        ASDファイルを保存するメソッド。Igor Proのコードと完全に一致させる。
        """
        try:
            y_pixels, x_pixels = height_map.shape
            x_scan_size = self.spinScanXNm.value()
            y_scan_size = self.spinScanYNm.value()

            # ★★★ 修正点2: comment_bytes を正しく使用する ★★★
            ope_name_bytes = "Nobody".encode('utf-8')
            comment_bytes = comment_string.encode('utf-8')
            
            # Igorコードの `165` は固定ヘッダーのバイト数
            file_header_size = 165 + len(ope_name_bytes) + len(comment_bytes)
            frame_header_size = 32

            now = datetime.datetime.now()
            
            with open(save_path, 'wb') as f:
                # --- ファイルヘッダー書き込み ---
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', file_header_size))
                f.write(struct.pack('<i', frame_header_size))
                f.write(struct.pack('<i', 932))
                f.write(struct.pack('<i', len(ope_name_bytes)))
                f.write(struct.pack('<i', len(comment_bytes))) # 正しいコメント長を書き込む
                f.write(struct.pack('<i', 20564))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', x_pixels))
                f.write(struct.pack('<i', y_pixels))
                f.write(struct.pack('<i', int(x_scan_size)))
                f.write(struct.pack('<i', int(y_scan_size)))
                f.write(struct.pack('<B', 0))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', now.year))
                f.write(struct.pack('<i', now.month))
                f.write(struct.pack('<i', now.day))
                f.write(struct.pack('<i', now.hour))
                f.write(struct.pack('<i', now.minute))
                f.write(struct.pack('<i', now.second))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<iiii', 0, 0, 0, 0))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', 262144))
                f.write(struct.pack('<i', 12))
                f.write(struct.pack('<f', 4000.0))
                f.write(struct.pack('<f', 1700.0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<f', 20.0))
                f.write(struct.pack('<f', 2.0))
                
                f.write(ope_name_bytes)
                f.write(comment_bytes) # ★★★ 正しいコメント本体を書き込む ★★★

                # --- フレームヘッダー書き込み ---
                max_data_raw = np.max(height_map)
                min_data_raw = np.min(height_map)

                f.write(struct.pack('<I', 0))
                f.write(struct.pack('<H', int(max_data_raw)))
                f.write(struct.pack('<H', int(min_data_raw)))
                f.write(struct.pack('<h', 0))
                f.write(struct.pack('<h', 0))
                f.write(struct.pack('<f', 0.0))
                f.write(struct.pack('<f', 0.0))
                f.write(struct.pack('<B', 0))
                f.write(struct.pack('<B', 0))
                f.write(struct.pack('<h', 0))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<i', 0))

                # --- 画像データ書き込み ---
                piezo_const_z = 20.0
                driver_gain_z = 2.0
                
                for y in range(y_pixels):
                    for x in range(x_pixels):
                        height_value = height_map[y, x]
                        # シミュレーターは凹凸データのみなので、nm → uint16の変換のみ
                        data = (5.0 - height_value / piezo_const_z / driver_gain_z) * 4096.0 / 10.0
                        f.write(struct.pack('<h', int(data)))
            return True
            
        except Exception as e:
            print(f"[ERROR] SaveASD failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def handle_save_3d_view(self):
        """現在の3Dビューを画像ファイルとして保存する"""
        if self.pdb_name == "":
            QMessageBox.warning(self, "No Data", "Please load a PDB file first.")
            return

        # --- ファイル保存ダイアログの準備 ---
        default_filename = f"{self.pdb_name}_3D_view.png"
        directory = ""
        if self.last_import_dir and os.path.isdir(self.last_import_dir):
            directory = self.last_import_dir
        
        default_save_path = os.path.join(directory, default_filename)
        
        # ユーザーにファイル名と保存形式を選択させる
        filters = "PNG Image (*.png);;TIFF Image (*.tif)"
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save 3D View As...", default_save_path, filters,
            options=QFileDialog.DontUseNativeDialog
        )

        if not save_path:
            return

        # --- PyMOLウィンドウのキャプチャと保存 ---
        try:
            if self._is_pymol_active():
                width = 1200
                height = 900
                if self.pymol_widget is not None:
                    width = max(400, int(self.pymol_widget.width()))
                    height = max(300, int(self.pymol_widget.height()))
                # PyMOLでPNG/TIFFを保存
                if not (save_path.endswith('.png') or save_path.endswith('.tif')):
                    if "png" in selected_filter:
                        save_path += ".png"
                    else:
                        save_path += ".tif"
                self.pymol_cmd.png(save_path, width, height, dpi=300, ray=0, quiet=1)
                QMessageBox.information(self, "Save Successful", f"3D view successfully saved to:\n{save_path}")
                return

            # 1. VTKウィンドウを画像データに変換するフィルターを作成
            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(self.vtk_widget.GetRenderWindow())
            # アルファチャンネル（透明度）を含めずにRGBのみをキャプチャ
            window_to_image_filter.SetInputBufferTypeToRGB() 
            # スケーリングを無効にし、ウィンドウの解像度でキャプチャ
            window_to_image_filter.SetScale(1) 
            window_to_image_filter.Update()

            # 2. 選択されたファイル形式に応じて適切なライターを選択
            if save_path.endswith('.png'):
                writer = vtk.vtkPNGWriter()
            elif save_path.endswith('.tif'):
                writer = vtk.vtkTIFFWriter()
            else:
                # ユーザーが拡張子を入力しなかった場合、選択したフィルターから判断
                if "png" in selected_filter:
                    save_path += ".png"
                    writer = vtk.vtkPNGWriter()
                else:
                    save_path += ".tif"
                    writer = vtk.vtkTIFFWriter()

            # 3. ファイルを書き出す
            writer.SetFileName(save_path)
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()
            
            QMessageBox.information(self, "Save Successful", f"3D view successfully saved to:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the 3D view:\n{e}")
    
    def handle_save_image(self):
        """Export one or more simulated AFM images (PNG) with optional incremental rotation."""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Data", "No simulation data available to save.")
            return
        
        # Build available (only those already simulated)
        available_keys = list(self.simulation_results.keys())
        display_names = {"XY_Frame": "XY View", "YZ_Frame": "YZ View", "ZX_Frame": "ZX View"}
        
        dlg = SaveAFMImageDialog(available_keys, display_names, self.get_active_dataset_id(), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        result = dlg.get_result()
        selected_view_keys = result['selected_views']
        rot_inc = result['drot']
        base_name = result['base_name']
        
        if not selected_view_keys:
            QMessageBox.warning(self, "No Selection", "No views selected.")
            return
        
        # Map for filename friendly
        def key_to_short(k):
            return {
                "XY_Frame": "XY",
                "YZ_Frame": "YZ",
                "ZX_Frame": "ZX"
            }.get(k, k.replace("_Frame", ""))
        
        # Prepare directory & ensure last_import_dir is valid
        directory = ""
        if self.last_import_dir and os.path.isdir(self.last_import_dir):
            directory = self.last_import_dir
        if not directory:
            directory = os.getcwd()
        
        # Save original rotation
        orig_rx = self.rotation_widgets['X']['spin'].value()
        orig_ry = self.rotation_widgets['Y']['spin'].value()
        orig_rz = self.rotation_widgets['Z']['spin'].value()
        
        apply_rotation = any(abs(v) > 1e-6 for v in rot_inc.values())
        
        try:
            if apply_rotation:
                # Apply incremental rotation (add to current)
                self.rotation_widgets['X']['spin'].setValue(self.normalize_angle(orig_rx + rot_inc['x']))
                self.rotation_widgets['Y']['spin'].setValue(self.normalize_angle(orig_ry + rot_inc['y']))
                self.rotation_widgets['Z']['spin'].setValue(self.normalize_angle(orig_rz + rot_inc['z']))
                # Force apply transform & run simulation for required views
                self.apply_structure_rotation()
                self.simulate_views_blocking(selected_view_keys)
            
            # Export each selected view
            export_count = 0
            for key in selected_view_keys:
                if key not in self.simulation_results:
                    continue
                data = self.simulation_results[key]
                # Normalize to 8-bit grayscale
                mn, mx = float(np.min(data)), float(np.max(data))
                if mx <= mn:
                    norm = np.zeros_like(data, dtype=np.uint8)
                else:
                    norm = ((data - mn) / (mx - mn) * 255).astype(np.uint8)
                
                # Resize to 512x512
                try:
                    from PIL import Image
                except ImportError:
                    QMessageBox.critical(self, "Missing Pillow", "Install Pillow to export images (pip install Pillow).")
                    return
                img = Image.fromarray(norm, mode='L')
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS
                img = img.resize((512, 512), resample=resample_filter)
                
                fname = f"{base_name}_{key_to_short(key)}_dx{rot_inc['x']:+.0f}_dy{rot_inc['y']:+.0f}_dz{rot_inc['z']:+.0f}.png"
                save_path = os.path.join(directory, fname)
                try:
                    img.save(save_path)
                    export_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to save {save_path}: {e}")
            
            if export_count:
                QMessageBox.information(self, "Export Complete", f"Exported {export_count} image(s) to:\n{directory}")
            else:
                QMessageBox.warning(self, "No Export", "No images were exported.")
        
        finally:
            # Restore original rotation if we changed it
            if apply_rotation:
                self.rotation_widgets['X']['spin'].setValue(orig_rx)
                self.rotation_widgets['Y']['spin'].setValue(orig_ry)
                self.rotation_widgets['Z']['spin'].setValue(orig_rz)
                self.apply_structure_rotation()
                # (Optionally regenerate original visible views if needed)
                # self.simulate_views_blocking(available_keys)
    
    def run_simulation_on_view_change(self, is_checked):
        """
        View選択チェックボックスがONになった時にシミュレーションを自動実行するスロット。
        """

        if not is_checked:
            return

        # If Interactive Update is enabled, keep auto-updates lightweight (XY only).
        try:
            if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
                self.trigger_interactive_simulation()
                return
        except Exception:
            pass

        # Otherwise, run a full simulation for the newly enabled views.
        if self.atoms_data is not None:
            if self.simulate_btn.text() == "Cancel":
                print("Note: Another simulation is already running.")
                return
            self.run_simulation()

    def _is_rectangle_lock_enabled(self):
        try:
            return bool(getattr(self, "rectangle_check", None) and self.rectangle_check.isChecked())
        except Exception:
            return False

    def _apply_rectangle_lock(self, *, enforce_values=True):
        """Apply the Rectangle lock (Y := X) to scan size and resolution UI."""
        if not hasattr(self, "spinScanXNm") or not hasattr(self, "spinScanYNm") or not hasattr(self, "spinNx") or not hasattr(self, "spinNy"):
            return

        locked = self._is_rectangle_lock_enabled()

        # Disable Y inputs while locked (still shows the value)
        try:
            self.spinScanYNm.setEnabled(not locked)
        except Exception:
            pass
        try:
            self.spinNy.setEnabled(not locked)
        except Exception:
            pass

        if not locked or not enforce_values:
            return

        # Enforce ScanY == ScanX and Ny == Nx
        try:
            x = float(self.spinScanXNm.value())
            if abs(float(self.spinScanYNm.value()) - x) > 1e-9:
                self.spinScanYNm.blockSignals(True)
                self.spinScanYNm.setValue(x)
                self.spinScanYNm.blockSignals(False)
        except Exception:
            try:
                self.spinScanYNm.blockSignals(False)
            except Exception:
                pass

        try:
            nx = int(self.spinNx.value())
            if int(self.spinNy.value()) != nx:
                self.spinNy.blockSignals(True)
                self.spinNy.setValue(nx)
                self.spinNy.blockSignals(False)
        except Exception:
            try:
                self.spinNy.blockSignals(False)
            except Exception:
                pass

    def _enforce_rectangle_lock_from_scan(self, _value=None):
        """Keep ScanY synced to ScanX when Rectangle lock is enabled."""
        if not self._is_rectangle_lock_enabled():
            return
        self._apply_rectangle_lock(enforce_values=True)

    def _enforce_rectangle_lock_from_resolution(self, _value=None):
        """Keep Ny synced to Nx when Rectangle lock is enabled."""
        if not self._is_rectangle_lock_enabled():
            return
        self._apply_rectangle_lock(enforce_values=True)

    def on_rectangle_lock_toggled(self, checked):
        """Handle Rectangle lock checkbox toggle."""
        self._apply_rectangle_lock(enforce_values=True)
        # Update simulation only when interactive update is enabled (keeps UI responsive).
        try:
            self.trigger_interactive_simulation()
        except Exception:
            pass

    def _apply_interactive_update_mode(self, enabled, *, show_message=False, run_initial=False):
        """Apply Interactive Update side-effects (UI + optional first simulation)."""
        try:
            enabled = bool(enabled)
        except Exception:
            enabled = False

        if enabled:
            # Remember current resolution for later high-res updates
            try:
                self.user_selected_resolution = self.resolution_combo.currentText()
            except Exception:
                pass

            # Ensure XY stays enabled/selected so real-time updates actually show something.
            try:
                if hasattr(self, 'afm_x_check') and not self.afm_x_check.isChecked():
                    self.afm_x_check.blockSignals(True)
                    self.afm_x_check.setChecked(True)
                    self.afm_x_check.blockSignals(False)
            except Exception:
                try:
                    self.afm_x_check.blockSignals(False)
                except Exception:
                    pass

            if show_message:
                try:
                    QMessageBox.information(
                        self, "Interactive Update Enabled",
                        "Interactive Update enabled.\n\n"
                        "• During rotation: Low resolution (64x64) for real-time updates (XY only)\n"
                        "• After rotation stops: High resolution (current setting) automatically generated (XY/YZ/ZX)\n"
                        "• If YZ/ZX are checked, they will be updated automatically after rotation stops\n"
                        "• You can change resolution anytime"
                    )
                except Exception:
                    pass

            if run_initial:
                # Prefer lightweight update
                try:
                    self.trigger_interactive_simulation()
                except Exception:
                    pass
        else:
            # Stop any pending high-res timer
            if hasattr(self, 'high_res_timer'):
                try:
                    self.high_res_timer.stop()
                except Exception:
                    pass

    def handle_interactive_update_toggle(self, is_checked):
        """「Interactive Update」チェックボックスの状態変化を処理する"""
        self._apply_interactive_update_mode(is_checked, show_message=True, run_initial=True)

    def on_resolution_changed(self, value):
        """NxまたはNyの数値が直接変更された時の処理"""
        # 通常のシミュレーションを実行
        self.trigger_interactive_simulation()

    def on_resolution_combo_changed(self, text):
        """Quick Resコンボボックスが変更された時の処理"""
        try:
            res = int(text.split('x')[0])
            # Nx, Nyを同時に更新（シグナルを一時ブロックして無限ループを避ける）
            self.spinNx.blockSignals(True)
            self.spinNy.blockSignals(True)
            self.spinNx.setValue(res)
            self.spinNy.setValue(res)
            self.spinNx.blockSignals(False)
            self.spinNy.blockSignals(False)
            
            # Interactive Updateが有効な場合、新しい解像度を高解像度として記憶
            if self.interactive_update_check.isChecked():
                self.user_selected_resolution = f"{res}x{res}"
            
            self.trigger_interactive_simulation()
        except (ValueError, IndexError):
            pass

    def run_simulation_silent(self, only_xy=False):
        """
        Interactive Update専用の軽量シミュレーション実行。
        UIの変更（ボタン、プログレスバー、ステータスなど）を行わない。
        """
        coords, mode = self.get_simulation_coords()
        if coords is None:
            return
        
        # 既に別のシミュレーションが実行中の場合は何もしない（改良版）
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker') or \
           self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            return

        base_coords = coords
        if base_coords is None:
            return

        # UIから共通パラメータを取得（ドラッグ中は常に64x64で計算）
        sim_params = {
            'scan_x_nm': self.spinScanXNm.value(),
            'scan_y_nm': self.spinScanYNm.value(),
            'nx': 64,  # ★★★ ドラッグ中は常に低解像度で計算 ★★★
            'ny': 64,
            'center_x': self.tip_x_slider.value() / 5.0,
            'center_y': self.tip_y_slider.value() / 5.0,
            'tip_radius': self.tip_radius_spin.value(),
            'minitip_radius': self.minitip_radius_spin.value(),
            'tip_angle': self.tip_angle_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'use_vdw': self.use_vdw_check.isChecked()
        }
        # Backward compatibility
        sim_params['scan_size'] = sim_params['scan_x_nm']
        sim_params['resolution'] = 64

        # チェックされた面の計算タスクを作成
        tasks = []
        if self.afm_x_check.isChecked():
            tasks.append({
                "name": "XY",
                "panel": self.afm_x_frame,
                "coords": base_coords
            })
        if (not only_xy) and self.afm_y_check.isChecked():
            x_scan = base_coords[:, 1]
            y_scan = base_coords[:, 2]
            z_scan = -base_coords[:, 0]
            tasks.append({
                "name": "YZ",
                "panel": self.afm_y_frame,
                "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)
            })
        if (not only_xy) and self.afm_z_check.isChecked():
            x_scan, y_scan, z_scan = base_coords[:, 0], base_coords[:, 2], -base_coords[:, 1]
            tasks.append({
                "name": "ZX",
                "panel": self.afm_z_frame,
                "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)
            })

        if not tasks:
            return

        # 既存の軽量ワーカーを停止（より安全に）
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            try:
                self.stop_worker(self.sim_worker_silent, timeout_ms=50, allow_terminate=True, worker_name="sim_worker_silent")
            except Exception as e:
                print(f"[WARNING] Error stopping sim_worker_silent: {e}")
        
        # 軽量ワーカーを作成（UI変更なし）
        self.sim_worker_silent = AFMSimulationWorker(
            self, sim_params, tasks,
            self.atoms_data['element'] if sim_params['use_vdw'] and self.atoms_data is not None else None,
            self.vdw_radii if sim_params['use_vdw'] and hasattr(self, 'vdw_radii') else None,
            silent_mode=True
        )
        self._connect_worker_delete_later(self.sim_worker_silent)
        self._track_worker_ref('sim_worker_silent', self.sim_worker_silent)

        # 最小限の接続のみ（プログレス、ステータス、ボタン変更なし）
        self.sim_worker_silent.task_done.connect(self.on_task_finished_silent)
        self.sim_worker_silent.done.connect(self.on_simulation_finished_silent)
        self.sim_worker_silent.start()

    def run_simulation_immediate(self):
        """
        Interactive Update用の即座実行版シミュレーション。
        ドラッグ中に使用され、タイマー遅延なしで実行される。
        """
        # データが読み込まれていない場合は何もしない
        if self.atoms_data is None and not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            return
        
        # 以前のタイマーが作動中であれば停止する
        if hasattr(self, 'interactive_timer'):
            self.interactive_timer.stop()
        
        # ★★★ 軽量シミュレーションを実行（ドラッグ中はXYのみ） ★★★
        self.run_simulation_silent(only_xy=True)

    def run_simulation_immediate_controlled(self):
        """
        ドラッグ中専用の制御付き即座実行版シミュレーション。
        前のシミュレーションが完了するまで待機し、スレッドの蓄積を防ぐ。
        """
        # データが読み込まれていない場合は何もしない
        if self.atoms_data is None and not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            return
        
        # 以前のタイマーが作動中であれば停止する
        if hasattr(self, 'interactive_timer'):
            self.interactive_timer.stop()
        
        # ★★★ 前のシミュレーションが完了するまで待機 ★★★
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            # 前のシミュレーションが実行中の場合は、ドラッグ中の更新をスキップ
            return
        
        # ★★★ 最小更新間隔の制御を強化 ★★★
        current_time = QTime.currentTime()
        if hasattr(self, 'last_drag_simulation_time'):
            time_diff = self.last_drag_simulation_time.msecsTo(current_time)
            if time_diff < 300:  # 300ms未満の場合はスキップ（200msから増加）
                return
        
        self.last_drag_simulation_time = current_time
        
        # 軽量シミュレーションを実行（ドラッグ中はXYのみ）
        self.run_simulation_silent(only_xy=True)

    def safe_final_simulation_update(self):
        """ドラッグ終了後の安全な最終更新"""
        try:
            # 他のシミュレーションが実行中でなければ更新
            if not self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
                self.run_simulation_silent()
        except Exception as e:
            print(f"[WARNING] Error in final simulation update: {e}")

    def schedule_high_res_simulation(self):
        """
        ドラッグ終了後、一定時間待ってから高解像度シミュレーションを実行する
        """
        # 既存のタイマーが動作中であれば停止
        if hasattr(self, 'high_res_timer'):
            self.high_res_timer.stop()
        
        # 新しいタイマーを設定（1秒後に実行）
        self.high_res_timer = QTimer(self)  # 親ウィンドウを設定
        self.high_res_timer.setSingleShot(True)
        self.high_res_timer.timeout.connect(self.run_high_res_simulation)
        self.high_res_timer.start(1000)  # 1秒待機

    def run_high_res_simulation(self):
        """
        一時的に高解像度でシミュレーションを実行し、その後64x64に戻す
        """
        if not self.interactive_update_check.isChecked():
            return
        
        # UI上の解像度表示は変更せず、内部で高解像度計算を実行
        if hasattr(self, 'user_selected_resolution') and self.user_selected_resolution:
            target_resolution = self.user_selected_resolution
        else:
            target_resolution = "256x256"  # デフォルト高解像度
   
        
        # 高解像度シミュレーションを実行（UI表示は変更しない）
        self.run_simulation_silent_high_res(target_resolution)

    def run_simulation_silent_high_res(self, target_resolution):
        """
        高解像度用の軽量シミュレーション（UI変更は最小限）
        """
        coords, mode = self.get_simulation_coords()
        if coords is None:
            return
        
        # 既に別のシミュレーションが実行中の場合は何もしない
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker'):
            return

        base_coords = coords
        if base_coords is None:
            return

        # UIから目標解像度と最新のスキャンサイズを取得
        try:
            if 'x' in target_resolution:
                target_nx = int(target_resolution.split('x')[0])
                target_ny = int(target_resolution.split('x')[int('x' in target_resolution)]) # logic for potentially split 
                # simplification:
                target_nx = int(target_resolution.split('x')[0])
                target_ny = int(target_resolution.split('x')[1])
            else:
                target_nx = self.spinNx.value()
                target_ny = self.spinNy.value()
        except:
            target_nx = self.spinNx.value()
            target_ny = self.spinNy.value()

        sim_params = {
            'scan_x_nm': self.spinScanXNm.value(),
            'scan_y_nm': self.spinScanYNm.value(),
            'nx': target_nx,
            'ny': target_ny,
            'center_x': self.tip_x_slider.value() / 5.0,
            'center_y': self.tip_y_slider.value() / 5.0,
            'tip_radius': self.tip_radius_spin.value(),
            'minitip_radius': self.minitip_radius_spin.value(),
            'tip_angle': self.tip_angle_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'use_vdw': self.use_vdw_check.isChecked()
        }
        # Backward compatibility
        sim_params['scan_size'] = sim_params['scan_x_nm']
        sim_params['resolution'] = sim_params['nx']

        # チェックされた面の計算タスクを作成（高解像度は選択された面をすべて計算）
        tasks = []
        if self.afm_x_check.isChecked():
            tasks.append({"name": "XY", "panel": self.afm_x_frame, "coords": base_coords})
        if self.afm_y_check.isChecked():
            x_scan = base_coords[:, 1]
            y_scan = base_coords[:, 2]
            z_scan = -base_coords[:, 0]
            tasks.append({"name": "YZ", "panel": self.afm_y_frame, "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)})
        if self.afm_z_check.isChecked():
            x_scan, y_scan, z_scan = base_coords[:, 0], base_coords[:, 2], -base_coords[:, 1]
            tasks.append({"name": "ZX", "panel": self.afm_z_frame, "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)})

        if not tasks:
            return

        # 既存の高解像度ワーカーを停止
        if self.is_worker_running(getattr(self, 'sim_worker_high_res', None), attr_name='sim_worker_high_res'):
            self.stop_worker(self.sim_worker_high_res, timeout_ms=300, allow_terminate=False, worker_name="sim_worker_high_res")
        
        # 高解像度ワーカーを作成
        self.sim_worker_high_res = AFMSimulationWorker(
            self, sim_params, tasks,
            self.atoms_data['element'] if sim_params['use_vdw'] and self.atoms_data is not None else None,
            self.vdw_radii if sim_params['use_vdw'] and hasattr(self, 'vdw_radii') else None,
            silent_mode=True
        )
        self._connect_worker_delete_later(self.sim_worker_high_res)
        self._track_worker_ref('sim_worker_high_res', self.sim_worker_high_res)

        # 完了時に解像度を戻すための特別なハンドラーを接続
        self.sim_worker_high_res.task_done.connect(self.on_task_finished_silent)
        self.sim_worker_high_res.done.connect(self.on_high_res_simulation_finished)
        self.sim_worker_high_res.start()

    def on_high_res_simulation_finished(self, result):
        """高解像度シミュレーション完了処理"""
        # 通常の完了処理
        if self.simulation_results:
            self.save_image_button.setEnabled(True)
            self.save_asd_button.setEnabled(True)
        
        # UI上の解像度表示は変更しない（既に正しい解像度が表示されている）
       

    def on_task_finished_silent(self, z_map, target_panel):
        """軽量シミュレーション用のタスク完了処理（UI変更最小限）"""
        if z_map is not None and target_panel is not None:
            image_key = target_panel.objectName()
            
            # 生データを保存し、表示更新関数を呼び出す
            self.raw_simulation_results[image_key] = z_map
            self.process_and_display_single_image(image_key)

    def on_simulation_finished_silent(self, result):
        """軽量シミュレーション用の完了処理（UI変更なし）"""
        # ★★★ ボタンやプログレスバーの変更は行わない ★★★
        # ★★★ 保存ボタンの有効化のみ行う ★★★
        if self.simulation_results:
            self.save_image_button.setEnabled(True)
            self.save_asd_button.setEnabled(True)

    def run_simulation_interactively(self):
        """
        インタラクティブモード用のシミュレーション実行関数。
        スライダー操作中に連続で実行されないよう、タイマーで遅延させる。
        """
        # PDBデータまたはMRCデータが読み込まれていない、または他のシミュレーションが実行中の場合は何もしない
        if self.atoms_data is None and not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            return
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker'):
            return
        
        # 以前のタイマーが作動中であれば停止する
        if hasattr(self, 'interactive_timer'):
            self.interactive_timer.stop()
        
        # 新しいタイマーを設定
        self.interactive_timer = QTimer(self)  # 親ウィンドウを設定
        self.interactive_timer.setSingleShot(True)  # 一度だけ実行
        # Interactive UpdateがONの時は軽量更新（XYのみ）にする
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            self.interactive_timer.timeout.connect(self.trigger_interactive_simulation)
        else:
            self.interactive_timer.timeout.connect(self.run_simulation)
        self.interactive_timer.start(300)

    def cleanup_threads(self):
        """実行中のスレッドを適切にクリーンアップする（完全版）"""
        try:
            print("Starting thread cleanup...")
            
            # スレッドのリストを作成
            workers = []
            if hasattr(self, 'sim_worker') and self.sim_worker:
                workers.append(('sim_worker', self.sim_worker))
            if hasattr(self, 'sim_worker_silent') and self.sim_worker_silent:
                workers.append(('sim_worker_silent', self.sim_worker_silent))
            if hasattr(self, 'sim_worker_high_res') and self.sim_worker_high_res:
                workers.append(('sim_worker_high_res', self.sim_worker_high_res))
            
            # 各ワーカーを停止
            for worker_name, worker in workers:
                try:
                    print(f"Stopping {worker_name}...")
                    stopped = self.stop_worker(worker, timeout_ms=300, allow_terminate=True, worker_name=worker_name)
                    if stopped:
                        print(f"Stopped {worker_name} gracefully")
                    else:
                        print(f"[WARNING] {worker_name} may still be running")
                        
                except Exception as e:
                    print(f"[WARNING] Error stopping {worker_name}: {e}")
            
            print("Thread cleanup completed")
                
        except Exception as e:
            print(f"[WARNING] Error during thread cleanup: {e}")

    def closeEvent(self, event):
        """ウィンドウが閉じられるときに自動的に呼び出される"""
        try:
            # ★★★ 全タイマーの停止を最初に実行 ★★★
            print("Stopping all timers...")
            timer_attrs = ['rotation_update_timer', 'filter_update_timer', 'interactive_timer', 'high_res_timer']
            for timer_attr in timer_attrs:
                if hasattr(self, timer_attr):
                    timer = getattr(self, timer_attr)
                    if timer:
                        try:
                            timer.stop()
                            timer.deleteLater()  # タイマーを完全に削除
                        except Exception as e:
                            print(f"[WARNING] Failed to stop {timer_attr}: {e}")
            
            # ★★★ スレッドの適切なクリーンアップ（同期的に実行） ★★★
            self.cleanup_threads()
            
            # ヘルプウィンドウを閉じる
            if hasattr(self, 'help_window') and self.help_window:
                try:
                    self.help_window.close()
                except RuntimeError:
                    print("[WARNING] Help window C++ object already deleted")
                except Exception as e:
                    print(f"[WARNING] Failed to close help window: {e}")

            # サブウィンドウを閉じる（メイン終了時に連動）
            for attr_name, label in (
                ('real_afm_window', 'Real AFM window'),
                ('afm_appearance_window', 'AFM Appearance window'),
            ):
                if hasattr(self, attr_name):
                    win = getattr(self, attr_name)
                    if win:
                        try:
                            win.close()
                        except RuntimeError:
                            print(f"[WARNING] {label} C++ object already deleted")
                        except Exception as e:
                            print(f"[WARNING] Failed to close {label}: {e}")
            
            # スタンドアロンアプリケーションなのでwindow_managerは使用しない
            
            # ウィンドウの位置とサイズを保存
            try:
                self.save_geometry()
            except Exception as e:
                print(f"[WARNING] Failed to save geometry: {e}")
            
            # 設定を保存
            try:
                self.save_settings()
            except Exception as e:
                print(f"[WARNING] Failed to save settings: {e}")
            
            # Qtのデフォルトのクローズ処理
            try:
                super().closeEvent(event)
            except RuntimeError:
                print("[WARNING] C++ object already deleted during super().closeEvent()")
            except Exception as e:
                print(f"[WARNING] Failed to call super().closeEvent(): {e}")
            
            event.accept()
            
        except Exception as e:
            print(f"[ERROR] Unexpected error in pyNuD_simulator closeEvent: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生してもイベントは受け入れる
            event.accept()

    def _import_mrc_internal(self, file_path):
        """MRCファイルの読み込み（内部メソッド）"""
        # 必要なライブラリのインポート
        import mrcfile
        from vtkmodules.util import numpy_support
        
        # PDBデータをクリア（MRCファイルimport時）
        self.clear_pdb_data()

        # MRCはVTK表示に切り替え
        self.current_structure_path = file_path
        self.current_structure_type = "mrc"
        self._set_render_backend("vtk")

        # 2. MRCファイル読み込みとボクセルサイズのスケール変換
        with mrcfile.open(file_path, permissive=True) as mrc:
            # 元のデータを保存
            self.mrc_data_original = mrc.data.copy()
            # デフォルトでZ flipを適用（読み込み時にFlipさせて管理）
            self.mrc_data = np.flip(self.mrc_data_original, axis=0).copy()
            
            if mrc.voxel_size.x:
                voxel_size_angstrom = mrc.voxel_size.x 
            else:
                voxel_size_angstrom = 1.0
            self.mrc_voxel_size_nm = voxel_size_angstrom / 10.0
            
        # MRCファイル名を表示
        self.mrc_name = os.path.basename(file_path)
        self.mrc_id = ""
        self.mrc_id = os.path.splitext(self.mrc_name)[0]
        self.file_label.setText(f"File Name: {self.mrc_name} (MRC)")
        
        self.mrc_group.setEnabled(True)
        # Z flipの状態に応じてmrc_surface_coordsを初期化
        self.mrc_surface_coords = self._get_mrc_surface_coords()
        self.update_mrc_display()
        self.simulate_btn.setEnabled(True)
        
        # 回転ウィジェットも有効化
        if hasattr(self, 'rotation_widgets'):
            for axis in ['X', 'Y', 'Z']:
                self.rotation_widgets[axis]['spin'].setEnabled(True)
                self.rotation_widgets[axis]['slider'].setEnabled(True)
        
        # チェックボックスの状態を確実に設定（デフォルトでTrue）
        if hasattr(self, 'mrc_z_flip_check'):
            self.mrc_z_flip_check.blockSignals(True)
            self.mrc_z_flip_check.setChecked(True)
            self.mrc_z_flip_check.blockSignals(False)
            self.mrc_z_flip = True
        
        # 回転状態をリセット（MRCファイル読み込み時）
        self.reset_structure_rotation()
        
        # Interactive Updateが有効な場合は初期シミュレーションを実行
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            self.run_simulation_interactively()

    def on_mrc_threshold_changed(self, value):
        """スライダーの値が変更されたときに呼ばれる（リアルタイム更新用）"""
        # ラベルを更新
        self.mrc_threshold_label.setText(f"Value: {value/100.0:.2f}")
        
        # Interactive Updateが有効な場合は疑似AFM像を自動更新
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            self.mrc_threshold = value / 100.0
            self.run_simulation_interactively()
    
    def on_mrc_threshold_released(self):
        """スライダーが離されたときに呼ばれ、しきい値を更新して再描画する"""
        self.mrc_threshold = self.mrc_threshold_slider.value() / 100.0
        self.update_mrc_display()
        
        # Interactive Updateが有効な場合は疑似AFM像も自動更新
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            self.run_simulation_interactively()

    def on_mrc_z_flip_changed(self, state):
        """Z軸フリップチェックボックスの状態変更時の処理"""
        self.mrc_z_flip = state == Qt.Checked
        
        # mrc_data_originalが存在しない場合は、現在のmrc_dataを元データとして使用
        if not hasattr(self, 'mrc_data_original') or self.mrc_data_original is None:
            if hasattr(self, 'mrc_data') and self.mrc_data is not None:
                # 現在のデータを元データとして保存（初回のみ）
                self.mrc_data_original = self.mrc_data.copy()
            else:
                return
        
        if self.mrc_data_original is not None:
            # フリップ状態変更時に回転状態をリセット（ジャンプを防ぐ）
            self.reset_structure_rotation()
            
            # チェック時：フリップ済みデータを使用（現在の状態を維持）
            # アンチェック時：元のデータを使用（元の向きに戻す）
            if self.mrc_z_flip:
                # チェック時：フリップ済みデータ（読み込み時のデフォルト状態）
                self.mrc_data = np.flip(self.mrc_data_original, axis=0).copy()
            else:
                # アンチェック時：元のデータ（元の向きに戻す）
                self.mrc_data = self.mrc_data_original.copy()
            
            # 座標データを再生成
            self.mrc_surface_coords = self._get_mrc_surface_coords()
            self.update_mrc_display()

    def _get_mrc_surface_coords(self):
        """MRCデータから表面座標を取得する"""
        if not hasattr(self, 'mrc_data') or self.mrc_data is None:
            return None
        
        from vtkmodules.util import numpy_support

        # 現在のフリップ状態に応じたデータを使用
        mask = (self.mrc_data > self.mrc_threshold).astype(np.uint8)

        vtk_data = vtk.vtkImageData()
        depth, height, width = mask.shape
        vtk_data.SetDimensions(width, height, depth)
        vtk_data.SetSpacing(self.mrc_voxel_size_nm, self.mrc_voxel_size_nm, self.mrc_voxel_size_nm)
        vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        flat = mask.transpose(2, 1, 0).flatten()
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_data.GetPointData().SetScalars(vtk_array)

        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_data)
        contour.SetValue(0, 0.5)
        contour.Update()

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(50)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputConnection(smoother.GetOutputPort())
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        center = centerOfMassFilter.GetCenter()

        transform = vtk.vtkTransform()
        transform.Translate(-center[0], -center[1], -center[2])

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(smoother.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        polydata = transformFilter.GetOutput()
        points = polydata.GetPoints()
        if points is not None:
            coords = numpy_support.vtk_to_numpy(points.GetData())  # shape: (N, 3)
            # ジッターを加える
            np.random.seed(42)
            jitter_amplitude = 0.01  # 0.01nmの範囲でジッター
            jitter = np.random.uniform(low=-jitter_amplitude, high=jitter_amplitude, size=coords.shape)
            coords_jittered = coords + jitter
            return coords_jittered
        else:
            return None

    def update_mrc_display(self):
        """現在のしきい値でMRCデータを3D表示する"""
        if self.mrc_data is None:
            return
        self._ensure_vtk_initialized()
        if not hasattr(self, 'renderer') or self.renderer is None:
            print("[WARNING] VTK renderer not initialized for MRC display.")
            return

        from vtkmodules.util import numpy_support

        # 現在のフリップ状態に応じたデータを使用
        mask = (self.mrc_data > self.mrc_threshold).astype(np.uint8)

        vtk_data = vtk.vtkImageData()
        depth, height, width = mask.shape
        vtk_data.SetDimensions(width, height, depth)
        vtk_data.SetSpacing(self.mrc_voxel_size_nm, self.mrc_voxel_size_nm, self.mrc_voxel_size_nm)
        vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        flat = mask.transpose(2, 1, 0).flatten()
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_data.GetPointData().SetScalars(vtk_array)

        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_data)
        contour.SetValue(0, 0.5)
        contour.Update()

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(50)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputConnection(smoother.GetOutputPort())
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        center = centerOfMassFilter.GetCenter()

        transform = vtk.vtkTransform()
        transform.Translate(-center[0], -center[1], -center[2])

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(smoother.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transformFilter.GetOutputPort())
        # スカラーデータの色マッピングを無効にして、アクターの色を使用
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        # MRCは常に選択された色を使用（カラースキームは関係ない）
        #print(f"Setting MRC color to: {self.current_single_color}")
        prop.SetColor(self.current_single_color[0], self.current_single_color[1], self.current_single_color[2])
        #rint(f"Actual MRC color set: {prop.GetColor()}")
        prop.SetOpacity(1.0)
        prop.SetAmbient(0.2)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.4)
        prop.SetSpecularPower(30.0)

        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            self.renderer.RemoveActor(self.mrc_actor)
        self.mrc_actor = actor
        self.renderer.AddActor(actor)
        # 新しいアクターにも現在の回転を適用
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            self.mrc_actor.SetUserTransform(self.molecule_transform)
        # カメラ視点をリセットしない（ResetCamera()を削除）
        self.vtk_widget.GetRenderWindow().Render()

        polydata = transformFilter.GetOutput()
        points = polydata.GetPoints()
        if points is not None:
            coords = numpy_support.vtk_to_numpy(points.GetData())  # shape: (N, 3)
            # ★★★ ジッターを加える ★★★
            np.random.seed(42)
            jitter_amplitude = 0.01  # 0.01nmの範囲でジッター
            jitter = np.random.uniform(low=-jitter_amplitude, high=jitter_amplitude, size=coords.shape)
            coords_jittered = coords + jitter
            self.mrc_surface_coords = coords_jittered
        else:
            self.mrc_surface_coords = None

    def get_simulation_coords(self):
        if hasattr(self, 'mrc_surface_coords') and self.mrc_surface_coords is not None:
            # 回転行列をnumpy配列に変換（combined_transformを使用）
            transform = self.combined_transform.GetMatrix()
            mat = np.array([[transform.GetElement(i, j) for j in range(4)] for i in range(4)])
            coords = self.mrc_surface_coords
            # 同次座標に変換
            coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
            coords_rot = (mat @ coords_h.T).T[:, :3]
            return coords_rot, 'mrc'
        elif self.atoms_data is not None:
            coords = self.get_rotated_atom_coords()
            return coords, 'pdb'
        else:
            return None, None
    
    def get_active_dataset_id(self):
        """
        Return an identifier for current dataset (PDB or MRC).
        優先順位: PDB > MRC > AFM
        """
        if getattr(self, 'pdb_id', ''):
            return self.pdb_id
        if getattr(self, 'mrc_id', ''):
            return self.mrc_id
        return "Unknown"

    def get_active_dataset_type(self):
        if getattr(self, 'pdb_id', ''):
            return "PDB"
        if getattr(self, 'mrc_id', ''):
            return "MRC"
        return "Unknown"   
    
class SaveAFMImageDialog(QDialog):
    """
    Custom dialog to select multiple AFM views and specify incremental rotations
    for export.
    """
    def __init__(self, available_keys, display_names, dataset_id, parent=None):
        super().__init__(parent)
        self.available_keys = available_keys
        self.display_names = display_names
        self.dataset_id = dataset_id
        self.setWindowTitle("Export Simulated AFM Images")
        self.setModal(True)
        self.setMinimumWidth(420)
        self._result = None
        self._build_ui()
    
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)
        
        # Views group
        views_group = QGroupBox("Select Views to Export")
        vg = QVBoxLayout(views_group)
        self.view_checks = {}
        for key in self.available_keys:
            cb = QCheckBox(self.display_names.get(key, key))
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_ok_state)
            self.view_checks[key] = cb
            vg.addWidget(cb)
        if not self.available_keys:
            note = QLabel("No simulated images are available.")
            note.setStyleSheet("color:#b00; font-style:italic;")
            vg.addWidget(note)
        # Select all / none buttons
        btn_row = QHBoxLayout()
        sel_all = QPushButton("All")
        sel_all.clicked.connect(lambda: self._set_all(True))
        sel_none = QPushButton("None")
        sel_none.clicked.connect(lambda: self._set_all(False))
        btn_row.addStretch()
        btn_row.addWidget(sel_all)
        btn_row.addWidget(sel_none)
        vg.addLayout(btn_row)
        main.addWidget(views_group)
        
        # Rotation increments
        rot_group = QGroupBox("Incremental Rotation (°)  (applied once before export)")
        rg = QGridLayout(rot_group)
        self.dx_spin = QDoubleSpinBox(); self._init_rot_spin(self.dx_spin)
        self.dy_spin = QDoubleSpinBox(); self._init_rot_spin(self.dy_spin)
        self.dz_spin = QDoubleSpinBox(); self._init_rot_spin(self.dz_spin)
        rg.addWidget(QLabel("ΔX:"), 0, 0); rg.addWidget(self.dx_spin, 0, 1)
        rg.addWidget(QLabel("ΔY:"), 0, 2); rg.addWidget(self.dy_spin, 0, 3)
        rg.addWidget(QLabel("ΔZ:"), 0, 4); rg.addWidget(self.dz_spin, 0, 5)
        main.addWidget(rot_group)
        
        # Base filename
        base_group = QGroupBox("Filename Base")
        bg = QHBoxLayout(base_group)
        bg.addWidget(QLabel("Base:"))
        self.base_edit = QLineEdit(self.dataset_id if self.dataset_id else "AFM")
        self.base_edit.setPlaceholderText("Base name (dataset id)")
        bg.addWidget(self.base_edit)
        main.addWidget(base_group)
        
        # Example label
        self.example_label = QLabel()
        self.example_label.setStyleSheet("color:#555; font-size:11px;")
        main.addWidget(self.example_label)
        self._update_example()
        for sp in (self.dx_spin, self.dy_spin, self.dz_spin, self.base_edit):
            if isinstance(sp, QDoubleSpinBox):
                sp.valueChanged.connect(self._update_example)
            else:
                sp.textChanged.connect(self._update_example)
        
        # Buttons
        btns = QHBoxLayout()
        btns.addStretch()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.ok_btn)
        btns.addWidget(cancel_btn)
        main.addLayout(btns)
        
        self._update_ok_state()
    
    def _init_rot_spin(self, spin):
        spin.setRange(-360.0, 360.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setValue(0.0)
        spin.setKeyboardTracking(False)
    
    def _set_all(self, state):
        for cb in self.view_checks.values():
            cb.setChecked(state)
        self._update_ok_state()
    
    def _update_ok_state(self):
        any_checked = any(cb.isChecked() for cb in self.view_checks.values())
        self.ok_btn.setEnabled(any_checked)
    
    def _update_example(self):
        base = self.base_edit.text().strip() or "AFM"
        dx = self.dx_spin.value(); dy = self.dy_spin.value(); dz = self.dz_spin.value()
        example = f"Example filename: {base}_XY_dx{dx:+.0f}_dy{dy:+.0f}_dz{dz:+.0f}.png"
        self.example_label.setText(example)
    
    def get_result(self):
        selected = [k for k, cb in self.view_checks.items() if cb.isChecked()]
        return {
            'selected_views': selected,
            'drot': {'x': self.dx_spin.value(), 'y': self.dy_spin.value(), 'z': self.dz_spin.value()},
            'base_name': self.base_edit.text().strip() or "AFM"
        }


class AFMSimulator(pyNuD_simulator):
    """pyNuD plugin variant with Real AFM synchronized to pyNuD selection."""

    def __init__(self, main_window=None):
        self.main_window = main_window
        self._pynud_frame_signal_connected = False
        self._pynud_pending_frame_index = None
        self._pynud_real_refresh_timer = None
        self._pynud_last_file_path = None
        self._pynud_trimmed_real_window_ref = None
        self._pymol_unavailable_warned = False
        super().__init__()
        self.setWindowTitle(PLUGIN_NAME)
        self._connect_main_window_signals()
        self._load_real_afm_from_pynud(frame_index=None, sync=False, show=False)
        if not bool(getattr(self, "pymol_available", False)):
            QTimer.singleShot(0, self._warn_pymol_unavailable)

    def _warn_pymol_unavailable(self):
        if self._pymol_unavailable_warned:
            return
        self._pymol_unavailable_warned = True
        QMessageBox.warning(
            self,
            "PyMOL Unavailable",
            "PyMOL is not available in this environment.\n"
            "PyMOL display modes are disabled.\n"
            "AFMSimulator will run with VTK (interactive) only.",
        )

    def _has_pynud_real_source(self):
        return bool(
            self.main_window is not None
            and hasattr(self.main_window, "getImageDataForFrame")
            and callable(getattr(self.main_window, "getImageDataForFrame", None))
        )

    def _current_pynud_file_path(self):
        try:
            if self.main_window is not None and hasattr(self.main_window, "getCurrentFilePath"):
                path = self.main_window.getCurrentFilePath()
                if path:
                    return str(path)
        except Exception:
            pass
        try:
            if hasattr(gv, "files") and hasattr(gv, "currentFileNum"):
                i = int(getattr(gv, "currentFileNum", -1))
                files = getattr(gv, "files", None)
                if files and 0 <= i < len(files):
                    return str(files[i])
        except Exception:
            pass
        return None

    def _map_scan_direction(self, val):
        try:
            v = int(val)
        except Exception:
            v = 0
        return "R2L" if v == 1 else "L2R"

    def _meta_from_pynud(self, real_image):
        h, w = real_image.shape
        try:
            nx = int(getattr(gv, "XPixel", w) or w)
        except Exception:
            nx = w
        try:
            ny = int(getattr(gv, "YPixel", h) or h)
        except Exception:
            ny = h
        nx = int(w) if w > 0 else nx
        ny = int(h) if h > 0 else ny
        try:
            scan_x_nm = float(getattr(gv, "XScanSize", 0.0) or 0.0)
        except Exception:
            scan_x_nm = 0.0
        try:
            scan_y_nm = float(getattr(gv, "YScanSize", 0.0) or 0.0)
        except Exception:
            scan_y_nm = 0.0
        if scan_x_nm <= 0 and scan_y_nm > 0:
            scan_x_nm = scan_y_nm
        if scan_y_nm <= 0 and scan_x_nm > 0:
            scan_y_nm = scan_x_nm
        return {
            "nx": nx,
            "ny": ny,
            "scan_x_nm": scan_x_nm,
            "scan_y_nm": scan_y_nm,
            "nm_per_pixel_x": (scan_x_nm / float(max(nx, 1))) if scan_x_nm > 0 else 0.0,
            "nm_per_pixel_y": (scan_y_nm / float(max(ny, 1))) if scan_y_nm > 0 else 0.0,
            "scan_direction": self._map_scan_direction(getattr(gv, "ScanDirection", 0)),
        }

    def _load_real_afm_from_pynud(self, frame_index=None, sync=False, show=False):
        if not self._has_pynud_real_source():
            return False
        try:
            if frame_index is None:
                frame_index = int(getattr(gv, "index", 0) or 0)
            else:
                frame_index = int(frame_index)
        except Exception:
            frame_index = 0
        try:
            real = self.main_window.getImageDataForFrame(frame_index, channel="1ch")
        except Exception as e:
            QMessageBox.critical(self, "Real AFM Load Error", f"Failed to load current pyNuD frame:\n{e}")
            return False
        if real is None:
            return False
        arr = np.asarray(real, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return False

        path = self._current_pynud_file_path()
        file_changed = path != self._pynud_last_file_path
        self._pynud_last_file_path = path
        if file_changed:
            self.real_afm_roi_px = None
            self._clear_real_afm_roi_overlay()

        self.real_afm_nm_full = arr
        self.real_meta_full = self._meta_from_pynud(arr)
        self.real_asd_path = path or "<pyNuD-current>"
        try:
            total = int(getattr(gv, "FrameNum", 0) or 0)
        except Exception:
            total = 0
        self.real_asd_frame_num = total
        if total > 0:
            frame_index = max(0, min(frame_index, total - 1))
        else:
            frame_index = max(0, frame_index)
        self.real_asd_frame_index = frame_index

        self._rebuild_real_afm_active_from_full()
        self.sim_aligned_nm = None
        self._ensure_real_afm_window(show=show)
        self.show_real_afm()
        self._update_real_afm_frame_controls()
        if sync:
            self.sync_sim_params_to_real()
        return True

    def _connect_main_window_signals(self):
        if self._pynud_frame_signal_connected:
            return
        if self.main_window is None or not hasattr(self.main_window, "frameChanged"):
            return
        try:
            self.main_window.frameChanged.connect(self._on_main_window_frame_changed)
            self._pynud_frame_signal_connected = True
        except Exception:
            self._pynud_frame_signal_connected = False

    def _disconnect_main_window_signals(self):
        if not self._pynud_frame_signal_connected:
            return
        if self.main_window is None or not hasattr(self.main_window, "frameChanged"):
            self._pynud_frame_signal_connected = False
            return
        try:
            self.main_window.frameChanged.disconnect(self._on_main_window_frame_changed)
        except Exception:
            pass
        self._pynud_frame_signal_connected = False

    def _on_main_window_frame_changed(self, frame_index):
        if not self._has_pynud_real_source():
            return
        if getattr(self, "real_afm_window", None) is None and getattr(self, "real_afm_nm", None) is None:
            return
        try:
            self._pynud_pending_frame_index = int(frame_index)
        except Exception:
            self._pynud_pending_frame_index = 0
        if self._pynud_real_refresh_timer is None:
            self._pynud_real_refresh_timer = QTimer(self)
            self._pynud_real_refresh_timer.setSingleShot(True)
            self._pynud_real_refresh_timer.timeout.connect(self._perform_pynud_real_refresh)
        self._pynud_real_refresh_timer.start(70)

    def _perform_pynud_real_refresh(self):
        frame_index = self._pynud_pending_frame_index
        if frame_index is None:
            return
        self._load_real_afm_from_pynud(frame_index=frame_index, sync=False, show=False)

    def _remove_child_widget(self, parent, object_name):
        if parent is None:
            return
        w = parent.findChild(QWidget, object_name)
        if w is None:
            return
        try:
            p = w.parentWidget()
            if p is not None and p.layout() is not None:
                p.layout().removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        except Exception:
            pass

    def _trim_real_afm_window_for_plugin(self):
        win = getattr(self, "real_afm_window", None)
        if win is None:
            return
        self._remove_child_widget(win, "REAL_AFM_FrameSliderRow")
        self._remove_child_widget(win, "SIM_ALIGNED_FrameControlSpacer")
        self.real_afm_frame_label = None
        self.real_afm_frame_slider = None
        self.real_afm_sim_info_label = None

    def _ensure_real_afm_window(self, show=False):
        super()._ensure_real_afm_window(show=show)
        win = getattr(self, "real_afm_window", None)
        if win is None:
            return
        if self._pynud_trimmed_real_window_ref is win:
            return
        self._trim_real_afm_window_for_plugin()
        self._pynud_trimmed_real_window_ref = win

    def on_load_real_asd(self):
        if self._has_pynud_real_source():
            if not self._load_real_afm_from_pynud(frame_index=None, sync=True, show=True):
                QMessageBox.information(self, "Real AFM", "pyNuDの現在選択ASDを取得できませんでした。")
            return
        super().on_load_real_asd()

    def open_real_afm_image_window(self):
        self._ensure_real_afm_window(show=True)
        if self._has_pynud_real_source():
            self._load_real_afm_from_pynud(frame_index=None, sync=False, show=False)

    def load_real_asd_file(self, filepath, sync=True):
        if self._has_pynud_real_source():
            self._load_real_afm_from_pynud(frame_index=None, sync=sync, show=True)
            return
        super().load_real_asd_file(filepath, sync=sync)

    def load_real_asd_frame(self, frame_index: int):
        if self._has_pynud_real_source():
            self._load_real_afm_from_pynud(frame_index=frame_index, sync=False, show=False)
            return
        super().load_real_asd_frame(frame_index)

    def closeEvent(self, event):
        self._disconnect_main_window_signals()
        try:
            if self.main_window is not None and hasattr(self.main_window, "plugin_actions"):
                action = self.main_window.plugin_actions.get(PLUGIN_NAME)
                if action is not None and hasattr(self.main_window, "setActionHighlight"):
                    self.main_window.setActionHighlight(action, False)
        except Exception:
            pass
        super().closeEvent(event)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setAttribute(Qt.AA_DontUseNativeMenuBar, True)
    app = QApplication(sys.argv)
    vtk.vtkObject.GlobalWarningDisplayOff()
    window = AFMSimulator()
    window.show()
    sys.exit(app.exec_())


def create_plugin(main_window):
    """Plugin entry point. Called from pyNuD Plugin menu."""
    return AFMSimulator(main_window=main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "AFMSimulator", "pyNuD_simulator"]


if __name__ == "__main__":
    main()
