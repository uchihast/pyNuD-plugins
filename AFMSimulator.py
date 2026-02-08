#!/usr/bin/env python3
# type: ignore
"""
AFM Simulator plugin for pyNuD.

This module reuses the full-featured simulator implementation from
`pyNuD_simulator.py` and adapts Real AFM loading to pyNuD's current selection.
"""

import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget

# Support standalone launch: use globalvals when run from pyNuD, else minimal stub
try:
    import globalvals as gv
except ModuleNotFoundError:
    class _GlobalValsStub:
        standardFont = "Helvetica"
        main_window = None
    gv = _GlobalValsStub()

# Ensure workspace root is importable when this file is loaded as a plugin from plugins/
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from pyNuD_simulator import pyNuD_simulator, vtk


PLUGIN_NAME = "AFM Simulator"


class AFMSimulator(pyNuD_simulator):
    """pyNuD plugin variant of the standalone simulator."""

    def __init__(self, main_window=None):
        self.main_window = main_window
        self._pynud_frame_signal_connected = False
        self._pynud_pending_frame_index = None
        self._pynud_real_refresh_timer = None
        self._pynud_last_file_path = None

        super().__init__()
        self.setWindowTitle("AFM Simulator")

        self._connect_main_window_signals()
        # Initialize Real AFM from pyNuD's current file/frame when available.
        self._load_real_afm_from_pynud(frame_index=None, sync=False, show=False)

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

        # Prefer actual image shape for consistency with processed data returned from pyNuD.
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
        """Load Real AFM from pyNuD current ASD selection using processed display data."""
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
        # Avoid unnecessary background refresh when Real AFM UI/state is not used yet.
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

        # Debounce rapid frame updates during playback.
        self._pynud_real_refresh_timer.start(70)

    def _perform_pynud_real_refresh(self):
        frame_index = self._pynud_pending_frame_index
        if frame_index is None:
            return
        self._load_real_afm_from_pynud(frame_index=frame_index, sync=False, show=False)

    def _remove_child_widget(self, parent, object_name):
        """Remove a child widget from layout by objectName."""
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
        """In pyNuD plugin mode, frame UI and meta row are unnecessary."""
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

        # Apply trimming once per currently active window instance.
        win = getattr(self, "real_afm_window", None)
        if win is None:
            return
        if getattr(self, "_pynud_trimmed_real_window_ref", None) is win:
            return

        self._trim_real_afm_window_for_plugin()
        self._pynud_trimmed_real_window_ref = win

    # -------- Real AFM overrides (pyNuD-coupled) --------
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
            # In plugin mode, Real AFM is always sourced from pyNuD current selection.
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

        # Keep Plugin menu highlight behavior consistent with other plugins.
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


__all__ = ["PLUGIN_NAME", "create_plugin", "AFMSimulator"]


if __name__ == "__main__":
    main()
