#!/usr/bin/env python3
"""Publish processed pyNuD AFM frames to the standalone pyNuD Simulator.

Single-file plugin: bridge protocol is embedded from simulator_bridge.py.
Regenerate with: python tools/embed_simulator_bridge_plugin.py
"""

from __future__ import annotations

import base64
import importlib.util
import os
import sys
import zlib
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

try:
    import globalvals as gv
except ModuleNotFoundError:
    gv = None  # type: ignore

_PLUGIN_DIR = Path(__file__).resolve().parent
_EMBEDDED_SIMULATOR_BRIDGE_B64 = (
(
    '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'
    'SxitPvkfocMwmw=='
)
)


def _import_simulator_bridge():
    """Load simulator_bridge from an adjacent dev file or the embedded source."""
    module_name = "simulator_bridge"
    adjacent = _PLUGIN_DIR / f"{module_name}.py"
    existing = sys.modules.get(module_name)
    existing_file = getattr(existing, "__file__", None) if existing is not None else None
    if existing is not None and existing_file:
        try:
            if Path(existing_file).resolve() == adjacent.resolve():
                return existing
        except Exception:
            pass

    if adjacent.is_file():
        spec = importlib.util.spec_from_file_location(module_name, adjacent)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    source = zlib.decompress(base64.b64decode(_EMBEDDED_SIMULATOR_BRIDGE_B64)).decode("utf-8")
    module = ModuleType(module_name)
    module.__file__ = str(adjacent)
    module.__loader__ = None
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(source, str(adjacent), "exec"), module.__dict__)
    return module


_bridge = _import_simulator_bridge()
consumer_is_running = _bridge.consumer_is_running
launch_standalone = _bridge.launch_standalone
mark_inactive = _bridge.mark_inactive
meta_from_pynud_image = _bridge.meta_from_pynud_image
publish_frame = _bridge.publish_frame
touch_state = _bridge.touch_state

PLUGIN_NAME = "Simulator Bridge"


class SimulatorBridgeWindow(QtWidgets.QWidget):
    """Lightweight pyNuD plugin that feeds Real AFM frames to pyNuD Simulator."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._bridge_seq = 0
        self._frame_signal_connected = False
        self._processed_signal_connected = False
        self._pending_frame_index: Optional[int] = None
        self._refresh_timer: Optional[QtCore.QTimer] = None
        self._last_file_path: Optional[str] = None
        self._last_label = "-"
        self._last_frame_index = 0

        self.setWindowTitle(PLUGIN_NAME)
        self.resize(420, 180)
        self._build_ui()
        self._connect_main_window_signals()

        self._heartbeat_timer = QtCore.QTimer(self)
        self._heartbeat_timer.setInterval(2000)
        self._heartbeat_timer.timeout.connect(touch_state)
        self._heartbeat_timer.start()

        QtCore.QTimer.singleShot(0, self.refresh_current_frame)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "While this window is open, the processed AFM frame shown in pyNuD "
            "is published to the standalone pyNuD Simulator.\n"
            'Enable "Live sync from pyNuD" in the Simulator to receive it.\n'
            "Frame and image changes are sent automatically. "
            "Close this window to stop publishing."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.status_label = QtWidgets.QLabel("Status: waiting for AFM data")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.launch_button = QtWidgets.QPushButton("Launch pyNuD Simulator")
        self.launch_button.setToolTip(
            "Start the standalone pyNuD Simulator if it is installed."
        )
        self.launch_button.clicked.connect(self._on_launch_standalone)
        layout.addWidget(self.launch_button)

        close_button = QtWidgets.QPushButton("Close")
        close_button.setToolTip("Stop publishing frames to the Simulator.")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    def _has_real_source(self) -> bool:
        return bool(
            self.main_window is not None
            and hasattr(self.main_window, "getImageDataForFrame")
            and callable(getattr(self.main_window, "getImageDataForFrame", None))
            and gv is not None
        )

    def _current_file_path(self) -> Optional[str]:
        try:
            if self.main_window is not None and hasattr(self.main_window, "getCurrentFilePath"):
                path = self.main_window.getCurrentFilePath()
                if path:
                    return str(path)
        except Exception:
            pass
        try:
            if gv is not None and hasattr(gv, "files") and hasattr(gv, "currentFileNum"):
                index = int(getattr(gv, "currentFileNum", -1))
                files = getattr(gv, "files", None)
                if files and 0 <= index < len(files):
                    return str(files[index])
        except Exception:
            pass
        return None

    def _current_frame_index(self, frame_index=None) -> int:
        if frame_index is not None:
            try:
                return max(0, int(frame_index))
            except (TypeError, ValueError):
                pass
        try:
            return max(0, int(getattr(gv, "index", 0) or 0))
        except Exception:
            return 0

    def refresh_current_frame(self, frame_index=None) -> bool:
        if not self._has_real_source():
            self._set_status("Status: pyNuD image API is unavailable.")
            return False

        index = self._current_frame_index(frame_index)
        try:
            real = self.main_window.getImageDataForFrame(index, channel="1ch")
        except Exception as exc:
            self._set_status(f"Status: failed to read frame {index + 1}: {exc}")
            return False
        if real is None:
            self._set_status("Status: no AFM frame is loaded in pyNuD.")
            return False

        arr = np.asarray(real, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            self._set_status("Status: current frame is empty.")
            return False

        path = self._current_file_path()
        if path != self._last_file_path:
            self._last_file_path = path
        label = os.path.basename(path) if path else "pyNuD-current"
        meta = meta_from_pynud_image(arr, gv)
        try:
            self._bridge_seq = publish_frame(
                arr,
                meta,
                label=label,
                frame_index=index,
                seq=self._bridge_seq,
            )
        except Exception as exc:
            self._set_status(f"Status: publish failed: {exc}")
            return False

        self._last_label = label
        self._last_frame_index = index
        consumer = "connected" if consumer_is_running() else "not detected"
        self._set_status(
            f"Status: publishing {label}  frame {index + 1}  "
            f"(bridge seq {self._bridge_seq}, simulator {consumer})"
        )
        return True

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _connect_main_window_signals(self) -> None:
        if self.main_window is None:
            return
        if not self._frame_signal_connected and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.connect(self._on_main_window_frame_changed)
                self._frame_signal_connected = True
            except Exception:
                self._frame_signal_connected = False
        if (
            not self._processed_signal_connected
            and hasattr(self.main_window, "processedImageChanged")
        ):
            try:
                self.main_window.processedImageChanged.connect(
                    self._on_main_window_frame_changed
                )
                self._processed_signal_connected = True
            except Exception:
                self._processed_signal_connected = False

    def _disconnect_main_window_signals(self) -> None:
        if self.main_window is None:
            self._frame_signal_connected = False
            self._processed_signal_connected = False
            return
        if self._frame_signal_connected and hasattr(self.main_window, "frameChanged"):
            try:
                self.main_window.frameChanged.disconnect(
                    self._on_main_window_frame_changed
                )
            except Exception:
                pass
        self._frame_signal_connected = False
        if (
            self._processed_signal_connected
            and hasattr(self.main_window, "processedImageChanged")
        ):
            try:
                self.main_window.processedImageChanged.disconnect(
                    self._on_main_window_frame_changed
                )
            except Exception:
                pass
        self._processed_signal_connected = False

    def _on_main_window_frame_changed(self, frame_index) -> None:
        if not self._has_real_source():
            return
        try:
            self._pending_frame_index = int(frame_index)
        except (TypeError, ValueError):
            self._pending_frame_index = 0
        if self._refresh_timer is None:
            self._refresh_timer = QtCore.QTimer(self)
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.timeout.connect(self._perform_refresh)
        self._refresh_timer.start(70)

    def _perform_refresh(self) -> None:
        if self._pending_frame_index is None:
            return
        self.refresh_current_frame(frame_index=self._pending_frame_index)

    def _on_launch_standalone(self) -> None:
        self.refresh_current_frame()
        if launch_standalone():
            self._set_status(
                "Status: launched pyNuD Simulator. Keep this window open for Live sync."
            )
            return
        QtWidgets.QMessageBox.warning(
            self,
            PLUGIN_NAME,
            "Installed pyNuD Simulator was not found.\n\n"
            "Install the standalone app, or set PYNUD_SIMULATOR_HOME.",
        )

    def closeEvent(self, event) -> None:
        self._disconnect_main_window_signals()
        mark_inactive()
        if (
            self.main_window is not None
            and hasattr(self.main_window, "plugin_actions")
            and hasattr(self.main_window, "setActionHighlight")
        ):
            try:
                action = self.main_window.plugin_actions.get(PLUGIN_NAME)
                if action is not None:
                    self.main_window.setActionHighlight(action, False)
            except Exception:
                pass
        super().closeEvent(event)


def create_plugin(main_window):
    return SimulatorBridgeWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "SimulatorBridgeWindow"]
