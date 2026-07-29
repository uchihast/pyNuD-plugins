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
            "pyNuD で表示中の処理済み AFM フレームを、"
            "standalone pyNuD Simulator へ送ります。\n"
            "Simulator 側で「Live sync from pyNuD」を ON にしてください。\n"
            "Publish または Launch をクリックするとウィンドウを閉じます。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.status_label = QtWidgets.QLabel("Status: waiting for AFM data")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        button_row = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Publish current frame")
        self.refresh_button.clicked.connect(self._on_publish_and_close)
        button_row.addWidget(self.refresh_button)

        self.launch_button = QtWidgets.QPushButton("Launch pyNuD Simulator")
        self.launch_button.clicked.connect(self._on_launch_standalone)
        button_row.addWidget(self.launch_button)
        layout.addLayout(button_row)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    def _close_after_action(self) -> None:
        self.close()

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
        if self._frame_signal_connected:
            return
        if self.main_window is None or not hasattr(self.main_window, "frameChanged"):
            return
        try:
            self.main_window.frameChanged.connect(self._on_main_window_frame_changed)
            self._frame_signal_connected = True
        except Exception:
            self._frame_signal_connected = False

    def _disconnect_main_window_signals(self) -> None:
        if not self._frame_signal_connected:
            return
        if self.main_window is None or not hasattr(self.main_window, "frameChanged"):
            self._frame_signal_connected = False
            return
        try:
            self.main_window.frameChanged.disconnect(self._on_main_window_frame_changed)
        except Exception:
            pass
        self._frame_signal_connected = False

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

    def _on_publish_and_close(self) -> None:
        if self.refresh_current_frame():
            self._close_after_action()

    def _on_launch_standalone(self) -> None:
        if launch_standalone():
            self._close_after_action()
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
