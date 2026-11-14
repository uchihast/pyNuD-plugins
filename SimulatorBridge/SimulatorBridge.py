#!/usr/bin/env python3
"""Publish processed pyNuD AFM frames to the standalone pyNuD Simulator.

Plugin bundle: the bridge protocol remains embedded in this entry module.
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
    from gui_i18n import QtWidgets
except ImportError:  # 古い pyNuD 本体には gui_i18n が無い。そのときは元の Qt の部品で動く。
    pass

try:
    import globalvals as gv
except ModuleNotFoundError:
    gv = None  # type: ignore

_PLUGIN_DIR = Path(__file__).resolve().parent
_EMBEDDED_SIMULATOR_BRIDGE_B64 = (
    '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'
)


def _import_simulator_bridge():
    """Load simulator_bridge from an adjacent dev file or the embedded source."""
    module_name = __package__ + ".simulator_bridge"
    adjacent = _PLUGIN_DIR / "simulator_bridge.py"
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
    module.__package__ = __package__
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
        import uuid
        super().__init__(parent)
        self.main_window = main_window
        self._bridge_seq = 0
        self._publisher_id = uuid.uuid4().hex
        self._closed = False
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
        self._heartbeat_timer.timeout.connect(
            lambda: touch_state(publisher_id=self._publisher_id))
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
        if self._closed:
            return False
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
                publisher_id=self._publisher_id,
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
        if self._closed:
            return
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
        if self._closed or self._pending_frame_index is None:
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
        _bridge.stop_publishing(self)
        self._disconnect_main_window_signals()
        mark_inactive(publisher_id=self._publisher_id)
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
