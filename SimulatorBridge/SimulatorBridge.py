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
    '+rUkHfE+XZt5NEo15+QrVK39GweIkIk='
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


def _pynud_ui_language():
    """pyNuD 本体の表示言語（"ja"/"en"）。言語の仕組みが無い古い本体では None。"""
    try:
        from gui_i18n import get_language, has_language_service
    except ImportError:
        return None
    try:
        return get_language() if has_language_service() else None
    except Exception:
        return None


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
            lambda: touch_state(publisher_id=self._publisher_id,
                                ui_language=_pynud_ui_language()))
        self._heartbeat_timer.start()
        # pyNuD の表示言語を Simulator に伝える（ハートビートでも送るが、切り替えはすぐ届ける）。
        try:
            from gui_i18n import add_language_listener
            add_language_listener(self._on_language_changed)
        except ImportError:
            pass

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
                ui_language=_pynud_ui_language(),
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

    def _on_language_changed(self, language) -> None:
        if not self._closed:
            touch_state(publisher_id=self._publisher_id, ui_language=_pynud_ui_language())

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
