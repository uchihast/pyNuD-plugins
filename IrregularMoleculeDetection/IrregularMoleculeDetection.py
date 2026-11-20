"""
Irregular Molecule Detection plugin for pyNuD.
不定形分子の検出プラグイン

Detects molecules without a fixed shape (for example mRNA: entangled clumps with
single-stranded tails) in HS-AFM height images and measures area and volume. Thin strands that a plain threshold cannot separate
from substrate roughness are picked up with a ridge filter and hysteresis thresholding, and
the outline is redrawn so that gaps between strands are not counted.
The algorithm lives in molecule_detect_core.py (method "E", chosen by visual comparison on 2026-11-19).

高速 AFM 像から、決まった形を持たない分子（例: mRNA の絡まった塊と一本鎖）を検出し、面積と体積を測る。
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

import csv
import os
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtWidgets as _NativeWidgets  # for text that pyNuD must not translate
from PyQt5.QtWidgets import QWidget as _NativeWidget
try:
    from gui_i18n import QtWidgets
except ImportError:  # 古い pyNuD 本体には gui_i18n が無い。
    pass

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import globalvals as gv

from .molecule_detect_core import DetectionParams, DetectionResult, analyze
from .molecule_i18n import add_language_listener, install as _install_strings, language, tr

_install_strings()  # before any widget is built, so pyNuD's language switch covers this panel

PLUGIN_NAME = "Irregular Molecule Detection"

RESULT_COLUMNS = [
    ("id", "ID"),
    ("area_nm2", "Area (nm²)"),
    ("volume_nm3", "Volume (nm³)"),
    ("max_height_nm", "Max height (nm)"),
    ("mean_height_nm", "Mean height (nm)"),
    ("skeleton_length_nm", "Skeleton length (nm)"),
    ("n_regions", "Regions"),
    ("x_nm", "X (nm)"),
    ("y_nm", "Y (nm)"),
]


def create_plugin(main_window):
    return IrregularMoleculeDetectionWindow(main_window)


def _plugin_version() -> str:
    try:
        import json
        with open(os.path.join(os.path.dirname(__file__), "plugin.json"), encoding="utf-8") as fh:
            return str(json.load(fh).get("version", ""))
    except Exception:
        return ""


class IrregularMoleculeDetectionWindow(QtWidgets.QWidget):
    PARAM_DELAY_MS = 400  # wait after the last parameter change before re-analyzing

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle(PLUGIN_NAME)  # pyNuD appends the manifest version
        self.resize(1100, 720)
        self.result: Optional[DetectionResult] = None
        self.result_frame: int = 0
        self.source_px = (1.0, 1.0)         # pixel size of the source frame (nm); the result has its own
        self.rows: List[Dict] = []          # rows for CSV (current frame or all frames)
        self.rows_scope = ""
        self._signal_connected = False
        self._summary = lambda: "Press Analyze Current Frame."  # rebuilt on language change
        self._tips = []  # (widget, English tooltip) that pyNuD itself does not retranslate
        self._setup_ui()
        self._connect_frame_signal()
        self._refresh_source_label()
        add_language_listener(self._on_language_changed)

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QWidget()
        left.setMaximumWidth(360)
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        self.source_label = QtWidgets.QLabel()
        self.source_label.setWordWrap(True)
        lv.addWidget(self.source_label)

        # --- input
        g_in = QtWidgets.QGroupBox("Input")
        f_in = QtWidgets.QFormLayout(g_in)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("Raw height", "raw")
        self.source_combo.addItem("Processed image", "processed")
        self._tip(self.source_combo, 
            "Raw height: the frame as stored in the file (recommended; the plugin levels it itself).\n"
            "Processed image: the image after pyNuD's filters, if any were applied.")
        f_in.addRow("Data:", self.source_combo)
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItem("Polynomial (cropped image)", "polynomial")
        self.bg_combo.addItem("Top-hat (raw wide field)", "tophat")
        self._tip(self.bg_combo, 
            "Polynomial: quadratic surface + scan-line median fitted outside the molecules.\n"
            "Use it for cropped, already flattened images; volumes of large clumps are preserved.\n"
            "Top-hat: removes structures wider than about twice the radius before detection.\n"
            "Use it for raw wide-field images with bumps or large particles; it also lowers the\n"
            "volume of clumps wider than about twice the radius.")
        f_in.addRow("Background:", self.bg_combo)
        self.tophat_spin = self._dspin(2, 50, 8.0, 0.5, " nm",
                                       "Top-hat radius. Structures wider than about twice this are treated as background.")
        f_in.addRow("Top-hat radius:", self.tophat_spin)
        self.bg_combo.currentIndexChanged.connect(self._update_enabled)
        lv.addWidget(g_in)

        # --- detection
        g_det = QtWidgets.QGroupBox("Detection")
        f_det = QtWidgets.QFormLayout(g_det)
        self.wmin_spin = self._dspin(0.5, 10, 1.0, 0.25, " nm", "Smallest ridge-filter scale (about the strand half-width).")
        self.wmax_spin = self._dspin(0.5, 10, 2.0, 0.25, " nm", "Largest ridge-filter scale.")
        f_det.addRow("Strand scale min:", self.wmin_spin)
        f_det.addRow("Strand scale max:", self.wmax_spin)
        self.klow_spin = self._dspin(0.5, 10, 2.0, 0.25, " σ",
                                     "Low threshold. Pixels above this are kept only if connected to a strong region.\n"
                                     "Lower values extend strands further but add noise.")
        self.khigh_spin = self._dspin(1, 20, 4.0, 0.25, " σ", "High threshold. Regions must contain pixels above this.")
        f_det.addRow("Low threshold:", self.klow_spin)
        f_det.addRow("High threshold:", self.khigh_spin)
        self.core_spin = self._dspin(0, 30, 6.0, 0.5, " σ",
                                     "A molecule must contain a part whose smoothed height exceeds this.\n"
                                     "It removes faint noise blobs. Set 0 to keep every region above the thresholds.")
        f_det.addRow("Core threshold:", self.core_spin)
        self.minarea_spin = self._dspin(1, 5000, 30.0, 5, " nm²", "Objects smaller than this are discarded.")
        f_det.addRow("Min area:", self.minarea_spin)
        lv.addWidget(g_det)

        # --- molecule definition
        g_mol = QtWidgets.QGroupBox("Molecule")
        f_mol = QtWidgets.QFormLayout(g_mol)
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("All molecules", "all")
        self.target_combo.addItem("Largest molecule", "largest")
        self._tip(self.target_combo, "Largest molecule: measure only the main clump (cropped single-molecule images).\n"
                                     "All molecules: measure every detected molecule.")
        f_mol.addRow("Target:", self.target_combo)
        self.group_gap_spin = self._dspin(0, 20, 4.0, 0.5, " nm",
                                          "Detected pieces closer than this are treated as one molecule.")
        f_mol.addRow("Grouping gap:", self.group_gap_spin)
        self.cut_gaps_check = QtWidgets.QCheckBox("Cut gaps between strands")
        self.cut_gaps_check.setChecked(True)
        self.cut_gaps_check.setToolTip("Redraw the outline on the lightly filtered height so that substrate\n"
                                       "between strands is not counted as part of the molecule.")
        f_mol.addRow(self.cut_gaps_check)
        self.island_gap_spin = self._dspin(0, 20, 3.0, 0.5, " nm",
                                           "Islands separated from the main body by at most this distance are candidates.")
        f_mol.addRow("Island gap:", self.island_gap_spin)
        self.island_len_spin = self._dspin(0, 200, 12.0, 1, " nm",
                                           "Candidate islands are kept only if their skeleton is at least this long\n"
                                           "(broken strands). Shorter stubs are dropped.")
        f_mol.addRow("Min island length:", self.island_len_spin)
        self.edge_check = QtWidgets.QCheckBox("Exclude molecules touching the edge")
        self.edge_check.setChecked(True)
        self.edge_check.setToolTip("Molecules cut by the scan edge or the drift-correction border are excluded,\n"
                                   "because their size and volume are incomplete.")
        f_mol.addRow(self.edge_check)
        lv.addWidget(g_mol)

        # --- actions (kept at the top, under the file name, so they are visible without scrolling)
        self.auto_check = QtWidgets.QCheckBox("Update on frame change")
        self.auto_check.setToolTip("Analyze the current frame again whenever the frame changes in pyNuD.")
        self.param_auto_check = QtWidgets.QCheckBox("Update on parameter change")
        self.param_auto_check.setChecked(True)
        self.param_auto_check.setToolTip("Analyze the current frame again whenever a parameter on the left changes.")
        self.analyze_button = QtWidgets.QPushButton("Analyze Current Frame")
        self.analyze_all_button = QtWidgets.QPushButton("Analyze All Frames")
        self.export_button = QtWidgets.QPushButton("Export CSV")
        self.save_image_button = QtWidgets.QPushButton("Save Image")
        self.defaults_button = QtWidgets.QPushButton("Restore Defaults")
        self.manual_button = QtWidgets.QPushButton("User Manual")
        self.manual_button.setToolTip("How to use the plugin and what each parameter means.")
        actions = QtWidgets.QGridLayout()
        actions.addWidget(self.analyze_button, 0, 0)
        actions.addWidget(self.analyze_all_button, 0, 1)
        actions.addWidget(self.auto_check, 1, 0, 1, 2)
        actions.addWidget(self.param_auto_check, 2, 0, 1, 2)
        actions.addWidget(self.export_button, 3, 0)
        actions.addWidget(self.save_image_button, 3, 1)
        actions.addWidget(self.defaults_button, 4, 0)
        actions.addWidget(self.manual_button, 4, 1)
        lv.insertLayout(1, actions)
        self.analyze_button.clicked.connect(self.analyze_current_frame)
        self.analyze_all_button.clicked.connect(self.analyze_all_frames)
        self.export_button.clicked.connect(self.export_csv)
        self.save_image_button.clicked.connect(self.save_image)
        self.defaults_button.clicked.connect(self.restore_defaults)
        self.manual_button.clicked.connect(self.show_manual)
        self.manual_dialog = None
        self._connect_param_signals()
        lv.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # pyNuD's panel scrolling sets every inner scroll area to minimum size 0 and the
        # Ignored size policy. In a plain QHBoxLayout the control panel then collapses
        # to zero width, so the panels sit in a splitter whose sizes are set explicitly.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(scroll)
        root.addWidget(splitter)

        # --- right: image + table
        right_panel = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        view_row = QtWidgets.QHBoxLayout()
        self.outline_check = QtWidgets.QCheckBox("Show outlines")
        self.outline_check.setChecked(True)
        self.ids_check = QtWidgets.QCheckBox("Show IDs")
        self.ids_check.setChecked(True)
        self.vmax_spin = self._dspin(0.1, 100, 2.2, 0.1, " nm", "Upper limit of the colour scale.")
        self.vmax_auto_check = QtWidgets.QCheckBox("Auto")
        self.vmax_auto_check.setChecked(True)
        self.vmax_auto_check.setToolTip("Set the colour range from the image: from -3σ to the height\n"
                                        "that 99.5 % of the molecule pixels stay below.")
        self.vmax_auto_check.toggled.connect(self._on_vmax_auto)
        self.vmax_spin.setEnabled(False)
        self.outline_check.toggled.connect(self._draw)
        self.ids_check.toggled.connect(self._draw)
        self.vmax_spin.valueChanged.connect(self._draw)
        view_row.addWidget(self.outline_check)
        view_row.addWidget(self.ids_check)
        view_row.addWidget(QtWidgets.QLabel("Colour max:"))
        view_row.addWidget(self.vmax_spin)
        view_row.addWidget(self.vmax_auto_check)
        view_row.addStretch(1)
        right.addWidget(self.toolbar)
        right.addLayout(view_row)
        right.addWidget(self.canvas, 3)
        self.summary_label = QtWidgets.QLabel("Press Analyze Current Frame.")
        # No word wrap: a wrapped label makes the panel height-for-width, and pyNuD's scroll
        # wrapper then sizes the panel to its preferred height (600 px canvas) and adds a
        # scrollbar. The summary breaks its own lines; Ignored lets long lines be clipped.
        self.summary_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        right.addWidget(self.summary_label)
        self.table = QtWidgets.QTableWidget(0, len(RESULT_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in RESULT_COLUMNS])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.table, 1)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 710])
        self._update_enabled()

    def show_manual(self) -> None:
        if self.manual_dialog is None:
            self.manual_dialog = ManualDialog(self)
        self.manual_dialog.set_language(language())
        self.manual_dialog.show()
        self.manual_dialog.raise_()
        self.manual_dialog.activateWindow()

    def _dspin(self, lo, hi, val, step, suffix, tip):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(2)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        self._tip(s, tip)
        return s

    def _tip(self, widget, text: str) -> None:
        # pyNuD (2.14) does not retranslate tooltips of combo boxes, or of spin boxes whose
        # suffix stays the same, so the plugin sets these itself.
        self._tips.append((widget, text))
        widget.setToolTip(text)  # keeps the English source if pyNuD does retranslate it
        _NativeWidget.setToolTip(widget, tr(text))

    def _connect_param_signals(self) -> None:
        """Re-analyze the current frame shortly after any detection parameter changes.

        One analysis takes about a second, so changes are collected for PARAM_DELAY_MS
        (typing a value, holding a spin-box arrow, Restore Defaults) and analyzed once.
        """
        self._param_timer = QtCore.QTimer(self)
        self._param_timer.setSingleShot(True)
        self._param_timer.setInterval(self.PARAM_DELAY_MS)
        self._param_timer.timeout.connect(self._on_params_settled)
        for combo in (self.source_combo, self.bg_combo, self.target_combo):
            combo.currentIndexChanged.connect(self._on_param_changed)
        for spin in (self.tophat_spin, self.wmin_spin, self.wmax_spin, self.klow_spin, self.khigh_spin,
                     self.core_spin, self.minarea_spin, self.group_gap_spin, self.island_gap_spin,
                     self.island_len_spin):
            spin.valueChanged.connect(self._on_param_changed)
        for check in (self.cut_gaps_check, self.edge_check):
            check.toggled.connect(self._on_param_changed)
        self.param_auto_check.toggled.connect(self._on_param_changed)

    def _on_param_changed(self, *_):
        if self.sender() is self.tophat_spin and not self.tophat_spin.isEnabled():
            return  # the radius is used only by the top-hat background
        if self.param_auto_check.isChecked():
            self._param_timer.start()  # restarts the delay if it is already running

    def _on_params_settled(self) -> None:
        if not (self.param_auto_check.isChecked() and self.isVisible()):
            return
        if not self.analyze_all_button.isEnabled():  # Analyze All Frames is running
            return
        if not self._current_file_path():
            return
        self.analyze_current_frame(quiet=True)

    def _update_enabled(self, *_):
        self.tophat_spin.setEnabled(self.bg_combo.currentData() == "tophat")

    def restore_defaults(self):
        p = DetectionParams()
        self.bg_combo.setCurrentIndex(0)
        self.source_combo.setCurrentIndex(0)
        self.tophat_spin.setValue(p.tophat_radius_nm)
        self.wmin_spin.setValue(p.strand_width_min_nm)
        self.wmax_spin.setValue(p.strand_width_max_nm)
        self.klow_spin.setValue(p.k_low)
        self.khigh_spin.setValue(p.k_high)
        self.core_spin.setValue(p.core_k)
        self.minarea_spin.setValue(p.min_area_nm2)
        self.target_combo.setCurrentIndex(self.target_combo.findData("largest" if p.largest_only else "all"))
        self.group_gap_spin.setValue(p.group_gap_nm)
        self.cut_gaps_check.setChecked(p.cut_gaps)
        self.island_gap_spin.setValue(p.island_gap_nm)
        self.island_len_spin.setValue(p.min_island_length_nm)
        self.edge_check.setChecked(p.exclude_edge)

    def params(self) -> DetectionParams:
        return DetectionParams(
            background=self.bg_combo.currentData(),
            tophat_radius_nm=self.tophat_spin.value(),
            strand_width_min_nm=min(self.wmin_spin.value(), self.wmax_spin.value()),
            strand_width_max_nm=max(self.wmin_spin.value(), self.wmax_spin.value()),
            k_low=min(self.klow_spin.value(), self.khigh_spin.value()),
            k_high=max(self.klow_spin.value(), self.khigh_spin.value()),
            core_k=self.core_spin.value(),
            min_area_nm2=self.minarea_spin.value(),
            largest_only=self.target_combo.currentData() == "largest",
            group_gap_nm=self.group_gap_spin.value(),
            cut_gaps=self.cut_gaps_check.isChecked(),
            island_gap_nm=self.island_gap_spin.value(),
            min_island_length_nm=self.island_len_spin.value(),
            exclude_edge=self.edge_check.isChecked(),
        )

    # ------------------------------------------------------------------ data access
    def _current_file_path(self) -> str:
        files = getattr(gv, "files", None)
        current = int(getattr(gv, "currentFileNum", 0) or 0)
        if not files or current < 0 or current >= len(files):
            return ""
        return str(files[current])

    def _pixel_size(self):
        xp = int(getattr(gv, "XPixel", 0) or 0)
        yp = int(getattr(gv, "YPixel", 0) or 0)
        xs = float(getattr(gv, "XScanSize", 0.0) or 0.0)
        ys = float(getattr(gv, "YScanSize", 0.0) or 0.0)
        if xp <= 0 or yp <= 0 or xs <= 0:
            raise RuntimeError(tr("The scan size or pixel count of the current file is unknown."))
        return xs / xp, (ys / yp if ys > 0 else xs / xp)

    def _frame_data(self) -> np.ndarray:
        data = None
        if self.source_combo.currentData() == "processed":
            data = getattr(gv, "aryData_processed_1ch", None)
        if data is None:
            data = getattr(gv, "aryData", None)
        if data is None:
            raise RuntimeError(tr("No height data is loaded. Open an ASD file in pyNuD first."))
        return np.array(data, dtype=float)

    def _refresh_source_label(self) -> None:
        path = self._current_file_path()
        if not path:
            self.source_label.setText("No ASD file loaded.")
            return
        xp, yp = int(getattr(gv, "XPixel", 0) or 0), int(getattr(gv, "YPixel", 0) or 0)
        xs, ys = float(getattr(gv, "XScanSize", 0) or 0), float(getattr(gv, "YScanSize", 0) or 0)
        self.source_label.setText(os.path.basename(path) + "\n" + tr(
            "Frame {frame} / {total}, {xp}x{yp} px, {xs:g}x{ys:g} nm ({dx:.3g}x{dy:.3g} nm/px)",
            frame=int(getattr(gv, "index", 0) or 0), total=int(getattr(gv, "FrameNum", 0) or 0),
            xp=xp, yp=yp, xs=xs, ys=ys, dx=xs / xp if xp else 0.0, dy=ys / yp if yp else 0.0))

    def _set_summary(self, build) -> None:
        """Show the summary; build() is called again when the pyNuD language changes."""
        self._summary = build
        self.summary_label.setText(build())

    def _on_language_changed(self, *_):
        for widget, text in self._tips:
            _NativeWidget.setToolTip(widget, tr(text))
        self._refresh_source_label()
        self.summary_label.setText(self._summary())
        self._draw()

    # ------------------------------------------------------------------ frame signal
    def _connect_frame_signal(self):
        mw = self.main_window
        if mw is not None and hasattr(mw, "frameChanged"):
            try:
                mw.frameChanged.connect(self._on_frame_changed)
                self._signal_connected = True
            except Exception:
                self._signal_connected = False

    def _on_frame_changed(self, *_):
        self._refresh_source_label()
        if self.auto_check.isChecked() and self.isVisible():
            self.analyze_current_frame(quiet=True)

    def closeEvent(self, event):
        if self._signal_connected:
            try:
                self.main_window.frameChanged.disconnect(self._on_frame_changed)
            except Exception:
                pass
            self._signal_connected = False
        self._param_timer.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------ analysis
    def analyze_current_frame(self, *_, quiet: bool = False) -> None:
        self._param_timer.stop()  # this analysis already uses the current parameters
        self._refresh_source_label()
        try:
            px, py = self._pixel_size()
            data = self._frame_data()
            res = analyze(data, px, py, self.params())
        except Exception as exc:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, PLUGIN_NAME, str(exc))
            self._set_summary(lambda e=str(exc): tr("Analysis failed: {error}", error=e))
            return
        frame = int(getattr(gv, "index", 0) or 0)
        self.result, self.result_frame, self.source_px = res, frame, (px, py)
        self.rows = [dict(frame=frame, **m) for m in res.molecules]
        self.rows_scope = "current"
        self._show_result()

    def analyze_all_frames(self) -> None:
        path = self._current_file_path()
        frame_num = int(getattr(gv, "FrameNum", 0) or 0)
        if not path or frame_num <= 0:
            QtWidgets.QMessageBox.warning(self, PLUGIN_NAME, tr("Open an ASD file in pyNuD first."))
            return
        try:
            from fileio import InitializeAryDataFallback, LoadFrame
            px, py = self._pixel_size()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, PLUGIN_NAME, str(exc))
            return
        params = self.params()
        self._param_timer.stop()
        progress = QtWidgets.QProgressDialog("Analyzing frames...", "Cancel", 0, frame_num, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(200)
        original_index = int(getattr(gv, "index", 0) or 0)
        rows: List[Dict] = []
        last = None
        self.analyze_all_button.setEnabled(False)
        try:
            for i in range(frame_num):
                progress.setValue(i)
                QtWidgets.QApplication.processEvents()
                if progress.wasCanceled():
                    self._set_summary(lambda: tr("Analysis canceled."))
                    return
                gv.index = i
                LoadFrame(path)
                InitializeAryDataFallback()
                res = analyze(self._frame_data(), px, py, params)
                rows.extend(dict(frame=i, **m) for m in res.molecules)
                if i == original_index:
                    last = res
            progress.setValue(frame_num)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, PLUGIN_NAME, str(exc))
            return
        finally:
            gv.index = original_index
            try:
                LoadFrame(path)
                InitializeAryDataFallback()
            except Exception:
                pass
            self.analyze_all_button.setEnabled(True)
        self.rows, self.rows_scope = rows, "all"
        if last is not None:
            self.result, self.result_frame, self.source_px = last, original_index, (px, py)
        self._show_result(all_frames=frame_num)

    # ------------------------------------------------------------------ display
    def _show_result(self, all_frames: int = 0) -> None:
        res = self.result
        if res is None:
            return
        mols = res.molecules
        frame, n_rows = self.result_frame, len(self.rows)
        vols = np.array([r["volume_nm3"] for r in self.rows], float)

        def build() -> str:
            text = tr("Frame {frame}: {n} molecule(s), background σ = {sigma:.3f} nm",
                      frame=frame, n=len(mols), sigma=res.sigma_nm)
            if res.excluded_edge:
                text += tr(", {n} excluded at the edge", n=res.excluded_edge)
            if abs(res.px_nm - self.source_px[0]) > 1e-9 or abs(res.py_nm - self.source_px[1]) > 1e-9:
                text += "\n" + tr("Non-square pixels: analyzed on {s:.3g} nm square pixels.", s=res.px_nm)
            if all_frames:
                text += "\n" + tr("All {frames} frames: {n} molecule(s)", frames=all_frames, n=n_rows)
                if vols.size:
                    text += tr(", volume mean {mean:.0f} nm³, SD {sd:.0f} nm³", mean=vols.mean(),
                               sd=vols.std(ddof=1) if vols.size > 1 else 0.0)
            return text

        self._set_summary(build)
        self.table.setRowCount(len(mols))
        for r, m in enumerate(mols):
            for c, (key, _) in enumerate(RESULT_COLUMNS):
                v = m[key]
                item = QtWidgets.QTableWidgetItem(str(v) if isinstance(v, int) else f"{v:.2f}")
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        self._draw()

    def _on_vmax_auto(self, auto: bool) -> None:
        self.vmax_spin.setEnabled(not auto)
        self._draw()

    def _colour_range(self):
        """Display range: -3σ to the 99.5th percentile of molecule heights (or the spin box)."""
        res = self.result
        vmin = -3.0 * res.sigma_nm
        if not self.vmax_auto_check.isChecked():
            return vmin, max(self.vmax_spin.value(), vmin + 1e-3)
        inside = res.height[res.labels > 0]
        values = inside if inside.size else res.height[res.valid]
        vmax = float(np.percentile(values, 99.5)) if values.size else 1.0
        vmax = max(vmax, vmin + 10 * res.sigma_nm, 0.1)
        blocker = QtCore.QSignalBlocker(self.vmax_spin)
        self.vmax_spin.setValue(round(vmax, 2))
        del blocker
        return vmin, vmax

    def _draw(self, *_):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        res = self.result
        if res is None:
            self.canvas.draw_idle()
            return
        px, py = res.px_nm, res.py_nm  # pixel size of the analyzed grid (square after resampling)
        h, w = res.height.shape
        ext = (0, w * px, 0, h * py)
        vmin, vmax = self._colour_range()
        im = ax.imshow(res.height, cmap="afmhot", origin="lower", extent=ext,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        # Plot text stays in English, as in pyNuD itself (graph axes are not translated, 2026-11-14).
        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Height (nm)")
        if self.outline_check.isChecked() and res.labels.any():
            xs = (np.arange(w) + 0.5) * px
            ys = (np.arange(h) + 0.5) * py
            ax.contour(xs, ys, (res.labels > 0).astype(float), [0.5], colors="cyan", linewidths=1.0)
        if self.ids_check.isChecked():
            for m in res.molecules:
                ax.text(m["x_nm"], m["y_nm"], str(m["id"]), color="white", fontsize=9,
                        ha="center", va="center", fontweight="bold")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        path = self._current_file_path()
        title = f"frame {self.result_frame}"
        ax.set_title(f"{os.path.basename(path)}  {title}" if path else title, fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ export
    def _default_base(self) -> str:
        path = self._current_file_path()
        return os.path.splitext(path)[0] if path else os.path.join(os.path.expanduser("~"), "molecules")

    def export_csv(self) -> None:
        if not self.rows and self.result is None:
            QtWidgets.QMessageBox.information(self, PLUGIN_NAME, tr("Run an analysis first."))
            return
        suffix = "_molecules_all.csv" if self.rows_scope == "all" else f"_molecules_f{self.result_frame}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("Export CSV"), self._default_base() + suffix,
                                                        tr("CSV files (*.csv)"))
        if not path:
            return
        try:
            self._write_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, PLUGIN_NAME, str(exc))
            return
        previous = self._summary
        self._set_summary(lambda: previous() + "\n" + tr("Saved: {path}", path=path))

    def _write_csv(self, path: str) -> None:
        keys = ["frame"] + [k for k, _ in RESULT_COLUMNS]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["# source", self._current_file_path()])
            w.writerow(["# plugin", f"{PLUGIN_NAME} {_plugin_version()}"])
            for k, v in self.params().to_dict().items():
                w.writerow([f"# {k}", v])
            w.writerow(["# data", self.source_combo.currentData()])
            if self.result is not None:
                w.writerow(["# pixel_size_nm (x, y)", *(f"{v:.6g}" for v in self.source_px)])
                w.writerow(["# analysis_pixel_size_nm (x, y)", f"{self.result.px_nm:.6g}", f"{self.result.py_nm:.6g}"])
            w.writerow(keys)
            for r in self.rows:
                w.writerow([r[k] if isinstance(r[k], int) else f"{r[k]:.4f}" for k in keys])

    def save_image(self) -> None:
        if self.result is None:
            QtWidgets.QMessageBox.information(self, PLUGIN_NAME, tr("Run an analysis first."))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Save Image"), self._default_base() + f"_molecules_f{self.result_frame}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.figure.savefig(path, dpi=200)


class ManualDialog(_NativeWidgets.QDialog):
    """Manual in Japanese or English (manual_ja.html / manual_en.html next to this file).

    It follows pyNuD's language switch and has its own 日本語 / English buttons.
    """

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.resize(760, 680)
        layout = _NativeWidgets.QVBoxLayout(self)
        row = _NativeWidgets.QHBoxLayout()
        self.buttons = {}
        for code, label in (("ja", "日本語"), ("en", "English")):
            button = _NativeWidgets.QPushButton(label)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.clicked.connect(lambda _=False, c=code: self.set_language(c))
            row.addWidget(button)
            self.buttons[code] = button
        row.addStretch(1)
        layout.addLayout(row)
        self.browser = _NativeWidgets.QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.document().setDefaultStyleSheet(
            "table { border-collapse: collapse; } th, td { vertical-align: top; } h2 { margin-top: 18px; }")
        layout.addWidget(self.browser, 1)
        self._pynud_sync_language = self.set_language  # called by pyNuD's language switch
        add_language_listener(self.set_language)
        self.language = ""

    def set_language(self, code: str) -> None:
        code = "ja" if code == "ja" else "en"
        self.buttons[code].setChecked(True)
        if code == self.language:
            return
        self.language = code
        self.setWindowTitle(f"{PLUGIN_NAME}（不定形分子の検出）マニュアル" if code == "ja" else f"{PLUGIN_NAME} Manual")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"manual_{code}.html")
        try:
            with open(path, encoding="utf-8") as fh:
                html = fh.read()
        except OSError as exc:
            html = f"<p>{exc}</p>"
        position = self.browser.verticalScrollBar().value()
        self.browser.setHtml(html)
        self.browser.verticalScrollBar().setValue(position if self.isVisible() else 0)


__all__ = ["PLUGIN_NAME", "create_plugin"]
