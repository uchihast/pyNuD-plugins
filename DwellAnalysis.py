import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import globalvals as gv

PLUGIN_NAME = "Dwell Analysis"

HELP_HTML_EN = """
<h1>Dwell Analysis</h1>
<h2>Overview</h2>
<p>Dwell Analysis quantifies how long a molecule (or feature) stays at a location across frames. You place marks on the main image view, save them per frame (Memorize), then run Finalization to link marks across frames and compute dwell times. Results can be exported as CSV and a dwell-time histogram is displayed.</p>

<h2>Access</h2>
<ul><li><strong>Plugin menu:</strong> Load Plugin... → select <code>plugins/DwellAnalysis.py</code>, then Plugin → Dwell Analysis</li></ul>

<h2>Main Features</h2>
<ul>
    <li><strong>Start Marking / Stop Marking:</strong> Toggle marking mode. When active, left-click on the main image to add a mark. Marks on the current frame are shown in blue (temporary) until you Memorize.</li>
    <li><strong>Memorize:</strong> Saves the current frame's temporary marks. Memorized marks are shown in cyan. Move to another frame and place more marks, then Memorize again.</li>
    <li><strong>Finalization:</strong> Links marks across frames into "molecules" using Mol Radius, computes dwell time per molecule, and shows a dwell-time histogram.</li>
    <li><strong>Export:</strong> Saves the results (molecule IDs, frame lists, dwell times) to a CSV file.</li>
    <li><strong>All Reset:</strong> Clears all marks and finalization results.</li>
</ul>

<h2>Parameters</h2>
<table class="param-table">
    <tr><th>Item</th><th>Description</th></tr>
    <tr><td>Mol Radius (nm)</td><td>Distance threshold in nm used to link the same molecule between consecutive frames. Marks within this distance are treated as the same molecule. Range: 0.1–100.0 nm. Set according to your scan size and expected drift.</td></tr>
    <tr><td>Mark Number</td><td>Number of marks on the current frame (temporary + memorized for this frame).</td></tr>
    <tr><td>Total Number</td><td>Number of molecules after Finalization (updated when you run Finalization).</td></tr>
</table>

<h2>Workflow</h2>
<div class="step"><strong>1:</strong> Load image data in pyNuD and open the Dwell Analysis plugin.</div>
<div class="step"><strong>2:</strong> Set <strong>Mol Radius</strong> (e.g. 5–20 nm depending on pixel size and drift).</div>
<div class="step"><strong>3:</strong> Click <strong>Start Marking</strong>, then left-click on the main image to place marks on the current frame. Click <strong>Memorize</strong> to save this frame's marks.</div>
<div class="step"><strong>4:</strong> Change frame (slider or keyboard), place marks on other frames, and click <strong>Memorize</strong> for each frame you edit.</div>
<div class="step"><strong>5:</strong> Click <strong>Finalization</strong> to link marks across frames and view the dwell-time histogram.</div>
<div class="step"><strong>6:</strong> Use <strong>Export</strong> to save the results to CSV.</div>

<div class="note"><strong>Tip:</strong> Blue markers = temporary (current frame only). Cyan = memorized for that frame. Use All Reset to clear everything and start over. If linking is wrong, adjust Mol Radius and run Finalization again.</div>
"""

HELP_HTML_JA = """
<h1>ドウェル解析</h1>
<h2>概要</h2>
<p>ドウェル解析は、分子（または特徴点）が同じ位置に何フレーム存在していたか（滞在時間）を解析する機能です。メインの画像ビューにマークを付け、フレームごとに Memorize で保存し、Finalization でフレーム間のマークをリンクして滞在時間を計算します。結果は CSV で出力でき、滞在時間のヒストグラムが表示されます。</p>

<h2>アクセス</h2>
<ul><li><strong>プラグインメニュー:</strong> Load Plugin... → <code>plugins/DwellAnalysis.py</code> を選択し、Plugin → Dwell Analysis</li></ul>

<h2>主な機能</h2>
<ul>
    <li><strong>Start Marking / Stop Marking:</strong> マーキングモードのオン/オフ。オンにするとメイン画像を左クリックしてマークを追加できます。Memorize するまで現在フレームのマークは青（仮）で表示されます。</li>
    <li><strong>Memorize:</strong> 現在フレームの仮マークを確定して保存します。確定したマークはシアンで表示されます。別のフレームに移動してマークを付け、また Memorize します。</li>
    <li><strong>Finalization:</strong> フレーム間のマークを Mol Radius に基づいて「分子」としてリンクし、分子ごとの滞在時間を計算してヒストグラムを表示します。</li>
    <li><strong>Export:</strong> 結果（分子ID、フレームリスト、滞在時間）を CSV で保存します。</li>
    <li><strong>All Reset:</strong> すべてのマークと Finalization の結果をクリアします。</li>
</ul>

<h2>パラメータ</h2>
<table class="param-table">
    <tr><th>項目</th><th>説明</th></tr>
    <tr><td>Mol Radius (nm)</td><td>隣接フレーム間で同じ分子とみなす距離の閾値（nm）。この距離以内のマークは同一分子としてリンクされます。0.1–100.0 nm。スキャンサイズやドリフトに合わせて設定してください。</td></tr>
    <tr><td>Mark Number</td><td>現在フレームのマーク数（このフレームの仮＋確定マーク）。</td></tr>
    <tr><td>Total Number</td><td>Finalization 後の分子数。Finalization 実行時に更新されます。</td></tr>
</table>

<h2>ワークフロー</h2>
<div class="step"><strong>1:</strong> pyNuD で画像データを読み込み、Dwell Analysis プラグインを開く。</div>
<div class="step"><strong>2:</strong> <strong>Mol Radius</strong> を設定する（ピクセルサイズやドリフトに応じて 5–20 nm 程度）。</div>
<div class="step"><strong>3:</strong> <strong>Start Marking</strong> をクリックし、メイン画像を左クリックして現在フレームにマークを付ける。<strong>Memorize</strong> でこのフレームのマークを保存。</div>
<div class="step"><strong>4:</strong> フレームを変え（スライダーやキー）、他のフレームにもマークを付けて、編集したフレームごとに <strong>Memorize</strong> をクリック。</div>
<div class="step"><strong>5:</strong> <strong>Finalization</strong> をクリックしてフレーム間をリンクし、滞在時間ヒストグラムを表示。</div>
<div class="step"><strong>6:</strong> <strong>Export</strong> で結果を CSV に保存。</div>

<div class="note"><strong>ヒント:</strong> 青マーカー＝仮（現在フレームのみ）。シアン＝そのフレームで確定済み。All Reset で全てクリアしてやり直せます。リンクがおかしい場合は Mol Radius を調整して再度 Finalization を実行してください。</div>
"""


class DwellAnalysisWindow(QtWidgets.QWidget):
    """
    Dwell Analysis Window / ドウェル解析ウィンドウ
    Migrated from DwellAnalysis.ipf
    """
    def __init__(self, main_window, parent=None):
        super(DwellAnalysisWindow, self).__init__(parent)
        self.parent = main_window
        
        # Register with window manager / ウィンドウ管理システムに登録
        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self, "sub")
        except ImportError:
            pass
            
        # Set as independent window / 独立したウィンドウとして設定
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinMaxButtonsHint)
        self.setWindowTitle("Dwell Analysis")
        
        # UI variables
        self.mol_radius = 5.0 # nm
        self.mark_num = 0
        self.total_num = 0
        
        # Initialization
        if not hasattr(gv, 'dwell_marks'):
            gv.dwell_marks = {} 
        if not hasattr(gv, 'dwell_temp_marks'):
            gv.dwell_temp_marks = []
        if not hasattr(gv, 'dwell_analysis_active'):
            gv.dwell_analysis_active = False
        if not hasattr(gv, 'dwell_mol_radius'):
            gv.dwell_mol_radius = 5.0
        if not hasattr(gv, 'dwell_molecules'):
            gv.dwell_molecules = []
        if not hasattr(gv, 'dwell_last_image_index'):
            gv.dwell_last_image_index = -1
            
        self.setupUI()
        self.restoreWindowSettings()
        
        # Connect to main window frame change signal if available
        if self.parent and hasattr(self.parent, 'frameChanged'):
            self.parent.frameChanged.connect(self.update_for_frame)
            
        self.update_for_frame(getattr(gv, 'index', 0))
        self.update_info()

    def setupUI(self):
        top_layout = QtWidgets.QVBoxLayout(self)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        help_menu = menu_bar.addMenu("Help" if QtCore.QLocale().language() != QtCore.QLocale.Japanese else "ヘルプ")
        manual_action = help_menu.addAction("Manual" if QtCore.QLocale().language() != QtCore.QLocale.Japanese else "マニュアル")
        manual_action.triggered.connect(self.showHelpDialog)
        top_layout.addWidget(menu_bar)

        main_layout = QtWidgets.QVBoxLayout()
        # Control Buttons
        btn_layout = QtWidgets.QGridLayout()
        
        self.start_marking_btn = QtWidgets.QPushButton("Start Marking")
        self.start_marking_btn.setCheckable(True)
        self.start_marking_btn.clicked.connect(self.toggle_marking)
        btn_layout.addWidget(self.start_marking_btn, 0, 0)
        
        self.reset_btn = QtWidgets.QPushButton("All Reset")
        self.reset_btn.clicked.connect(self.all_reset)
        btn_layout.addWidget(self.reset_btn, 0, 1)
        
        self.memorize_btn = QtWidgets.QPushButton("Memorize")
        self.memorize_btn.clicked.connect(self.memorize)
        btn_layout.addWidget(self.memorize_btn, 1, 0)
        
        self.finalization_btn = QtWidgets.QPushButton("Finalization")
        self.finalization_btn.clicked.connect(self.finalize)
        btn_layout.addWidget(self.finalization_btn, 1, 1)

        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.clicked.connect(self.export_results)
        btn_layout.addWidget(self.export_btn, 2, 0, 1, 2) # Span 2 columns
        
        main_layout.addLayout(btn_layout)
        
        # Parameters
        param_layout = QtWidgets.QFormLayout()
        
        self.mol_radius_sb = QtWidgets.QDoubleSpinBox()
        self.mol_radius_sb.setRange(0.1, 100.0)
        self.mol_radius_sb.setValue(gv.dwell_mol_radius)
        self.mol_radius_sb.setSuffix(" nm")
        self.mol_radius_sb.valueChanged.connect(self.on_radius_changed)
        param_layout.addRow("Mol Radius:", self.mol_radius_sb)
        
        self.mark_num_label = QtWidgets.QLabel("0")
        param_layout.addRow("Mark Number:", self.mark_num_label)
        
        self.total_num_label = QtWidgets.QLabel("0")
        param_layout.addRow("Total Number:", self.total_num_label)
        
        main_layout.addLayout(param_layout)
        top_layout.addLayout(main_layout, 1)
        
    def showHelpDialog(self):
        """Help → Manual でマニュアルを表示（日本語/English 切替可能）"""
        dialog = QtWidgets.QDialog(self)
        dialog.setMinimumSize(500, 500)
        dialog.resize(600, 650)
        layout_dlg = QtWidgets.QVBoxLayout(dialog)
        lang_row = QtWidgets.QHBoxLayout()
        lang_row.addWidget(QtWidgets.QLabel("Language / 言語:"))
        btn_ja = QtWidgets.QPushButton("日本語", dialog)
        btn_en = QtWidgets.QPushButton("English", dialog)
        btn_ja.setCheckable(True)
        btn_en.setCheckable(True)
        lang_grp = QtWidgets.QButtonGroup(dialog)
        lang_grp.addButton(btn_ja)
        lang_grp.addButton(btn_en)
        lang_grp.setExclusive(True)
        _BTN_SELECTED = "QPushButton { background-color: #007aff; color: white; font-weight: bold; }"
        _BTN_NORMAL = "QPushButton { background-color: #e5e5e5; color: black; }"
        lang_row.addWidget(btn_ja)
        lang_row.addWidget(btn_en)
        lang_row.addStretch()
        layout_dlg.addLayout(lang_row)
        browser = QtWidgets.QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        css = """
        body { font-size: 15px; line-height: 1.6; }
        .step { margin: 8px 0; padding: 6px 0; font-size: 15px; }
        .note { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 14px; border-radius: 4px; margin: 14px 0; font-size: 15px; }
        h1 { font-size: 22px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { font-size: 18px; color: #2c3e50; margin-top: 18px; }
        ul { padding-left: 24px; font-size: 15px; }
        table.param-table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
        table.param-table th, table.param-table td { border: 1px solid #ddd; padding: 10px 12px; text-align: left; }
        table.param-table th { background-color: #f8f9fa; font-weight: bold; }
        """
        browser.document().setDefaultStyleSheet(css)
        close_btn = QtWidgets.QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)

        def set_lang(use_ja):
            btn_ja.setChecked(use_ja)
            btn_en.setChecked(not use_ja)
            btn_ja.setStyleSheet(_BTN_SELECTED if use_ja else _BTN_NORMAL)
            btn_en.setStyleSheet(_BTN_SELECTED if not use_ja else _BTN_NORMAL)
            if use_ja:
                browser.setHtml("<html><body>" + HELP_HTML_JA.strip() + "</body></html>")
                dialog.setWindowTitle("ドウェル解析 - マニュアル")
                close_btn.setText("閉じる")
            else:
                browser.setHtml("<html><body>" + HELP_HTML_EN.strip() + "</body></html>")
                dialog.setWindowTitle("Dwell Analysis - Manual")
                close_btn.setText("Close")

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        layout_dlg.addWidget(browser)
        layout_dlg.addWidget(close_btn)
        set_lang(False)  # デフォルトは英語
        dialog.exec_()
        
    def restoreWindowSettings(self):
        from helperFunctions import restore_window_geometry
        restore_window_geometry(self, 'DwellAnalysisWindow', 300, 300, 250, 200)

    def closeEvent(self, event):
        gv.dwell_analysis_active = False
        if self.start_marking_btn.isChecked():
            self.start_marking_btn.setChecked(False)
        self.saveWindowSettings()
        # プラグインとして開いた場合のツールバーアクションのハイライトを解除
        try:
            if self.parent and hasattr(self.parent, 'setActionHighlight') and hasattr(self.parent, 'plugin_actions'):
                action = self.parent.plugin_actions.get("Dwell Analysis")
                if action is not None:
                    self.parent.setActionHighlight(action, False)
        except Exception:
            pass
        if self.parent and hasattr(self.parent, 'UpdateDisplayImage'):
            self.parent.UpdateDisplayImage()
            self.parent.showDisplayImage()
        event.accept()

    def saveWindowSettings(self):
        try:
            geometry = self.geometry()
            window_settings = getattr(gv, 'windowSettings', {})
            window_settings['DwellAnalysisWindow'] = {
                'width': geometry.width(),
                'height': geometry.height(),
                'x': geometry.x(),
                'y': geometry.y(),
                'visible': False,
                'title': self.windowTitle(),
                'class_name': 'DwellAnalysisWindow'
            }
            gv.windowSettings = window_settings
            if self.parent and hasattr(self.parent, 'saveAllInitialParams'):
                self.parent.saveAllInitialParams()
        except Exception as e:
            print(f"[WARNING] Failed to save Dwell Analysis window settings: {e}")

    def toggle_marking(self, checked):
        gv.dwell_analysis_active = checked
        if checked:
            self.start_marking_btn.setText("Stop Marking")
            # Ensure we have image data
            if not hasattr(gv, 'aryData') or gv.aryData is None:
                QtWidgets.QMessageBox.warning(self, "No Data", "Please load image data first.")
                self.start_marking_btn.setChecked(False)
                gv.dwell_analysis_active = False
                self.start_marking_btn.setText("Start Marking")
                return
        else:
            self.start_marking_btn.setText("Start Marking")
        
        if self.parent and hasattr(self.parent, 'UpdateDisplayImage'):
            self.parent.UpdateDisplayImage()
            self.parent.showDisplayImage()

    def update_for_frame(self, frame_index):
        """Update temporary marks when frame changes"""
        # If marks exist for this frame, load them
        if frame_index in gv.dwell_marks:
            gv.dwell_temp_marks = list(gv.dwell_marks[frame_index])
        else:
            # If marks exist for the previous frame, carry them over as temporary (Blue)
            if frame_index > 0 and (frame_index - 1) in gv.dwell_marks:
                gv.dwell_temp_marks = list(gv.dwell_marks[frame_index - 1])
            else:
                gv.dwell_temp_marks = []
        
        gv.dwell_last_image_index = frame_index
        self.update_info()
        
        if self.parent and hasattr(self.parent, 'UpdateDisplayImage'):
            self.parent.UpdateDisplayImage()
            self.parent.showDisplayImage()

    def on_radius_changed(self, value):
        gv.dwell_mol_radius = value

    def all_reset(self):
        reply = QtWidgets.QMessageBox.question(self, 'Reset', "Clear all marks and results?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            gv.dwell_marks = {}
            gv.dwell_temp_marks = []
            gv.dwell_molecules = []
            self.update_info()
            if self.parent and hasattr(self.parent, 'UpdateDisplayImage'):
                self.parent.UpdateDisplayImage()
                self.parent.showDisplayImage()

    def memorize(self):
        current_frame = getattr(gv, 'index', 0)
        # Save current temporary marks to the permanent marks for this frame
        gv.dwell_marks[current_frame] = list(gv.dwell_temp_marks)
        
        self.update_info()
        if self.parent and hasattr(self.parent, 'UpdateDisplayImage'):
            self.parent.UpdateDisplayImage()
            self.parent.showDisplayImage()

    def finalize(self):
        """
        Equivalent to proc_ButtonFinal in Igor.
        Builds the global tracking results across all frames.
        """
        if not gv.dwell_marks:
            QtWidgets.QMessageBox.warning(self, "No Data", "No marks to process.")
            return

        sorted_frames = sorted(gv.dwell_marks.keys())
        radius = gv.dwell_mol_radius
        
        # nm per pixel
        dx_nm = getattr(gv, 'XScanSize', 0) / getattr(gv, 'XPixel', 1)
        dy_nm = getattr(gv, 'YScanSize', 0) / getattr(gv, 'YPixel', 1)

        # molecules: list of dicts {frames: [frame_idx], positions: [(x_px, y_px)]}
        molecules = []
        
        for frame in sorted_frames:
            current_marks = gv.dwell_marks[frame]
            
            if not molecules:
                # First frame with marks
                for m in current_marks:
                    molecules.append({'frames': [frame], 'positions': [m]})
                continue
            
            # Match current marks with existing molecules' last positions
            matched_marks = [False] * len(current_marks)
            
            for mol in molecules:
                if mol['frames'][-1] == frame - 1:
                    # Look for a match in current marks
                    last_x_px, last_y_px = mol['positions'][-1]
                    last_x_nm = last_x_px * dx_nm
                    last_y_nm = last_y_px * dy_nm
                    
                    for i, (curr_x_px, curr_y_px) in enumerate(current_marks):
                        if not matched_marks[i]:
                            curr_x_nm = curr_x_px * dx_nm
                            curr_y_nm = curr_y_px * dy_nm
                            
                            dist_nm = np.sqrt((curr_x_nm - last_x_nm)**2 + (curr_y_nm - last_y_nm)**2)
                            if dist_nm <= radius:
                                mol['frames'].append(frame)
                                mol['positions'].append((curr_x_px, curr_y_px))
                                matched_marks[i] = True
                                break
            
            # Unmatched current marks become new molecules
            for i, matched in enumerate(matched_marks):
                if not matched:
                    molecules.append({'frames': [frame], 'positions': [current_marks[i]]})
        
        gv.dwell_molecules = molecules
        self.total_num = len(molecules)
        self.update_info()
        
        # Calculate dwell times
        frame_time = getattr(gv, 'FrameTime', 0) # ms
        if frame_time == 0:
            frame_time = 100 # default fallback
            
        dwell_times = []
        results_str = "Molecules Tracking Finalized:\n"
        for idx, mol in enumerate(molecules):
            dwell_frames = len(mol['frames'])
            dt = dwell_frames * frame_time / 1000.0 # seconds
            dwell_times.append(dt)
            results_str += f"Mol {idx+1}: {dwell_frames} frames ({dt:.2f} s)\n"
            
        print(results_str)
        
        # Show Histogram
        self.show_histogram(dwell_times)
        
        QtWidgets.QMessageBox.information(self, "Finalization", f"Processed {len(molecules)} molecules.")

    def export_results(self):
        """Export the finalized time sequence result to a CSV file."""
        if not hasattr(gv, 'dwell_molecules') or not gv.dwell_molecules:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please run 'Finalization' first to generate results.")
            return

        # Get number of frames
        if hasattr(gv, 'aryData') and gv.aryData is not None:
            num_frames = gv.aryData.shape[0]
        else:
            # Fallback to max frame index in marks
            if gv.dwell_marks:
                num_frames = max(gv.dwell_marks.keys()) + 1
            else:
                num_frames = 1

        # File dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Dwell Analysis Results", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Metadata Row 1
                particle_counts = len(gv.dwell_molecules)
                # Ensure row has at least num_frames columns if needed, 
                # but metadata rows can be shorter. 
                # The user says dimension is (N+2) by frame-number.
                meta1 = ["Total particle:", particle_counts]
                meta1 += [""] * (num_frames - len(meta1))
                writer.writerow(meta1)
                
                # Metadata Row 2
                frame_time_ms = getattr(gv, 'FrameTime', 0)
                frame_rate_sec = frame_time_ms / 1000.0 # interval in seconds
                meta2 = ["Frame rate:", frame_rate_sec, "Unit:", "second"]
                meta2 += [""] * (num_frames - len(meta2))
                writer.writerow(meta2)
                
                # Particle Data Matrix
                for mol in gv.dwell_molecules:
                    # Create a binary sequence for frames
                    row = [0] * num_frames
                    for frame_idx in mol['frames']:
                        if 0 <= frame_idx < num_frames:
                            row[frame_idx] = 1
                    writer.writerow(row)
                    
            QtWidgets.QMessageBox.information(self, "Export Success", f"Results exported to {os.path.basename(file_path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    def show_histogram(self, data):
        """Generate and show a histogram of dwell times"""
        if not data:
            return
            
        import matplotlib.pyplot as plt
        plt.figure("Dwell Time Histogram")
        plt.hist(data, bins='auto', color='skyblue', edgecolor='black')
        plt.title("Dwell Time Distribution")
        plt.xlabel("Dwell Time (s)")
        plt.ylabel("Counts")
        plt.grid(True, alpha=0.3)
        plt.show()

    def update_info(self):
        # Use temp_marks for display info
        self.mark_num = len(gv.dwell_temp_marks)
        self.mark_num_label.setText(str(self.mark_num))
        self.total_num_label.setText(str(len(getattr(gv, 'dwell_molecules', []))))

    def add_mark(self, px, py):
        # Interact with dwell_temp_marks
        found_idx = -1
        for i, (mx, my) in enumerate(gv.dwell_temp_marks):
            dist_px = np.sqrt((mx - px)**2 + (my - py)**2)
            if dist_px < 5: # 5 pixels radius for removal (Desorption)
                found_idx = i
                break
        
        if found_idx >= 0:
            gv.dwell_temp_marks.pop(found_idx)
        else:
            gv.dwell_temp_marks.append((px, py))
            
        self.update_info()
        if self.parent and hasattr(self.parent, 'UpdateDisplayImage'):
            self.parent.UpdateDisplayImage()
            self.parent.showDisplayImage()

    def get_marks(self, frame_index):
        return gv.dwell_marks.get(frame_index, [])


def create_plugin(main_window):
    """プラグインエントリポイント。pyNuD の Plugin メニューから呼ばれる。"""
    return DwellAnalysisWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "DwellAnalysisWindow"]
