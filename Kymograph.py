#!/usr/bin/env python3
"""
Kymograph Module for pyNuD
pyNuD用Kymographモジュール

AFM画像からkymographを作成する機能を提供します。
Provides functionality to create kymographs from AFM images.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import cv2
import pandas as pd
import os
import sys
from scipy.ndimage import rotate, gaussian_filter
from scipy import signal
from scipy.optimize import curve_fit
import warnings

# 粒子検出用のインポート
try:
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
except ImportError:
    peak_local_max = None
    threshold_otsu = None
    print("Warning: skimage features not available for particle detection")

# グローバル変数のインポート
try:
    import globalvals as gv
except ImportError:
    gv = None
    print("Warning: globalvals not available")

# fileio.pyから必要な関数をインポート
try:
    from fileio import LoadFrame, InitializeAryDataFallback
except ImportError:
    LoadFrame = None
    InitializeAryDataFallback = None
    print("Warning: fileio functions not available")

# 追加のエラーハンドリング用の関数
def safe_load_frame(file_path, frame_index):
    """安全なフレーム読み込み関数"""
    try:
        if LoadFrame is None:
            return None
        LoadFrame(file_path)
        InitializeAryDataFallback()
        return True
    except Exception as e:
        print(f"Error loading frame {frame_index}: {e}")
        return False

def safe_get_ary_data():
    """安全な画像データ取得関数"""
    try:
        if gv is None or not hasattr(gv, 'aryData') or gv.aryData is None:
            return None
        return gv.aryData.copy()
    except Exception as e:
        print(f"Error getting aryData: {e}")
        return None


HELP_HTML_EN = """
<h1>Kymograph Analysis</h1>
<h2>Overview</h2>
<p>Extracts intensity along user-defined lines (polylines) and visualizes them as time-distance images (kymographs). Ideal for analyzing fibril elongation/contraction and movement along rails.</p>

<h2>Access</h2>
<ul><li><strong>Plugin menu:</strong> Load Plugin... → select <code>plugins/Kymograph.py</code>, then Plugin → Kymograph</li></ul>

<h2>Main Features</h2>
<ul>
    <li>Draw lines on images and edit to multi-point paths</li>
    <li>Specify rectangular ROI with Δh/Δw for width averaging</li>
    <li>Normalization reduces brightness variations between frames</li>
    <li>Automatic analysis of Fibril end / Particle on kymograph</li>
    <li>Semi-automatic tracking with reference path + search radius</li>
    <li>Display distance-time graphs and CSV export, image saving</li>
</ul>

<h2>Parameters (Excerpt)</h2>
<table class="param-table">
    <tr><th>Item</th><th>Description</th></tr>
    <tr><td>Δh (nm)</td><td>Half-width perpendicular to line (total width = 2×Δh)</td></tr>
    <tr><td>Δw (nm)</td><td>Extra length at both ends of line (line direction)</td></tr>
    <tr><td>Enable Normalization</td><td>Normalize ROI per frame before accumulation</td></tr>
    <tr><td>Gaussian Sigma</td><td>Gradient smoothing (0.5–3.0)</td></tr>
    <tr><td>Min Distance (px)</td><td>Minimum distance between peaks (1–50)</td></tr>
    <tr><td>Threshold Factor</td><td>Otsu threshold multiplier (0.1–2.0)</td></tr>
    <tr><td>Distance/Time Threshold</td><td>Frame-to-frame link distance/gap tolerance</td></tr>
    <tr><td>Use Kalman Filter</td><td>Stabilize links with predictive filtering</td></tr>
    <tr><td>Search Radius</td><td>Search radius around reference path (px)</td></tr>
</table>

<h2>Workflow</h2>
<div class="step"><strong>1:</strong> Draw line and create rectangular ROI with Δh/Δw if needed</div>
<div class="step"><strong>2:</strong> Enable normalization if needed and execute Make Kymograph</div>
<div class="step"><strong>3:</strong> Select Fibril end / Particle and auto-detect (adjust Sigma/Min Distance/Threshold)</div>
<div class="step"><strong>4:</strong> For difficult cases, use reference path + Search Radius for semi-automatic</div>
<div class="step"><strong>5:</strong> Show Graph, set Start Point, Save Data/image</div>

<div class="note"><strong>Tip:</strong> Distance axis depends on scan size. Use Sigma=1.5, Min Distance=5 px, Threshold=1.0 as initial values. Use Kalman Filter for noisy data.</div>
"""

HELP_HTML_JA = """
<h1>キモグラフ解析</h1>
<h2>概要</h2>
<p>線（折れ線）に沿った強度をフレーム方向に積み重ね、時間–距離画像を生成して解析します。線維の伸長/収縮やレール上の移動の可視化・定量に適しています。</p>

<h2>アクセス</h2>
<ul><li><strong>プラグインメニュー:</strong> Load Plugin... → <code>plugins/Kymograph.py</code> を選択し、Plugin → Kymograph</li></ul>

<h2>主な機能</h2>
<ul>
    <li>画像上に線を描画し、複数点パスへ編集</li>
    <li>Δh/Δw で矩形ROIを指定し幅方向に平均化</li>
    <li>正規化（Normalize）でフレーム間の輝度変動を低減</li>
    <li>キモグラフ上で Fibril end / Particle を自動解析</li>
    <li>参照パス + 検索半径で半自動トラッキング</li>
    <li>距離–時間グラフの表示とCSV出力、画像保存</li>
</ul>

<h2>パラメータ（抜粋）</h2>
<table class="param-table">
    <tr><th>項目</th><th>説明</th></tr>
    <tr><td>Δh (nm)</td><td>線に直交する半幅（総幅=2×Δh）</td></tr>
    <tr><td>Δw (nm)</td><td>線の両端の余長（線方向）</td></tr>
    <tr><td>Enable Normalization</td><td>フレーム毎のROIを正規化してから積算</td></tr>
    <tr><td>Gaussian Sigma</td><td>勾配平滑化（0.5–3.0）</td></tr>
    <tr><td>Min Distance (px)</td><td>ピーク間の最小距離（1–50）</td></tr>
    <tr><td>Threshold Factor</td><td>Otsu閾値の倍率（0.1–2.0）</td></tr>
    <tr><td>Distance/Time Threshold</td><td>フレーム間リンク距離/ギャップ許容</td></tr>
    <tr><td>Use Kalman Filter</td><td>予測フィルタでリンクを安定化</td></tr>
    <tr><td>Search Radius</td><td>参照パス周囲の探索半径（px）</td></tr>
</table>

<h2>ワークフロー</h2>
<div class="step"><strong>1:</strong> 線を描き、必要に応じて Δh/Δw で矩形ROIを作成</div>
<div class="step"><strong>2:</strong> 必要なら正規化を有効化し、Make Kymograph を実行</div>
<div class="step"><strong>3:</strong> Fibril end / Particle を選び、自動検出（Sigma/Min Distance/Threshold を調整）</div>
<div class="step"><strong>4:</strong> 難しい場合は参照パス + Search Radius で半自動</div>
<div class="step"><strong>5:</strong> Show Graph、Start Point 設定、Save Data/画像保存</div>

<div class="note"><strong>ヒント:</strong> 距離軸はスキャンサイズに依存します。Sigma=1.5、Min Distance=5 px、Threshold=1.0 を初期値の目安に。ノイズが多ければ Kalman Filter を使用。</div>
"""


PLUGIN_NAME = "Kymograph"


class KymographWindow(QtWidgets.QWidget):
    """
    Kymograph作成ウィンドウ
    AFM画像からkymographを作成するためのGUI
    """
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Kymograph Creator - Kymograph作成")
        self.setWindowFlags(QtCore.Qt.Window)
        
        # ウィンドウ管理システムに登録
        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self, "sub")
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARNING] Failed to register KymographWindow: {e}")
        
        # 閉じられたことを通知するシグナル
        self.was_closed = False
        
        # データ
        self.image_stack = []
        self.current_frame = 0
        self.roi_line = None
        self.roi_rect = None
        self.kymograph_data = None
        
        # パラメータ
        self.line_width = 5
        self.interpolation_method = 'linear'
        
        # 選択モード用の変数
        self.line_selection_mode = False
        self.line_start_point = None
        self.line_end_point = None
        
        # ドラッグ機能用の変数
        self.drag_mode = False
        self.drag_point = None  # 'start' or 'end'
        self.drag_start_pos = None
        self.endpoint_radius = 4  # 端点の検出半径を小さくする
        self.endpoint_visible = False  # 端点の表示フラグ
        
        # Draw Box function variables
        self.dh_nm = 10.0  # Δh (nm)
        self.dw_nm = 0.0   # Δw (nm)
        self.drawn_box = None  # 描画された矩形のデータ
        
        # 正規化設定
        self.normalize_enabled = False  # 正規化の有効/無効
        
        # 実際のフレーム高さを記録
        self.actual_frame_height = None  # 実際のフレーム高さを記録
        
        # 線維成長解析用の変数
        self.fiber_growth_data = None  # 線維成長データ
        self.growth_trajectory = None  # 成長軌跡
        self.growth_speed = None  # 成長速度
        
        # 複数点ライン機能用の変数
        self.multi_point_line = None  # 複数点ラインのデータ
        self.multi_point_mode = False  # 複数点編集モード
        self.selected_point_index = None  # 選択された点のインデックス
        self.point_radius = 6  # 点の検出半径
        
        # デバッグウィンドウ用の変数
        self.debug_window = None
        self.test_window = None
        
        # グラフウィンドウ用の変数を初期化
        self.graph_window = None
        self.graph_figure = None
        self.graph_canvas = None
        
        # Kymographズーム機能の初期化
        self.kymograph_original_limits = None
        
        # セミオート追跡用の変数
        self.semi_auto_mode = False
        self.reference_line_points = []  # 参照線の点 [(x, y), ...]
        self.reference_line_complete = False
        self.search_radius = 10  # 検索半径（ピクセル）
        self.semi_auto_results = None
        self.drawing_reference_line = False
        self.shift_pressed = False  # Shiftキーの状態
        
        # 粒子線描画用の変数
        self.particle_lines = []  # 粒子検出用の複数線を保存
        self.current_line_points = []  # 現在描画中の線の点
        self.is_drawing_particle_line = False  # 粒子線描画中かどうか
        
        # 更新タイマー
        self.update_timer = QtCore.QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.updateImageDisplay)
        
        # メインウィンドウのframeChangedシグナルに接続
        if hasattr(self.main_window, 'frameChanged'):
            self.main_window.frameChanged.connect(self.onFrameChanged)
        
        self.setupUI()
        self.loadWindowSettings()  # ウィンドウ設定を読み込み
        
        # ウィンドウの最小サイズを設定（より自由にリサイズ可能）
        self.setMinimumSize(500, 300)
        
        # 初期画像を表示（遅延実行でラベルのサイズが正しく取得できるように）
        QtCore.QTimer.singleShot(100, self.updateImageDisplay)
        
    def setupUI(self):
        """UIの初期化"""
        top_layout = QtWidgets.QVBoxLayout(self)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        # Help メニュー（ウィンドウ内に表示するため setNativeMenuBar(False)）
        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        help_menu = menu_bar.addMenu("Help" if QtCore.QLocale().language() != QtCore.QLocale.Japanese else "ヘルプ")
        manual_action = help_menu.addAction("Manual" if QtCore.QLocale().language() != QtCore.QLocale.Japanese else "マニュアル")
        manual_action.triggered.connect(self.showHelpDialog)
        top_layout.addWidget(menu_bar)

        # スプリッターを作成
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # 左パネル（コントロール）
        self.left_panel = self.createLeftPanel()
        
        # 中央パネル（画像表示）
        self.center_panel = self.createCenterPanel()
        
        # 右パネル（結果表示）
        self.right_panel = self.createRightPanel()
        
        # スプリッターに各パネルを追加
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.center_panel)
        self.main_splitter.addWidget(self.right_panel)
        
        # スプリッターの初期サイズを設定（より小さなウィンドウに対応）
        self.main_splitter.setSizes([120, 300, 200])
        
        # メニューバーは高さ固定、縦拡大時はスプリッターのみ伸ばす（stretch=1）
        top_layout.addWidget(self.main_splitter, 1)

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
        .feature-box { margin: 12px 0; padding: 14px; border: 1px solid #ccc; border-radius: 4px; font-size: 15px; }
        .step { margin: 8px 0; padding: 6px 0; font-size: 15px; }
        .note { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 14px; border-radius: 4px; margin: 14px 0; font-size: 15px; }
        h1 { font-size: 22px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { font-size: 18px; color: #2c3e50; margin-top: 18px; }
        h3 { font-size: 16px; color: #34495e; margin-top: 12px; }
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
                dialog.setWindowTitle("キモグラフ - マニュアル")
                close_btn.setText("閉じる")
            else:
                browser.setHtml("<html><body>" + HELP_HTML_EN.strip() + "</body></html>")
                dialog.setWindowTitle("Kymograph - Manual")
                close_btn.setText("Close")

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        layout_dlg.addWidget(browser)
        layout_dlg.addWidget(close_btn)
        set_lang(False)  # デフォルトは英語
        dialog.exec_()
        
    def createLeftPanel(self):
        """左パネル（コントロール）の作成"""
        # スクロールエリアを作成
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # メインウィジェット
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)  # マージンを小さく
        
        # Select Targetセクション
        target_group = QtWidgets.QGroupBox("Select Target")
        target_group.setStyleSheet("font-size: 9pt;")
        target_layout = QtWidgets.QVBoxLayout(target_group)
        target_layout.setContentsMargins(5, 5, 5, 5)  # マージンを小さく
        
        # Draw Line button
        self.select_line_button = QtWidgets.QPushButton("Draw Line")
        self.select_line_button.setMaximumWidth(120)  # 横幅を制限
        self.select_line_button.setStyleSheet("font-size: 9pt;")
        self.select_line_button.setToolTip("Draw a line on the image to select target region\n画像上に線を描いて対象領域を選択")
        self.select_line_button.clicked.connect(self.selectLine)
        target_layout.addWidget(self.select_line_button)
        
        # Clear Line button
        self.clear_roi_button = QtWidgets.QPushButton("Clear Line")
        self.clear_roi_button.setMaximumWidth(120)  # 横幅を制限
        self.clear_roi_button.setStyleSheet("font-size: 9pt;")
        self.clear_roi_button.setToolTip("Clear the drawn line\n描いた線をクリア")
        self.clear_roi_button.clicked.connect(self.clearROI)
        target_layout.addWidget(self.clear_roi_button)
        
        layout.addWidget(target_group)
        
        layout.addSpacing(10)  # スペースを小さく
        
        # Draw Box parameters
        box_group = QtWidgets.QGroupBox("Draw Box")
        box_group.setStyleSheet("font-size: 9pt;")
        box_layout = QtWidgets.QVBoxLayout(box_group)
        box_layout.setContentsMargins(5, 5, 5, 5)  # マージンを小さく
        
        # Δh (nm)
        dh_layout = QtWidgets.QHBoxLayout()
        dh_layout.setContentsMargins(0, 0, 0, 0)  # マージンを0に
        dh_layout.setSpacing(5)  # 間隔を小さく
        dh_label = QtWidgets.QLabel("Δh (nm):")
        dh_label.setMinimumWidth(50)  # ラベルの最小幅を設定
        dh_label.setStyleSheet("font-size: 9pt;")
        dh_layout.addWidget(dh_label)
        self.dh_spin = QtWidgets.QDoubleSpinBox()
        self.dh_spin.setRange(1.0, 1000.0)
        self.dh_spin.setValue(10.0)
        self.dh_spin.setDecimals(1)
        self.dh_spin.setMaximumWidth(80)  # スピンボックスの横幅を制限
        self.dh_spin.setStyleSheet("font-size: 9pt;")
        self.dh_spin.setToolTip("Height of the box to draw (nm)\n描くボックスの高さ（nm）")
        self.dh_spin.valueChanged.connect(self.onDhChanged)
        dh_layout.addWidget(self.dh_spin)
        dh_layout.addStretch()  # 右側に伸縮スペースを追加
        box_layout.addLayout(dh_layout)
        
        # Δw (nm)
        dw_layout = QtWidgets.QHBoxLayout()
        dw_layout.setContentsMargins(0, 0, 0, 0)  # マージンを0に
        dw_layout.setSpacing(5)  # 間隔を小さく
        dw_label = QtWidgets.QLabel("Δw (nm):")
        dw_label.setMinimumWidth(50)  # ラベルの最小幅を設定
        dw_label.setStyleSheet("font-size: 9pt;")
        dw_layout.addWidget(dw_label)
        self.dw_spin = QtWidgets.QDoubleSpinBox()
        self.dw_spin.setRange(0.0, 1000.0)
        self.dw_spin.setValue(0.0)
        self.dw_spin.setDecimals(1)
        self.dw_spin.setMaximumWidth(80)  # スピンボックスの横幅を制限
        self.dw_spin.setStyleSheet("font-size: 9pt;")
        self.dw_spin.setToolTip("Width of the box to draw (nm)\n描くボックスの幅（nm）")
        self.dw_spin.valueChanged.connect(self.onDwChanged)
        dw_layout.addWidget(self.dw_spin)
        dw_layout.addStretch()  # 右側に伸縮スペースを追加
        box_layout.addLayout(dw_layout)
        
        # Draw Box button
        self.draw_box_button = QtWidgets.QPushButton("Draw Box")
        self.draw_box_button.setMaximumWidth(120)  # 横幅を制限
        self.draw_box_button.setStyleSheet("font-size: 9pt;")
        self.draw_box_button.setToolTip("Draw a box on the image with specified dimensions\n指定した寸法で画像上にボックスを描く")
        self.draw_box_button.clicked.connect(self.drawBox)
        self.draw_box_button.setEnabled(False)
        box_layout.addWidget(self.draw_box_button)
        
        layout.addWidget(box_group)
        
        layout.addSpacing(10)  # スペースを小さく
        
        # 正規化オプション
        normalize_group = QtWidgets.QGroupBox("Normalization")
        normalize_group.setStyleSheet("font-size: 9pt;")
        normalize_layout = QtWidgets.QVBoxLayout(normalize_group)
        normalize_layout.setContentsMargins(5, 5, 5, 5)  # マージンを小さく
        
        # 正規化チェックボックス
        self.normalize_checkbox = QtWidgets.QCheckBox("Enable Normalization")
        self.normalize_checkbox.setStyleSheet("font-size: 9pt;")
        self.normalize_checkbox.setToolTip("Normalize cropped data before creating kymograph\nキモグラフ作成前にクロップデータを正規化")
        self.normalize_checkbox.stateChanged.connect(self.onNormalizeChanged)
        normalize_layout.addWidget(self.normalize_checkbox)
        
        layout.addWidget(normalize_group)
        
        layout.addSpacing(10)  # スペースを小さく
        
        # 処理ボタン
        self.create_kymograph_button = QtWidgets.QPushButton("Make Kymograph")
        self.create_kymograph_button.setMaximumWidth(140)  # 横幅を少し大きく
        self.create_kymograph_button.setStyleSheet("font-size: 9pt;")
        self.create_kymograph_button.setToolTip("Create kymograph from selected region\n選択した領域からキモグラフを作成")
        self.create_kymograph_button.clicked.connect(self.createKymograph)
        self.create_kymograph_button.setEnabled(False)
        layout.addWidget(self.create_kymograph_button)
        
        # Save button
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setMaximumWidth(120)  # 横幅を制限
        self.save_button.setStyleSheet("font-size: 9pt;")
        self.save_button.setToolTip("Save kymograph image\nキモグラフ画像を保存")
        self.save_button.clicked.connect(self.saveKymograph)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        
        layout.addSpacing(10)  # スペースを小さく
        
        # Analysis セクション
        analysis_group = QtWidgets.QGroupBox("Analysis")
        analysis_group.setStyleSheet("font-size: 9pt;")
        analysis_layout = QtWidgets.QVBoxLayout(analysis_group)
        analysis_layout.setContentsMargins(5, 5, 5, 5)
        
        # detection コンボボックス
        detection_layout = QtWidgets.QHBoxLayout()
        detection_label = QtWidgets.QLabel("detection")
        detection_label.setStyleSheet("font-size: 9pt;")
        detection_layout.addWidget(detection_label)
        
        self.detection_combo = QtWidgets.QComboBox()
        self.detection_combo.addItems(["Fibril end", "Particle"])
        self.detection_combo.setCurrentText("Fibril end")
        self.detection_combo.setStyleSheet("font-size: 9pt;")
        self.detection_combo.setMaximumWidth(120)
        self.detection_combo.setToolTip("Select detection type\n検出タイプを選択")
        detection_layout.addWidget(self.detection_combo)
        detection_layout.addStretch()
        analysis_layout.addLayout(detection_layout)
        
        # Auto detection ボタン（元のAnalyze Growth）
        self.auto_detection_button = QtWidgets.QPushButton("Auto detection")
        self.auto_detection_button.setMaximumWidth(120)
        self.auto_detection_button.setStyleSheet("font-size: 9pt;")
        self.auto_detection_button.clicked.connect(self.analyzeFiberGrowthFromUI)
        self.detection_combo.currentTextChanged.connect(self.onDetectionTypeChanged)
        self.auto_detection_button.setEnabled(False)
        self.auto_detection_button.setToolTip("Analyze fiber growth or detect particles and track them automatically\n線維成長を解析または粒子を検出・追跡")
        analysis_layout.addWidget(self.auto_detection_button)
        
        layout.addWidget(analysis_group)
        
        layout.addSpacing(10)  # スペースを小さく
        
        # 粒子検出パラメータセクション（tracker.pyのLocal Max手法を使用）
        self.particle_params_group = QtWidgets.QGroupBox("Particle Detection Parameters")
        self.particle_params_group.setStyleSheet("font-size: 9pt; background-color: #f0f8ff;")
        particle_params_layout = QtWidgets.QVBoxLayout(self.particle_params_group)
        particle_params_layout.setContentsMargins(5, 5, 5, 5)
        
        # ガウシアンシグマパラメータ
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_label = QtWidgets.QLabel("Gaussian Sigma:")
        sigma_label.setMinimumWidth(100)
        sigma_label.setStyleSheet("font-size: 9pt;")
        self.gradient_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.gradient_sigma_spin.setRange(0.5, 3.0)
        self.gradient_sigma_spin.setValue(1.5)
        self.gradient_sigma_spin.setSingleStep(0.1)
        self.gradient_sigma_spin.setMaximumWidth(60)
        self.gradient_sigma_spin.setStyleSheet("font-size: 9pt;")
        self.gradient_sigma_spin.setToolTip("Gaussian filter sigma value\nSmall values: Preserve fine features\nLarge values: Remove noise and smooth\nガウシアンフィルタのシグマ値\n小さい値: より細かい特徴を保持\n大きい値: ノイズを除去し、滑らかに")
        sigma_layout.addWidget(sigma_label)
        sigma_layout.addWidget(self.gradient_sigma_spin)
        sigma_layout.addStretch()
        particle_params_layout.addLayout(sigma_layout)
        
        # 最小距離パラメータ
        min_distance_layout = QtWidgets.QHBoxLayout()
        min_distance_label = QtWidgets.QLabel("Min Distance (px):")
        min_distance_label.setMinimumWidth(100)
        min_distance_label.setStyleSheet("font-size: 9pt;")
        self.min_peak_distance_spin = QtWidgets.QSpinBox()
        self.min_peak_distance_spin.setRange(1, 50)
        self.min_peak_distance_spin.setValue(5)
        self.min_peak_distance_spin.setMaximumWidth(60)
        self.min_peak_distance_spin.setStyleSheet("font-size: 9pt;")
        self.min_peak_distance_spin.setToolTip("Minimum distance between peaks (pixels)\nSmall values: Detect more peaks\nLarge values: Detect fewer peaks\nピーク間の最小距離（ピクセル）\n小さい値: より多くのピークを検出\n大きい値: より少ないピークを検出")
        min_distance_layout.addWidget(min_distance_label)
        min_distance_layout.addWidget(self.min_peak_distance_spin)
        min_distance_layout.addStretch()
        particle_params_layout.addLayout(min_distance_layout)
        
        # 閾値パラメータ
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_label = QtWidgets.QLabel("Threshold Factor:")
        threshold_label.setMinimumWidth(100)
        threshold_label.setStyleSheet("font-size: 9pt;")
        self.threshold_factor_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_factor_spin.setRange(0.1, 2.0)
        self.threshold_factor_spin.setValue(1.0)
        self.threshold_factor_spin.setSingleStep(0.1)
        self.threshold_factor_spin.setMaximumWidth(60)
        self.threshold_factor_spin.setStyleSheet("font-size: 9pt;")
        self.threshold_factor_spin.setToolTip("Otsu threshold multiplier\nSmall values: Detect more peaks\nLarge values: Detect fewer peaks\nOtsu閾値の倍率\n小さい値: より多くのピークを検出\n大きい値: より少ないピークを検出")
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_factor_spin)
        threshold_layout.addStretch()
        particle_params_layout.addLayout(threshold_layout)
        
        # 初期状態では非表示
        self.particle_params_group.setVisible(False)
        layout.addWidget(self.particle_params_group)
        
        # トラッキングパラメータセクション
        self.tracking_params_group = QtWidgets.QGroupBox("Tracking Parameters")
        self.tracking_params_group.setStyleSheet("font-size: 9pt; background-color: #e6f3ff;")
        tracking_params_layout = QtWidgets.QVBoxLayout(self.tracking_params_group)
        tracking_params_layout.setContentsMargins(5, 5, 5, 5)
        
        # 距離閾値
        distance_layout = QtWidgets.QHBoxLayout()
        distance_label = QtWidgets.QLabel("Distance Threshold:")
        distance_label.setStyleSheet("font-size: 9pt;")
        distance_layout.addWidget(distance_label)
        
        self.distance_threshold_spin = QtWidgets.QSpinBox()
        self.distance_threshold_spin.setRange(1, 50)
        self.distance_threshold_spin.setValue(15)
        self.distance_threshold_spin.setStyleSheet("font-size: 9pt;")
        self.distance_threshold_spin.setMaximumWidth(60)
        self.distance_threshold_spin.setToolTip("Maximum distance between particles in consecutive frames (pixels)\n\nSmall values (1-10): Strict tracking, fewer false connections\nLarge values (20-50): Loose tracking, more connections\nRecommended: 10-20\n\n連続フレーム間の粒子の最大距離（ピクセル）\n\n小さい値（1-10）: 厳密な追跡、誤接続が少ない\n大きい値（20-50）: 緩い追跡、より多くの接続\n推奨: 10-20")
        self.distance_threshold_spin.valueChanged.connect(self.onDistanceThresholdChanged)
        distance_layout.addWidget(self.distance_threshold_spin)
        distance_layout.addStretch()
        tracking_params_layout.addLayout(distance_layout)
        
        # 時間閾値
        time_layout = QtWidgets.QHBoxLayout()
        time_label = QtWidgets.QLabel("Time Threshold:")
        time_label.setStyleSheet("font-size: 9pt;")
        time_layout.addWidget(time_label)
        
        self.time_threshold_spin = QtWidgets.QSpinBox()
        self.time_threshold_spin.setRange(1, 15)
        self.time_threshold_spin.setValue(5)
        self.time_threshold_spin.setStyleSheet("font-size: 9pt;")
        self.time_threshold_spin.setMaximumWidth(60)
        self.time_threshold_spin.setToolTip("Maximum frame gap for tracking (frames)\n\nSmall values (1-3): Short tracks, fewer gaps\nLarge values (5-15): Long tracks, more gaps allowed\nRecommended: 3-8\n\n追跡の最大フレームギャップ（フレーム）\n\n小さい値（1-3）: 短い軌跡、ギャップが少ない\n大きい値（5-15）: 長い軌跡、より多くのギャップを許可\n推奨: 3-8")
        self.time_threshold_spin.valueChanged.connect(self.onTimeThresholdChanged)
        time_layout.addWidget(self.time_threshold_spin)
        time_layout.addStretch()
        tracking_params_layout.addLayout(time_layout)
        
        # Kalman Filter チェックボックス
        kalman_layout = QtWidgets.QHBoxLayout()
        self.kalman_filter_check = QtWidgets.QCheckBox("Use Kalman Filter")
        self.kalman_filter_check.setChecked(True)
        self.kalman_filter_check.setStyleSheet("font-size: 9pt;")
        self.kalman_filter_check.setToolTip("Use Kalman Filter for improved tracking accuracy\n追跡精度を向上させるためにカルマンフィルターを使用")
        self.kalman_filter_check.stateChanged.connect(self.onKalmanFilterChanged)
        kalman_layout.addWidget(self.kalman_filter_check)
        kalman_layout.addStretch()
        tracking_params_layout.addLayout(kalman_layout)
        
        # 初期状態では非表示
        self.tracking_params_group.setVisible(False)
        layout.addWidget(self.tracking_params_group)
        
        # Particle検出結果を管理する変数
        self.particle_detection_results = None
        self.particle_click_radius = 10  # クリック判定の半径（ピクセル）
        
        # 軌跡追跡用の変数
        self.particle_tracks = None  # 追跡された軌跡データ
        self.tracking_distance_threshold = 15  # 軌跡追跡の距離閾値（ピクセル）- 中間値
        self.tracking_time_threshold = 5  # 軌跡追跡の時間閾値（フレーム数）- 中間値
        self.use_trackpy = True  # trackpyを使用するかどうか
        self.use_kalman_filter = True  # Kalman Filterを使用するかどうか
        
        # 始点選択用の変数
        self.selected_start_point = "Blue"  # デフォルトは青線
        
        # グラフウィンドウの位置とサイズを保存する変数
        self.graph_window_geometry = None
        
        # 複数線描画用の変数
        self.particle_lines = []  # 粒子検出用の複数線を保存
        self.current_line_points = []  # 現在描画中の線の点
        self.is_drawing_particle_line = False  # 粒子線描画中かどうか
        

        
        # セミオートマチック追跡セクション
        semi_auto_group = QtWidgets.QGroupBox("Semi-Automatic Tracking")
        if (gv.darkmode == True):
            pass
        else:
            semi_auto_group.setStyleSheet("font-size: 9pt; background-color: #f0f0f0;")
        semi_auto_layout = QtWidgets.QVBoxLayout(semi_auto_group)
        semi_auto_layout.setContentsMargins(5, 5, 5, 5)
        
        # 描画ボタン（共通）
        self.draw_line_button = QtWidgets.QPushButton("Draw Line")
        self.draw_line_button.setMaximumWidth(140)
        self.draw_line_button.setStyleSheet("font-size: 9pt;")
        self.draw_line_button.setToolTip("Draw a line for semi-automatic detection\n半自動検出用の線を描く")
        self.draw_line_button.clicked.connect(self.startDrawingLine)
        self.draw_line_button.setEnabled(True)  # デフォルトで有効化
        semi_auto_layout.addWidget(self.draw_line_button)
        
        # 検索半径設定
        radius_layout = QtWidgets.QHBoxLayout()
        radius_label = QtWidgets.QLabel("Search Radius:")
        radius_label.setMinimumWidth(80)
        radius_label.setStyleSheet("font-size: 9pt;")
        self.radius_spin = QtWidgets.QSpinBox()
        self.radius_spin.setRange(3, 50)
        self.radius_spin.setValue(10)
        self.radius_spin.setSuffix(" px")
        self.radius_spin.setMaximumWidth(70)
        self.radius_spin.setStyleSheet("font-size: 9pt;")
        self.radius_spin.setToolTip("Search radius around the drawn line (pixels)\n描いた線の周囲の検索半径（ピクセル）")
        self.radius_spin.valueChanged.connect(self.onRadiusChanged)
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_spin)
        radius_layout.addStretch()
        semi_auto_layout.addLayout(radius_layout)
        
        # コントロールボタン
        control_layout = QtWidgets.QHBoxLayout()
        
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.setMaximumWidth(60)
        self.clear_button.setStyleSheet("font-size: 9pt;")
        self.clear_button.setToolTip("Clear all drawn lines\n描いた線をすべてクリア")
        self.clear_button.clicked.connect(self.clearLines)
        self.clear_button.setEnabled(True)  # デフォルトで有効化
        control_layout.addWidget(self.clear_button)
        
        self.auto_detect_button = QtWidgets.QPushButton("Auto Detect")
        self.auto_detect_button.setMaximumWidth(80)
        self.auto_detect_button.setStyleSheet("font-size: 9pt;")
        self.auto_detect_button.setToolTip("Perform automatic detection along drawn lines\n描いた線に沿って自動検出を実行")
        self.auto_detect_button.clicked.connect(self.performSemiAutoDetection)
        self.auto_detect_button.setEnabled(False)
        control_layout.addWidget(self.auto_detect_button)
        
        semi_auto_layout.addLayout(control_layout)
        
        # 検出設定
        detection_layout = QtWidgets.QHBoxLayout()
        method_label = QtWidgets.QLabel("Method:")
        method_label.setMinimumWidth(50)
        method_label.setStyleSheet("font-size: 9pt;")
        self.detection_method_combo = QtWidgets.QComboBox()
        self.detection_method_combo.addItems(["Edge Detector", "Max Intensity", "Centroid"])
        self.detection_method_combo.setCurrentText("Edge Detector")
        self.detection_method_combo.setMaximumWidth(100)
        self.detection_method_combo.setStyleSheet("font-size: 9pt;")
        detection_layout.addWidget(method_label)
        detection_layout.addWidget(self.detection_method_combo)
        detection_layout.addStretch()
        semi_auto_layout.addLayout(detection_layout)
        
        # 説明テキスト
        self.help_label = QtWidgets.QLabel("1. Draw approximate fiber edge line\n2. Auto detect precise positions")
        self.help_label.setStyleSheet("color: blue; font-size: 9pt;")
        self.help_label.setWordWrap(True)
        semi_auto_layout.addWidget(self.help_label)
        
        layout.addWidget(semi_auto_group)
        
        # Show Graph と Save Data ボタン
        bottom_layout = QtWidgets.QHBoxLayout()
        
        self.show_graph_button = QtWidgets.QPushButton("Show Graph")
        self.show_graph_button.setMaximumWidth(80)
        self.show_graph_button.setStyleSheet("font-size: 9pt;")
        self.show_graph_button.setEnabled(False)
        self.show_graph_button.clicked.connect(self.showGraph)  # メソッド名を変更
        bottom_layout.addWidget(self.show_graph_button)
        
        self.save_data_button = QtWidgets.QPushButton("Save Data")
        self.save_data_button.setMaximumWidth(80)
        self.save_data_button.setStyleSheet("font-size: 9pt;")
        self.save_data_button.setEnabled(False)
        self.save_data_button.clicked.connect(self.saveGraphData)
        bottom_layout.addWidget(self.save_data_button)
        
        # 始点選択セクション
        start_point_layout = QtWidgets.QHBoxLayout()
        start_point_label = QtWidgets.QLabel("Start Point:")
        start_point_label.setStyleSheet("font-size: 9pt;")
        start_point_layout.addWidget(start_point_label)
        
        self.start_point_combo = QtWidgets.QComboBox()
        self.start_point_combo.addItems(["Blue", "Red"])
        self.start_point_combo.setCurrentText("Blue")
        self.start_point_combo.setStyleSheet("font-size: 9pt;")
        self.start_point_combo.setMaximumWidth(60)
        self.start_point_combo.setToolTip("Select reference line color for distance calculation")
        self.start_point_combo.currentTextChanged.connect(self.onStartPointChanged)
        start_point_layout.addWidget(self.start_point_combo)
        start_point_layout.addStretch()
        layout.addLayout(start_point_layout)
        
        layout.addLayout(bottom_layout)
        
        layout.addStretch()
        
        # スクロールエリアにパネルを設定
        scroll_area.setWidget(panel)
        
        return scroll_area
        
    def createCenterPanel(self):
        """中央パネル（画像表示）の作成"""
        # メインウィジェット
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 画像表示用のQLabel
        self.image_label = QtWidgets.QLabel()
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 150)  # より小さな最小サイズ
        self.image_label.setStyleSheet("background-color: white; border: 1px solid gray;")
        self.image_label.mousePressEvent = self.imageMousePressEvent
        self.image_label.mouseMoveEvent = self.imageMouseMoveEvent
        self.image_label.mouseReleaseEvent = self.imageMouseReleaseEvent
        
        # 右クリックメニューとキーボードショートカットを有効にする
        self.image_label.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.showImageContextMenu)
        self.image_label.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.image_label.keyPressEvent = self.onImageKeyPress
        
        layout.addWidget(self.image_label)
        
        # パネルを直接返す（スクロールエリアは使用しない）
        return panel
        
    def createRightPanel(self):
        """右パネル（結果表示）の作成"""
        # スクロールエリアを作成
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # メインウィジェット
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        
        # 結果表示用のmatplotlibキャンバス
        self.result_figure = Figure(figsize=(4, 6))  # より小さなサイズ
        self.result_canvas = FigureCanvas(self.result_figure)
        
        # マウス操作を有効にする
        self.result_canvas.mpl_connect('button_press_event', self.onKymographMousePress)
        self.result_canvas.mpl_connect('button_release_event', self.onKymographMouseRelease)
        self.result_canvas.mpl_connect('motion_notify_event', self.onKymographMouseMove)
        self.result_canvas.mpl_connect('scroll_event', self.onKymographScroll)
        

        
        # 右クリックメニュー用のコンテキストメニューイベントを追加
        self.result_canvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.result_canvas.customContextMenuRequested.connect(self.showKymographContextMenuQt)
        
        # キーボードショートカットを有効にする
        self.result_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.result_canvas.keyPressEvent = self.onKymographKeyPress
        
        layout.addWidget(self.result_canvas)
        
        # 情報表示
        self.info_label = QtWidgets.QLabel("Please draw a line")
        layout.addWidget(self.info_label)
        
        # スクロールエリアにパネルを設定
        scroll_area.setWidget(panel)
        
        return scroll_area
        
    def selectLine(self):
        """線選択モードを開始"""
        # 他の選択モードを終了
        self.line_selection_mode = False
        self.line_start_point = None
        self.line_end_point = None
        
        # 線選択モードを開始
        self.line_selection_mode = True
        self.image_label.setCursor(QtCore.Qt.CrossCursor)
        
        # 情報を更新
        self.updateInfo()
            
    def onLineSelected(self, line_data):
        """線が選択された時の処理"""
        self.roi_line = line_data
        self.roi_rect = None
        self.updateInfo()
        self.create_kymograph_button.setEnabled(True)
        # Draw Boxボタンの状態はupdateInfo()内のupdateDrawBoxButtonState()で管理
        
        # 線が選択された後、画像を再表示して端点を表示
        self.updateImageDisplay()
    
    def finishLineSelection(self):
        """線選択完了処理"""
        if self.line_start_point and self.line_end_point:
            # 画像座標に変換
            start_x, start_y = self.widgetToImageCoords(self.line_start_point)
            end_x, end_y = self.widgetToImageCoords(self.line_end_point)
            
            # 線データを作成
            line_data = {
                'start_x': start_x,
                'start_y': start_y,
                'end_x': end_x,
                'end_y': end_y
            }
            
            # 複数点ラインを初期化（直線として）
            self.multi_point_line = {
                'points': [(start_x, start_y), (end_x, end_y)],
                'is_straight': True
            }
            
            # 選択モードを終了
            self.line_selection_mode = False
            self.image_label.unsetCursor()
            
            # 内部で処理
            self.onLineSelected(line_data)
            
            # 情報を更新
            self.updateInfo()
        
    def clearROI(self):
        """ROIをクリア"""
        # 選択モードを終了
        self.line_selection_mode = False
        self.line_start_point = None
        self.line_end_point = None
        
        # ドラッグモードをクリア
        self.drag_mode = False
        self.drag_point = None
        self.drag_start_pos = None
        
        # ROIデータをクリア
        self.roi_line = None
        self.roi_rect = None
        self.drawn_box = None  # 描画された矩形もクリア
        
        # 複数点ラインもクリア
        self.multi_point_line = None
        self.multi_point_mode = False
        self.selected_point_index = None
        
        # カーソルをリセット
        self.image_label.unsetCursor()
        
        # 情報を更新
        self.updateInfo()
        self.create_kymograph_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.draw_box_button.setEnabled(False)  # Disable Draw Box button
        
        # 画像を再表示（選択表示をクリア）
        self.updateImageDisplay()
        
    def onFrameChanged(self, frame_index):
        """メインウィンドウのフレーム変更時に呼ばれる"""
        # 画像表示を更新
        self.updateImageDisplay()
        
    def updateInfo(self):
        """情報表示を更新"""
        info_text = ""
        if self.roi_line:
            info_text += "Line is selected"
        else:
            info_text += "Please draw a line"
        
        # 正規化の状態を追加
        if self.normalize_enabled:
            info_text += " [Normalized]"
        
        self.info_label.setText(info_text)
        
        # Draw Boxボタンの状態を更新
        self.updateDrawBoxButtonState()
    
    def updateDrawBoxButtonState(self):
        """Draw Boxボタンの状態を更新"""
        if hasattr(self, 'draw_box_button'):
            # 直線が選択されており、複数点ラインが直線モードの場合のみ有効
            if (self.roi_line and 
                self.multi_point_line and 
                self.multi_point_line['is_straight']):
                self.draw_box_button.setEnabled(True)
            else:
                self.draw_box_button.setEnabled(False)
    
    def updateImageDisplay(self):
        """AFM画像を表示"""
        # 安全なデータ取得
        image_data = safe_get_ary_data()
        if image_data is None:
            self.image_label.setText("画像データがありません")
            return
        
        try:
            # 画像の上下を反転（AFM画像の正しい向きに修正）
            # 注意: この反転により、座標変換でY座標の反転は不要になる
            image_data = np.flipud(image_data)
            
            # 画像の正規化
            if image_data.max() != image_data.min():
                normalized_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
            else:
                normalized_data = image_data * 255
            
            # uint8に変換
            image_uint8 = normalized_data.astype(np.uint8)
            
            # OpenCVのBGR形式に変換（グレースケール）
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
            
            # QImageに変換
            height, width, channel = image_bgr.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image_bgr.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            q_image = q_image.rgbSwapped()  # BGR to RGB
            
            # QPixmapに変換して表示
            pixmap = QtGui.QPixmap.fromImage(q_image)
            
            # 物理サイズのアスペクト比を取得（必須）
            physical_aspect_ratio = self._get_physical_aspect_ratio()
            if physical_aspect_ratio == 1.0:
                # 物理サイズ情報が利用できない場合の警告
                if not hasattr(gv, 'XScanSize') or not hasattr(gv, 'YScanSize') or gv.YScanSize <= 0:
                    QtWidgets.QMessageBox.critical(
                        self, 
                        "エラー", 
                        "物理サイズ情報が利用できません。\n"
                        "XScanSizeまたはYScanSizeが設定されていないか、無効な値です。\n"
                        "画像データを正しく読み込んでから再試行してください。"
                    )
                    return
            #print(f"DEBUG: Physical aspect ratio - XScanSize: {gv.XScanSize}, YScanSize: {gv.YScanSize}, Ratio: {physical_aspect_ratio:.3f}")
            
            # ラベルのサイズに合わせてスケーリング（物理サイズのアスペクト比を考慮）
            label_size = self.image_label.size()
            
            # 初期化時やラベルのサイズが取得できない場合の処理
            if label_size.width() <= 0 or label_size.height() <= 0:
                # ラベルのサイズが取得できない場合は、適切なサイズを推定
                # 画像のサイズに基づいて適切な表示サイズを計算
                image_height, image_width = image_data.shape
                
                # 物理サイズのアスペクト比に基づいて表示サイズを計算
                if physical_aspect_ratio > 1.0:
                    # 横長の場合
                    estimated_height = 400  # 推定高さ
                    display_width = int(estimated_height * physical_aspect_ratio)
                    display_height = estimated_height
                else:
                    # 縦長の場合
                    estimated_width = 400  # 推定幅
                    display_height = int(estimated_width / physical_aspect_ratio)
                    display_width = estimated_width
                
                #G: Estimated display size - Display: {display_width}x{display_height}")
                
                # 物理サイズのアスペクト比で強制的にスケーリング
                scaled_pixmap = pixmap.scaled(display_width, display_height, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                background_pixmap = scaled_pixmap
            else:
                # ラベルのサイズが取得できる場合
                # 物理サイズのアスペクト比に基づいて表示サイズを計算
                if physical_aspect_ratio > 1.0:
                    # 横長の場合
                    display_width = min(label_size.width(), int(label_size.height() * physical_aspect_ratio))
                    display_height = int(display_width / physical_aspect_ratio)
                else:
                    # 縦長の場合
                    display_height = min(label_size.height(), int(label_size.width() / physical_aspect_ratio))
                    display_width = int(display_height * physical_aspect_ratio)
                
                #print(f"DEBUG: Display size - Label: {label_size.width()}x{label_size.height()}, Display: {display_width}x{display_height}")
                
                # 物理サイズのアスペクト比で強制的にスケーリング（ピクセルアスペクト比を無視）
                scaled_pixmap = pixmap.scaled(display_width, display_height, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                
                # 白い背景のQPixmapを作成
                background_pixmap = QtGui.QPixmap(label_size)
                background_pixmap.fill(QtCore.Qt.white)
                
                # 背景に画像を描画（中央配置）
                painter = QtGui.QPainter(background_pixmap)
                x_offset = (background_pixmap.width() - scaled_pixmap.width()) // 2
                y_offset = (background_pixmap.height() - scaled_pixmap.height()) // 2
                painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
                painter.end()
            
            # 選択中の線や矩形を描画
            if self.line_selection_mode and self.line_start_point and self.line_end_point:
                painter = QtGui.QPainter(background_pixmap)
                painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
                
                # マウス座標を直接使用（lineprofile.pyと同じ方法）
                start_pos = self.line_start_point
                end_pos = self.line_end_point
                
                painter.drawLine(start_pos, end_pos)
                painter.end()
            
            # 既存の線とその端点を描画
            if self.roi_line and not self.line_selection_mode:
                painter = QtGui.QPainter(background_pixmap)
                
                # 線を描画
                painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
                
                # 画像座標をウィジェット座標に変換
                start_x, start_y = self.roi_line['start_x'], self.roi_line['start_y']
                end_x, end_y = self.roi_line['end_x'], self.roi_line['end_y']
                
                # 画像座標をウィジェット座標に変換
                start_widget_x, start_widget_y = self.imageToWidgetCoords(start_x, start_y)
                end_widget_x, end_widget_y = self.imageToWidgetCoords(end_x, end_y)
                
                # ウィジェット座標を直接使用（lineprofile.pyと同じ方法）
                start_pos = QtCore.QPoint(int(start_widget_x), int(start_widget_y))
                end_pos = QtCore.QPoint(int(end_widget_x), int(end_widget_y))
                
                painter.drawLine(start_pos, end_pos)
                
                # 端点を描画（ドラッグ可能であることを示す）
                # 開始点（赤色）
                painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
                painter.setBrush(QtGui.QBrush(QtCore.Qt.red))
                painter.drawEllipse(start_pos, self.endpoint_radius, self.endpoint_radius)
                
                # 終了点（青色）
                painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
                painter.setBrush(QtGui.QBrush(QtCore.Qt.blue))
                painter.drawEllipse(end_pos, self.endpoint_radius, self.endpoint_radius)
                
                painter.end()
            
            # 複数点ラインを描画
            if self.multi_point_line and not self.multi_point_line['is_straight'] and not self.line_selection_mode:
                painter = QtGui.QPainter(background_pixmap)
                
                # 線分を描画
                painter.setPen(QtGui.QPen(QtCore.Qt.magenta, 2))
                points = self.multi_point_line['points']
                
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i + 1]
                    
                    # 画像座標をウィジェット座標に変換
                    widget_x1, widget_y1 = self.imageToWidgetCoords(p1[0], p1[1])
                    widget_x2, widget_y2 = self.imageToWidgetCoords(p2[0], p2[1])
                    
                    start_pos = QtCore.QPoint(int(widget_x1), int(widget_y1))
                    end_pos = QtCore.QPoint(int(widget_x2), int(widget_y2))
                    
                    painter.drawLine(start_pos, end_pos)
                
                # 各点を描画
                for i, point in enumerate(points):
                    widget_x, widget_y = self.imageToWidgetCoords(point[0], point[1])
                    point_pos = QtCore.QPoint(int(widget_x), int(widget_y))
                    
                    # 端点は異なる色で描画
                    if i == 0:
                        painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.red))
                    elif i == len(points) - 1:
                        painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.blue))
                    else:
                        painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.yellow))
                    
                    painter.drawEllipse(point_pos, self.point_radius, self.point_radius)
                
                painter.end()
            
            # 描画された矩形を表示（複数点ラインが存在する場合は描画しない）
            if (self.drawn_box and not self.line_selection_mode and 
                (not self.multi_point_line or self.multi_point_line['is_straight'])):
                painter = QtGui.QPainter(background_pixmap)
                painter.setPen(QtGui.QPen(QtCore.Qt.green, 2))
                
                # Convert saved rectangle corners to widget coordinates
                corners_widget = []
                for corner_x, corner_y in self.drawn_box['corners']:
                    # 画像座標をウィジェット座標に変換
                    widget_x, widget_y = self.imageToWidgetCoords(corner_x, corner_y)
                    corners_widget.append(QtCore.QPoint(int(widget_x), int(widget_y)))
                
                # 矩形を描画
                for i in range(len(corners_widget)):
                    start_pos = corners_widget[i]
                    end_pos = corners_widget[(i + 1) % len(corners_widget)]
                    painter.drawLine(start_pos, end_pos)
                
                painter.end()
            
            self.image_label.setPixmap(background_pixmap)
            
            # AFM画像が表示されたらフォーカスを設定
            self.image_label.setFocus()
            
        except Exception as e:
            print(f"Image display error: {e}")
            self.image_label.setText(f"Image display error: {e}")
    
    def resizeEvent(self, event):
        """ウィンドウサイズ変更時の処理"""
        super().resizeEvent(event)
        # 少し遅延させてから画像を再表示（ラベルのサイズが正しく取得できるように）
        QtCore.QTimer.singleShot(10, self.updateImageDisplay)
    
    def imageMousePressEvent(self, event):
        """画像上のマウスプレスイベント"""
        if event.button() == QtCore.Qt.LeftButton:
            #print(f"DEBUG: Mouse press at ({event.pos().x()}, {event.pos().y()})")
            if self.line_selection_mode:
                # 線選択モード
                pos = event.pos()
                self.line_start_point = pos
                self.line_end_point = None
                self.image_label.setCursor(QtCore.Qt.CrossCursor)
                #print("DEBUG: Line selection mode")
            elif self.multi_point_mode and self.multi_point_line:
                # 複数点編集モード
                pos = event.pos()
                point_index = self.checkPointClick(pos)
                if point_index is not None:
                    self.drag_mode = True
                    self.drag_point = f'point_{point_index}'
                    self.selected_point_index = point_index
                    self.drag_start_pos = pos
                    self.image_label.setCursor(QtCore.Qt.SizeAllCursor)
                else:
                    # 新しい点を追加
                    self.addPointToLine(pos)
            elif self.roi_line and not self.line_selection_mode:
                # 線が既に存在する場合、端点のドラッグをチェック
                pos = event.pos()
                endpoint = self.checkEndpointClick(pos)
                if endpoint:
                    self.drag_mode = True
                    self.drag_point = endpoint
                    self.drag_start_pos = pos
                    self.image_label.setCursor(QtCore.Qt.SizeAllCursor)
                    #print(f"DEBUG: Endpoint drag started: {endpoint}")
                elif self.checkLineCenterClick(pos):
                    self.drag_mode = True
                    self.drag_point = 'center'
                    self.drag_start_pos = pos
                    self.image_label.setCursor(QtCore.Qt.SizeAllCursor)
                    #print("DEBUG: Center drag started")
                #else:
                    #print("DEBUG: No drag action detected")
    
    def imageMouseMoveEvent(self, event):
        """画像上のマウス移動イベント"""
        if self.line_selection_mode and self.line_start_point:
            # 線選択中の表示更新
            self.line_end_point = event.pos()
            self.updateImageDisplay()  # リアルタイム更新を復活
        elif self.drag_mode and self.drag_point == 'center':
            # 線の中央ドラッグ中
            new_pos = event.pos()
            #print(f"DEBUG: Center drag at ({new_pos.x()}, {new_pos.y()})")
            self.updateLinePosition(new_pos)
            self.updateImageDisplay()  # ドラッグ中もリアルタイム更新
        elif self.drag_mode and self.drag_point in ['start', 'end']:
            # 線の端点ドラッグ中
            new_pos = event.pos()
            #print(f"DEBUG: Endpoint drag at ({new_pos.x()}, {new_pos.y()})")
            self.updateLineEndpoint(new_pos)
            self.updateImageDisplay()  # ドラッグ中もリアルタイム更新
        elif self.drag_mode and self.drag_point.startswith('point_'):
            # 複数点ラインの点ドラッグ中
            new_pos = event.pos()
            point_index = int(self.drag_point.split('_')[1])
            self.updateMultiPointPosition(new_pos, point_index)
            self.updateImageDisplay()  # ドラッグ中もリアルタイム更新
        elif self.roi_line and not self.line_selection_mode and not self.drag_mode:
            # 線が存在し、選択モードでもドラッグモードでもない場合
            pos = event.pos()
            
            # 端点のチェック
            endpoint = self.checkEndpointClick(pos)
            if endpoint:
                self.image_label.setCursor(QtCore.Qt.SizeAllCursor)
            # 線の中央部分のチェック
            elif self.checkLineCenterClick(pos):
                self.image_label.setCursor(QtCore.Qt.CrossCursor)
            else:
                self.image_label.unsetCursor()
    
    def imageMouseReleaseEvent(self, event):
        """画像上のマウスリリースイベント"""
        if event.button() == QtCore.Qt.LeftButton:
            if self.line_selection_mode and self.line_start_point:
                # 線選択完了
                self.line_end_point = event.pos()
                self.finishLineSelection()
            elif self.drag_mode:
                # ドラッグ完了
                self.drag_mode = False
                self.drag_point = None
                self.drag_start_pos = None
                self.image_label.unsetCursor()
                self.updateInfo()
                # ドラッグ完了時に画像を再表示
                self.updateImageDisplay()
    

    
    def _get_physical_aspect_ratio(self):
        """物理サイズのアスペクト比を取得"""
        if hasattr(gv, 'XScanSize') and hasattr(gv, 'YScanSize') and gv.YScanSize > 0:
            return gv.XScanSize / gv.YScanSize
        return 1.0
    
    def widgetToImageCoords(self, widget_pos):
        """ウィジェット座標を画像座標に変換（物理サイズのアスペクト比を考慮）"""
        if not hasattr(gv, 'aryData') or gv.aryData is None:
            return 0, 0
        
        # 画像のサイズ
        image_height, image_width = gv.aryData.shape
        
        # ラベルのサイズ
        label_size = self.image_label.size()
        
        # ラベルのサイズが取得できない場合はデフォルト値を使用
        if label_size.width() <= 0 or label_size.height() <= 0:
            return 0, 0
        
        # 物理サイズのアスペクト比を取得
        physical_aspect_ratio = self._get_physical_aspect_ratio()
        
        # 物理サイズのアスペクト比に基づいて表示サイズを計算
        if physical_aspect_ratio > 1.0:
            # 横長の場合
            display_width = min(label_size.width(), int(label_size.height() * physical_aspect_ratio))
            display_height = int(display_width / physical_aspect_ratio)
        else:
            # 縦長の場合
            display_height = min(label_size.height(), int(label_size.width() / physical_aspect_ratio))
            display_width = int(display_height * physical_aspect_ratio)
        
        # オフセット計算（画像を中央に配置）
        offset_x = (label_size.width() - display_width) / 2
        offset_y = (label_size.height() - display_height) / 2
        
        # マウス座標から画像表示座標への変換
        image_display_x = widget_pos.x() - offset_x
        image_display_y = widget_pos.y() - offset_y
        
        # 画像表示座標から実際の画像座標への変換
        if display_width > 0 and display_height > 0:
            scale_x = image_width / display_width
            scale_y = image_height / display_height
            
            image_x = int(image_display_x * scale_x)
            image_y = int(image_display_y * scale_y)
        else:
            image_x = 0
            image_y = 0
        
        # 境界チェック
        image_x = max(0, min(image_x, image_width - 1))
        image_y = max(0, min(image_y, image_height - 1))
        
        # 画像表示でflipudしているため、Y軸の反転が必要
        image_y = image_height - 1 - image_y
        
        return image_x, image_y
    
    def mouseToDisplayCoords(self, mouse_pos):
        """マウス座標を画像表示座標に変換"""
        # ラベルのサイズ
        label_size = self.image_label.size()
        
        # 画像の表示サイズを計算
        pixmap = self.image_label.pixmap()
        if pixmap:
            display_size = pixmap.size()
        else:
            display_size = label_size
        
        # 画像がラベル内で中央に配置されている場合のオフセットを計算
        offset_x = (label_size.width() - display_size.width()) / 2
        offset_y = (label_size.height() - display_size.height()) / 2
        
        # マウス座標から画像表示座標への変換
        display_x = mouse_pos.x() - offset_x
        display_y = mouse_pos.y() - offset_y
        
        return QtCore.QPoint(int(display_x), int(display_y))
            
    def createKymograph(self):
        """Kymographを作成"""
        if not self.roi_line:
            QtWidgets.QMessageBox.warning(self, "Error", "ROIが選択されていません。")
            return
            
        try:
            # Kymograph作成時に既存の検出結果をクリア
            self.clearDetectionResults()
            
            # 画像スタックを読み込み
            self.loadImageStack()
            
            if not self.image_stack:
                QtWidgets.QMessageBox.warning(self, "Error", "画像データが読み込めません。")
                return
                
            # Kymograph作成
            self.kymograph_data = self.createLineKymograph()
            
            if self.kymograph_data is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Kymographの作成に失敗しました。")
                return
                
            # 結果表示
            self.displayKymograph()
            self.save_button.setEnabled(True)
            
            # セミオート追跡ボタンを有効化
            if hasattr(self, 'draw_reference_button'):
                self.draw_reference_button.setEnabled(True)
            
            # 線維成長解析ボタンを有効化
            if hasattr(self, 'analyze_growth_button'):
                self.analyze_growth_button.setEnabled(True)
            
        except Exception as e:
            print(f"Error in createKymograph: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while creating the kymograph:\n{e}")
    
    def clearDetectionResults(self):
        """Kymograph上に描画されている検出結果をすべてクリア"""
        # 線維成長解析データをクリア
        if hasattr(self, 'fiber_growth_data'):
            self.fiber_growth_data = None
        
        # 粒子検出結果をクリア
        if hasattr(self, 'particle_detection_results'):
            self.particle_detection_results = None
        
        # 軌跡データをクリア
        if hasattr(self, 'particle_tracks'):
            self.particle_tracks = None
        
        # セミオート検出結果をクリア
        if hasattr(self, 'semi_auto_results'):
            self.semi_auto_results = None
        
        # 参照線をクリア
        if hasattr(self, 'reference_line_points'):
            self.reference_line_points = []
            self.reference_line_complete = False
        
        # 描画中の参照線フラグをリセット
        if hasattr(self, 'drawing_reference_line'):
            self.drawing_reference_line = False
        
        # グラフウィンドウを閉じる
        if hasattr(self, 'graph_window') and self.graph_window is not None:
            self.graph_window.close()
            self.graph_window = None
        
        # デバッグウィンドウを閉じる
        if hasattr(self, 'debug_window') and self.debug_window is not None:
            self.debug_window.close()
            self.debug_window = None
        
        # テストウィンドウを閉じる
        if hasattr(self, 'test_window') and self.test_window is not None:
            self.test_window.close()
            self.test_window = None
        
        #print("DEBUG: All detection results cleared")
        
    def loadImageStack(self):
        """画像スタックを読み込み"""
        self.image_stack = []
        
        # グローバル変数の安全性チェック
        if gv is None:
            print("Error: globalvals not available")
            return
            
        if not hasattr(gv, 'files') or len(gv.files) == 0:
            print("Error: No files available")
            return
            
        if not hasattr(gv, 'FrameNum') or gv.FrameNum <= 0:
            print("Error: Invalid FrameNum")
            return
            
        if not hasattr(gv, 'currentFileNum') or gv.currentFileNum >= len(gv.files):
            print("Error: Invalid currentFileNum")
            return
            
        # 元のインデックスを保存
        original_index = gv.index if hasattr(gv, 'index') else 0
        
        try:
            # メモリ使用量を制限するために、最大フレーム数を制限
            max_frames = min(gv.FrameNum, 1000)  # 最大1000フレームまで
            
            for frame_idx in range(max_frames):
                # インデックスを設定
                gv.index = frame_idx
                
                # 安全なフレーム読み込み
                if not safe_load_frame(gv.files[gv.currentFileNum], frame_idx):
                    print(f"Warning: Failed to load frame {frame_idx}")
                    continue
                
                # 安全なデータ取得
                frame_data = safe_get_ary_data()
                if frame_data is not None:
                    # メモリ使用量を削減するために、データ型を最適化
                    if frame_data.dtype != np.float32:
                        frame_data = frame_data.astype(np.float32)
                    self.image_stack.append(frame_data)
                else:
                    print(f"Warning: No data for frame {frame_idx}")
                    
                # メモリ使用量を監視（オプション）
                #if len(self.image_stack) % 100 == 0:
                    #print(f"Loaded {len(self.image_stack)} frames...")
                    
        except Exception as e:
            print(f"Error in loadImageStack: {e}")
            self.image_stack = []
            
        finally:
            # 元のフレームに戻す
            try:
                gv.index = original_index
                safe_load_frame(gv.files[gv.currentFileNum], original_index)
            except Exception as e:
                print(f"Error restoring original frame: {e}")
        
         #print(f"Loaded {len(self.image_stack)} frames out of {gv.FrameNum}")
        
    def createLineKymograph(self):
        """隙間のないマスクベース矩形Kymograph作成"""
        if not self.roi_line or not self.image_stack:
            return None
            
        # drawn_boxが存在しない場合は直線に沿ったkymographを作成
        if not hasattr(self, 'drawn_box') or self.drawn_box is None:
            self.actual_frame_height = None  # 直線の場合は1行/フレーム
            return self.createLineBasedKymograph()
        
        #print(f"DEBUG: Starting seamless mask-based kymograph creation")
        #  print(f"DEBUG: Box corners: {self.drawn_box['corners']}")
        
        # 矩形マスクを作成
        image_height, image_width = self.image_stack[0].shape
        mask = self.createRectangleMask(image_height, image_width)
        
        if mask is None or np.sum(mask) == 0:
            # print("DEBUG: Invalid mask, falling back to line-based method")
            self.actual_frame_height = None
            return self.createLineBasedKymograph()
        
        #print(f"DEBUG: Mask created with {np.sum(mask)} pixels")
        
        # 矩形の角度を計算（水平にする角度）
        box_angle = self.drawn_box['angle']
        rotation_angle = box_angle
        
        #print(f"DEBUG: Box angle: {np.degrees(box_angle):.1f} degrees")
        #print(f"DEBUG: Rotation angle: {np.degrees(rotation_angle):.1f} degrees")
        
        # 最初のフレームを使って基準となるクロップ領域を決定
        first_frame = self.image_stack[0]
        
        # 最初のフレームでマスクを適用して回転
        masked_first = first_frame * mask
        rotated_first = rotate(masked_first, np.degrees(rotation_angle), 
                              reshape=True, order=1, prefilter=False)
        rotated_mask_first = rotate(mask.astype(float), np.degrees(rotation_angle), 
                                   reshape=True, order=1, prefilter=False)
        
        # 有効領域を特定（最初のフレームを基準に）
        valid_region_first = rotated_mask_first > 0.1
        
        if np.sum(valid_region_first) == 0:
            #print("DEBUG: No valid region in first frame")
            self.actual_frame_height = None
            return self.createLineBasedKymograph()
        
        # 有効領域の境界を計算（すべてのフレームで共通使用）
        rows, cols = np.where(valid_region_first)
        crop_min_row = max(0, np.min(rows))
        crop_max_row = min(rotated_first.shape[0], np.max(rows) + 1)
        crop_min_col = max(0, np.min(cols))
        crop_max_col = min(rotated_first.shape[1], np.max(cols) + 1)
        
        # クロップサイズを確定
        crop_height = crop_max_row - crop_min_row
        crop_width = crop_max_col - crop_min_col
        
        # 実際のフレーム高さを記録（追加）
        self.actual_frame_height = crop_height
        
        #print(f"DEBUG: Fixed crop region: ({crop_min_row}, {crop_min_col}) to ({crop_max_row}, {crop_max_col})")
        #print(f"DEBUG: Fixed crop size: {crop_height} x {crop_width}")
        
        # すべてのフレームを同じ領域でクロップ
        processed_frames = []
        
        for frame_idx, frame_data in enumerate(self.image_stack):
            #print(f"DEBUG: Processing frame {frame_idx} with fixed crop region")
            
            # マスクを適用
            masked_data = frame_data * mask
            
            # 同じ角度で回転
            rotated_frame = rotate(masked_data, np.degrees(rotation_angle), 
                                 reshape=True, order=1, prefilter=False)
            
            # 回転後の画像サイズをチェック
            if (rotated_frame.shape[0] < crop_max_row or 
                rotated_frame.shape[1] < crop_max_col):
               # print(f"DEBUG: Warning - Frame {frame_idx} rotated size {rotated_frame.shape} "
                     # f"is smaller than expected crop region")
                # サイズが足りない場合はパディング
                padded_frame = np.zeros((max(rotated_frame.shape[0], crop_max_row),
                                       max(rotated_frame.shape[1], crop_max_col)))
                h, w = rotated_frame.shape
                padded_frame[:h, :w] = rotated_frame
                rotated_frame = padded_frame
            
            # 固定された領域でクロップ
            cropped_frame = rotated_frame[crop_min_row:crop_max_row, 
                                         crop_min_col:crop_max_col]
            
            # サイズチェック
            if cropped_frame.shape != (crop_height, crop_width):
                #print(f"DEBUG: Warning - Frame {frame_idx} cropped size {cropped_frame.shape} "
                      #f"differs from expected {(crop_height, crop_width)}")
                # サイズを強制的に合わせる
                fixed_frame = np.zeros((crop_height, crop_width))
                min_h = min(cropped_frame.shape[0], crop_height)
                min_w = min(cropped_frame.shape[1], crop_width)
                fixed_frame[:min_h, :min_w] = cropped_frame[:min_h, :min_w]
                cropped_frame = fixed_frame
            
            # 正規化を適用
            if self.normalize_enabled:
                cropped_frame = self.normalizeFrame(cropped_frame)
            
            processed_frames.append(cropped_frame)
            
            #print(f"DEBUG: Frame {frame_idx} - Final crop shape: {cropped_frame.shape}")
            #print(f"DEBUG: Frame {frame_idx} - Data range: {cropped_frame.min():.2f} to {cropped_frame.max():.2f}")
        
        if not processed_frames:
            #print("DEBUG: No processed frames")
            self.actual_frame_height = None
            return self.createLineBasedKymograph()
        
        # 全フレームのサイズが同じことを確認
        expected_shape = (crop_height, crop_width)
        #for i, frame in enumerate(processed_frames):
            #if frame.shape != expected_shape:
                #print(f"DEBUG: Error - Frame {i} has wrong shape: {frame.shape}, expected: {expected_shape}")
        
        #print(f"DEBUG: All frames have consistent shape: {expected_shape}")
        #print(f"DEBUG: Total frames: {len(processed_frames)}")
        
        # 隙間なく縦に連結
        kymograph = np.vstack(processed_frames)
        
        #print(f"DEBUG: Final seamless kymograph shape: {kymograph.shape}")
        #print(f"DEBUG: Expected height: {len(processed_frames) * crop_height}")
        #print(f"DEBUG: Final kymograph range: {kymograph.min():.2f} to {kymograph.max():.2f}")
        
        return kymograph
    
    def createLineBasedKymograph(self):
        """直線に沿ったkymograph作成（修正版）"""
        if not self.roi_line or not self.image_stack:
            return None
            
        # メソッドの最初に追加
        self.actual_frame_height = None  # 直線の場合は使用しない
        
        # 複数点ラインが存在する場合は複数点版を使用
        if self.multi_point_line and not self.multi_point_line['is_straight']:
            return self.createMultiPointKymograph()
            
        # 線の端点を取得（画像座標）
        start_x = self.roi_line['start_x']
        start_y = self.roi_line['start_y']
        end_x = self.roi_line['end_x'] 
        end_y = self.roi_line['end_y']
        
        #print(f"DEBUG: Line coordinates - start: ({start_x:.1f}, {start_y:.1f}), end: ({end_x:.1f}, {end_y:.1f})")
        
        # 線の長さを計算（ピクセル）
        line_length_pixels = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # 物理サイズベースのサンプリング点数を決定（必須）
        if not hasattr(gv, 'XScanSize') or not hasattr(gv, 'YScanSize') or gv.XScanSize <= 0 or gv.YScanSize <= 0:
            QtWidgets.QMessageBox.critical(
                self, 
                "エラー", 
                "物理サイズ情報が利用できません。\n"
                "XScanSizeまたはYScanSizeが設定されていないか、無効な値です。\n"
                "画像データを正しく読み込んでから再試行してください。"
            )
            return None
        
        # 物理サイズが利用可能な場合
        image_height, image_width = gv.aryData.shape
        pixel_size_x = gv.XScanSize / image_width  # nm/pixel
        pixel_size_y = gv.YScanSize / image_height  # nm/pixel
        
        # 線の物理的な長さを計算（nm）
        line_length_nm = line_length_pixels * np.sqrt(pixel_size_x**2 + pixel_size_y**2)
        
        # 物理サイズに基づいてサンプリング点数を決定（1nmあたり1点程度）
        num_points = max(int(line_length_nm), 10)
        
        #print(f"DEBUG: Line length: {line_length:.1f} pixels, Sampling points: {num_points}")
        
        # 線に沿った座標を生成（start → end の方向）
        seg_x = np.linspace(start_x, end_x, num_points)
        seg_y = np.linspace(start_y, end_y, num_points)
        
        # デバッグ用：座標の確認
        #print(f"DEBUG: First 3 points: x={seg_x[:3]}, y={seg_y[:3]}")
        #print(f"DEBUG: Last 3 points: x={seg_x[-3:]}, y={seg_y[-3:]}")
        
        # kymographデータを初期化
        if len(self.image_stack) == 1:
            kymograph = np.zeros((1, num_points))
        else:
            kymograph = np.zeros((len(self.image_stack), num_points))
        
        # 各フレームで線に沿ったデータを抽出
        for frame_idx, frame_data in enumerate(self.image_stack):
            #print(f"DEBUG: Processing frame {frame_idx}")
            #print(f"DEBUG: Frame shape: {frame_data.shape}")
            #print(f"DEBUG: Frame data range: {frame_data.min():.2f} to {frame_data.max():.2f}")
            
            # 各点でのデータ値を取得
            extracted_data = []
            for i in range(num_points):
                x, y = seg_x[i], seg_y[i]
                
                # 境界チェック
                if 0 <= x < frame_data.shape[1] and 0 <= y < frame_data.shape[0]:
                    # 双線形補間
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0 + 1, frame_data.shape[1] - 1), min(y0 + 1, frame_data.shape[0] - 1)
                    
                    # 補間係数
                    fx, fy = x - x0, y - y0
                    
                    # 4つの隣接ピクセルの値
                    v00 = frame_data[y0, x0]
                    v01 = frame_data[y0, x1] if x1 != x0 else v00
                    v10 = frame_data[y1, x0] if y1 != y0 else v00
                    v11 = frame_data[y1, x1] if (x1 != x0 and y1 != y0) else v00
                    
                    # 双線形補間
                    interpolated_value = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01 + (1-fx)*fy*v10 + fx*fy*v11
                    extracted_data.append(interpolated_value)
                else:
                    # 境界外の場合は0
                    extracted_data.append(0)
            
            extracted_data = np.array(extracted_data)
            
            # 正規化を適用
            if self.normalize_enabled:
                extracted_data = self.normalizeFrame(extracted_data)
            
            if len(self.image_stack) == 1:
                kymograph[0, :] = extracted_data
            else:
                kymograph[frame_idx, :] = extracted_data
            
            #print(f"DEBUG: Frame {frame_idx} - Extracted data range: {extracted_data.min():.2f} to {extracted_data.max():.2f}")
            
            # 線の両端のデータ値を確認
            #print(f"DEBUG: Frame {frame_idx} - Start value: {extracted_data[0]:.2f}, End value: {extracted_data[-1]:.2f}")
        
        #print(f"DEBUG: Final kymograph shape: {kymograph.shape}")
        #print(f"DEBUG: Final kymograph range: {kymograph.min():.2f} to {kymograph.max():.2f}")
        
        return kymograph
    
    def createMultiPointKymograph(self):
        """複数点ラインに沿ったkymograph作成"""
        if not self.multi_point_line or not self.image_stack:
            return None
        
        points = self.multi_point_line['points']
        if len(points) < 2:
            return None
        
        # 各線分の長さを計算して総長を求める（ピクセル）
        total_length_pixels = 0
        segment_lengths_pixels = []
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            segment_lengths_pixels.append(segment_length)
            total_length_pixels += segment_length
        
        # 物理サイズベースのサンプリング点数を決定（必須）
        if not hasattr(gv, 'XScanSize') or not hasattr(gv, 'YScanSize') or gv.XScanSize <= 0 or gv.YScanSize <= 0:
            QtWidgets.QMessageBox.critical(
                self, 
                "エラー", 
                "物理サイズ情報が利用できません。\n"
                "XScanSizeまたはYScanSizeが設定されていないか、無効な値です。\n"
                "画像データを正しく読み込んでから再試行してください。"
            )
            return None
        
        # 物理サイズが利用可能な場合
        image_height, image_width = gv.aryData.shape
        pixel_size_x = gv.XScanSize / image_width  # nm/pixel
        pixel_size_y = gv.YScanSize / image_height  # nm/pixel
        
        # 複数点ラインの物理的な総長を計算（nm）
        total_length_nm = total_length_pixels * np.sqrt(pixel_size_x**2 + pixel_size_y**2)
        
        # 物理サイズに基づいてサンプリング点数を決定（1nmあたり1点程度）
        num_points = max(int(total_length_nm), 10)
        
        # kymographデータを初期化
        if len(self.image_stack) == 1:
            kymograph = np.zeros((1, num_points))
        else:
            kymograph = np.zeros((len(self.image_stack), num_points))
        
        # 各フレームで複数点ラインに沿ったデータを抽出
        for frame_idx, frame_data in enumerate(self.image_stack):
            extracted_data = []
            
            # 各サンプリング点でのデータ値を取得
            for i in range(num_points):
                # サンプリング点の位置を計算（0から1の範囲）
                t = i / (num_points - 1) if num_points > 1 else 0
                
                # 対応する線分と位置を特定
                current_length = 0
                target_length = t * total_length_pixels
                
                for seg_idx, seg_length in enumerate(segment_lengths_pixels):
                    if current_length + seg_length >= target_length:
                        # この線分内の位置を計算
                        local_t = (target_length - current_length) / seg_length
                        p1 = points[seg_idx]
                        p2 = points[seg_idx + 1]
                        
                        # 線形補間で座標を計算
                        x = p1[0] + local_t * (p2[0] - p1[0])
                        y = p1[1] + local_t * (p2[1] - p1[1])
                        break
                    current_length += seg_length
                else:
                    # 最後の点
                    x, y = points[-1]
                
                # 境界チェック
                if 0 <= x < frame_data.shape[1] and 0 <= y < frame_data.shape[0]:
                    # 双線形補間
                    x0, y0 = int(x), int(y)
                    x1, y1 = min(x0 + 1, frame_data.shape[1] - 1), min(y0 + 1, frame_data.shape[0] - 1)
                    
                    # 補間係数
                    fx, fy = x - x0, y - y0
                    
                    # 4つの隣接ピクセルの値
                    v00 = frame_data[y0, x0]
                    v01 = frame_data[y0, x1] if x1 != x0 else v00
                    v10 = frame_data[y1, x0] if y1 != y0 else v00
                    v11 = frame_data[y1, x1] if (x1 != x0 and y1 != y0) else v00
                    
                    # 双線形補間
                    interpolated_value = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01 + (1-fx)*fy*v10 + fx*fy*v11
                    extracted_data.append(interpolated_value)
                else:
                    # 境界外の場合は0
                    extracted_data.append(0)
            
            extracted_data = np.array(extracted_data)
            
            # 正規化を適用
            if self.normalize_enabled:
                extracted_data = self.normalizeFrame(extracted_data)
            
            if len(self.image_stack) == 1:
                kymograph[0, :] = extracted_data
            else:
                kymograph[frame_idx, :] = extracted_data
        
        return kymograph
    
    def createRectangleMask(self, height, width):
        """矩形マスクを作成"""
        if not hasattr(self, 'drawn_box') or self.drawn_box is None:
            return None
        
        # マスク初期化
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 矩形の角座標
        corners = self.drawn_box['corners']
        
        #print(f"DEBUG: Creating mask for corners: {corners}")
        
        # OpenCVを使って矩形を塗りつぶし
        try:
            import cv2
            # 座標を整数に変換
            pts = np.array([(int(x), int(y)) for x, y in corners], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
            #print(f"DEBUG: OpenCV mask created with {np.sum(mask)} pixels")
            return mask
        except ImportError:
            #print("DEBUG: OpenCV not available, using manual method")
            return self.createPolygonMaskManual(height, width, corners)
    
    def createPolygonMaskManual(self, height, width, corners):
        """手動でポリゴンマスクを作成（OpenCV不使用）"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        def point_in_polygon(x, y, corners):
            """点が多角形内にあるかチェック（Ray casting algorithm）"""
            n = len(corners)
            inside = False
            p1x, p1y = corners[0]
            for i in range(1, n + 1):
                p2x, p2y = corners[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        
        # 境界ボックス内の全点をチェック
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        
        min_x = max(0, int(min(x_coords)))
        max_x = min(width, int(max(x_coords)) + 1)
        min_y = max(0, int(min(y_coords)))
        max_y = min(height, int(max(y_coords)) + 1)
        
        count = 0
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if point_in_polygon(x, y, corners):
                    mask[y, x] = 1
                    count += 1
        
        # print(f"DEBUG: Manual mask created with {count} pixels")
        return mask
    
    def debugRectangleMask(self):
        """デバッグ用：矩形マスクの確認"""
        if not hasattr(self, 'drawn_box') or self.drawn_box is None:
            print("No drawn box to debug")
            return
        
        if not hasattr(gv, 'aryData') or gv.aryData is None:
            print("No image data available")
            return
        
        # マスクを作成
        image_height, image_width = gv.aryData.shape
        mask = self.createRectangleMask(image_height, image_width)
        
        if mask is None:
            print("Failed to create mask")
            return
        
        #print(f"=== Rectangle Mask Debug ===")
        #print(f"Image size: {image_width} x {image_height}")
        #print(f"Mask pixels: {np.sum(mask)}")
        #print(f"Box corners: {self.drawn_box['corners']}")
        
        # マスクを可視化
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 元画像
            axes[0, 0].imshow(gv.aryData, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # マスク
            axes[0, 1].imshow(mask, cmap='gray')
            axes[0, 1].set_title('Rectangle Mask')
            axes[0, 1].axis('off')
            
            # マスクされた画像
            masked_image = gv.aryData * mask
            axes[1, 0].imshow(masked_image, cmap='gray')
            axes[1, 0].set_title('Masked Image')
            axes[1, 0].axis('off')
            
            # 矩形の輪郭を描画
            axes[1, 1].imshow(gv.aryData, cmap='gray')
            corners = self.drawn_box['corners']
            # 矩形を描画
            for i in range(len(corners)):
                start = corners[i]
                end = corners[(i + 1) % len(corners)]
                axes[1, 1].plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=2)
            axes[1, 1].set_title('Rectangle Outline')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig('debug_rectangle_mask.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            #print("Debug images saved as 'debug_rectangle_mask.png'")
            
            # マスク内の統計情報
            masked_values = gv.aryData[mask > 0]
            #if len(masked_values) > 0:
                #print(f"Masked region statistics:")
                #print(f"  Min: {masked_values.min():.2f}")
                #print(f"  Max: {masked_values.max():.2f}")
                #print(f"  Mean: {masked_values.mean():.2f}")
                #print(f"  Std: {masked_values.std():.2f}")
        
        except ImportError:
            print("Matplotlib not available for visualization")
    
    def testRectangleCoordinates(self):
        """座標の詳細テスト"""
        if not self.roi_line or not self.drawn_box:
            print("No line or box to test")
            return
        
        #print("=== Rectangle Coordinate Test ===")
        
        # 線の情報
        #print(f"Line: start=({self.roi_line['start_x']:.1f}, {self.roi_line['start_y']:.1f}), "
              #f"end=({self.roi_line['end_x']:.1f}, {self.roi_line['end_y']:.1f})")
        
        # 矩形の情報
            #print(f"Box center: ({self.drawn_box['center_x']:.1f}, {self.drawn_box['center_y']:.1f})")
        #print(f"Box angle: {np.degrees(self.drawn_box['angle']):.1f} degrees")
        #print(f"Box size: {self.drawn_box['box_length']:.1f} x {self.drawn_box['box_width']:.1f}")
        
        # 各角の座標とピクセル値
        corners = self.drawn_box['corners']
        #print("Corner coordinates and pixel values:")
        for i, (x, y) in enumerate(corners):
            #print(f"  Corner {i+1}: ({x:.1f}, {y:.1f})")
            
            # 画像内かチェック
            if hasattr(gv, 'aryData'):
                h, w = gv.aryData.shape
                if 0 <= x < w and 0 <= y < h:
                    pixel_value = gv.aryData[int(y), int(x)]
                    print(f"    Pixel value: {pixel_value:.2f}")
                else:
                    print(f"    Out of bounds (image: {w}x{h})")
        
        # 線の中心での値
        center_x = (self.roi_line['start_x'] + self.roi_line['end_x']) / 2
        center_y = (self.roi_line['start_y'] + self.roi_line['end_y']) / 2
        if hasattr(gv, 'aryData'):
            h, w = gv.aryData.shape
            if 0 <= center_x < w and 0 <= center_y < h:
                center_value = gv.aryData[int(center_y), int(center_x)]
                print(f"Line center ({center_x:.1f}, {center_y:.1f}): {center_value:.2f}")
        
        # 矩形の境界ボックス
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        #print(f"Bounding box: ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})")
        #print(f"Bbox size: {bbox[2]-bbox[0]:.1f} x {bbox[3]-bbox[1]:.1f}")
    
    def forceMaskBasedKymograph(self):
        """マスクベースKymographを強制実行"""
        if not self.roi_line:
            QtWidgets.QMessageBox.warning(self, "Error", "No line selected")
            return
        
        if not hasattr(self, 'drawn_box') or self.drawn_box is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No box drawn")
            return
        
        try:
            # 画像スタックを確認
            if not self.image_stack:
                self.loadImageStack()
            
            if not self.image_stack:
                QtWidgets.QMessageBox.warning(self, "Error", "No image stack available")
                return
            
            print("=== Forcing Mask-based Kymograph ===")
            
            # マスクベースでkymographを作成
            self.kymograph_data = self.createLineKymograph()
            
            if self.kymograph_data is not None:
                self.displayKymograph()
                self.save_button.setEnabled(True)
                QtWidgets.QMessageBox.information(self, "Success", "Mask-based kymograph created successfully")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to create mask-based kymograph")
        
        except Exception as e:
            print(f"ERROR in forceMaskBasedKymograph: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error in mask-based kymograph: {e}")
    
    def compareMethods(self):
        """異なる手法でのkymograph作成を比較"""
        if not self.roi_line or not self.image_stack:
            QtWidgets.QMessageBox.warning(self, "Error", "No line or image stack available")
            return
        
        try:
            print("=== Comparing Kymograph Methods ===")
            
            # 1. 直線ベース
            print("Creating line-based kymograph...")
            line_kymo = self.createLineBasedKymograph()
            
            # 2. マスクベース（矩形がある場合）
            mask_kymo = None
            if hasattr(self, 'drawn_box') and self.drawn_box is not None:
                print("Creating mask-based kymograph...")
                mask_kymo = self.createLineKymograph()  # 修正版のマスクベース手法
            
            # 結果表示
            if mask_kymo is not None:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # 直線ベース
                im1 = axes[0].imshow(line_kymo, aspect='auto', cmap='gray')
                axes[0].set_title('Line-based Kymograph')
                axes[0].set_xlabel('Position')
                axes[0].set_ylabel('Frame')
                
                # マスクベース
                im2 = axes[1].imshow(mask_kymo, aspect='auto', cmap='gray')
                axes[1].set_title('Mask-based Kymograph')
                axes[1].set_xlabel('Position')
                axes[1].set_ylabel('Frame')
                
                plt.tight_layout()
                plt.savefig('kymograph_comparison.png', dpi=150, bbox_inches='tight')
                plt.show()
                
                print("Comparison saved as 'kymograph_comparison.png'")
                print(f"Line-based shape: {line_kymo.shape}, range: {line_kymo.min():.2f}-{line_kymo.max():.2f}")
                print(f"Mask-based shape: {mask_kymo.shape}, range: {mask_kymo.min():.2f}-{mask_kymo.max():.2f}")
            else:
                print("Only line-based method available (no rectangle drawn)")
                print(f"Line-based shape: {line_kymo.shape}, range: {line_kymo.min():.2f}-{line_kymo.max():.2f}")
                
            QtWidgets.QMessageBox.information(self, "Comparison Complete", 
                                            "Methods compared. Check console and saved images.")
        
        except Exception as e:
            print(f"ERROR in compareMethods: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error in method comparison: {e}")
    
    def testRotationAngle(self):
        """回転角度のテスト"""
        if not self.roi_line:
            print("No line to test rotation")
            return
        
        print("=== Rotation Angle Test ===")
        
        # 線の情報
        start_x, start_y = self.roi_line['start_x'], self.roi_line['start_y']
        end_x, end_y = self.roi_line['end_x'], self.roi_line['end_y']
        
        print(f"Line: start=({start_x:.1f}, {start_y:.1f}), end=({end_x:.1f}, {end_y:.1f})")
        
        # ベクトル計算
        dx = end_x - start_x
        dy = end_y - start_y
        
        print(f"Vector: ({dx:.1f}, {dy:.1f})")
        
        # 線の角度計算
        line_angle_rad = np.arctan2(dy, dx)
        line_angle_deg = np.degrees(line_angle_rad)
        
        print(f"Line angle: {line_angle_deg:.1f} degrees")
        
        # 矩形情報（ある場合）
        if hasattr(self, 'drawn_box') and self.drawn_box is not None:
            box_angle_rad = self.drawn_box['angle']
            box_angle_deg = np.degrees(box_angle_rad)
            
            print(f"Box angle: {box_angle_deg:.1f} degrees")
            print(f"Box dimensions: {self.drawn_box['box_length']:.1f} x {self.drawn_box['box_width']:.1f}")
            
            # 矩形を水平にする回転角度
            rotation_angle_rad = box_angle_rad
            rotation_angle_deg = np.degrees(rotation_angle_rad)
            
            print(f"Rotation angle: {rotation_angle_deg:.1f} degrees")
            
            # 期待される結果
            expected_horizontal = box_angle_deg + rotation_angle_deg
            print(f"Expected horizontal angle: {expected_horizontal:.1f} degrees")
            
            QtWidgets.QMessageBox.information(self, "Rotation Test", 
                                            f"Line angle: {line_angle_deg:.1f}°\n"
                                            f"Box angle: {box_angle_deg:.1f}°\n"
                                            f"Rotation angle: {rotation_angle_deg:.1f}°\n"
                                            f"Expected horizontal: {expected_horizontal:.1f}°")
        else:
            print("No box drawn")
            QtWidgets.QMessageBox.information(self, "Rotation Test", 
                                            f"Line angle: {line_angle_deg:.1f}°\n"
                                            f"No box drawn")
        
        # 線の長さ
        length = np.sqrt(dx*dx + dy*dy)
        print(f"Line length: {length:.1f} pixels")
        
    def displayKymograph(self):
        """Kymographを表示"""
        if self.kymograph_data is None:
            return
            
        # 結果表示用のフィギュアをクリア
        self.result_figure.clear()
        self.result_ax = self.result_figure.add_subplot(111)
        ax = self.result_ax
        
        # コントラスト設定を決定
        if hasattr(self, 'current_contrast') and self.current_contrast is not None:
            # 保存されたコントラスト設定を使用
            vmin, vmax = self.current_contrast
    
        else:
            # デフォルトのコントラスト最適化
            data_min = np.min(self.kymograph_data)
            data_max = np.max(self.kymograph_data)
            
            # パーセントイルを使用してコントラストを調整
            # 下位0.5%と上位99.5%の値を基準にする（より厳しいコントラスト）
            vmin = np.percentile(self.kymograph_data, 0.5)
            vmax = np.percentile(self.kymograph_data, 99.5)
            
            # 現在のコントラスト設定を保存
            self.current_contrast = (vmin, vmax)
    
        
        
        
        # Kymographを表示（コントラスト調整済み）
        im = ax.imshow(self.kymograph_data, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Position')
        
        # 線維先端検出結果を描画（修正版）
        if hasattr(self, 'fiber_growth_data') and self.fiber_growth_data is not None:
            if not hasattr(self.fiber_growth_data, 'manual_points'):  # 自動検出の場合のみ
                edge_positions = self.fiber_growth_data['edge_positions']
                time_points = self.fiber_growth_data['time_points']
                
                # 矩形モードかどうかを判定
                is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                     self.actual_frame_height is not None and 
                                     self.actual_frame_height > 1)
                
        
                
                for i, (t, pos) in enumerate(zip(time_points, edge_positions)):
                    if 0 <= pos < self.kymograph_data.shape[1]:  # 位置が有効範囲内かチェック
                        
                        if is_rectangular_mode:
                            # 矩形モード：各フレームの中央位置に描画
                            frame_height = int(self.actual_frame_height)
                            frame_center_y = i * frame_height + frame_height / 2
                            
                            # kymographの範囲内かチェック
                            if frame_center_y < self.kymograph_data.shape[0]:
                                ax.plot(pos, frame_center_y, 'y+', markersize=8, markeredgewidth=2)
                    
                        else:
                            # 直線モード：フレームインデックスをそのまま使用
                            ax.plot(pos, i, 'y+', markersize=8, markeredgewidth=2)
                            
        

        
        # 参照線を描画
        if hasattr(self, 'reference_line_points') and self.reference_line_points:
            ref_x = [p[0] for p in self.reference_line_points]
            ref_y = [p[1] for p in self.reference_line_points]
            
            if self.drawing_reference_line:
                # 描画中は点と線を表示
                ax.scatter(ref_x, ref_y, c='orange', s=80, marker='o', 
                          edgecolors='white', linewidth=2, label='Reference Points', zorder=10)
                
                if len(self.reference_line_points) > 1:
                    ax.plot(ref_x, ref_y, 'orange', linewidth=3, alpha=0.7, 
                           label='Reference Line', zorder=9)
            else:
                # 完成した参照線
                ax.plot(ref_x, ref_y, 'lime', linewidth=2, alpha=0.8, 
                       label='Reference Line', zorder=9)
                ax.scatter(ref_x, ref_y, c='lime', s=60, marker='s', 
                          edgecolors='white', linewidth=1, zorder=10)
        
        # 粒子線を描画
        if hasattr(self, 'particle_lines') and self.particle_lines:
            for line_data in self.particle_lines:
                line_x = [p[0] for p in line_data['points']]
                line_y = [p[1] for p in line_data['points']]
                
                # 完成した粒子線
                ax.plot(line_x, line_y, color='magenta', linewidth=2, alpha=0.8, 
                       label=f'Particle Line {line_data["line_id"]}', zorder=9)
                ax.scatter(line_x, line_y, c='magenta', s=60, marker='o', 
                          edgecolors='white', linewidth=1, zorder=10)
        
        # 現在描画中の粒子線を描画
        if hasattr(self, 'current_line_points') and self.current_line_points and self.is_drawing_particle_line:
            current_x = [p[0] for p in self.current_line_points]
            current_y = [p[1] for p in self.current_line_points]
            
            if len(self.current_line_points) > 1:
                ax.plot(current_x, current_y, 'magenta', linewidth=3, alpha=0.7, 
                       label='Drawing Particle Line', zorder=9)
            ax.scatter(current_x, current_y, c='magenta', s=80, marker='o', 
                      edgecolors='white', linewidth=2, zorder=10)
        
        # 粒子検出結果を描画
        if hasattr(self, 'particle_detection_results') and self.particle_detection_results is not None:
            #print("DEBUG: Drawing particle detection results")
            
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            for particle in self.particle_detection_results:
                frame_idx = particle['frame']
                position = particle['position']
                confidence = particle['confidence']
                
                if is_rectangular_mode:
                    # 矩形モード：各フレームの中央位置
                    frame_height = int(self.actual_frame_height)
                    frame_center_y = frame_idx * frame_height + frame_height / 2
                    
                    if frame_center_y < self.kymograph_data.shape[0]:
                        # 信頼度に応じて色を変更
                        if confidence > 0.7:
                            color = 'red'
                            marker = 'o'
                        elif confidence > 0.3:
                            color = 'orange'
                            marker = 's'
                        else:
                            color = 'yellow'
                            marker = '^'
                        
                        ax.plot(position, frame_center_y, color=color, marker=marker, 
                               markersize=6, markeredgewidth=1, markeredgecolor='white')
                else:
                    # 直線モード：フレームインデックスをそのまま使用
                    if confidence > 0.7:
                        color = 'red'
                        marker = 'o'
                    elif confidence > 0.3:
                        color = 'orange'
                        marker = 's'
                    else:
                        color = 'yellow'
                        marker = '^'
                    
                    ax.plot(position, frame_idx, color=color, marker=marker, 
                           markersize=6, markeredgewidth=1, markeredgecolor='white')
        
        # トラッキング結果を線で描画
        if hasattr(self, 'particle_tracks') and self.particle_tracks is not None:
            #print("DEBUG: Drawing particle tracks")
            
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            # 各トラックを描画
            for track in self.particle_tracks:
                if len(track['particles']) < 2:
                    continue  # 2点未満のトラックは描画しない
                
                # トラックの座標を抽出
                track_x = []
                track_y = []
                
                for particle in track['particles']:
                    frame_idx = particle['frame']
                    position = particle['position']
                    
                    if is_rectangular_mode:
                        # 矩形モード：各フレームの中央位置
                        frame_height = int(self.actual_frame_height)
                        frame_center_y = frame_idx * frame_height + frame_height / 2
                        
                        if frame_center_y < self.kymograph_data.shape[0]:
                            track_x.append(position)
                            track_y.append(frame_center_y)
                    else:
                        # 直線モード：フレームインデックスをそのまま使用
                        track_x.append(position)
                        track_y.append(frame_idx)
                
                # トラックを線で描画（ピンク色で統一）
                if len(track_x) >= 2:
                    # トラックの線を描画
                    ax.plot(track_x, track_y, color='pink', linewidth=1, alpha=0.9, zorder=5)
                    # トラックの開始点を円で強調
                    ax.plot(track_x[0], track_y[0], color='pink', marker='o', 
                           markersize=6, markeredgecolor='white', markeredgewidth=1, zorder=6)
                    # トラックの終了点を四角で強調
                    ax.plot(track_x[-1], track_y[-1], color='pink', marker='s', 
                           markersize=6, markeredgecolor='white', markeredgewidth=1, zorder=6)
        
        # セミオート検出結果を描画（修正版）
        if (hasattr(self, 'semi_auto_results') and self.semi_auto_results is not None and 
            'reference_line_points' in self.semi_auto_results):
            
            edge_positions = self.semi_auto_results['edge_positions']
            time_points = self.semi_auto_results['time_points']
            
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            # 検出された点を表示
            if is_rectangular_mode:
                # 矩形モード：各フレームの中央位置
                frame_height = int(self.actual_frame_height)
                frame_indices = []
                for i in range(len(edge_positions)):
                    frame_center_y = i * frame_height + frame_height / 2
                    if frame_center_y < self.kymograph_data.shape[0]:
                        frame_indices.append(frame_center_y)
                    else:
                        frame_indices.append(i * frame_height)  # フォールバック
                
                ax.scatter(edge_positions[:len(frame_indices)], frame_indices, c='cyan', s=30, marker='+', 
                          linewidth=2, zorder=8)
                
                # 軌跡線を表示
                ax.plot(edge_positions[:len(frame_indices)], frame_indices, 'cyan', linewidth=1, 
                       alpha=0.6, zorder=7)
            else:
                # 直線モード：従来通り
                frame_indices = np.arange(len(edge_positions))
                ax.scatter(edge_positions, frame_indices, c='cyan', s=30, marker='+', 
                          linewidth=2, zorder=8)
                
                # 軌跡線を表示
                ax.plot(edge_positions, frame_indices, 'cyan', linewidth=1, 
                       alpha=0.6, zorder=7)
        
        # 凡例は表示しない
        
        # 縦軸をフレーム数×FrameTime（秒）に設定
        if hasattr(gv, 'FrameTime'):
            frame_time = gv.FrameTime / 1000.0  # msを秒に変換
            total_frames = len(self.image_stack)
            total_time = total_frames * frame_time  # 総時間（秒）
            ax.set_ylabel(f'Time (s)')
            
            # 矩形モードか直線モードかを判定
            is_rectangular_mode = (hasattr(self, 'drawn_box') and self.drawn_box is not None and 
                                 hasattr(self, 'actual_frame_height') and self.actual_frame_height is not None)
            
            frame_height = self.kymograph_data.shape[0] / total_frames  # 1フレームあたりの高さ
            
            if is_rectangular_mode:
                # 矩形モード: 各フレームの中心が時間軸に対応
                y_ticks = []
                y_tick_labels = []
                
                # フレーム数に応じてtick数を調整
                max_ticks = min(10, total_frames)  # 最大10個のtick
                tick_interval = max(1, total_frames // max_ticks)  # 間隔を計算
                
                for i in range(0, total_frames, tick_interval):
                    y_pos = i * frame_height + frame_height / 2  # フレームの中心
                    if y_pos < len(self.kymograph_data):
                        y_ticks.append(y_pos)
                        time_sec = i * frame_time
                        y_tick_labels.append(f'{time_sec:.1f}')
                
                # 最後のフレームも含める（間隔に含まれていない場合）
                if total_frames > 0 and (total_frames - 1) % tick_interval != 0:
                    final_y_pos = (total_frames - 1) * frame_height + frame_height / 2
                    if final_y_pos < len(self.kymograph_data):
                        y_ticks.append(final_y_pos)
                        final_time_sec = (total_frames - 1) * frame_time
                        y_tick_labels.append(f'{final_time_sec:.1f}')
                
                
            else:
                # 直線モード: 各フレームの開始時刻が時間軸に対応
                y_ticks = []
                y_tick_labels = []
                
                # フレーム数に応じてtick数を調整
                max_ticks = min(10, total_frames + 1)  # 最大10個のtick
                tick_interval = max(1, (total_frames + 1) // max_ticks)  # 間隔を計算
                
                for i in range(0, total_frames + 1, tick_interval):
                    y_pos = i * frame_height
                    if y_pos <= len(self.kymograph_data):
                        y_ticks.append(y_pos)
                        time_sec = i * frame_time
                        y_tick_labels.append(f'{time_sec:.1f}')
                
                
            
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)
        else:
            ax.set_ylabel('Frame')
            
        # 動的にフォントサイズを調整
        title = 'Kymograph'
        fontsize = self.calculateOptimalFontSize(ax, title, max_width_ratio=0.9)
        ax.set_title(title, fontsize=fontsize, pad=5)
        
        # レイアウトを調整してラベルが切れないようにする
        self.result_figure.tight_layout(pad=1.0)  # パディングを小さく
        
        self.result_canvas.draw()
        
        # Kymographが表示されたらフォーカスを設定
        self.result_canvas.setFocus()
        
        # 線維成長解析ボタンを有効にする
        if hasattr(self, 'auto_detection_button'):
            self.auto_detection_button.setEnabled(True)
            #print("DEBUG: auto_detection_button enabled")
        
        # Show Graphボタンを有効にする
        if hasattr(self, 'show_graph_button'):
            self.show_graph_button.setEnabled(True)
            #print("DEBUG: show_graph_button enabled")
    
    def calculateOptimalFontSize(self, ax, text, max_width_ratio=0.9, min_fontsize=6, max_fontsize=16):
        """ウィンドウサイズに応じて最適なフォントサイズを計算"""
        # キャンバスの幅を取得
        canvas_width = self.result_canvas.width()
        if canvas_width <= 0:
            return 10  # デフォルトサイズ
            
        # 利用可能な幅を計算（グラフの幅の90%まで）
        available_width = canvas_width * max_width_ratio
        
        # フォントサイズを二分探索で最適化
        left, right = min_fontsize, max_fontsize
        optimal_size = min_fontsize
        
        while left <= right:
            mid = (left + right) / 2
            ax.set_title(text, fontsize=mid, pad=5)
            
            # タイトルのバウンディングボックスを取得
            title_bbox = ax.title.get_window_extent()
            title_width = title_bbox.width
            
            if title_width <= available_width:
                optimal_size = mid
                left = mid + 0.5
            else:
                right = mid - 0.5
                
        return optimal_size
        
    def onKymographMousePress(self, event):
        """Kymographのマウスプレスイベント"""
        if not event.inaxes:
            return
        
        # クリック位置を取得
        click_x = event.xdata
        click_y = event.ydata
        
        if click_x is None or click_y is None:
            return
        
        # 修飾キーの状態をQtから直接取得する、最も確実な方法
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        shift_pressed = modifiers & QtCore.Qt.ShiftModifier
        ctrl_pressed = modifiers & QtCore.Qt.ControlModifier
        
        # 検出タイプを取得
        detection_type = self.detection_combo.currentText() if hasattr(self, 'detection_combo') else "Fibril end"
        
        # Particle検出結果がある場合の処理
        if self.particle_detection_results is not None:
            # Shift+左クリック: 粒子の削除
            if event.button == 1 and shift_pressed:
                nearest_particle, particle_index = self.findNearestParticle(click_x, click_y)
                if nearest_particle is not None:
                    self.removeParticle(particle_index)
                else:
                    pass
              
              # Ctrl+左クリック: 粒子の追加
            elif event.button == 1 and ctrl_pressed:
                self.addParticleAtPosition(click_x, click_y)

        
        # Particleモードで描画した線の位置削除
        if detection_type == "Particle" and hasattr(self, 'particle_lines') and self.particle_lines:
            if event.button == 1 and shift_pressed:
                # 描画中の線の位置を削除
                if hasattr(self, 'current_line_points') and self.current_line_points:
                    removed_point = self.removePointFromCurrentLine(click_x, click_y)
                    if removed_point:

                        return
                
                # 完成した線の位置を削除
                removed_point = self.removePointFromParticleLines(click_x, click_y)
                if removed_point:

                    return
        
    def onKymographMouseRelease(self, event):
        """Kymographのマウスリリースイベント"""
        pass
    

        
    def onKymographMouseMove(self, event):
        """Kymographのマウス移動イベント"""
        pass
        
    def onKymographScroll(self, event):
        """Kymographのスクロールイベント（ズーム機能）"""
        if event.inaxes:
            # マウスホイールでズーム
            ax = event.inaxes
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            # ズーム係数
            base_scale = 1.1
            if event.button == 'up':
                scale_factor = 1 / base_scale
            else:
                scale_factor = base_scale
            
            # 新しい表示範囲を計算
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            # マウス位置を中心にズーム
            relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
            
            ax.set_xlim([event.xdata - new_width * (1 - relx), event.xdata + new_width * relx])
            ax.set_ylim([event.ydata - new_height * (1 - rely), event.ydata + new_height * rely])
            
            # キャンバスを更新
            self.result_canvas.draw()
            
    def onKymographKeyPress(self, event):
        """Kymographのキーボードイベント処理"""
        # Ctrl+C でクリップボードにコピー
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            self.copyKymographToClipboard()
        else:
            # 他のキーイベントは無視する
            event.ignore()
            
    def onImageKeyPress(self, event):
        """AFM画像のキーボードイベント処理"""
        # Ctrl+C でクリップボードにコピー
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            self.copyImageToClipboard()
        else:
            # 他のキーイベントは無視する（QLabelにはkeyPressEventがないため）
            event.ignore()
            
    def showImageContextMenu(self, pos):
        """AFM画像の右クリックメニューを表示"""
        menu = QtWidgets.QMenu(self)
        
        # ラインが存在する場合のみ複数点機能を表示
        if self.roi_line and self.multi_point_line:
            # Multiple Points メニュー
            if self.multi_point_line['is_straight']:
                multi_point_action = menu.addAction("Multiple Points")
                multi_point_action.triggered.connect(self.convertToMultiPointLine)
            else:
                reset_action = menu.addAction("Reset to Straight Line")
                reset_action.triggered.connect(self.resetToStraightLine)
            
            menu.addSeparator()
        
        # Copy メニュー
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.copyImageToClipboard)
        
        # メニューを表示
        menu.exec_(self.image_label.mapToGlobal(pos))
        
    def showKymographContextMenu(self, event):
        """Kymographの右クリックメニューを表示"""
        if not event.inaxes:
            return
            
        # 現在のマウス位置を取得
        cursor_pos = QtWidgets.QApplication.primaryScreen().geometry().center()
        
        # ポップアップメニューを作成
        menu = QtWidgets.QMenu(self)
        
        # Reset アクション
        reset_action = menu.addAction("Reset")
        reset_action.triggered.connect(self.resetKymographView)
        
        # Copy アクション
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.copyKymographToClipboard)
        
        # Contrast サブメニュー
        contrast_menu = menu.addMenu("Contrast")
        
        # ヒストグラムパーセンタイルオプション
        percentiles = [1, 5, 10]
        for p in percentiles:
            action = contrast_menu.addAction(f"Percentile {p}-{100-p}")
            action.triggered.connect(lambda checked, p=p: self.optimizeContrastPercentile(p))
        
        # Otsu二値化 オプション
        otsu_action = contrast_menu.addAction("Otsu Thresholding")
        otsu_action.triggered.connect(self.optimizeContrastOtsu)
        
        # ヒストグラム オプション
        histogram_action = contrast_menu.addAction("Histogram")
        histogram_action.triggered.connect(self.showHistogram)
        
        # メニューを表示（現在のマウス位置で表示）
        menu.exec_(QtGui.QCursor.pos())
        
    def showKymographContextMenuQt(self, pos):
        """Qtのコンテキストメニューイベント用"""
        # ポップアップメニューを作成
        menu = QtWidgets.QMenu(self)
        
        # Reset アクション
        reset_action = menu.addAction("Reset")
        reset_action.triggered.connect(self.resetKymographView)
        
        # Copy アクション
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.copyKymographToClipboard)
        
        # Contrast サブメニュー
        contrast_menu = menu.addMenu("Contrast")
        
        # ヒストグラムパーセンタイルオプション
        percentiles = [1, 5, 10]
        for p in percentiles:
            action = contrast_menu.addAction(f"Percentile {p}-{100-p}")
            action.triggered.connect(lambda checked, p=p: self.optimizeContrastPercentile(p))
        
        # Otsu二値化 オプション
        otsu_action = contrast_menu.addAction("Otsu Thresholding")
        otsu_action.triggered.connect(self.optimizeContrastOtsu)
        
        # ヒストグラム オプション
        histogram_action = contrast_menu.addAction("Histogram")
        histogram_action.triggered.connect(self.showHistogram)
        
        # メニューを表示
        menu.exec_(self.result_canvas.mapToGlobal(pos))
        
    def resetKymographView(self):
        """Kymographの表示を元のサイズにリセット"""
        if hasattr(self, 'result_figure') and self.result_figure:
            ax = self.result_figure.axes[0]
            if ax:
                # 元の表示範囲に戻す
                ax.autoscale()
                self.result_canvas.draw()
                
    def copyKymographToClipboard(self):
        """Kymographをクリップボードにコピー"""
        if hasattr(self, 'result_figure') and self.result_figure:
            # フィギュアをクリップボードにコピー
            self.result_canvas.draw()
            pixmap = self.result_canvas.grab()
            QtWidgets.QApplication.clipboard().setPixmap(pixmap)
            
    def copyImageToClipboard(self):
        """AFM画像をクリップボードにコピー"""
        if not hasattr(gv, 'aryData') or gv.aryData is None:
            return
            
        try:
            # 現在表示されている画像を取得
            current_pixmap = self.image_label.pixmap()
            if current_pixmap is None:
                return
                
            # クリップボードにコピー
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setPixmap(current_pixmap)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Error copying image to clipboard: {e}")
            
    def optimizeContrastPercentile(self, percentile):
        """パーセンタイルベースのコントラスト最適化"""
        if self.kymograph_data is None:
            return
            
        # パーセンタイルを計算
        vmin = np.percentile(self.kymograph_data, percentile)
        vmax = np.percentile(self.kymograph_data, 100 - percentile)
        
        # コントラスト設定を保存
        self.current_contrast = (vmin, vmax)

        
        # displayKymographを呼び出して粒子トラッキング結果も含めて再描画
        self.displayKymograph()

    def _require_skimage(self, feature_name="This feature"):
        """skimage が利用可能なら True。未インストールの場合はインストールを促して False。"""
        try:
            import skimage
            return True
        except ImportError:
            _frozen = getattr(sys, "frozen", False)
            if _frozen:
                msg = (
                    f"{feature_name} requires scikit-image.\n"
                    "この機能には scikit-image が必要です。モジュールがインストールされていません。\n\n"
                    "scikit-image is not bundled with this installation.\n"
                    "このモジュールはこのパッケージに含まれていません。"
                )
            else:
                msg = (
                    f"{feature_name} requires scikit-image.\n"
                    "この機能には scikit-image が必要です。モジュールがインストールされていません。\n\n"
                    "Install with: pip install scikit-image"
                )
            QtWidgets.QMessageBox.warning(self, "Package required", msg)
            return False
        
    def optimizeContrastCLAHE(self):
        """CLAHEベースのコントラスト最適化"""
        if self.kymograph_data is None:
            return
        if not self._require_skimage("CLAHE contrast optimization"):
            return
        try:
            from skimage import exposure
            
            # データを0-1の範囲に正規化
            data_normalized = (self.kymograph_data - np.min(self.kymograph_data)) / (np.max(self.kymograph_data) - np.min(self.kymograph_data))
            
            # CLAHEを適用
            kymograph_clahe = exposure.equalize_adapthist(data_normalized, clip_limit=0.02)
            
            # コントラスト設定を保存（CLAHEの場合は0-1の範囲）
            self.current_contrast = (0.0, 1.0)
    
            
            # displayKymographを呼び出して粒子トラッキング結果も含めて再描画
            self.displayKymograph()
            
        except ImportError:
            self._require_skimage("CLAHE contrast optimization")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Error occurred during CLAHE processing: {e}")
        
    def optimizeContrastOtsu(self):
        """Otsu二値化ベースのコントラスト最適化"""
        if self.kymograph_data is None:
            return
        if not self._require_skimage("Otsu thresholding"):
            return
        try:
            from skimage import filters
            
            # データを0-1の範囲に正規化
            data_normalized = (self.kymograph_data - np.min(self.kymograph_data)) / (np.max(self.kymograph_data) - np.min(self.kymograph_data))
            
            # Otsu閾値を計算
            otsu_threshold = filters.threshold_otsu(data_normalized)
            
            # 二値化を適用
            kymograph_binary = data_normalized > otsu_threshold
            
            # コントラスト設定を保存（Otsuの場合は0-1の範囲）
            self.current_contrast = (0.0, 1.0)
    
            
            # displayKymographを呼び出して粒子トラッキング結果も含めて再描画
            self.displayKymograph()
            
        except ImportError:
            self._require_skimage("Otsu thresholding")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Error occurred during Otsu thresholding: {e}")
    
    def showHistogram(self):
        """ヒストグラムウィンドウを表示"""
        if self.kymograph_data is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No kymograph data available.")
            return
            
        # 既にヒストグラムウィンドウが開いている場合は前面に表示
        if hasattr(self, 'histogram_window') and self.histogram_window is not None:
            if self.histogram_window.isVisible():
                self.histogram_window.raise_()
                self.histogram_window.activateWindow()
                return
            else:
                # ウィンドウが閉じられている場合は削除
                self.histogram_window = None
        
        # 新しいヒストグラムウィンドウを作成
        self.histogram_window = KymographHistogramWindow(self)
        self.histogram_window.contrastChanged.connect(self.updateKymographContrast)
        self.histogram_window.updateHistogramData(self.kymograph_data.flatten())
        self.histogram_window.show()
    
    def updateKymographContrast(self, min_val, max_val):
        """ヒストグラムからコントラスト変更を受け取ってkymographを更新"""
        if self.kymograph_data is None:
            return
        
        # コントラスト設定を保存
        self.current_contrast = (min_val, max_val)

        
        # displayKymographを呼び出して粒子トラッキング結果も含めて再描画
        self.displayKymograph()
        
        # Kymographが表示されたらフォーカスを設定
        self.result_canvas.setFocus()
        
    def getDefaultSaveFolder(self):
        """デフォルトの保存フォルダを取得（File Listの選択ファイルと同じディレクトリ）"""
        
        try:
            if hasattr(gv, 'files') and gv.files and hasattr(gv, 'currentFileNum'):
                if 0 <= gv.currentFileNum < len(gv.files):
                    current_file = gv.files[gv.currentFileNum]
                    folder = os.path.dirname(current_file)
                    return folder
        except Exception as e:
            print(f"[WARNING] Failed to get default save folder: {e}")
        
        # フォールバック: 現在のディレクトリまたはホームディレクトリ
        if hasattr(gv, 'directoryName') and gv.directoryName:
            return gv.directoryName
        else:
            home_dir = os.path.expanduser("~")
            return home_dir

    def getDefaultFilename(self):
        """デフォルトのファイル名を生成（asdファイル名+_frフレーム番号_kymograph）"""
        
        try:
            if hasattr(gv, 'files') and gv.files and hasattr(gv, 'currentFileNum'):
                if 0 <= gv.currentFileNum < len(gv.files):
                    current_file = gv.files[gv.currentFileNum]
                    # asdファイル名を取得（拡張子なし）
                    asd_filename = os.path.splitext(os.path.basename(current_file))[0]
                    # フレーム番号を取得
                    frame_num = getattr(gv, 'index', 0)
                    filename = f"{asd_filename}_fr{frame_num:02d}_kymograph"
                    return filename
        except Exception as e:
            print(f"[WARNING] Failed to get default filename: {e}")
        
        # フォールバック: デフォルトファイル名
        frame_num = getattr(gv, 'index', 0)
        fallback_filename = f"kymograph_fr{frame_num:02d}"
        return fallback_filename

    def saveKymograph(self):
        """Save Kymograph"""
        if self.kymograph_data is None:
            return
            
        default_folder = self.getDefaultSaveFolder()
        default_filename = self.getDefaultFilename() + ".csv"  # デフォルトでCSV拡張子を追加
        
        # ファイル名を取得
        # macOSのNSSavePanel警告を抑制
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Kymograph", 
                os.path.join(default_folder, default_filename),
                "CSV Files (*.csv);;NPZ Files (*.npz);;All Files (*)"
            )
        
        if not file_path:
            return
            
        try:
            if file_path.endswith('.csv'):
                # Save as CSV format with frame information
                df = pd.DataFrame(self.kymograph_data)
                
                # ヘッダーに位置情報を追加
                if hasattr(self, 'drawn_box') and self.drawn_box is not None:
                    # 矩形の場合
                    box_width = self.drawn_box['box_width']
                    box_length = self.drawn_box['box_length']
                    header_info = [f"Position_{i+1}" for i in range(df.shape[1])]
                    df.columns = header_info
                else:
                    # 直線の場合
                    header_info = [f"Position_{i+1}" for i in range(df.shape[1])]
                    df.columns = header_info
                
                # 矩形の場合はフレームごとにグループ化して保存
                if (hasattr(self, 'drawn_box') and self.drawn_box is not None and 
                    hasattr(self, 'actual_frame_height') and self.actual_frame_height is not None):
                    # 矩形の場合：実際のフレーム高さを使用
                    with open(file_path, 'w', newline='') as f:
                        # ヘッダーを書き込み
                        f.write(',' + ','.join(header_info) + '\n')
                        
                        # 実際のフレーム高さを使用（重要な修正点！）
                        frame_height = int(self.actual_frame_height)
                        
                        # 各フレームのデータを書き込み
                        frame_idx = 0
                        for i in range(0, len(self.kymograph_data), frame_height):
                            if hasattr(gv, 'FrameTime'):
                                frame_time = gv.FrameTime / 1000.0  # msを秒に変換
                                time_sec = frame_idx * frame_time
                                frame_label = f"Frame_{frame_idx+1}_Time_{time_sec:.3f}s"
                            else:
                                frame_label = f"Frame_{frame_idx+1}"
                            
                            # フレームラベル行を書き込み（データなし）
                            f.write(frame_label + ',' + ','.join([''] * len(header_info)) + '\n')
                            
                            # フレームのデータ行を書き込み（ラベルなし）
                            for j in range(frame_height):
                                if i + j < len(self.kymograph_data):
                                    row_data = self.kymograph_data[i + j]
                                    f.write(',' + ','.join([str(val) for val in row_data]) + '\n')
                            
                            # フレーム間で1行空ける（最後のフレーム以外）
                            total_frames = len(self.kymograph_data) // frame_height
                            if frame_idx < total_frames - 1:
                                f.write('\n')
                            
                            frame_idx += 1
                else:
                    # 直線の場合：従来通り
                    frame_info = []
                    for i in range(len(self.kymograph_data)):
                        if hasattr(gv, 'FrameTime'):
                            frame_time = gv.FrameTime / 1000.0  # msを秒に変換
                            time_sec = i * frame_time
                            frame_info.append(f"Frame_{i+1}_Time_{time_sec:.3f}s")
                        else:
                            frame_info.append(f"Frame_{i+1}")
                    
                    # インデックスにフレーム情報を設定
                    df.index = frame_info
                    df.to_csv(file_path, index=True, header=True)
                    
            elif file_path.endswith('.npz'):
                # Save as NPZ format
                np.savez(file_path, kymograph=self.kymograph_data)
            else:
                # デフォルトはCSV（フレーム情報付き）
                df = pd.DataFrame(self.kymograph_data)
                
                # ヘッダーに位置情報を追加
                if hasattr(self, 'drawn_box') and self.drawn_box is not None:
                    # 矩形の場合
                    box_width = self.drawn_box['box_width']
                    box_length = self.drawn_box['box_length']
                    header_info = [f"Position_{i+1}" for i in range(df.shape[1])]
                    df.columns = header_info
                else:
                    # 直線の場合
                    header_info = [f"Position_{i+1}" for i in range(df.shape[1])]
                    df.columns = header_info
                
                # 矩形の場合はフレームごとにグループ化して保存
                if (hasattr(self, 'drawn_box') and self.drawn_box is not None and 
                    hasattr(self, 'actual_frame_height') and self.actual_frame_height is not None):
                    # 矩形の場合：実際のフレーム高さを使用
                    with open(file_path, 'w', newline='') as f:
                        # ヘッダーを書き込み
                        f.write(',' + ','.join(header_info) + '\n')
                        
                        # 実際のフレーム高さを使用（重要な修正点！）
                        frame_height = int(self.actual_frame_height)
                        
                        # 各フレームのデータを書き込み
                        frame_idx = 0
                        for i in range(0, len(self.kymograph_data), frame_height):
                            if hasattr(gv, 'FrameTime'):
                                frame_time = gv.FrameTime / 1000.0  # msを秒に変換
                                time_sec = frame_idx * frame_time
                                frame_label = f"Frame_{frame_idx+1}_Time_{time_sec:.3f}s"
                            else:
                                frame_label = f"Frame_{frame_idx+1}"
                            
                            # フレームラベル行を書き込み（データなし）
                            f.write(frame_label + ',' + ','.join([''] * len(header_info)) + '\n')
                            
                            # フレームのデータ行を書き込み（ラベルなし）
                            for j in range(frame_height):
                                if i + j < len(self.kymograph_data):
                                    row_data = self.kymograph_data[i + j]
                                    f.write(',' + ','.join([str(val) for val in row_data]) + '\n')
                            
                            # フレーム間で1行空ける（最後のフレーム以外）
                            total_frames = len(self.kymograph_data) // frame_height
                            if frame_idx < total_frames - 1:
                                f.write('\n')
                            
                            frame_idx += 1
                else:
                    # 直線の場合：従来通り
                    frame_info = []
                    for i in range(len(self.kymograph_data)):
                        if hasattr(gv, 'FrameTime'):
                            frame_time = gv.FrameTime / 1000.0  # msを秒に変換
                            time_sec = i * frame_time
                            frame_info.append(f"Frame_{i+1}_Time_{time_sec:.3f}s")
                        else:
                            frame_info.append(f"Frame_{i+1}")
                    
                    # インデックスにフレーム情報を設定
                    df.index = frame_info
                    df.to_csv(file_path, index=True, header=True)
                
            QtWidgets.QMessageBox.information(self, "Save Complete", f"Kymograph saved:\n{file_path}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error occurred during save:\n{e}")
            
    def loadWindowSettings(self):
        """ウィンドウ設定を読み込み"""
        try:
            if hasattr(gv, 'windowSettings') and 'KymographWindow' in gv.windowSettings:
                settings = gv.windowSettings['KymographWindow']
                if 'geometry' in settings:
                    x, y, width, height = settings['geometry']
                    self.setGeometry(x, y, width, height)
                if 'visible' in settings and settings['visible']:
                    self.show()
                if 'normalize_enabled' in settings:
                    self.normalize_enabled = settings['normalize_enabled']
                    if hasattr(self, 'normalize_checkbox'):
                        self.normalize_checkbox.setChecked(self.normalize_enabled)
                if 'dh_nm' in settings:
                    self.dh_nm = settings['dh_nm']
                    if hasattr(self, 'dh_spin'):
                        self.dh_spin.setValue(self.dh_nm)
                if 'dw_nm' in settings:
                    self.dw_nm = settings['dw_nm']
                    if hasattr(self, 'dw_spin'):
                        self.dw_spin.setValue(self.dw_nm)
        except Exception as e:
            print(f"Error loading window settings: {e}")
            # デフォルトの位置とサイズを設定
            self.setGeometry(150, 150, 1000, 800)
            
    def saveWindowSettings(self):
        """Save window settings"""
        try:
            if not hasattr(gv, 'windowSettings'):
                gv.windowSettings = {}
            gv.windowSettings['KymographWindow'] = {
                'geometry': self.geometry().getRect(),
                'visible': self.isVisible(),
                'normalize_enabled': self.normalize_enabled,
                'dh_nm': self.dh_nm,
                'dw_nm': self.dw_nm
            }
        except Exception as e:
            print(f"Error saving window settings: {e}")
            
    def closeEvent(self, event):
        """ウィンドウが閉じられる時の処理"""
        try:
            # サブウィンドウを閉じる
            sub_windows = ['histogram_window', 'graph_window', 'debug_window', 'test_window']
            for window_attr in sub_windows:
                if hasattr(self, window_attr) and getattr(self, window_attr) is not None:
                    try:
                        getattr(self, window_attr).close()
                        setattr(self, window_attr, None)
                    except RuntimeError:
                        print(f"[WARNING] {window_attr} C++ object already deleted")
                    except Exception as e:
                        print(f"[WARNING] Failed to close {window_attr}: {e}")
            
            # ウィンドウ管理システムから登録を削除
            try:
                from window_manager import unregister_pyNuD_window
                unregister_pyNuD_window(self)
            except ImportError:
                pass
            except Exception as e:
                print(f"[WARNING] Failed to unregister KymographWindow: {e}")
            
            # 設定を保存
            try:
                self.saveWindowSettings()
            except Exception as e:
                print(f"[WARNING] Failed to save window settings: {e}")
            
            self.was_closed = True  # 閉じられたことを通知
            
            # プラグインとして開いた場合のツールバーアクションのハイライトを解除
            try:
                if gv is not None and hasattr(gv, 'main_window') and gv.main_window:
                    mw = gv.main_window
                    if hasattr(mw, 'setActionHighlight') and hasattr(mw, 'plugin_actions'):
                        action = mw.plugin_actions.get("Kymograph")
                        if action is not None:
                            mw.setActionHighlight(action, False)
            except Exception as e:
                print(f"[WARNING] Failed to reset kymograph action highlight: {e}")
            
            # Qtのデフォルトのクローズ処理
            try:
                super().closeEvent(event)
            except RuntimeError:
                print("[WARNING] C++ object already deleted during super().closeEvent()")
            except Exception as e:
                print(f"[WARNING] Failed to call super().closeEvent(): {e}")
            
            event.accept()
            
        except Exception as e:
            print(f"[ERROR] Unexpected error in KymographWindow closeEvent: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生してもイベントは受け入れる
            event.accept()

    def checkEndpointClick(self, pos):
        """マウス位置が線の端点かどうかをチェック"""
        if not self.roi_line:
            return None
        
        # 画像座標に変換
        mouse_x, mouse_y = self.widgetToImageCoords(pos)
        
        # 線の端点を画像座標で取得
        start_x, start_y = self.roi_line['start_x'], self.roi_line['start_y']
        end_x, end_y = self.roi_line['end_x'], self.roi_line['end_y']
        
        # 端点までの距離を計算
        dist_to_start = ((mouse_x - start_x) ** 2 + (mouse_y - start_y) ** 2) ** 0.5
        dist_to_end = ((mouse_x - end_x) ** 2 + (mouse_y - end_y) ** 2) ** 0.5
        
        #print(f"DEBUG: Mouse pos: ({mouse_x}, {mouse_y})")
        #print(f"DEBUG: Start pos: ({start_x}, {start_y}), End pos: ({end_x}, {end_y})")
        #print(f"DEBUG: Dist to start: {dist_to_start}, Dist to end: {dist_to_end}, Radius: {self.endpoint_radius}")
        
        # 端点の検出半径内かチェック
        if dist_to_start <= self.endpoint_radius:
            #print("DEBUG: Start endpoint detected")
            return 'start'
        elif dist_to_end <= self.endpoint_radius:
            #print("DEBUG: End endpoint detected")
            return 'end'
        
        #print("DEBUG: No endpoint detected")
        return None
    
    def checkLineCenterClick(self, pos):
        """マウス位置が線の中央部分かどうかをチェック"""
        if not self.roi_line:
            return False
        
        # 画像座標に変換
        mouse_x, mouse_y = self.widgetToImageCoords(pos)
        
        # 線の端点を画像座標で取得
        start_x, start_y = self.roi_line['start_x'], self.roi_line['start_y']
        end_x, end_y = self.roi_line['end_x'], self.roi_line['end_y']
        
        # 線の中心点を計算
        center_x = (start_x + end_x) / 2
        center_y = (start_y + end_y) / 2
        
        # 線の長さを計算
        line_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
        
        # 線の中央部分の範囲を定義（線の長さの20%）
        center_range = line_length * 0.2
        
        # マウス位置から線の中心までの距離を計算
        dist_to_center = np.sqrt((mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2)
        
        #print(f"DEBUG: Mouse pos: ({mouse_x}, {mouse_y})")
        #print(f"DEBUG: Center pos: ({center_x}, {center_y})")
        #print(f"DEBUG: Line length: {line_length}, Center range: {center_range}")
        #print(f"DEBUG: Dist to center: {dist_to_center}")
        
        # 線の中央部分の範囲内かチェック
        if dist_to_center <= center_range:
            #print("DEBUG: Center area detected")
            return True
        
        #print("DEBUG: Center area not detected")
        return False
    
    def updateLinePosition(self, new_pos):
        """線全体の位置を更新（中央ドラッグ）"""
        if not self.roi_line or not self.drag_start_pos:
            return
        
        # 現在のマウス位置を画像座標に変換
        current_image_x, current_image_y = self.widgetToImageCoords(new_pos)
        
        # ドラッグ開始位置を画像座標に変換
        start_image_x, start_image_y = self.widgetToImageCoords(self.drag_start_pos)
        
        # 画像座標での移動量を計算
        dx = current_image_x - start_image_x
        dy = current_image_y - start_image_y
        
        #print(f"DEBUG: Line position update - dx: {dx}, dy: {dy}")
        #print(f"DEBUG: Current image pos: ({current_image_x}, {current_image_y})")
        #print(f"DEBUG: Start image pos: ({start_image_x}, {start_image_y})")
        
        # 線の両端点を同じ量だけ移動
        self.roi_line['start_x'] += dx
        self.roi_line['start_y'] += dy
        self.roi_line['end_x'] += dx
        self.roi_line['end_y'] += dy
        
        #print(f"DEBUG: New line start: ({self.roi_line['start_x']}, {self.roi_line['start_y']})")
        #print(f"DEBUG: New line end: ({self.roi_line['end_x']}, {self.roi_line['end_y']})")
        
        # ドラッグ開始位置を更新
        self.drag_start_pos = new_pos
        
        # 矩形も更新
        if self.drawn_box:
            self.drawBox()
    
    def updateLineEndpoint(self, new_pos):
        """線の端点を更新"""
        if not self.roi_line or not self.drag_point:
            return
        
        # 新しい座標を画像座標に変換
        new_x, new_y = self.widgetToImageCoords(new_pos)
        
        # 端点を更新
        if self.drag_point == 'start':
            self.roi_line['start_x'] = new_x
            self.roi_line['start_y'] = new_y
        elif self.drag_point == 'end':
            self.roi_line['end_x'] = new_x
            self.roi_line['end_y'] = new_y
        
        # 複数点ラインも更新
        if self.multi_point_line:
            if self.drag_point == 'start':
                self.multi_point_line['points'][0] = (new_x, new_y)
            elif self.drag_point == 'end':
                self.multi_point_line['points'][-1] = (new_x, new_y)
        
        # 矩形も更新
        if self.drawn_box:
            self.drawBox()

    def convertToMultiPointLine(self):
        """直線を複数点ラインに変換"""
        if not self.multi_point_line or self.multi_point_line['is_straight']:
            # 直線を複数点に変換（中間点を追加）
            start_point = self.multi_point_line['points'][0]
            end_point = self.multi_point_line['points'][1]
            
            # 中間点を追加
            mid_x = (start_point[0] + end_point[0]) / 2
            mid_y = (start_point[1] + end_point[1]) / 2
            
            self.multi_point_line['points'] = [start_point, (mid_x, mid_y), end_point]
            self.multi_point_line['is_straight'] = False
            
            # 複数点編集モードを有効化
            self.multi_point_mode = True
            
            # 矩形BOXを消去（複数点ラインに変換するため）
            if hasattr(self, 'drawn_box') and self.drawn_box is not None:
                self.drawn_box = None
                #print("DEBUG: Rectangle box cleared due to multi-point line conversion")
            
            # Draw Boxボタンを無効化（複数点ラインのため）
            if hasattr(self, 'draw_box_button'):
                self.draw_box_button.setEnabled(False)
            
            # 画像を再表示
            self.updateImageDisplay()
    
    def resetToStraightLine(self):
        """複数点ラインを直線にリセット"""
        if self.multi_point_line and not self.multi_point_line['is_straight']:
            # 最初と最後の点のみを使用
            start_point = self.multi_point_line['points'][0]
            end_point = self.multi_point_line['points'][-1]
            
            self.multi_point_line['points'] = [start_point, end_point]
            self.multi_point_line['is_straight'] = True
            self.multi_point_mode = False
            self.selected_point_index = None
            
            # ROIラインも更新
            if self.roi_line:
                self.roi_line['start_x'] = start_point[0]
                self.roi_line['start_y'] = start_point[1]
                self.roi_line['end_x'] = end_point[0]
                self.roi_line['end_y'] = end_point[1]
            
            # Draw Boxボタンを有効化（直線に戻ったため）
            if hasattr(self, 'draw_box_button'):
                self.draw_box_button.setEnabled(True)
            
            # 画像を再表示
            self.updateImageDisplay()
    
    def addPointToLine(self, pos):
        """ラインに新しい点を追加"""
        if not self.multi_point_line or self.multi_point_mode:
            # 画像座標に変換
            image_x, image_y = self.widgetToImageCoords(pos)
            
            # 最も近い線分を見つけて、その位置に点を挿入
            points = self.multi_point_line['points']
            min_dist = float('inf')
            insert_index = 1
            
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                # 点から線分までの距離を計算
                dist = self.pointToLineDistance((image_x, image_y), p1, p2)
                
                if dist < min_dist:
                    min_dist = dist
                    insert_index = i + 1
            
            # 新しい点を挿入
            self.multi_point_line['points'].insert(insert_index, (image_x, image_y))
            self.multi_point_line['is_straight'] = False
            
            # 矩形BOXを消去（複数点ラインになったため）
            if hasattr(self, 'drawn_box') and self.drawn_box is not None:
                self.drawn_box = None
                #print("DEBUG: Rectangle box cleared due to adding point to line")
            
            # Draw Boxボタンを無効化（複数点ラインになったため）
            if hasattr(self, 'draw_box_button'):
                self.draw_box_button.setEnabled(False)
            
            # 画像を再表示
            self.updateImageDisplay()
    
    def pointToLineDistance(self, point, line_start, line_end):
        """点から線分までの距離を計算"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 線分の長さ
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # 点から線分への垂線の足を計算
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length**2)))
        
        # 垂線の足の座標
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # 距離を計算
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def checkPointClick(self, pos):
        """点のクリックをチェック"""
        if not self.multi_point_line or not self.multi_point_mode:
            return None
        
        # 画像座標に変換
        image_x, image_y = self.widgetToImageCoords(pos)
        
        # 各点との距離をチェック
        for i, point in enumerate(self.multi_point_line['points']):
            dist = np.sqrt((image_x - point[0])**2 + (image_y - point[1])**2)
            if dist <= self.point_radius:
                return i
        
        return None
    
    def updateMultiPointPosition(self, new_pos, point_index):
        """複数点ラインの点の位置を更新"""
        if not self.multi_point_line or point_index >= len(self.multi_point_line['points']):
            return
        
        # 新しい座標を画像座標に変換
        new_x, new_y = self.widgetToImageCoords(new_pos)
        
        # 点の位置を更新
        self.multi_point_line['points'][point_index] = (new_x, new_y)
        
        # 端点の場合はROIラインも更新
        if point_index == 0 and self.roi_line:
            self.roi_line['start_x'] = new_x
            self.roi_line['start_y'] = new_y
        elif point_index == len(self.multi_point_line['points']) - 1 and self.roi_line:
            self.roi_line['end_x'] = new_x
            self.roi_line['end_y'] = new_y

    def imageToWidgetCoords(self, image_x, image_y):
        """画像座標をウィジェット座標に変換（物理サイズのアスペクト比を考慮）"""
        if not hasattr(gv, 'aryData') or gv.aryData is None:
            return 0, 0
        
        # 画像のサイズ
        image_height, image_width = gv.aryData.shape
        
        # ラベルのサイズ
        label_size = self.image_label.size()
        
        # ラベルのサイズが取得できない場合はデフォルト値を使用
        if label_size.width() <= 0 or label_size.height() <= 0:
            return 0, 0
        
        # 物理サイズのアスペクト比を取得（必須）
        if not hasattr(gv, 'XScanSize') or not hasattr(gv, 'YScanSize') or gv.YScanSize <= 0:
            print("ERROR: Physical size information not available in imageToWidgetCoords")
            return 0, 0
        
        physical_aspect_ratio = gv.XScanSize / gv.YScanSize
        
        # 物理サイズのアスペクト比に基づいて表示サイズを計算
        if physical_aspect_ratio > 1.0:
            # 横長の場合
            display_width = min(label_size.width(), int(label_size.height() * physical_aspect_ratio))
            display_height = int(display_width / physical_aspect_ratio)
        else:
            # 縦長の場合
            display_height = min(label_size.height(), int(label_size.width() / physical_aspect_ratio))
            display_width = int(display_height * physical_aspect_ratio)
        
        # オフセット計算（画像を中央に配置）
        offset_x = (label_size.width() - display_width) / 2
        offset_y = (label_size.height() - display_height) / 2
        
        # 画像表示でflipudしているため、Y軸の反転が必要
        image_y = image_height - 1 - image_y
        
        # ピクセルベースの変換
        scale_x = display_width / image_width
        scale_y = display_height / image_height
        
        widget_x = image_x * scale_x + offset_x
        widget_y = image_y * scale_y + offset_y
        
        return widget_x, widget_y

    def onDhChanged(self, value):
        """Δhが変更された時の処理"""
        self.dh_nm = value
        #print(f"DEBUG: onDhChanged - New dh_nm: {self.dh_nm}")
        # 矩形を再描画
        if self.roi_line:
            self.drawBox()
            # kymographを再作成（矩形サイズが変更されたため）
            if self.kymograph_data is not None:
                self.createKymograph()
        
    def onDwChanged(self, value):
        """Δwが変更された時の処理"""
        self.dw_nm = value
        #print(f"DEBUG: onDwChanged - New dw_nm: {self.dw_nm}")
        # 矩形を再描画
        if self.roi_line:
            self.drawBox()
            # kymographを再作成（矩形サイズが変更されたため）
            if self.kymograph_data is not None:
                self.createKymograph()
        
    def drawBox(self):
        """線を中心に矩形を描画（物理サイズベース）"""
        if not self.roi_line:
            return
        
        # 線の中心点を基準に計算（元の方法に戻す）
        center_x = (self.roi_line['start_x'] + self.roi_line['end_x']) / 2
        center_y = (self.roi_line['start_y'] + self.roi_line['end_y']) / 2
        
        # 線の角度を計算（最も水平になる小さな回転角度）
        dx = self.roi_line['end_x'] - self.roi_line['start_x']
        dy = self.roi_line['end_y'] - self.roi_line['start_y']
        
        # 角度を計算（-π/2 から π/2 の範囲に正規化）
        angle = np.arctan2(dy, dx)
        
        # 角度を最も水平になる方向に調整
        # 角度が π/2 より大きい場合は π を引く
        # 角度が -π/2 より小さい場合は π を足す
        if angle > np.pi/2:
            angle -= np.pi
        elif angle < -np.pi/2:
            angle += np.pi
        
        # 線の長さを計算（ピクセル）
        line_length_pixels = np.sqrt(dx**2 + dy**2)
        
        # 物理サイズからピクセルサイズへの変換
        if hasattr(gv, 'XScanSize') and hasattr(gv, 'YScanSize') and gv.XScanSize > 0 and gv.YScanSize > 0:
            # 物理サイズが利用可能な場合
            image_height, image_width = gv.aryData.shape
            pixel_size_x = gv.XScanSize / image_width  # nm/pixel
            pixel_size_y = gv.YScanSize / image_height  # nm/pixel
            
            # nmからピクセル数に変換
            dh_pixels = self.dh_nm / pixel_size_y
            dw_pixels = self.dw_nm / pixel_size_x
        else:
            # 物理サイズが利用できない場合、デフォルト値を使用
            dh_pixels = self.dh_nm / 10.0  # 仮のスケール
            dw_pixels = self.dw_nm / 10.0  # 仮のスケール
        
        # 矩形のサイズを計算（線の長さ + 余白）
        # 線に平行な方向のサイズ = 線の長さ + 2*Δw
        # 線に垂直な方向のサイズ = 2*Δh
        box_length = line_length_pixels + 2 * dw_pixels  # 線の長さ + 両側の余白
        box_width = 2 * dh_pixels  # 上下の余白
        
        # 矩形の4つの角を計算（ピクセル座標）
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # 矩形の角の相対座標（線に垂直・平行な方向）
        corners_relative = [
            (-box_length/2, -box_width/2),  # 左上
            (box_length/2, -box_width/2),   # 右上
            (box_length/2, box_width/2),    # 右下
            (-box_length/2, box_width/2)    # 左下
        ]
        
        # 回転と平行移動を適用（ピクセル座標）
        corners = []
        for dx_rel, dy_rel in corners_relative:
            # 回転（線の角度に合わせて）
            x_rot = dx_rel * cos_angle - dy_rel * sin_angle
            y_rot = dx_rel * sin_angle + dy_rel * cos_angle
            
            # 平行移動（線の中心に）
            x_abs = center_x + x_rot
            y_abs = center_y + y_rot
            
            corners.append((x_abs, y_abs))
        
        # Save rectangle data
        self.drawn_box = {
            'center_x': center_x,
            'center_y': center_y,
            'angle': angle,
            'corners': corners,
            'box_length': box_length,
            'box_width': box_width
        }
        
        #print(f"DEBUG: drawBox - Center: ({center_x:.1f}, {center_y:.1f})")
        #print(f"DEBUG: drawBox - Angle: {np.degrees(angle):.1f} degrees")
        #print(f"DEBUG: drawBox - Box size: {box_length:.1f} x {box_width:.1f} pixels")
        #print(f"DEBUG: drawBox - Corners: {[(c[0], c[1]) for c in corners]}")
        
        # 追加のデータを保存
        self.drawn_box.update({
            'dh_pixels': dh_pixels,
            'dw_pixels': dw_pixels
        })
        
        # 画像を再表示
        self.updateImageDisplay()
    
    def createFixedSizeKymograph(self):
        """固定サイズクロップでより確実な隙間なしKymograph作成"""
        if not self.roi_line or not self.image_stack:
            return None
            
        if not hasattr(self, 'drawn_box') or self.drawn_box is None:
            return self.createLineBasedKymograph()
        
        #print(f"DEBUG: Starting fixed-size kymograph creation")
        
        # 矩形情報を取得
        box_center_x = self.drawn_box['center_x']
        box_center_y = self.drawn_box['center_y']
        box_angle = self.drawn_box['angle']
        box_length = self.drawn_box['box_length']
        box_width = self.drawn_box['box_width']
        
        #print(f"DEBUG: Box center: ({box_center_x:.1f}, {box_center_y:.1f})")
        #print(f"DEBUG: Box size: {box_length:.1f} x {box_width:.1f}")
        #print(f"DEBUG: Box angle: {np.degrees(box_angle):.1f} degrees")
        
        # 回転角度
        rotation_angle = box_angle
        
        # 固定クロップサイズを決定（整数に丸める）
        fixed_width = int(box_length)
        fixed_height = int(box_width)
        
        #print(f"DEBUG: Fixed crop size: {fixed_height} x {fixed_width}")
        
        # 各フレームを処理
        processed_frames = []
        
        for frame_idx, frame_data in enumerate(self.image_stack):
            #print(f"DEBUG: Processing frame {frame_idx}")
            
            # 画像全体を回転
            rotated_frame = rotate(frame_data, np.degrees(rotation_angle), 
                                 reshape=True, order=1, prefilter=False)
            
            # 元の中心からの回転後中心を計算
            orig_center_y, orig_center_x = np.array(frame_data.shape) / 2
            rot_center_y, rot_center_x = np.array(rotated_frame.shape) / 2
            
            # 矩形中心の回転後座標を計算
            rel_x = box_center_x - orig_center_x
            rel_y = box_center_y - orig_center_y
            
            cos_angle = np.cos(rotation_angle)
            sin_angle = np.sin(rotation_angle)
            
            rot_rel_x = rel_x * cos_angle - rel_y * sin_angle
            rot_rel_y = rel_x * sin_angle + rel_y * cos_angle
            
            rot_box_center_x = rot_center_x + rot_rel_x
            rot_box_center_y = rot_center_y + rot_rel_y
            
            # 固定サイズでクロップ
            start_x = int(rot_box_center_x - fixed_width // 2)
            end_x = start_x + fixed_width
            start_y = int(rot_box_center_y - fixed_height // 2)
            end_y = start_y + fixed_height
            
            # 境界チェック
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(rotated_frame.shape[1], end_x)
            end_y = min(rotated_frame.shape[0], end_y)
            
            # 実際のクロップサイズを計算
            actual_width = end_x - start_x
            actual_height = end_y - start_y
            
            # 固定サイズのフレームを作成
            fixed_frame = np.zeros((fixed_height, fixed_width))
            
            if actual_width > 0 and actual_height > 0:
                # クロップしたデータを中央に配置
                paste_start_y = (fixed_height - actual_height) // 2
                paste_start_x = (fixed_width - actual_width) // 2
                
                cropped = rotated_frame[start_y:end_y, start_x:end_x]
                fixed_frame[paste_start_y:paste_start_y+actual_height, 
                           paste_start_x:paste_start_x+actual_width] = cropped
                
                # 正規化を適用
                if self.normalize_enabled:
                    fixed_frame = self.normalizeFrame(fixed_frame)
            
            processed_frames.append(fixed_frame)
            
            #print(f"DEBUG: Frame {frame_idx} - Crop region: ({start_y}, {start_x}) to ({end_y}, {end_x})")
            #print(f"DEBUG: Frame {frame_idx} - Actual size: {actual_height} x {actual_width}")
            #print(f"DEBUG: Frame {frame_idx} - Fixed frame shape: {fixed_frame.shape}")
        
        if not processed_frames:
            #print("DEBUG: No processed frames")
            return self.createLineBasedKymograph()
        
        # すべてのフレームが同じサイズであることを確認
        #for i, frame in enumerate(processed_frames):
            #if frame.shape != (fixed_height, fixed_width):
                #print(f"DEBUG: Error - Frame {i} has wrong shape: {frame.shape}")
        
        #print  (f"DEBUG: All frames have shape: ({fixed_height}, {fixed_width})")
        
        # 隙間なく縦に連結
        kymograph = np.vstack(processed_frames)
        
        #print(f"DEBUG: Final kymograph shape: {kymograph.shape}")
        #print(f"DEBUG: Expected shape: ({len(processed_frames) * fixed_height}, {fixed_width})")
        #print(f"DEBUG: Final kymograph range: {kymograph.min():.2f} to {kymograph.max():.2f}")
        
        return kymograph

    def testSeamlessKymograph(self):
        """隙間なしKymographをテスト"""
        if not self.roi_line:
            QtWidgets.QMessageBox.warning(self, "Error", "No line selected")
            return
        
        if not hasattr(self, 'drawn_box') or self.drawn_box is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No box drawn")
            return
        
        try:
            if not self.image_stack:
                self.loadImageStack()
            
            if not self.image_stack:
                QtWidgets.QMessageBox.warning(self, "Error", "No image stack available")
                return
            
            #print("=== Testing Seamless Kymograph ===")
            
            # 固定サイズ版を試す
            seamless_kymo = self.createFixedSizeKymograph()
            
            if seamless_kymo is not None:
                # 現在のkymographと置き換えて表示
                self.kymograph_data = seamless_kymo
                self.displayKymograph()
                self.save_button.setEnabled(True)
                QtWidgets.QMessageBox.information(self, "Success", "Seamless kymograph created")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to create seamless kymograph")
        
        except Exception as e:
            print(f"ERROR in testSeamlessKymograph: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Error: {e}")

    def debugFrameSizes(self):
        """フレームサイズをデバッグ"""
        if not self.image_stack:
            print("No image stack to debug")
            return
        
        print("=== Frame Size Debug ===")
        
        for i, frame in enumerate(self.image_stack):
            print(f"Frame {i}: shape={frame.shape}, range={frame.min():.2f}-{frame.max():.2f}")
        
        if hasattr(self, 'drawn_box') and self.drawn_box is not None:
            print(f"Box dimensions: {self.drawn_box['box_length']:.1f} x {self.drawn_box['box_width']:.1f}")
            print(f"Box angle: {np.degrees(self.drawn_box['angle']):.1f} degrees")
    
    def onNormalizeChanged(self, state):
        """正規化チェックボックスの状態変更を処理"""
        self.normalize_enabled = state == QtCore.Qt.Checked
        #print(f"DEBUG: Normalization enabled: {self.normalize_enabled}")

    def normalizeFrame(self, frame_data):
        """フレームデータを正規化"""
        if not self.normalize_enabled:
            return frame_data
        
        # データの範囲を取得
        data_min = np.min(frame_data)
        data_max = np.max(frame_data)
        
        # ゼロ除算を避ける
        if data_max == data_min:
            return frame_data
        
        # 0-1の範囲に正規化
        normalized_data = (frame_data - data_min) / (data_max - data_min)
        
        #print(f"DEBUG: Normalized frame - Original range: {data_min:.3f} to {data_max:.3f}")
        #print(f"DEBUG: Normalized frame - New range: {np.min(normalized_data):.3f} to {np.max(normalized_data):.3f}")
        
        return normalized_data
    
    def detectFiberEdge(self, kymograph_data, threshold_method='adaptive', smoothing=True, detection_method='leading_edge'):
        """
        線維端面を自動検出（矩形対応版）
        
        Args:
            kymograph_data: キモグラフデータ
            threshold_method: 閾値決定方法 ('adaptive', 'otsu', 'percentile')
            smoothing: スムージングを適用するかどうか
            detection_method: 検出方法 ('leading_edge', 'centroid', 'max_intensity', '2d_edge_detection')
        
        Returns:
            edge_positions: 各時間での端面位置
            confidence: 検出信頼度
        """

        
        if kymograph_data is None:
            #print("DEBUG: kymograph_data is None in detectFiberEdge")
            return None, None
        
        # 矩形モードかどうかを判定
        is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                              self.actual_frame_height is not None and 
                              self.actual_frame_height > 1)
        
        if is_rectangular_mode:
    
            return self.detectFiberEdgeRectangular(kymograph_data, threshold_method, smoothing, detection_method)
        else:
            #print("DEBUG: Using linear detection mode")
            return self.detectFiberEdgeLinear(kymograph_data, threshold_method, smoothing, detection_method)
    
    def detectFiberEdgeLinear(self, kymograph_data, threshold_method='adaptive', smoothing=True, detection_method='leading_edge'):
        """
        直線ベースkymographでの端面検出（既存のロジック）
        """
        # データの前処理
        processed_data = kymograph_data.copy()
        
        # スムージングを適用
        if smoothing:
            #print("DEBUG: applying smoothing")
            processed_data = signal.medfilt2d(processed_data, kernel_size=3)
        
        # 成長方向を自動検出
        growth_direction = self._detectGrowthDirection(processed_data)

        
        # 各時間フレームで端面を検出
        edge_positions = []
        confidence_scores = []
        

        
        for t in range(processed_data.shape[0]):
            # 時間tでの1次元プロファイルを取得
            profile = processed_data[t, :]
            
            # 検出方法に応じて端面位置を決定
            if detection_method == 'leading_edge':
                edge_pos, confidence = self._detectLeadingEdge(profile, growth_direction, threshold_method)
            elif detection_method == 'centroid':
                edge_pos, confidence = self._detectCentroid(profile, growth_direction)
            elif detection_method == 'max_intensity':
                edge_pos, confidence = self._detectMaxIntensity(profile, growth_direction)
            else:
                edge_pos, confidence = self._detectLeadingEdge(profile, growth_direction, threshold_method)
            
            edge_positions.append(edge_pos)
            confidence_scores.append(confidence)
        
        # 検出結果の後処理（異常値除去とスムージング）
        edge_positions = self._postProcessEdgePositions(edge_positions)
        
        return np.array(edge_positions), np.array(confidence_scores)
    
    def detectFiberEdgeRectangular(self, kymograph_data, threshold_method='adaptive', smoothing=True, detection_method='leading_edge'):
        """
        矩形ベースkymographでの端面検出
        """

        
        # フレーム高さを取得
        frame_height = int(self.actual_frame_height)
        total_frames = len(self.image_stack)
        

        
        # データの前処理
        processed_data = kymograph_data.copy()
        
        # 各フレームで端面を検出
        edge_positions = []
        confidence_scores = []
        
        for frame_idx in range(total_frames):
            # フレームデータを抽出
            start_row = frame_idx * frame_height
            end_row = start_row + frame_height
            

                
            frame_data = processed_data[start_row:end_row, :]

            
            # フレーム内で端面位置を検出
            edge_pos, confidence = self._detectEdgeInFrame(frame_data, detection_method, threshold_method, smoothing)
            
            edge_positions.append(edge_pos)
            confidence_scores.append(confidence)
        

        
        # 検出結果の後処理
        edge_positions = self._postProcessEdgePositions(edge_positions)
        
        return np.array(edge_positions), np.array(confidence_scores)
    
    def _detectEdgeInFrame(self, frame_data, detection_method, threshold_method, smoothing):
        """
        矩形フレーム内での端面検出
        
        Args:
            frame_data: フレームデータ (height x width)
            detection_method: 検出方法
            threshold_method: 閾値決定方法
            smoothing: スムージング適用フラグ
        
        Returns:
            edge_position: 端面位置
            confidence: 信頼度
        """
        if frame_data.size == 0:
            return 0, 0.0
        
        # 方法1: 各行での1D検出結果を統合
        if detection_method in ['leading_edge', 'centroid', 'max_intensity']:
            return self._detectEdgeByRowAggregation(frame_data, detection_method, threshold_method, smoothing)
        
        # 方法2: 2D画像解析による検出
        elif detection_method == '2d_edge_detection':
            return self._detectEdgeBy2DAnalysis(frame_data, threshold_method, smoothing)
        
        else:
            # デフォルトは行集約方式
            return self._detectEdgeByRowAggregation(frame_data, detection_method, threshold_method, smoothing)
    
    def _detectEdgeByRowAggregation(self, frame_data, detection_method, threshold_method, smoothing):
        """
        各行での1D検出結果を統合して端面位置を決定
        """
        height, width = frame_data.shape
        
        # 各行で端面検出を実行
        row_positions = []
        row_confidences = []
        
        # 成長方向を判定（フレーム全体から）
        frame_profile = np.mean(frame_data, axis=0)
        growth_direction = self._detectGrowthDirectionFromProfile(frame_profile)
        
        for row_idx in range(height):
            profile = frame_data[row_idx, :]
            
            # スムージング適用
            if smoothing and len(profile) > 5:
                profile = signal.savgol_filter(profile, window_length=min(5, len(profile)//2*2+1), polyorder=2)
            
            # 各行で端面検出
            if detection_method == 'leading_edge':
                edge_pos, confidence = self._detectLeadingEdge(profile, growth_direction, threshold_method)
            elif detection_method == 'centroid':
                edge_pos, confidence = self._detectCentroid(profile, growth_direction)
            elif detection_method == 'max_intensity':
                edge_pos, confidence = self._detectMaxIntensity(profile, growth_direction)
            else:
                edge_pos, confidence = self._detectLeadingEdge(profile, growth_direction, threshold_method)
            
            # 有効な検出結果のみを保存
            if confidence > 0.1:  # 最小信頼度閾値
                row_positions.append(edge_pos)
                row_confidences.append(confidence)
        
        if not row_positions:
            return width // 2, 0.0
        
        # 結果を統合
        row_positions = np.array(row_positions)
        row_confidences = np.array(row_confidences)
        
        # 重み付き平均で最終位置を決定
        total_confidence = np.sum(row_confidences)
        if total_confidence > 0:
            weighted_position = np.sum(row_positions * row_confidences) / total_confidence
            avg_confidence = np.mean(row_confidences)
        else:
            weighted_position = np.mean(row_positions)
            avg_confidence = 0.0
        
        #print(f"DEBUG: Row aggregation - positions: {len(row_positions)}, weighted_pos: {weighted_position:.1f}, confidence: {avg_confidence:.3f}")
        
        return weighted_position, avg_confidence
    
    def _detectEdgeBy2DAnalysis(self, frame_data, threshold_method, smoothing):
        """
        2D画像解析による端面検出
        """
        try:
            from skimage import filters, measure, morphology
            from scipy import ndimage
        except ImportError:
            self._require_skimage("Fiber edge detection")
            return frame_data.shape[1] // 2, 0.0

        try:
            # スムージング適用
            if smoothing:
                frame_data = filters.gaussian(frame_data, sigma=1.0)
            
            # 閾値設定
            if threshold_method == 'otsu':
                threshold = filters.threshold_otsu(frame_data)
            else:
                threshold = np.percentile(frame_data, 75)
            
            # 二値化
            binary_mask = frame_data > threshold
            
            # ノイズ除去
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=5)
            
            # 連続成分を検出
            labeled_regions = measure.label(binary_mask)
            regions = measure.regionprops(labeled_regions)
            
            if not regions:
                return frame_data.shape[1] // 2, 0.0
            
            # 最大領域を選択
            largest_region = max(regions, key=lambda r: r.area)
            
            # 成長方向に応じて端面を決定
            bbox = largest_region.bbox  # (min_row, min_col, max_row, max_col)
            
            # 右端または左端を検出（成長方向による）
            frame_profile = np.mean(frame_data, axis=0)
            growth_direction = self._detectGrowthDirectionFromProfile(frame_profile)
            
            if growth_direction == 'right':
                edge_position = bbox[3] - 1  # 右端
            else:
                edge_position = bbox[1]      # 左端
            
            # 信頼度は領域のcompactnessと面積比で評価
            confidence = min(1.0, largest_region.area / (frame_data.shape[0] * frame_data.shape[1]) * 10)
            
            #print(f"DEBUG: 2D analysis - edge_pos: {edge_position}, confidence: {confidence:.3f}")
            
            return edge_position, confidence
            
        except ImportError:
            #print("DEBUG: scikit-image not available, falling back to row aggregation")
            return self._detectEdgeByRowAggregation(frame_data, 'leading_edge', threshold_method, smoothing)
        except Exception as e:
            #print(f"DEBUG: Error in 2D analysis: {e}")
            return self._detectEdgeByRowAggregation(frame_data, 'leading_edge', threshold_method, smoothing)
    
    def _detectGrowthDirectionFromProfile(self, profile):
        """
        1Dプロファイルから成長方向を検出
        """
        try:
            # プロファイルの左半分と右半分の強度を比較
            mid_point = len(profile) // 2
            left_intensity = np.mean(profile[:mid_point])
            right_intensity = np.mean(profile[mid_point:])
            
            # 線維がより多く分布している側の反対方向が成長方向
            return 'right' if left_intensity > right_intensity else 'left'
            
        except Exception as e:
            #print(f"DEBUG: Error in growth direction detection: {e}")
            return 'right'  # デフォルト
    
    def _otsu_threshold(self, histogram):
        """Otsu法で閾値を決定"""
        total = np.sum(histogram)
        if total == 0:
            return 0
        
        # 累積分布を計算
        cumsum = np.cumsum(histogram)
        cumsum_sq = np.cumsum(histogram * np.arange(len(histogram)))
        
        # 各閾値での分散を計算
        mean_total = cumsum_sq[-1] / total
        variance = np.zeros(len(histogram))
        
        for i in range(len(histogram)):
            if cumsum[i] == 0 or cumsum[i] == total:
                continue
            
            mean_below = cumsum_sq[i] / cumsum[i]
            mean_above = (cumsum_sq[-1] - cumsum_sq[i]) / (total - cumsum[i])
            
            variance[i] = (cumsum[i] * (total - cumsum[i]) * 
                          (mean_below - mean_above) ** 2) / (total ** 2)
        
        # 最大分散を与える閾値を返す
        return np.argmax(variance)
    
    def _detectGrowthDirection(self, kymograph_data):
        """成長方向を自動検出"""
        try:
            # 最初と最後の数フレームを比較
            n_frames = min(5, kymograph_data.shape[0] // 4)
            
            # 最初の数フレームの平均
            early_frames = np.mean(kymograph_data[:n_frames], axis=0)
            # 最後の数フレームの平均
            late_frames = np.mean(kymograph_data[-n_frames:], axis=0)
            
            # 線維の重心位置を計算
            def calculate_centroid(profile):
                # プロファイルの閾値以上の部分のみを考慮
                threshold = np.percentile(profile, 70)  # 上位30%の強度
                mask = profile >= threshold
                if np.sum(mask) == 0:
                    return len(profile) // 2
                
                positions = np.arange(len(profile))
                weights = profile * mask
                if np.sum(weights) == 0:
                    return len(profile) // 2
                
                centroid = np.sum(positions * weights) / np.sum(weights)
                return centroid
            
            early_centroid = calculate_centroid(early_frames)
            late_centroid = calculate_centroid(late_frames)
            
            centroid_shift = late_centroid - early_centroid
            
            #print(f"DEBUG: Early centroid: {early_centroid:.1f}, Late centroid: {late_centroid:.1f}")
            #print(f"DEBUG: Centroid shift: {centroid_shift:.1f}")
            
            # 成長方向を判定
            if abs(centroid_shift) > 2:  # 有意な変化がある場合
                return 'right' if centroid_shift > 0 else 'left'
            else:
                # 変化が小さい場合は線維の分布から判定
                # 線維がより多く分布している側の反対方向が成長方向
                left_intensity = np.mean(early_frames[:len(early_frames)//2])
                right_intensity = np.mean(early_frames[len(early_frames)//2:])
                
                return 'right' if left_intensity > right_intensity else 'left'
                
        except Exception as e:
    
            return 'right'  # デフォルト
    
    def _detectLeadingEdge(self, profile, growth_direction, threshold_method):
        """先端エッジ検出（改良版）"""
        try:
            # プロファイルをスムージング
            profile_smooth = signal.savgol_filter(profile, window_length=min(11, len(profile)//4*2+1), polyorder=3)
            
            # 線維領域を特定（適応的閾値）
            if threshold_method == 'adaptive':
                # Otsu法で基本閾値を決定
                hist, bins = np.histogram(profile_smooth, bins=50)
                threshold_base = self._otsu_threshold_value(profile_smooth, hist, bins)
                # より厳しい閾値を設定（線維の明確な部分のみ）
                threshold = threshold_base + (np.max(profile_smooth) - threshold_base) * 0.3
            else:
                threshold = np.percentile(profile_smooth, 75)
            
            # 線維領域のマスクを作成
            fiber_mask = profile_smooth >= threshold
            
            if np.sum(fiber_mask) == 0:
                return len(profile) // 2, 0.0
            
            # 連続した線維領域を特定
            fiber_regions = self._findConnectedRegions(fiber_mask)
            
            if len(fiber_regions) == 0:
                return len(profile) // 2, 0.0
            
            # 最大の線維領域を選択
            largest_region = max(fiber_regions, key=lambda x: x[1] - x[0])
            
            # 成長方向に応じて先端を決定
            if growth_direction == 'right':
                # 右方向成長の場合、最も右端が成長端
                leading_edge = largest_region[1] - 1  # 領域の右端
            else:
                # 左方向成長の場合、最も左端が成長端
                leading_edge = largest_region[0]  # 領域の左端
            
            # 信頼度を計算（線維領域の明確さと連続性に基づく）
            region_intensity = np.mean(profile_smooth[largest_region[0]:largest_region[1]])
            background_intensity = np.mean(profile_smooth[~fiber_mask]) if np.sum(~fiber_mask) > 0 else 0
            
            contrast = (region_intensity - background_intensity) / (region_intensity + background_intensity + 1e-10)
            region_size_ratio = (largest_region[1] - largest_region[0]) / len(profile)
            
            confidence = min(1.0, contrast * region_size_ratio * 2)
            
            return leading_edge, confidence
            
        except Exception as e:
    
            return len(profile) // 2, 0.0
    
    def _detectCentroid(self, profile, growth_direction):
        """重心ベース検出"""
        try:
            # 線維領域の重心を計算
            threshold = np.percentile(profile, 70)
            mask = profile >= threshold
            
            if np.sum(mask) == 0:
                return len(profile) // 2, 0.0
            
            positions = np.arange(len(profile))
            weights = profile * mask
            centroid = np.sum(positions * weights) / np.sum(weights)
            
            # 信頼度は重み付き分散の逆数
            variance = np.sum(weights * (positions - centroid)**2) / np.sum(weights)
            confidence = 1.0 / (1.0 + variance / len(profile))
            
            return int(centroid), confidence
            
        except Exception as e:
    
            return len(profile) // 2, 0.0
    
    def _detectMaxIntensity(self, profile, growth_direction):
        """最大強度ベース検出"""
        try:
            # 最大強度位置を検出
            max_pos = np.argmax(profile)
            max_intensity = profile[max_pos]
            
            # 信頼度は最大強度の相対値
            confidence = (max_intensity - np.min(profile)) / (np.max(profile) - np.min(profile) + 1e-10)
            
            return max_pos, confidence
            
        except Exception as e:
    
            return len(profile) // 2, 0.0
    
    def _findConnectedRegions(self, mask):
        """連続した領域を検出"""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                # 領域開始
                start = i
                in_region = True
            elif not val and in_region:
                # 領域終了
                regions.append((start, i))
                in_region = False
        
        # 最後が領域内で終わっている場合
        if in_region:
            regions.append((start, len(mask)))
        
        return regions
    
    def _otsu_threshold_value(self, data, hist, bins):
        """Otsu法で実際の閾値を計算"""
        bin_centers = (bins[:-1] + bins[1:]) / 2
        total = np.sum(hist)
        
        if total == 0:
            return np.mean(data)
        
        # 累積分布を計算
        cumsum = np.cumsum(hist)
        cumsum_sq = np.cumsum(hist * bin_centers)
        
        max_variance = 0
        threshold = bin_centers[0]
        
        for i in range(len(hist)):
            if cumsum[i] == 0 or cumsum[i] == total:
                continue
            
            mean_below = cumsum_sq[i] / cumsum[i]
            mean_above = (cumsum_sq[-1] - cumsum_sq[i]) / (total - cumsum[i])
            
            variance = (cumsum[i] * (total - cumsum[i]) * (mean_below - mean_above) ** 2) / (total ** 2)
            
            if variance > max_variance:
                max_variance = variance
                threshold = bin_centers[i]
        
        return threshold
    
    def _postProcessEdgePositions(self, edge_positions):
        """検出結果の後処理"""
        if len(edge_positions) < 3:
            return edge_positions
        
        edge_positions = np.array(edge_positions)
        
        # 異常値検出と修正（中央値フィルタ）
        window_size = min(5, len(edge_positions)//3)
        if window_size >= 3:
            edge_positions_filtered = signal.medfilt(edge_positions, kernel_size=window_size)
        else:
            edge_positions_filtered = edge_positions
        
        # 大きな跳躍の修正
        diff_threshold = np.std(np.diff(edge_positions_filtered)) * 3
        corrected_positions = [edge_positions_filtered[0]]
        
        for i in range(1, len(edge_positions_filtered)):
            diff = edge_positions_filtered[i] - corrected_positions[-1]
            if abs(diff) > diff_threshold:
                # 大きな跳躍の場合は線形補間
                corrected_pos = corrected_positions[-1] + np.sign(diff) * diff_threshold
                corrected_positions.append(corrected_pos)
            else:
                corrected_positions.append(edge_positions_filtered[i])
        
        return np.array(corrected_positions)
    
    def analyzeFiberGrowth(self, kymograph_data, time_interval=1.0):
        """
        線維成長を解析
        
        Args:
            kymograph_data: キモグラフデータ
            time_interval: 時間間隔（秒）
        
        Returns:
            growth_data: 成長解析結果の辞書
        """
        #print(f"DEBUG: analyzeFiberGrowth called with shape {kymograph_data.shape}")
        
        if kymograph_data is None:
            #print("DEBUG: kymograph_data is None")
            return None
        
        # 線維端面を検出
        #print("DEBUG: calling detectFiberEdge")
        edge_positions, confidence = self.detectFiberEdge(kymograph_data)
        
        if edge_positions is None:
            #print("DEBUG: detectFiberEdge returned None")
            return None
        
        #print(f"DEBUG: edge_positions shape: {edge_positions.shape}")
        #print(f"DEBUG: confidence shape: {confidence.shape}")
        
        # 時間軸を生成
        time_points = np.arange(len(edge_positions)) * time_interval
        #print(f"DEBUG: time_points shape: {time_points.shape}")
        
        # 成長軌跡をフィッティング
        #print("DEBUG: calling _fitGrowthTrajectory")
        growth_params = self._fitGrowthTrajectory(time_points, edge_positions)
        
        # 成長速度を計算
        #print("DEBUG: calling _calculateGrowthSpeed")
        growth_speed = self._calculateGrowthSpeed(time_points, edge_positions)
        
        # 結果をまとめる
        growth_data = {
            'time_points': time_points,
            'edge_positions': edge_positions,
            'confidence': confidence,
            'growth_params': growth_params,
            'growth_speed': growth_speed,
            'total_growth': edge_positions[-1] - edge_positions[0] if len(edge_positions) > 1 else 0,
            'average_speed': np.mean(growth_speed) if growth_speed is not None else 0
        }
        
        #print("DEBUG: analyzeFiberGrowth completed successfully")
        return growth_data
    
    def _fitGrowthTrajectory(self, time_points, edge_positions):
        """成長軌跡をフィッティング"""
        if len(time_points) < 3:
            return None
        
        try:
            # 線形フィッティング
            def linear_func(t, a, b):
                return a * t + b
            
            # フィッティングを実行
            popt, pcov = curve_fit(linear_func, time_points, edge_positions)
            
            return {
                'slope': popt[0],  # 成長速度
                'intercept': popt[1],  # 初期位置
                'r_squared': self._calculate_r_squared(time_points, edge_positions, linear_func, popt)
            }
        except:
            return None
    
    def _calculateGrowthSpeed(self, time_points, edge_positions):
        """成長速度を計算"""
        if len(time_points) < 2:
            return None
        
        # 差分法で速度を計算
        dt = np.diff(time_points)
        dx = np.diff(edge_positions)
        
        # ゼロ除算を避ける
        dt = np.where(dt == 0, 1e-10, dt)
        speed = dx / dt
        
        return speed
    
    def _calculate_r_squared(self, x, y, func, popt):
        """決定係数を計算"""
        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        return 1 - (ss_res / ss_tot)
    
    def displayFiberGrowthAnalysis(self, growth_data):
        """線維成長解析結果を表示"""
        #print("DEBUG: displayFiberGrowthAnalysis called")
        
        if growth_data is None:
            #print("DEBUG: growth_data is None in displayFiberGrowthAnalysis")
            return
        
        #print("DEBUG: calling createFiberGrowthReport")
        # 改善されたレポートウィンドウを作成
        report_window = self.createFiberGrowthReport(growth_data)
        if report_window:
            #print("DEBUG: report_window created, calling show()")
            report_window.show()
            #print("DEBUG: report_window.show() called")
        #else:
            #print("DEBUG: report_window is None")
    
    def _formatGrowthResults(self, growth_data):
        """成長解析結果をフォーマット"""
        result_str = "=== 線維成長解析結果 ===\n\n"
        
        result_str += f"総成長距離: {growth_data['total_growth']:.2f} ピクセル\n"
        result_str += f"平均成長速度: {growth_data['average_speed']:.2f} ピクセル/秒\n"
        
        if growth_data['growth_params']:
            params = growth_data['growth_params']
            result_str += f"フィッティング結果:\n"
            result_str += f"  成長速度: {params['slope']:.2f} ピクセル/秒\n"
            result_str += f"  初期位置: {params['intercept']:.2f} ピクセル\n"
            result_str += f"  決定係数: {params['r_squared']:.3f}\n"
        
        result_str += f"\n検出信頼度:\n"
        result_str += f"  平均: {np.mean(growth_data['confidence']):.3f}\n"
        result_str += f"  最小: {np.min(growth_data['confidence']):.3f}\n"
        result_str += f"  最大: {np.max(growth_data['confidence']):.3f}\n"
        
        return result_str
    
    def _plotGrowthTrajectory(self, growth_data):
        """成長軌跡をプロット"""
        if growth_data is None:
            return
        
        # 新しいウィンドウでプロット
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 成長軌跡をプロット
        ax.plot(growth_data['time_points'], growth_data['edge_positions'], 
                'bo-', label='検出された端面位置')
        
        # フィッティング結果をプロット
        if growth_data['growth_params']:
            params = growth_data['growth_params']
            t_fit = growth_data['time_points']
            y_fit = params['slope'] * t_fit + params['intercept']
            ax.plot(t_fit, y_fit, 'r--', label=f'フィッティング (速度: {params["slope"]:.2f})')
        
        ax.set_xlabel('時間 (秒)')
        ax.set_ylabel('位置 (ピクセル)')
        ax.set_title('線維成長軌跡')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _saveGrowthData(self, growth_data):
        """成長解析データを保存"""
        if growth_data is None:
            return
        
        # ファイル名を取得
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "成長解析データを保存", 
            "fiber_growth_analysis.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            # データフレームを作成
            df = pd.DataFrame({
                'Time(s)': growth_data['time_points'],
                'EdgePosition(pixel)': growth_data['edge_positions'],
                'Confidence': growth_data['confidence']
            })
            
            # CSVに保存
            df.to_csv(file_path, index=False)
            
            QtWidgets.QMessageBox.information(
                self, "保存完了", 
                f"成長解析データを保存しました:\n{file_path}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                        self, "Error",
        f"An error occurred while saving data:\n{e}"
            )
    
    def analyzeFiberGrowthFromUI(self):
        """UIから線維成長解析を実行（矩形対応版）"""
        #print("DEBUG: analyzeFiberGrowthFromUI called")
        
        if self.kymograph_data is None:
            #print("DEBUG: kymograph_data is None")
            QtWidgets.QMessageBox.warning(self, "Error", "No kymograph data available.")
            return
        
        #print(f"DEBUG: kymograph_data shape: {self.kymograph_data.shape}")
        
        # 矩形モードかどうかを判定
        is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                              self.actual_frame_height is not None and 
                              self.actual_frame_height > 1)
        
        # セミオート検出結果と参照線をクリア
        self.semi_auto_results = None
        self.reference_line_points = []
        self.reference_line_complete = False
        self.drawing_reference_line = False
        
        # detectionコンボボックスの選択を取得
        detection_type = self.detection_combo.currentText()
        
        # 検出タイプに応じて検出方法を設定
        if detection_type == "Fibril end":
            # 線維端検出の場合は現在の実装を使用
            if is_rectangular_mode:
                methods = ['leading_edge', 'centroid', 'max_intensity', '2d_edge_detection']
                method_names = ['Leading Edge Detection', 'Centroid Based', 'Maximum Intensity Based', '2D Image Analysis']
                dialog_title = "Detection Method Selection (Rectangular Mode)"
                dialog_text = "Please select the fiber end detection method for rectangular region:"
            else:
                methods = ['leading_edge', 'centroid', 'max_intensity']
                method_names = ['Leading Edge Detection', 'Centroid Based', 'Maximum Intensity Based']
                dialog_title = "Detection Method Selection (Linear Mode)"
                dialog_text = "Please select the fiber end detection method:"
            
            method, ok = QtWidgets.QInputDialog.getItem(
                self, dialog_title, dialog_text, method_names, 0, False)
            
            if not ok:
                return
            
            detection_method = methods[method_names.index(method)]
            
            try:
                # 時間間隔を取得（デフォルト値1.0）
                time_interval = 1.0
                #print(f"DEBUG: time_interval = {time_interval}, detection_method = {detection_method}")
                #print(f"DEBUG: rectangular_mode = {is_rectangular_mode}")
                
                # Execute fiber growth analysis
                #print("DEBUG: calling analyzeFiberGrowth with rectangular support")
                growth_data = self.analyzeFiberGrowthImproved(self.kymograph_data, time_interval, detection_method)
                
                if growth_data is None:
                    #print("DEBUG: analyzeFiberGrowth returned None")
                    QtWidgets.QMessageBox.warning(self, "Error", "Fiber growth analysis failed.")
                    return
                
                #print("DEBUG: analyzeFiberGrowth successful")
                
                # Save analysis results
                self.fiber_growth_data = growth_data
                
                # Display analysis results
                #print("DEBUG: calling displayFiberGrowthAnalysis")
                self.displayFiberGrowthAnalysis(growth_data)
                
                # Redraw kymograph with detection results
                #print("DEBUG: updating kymograph display with detection results")
                self.displayKymograph()
                
            except Exception as e:
                #print(f"DEBUG: Exception in analyzeFiberGrowthFromUI: {e}")
                import traceback
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during fiber growth analysis:\n{e}")
                
        elif detection_type == "Particle":
            # 粒子検出の場合は新しい実装を使用
            try:
                #print("DEBUG: Starting particle detection")
                self.detectParticlesInKymograph()
            except Exception as e:
                #print(f"DEBUG: Exception in particle detection: {e}")
                import traceback
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during particle detection:\n{e}")
            return  # Exit here for Particle mode
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid detection type.")
            return
        
        # Execute the following processing only for Fibril end mode
        try:
            # 時間間隔を取得（デフォルト値1.0）
            time_interval = 1.0
            #print(f"DEBUG: time_interval = {time_interval}, detection_method = {detection_method}")
            #print(f"DEBUG: rectangular_mode = {is_rectangular_mode}")
            
            # Execute fiber growth analysis
            #print("DEBUG: calling analyzeFiberGrowth with rectangular support")
            growth_data = self.analyzeFiberGrowthImproved(self.kymograph_data, time_interval, detection_method)
            
            if growth_data is None:
                #print("DEBUG: analyzeFiberGrowth returned None")
                QtWidgets.QMessageBox.warning(self, "Error", "Fiber growth analysis failed.")
                return
            
            #print("DEBUG: analyzeFiberGrowth successful")
            
            # Save analysis results
            self.fiber_growth_data = growth_data
            
            # Display analysis results
            #print("DEBUG: calling displayFiberGrowthAnalysis")
            self.displayFiberGrowthAnalysis(growth_data)
            
            # Redraw kymograph with detection results
            #print("DEBUG: updating kymograph display with detection results")
            self.displayKymograph()
            
        except Exception as e:
            #print(f"DEBUG: Exception in analyzeFiberGrowthFromUI: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during fiber growth analysis:\n{e}")
    
    def createFiberGrowthReport(self, growth_data):
        """Create fiber growth analysis report"""
        #print("DEBUG: createFiberGrowthReport called")
        
        if growth_data is None:
            #print("DEBUG: growth_data is None in createFiberGrowthReport")
            return None
        
        #print("DEBUG: creating report window")
        # Create report window
        report_window = QtWidgets.QWidget()
        report_window.setWindowTitle("Fiber Growth Analysis Report")
        report_window.setGeometry(200, 200, 1000, 700)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout()
        
        # Create tab widget
        tab_widget = QtWidgets.QTabWidget()
        
        # 結果タブ
        results_tab = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results_tab)
        
        # 結果テキスト
        results_text = QtWidgets.QTextEdit()
        results_text.setReadOnly(True)
        results_str = self._formatGrowthResults(growth_data)
        results_text.setPlainText(results_str)
        results_layout.addWidget(results_text)
        
        tab_widget.addTab(results_tab, "解析結果")
        
        # グラフタブ
        graph_tab = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(graph_tab)
        
        # matplotlibキャンバスを作成
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 成長軌跡をプロット
        ax.plot(growth_data['time_points'], growth_data['edge_positions'], 
                'bo-', label='検出された端面位置', markersize=4)
        
        # フィッティング結果をプロット
        if growth_data['growth_params']:
            params = growth_data['growth_params']
            t_fit = growth_data['time_points']
            y_fit = params['slope'] * t_fit + params['intercept']
            ax.plot(t_fit, y_fit, 'r--', linewidth=2, 
                   label=f'フィッティング (速度: {params["slope"]:.2f} ピクセル/秒)')
        
        ax.set_xlabel('時間 (秒)')
        ax.set_ylabel('位置 (ピクセル)')
        ax.set_title('線維成長軌跡')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # キャンバスをQtウィジェットに埋め込み
        canvas = FigureCanvas(fig)
        graph_layout.addWidget(canvas)
        
        tab_widget.addTab(graph_tab, "成長軌跡")
        
        # 信頼度タブ
        confidence_tab = QtWidgets.QWidget()
        confidence_layout = QtWidgets.QVBoxLayout(confidence_tab)
        
        # 信頼度グラフ
        fig_conf, ax_conf = plt.subplots(figsize=(8, 6))
        ax_conf.plot(growth_data['time_points'], growth_data['confidence'], 
                    'go-', markersize=4)
        ax_conf.set_xlabel('時間 (秒)')
        ax_conf.set_ylabel('検出信頼度')
        ax_conf.set_title('端面検出信頼度')
        ax_conf.grid(True, alpha=0.3)
        ax_conf.set_ylim(0, 1)
        
        canvas_conf = FigureCanvas(fig_conf)
        confidence_layout.addWidget(canvas_conf)
        
        tab_widget.addTab(confidence_tab, "検出信頼度")
        
        layout.addWidget(tab_widget)
        
        # ボタン
        button_layout = QtWidgets.QHBoxLayout()
        
        save_button = QtWidgets.QPushButton("データ保存")
        save_button.clicked.connect(lambda: self._saveGrowthData(growth_data))
        button_layout.addWidget(save_button)
        
        export_button = QtWidgets.QPushButton("レポート出力")
        export_button.clicked.connect(lambda: self._exportGrowthReport(growth_data))
        button_layout.addWidget(export_button)
        
        close_button = QtWidgets.QPushButton("閉じる")
        close_button.clicked.connect(report_window.close)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        report_window.setLayout(layout)
        #print("DEBUG: report window layout set, returning window")
        return report_window
    
    def _exportGrowthReport(self, growth_data):
        """線維成長解析レポートをファイルに出力"""
        if growth_data is None:
            return
        
        # ファイル名を取得
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "レポートを保存", 
            "fiber_growth_report.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=== 線維成長解析レポート ===\n\n")
                f.write(f"解析日時: {QtCore.QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')}\n\n")
                
                # 基本情報
                f.write("【基本情報】\n")
                f.write(f"総成長距離: {growth_data['total_growth']:.2f} ピクセル\n")
                f.write(f"平均成長速度: {growth_data['average_speed']:.2f} ピクセル/秒\n")
                f.write(f"解析時間範囲: {growth_data['time_points'][0]:.1f} - {growth_data['time_points'][-1]:.1f} 秒\n")
                f.write(f"データポイント数: {len(growth_data['time_points'])}\n\n")
                
                # フィッティング結果
                if growth_data['growth_params']:
                    params = growth_data['growth_params']
                    f.write("【フィッティング結果】\n")
                    f.write(f"成長速度: {params['slope']:.2f} ピクセル/秒\n")
                    f.write(f"初期位置: {params['intercept']:.2f} ピクセル\n")
                    f.write(f"決定係数: {params['r_squared']:.3f}\n\n")
                
                # 信頼度統計
                f.write("【検出信頼度】\n")
                f.write(f"平均信頼度: {np.mean(growth_data['confidence']):.3f}\n")
                f.write(f"最小信頼度: {np.min(growth_data['confidence']):.3f}\n")
                f.write(f"最大信頼度: {np.max(growth_data['confidence']):.3f}\n")
                f.write(f"信頼度標準偏差: {np.std(growth_data['confidence']):.3f}\n\n")
                
                # 詳細データ
                f.write("【詳細データ】\n")
                f.write("時間(秒), 位置(ピクセル), 信頼度\n")
                for i in range(len(growth_data['time_points'])):
                    f.write(f"{growth_data['time_points'][i]:.3f}, "
                           f"{growth_data['edge_positions'][i]:.2f}, "
                           f"{growth_data['confidence'][i]:.3f}\n")
            
            QtWidgets.QMessageBox.information(
                self, "保存完了", 
                f"レポートを保存しました:\n{file_path}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                        self, "Error",
        f"An error occurred while saving the report:\n{e}"
            )

    def analyzeFiberGrowthImproved(self, kymograph_data, time_interval=1.0, detection_method='leading_edge'):
        """
        Analyze fiber growth (improved version)
        """
        #print(f"DEBUG: analyzeFiberGrowthImproved called with shape {kymograph_data.shape}")
        
        if kymograph_data is None:
            #print("DEBUG: kymograph_data is None")
            return None
        
        # Detect fiber edge (improved version)
        #print("DEBUG: calling improved detectFiberEdge")
        edge_positions, confidence = self.detectFiberEdge(
            kymograph_data, 
            threshold_method='adaptive',
            detection_method=detection_method
        )
        
        if edge_positions is None:
            #print("DEBUG: detectFiberEdge returned None")
            return None
        
        #print(f"DEBUG: edge_positions shape: {edge_positions.shape}")
        
        # Generate time axis
        time_points = np.arange(len(edge_positions)) * time_interval
        
        # Fit growth trajectory
        growth_params = self._fitGrowthTrajectory(time_points, edge_positions)
        
        # Calculate growth speed
        growth_speed = self._calculateGrowthSpeed(time_points, edge_positions)
        
        # Compile results
        growth_data = {
            'time_points': time_points,
            'edge_positions': edge_positions,
            'confidence': confidence,
            'growth_params': growth_params,
            'growth_speed': growth_speed,
            'total_growth': edge_positions[-1] - edge_positions[0] if len(edge_positions) > 1 else 0,
            'average_speed': np.mean(growth_speed) if growth_speed is not None else 0,
            'detection_method': detection_method
        }
        
        #print("DEBUG: analyzeFiberGrowthImproved completed successfully")
        return growth_data
    


    # Semi-automatic tracking related methods
    def startDrawingReferenceLine(self):
        """Start drawing reference line"""
        if not self.drawing_reference_line:
            # Start drawing
            self.drawing_reference_line = True
            self.semi_auto_mode = True
            self.reference_line_points = []
            self.reference_line_complete = False
            self.shift_pressed = False  # Initialize
            
            # Clear auto detection results and semi-auto detection results
            self.fiber_growth_data = None
            self.semi_auto_results = None
            
            self.draw_line_button.setText("Finish Line")
            self.clear_button.setEnabled(True)
            
            # Connect mouse events and keyboard events
            self.canvas_click_cid = self.result_canvas.mpl_connect('button_press_event', self.onReferenceLineClick)
            self.canvas_move_cid = self.result_canvas.mpl_connect('motion_notify_event', self.onReferenceLineMove)
            self.canvas_key_press_cid = self.result_canvas.mpl_connect('key_press_event', self.onReferenceLineKeyPress)
            self.canvas_key_release_cid = self.result_canvas.mpl_connect('key_release_event', self.onReferenceLineKeyRelease)
            
            # Set keyboard focus
            self.result_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.result_canvas.setFocus()
            
            self.result_canvas.setCursor(QtCore.Qt.CrossCursor)
            
            # Redraw kymograph (remove detection results)
            if self.kymograph_data is not None:
                self.displayKymograph()
            
            QtWidgets.QMessageBox.information(
                self, "Draw Reference Line", 
                "Click points along the approximate fiber edge trajectory.\n"
                "The line will guide automatic detection in each frame.\n\n"
                "Left click: Add point\n"
                "Shift + Left click: Remove nearest point\n"
                "Click 'Finish Line' when done"
            )
        else:
            # Finish drawing
            self.finishReferenceLine()

    def finishReferenceLine(self):
        """参照線の描画を終了"""
        if len(self.reference_line_points) < 2:
            QtWidgets.QMessageBox.warning(self, "Error", "At least 2 points are required for reference line.")
            return
        
        self.drawing_reference_line = False
        self.reference_line_complete = True
        self.draw_line_button.setText("Draw Line")
        self.auto_detect_button.setEnabled(True)
        
        self.result_canvas.unsetCursor()
        
        # 参照線を表示
        self.displayKymograph()
        


    def clearReferenceLine(self):
        """参照線をクリア"""
        self.reference_line_points = []
        self.reference_line_complete = False
        self.drawing_reference_line = False
        self.semi_auto_results = None
        self.shift_pressed = False  # フラグもリセット
        
        self.draw_line_button.setText("Draw Line")
        self.clear_button.setEnabled(False)
        self.auto_detect_button.setEnabled(False)
        
        # イベント接続を切断
        try:
            if hasattr(self, 'canvas_click_cid'):
                self.result_canvas.mpl_disconnect(self.canvas_click_cid)
            if hasattr(self, 'canvas_move_cid'):
                self.result_canvas.mpl_disconnect(self.canvas_move_cid)
            if hasattr(self, 'canvas_key_press_cid'):
                self.result_canvas.mpl_disconnect(self.canvas_key_press_cid)
            if hasattr(self, 'canvas_key_release_cid'):
                self.result_canvas.mpl_disconnect(self.canvas_key_release_cid)
        except Exception as e:
    
            pass
        
        self.result_canvas.unsetCursor()
        
        # kymographを再描画
        if self.kymograph_data is not None:
            self.displayKymograph()

    def onReferenceLineClick(self, event):
        """参照線描画時のクリック処理"""
        if not self.drawing_reference_line or not event.inaxes:
            return
        
        if event.button == 1:  # 左クリック
            # 複数の方法でShiftキーの状態をチェック
            shift_pressed = False
            
            # 方法1: matplotlibのイベントから検出
            try:
                if hasattr(event, 'key') and event.key == 'shift':
                    shift_pressed = True
                    print("DEBUG: Shift detected via event.key")
            except:
                pass
            
            # 方法2: Qtのアプリケーションからモディファイアを取得
            try:
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if modifiers & QtCore.Qt.ShiftModifier:
                    shift_pressed = True
                    print("DEBUG: Shift detected via Qt keyboard modifiers")
            except Exception as e:
        
                pass
            
            # 方法3: 内部フラグをチェック（フォールバック）
            if not shift_pressed and hasattr(self, 'shift_pressed'):
                shift_pressed = self.shift_pressed
                if shift_pressed:
                    print("DEBUG: Shift detected via internal flag")
            
    
            
            if shift_pressed:  # Shift+左クリック - 点を削除
                if self.reference_line_points:
                    # クリック位置に最も近い点を検索
                    clicked_x, clicked_y = event.xdata, event.ydata
                    min_distance = float('inf')
                    nearest_idx = -1
                    
                    for i, (ref_x, ref_y) in enumerate(self.reference_line_points):
                        distance = np.sqrt((clicked_x - ref_x)**2 + (clicked_y - ref_y)**2)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_idx = i
                    
                    # 近い点が見つかった場合（検索半径内）
                    if nearest_idx >= 0 and min_distance < 20:  # 20ピクセル以内
                        removed_point = self.reference_line_points.pop(nearest_idx)
                
                        
                        # リアルタイム表示更新
                        self.displayKymograph()
                    else:
                        # 最後の点を削除（従来の動作）
                        if self.reference_line_points:
                            removed_point = self.reference_line_points.pop()
                    
                            
                            # リアルタイム表示更新
                            self.displayKymograph()
            else:  # 通常の左クリック - 点を追加
                # 境界チェック
                if (0 <= event.xdata < self.kymograph_data.shape[1] and 
                    0 <= event.ydata < self.kymograph_data.shape[0]):
                    
                    self.reference_line_points.append((event.xdata, event.ydata))
            
                    
                    # リアルタイム表示更新
                    self.displayKymograph()
        


    def onReferenceLineMove(self, event):
        """参照線描画時のマウス移動処理"""
        # 現在は特に処理なし（将来的にプレビュー線を表示可能）
        pass
    
    def onReferenceLineKeyPress(self, event):
        """参照線描画時のキー押下処理"""
        if event.key == 'shift':
            #print("DEBUG: Shift key pressed")
            self.shift_pressed = True
    
    def onReferenceLineKeyRelease(self, event):
        """参照線描画時のキーリリース処理"""
        if event.key == 'shift':
            #print("DEBUG: Shift key released")
            self.shift_pressed = False

    def startDrawingLine(self):
        """描画を開始（検出タイプに応じて適切な機能を呼び出し）"""
        detection_type = self.detection_combo.currentText()
        
        if detection_type == "Particle":
            # Start particle line drawing for particle detection mode
            self.startDrawingParticleLine()
        else:
            # Start reference line drawing for fiber end detection mode
            self.startDrawingReferenceLine()

    def clearLines(self):
        """線をクリア（検出タイプに応じて適切な機能を呼び出し）"""
        detection_type = self.detection_combo.currentText()
        
        if detection_type == "Particle":
            # Clear particle lines for particle detection mode
            self.clearParticleLines()
        else:
            # Clear reference line for fiber end detection mode
            self.clearReferenceLine()

    def startDrawingParticleLine(self):
        """粒子線の描画を開始"""
        if not self.is_drawing_particle_line:
            # kymographが作成されているかチェック
            if self.kymograph_data is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Please create a kymograph first.")
                return
            
            # 描画開始
            self.is_drawing_particle_line = True
            self.current_line_points = []
            
            self.draw_line_button.setText("Finish Line")
            self.clear_button.setEnabled(True)
            
            # マウスイベントとキーボードイベントを接続
            try:
                self.canvas_click_cid = self.result_canvas.mpl_connect('button_press_event', self.onParticleLineClick)
                self.canvas_move_cid = self.result_canvas.mpl_connect('motion_notify_event', self.onParticleLineMove)
                self.canvas_key_press_cid = self.result_canvas.mpl_connect('key_press_event', self.onParticleLineKeyPress)
                self.canvas_key_release_cid = self.result_canvas.mpl_connect('key_release_event', self.onParticleLineKeyRelease)
            except Exception as e:
                print(f"DEBUG: Error connecting particle line events: {e}")
                self.is_drawing_particle_line = False
                self.draw_line_button.setText("Draw Line")
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to start particle line drawing: {e}")
                return
            
            # キーボードフォーカスを設定
            self.result_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.result_canvas.setFocus()
            
            self.result_canvas.setCursor(QtCore.Qt.CrossCursor)
            
            # kymographを再描画
            if self.kymograph_data is not None:
                self.displayKymograph()
            
            QtWidgets.QMessageBox.information(
                self, "Draw Particle Line", 
                "Click points along the particle trajectory.\n"
                "The line will guide particle detection in each frame.\n\n"
                "Left click: Add point\n"
                "Shift + Left click: Remove nearest point\n"
                "Click 'Finish Line' when done"
            )
        else:
            # 描画終了
            self.finishParticleLine()

    def finishParticleLine(self):
        """粒子線の描画を終了"""
        if len(self.current_line_points) < 2:
            QtWidgets.QMessageBox.warning(self, "Error", "At least 2 points are required for particle line.")
            return
        
        # 現在の線を粒子線リストに追加
        if self.particle_lines is None:
            self.particle_lines = []
        
        line_data = {
            'points': self.current_line_points.copy(),
            'color': (255, 0, 255),  # マゼンタ色
            'line_id': len(self.particle_lines)
        }
        self.particle_lines.append(line_data)
        
        self.is_drawing_particle_line = False
        self.current_line_points = []
        self.draw_line_button.setText("Draw Line")
        self.auto_detect_button.setEnabled(True)
        
        self.result_canvas.unsetCursor()
        
        # 粒子線を表示
        self.displayKymograph()
        
        print(f"DEBUG: Particle line completed with {len(line_data['points'])} points")

    def clearParticleLines(self):
        """粒子線をクリア"""
        self.particle_lines = []
        self.current_line_points = []
        self.is_drawing_particle_line = False
        
        self.draw_line_button.setText("Draw Line")
        self.clear_button.setEnabled(False)
        self.auto_detect_button.setEnabled(False)
        
        # イベント接続を切断
        try:
            if hasattr(self, 'canvas_click_cid'):
                self.result_canvas.mpl_disconnect(self.canvas_click_cid)
            if hasattr(self, 'canvas_move_cid'):
                self.result_canvas.mpl_disconnect(self.canvas_move_cid)
            if hasattr(self, 'canvas_key_press_cid'):
                self.result_canvas.mpl_disconnect(self.canvas_key_press_cid)
            if hasattr(self, 'canvas_key_release_cid'):
                self.result_canvas.mpl_disconnect(self.canvas_key_release_cid)
        except Exception as e:
            pass
        
        self.result_canvas.unsetCursor()
        
        # kymographを再描画
        if self.kymograph_data is not None:
            self.displayKymograph()

    def onParticleLineClick(self, event):
        """粒子線描画時のクリックイベント"""
        if not self.is_drawing_particle_line or not hasattr(self, 'result_ax') or event.inaxes != self.result_ax:
            return
        
        if event.button == 1:  # 左クリック
            if self.shift_pressed:
                # Shift + 左クリック: 最も近い点を削除
                if self.current_line_points:
                    click_pos = (event.xdata, event.ydata)
                    distances = [np.sqrt((p[0] - click_pos[0])**2 + (p[1] - click_pos[1])**2) 
                               for p in self.current_line_points]
                    nearest_idx = np.argmin(distances)
                    if distances[nearest_idx] < 20:  # 20ピクセル以内
                        self.current_line_points.pop(nearest_idx)
                        self.displayKymograph()
            else:
                # 通常の左クリック: 点を追加
                new_point = (event.xdata, event.ydata)
                self.current_line_points.append(new_point)
                self.displayKymograph()

    def onParticleLineMove(self, event):
        """粒子線描画時のマウス移動イベント"""
        if not self.is_drawing_particle_line or not hasattr(self, 'result_ax') or event.inaxes != self.result_ax:
            return
        
        # プレビュー表示（必要に応じて実装）
        pass

    def onParticleLineKeyPress(self, event):
        """粒子線描画時のキー押下イベント"""
        if event.key == 'shift':
            self.shift_pressed = True

    def onParticleLineKeyRelease(self, event):
        """粒子線描画時のキー解放イベント"""
        if event.key == 'shift':
            self.shift_pressed = False

    def onRadiusChanged(self, value):
        """検索半径変更時の処理"""
        self.search_radius = value
        #print(f"DEBUG: Search radius changed to {value}")

    def onDetectionTypeChanged(self, detection_type):
        """検出タイプが変更された時の処理"""
        print(f"DEBUG: Detection type changed to {detection_type}")
        
        # Particle parameters UIの表示/非表示を切り替え
        if hasattr(self, 'particle_params_group'):
            print(f"DEBUG: particle_params_group found, current visibility: {self.particle_params_group.isVisible()}")
            if detection_type == "Particle":
                self.particle_params_group.setVisible(True)
                print("DEBUG: Setting particle_params_group to visible")
            else:
                self.particle_params_group.setVisible(False)
                print("DEBUG: Setting particle_params_group to invisible")
        else:
            print("DEBUG: particle_params_group not found")
        
        # 検出タイプに応じて検出方法を設定
        if detection_type == "Fibril end":
            # 線維端検出の場合は現在の実装を使用
            if hasattr(self, 'auto_detection_button'):
                self.auto_detection_button.setText("Auto detection")
                self.auto_detection_button.setToolTip("Analyze fiber growth from kymograph data")
            # トラッキングパラメータセクションを非表示
            if hasattr(self, 'tracking_params_group'):
                print("DEBUG: Hiding tracking parameters group")
                self.tracking_params_group.setVisible(False)
            else:
                print("DEBUG: tracking_params_group not found")
            # Semi-auto detectionボタンを有効化（Fibril endでも使用可能）
            if hasattr(self, 'draw_line_button'):
                self.draw_line_button.setEnabled(True)
            if hasattr(self, 'clear_button'):
                self.clear_button.setEnabled(True)
            # 説明テキストを更新
            if hasattr(self, 'help_label'):
                self.help_label.setText("1. Draw approximate fiber edge line\n2. Auto detect precise positions")
        elif detection_type == "Particle":
            # 粒子検出の場合は新しい実装を使用
            print("DEBUG: Particle detection mode activated")
            # Auto detectionボタンのテキストとツールチップを変更
            if hasattr(self, 'auto_detection_button'):
                self.auto_detection_button.setText("Auto Detect")
                self.auto_detection_button.setToolTip("Detect particles and track them automatically")
            # トラッキングパラメータセクションを表示
            if hasattr(self, 'tracking_params_group'):
                print("DEBUG: Showing tracking parameters group")
                self.tracking_params_group.setVisible(True)
            else:
                print("DEBUG: tracking_params_group not found")
            # Semi-auto detectionボタンを有効化
            if hasattr(self, 'draw_line_button'):
                self.draw_line_button.setEnabled(True)
            if hasattr(self, 'clear_button'):
                self.clear_button.setEnabled(True)
            # 説明テキストを更新
            if hasattr(self, 'help_label'):
                self.help_label.setText("1. Draw particle lines\n2. Auto detect particles along lines")
        else:
            # デフォルトの場合は元のテキストに戻す
            if hasattr(self, 'auto_detection_button'):
                self.auto_detection_button.setText("Auto detection")
                self.auto_detection_button.setToolTip("Analyze fiber growth from kymograph data")
            # トラッキングパラメータセクションを非表示
            if hasattr(self, 'tracking_params_group'):
                self.tracking_params_group.setVisible(False)
            # Semi-auto detectionボタンを無効化
            if hasattr(self, 'draw_line_button'):
                self.draw_line_button.setEnabled(False)
            if hasattr(self, 'clear_button'):
                self.clear_button.setEnabled(False)
            # 説明テキストを元に戻す
            if hasattr(self, 'help_label'):
                self.help_label.setText("1. Draw approximate fiber edge line\n2. Auto detect precise positions")

    def onDistanceThresholdChanged(self, value):
        """距離閾値が変更された時の処理"""
        self.tracking_distance_threshold = value
        print(f"DEBUG: Distance threshold changed to {value}")

    def onTimeThresholdChanged(self, value):
        """時間閾値が変更された時の処理"""
        self.tracking_time_threshold = value
        print(f"DEBUG: Time threshold changed to {value}")

    def onKalmanFilterChanged(self, state):
        """Kalman Filterの使用設定が変更された時の処理"""
        self.use_kalman_filter = (state == QtCore.Qt.Checked)
        print(f"DEBUG: Kalman Filter {'enabled' if self.use_kalman_filter else 'disabled'}")
        
    def onStartPointChanged(self, start_point):
        """始点選択が変更された時の処理"""
        self.selected_start_point = start_point
        print(f"DEBUG: Start point changed to {start_point}")
        
        # 参照位置と方向を確認
        reference_position = self._getReferencePosition()
        direction_multiplier = self._getDirectionMultiplier()
        print(f"DEBUG: Reference position: {reference_position}, Direction multiplier: {direction_multiplier}")
        
        # グラフが表示されている場合は再描画
        if hasattr(self, 'graph_window') and self.graph_window and self.graph_window.isVisible():
            self.showGraph()

    def showGraph(self):
        """グラフを表示（検出結果はクリアしない）"""
        try:
            print("DEBUG: showGraph called")
            
            # グラフデータを準備
            graph_data = self.prepareGraphData()
            
            if graph_data is None:
                QtWidgets.QMessageBox.warning(self, "Warning", "No graph data available to display.")
                return
            
            # グラフを表示
            self.displayGraph(graph_data)
            
            # 検出結果はクリアしない（kymograph上の表示を保持）
            print("DEBUG: Graph displayed, keeping detection results")
            
        except Exception as e:
            print(f"DEBUG: Error in showGraph: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while displaying the graph:\n{e}")

    def prepareGraphData(self):
        """グラフ表示用のデータを準備"""
        try:
            print("DEBUG: prepareGraphData called")
            
            # 時間データを準備
            if hasattr(gv, 'FrameTime'):
                frame_time = gv.FrameTime / 1000.0  # msを秒に変換
            else:
                frame_time = 1.0  # デフォルト値
            
            total_frames = len(self.image_stack)
            time_points = np.arange(total_frames) * frame_time
            
            print(f"DEBUG: Time data prepared - {total_frames} frames, frame_time: {frame_time}s")
            
            # 検出データを取得
            detection_data = None
            detection_type = "Unknown"
            
            # 各データの存在を確認
            print(f"DEBUG: Checking data availability:")
            print(f"  - fiber_growth_data: {hasattr(self, 'fiber_growth_data') and self.fiber_growth_data is not None}")
            print(f"  - semi_auto_results: {hasattr(self, 'semi_auto_results') and self.semi_auto_results is not None}")
            print(f"  - particle_detection_results: {hasattr(self, 'particle_detection_results') and self.particle_detection_results is not None}")
            
            # 線維端検出データを確認
            if hasattr(self, 'fiber_growth_data') and self.fiber_growth_data is not None:
                if 'edge_positions' in self.fiber_growth_data:
                    detection_data = self.fiber_growth_data['edge_positions']
                    detection_type = "Fiber End"
                    print(f"DEBUG: Using fiber growth data, {len(detection_data)} points")
                else:
                    print("DEBUG: fiber_growth_data exists but no edge_positions")
            
            # セミオート検出データを確認
            elif hasattr(self, 'semi_auto_results') and self.semi_auto_results is not None:
                if 'edge_positions' in self.semi_auto_results:
                    detection_data = self.semi_auto_results['edge_positions']
                    detection_type = "Semi-Auto Detection"
                    print(f"DEBUG: Using semi-auto results, {len(detection_data)} points")
                else:
                    print("DEBUG: semi_auto_results exists but no edge_positions")
            
            # 粒子検出データを確認
            elif hasattr(self, 'particle_tracks') and self.particle_tracks is not None:
                # 軌跡データがある場合は軌跡データを返す
                track_data = self.getTrackDataForGraph()
                if track_data is not None and len(track_data) > 0:
                    return {
                        'track_data': track_data,
                        'detection_type': 'Particle Tracks',
                        'frame_time': frame_time,
                        'total_frames': total_frames
                    }
                else:
                    print("DEBUG: No track data available")
                    return None
            elif hasattr(self, 'particle_detection_results') and self.particle_detection_results is not None:
                # 軌跡データがない場合は個別の粒子データを使用
                detection_data = [particle['position'] for particle in self.particle_detection_results]
                detection_type = "Particle"
                print(f"DEBUG: Using particle detection results, {len(detection_data)} points")
            
            if detection_data is None:
                print("DEBUG: No detection data available")
                return None
            
            # 選択された始点からの距離に変換
            converted_positions = self.convertToDistanceFromStart(detection_data)
            
            # グラフデータを構築
            graph_data = {
                'time_points': time_points,
                'positions': converted_positions,
                'detection_type': detection_type,
                'frame_time': frame_time,
                'total_frames': total_frames
            }
            
            print(f"DEBUG: Graph data prepared - {detection_type}, {len(converted_positions)} points")
            return graph_data
            
        except Exception as e:
            print(f"DEBUG: Error in prepareGraphData: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convertToDistanceFromStart(self, positions):
        """位置データを選択された始点からの距離（nm）に変換（始点によって方向を変更）"""
        try:
            # 選択された始点に基づいて参照位置と方向を決定
            reference_position = self._getReferencePosition()
            direction_multiplier = self._getDirectionMultiplier()
            
            # ピクセルからnmへの変換係数を取得
            pixel_to_nm = self.getPixelToNmConversion()
            
            # 各位置を参照位置からの距離（nm）に変換
            distances = []
            for pos in positions:
                distance_pixels = (pos - reference_position) * direction_multiplier
                distance_nm = distance_pixels * pixel_to_nm
                distances.append(distance_nm)
            
            print(f"DEBUG: Converted {len(distances)} positions, reference_position: {reference_position}, direction: {direction_multiplier}, pixel_to_nm: {pixel_to_nm:.3f}")
            print(f"DEBUG: Distance range: {min(distances):.2f} to {max(distances):.2f} nm")
            return np.array(distances)
            
        except Exception as e:
            print(f"DEBUG: Error in convertToDistanceFromStart: {e}")
            return np.array(positions)

    def getPixelToNmConversion(self):
        """ピクセルからnmへの変換係数を取得"""
        try:
            # 画像の物理サイズ情報から変換係数を計算
            if (hasattr(gv, 'XScanSize') and hasattr(gv, 'YScanSize') and 
                hasattr(gv, 'aryData') and gv.aryData is not None and
                gv.XScanSize > 0 and gv.YScanSize > 0):
                
                image_height, image_width = gv.aryData.shape
                pixel_size_x = gv.XScanSize / image_width  # nm/pixel
                pixel_size_y = gv.YScanSize / image_height  # nm/pixel
                
                # X方向の変換係数を使用（kymographでは主にX方向の移動を測定）
                pixel_to_nm = pixel_size_x
                print(f"DEBUG: Calculated pixel_to_nm from scan size: {pixel_to_nm:.3f} nm/pixel")
                return pixel_to_nm
                
            # フォールバック: デフォルト値
            else:
                print("DEBUG: Using default pixel size (1.0 nm/pixel)")
                return 1.0
                
        except Exception as e:
            print(f"DEBUG: Error getting pixel to nm conversion: {e}")
            return 1.0

    def displayGraph(self, graph_data):
        """グラフを表示"""
        try:
            print("DEBUG: displayGraph called")
            
            # 既存のグラフウィンドウがある場合は再利用、なければ新規作成
            if hasattr(self, 'graph_window') and self.graph_window is not None and self.graph_window.isVisible():
                # 既存のウィンドウを再利用
                print("DEBUG: Reusing existing graph window")
                # 既存のグラフをクリア
                self.graph_figure.clear()
            else:
                # 新しいウィンドウを作成
                print("DEBUG: Creating new graph window")
                if hasattr(self, 'graph_window') and self.graph_window is not None:
                    try:
                        # 現在の位置とサイズを保存
                        self.graph_window_geometry = self.graph_window.geometry()
                        print(f"DEBUG: Saving window geometry: {self.graph_window_geometry}")
                        self.graph_window.close()
                    except:
                        pass
                
                # 新しいウィンドウを作成（インスタンス変数として保存）
                self.graph_window = QtWidgets.QWidget()
                self.graph_window.setWindowTitle("Fiber Growth Analysis")
                self.graph_window.setMinimumSize(700, 500)
                self.graph_window.setWindowFlags(QtCore.Qt.Window)
                
                # 親ウィンドウを設定（重要！）
                self.graph_window.setParent(self, QtCore.Qt.Window)
                
                # レイアウトを作成
                layout = QtWidgets.QVBoxLayout()
                self.graph_window.setLayout(layout)
                
                # フィギュアとキャンバスを作成
                self.graph_figure = Figure(figsize=(10, 6))
                self.graph_canvas = FigureCanvas(self.graph_figure)
                layout.addWidget(self.graph_canvas)
                
                # ウィンドウの移動イベントを監視して位置を保存
                self.graph_window.moveEvent = self.onGraphWindowMoved
                
                # 保存された位置とサイズを復元（レイアウト作成後に設定）
                if self.graph_window_geometry is not None:
                    # 遅延実行で位置を確実に設定
                    QtCore.QTimer.singleShot(100, lambda: self.restoreGraphWindowPosition())
                    print(f"DEBUG: Scheduling geometry restoration: {self.graph_window_geometry}")
            
            # 新しいグラフを作成
            ax = self.graph_figure.add_subplot(111)
            
            # データをプロット
            detection_type = graph_data['detection_type']
            
            if detection_type == 'Particle Tracks' and 'track_data' in graph_data:
                # 軌跡データの場合、各軌跡を個別の線としてプロット
                track_data = graph_data['track_data']
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                
                print(f"DEBUG: Plotting {len(track_data)} particle tracks")
                
                for i, track in enumerate(track_data):
                    time_points = track['time_points']
                    positions = track['positions']
                    color = colors[i % len(colors)]
                    
                    if len(time_points) > 0 and len(positions) > 0:
                        # 位置をnmに変換（ピクセルからnmへの変換）
                        pixel_to_nm = self.getPixelToNmConversion()
                        converted_positions = np.array(positions) * pixel_to_nm
                        
                        # 各軌跡を個別の線としてプロット（凡例なし）
                        ax.plot(time_points, converted_positions, color=color, linewidth=1, 
                               alpha=0.8)
                        ax.plot(time_points, converted_positions, 'o', color=color, 
                               markersize=3, alpha=0.8)
                        
                        print(f"DEBUG: Track {i+1}: {len(time_points)} points, position range: {min(converted_positions):.2f} to {max(converted_positions):.2f} nm")
                

                
            else:
                # 従来のデータ（単一の線）
                time_points = graph_data['time_points']
                positions = graph_data['positions']
                
                print(f"DEBUG: Plotting data - time_points: {len(time_points)}, positions: {len(positions)}")
                print(f"DEBUG: Time range: {time_points[0]:.2f} to {time_points[-1]:.2f} s")
                print(f"DEBUG: Position range: {positions[0]:.2f} to {positions[-1]:.2f} nm")
                
                # プロット（線と点の両方、凡例なし）
                ax.plot(time_points, positions, 'b-', linewidth=2, alpha=0.7)
                ax.plot(time_points, positions, 'ro', markersize=4)
                

            
            # 軸ラベルとタイトル
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Distance from Line Start (nm)', fontsize=12)
            ax.set_title(f'Fiber Growth Analysis - {detection_type}', fontsize=14, pad=15)
            
            # グリッドを追加
            ax.grid(True, alpha=0.3)
            
            # 統計情報ボックスは削除（邪魔になるため）
            
            # レイアウトを調整
            self.graph_figure.tight_layout()
            
            # キャンバスを更新
            self.graph_canvas.draw()
            
            # ウィンドウを表示（複数の方法で確実に表示）
            self.graph_window.show()
            self.graph_window.raise_()
            self.graph_window.activateWindow()
            
            # Save Dataボタンを有効化
            if hasattr(self, 'save_data_button'):
                self.save_data_button.setEnabled(True)
                print("DEBUG: Save Data button enabled")
            
            print(f"DEBUG: Graph window created and displayed - {detection_type}, {len(positions)} points")
            print(f"DEBUG: Window visible: {self.graph_window.isVisible()}")
            
            # ウィンドウが閉じられた時の処理を設定
            self.graph_window.closeEvent = self.onGraphWindowClosed
            
        except Exception as e:
            print(f"DEBUG: Error in displayGraph: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while displaying the graph:\n{e}")

    def onGraphWindowClosed(self, event):
        """グラフウィンドウが閉じられた時の処理"""
        self.graph_window = None
        
        # Save Dataボタンを無効化
        if hasattr(self, 'save_data_button'):
            self.save_data_button.setEnabled(False)
            print("DEBUG: Save Data button disabled")
        
    def onGraphWindowMoved(self, event):
        """グラフウィンドウが移動された時の処理"""
        if hasattr(self, 'graph_window') and self.graph_window is not None:
            # 新しい位置とサイズを保存
            self.graph_window_geometry = self.graph_window.geometry()
            #print(f"DEBUG: Window moved, saving geometry: {self.graph_window_geometry}")
        event.accept()
        
    def restoreGraphWindowPosition(self):
        """グラフウィンドウの位置を復元"""
        if (hasattr(self, 'graph_window') and self.graph_window is not None and 
            hasattr(self, 'graph_window_geometry') and self.graph_window_geometry is not None):
            self.graph_window.setGeometry(self.graph_window_geometry)
            #print(f"DEBUG: Restored window geometry: {self.graph_window_geometry}")
        
    def saveGraphData(self):
        """グラフデータをCSVファイルとして保存"""
        try:
            #print("DEBUG: saveGraphData called")
            
            # グラフデータを取得
            graph_data = self.prepareGraphData()
            if graph_data is None:
                QtWidgets.QMessageBox.warning(self, "Warning", "No data available to save.")
                return
            
            # デフォルトフォルダを取得
            default_folder = self.getDefaultSaveFolder()
            if not default_folder:
                default_folder = os.path.expanduser("~/Desktop")
            
            # ファイル名を生成
            detection_type = graph_data['detection_type']
            if detection_type == 'Particle Tracks':
                default_filename = "particle_tracks_data.csv"
            else:
                default_filename = "fiber_growth_data.csv"
            
            # ファイル保存ダイアログを表示
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 
                "Save Graph Data", 
                os.path.join(default_folder, default_filename),
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return  # ユーザーがキャンセル
            
            # データをCSVファイルとして保存
            self._exportGraphDataToCSV(graph_data, file_path)
            
            QtWidgets.QMessageBox.information(self, "Complete", f"Data saved successfully:\n{file_path}")
            
        except Exception as e:
            #print(f"DEBUG: Error in saveGraphData: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred while saving data:\n{e}")
        
    def _exportGraphDataToCSV(self, graph_data, file_path):
        """グラフデータをCSVファイルとしてエクスポート"""
        try:
            import pandas as pd
            
            detection_type = graph_data['detection_type']
            
            if detection_type == 'Particle Tracks' and 'track_data' in graph_data:
                # 粒子トラッキングデータの場合
                track_data = graph_data['track_data']
                
                # 時間データを準備
                all_time_points = set()
                for track in track_data:
                    all_time_points.update(track['time_points'])
                time_points = sorted(list(all_time_points))
                
                # データフレームを作成
                df_data = {'Time (s)': time_points}
                
                # 各軌跡のデータを列として追加
                for i, track in enumerate(track_data):
                    track_positions = dict(zip(track['time_points'], track['positions']))
                    
                    # ピクセルからnmへの変換
                    pixel_to_nm = self.getPixelToNmConversion()
                    
                    # 各時間点での位置を取得（ない場合はNaN）
                    positions = []
                    for t in time_points:
                        if t in track_positions:
                            pos_nm = track_positions[t] * pixel_to_nm
                            positions.append(pos_nm)
                        else:
                            positions.append(float('nan'))
                    
                    df_data[f'Particle_{i+1} (nm)'] = positions
                
                df = pd.DataFrame(df_data)
                
            else:
                # Fiber end検出データの場合
                time_points = graph_data['time_points']
                positions = graph_data['positions']
                
                # データフレームを作成
                df = pd.DataFrame({
                    'Time (s)': time_points,
                    'Position (nm)': positions
                })
            
            # 数値を小数点第2位までに制限
            for column in df.columns:
                if column != 'Time (s)':  # 時間列以外の数値列を処理
                    df[column] = df[column].round(2)
                else:
                    # 時間列も小数点第2位までに制限
                    df[column] = df[column].round(2)
            
            # CSVファイルとして保存
            df.to_csv(file_path, index=False, float_format='%.2f')
            #print(f"DEBUG: Data saved to {file_path}")
            #print(f"DEBUG: DataFrame shape: {df.shape}")
            
        except Exception as e:
            #print(f"DEBUG: Error in _exportGraphDataToCSV: {e}")
            raise
        
    def performSemiAutoDetection(self):
        """セミオートマチック検出を実行"""
        detection_type = self.detection_combo.currentText()
        
        if detection_type == "Particle":
            # 粒子検出モードの場合
            if not self.particle_lines or len(self.particle_lines) == 0:
                QtWidgets.QMessageBox.warning(self, "Error", "Please draw particle lines first.")
                return
            
            try:
                # 粒子検出結果をクリア
                self.particle_detection_results = None
                self.particle_tracks = None
                
                # 各粒子線に沿って粒子検出を実行
                all_particles = []
                for line_data in self.particle_lines:
                    particles = self.detectParticlesAlongParticleLine(line_data)
                    if particles:
                        all_particles.extend(particles)
                
                if not all_particles:
                    QtWidgets.QMessageBox.warning(self, "Warning", "No particles detected along the lines.")
                    return
                
                # 検出結果を保存
                self.particle_detection_results = all_particles
                
                # 軌跡追跡を実行
                self.trackParticles()
                
                # kymographを再表示
                self.displayKymograph()
                
                # 結果を表示
                track_count = len(self.particle_tracks) if self.particle_tracks else 0
                message = f"Semi-auto particle detection completed.\n\n"
                message += f"Detected particles: {len(all_particles)}个\n"
                message += f"Tracked trajectories: {track_count}个\n\n"
                message += "Tracked particles are displayed with pink lines."
                
                QtWidgets.QMessageBox.information(self, "Semi-Auto Detection Complete", message)
                
            except Exception as e:
                #print(f"DEBUG: Exception in particle semi-auto detection: {e}")
                import traceback
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(self, "Error", f"Error in particle semi-auto detection:\n{e}")
        else:
            # 線維端検出モードの場合（従来の処理）
            if not self.reference_line_complete or len(self.reference_line_points) < 2:
                QtWidgets.QMessageBox.warning(self, "Error", "Please draw a reference line first.")
                return
            
            try:
                # Auto検出結果をクリア
                self.fiber_growth_data = None
                
                detection_method = self.detection_method_combo.currentText().lower().replace(" ", "_")
                
                # 検出を実行
                edge_positions, confidence_scores = self.detectAlongReferenceLine(detection_method)
                
                if edge_positions is None:
                    QtWidgets.QMessageBox.warning(self, "Error", "Semi-automatic detection failed.")
                    return
                
                # 時間軸を生成（デフォルト値1.0）
                time_interval = 1.0
                time_points = np.arange(len(edge_positions)) * time_interval
                
                # 成長解析データを作成
                growth_data = self.createSemiAutoGrowthData(
                    time_points, edge_positions, confidence_scores, detection_method
                )
                
                # 結果を保存
                self.semi_auto_results = growth_data
                self.fiber_growth_data = growth_data
                
                # 結果を表示
                self.displayFiberGrowthAnalysis(growth_data)
                self.displayKymograph()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(self, "Error", f"Error in semi-automatic detection:\n{e}")

    def detectAlongReferenceLine(self, detection_method):
        """参照線に沿った自動検出（矩形モード対応）"""
        try:
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            total_frames = len(self.image_stack)
            
            # detectionコンボボックスの選択を確認
            detection_type = self.detection_combo.currentText()
            
            if detection_type == "Particle":
                # 粒子検出の場合は参照点付近で検出
                #print("DEBUG: Semi-auto particle detection")
                return self.detectParticlesAlongReferenceLine(total_frames)
            else:
                # 線維端検出の場合は従来通り
                if is_rectangular_mode:
                    #print(f"DEBUG: Semi-auto detection in rectangular mode, frame_height: {self.actual_frame_height}")
                    return self.detectAlongReferenceLineRectangular(detection_method, total_frames)
                else:
                    #print("DEBUG: Semi-auto detection in linear mode")
                    return self.detectAlongReferenceLineLinear(detection_method, total_frames)
                
        except Exception as e:
            #print(f"DEBUG: Error in detectAlongReferenceLine: {e}")
            return None, None

    def detectAlongReferenceLineLinear(self, detection_method, total_frames):
        """直線モードでの参照線に沿った検出"""
        # 参照線から各フレームでの予想位置を計算
        predicted_positions = self.interpolateReferenceLine()
        
        edge_positions = []
        confidence_scores = []
        
        for frame_idx in range(total_frames):
            # このフレームでの予想位置
            predicted_x = predicted_positions[frame_idx]
            
            # フレームデータを取得
            frame_profile = self.kymograph_data[frame_idx, :]
            
            # 検索範囲を設定
            search_start = max(0, int(predicted_x - self.search_radius))
            search_end = min(len(frame_profile), int(predicted_x + self.search_radius + 1))
            
            if search_start >= search_end:
                edge_positions.append(predicted_x)
                confidence_scores.append(0.0)
                continue
            
            # 検索範囲内で最適位置を検出
            search_profile = frame_profile[search_start:search_end]
            search_positions = np.arange(search_start, search_end)
            
            local_pos, confidence = self.detectInLocalRegion(search_profile, search_positions, detection_method)
            
            edge_positions.append(local_pos)
            confidence_scores.append(confidence)
        
        return np.array(edge_positions), np.array(confidence_scores)

    def detectAlongReferenceLineRectangular(self, detection_method, total_frames):
        """矩形モードでの参照線に沿った検出"""
        # 参照線から各フレームでの予想位置を計算
        predicted_positions = self.interpolateReferenceLine()
        
        edge_positions = []
        confidence_scores = []
        
        frame_height = int(self.actual_frame_height)
        
        for frame_idx in range(total_frames):
            # このフレームでの予想位置
            predicted_x = predicted_positions[frame_idx]
            
            # フレームデータを抽出
            start_row = frame_idx * frame_height
            end_row = start_row + frame_height
            
            #if end_row > self.kymograph_data.shape[0]:
                #print(f"DEBUG: Frame {frame_idx} exceeds data bounds")
                #break
                
            frame_data = self.kymograph_data[start_row:end_row, :]
            
            # フレーム内で検索範囲を設定
            search_start = max(0, int(predicted_x - self.search_radius))
            search_end = min(frame_data.shape[1], int(predicted_x + self.search_radius + 1))
            
            if search_start >= search_end:
                edge_positions.append(predicted_x)
                confidence_scores.append(0.0)
                continue
            
            # フレーム内の検索領域を抽出
            search_region = frame_data[:, search_start:search_end]
            
            # 検索領域内で端面検出
            local_pos, confidence = self.detectInFrameRegion(search_region, search_start, detection_method)
            
            edge_positions.append(local_pos)
            confidence_scores.append(confidence)
            
            #print(f"DEBUG: Frame {frame_idx} - predicted: {predicted_x:.1f}, detected: {local_pos:.1f}, confidence: {confidence:.3f}")
        
        return np.array(edge_positions), np.array(confidence_scores)

    def detectInFrameRegion(self, search_region, search_start, detection_method):
        """矩形フレーム内の検索領域での端面検出"""
        try:
            if search_region.size == 0:
                return search_start + search_region.shape[1] // 2, 0.0
            
            height, width = search_region.shape
            
            if detection_method == "edge_detection":
                # 各行でエッジ検出を行い、結果を統合
                row_positions = []
                row_confidences = []
                
                for row_idx in range(height):
                    profile = search_region[row_idx, :]
                    gradient = np.gradient(profile)
                    
                    if len(gradient) > 0:
                        max_grad_idx = np.argmax(np.abs(gradient))
                        row_positions.append(max_grad_idx)
                        
                        max_gradient = np.abs(gradient[max_grad_idx])
                        mean_gradient = np.mean(np.abs(gradient))
                        confidence = min(1.0, max_gradient / (mean_gradient + 1e-10))
                        row_confidences.append(confidence)
                
                if row_positions:
                    # 重み付き平均で位置を決定
                    row_positions = np.array(row_positions)
                    row_confidences = np.array(row_confidences)
                    
                    if np.sum(row_confidences) > 0:
                        weighted_pos = np.sum(row_positions * row_confidences) / np.sum(row_confidences)
                        avg_confidence = np.mean(row_confidences)
                    else:
                        weighted_pos = np.mean(row_positions)
                        avg_confidence = 0.0
                    
                    global_pos = search_start + weighted_pos
                    return global_pos, avg_confidence
                
            elif detection_method == "max_intensity":
                # 最大強度位置を検出
                max_pos = np.unravel_index(np.argmax(search_region), search_region.shape)
                global_pos = search_start + max_pos[1]  # x座標
                
                max_intensity = search_region[max_pos]
                mean_intensity = np.mean(search_region)
                confidence = min(1.0, (max_intensity - mean_intensity) / (max_intensity + 1e-10))
                
                return global_pos, confidence
                
            elif detection_method == "centroid":
                # 重心ベース検出
                threshold = np.percentile(search_region, 70)
                mask = search_region >= threshold
                
                if np.sum(mask) > 0:
                    y_coords, x_coords = np.where(mask)
                    weights = search_region[mask]
                    
                    if np.sum(weights) > 0:
                        centroid_x = np.sum(x_coords * weights) / np.sum(weights)
                        global_pos = search_start + centroid_x
                        
                        confidence = min(1.0, np.sum(weights) / (search_region.size * np.max(search_region) + 1e-10))
                        return global_pos, confidence
            
            # デフォルト：中央位置
            return search_start + width // 2, 0.0
            
        except Exception as e:
            #print(f"DEBUG: Error in detectInFrameRegion: {e}")
            return search_start + search_region.shape[1] // 2 if search_region.size > 0 else search_start, 0.0

    def detectInLocalRegion(self, search_profile, search_positions, detection_method):
        """1D検索領域での検出（直線モード用）"""
        try:
            if detection_method == "edge_detection":
                return self.detectEdgeInRegion(search_profile, search_positions)
            elif detection_method == "max_intensity":
                return self.detectMaxIntensityInRegion(search_profile, search_positions)
            elif detection_method == "centroid":
                return self.detectCentroidInRegion(search_profile, search_positions)
            else:
                return self.detectEdgeInRegion(search_profile, search_positions)
                
        except Exception as e:
            #print(f"DEBUG: Error in detectInLocalRegion: {e}")
            return search_positions[len(search_positions)//2], 0.0

    def interpolateReferenceLine(self):
        """参照線から各フレームでの予想位置を計算（矩形モード対応）"""
        try:
            # 参照線の点を時間順でソート
            sorted_points = sorted(self.reference_line_points, key=lambda p: p[1])  # y座標（時間）でソート
            
            ref_y = np.array([p[1] for p in sorted_points])  # 時間（kymograph上のy座標）
            ref_x = np.array([p[0] for p in sorted_points])  # 位置
            
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            total_frames = len(self.image_stack)
            
            if is_rectangular_mode:
                # 矩形モード：各フレームの中央位置を基準に補間
                frame_height = int(self.actual_frame_height)
                frame_center_positions = []
                
                for frame_idx in range(total_frames):
                    frame_center_y = frame_idx * frame_height + frame_height / 2
                    frame_center_positions.append(frame_center_y)
                
                frame_center_positions = np.array(frame_center_positions)
                
                # フレーム中央位置で補間
                predicted_positions = np.interp(frame_center_positions, ref_y, ref_x)
                
                #print(f"DEBUG: Rectangular mode interpolation - {total_frames} frames, frame_height: {frame_height}")
                #print(f"DEBUG: Frame centers: {frame_center_positions[:5]}...")  # 最初の5個を表示
                
            else:
                # 直線モード：従来通り
                frame_times = np.arange(total_frames)
                predicted_positions = np.interp(frame_times, ref_y, ref_x)
                
                #print(f"DEBUG: Linear mode interpolation - {total_frames} frames")
            
            #print(f"DEBUG: Interpolated {len(predicted_positions)} positions")
            
            return predicted_positions
            
        except Exception as e:
            #print(f"DEBUG: Error in interpolateReferenceLine: {e}")
            total_frames = len(self.image_stack) if hasattr(self, 'image_stack') else self.kymograph_data.shape[0]
            return np.zeros(total_frames)

    def detectEdgeInRegion(self, profile, positions):
        """局所領域でのエッジ検出"""
        try:
            if len(profile) < 3:
                return positions[len(positions)//2], 0.0
            
            # 勾配を計算
            gradient = np.gradient(profile)
            
            # 最大勾配位置を検出（線維端での急激な変化）
            max_grad_idx = np.argmax(np.abs(gradient))
            detected_pos = positions[max_grad_idx]
            
            # 信頼度を計算
            max_gradient = np.abs(gradient[max_grad_idx])
            mean_gradient = np.mean(np.abs(gradient))
            confidence = min(1.0, max_gradient / (mean_gradient + 1e-10))
            
            return detected_pos, confidence
            
        except Exception as e:
            #print(f"DEBUG: Error in detectEdgeInRegion: {e}")
            return positions[len(positions)//2], 0.0

    def detectMaxIntensityInRegion(self, profile, positions):
        """局所領域での最大強度検出"""
        try:
            max_idx = np.argmax(profile)
            detected_pos = positions[max_idx]
            
            # 信頼度を計算
            max_intensity = profile[max_idx]
            mean_intensity = np.mean(profile)
            confidence = min(1.0, (max_intensity - mean_intensity) / (max_intensity + 1e-10))
            
            return detected_pos, confidence
            
        except Exception as e:
            #print(f"DEBUG: Error in detectMaxIntensityInRegion: {e}")
            return positions[len(positions)//2], 0.0

    def detectCentroidInRegion(self, profile, positions):
        """局所領域での重心検出"""
        try:
            # 閾値以上の領域で重心を計算
            threshold = np.percentile(profile, 70)
            mask = profile >= threshold
            
            if np.sum(mask) == 0:
                return positions[len(positions)//2], 0.0
            
            weights = profile * mask
            total_weight = np.sum(weights)
            
            if total_weight == 0:
                return positions[len(positions)//2], 0.0
            
            centroid_idx = np.sum(np.arange(len(profile)) * weights) / total_weight
            detected_pos = positions[int(centroid_idx)]
            
            # 信頼度を計算
            confidence = min(1.0, total_weight / (len(profile) * np.max(profile) + 1e-10))
            
            return detected_pos, confidence
            
        except Exception as e:
            #print(f"DEBUG: Error in detectCentroidInRegion: {e}")
            return positions[len(positions)//2], 0.0

    def postProcessSemiAutoResults(self, edge_positions, predicted_positions):
        """セミオート結果の後処理"""
        try:
            edge_positions = np.array(edge_positions)
            predicted_positions = np.array(predicted_positions)
            
            # 異常値検出（参照線から大きく外れた点を修正）
            deviations = np.abs(edge_positions - predicted_positions)
            threshold = np.std(deviations) * 2.5
            
            outlier_mask = deviations > threshold
            if np.sum(outlier_mask) > 0:
                #print(f"DEBUG: Found {np.sum(outlier_mask)} outliers, correcting...")
                edge_positions[outlier_mask] = predicted_positions[outlier_mask]
            
            # 軽微なスムージング
            if len(edge_positions) >= 5:
                from scipy.ndimage import median_filter
                edge_positions = median_filter(edge_positions, size=3)
            
            return edge_positions
            
        except Exception as e:
            #print(f"DEBUG: Error in postProcessSemiAutoResults: {e}")
            return edge_positions

    def createSemiAutoGrowthData(self, time_points, edge_positions, confidence_scores, detection_method):
        """セミオート検出結果から成長データを作成"""
        try:
            # 成長速度を計算
            growth_speed = np.gradient(edge_positions, time_points[1] - time_points[0])
            
            # フィッティングパラメータを計算
            growth_params = self._fitGrowthTrajectory(time_points, edge_positions)
            
            # 結果をまとめる
            growth_data = {
                'time_points': time_points,
                'edge_positions': edge_positions,
                'confidence': confidence_scores,
                'growth_params': growth_params,
                'growth_speed': growth_speed,
                'total_growth': edge_positions[-1] - edge_positions[0],
                'average_speed': np.mean(growth_speed),
                'detection_method': f'semi_auto_{detection_method}',
                'reference_line_points': self.reference_line_points,
                'search_radius': self.search_radius
            }
            
            return growth_data
            
        except Exception as e:
            #print(f"DEBUG: Error in createSemiAutoGrowthData: {e}")
            return None

    def detectParticlesInKymograph(self):
        """Kymographでの粒子検出（Auto mode）"""
        try:
            #print("DEBUG: detectParticlesInKymograph called")
            
            if self.kymograph_data is None:
                #print("DEBUG: kymograph_data is None")
                return
            
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            if is_rectangular_mode:
                particle_positions = self.detectParticlesRectangular()
            else:
                particle_positions = self.detectParticlesLinear()
            
            if particle_positions is None or len(particle_positions) == 0:
                QtWidgets.QMessageBox.warning(self, "Warning", "No particles were detected.")
                return
            
            # 検出結果を保存
            self.particle_detection_results = particle_positions
            
            # 軌跡追跡を実行
            self.trackParticles()
            
            # Kymographを再表示して検出結果を描画
            self.displayKymograph()
            
            # 結果を表示
            track_count = len(self.particle_tracks) if self.particle_tracks else 0
            message = f"粒子検出とトラッキングが完了しました。\n\n"
            message += f"検出された粒子: {len(particle_positions)}個\n"
            message += f"追跡された軌跡: {track_count}個\n\n"
            message += "トラッキングされた粒子はピンク色の線で結ばれて表示されています。"
            
            QtWidgets.QMessageBox.information(self, "Auto Detect 完了", message)
            
        except Exception as e:
            #print(f"DEBUG: Error in detectParticlesInKymograph: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during particle detection:\n{e}")

    def findNearestParticle(self, click_x, click_y):
        """クリック位置に最も近い粒子を検索"""
        if self.particle_detection_results is None:
            return None, -1

        min_distance = float('inf')
        nearest_particle = None
        nearest_index = -1
        
        # 矩形モードかどうかを判定
        is_rectangular_mode = (hasattr(self, 'actual_frame_height') and
                             self.actual_frame_height is not None and
                             self.actual_frame_height > 1)

        for i, particle in enumerate(self.particle_detection_results):
            # 粒子の座標を取得
            particle_x = particle['position']
            
            if is_rectangular_mode:
                # 矩形モードの場合、Y座標を描画時と同じ方法で計算
                frame_height = int(self.actual_frame_height)
                particle_y = particle['frame'] * frame_height + frame_height / 2
            else:
                # 直線モードの場合、フレーム番号をそのままY座標として使用
                particle_y = particle['frame']

            # 距離を計算
            distance = ((click_x - particle_x) ** 2 + (click_y - particle_y) ** 2) ** 0.5

            if distance < min_distance and distance <= self.particle_click_radius:
                min_distance = distance
                nearest_particle = particle
                nearest_index = i

        return nearest_particle, nearest_index

    def addParticleAtPosition(self, click_x, click_y):
        """指定位置近傍で局所最大値を検索して粒子を追加"""
        try:
            # 矩形モードかどうかでフレーム番号の計算方法を変える
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and
                                 self.actual_frame_height is not None and
                                 self.actual_frame_height > 1)

            if is_rectangular_mode:
                frame_height = int(self.actual_frame_height)
                # クリックされたY座標からフレーム番号を逆算
                frame_idx = int(click_y / frame_height)
            else:
                # クリック位置のフレーム番号を取得
                frame_idx = int(click_y)

            if frame_idx < 0 or frame_idx >= len(self.image_stack):
                #print(f"DEBUG: Invalid frame index: {frame_idx}")
                return

            # フレームデータを取得
            if is_rectangular_mode:
                # 矩形モード
                frame_height = int(self.actual_frame_height)
                start_row = frame_idx * frame_height
                end_row = start_row + frame_height
                
                if end_row > self.kymograph_data.shape[0]:
                    return
                
                frame_data = self.kymograph_data[start_row:end_row, :]
            else:
                # 直線モード
                frame_data = self.kymograph_data[frame_idx, :]
            
            # 局所最大値を検索する範囲を設定
            search_radius = 20  # ピクセル
            x_start = max(0, int(click_x) - search_radius)
            x_end = min(frame_data.shape[1] if len(frame_data.shape) > 1 else len(frame_data), 
                       int(click_x) + search_radius)
            
            if len(frame_data.shape) > 1:
                # 2次元データ（矩形モード）
                search_region = frame_data[:, x_start:x_end]
                max_pos = np.unravel_index(np.argmax(search_region), search_region.shape)
                max_x = x_start + max_pos[1]
                max_y = frame_idx
            else:
                # 1次元データ（直線モード）
                search_region = frame_data[x_start:x_end]
                max_x = x_start + np.argmax(search_region)
                max_y = frame_idx
            
            # 新しい粒子を作成
            new_particle = {
                'frame': frame_idx,
                'position': max_x,
                'intensity': frame_data[max_y, max_x] if len(frame_data.shape) > 1 else frame_data[max_x],
                'confidence': 1.0,  # 手動追加なので信頼度は1.0
                'size': 1
            }
            
            # 検出結果に追加
            if self.particle_detection_results is None:
                self.particle_detection_results = []
            
            self.particle_detection_results.append(new_particle)
            
            # 軌跡データも更新
            if self.particle_tracks is not None:
                self.updateTracksAfterParticleAddition(new_particle)
            
            # kymographを再表示
            self.displayKymograph()
            
            #print(f"DEBUG: Added particle at frame {frame_idx}, position {max_x}")
            
        except Exception as e:
            #print(f"DEBUG: Error in addParticleAtPosition: {e}")
            import traceback
            traceback.print_exc()

    def updateTracksAfterParticleAddition(self, new_particle):
        """粒子追加後に軌跡データを更新"""
        if self.particle_tracks is None:
            return
        
        # 新しい粒子を既存の軌跡に追加するか、新しい軌跡を作成するかを決定
        best_track = None
        best_distance = float('inf')
        
        for track in self.particle_tracks:
            # 軌跡の最後の粒子との距離をチェック
            if len(track['particles']) > 0:
                last_particle = track['particles'][-1]
                distance = abs(new_particle['position'] - last_particle['position'])
                frame_diff = new_particle['frame'] - last_particle['frame']
                
                # 距離と時間の条件をチェック
                if (distance <= self.tracking_distance_threshold and 
                    frame_diff <= self.tracking_time_threshold and
                    frame_diff > 0 and  # 同じフレームは除外
                    distance < best_distance):
                    best_distance = distance
                    best_track = track
        
        if best_track is not None:
            # 既存の軌跡に追加
            best_track['particles'].append(new_particle)
            best_track['end_frame'] = new_particle['frame']
            #print(f"DEBUG: Added particle to existing track {best_track['track_id']}")
        else:
            # 新しい軌跡を作成
            new_track = {
                'track_id': len(self.particle_tracks),
                'particles': [new_particle],
                'start_frame': new_particle['frame'],
                'end_frame': new_particle['frame']
            }
            self.particle_tracks.append(new_track)
            #print(f"DEBUG: Created new track {new_track['track_id']} for added particle")

    def removeParticle(self, particle_index):
        """指定されたインデックスの粒子を削除"""
        if self.particle_detection_results is None or particle_index < 0:
            return
        
        if particle_index < len(self.particle_detection_results):
            removed_particle = self.particle_detection_results.pop(particle_index)
            #print(f"DEBUG: Removed particle at frame {removed_particle['frame']}, position {removed_particle['position']}")
            
            # 軌跡データも更新
            if self.particle_tracks is not None:
                self.updateTracksAfterParticleRemoval(removed_particle)
            
            # kymographを再表示
            self.displayKymograph()

    def updateTracksAfterParticleRemoval(self, removed_particle):
        """粒子削除後に軌跡データを更新"""
        if self.particle_tracks is None:
            return
        
        updated_tracks = []
        
        for track in self.particle_tracks:
            # 削除された粒子を含む軌跡から、その粒子を除去
            updated_particles = []
            for particle in track['particles']:
                if (particle['frame'] != removed_particle['frame'] or 
                    particle['position'] != removed_particle['position']):
                    updated_particles.append(particle)
            
            # 軌跡に粒子が残っている場合は保持
            if len(updated_particles) > 0:
                track['particles'] = updated_particles
                track['start_frame'] = min(p['frame'] for p in updated_particles)
                track['end_frame'] = max(p['frame'] for p in updated_particles)
                updated_tracks.append(track)
        
        self.particle_tracks = updated_tracks
        #print(f"DEBUG: Updated tracks after particle removal. Remaining tracks: {len(updated_tracks)}")

    def trackParticles(self):
        """検出された粒子を軌跡として追跡（trackpy使用）"""
        if self.particle_detection_results is None or len(self.particle_detection_results) == 0:
            #print("DEBUG: No particles to track")
            return
        
        try:
            import trackpy as tp
            #print("DEBUG: Starting particle tracking with trackpy")
            
            # trackpy用のデータフレームを作成（高信頼度の粒子のみ）
            features_list = []
            for particle in self.particle_detection_results:
                # 信頼度が0.7以上の粒子のみを使用
                if particle['confidence'] >= 0.7:
                    features_list.append({
                        'frame': particle['frame'],
                        'x': particle['position'],
                        'y': 0,  # kymographは1次元なのでy=0
                        'intensity': particle['intensity'],
                        'size': particle['size'],
                        'ecc': 0,  # 楕円率（1次元なので0）
                        'signal': particle['confidence'],
                        'raw_mass': particle['intensity']
                    })
            
            if not features_list:
                #print("DEBUG: No features to track")
                return
            
            # pandas DataFrameに変換
            import pandas as pd
            features_df = pd.DataFrame(features_list)
            
           # print(f"DEBUG: Created features DataFrame with {len(features_df)} particles")
            
            # trackpyで軌跡追跡
            # パラメータを設定
            search_range = self.tracking_distance_threshold
            memory = self.tracking_time_threshold
            use_trackpy = self.use_trackpy
            
            #print(f"DEBUG: Tracking parameters - search_range: {search_range}, memory: {memory}, use_trackpy: {use_trackpy}")
            
            if not use_trackpy:
                #print("DEBUG: trackpy disabled, using fallback tracking")
                self._trackParticlesFallback()
                return
            
            # Kalman Filterを使用した軌跡追跡
            if self.use_kalman_filter:
                trajectories = self._trackWithKalmanFilter(features_df, search_range, memory)
            else:
                # 通常のtrackpy軌跡追跡
                trajectories = tp.link(features_df, search_range=search_range, memory=memory)
            
            #print(f"DEBUG: Trackpy found {len(trajectories)} trajectories")
            
            # 結果を内部形式に変換
            tracks = []
            for track_id in trajectories['particle'].unique():
                track_data = trajectories[trajectories['particle'] == track_id]
                
                track = {
                    'track_id': int(track_id),
                    'particles': [],
                    'start_frame': int(track_data['frame'].min()),
                    'end_frame': int(track_data['frame'].max())
                }
                
                # 各フレームの粒子データを変換
                for _, row in track_data.iterrows():
                    particle = {
                        'frame': int(row['frame']),
                        'position': float(row['x']),
                        'intensity': float(row['intensity']),
                        'confidence': float(row['signal']),
                        'size': int(row['size'])
                    }
                    track['particles'].append(particle)
                
                tracks.append(track)
            
            self.particle_tracks = tracks
            #print(f"DEBUG: Converted to {len(tracks)} internal tracks")
            
            # 軌跡情報を表示
            #for i, track in enumerate(tracks):
                #print(f"DEBUG: Track {i}: {len(track['particles'])} particles from frame {track['start_frame']} to {track['end_frame']}")
                
        except ImportError:
            #print("DEBUG: trackpy not available, using fallback tracking")
            self._trackParticlesFallback()
        except Exception as e:
            #print(f"DEBUG: Error in trackpy tracking: {e}")
            #print("DEBUG: Using fallback tracking")
            self._trackParticlesFallback()

    def _trackWithKalmanFilter(self, features_df, search_range, memory):
        """Kalman Filterを使用した軌跡追跡"""
        try:
            import cv2
            #print("DEBUG: Starting Kalman Filter tracking")
            
            # フレームごとに粒子をグループ化
            frames = sorted(features_df['frame'].unique())
            tracks = []
            track_id = 0
            
            # 各フレームの粒子を処理
            for frame_idx in frames:
                frame_particles = features_df[features_df['frame'] == frame_idx]
                
                for _, particle in frame_particles.iterrows():
                    # 新しい軌跡を開始
                    track = {
                        'particle': track_id,
                        'frame': particle['frame'],
                        'x': particle['x'],
                        'y': particle['y'],
                        'intensity': particle['intensity'],
                        'size': particle['size'],
                        'ecc': particle['ecc'],
                        'signal': particle['signal'],
                        'raw_mass': particle['raw_mass']
                    }
                    
                    # Kalman Filterを初期化
                    kalman = cv2.KalmanFilter(4, 2)  # 状態: [x, y, vx, vy], 観測: [x, y]
                    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                       [0, 1, 0, 0]], np.float32)
                    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                      [0, 1, 0, 1],
                                                      [0, 0, 1, 0],
                                                      [0, 0, 0, 1]], np.float32)
                    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0],
                                                      [0, 0, 1, 0],
                                                      [0, 0, 0, 1]], np.float32) * 0.03
                    
                    # 初期状態を設定
                    kalman.statePre = np.array([[particle['x']], [particle['y']], [0], [0]], np.float32)
                    kalman.statePost = np.array([[particle['x']], [particle['y']], [0], [0]], np.float32)
                    
                    # 次のフレームで同じ軌跡を探す
                    current_frame = particle['frame']
                    current_pos = np.array([particle['x'], particle['y']])
                    
                    track_particles = [track]
                    
                    for next_frame in range(current_frame + 1, min(current_frame + memory + 1, max(frames) + 1)):
                        if next_frame not in features_df['frame'].unique():
                            break
                        
                        # 予測
                        prediction = kalman.predict()
                        predicted_pos = prediction[:2].flatten()
                        
                        # 次のフレームの粒子を取得
                        next_particles = features_df[features_df['frame'] == next_frame]
                        
                        # 最も近い粒子を探す
                        nearest_particle = None
                        min_distance = float('inf')
                        
                        for _, next_p in next_particles.iterrows():
                            next_pos = np.array([next_p['x'], next_p['y']])
                            distance = np.linalg.norm(next_pos - predicted_pos)
                            
                            # より厳密な条件
                            if (distance < min_distance and 
                                distance <= search_range and
                                distance > 0):
                                
                                min_distance = distance
                                nearest_particle = next_p
                        
                        if nearest_particle is not None:
                            # 観測値を更新
                            measurement = np.array([[nearest_particle['x']], [nearest_particle['y']]], np.float32)
                            kalman.correct(measurement)
                            
                            # 軌跡に追加
                            next_track = {
                                'particle': track_id,
                                'frame': nearest_particle['frame'],
                                'x': nearest_particle['x'],
                                'y': nearest_particle['y'],
                                'intensity': nearest_particle['intensity'],
                                'size': nearest_particle['size'],
                                'ecc': nearest_particle['ecc'],
                                'signal': nearest_particle['signal'],
                                'raw_mass': nearest_particle['raw_mass']
                            }
                            track_particles.append(next_track)
                        else:
                            # 予測位置を使用して軌跡を継続
                            next_track = {
                                'particle': track_id,
                                'frame': next_frame,
                                'x': predicted_pos[0],
                                'y': predicted_pos[1],
                                'intensity': particle['intensity'],
                                'size': particle['size'],
                                'ecc': particle['ecc'],
                                'signal': particle['signal'] * 0.8,  # 信頼度を下げる
                                'raw_mass': particle['raw_mass']
                            }
                            track_particles.append(next_track)
                    
                    tracks.extend(track_particles)
                    track_id += 1
            
            # DataFrameに変換
            import pandas as pd
            trajectories = pd.DataFrame(tracks)
            
            #print(f"DEBUG: Kalman Filter found {len(trajectories)} trajectory points")
            return trajectories
            
        except Exception as e:
            #print(f"DEBUG: Error in Kalman Filter tracking: {e}")
            # フォールバック: 通常のtrackpyを使用
            import trackpy as tp
            return tp.link(features_df, search_range=search_range, memory=memory)

    def _trackParticlesFallback(self):
        """trackpyが利用できない場合のフォールバック追跡"""
        if self.particle_detection_results is None or len(self.particle_detection_results) == 0:
            #print("DEBUG: No particles to track")
            return
        
        #print("DEBUG: Starting fallback particle tracking")
        
        # フレームごとに粒子をグループ化
        particles_by_frame = {}
        for particle in self.particle_detection_results:
            frame = particle['frame']
            if frame not in particles_by_frame:
                particles_by_frame[frame] = []
            particles_by_frame[frame].append(particle)
        
        # 軌跡追跡アルゴリズム（簡易版）
        tracks = []
        used_particles = set()
        
        for frame in sorted(particles_by_frame.keys()):
            for particle in particles_by_frame[frame]:
                particle_id = (particle['frame'], particle['position'])
                if particle_id in used_particles:
                    continue
                
                # 新しい軌跡を開始
                track = {
                    'track_id': len(tracks),
                    'particles': [particle],
                    'start_frame': particle['frame'],
                    'end_frame': particle['frame']
                }
                used_particles.add(particle_id)
                
                # 次のフレームで同じ軌跡を探す
                current_frame = particle['frame']
                current_pos = particle['position']
                
                # より厳密な軌跡追跡
                for next_frame in range(current_frame + 1, current_frame + self.tracking_time_threshold + 1):
                    if next_frame not in particles_by_frame:
                        break
                    
                    # 最も近い粒子を探す（より厳密な条件）
                    nearest_particle = None
                    min_distance = float('inf')
                    
                    for next_particle in particles_by_frame[next_frame]:
                        next_particle_id = (next_particle['frame'], next_particle['position'])
                        if next_particle_id in used_particles:
                            continue
                        
                        # 距離と移動方向を考慮
                        distance = abs(next_particle['position'] - current_pos)
                        
                        # より厳密な条件：距離が閾値以内で、移動方向が一貫している
                        if (distance < min_distance and 
                            distance <= self.tracking_distance_threshold and
                            distance > 0):  # 同じ位置の粒子は除外
                            
                            min_distance = distance
                            nearest_particle = next_particle
                    
                    if nearest_particle is not None:
                        # 軌跡に追加
                        track['particles'].append(nearest_particle)
                        track['end_frame'] = nearest_particle['frame']
                        used_particles.add((nearest_particle['frame'], nearest_particle['position']))
                        current_pos = nearest_particle['position']
                    else:
                        # 次のフレームで見つからない場合は軌跡を終了
                        break
                
                tracks.append(track)
        
        self.particle_tracks = tracks
        #print(f"DEBUG: Fallback tracked {len(tracks)} particle trajectories")
        
        # 軌跡情報を表示
        #for i, track in enumerate(tracks):
            #print(f"DEBUG: Track {i}: {len(track['particles'])} particles from frame {track['start_frame']} to {track['end_frame']}")

    def getTrackDataForGraph(self):
        """グラフ表示用の軌跡データを準備（選択された始点からの絶対距離）"""
        if self.particle_tracks is None:
            return None
        
        track_data = []
        
        # 選択された始点に基づいて参照位置と方向を決定
        reference_position = self._getReferencePosition()
        direction_multiplier = self._getDirectionMultiplier()
        #print(f"DEBUG: getTrackDataForGraph - Reference position: {reference_position}, Direction: {direction_multiplier}")
        
        for track in self.particle_tracks:
            track_info = {
                'track_id': track['track_id'],
                'time_points': [],
                'positions': [],
                'intensities': [],
                'confidences': []
            }
            
            for particle in track['particles']:
                # 時間を秒に変換
                if hasattr(gv, 'FrameTime'):
                    frame_time = gv.FrameTime / 1000.0  # msを秒に変換
                    time_sec = particle['frame'] * frame_time
                else:
                    time_sec = particle['frame']  # フレーム番号をそのまま使用
                
                # 位置を選択された始点からの絶対距離に変換
                absolute_position = (particle['position'] - reference_position) * direction_multiplier
                
                track_info['time_points'].append(time_sec)
                track_info['positions'].append(absolute_position)
                track_info['intensities'].append(particle['intensity'])
                track_info['confidences'].append(particle['confidence'])
            
            track_data.append(track_info)
        
        return track_data
        
    def _getReferencePosition(self):
        """選択された始点に基づいて参照位置を取得"""
        if not hasattr(self, 'selected_start_point'):
            self.selected_start_point = "Blue"
        
        # 直線の色に基づいて参照位置を決定
        if self.selected_start_point == "Blue":
            # 青線の位置を参照（通常は左端）
            if hasattr(self, 'line_data') and self.line_data:
                # 直線の開始点（青線）を参照
                return self.line_data[0]['x']  # 青線のx座標
            elif hasattr(self, 'roi_line') and self.roi_line is not None:
                # Draw Lineが存在する場合は、その始点を使用
                return self.roi_line['start_x']
            elif hasattr(self, 'reference_line_points') and len(self.reference_line_points) >= 2:
                # 参照線が存在する場合は、その始点を使用
                start_point = self.reference_line_points[0]
                return start_point[0]
            else:
                # デフォルト値
                return 0
        elif self.selected_start_point == "Red":
            # 赤線の位置を参照（通常は右端）
            if hasattr(self, 'line_data') and self.line_data and len(self.line_data) > 1:
                # 直線の終了点（赤線）を参照
                return self.line_data[-1]['x']  # 赤線のx座標
            elif hasattr(self, 'roi_line') and self.roi_line is not None:
                # Draw Lineが存在する場合は、その終点を使用
                return self.roi_line['end_x']
            elif hasattr(self, 'reference_line_points') and len(self.reference_line_points) >= 2:
                # 参照線が存在する場合は、その終点を使用
                end_point = self.reference_line_points[-1]
                return end_point[0]
            else:
                # デフォルト値
                return 0
        else:
            # デフォルトは青線
            return 0
            
    def _getDirectionMultiplier(self):
        """選択された始点に基づいて方向の乗数を取得"""
        if not hasattr(self, 'selected_start_point'):
            self.selected_start_point = "Blue"
        
        # 始点の選択に基づいて方向を決定
        if self.selected_start_point == "Blue":
            # Blue選択：Blue->Redの向きが正（右方向を正とする）
            return 1
        elif self.selected_start_point == "Red":
            # Red選択：Red->Blueの向きが正（左方向を正とする）
            return -1
        else:
            # デフォルトは青線
            return 1

    def detectParticlesLinear(self):
        """直線モードでの粒子検出（tracker.pyのLocal Max手法を使用）"""
        try:
            #print("DEBUG: detectParticlesLinear called")
            
            # 必要なライブラリのインポート
            if not self._require_skimage("Particle detection (Linear)"):
                return None
            try:
                from scipy.ndimage import gaussian_filter
                from skimage.feature import peak_local_max
                from skimage.filters import threshold_otsu
            except ImportError:
                self._require_skimage("Particle detection (Linear)")
                return None
            
            # パラメータを取得
            sigma = self.gradient_sigma_spin.value()
            min_distance = self.min_peak_distance_spin.value()
            threshold_factor = self.threshold_factor_spin.value()
            
            #print(f"DEBUG: Local Max parameters - sigma: {sigma}, min_distance: {min_distance}, threshold_factor: {threshold_factor}")
            
            particle_positions = []
            total_frames = self.kymograph_data.shape[0]
            
            for frame_idx in range(total_frames):
                # フレームデータを取得
                frame_data = self.kymograph_data[frame_idx, :]
                
                # データの正規化（負の値を含む場合）
                processed_data = frame_data.copy()
                original_min = np.min(processed_data)
                original_max = np.max(processed_data)
                
                if original_min < 0:
                    # 負の値を0にシフト
                    processed_data = processed_data - original_min
                
                # ガウシアンフィルターで平滑化（tracker.pyと同じ手法）
                smoothed_data = gaussian_filter(processed_data, sigma=sigma)
                
                # Otsu法で閾値を計算（tracker.pyと同じ手法）
                otsu_threshold = threshold_otsu(smoothed_data)
                final_threshold = otsu_threshold * threshold_factor
                
                # ピーク検出（tracker.pyと同じ手法）
                coordinates = peak_local_max(smoothed_data, 
                                          min_distance=min_distance,
                                          threshold_abs=final_threshold,
                                          exclude_border=True)
                
                # 品質チェック（強度比フィルタリング）
                if len(coordinates) > 0:
                    valid_coordinates = []
                    for coord in coordinates:
                        y = coord[0]  # 1次元データなのでy座標のみ
                        # 元の画像での強度をチェック
                        original_intensity = processed_data[y]
                        mean_intensity = np.mean(processed_data)
                        
                        # 平均値の1.0倍以上を要求（tracker.pyと同じ）
                        if original_intensity > mean_intensity * 1.0:
                            valid_coordinates.append(coord)
                    
                    coordinates = np.array(valid_coordinates) if valid_coordinates else np.empty((0, 1), dtype=int)
                
                # 検出されたすべてのピークを保存
                if len(coordinates) > 0:
                    for coord in coordinates:
                        y = coord[0]  # 1次元データなのでy座標のみ
                        intensity = smoothed_data[y]
                        
                        # 信頼度を計算（強度に基づく）
                        max_intensity = np.max(smoothed_data)
                        confidence = intensity / max_intensity if max_intensity > 0 else 0.0
                        
                        particle_positions.append({
                            'frame': frame_idx,
                            'position': y,
                            'intensity': intensity,
                            'confidence': confidence,
                            'size': 1  # Local Maxではサイズは1
                        })
                else:
                    # ピークが見つからない場合は中央位置を使用
                    particle_positions.append({
                        'frame': frame_idx,
                        'position': len(frame_data) // 2,
                        'intensity': np.max(smoothed_data),
                        'confidence': 0.0,
                        'size': 0
                    })
            
            #print(f"DEBUG: Total particles detected with Local Max: {len(particle_positions)}")
            return particle_positions
            
        except Exception as e:
            #print(f"DEBUG: Error in detectParticlesLinear: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detectParticlesRectangular(self):
        """矩形モードでの粒子検出（tracker.pyのLocal Max手法を使用）"""
        try:
            #print("DEBUG: detectParticlesRectangular called")
            
            if not self._require_skimage("Particle detection (Rectangular)"):
                return None
            try:
                from scipy.ndimage import gaussian_filter
                from skimage.feature import peak_local_max
                from skimage.filters import threshold_otsu
            except ImportError:
                self._require_skimage("Particle detection (Rectangular)")
                return None
            
            # パラメータを取得
            sigma = self.gradient_sigma_spin.value()
            min_distance = self.min_peak_distance_spin.value()
            threshold_factor = self.threshold_factor_spin.value()
            
            #print(f"DEBUG: Local Max parameters - sigma: {sigma}, min_distance: {min_distance}, threshold_factor: {threshold_factor}")
            
            particle_positions = []
            total_frames = len(self.image_stack)
            frame_height = int(self.actual_frame_height)
            
            for frame_idx in range(total_frames):
                # フレームデータを抽出
                start_row = frame_idx * frame_height
                end_row = start_row + frame_height
                
                if end_row > self.kymograph_data.shape[0]:
                    break
                
                frame_data = self.kymograph_data[start_row:end_row, :]
                
                # データの正規化（負の値を含む場合）
                processed_data = frame_data.copy()
                original_min = np.min(processed_data)
                original_max = np.max(processed_data)
                
                if original_min < 0:
                    # 負の値を0にシフト
                    processed_data = processed_data - original_min
                
                # ガウシアンフィルターで平滑化（tracker.pyと同じ手法）
                smoothed_data = gaussian_filter(processed_data, sigma=sigma)
                
                # Otsu法で閾値を計算（tracker.pyと同じ手法）
                otsu_threshold = threshold_otsu(smoothed_data)
                final_threshold = otsu_threshold * threshold_factor
                
                # ピーク検出（tracker.pyと同じ手法）
                coordinates = peak_local_max(smoothed_data, 
                                          min_distance=min_distance,
                                          threshold_abs=final_threshold,
                                          exclude_border=True)
                
                # 品質チェック（強度比フィルタリング）
                if len(coordinates) > 0:
                    valid_coordinates = []
                    for coord in coordinates:
                        y, x = coord
                        # 元の画像での強度をチェック
                        original_intensity = processed_data[y, x]
                        mean_intensity = np.mean(processed_data)
                        
                        # 平均値の1.0倍以上を要求（tracker.pyと同じ）
                        if original_intensity > mean_intensity * 1.0:
                            valid_coordinates.append(coord)
                    
                    coordinates = np.array(valid_coordinates) if valid_coordinates else np.empty((0, 2), dtype=int)
                
                # 検出されたすべてのピークを保存
                if len(coordinates) > 0:
                    for coord in coordinates:
                        y, x = coord
                        intensity = smoothed_data[y, x]
                        
                        # 信頼度を計算（強度に基づく）
                        max_intensity = np.max(smoothed_data)
                        confidence = intensity / max_intensity if max_intensity > 0 else 0.0
                        
                        particle_positions.append({
                            'frame': frame_idx,
                            'position': x,  # x座標
                            'intensity': intensity,
                            'confidence': confidence,
                            'size': 1  # Local Maxではサイズは1
                        })
                else:
                    # ピークが見つからない場合は中央位置を使用
                    particle_positions.append({
                        'frame': frame_idx,
                        'position': frame_data.shape[1] // 2,
                        'intensity': np.max(smoothed_data),
                        'confidence': 0.0,
                        'size': 0
                    })
            
            print(f"DEBUG: Total particles detected with Local Max: {len(particle_positions)}")
            return particle_positions
            
        except Exception as e:
            print(f"DEBUG: Error in detectParticlesRectangular: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detectParticleNearReferencePoint(self, reference_point, frame_idx):
        """参照点付近の粒子検出（Semi Auto mode）"""
        try:
            #print(f"DEBUG: detectParticleNearReferencePoint called for frame {frame_idx}")
            
            if peak_local_max is None or threshold_otsu is None:
                self._require_skimage("Particle detection (Semi Auto)")
                return None, 0.0
            
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            if is_rectangular_mode:
                # 矩形モードでの検出
                frame_height = int(self.actual_frame_height)
                start_row = frame_idx * frame_height
                end_row = start_row + frame_height
                
                if end_row > self.kymograph_data.shape[0]:
                    return reference_point, 0.0
                
                frame_data = self.kymograph_data[start_row:end_row, :]
            else:
                # 直線モードでの検出
                frame_data = self.kymograph_data[frame_idx, :].reshape(1, -1)
            
            # 検索範囲を設定
            search_radius = self.search_radius
            ref_x = int(reference_point)
            
            search_start = max(0, ref_x - search_radius)
            search_end = min(frame_data.shape[1], ref_x + search_radius + 1)
            
            if search_start >= search_end:
                return reference_point, 0.0
            
            # 検索領域を抽出
            search_region = frame_data[:, search_start:search_end]
            
            # データを平滑化
            smoothed_region = gaussian_filter(search_region, sigma=1.0)
            
            # 適応的閾値を計算
            otsu_threshold = threshold_otsu(smoothed_region)
            
            # ピーク検出
            coordinates = peak_local_max(smoothed_region, 
                                      min_distance=2,
                                      threshold_abs=otsu_threshold,
                                      exclude_border=False)
            
            if len(coordinates) > 0:
                # 最大強度のピークを選択
                peak_intensities = [smoothed_region[coord[0], coord[1]] for coord in coordinates]
                max_idx = np.argmax(peak_intensities)
                best_peak = coordinates[max_idx]
                
                # グローバル座標に変換
                global_x = search_start + best_peak[1]
                
                # 信頼度を計算
                confidence = min(1.0, peak_intensities[max_idx] / (np.max(smoothed_region) + 1e-10))
                
                return global_x, confidence
            else:
                # ピークが見つからない場合は参照点を使用
                return reference_point, 0.0
                
        except Exception as e:
            #print(f"DEBUG: Error in detectParticleNearReferencePoint: {e}")
            return reference_point, 0.0

    def detectParticlesAlongReferenceLine(self, total_frames):
        """参照線に沿った粒子検出（Semi Auto mode）"""
        try:
            #print("DEBUG: detectParticlesAlongReferenceLine called")
            
            if not self.reference_line_complete or len(self.reference_line_points) < 2:
                #print("DEBUG: Reference line not complete")
                return None, None
            
            # 参照線から各フレームでの予想位置を計算
            predicted_positions = self.interpolateReferenceLine()
            
            particle_positions = []
            confidence_scores = []
            
            for frame_idx in range(total_frames):
                # このフレームでの予想位置
                predicted_x = predicted_positions[frame_idx]
                
                # 参照点付近で粒子検出
                detected_pos, confidence = self.detectParticleNearReferencePoint(predicted_x, frame_idx)
                
                if detected_pos is not None:
                    particle_positions.append(detected_pos)
                    confidence_scores.append(confidence)
                else:
                    particle_positions.append(predicted_x)
                    confidence_scores.append(0.0)
                
                #print(f"DEBUG: Frame {frame_idx} - predicted: {predicted_x:.1f}, detected: {detected_pos:.1f}, confidence: {confidence:.3f}")
            
            return np.array(particle_positions), np.array(confidence_scores)
            
        except Exception as e:
            #print(f"DEBUG: Error in detectParticlesAlongReferenceLine: {e}")
            return None, None

    def detectParticlesAlongParticleLine(self, line_data):
        """粒子線に沿った粒子検出"""
        try:

            
            if not line_data['points'] or len(line_data['points']) < 2:
                return []
            
            # 粒子線の点を補間して各フレームでの位置を取得
            total_frames = len(self.image_stack)
            interpolated_positions = self.interpolateParticleLine(line_data['points'], total_frames)
            
            particles = []
            
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            # 有効なフレームのみで検出実行
            valid_frame_count = 0
            for frame_idx in range(total_frames):
                predicted_x = interpolated_positions[frame_idx]
                
                # Noneの場合はスキップ（線の範囲外）
                if predicted_x is None:
                    continue
                
                valid_frame_count += 1
                
                # 参照点付近で粒子検出
                detected_particles = self.detectParticlesNearLinePoint(predicted_x, frame_idx, line_data['line_id'])
                
                if detected_particles:
                    particles.extend(detected_particles)
                

            return particles
            
        except Exception as e:
    
            return []

    def interpolateParticleLine(self, line_points, total_frames):
        """粒子線の点を補間して各フレームでの位置を取得（修正版）"""
        try:
            if len(line_points) < 2:
                return [None] * total_frames  # Noneを返して範囲外を明示
            
            # 点の座標を抽出
            x_coords = [p[0] for p in line_points]
            y_coords = [p[1] for p in line_points]
            
            # フレーム座標に変換（より精密な計算）
            frame_coords = []
            for y in y_coords:
                if hasattr(self, 'actual_frame_height') and self.actual_frame_height is not None:
                    # 矩形モード：フレームの中心を基準に計算
                    frame_height = float(self.actual_frame_height)
                    frame_idx = y / frame_height  # 小数点も保持
                    frame_coords.append(frame_idx)
                else:
                    # 直線モード
                    frame_coords.append(float(y))
            
            # フレーム座標をソート（時間順に並べる）
            sorted_indices = np.argsort(frame_coords)
            sorted_frame_coords = [frame_coords[i] for i in sorted_indices]
            sorted_x_coords = [x_coords[i] for i in sorted_indices]
            
            # 線の有効範囲を取得
            min_frame = min(sorted_frame_coords)
            max_frame = max(sorted_frame_coords)
            

            
            # 各フレームで補間（範囲内のみ）
            interpolated_positions = []
            for frame_idx in range(total_frames):
                if min_frame <= frame_idx <= max_frame:
                    # 範囲内：補間実行
                    interpolated_x = np.interp(frame_idx, sorted_frame_coords, sorted_x_coords)
                    interpolated_positions.append(interpolated_x)
                else:
                    # 範囲外：Noneを設定（検出スキップ）
                    interpolated_positions.append(None)
            
            return interpolated_positions
            
        except Exception as e:
    
            return [None] * total_frames

    def detectParticlesNearLinePoint(self, reference_x, frame_idx, line_id):
        """線上の点付近の粒子検出（改良版）"""
        try:
            # 線の範囲外の場合は検出しない
            if reference_x is None:
                return []
            if not self._require_skimage("Particle detection (Near line point)"):
                return []
            
            # パラメータを取得
            min_distance = self.min_peak_distance_spin.value() if hasattr(self, 'min_peak_distance_spin') else 5
            threshold_factor = self.threshold_factor_spin.value() if hasattr(self, 'threshold_factor_spin') else 0.8
            sigma = self.gradient_sigma_spin.value() if hasattr(self, 'gradient_sigma_spin') else 0.5
            
            # 検索範囲をMin Distanceパラメータに基づいて設定
            search_radius = min_distance  # Min Distanceを検索半径として使用
            ref_x = int(reference_x)
            

            
            # 矩形モードかどうかを判定
            is_rectangular_mode = (hasattr(self, 'actual_frame_height') and 
                                 self.actual_frame_height is not None and 
                                 self.actual_frame_height > 1)
            
            if is_rectangular_mode:
                # 矩形モードでの検出
                frame_height = int(self.actual_frame_height)
                start_row = frame_idx * frame_height
                end_row = start_row + frame_height
                
                if end_row > self.kymograph_data.shape[0]:
    
                    return []
                
                frame_data = self.kymograph_data[start_row:end_row, :]
                
                # 検索範囲を設定
                search_start_x = max(0, ref_x - search_radius)
                search_end_x = min(frame_data.shape[1], ref_x + search_radius + 1)
                search_start_y = 0
                search_end_y = frame_height
                

                
            else:
                # 1次元線モードでの検出
                frame_data = self.kymograph_data[frame_idx, :].reshape(1, -1)
                
                # 水平方向のみの検索範囲
                search_start_x = max(0, ref_x - search_radius)
                search_end_x = min(frame_data.shape[1], ref_x + search_radius + 1)
                search_start_y = 0
                search_end_y = 1
            
            if search_start_x >= search_end_x:
                return []
            
            # 検索領域を抽出
            search_region = frame_data[search_start_y:search_end_y, search_start_x:search_end_x]
            
            # データを平滑化
            from scipy.ndimage import gaussian_filter
            try:
                from skimage.filters import threshold_otsu
            except ImportError:
                self._require_skimage("Particle detection (Near line point)")
                return []
            
            smoothed_region = gaussian_filter(search_region, sigma=sigma)
            
            # 適応的閾値を計算
            otsu_threshold = threshold_otsu(smoothed_region)
            detection_threshold = otsu_threshold * threshold_factor
            
            # 最大値の位置を検索
            max_pos = None
            max_value = -float('inf')
            
            for y in range(search_region.shape[0]):
                for x in range(search_region.shape[1]):
                    if smoothed_region[y, x] > max_value:
                        max_value = smoothed_region[y, x]
                        max_pos = (y, x)
            
            particles = []
            if max_pos is not None and max_value >= detection_threshold:
                # 絶対座標に変換
                abs_x = search_start_x + max_pos[1]
                
                # 線からの距離を計算
                distance_from_line = abs(abs_x - ref_x)
                
                # 線からsearch_radius以内の粒子を検出
                if distance_from_line <= search_radius:
                    # 粒子データを作成
                    particle = {
                        'frame': frame_idx,
                        'position': abs_x,
                        'intensity': max_value,
                        'confidence': 1.0,
                        'size': 1,
                        'line_id': line_id
                    }
                    
                    particles.append(particle)

            
            return particles
            
        except Exception as e:
    
            return []

    def detectLocalMaximaInRegion(self, search_region, search_start, frame_idx, line_id, ref_x):
        """領域内の最大値を検出（シンプルな方法）"""
        try:
            # 最大値の位置を検索
            max_pos = None
            max_value = -float('inf')
            
            for y in range(search_region.shape[0]):
                for x in range(search_region.shape[1]):
                    if search_region[y, x] > max_value:
                        max_value = search_region[y, x]
                        max_pos = (y, x)
            
            particles = []
            if max_pos is not None:
                # 絶対座標に変換
                abs_x = search_start + max_pos[1]
                abs_y = frame_idx
                
                # 粒子データを作成
                particle = {
                    'frame': frame_idx,
                    'position': abs_x,
                    'intensity': max_value,
                    'confidence': 1.0,
                    'size': 1,
                    'line_id': line_id
                }
                
                particles.append(particle)
                #print(f"DEBUG: Frame {frame_idx} - detected particle at position {abs_x} with intensity {max_value:.2f}")
            #else:
                #print(f"DEBUG: Frame {frame_idx} - no particle detected")
            
            return particles
            
        except Exception as e:
            #print(f"DEBUG: Error in detectLocalMaximaInRegion: {e}")
            return []

    def removePointFromCurrentLine(self, click_x, click_y):
        """描画中の線から最も近い点を削除"""
        if not hasattr(self, 'current_line_points') or not self.current_line_points:
            return False
        
        # 最も近い点を見つける
        min_distance = float('inf')
        nearest_index = -1
        
        for i, point in enumerate(self.current_line_points):
            distance = ((point[0] - click_x) ** 2 + (point[1] - click_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        # 閾値内の点を削除（10ピクセル以内）
        if min_distance <= 10 and nearest_index >= 0:
            removed_point = self.current_line_points.pop(nearest_index)
            #print(f"DEBUG: Removed point {nearest_index} from current line: {removed_point}")
            
            # キャンバスを更新
            self.displayKymograph()
            return True
        
        return False

    def removePointFromParticleLines(self, click_x, click_y):
        """完成した粒子線から最も近い点を削除"""
        if not hasattr(self, 'particle_lines') or not self.particle_lines:
            return False
        
        # 最も近い点を見つける
        min_distance = float('inf')
        nearest_line_index = -1
        nearest_point_index = -1
        
        for line_idx, line_data in enumerate(self.particle_lines):
            if 'points' in line_data:
                for point_idx, point in enumerate(line_data['points']):
                    distance = ((point[0] - click_x) ** 2 + (point[1] - click_y) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        nearest_line_index = line_idx
                        nearest_point_index = point_idx
        
        # 閾値内の点を削除（10ピクセル以内）
        if min_distance <= 10 and nearest_line_index >= 0 and nearest_point_index >= 0:
            removed_point = self.particle_lines[nearest_line_index]['points'].pop(nearest_point_index)
            #print(f"DEBUG: Removed point {nearest_point_index} from line {nearest_line_index}: {removed_point}")
            
            # 線が空になった場合は削除
            if not self.particle_lines[nearest_line_index]['points']:
                self.particle_lines.pop(nearest_line_index)
                #print(f"DEBUG: Removed empty line {nearest_line_index}")
            
            # キャンバスを更新
            self.displayKymograph()
            return True
        
        return False

# エクスポートするクラスを明示的に指定
__all__ = ['KymographWindow'] 

class KymographHistogramWindow(QtWidgets.QWidget):
    """Kymograph用ヒストグラムウィンドウ"""
    contrastChanged = QtCore.pyqtSignal(float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kymograph Histogram - Manual Contrast")
        self.setMinimumSize(300, 250)  # 最小サイズを小さく
        self.resize(400, 300)  # デフォルトサイズを小さく
        self.setWindowFlags(QtCore.Qt.Window)
        
        # Windows固有の設定
        #import sys
        #if sys.platform.startswith('win'):
            # Windowsでの安定性向上のための設定
        #    self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        #    self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        
        # ウィンドウ管理システムに登録
        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self, "sub")
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARNING] Failed to register KymographHistogramWindow: {e}")
        
        self.data_min = 0.0
        self.data_max = 1.0
        self.min_value = 0.0
        self.max_value = 1.0
        
        self.setupUI()
        
        self.min_line = None
        self.max_line = None
        self._dragging_vline = False
        
        # 親ウィンドウへの参照を保存
        self.parent_window = parent
        
    def closeEvent(self, event):
        """ウィンドウが閉じられる時の処理"""
        try:
            # ウィンドウ管理システムから登録を削除
            try:
                from window_manager import unregister_pyNuD_window
                unregister_pyNuD_window(self)
            except ImportError:
                pass
            except Exception as e:
                print(f"[WARNING] Failed to unregister KymographHistogramWindow: {e}")
            
            # Qtのデフォルトのクローズ処理
            try:
                super().closeEvent(event)
            except RuntimeError:
                print("[WARNING] C++ object already deleted during super().closeEvent()")
            except Exception as e:
                print(f"[WARNING] Failed to call super().closeEvent(): {e}")
            
            event.accept()
            
        except Exception as e:
            print(f"[ERROR] Unexpected error in KymographHistogramWindow closeEvent: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生してもイベントは受け入れる
            event.accept()
    
    def setupUI(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # キャンバスとツールバー
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))  # サイズを小さく
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QtCore.QSize(12, 12))
        
        # HOMEボタンのイベントを接続
        for action in self.toolbar.actions():
            if action.text() == "Home":
                action.triggered.connect(self.onHomeButtonClicked)
                action.setToolTip("Reset contrast range to data min/max values")
                break
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # コントロールパネル
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QGridLayout(control_widget)
        control_layout.setContentsMargins(2, 2, 2, 2)  # マージンをさらに小さく
        control_layout.setSpacing(2)  # 間隔をさらに小さく
        
        # Min/Max スピンボックス
        min_label = QtWidgets.QLabel("Min:")
        min_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: red;")
        min_label.setMinimumWidth(30)  # 適切な最小幅に戻す
        min_label.setMaximumWidth(35)  # 最大幅も適切に設定
        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setRange(-999999.0, 999999.0)
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setValue(0.0)
        self.min_spinbox.setFixedWidth(70)  # スピンボックスの幅をさらに小さく
        self.min_spinbox.valueChanged.connect(self.onMinSpinboxChanged)
        
        max_label = QtWidgets.QLabel("Max:")
        max_label.setStyleSheet("font-weight: bold; font-size: 10pt; color: blue;")
        max_label.setMinimumWidth(30)  # 適切な最小幅に戻す
        max_label.setMaximumWidth(35)  # 最大幅も適切に設定
        self.max_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_spinbox.setRange(-999999.0, 999999.0)
        self.max_spinbox.setDecimals(3)
        self.max_spinbox.setValue(1.0)
        self.max_spinbox.setFixedWidth(70)  # スピンボックスの幅をさらに小さく
        self.max_spinbox.valueChanged.connect(self.onMaxSpinboxChanged)
        
        # 説明ラベル
        self.instruction_label = QtWidgets.QLabel("Drag lines or enter values. HOME button resets to data range.")
        self.instruction_label.setStyleSheet("color: green; font-size: 9pt;")
        
        # 水平レイアウトでMin/Maxを配置
        min_layout = QtWidgets.QHBoxLayout()
        min_layout.setContentsMargins(0, 0, 0, 0)
        min_layout.setSpacing(5)  # 適切な間隔に調整
        min_layout.addWidget(min_label)
        min_layout.addWidget(self.min_spinbox)
        min_layout.addStretch()  # 右側に伸縮スペース
        
        max_layout = QtWidgets.QHBoxLayout()
        max_layout.setContentsMargins(0, 0, 0, 0)
        max_layout.setSpacing(5)  # 適切な間隔に調整
        max_layout.addWidget(max_label)
        max_layout.addWidget(self.max_spinbox)
        max_layout.addStretch()  # 右側に伸縮スペース
        
        control_layout.addLayout(min_layout, 0, 0)
        control_layout.addLayout(max_layout, 1, 0)
        control_layout.addWidget(self.instruction_label, 2, 0)
        
        layout.addWidget(control_widget)
        
        # ヒストグラム用のaxes
        self.axes = self.canvas.figure.subplots()
        self.axes.set_xlabel("Intensity")
        self.axes.set_ylabel("Frequency")
        
    def updateHistogramData(self, data, bins=100):
        """ヒストグラムデータを更新"""
        self.data = data
        self.data_min = np.min(data)
        self.data_max = np.max(data)
        
        # 初期値をデータ範囲に設定
        self.min_value = self.data_min
        self.max_value = self.data_max
        
        # スピンボックスの範囲を更新
        self.min_spinbox.setRange(self.data_min, self.data_max)
        self.max_spinbox.setRange(self.data_min, self.data_max)
        self.min_spinbox.setValue(self.min_value)
        self.max_spinbox.setValue(self.max_value)
        
        # ヒストグラムを計算
        self.hist, self.bin_edges = np.histogram(data, bins=bins)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # ヒストグラムを描画
        self.updateHistogram()
        
    def updateHistogram(self):
        """ヒストグラムを描画"""
        self.axes.clear()
        
        # ヒストグラムを描画
        self.axes.bar(self.bin_centers, self.hist, width=self.bin_edges[1] - self.bin_edges[0], 
                      alpha=0.7, color='gray')
        
        # ドラッグ可能な線を描画
        if self.min_line is None:
            self.min_line = self.axes.axvline(self.min_value, color='red', linewidth=3, 
                                             picker=True, pickradius=15, alpha=0.8, zorder=10)
        else:
            self.min_line.set_xdata([self.min_value, self.min_value])
        
        if self.max_line is None:
            self.max_line = self.axes.axvline(self.max_value, color='blue', linewidth=3, 
                                             picker=True, pickradius=15, alpha=0.8, zorder=10)
        else:
            self.max_line.set_xdata([self.max_value, self.max_value])
        
        # イベントを接続
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.axes.set_xlabel("Intensity")
        self.axes.set_ylabel("Frequency")
        self.axes.set_title("Kymograph Intensity Histogram")
        
        self.canvas.draw()
        
    def on_press(self, event):
        """マウスプレスイベント"""
        if event.inaxes != self.axes:
            return
            
        if event.button == 1:  # 左クリック
            # 線の近くをクリックしたかチェック
            if self.min_line and self.min_line.contains(event)[0]:
                self._dragging_vline = 'min'
            elif self.max_line and self.max_line.contains(event)[0]:
                self._dragging_vline = 'max'
                
    def on_motion(self, event):
        """マウス移動イベント"""
        if event.inaxes != self.axes or not self._dragging_vline:
            return
        if event.xdata is None:
            return
        # 線をドラッグ
        if self._dragging_vline == 'min':
            new_value = max(self.data_min, min(event.xdata, self.max_value))
            self.min_value = new_value
            self.min_spinbox.setValue(new_value)
            self.min_line.set_xdata([new_value, new_value])
            # ドラッグ中はkymograph更新を停止（パフォーマンス向上）
        elif self._dragging_vline == 'max':
            new_value = min(self.data_max, max(event.xdata, self.min_value))
            self.max_value = new_value
            self.max_spinbox.setValue(new_value)
            self.max_line.set_xdata([new_value, new_value])
            # ドラッグ中はkymograph更新を停止（パフォーマンス向上）
        self.canvas.draw()
        
    def on_release(self, event):
        """マウスリリースイベント"""
        if self._dragging_vline:
            # ドラッグ終了時にkymographを更新
            self.contrastChanged.emit(self.min_value, self.max_value)
        self._dragging_vline = False
        
    def onMinSpinboxChanged(self, value):
        """Minスピンボックス変更"""
        if value >= self.max_value:
            value = self.max_value - (self.max_value - self.data_min) * 0.01
            self.min_spinbox.setValue(value)
        self.min_value = value
        if self.min_line:
            self.min_line.set_xdata([value, value])
        # スピンボックス変更時は即座にkymograph更新
        self.contrastChanged.emit(self.min_value, self.max_value)
        self.canvas.draw()
        
    def onMaxSpinboxChanged(self, value):
        """Maxスピンボックス変更"""
        if value <= self.min_value:
            value = self.min_value + (self.data_max - self.min_value) * 0.01
            self.max_spinbox.setValue(value)
        self.max_value = value
        if self.max_line:
            self.max_line.set_xdata([value, value])
        # スピンボックス変更時は即座にkymograph更新
        self.contrastChanged.emit(self.min_value, self.max_value)
        self.canvas.draw()
        
    def onHomeButtonClicked(self):
        """HOMEボタンクリック"""
        self.min_value = self.data_min
        self.max_value = self.data_max
        self.min_spinbox.setValue(self.min_value)
        self.max_spinbox.setValue(self.max_value)
        self.contrastChanged.emit(self.min_value, self.max_value)
        self.updateHistogram()
        
    def closeEvent(self, event):
        """ウィンドウが閉じられた時の処理"""
        # 親ウィンドウの参照をクリア
        if hasattr(self.parent_window, 'histogram_window'):
            self.parent_window.histogram_window = None
        event.accept()


def create_plugin(main_window):
    """プラグインエントリポイント。pyNuD の Plugin メニューから呼ばれる。"""
    return KymographWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "KymographWindow"]
