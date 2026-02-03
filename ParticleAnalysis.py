#!/usr/bin/env python3
"""
Particle Analysis Module for pyNuD
pyNuD用粒子解析モジュール

AFM画像から粒子を検出し、そのサイズ、形状、分布などの解析を行う機能を提供します。
Provides particle detection and analysis functionality for AFM images.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import os
import sys
import json
from scipy import ndimage
from skimage import measure, morphology, filters
try:
    from skimage.feature import peak_local_max
except ImportError:
    from skimage.feature import peak_local_maxima as peak_local_max

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

# プラグイン内ヘルプ用 HTML（EN/JA）
HELP_HTML_EN = """
    <h1>Particle Analysis</h1>
    <h2>Overview</h2>
    <p>Particle Analysis is a powerful tool for detecting and analyzing particles in AFM images. It provides comprehensive capabilities for identifying particles, measuring their properties, and analyzing their distribution patterns.</p>
    <h2>Access</h2>
    <ul>
        <li><strong>Plugin menu:</strong> Load Plugin... → select <code>plugins/ParticleAnalysis.py</code>, then Plugin → Particle Analysis</li>
    </ul>

    <h2>Mode</h2>
    <table class="param-table">
        <tr><th>Parameter</th><th>Options</th><th>Description</th></tr>
        <tr><td>Mode</td><td>All Particles / Single Particle</td><td>All Particles: detect in full image. Single Particle: select ROI and detect in magnified region only.</td></tr>
    </table>

    <h2>Preprocessing Settings</h2>
    <h3>Background Subtraction</h3>
    <table class="param-table">
        <tr><th>Method</th><th>Parameter</th><th>Purpose</th></tr>
        <tr><td>None</td><td>-</td><td>No background subtraction</td></tr>
        <tr><td>Rolling Ball</td><td>Radius (nm): range 1–1000, default 10</td><td>Remove non-uniform background</td></tr>
        <tr><td>Polynomial Fit</td><td>-</td><td>Remove low-frequency components</td></tr>
    </table>
    <h3>Smoothing</h3>
    <table class="param-table">
        <tr><th>Method</th><th>Parameter</th><th>Purpose</th></tr>
        <tr><td>None</td><td>-</td><td>No smoothing</td></tr>
        <tr><td>Gaussian</td><td>Sigma: 0.1–3.0 (default 1.0)</td><td>Noise removal (edge preserving)</td></tr>
        <tr><td>Median</td><td>Filter size: integer (e.g. 1–9)</td><td>Spike noise removal</td></tr>
    </table>

    <h2>Particle Detection Settings</h2>
    <h3>Detection Methods</h3>
    <table class="param-table">
        <tr><th>Method</th><th>Feature</th><th>Parameters</th></tr>
        <tr><td>Threshold</td><td>Otsu / Manual / Adaptive</td><td>Manual: threshold 0.0–1.0 (or data range). When Manual is selected, use Interactive Histogram to adjust.</td></tr>
        <tr><td>Peak Detection</td><td>Watershed or Contour Level</td><td>Min Distance (px): 1–50 (default 5). Gaussian Sigma: 0.5–3.0 (default 1.5). Boundary Method: Watershed or Contour Level. Watershed threshold (%): 10–99 (default 70). Contour Level (%): 10–90 (default 30).</td></tr>
        <tr><td>Hessian Blob</td><td>Blob detection via Hessian matrix</td><td>Min Sigma: 0.1–10.0 (default 1.0). Max Sigma: 1.0–20.0 (default 5.0). Hessian threshold: 0.0–0.5 (default 0.05). Recommended: 0.05–0.25.</td></tr>
        <tr><td>Ring Detection</td><td>Ring structures via radial profile</td><td>Min Radius (nm): 1–1000 (default 5). Max Radius (nm): 1–10000 (default 100). Center Blend % Weighted: 0–100 (default 50). Inner Drop % of peak: 5–90 (default 5). Recommended for inner: 15–35%.</td></tr>
    </table>

    <h3>Size Filtering</h3>
    <ul>
        <li><strong>Min Particle Size:</strong> 1–1000 pixels (default: 10). Exclude smaller regions. Recommended: 5–50.</li>
        <li><strong>Max Particle Size:</strong> 1–10000 pixels (default: 1000). Exclude larger regions. Recommended: 100–5000.</li>
        <li><strong>Exclude particles touching image edges:</strong> Recommended: enabled. Excludes incomplete particles at image borders.</li>
    </ul>

    <h2>Measurement Parameters</h2>
    <table class="param-table">
        <tr><th>Parameter</th><th>Meaning</th><th>Unit</th></tr>
        <tr><td>Area</td><td>Particle area</td><td>nm²</td></tr>
        <tr><td>Perimeter</td><td>Particle perimeter</td><td>nm</td></tr>
        <tr><td>Circularity</td><td>Shape circularity (4π×area/perimeter²)</td><td>-</td></tr>
        <tr><td>Max Height</td><td>Maximum height of particle</td><td>nm</td></tr>
        <tr><td>Mean Height</td><td>Average height of particle</td><td>nm</td></tr>
        <tr><td>Volume</td><td>Particle volume (area × average height)</td><td>nm³</td></tr>
        <tr><td>Centroid X/Y</td><td>Particle centroid coordinates</td><td>nm</td></tr>
    </table>

    <h2>Analysis Operations</h2>
    <table class="param-table">
        <tr><th>Function</th><th>Operation</th><th>Result</th></tr>
        <tr><td>Detect</td><td>Click button</td><td>Detect particles in current frame</td></tr>
        <tr><td>All Frames</td><td>Click button</td><td>Batch analysis of all frames</td></tr>
        <tr><td>Threshold adjustment</td><td>Mouse on histogram (when Manual threshold)</td><td>Adjust threshold with real-time preview</td></tr>
        <tr><td>Export Results</td><td>Click button after detection</td><td>Save particle/ring data as CSV</td></tr>
    </table>

    <div class="feature-box">
        <h3>Ring Detection (New)</h3>
        <ul>
            <li><strong>Center estimation:</strong> Blend intensity minimum with geometric center distance (Center Blend %).</li>
            <li><strong>Inner diameter:</strong> On inverted data, detect isocontour at Inner Drop % of peak (default 5%).</li>
            <li><strong>Ring diameter:</strong> Watershed from center on inverted data to obtain mid contour (fallback from multi-directional peaks).</li>
            <li><strong>Ellipse fitting & display:</strong> Fit ellipses to inner/mid contours; dashed overlays. Results: Inner Diameter (nm), Ring Diameter (nm), Circularity (ellipse-based).</li>
            <li><strong>Row deletion:</strong> Right-click “Delete Particle” in Results to remove row and renumber (particle and ring).</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>Interactive Histogram</h3>
        <p>When Threshold method is Manual, the histogram is shown. Click and drag on the histogram to adjust the threshold with real-time preview.</p>
    </div>

    <div class="feature-box">
        <h3>Particle Properties</h3>
        <p>Area (nm²), Perimeter (nm), Circularity, Max Height (nm), Mean Height (nm), Volume (nm³). Enable/disable via “Calculate Particle Properties” and the property checkboxes.</p>
    </div>

    <h2>Analysis Workflow</h2>
    <div class="step">Step 1: Load an AFM image and open Particle Analysis (Plugin → Particle Analysis)</div>
    <div class="step">Step 2: Configure preprocessing (background subtraction, smoothing)</div>
    <div class="step">Step 3: Choose detection method (Threshold, Peak Detection, Hessian Blob, or Ring Detection)</div>
    <div class="step">Step 4: Adjust parameters (use Interactive Histogram when Manual threshold is selected)</div>
    <div class="step">Step 5: Set particle size limits and exclusion criteria</div>
    <div class="step">Step 6: Click "Detect" to analyze particles</div>
    <div class="step">Step 7: Review results and export data</div>

    <h2>Advanced Features</h2>
    <ul>
        <li><strong>All Frames:</strong> Analyze particles across all frames in time-series data.</li>
        <li><strong>Frame Navigation:</strong> Use arrow buttons to move between frames.</li>
        <li><strong>Export Results:</strong> Save particle/ring properties as CSV.</li>
        <li><strong>Show Legend:</strong> Toggle legend on the image.</li>
    </ul>

    <h2>Parameter Guidelines</h2>
    <ul>
        <li><strong>Min Particle Size:</strong> Typically 10+ pixels; increase to exclude noise.</li>
        <li><strong>Max Particle Size:</strong> Typically 1000 pixels; decrease to exclude large background.</li>
        <li><strong>Edge Exclusion:</strong> Recommended enabled for accurate analysis.</li>
    </ul>

    <div class="note">
        <strong>Pro Tips:</strong>
        <ul>
            <li>Start with Otsu and fine-tune manually if needed.</li>
            <li>Use background subtraction for images with significant background variations.</li>
            <li>Adjust smoothing to balance noise reduction and feature preservation.</li>
            <li>For time-series analysis, use consistent parameters across all frames.</li>
            <li>Export results regularly to avoid data loss.</li>
        </ul>
    </div>
"""

HELP_HTML_JA = """
    <h1>粒子解析</h1>
    <h2>概要</h2>
    <p>粒子解析は、AFM画像から粒子を検出し解析する強力なツールです。粒子の識別、特性測定、分布パターンの解析など、包括的な機能を提供します。</p>
    <h2>アクセス方法</h2>
    <ul>
        <li><strong>プラグイン:</strong> Load Plugin... → <code>plugins/ParticleAnalysis.py</code> を選択し、Plugin → Particle Analysis で開く</li>
    </ul>

    <h2>モード</h2>
    <table class="param-table">
        <tr><th>パラメータ</th><th>選択肢</th><th>説明</th></tr>
        <tr><td>Mode</td><td>All Particles / Single Particle</td><td>All Particles: 全画面で粒子検出。Single Particle: ROI を選択し拡大領域のみで検出。</td></tr>
    </table>

    <h2>前処理設定</h2>
    <h3>背景除去</h3>
    <table class="param-table">
        <tr><th>方法</th><th>パラメータ</th><th>用途</th></tr>
        <tr><td>None</td><td>-</td><td>背景除去なし</td></tr>
        <tr><td>Rolling Ball</td><td>Radius (nm): 1–1000、デフォルト 10</td><td>均一でない背景を除去</td></tr>
        <tr><td>Polynomial Fit</td><td>-</td><td>低周波成分を除去</td></tr>
    </table>
    <h3>平滑化</h3>
    <table class="param-table">
        <tr><th>方法</th><th>パラメータ</th><th>用途</th></tr>
        <tr><td>None</td><td>-</td><td>平滑化なし</td></tr>
        <tr><td>Gaussian</td><td>Sigma: 0.1–3.0（デフォルト 1.0）</td><td>ノイズ除去（エッジ保持）</td></tr>
        <tr><td>Median</td><td>フィルタサイズ: 整数（例 1–9）</td><td>スパイクノイズ除去</td></tr>
    </table>

    <h2>粒子検出設定</h2>
    <h3>検出方法</h3>
    <table class="param-table">
        <tr><th>方法</th><th>特徴</th><th>パラメータ</th></tr>
        <tr><td>Threshold（閾値法）</td><td>Otsu / Manual / Adaptive</td><td>Manual: 閾値 0.0–1.0（またはデータ範囲）。Manual 選択時はインタラクティブヒストグラムで調整。</td></tr>
        <tr><td>Peak Detection（ピーク検出）</td><td>Watershed または Contour Level</td><td>Min Distance (px): 1–50（デフォルト 5）。Gaussian Sigma: 0.5–3.0（デフォルト 1.5）。Boundary Method: Watershed / Contour Level。Watershed threshold (%): 10–99（デフォルト 70）。Contour Level (%): 10–90（デフォルト 30）。</td></tr>
        <tr><td>Hessian Blob</td><td>Hessian 行列によるブロブ検出</td><td>Min Sigma: 0.1–10.0（デフォルト 1.0）。Max Sigma: 1.0–20.0（デフォルト 5.0）。Hessian threshold: 0.0–0.5（デフォルト 0.05）。推奨: 0.05–0.25。</td></tr>
        <tr><td>Ring Detection（リング検出）</td><td>放射プロファイルによるリング構造検出</td><td>Min Radius (nm): 1–1000（デフォルト 5）。Max Radius (nm): 1–10000（デフォルト 100）。Center Blend % Weighted: 0–100（デフォルト 50）。Inner Drop % of peak: 5–90（デフォルト 5）。内径推奨: 15–35%。</td></tr>
    </table>

    <h3>サイズフィルタリング</h3>
    <ul>
        <li><strong>Min Particle Size（最小粒子サイズ）:</strong> 1–1000 ピクセル（デフォルト 10）。これより小さい領域を除外。推奨: 5–50。</li>
        <li><strong>Max Particle Size（最大粒子サイズ）:</strong> 1–10000 ピクセル（デフォルト 1000）。これより大きい領域を除外。推奨: 100–5000。</li>
        <li><strong>Exclude particles touching image edges（エッジ粒子除外）:</strong> 推奨: 有効。画像端に接する不完全な粒子を除外。</li>
    </ul>

    <h2>測定パラメータ</h2>
    <table class="param-table">
        <tr><th>パラメータ</th><th>意味</th><th>単位</th></tr>
        <tr><td>Area</td><td>粒子の面積</td><td>nm²</td></tr>
        <tr><td>Perimeter</td><td>粒子の周長</td><td>nm</td></tr>
        <tr><td>Circularity</td><td>形状の円形度（4π×面積/周長²）</td><td>-</td></tr>
        <tr><td>Max Height</td><td>粒子の最大高さ</td><td>nm</td></tr>
        <tr><td>Mean Height</td><td>粒子の平均高さ</td><td>nm</td></tr>
        <tr><td>Volume</td><td>粒子の体積（面積×平均高さ）</td><td>nm³</td></tr>
        <tr><td>Centroid X/Y</td><td>粒子の重心座標</td><td>nm</td></tr>
    </table>

    <h2>解析操作</h2>
    <table class="param-table">
        <tr><th>機能</th><th>操作方法</th><th>結果</th></tr>
        <tr><td>Detect</td><td>ボタンクリック</td><td>現在フレームの粒子検出</td></tr>
        <tr><td>All Frames</td><td>ボタンクリック</td><td>全フレームの一括解析</td></tr>
        <tr><td>閾値調整</td><td>ヒストグラム上でマウス操作（Manual 閾値時）</td><td>リアルタイムプレビューで閾値を調整</td></tr>
        <tr><td>Export Results</td><td>検出後にボタンクリック</td><td>粒子/リングデータを CSV で保存</td></tr>
    </table>

    <div class="feature-box">
        <h3>リング検出（新機能）</h3>
        <ul>
            <li><strong>中心推定:</strong> 強度最小と幾何中心距離をブレンドして中心を推定（Center Blend %）。</li>
            <li><strong>内径:</strong> 反転データ上でピーク高さに対する Inner Drop % of peak（初期値 5%）の等高線を検出。</li>
            <li><strong>リング径:</strong> 反転データに対する中心起点の Watershed で中間輪郭を取得（全方位ピークからのフォールバックあり）。</li>
            <li><strong>楕円フィットと表示:</strong> 内径/中間輪郭に楕円フィットし、破線でオーバーレイ表示。結果: Inner Diameter (nm)、Ring Diameter (nm)、Circularity（楕円ベース）。</li>
            <li><strong>行削除:</strong> Results 右クリック「Delete Particle」で行削除＆番号詰め（粒子・リング両方）。</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>インタラクティブヒストグラム</h3>
        <p>閾値方法が Manual のときヒストグラムが表示されます。ヒストグラム上でクリック＆ドラッグすると閾値をリアルタイムプレビューで調整できます。</p>
    </div>

    <div class="feature-box">
        <h3>粒子特性</h3>
        <p>面積 (nm²)、周長 (nm)、円形度、最大高さ (nm)、平均高さ (nm)、体積 (nm³)。「Calculate Particle Properties」と各チェックボックスで有効/無効を切り替えます。</p>
    </div>

    <h2>解析ワークフロー</h2>
    <div class="step">ステップ1: AFM画像を読み込み、Plugin → Particle Analysis で開く</div>
    <div class="step">ステップ2: 前処理を設定（背景除去、平滑化）</div>
    <div class="step">ステップ3: 検出方法を選択（閾値／ピーク検出／Hessian Blob／リング検出）</div>
    <div class="step">ステップ4: パラメータを調整（Manual 閾値時はインタラクティブヒストグラムを使用）</div>
    <div class="step">ステップ5: 粒子サイズ制限と除外基準を設定</div>
    <div class="step">ステップ6: 「Detect」で粒子を解析</div>
    <div class="step">ステップ7: 結果を確認しエクスポート</div>

    <h2>高度な機能</h2>
    <ul>
        <li><strong>All Frames:</strong> 時系列データの全フレームで粒子を解析。</li>
        <li><strong>フレームナビゲーション:</strong> 矢印ボタンでフレーム間を移動。</li>
        <li><strong>Export Results:</strong> 粒子/リング特性を CSV で保存。</li>
        <li><strong>Show Legend:</strong> 画像上の凡例の表示/非表示。</li>
    </ul>

    <h2>パラメータガイドライン</h2>
    <ul>
        <li><strong>最小粒子サイズ:</strong> 通常 10+ ピクセル。ノイズを除外するには大きくする。</li>
        <li><strong>最大粒子サイズ:</strong> 通常 1000 ピクセル。大きな背景を除外するには小さくする。</li>
        <li><strong>エッジ除外:</strong> 正確な解析のため有効を推奨。</li>
    </ul>

    <div class="note">
        <strong>プロのヒント:</strong>
        <ul>
            <li>Otsu から開始し、必要に応じて手動で微調整。</li>
            <li>背景変動が大きい画像には背景除去を使用。</li>
            <li>ノイズ除去と特徴保持のバランスのため平滑化を調整。</li>
            <li>時系列解析では全フレームで一貫したパラメータを使用。</li>
            <li>定期的に結果をエクスポートしてデータ損失を防ぐ。</li>
        </ul>
    </div>
"""

# プラグイン契約: Plugin メニューから Load Plugin で読み込み
PLUGIN_NAME = "Particle Analysis"


def create_plugin(main_window):
    """pyNuD から呼び出されるエントリポイント。"""
    return ParticleAnalysisWindow(main_window)


class ParticleAnalysisWindow(QtWidgets.QWidget):
    """
    Particle analysis window for AFM image analysis
    AFM画像の粒子解析ウィンドウ
    """
    
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle("Particle Analysis")
        self.setMinimumSize(400, 600)
        
        # ウィンドウ管理システムに登録
        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self, "sub")
        except ImportError:
            pass
        
        #print("[DEBUG] ParticleAnalysisWindow.__init__ - Starting initialization")
        
        # 解析用のデータ
        self.filtered_data = None  # 前処理後のデータ
        self.current_data = None   # 後方互換性のため残す
        self.roi_rect = None
        self.roi_shape = "Rectangle"
        self.roi_data = None  # ROI内の拡大画像データ
        self.analysis_mode = "All Particles"  # 解析モード: "All Particles" or "Single Particle"
        self.roi_scale_info = None  # ROIの物理サイズとピクセル情報
        self.ring_last_avg_info = None
        self.ring_detection_history = []  # Ring Detection の履歴（複数回の検出を保持）
        self.current_threshold_method = "Otsu"
        
        # 粒子検出結果
        self.detected_particles = []
        self.particle_properties = []
        
        # グラフウィンドウの参照
        self.graph_window = None
        
        # オーバーレイ（重ね描き）されたプロットを保存するリスト
        self.overlay_artists = []
        
        #print("[DEBUG] ParticleAnalysisWindow.__init__ - Calling setupUI")
        self.setupUI()
        #print("[DEBUG] ParticleAnalysisWindow.__init__ - Calling loadWindowSettings")
        self.loadWindowSettings()
        
        # ウィンドウが開かれた時に自動的にデータを読み込む
        #print("[DEBUG] ParticleAnalysisWindow.__init__ - Setting up timer for initializeData")
        QtCore.QTimer.singleShot(100, self.initializeData)
        
        #print("[DEBUG] ParticleAnalysisWindow.__init__ - Initialization complete")
        
    def initializeData(self):
        """ウィンドウ初期化時にデータを読み込む"""
        
        #print("[DEBUG] initializeData - Starting")
        
        # setupUIで既にデータが処理されている場合はスキップ
        if hasattr(self, 'data_initialized') and self.data_initialized:
            #print("[DEBUG] initializeData - Data already initialized, but applying detection method")
            self.applyDetectionMethod()
            return
            
        #print("[DEBUG] initializeData - Calling getCurrentData")
        if self.getCurrentData():
            #print("[DEBUG] initializeData - Data obtained successfully")
            #print(f"[DEBUG] initializeData - Current data shape: {self.current_data.shape}")
            #print(f"[DEBUG] initializeData - Current data range: {np.min(self.current_data):.3f} to {np.max(self.current_data):.3f}")
            
            # gv.aryDataをfiltered_dataにコピー（起動時は元データ）
            #print("[DEBUG] initializeData - Copying gv.aryData to filtered_data")
            self.filtered_data = self.current_data.copy()
            #print(f"[DEBUG] initializeData - Filtered data shape: {self.filtered_data.shape}")
            #print(f"[DEBUG] initializeData - Filtered data range: {np.min(self.filtered_data):.3f} to {np.max(self.filtered_data):.3f}")
            
            # filtered_dataを表示
            #print("[DEBUG] initializeData - Displaying filtered_data")
            self.displayFilteredImage()
            
            # 起動時に自動的にthreshold/peak detectionを適用
            #print("[DEBUG] initializeData - Applying automatic detection")
            #print(f"[DEBUG] initializeData - Current smoothing method: {self.smooth_combo.currentText()}")
            #print(f"[DEBUG] initializeData - Current smooth parameter: {self.smooth_param_spin.value()}")
            #print(f"[DEBUG] initializeData - Current smooth parameter: {self.smooth_param_spin.value()}")
            #print(f"[DEBUG] initializeData - Smooth parameter spinbox visible: {self.smooth_param_spin.isVisible()}")
            self.applyDetectionMethod()
            
        else:
            #print("[DEBUG] initializeData - Failed to get current data")
            pass
            
        #print("[DEBUG] initializeData - Complete")
        
    def setupUI(self):
        """UIの初期化"""
        top_layout = QtWidgets.QVBoxLayout(self)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)  # ウィンドウ内にメニューを表示（macOS で見えない対策）
        help_menu = menu_bar.addMenu("Help" if QtCore.QLocale().language() != QtCore.QLocale.Japanese else "ヘルプ")
        manual_action = help_menu.addAction("Manual" if QtCore.QLocale().language() != QtCore.QLocale.Japanese else "マニュアル")
        manual_action.triggered.connect(self.showHelpDialog)
        top_layout.addWidget(menu_bar)
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # 左側：パラメータ設定エリア
        left_widget = QtWidgets.QWidget()
        left_widget.setFixedWidth(400)
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        
        # スクロールエリアを作成
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setMinimumHeight(600)
        
        # スクロールエリアの中身となるウィジェットを作成
        content_widget = QtWidgets.QWidget()
        main_layout_content = QtWidgets.QVBoxLayout(content_widget)
        main_layout_content.setContentsMargins(5, 5, 5, 5)
        main_layout_content.setSpacing(5)
        
        # Analysis Mode Selection (一番上に配置)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["All Particles", "Single Particle"])
        self.mode_combo.setFixedWidth(150)
        self.mode_combo.setToolTip("解析モードを選択します。\nAll Particles: 全画面で粒子検出\nSingle Particle: ROI選択して拡大領域のみで粒子検出\n\nSelect analysis mode.\nAll Particles: Detect particles in full image\nSingle Particle: Select ROI and detect particles in magnified region")
        self.mode_combo.currentTextChanged.connect(self.onModeChanged)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        main_layout_content.addLayout(mode_row)
        
        # Preprocessing Group
        preprocess_group = QtWidgets.QGroupBox("Preprocessing")
        preprocess_layout = QtWidgets.QVBoxLayout()
        
        # Background subtraction
        bg_row = QtWidgets.QHBoxLayout()
        bg_row.addWidget(QtWidgets.QLabel("Background Subtraction:"))
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItems(["None", "Rolling Ball", "Polynomial Fit"])
        self.bg_combo.setFixedWidth(120)
        self.bg_combo.currentTextChanged.connect(self.onPreprocessingChanged)
        bg_row.addWidget(self.bg_combo)
        bg_row.addStretch()
        preprocess_layout.addLayout(bg_row)
        
        # Rolling Ball radius (only show when Rolling Ball is selected)
        self.rolling_radius_row = QtWidgets.QHBoxLayout()
        self.rolling_radius_label = QtWidgets.QLabel("Rolling Ball Radius (nm):")
        self.rolling_radius_spin = QtWidgets.QSpinBox()
        self.rolling_radius_spin.setRange(1, 1000)
        self.rolling_radius_spin.setValue(10)  # デフォルト値を50から10に変更
        self.rolling_radius_spin.setFixedWidth(60)
        self.rolling_radius_spin.valueChanged.connect(self.onPreprocessingChanged)
        self.rolling_radius_row.addWidget(self.rolling_radius_label)
        self.rolling_radius_row.addWidget(self.rolling_radius_spin)
        self.rolling_radius_row.addStretch()
        # 初期状態は非表示（ラベルとスピンボックスを非表示）
        self.rolling_radius_label.setVisible(False)
        self.rolling_radius_spin.setVisible(False)
        preprocess_layout.addLayout(self.rolling_radius_row)
        
        # Smoothing
        smooth_row = QtWidgets.QHBoxLayout()
        smooth_row.addWidget(QtWidgets.QLabel("Smoothing:"))
        self.smooth_combo = QtWidgets.QComboBox()
        self.smooth_combo.addItems(["None", "Gaussian", "Median"])
        self.smooth_combo.setFixedWidth(120)
        # Smoothing methodのデバッグ用コールバックを追加
        def onSmoothingMethodChanged(method):
            #print(f"[DEBUG] ===== Smoothing method changed to: {method} =====")
            #print(f"[DEBUG] Current smooth parameter: {self.smooth_param_spin.value()}")
            self.updatePreprocessingUI()
            self.onPreprocessingChanged()
            #print(f"[DEBUG] ===== Smoothing method change completed =====")
        
        self.smooth_combo.currentTextChanged.connect(onSmoothingMethodChanged)
        smooth_row.addWidget(self.smooth_combo)
        smooth_row.addStretch()
        preprocess_layout.addLayout(smooth_row)
        
        # Smoothing parameter
        smooth_param_row = QtWidgets.QHBoxLayout()

        self.smooth_param_spin = QtWidgets.QDoubleSpinBox()
        self.smooth_param_spin.setRange(0.1, 3.0)  # Gaussian Sigma用の範囲
        self.smooth_param_spin.setValue(1.0)  # デフォルト値
        self.smooth_param_spin.setSingleStep(0.1)  # 0.1刻み
        self.smooth_param_spin.setDecimals(1)  # 小数点1桁
        self.smooth_param_spin.setFixedWidth(60)
        self.smooth_param_spin.valueChanged.connect(self.onSmoothingParameterChanged)
        smooth_param_row.addWidget(self.smooth_param_spin)
        smooth_param_row.addStretch()
        # 初期状態は非表示（Smoothing methodが選択されたときに表示される）
        # QHBoxLayoutにはsetVisibleがないため、各ウィジェットを個別に制御
        self.smooth_param_label = QtWidgets.QLabel("Gaussian Sigma:")
        self.smooth_param_label.setVisible(False)
        smooth_param_row.insertWidget(0, self.smooth_param_label)
        self.smooth_param_spin.setVisible(False)
        preprocess_layout.addLayout(smooth_param_row)
        
        preprocess_group.setLayout(preprocess_layout)
        main_layout_content.addWidget(preprocess_group)
        
        # Particle Detection Group
        detection_group = QtWidgets.QGroupBox("Particle Detection")
        self.detection_layout = QtWidgets.QVBoxLayout()  # クラス変数として保存
        
        # Detection method selection
        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(QtWidgets.QLabel("Detection Method:"))
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["Threshold", "Peak Detection", "Hessian Blob", "Ring Detection"])
        self.method_combo.setFixedWidth(120)
        self.method_combo.setToolTip("粒子検出方法を選択します。\nThreshold: 閾値ベースの検出\nPeak Detection: ピーク検出とWatershed分割\nHessian Blob: Hessian行列によるブロブ検出\nRing Detection: 円形ハフ変換によるリング検出\n\nSelect particle detection method.\nThreshold: Threshold-based detection\nPeak Detection: Peak detection with Watershed segmentation\nHessian Blob: Blob detection using Hessian matrix\nRing Detection: Ring detection using Circular Hough Transform")
        self.method_combo.currentTextChanged.connect(self.onDetectionMethodChanged)
        method_row.addWidget(self.method_combo)
        method_row.addStretch()
        self.detection_layout.addLayout(method_row)
        
        # Particle detection status label (below Detection Method) - for Single Particle mode
        self.particle_status_label = QtWidgets.QLabel("")
        self.particle_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
        self.particle_status_label.setVisible(False)  # 初期状態では非表示
        particle_status_row = QtWidgets.QHBoxLayout()
        particle_status_row.addWidget(self.particle_status_label)
        particle_status_row.addStretch()
        self.detection_layout.addLayout(particle_status_row)
        
        # Threshold parameters
        threshold_row = QtWidgets.QHBoxLayout()
        threshold_row.addWidget(QtWidgets.QLabel("Threshold:"))
        self.threshold_combo = QtWidgets.QComboBox()
        self.threshold_combo.addItems(["Otsu", "Manual", "Adaptive"])
        self.threshold_combo.setFixedWidth(100)
        self.threshold_combo.setToolTip("閾値設定方法を選択します。\nOtsu: 自動的に最適な閾値を計算\nManual: 手動で閾値を設定\nAdaptive: 局所的に適応的な閾値を計算\n\nSelect threshold method.\nOtsu: Automatically calculate optimal threshold\nManual: Set threshold manually\nAdaptive: Calculate adaptive threshold locally")
        self.threshold_combo.currentTextChanged.connect(self.onThresholdMethodChanged)
        threshold_row.addWidget(self.threshold_combo)
        
        threshold_row.addStretch()
        self.detection_layout.addLayout(threshold_row)
        
        # Manual threshold value
        manual_thresh_row = QtWidgets.QHBoxLayout()
        manual_thresh_row.addWidget(QtWidgets.QLabel("Manual Threshold:"))
        self.manual_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.manual_thresh_spin.setRange(0.0, 1.0)
        self.manual_thresh_spin.setSingleStep(0.01)
        self.manual_thresh_spin.setValue(0.5)
        self.manual_thresh_spin.setFixedWidth(80)
        self.manual_thresh_spin.setToolTip("手動で設定する閾値です。\n0.0: すべてのピクセルを粒子として検出\n1.0: 最も明るいピクセルのみを検出\n推奨値: 0.1-0.8\n\nManual threshold value.\n0.0: Detect all pixels as particles\n1.0: Detect only brightest pixels\nRecommended: 0.1-0.8")
        self.manual_thresh_spin.setEnabled(False)  # 初期状態では無効化
        self.manual_thresh_spin.valueChanged.connect(self.onManualThresholdChanged)
        manual_thresh_row.addWidget(self.manual_thresh_spin)
        manual_thresh_row.addStretch()
        self.detection_layout.addLayout(manual_thresh_row)
        
        # Peak Detection Parameters Group
        self.peak_group = QtWidgets.QWidget()  # QGroupBoxからQWidgetに変更
        peak_layout = QtWidgets.QVBoxLayout()
        
        # Peak detection parameters
        peak_params_row = QtWidgets.QHBoxLayout()
        peak_params_row.addWidget(QtWidgets.QLabel("Min Distance (pixels):"))
        self.min_peak_distance_spin = QtWidgets.QSpinBox()
        self.min_peak_distance_spin.setRange(1, 50)
        self.min_peak_distance_spin.setValue(5)
        self.min_peak_distance_spin.setFixedWidth(30)
        self.min_peak_distance_spin.setToolTip("ピーク間の最小距離（ピクセル）です。\n小さい値: より多くのピークを検出\n大きい値: より少ないピークを検出\n推奨値: 3-10\n\nMinimum distance between peaks in pixels.\nSmall values: Detect more peaks\nLarge values: Detect fewer peaks\nRecommended: 3-10")
        peak_params_row.addWidget(self.min_peak_distance_spin)
        
        peak_params_row.addWidget(QtWidgets.QLabel("Gaussian Sigma:"))
        self.gradient_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.gradient_sigma_spin.setRange(0.5, 3.0)
        self.gradient_sigma_spin.setValue(1.5)  # 1.5に調整
        self.gradient_sigma_spin.setSingleStep(0.1)
        self.gradient_sigma_spin.setFixedWidth(50)
        self.gradient_sigma_spin.setToolTip("ガウシアンフィルタのシグマ値です。\n小さい値: より細かい特徴を保持\n大きい値: ノイズを除去し、滑らかに\n推奨値: 0.5-2.0\n\nGaussian filter sigma value.\nSmall values: Preserve fine details\nLarge values: Remove noise and smooth\nRecommended: 0.5-2.0")
        peak_params_row.addWidget(self.gradient_sigma_spin)
        
        peak_params_row.addStretch()
        peak_layout.addLayout(peak_params_row)
        
        # Boundary method selection
        boundary_method_row = QtWidgets.QHBoxLayout()
        boundary_method_row.addWidget(QtWidgets.QLabel("Boundary Method:"))
        self.boundary_method_combo = QtWidgets.QComboBox()
        self.boundary_method_combo.addItems(["Watershed", "Contour Level"])
        self.boundary_method_combo.setFixedWidth(120)
        self.boundary_method_combo.setToolTip("粒子境界の設定方法を選択します。\nWatershed: 従来のwatershed分割\nContour Level: 等高線による境界設定\n\nSelect particle boundary method.\nWatershed: Traditional watershed segmentation\nContour Level: Boundary setting using contour levels")
        boundary_method_row.addWidget(self.boundary_method_combo)
        boundary_method_row.addStretch()
        peak_layout.addLayout(boundary_method_row)
        
        # Watershed threshold parameter
        watershed_threshold_row = QtWidgets.QHBoxLayout()
        watershed_threshold_row.addWidget(QtWidgets.QLabel("Watershed threshold (%):"))
        self.watershed_threshold_spin = QtWidgets.QSpinBox()
        self.watershed_threshold_spin.setRange(10, 99)
        self.watershed_threshold_spin.setValue(70)  # デフォルト値を70%に変更（より緩い）
        self.watershed_threshold_spin.setSingleStep(1)
        self.watershed_threshold_spin.setFixedWidth(60)
        self.watershed_threshold_spin.setToolTip("Threshold percentile for watershed mask (higher = smaller regions) / Watershedマスクの閾値パーセンタイル（高いほど領域が小さい）")
        watershed_threshold_row.addWidget(self.watershed_threshold_spin)
        watershed_threshold_row.addStretch()
        peak_layout.addLayout(watershed_threshold_row)
        
        # Contour level parameter
        contour_level_row = QtWidgets.QHBoxLayout()
        contour_level_row.addWidget(QtWidgets.QLabel("Contour Level (%):"))
        self.contour_level_spin = QtWidgets.QSpinBox()
        self.contour_level_spin.setRange(10, 90)
        self.contour_level_spin.setValue(30)  # より適切なデフォルト値
        self.contour_level_spin.setSingleStep(5)
        self.contour_level_spin.setFixedWidth(60)
        self.contour_level_spin.setToolTip("ピーク高さから下がる割合（%）です。\n小さい値: ピークに近い高さで等高線（大きな粒子境界）\n大きい値: ピークから遠い高さで等高線（小さな粒子境界）\n推奨値: 10-50\n\nPercentage to drop from peak height.\nSmall values: Contour near peak (larger boundaries)\nLarge values: Contour far from peak (smaller boundaries)\nRecommended: 10-50")
        contour_level_row.addWidget(self.contour_level_spin)
        contour_level_row.addStretch()
        peak_layout.addLayout(contour_level_row)
        
        # 初期状態でContour Levelパラメータを非表示
        self.contour_level_spin.setVisible(False)
        # Contour Levelのラベルも非表示
        for i in range(contour_level_row.count()):
            item = contour_level_row.itemAt(i)
            if item.widget():
                widget = item.widget()
                if isinstance(widget, QtWidgets.QLabel) and "Contour Level" in widget.text():
                    widget.setVisible(False)
                    break
        
        # Auto-adjustment info
        auto_info_row = QtWidgets.QHBoxLayout()
        auto_info_label = QtWidgets.QLabel("Auto-adjustment: Adaptive threshold automatically optimized")
        auto_info_label.setStyleSheet("color: blue; font-size: 10px;")
        auto_info_row.addWidget(auto_info_label)
        auto_info_row.addStretch()
        peak_layout.addLayout(auto_info_row)
        
        # パラメーター変更時のシグナル接続
        self.min_peak_distance_spin.valueChanged.connect(self.onPeakParametersChanged)
        self.gradient_sigma_spin.valueChanged.connect(self.onPeakParametersChanged)
        self.watershed_threshold_spin.valueChanged.connect(self.onPeakParametersChanged)
        self.boundary_method_combo.currentTextChanged.connect(self.onBoundaryMethodChanged)
        self.contour_level_spin.valueChanged.connect(self.onPeakParametersChanged)
        
        self.peak_group.setLayout(peak_layout)
        self.peak_group.setVisible(False)  # 初期状態では非表示
        self.detection_layout.addWidget(self.peak_group)
        
        # Hessian Blob Parameters Group
        self.hessian_group = QtWidgets.QWidget()
        hessian_layout = QtWidgets.QVBoxLayout()
        
        # Hessian Blob parameters
        hessian_params_row = QtWidgets.QHBoxLayout()
        hessian_params_row.addWidget(QtWidgets.QLabel("Min Sigma:"))
        self.hessian_min_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.hessian_min_sigma_spin.setRange(0.1, 10.0)
        self.hessian_min_sigma_spin.setValue(1.0)
        self.hessian_min_sigma_spin.setSingleStep(0.1)
        self.hessian_min_sigma_spin.setFixedWidth(50)
        self.hessian_min_sigma_spin.setToolTip("ブロブ検出の最小シグマ値です。\n小さい値: 小さな粒子を検出\n大きい値: 大きな粒子のみを検出\n推奨値: 0.5-2.0\n\nMinimum sigma for blob detection.\nSmall values: Detect small particles\nLarge values: Detect only large particles\nRecommended: 0.5-2.0")
        hessian_params_row.addWidget(self.hessian_min_sigma_spin)
        
        hessian_params_row.addWidget(QtWidgets.QLabel("Max Sigma:"))
        self.hessian_max_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.hessian_max_sigma_spin.setRange(1.0, 20.0)
        self.hessian_max_sigma_spin.setValue(5.0)
        self.hessian_max_sigma_spin.setSingleStep(0.5)
        self.hessian_max_sigma_spin.setFixedWidth(50)
        self.hessian_max_sigma_spin.setToolTip("ブロブ検出の最大シグマ値です。\n小さい値: 小さな粒子のみを検出\n大きい値: 大きな粒子も検出\n推奨値: 3.0-10.0\n\nMaximum sigma for blob detection.\nSmall values: Detect only small particles\nLarge values: Detect large particles too\nRecommended: 3.0-10.0")
        hessian_params_row.addWidget(self.hessian_max_sigma_spin)
        
        hessian_params_row.addStretch()
        hessian_layout.addLayout(hessian_params_row)
        
        # Hessian threshold parameter
        hessian_threshold_row = QtWidgets.QHBoxLayout()
        hessian_threshold_row.addWidget(QtWidgets.QLabel("Hessian threshold:"))
        self.hessian_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.hessian_threshold_spin.setRange(0.0, 0.5)
        self.hessian_threshold_spin.setValue(0.05)
        self.hessian_threshold_spin.setSingleStep(0.05)
        self.hessian_threshold_spin.setDecimals(2)
        self.hessian_threshold_spin.setFixedWidth(60)
        self.hessian_threshold_spin.setToolTip("Hessianブロブ検出の閾値です。\n小さい値: より多くのブロブを検出\n大きい値: より少ないブロブを検出\n推奨値: 0.05-0.25\n\nThreshold for Hessian blob detection.\nSmall values: Detect more blobs\nLarge values: Detect fewer blobs\nRecommended: 0.05-0.25")
        hessian_threshold_row.addWidget(self.hessian_threshold_spin)
        hessian_threshold_row.addStretch()
        hessian_layout.addLayout(hessian_threshold_row)
        
        # Hessian info
        hessian_info_row = QtWidgets.QHBoxLayout()
        hessian_info_label = QtWidgets.QLabel("Hessian Blob: Detects particles using Hessian matrix analysis")
        hessian_info_label.setStyleSheet("color: blue; font-size: 10px;")
        hessian_info_row.addWidget(hessian_info_label)
        hessian_info_row.addStretch()
        hessian_layout.addLayout(hessian_info_row)
        
        # パラメーター変更時のシグナル接続
        self.hessian_min_sigma_spin.valueChanged.connect(self.onHessianParametersChanged)
        self.hessian_max_sigma_spin.valueChanged.connect(self.onHessianParametersChanged)
        self.hessian_threshold_spin.valueChanged.connect(self.onHessianParametersChanged)
        
        # Hessian thresholdに右クリックメニューを追加
        self.hessian_threshold_spin.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.hessian_threshold_spin.customContextMenuRequested.connect(self.showHessianThresholdContextMenu)
        
        self.hessian_group.setLayout(hessian_layout)
        self.hessian_group.setVisible(False)  # 初期状態では非表示
        self.detection_layout.addWidget(self.hessian_group)
        
        # Ring Detection Parameters Group
        self.ring_group = QtWidgets.QWidget()
        ring_layout = QtWidgets.QVBoxLayout()
        
        # Ring detection parameters
        ring_params_row1 = QtWidgets.QHBoxLayout()
        ring_params_row1.addWidget(QtWidgets.QLabel("Min Radius (nm):"))
        self.ring_min_radius_spin = QtWidgets.QDoubleSpinBox()
        self.ring_min_radius_spin.setRange(1.0, 1000.0)
        self.ring_min_radius_spin.setValue(5.0)
        self.ring_min_radius_spin.setSingleStep(1.0)
        self.ring_min_radius_spin.setDecimals(1)
        self.ring_min_radius_spin.setFixedWidth(70)
        self.ring_min_radius_spin.setToolTip("検出するリングの最小半径（nm）です。\n小さい値: より小さなリングも検出\n大きい値: 大きなリングのみを検出\n推奨値: 5-50 nm\n\nMinimum ring radius in nm.\nSmall values: Detect smaller rings too\nLarge values: Detect only large rings\nRecommended: 5-50 nm")
        ring_params_row1.addWidget(self.ring_min_radius_spin)
        
        ring_params_row1.addWidget(QtWidgets.QLabel("Max Radius (nm):"))
        self.ring_max_radius_spin = QtWidgets.QDoubleSpinBox()
        self.ring_max_radius_spin.setRange(1.0, 10000.0)
        self.ring_max_radius_spin.setValue(100.0)
        self.ring_max_radius_spin.setSingleStep(5.0)
        self.ring_max_radius_spin.setDecimals(1)
        self.ring_max_radius_spin.setFixedWidth(70)
        self.ring_max_radius_spin.setToolTip("検出するリングの最大半径（nm）です。\n小さい値: 小さなリングのみを検出\n大きい値: 大きなリングも検出\n推奨値: 50-500 nm\n\nMaximum ring radius in nm.\nSmall values: Detect only small rings\nLarge values: Detect large rings too\nRecommended: 50-500 nm")
        ring_params_row1.addWidget(self.ring_max_radius_spin)
        ring_params_row1.addStretch()
        ring_layout.addLayout(ring_params_row1)
        
        # Ring detection parameters (center & width control)
        ring_form_layout = QtWidgets.QFormLayout()
        ring_form_layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        ring_form_layout.setFormAlignment(QtCore.Qt.AlignLeft)
        self.ring_center_blend_spin = QtWidgets.QDoubleSpinBox()
        self.ring_center_blend_spin.setRange(0.0, 100.0)
        self.ring_center_blend_spin.setValue(50.0)
        self.ring_center_blend_spin.setSingleStep(5.0)
        self.ring_center_blend_spin.setDecimals(1)
        self.ring_center_blend_spin.setFixedWidth(80)
        self.ring_center_blend_spin.setToolTip(
            "幾何学中心と重み付き中心をどの割合で混合するかを指定します。\n"
            "0%: 幾何学中心のみ\n100%: 重み付き中心のみ\n推奨: 50%\n\n"
            "Blend ratio between geometric center and intensity-weighted center.\n"
            "0%: Pure geometric center\n100%: Pure weighted center\nRecommended: 50%")
        ring_form_layout.addRow("Center Blend % Weighted:", self.ring_center_blend_spin)

        self.ring_inner_drop_spin = QtWidgets.QDoubleSpinBox()
        self.ring_inner_drop_spin.setRange(5.0, 90.0)
        self.ring_inner_drop_spin.setValue(5.0)
        self.ring_inner_drop_spin.setSingleStep(1.0)
        self.ring_inner_drop_spin.setDecimals(1)
        self.ring_inner_drop_spin.setFixedWidth(80)
        self.ring_inner_drop_spin.setToolTip(
            "内径を判定する際にピーク高さから何%低下した位置を基準にするか指定します。\n"
            "小さい値: 内径が大きく（リングが太く）なる傾向\n大きい値: 内径が小さくなる傾向\n推奨: 15-35%\n\n"
            "Controls the drop level used to determine the inner boundary of the ring.\n"
            "Smaller values yield larger inner radii (thicker rings); larger values yield smaller inner radii.\nRecommended: 15-35%")
        ring_form_layout.addRow("Inner Drop % of peak:", self.ring_inner_drop_spin)

        # Outer Drop 控えのUIは不要のため削除
        ring_layout.addLayout(ring_form_layout)
        
        # Ring detection info
        ring_info_row = QtWidgets.QHBoxLayout()
        ring_info_label = QtWidgets.QLabel("Ring Detection: Detects ring structures via radial profile analysis")
        ring_info_label.setStyleSheet("color: blue; font-size: 10px;")
        ring_info_row.addWidget(ring_info_label)
        ring_info_row.addStretch()
        ring_layout.addLayout(ring_info_row)
        
        # パラメーター変更時のシグナル接続
        self.ring_min_radius_spin.valueChanged.connect(self.onRingParametersChanged)
        self.ring_max_radius_spin.valueChanged.connect(self.onRingParametersChanged)
        self.ring_center_blend_spin.valueChanged.connect(self.onRingParametersChanged)
        self.ring_inner_drop_spin.valueChanged.connect(self.onRingParametersChanged)
        # 外径しきい値UIは削除済み
        
        self.ring_group.setLayout(ring_layout)
        self.ring_group.setVisible(False)  # 初期状態では非表示
        self.detection_layout.addWidget(self.ring_group)
        
        # Interactive histogram for manual threshold
        self.histogram_group = QtWidgets.QGroupBox("Interactive Histogram")
        histogram_layout = QtWidgets.QVBoxLayout()
        
        # Histogram canvas
        self.hist_figure = Figure(figsize=(6, 3.3))  # 高さを2/3程度に調整
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_axes = self.hist_figure.add_subplot(111)
        
        # マージンを設定してラベルがはみ出さないようにする
        self.hist_figure.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
        
        # Threshold line on histogram
        self.threshold_line = None
        self.hist_data = None
        
        histogram_layout.addWidget(self.hist_canvas)
        self.histogram_group.setLayout(histogram_layout)
        self.histogram_group.setMinimumHeight(200)  # 最小高さを2/3程度に調整
        self.histogram_group.setVisible(False)  # 初期状態では非表示
        self.detection_layout.addWidget(self.histogram_group)
        
        # Minimum particle size
        min_size_row = QtWidgets.QHBoxLayout()
        min_size_row.addWidget(QtWidgets.QLabel("Min Particle Size (pixels):"))
        self.min_size_spin = QtWidgets.QSpinBox()
        self.min_size_spin.setRange(1, 1000)
        self.min_size_spin.setValue(10)
        self.min_size_spin.setFixedWidth(80)
        self.min_size_spin.setToolTip("検出する粒子の最小サイズ（ピクセル）です。\n小さい値: より小さな粒子も検出\n大きい値: 大きな粒子のみを検出\n推奨値: 5-50\n\nMinimum particle size in pixels.\nSmall values: Detect smaller particles\nLarge values: Detect only large particles\nRecommended: 5-50")
        self.min_size_spin.valueChanged.connect(self.onParticleSizeChanged)
        min_size_row.addWidget(self.min_size_spin)
        min_size_row.addStretch()
        self.detection_layout.addLayout(min_size_row)
        
        # Maximum particle size
        max_size_row = QtWidgets.QHBoxLayout()
        max_size_row.addWidget(QtWidgets.QLabel("Max Particle Size (pixels):"))
        self.max_size_spin = QtWidgets.QSpinBox()
        self.max_size_spin.setRange(1, 10000)
        self.max_size_spin.setValue(1000)
        self.max_size_spin.setFixedWidth(80)
        self.max_size_spin.setToolTip("検出する粒子の最大サイズ（ピクセル）です。\n小さい値: 小さな粒子のみを検出\n大きい値: 大きな粒子も検出\n推奨値: 100-5000\n\nMaximum particle size in pixels.\nSmall values: Detect only small particles\nLarge values: Detect large particles too\nRecommended: 100-5000")
        self.max_size_spin.valueChanged.connect(self.onParticleSizeChanged)
        max_size_row.addWidget(self.max_size_spin)
        max_size_row.addStretch()
        self.detection_layout.addLayout(max_size_row)
        
        # Edge particle exclusion
        edge_exclusion_row = QtWidgets.QHBoxLayout()
        self.exclude_edge_check = QtWidgets.QCheckBox("Exclude particles touching image edges")
        self.exclude_edge_check.setChecked(True)  # デフォルトで有効
        self.exclude_edge_check.setToolTip("画像の端に接する粒子を除外します。\n有効: 不完全な粒子を除外して精度を向上\n無効: すべての粒子を検出\n\nExclude particles touching image edges.\nEnabled: Exclude incomplete particles for better accuracy\nDisabled: Detect all particles")
        self.exclude_edge_check.toggled.connect(self.onEdgeExclusionChanged)
        edge_exclusion_row.addWidget(self.exclude_edge_check)
        edge_exclusion_row.addStretch()
        self.detection_layout.addLayout(edge_exclusion_row)
        
        detection_group.setLayout(self.detection_layout)
        main_layout_content.addWidget(detection_group)
        
        # Analysis Group
        analysis_group = QtWidgets.QGroupBox("Analysis Parameters")
        analysis_layout = QtWidgets.QVBoxLayout()
        
        # Calculate properties
        self.calc_props_check = QtWidgets.QCheckBox("Calculate Particle Properties")
        self.calc_props_check.setChecked(True)
        self.calc_props_check.setToolTip("粒子の特性を計算します。\n有効: 面積、周長、円形度、高さ、体積を計算\n無効: 検出のみ実行\n\nCalculate particle properties.\nEnabled: Calculate area, perimeter, circularity, height, volume\nDisabled: Detection only")
        analysis_layout.addWidget(self.calc_props_check)
        
        # Properties to calculate
        props_label = QtWidgets.QLabel("Properties:")
        props_label.setStyleSheet("font-weight: bold;")
        analysis_layout.addWidget(props_label)
        
        # 1行目
        props_row1 = QtWidgets.QHBoxLayout()
        self.area_check = QtWidgets.QCheckBox("Area (nm²)")
        self.area_check.setChecked(True)
        self.area_check.setToolTip("粒子の面積を計算します（nm²）\nCalculate particle area (nm²)")
        self.perimeter_check = QtWidgets.QCheckBox("Perimeter (nm)")
        self.perimeter_check.setChecked(True)
        self.perimeter_check.setToolTip("粒子の周長を計算します（nm）\nCalculate particle perimeter (nm)")
        self.circularity_check = QtWidgets.QCheckBox("Circularity")
        self.circularity_check.setChecked(True)
        self.circularity_check.setToolTip("粒子の円形度を計算します（4π×面積/周長²）\nCalculate particle circularity (4π×area/perimeter²)")
        props_row1.addWidget(self.area_check)
        props_row1.addWidget(self.perimeter_check)
        props_row1.addWidget(self.circularity_check)
        props_row1.addStretch()
        analysis_layout.addLayout(props_row1)
        
        # 2行目
        props_row2 = QtWidgets.QHBoxLayout()
        self.max_height_check = QtWidgets.QCheckBox("Max Height (nm)")
        self.max_height_check.setChecked(True)
        self.max_height_check.setToolTip("粒子の最大高さを計算します（nm）\nCalculate particle maximum height (nm)")
        self.mean_height_check = QtWidgets.QCheckBox("Mean Height (nm)")
        self.mean_height_check.setChecked(True)
        self.mean_height_check.setToolTip("粒子の平均高さを計算します（nm）\nCalculate particle mean height (nm)")
        self.volume_check = QtWidgets.QCheckBox("Volume (nm³)")
        self.volume_check.setChecked(True)
        self.volume_check.setToolTip("粒子の体積を計算します（nm³）\nCalculate particle volume (nm³)")
        props_row2.addWidget(self.max_height_check)
        props_row2.addWidget(self.mean_height_check)
        props_row2.addWidget(self.volume_check)
        props_row2.addStretch()
        analysis_layout.addLayout(props_row2)
        
        analysis_group.setLayout(analysis_layout)
        main_layout_content.addWidget(analysis_group)
        
        # Action Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.detect_button = QtWidgets.QPushButton("Detect")
        self.detect_button.clicked.connect(self.detectParticles)
        button_layout.addWidget(self.detect_button)
        
        self.all_frames_button = QtWidgets.QPushButton("All Frames")
        self.all_frames_button.clicked.connect(self.detectAllFrames)
        button_layout.addWidget(self.all_frames_button)
        
        self.export_button = QtWidgets.QPushButton("Export Results")
        self.export_button.clicked.connect(self.exportResults)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)
        
        main_layout_content.addLayout(button_layout)
        
        # Frame Navigation Buttons
        nav_layout = QtWidgets.QHBoxLayout()
        
        self.prev_frame_button = QtWidgets.QPushButton("←")
        self.prev_frame_button.setFixedWidth(40)
        self.prev_frame_button.clicked.connect(self.previousFrame)
        self.prev_frame_button.setEnabled(False)
        nav_layout.addWidget(self.prev_frame_button)
        
        self.next_frame_button = QtWidgets.QPushButton("→")
        self.next_frame_button.setFixedWidth(40)
        self.next_frame_button.clicked.connect(self.nextFrame)
        self.next_frame_button.setEnabled(False)
        nav_layout.addWidget(self.next_frame_button)
        
        nav_layout.addStretch()
        main_layout_content.addLayout(nav_layout)
        
        # 左側を上下に分割
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # 上部：パラメーター設定（スクロール可能）
        param_scroll = QtWidgets.QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        param_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        param_scroll.setWidget(content_widget)
        
        # 下部：Results（スクロール可能）
        results_scroll = QtWidgets.QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        results_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # Results用のウィジェットを作成
        results_content = QtWidgets.QWidget()
        results_content_layout = QtWidgets.QVBoxLayout(results_content)
        
        # Results display with splitter
        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout()
        
        # 分割可能なウィジェットを作成
        self.results_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # 上部：統計情報
        stats_widget = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(stats_widget)
        self.stats_label = QtWidgets.QLabel("Statistics: No particles detected")
        self.stats_label.setStyleSheet("font-weight: bold; color: blue;")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        
        # 下部：結果テーブル（スクロール可能）
        table_widget = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_widget)
        
        # テーブル上部のコントロールエリア
        table_controls_layout = QtWidgets.QHBoxLayout()
        
        # LEGEND表示チェックボックス
        self.show_legend_check = QtWidgets.QCheckBox("Show Legend")
        self.show_legend_check.setChecked(True)  # デフォルトで有効
        self.show_legend_check.stateChanged.connect(self.onLegendVisibilityChanged)
        table_controls_layout.addWidget(self.show_legend_check)
        
        table_controls_layout.addStretch()
        table_layout.addLayout(table_controls_layout)
        
        self.results_table = QtWidgets.QTableWidget()
        
        # テーブルの設定
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            "Particle No.", "Area (nm²)", "Perimeter (nm)", 
            "Circularity", "Max Height (nm)", "Mean Height (nm)", 
            "Volume (nm³)", "Centroid X (nm)", "Centroid Y (nm)"
        ])
        
        # テーブルの列幅を調整
        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(True)  # 最後の列の自動伸縮を有効化
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)  # 手動リサイズを有効化
        header.setHighlightSections(True)  # セクションのハイライトを有効化
        header.setDefaultAlignment(QtCore.Qt.AlignLeft)  # デフォルトの配置を左寄せに設定
        header.setMinimumSectionSize(50)  # 最小セクションサイズを設定
        
        # 各列の幅を設定（より柔軟な設定）
        column_widths = [80, 100, 120, 80, 120, 120, 120, 120, 120]  # ピクセル単位
        for i, width in enumerate(column_widths):
            self.results_table.setColumnWidth(i, width)
        
        # スクロールバーの設定
        self.results_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.results_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.results_table.setAlternatingRowColors(True)
        
        # テーブルのサイズポリシーを設定（より柔軟に）
        self.results_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        # テーブルウィジェットのサイズポリシーも設定
        table_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        # テーブルに右クリックメニューを追加
        self.results_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.showTableContextMenu)
        
        # テーブルのリサイズイベントを接続
        self.results_table.resizeEvent = self.onTableResize
        
        # テーブルをクリック可能にしてフォーカスを設定しやすくする
        self.results_table.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        # テーブルを直接レイアウトに追加（スクロールエリアを使わない）
        table_layout.addWidget(self.results_table)
        
        # 分割可能なウィジェットに追加
        self.results_splitter.addWidget(stats_widget)
        self.results_splitter.addWidget(table_widget)
        
        # 初期サイズを設定（上部を小さく、下部を大きく）
        self.results_splitter.setSizes([50, 200])
        self.results_splitter.setStretchFactor(0, 0)  # 統計エリアは固定サイズ
        self.results_splitter.setStretchFactor(1, 1)  # テーブルエリアは伸縮可能
        self.results_splitter.setChildrenCollapsible(False)  # 子ウィジェットの折りたたみを無効化
        self.results_splitter.setHandleWidth(8)  # 分割ハンドルの幅を設定
        
        results_layout.addWidget(self.results_splitter)
        results_group.setLayout(results_layout)
        results_content_layout.addWidget(results_group)
        
        # 結果コンテンツウィジェットのサイズポリシーを設定
        results_content.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        results_scroll.setWidget(results_content)
        
        # 結果スクロールエリアのサイズポリシーも設定
        results_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        # 分割可能なウィジェットに追加
        left_splitter.addWidget(param_scroll)
        left_splitter.addWidget(results_scroll)
        
        # 初期サイズを設定（上部を大きく、下部を小さく）
        left_splitter.setSizes([400, 200])
        left_splitter.setStretchFactor(0, 1)  # パラメータエリアは伸縮可能
        left_splitter.setStretchFactor(1, 1)  # 結果エリアも伸縮可能
        left_splitter.setChildrenCollapsible(False)  # 子ウィジェットの折りたたみを無効化
        left_splitter.setHandleWidth(8)  # 分割ハンドルの幅を設定
        
        # 左側のレイアウトに分割可能なウィジェットを追加
        left_layout.addWidget(left_splitter)
        
        # 右側：AFM画像表示エリア
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)
        
        # Matplotlibフィギュアを作成
        self.image_figure = Figure(figsize=(8, 6))
        self.image_canvas = FigureCanvas(self.image_figure)
        self.image_axes = self.image_figure.add_subplot(111)
        
        # リサイズイベントを追加
        self.image_canvas.resizeEvent = self.onCanvasResize
        
        right_layout.addWidget(self.image_canvas)
        
        # マウスイベントを接続
        self.image_canvas.mpl_connect('button_press_event', self.onImageClick)
        
        # メインレイアウトに左右のウィジェットを追加
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        top_layout.addLayout(main_layout)
        
        # 保存された設定を復元
        self.loadWindowSettings()
        
        # データを確実に取得
        if self.getCurrentData():
            # AFM画像を表示
            self.displayAFMImage()
            
            # 復元された設定に応じて適切な処理を実行
            detection_method = self.method_combo.currentText()
            
            if detection_method == "Peak Detection":
                # Peak Detectionが復元された場合
                # Peak Detectionの処理を実行
                self.peakDetection(self.current_data)
                # ピーク位置をオーバーレイ表示
                self.displayPeakOverlay()
            else:
                # Thresholdが復元された場合（デフォルト）
                # 前処理を実行
                processed_data = self.preprocessData(self.current_data)
                # 2値化を実行
                threshold_result = self.thresholdDetection(processed_data)
                if threshold_result is not None:
                    # マスクをオーバーレイ（displayAFMImageは呼ばない）
                    self.updateThresholdPreviewForMethod("Otsu")
                else:
                    pass
        else:
            # データが取得できない場合は空の画像を表示
            print("[WARNING] No data available during setup, displaying empty image")
            self.displayEmptyImage()
            
        # データ処理完了フラグを設定
        self.data_initialized = True
        
        # キーボードショートカットを設定
        self.setupKeyboardShortcuts()
        
        # 前処理UIの初期化
        self.updatePreprocessingUI()
        
    def showHelpDialog(self):
        """Help → Manual でマニュアルを表示（日本語/English 切替可能）"""
        dialog = QtWidgets.QDialog(self)
        dialog.setMinimumSize(500, 500)
        dialog.resize(600, 650)
        layout = QtWidgets.QVBoxLayout(dialog)
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
        lang_row.addWidget(btn_ja)
        lang_row.addWidget(btn_en)
        lang_row.addStretch()
        layout.addLayout(lang_row)
        browser = QtWidgets.QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        css = """
        body { font-size: 15px; line-height: 1.6; }
        p { font-size: 15px; margin: 8px 0; }
        .feature-box { margin: 12px 0; padding: 14px; border: 1px solid #ccc; border-radius: 4px; font-size: 15px; }
        .step { margin: 8px 0; padding: 6px 0; font-size: 15px; }
        .note { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 14px; border-radius: 4px; margin: 14px 0; font-size: 15px; }
        h1 { font-size: 22px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 12px; }
        h2 { font-size: 18px; color: #2c3e50; margin-top: 18px; margin-bottom: 8px; }
        h3 { font-size: 16px; color: #34495e; margin-top: 12px; margin-bottom: 6px; }
        ul { padding-left: 24px; font-size: 15px; }
        li { margin: 4px 0; }
        table.param-table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
        table.param-table th, table.param-table td { border: 1px solid #ddd; padding: 10px 12px; text-align: left; }
        table.param-table th { background-color: #f8f9fa; font-weight: bold; font-size: 14px; }
        table.param-table tr:nth-child(even) { background-color: #f9f9f9; }
        """
        browser.document().setDefaultStyleSheet(css)
        close_btn = QtWidgets.QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        _BTN_SELECTED = "QPushButton { background-color: #007aff; color: white; font-weight: bold; }"
        _BTN_NORMAL = "QPushButton { background-color: #e5e5e5; color: black; }"

        def set_lang(use_ja):
            btn_ja.setChecked(use_ja)
            btn_en.setChecked(not use_ja)
            btn_ja.setStyleSheet(_BTN_SELECTED if use_ja else _BTN_NORMAL)
            btn_en.setStyleSheet(_BTN_SELECTED if not use_ja else _BTN_NORMAL)
            if use_ja:
                browser.setHtml("<html><body>" + HELP_HTML_JA.strip() + "</body></html>")
                dialog.setWindowTitle("粒子解析 - マニュアル")
                close_btn.setText("閉じる")
            else:
                browser.setHtml("<html><body>" + HELP_HTML_EN.strip() + "</body></html>")
                dialog.setWindowTitle("Particle Analysis - Manual")
                close_btn.setText("Close")
        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        layout.addWidget(browser)
        layout.addWidget(close_btn)
        set_lang(False)  # デフォルトは英語
        dialog.exec_()
        
    def setupKeyboardShortcuts(self):
        """キーボードショートカットを設定"""
        try:
            # Ctrl+Cでフォーカスに応じて適切なコピー機能を実行
            copy_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+C"), self)
            copy_shortcut.activated.connect(self.handleCopyShortcut)
            
        except Exception as e:
            print(f"[ERROR] Failed to setup keyboard shortcuts: {e}")
            import traceback
            traceback.print_exc()
    
    def handleCopyShortcut(self):
        """Ctrl+Cショートカットの処理"""
        try:
            # 現在フォーカスされているウィジェットを取得
            focused_widget = QtWidgets.QApplication.focusWidget()
            
            if focused_widget == self.results_table:
                # テーブルにフォーカスがある場合
                # 選択された項目があるかチェック
                selected_ranges = self.results_table.selectedRanges()
                if selected_ranges:
                    # 選択された項目がある場合は選択項目のみをコピー
                    self.copySelectedTableDataToClipboard()
                else:
                    # 選択された項目がない場合は全データをコピー
                    self.copyTableDataToClipboard()
            else:
                # その他の場合は画像をコピー
                self.copyImageToClipboard()
                
        except Exception as e:
            print(f"[ERROR] Failed to handle copy shortcut: {e}")
            import traceback
            traceback.print_exc()
    
    def onThresholdMethodChanged(self, method):
        """閾値法が変更された時の処理"""
        
        self.current_threshold_method = method
        
        self.current_threshold_method = method
        if method == "Manual":
            self.histogram_group.setVisible(True)
            self.manual_thresh_spin.setEnabled(True)
            self.updateHistogram()
        else:
            self.histogram_group.setVisible(False)
            self.manual_thresh_spin.setEnabled(False)
            self.updateThresholdPreviewForMethod(method)
        
    def onDetectionMethodChanged(self, method):
        """検出方法が変更された時の処理"""
        
        # 結果テーブルをクリア（新しい検出方法に切り替わったため）
        self.results_table.setRowCount(0)
        
        # Ring Detection 履歴をクリア
        self.ring_detection_history = []
        
        # すべてのUI要素を初期状態にリセット
        self.threshold_combo.setVisible(False)
        self.manual_thresh_spin.setVisible(False)
        self.manual_thresh_spin.setEnabled(False)
        self.histogram_group.setVisible(False)
        self.peak_group.setVisible(False)
        self.hessian_group.setVisible(False)
        self.ring_group.setVisible(False)
        
        # Particle detection status labelをリセット（メソッド変更時は一旦非表示）
        self.particle_status_label.setVisible(False)
        self.particle_status_label.setText("")
        
        # threshold関連のラベルを非表示
        for i in range(self.detection_layout.count()):
            item = self.detection_layout.itemAt(i)
            if item.widget():
                widget = item.widget()
                if isinstance(widget, QtWidgets.QLabel):
                    if "Threshold:" in widget.text() or "Manual Threshold:" in widget.text():
                        widget.setVisible(False)
            elif item.layout():
                # レイアウト内のウィジェットもチェック
                for j in range(item.layout().count()):
                    sub_item = item.layout().itemAt(j)
                    if sub_item.widget():
                        sub_widget = sub_item.widget()
                        if isinstance(sub_widget, QtWidgets.QLabel):
                            if "Threshold:" in sub_widget.text() or "Manual Threshold:" in sub_widget.text():
                                sub_widget.setVisible(False)
        
        # ピークマーカーをクリア
        if hasattr(self, 'peak_markers'):
            self.peak_markers = None
        
        # AFM Imageを再表示（マスクをクリア）
        # Single ParticleモードでROI選択済みの場合はスキップ（ROI画像を維持）
        if (self.analysis_mode != "Single Particle" or self.roi_data is None):
            if hasattr(self, 'current_data') and self.current_data is not None:
                self.displayAFMImage()
        
        # 選択された方法に応じてUIを設定
        if method == "Peak Detection":
            # テーブルヘッダーを通常の粒子検出用にリセット
            self.resetTableHeadersForParticles()
            
            # Analysis Parametersを通常の粒子検出用にリセット
            self.updateAnalysisParametersForParticleDetection()
            
            # Peak Detection Parametersを表示
            self.peak_group.setVisible(True)
            
            # Modeに応じてROI選択を有効/無効にする
            self.updateROISelectionMode()
            
            # 自動的にPeak Detectionを実行
            # Single ParticleモードでROI選択済み、またはcurrent_dataが存在する場合
            if ((self.analysis_mode == "Single Particle" and self.roi_data is not None) or 
                (hasattr(self, 'current_data') and self.current_data is not None)):
                #print(f"[DEBUG] Detection method changed to Peak Detection")
                # 少し遅延を入れてから検出を実行（UI更新の完了を待つ）
                QtCore.QTimer.singleShot(100, self.onPeakParametersChanged)
                
        elif method == "Hessian Blob":
            # テーブルヘッダーを通常の粒子検出用にリセット
            self.resetTableHeadersForParticles()
            
            # Analysis Parametersを通常の粒子検出用にリセット
            self.updateAnalysisParametersForParticleDetection()
            
            # Hessian Blob Parametersを表示
            self.hessian_group.setVisible(True)
            
            # Modeに応じてROI選択を有効/無効にする
            self.updateROISelectionMode()
            
            # 自動的にHessian Blob検出を実行
            # Single ParticleモードでROI選択済み、またはcurrent_dataが存在する場合
            if ((self.analysis_mode == "Single Particle" and self.roi_data is not None) or 
                (hasattr(self, 'current_data') and self.current_data is not None)):
                #print(f"[DEBUG] Detection method changed to Hessian Blob")
                # 少し遅延を入れてから検出を実行（UI更新の完了を待つ）
                QtCore.QTimer.singleShot(100, self.onHessianParametersChanged)
                
        elif method == "Ring Detection":
            # Ring Detection Parametersを表示
            self.ring_group.setVisible(True)
            
            # Ring Detection用にテーブルヘッダーを変更（Ring No. + 3列）
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels([
                "Ring No.", "Inner Diameter (nm)", "Ring Diameter (nm)", "Circularity"
            ])
            
            # 列幅を調整
            column_widths = [80, 150, 150, 150]
            for i, width in enumerate(column_widths):
                self.results_table.setColumnWidth(i, width)
            
            # Ring Detection用にAnalysis Parametersを変更
            self.updateAnalysisParametersForRingDetection()
            
            # Modeに応じてROI選択を有効/無効にする
            self.updateROISelectionMode()
            
            # 自動的にRing Detectionを実行
            # Single ParticleモードでROI選択済み、またはcurrent_dataが存在する場合
            if ((self.analysis_mode == "Single Particle" and self.roi_data is not None) or 
                (hasattr(self, 'current_data') and self.current_data is not None)):
                QtCore.QTimer.singleShot(100, self.onRingParametersChanged)
                
        else:  # Threshold
            # テーブルヘッダーを通常の粒子検出用にリセット
            self.resetTableHeadersForParticles()
            
            # Analysis Parametersを通常の粒子検出用にリセット
            self.updateAnalysisParametersForParticleDetection()
            
            # threshold関連のUI要素を可視にする
            self.threshold_combo.setVisible(True)
            self.manual_thresh_spin.setVisible(True)
            
            # Modeに応じてROI選択を有効/無効にする
            self.updateROISelectionMode()
            
            # threshold関連のラベルも可視にする
            for i in range(self.detection_layout.count()):
                item = self.detection_layout.itemAt(i)
                if item.widget():
                    widget = item.widget()
                    if isinstance(widget, QtWidgets.QLabel):
                        if "Threshold:" in widget.text() or "Manual Threshold:" in widget.text():
                            widget.setVisible(True)
                elif item.layout():
                    # レイアウト内のウィジェットもチェック
                    for j in range(item.layout().count()):
                        sub_item = item.layout().itemAt(j)
                        if sub_item.widget():
                            sub_widget = sub_item.widget()
                            if isinstance(sub_widget, QtWidgets.QLabel):
                                if "Threshold:" in sub_widget.text() or "Manual Threshold:" in sub_widget.text():
                                    sub_widget.setVisible(True)
            
            # 常にマスクをオーバーレイ
            self.updateOverlay()
            
            # 現在の閾値モードに応じてUIを更新
            current_method = self.threshold_combo.currentText()
            self.onThresholdMethodChanged(current_method)
        
    def detectPeaksOnly(self):
        """ピーク検出のみを実行"""
        try:
            # 現在のデータを取得
            if not self.getCurrentData():
                return
                
            # 前処理
            processed_data = self.preprocessData(self.current_data)
            
            # ピーク検出を実行
            result = self.peakDetection(processed_data)
            
            if result is not None:
                # ピーク検出が成功した場合、オーバーレイを表示
                self.displayPeakOverlay()
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", 
                    "No peaks detected.\nピークが検出されませんでした。")
                    
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", 
                f"Error during peak detection: {str(e)}\nピーク検出中にエラーが発生しました: {str(e)}")
            
    def onPeakParametersChanged(self):
        """ピーク検出パラメーターが変更された時の処理"""
        try:
            # Peak Detectionが選択されている場合のみ更新
            if self.method_combo.currentText() == "Peak Detection":
                #print(f"[DEBUG] Peak parameters changed: min_distance={self.min_peak_distance_spin.value()}, sigma={self.gradient_sigma_spin.value()}, watershed_threshold={self.watershed_threshold_spin.value()}")
                
                # Modeに応じて使用するデータを決定
                if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                    # Single ParticleモードでROIが選択されている場合
                    processed_data = self.preprocessData(self.roi_data)
                    self.filtered_data = processed_data
                elif hasattr(self, 'filtered_data') and self.filtered_data is not None:
                    # filtered_dataが既に存在する場合はそれを使用
                    processed_data = self.filtered_data
                elif hasattr(self, 'current_data') and self.current_data is not None:
                    # filtered_dataがない場合は前処理を実行
                    processed_data = self.preprocessData(self.current_data)
                    self.filtered_data = processed_data  # 結果を保存
                else:
                    print("[ERROR] No data available for peak detection")
                    return
                    
                # Boundary Methodに応じて適切な検出方法を選択
                boundary_method = self.boundary_method_combo.currentText()
                if boundary_method == "Contour Level":
                    result = self.contourLevelFromPeaks(processed_data)
                    # Contour Levelが失敗した場合はWatershedにフォールバック
                    if result is None or np.sum(result) == 0:
                        print("[WARNING] Contour Level failed, falling back to Watershed")
                        result = self.watershedFromPeaks(processed_data)
                else:  # Watershed
                    result = self.watershedFromPeaks(processed_data)
                if result is not None:
                    result = self._apply_edge_exclusion(result)
                    # 検出結果を保存
                    self.detected_particles = result
                    # 粒子特性を再計算
                    if hasattr(self, 'calc_props_check') and self.calc_props_check.isChecked():
                        self.particle_properties = self.calculateParticleProperties()
                    # オーバーレイを更新（粒子境界とピーク位置の両方を表示）
                    self.updateParticleOverlay()
                    #print(f"[DEBUG] Peak parameter update completed - {len(np.unique(result)) - 1} particles")
                else:
                    # 検出に失敗した場合は、ベース画像のみ表示
                    self.displayFilteredImage()
                    print("[WARNING] Peak parameter update failed - no particles detected")
                    
        except Exception as e:
            print(f"[ERROR] Failed to handle peak parameter change: {e}")
            import traceback
            traceback.print_exc()
            
    def onBoundaryMethodChanged(self, method):
        """境界設定方法が変更された時の処理"""
        try:
            #print(f"[DEBUG] Boundary method changed to: {method}")
            
            # Contour Levelが選択された場合はContour Levelパラメータを表示、Watershed thresholdを非表示
            if method == "Contour Level":
                self.contour_level_spin.setVisible(True)
                self.watershed_threshold_spin.setVisible(False)
                # Contour Levelのラベルを表示、Watershed thresholdのラベルを非表示
                for i in range(self.peak_group.layout().count()):
                    item = self.peak_group.layout().itemAt(i)
                    if item.layout():
                        for j in range(item.layout().count()):
                            sub_item = item.layout().itemAt(j)
                            if sub_item.widget():
                                widget = sub_item.widget()
                                if isinstance(widget, QtWidgets.QLabel):
                                    if "Contour Level" in widget.text():
                                        widget.setVisible(True)
                                    elif "Watershed threshold" in widget.text():
                                        widget.setVisible(False)
            else:  # Watershed
                self.contour_level_spin.setVisible(False)
                self.watershed_threshold_spin.setVisible(True)
                # Contour Levelのラベルを非表示、Watershed thresholdのラベルを表示
                for i in range(self.peak_group.layout().count()):
                    item = self.peak_group.layout().itemAt(i)
                    if item.layout():
                        for j in range(item.layout().count()):
                            sub_item = item.layout().itemAt(j)
                            if sub_item.widget():
                                widget = sub_item.widget()
                                if isinstance(widget, QtWidgets.QLabel):
                                    if "Contour Level" in widget.text():
                                        widget.setVisible(False)
                                    elif "Watershed threshold" in widget.text():
                                        widget.setVisible(True)
            
            # ピークパラメータの変更をトリガーして再検出
            self.onPeakParametersChanged()
            
        except Exception as e:
            print(f"[ERROR] Failed to handle boundary method change: {e}")
            import traceback
            traceback.print_exc()
            
    def onHessianParametersChanged(self):
        """Hessian Blob検出パラメータが変更された時の処理"""
        try:
            # Hessian Blobが選択されている場合のみ更新
            if self.method_combo.currentText() == "Hessian Blob":
                #print(f"[DEBUG] Hessian parameters changed: min_sigma={self.hessian_min_sigma_spin.value()}, max_sigma={self.hessian_max_sigma_spin.value()}, threshold={self.hessian_threshold_spin.value()}")
                
                # Modeに応じて使用するデータを決定
                if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                    # Single ParticleモードでROIが選択されている場合
                    processed_data = self.preprocessData(self.roi_data)
                    self.filtered_data = processed_data
                elif hasattr(self, 'filtered_data') and self.filtered_data is not None:
                    # filtered_dataが既に存在する場合はそれを使用
                    processed_data = self.filtered_data
                elif hasattr(self, 'current_data') and self.current_data is not None:
                    # filtered_dataがない場合は前処理を実行
                    processed_data = self.preprocessData(self.current_data)
                    self.filtered_data = processed_data  # 結果を保存
                else:
                    print("[ERROR] No data available for Hessian blob detection")
                    return
                    
                # Hessian Blob検出を実行
                result = self.hessianBlobDetection(processed_data)
                if result is not None:
                    result = self._apply_edge_exclusion(result)
                    # 検出結果を保存
                    self.detected_particles = result
                    # 粒子特性を再計算
                    if hasattr(self, 'calc_props_check') and self.calc_props_check.isChecked():
                        self.particle_properties = self.calculateParticleProperties()
                    # オーバーレイを更新
                    self.updateParticleOverlay()
                    #print(f"[DEBUG] Hessian parameter update completed - {len(np.unique(result)) - 1} particles")
                else:
                    # 検出に失敗した場合は、ベース画像のみ表示
                    self.displayFilteredImage()
                    #print("[WARNING] Hessian parameter update failed - no particles detected")
                    
        except Exception as e:
            print(f"[ERROR] Failed to handle Hessian parameter change: {e}")
            import traceback
            traceback.print_exc()
    
    def onRingParametersChanged(self):
        """Ring Detectionパラメータが変更された時の処理"""
        try:
            # Ring Detectionが選択されている場合のみ更新
            if self.method_combo.currentText() == "Ring Detection":
                # Modeに応じて使用するデータを決定
                if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                    # Single ParticleモードでROIが選択されている場合
                    processed_data = self.preprocessData(self.roi_data)
                    self.filtered_data = processed_data
                elif hasattr(self, 'filtered_data') and self.filtered_data is not None:
                    # filtered_dataが既に存在する場合はそれを使用
                    processed_data = self.filtered_data
                elif hasattr(self, 'current_data') and self.current_data is not None:
                    # filtered_dataがない場合は前処理を実行
                    processed_data = self.preprocessData(self.current_data)
                    self.filtered_data = processed_data  # 結果を保存
                else:
                    print("[ERROR] No data available for ring detection")
                    return
                    
                # Ring Detectionを実行
                result = self.ringDetection(processed_data)
                if result is not None:
                    result = self._apply_edge_exclusion(result)
                    # 検出結果を保存
                    self.detected_particles = result
                    # 粒子特性を再計算
                    if hasattr(self, 'calc_props_check') and self.calc_props_check.isChecked():
                        self.particle_properties = self.calculateParticleProperties()
                    # オーバーレイを更新
                    self.updateParticleOverlay()
                else:
                    # 検出に失敗しても中心表示のために空ラベルでオーバーレイを更新
                    try:
                        zeros = np.zeros_like(processed_data, dtype=np.int32)
                    except Exception:
                        # フォールバック（念のため current_data を使用）
                        base = processed_data if 'processed_data' in locals() else (self.filtered_data if hasattr(self, 'filtered_data') and self.filtered_data is not None else self.current_data)
                        zeros = np.zeros_like(base, dtype=np.int32) if base is not None else None
                    if zeros is not None:
                        self.detected_particles = zeros
                        self.updateParticleOverlay()
                    else:
                        self.displayFilteredImage()
                    
        except Exception as e:
            print(f"[ERROR] Failed to handle ring parameter change: {e}")
            import traceback
            traceback.print_exc()
    
    def onModeChanged(self, mode):
        """解析モードが変更された時の処理"""
        try:
            self.analysis_mode = mode
            
            # Modeに応じてROI選択を有効/無効にする
            self.updateROISelectionMode()
            
            # 「All Frames」ボタンの有効/無効を設定
            # All ParticlesモードでのみAll Framesボタンを有効にする
            if hasattr(self, 'all_frames_button'):
                self.all_frames_button.setEnabled(mode == "All Particles")
            
            # データが利用可能な場合は検出を再実行
            if hasattr(self, 'current_data') and self.current_data is not None:
                detection_method = self.method_combo.currentText()
                if detection_method == "Threshold":
                    QtCore.QTimer.singleShot(100, self.detectParticles)
                elif detection_method == "Peak Detection":
                    QtCore.QTimer.singleShot(100, self.onPeakParametersChanged)
                elif detection_method == "Hessian Blob":
                    QtCore.QTimer.singleShot(100, self.onHessianParametersChanged)
                elif detection_method == "Ring Detection":
                    QtCore.QTimer.singleShot(100, self.onRingParametersChanged)
        except Exception as e:
            print(f"[ERROR] Failed to handle mode change: {e}")
            import traceback
            traceback.print_exc()
    
    def updateROISelectionMode(self):
        """Modeに応じてROI選択モードを有効/無効にする"""
        try:
            if self.analysis_mode == "Single Particle":
                # Single Particleモード: ROI選択を有効にする
                if hasattr(self.main, 'image_window') and self.main.image_window:
                    # 既存の接続を切断してから新しい接続を設定
                    try:
                        self.main.image_window.roi_selected.disconnect(self.onROISelected)
                    except:
                        pass  # 接続が存在しない場合は無視
                    self.main.image_window.roi_selected.connect(self.onROISelected)
                    
                    # ROI選択モードを開始
                    self.main.image_window.set_roi_selection_mode(True, "Rectangle")
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Image View Required",
                        "Image Viewウィンドウが開かれていません。\n"
                        "まずImage Viewウィンドウを開いてください。\n\n"
                        "Image View window is not open.\n"
                        "Please open the Image View window first."
                    )
            else:
                # All Particlesモード: ROI選択を無効にする
                if hasattr(self.main, 'image_window') and self.main.image_window:
                    try:
                        self.main.image_window.set_roi_selection_mode(False)
                        self.main.image_window.roi_selected.disconnect(self.onROISelected)
                    except:
                        pass
                    # ROIデータをクリア
                    self.clearROI()
        except Exception as e:
            print(f"[ERROR] Failed to update ROI selection mode: {e}")
            import traceback
            traceback.print_exc()
    
    def clearROI(self):
        """ROIをクリア"""
        self.roi_rect = None
        self.roi_data = None
        self.roi_scale_info = None
    
    def onROISelected(self, roi_data):
        """ROIが選択された時の処理（全検出方法共通）"""
        try:
            # ROIデータを保存
            if isinstance(roi_data, dict):
                self.roi_rect = roi_data['rect']
                self.roi_shape = roi_data['shape']
            else:
                self.roi_rect = roi_data
                self.roi_shape = "Rectangle"
            
            # ROI内の画像を抽出
            if self.roi_rect and self.roi_rect.width() > 0 and self.roi_rect.height() > 0:
                # 現在のデータからROI内の画像を抽出
                if hasattr(self, 'current_data') and self.current_data is not None:
                    # ImageWindowのmouseReleaseEventでgv.dspimgの座標に変換されている
                    # gv.dspimgの座標をgv.aryDataの座標に変換する必要がある
                    # gv.dspimgのサイズはgv.dspsize、gv.aryDataのサイズはgv.YPixel x gv.XPixel
                    # gv.dspimgはcv2.flip(resized_img, 0)でY軸が反転されているため、Y軸の反転を考慮する必要がある
                    if hasattr(gv, 'dspsize') and gv.dspsize and len(gv.dspsize) == 2:
                        dsp_w, dsp_h = gv.dspsize
                        if hasattr(gv, 'XPixel') and hasattr(gv, 'YPixel') and gv.XPixel > 0 and gv.YPixel > 0:
                            # gv.dspimgの座標をgv.aryDataの座標に変換
                            scale_x = gv.XPixel / dsp_w if dsp_w > 0 else 1.0
                            scale_y = gv.YPixel / dsp_h if dsp_h > 0 else 1.0
                            
                            # X軸はそのまま変換
                            roi_x = int(self.roi_rect.x() * scale_x)
                            roi_w = int(self.roi_rect.width() * scale_x)
                            
                            # Y軸は反転を考慮して変換
                            # gv.dspimgではY=0が下、Y=height-1が上
                            # gv.aryDataではY=0が上、Y=height-1が下
                            # したがって、roi_yは反転する必要がある
                            roi_y_dsp = self.roi_rect.y()
                            roi_h_dsp = self.roi_rect.height()
                            # Y軸反転: dspimg座標系からaryData座標系へ
                            roi_y = int((dsp_h - roi_y_dsp - roi_h_dsp) * scale_y)
                            roi_h = int(roi_h_dsp * scale_y)
                        else:
                            # フォールバック: そのまま使用
                            roi_x = self.roi_rect.x()
                            roi_y = self.roi_rect.y()
                            roi_w = self.roi_rect.width()
                            roi_h = self.roi_rect.height()
                    else:
                        # フォールバック: そのまま使用
                        roi_x = self.roi_rect.x()
                        roi_y = self.roi_rect.y()
                        roi_w = self.roi_rect.width()
                        roi_h = self.roi_rect.height()
                    
                    # 画像の範囲内にクリップ
                    h, w = self.current_data.shape
                    roi_x = max(0, min(roi_x, w - 1))
                    roi_y = max(0, min(roi_y, h - 1))
                    roi_w = min(roi_w, w - roi_x)
                    roi_h = min(roi_h, h - roi_y)
                    
                    if roi_w > 0 and roi_h > 0:
                        # ROIの物理サイズ情報を計算
                        roi_pixels_x = roi_w
                        roi_pixels_y = roi_h
                        total_pixels_x = getattr(gv, 'XPixel', 0)
                        total_pixels_y = getattr(gv, 'YPixel', 0)
                        total_scan_x = getattr(gv, 'XScanSize', 0.0)
                        total_scan_y = getattr(gv, 'YScanSize', 0.0)

                        if total_pixels_x and total_scan_x:
                            roi_scan_size_x = (roi_pixels_x / total_pixels_x) * total_scan_x
                        else:
                            roi_scan_size_x = float(roi_pixels_x)

                        if total_pixels_y and total_scan_y:
                            roi_scan_size_y = (roi_pixels_y / total_pixels_y) * total_scan_y
                        else:
                            roi_scan_size_y = float(roi_pixels_y)

                        self.roi_scale_info = {
                            'scan_size_x': roi_scan_size_x,
                            'scan_size_y': roi_scan_size_y,
                            'pixels_x': roi_pixels_x,
                            'pixels_y': roi_pixels_y
                        }

                        # ROI内の画像を抽出
                        self.roi_data = self.current_data[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
                        
                        # ROI内の画像をfiltered_dataとして設定（拡大表示）
                        self.filtered_data = self.preprocessData(self.roi_data)
                        
                        # 画像を表示
                        self.displayFilteredImage()
                        
                        # 現在の検出方法に応じて自動的に検出を実行
                        detection_method = self.method_combo.currentText()
                        if detection_method == "Threshold":
                            QtCore.QTimer.singleShot(100, self.detectParticles)
                        elif detection_method == "Peak Detection":
                            QtCore.QTimer.singleShot(100, self.onPeakParametersChanged)
                        elif detection_method == "Hessian Blob":
                            QtCore.QTimer.singleShot(100, self.onHessianParametersChanged)
                        elif detection_method == "Ring Detection":
                            QtCore.QTimer.singleShot(100, self.onRingParametersChanged)
            
            # Single Particleモードの場合は、ROI選択モードを維持して再度ROIを描けるようにする
            if self.analysis_mode == "Single Particle":
                if hasattr(self.main, 'image_window') and self.main.image_window:
                    # 既存のROI表示をクリア
                    self.main.image_window.clear_roi_display()
                    # ROI選択モードを再度有効にする
                    QtCore.QTimer.singleShot(100, lambda: self.main.image_window.set_roi_selection_mode(True, "Rectangle"))
        except Exception as e:
            print(f"[ERROR] Failed to handle ROI selection: {e}")
            import traceback
            traceback.print_exc()
            
    def onEdgeExclusionChanged(self, checked):
        """画像端の粒子除外設定が変更された時の処理"""
        try:
            #print(f"[DEBUG] Edge exclusion changed to: {checked}")
            
            # 現在の検出方法を取得
            detection_method = self.method_combo.currentText()
            
            # データが利用可能で、既に検出が実行されている場合のみ再実行
            if (hasattr(self, 'current_data') and self.current_data is not None and
                hasattr(self, 'detected_particles') and self.detected_particles is not None):
                
                # 前処理済みデータを使用
                if hasattr(self, 'filtered_data') and self.filtered_data is not None:
                    processed_data = self.filtered_data
                else:
                    processed_data = self.preprocessData(self.current_data)
                    self.filtered_data = processed_data
                
                # 検出方法に応じて再実行
                if detection_method == "Threshold":
                    result = self.thresholdDetection(processed_data)
                elif detection_method == "Peak Detection":
                    result = self.watershedFromPeaks(processed_data)
                elif detection_method == "Hessian Blob":
                    result = self.hessianBlobDetection(processed_data)
                else:
                    return
                
                if result is not None:
                    result = self._apply_edge_exclusion(result)
                    self.detected_particles = result
                    # 粒子特性を再計算
                    if hasattr(self, 'calc_props_check') and self.calc_props_check.isChecked():
                        self.particle_properties = self.calculateParticleProperties()
                    # オーバーレイを更新
                    self.updateParticleOverlay()
                    #print(f"[DEBUG] Edge exclusion update completed - {len(np.unique(result)) - 1} particles")
                else:
                    #print("[WARNING] Edge exclusion update failed - no particles detected")
                    # 検出に失敗した場合は、ベース画像のみ表示
                    self.displayFilteredImage()
                    
        except Exception as e:
            print(f"[ERROR] Failed to handle edge exclusion change: {e}")
            import traceback
            traceback.print_exc()
            
    def onParticleSizeChanged(self, value):
        """粒子サイズ設定が変更された時の処理"""
        try:
            #print(f"[DEBUG] Particle size changed to: min={self.min_size_spin.value()}, max={self.max_size_spin.value()}")
            
            # 現在の検出方法を取得
            detection_method = self.method_combo.currentText()
            
            # データが利用可能で、既に検出が実行されている場合のみ再実行
            if (hasattr(self, 'current_data') and self.current_data is not None and
                hasattr(self, 'detected_particles') and self.detected_particles is not None):
                
                # 前処理済みデータを使用
                if hasattr(self, 'filtered_data') and self.filtered_data is not None:
                    processed_data = self.filtered_data
                else:
                    processed_data = self.preprocessData(self.current_data)
                    self.filtered_data = processed_data
                
                # 検出方法に応じて再実行
                if detection_method == "Threshold":
                    result = self.thresholdDetection(processed_data)
                elif detection_method == "Peak Detection":
                    result = self.watershedFromPeaks(processed_data)
                elif detection_method == "Hessian Blob":
                    result = self.hessianBlobDetection(processed_data)
                else:
                    return
                
                if result is not None:
                    result = self._apply_edge_exclusion(result)
                    self.detected_particles = result
                    # 粒子特性を再計算
                    if hasattr(self, 'calc_props_check') and self.calc_props_check.isChecked():
                        self.particle_properties = self.calculateParticleProperties()
                    # オーバーレイを更新
                    self.updateParticleOverlay()
                    #print(f"[DEBUG] Particle size update completed - {len(np.unique(result)) - 1} particles")
                else:
                    print("[WARNING] Particle size update failed - no particles detected")
                    # 検出に失敗した場合は、ベース画像のみ表示
                    self.displayFilteredImage()
                    
        except Exception as e:
            print(f"[ERROR] Failed to handle particle size change: {e}")
            import traceback
            traceback.print_exc()
            
    def onOverlayChanged(self, state):
        """Overlayチェックボックスの状態が変更された時の処理（無効化）"""
        # Overlayチェックボックスは使用しないため、このメソッドは無効化
        pass
        
    def onPreprocessingChanged(self):
        """前処理パラメータが変更された時の処理"""
        
        if self.sender() == self.rolling_radius_spin:
            #print(f"[DEBUG] Called by Rolling Radius spinbox!")
            pass
        elif self.sender() == self.smooth_param_spin:
            #print(f"[DEBUG] Called by Smooth Parameter spinbox!")
            pass
        else:
            #print(f"[DEBUG] Called by unknown sender: {self.sender()}")
            if self.sender() is not None:
                #print(f"[DEBUG] Sender object name: {self.sender().objectName()}")
                #print(f"[DEBUG] Sender class name: {self.sender().__class__.__name__}")
                pass
       
        #print(f"[DEBUG] ===== onPreprocessingChanged processing started =====")
        #print(f"[DEBUG] About to start the main processing block")
        try:
           
            self.updatePreprocessingUI()
           
            # 前処理を実行（Modeに応じて元データを選択）
           
            try:
                # Modeに応じて使用するデータを決定
                if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                    # Single ParticleモードでROIが選択されている場合
                    #print(f"[DEBUG] onPreprocessingChanged - Using roi_data for preprocessing")
                    processed_data = self.preprocessData(self.roi_data)
                elif hasattr(self, 'current_data') and self.current_data is not None:
                    #print(f"[DEBUG] onPreprocessingChanged - Using current_data for preprocessing")
                    processed_data = self.preprocessData(self.current_data)
                else:
                    # current_dataがない場合はgv.aryDataを使用
                    if hasattr(gv, 'aryData') and gv.aryData is not None:
                        #print(f"[DEBUG] onPreprocessingChanged - Using gv.aryData for preprocessing")
                        processed_data = self.preprocessData(gv.aryData)
                    else:
                        print(f"[ERROR] No data available for preprocessing")
                        return
                
            except Exception as e:
                print(f"[ERROR] Exception in preprocessData call: {e}")
                import traceback
                traceback.print_exc()
                return
             
            if processed_data is not None:
             
                if hasattr(self, 'filtered_data') and self.filtered_data is not None:
                    data_changed = not np.array_equal(self.filtered_data, processed_data)
                    
                self.filtered_data = processed_data.copy()
                
                # Background Subtraction変更時は検出結果をリセット
                self.detected_particles = None
                self.particle_properties = None
                if hasattr(self, 'peak_markers'):
                    self.peak_markers = None
                 
                if self.filtered_data is not None:
                    #print(f"[DEBUG] filtered_data shape: {self.filtered_data.shape}")
                    #print(f"[DEBUG] filtered_data range: {np.min(self.filtered_data):.3f} to {np.max(self.filtered_data):.3f}")
                    pass
                try:
                    #print(f"[DEBUG] Calling displayFilteredImage...")
                    self.displayFilteredImage()
                    #print(f"[DEBUG] displayFilteredImage completed successfully")
                except Exception as e:
                    print(f"[ERROR] Exception in displayFilteredImage: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 現在の検出方法を適用（プレビューのみ）
                #print(f"[DEBUG] About to apply detection method")
                self.applyDetectionMethod()
                #print(f"[DEBUG] applyDetectionMethod completed")
                
                # 現在の検出方法に応じて即座に検出を再実行
                detection_method = self.method_combo.currentText()
                if detection_method == "Peak Detection":
                    # 少し遅延を入れてから検出を再実行（前処理の完了を待つ）
                    QtCore.QTimer.singleShot(100, self.onPeakParametersChanged)
                elif detection_method == "Hessian Blob":
                    # 少し遅延を入れてから検出を再実行（前処理の完了を待つ）
                    QtCore.QTimer.singleShot(100, self.onHessianParametersChanged)
                
                # 粒子検出は実行しない（Detect/All Framesボタンでのみ実行）
                #print(f"[DEBUG] ===== onPreprocessingChanged completed =====")
                #print(f"[DEBUG] ===== onPreprocessingChanged SUCCESSFULLY COMPLETED =====")
                #print(f"[DEBUG] ===== onPreprocessingChanged EXITING METHOD =====")
            else:
                #print(f"[DEBUG] ===== ENTERING processed_data is None block =====")
                print(f"[ERROR] Preprocessing failed - processed_data is None")
                return
                
        except Exception as e:
            print(f"[ERROR] Exception in onPreprocessingChanged: {e}")
            import traceback
            traceback.print_exc()
    
    def updatePreprocessingUI(self):
        """前処理UIの表示/非表示を更新"""
        try:
            #print(f"[DEBUG] ===== updatePreprocessingUI called =====")
            #print(f"[DEBUG] Background method: {self.bg_combo.currentText()}")
            #print(f"[DEBUG] Smoothing method: {self.smooth_combo.currentText()}")
            
            # 背景除去方法に応じてRolling Ballパラメーターの表示/非表示を切り替え
            bg_method = self.bg_combo.currentText()
            if bg_method == "Rolling Ball":
                self.rolling_radius_label.setVisible(True)
                self.rolling_radius_spin.setVisible(True)
            else:
                self.rolling_radius_label.setVisible(False)
                self.rolling_radius_spin.setVisible(False)
            
            # スムージング方法に応じてパラメーターの設定と表示/非表示を切り替え
            smooth_method = self.smooth_combo.currentText()
            #print(f"[DEBUG] Setting visibility for smoothing method: {smooth_method}")
            
            if smooth_method == "Gaussian":
                # Gaussianが選択された場合、Gaussian Sigmaを表示
                #print(f"[DEBUG] Setting up Gaussian Sigma control")
                self.smooth_param_label.setText("Gaussian Sigma:")
                self.smooth_param_spin.setDecimals(1)
                self.smooth_param_spin.setRange(0.1, 3.0)
                self.smooth_param_spin.setSingleStep(0.1)
                # 現在の値が範囲外の場合は適切な値に調整
                current_value = self.smooth_param_spin.value()
                #print(f"[DEBUG] Current value before adjustment: {current_value}")
                if current_value < 0.1 or current_value > 3.0:
                    self.smooth_param_spin.setValue(1.0)
                    #print(f"[DEBUG] Value adjusted to 1.0")
                # スピンボックスを有効化して確実に操作可能にする
                self.smooth_param_spin.setEnabled(True)
                self.smooth_param_label.setVisible(True)
                self.smooth_param_spin.setVisible(True)
                #print(f"[DEBUG] Gaussian Sigma control setup completed")
                #print(f"[DEBUG] Final value: {self.smooth_param_spin.value()}")
                #print(f"[DEBUG] Range: {self.smooth_param_spin.minimum()} - {self.smooth_param_spin.maximum()}")
                #print(f"[DEBUG] Step: {self.smooth_param_spin.singleStep()}")
                #print(f"[DEBUG] Enabled: {self.smooth_param_spin.isEnabled()}")
                #print(f"[DEBUG] Visible: {self.smooth_param_spin.isVisible()}")
                #print(f"[DEBUG] Gaussian selected - showing Gaussian Sigma control")
                
            elif smooth_method == "Median":
                # Medianが選択された場合、NxNサイズを表示
                self.smooth_param_label.setText("NxN Size:")
                self.smooth_param_spin.setDecimals(0)
                self.smooth_param_spin.setRange(1, 9)
                self.smooth_param_spin.setSingleStep(2)
                # 現在の値が範囲外の場合は適切な値に調整
                current_value = self.smooth_param_spin.value()
                if current_value < 1 or current_value > 9 or current_value % 2 == 0:
                    self.smooth_param_spin.setValue(3)
                # スピンボックスを有効化して確実に操作可能にする
                self.smooth_param_spin.setEnabled(True)
                self.smooth_param_label.setVisible(True)
                self.smooth_param_spin.setVisible(True)
                #print(f"[DEBUG] Median selected - showing NxN Size control")
                
            else:
                # Noneが選択された場合、非表示・無効化
                self.smooth_param_label.setVisible(False)
                self.smooth_param_spin.setVisible(False)
                self.smooth_param_spin.setEnabled(False)
                #print(f"[DEBUG] None selected - hiding smoothing control")
            
            #print(f"[DEBUG] ===== updatePreprocessingUI completed =====")
                
        except Exception as e:
            print(f"[ERROR] Failed to update preprocessing UI: {e}")
            import traceback
            traceback.print_exc()
        
    def onSmoothingParameterChanged(self, value):
        """Smoothingパラメーターが変更された時の処理"""
        smooth_method = self.smooth_combo.currentText()
        #print(f"[DEBUG] ===== Smoothing parameter changed to: {value} =====")
        #print(f"[DEBUG] Current smooth method: {smooth_method}")
        
        # パラメーターが変更された場合は前処理を更新
        self.onPreprocessingChanged()
        
        # 現在の検出方法に応じて即座に検出を再実行
        detection_method = self.method_combo.currentText()
        if detection_method == "Peak Detection":
            # 少し遅延を入れてから検出を再実行（前処理の完了を待つ）
            QtCore.QTimer.singleShot(100, self.onPeakParametersChanged)
        elif detection_method == "Hessian Blob":
            # 少し遅延を入れてから検出を再実行（前処理の完了を待つ）
            QtCore.QTimer.singleShot(100, self.onHessianParametersChanged)
        elif detection_method == "Ring Detection":
            # 少し遅延を入れてから検出を再実行（前処理の完了を待つ）
            QtCore.QTimer.singleShot(100, self.onRingParametersChanged)
        #print(f"[DEBUG] onPreprocessingChanged completed")
        #print(f"[DEBUG] ===== Smoothing parameter change completed =====")
    

            
    def applyDetectionMethod(self):
        """現在の検出方法を適用"""
        try:
            #print(f"[DEBUG] ===== applyDetectionMethod called =====")
            
            # Modeに応じて使用するデータを決定
            if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                # Single ParticleモードでROIが選択されている場合は常にROIデータで更新
                self.filtered_data = self.preprocessData(self.roi_data)
            elif self.analysis_mode == "All Particles" or not hasattr(self, 'filtered_data') or self.filtered_data is None:
                # filtered_dataが存在しない場合は初期化
                #print(f"[DEBUG] applyDetectionMethod - filtered_data not available, initializing")
                if hasattr(self, 'current_data') and self.current_data is not None:
                    self.filtered_data = self.preprocessData(self.current_data)  # 前処理を実行
                    #print(f"[DEBUG] applyDetectionMethod - filtered_data initialized from current_data")
                elif hasattr(gv, 'aryData') and gv.aryData is not None:
                    self.filtered_data = self.preprocessData(gv.aryData)  # 前処理を実行
                    #print(f"[DEBUG] applyDetectionMethod - filtered_data initialized from gv.aryData")
                else:
                    print(f"[ERROR] applyDetectionMethod - No data available")
                    return
            
            detection_method = self.method_combo.currentText()
            #print(f"[DEBUG] Detection method: {detection_method}")
            
            if detection_method == "Threshold":
                threshold_method = self.threshold_combo.currentText()
                #print(f"[DEBUG] Threshold method: {threshold_method}")
                if threshold_method == "Manual":
                    self.histogram_group.setVisible(True)
                    self.manual_thresh_spin.setEnabled(True)
                    self.updateHistogram()
                else:
                    self.histogram_group.setVisible(False)
                    self.manual_thresh_spin.setEnabled(False)
                    self.updateThresholdPreviewForMethod(threshold_method)
            elif detection_method == "Peak Detection":
                # Peak Detectionの場合はModeに応じてデータを使用
                if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                    # Single ParticleモードでROIが選択されている場合
                    processed_data = self.preprocessData(self.roi_data)
                else:
                    # All ParticlesモードまたはROIが選択されていない場合
                    processed_data = self.filtered_data if self.filtered_data is not None else self.current_data
                
                result = self.peakDetection(processed_data)
                if result is not None:
                    self.displayPeakOverlay()
                else:
                    self.displayFilteredImage()
            elif detection_method == "Hessian Blob":
                # Hessian Blobの場合はModeに応じてデータを使用
                if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                    # Single ParticleモードでROIが選択されている場合
                    processed_data = self.preprocessData(self.roi_data)
                else:
                    # All ParticlesモードまたはROIが選択されていない場合
                    processed_data = self.filtered_data if self.filtered_data is not None else self.current_data
                
                result = self.hessianBlobDetection(processed_data)
                if result is not None:
                    result = self._apply_edge_exclusion(result)
                    self.detected_particles = result
                    self.updateParticleOverlay()
                else:
                    self.displayFilteredImage()
            elif detection_method == "Ring Detection":
                # Ring Detectionの場合はModeに応じてデータを使用
                if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                    # Single ParticleモードでROIが選択されている場合
                    processed_data = self.preprocessData(self.roi_data)
                else:
                    # All ParticlesモードまたはROIが選択されていない場合
                    processed_data = self.preprocessData(self.filtered_data if self.filtered_data is not None else self.current_data)
                
                result = self.ringDetection(processed_data)
                if result is not None:
                    result = self._apply_edge_exclusion(result)
                    self.detected_particles = result
                else:
                    self.detected_particles = np.zeros_like(processed_data, dtype=np.int32)
                self.updateParticleOverlay()
            else:
                #print(f"[DEBUG] Unknown detection method: {detection_method}")
                pass
            #print(f"[DEBUG] ===== applyDetectionMethod completed =====")
                
        except Exception as e:
            print(f"[ERROR] Exception in applyDetectionMethod: {e}")
            import traceback
            traceback.print_exc()
            
    def displayFilteredImage(self):
        """filtered_dataを表示（Background Subtraction適用済み）"""
        try:
            if self.filtered_data is None:
                #print(f"[DEBUG] displayFilteredImage - filtered_data is None, preprocessing...")
                # filtered_dataが存在しない場合は前処理を実行
                # Modeに応じて使用するデータを決定
                if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                    # Single ParticleモードでROIが選択されている場合
                    self.filtered_data = self.preprocessData(self.roi_data)
                elif hasattr(self, 'current_data') and self.current_data is not None:
                    # All ParticlesモードまたはROI未選択の場合
                    self.filtered_data = self.preprocessData(self.current_data)
                else:
                    print(f"[ERROR] displayFilteredImage - No current_data available")
                    return
            
            #print(f"[DEBUG] displayFilteredImage - filtered_data range: {np.min(self.filtered_data):.3f} to {np.max(self.filtered_data):.3f}")
            
            # 安全なオーバーレイ削除
            for artist in self.overlay_artists[:]:  # リストのコピーを作成
                try:
                    if hasattr(artist, 'remove'):
                        artist.remove()
                except (NotImplementedError, ValueError) as e:
                    # 削除できないアーティストは無視
                    pass
            self.overlay_artists.clear()  # リストをクリア

            # 軸をクリア（フィギュア全体ではなく）
            self.image_axes.clear()
            
            # 検出結果がリセットされた場合はオーバーレイもクリア
            #if not hasattr(self, 'detected_particles') or self.detected_particles is None:
                #print(f"[DEBUG] displayFilteredImage - No detected particles, clearing overlay")
            
            # 物理サイズを取得
            # Single ParticleモードでROIが選択されている場合はROIの物理サイズを使用
            if (self.analysis_mode == "Single Particle" and 
                self.roi_scale_info is not None and 
                hasattr(self, 'filtered_data') and self.filtered_data is not None):
                # ROIの物理サイズを使用
                scan_size_x = self.roi_scale_info['scan_size_x']
                scan_size_y = self.roi_scale_info['scan_size_y']
                # ROIのアスペクト比を計算
                roi_aspect_ratio = scan_size_y / scan_size_x if scan_size_x > 0 else 1.0
            else:
                # 全画面の物理サイズを使用
                scan_size_x = self.scan_size_x
                scan_size_y = self.scan_size_y
                roi_aspect_ratio = None
            
            # filtered_dataを表示（Background Subtraction適用済み）
            self.image_axes.imshow(self.filtered_data, cmap='viridis', 
                                      extent=[0, scan_size_x, 0, scan_size_y],
                                      aspect='equal', origin='lower')
            
            # ROIのアスペクト比に合わせてaxesのボックスアスペクト比を設定
            if roi_aspect_ratio is not None:
                try:
                    self.image_axes.set_box_aspect(roi_aspect_ratio)
                except:
                    # set_box_aspectがサポートされていない場合は無視
                    pass
            else:
                # 全画面表示の場合は1:1のアスペクト比
                try:
                    self.image_axes.set_box_aspect(1.0)
                except:
                    pass
            
            # 軸ラベルを再設定
            self.image_axes.set_xlabel('X (nm)')
            self.image_axes.set_ylabel('Y (nm)')
            self.image_axes.set_title('AFM Image (Background Subtracted)')

            # 画像を描画して画面に反映
            self.image_canvas.draw()
            
        except Exception as e:
            print(f"[ERROR] Failed to display filtered image: {e}")
            import traceback
            traceback.print_exc()

    def displayProcessedImage(self, processed_data):
        """前処理後の画像を表示（後方互換性のため残す）"""
        try:
            #print(f"[DEBUG] ===== displayProcessedImage called =====")
            #print(f"[DEBUG] Processed data shape: {processed_data.shape}")
            #print(f"[DEBUG] Processed data range: {np.min(processed_data):.3f} to {np.max(processed_data):.3f}")
            
            # フィギュアをクリア
            #print(f"[DEBUG] Clearing image figure")
            self.image_figure.clear()
            self.image_axes = self.image_figure.add_subplot(111)
            
            # 物理サイズを取得
            scan_size_x = self.scan_size_x
            scan_size_y = self.scan_size_y
            #print(f"[DEBUG] Scan size: X={scan_size_x} nm, Y={scan_size_y} nm")
            
            # 処理後の画像を表示
            #print(f"[DEBUG] Creating imshow with processed data")
            im = self.image_axes.imshow(processed_data, cmap='viridis', 
                                      extent=[0, scan_size_x, 0, scan_size_y],
                                      aspect='equal', origin='lower')
            
            # 軸ラベルを設定
            #print(f"[DEBUG] Setting axis labels")
            self.image_axes.set_xlabel('X (nm)')
            self.image_axes.set_ylabel('Y (nm)')
            self.image_axes.set_title('AFM Image (Processed)')
            
            # 画像を更新
            #print(f"[DEBUG] Drawing canvas")
            self.image_canvas.draw()
            
            # カラーバーのサイズを調整
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                try:
                    self.colorbar.set_size(0.046)
                except (KeyError, ValueError):
                    # カラーバーが無効な場合は無視
                    pass
            self.image_figure.tight_layout()
            #print(f"[DEBUG] ===== displayProcessedImage completed =====")
            
        except Exception as e:
            print(f"[ERROR] Failed to display processed image: {e}")
            import traceback
            traceback.print_exc()
            
    def onManualThresholdChanged(self, value):
        """Manual threshold値が変更された時の処理"""
        
        # ヒストグラムの閾値線を更新
        if self.threshold_line is not None:
            self.threshold_line.set_xdata([value, value])  # 配列として渡す
            self.hist_canvas.draw()
        
        # 常にプレビューを更新
        self.updateThresholdPreview(value)
        
    def updateHistogram(self):
        """ヒストグラムを更新"""
        try:
            if not hasattr(self, 'filtered_data') or self.filtered_data is None:
                return
                
            # ヒストグラムフィギュアをクリア
            self.hist_figure.clear()
            self.hist_axes = self.hist_figure.add_subplot(111)
            
            # マージンを設定してラベルがはみ出さないようにする
            self.hist_figure.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
            
            # データを絶対値で使用（AFMの高さ情報）
            data_min = np.min(self.filtered_data)
            data_max = np.max(self.filtered_data)
            
            # ヒストグラムを計算（絶対値で）
            hist, bins, _ = self.hist_axes.hist(self.filtered_data.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            self.hist_data = (hist, bins)
            
            # 現在の閾値を絶対値で使用
            current_threshold = self.manual_thresh_spin.value()
            # 閾値がデータ範囲内にあるかチェック
            if current_threshold < data_min or current_threshold > data_max:
                # 範囲外の場合は中央値に設定
                current_threshold = data_min + (data_max - data_min) * 0.5
                self.manual_thresh_spin.setValue(current_threshold)
            
            # 閾値線を描画（絶対値で）
            self.threshold_line = self.hist_axes.axvline(current_threshold, color='red', linestyle='--', linewidth=2)
            
            # 軸ラベルを設定（絶対値）
            self.hist_axes.set_xlabel('AFM Height (nm)')
            self.hist_axes.set_ylabel('')  # 縦軸ラベルを削除
            self.hist_axes.set_title('')   # タイトルを削除
            
            # Y軸の範囲を調整して見やすくする
            self.hist_axes.set_ylim(0, np.max(hist) * 1.1)
            
            # 縦軸の数値を削除
            self.hist_axes.set_yticks([])
            
            # グリッドを追加
            self.hist_axes.grid(True, alpha=0.3)
            
            # マウスイベントを接続
            self.hist_canvas.mpl_connect('button_press_event', self.onHistogramClick)
            self.hist_canvas.mpl_connect('motion_notify_event', self.onHistogramDrag)
            self.hist_canvas.mpl_connect('button_release_event', self.onHistogramRelease)
            
            # データの範囲を保存
            self.data_min = data_min
            self.data_max = data_max
            
            # スピンボックスの範囲を調整（絶対値で）
            self.manual_thresh_spin.setRange(data_min, data_max)
            
            # 現在の閾値を保持（範囲外の場合は中央値に設定）
            current_threshold = self.manual_thresh_spin.value()
            if current_threshold < data_min or current_threshold > data_max:
                # 範囲外の場合は中央値に設定
                new_threshold = data_min + (data_max - data_min) * 0.5
                self.manual_thresh_spin.setValue(new_threshold)
            
            self.hist_canvas.draw()
            
        except Exception as e:
            print(f"[ERROR] Failed to update histogram: {e}")
            import traceback
            traceback.print_exc()
            
    def onHistogramClick(self, event):
        """ヒストグラムクリック時の処理"""
        if event.inaxes == self.hist_axes:
            self.dragging = True
            self.updateThresholdFromHistogram(event.xdata)
            
    def onHistogramDrag(self, event):
        """ヒストグラムドラッグ時の処理"""
        if hasattr(self, 'dragging') and self.dragging and event.inaxes == self.hist_axes:
            self.updateThresholdFromHistogram(event.xdata)
            
    def onHistogramRelease(self, event):
        """ヒストグラムリリース時の処理"""
        self.dragging = False
        
    def updateThresholdFromHistogram(self, x_value):
        """ヒストグラムから閾値を更新（絶対値で）"""
        if x_value is not None:
            # 絶対値で直接使用
            if hasattr(self, 'data_min') and hasattr(self, 'data_max'):
                # データ範囲内に制限
                threshold_value = np.clip(x_value, self.data_min, self.data_max)
                
                # スピンボックスの値を更新（信号を一時的に無効化）
                self.manual_thresh_spin.blockSignals(True)
                self.manual_thresh_spin.setValue(float(threshold_value))
                self.manual_thresh_spin.blockSignals(False)
                
                # ヒストグラムの閾値線を更新
                if self.threshold_line is not None:
                    self.threshold_line.set_xdata([threshold_value, threshold_value])  # 配列として渡す
                    self.hist_canvas.draw()
                
                # プレビューを更新
                self.updateThresholdPreview(threshold_value)
        
    def updateThresholdPreviewForMethod(self, method):
        """閾値法に応じたプレビューを更新"""
        #print(f"[DEBUG] ===== updateThresholdPreviewForMethod - Starting with method: {method} =====")
        #print(f"[DEBUG] updateThresholdPreviewForMethod - Filtered data shape: {self.filtered_data.shape if hasattr(self, 'filtered_data') and self.filtered_data is not None else 'None'}")
        #print(f"[DEBUG] updateThresholdPreviewForMethod - Filtered data range: {np.min(self.filtered_data):.3f} to {np.max(self.filtered_data):.3f}" if hasattr(self, 'filtered_data') and self.filtered_data is not None else 'None')
        #print(f"[DEBUG] updateThresholdPreviewForMethod - Filtered data statistics:")
        
        
        try:
                 
            
            self.displayFilteredImage()
            
            
            if method == "Otsu":
                threshold_value = filters.threshold_otsu(self.filtered_data)
            elif method == "Adaptive":
                threshold_value = filters.threshold_local(self.filtered_data, block_size=35)
                #print(f"[DEBUG] updateThresholdPreviewForMethod - Adaptive threshold calculated")
            else:
                #print(f"[DEBUG] updateThresholdPreviewForMethod - Unknown method, returning")
                return  # Manualの場合は別のメソッドで処理
                
            #print(f"[DEBUG] updateThresholdPreviewForMethod - Creating mask for threshold: {threshold_value}")
            # 閾値以下の領域を黄色でオーバーレイ
            mask = self.filtered_data <= threshold_value
            if self.exclude_edge_check.isChecked():
                from skimage import measure
                labels_mask = measure.label(mask)
                labels_mask = self._apply_edge_exclusion(labels_mask)
                mask = labels_mask > 0
            #print(f"[DEBUG] updateThresholdPreviewForMethod - Mask shape: {mask.shape}")
            #print(f"[DEBUG] updateThresholdPreviewForMethod - Mask sum: {np.sum(mask)}")
            #print(f"[DEBUG] updateThresholdPreviewForMethod - Mask percentage: {np.sum(mask) / mask.size * 100:.1f}%")
            
            # 物理サイズを取得
            scan_size_x = self.scan_size_x
            scan_size_y = self.scan_size_y
            
            #print(f"[DEBUG] updateThresholdPreviewForMethod - Scan size: X={scan_size_x}, Y={scan_size_y}")
            
            # マスクを浮動小数点に変換
            mask_float = mask.astype(float)
            
            #print("[DEBUG] updateThresholdPreviewForMethod - Adding mask overlay")
            # 閾値以下の領域を赤色でオーバーレイ表示（透明度を下げてAFM画像が見えるように）
            mask_overlay = self.image_axes.imshow(mask_float, cmap='Reds', alpha=0.15, 
                                 extent=[0, scan_size_x, 0, scan_size_y],
                                 aspect='equal', origin='lower')
            self.overlay_artists.append(mask_overlay)  # ←追加
            
            # 閾値の等高線を表示（物理サイズで）
            from skimage import measure
            from skimage.segmentation import watershed
            if method == "Adaptive":
                # Adaptive thresholdの場合は等高線を表示しない
                pass
            else:
                # OtsuとManualの場合は等高線を表示
                try:
                    contours = measure.find_contours(self.filtered_data, threshold_value)
                    #print(f"[DEBUG] updateThresholdPreviewForMethod - Found {len(contours)} contours")
                    
                    for i, contour in enumerate(contours):
                        # ピクセル座標を物理座標に変換
                        x_coords = contour[:, 1] * scan_size_x / self.filtered_data.shape[1]
                        y_coords = contour[:, 0] * scan_size_y / self.filtered_data.shape[0]
                        contour_line, = self.image_axes.plot(x_coords, y_coords, 'r-', linewidth=2)
                        self.overlay_artists.append(contour_line)  # ←追加
                        #print(f"[DEBUG] updateThresholdPreviewForMethod - Added contour {i+1}")
                except Exception as e:
                    # 等高線表示に失敗してもマスクは表示される
                    #print(f"[DEBUG] updateThresholdPreviewForMethod - Contour error: {e}")
                    pass
            
            #print("[DEBUG] updateThresholdPreviewForMethod - Drawing canvas")
            # 画像を更新
            self.image_canvas.draw()
            
            #print("[DEBUG] updateThresholdPreviewForMethod - Adjusting colorbar")
            # カラーバーのサイズを調整
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                self.colorbar.set_size(0.046)
            self.image_figure.tight_layout()
            
            #print("[DEBUG] updateThresholdPreviewForMethod - Complete")
            
        except Exception as e:
            print(f"[ERROR] updateThresholdPreviewForMethod failed: {e}")
            import traceback
            traceback.print_exc()
        
    def updateThresholdPreview(self, threshold_value):
        """閾値プレビューを更新"""
        try:
            if not hasattr(self, 'filtered_data') or self.filtered_data is None:
                return
                
            # filtered_dataを表示
            self.displayFilteredImage()
            
            # 閾値以下の領域を黄色でオーバーレイ
            mask = self.filtered_data <= threshold_value
            if self.exclude_edge_check.isChecked():
                from skimage import measure
                labels_mask = measure.label(mask)
                labels_mask = self._apply_edge_exclusion(labels_mask)
                mask = labels_mask > 0
            
            # 閾値以下の領域を黄色でオーバーレイ表示（物理サイズで）
            # 物理サイズを取得
            scan_size_x = self.scan_size_x
            scan_size_y = self.scan_size_y
            
            # マスクを適切に表示（閾値以下の領域を黄色で）
            # マスクを浮動小数点に変換
            mask_float = mask.astype(float)
            
            # 閾値以下の領域を黄色でオーバーレイ表示
            mask_overlay = self.image_axes.imshow(mask_float, cmap='YlOrRd', alpha=0.3, 
                                 extent=[0, scan_size_x, 0, scan_size_y],
                                 aspect='equal', origin='lower')
            self.overlay_artists.append(mask_overlay)  # ←追加
            
            # 閾値の等高線を表示（物理サイズで）
            from skimage import measure
            try:
                contours = measure.find_contours(self.filtered_data, threshold_value)
                #print(f"[DEBUG] Found {len(contours)} contours for threshold {threshold_value}")
                
                for i, contour in enumerate(contours):
                    # ピクセル座標を物理座標に変換
                    x_coords = contour[:, 1] * scan_size_x / self.filtered_data.shape[1]
                    y_coords = contour[:, 0] * scan_size_y / self.filtered_data.shape[0]
                    contour_line, = self.image_axes.plot(x_coords, y_coords, 'r-', linewidth=2)
                    self.overlay_artists.append(contour_line)  # ←追加
                    #print(f"[DEBUG] Plotted contour {i+1} with {len(contour)} points")
            except Exception as e:
                print(f"[ERROR] Failed to plot contours: {e}")
                # 等高線表示に失敗してもマスクは表示される
                pass
            
            self.image_canvas.draw()
            
            # カラーバーのサイズを調整
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                self.colorbar.set_size(0.046)
            self.image_figure.tight_layout()
            
        except Exception as e:
            pass
        
    def displayAFMImage(self):
        """AFM画像を表示"""
        
        #print("[DEBUG] displayAFMImage - Starting")
        
        if not hasattr(self, 'current_data') or self.current_data is None:
            #print("[DEBUG] displayAFMImage - No current_data available")
            return
            
        try:
            #print("[DEBUG] displayAFMImage - Clearing image axes")
            # 画像をクリア
            self.image_axes.clear()
            
            # スキャンサイズ情報を使用（getCurrentDataで取得済み）
            scan_size_x = self.scan_size_x
            scan_size_y = self.scan_size_y
            x_pixels = self.x_pixels
            y_pixels = self.y_pixels
            
            #print(f"[DEBUG] displayAFMImage - Scan size: X={scan_size_x} nm, Y={scan_size_y} nm")
            #print(f"[DEBUG] displayAFMImage - Pixels: X={x_pixels}, Y={y_pixels}")
            #print(f"[DEBUG] displayAFMImage - Image shape: {self.current_data.shape}")
            
            # スキャンサイズが0の場合はエラー
            if scan_size_x == 0 or scan_size_y == 0:
                print(f"[ERROR] Scan size is 0, cannot display image")
                print(f"[ERROR] Please load AFM data first to get scan size information")
                return
            
            #print("[DEBUG] displayAFMImage - Creating imshow")
            # AFM画像を表示
            im = self.image_axes.imshow(self.current_data, cmap='viridis', 
                                      extent=[0, scan_size_x, 0, scan_size_y],
                                      aspect='equal', origin='lower')
            
            #print("[DEBUG] displayAFMImage - Setting axis labels")
            # 軸ラベルを設定
            self.image_axes.set_xlabel('X (nm)')
            self.image_axes.set_ylabel('Y (nm)')
            self.image_axes.set_title('AFM Image')
            
            #print("[DEBUG] displayAFMImage - Drawing canvas")
            # 画像を更新
            self.image_canvas.draw()
            
            #print("[DEBUG] displayAFMImage - Complete")
            
        except Exception as e:
            print(f"[ERROR] Failed to display AFM image: {e}")
            import traceback
            traceback.print_exc()
    
    def displayEmptyImage(self):
        """データがない場合の空の画像を表示"""
        try:
            # 画像をクリア
            self.image_axes.clear()
            
            # 既存のカラーバーを削除
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                try:
                    self.colorbar.remove()
                except (KeyError, ValueError):
                    # カラーバーが既に削除されている場合は無視
                    pass
                self.colorbar = None
            
            # スキャンサイズが取得できない場合はエラー
            if not hasattr(self, 'scan_size_x') or not hasattr(self, 'scan_size_y'):
                print("[ERROR] Scan size not available for empty image display")
                print("[ERROR] Please load AFM data first to get scan size information")
                return
            scan_size_x = self.scan_size_x
            scan_size_y = self.scan_size_y
            
            # スキャンサイズが0の場合はエラー
            if scan_size_x == 0 or scan_size_y == 0:
                print("[ERROR] Scan size is 0, cannot display empty image")
                print("[ERROR] Please load AFM data first to get scan size information")
                return
                
            # ピクセル数が取得できない場合はエラー
            if not hasattr(self, 'x_pixels') or not hasattr(self, 'y_pixels'):
                print("[ERROR] Pixel count not available for empty image display")
                print("[ERROR] Please load AFM data first to get pixel information")
                return
            x_pixels = self.x_pixels
            y_pixels = self.y_pixels
            
            # ピクセル数が0の場合はエラー
            if x_pixels == 0 or y_pixels == 0:
                print("[ERROR] Pixel count is 0, cannot display empty image")
                print("[ERROR] Please load AFM data first to get pixel information")
                return
            
            # 空の画像を表示（実際のピクセル数を使用）
            empty_data = np.zeros((y_pixels, x_pixels))
            im = self.image_axes.imshow(empty_data, cmap='viridis', 
                                      extent=[0, scan_size_x, 0, scan_size_y],
                                      aspect='equal', origin='lower')
            
            # カラーバーを追加
            self.colorbar = self.image_figure.colorbar(im, ax=self.image_axes)
            self.colorbar.set_label('Height (nm)')
            
            # 軸ラベルを設定
            self.image_axes.set_xlabel('X (nm)')
            self.image_axes.set_ylabel('Y (nm)')
            self.image_axes.set_title('AFM Image (No Data)')
            
            # レイアウトを調整
            self.image_figure.tight_layout()
            
            # 画像を更新
            self.image_canvas.draw()
            
        except Exception as e:
            print(f"[ERROR] Failed to display empty image: {e}")
            import traceback
            traceback.print_exc()
        
    def updateParticleOverlay(self):
        """粒子検出結果を画像上にオーバーレイ"""
        try:
            # 以前のオーバーレイを安全に削除
            for artist in self.overlay_artists:
                try:
                    if artist in self.image_axes.get_children():
                        artist.remove()
                except Exception as e:
                    pass
            self.overlay_artists = []

            is_ring_detection = (self.method_combo.currentText() == "Ring Detection")
            has_detected_array = hasattr(self, 'detected_particles') and self.detected_particles is not None
            has_ring_info = hasattr(self, 'ring_last_avg_info') and self.ring_last_avg_info is not None

            if (not has_detected_array) or (np.max(self.detected_particles) == 0 and not (is_ring_detection and has_ring_info)):
                self.image_canvas.draw()  # オーバーレイが無い場合も描画を更新
                return

            # スキャンサイズ情報とピクセルサイズを決定
            if (self.analysis_mode == "Single Particle" and
                self.roi_scale_info is not None and
                hasattr(self, 'filtered_data') and self.filtered_data is not None):
                scan_size_x = self.roi_scale_info['scan_size_x']
                scan_size_y = self.roi_scale_info['scan_size_y']
                height, width = self.filtered_data.shape
            else:
                scan_size_x = self.scan_size_x
                scan_size_y = self.scan_size_y
                if self.current_data is not None:
                    height, width = self.current_data.shape
                elif hasattr(self, 'filtered_data') and self.filtered_data is not None:
                    height, width = self.filtered_data.shape
                else:
                    return

            if scan_size_x == 0 or scan_size_y == 0:
                return

            pixel_size_x = scan_size_x / width if width else 1.0
            pixel_size_y = scan_size_y / height if height else 1.0
            
            # 各粒子の境界を描画
            regions = measure.regionprops(self.detected_particles)
            
            for i, region in enumerate(regions):
                try:
                    contour = measure.find_contours(self.detected_particles == region.label, 0.5)
                    
                    if len(contour) > 0:
                        if is_ring_detection:
                            contour_list = sorted(contour, key=lambda c: c.shape[0], reverse=True)
                        else:
                            contour_list = [contour[0]]

                        for contour_idx, boundary_coords in enumerate(contour_list):
                            y_coords, x_coords = boundary_coords[:, 0], boundary_coords[:, 1]
                            
                            # Single Particleモードの場合はROIオフセットを適用しない
                            if (hasattr(self, 'roi_rect') and self.roi_rect is not None and 
                                self.analysis_mode != "Single Particle"):
                                from PyQt5.QtCore import QRect
                                if isinstance(self.roi_rect, QRect):
                                    x_offset = self.roi_rect.x()
                                    y_offset = self.roi_rect.y()
                                else:
                                    x_offset, y_offset = self.roi_rect[0], self.roi_rect[1]
                                x_coords += x_offset
                                y_coords += y_offset
                            
                            # ピクセル中心座標を物理座標に変換
                            x_phys = (x_coords + 0.5) * pixel_size_x
                            y_phys = (y_coords + 0.5) * pixel_size_y
                            
                            # 境界線を描画（外側=赤、内側=黄色）
                            line_color = 'r-' if (not is_ring_detection or contour_idx == 0) else 'y-'
                            line, = self.image_axes.plot(x_phys, y_phys, line_color, linewidth=1.5, alpha=0.9)
                            self.overlay_artists.append(line)
                        
                        # 重心を描画し、artistをリストに追加
                        centroid = region.centroid
                        # Single Particleモードの場合はROIオフセットを適用しない
                        if (hasattr(self, 'roi_rect') and self.roi_rect is not None and 
                            self.analysis_mode != "Single Particle"):
                            # QRectオブジェクトの場合とタプルの場合の両方に対応
                            from PyQt5.QtCore import QRect
                            if isinstance(self.roi_rect, QRect):
                                x_offset = self.roi_rect.x()
                                y_offset = self.roi_rect.y()
                            else:
                                x_offset, y_offset = self.roi_rect[0], self.roi_rect[1]
                            centroid = (centroid[0] + y_offset, centroid[1] + x_offset)
                        
                        # ピクセル中心座標を物理座標に変換
                        centroid_x_phys = (centroid[1] + 0.5) * pixel_size_x
                        centroid_y_phys = (centroid[0] + 0.5) * pixel_size_y
                        point, = self.image_axes.plot(centroid_x_phys, centroid_y_phys, 'go', 
                                                      markersize=5, markeredgecolor='black', markeredgewidth=0.5)
                        self.overlay_artists.append(point)
                        
                        # 粒子番号を描画し、artistをリストに追加
                        text = self.image_axes.text(centroid_x_phys, centroid_y_phys, str(i+1), 
                                           color='white', fontsize=8, ha='center', va='center',
                                                    bbox=dict(boxstyle="round,pad=0.2", fc='black', ec='none', alpha=0.6))
                        self.overlay_artists.append(text)
                except Exception as e:
                    print(f"[WARNING] Failed to draw particle {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if (is_ring_detection and 
                hasattr(self, 'ring_last_avg_info') and 
                self.ring_last_avg_info is not None):
                info = self.ring_last_avg_info
                center_x = info.get('center_x')
                center_y = info.get('center_y')
                outer_contour = info.get('outer_contour')
                inner_contour = info.get('inner_contour')
                
                # 中心点にバツ印を表示
                if center_x is not None and center_y is not None:
                    center_x_phys = (center_x + 0.5) * pixel_size_x
                    center_y_phys = (center_y + 0.5) * pixel_size_y
                    
                    # バツ印を描画（上から下、左から右）
                    cross_size_nm = 3.0  # nm単位のバツ印のサイズ
                    cross_size_x = cross_size_nm / pixel_size_x
                    cross_size_y = cross_size_nm / pixel_size_y
                    
                    # 上から下のライン
                    cross_v, = self.image_axes.plot(
                        [center_x_phys, center_x_phys],
                        [center_y_phys - cross_size_y, center_y_phys + cross_size_y],
                        color='c', linewidth=2.0, alpha=0.9
                    )
                    self.overlay_artists.append(cross_v)
                    
                    # 左から右のライン
                    cross_h, = self.image_axes.plot(
                        [center_x_phys - cross_size_x, center_x_phys + cross_size_x],
                        [center_y_phys, center_y_phys],
                        color='c', linewidth=2.0, alpha=0.9
                    )
                    self.overlay_artists.append(cross_h)
                    
                
                # リング検出成功時の輪郭描画（外径は描画しない）
                # 追加: 輪郭に対して楕円フィッティングし、破線で描画するユーティリティ
                def _fit_and_draw_ellipse(contour_np, color_code):
                    try:
                        import cv2
                        if contour_np is None or len(contour_np) < 5:
                            return None
                        # contour: (N,2) with (y,x) → OpenCV expects (x,y)
                        pts_xy = np.ascontiguousarray(np.column_stack((contour_np[:, 1], contour_np[:, 0])).astype(np.float32))
                        ellipse = cv2.fitEllipse(pts_xy)  # ((cx, cy), (w, h), angle_deg)
                        (cx, cy), (w, h), ang = ellipse
                        # パラメトリック生成（物理座標へ変換）
                        t = np.linspace(0, 2*np.pi, 181)
                        a = w / 2.0
                        b = h / 2.0
                        cos_a = np.cos(np.deg2rad(ang))
                        sin_a = np.sin(np.deg2rad(ang))
                        ex = cx + a*np.cos(t)*cos_a - b*np.sin(t)*sin_a
                        ey = cy + a*np.cos(t)*sin_a + b*np.sin(t)*cos_a
                        x_phys_e = (ex + 0.5) * pixel_size_x
                        y_phys_e = (ey + 0.5) * pixel_size_y
                        eline, = self.image_axes.plot(x_phys_e, y_phys_e, linestyle='--', color=color_code, linewidth=1.0, alpha=0.9)
                        self.overlay_artists.append(eline)
                        return True
                    except Exception as _:
                        return None
                
                # 内側の輪郭を黄色で描画
                if inner_contour is not None and len(inner_contour) > 0:
                    y_coords, x_coords = inner_contour[:, 0], inner_contour[:, 1]
                    x_phys = (x_coords + 0.5) * pixel_size_x
                    y_phys = (y_coords + 0.5) * pixel_size_y
                    
                    # 輪郭の始点と終点を接続（閉じた曲線にする）
                    x_phys_closed = np.append(x_phys, x_phys[0])
                    y_phys_closed = np.append(y_phys, y_phys[0])
                    
                    # 内側の輪郭を黄色で描画
                    inner_line, = self.image_axes.plot(x_phys_closed, y_phys_closed, color='y', linewidth=1.5, alpha=0.9)
                    self.overlay_artists.append(inner_line)
                    
                    # 楕円フィット（破線、黄色）
                    _fit_and_draw_ellipse(inner_contour, 'y')
                
                # 中間の紫線を描画
                # 1) 優先: ringDetection で保存した mid_contour（Watershed 由来の輪郭）
                mid_contour_saved = info.get('mid_contour')
                if mid_contour_saved is not None and len(mid_contour_saved) > 0:
                    y_coords, x_coords = mid_contour_saved[:, 0], mid_contour_saved[:, 1]
                    x_phys = (x_coords + 0.5) * pixel_size_x
                    y_phys = (y_coords + 0.5) * pixel_size_y
                    x_phys_closed = np.append(x_phys, x_phys[0])
                    y_phys_closed = np.append(y_phys, y_phys[0])
                    mid_line, = self.image_axes.plot(x_phys_closed, y_phys_closed, color='m', linewidth=1.4, alpha=0.9)
                    self.overlay_artists.append(mid_line)
                    
                    # 楕円フィット（破線、紫）
                    _fit_and_draw_ellipse(mid_contour_saved, 'm')
                # 2) 代替: 外側と内側の中点から生成（mid_contour が無い場合のみ）
                elif outer_contour is not None and len(outer_contour) > 0 and inner_contour is not None and len(inner_contour) > 0:
                    try:
                        # 外側と内側の対応する位置のポイントを正確に計算
                        # 両輪郭を正規化して、同じパラメータでサンプリング
                        from scipy.interpolate import interp1d
                        
                        # 外側の輪郭の曲線パラメータ（弧長に基づく）
                        outer_diffs = np.sqrt(np.sum(np.diff(outer_contour, axis=0)**2, axis=1))
                        outer_cumlen = np.concatenate(([0], np.cumsum(outer_diffs)))
                        if outer_cumlen[-1] > 0:
                            outer_cumlen_normalized = outer_cumlen / outer_cumlen[-1]
                        else:
                            outer_cumlen_normalized = np.linspace(0, 1, len(outer_cumlen))
                        
                        # 内側の輪郭の曲線パラメータ
                        inner_diffs = np.sqrt(np.sum(np.diff(inner_contour, axis=0)**2, axis=1))
                        inner_cumlen = np.concatenate(([0], np.cumsum(inner_diffs)))
                        if inner_cumlen[-1] > 0:
                            inner_cumlen_normalized = inner_cumlen / inner_cumlen[-1]
                        else:
                            inner_cumlen_normalized = np.linspace(0, 1, len(inner_cumlen))
                        
                        # 均一なパラメータでリサンプリング
                        n_samples = max(len(outer_contour), len(inner_contour))
                        param = np.linspace(0, 1, n_samples)
                        
                        # 両輪郭を補間
                        outer_interp = interp1d(outer_cumlen_normalized, outer_contour, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=True)
                        inner_interp = interp1d(inner_cumlen_normalized, inner_contour, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=True)
                        
                        outer_resampled = outer_interp(param)
                        inner_resampled = inner_interp(param)
                        
                        # 中点を計算
                        mid_contour = (outer_resampled + inner_resampled) / 2.0
                        
                        y_coords, x_coords = mid_contour[:, 0], mid_contour[:, 1]
                        x_phys = (x_coords + 0.5) * pixel_size_x
                        y_phys = (y_coords + 0.5) * pixel_size_y
                        
                        # 始点と終点を接続
                        x_phys_closed = np.append(x_phys, x_phys[0])
                        y_phys_closed = np.append(y_phys, y_phys[0])
                        
                        # 中間の輪郭を紫で描画
                        mid_line, = self.image_axes.plot(x_phys_closed, y_phys_closed, color='m', linewidth=1.2, alpha=0.9)
                        self.overlay_artists.append(mid_line)
                    except Exception as e:
                        print(f"[WARNING] Failed to draw middle ring contour: {e}")
                        import traceback
                        traceback.print_exc()
                elif outer_contour is not None and len(outer_contour) > 0:
                    # 内側の輪郭がない場合は外側のみで中間を計算
                    y_coords, x_coords = outer_contour[:, 0], outer_contour[:, 1]
                    x_phys = (x_coords + 0.5) * pixel_size_x
                    y_phys = (y_coords + 0.5) * pixel_size_y
                    
                    x_phys_closed = np.append(x_phys, x_phys[0])
                    y_phys_closed = np.append(y_phys, y_phys[0])
                    
                    mid_line, = self.image_axes.plot(x_phys_closed, y_phys_closed, color='m', linewidth=1.2, alpha=0.9)
                    self.overlay_artists.append(mid_line)
                    
            
            # Peak Detectionの場合、ピーク位置も表示
            if (self.method_combo.currentText() == "Peak Detection" and 
                hasattr(self, 'peak_markers') and self.peak_markers is not None):
                self.displayPeakPositions()
            
            # 画像全体を一度だけ更新
            self.image_canvas.draw()
            
        except Exception as e:
            print(f"[ERROR] Failed to update particle overlay: {e}")
            import traceback
            traceback.print_exc()

    def _apply_edge_exclusion(self, labels):
        """Exclude labels touching image edges when the checkbox is enabled."""
        if labels is None:
            return None
        if not self.exclude_edge_check.isChecked():
            return labels
        if np.max(labels) == 0:
            return labels
        
        labels = labels.copy()
        mask = labels > 0
        from skimage import measure
        labeled_mask = measure.label(mask)
        if labeled_mask.max() == 0:
            return np.zeros_like(labels)
        
        keep_mask = np.zeros_like(mask, dtype=bool)
        nrows, ncols = mask.shape
        for region in measure.regionprops(labeled_mask):
            minr, minc, maxr, maxc = region.bbox
            if minr == 0 or minc == 0 or maxr == nrows or maxc == ncols:
                continue
            keep_mask[labeled_mask == region.label] = True
        
        if not np.any(keep_mask):
            return np.zeros_like(labels)
        
        new_labels = measure.label(keep_mask).astype(labels.dtype)
        # Map back onto original labels to preserve label order if needed
        final_labels = np.zeros_like(labels)
        unique_new = np.unique(new_labels)
        unique_new = unique_new[unique_new != 0]
        for idx, lbl in enumerate(unique_new, start=1):
            final_labels[new_labels == lbl] = idx
        return final_labels
        
    def detectParticles(self):
        """粒子検出を実行"""
        try:
            # 現在のデータを取得
            if not self.getCurrentData():
                return
            
            # Modeに応じて使用するデータを決定
            if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                # Single ParticleモードでROIが選択されている場合
                processed_data = self.preprocessData(self.roi_data)
                self.filtered_data = processed_data.copy()
            else:
                # All ParticlesモードまたはROIが選択されていない場合
                # 常に前処理を実行して、Background Subtractionを適用
                #print(f"[DEBUG] detectParticles - Preprocessing data with background subtraction")
                processed_data = self.preprocessData(self.current_data)
                
                # 前処理後のデータをfiltered_dataとして保存
                self.filtered_data = processed_data.copy()
            
            # 粒子検出
            detected_result = self.detectParticlesFromData(processed_data)
            
            #print(f"[DEBUG] detectParticles - detected_result type: {type(detected_result)}")
            #if detected_result is not None:
                #print(f"[DEBUG] detectParticles - detected_result shape: {detected_result.shape}")
                #print(f"[DEBUG] detectParticles - max label: {np.max(detected_result)}")
                #print(f"[DEBUG] detectParticles - unique labels: {len(np.unique(detected_result))}")
            
            # 検出結果をチェック
            if detected_result is None:
                QtWidgets.QMessageBox.warning(self, "Warning", 
                    "No particles detected.\n粒子が検出されませんでした。")
                return
                
            self.detected_particles = detected_result
            #print(f"[DEBUG] detectParticles - saved detected_particles, max: {np.max(self.detected_particles)}")
            
            # 検出後の閾値を再確認
            new_threshold = self.manual_thresh_spin.value()
            
            # 粒子の特性を計算
            self.particle_properties = self.calculateParticleProperties()
            
            # 結果表示をクリアして更新
            self.displayParticleResults()
            
            # 画像上に粒子検出結果をオーバーレイ
            self.updateParticleOverlay()
            
            # Peak Detectionの場合、ピーク位置も表示
            if (self.method_combo.currentText() == "Peak Detection" and 
                hasattr(self, 'peak_markers') and self.peak_markers is not None):
                self.displayPeakPositions()
            
            # エクスポートボタンとナビゲーションボタンを有効化
            self.export_button.setEnabled(True)
            self.prev_frame_button.setEnabled(True)
            self.next_frame_button.setEnabled(True)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", 
                f"Error during particle detection: {str(e)}\n粒子検出中にエラーが発生しました: {str(e)}")
                
    def detectAllFrames(self):
        """全フレームで粒子検出を実行"""
        try:
            if not hasattr(gv, 'files') or not gv.files:
                QtWidgets.QMessageBox.warning(self, "Warning", 
                    "No files loaded.\nファイルが読み込まれていません。")
                return
                
            if not hasattr(gv, 'FrameNum') or gv.FrameNum <= 0:
                QtWidgets.QMessageBox.warning(self, "Warning", 
                    "No frames available.\nフレームが利用できません。")
                return
                
            # 現在のファイルとフレームを保存
            current_file_num = gv.currentFileNum
            current_frame = gv.index
            
            # 全フレームの結果を保存する辞書
            self.all_frame_results = {}
            
            # プログレスダイアログを作成
            progress = QtWidgets.QProgressDialog("Processing all frames...", "Cancel", 0, gv.FrameNum, self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setAutoClose(True)
            progress.setAutoReset(True)
            
            for frame_idx in range(gv.FrameNum):
                progress.setValue(frame_idx)
                if progress.wasCanceled():
                    break
                    
                # フレームを変更
                gv.index = frame_idx
                self.main.frameSlider.setValue(frame_idx)
                self.main.updateFrame()
                
                # データを取得
                if not self.getCurrentData():
                    continue
                    
                # 各フレームでPeak Detectionを実行
                if hasattr(gv, 'aryData') and gv.aryData is not None:
                    # 1. Peak Detectionを実行
                    if self.method_combo.currentText() == "Peak Detection":
                        # ピーク検出を実行
                        peak_result = self.peakDetection(gv.aryData)
                        if peak_result is not None:
                            self.peak_markers = peak_result
                        else:
                            self.peak_markers = None
                    
                    # 2. Detectと同じ処理を実行
                    # 前処理を実行
                    processed_data = self.preprocessData(gv.aryData)
                    
                    # 粒子検出
                    detected_particles = self.detectParticlesFromData(processed_data)
                    
                    # 粒子の特性を計算
                    if detected_particles is not None and np.max(detected_particles) > 0:
                        self.detected_particles = detected_particles
                        particle_properties = self.calculateParticleProperties()
                        self.all_frame_results[frame_idx] = {
                            'detected_particles': detected_particles,
                            'particle_properties': particle_properties,
                            'original_data': gv.aryData.copy(),
                            'processed_data': processed_data.copy(),
                            'filtered_data': processed_data.copy()  # 前処理されたデータも保存
                        }
                    else:
                        self.all_frame_results[frame_idx] = {
                            'detected_particles': None,
                            'particle_properties': [],
                            'original_data': gv.aryData.copy(),
                            'processed_data': processed_data.copy(),
                            'filtered_data': processed_data.copy()  # 前処理されたデータも保存
                        }
                else:
                    print(f"[ERROR] No original data available for frame {frame_idx}")
                    continue
                    
            progress.setValue(gv.FrameNum)
            
            # 元のフレームに戻す
            gv.currentFileNum = current_file_num
            gv.index = current_frame
            self.main.frameSlider.setValue(current_frame)
            self.main.updateFrame()
            
            # 現在のフレームの結果を表示
            if current_frame in self.all_frame_results:
                self.detected_particles = self.all_frame_results[current_frame]['detected_particles']
                self.particle_properties = self.all_frame_results[current_frame]['particle_properties']
                self.displayParticleResults()
                self.updateParticleOverlay()
                
            # ボタンを有効化
            self.export_button.setEnabled(True)
            self.prev_frame_button.setEnabled(True)
            self.next_frame_button.setEnabled(True)
            
            QtWidgets.QMessageBox.information(self, "Success", 
                f"Particle detection completed for all {gv.FrameNum} frames.\n全{gv.FrameNum}フレームの粒子検出が完了しました。")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", 
                f"Error during all frames detection: {str(e)}\n全フレーム検出中にエラーが発生しました: {str(e)}")
                
    def previousFrame(self):
        """前のフレームに移動"""
        try:
            if not hasattr(self, 'all_frame_results') or not self.all_frame_results:
                QtWidgets.QMessageBox.information(self, "Information", 
                    "Please run 'All Frames' detection first.\n先に'All Frames'検出を実行してください。")
                return
                
            current_frame = gv.index
            if current_frame > 0:
                # 前のフレームに移動
                gv.index = current_frame - 1
                self.main.frameSlider.setValue(gv.index)
                self.main.updateFrame()
                
                # 少し待ってから結果を表示（フレーム更新の完了を待つ）
                QtCore.QTimer.singleShot(100, lambda: self.displayFrameResults(current_frame - 1))
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", 
                f"Error moving to previous frame: {str(e)}\n前のフレームへの移動中にエラーが発生しました: {str(e)}")
                
    def nextFrame(self):
        """次のフレームに移動"""
        try:
            if not hasattr(self, 'all_frame_results') or not self.all_frame_results:
                QtWidgets.QMessageBox.information(self, "Information", 
                    "Please run 'All Frames' detection first.\n先に'All Frames'検出を実行してください。")
                return
                
            current_frame = gv.index
            if current_frame < gv.FrameNum - 1:
                # 次のフレームに移動
                gv.index = current_frame + 1
                self.main.frameSlider.setValue(gv.index)
                self.main.updateFrame()
                
                # 少し待ってから結果を表示（フレーム更新の完了を待つ）
                QtCore.QTimer.singleShot(100, lambda: self.displayFrameResults(current_frame + 1))
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", 
                f"Error moving to next frame: {str(e)}\n次のフレームへの移動中にエラーが発生しました: {str(e)}")
                
    def displayFrameResults(self, frame_idx):
        """指定フレームの結果を表示"""
        try:
            if frame_idx in self.all_frame_results:
                result = self.all_frame_results[frame_idx]
                self.detected_particles = result['detected_particles']
                self.particle_properties = result['particle_properties']
                
                # 現在のフレームのデータを取得
                self.getCurrentData()
                
                # メインウィンドウから直接データを取得して確実に更新
                if hasattr(gv, 'aryData') and gv.aryData is not None:
                    self.current_data = gv.aryData.copy()
                    #print(f"[DEBUG] Updated current_data for frame {frame_idx}: shape={self.current_data.shape}")
                
                # All Frames後の表示なので、前処理は実行せず、保存された粒子情報を直接表示
                # 前処理されたデータは既に保存されているはず
                if 'filtered_data' in result:
                    self.filtered_data = result['filtered_data']
                else:
                    # 念のため、前処理を実行（初回のみ）
                    self.filtered_data = self.preprocessData(None)
                
                # フィルタリングされた画像を表示
                self.displayFilteredImage()
                
                # 結果を表示
                self.displayParticleResults()
                
                # 画像上に粒子検出結果をオーバーレイ（保存された情報を使用）
                if result['detected_particles'] is not None:
                    # All Frames後の表示なので、ピーク位置表示は行わず、粒子領域のみ表示
                    self.updateParticleOverlay()
                    
                # 統計情報を更新
                num_particles = len(self.particle_properties)
                self.stats_label.setText(f"Frame {frame_idx + 1}: {num_particles} particles detected")
                
        except Exception as e:
            print(f"[ERROR] Failed to display frame results: {e}")
            import traceback
            traceback.print_exc()
    
    def onImageClick(self, event):
        """画像クリックイベントハンドラー"""
        if event.button == 3:  # 右クリック
            self.showImageContextMenu(event)
                
    def showImageContextMenu(self, event):
        """画像の右クリックメニューを表示"""
        try:
            # コンテキストメニューを作成
            context_menu = QtWidgets.QMenu(self)
            
            # 粒子が検出されている場合のみ粒子削除メニューを追加
            if hasattr(self, 'detected_particles') and self.detected_particles is not None:
                # クリックされた位置を取得（MatplotlibのMouseEventオブジェクト）
                x_data, y_data = event.xdata, event.ydata
                
                #print(f"[DEBUG] Right click at: x={x_data}, y={y_data}")
                
                
                
                # 物理座標（nm）からピクセル座標に変換
                if (x_data is not None and y_data is not None and
                    hasattr(self, 'current_data') and self.current_data is not None):
                    
                    # スキャンサイズとピクセル数を取得
                    scan_size_x = self.scan_size_x
                    scan_size_y = self.scan_size_y
                    data_width = self.current_data.shape[1]
                    data_height = self.current_data.shape[0]
                    
                   
                    # 物理座標をピクセル座標に変換
                    pixel_x = int((x_data / scan_size_x) * data_width)
                    pixel_y = int((y_data / scan_size_y) * data_height)
                       
                    # ピクセル座標が範囲内かチェック
                    if (0 <= pixel_x < data_width and 0 <= pixel_y < data_height):
                    # 粒子ラベルを取得
                        label = self.detected_particles[pixel_y, pixel_x]
                           
                    if label > 0:  # 背景以外の場合
                        # 削除アクションを追加
                        delete_action = context_menu.addAction("Delete Particle")
                        delete_action.triggered.connect(lambda: self.deleteParticle(label))
                           
            # メニューを表示（マウスカーソルの現在位置を取得）
            cursor = QtGui.QCursor()
            global_pos = cursor.pos()
            context_menu.exec_(global_pos)
                        
        except Exception as e:
            print(f"[ERROR] Failed to show image context menu: {e}")
            import traceback
            traceback.print_exc()
    
    def showTableContextMenu(self, position):
        """テーブルの右クリックメニューを表示"""
        try:
            # コンテキストメニューを作成
            context_menu = QtWidgets.QMenu(self)
            
            # 選択された項目があるかチェック
            selected_ranges = self.results_table.selectedRanges()
            has_selection = len(selected_ranges) > 0
            
            if has_selection:
                # 選択された項目のみをコピーするアクションを追加
                copy_selected_action = context_menu.addAction("Copy Selected Items")
                copy_selected_action.triggered.connect(self.copySelectedTableDataToClipboard)
                
                # セパレーターを追加
                context_menu.addSeparator()
                # 行削除アクション（Ring/Particle共通）
                delete_action = context_menu.addAction("Delete Particle")
                delete_action.triggered.connect(self.deleteSelectedResultRows)
                context_menu.addSeparator()
            
            # 全データをクリップボードにコピーするアクションを追加
            copy_all_action = context_menu.addAction("Copy All Data to Clipboard")
            copy_all_action.triggered.connect(self.copyTableDataToClipboard)
            
            # メニューを表示
            context_menu.exec_(self.results_table.viewport().mapToGlobal(position))
            
        except Exception as e:
            print(f"[ERROR] Failed to show table context menu: {e}")
            import traceback
            traceback.print_exc()
    
    def deleteSelectedResultRows(self):
        """選択行を削除。Ring Detection時はRing No.、Particle Detection時はParticle No.を詰める。"""
        try:
            is_ring = (hasattr(self, 'method_combo') and self.method_combo.currentText() == "Ring Detection")
            
            selected_ranges = self.results_table.selectedRanges()
            if not selected_ranges:
                return
            
            rows_to_delete = set()
            for r in selected_ranges:
                for row in range(r.topRow(), r.bottomRow() + 1):
                    rows_to_delete.add(row)
            if not rows_to_delete:
                return
            
            if is_ring:
                # 上から消すとインデックスがずれるため降順で削除
                for row in sorted(rows_to_delete, reverse=True):
                    self._delete_ring_result_row(row)
            else:
                # Particle Detection: ラベルを消去してから行を削除
                # ピクセルサイズを推定（calculateParticleProperties と同じロジック）
                # 画像サイズ
                if (self.analysis_mode == "Single Particle" and
                    self.roi_scale_info is not None and
                    hasattr(self, 'filtered_data') and self.filtered_data is not None):
                    height, width = self.filtered_data.shape
                    scan_size_x = self.roi_scale_info['scan_size_x']
                    scan_size_y = self.roi_scale_info['scan_size_y']
                else:
                    if self.current_data is not None:
                        height, width = self.current_data.shape
                    elif hasattr(self, 'filtered_data') and self.filtered_data is not None:
                        height, width = self.filtered_data.shape
                    else:
                        height = width = 0
                    scan_size_x = getattr(self, 'scan_size_x', 0)
                    scan_size_y = getattr(self, 'scan_size_y', 0)
                if width and height and scan_size_x and scan_size_y:
                    pixel_size_x = scan_size_x / width
                    pixel_size_y = scan_size_y / height
                else:
                    pixel_size_x = pixel_size_y = 1.0
                
                # ラベル配列がある場合のみ処理
                has_labels = hasattr(self, 'detected_particles') and self.detected_particles is not None
                
                for row in sorted(rows_to_delete, reverse=True):
                    # テーブルのCentroid X/Y (nm) からラベルを推定して削除
                    try:
                        x_item = self.results_table.item(row, 7)
                        y_item = self.results_table.item(row, 8)
                        if has_labels and x_item and y_item:
                            x_nm = float(x_item.text())
                            y_nm = float(y_item.text())
                            ix = int(max(0, min(width - 1, round(x_nm / pixel_size_x - 0.5))))
                            iy = int(max(0, min(height - 1, round(y_nm / pixel_size_y - 0.5))))
                            label_val = int(self.detected_particles[iy, ix]) if self.detected_particles is not None else 0
                            if label_val > 0:
                                self.detected_particles[self.detected_particles == label_val] = 0
                    except Exception:
                        pass
                    # 行削除
                    self.results_table.removeRow(row)
            
            # No. を 1..N に振り直し（列0）
            for i in range(self.results_table.rowCount()):
                self.results_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            
            # オーバーレイ更新
            self.updateParticleOverlay()
        except Exception as e:
            print(f"[ERROR] Failed to delete selected result rows: {e}")
            import traceback
            traceback.print_exc()
    
    def _delete_ring_result_row(self, row_index):
        """指定行（Ring Detection）のデータを削除し、履歴と現在の表示を整合させる。"""
        try:
            row_count = self.results_table.rowCount()
            if row_index < 0 or row_index >= row_count:
                return
            
            # 履歴からも削除（存在すれば）
            if hasattr(self, 'ring_detection_history') and isinstance(self.ring_detection_history, list):
                if 0 <= row_index < len(self.ring_detection_history):
                    del self.ring_detection_history[row_index]
            
            # テーブルから行を削除
            self.results_table.removeRow(row_index)
            
            # ring_last_avg_info を最新（履歴の最後）に更新、履歴が空ならNone
            if hasattr(self, 'ring_detection_history') and len(self.ring_detection_history) > 0:
                self.ring_last_avg_info = self.ring_detection_history[-1]
            else:
                self.ring_last_avg_info = None
                # Ring Detection のオーバーレイを消すため、detected_particlesが空ならそのまま、そうでなければ0配列に
                if hasattr(self, 'detected_particles') and self.detected_particles is not None:
                    if np.max(self.detected_particles) != 0:
                        self.detected_particles = np.zeros_like(self.detected_particles, dtype=np.int32)
        except Exception as e:
            print(f"[ERROR] Failed to delete ring result row {row_index}: {e}")
            import traceback
            traceback.print_exc()
    
    def copyTableDataToClipboard(self):
        """テーブルのデータをクリップボードにコピー"""
        try:
            # テーブルからデータを取得
            rows = self.results_table.rowCount()
            cols = self.results_table.columnCount()
            
            if rows == 0:
                return
            
            clipboard_text = []
            
            # LEGENDが有効な場合のみヘッダー行を追加
            if self.show_legend_check.isChecked():
                header_row = []
                for col in range(cols):
                    header_item = self.results_table.horizontalHeaderItem(col)
                    if header_item:
                        header_row.append(header_item.text())
                    else:
                        header_row.append(f"Column {col}")
                clipboard_text.append("\t".join(header_row))
            
            # データ行を追加
            for row in range(rows):
                row_data = []
                for col in range(cols):
                    item = self.results_table.item(row, col)
                    if item:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                clipboard_text.append("\t".join(row_data))
            
            # クリップボードにコピー
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText("\n".join(clipboard_text))
            
        except Exception as e:
            print(f"[ERROR] Failed to copy table data to clipboard: {e}")
            import traceback
            traceback.print_exc()
    
    def copySelectedTableDataToClipboard(self):
        """テーブルの選択された項目のみをクリップボードにコピー"""
        try:
            # 選択された範囲を取得
            selected_ranges = self.results_table.selectedRanges()
            
            if not selected_ranges:
                return
            
            clipboard_text = []
            
            # LEGENDが有効な場合のみヘッダー行を追加
            if self.show_legend_check.isChecked():
                header_row = []
                for col in range(self.results_table.columnCount()):
                    header_item = self.results_table.horizontalHeaderItem(col)
                    if header_item:
                        header_row.append(header_item.text())
                    else:
                        header_row.append(f"Column {col}")
                clipboard_text.append("\t".join(header_row))
            
            # 選択された範囲のデータを追加
            for range_obj in selected_ranges:
                top_row = range_obj.topRow()
                bottom_row = range_obj.bottomRow()
                left_col = range_obj.leftColumn()
                right_col = range_obj.rightColumn()
                
                for row in range(top_row, bottom_row + 1):
                    row_data = []
                    for col in range(left_col, right_col + 1):
                        item = self.results_table.item(row, col)
                        if item:
                            row_data.append(item.text())
                        else:
                            row_data.append("")
                    clipboard_text.append("\t".join(row_data))
            
            # クリップボードにコピー
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText("\n".join(clipboard_text))
            
        except Exception as e:
            print(f"[ERROR] Failed to copy selected table data to clipboard: {e}")
            import traceback
            traceback.print_exc()
    
    def copyImageToClipboard(self):
        """AFM画像をクリップボードにコピー"""
        try:
            # 現在のフィギュアをPixmapに変換
            canvas = self.image_canvas
            pixmap = canvas.grab()
            
            # クリップボードにコピー
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setPixmap(pixmap)
            
        except Exception as e:
            print(f"[ERROR] Failed to copy image to clipboard: {e}")
            import traceback
            traceback.print_exc()
    
    def onLegendVisibilityChanged(self, state):
        """LEGENDの表示/非表示を制御"""
        try:
            if state == QtCore.Qt.Checked:
                # LEGENDを表示
                self.results_table.horizontalHeader().setVisible(True)
            else:
                # LEGENDを非表示
                self.results_table.horizontalHeader().setVisible(False)
        except Exception as e:
            print(f"[ERROR] Failed to change legend visibility: {e}")
            import traceback
            traceback.print_exc()
    
    def deleteParticle(self, particle_label):
        """指定された粒子を削除"""
        try:
                    
            if not hasattr(self, 'detected_particles') or self.detected_particles is None:
                #print("[DEBUG] No detected_particles found")
                return
                
            # 指定されたラベルの粒子を背景（0）に変更
            self.detected_particles[self.detected_particles == particle_label] = 0
                       
            # 粒子ラベルを再整理
            self.reorganizeParticleLabels()
                  
            # 粒子の特性を再計算
            self.particle_properties = self.calculateParticleProperties()
                   
            # 結果表示を更新
            self.displayParticleResults()
                 
            # オーバーレイを更新
            self.updateParticleOverlay()
               
        except Exception as e:
            import traceback
            traceback.print_exc()
            
    def reorganizeParticleLabels(self):
        """粒子ラベルを連続的に再整理"""
        try:
            if not hasattr(self, 'detected_particles') or self.detected_particles is None:
                return
                
            # 現在のラベルを取得
            current_labels = np.unique(self.detected_particles)
            current_labels = current_labels[current_labels > 0]  # 背景（0）を除外
            
            # 新しいラベルで置き換え
            new_labels = np.zeros_like(self.detected_particles)
            for new_label, old_label in enumerate(current_labels, 1):
                new_labels[self.detected_particles == old_label] = new_label
                
            self.detected_particles = new_labels
            
        except Exception as e:
            print(f"[ERROR] Failed to reorganize particle labels: {e}")
            import traceback
            traceback.print_exc()
        
    def preprocessData(self, data):
        """データの前処理"""
        try:
             
            # データが指定されている場合はそれを使用、そうでなければgv.aryDataを使用
            if data is not None:
                processed_data = data.copy()
            elif hasattr(gv, 'aryData') and gv.aryData is not None:
                processed_data = gv.aryData.copy()
            else:
                print(f"[ERROR] No data available for preprocessing")
                return None
            
            # スキャンサイズ情報を取得
            # Single ParticleモードでROI選択済みの場合は、ROIの物理サイズを使用
            if (self.analysis_mode == "Single Particle" and 
                self.roi_scale_info is not None and 
                self.roi_data is not None):
                # ROIデータを処理中の場合は、ROIの物理サイズとピクセル情報を使用
                self.scan_size_x = self.roi_scale_info['scan_size_x']
                self.scan_size_y = self.roi_scale_info['scan_size_y']
                self.x_pixels = self.roi_scale_info['pixels_x']
                self.y_pixels = self.roi_scale_info['pixels_y']
                #print(f"[DEBUG] Using ROI scale info: scan_size_x={self.scan_size_x}, scan_size_y={self.scan_size_y}")
            else:
                # 全画面モードの場合は、グローバル変数から取得
                self.scan_size_x = getattr(gv, 'XScanSize', 0)
                self.scan_size_y = getattr(gv, 'YScanSize', 0)
                self.x_pixels = getattr(gv, 'XPixel', 0)
                self.y_pixels = getattr(gv, 'YPixel', 0)
                #print(f"[DEBUG] Using global scale info: scan_size_x={self.scan_size_x}, scan_size_y={self.scan_size_y}")
                     
            # Step 1: Background subtraction (常に元データに対して)
            bg_method = self.bg_combo.currentText()
            if bg_method == "Rolling Ball":
                radius_nm = self.rolling_radius_spin.value()
                # ピクセルサイズを計算
                scan_size_x = self.scan_size_x
                scan_size_y = self.scan_size_y
                
                # スキャンサイズが0の場合はエラー
                if scan_size_x == 0 or scan_size_y == 0:
                    print(f"[ERROR] Scan size is 0, cannot preprocess data")
                    print(f"[ERROR] Please load AFM data first to get scan size information")
                    return data
                
                pixel_size = min(scan_size_x / processed_data.shape[1], scan_size_y / processed_data.shape[0])
                radius_pixels = max(1, int(radius_nm / pixel_size))
                 
                from skimage import restoration
                processed_data = restoration.rolling_ball(processed_data, radius=radius_pixels)
                 
            elif bg_method == "Polynomial Fit":
                # 1次平面フィッティングによる背景除去（傾き補正）
                from scipy import ndimage
                from scipy.optimize import curve_fit
                import numpy as np
                
                # 1次平面フィッティング（傾き補正）
                height, width = processed_data.shape
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                
                # データを1次元にフラット化
                x_flat = x_coords.flatten()
                y_flat = y_coords.flatten()
                z_flat = processed_data.flatten()
                
                # 1次平面フィッティング関数（z = a + b*x + c*y）
                def plane(xy, a, b, c):
                    x, y = xy
                    return a + b*x + c*y
                
                try:
                    # 平面フィッティングを実行
                    popt, _ = curve_fit(plane, (x_flat, y_flat), z_flat)
                    
                    # 背景面を再構築
                    background = plane((x_coords, y_coords), *popt)
                    
                    # 背景を除去
                    processed_data = processed_data - background
                    
                except Exception as e:
                    print(f"[WARNING] Plane fitting failed: {e}")
                    # フィッティングが失敗した場合はガウシアンフィルタを使用
                    processed_data = ndimage.gaussian_filter(processed_data, sigma=10)
                    processed_data = processed_data - ndimage.gaussian_filter(processed_data, sigma=10)
                 
            # Step 2: Smoothing (Rolling Ball処理後のデータに対して)
            smooth_method = self.smooth_combo.currentText()
            if smooth_method == "Gaussian":
                sigma = self.smooth_param_spin.value()
                from scipy.ndimage import gaussian_filter
                processed_data = gaussian_filter(processed_data, sigma=sigma)
                
                # Intensityを1.0に固定
                intensity = 1.0
                processed_data = processed_data * intensity
                
                    
            elif smooth_method == "Median":
                # NxNサイズを直接使用
                size = int(self.smooth_param_spin.value())
                from scipy.ndimage import median_filter
                processed_data = median_filter(processed_data, size=size)
            
            return processed_data
            
        except Exception as e:
            print(f"[ERROR] Failed to preprocess data: {e}")
            import traceback
            traceback.print_exc()
            return data
        
    def displayParticleResults(self):
        """粒子検出結果をテーブルに表示（物理単位）"""
        try:
            if not hasattr(self, 'detected_particles') or self.detected_particles is None:
                #print(f"[DEBUG] displayParticleResults - No detected_particles")
                self.stats_label.setText("Statistics: No particles detected")
                return
            
            # 検出方法を確認
            detection_method = self.method_combo.currentText() if hasattr(self, 'method_combo') else "Threshold"
            
            if detection_method == "Ring Detection":
                # Ring Detection の結果を表示
                self.displayRingDetectionResults()
            else:
                # 通常の粒子検出結果を表示
                # 粒子の特性を計算
                if not hasattr(self, 'particle_properties') or self.particle_properties is None:
                    self.particle_properties = self.calculateParticleProperties()
                
                if not self.particle_properties:
                    #print(f"[DEBUG] displayParticleResults - No particle_properties")
                    self.stats_label.setText("Statistics: No particles detected")
                    return
                
                # 現在のテーブル行数を取得
                current_row_count = self.results_table.rowCount()
                
                # テーブルに行を追加（新しい行を追加、既存行は保持）
                self.results_table.setRowCount(current_row_count + len(self.particle_properties))
                
                for i, prop in enumerate(self.particle_properties):
                    row_idx = current_row_count + i
                    
                    # Particle No. - 現在の行数 + 1
                    particle_no = row_idx + 1
                    self.results_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(particle_no)))
                    
                    # Area (nm²)
                    area = prop.get('area', 0)
                    self.results_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{area:.2f}"))
                    
                    # Perimeter (nm)
                    perimeter = prop.get('perimeter', 0)
                    self.results_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(f"{perimeter:.2f}"))
                    
                    # Circularity
                    circularity = prop.get('circularity', 0)
                    self.results_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(f"{circularity:.3f}"))
                    
                    # Max Height (nm)
                    max_height = prop.get('max_height', 0)
                    self.results_table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(f"{max_height:.2f}"))
                    
                    # Mean Height (nm)
                    mean_height = prop.get('mean_height', 0)
                    self.results_table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(f"{mean_height:.2f}"))
                    
                    # Volume (nm³)
                    volume = prop.get('volume', 0)
                    self.results_table.setItem(row_idx, 6, QtWidgets.QTableWidgetItem(f"{volume:.2f}"))
                    
                    # Centroid X (nm)
                    centroid = prop.get('centroid', (0, 0))
                    self.results_table.setItem(row_idx, 7, QtWidgets.QTableWidgetItem(f"{centroid[1]:.1f}"))
                    
                    # Centroid Y (nm)
                    self.results_table.setItem(row_idx, 8, QtWidgets.QTableWidgetItem(f"{centroid[0]:.1f}"))
                
                # 統計情報を計算（全行対象）
                all_areas = []
                all_perimeters = []
                all_circularities = []
                all_max_heights = []
                all_mean_heights = []
                all_volumes = []
                
                for row in range(self.results_table.rowCount()):
                    # 各行のデータを取得
                    try:
                        area = float(self.results_table.item(row, 1).text())
                        perimeter = float(self.results_table.item(row, 2).text())
                        circularity = float(self.results_table.item(row, 3).text())
                        max_height = float(self.results_table.item(row, 4).text())
                        mean_height = float(self.results_table.item(row, 5).text())
                        volume = float(self.results_table.item(row, 6).text())
                        
                        all_areas.append(area)
                        all_perimeters.append(perimeter)
                        all_circularities.append(circularity)
                        all_max_heights.append(max_height)
                        all_mean_heights.append(mean_height)
                        all_volumes.append(volume)
                    except (ValueError, AttributeError):
                        pass
                
                #print(f"[DEBUG] displayParticleResults - Found {len(self.particle_properties)} particle properties")
                
                stats_text = f"Statistics: {self.results_table.rowCount()} particles detected"
                if all_areas:
                    stats_text += f" | Area: {np.mean(all_areas):.1f}±{np.std(all_areas):.1f} nm²"
                if all_max_heights:
                    stats_text += f" | Max Height: {np.mean(all_max_heights):.1f}±{np.std(all_max_heights):.1f} nm"
                if all_volumes:
                    stats_text += f" | Volume: {np.mean(all_volumes):.1f}±{np.std(all_volumes):.1f} nm³"
                    
                self.stats_label.setText(stats_text)
            
        except Exception as e:
            print(f"[ERROR] Failed to display particle results: {e}")
            import traceback
            traceback.print_exc()
        
    def displayRingDetectionResults(self):
        """Ring Detection の結果をテーブルに表示"""
        try:
            # テーブルのヘッダーをRing Detection用に変更（Ring No. + 3列）
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels([
                "Ring No.", "Inner Diameter (nm)", "Ring Diameter (nm)", "Circularity"
            ])
            
            # 列幅を調整
            column_widths = [80, 160, 160, 120]
            for i, width in enumerate(column_widths):
                self.results_table.setColumnWidth(i, width)
            
            # Ring detection の結果を取得
            if not hasattr(self, 'ring_last_avg_info') or self.ring_last_avg_info is None:
                self.stats_label.setText("Statistics: No rings detected")
                return
            
            ring_info = self.ring_last_avg_info
            # 検出有無に関わらず、ring_diameter があれば表示する
            
            # 履歴に追加
            if not hasattr(self, 'ring_detection_history'):
                self.ring_detection_history = []
            self.ring_detection_history.append(ring_info)
            
            # テーブルに行を追加（新しい行を追加、上書きしない）
            current_row_count = self.results_table.rowCount()
            self.results_table.setRowCount(current_row_count + 1)
            
            # Ring No. - インクリメントされた番号
            ring_no = current_row_count + 1
            self.results_table.setItem(current_row_count, 0, QtWidgets.QTableWidgetItem(str(ring_no)))
            
            # Inner Diameter (nm)
            inner_diameter = ring_info.get('inner_diameter_avg', None)
            if inner_diameter is None:
                inner_diameter = 0.0
            self.results_table.setItem(current_row_count, 1, QtWidgets.QTableWidgetItem(f"{inner_diameter:.2f}"))
            
            # Ring Diameter (nm) - 楕円面積等価直径（フィット成功時）、フォールバックは平均半径×2
            ring_diameter = ring_info.get('ring_diameter_nm', None)
            if ring_diameter is None:
                ring_diameter = 0.0
            self.results_table.setItem(current_row_count, 2, QtWidgets.QTableWidgetItem(f"{ring_diameter:.2f}"))
            
            # Circularity
            ring_circ = ring_info.get('ring_circularity', None)
            if ring_circ is None:
                ring_circ = 0.0
            self.results_table.setItem(current_row_count, 3, QtWidgets.QTableWidgetItem(f"{ring_circ:.3f}"))
            
            # 統計情報を表示
            stats_text = f"Ring {ring_no}: Inner {inner_diameter:.1f} nm | Ring {ring_diameter:.1f} nm | Circ {ring_circ:.3f}"
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            print(f"[ERROR] Failed to display ring detection results: {e}")
            import traceback
            traceback.print_exc()
    
    def resetTableHeadersForParticles(self):
        """テーブルヘッダーを通常の粒子検出用にリセット"""
        try:
            self.results_table.setColumnCount(9)
            self.results_table.setHorizontalHeaderLabels([
                "Particle No.", "Area (nm²)", "Perimeter (nm)", 
                "Circularity", "Max Height (nm)", "Mean Height (nm)", 
                "Volume (nm³)", "Centroid X (nm)", "Centroid Y (nm)"
            ])
            
            # 列幅を調整
            column_widths = [80, 100, 120, 80, 120, 120, 120, 120, 120]
            for i, width in enumerate(column_widths):
                self.results_table.setColumnWidth(i, width)
                
        except Exception as e:
            print(f"[ERROR] Failed to reset table headers: {e}")
    
    def updateAnalysisParametersForParticleDetection(self):
        """Analysis Parametersを通常の粒子検出用に更新"""
        try:
            # チェックボックスのラベルを通常の粒子用に変更
            self.calc_props_check.setText("Calculate Particle Properties")
            self.area_check.setText("Area (nm²)")
            self.perimeter_check.setText("Perimeter (nm)")
            self.circularity_check.setText("Circularity")
            self.max_height_check.setText("Max Height (nm)")
            self.mean_height_check.setText("Mean Height (nm)")
            self.volume_check.setText("Volume (nm³)")
            
            # すべてのチェックボックスを表示
            self.area_check.setVisible(True)
            self.perimeter_check.setVisible(True)
            self.circularity_check.setVisible(True)
            self.max_height_check.setVisible(True)
            self.mean_height_check.setVisible(True)
            self.volume_check.setVisible(True)
            
            # すべてをチェック
            self.area_check.setChecked(True)
            self.perimeter_check.setChecked(True)
            self.circularity_check.setChecked(True)
            self.max_height_check.setChecked(True)
            self.mean_height_check.setChecked(True)
            self.volume_check.setChecked(True)
            
        except Exception as e:
            print(f"[ERROR] Failed to update analysis parameters for particle detection: {e}")
    
    def updateAnalysisParametersForRingDetection(self):
        """Analysis Parametersを Ring Detection 用に更新"""
        try:
            # チェックボックスのラベルを Ring Detection 用に変更
            self.calc_props_check.setText("Calculate Ring Properties")
            self.area_check.setText("")  # 未使用（非表示）
            self.perimeter_check.setText("Inner Diameter (nm)")
            self.circularity_check.setText("Ring Diameter (nm)")
            self.max_height_check.setText("Circularity")
            self.mean_height_check.setText("")  # 未使用
            self.volume_check.setText("")  # 未使用
            
            # Ring Detection 用のチェックボックスを表示
            self.area_check.setVisible(False)
            self.perimeter_check.setVisible(True)
            self.circularity_check.setVisible(True)
            self.max_height_check.setVisible(True)
            self.mean_height_check.setVisible(False)  # 隠す
            self.volume_check.setVisible(False)  # 隠す
            
            # すべてをチェック
            self.area_check.setChecked(False)
            self.perimeter_check.setChecked(True)
            self.circularity_check.setChecked(True)
            self.max_height_check.setChecked(True)
            
        except Exception as e:
            print(f"[ERROR] Failed to update analysis parameters for ring detection: {e}")

    def detectParticlesFromData(self, data):
        """データから粒子を検出"""
        try:
            method = self.method_combo.currentText()
            
            if data is None:
                return None
                
            if method == "Threshold":
                result = self.thresholdDetection(data)
            elif method == "Peak Detection":
                # Peak Detectionの場合はBoundary Methodに応じて処理を選択
                boundary_method = self.boundary_method_combo.currentText()
                if boundary_method == "Contour Level":
                    result = self.contourLevelFromPeaks(data)
                else:  # Watershed
                    result = self.watershedFromPeaks(data)
            elif method == "Hessian Blob":
                # Hessian Blob検出を実行
                result = self.hessianBlobDetection(data)
            elif method == "Ring Detection":
                # Ring Detectionを実行
                result = self.ringDetection(data)
            else:
                return None
                
            return result
            
        except Exception as e:
            print(f"[ERROR] detectParticlesFromData failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def thresholdDetection(self, data):
        """閾値法による粒子検出"""
        self.ring_last_avg_info = None
        threshold_method = self.threshold_combo.currentText()
        
        if threshold_method == "Otsu":
            threshold = filters.threshold_otsu(data)
        elif threshold_method == "Manual":
            threshold = self.manual_thresh_spin.value()
        elif threshold_method == "Adaptive":
            threshold = filters.threshold_adaptive(data)
        else:
            return None
            
        # 二値化
        binary = data > threshold
        
        # シンプルな形態学処理（ピクセルベース）
        min_size = self.min_size_spin.value()
        max_size = self.max_size_spin.value()
        
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        binary = morphology.remove_small_holes(binary, area_threshold=max_size)
        
        # 最終的なラベリング
        labels = measure.label(binary)
        
        # 画像端の粒子を除外（チェックボックスが有効な場合）
        labels = self._apply_edge_exclusion(labels)
        
        # ステータスラベルを更新（Single Particleモードの場合のみ）
        self.updateParticleDetectionStatus(labels, "Threshold")
        
        return labels
        
    def hessianBlobDetection(self, data):
        """Hessian Blob検出による粒子検出"""
        self.ring_last_avg_info = None
        try:
            from skimage.feature import blob_log, blob_dog, blob_doh
            from skimage.filters import gaussian
            from scipy import ndimage
            import numpy as np
            
            # パラメータ取得
            min_sigma = self.hessian_min_sigma_spin.value()
            max_sigma = self.hessian_max_sigma_spin.value()
            threshold = self.hessian_threshold_spin.value()
            
            
            
            # データの正規化（0-1範囲）
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            #print(f"[DEBUG] Normalized data range: {np.min(data_normalized):.3f} to {np.max(data_normalized):.3f}")
            
            # まずskimageのblob_logを使用してブロブ検出を試行
            #print(f"[DEBUG] Trying blob_log with min_sigma={min_sigma}, max_sigma={max_sigma}")
            blobs = blob_log(data_normalized, min_sigma=min_sigma, max_sigma=max_sigma, 
                           num_sigma=10, threshold=threshold)
            
            if len(blobs) > 0:
                #print(f"[DEBUG] blob_log detected {len(blobs)} blobs")
                # ブロブの座標からマスクを作成
                blob_mask = np.zeros_like(data_normalized, dtype=bool)
                for blob in blobs:
                    y, x, r = blob
                    y, x, r = int(y), int(x), int(r)
                    # 円形のマスクを作成
                    yy, xx = np.ogrid[:data_normalized.shape[0], :data_normalized.shape[1]]
                    mask = (xx - x)**2 + (yy - y)**2 <= r**2
                    blob_mask |= mask
                
                # ラベリング
                labels = measure.label(blob_mask)
                num_particles = len(np.unique(labels)) - 1  # 背景を除く
                #print(f"[DEBUG] blob_log - initial labels: {num_particles} particles")
                
                # サイズフィルタリングを適用
                if num_particles > 0:
                    min_size = self.min_size_spin.value()
                    max_size = self.max_size_spin.value()
                    
                    unique_labels = np.unique(labels)
                    filtered_labels = np.zeros_like(labels)
                    
                    valid_particles = 0
                    for label in unique_labels:
                        if label == 0:  # 背景
                            continue
                            
                        label_size = np.sum(labels == label)
                        if min_size <= label_size <= max_size:
                            filtered_labels[labels == label] = label
                            valid_particles += 1
                    
                    labels = filtered_labels
                    num_particles = valid_particles
                    #print(f"[DEBUG] blob_log - after size filtering: {num_particles} particles")
                
                # 画像端の粒子を除外
                if self.exclude_edge_check.isChecked() and num_particles > 0:
                    #print(f"[DEBUG] blob_log - Excluding edge particles...")
                    # 画像の境界に接するラベルを特定
                    edge_labels = set()
                    height, width = labels.shape
                    
                    # 上下の境界
                    edge_labels.update(labels[0, :])  # 上端
                    edge_labels.update(labels[-1, :])  # 下端
                    # 左右の境界
                    edge_labels.update(labels[:, 0])  # 左端
                    edge_labels.update(labels[:, -1])  # 右端
                    
                    # 背景ラベル（0）を除外
                    edge_labels.discard(0)
                    
                    #print(f"[DEBUG] blob_log - Found {len(edge_labels)} edge labels: {edge_labels}")
                    
                    # 境界に接する粒子を除去
                    for label in edge_labels:
                        labels[labels == label] = 0
                    
                    # 再ラベリング
                    labels = measure.label(labels > 0)
                    num_particles = len(np.unique(labels)) - 1  # 背景を除く
                    #print(f"[DEBUG] blob_log - After edge exclusion: {num_particles} particles")
                
                if num_particles > 0:
                    return labels
            
            # blob_logで検出されない場合は、Hessian行列による検出を試行
            #print(f"[DEBUG] blob_log failed, trying Hessian matrix detection...")
            # ガウシアンフィルタでノイズ除去（Min Sigmaを使用）
            smoothing_sigma = min_sigma / 2.0  # Min Sigmaの半分を使用
            data_smoothed = gaussian(data_normalized, sigma=smoothing_sigma)
            
            # Hessian行列の計算（Max Sigmaを使用してスケールを調整）
            scale_factor = max_sigma / min_sigma if min_sigma > 0 else 1.0
            hessian_sigma = min_sigma * scale_factor
            
            # 2次微分を計算
            dxx = ndimage.gaussian_filter(data_smoothed, sigma=hessian_sigma, order=[2, 0])
            dyy = ndimage.gaussian_filter(data_smoothed, sigma=hessian_sigma, order=[0, 2])
            dxy = ndimage.gaussian_filter(data_smoothed, sigma=hessian_sigma, order=[1, 1])
            
            # Hessian行列の固有値を計算
            # H = [dxx dxy; dxy dyy]
            # 固有値: λ1, λ2 = (dxx + dyy ± sqrt((dxx - dyy)^2 + 4*dxy^2)) / 2
            trace = dxx + dyy
            det = dxx * dyy - dxy * dxy
            
            # 判別式
            discriminant = trace * trace - 4 * det
            discriminant = np.maximum(discriminant, 0)  # 負の値を0に
            
            # 固有値
            lambda1 = (trace + np.sqrt(discriminant)) / 2
            lambda2 = (trace - np.sqrt(discriminant)) / 2
            
            #print(f"[DEBUG] Hessian - Eigenvalue ranges: λ1 [{np.min(lambda1):.3f}, {np.max(lambda1):.3f}], λ2 [{np.min(lambda2):.3f}, {np.max(lambda2):.3f}]")
            
            # ブロブ検出条件（Hessian thresholdを直接使用）
            # 両方の固有値が負（局所的最大値）かつ閾値以上
            blob_mask = (lambda1 < -threshold) & (lambda2 < -threshold)
            
            #print(f"[DEBUG] Hessian - Initial blob mask: {np.sum(blob_mask)} pixels")
            
            # 形態学処理でノイズ除去
            from skimage import morphology
            # Min Sigmaに基づいて最小サイズを設定
            min_object_size = max(3, int(min_sigma))
            blob_mask = morphology.remove_small_objects(blob_mask, min_size=min_object_size)
            blob_mask = morphology.binary_closing(blob_mask, morphology.disk(1))
            
            #print(f"[DEBUG] Hessian - After morphology: {np.sum(blob_mask)} pixels")
            
            # ラベリング
            labels = measure.label(blob_mask)
            num_particles = len(np.unique(labels)) - 1  # 背景を除く
            
            #print(f"[DEBUG] Hessian - detected {num_particles} particles")
            #if num_particles > 0:
                #print(f"[DEBUG] Hessian - labels shape: {labels.shape}")
                #print(f"[DEBUG] Hessian - unique labels: {np.unique(labels)}")
                #print(f"[DEBUG] Hessian - label range: {np.min(labels)} to {np.max(labels)}")
            
            # 粒子サイズフィルタリング
            min_size = self.min_size_spin.value()
            max_size = self.max_size_spin.value()
            
            #print(f"[DEBUG] Hessian - size filtering: min={min_size}, max={max_size}")
            
            # 各ラベルのサイズをチェック
            unique_labels = np.unique(labels)
            filtered_labels = np.zeros_like(labels)
            
            valid_particles = 0
            size_distribution = []
            for label in unique_labels:
                if label == 0:  # 背景
                    continue
                    
                # ラベルのサイズを計算
                label_size = np.sum(labels == label)
                size_distribution.append(label_size)
                
                if min_size <= label_size <= max_size:
                    filtered_labels[labels == label] = label
                    valid_particles += 1
            
            #print(f"[DEBUG] Hessian - size distribution: min={min(size_distribution) if size_distribution else 0}, max={max(size_distribution) if size_distribution else 0}, mean={np.mean(size_distribution) if size_distribution else 0:.1f}")
            #print(f"[DEBUG] Hessian - after size filtering: {valid_particles} particles")
            
            # フィルタリング後のラベルを使用
            labels = filtered_labels
            num_particles = valid_particles
            
            # 画像端の粒子を除外
            if self.exclude_edge_check.isChecked() and num_particles > 0:
                #print(f"[DEBUG] Hessian - Excluding edge particles...")
                # 画像の境界に接するラベルを特定
                edge_labels = set()
                height, width = labels.shape
                
                # 上下の境界
                edge_labels.update(labels[0, :])  # 上端
                edge_labels.update(labels[-1, :])  # 下端
                # 左右の境界
                edge_labels.update(labels[:, 0])  # 左端
                edge_labels.update(labels[:, -1])  # 右端
                
                # 背景ラベル（0）を除外
                edge_labels.discard(0)
                
                #print(f"[DEBUG] Hessian - Found {len(edge_labels)} edge labels: {edge_labels}")
                
                # 境界に接する粒子を除去
                for label in edge_labels:
                    labels[labels == label] = 0
                
                # 再ラベリング
                labels = measure.label(labels > 0)
                num_particles = len(np.unique(labels)) - 1  # 背景を除く
                #print(f"[DEBUG] Hessian - After edge exclusion: {num_particles} particles")
            
            #if num_particles == 0:
                #print("[WARNING] No particles detected with current parameters.")
                #print("[DEBUG] Try adjusting Min Sigma, Max Sigma, or threshold values.")
            
            # ステータスラベルを更新（Single Particleモードの場合のみ）
            self.updateParticleDetectionStatus(labels, "Hessian Blob")
            
            labels = self._apply_edge_exclusion(labels)
            return labels
            
        except Exception as e:
            print(f"[ERROR] Hessian Blob detection failed: {e}")
            if self.analysis_mode == "Single Particle":
                self.particle_status_label.setText("✗ Detection failed")
                self.particle_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
                self.particle_status_label.setVisible(True)
            import traceback
            traceback.print_exc()
            return None
    
    def updateParticleDetectionStatus(self, labels, detection_method):
        """
        粒子検出のステータスを更新（Single Particleモード専用）
        
        Args:
            labels: 検出結果のラベル配列
            detection_method: 検出メソッド名（"Ring Detection", "Threshold"など）
        """
        if self.analysis_mode != "Single Particle":
            return
        
        if labels is None:
            self.particle_status_label.setText("✗ Detection failed")
            self.particle_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
            self.particle_status_label.setVisible(True)
            return
        
        num_detected = np.max(labels) if labels is not None else 0
        
        if num_detected > 0:
            # 粒子/リングが検出された
            if detection_method == "Ring Detection":
                msg = "✓ Ring detected"
            else:
                msg = f"✓ Particle detected ({num_detected})"
            self.particle_status_label.setText(msg)
            self.particle_status_label.setStyleSheet("color: green; font-weight: bold; font-size: 11px;")
            self.particle_status_label.setVisible(True)
        else:
            # 粒子/リングが検出されなかった
            if detection_method == "Ring Detection":
                msg = "✗ No ring detected"
            else:
                msg = "✗ No particle detected"
            self.particle_status_label.setText(msg)
            self.particle_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
            self.particle_status_label.setVisible(True)
    
    def ringDetection(self, data):
        """
        改善版リング検出アルゴリズム：
        1. 中心周辺の最小値を見つけてリング中心を決定
        2. 全方位に徐々に離れていってピークを探す（ピーク位置=リング中心線）
        3. ピークからさらに外側で下がる部分がピークの指定%の位置を外径に
        4. ピークから内側で下がる部分がピークの指定%の位置を内径に
        """
        try:
            import cv2
            import numpy as np
            from skimage import measure
            from scipy.ndimage import gaussian_filter1d, gaussian_filter, minimum_filter
            from scipy.signal import find_peaks
            # watershed の互換インポート（segmentation → morphology フォールバック）
            try:
                from skimage.segmentation import watershed as sk_watershed
            except Exception:
                try:
                    from skimage.morphology import watershed as sk_watershed
                except Exception:
                    sk_watershed = None
            
            
            
            # パラメータ取得
            min_radius_nm = self.ring_min_radius_spin.value()
            max_radius_nm = self.ring_max_radius_spin.value()
            center_blend_percent = self.ring_center_blend_spin.value()
            inner_percent = self.ring_inner_drop_spin.value()
            inner_fraction = max(0.01, min(inner_percent / 100.0, 0.9))
            
            
            
            # スケール情報を取得（nm/pixel）
            # Single ParticleモードでROI選択済みの場合は、ROIのスケールを使用
            if (self.analysis_mode == "Single Particle" and 
                self.roi_scale_info is not None and 
                self.roi_data is not None):
                # ROIのスケール情報を使用
                roi_scan_size_x = self.roi_scale_info['scan_size_x']
                roi_pixels_x = self.roi_scale_info['pixels_x']
                if roi_pixels_x > 0:
                    nm_per_pixel = roi_scan_size_x / roi_pixels_x
                else:
                    nm_per_pixel = 1.0
                
            # 全画面モードの場合は、グローバル変数から取得
            elif hasattr(gv, 'scan_size') and gv.scan_size is not None and gv.scan_size > 0:
                if hasattr(gv, 'pixelNum') and gv.pixelNum is not None and gv.pixelNum > 0:
                    nm_per_pixel = gv.scan_size / gv.pixelNum
                else:
                    nm_per_pixel = 1.0
                
            else:
                nm_per_pixel = 1.0
                
            
            # 半径をピクセル単位に変換
            min_radius_px = int(min_radius_nm / nm_per_pixel)
            max_radius_px = int(max_radius_nm / nm_per_pixel)
            
            
            
            # --- ステップ1: 拡大画像の中心周囲（30%の面積内）から最小値を探してリング中心を決定 ---
            height, width = data.shape
            center_y_geom = height / 2.0
            center_x_geom = width / 2.0
            
            
            
            # 拡大画像の30%の面積を中心周囲で探索（距離と強度をブレンド）
            # 面積が30%なら、一辺は sqrt(0.3) ≈ 0.548 倍
            search_fraction = np.sqrt(0.30)  # 約 0.548
            search_radius_x = int(width * search_fraction / 2.0)
            search_radius_y = int(height * search_fraction / 2.0)
            
            y_start = max(0, int(center_y_geom - search_radius_y))
            y_end = min(height, int(center_y_geom + search_radius_y + 1))
            x_start = max(0, int(center_x_geom - search_radius_x))
            x_end = min(width, int(center_x_geom + search_radius_x + 1))
            
            center_search_region = data[y_start:y_end, x_start:x_end]
            
            
            
            if center_search_region.size > 0:
                # ノイズを抑えるために軽く平滑化
                smoothed_region = gaussian_filter(center_search_region, sigma=1.2)
                
                # 強度を0-1に正規化（小さいほど中心候補として有利）
                region_min = np.min(smoothed_region)
                region_max = np.max(smoothed_region)
                region_range = max(region_max - region_min, 1e-6)
                normalized_intensity = (smoothed_region - region_min) / region_range
                
                # 幾何学的中心からの距離を計算し、0-1に正規化（小さいほど有利）
                yy_local, xx_local = np.indices(smoothed_region.shape)
                global_x = x_start + xx_local
                global_y = y_start + yy_local
                distance_map_center = np.sqrt((global_x - center_x_geom)**2 + (global_y - center_y_geom)**2)
                max_distance = max(np.max(distance_map_center), 1e-6)
                normalized_distance = distance_map_center / max_distance
                
                blend_weight = np.clip(center_blend_percent / 100.0, 0.0, 1.0)
                
                
                # コスト関数を計算（小さいほど中心に適する）
                cost_map = blend_weight * normalized_intensity + (1.0 - blend_weight) * normalized_distance
                
                min_idx = np.argmin(cost_map)
                min_y_local, min_x_local = np.unravel_index(min_idx, cost_map.shape)
                center_y = y_start + min_y_local
                center_x = x_start + min_x_local
                min_value_original = center_search_region[min_y_local, min_x_local]
                distance_from_geom = distance_map_center[min_y_local, min_x_local]
                
                
            else:
                center_y = int(center_y_geom)
                center_x = int(center_x_geom)
                
            
            
            
            # 検出成否に関わらず中心情報を保持するためのベース辞書を初期化
            ring_info = {
                'center_x': float(center_x),
                'center_y': float(center_y),
                'inner_radius': None,
                'outer_radius': None,
                'outer_diameter_avg': None,
                'inner_diameter_avg': None,
                'avg_diameter': None,
                'outer_circularity': None,
                'inner_circularity': None,
                'outer_contour': None,
                'inner_contour': None,
                'detected': False
            }
            self.ring_last_avg_info = ring_info
            
            # --- ステップ2: ラジアルプロファイル作成 ---
            y_coords, x_coords = np.ogrid[0:height, 0:width]
            distance_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # 距離の最大値を計算
            max_distance = int(np.sqrt((width/2)**2 + (height/2)**2)) + 1
            
            # 距離ごとの平均強度を計算
            radial_profile = []
            radial_std = []
            distances = []
            
            for r in range(max_distance):
                mask = (distance_map >= r) & (distance_map < r + 1)
                if np.sum(mask) > 0:
                    values = data[mask]
                    radial_profile.append(np.mean(values))
                    radial_std.append(np.std(values))
                    distances.append(r)
                else:
                    radial_profile.append(0)
                    radial_std.append(0)
                    distances.append(r)
            
            radial_profile = np.array(radial_profile)
            radial_std = np.array(radial_std)
            distances = np.array(distances)
            
            
            
            # --- ステップ3: プロファイルを平滑化 ---
            # ノイズ除去のためにガウシアン平滑化
            smoothed_profile = gaussian_filter1d(radial_profile, sigma=2.0)
            
            # --- ステップ4: リング検出（ピーク検出） ---
            # リング構造の特徴：中心が低く、ある半径で高くなる
            # 最小半径から最大半径の範囲でピークを探す
            
            # 検索範囲を制限
            search_start = max(0, min_radius_px - 2)
            search_end = min(len(smoothed_profile), max_radius_px + 2)
            
            if search_end <= search_start:
                print(f"[WARNING] Invalid search range: {search_start} - {search_end}")
                return np.zeros_like(data, dtype=np.int32)
            
            search_profile = smoothed_profile[search_start:search_end]
            
            # ピーク検出（prominence: ピークの顕著性）
            # heightは相対的な高さ（中央値より高い）
            median_height = np.median(smoothed_profile)
            peaks, properties = find_peaks(search_profile, 
                                          prominence=np.std(search_profile) * 0.3,
                                          height=median_height * 0.9)
            
            
            
            if len(peaks) == 0:
                # ピークが見つからない場合、最大値の位置をリング半径とする
                max_idx = np.argmax(search_profile)
                ring_radius_px = search_start + max_idx
                
            else:
                # 最も顕著なピークを選択
                peak_prominences = properties['prominences']
                best_peak_idx = np.argmax(peak_prominences)
                ring_radius_px = search_start + peaks[best_peak_idx]
                
                
                
                # 全てのピークを表示
                for i, peak in enumerate(peaks):
                    r = search_start + peak
            
            # リング半径をnmに変換
            ring_radius_nm = ring_radius_px * nm_per_pixel
            
            
            # --- ステップ5: 全方位に徐々に離れていってピークを探す ---
            # 複数の方向（8方向以上）でラジアルプロファイルを取得してピークを探す
            n_angles = 16  # 16方向で探索
            angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
            
            all_radii = []  # 各方向で見つかったピーク半径
            ray_peak_radii = []  # 各方向の個別ピーク半径（ミッド輪郭のフォールバック用）
            
            for angle in angles:
                # 各方向でのラジアルプロファイルを取得
                ray_profile = []
                for r in range(max_distance):
                    # その方向の直線上のピクセル
                    x = center_x + r * np.cos(angle)
                    y = center_y + r * np.sin(angle)
                    
                    if 0 <= int(x) < width and 0 <= int(y) < height:
                        # 双線形補間で値を取得
                        x0, x1 = int(np.floor(x)), int(np.ceil(x))
                        y0, y1 = int(np.floor(y)), int(np.ceil(y))
                        x0 = max(0, min(x0, width - 1))
                        x1 = max(0, min(x1, width - 1))
                        y0 = max(0, min(y0, height - 1))
                        y1 = max(0, min(y1, height - 1))
                        
                        v = data[y0, x0] * (1 - (x - x0)) * (1 - (y - y0)) + \
                            data[y0, x1] * ((x - x0)) * (1 - (y - y0)) + \
                            data[y1, x0] * (1 - (x - x0)) * ((y - y0)) + \
                            data[y1, x1] * ((x - x0)) * ((y - y0))
                        ray_profile.append(v)
                    else:
                        if len(ray_profile) > 0:
                            break
                
                if len(ray_profile) > min_radius_px + 2:
                    ray_profile = np.array(ray_profile)
                    # 平滑化
                    smoothed_ray = gaussian_filter1d(ray_profile, sigma=2.0)
                    # min_radius_px から max_radius_px の範囲でピークを探す
                    search_start_ray = max(min_radius_px - 2, 0)
                    search_end_ray = min(max_radius_px + 2, len(smoothed_ray))
                    
                    if search_end_ray > search_start_ray:
                        search_profile_ray = smoothed_ray[search_start_ray:search_end_ray]
                        # ピークを探す
                        peaks_ray, _ = find_peaks(search_profile_ray, prominence=np.std(search_profile_ray) * 0.2)
                        if len(peaks_ray) > 0:
                            peak_idx = peaks_ray[np.argmax(search_profile_ray[peaks_ray])]
                            peak_radius = search_start_ray + peak_idx
                            all_radii.append(peak_radius)
                            ray_peak_radii.append(peak_radius)
                        else:
                            ray_peak_radii.append(None)
                    else:
                        ray_peak_radii.append(None)
            
            # 各方向のピーク半径の平均を取る
            if len(all_radii) > 0:
                ring_radius_px = int(np.median(all_radii))
                
            else:
                print(f"[WARNING] No ring peaks found, using estimated radius")
                ring_radius_px = (min_radius_px + max_radius_px) // 2
            
            # --- ステップ6: ピークの高さを基準に内径・外径を決定 ---
            # ピーク位置での高さ
            y_peak = center_y + ring_radius_px * np.sin(0)
            x_peak = center_x + ring_radius_px * np.cos(0)
            if 0 <= int(x_peak) < width and 0 <= int(y_peak) < height:
                peak_height = data[int(y_peak), int(x_peak)]
            else:
                peak_height = np.max(data)
            
            # ラジアルプロファイルからピークの高さを取得
            radial_mask = (distance_map >= ring_radius_px - 1) & (distance_map <= ring_radius_px + 1)
            if np.sum(radial_mask) > 0:
                peak_height = np.mean(data[radial_mask])
            
            baseline = np.min(data)
            amplitude = max(peak_height - baseline, 1e-6)
            
            
            
            # 内径：ピーク位置からピークの指定%以下の内側（画像反転＋等高線法）
            inner_drop_threshold = peak_height * (1.0 - inner_fraction)
            
            
            
            inner_radius = None
            inner_contour_custom = None
            inner_region_mask = None
            inner_candidate_list = []
            
            data_min = float(np.min(data))
            data_max = float(np.max(data))
            inv_range = max(data_max - data_min, 1e-6)
            inv_data = (data_max - data)  # 反転（中心が高くなる）
            inv_norm = inv_data / inv_range
            center_y_idx = int(np.clip(round(center_y), 0, height - 1))
            center_x_idx = int(np.clip(round(center_x), 0, width - 1))
            center_peak_val_norm = inv_norm[center_y_idx, center_x_idx]
            
            inner_level_inv = max(0.0, data_max - inner_drop_threshold)
            inner_level_norm = inner_level_inv / inv_range
            inner_level_norm = float(np.clip(inner_level_norm, 1e-4, min(center_peak_val_norm * 0.995 + 1e-4, 0.995)))
            
            
            
            inner_candidates = measure.find_contours(inv_norm, inner_level_norm)
            if inner_candidates:
                try:
                    from matplotlib.path import Path
                except ImportError:
                    Path = None
                    print("[WARNING] matplotlib.path.Path unavailable; skipping inner contour selection")
                
                center_point = (center_x, center_y)
                best_area = None
                for contour in inner_candidates:
                    if contour.shape[0] < 3:
                        continue
                    if Path is not None:
                        path = Path(np.column_stack((contour[:, 1], contour[:, 0])))
                        if not path.contains_point(center_point, radius=1e-6):
                            continue
                    # 面積を計算して最も小さい（中心を囲む）輪郭を選択
                    x_coords = contour[:, 1]
                    y_coords = contour[:, 0]
                    area = 0.5 * abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1)))
                    if area <= 0:
                        continue
                    inner_candidate_list.append({
                        'contour': contour,
                        'area': area
                    })
            
            # 内径候補を検証（外径は用いず、中心を囲み面積が最小の候補を優先）
            if inner_candidate_list:
                inner_candidate_list.sort(key=lambda c: c['area'])
                for candidate in inner_candidate_list:
                    contour = candidate['contour']
                    distances_inner = np.sqrt(
                        (contour[:, 1] - center_x) ** 2 +
                        (contour[:, 0] - center_y) ** 2
                    )
                    if distances_inner.size == 0:
                        continue
                    mean_distance_inner = float(np.mean(distances_inner))
                    
                    try:
                        from skimage.draw import polygon
                        candidate_mask = np.zeros_like(data, dtype=bool)
                        poly_y = np.clip(contour[:, 0], 0, height - 1)
                        poly_x = np.clip(contour[:, 1], 0, width - 1)
                        rr, cc = polygon(poly_y, poly_x, shape=data.shape)
                        candidate_mask[rr, cc] = True
                    except Exception as e:
                        print(f"[WARNING] Failed to rasterize inner contour candidate: {e}")
                        candidate_mask = None
                    
                    inner_radius = mean_distance_inner
                    inner_contour_custom = contour
                    inner_region_mask = candidate_mask if candidate_mask is not None else (distance_map <= inner_radius)
                    
                    break
            
            if inner_radius is None:
                
                for r in range(ring_radius_px, max(0, ring_radius_px - 20), -1):
                    radial_mask_inner = (distance_map >= r - 0.5) & (distance_map <= r + 0.5)
                    if np.sum(radial_mask_inner) > 0:
                        height_at_r = np.mean(data[radial_mask_inner])
                        if height_at_r <= inner_drop_threshold:
                            inner_radius = r
                            break
                if inner_radius is None:
                    inner_radius = max(0, ring_radius_px - int(ring_radius_px * 0.4))
                inner_region_mask = (distance_map <= inner_radius)
                inner_contour_custom = None
            
            # 外径検出は一旦無効化（ユーザー要望により削除）
            ring_info['outer_contour'] = None
            ring_info['outer_radius'] = None
            ring_info['outer_diameter_avg'] = None
            
            # ===== 内径のみをまず確実に表示（外径は一旦スキップ）=====
            ring_info['inner_radius'] = float(inner_radius) if inner_radius is not None else None
            if inner_radius is not None:
                ring_info['inner_diameter_avg'] = inner_radius * 2 * nm_per_pixel
            if inner_contour_custom is not None:
                ring_info['inner_contour'] = inner_contour_custom
            else:
                # 既定のアルゴリズムで輪郭候補が得られない場合でも、表示が消えないように
                # 内径マスクの境界から輪郭を生成（描画フォールバック：検出ロジックは変更しない）
                if inner_region_mask is not None:
                    try:
                        inner_contours_fb = measure.find_contours(inner_region_mask.astype(float), 0.5)
                        if inner_contours_fb:
                            # 最長の輪郭を採用（内側の近似円に相当）
                            inner_contour_fb = max(inner_contours_fb, key=lambda c: c.shape[0])
                            ring_info['inner_contour'] = inner_contour_fb
                        else:
                            ring_info['inner_contour'] = None
                    except Exception as e:
                        print(f"[WARNING] Inner contour fallback failed: {e}")
                        ring_info['inner_contour'] = None
                else:
                    ring_info['inner_contour'] = None
            
            # --- 追加: 反転データを用いた中心起点のWatershedでリング形状（中間輪郭）を抽出 ---
            try:
                if sk_watershed is not None:
                    elev = 1.0 - inv_norm  # 中心が最小になるように（流域が中心に集まる）
                    markers = np.zeros_like(data, dtype=np.int32)
                    markers[center_y_idx, center_x_idx] = 1  # 中心シード
                    # 画像の外周を背景シードにする
                    markers[0, :] = 2
                    markers[-1, :] = 2
                    markers[:, 0] = 2
                    markers[:, -1] = 2
                    
                    ws_labels = sk_watershed(elev, markers=markers)
                    center_region = (ws_labels == 1)
                    
                    # 中心領域の境界をリング形状（中間輪郭）として取得
                    ws_contours = measure.find_contours(center_region.astype(float), 0.5)
                    if ws_contours:
                        # 最長の閉曲線を採用
                        ws_contours_sorted = sorted(ws_contours, key=lambda c: c.shape[0], reverse=True)
                        mid_contour = ws_contours_sorted[0]
                        ring_info['mid_contour'] = mid_contour
                        # 中間輪郭の平均半径からリング径[nm]を算出して保存
                        mid_dist = np.sqrt((mid_contour[:, 1] - center_x) ** 2 + (mid_contour[:, 0] - center_y) ** 2)
                        if mid_dist.size > 0:
                            ring_info['ring_diameter_nm'] = float(2.0 * np.mean(mid_dist) * nm_per_pixel)
                        else:
                            ring_info['ring_diameter_nm'] = None
                    else:
                        ring_info['mid_contour'] = None
                        ring_info['ring_diameter_nm'] = None
                else:
                    # フォールバック: 複数方向のピーク半径から輪郭点を生成
                    mid_pts = []
                    for ang, r in zip(angles, ray_peak_radii):
                        if r is None:
                            continue
                        y = center_y + r * np.sin(ang)
                        x = center_x + r * np.cos(ang)
                        if 0 <= x < width and 0 <= y < height:
                            mid_pts.append([y, x])
                    if len(mid_pts) >= 3:
                        ring_info['mid_contour'] = np.array(mid_pts, dtype=float)
                        mid_dist = np.sqrt((ring_info['mid_contour'][:, 1] - center_x) ** 2 + (ring_info['mid_contour'][:, 0] - center_y) ** 2)
                        ring_info['ring_diameter_nm'] = float(2.0 * np.mean(mid_dist) * nm_per_pixel) if mid_dist.size > 0 else None
                    else:
                        ring_info['mid_contour'] = None
                        ring_info['ring_diameter_nm'] = None
            except Exception as e:
                ring_info['mid_contour'] = None
                ring_info['ring_diameter_nm'] = None
                print(f"[WARNING] Watershed/ray mid contour extraction failed: {e}")
            
            # --- 楕円フィッティングに基づく等価円直径とCircularityの計算 ---
            try:
                # 内径：内側輪郭の楕円フィット → 面積等価円直径（px）d = sqrt(w*h)
                inner_contour_fit = ring_info.get('inner_contour')
                if inner_contour_fit is not None and len(inner_contour_fit) >= 5:
                    pts_inner = np.ascontiguousarray(
                        np.column_stack((inner_contour_fit[:, 1], inner_contour_fit[:, 0])).astype(np.float32)
                    )
                    ellipse_inner = cv2.fitEllipse(pts_inner)  # ((cx,cy),(w,h),angle)
                    (icx, icy), (iw, ih), iang = ellipse_inner
                    d_eq_inner_px = float(np.sqrt(max(iw, 0.0) * max(ih, 0.0)))
                    ring_info['inner_diameter_avg'] = d_eq_inner_px * nm_per_pixel
                    ring_info['inner_ellipse'] = ((float(icx), float(icy)), (float(iw), float(ih)), float(iang))
            except Exception as e:
                print(f"[WARNING] Inner ellipse fitting failed: {e}")
            
            try:
                # リング径（中間輪郭）：楕円フィット → 面積等価円直径と楕円ベースのCircularity
                mid_contour_fit = ring_info.get('mid_contour')
                if mid_contour_fit is not None and len(mid_contour_fit) >= 5:
                    pts_mid = np.ascontiguousarray(
                        np.column_stack((mid_contour_fit[:, 1], mid_contour_fit[:, 0])).astype(np.float32)
                    )
                    ellipse_mid = cv2.fitEllipse(pts_mid)  # ((cx,cy),(w,h),angle)
                    (mcx, mcy), (mw, mh), mang = ellipse_mid
                    # 面積等価円直径 d = sqrt(w*h) （w,h は楕円の長短軸の「長さ」[px]）
                    d_eq_ring_px = float(np.sqrt(max(mw, 0.0) * max(mh, 0.0)))
                    ring_info['ring_diameter_nm'] = d_eq_ring_px * nm_per_pixel
                    ring_info['mid_ellipse'] = ((float(mcx), float(mcy)), (float(mw), float(mh)), float(mang))
                    # 楕円面積と周長（Ramanujan近似）からCircularity = 4πA / P^2
                    a = max(mw, 0.0) / 2.0
                    b = max(mh, 0.0) / 2.0
                    area_ell = np.pi * a * b
                    # Ramanujan approximation for ellipse perimeter
                    perim_ell = np.pi * (3.0 * (a + b) - np.sqrt(max((3.0 * a + b) * (a + 3.0 * b), 0.0)))
                    if perim_ell > 1e-9:
                        circ = float(4.0 * np.pi * area_ell / (perim_ell ** 2))
                        ring_info['ring_circularity'] = min(max(circ, 0.0), 1.0)
                        ring_info['ring_perimeter_nm'] = perim_ell * nm_per_pixel  # 参考値（表示はしない）
            except Exception as e:
                print(f"[WARNING] Mid ellipse fitting failed: {e}")
            
            # 外径は不要のため未計算・未描画
            ring_info['outer_contour'] = None
            ring_info['outer_radius'] = None
            ring_info['outer_diameter_avg'] = None
            
            self.ring_last_avg_info = ring_info
            
            # ステータス更新と空ラベル返却（中心と内径輪郭は ring_last_avg_info から描画）
            labels = np.zeros_like(data, dtype=np.int32)
            self.updateParticleDetectionStatus(labels, "Ring Detection")
            return labels
            
            ring_width_px = max(1, outer_radius - inner_radius)
            
            
            ring_info['inner_radius'] = float(inner_radius) if inner_radius is not None else None
            ring_info['outer_radius'] = float(outer_radius) if outer_radius is not None else None
            
            outer_diameter_avg = (outer_radius * 2 * nm_per_pixel) if outer_radius is not None else None
            inner_diameter_avg = (inner_radius * 2 * nm_per_pixel) if inner_radius is not None else None
            avg_diameter = None
            if outer_diameter_avg is not None and inner_diameter_avg is not None:
                avg_diameter = (outer_diameter_avg + inner_diameter_avg) / 2.0
            
            ring_info['outer_diameter_avg'] = outer_diameter_avg
            ring_info['inner_diameter_avg'] = inner_diameter_avg
            ring_info['avg_diameter'] = avg_diameter
            
            # --- ステップ6: ラベルマップを作成 ---
            labels = np.zeros_like(data, dtype=np.int32)
            
            # リング領域をラベル化（アニュラス領域）
            outer_region_mask = (distance_map <= outer_radius)
            if inner_region_mask is None:
                inner_region_mask = (distance_map <= inner_radius)
            ring_mask = outer_region_mask & (~inner_region_mask)
            labels[ring_mask] = 1
            
            
            
            # デバッグ: リング領域の重心を計算
            if np.sum(ring_mask) > 0:
                y_indices, x_indices = np.where(ring_mask)
                centroid_y = np.mean(y_indices)
                centroid_x = np.mean(x_indices)
                
                
                dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                
                # 外側の領域マスクと内側（孔）マスク
                outer_mask_full = dist <= outer_radius
                inner_mask_full = inner_region_mask.copy()
                
                # 円形度を計算
                # 円形度 = 4π * 面積 / 周長^2 (完全な円は1.0)
                def calculate_circularity(mask):
                    if np.sum(mask) == 0:
                        return 0.0
                    # contourを取得して周長を計算
                    contours = measure.find_contours(mask.astype(float), 0.5)
                    if len(contours) == 0:
                        return 0.0
                    contour = contours[0]
                    area = np.sum(mask)
                    # 周長を計算（contour間の距離の合計）
                    perimeter = np.sum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
                    if perimeter < 1e-6:
                        return 0.0
                    circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
                    return min(circularity, 1.0)
                
                outer_circularity = calculate_circularity(outer_mask_full)
                inner_circularity = calculate_circularity(inner_mask_full)
                ring_info['outer_circularity'] = outer_circularity
                ring_info['inner_circularity'] = inner_circularity
                
                # 外径・内径を計算（平均値）
                
                
                # リング領域の外側と内側の輪郭を別々に取得
                outer_contour = None
                inner_contour = None
                
                # 実際のリング領域からの輪郭を取得
                ring_contours = measure.find_contours(ring_mask.astype(float), 0.5)
                
                
                
                if len(ring_contours) >= 2:
                    # 複数の輪郭が見つかった場合、最も長い2つを外側と内側とする
                    sorted_contours = sorted(ring_contours, key=len, reverse=True)
                    outer_contour = sorted_contours[0]  # 最も長い＝外側
                    inner_contour = sorted_contours[1]  # 次に長い＝内側
                    
                elif len(ring_contours) == 1:
                    # 1つだけの場合、それが外側の輪郭
                    outer_contour = ring_contours[0]
                    
                else:
                    pass
                
                if inner_contour_custom is not None:
                    inner_contour = inner_contour_custom
                    
                
                ring_info['outer_contour'] = outer_contour  # 外側の輪郭（赤）
                ring_info['inner_contour'] = inner_contour  # 内側の輪郭（黄色）
                ring_info['detected'] = True
            else:
                pass
            
            # サイズフィルタリングを適用
            min_size = self.min_size_spin.value()
            max_size = self.max_size_spin.value()
            
            ring_size = np.sum(labels == 1)
            if ring_size < min_size or ring_size > max_size:
                
                labels = np.zeros_like(data, dtype=np.int32)
            else:
                pass
            
            # Ring Detectionでは、エッジ除外を無視する（Single Particle モード専用で、ROI全体が対象）
            # labels = self._apply_edge_exclusion(labels)  # Ring Detection では適用しない
            
            # Note: ring_last_avg_info は既に設定済みなので、ここでクリアしない
            # （サイズフィルターで labels が 0 になった場合でも、ring_last_avg_info は保持）
            if np.max(labels) == 0:
                pass
            else:
                pass
            
            # ステータスラベルを更新（Single Particleモードの場合のみ）
            self.updateParticleDetectionStatus(labels, "Ring Detection")
            
            return labels
            
        except ImportError as ie:
            print(f"[ERROR] Missing module for ring detection: {ie}")
            if self.analysis_mode == "Single Particle":
                self.particle_status_label.setText("✗ Error: Missing module")
                self.particle_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
                self.particle_status_label.setVisible(True)
            _frozen = getattr(sys, "frozen", False)
            if _frozen:
                msg = (
                    f"Required module for ring detection is missing:\n{ie}\n\n"
                    "リング検出に必要なモジュールがインストールされていません。\n\n"
                    "This module is not bundled with this installation.\n"
                    "このモジュールはこのパッケージに含まれていません。"
                )
            else:
                msg = (
                    f"Required module for ring detection is missing:\n{ie}\n\n"
                    "リング検出に必要なモジュールがインストールされていません。\n\n"
                    "Please install scipy and opencv-python (pip install scipy opencv-python)."
                )
            QtWidgets.QMessageBox.warning(self, "Module Not Found", msg)
            return None
        except Exception as e:
            print(f"[ERROR] Ring detection failed: {e}")
            if self.analysis_mode == "Single Particle":
                self.particle_status_label.setText("✗ Detection failed")
                self.particle_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
                self.particle_status_label.setVisible(True)
            # 中心情報までは計算済みの可能性があるため、ring_last_avg_info は保持する
            import traceback
            traceback.print_exc()
            return None
        
    def peakDetection(self, data):
        """ピーク検出のみ（2段階処理の第1段階）- 改善版"""
        try:
            # データがNoneでないかチェック
            if data is None:
                return None
            
            
            
            # 前回のpeak_markersをクリア
            self.peak_markers = None
                
            # より適切なピーク検出
            from scipy.ndimage import gaussian_filter
            from skimage.feature import peak_local_max
            from skimage.filters import threshold_otsu
            
            # データを平滑化
            sigma = self.gradient_sigma_spin.value()
            smoothed_data = gaussian_filter(data, sigma=sigma)
            
            # 適応的閾値を計算
            otsu_threshold = threshold_otsu(smoothed_data)
            
            # ピーク検出パラメータを取得
            min_distance = self.min_peak_distance_spin.value()
            
            # 閾値ベースでピークを検出
            coordinates = peak_local_max(smoothed_data, 
                                      min_distance=min_distance,
                                      threshold_abs=otsu_threshold,
                                      exclude_border=False)
            
            # マスクを適用（座標を取得後にフィルタリング）
            if len(coordinates) > 0:
                mask = smoothed_data > otsu_threshold
                valid_coordinates = []
                for coord in coordinates:
                    y, x = coord
                    if mask[y, x]:  # マスク内の座標のみを保持
                        valid_coordinates.append(coord)
                coordinates = np.array(valid_coordinates) if valid_coordinates else np.empty((0, 2), dtype=int)
            
            # ピークが少ない場合は閾値を調整
            if len(coordinates) < 3:
                lower_threshold = otsu_threshold * 0.8  # 20%下げる
                coordinates = peak_local_max(smoothed_data,
                                          min_distance=min_distance,
                                          threshold_abs=lower_threshold,
                                          exclude_border=False)
                
                # マスクを適用
                if len(coordinates) > 0:
                    mask = smoothed_data > lower_threshold
                    valid_coordinates = []
                    for coord in coordinates:
                        y, x = coord
                        if mask[y, x]:
                            valid_coordinates.append(coord)
                    coordinates = np.array(valid_coordinates) if valid_coordinates else np.empty((0, 2), dtype=int)
            
            # ピークが多すぎる場合は閾値を厳しく
            if len(coordinates) > 20:
                higher_threshold = otsu_threshold * 1.2  # 20%上げる
                coordinates = peak_local_max(smoothed_data,
                                          min_distance=min_distance,
                                          threshold_abs=higher_threshold,
                                          exclude_border=False)
                
                # マスクを適用
                if len(coordinates) > 0:
                    mask = smoothed_data > higher_threshold
                    valid_coordinates = []
                    for coord in coordinates:
                        y, x = coord
                        if mask[y, x]:
                            valid_coordinates.append(coord)
                    coordinates = np.array(valid_coordinates) if valid_coordinates else np.empty((0, 2), dtype=int)
              
            if len(coordinates) == 0:
                return None
                
            # ピーク位置をマーカーとして保存
            self.peak_markers = np.zeros_like(data, dtype=int)
            for i, (y, x) in enumerate(coordinates):
                self.peak_markers[y, x] = i + 1  # 1から始まるラベル
            
            return self.peak_markers
            
        except Exception as e:
            print(f"[ERROR] peakDetection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def watershedFromPeaks(self, data):
        """ピークからWatershedによる粒子検出（改善版）"""
        try:
            
            
            if not hasattr(self, 'peak_markers') or self.peak_markers is None:
                peak_result = self.peakDetection(data)
                if peak_result is None:
                    return None
                self.peak_markers = peak_result
                
            # より適切な前処理
            from scipy.ndimage import gaussian_filter, distance_transform_edt
            from skimage.filters import sobel
            from skimage.segmentation import watershed
            
            # データを適度に平滑化（ノイズ除去）
            sigma = 0.5  # より小さな値
            smoothed_data = gaussian_filter(data, sigma=sigma)
            
            # より適切な勾配計算
            # AFMデータでは、sobelフィルターの方が適している
            gradient = sobel(smoothed_data)
            
            # ピークマーカーを改善
            # 各ピークの周囲に小さな領域を作成
            improved_markers = np.zeros_like(data, dtype=int)
            peak_positions = np.where(self.peak_markers > 0)
            
            for i, (y, x) in enumerate(zip(peak_positions[0], peak_positions[1])):
                # ピーク周囲の小さな領域を作成（3x3ピクセル）
                y_min = max(0, y-1)
                y_max = min(data.shape[0], y+2)
                x_min = max(0, x-1)
                x_max = min(data.shape[1], x+2)
                
                improved_markers[y_min:y_max, x_min:x_max] = i + 1
            
            # より適切なマスク条件
            # UIで設定されたwatershed threshold値を使用
            watershed_threshold_percentile = self.watershed_threshold_spin.value()
            threshold = np.percentile(smoothed_data, watershed_threshold_percentile)
            mask = smoothed_data > threshold
            
            #print(f"[DEBUG] Watershed - threshold_percentile: {watershed_threshold_percentile}%, threshold_value: {threshold:.3f}")
            #print(f"[DEBUG] Watershed - mask coverage: {np.sum(mask)}/{mask.size} pixels ({100*np.sum(mask)/mask.size:.1f}%)")
            #print(f"[DEBUG] Watershed - peak markers: {len(peak_positions[0])} peaks")
            
            # Watershed実行（より緩いパラメータ）
            compactness = 0.1  # より緩い境界
            labels = watershed(gradient, improved_markers, 
                             mask=mask, 
                             compactness=compactness)
            
            #print(f"[DEBUG] Watershed - initial labels: {len(np.unique(labels)) - 1} particles")
            
            # 粒子サイズフィルタリング
            min_size = self.min_size_spin.value()
            max_size = self.max_size_spin.value()
            
            #print(f"[DEBUG] Watershed - size filtering: min={min_size}, max={max_size}")
            
            # 各ラベルのサイズをチェック
            unique_labels = np.unique(labels)
            filtered_labels = np.zeros_like(labels)
            
            valid_particles = 0
            size_distribution = []
            for label in unique_labels:
                if label == 0:  # 背景
                    continue
                    
                # ラベルのサイズを計算
                label_size = np.sum(labels == label)
                size_distribution.append(label_size)
                
                if min_size <= label_size <= max_size:
                    filtered_labels[labels == label] = label
                    valid_particles += 1
            
            #print(f"[DEBUG] Watershed - size distribution: min={min(size_distribution)}, max={max(size_distribution)}, mean={np.mean(size_distribution):.1f}")
            #print(f"[DEBUG] Watershed - after size filtering: {valid_particles} particles")
            
              
            # 画像端の粒子を除外
            if self.exclude_edge_check.isChecked() and valid_particles > 0:
                # 画像の境界に接するラベルを特定
                # 注: filtered_labels.shapeを使用するため、ROI画像の場合はROIのサイズで判定される
                edge_labels = set()
                height, width = filtered_labels.shape
                
                
                # 上下の境界
                edge_labels.update(filtered_labels[0, :])  # 上端
                edge_labels.update(filtered_labels[-1, :])  # 下端
                # 左右の境界
                edge_labels.update(filtered_labels[:, 0])  # 左端
                edge_labels.update(filtered_labels[:, -1])  # 右端
                
                # 背景ラベル（0）を除外
                edge_labels.discard(0)
                
                
                
                # 境界に接する粒子を除去
                for label in edge_labels:
                    filtered_labels[filtered_labels == label] = 0
                
                # 再ラベリング
                filtered_labels = measure.label(filtered_labels > 0)
                valid_particles = len(np.unique(filtered_labels)) - 1  # 背景を除く
                #print(f"[DEBUG] Watershed - After edge exclusion: {valid_particles} particles")
            
            # 結果が空でないかチェック
            if np.sum(filtered_labels) == 0:
                # パラメータを緩和して再試行
                threshold_relaxed = np.percentile(smoothed_data, max(50, watershed_threshold_percentile - 10))  # より緩い閾値
                mask_relaxed = smoothed_data > threshold_relaxed
                
                compactness_relaxed = 0.1  # より緩い境界
                labels = watershed(gradient, improved_markers, 
                                 mask=mask_relaxed, 
                                 compactness=compactness_relaxed)
                
                # サイズフィルタリングを緩和
                min_size_relaxed = max(1, min_size // 2)
                max_size_relaxed = max_size * 2
                
                 
                for label in unique_labels:
                    if label == 0:
                        continue
                    label_size = np.sum(labels == label)
                    if min_size_relaxed <= label_size <= max_size_relaxed:
                        filtered_labels[labels == label] = label
                        valid_particles += 1
            
            # 画面端の粒子を除外
            if self.exclude_edge_check.isChecked():
                # 画像の境界に接するラベルを特定
                edge_labels = set()
                height, width = filtered_labels.shape
                
                # 上下の境界
                edge_labels.update(filtered_labels[0, :])  # 上端
                edge_labels.update(filtered_labels[-1, :])  # 下端
                # 左右の境界
                edge_labels.update(filtered_labels[:, 0])  # 左端
                edge_labels.update(filtered_labels[:, -1])  # 右端
                
                # 背景ラベル（0）を除外
                edge_labels.discard(0)
                
                # 境界に接する粒子を除去
                for label in edge_labels:
                    filtered_labels[filtered_labels == label] = 0
            
            filtered_labels = self._apply_edge_exclusion(filtered_labels)
            
            # ステータスラベルを更新（Single Particleモードの場合のみ）
            self.updateParticleDetectionStatus(filtered_labels, "Peak Detection")
            
            return filtered_labels
            
        except Exception as e:
            print(f"[ERROR] Watershed from peaks failed: {e}")
            if self.analysis_mode == "Single Particle":
                self.particle_status_label.setText("✗ Detection failed")
                self.particle_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
                self.particle_status_label.setVisible(True)
            import traceback
            traceback.print_exc()
            return None
            
    def contourLevelFromPeaks(self, data):
        """ピークからContour Levelによる粒子検出"""
        try:
            #print(f"[DEBUG] contourLevelFromPeaks - Starting with data shape: {data.shape}")
            
            if not hasattr(self, 'peak_markers') or self.peak_markers is None:
                #print("[DEBUG] contourLevelFromPeaks - No peak markers, running peak detection")
                peak_result = self.peakDetection(data)
                if peak_result is None:
                    #print("[DEBUG] contourLevelFromPeaks - Peak detection failed")
                    return None
                self.peak_markers = peak_result
                
            # ピーク位置を取得
            peak_positions = np.where(self.peak_markers > 0)
            #print(f"[DEBUG] contourLevelFromPeaks - Found {len(peak_positions[0])} peaks")
            if len(peak_positions[0]) == 0:
                print("[WARNING] No peaks detected for contour level analysis")
                return None
                
            # Contour Levelパラメータを取得
            contour_level_percent = self.contour_level_spin.value()
            
            # ガウシアンフィルタで平滑化（ノイズ除去）
            from scipy.ndimage import gaussian_filter
            sigma = 0.5
            smoothed_data = gaussian_filter(data, sigma=sigma)
            
            #print(f"[DEBUG] contourLevelFromPeaks - Original data range: min={np.min(data):.3f}, max={np.max(data):.3f}")
            #print(f"[DEBUG] contourLevelFromPeaks - Smoothed data range: min={np.min(smoothed_data):.3f}, max={np.max(smoothed_data):.3f}")
            
            # 各ピークに対して等高線を計算
            from skimage import measure
            from scipy.spatial.distance import cdist
            
            # 結果のラベル画像を初期化
            result_labels = np.zeros_like(data, dtype=int)
            current_label = 1
            
            #print(f"[DEBUG] Contour Level - Processing {len(peak_positions[0])} peaks with {contour_level_percent}% contour level")
            
            for i, (y, x) in enumerate(zip(peak_positions[0], peak_positions[1])):
                # ピーク位置での高さを取得
                peak_height = smoothed_data[y, x]
                
                # 等高線の閾値を計算（ピーク高さから設定された割合だけ下がった値）
                # 例：ピーク高さZ、Contour Level 10%の場合、閾値はZ * 0.9（ピーク高さから10%下がった値）
                contour_threshold = peak_height * (1.0 - contour_level_percent / 100.0)
                
                # ピーク周囲の局所的な領域を抽出（半径20ピクセル程度）
                radius = 20
                y_min = max(0, y - radius)
                y_max = min(data.shape[0], y + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(data.shape[1], x + radius + 1)
                
                local_data = smoothed_data[y_min:y_max, x_min:x_max]
                local_peak_y = y - y_min
                local_peak_x = x - x_min
                
                #print(f"[DEBUG] Peak {i+1} at ({y}, {x}): height={peak_height:.3f}, contour_threshold={contour_threshold:.3f}")
                #print(f"[DEBUG] Peak {i+1} - Local data range: min={np.min(local_data):.3f}, max={np.max(local_data):.3f}")
                
                # 等高線を検出（ピークから下がった等高線を探す）
                try:
                    # 元のデータで等高線を検出
                    contours = measure.find_contours(local_data, contour_threshold)
                    
                    #print(f"[DEBUG] Peak {i+1} - Found {len(contours)} contours at threshold {contour_threshold:.3f}")
                    #print(f"[DEBUG] Peak {i+1} - Local data shape: {local_data.shape}, peak position: ({local_peak_y}, {local_peak_x})")
                    
                    # 等高線が見つからない場合、最も近い高さで等高線を探す
                    if len(contours) == 0:
                        #print(f"[DEBUG] Peak {i+1}: No contours found at initial threshold {contour_threshold:.3f}")
                        # 局所データの範囲を取得
                        local_min = np.min(local_data)
                        local_max = np.max(local_data)
                        
                        # 等高線の閾値が範囲内にあるかチェック
                        if contour_threshold < local_min:
                            # 閾値が最小値より小さい場合、最小値で等高線を探す
                            contour_threshold = local_min
                            #print(f"[DEBUG] Peak {i+1} - Threshold below local minimum, using local_min: {contour_threshold:.3f}")
                        elif contour_threshold > local_max:
                            # 閾値が最大値より大きい場合、最大値で等高線を探す
                            contour_threshold = local_max
                           # print(f"[DEBUG] Peak {i+1} - Threshold above local maximum, using local_max: {contour_threshold:.3f}")
                        else:
                            # 閾値が範囲内にあるが等高線が見つからない場合、最も近い高さを探す
                            # 局所データの一意な値を取得
                            unique_heights = np.unique(local_data)
                            # 閾値に最も近い高さを見つける
                            closest_height_idx = np.argmin(np.abs(unique_heights - contour_threshold))
                            contour_threshold = unique_heights[closest_height_idx]
                            #print(f"[DEBUG] Peak {i+1} - No contours found, using closest height: {contour_threshold:.3f}")
                        
                        # 新しい閾値で等高線を再検出
                        contours = measure.find_contours(local_data, contour_threshold)
                        #print(f"[DEBUG] Peak {i+1} - Retry with adjusted threshold: {len(contours)} contours found")
                    
                    if len(contours) > 0:
                        # ピーク位置に最も近い等高線を選択
                        best_contour = None
                        min_distance = float('inf')
                        
                        # 最も近い等高線を選択（ピークを囲む等高線を優先）
                        for contour in contours:
                            # 等高線がピーク位置を囲んでいるかチェック
                            from matplotlib.path import Path
                            contour_path = Path(contour)
                            if contour_path.contains_point([local_peak_x, local_peak_y]):
                                # ピークを囲む等高線を選択
                                best_contour = contour
                                #print(f"[DEBUG] Peak {i+1} - Found contour that contains peak")
                                break
                        
                        # ピークを囲む等高線が見つからない場合、最も近い等高線を選択
                        if best_contour is None:
                            for contour in contours:
                                distances = cdist(contour, [[local_peak_x, local_peak_y]])
                                min_dist = np.min(distances)
                                
                                if min_dist < min_distance:
                                    min_distance = min_dist
                                    best_contour = contour
                            #print(f"[DEBUG] Peak {i+1} - Selected nearest contour with distance {min_distance:.2f}")
                        
                        if best_contour is not None:
                            # 等高線をポリゴンマスクに変換
                            from skimage.draw import polygon
                            
                            # 等高線を整数座標に変換
                            contour_coords = best_contour.astype(int)
                            
                            # ポリゴンの内部を塗りつぶし
                            rr, cc = polygon(contour_coords[:, 0], contour_coords[:, 1], 
                                           shape=local_data.shape)
                            
                            # グローバル座標に変換
                            global_rr = rr + y_min
                            global_cc = cc + x_min
                            
                            # 境界チェック
                            valid_mask = (global_rr >= 0) & (global_rr < data.shape[0]) & \
                                       (global_cc >= 0) & (global_cc < data.shape[1])
                            
                            if np.any(valid_mask):
                                global_rr = global_rr[valid_mask]
                                global_cc = global_cc[valid_mask]
                                
                                # 結果にラベルを設定
                                result_labels[global_rr, global_cc] = current_label
                                current_label += 1
                                
                                #print(f"[DEBUG] Peak {i+1}: Contour found with {len(global_rr)} pixels")
                            #else:
                               # print(f"[DEBUG] Peak {i+1}: Contour outside image bounds")
                        #else:
                            #print(f"[DEBUG] Peak {i+1}: No suitable contour found")
                    #else:
                        #print(f"[DEBUG] Peak {i+1}: No contours found even after adjustment")
                        #print(f"[DEBUG] Peak {i+1}: This peak will not have a particle boundary")
                        
                except Exception as e:
                    print(f"[ERROR] Error processing peak {i+1}: {e}")
                    continue
            
            # 粒子サイズフィルタリング
            min_size = self.min_size_spin.value()
            max_size = self.max_size_spin.value()
            
            #print(f"[DEBUG] Contour Level - Size filtering: min={min_size}, max={max_size}")
            
            # 各ラベルのサイズをチェック
            unique_labels = np.unique(result_labels)
            #print(f"[DEBUG] contourLevelFromPeaks - Initial labels: {len(unique_labels) - 1} (excluding background)")
            #print(f"[DEBUG] contourLevelFromPeaks - Unique labels: {unique_labels}")
            filtered_labels = np.zeros_like(result_labels)
            
            valid_particles = 0
            size_distribution = []
            for label in unique_labels:
                if label == 0:  # 背景
                    continue
                    
                # ラベルのサイズを計算
                label_size = np.sum(result_labels == label)
                size_distribution.append(label_size)
                #print(f"[DEBUG] contourLevelFromPeaks - Label {label}: size={label_size}")
                
                if min_size <= label_size <= max_size:
                    filtered_labels[result_labels == label] = label
                    valid_particles += 1
                    #print(f"[DEBUG] contourLevelFromPeaks - Label {label}: ACCEPTED (size {label_size} within range {min_size}-{max_size})")
                #else:
                    #print(f"[DEBUG] contourLevelFromPeaks - Label {label}: REJECTED (size {label_size} outside range {min_size}-{max_size})")
            
           # if size_distribution:
                #print(f"[DEBUG] Contour Level - Size distribution: min={min(size_distribution)}, max={max(size_distribution)}, mean={np.mean(size_distribution):.1f}")
            #print(f"[DEBUG] Contour Level - After size filtering: {valid_particles} particles")
            
            # 画像端の粒子を除外
            if self.exclude_edge_check.isChecked() and valid_particles > 0:
                # 画像の境界に接するラベルを特定
                # 注: filtered_labels.shapeを使用するため、ROI画像の場合はROIのサイズで判定される
                edge_labels = set()
                height, width = filtered_labels.shape
                
                
                # 上下の境界
                edge_labels.update(filtered_labels[0, :])  # 上端
                edge_labels.update(filtered_labels[-1, :])  # 下端
                # 左右の境界
                edge_labels.update(filtered_labels[:, 0])  # 左端
                edge_labels.update(filtered_labels[:, -1])  # 右端
                
                # 背景ラベル（0）を除外
                edge_labels.discard(0)
                
                
                
                # 境界に接する粒子を除去
                for label in edge_labels:
                    filtered_labels[filtered_labels == label] = 0
                
                # 再ラベリング
                filtered_labels = measure.label(filtered_labels > 0)
                valid_particles = len(np.unique(filtered_labels)) - 1  # 背景を除く
                #print(f"[DEBUG] Contour Level - After edge exclusion: {valid_particles} particles")
            
            filtered_labels = self._apply_edge_exclusion(filtered_labels)
            
            # ステータスラベルを更新（Single Particleモードの場合のみ）
            self.updateParticleDetectionStatus(filtered_labels, "Peak Detection")
            
            return filtered_labels
            
        except Exception as e:
            print(f"[ERROR] Contour level from peaks failed: {e}")
            if self.analysis_mode == "Single Particle":
                self.particle_status_label.setText("✗ Detection failed")
                self.particle_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
                self.particle_status_label.setVisible(True)
            import traceback
            traceback.print_exc()
            return None
            
    def displayPeakOverlay(self):
        """ピーク位置をオーバーレイ表示（独立した表示用）"""
        try:
            if not hasattr(self, 'peak_markers') or self.peak_markers is None:
                return
                
            # サイズの整合性をチェック
            if not hasattr(self, 'filtered_data') or self.filtered_data is None:
                print("[ERROR] displayPeakOverlay - filtered_data is not available")
                return
                
            if self.peak_markers.shape != self.filtered_data.shape:
                # 画像サイズ変更後の自動再検出を試みる
                try:
                    peak_result = self.peakDetection(self.filtered_data)
                    if peak_result is not None and peak_result.shape == self.filtered_data.shape:
                        self.peak_markers = peak_result
                    else:
                        return
                except Exception:
                    return
                
            # ピーク位置を取得（最新のpeak_markersから）
            peak_positions = np.where(self.peak_markers > 0)
            
            if len(peak_positions[0]) > 0:
                # ベース画像を再表示（安全な方法で）
                self.displayFilteredImage()
                
                # ピーク位置を表示
                self.displayPeakPositions()
                
        except Exception as e:
            print(f"[ERROR] displayPeakOverlay failed: {e}")
            import traceback
            traceback.print_exc()
    
    def displayPeakPositions(self):
        """ピーク位置を表示（粒子境界と一緒に表示）"""
        try:
            if not hasattr(self, 'peak_markers') or self.peak_markers is None:
                return
                
            # サイズの整合性をチェック
            if not hasattr(self, 'filtered_data') or self.filtered_data is None:
                print("[ERROR] displayPeakPositions - filtered_data is not available")
                return
                
            if self.peak_markers.shape != self.filtered_data.shape:
                # 画像サイズ変更後の自動再検出を試みる
                try:
                    peak_result = self.peakDetection(self.filtered_data)
                    if peak_result is not None and peak_result.shape == self.filtered_data.shape:
                        self.peak_markers = peak_result
                    else:
                        return
                except Exception:
                    return
                
            # ピーク位置を取得
            peak_positions = np.where(self.peak_markers > 0)
            
            if len(peak_positions[0]) > 0:
                # スキャンサイズ情報を使用
                scan_size_x = self.scan_size_x
                scan_size_y = self.scan_size_y
                
                # ピクセルサイズを計算（nm/pixel）
                pixel_size_x = scan_size_x / self.filtered_data.shape[1]
                pixel_size_y = scan_size_y / self.filtered_data.shape[0]
                
                # ピクセル座標を物理座標に変換
                x_coords = peak_positions[1] * pixel_size_x
                y_coords = peak_positions[0] * pixel_size_y
                
                # ピーク位置をプロット（黄色の十字で表示）
                peak_plot = self.image_axes.plot(x_coords, y_coords, 'y+', 
                                               markersize=12, markeredgewidth=2, alpha=0.8)
                self.overlay_artists.extend(peak_plot)
                
                #print(f"[DEBUG] displayPeakPositions - Displayed {len(peak_positions[0])} peak positions")
                
        except Exception as e:
            print(f"[ERROR] displayPeakPositions failed: {e}")
            import traceback
            traceback.print_exc()

    def calculateParticleProperties(self):
        """粒子の特性を計算（物理単位）"""
        try:
            properties = []
            
            # 検出された粒子がNoneでないかチェック
            if not hasattr(self, 'detected_particles') or self.detected_particles is None:
                return []
                
            # 元データ（gv.aryData）を使用
            if not hasattr(gv, 'aryData') or gv.aryData is None:
                print(f"[ERROR] No original data available in gv.aryData")
                return []
            
            # 使用する強度画像とスキャンサイズを決定
            if (self.analysis_mode == "Single Particle" and
                self.roi_scale_info is not None and
                hasattr(self, 'filtered_data') and self.filtered_data is not None):
                intensity_image = self.filtered_data
                scan_size_x = self.roi_scale_info['scan_size_x']
                scan_size_y = self.roi_scale_info['scan_size_y']
            else:
                intensity_image = gv.aryData
                scan_size_x = self.scan_size_x
                scan_size_y = self.scan_size_y
            
            if intensity_image is None:
                return []
            
            # スキャンサイズが0の場合はエラー
            if scan_size_x == 0 or scan_size_y == 0:
                print(f"[ERROR] Scan size is 0, cannot calculate particle properties")
                print(f"[ERROR] Please load AFM data first to get scan size information")
                return []
            
            # ピクセルサイズを計算（nm/pixel）
            pixel_size_x = scan_size_x / intensity_image.shape[1]
            pixel_size_y = scan_size_y / intensity_image.shape[0]
            
             
            # 各粒子の特性を計算（元データを使用）
            for region in measure.regionprops(self.detected_particles, intensity_image=intensity_image):
                prop = {}
                
                # 物理単位での面積（nm²）
                if self.area_check.isChecked():
                    prop['area'] = region.area * pixel_size_x * pixel_size_y
                    
                # 物理単位での周長（nm）
                if self.perimeter_check.isChecked():
                    # 周長は境界の長さなので、各方向のピクセルサイズを考慮
                    # 境界の各ピクセルがX方向とY方向のどちらに移動するかを考慮する必要があるが、
                    # 簡易的に平均ピクセルサイズを使用（より正確な計算は複雑）
                    avg_pixel_size = (pixel_size_x + pixel_size_y) / 2
                    prop['perimeter'] = region.perimeter * avg_pixel_size
                    
                # 円形度
                if self.circularity_check.isChecked():
                    if region.area > 0:
                        prop['circularity'] = 4 * np.pi * region.area / (region.perimeter ** 2)
                    else:
                        prop['circularity'] = 0
                        
                # 最大高さ（nm）
                if self.max_height_check.isChecked():
                    prop['max_height'] = region.max_intensity
                    
                # 平均高さ（nm）
                if self.mean_height_check.isChecked():
                    prop['mean_height'] = region.mean_intensity
                    
                # 体積（nm³）
                if self.volume_check.isChecked():
                    # 体積 = 面積 × 平均高さ（高さはnm単位なので、面積の単位変換のみ）
                    prop['volume'] = region.area * region.mean_intensity * pixel_size_x * pixel_size_y
                    
                # 重心座標（物理単位）
                centroid = region.centroid
                prop['centroid_x'] = centroid[1] * pixel_size_x  # X座標
                prop['centroid_y'] = centroid[0] * pixel_size_y  # Y座標
                
                properties.append(prop)
                
                
            return properties
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate particle properties: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def exportResults(self):
        """結果をエクスポート（物理単位）"""
        if not hasattr(self, 'particle_properties') or not self.particle_properties:
            QtWidgets.QMessageBox.warning(self, "Warning", 
                "No analysis results to export.\nエクスポートする解析結果がありません。")
            return
            
        try:
            # 現在選択されているASDファイルの情報を取得
            if not hasattr(gv, 'files') or not gv.files or not hasattr(gv, 'currentFileNum'):
                QtWidgets.QMessageBox.warning(self, "Warning", 
                    "No file selected.\nファイルが選択されていません。")
                return
                
            if gv.currentFileNum >= len(gv.files):
                QtWidgets.QMessageBox.warning(self, "Warning", 
                    "Invalid file selection.\n無効なファイル選択です。")
                return
            
            # 現在選択されているASDファイルのパスとファイル名を取得
            current_file_path = gv.files[gv.currentFileNum]
            current_file_dir = os.path.dirname(current_file_path)
            current_file_name = os.path.splitext(os.path.basename(current_file_path))[0]
            
            # デフォルトの保存ファイル名を設定
            default_filename = f"{current_file_name}_particle.csv"
            default_save_path = os.path.join(current_file_dir, default_filename)
            
            # ファイル保存ダイアログを表示
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export Particle Analysis Results", 
                default_save_path, 
                "CSV Files (*.csv)"
            )
            
            if not filename:
                return
            
            # 全フレーム結果があるかチェック
            has_all_frames = hasattr(self, 'all_frame_results') and self.all_frame_results
            
            if has_all_frames:
                # 全フレームのデータを統合（指定された形式）
                all_data = []
                
                # ヘッダー行（1行目）
                headers = ["Frame Number", "Particle No.", "Area (nm²)", "Perimeter (nm)", 
                          "Circularity", "Max Height (nm)", "Mean Height (nm)", 
                          "Volume (nm³)", "Centroid X (nm)", "Centroid Y (nm)"]
                all_data.append(headers)
                 
                # データ行（2行目以降）
                for frame_idx in range(gv.FrameNum):
                    if frame_idx in self.all_frame_results:
                        result = self.all_frame_results[frame_idx]
                        particle_properties = result['particle_properties']
                        
                        # 各粒子のデータを追加
                        for i, prop in enumerate(particle_properties):
                            # 有効な粒子のみエクスポート（削除された粒子は除外）
                            if prop.get('area', 0) > 0:  # 面積が0より大きい粒子のみ
                                row_data = [
                                    str(frame_idx + 1),  # Frame Number
                                    str(i + 1),  # Particle No.
                                    f"{prop.get('area', 0):.2f}",  # Area (nm²)
                                    f"{prop.get('perimeter', 0):.2f}",  # Perimeter (nm)
                                    f"{prop.get('circularity', 0):.3f}",  # Circularity
                                    f"{prop.get('max_height', 0):.2f}",  # Max Height (nm)
                                    f"{prop.get('mean_height', 0):.2f}",  # Mean Height (nm)
                                    f"{prop.get('volume', 0):.2f}",  # Volume (nm³)
                                    f"{prop.get('centroid_x', 0):.1f}",  # Centroid X (nm)
                                    f"{prop.get('centroid_y', 0):.1f}"   # Centroid Y (nm)
                                ]
                                all_data.append(row_data)
                
                
                # CSVファイルに保存
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    import csv
                    writer = csv.writer(csvfile)
                    
                    # 1行ずつ出力してデバッグ
                    for i, row in enumerate(all_data):
                        writer.writerow(row)
                        
                
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                                                
                except Exception as e:
                    print(f"[ERROR] CSVファイル読み込みエラー: {e}")
                    
                    QtWidgets.QMessageBox.information(self, "Success", 
                        f"All frames results exported to {filename}\n全フレーム結果を{filename}にエクスポートしました")
            else:
                # 現在のフレームのみエクスポート（同じ形式）
                all_data = []
                
                # ヘッダー行（1行目）
                headers = ["Frame Number", "Particle No.", "Area (nm²)", "Perimeter (nm)", 
                          "Circularity", "Max Height (nm)", "Mean Height (nm)", 
                          "Volume (nm³)", "Centroid X (nm)", "Centroid Y (nm)"]
                all_data.append(headers)
                
                # データ行（2行目以降）
                current_frame = gv.index + 1  # 現在のフレーム番号
                for i, prop in enumerate(self.particle_properties):
                    row_data = [
                        str(current_frame),  # Frame Number
                        str(i + 1),  # Particle No.
                        f"{prop.get('area', 0):.2f}",  # Area (nm²)
                        f"{prop.get('perimeter', 0):.2f}",  # Perimeter (nm)
                        f"{prop.get('circularity', 0):.3f}",  # Circularity
                        f"{prop.get('max_height', 0):.2f}",  # Max Height (nm)
                        f"{prop.get('mean_height', 0):.2f}",  # Mean Height (nm)
                        f"{prop.get('volume', 0):.2f}",  # Volume (nm³)
                        f"{prop.get('centroid_x', 0):.1f}",  # Centroid X (nm)
                        f"{prop.get('centroid_y', 0):.1f}"   # Centroid Y (nm)
                    ]
                    all_data.append(row_data)
                  
                # CSVファイルに保存
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    import csv
                    writer = csv.writer(csvfile)
                    
                    # 1行ずつ出力してデバッグ
                    for i, row in enumerate(all_data):
                        writer.writerow(row)
                       
                # 出力されたファイルの内容を確認
                
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                        
                except Exception as e:
                    print(f"[ERROR] CSVファイル読み込みエラー: {e}")
                    
                    QtWidgets.QMessageBox.information(self, "Success", 
                        f"Results exported to {filename}\n結果を{filename}にエクスポートしました")
                    
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", 
                f"Error during export: {str(e)}\nエクスポート中にエラーが発生しました: {str(e)}")
                
    def getCurrentData(self):
        """現在のデータを取得"""
        try:          
            
            # gvからデータを直接取得（ファイル変更時は常に更新）
            if hasattr(gv, 'aryData') and gv.aryData is not None:
                
                # ファイル変更時は常にcurrent_dataを更新
                self.current_data = gv.aryData.copy()
                           
                # filtered_dataも初期化（ファイル変更時はリセット）
                self.filtered_data = None
                
                # 検出結果もリセット（ファイル変更時）
                self.detected_particles = None
                self.particle_properties = None
                if hasattr(self, 'peak_markers'):
                    self.peak_markers = None
            else:                
                return False
                
            if self.current_data is None:
                print("[ERROR] No data available")
                return False
 
            
            # gvの全属性を確認
            gv_attrs = [attr for attr in dir(gv) if not attr.startswith('_')]
            scan_attrs = [attr for attr in gv_attrs if 'scan' in attr.lower() or 'size' in attr.lower()]
            pixel_attrs = [attr for attr in gv_attrs if 'pixel' in attr.lower()]
              
            # スキャンサイズ情報を正確に取得
            self.scan_size_x = getattr(gv, 'XScanSize', 0)
            self.scan_size_y = getattr(gv, 'YScanSize', 0)
            self.x_pixels = getattr(gv, 'XPixel', 0)
            self.y_pixels = getattr(gv, 'YPixel', 0)
            
            # スキャンサイズが0の場合はエラー
            if self.scan_size_x == 0 or self.scan_size_y == 0:
                print(f"[ERROR] Scan size not available from gv: XScanSize={self.scan_size_x}, YScanSize={self.scan_size_y}")
                print(f"[ERROR] Please load AFM data first to get scan size information")
                return False
                
            # ピクセル数が0の場合はエラー
            if self.x_pixels == 0 or self.y_pixels == 0:
                print(f"[ERROR] Pixel count not available from gv: XPixel={self.x_pixels}, YPixel={self.y_pixels}")
                print(f"[ERROR] Please load AFM data first to get pixel information")
                return False
              
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to get current data: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def loadWindowSettings(self):
        """ウィンドウ設定を読み込み"""
        try:
            settings = QtCore.QSettings()
            
            # ウィンドウの位置とサイズを復元
            geometry = settings.value('particle_analysis/geometry')
            if geometry:
                self.restoreGeometry(geometry)
            
            # 検出方法を復元
            detection_method = settings.value('particle_analysis/detection_method', 'Threshold')
            self.method_combo.setCurrentText(detection_method)
            
            # 閾値法を復元
            threshold_method = settings.value('particle_analysis/threshold_method', 'Otsu')
            self.threshold_combo.setCurrentText(threshold_method)
            
            # LEGEND表示設定を復元
            show_legend = settings.value('particle_analysis/show_legend', True, type=bool)
            self.show_legend_check.setChecked(show_legend)
            
            # Peak Detection パラメータを復元
            min_peak_distance = settings.value('particle_analysis/min_peak_distance', 5, type=int)
            self.min_peak_distance_spin.setValue(min_peak_distance)
            
            gradient_sigma = settings.value('particle_analysis/gradient_sigma', 1.5, type=float)
            self.gradient_sigma_spin.setValue(gradient_sigma)
            
            watershed_threshold = settings.value('particle_analysis/watershed_threshold', 70, type=int)
            self.watershed_threshold_spin.setValue(watershed_threshold)
            
            # Hessian Blob パラメータを復元
            hessian_min_sigma = settings.value('particle_analysis/hessian_min_sigma', 1.0, type=float)
            self.hessian_min_sigma_spin.setValue(hessian_min_sigma)
            
            hessian_max_sigma = settings.value('particle_analysis/hessian_max_sigma', 5.0, type=float)
            self.hessian_max_sigma_spin.setValue(hessian_max_sigma)
            
            hessian_threshold = settings.value('particle_analysis/hessian_threshold', 0.01, type=float)
            self.hessian_threshold_spin.setValue(hessian_threshold)
            
        except Exception as e:
            pass
        
    def saveWindowSettings(self):
        """ウィンドウ設定を保存"""
        try:
            if not hasattr(gv, 'windowSettings'):
                gv.windowSettings = {}
            gv.windowSettings['ParticleAnalysisWindow'] = {
                'x': self.x(),
                'y': self.y(),
                'width': self.width(),
                'height': self.height(),
                'detection_method': self.method_combo.currentText(),  # 検出方法を保存
                'threshold_method': self.threshold_combo.currentText(),  # 閾値法を保存
                'show_legend': self.show_legend_check.isChecked()  # LEGEND表示設定を保存
            }
            
            # Peak Detection パラメータを保存
            settings = QtCore.QSettings()
            settings.setValue('particle_analysis/min_peak_distance', self.min_peak_distance_spin.value())
            settings.setValue('particle_analysis/gradient_sigma', self.gradient_sigma_spin.value())
            settings.setValue('particle_analysis/watershed_threshold', self.watershed_threshold_spin.value())
            
            # Hessian Blob パラメータを保存
            settings.setValue('particle_analysis/hessian_min_sigma', self.hessian_min_sigma_spin.value())
            settings.setValue('particle_analysis/hessian_max_sigma', self.hessian_max_sigma_spin.value())
            settings.setValue('particle_analysis/hessian_threshold', self.hessian_threshold_spin.value())
            
        except Exception as e:
            print(f"Warning: Could not save particle analysis settings: {e}")
            
    def closeEvent(self, event):
        """ウィンドウが閉じられる時の処理"""
        self.saveWindowSettings()
        
        # MainWindowのsaveAllInitialParams()を呼び出してJSONファイルに保存
        if hasattr(gv, 'main_window') and gv.main_window and hasattr(gv.main_window, 'saveAllInitialParams'):
            gv.main_window.saveAllInitialParams()
        
        # ツールバーアクションのハイライトを解除
        try:
            if hasattr(gv, 'main_window') and gv.main_window:
                if hasattr(gv.main_window, 'setActionHighlight') and hasattr(gv.main_window, 'particles_action'):
                    gv.main_window.setActionHighlight(gv.main_window.particles_action, False)
        except Exception as e:
            print(f"[WARNING] Failed to reset particles action highlight: {e}")
        
        event.accept()
        
    def updateFromMainWindow(self):
        """メインウィンドウからの更新"""
        try:
                     
            # データを更新
            if self.getCurrentData():                               
                # All Frames後の場合は前処理をスキップ
                if hasattr(self, 'all_frame_results') and self.all_frame_results and gv.index in self.all_frame_results:
                    result = self.all_frame_results[gv.index]
                    self.filtered_data = result['filtered_data']
                    self.detected_particles = result['detected_particles']
                    self.particle_properties = result['particle_properties']
                    
                    # フィルタリングされた画像を表示
                    self.displayFilteredImage()
                    
                    # 粒子検出結果をオーバーレイ（保存された情報を使用）
                    if result['detected_particles'] is not None:
                        self.updateParticleOverlay()
                else:
                    # 通常の更新処理
                    # Modeに応じて使用するデータを決定
                    if self.analysis_mode == "Single Particle" and self.roi_data is not None:
                        # Single ParticleモードでROIが選択されている場合
                        # フレーム変更時はROI領域を再抽出する必要があるかチェック
                        # ここではROIデータを維持したまま前処理のみ更新
                        self.filtered_data = self.preprocessData(self.roi_data)
                    elif hasattr(self, 'current_data') and self.current_data is not None:
                        # All ParticlesモードまたはROI未選択の場合
                        # current_dataが存在する場合はそれを使用
                        self.filtered_data = self.preprocessData(self.current_data)
                    else:
                        # current_dataがない場合はgv.aryDataを使用
                        if hasattr(gv, 'aryData') and gv.aryData is not None:
                            self.filtered_data = self.preprocessData(gv.aryData)
                        else:
                            print(f"[ERROR] No data available for preprocessing")
                            return
                    
                    # フィルタリングされた画像を表示
                    self.displayFilteredImage()
                    
                    # 検出方法に応じて処理
                    detection_method = self.method_combo.currentText()
                    
                    if detection_method == "Peak Detection":
                        self.peakDetection(self.filtered_data)
                        # peakDetectionメソッド内でdisplayPeakOverlayが呼ばれるので、ここでは呼ばない
                    elif detection_method == "Hessian Blob":
                        self.hessianBlobDetection(self.filtered_data)
                        # 検出結果をオーバーレイ表示
                        self.updateOverlay()
                    else:
                        # 閾値法の場合はオーバーレイを更新
                        self.updateOverlay()
                    
                              
        except Exception as e:
            print(f"[ERROR] updateFromMainWindow failed: {e}")
            import traceback
            traceback.print_exc()

    def loadData(self, data):
        """データを読み込み"""
        try:
            if data is not None:
                self.current_data = data.copy()
                
                # ヒストグラムを更新
                self.updateHistogram()
                
                # 現在の検出方法に応じてプレビューを更新
                detection_method = self.method_combo.currentText()
                if detection_method == "Threshold":
                    threshold_method = self.threshold_combo.currentText()
                    if threshold_method == "Manual":
                        # Manualの場合はヒストグラムを更新
                        self.updateHistogram()
                    else:
                        # その他の場合は閾値プレビューを更新
                        self.updateThresholdPreviewForMethod(threshold_method)
                elif detection_method == "Peak Detection":
                    # Peak Detectionの場合はピークオーバーレイを表示
                    self.detectPeaksOnly()
                else:
                    # その他の場合は、マスクなしで画像を再表示
                    self.displayAFMImage()
                    
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            import traceback
            traceback.print_exc()

    def updateOverlay(self):
        """オーバーレイを更新"""
        try:
            # 検出方法を確認
            detection_method = self.method_combo.currentText()
            
            if detection_method == "Threshold":
                # 閾値法を確認
                threshold_method = self.threshold_combo.currentText()
                
                if threshold_method == "Manual":
                    # Manualの場合は閾値プレビューを更新
                    self.updateThresholdPreview(self.manual_thresh_spin.value())
                else:
                    # その他の場合は閾値プレビューを更新
                    self.updateThresholdPreviewForMethod(threshold_method)
            else:
                pass
                
        except Exception as e:
            pass

    def onCanvasResize(self, event):
        """キャンバスがリサイズされた時の処理"""
        # 元のリサイズイベントを呼び出し
        super(FigureCanvas, self.image_canvas).resizeEvent(event)

    def onTableResize(self, event):
        """テーブルがリサイズされた時の処理"""
        try:
            # テーブルのサイズが変更された時に列幅を調整
            if hasattr(self, 'results_table'):
                # テーブルの幅に応じて列幅を調整
                table_width = self.results_table.width()
                if table_width > 0:
                    # 利用可能な幅に応じて列幅を調整
                    header = self.results_table.horizontalHeader()
                    total_width = sum([self.results_table.columnWidth(i) for i in range(self.results_table.columnCount())])
                    
                    if total_width > 0 and table_width > total_width:
                        # テーブルが広くなった場合、最後の列を伸縮
                        header.setStretchLastSection(True)
            
            # 元のリサイズイベントを呼び出し
            super(QtWidgets.QTableWidget, self.results_table).resizeEvent(event)
            
        except Exception as e:
            # エラーが発生した場合は元のイベントのみ呼び出し
            super(QtWidgets.QTableWidget, self.results_table).resizeEvent(event)

    def showHessianThresholdContextMenu(self, position):
        """Hessian thresholdの右クリックメニューを表示"""
        try:
            menu = QtWidgets.QMenu(self)
            
            # ステップ値の選択メニュー
            step_menu = menu.addMenu("Step Value / ステップ値")
            
            # 0.001ステップ
            action_0001 = step_menu.addAction("0.001")
            action_0001.triggered.connect(lambda: self.setHessianThresholdStep(0.001))
            
            # 0.005ステップ
            action_0005 = step_menu.addAction("0.005")
            action_0005.triggered.connect(lambda: self.setHessianThresholdStep(0.005))
            
            # 0.01ステップ
            action_001 = step_menu.addAction("0.01")
            action_001.triggered.connect(lambda: self.setHessianThresholdStep(0.01))
            
            # 0.02ステップ
            action_002 = step_menu.addAction("0.02")
            action_002.triggered.connect(lambda: self.setHessianThresholdStep(0.02))
            
            # 現在のステップ値を表示
            current_step = self.hessian_threshold_spin.singleStep()
            step_menu.addSeparator()
            current_action = step_menu.addAction(f"Current: {current_step}")
            current_action.setEnabled(False)
            
            # メニューを表示
            menu.exec_(self.hessian_threshold_spin.mapToGlobal(position))
            
        except Exception as e:
            print(f"[ERROR] Failed to show Hessian threshold context menu: {e}")
            import traceback
            traceback.print_exc()
            
    def setHessianThresholdStep(self, step_value):
        """Hessian thresholdのステップ値を設定"""
        try:
            # 現在の値を取得
            current_value = self.hessian_threshold_spin.value()
            
            # ステップ値を変更
            self.hessian_threshold_spin.setSingleStep(step_value)
            
            # 小数点桁数を調整
            if step_value == 0.001:
                self.hessian_threshold_spin.setDecimals(3)
            elif step_value == 0.005:
                self.hessian_threshold_spin.setDecimals(3)
            elif step_value == 0.01:
                self.hessian_threshold_spin.setDecimals(2)
            elif step_value == 0.02:
                self.hessian_threshold_spin.setDecimals(2)
            
            #print(f"[DEBUG] Hessian threshold step changed to: {step_value}")
            
            # 値が変更された場合は自動的に検出を再実行
            if hasattr(self, 'current_data') and self.current_data is not None:
                self.onHessianParametersChanged()
                
        except Exception as e:
            print(f"[ERROR] Failed to set Hessian threshold step: {e}")
            import traceback
            traceback.print_exc()


__all__ = ["PLUGIN_NAME", "create_plugin", "ParticleAnalysisWindow"]
