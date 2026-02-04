import sys
import os
import numpy as np
import cv2
import traceback
from PyQt5 import QtWidgets, QtCore, QtGui
import globalvals as gv
from scipy.signal import find_peaks
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import least_squares
from helperFunctions import restore_window_geometry


try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import cm
    import matplotlib.colors as mcolors
    from matplotlib.patches import Ellipse
except ImportError:
    FigureCanvas = NavigationToolbar = Figure = cm = mcolors = Ellipse = None


HELP_HTML_EN = """
<h1>Particle Cluster Analysis</h1>
<h2>Overview</h2>
<p>This plugin detects particles in AFM images, computes the radial distribution function g(r), structural order parameters (Z, Psi6/Psi4/Psi2, Finger Tensor), segments clustered particles, and fits 2D ellipses. You can export and load results.</p>
<p><strong>How to open:</strong> From the pyNuD menu bar: <strong>Plugin → Particle Cluster Analysis</strong>. Use it with an image already opened in the main window; the <strong>current frame</strong> is used as the analysis source.</p>

<h2>Basic flow (flowchart)</h2>
<div class="step"><strong>1.</strong> Open image in main window</div>
<div class="step"><strong>2.</strong> P1: Click <strong>Detect Particles</strong></div>
<div class="step"><strong>3.</strong> (Optional) P2: Click <strong>Calculate g(r)</strong>, then set R1/V1/V2 from the plot if you need structure analysis</div>
<div class="step"><strong>4.</strong> P3–P5: Click <strong>Analyze Structure &amp; Tensor</strong></div>
<div class="step"><strong>5.</strong> (Optional) P7: <strong>Segment Clustered Particles</strong>, then P8: <strong>Identify 2D Ellipses</strong></div>
<div class="step"><strong>6.</strong> <strong>Export</strong> to save results</div>
<p>If you only need particle positions and g(r), you can stop after step 2 or 3. If you need structural maps (Z, Psi6, etc.), do step 4. If you need ellipse fits per particle, do steps 5 and 6.</p>

<h2>Parameters (detailed)</h2>
<h3>P1: Particle Extraction</h3>
<table class="param-table">
<tr><th>Item</th><th>Description</th><th>Range / Default</th><th>Tip</th></tr>
<tr><td>Est. Diameter (nm)</td><td>Estimated particle size used for detection (LoG filter scale).</td><td>1–500 nm, default 20</td><td>Set close to your actual particle diameter.</td></tr>
<tr><td>Threshold</td><td>Sensitivity for picking local maxima (particle peaks). Higher = fewer, stronger peaks.</td><td>0–1000, default 100</td><td>Adjust if too many or too few particles are detected.</td></tr>
<tr><td>Polarity</td><td>Bright: particles are higher (protrusions). Dark: particles are lower (depressions).</td><td>Bright / Dark</td><td>Match your image contrast.</td></tr>
<tr><td>Detect Particles</td><td>Runs detection on the current frame.</td><td>—</td><td>Run after setting Est. Diameter and Threshold.</td></tr>
<tr><td>Delete Edges</td><td>Removes particles near image edges (to avoid edge artifacts).</td><td>—</td><td>Use after detection if needed.</td></tr>
<tr><td>Manual Edit</td><td>Toggle: add or remove particles by clicking on the main image.</td><td>—</td><td>Turn on to correct detection mistakes.</td></tr>
<tr><td>Make Checkwave</td><td>Generates a checkwave from current particle list (for alignment checks).</td><td>—</td><td>Optional.</td></tr>
</table>
<h3>P2: g(r) Analysis</h3>
<p><strong>g(r)</strong> = radial distribution function: how particle density varies with distance from a reference particle.</p>
<table class="param-table">
<tr><th>Item</th><th>Description</th><th>Range / Default</th><th>Tip</th></tr>
<tr><td>dr (nm)</td><td>Bin width for g(r). Smaller = finer resolution, noisier.</td><td>1–50 nm, default 1</td><td>1–2 nm is often enough.</td></tr>
<tr><td>Calculate g(r)</td><td>Computes g(r) from current particle positions.</td><td>—</td><td>Run after P1 detection.</td></tr>
<tr><td>Pick None / R1 / V1 / V2</td><td>Click on g(r) plot to set R1 (first peak), V1, V2. These are used as cutoffs for structure analysis.</td><td>—</td><td>R1 = first neighbor shell; V1/V2 = distance range for Finger Tensor.</td></tr>
<tr><td>R1, V1, V2</td><td>Distance values (nm). R1 from g(r) first peak; V1/V2 set analysis range.</td><td>Manual input</td><td>You can type values or pick from the plot.</td></tr>
</table>
<h3>P3–P5: Mapping</h3>
<p><strong>Z</strong> = coordination number (number of neighbors). <strong>Psi6 / Psi4 / Psi2</strong> = bond-orientational order (hexagonal, rectangular, linear). <strong>Finger Tensor</strong> = local anisotropy (lambda1, lambda2, aspect).</p>
<table class="param-table">
<tr><th>Item</th><th>Description</th><th>Range / Default</th><th>Tip</th></tr>
<tr><td>Analyze Structure &amp; Tensor</td><td>Computes Z, Psi6/Psi4/Psi2, Finger Tensor and updates the structural maps on the right panel.</td><td>—</td><td>Uses R1, V1, V2. Run after P2 if you use g(r) cutoffs.</td></tr>
</table>
<h3>P6: Particle Navigation</h3>
<table class="param-table">
<tr><th>Item</th><th>Description</th><th>Range / Default</th><th>Tip</th></tr>
<tr><td>Particle ID</td><td>Select which particle to show in the zoom view.</td><td>0 to N−1</td><td>Change to inspect individual particles.</td></tr>
<tr><td>Z Range (nm)</td><td>Height range (relative to particle peak) for the zoom view.</td><td>Low–High nm, default 2–20</td><td>Narrow range shows a slice; wide range shows full height.</td></tr>
<tr><td>Z Tolerance (nm)</td><td>Slice thickness for height-based display.</td><td>1–50 nm, default 1</td><td>Smaller = thinner slice.</td></tr>
</table>
<h3>P7: Segmentation</h3>
<table class="param-table">
<tr><th>Item</th><th>Description</th><th>Range / Default</th><th>Tip</th></tr>
<tr><td>Basic / Advanced</td><td>Basic: single Gating Thresh. Advanced: Alpha (distance) and Beta (curvature) for weighted watershed.</td><td>—</td><td>Try Basic first.</td></tr>
<tr><td>Gating Thresh</td><td>Normalized curvature threshold (Basic mode). Higher = stricter, fewer regions.</td><td>0–1, default 0.2</td><td>Adjust if over/under-segmented.</td></tr>
<tr><td>Alpha (Dist)</td><td>Distance weight in Advanced mode.</td><td>0.1–5, default 1</td><td>Used with Beta.</td></tr>
<tr><td>Beta (Curv)</td><td>Curvature weight in Advanced mode.</td><td>0.1–20, default 5</td><td>Higher = more curvature-sensitive.</td></tr>
<tr><td>Sigma (Blur)</td><td>Blur (nm) before segmentation. Denoise: optional smoothing.</td><td>0.1–5 nm, default 1.2</td><td>Reduces noise; too large blurs boundaries.</td></tr>
<tr><td>Show Tags</td><td>Show segment IDs on the main image.</td><td>—</td><td>Turn on to check segmentation.</td></tr>
<tr><td>Conf</td><td>Confidence threshold for displayed segments.</td><td>0–1, default 0.75</td><td>Higher = only high-confidence segments.</td></tr>
<tr><td>Segment Clustered Particles</td><td>Runs segmentation and overlays labels on the image.</td><td>—</td><td>Run after P1 detection. Required before P8 ellipse fitting.</td></tr>
</table>
<h3>P8: Ellipse Fitting</h3>
<table class="param-table">
<tr><th>Item</th><th>Description</th><th>Range / Default</th><th>Tip</th></tr>
<tr><td>Identify 2D Ellipses</td><td>Fits an ellipse to each segmented particle. Results appear in the table (Axis A, B, Angle, RMSE) and in Fit Aspect / Orientation maps.</td><td>—</td><td>Run after P7 segmentation.</td></tr>
</table>
<h3>Export / Load</h3>
<table class="param-table">
<tr><th>Item</th><th>Description</th><th>Range / Default</th><th>Tip</th></tr>
<tr><td>Export</td><td>Saves particle list, g(r), structural results, and ellipse fits to a file.</td><td>—</td><td>Choose a path and filename.</td></tr>
<tr><td>Load</td><td>Loads a previously exported result file.</td><td>—</td><td>Restores particles and results for that file.</td></tr>
</table>

<h2>Workflow (step by step)</h2>
<div class="step"><strong>Step 1:</strong> Open your AFM image in the main window and go to the frame you want to analyze.</div>
<div class="step"><strong>Step 2:</strong> Set <strong>Est. Diameter</strong> (nm) to your particle size, adjust <strong>Threshold</strong> if needed, then click <strong>Detect Particles</strong>.</div>
<div class="step"><strong>Step 3:</strong> (Optional) Click <strong>Calculate g(r)</strong>. On the g(r) plot, choose <strong>R1</strong> (or V1/V2) and click to set the value, or type R1/V1/V2 manually.</div>
<div class="step"><strong>Step 4:</strong> Click <strong>Analyze Structure &amp; Tensor</strong> to compute order parameters and view maps in the right panel.</div>
<div class="step"><strong>Step 5:</strong> (Optional) Set P7 parameters and click <strong>Segment Clustered Particles</strong>, then click <strong>Identify 2D Ellipses</strong> in P8.</div>
<div class="step"><strong>Step 6:</strong> Click <strong>Export</strong> to save results to a file.</div>

<div class="note"><strong>Note:</strong> The analysis uses the <strong>current frame</strong> of the image in the main window. For 2ch data, make sure the correct channel is selected before running detection or analysis.</div>
"""

HELP_HTML_JA = """
<h1>Particle Cluster Analysis（粒子クラスター解析）</h1>
<h2>概要</h2>
<p>本プラグインは、AFM画像から粒子を検出し、動径分布関数 g(r)、構造秩序パラメータ（Z, Psi6/Psi4/Psi2, Finger Tensor）、クラスター粒子のセグメンテーション、2D楕円フィットを行います。結果のエクスポート・読み込みが可能です。</p>
<p><strong>起動方法:</strong> pyNuDメニュー <strong>Plugin → Particle Cluster Analysis</strong>。メインウィンドウで画像を開いた状態で使用し、<strong>現在のフレーム</strong>が解析対象になります。</p>

<h2>基本の流れ（フローチャート）</h2>
<div class="step"><strong>1.</strong> メインウィンドウで画像を開く</div>
<div class="step"><strong>2.</strong> P1: <strong>Detect Particles</strong> をクリック</div>
<div class="step"><strong>3.</strong>（任意）P2: <strong>Calculate g(r)</strong> をクリックし、必要ならプロットから R1/V1/V2 を設定</div>
<div class="step"><strong>4.</strong> P3–P5: <strong>Analyze Structure &amp; Tensor</strong> をクリック</div>
<div class="step"><strong>5.</strong>（任意）P7: <strong>Segment Clustered Particles</strong>、続けて P8: <strong>Identify 2D Ellipses</strong></div>
<div class="step"><strong>6.</strong> <strong>Export</strong> で結果を保存</div>
<p>粒子位置と g(r) だけ必要な場合は 2 または 3 で終了できます。構造マップ（Z, Psi6 など）が必要なら 4 を実行。粒子ごとの楕円フィットが必要なら 5 と 6 を実行してください。</p>

<h2>パラメータ（詳しい説明）</h2>
<h3>P1: Particle Extraction</h3>
<table class="param-table">
<tr><th>項目</th><th>説明</th><th>範囲・初期値</th><th>コツ</th></tr>
<tr><td>Est. Diameter (nm)</td><td>検出に使う粒子径の目安（LoGフィルタのスケール）。</td><td>1–500 nm、初期値 20</td><td>実際の粒子径に近く設定。</td></tr>
<tr><td>Threshold</td><td>局所極大（粒子ピーク）を選ぶ感度。大きいほど少なく・強いピークのみ。</td><td>0–1000、初期値 100</td><td>検出数が多すぎ・少なすぎなら調整。</td></tr>
<tr><td>Polarity</td><td>Bright: 粒子が高い（突出）。Dark: 粒子が低い（窪み）。</td><td>Bright / Dark</td><td>画像のコントラストに合わせる。</td></tr>
<tr><td>Detect Particles</td><td>現在フレームで検出を実行。</td><td>—</td><td>Est. Diameter と Threshold を設定してから実行。</td></tr>
<tr><td>Delete Edges</td><td>画像端付近の粒子を除外（端の影響を避ける）。</td><td>—</td><td>必要なら検出後に実行。</td></tr>
<tr><td>Manual Edit</td><td>オンにするとメイン画像をクリックして粒子を追加・削除できる。</td><td>—</td><td>検出ミスを直すときに使う。</td></tr>
<tr><td>Make Checkwave</td><td>現在の粒子リストからチェック波を生成（位置合わせの確認用）。</td><td>—</td><td>任意。</td></tr>
</table>
<h3>P2: g(r) Analysis</h3>
<p><strong>g(r)</strong> = 動径分布関数：基準粒子からの距離に対する粒子密度の変化。</p>
<table class="param-table">
<tr><th>項目</th><th>説明</th><th>範囲・初期値</th><th>コツ</th></tr>
<tr><td>dr (nm)</td><td>g(r) のビン幅。小さくすると細かく、ノイズは増える。</td><td>1–50 nm、初期値 1</td><td>1–2 nm で十分なことが多い。</td></tr>
<tr><td>Calculate g(r)</td><td>現在の粒子位置から g(r) を計算。</td><td>—</td><td>P1 検出後に実行。</td></tr>
<tr><td>Pick None / R1 / V1 / V2</td><td>g(r) プロットをクリックして R1（第1ピーク）、V1、V2 を設定。構造解析の距離範囲に使う。</td><td>—</td><td>R1 = 第1隣接殻。V1/V2 = Finger Tensor の範囲。</td></tr>
<tr><td>R1, V1, V2</td><td>距離（nm）。R1 は g(r) の第1ピーク。V1/V2 で解析範囲を指定。</td><td>手入力可</td><td>プロットでピックするか、数値入力。</td></tr>
</table>
<h3>P3–P5: Mapping</h3>
<p><strong>Z</strong> = 配位数（隣接粒子の数）。<strong>Psi6 / Psi4 / Psi2</strong> = 結合配向秩序（六方・四角・直線）。<strong>Finger Tensor</strong> = 局所異方性（lambda1, lambda2, aspect）。</p>
<table class="param-table">
<tr><th>項目</th><th>説明</th><th>範囲・初期値</th><th>コツ</th></tr>
<tr><td>Analyze Structure &amp; Tensor</td><td>Z, Psi6/Psi4/Psi2, Finger Tensor を計算し、右パネルの構造マップを更新。</td><td>—</td><td>R1, V1, V2 を使用。g(r) から決める場合は P2 の後に実行。</td></tr>
</table>
<h3>P6: Particle Navigation</h3>
<table class="param-table">
<tr><th>項目</th><th>説明</th><th>範囲・初期値</th><th>コツ</th></tr>
<tr><td>Particle ID</td><td>拡大表示する粒子を選択。</td><td>0 ～ N−1</td><td>個々の粒子を確認するときに変更。</td></tr>
<tr><td>Z Range (nm)</td><td>拡大ビューの高さ範囲（粒子ピークからの相対）。</td><td>Low–High nm、初期値 2–20</td><td>狭いとスライス、広いと全体の高さを表示。</td></tr>
<tr><td>Z Tolerance (nm)</td><td>高さスライスの厚さ。</td><td>1–50 nm、初期値 1</td><td>小さくすると薄いスライス。</td></tr>
</table>
<h3>P7: Segmentation</h3>
<table class="param-table">
<tr><th>項目</th><th>説明</th><th>範囲・初期値</th><th>コツ</th></tr>
<tr><td>Basic / Advanced</td><td>Basic: Gating Thresh のみ。Advanced: Alpha（距離）と Beta（曲率）で重み付き watershed。</td><td>—</td><td>まず Basic を試す。</td></tr>
<tr><td>Gating Thresh</td><td>正規化曲率閾値（Basic）。大きいほど厳しく、領域は少なく。</td><td>0–1、初期値 0.2</td><td>過剰・不足セグメントなら調整。</td></tr>
<tr><td>Alpha (Dist)</td><td>Advanced モードの距離の重み。</td><td>0.1–5、初期値 1</td><td>Beta と併用。</td></tr>
<tr><td>Beta (Curv)</td><td>Advanced モードの曲率の重み。</td><td>0.1–20、初期値 5</td><td>大きくすると曲率に敏感。</td></tr>
<tr><td>Sigma (Blur)</td><td>セグメンテーション前のぼかし（nm）。Denoise で平滑化オプション。</td><td>0.1–5 nm、初期値 1.2</td><td>ノイズ低減。大きすぎると境界がぼける。</td></tr>
<tr><td>Show Tags</td><td>メイン画像にセグメントIDを表示。</td><td>—</td><td>セグメント結果の確認用。</td></tr>
<tr><td>Conf</td><td>表示するセグメントの信頼度閾値。</td><td>0–1、初期値 0.75</td><td>大きくすると高信頼のみ表示。</td></tr>
<tr><td>Segment Clustered Particles</td><td>セグメンテーションを実行し、画像にラベルを重ねる。</td><td>—</td><td>P1 検出後に実行。P8 楕円フィットの前に必要。</td></tr>
</table>
<h3>P8: Ellipse Fitting</h3>
<table class="param-table">
<tr><th>項目</th><th>説明</th><th>範囲・初期値</th><th>コツ</th></tr>
<tr><td>Identify 2D Ellipses</td><td>各セグメント粒子に楕円をフィット。結果はテーブル（Axis A, B, Angle, RMSE）と Fit Aspect / Orientation マップに表示。</td><td>—</td><td>P7 セグメンテーションの後に実行。</td></tr>
</table>
<h3>Export / Load</h3>
<table class="param-table">
<tr><th>項目</th><th>説明</th><th>範囲・初期値</th><th>コツ</th></tr>
<tr><td>Export</td><td>粒子リスト、g(r)、構造結果、楕円フィットをファイルに保存。</td><td>—</td><td>保存先とファイル名を指定。</td></tr>
<tr><td>Load</td><td>以前エクスポートした結果ファイルを読み込む。</td><td>—</td><td>粒子と結果を復元。</td></tr>
</table>

<h2>ワークフロー（手順）</h2>
<div class="step"><strong>ステップ 1:</strong> メインウィンドウでAFM画像を開き、解析したいフレームに移動する。</div>
<div class="step"><strong>ステップ 2:</strong> <strong>Est. Diameter</strong> (nm) を粒子サイズに合わせ、必要なら <strong>Threshold</strong> を調整してから <strong>Detect Particles</strong> をクリック。</div>
<div class="step"><strong>ステップ 3:</strong>（任意）<strong>Calculate g(r)</strong> をクリック。g(r) プロットで <strong>R1</strong>（または V1/V2）を選びクリックして値を設定するか、R1/V1/V2 を手入力。</div>
<div class="step"><strong>ステップ 4:</strong> <strong>Analyze Structure &amp; Tensor</strong> をクリックして秩序パラメータを計算し、右パネルでマップを確認。</div>
<div class="step"><strong>ステップ 5:</strong>（任意）P7 のパラメータを設定し <strong>Segment Clustered Particles</strong> をクリック、続けて P8 の <strong>Identify 2D Ellipses</strong> をクリック。</div>
<div class="step"><strong>ステップ 6:</strong> <strong>Export</strong> をクリックして結果をファイルに保存。</div>

<div class="note"><strong>注意:</strong> 解析対象はメインウィンドウで表示している<strong>現在フレーム</strong>の画像です。2chデータの場合は、検出・解析前に正しいチャネルが選択されているか確認してください。</div>
"""


class QRangeSlider(QtWidgets.QWidget):
    """
    A custom dual-handle range slider widget for selecting a Z-height range.
    """
    valueChanged = QtCore.pyqtSignal(int, int)

    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._low = 2
        self._high = 20
        self._handle_radius = 7
        self._active_handle = None
        self.setMinimumHeight(24)
        self.setContentsMargins(5, 0, 5, 0)

    def setRange(self, min_val, max_val):
        self._min = min_val
        self._max = max_val
        # Enforce minimum gap of 1
        self._low = max(self._min, min(self._low, self._max - 1))
        self._high = max(self._low + 1, min(self._high, self._max))
        self.update()

    def setValue(self, low, high):
        # Enforce minimum gap of 1
        self._low = max(self._min, min(low, self._max - 1))
        self._high = max(self._low + 1, min(high, self._max))
        self.update()
        self.valueChanged.emit(self._low, self._high)

    def values(self): return self._low, self._high
    def low(self): return self._low
    def high(self): return self._high

    def _px_to_val(self, px):
        w = self.width() - 2 * self._handle_radius
        if w <= 0: return self._min
        ratio = (px - self._handle_radius) / w
        return int(self._min + ratio * (self._max - self._min))

    def _val_to_px(self, val):
        w = self.width() - 2 * self._handle_radius
        if self._max <= self._min: return self._handle_radius
        ratio = (val - self._min) / (self._max - self._min)
        return self._handle_radius + int(ratio * w)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        cy = self.height() // 2
        w = self.width()
        
        # Track
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(220, 220, 220))
        painter.drawRoundedRect(self._handle_radius, cy - 2, w - 2*self._handle_radius, 4, 2, 2)
        
        # Highlighted Range
        px_low = self._val_to_px(self._low)
        px_high = self._val_to_px(self._high)
        painter.setBrush(QtGui.QColor(59, 130, 246, 180)) # PRIMARY_500 with alpha
        painter.drawRect(px_low, cy - 2, px_high - px_low, 4)
        
        # Handles
        painter.setBrush(QtCore.Qt.white)
        painter.setPen(QtGui.QPen(QtGui.QColor(160, 160, 160), 1))
        painter.drawEllipse(QtCore.QPoint(px_low, cy), self._handle_radius, self._handle_radius)
        painter.drawEllipse(QtCore.QPoint(px_high, cy), self._handle_radius, self._handle_radius)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            x = event.x()
            p_low, p_high = self._val_to_px(self._low), self._val_to_px(self._high)
            if abs(x - p_low) < 15: self._active_handle = 'low'
            elif abs(x - p_high) < 15: self._active_handle = 'high'
            else: self._active_handle = None

    def mouseMoveEvent(self, event):
        if self._active_handle:
            val = max(self._min, min(self._max, self._px_to_val(event.x())))
            if self._active_handle == 'low':
                self._low = min(val, self._high - 1)
            else:
                self._high = max(self._low + 1, val)
            self.update()
            self.valueChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        self._active_handle = None

class SegmentationWorker(QtCore.QThread):
    """
    Worker thread to handle the Curvature-Distance Weighted Watershed computation.
    """
    finished = QtCore.pyqtSignal(object, object, object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, mode, height_data, particles_px, v1_nm, scan_size_nm, alpha, beta, sigma, denoise_on, threshold=0.2):
        super().__init__()
        self.mode = mode
        self.height_data = height_data
        self.particles_px = particles_px
        self.v1_nm = v1_nm
        self.scan_size_nm = scan_size_nm
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.denoise_on = denoise_on
        self.threshold = threshold

    def run(self):
        try:
            if self.mode == 'basic':
                # Local Curvature Gating (Basic Mode)
                labels, confidence, curv = local_curvature_gating_seg(
                    self.height_data, self.particles_px, self.v1_nm, self.scan_size_nm,
                    self.threshold, self.sigma, self.denoise_on
                )
            else:
                # Weighted Watershed ROI search (Advanced Mode)
                labels, confidence, curv = weighted_watershed_seg(
                    self.height_data, self.particles_px, self.v1_nm, self.scan_size_nm,
                    self.alpha, self.beta, self.sigma, self.denoise_on
                )
            self.finished.emit(labels, confidence, curv)
        except Exception as e:
            self.error.emit(str(e))

class EllipseFittingWorker(QtCore.QThread):
    """
    Worker thread to handle 2D Ellipse Fitting for all segmented particle masks.
    / すべてのセグメント化された粒子マスクに対して2D楕円フィッティングを処理するワーカースレッド。
    """
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, labels, px_size_nm, r1_nm, v1_nm, v2_nm):
        super().__init__()
        self.labels = labels
        self.px_size_nm = px_size_nm
        self.r1_nm = r1_nm
        self.v1_nm = v1_nm
        self.v2_nm = v2_nm

    def run(self):
        try:
            results = []
            if self.labels is None:
                self.finished.emit([])
                return
                
            unique_labels = np.unique(self.labels)
            particle_ids = unique_labels[unique_labels > 0]
            
            for pid in particle_ids:
                mask = (self.labels == pid)
                a, b, ang, rmse, x0, y0 = fit_ellipse_to_mask(
                    mask, self.px_size_nm, self.r1_nm, self.v1_nm, self.v2_nm
                )
                results.append({
                    'id': int(pid),
                    'a': a, 'b': b,
                    'angle': ang, 'rmse': rmse,
                    'x0': x0, 'y0': y0
                })
            self.finished.emit(results)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

class ParticleClusterWindow(QtWidgets.QMainWindow):
    """
    Particle Cluster Analysis Window for grain extraction and structural analysis.

    This window provides a GUI for detecting particles in AFM images,
    calculating radial distribution functions (g(r)), and generating 
    structural maps based on order parameters and the Finger Tensor.
    """
    def __init__(self, parent=None):
        super(ParticleClusterWindow, self).__init__(parent)
        self.parent = parent
        
        # Register with window manager
        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self, "sub")
        except ImportError:
            pass
            
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinMaxButtonsHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Particle Cluster Analysis / 粒子クラスター解析")
        
        # UI variables
        self.particles = []
        self.gr_data = None
        self.struct_results = {} # Store structural parameters (Coordination, Psi6, etc.)
        self.struct_maps = {}
        self.seg_labels = None
        self.seg_confidence = None
        self.fitting_results = {} # Store ellipsoid params indexed by particle ID
        
        self.setupUI()
        self._setup_menu_bar()
        self.restoreWindowSettings()

        # Connect to main window frame change signal
        if self.parent and hasattr(self.parent, 'frameChanged'):
            self.parent.frameChanged.connect(self.on_frame_changed)
            
        # Initial detection if image exists
        self.on_frame_changed()

    def setupUI(self):
        """
        Initializes the user interface components of the P-Film Analysis window.
        
        Sets up the three main panels:
        1. Controls: Extraction, g(r) parameters, and structural mapping.
        2. Visualizations: g(r) plot and individual particle zoom.
        3. Structural Maps: Tabbed views of various structural order parameters.
        """
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        self._setup_first_panel()
        self._setup_second_panel()
        self._setup_third_panel()

        self.splitter.setStretchFactor(0, 0); self.splitter.setStretchFactor(1, 1); self.splitter.setStretchFactor(2, 2)

    def _setup_menu_bar(self):
        """Add Help menu with Manual and About."""
        menubar = self.menuBar()
        menubar.setVisible(True)
        try:
            menubar.setNativeMenuBar(False)  # Show menu bar inside window (e.g. on macOS)
        except Exception:
            pass
        help_menu = menubar.addMenu("&Help")
        manual_action = QtWidgets.QAction("&Manual", self)
        manual_action.setShortcut("F1")
        manual_action.triggered.connect(self._show_help)
        help_menu.addAction(manual_action)
        help_menu.addSeparator()
        about_action = QtWidgets.QAction("&About Particle Cluster Analysis", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        menubar.update()

    def _show_help(self):
        """Help → Manual: show manual in QDialog (ja/English toggle), same style as Dwell Analysis."""
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
                dialog.setWindowTitle("粒子クラスター解析 - マニュアル")
                close_btn.setText("閉じる")
            else:
                browser.setHtml("<html><body>" + HELP_HTML_EN.strip() + "</body></html>")
                dialog.setWindowTitle("Particle Cluster Analysis - Manual")
                close_btn.setText("Close")

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        layout_dlg.addWidget(browser)
        layout_dlg.addWidget(close_btn)
        set_lang(False)  # default: English
        dialog.exec_()

    def _show_about(self):
        """Show About dialog."""
        QtWidgets.QMessageBox.about(
            self,
            "About Particle Cluster Analysis",
            "Particle Cluster Analysis / 粒子クラスター解析\n\n"
            "AFM image particle detection, g(r), structural order parameters, "
            "segmentation, and ellipse fitting.\n\n"
            "pyNuD plugin."
        )

    def showEvent(self, event):
        """Ensure menu bar is visible when window is shown (e.g. macOS)."""
        super(ParticleClusterWindow, self).showEvent(event)
        try:
            menubar = self.menuBar()
            if not menubar.isVisible() or not menubar.actions():
                menubar.setVisible(True)
                try:
                    menubar.setNativeMenuBar(False)
                except Exception:
                    pass
                if not menubar.actions():
                    menubar.clear()
                    self._setup_menu_bar()
        except Exception:
            pass

    def _setup_first_panel(self):
        """Setup the first panel (Controls) / 第1パネル（コントロール）のセットアップ"""
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        self.splitter.addWidget(left_panel)

        # P1: Extraction
        p1_group = QtWidgets.QGroupBox("P1: Particle Extraction")
        p1_layout = QtWidgets.QVBoxLayout(p1_group)
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel("Est. Diameter (nm):"))
        self.est_diameter = QtWidgets.QDoubleSpinBox()
        self.est_diameter.setRange(1.0, 500.0); self.est_diameter.setValue(20.0)
        sigma_layout.addWidget(self.est_diameter)
        p1_layout.addLayout(sigma_layout)

        thresh_layout = QtWidgets.QVBoxLayout()
        thresh_layout.addWidget(QtWidgets.QLabel("Threshold:"))
        self.thresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.thresh_slider.setRange(0, 1000); self.thresh_slider.setValue(100)
        self.thresh_slider.valueChanged.connect(self.on_param_changed)
        thresh_layout.addWidget(self.thresh_slider)
        p1_layout.addLayout(thresh_layout)

        polarity_layout = QtWidgets.QHBoxLayout()
        polarity_layout.addWidget(QtWidgets.QLabel("Polarity:"))
        self.polarity_bright = QtWidgets.QRadioButton("Bright"); self.polarity_bright.setChecked(True)
        self.polarity_dark = QtWidgets.QRadioButton("Dark")
        self.polarity_bright.toggled.connect(self.on_param_changed)
        polarity_layout.addWidget(self.polarity_bright); polarity_layout.addWidget(self.polarity_dark)
        p1_layout.addLayout(polarity_layout)

        self.detect_btn = QtWidgets.QPushButton("Detect Particles")
        self.detect_btn.clicked.connect(self.run_detection)
        p1_layout.addWidget(self.detect_btn)
        self.delete_edge_btn = QtWidgets.QPushButton("Delete Edges")
        self.delete_edge_btn.clicked.connect(self.delete_edges)
        p1_layout.addWidget(self.delete_edge_btn)
        
        self.manual_edit_btn = QtWidgets.QPushButton("Manual Edit")
        self.manual_edit_btn.setCheckable(True)
        self.manual_edit_btn.setStyleSheet("QPushButton:checked { background-color: #ffaa00; font-weight: bold; }")
        p1_layout.addWidget(self.manual_edit_btn)

        self.make_checkwave_btn = QtWidgets.QPushButton("Make Checkwave")
        self.make_checkwave_btn.clicked.connect(self.make_checkwave)
        p1_layout.addWidget(self.make_checkwave_btn)
        left_layout.addWidget(p1_group)

        # P2: g(r) Controls
        p2_group = QtWidgets.QGroupBox("P2: g(r) Analysis")
        p2_layout = QtWidgets.QVBoxLayout(p2_group)
        # Combined dr setting and Calculate button into one row
        # Spinbox stays next to label, button expands to fill remaining space
        dr_calc_layout = QtWidgets.QHBoxLayout()
        dr_calc_layout.setSpacing(8)
        dr_calc_layout.addWidget(QtWidgets.QLabel("dr(nm):"))
        self.gr_dr = QtWidgets.QSpinBox(); self.gr_dr.setRange(1, 50); self.gr_dr.setValue(1)
        self.gr_dr.setFixedWidth(50)
        dr_calc_layout.addWidget(self.gr_dr)
        self.calc_gr_btn = QtWidgets.QPushButton("Calculate g(r)")
        self.calc_gr_btn.clicked.connect(self.run_gr_analysis)
        dr_calc_layout.addWidget(self.calc_gr_btn, 1)
        p2_layout.addLayout(dr_calc_layout)
        pick_layout = QtWidgets.QHBoxLayout(); self.pick_mode_group = QtWidgets.QButtonGroup(self)
        self.pick_none = QtWidgets.QRadioButton("None"); self.pick_none.setChecked(True)
        self.pick_r1 = QtWidgets.QRadioButton("R1"); self.pick_v1 = QtWidgets.QRadioButton("V1"); self.pick_v2 = QtWidgets.QRadioButton("V2")
        for b in [self.pick_none, self.pick_r1, self.pick_v1, self.pick_v2]: pick_layout.addWidget(b); self.pick_mode_group.addButton(b)
        p2_layout.addLayout(pick_layout)
        # R1, V1, V2 inputs in a single row
        res_layout = QtWidgets.QHBoxLayout()
        self.res_r1 = QtWidgets.QLineEdit()
        self.res_v1 = QtWidgets.QLineEdit()
        self.res_v2 = QtWidgets.QLineEdit()
        self.res_v1.setText("700") # Default zoom/analysis range
        
        for label_text, widget in [("R1:", self.res_r1), ("V1:", self.res_v1), ("V2:", self.res_v2)]:
            res_layout.addWidget(QtWidgets.QLabel(label_text))
            res_layout.addWidget(widget)
            widget.setMaximumWidth(60) # Compact size for single row
            widget.editingFinished.connect(self.update_plot)
            
        p2_layout.addLayout(res_layout)
        left_layout.addWidget(p2_group)

        # P3-P5: Mapping Controls
        p3_group = QtWidgets.QGroupBox("P3-P5: Mapping")
        p3_layout = QtWidgets.QVBoxLayout(p3_group)
        self.analyze_struct_btn = QtWidgets.QPushButton("Analyze Structure & Tensor")
        self.analyze_struct_btn.clicked.connect(self.run_structural_analysis)
        p3_layout.addWidget(self.analyze_struct_btn)
        left_layout.addWidget(p3_group)
        
        # P6: Individual Analysis (Navigation)
        p6_group = QtWidgets.QGroupBox("P6: Particle Navigation")
        p6_layout = QtWidgets.QFormLayout(p6_group)
        self.p_spin = QtWidgets.QSpinBox()
        self.p_spin.setRange(0, 0)
        self.p_spin.valueChanged.connect(self.update_particle_zoom)
        p6_layout.addRow("Particle ID:", self.p_spin)
        
        # Range Slider for Z (Replaces two separate sliders)
        self.z_range_slider = QRangeSlider()
        self.z_range_slider.setRange(0, 100)
        self.z_range_slider.setValue(2, 20)
        self.z_range_lbl = QtWidgets.QLabel("2 - 20 nm")
        self.z_range_slider.valueChanged.connect(lambda l, h: (self.z_range_lbl.setText(f"{l} - {h} nm"), self.update_particle_zoom()))
        p6_layout.addRow("Z Range (nm):", self.z_range_slider)
        p6_layout.addRow("", self.z_range_lbl)

        self.z_tol_spin = QtWidgets.QSpinBox()
        self.z_tol_spin.setRange(1, 50); self.z_tol_spin.setValue(1); self.z_tol_spin.setSingleStep(1)
        self.z_tol_spin.valueChanged.connect(self.update_particle_zoom)
        p6_layout.addRow("Z Tolerance (nm):", self.z_tol_spin)
        
        left_layout.addWidget(p6_group)
        
        # P7: Segmentation Controls
        self._setup_segmentation_controls(left_layout)
        
        # P8: Ellipse Fitting (Separate Process)
        p8_group = QtWidgets.QGroupBox("P8: Ellipse Fitting")
        p8_layout = QtWidgets.QVBoxLayout(p8_group)
        self.fit_ellipse_btn = QtWidgets.QPushButton("Identify 2D Ellipses")
        self.fit_ellipse_btn.setMinimumHeight(36)
        self.fit_ellipse_btn.setStyleSheet("background-color: #10B981; color: white; font-weight: bold; border-radius: 4px;")
        self.fit_ellipse_btn.clicked.connect(self.run_ellipse_fitting)
        p8_layout.addWidget(self.fit_ellipse_btn)
        left_layout.addWidget(p8_group)

        # Export / Load controls
        export_load_layout = QtWidgets.QHBoxLayout()
        export_load_layout.setContentsMargins(0, 10, 0, 0)
        
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.setFixedSize(120, 40)
        self.export_btn.setStyleSheet("background-color: #6366F1; color: white; font-weight: bold; border-radius: 4px;")
        self.export_btn.clicked.connect(self.export_results)
        
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setFixedSize(120, 40)
        self.load_btn.setStyleSheet("background-color: #4B5563; color: white; font-weight: bold; border-radius: 4px;")
        self.load_btn.clicked.connect(self.load_results)
        
        export_load_layout.addWidget(self.export_btn)
        export_load_layout.addWidget(self.load_btn)
        export_load_layout.addStretch()
        
        left_layout.addLayout(export_load_layout)
        left_layout.addStretch()

    def _setup_segmentation_controls(self, layout):
        """Setup the P7: Segmentation Controls / 第7パネル（セグメンテーション制御）のセットアップ"""
        p7_group = QtWidgets.QGroupBox("P7: Segmentation Controls")
        p7_layout = QtWidgets.QVBoxLayout(p7_group)
        p7_layout.setContentsMargins(8, 16, 8, 8)
        p7_layout.setSpacing(8)

        # Strategy Switcher / 戦略セレクター
        mode_layout = QtWidgets.QHBoxLayout()
        self.seg_mode_basic = QtWidgets.QRadioButton("Basic")
        self.seg_mode_advanced = QtWidgets.QRadioButton("Advanced")
        self.seg_mode_basic.setChecked(True)
        mode_layout.addWidget(self.seg_mode_basic)
        mode_layout.addWidget(self.seg_mode_advanced)
        p7_layout.addLayout(mode_layout)

        # Form layout for parameters
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(6)

        # Basic Mode: Gating Threshold
        self.seg_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.seg_thresh_spin.setRange(0.0, 1.0); self.seg_thresh_spin.setValue(0.2)
        self.seg_thresh_spin.setSingleStep(0.05)
        self.seg_thresh_spin.setToolTip("Normalized curvature threshold (0.0=include all negative, 1.0=only peaks)")
        self.seg_thresh_row = (QtWidgets.QLabel("Gating Thresh:"), self.seg_thresh_spin)
        form_layout.addRow(self.seg_thresh_row[0], self.seg_thresh_row[1])

        # Advanced Mode: Alpha (Dist)
        self.seg_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.seg_alpha_spin.setRange(0.1, 5.0); self.seg_alpha_spin.setValue(1.0)
        self.seg_alpha_row = (QtWidgets.QLabel("Alpha (Dist):"), self.seg_alpha_spin)
        form_layout.addRow(self.seg_alpha_row[0], self.seg_alpha_row[1])

        # Advanced Mode: Beta (Curv)
        self.seg_beta_spin = QtWidgets.QDoubleSpinBox()
        self.seg_beta_spin.setRange(0.1, 20.0); self.seg_beta_spin.setValue(5.0)
        self.seg_beta_row = (QtWidgets.QLabel("Beta (Curv):"), self.seg_beta_spin)
        form_layout.addRow(self.seg_beta_row[0], self.seg_beta_row[1])

        # Shared: Sigma (Blur)
        self.seg_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.seg_sigma_spin.setRange(0.1, 5.0); self.seg_sigma_spin.setValue(1.2); self.seg_sigma_spin.setSuffix(" nm")
        self.seg_denoise_chk = QtWidgets.QCheckBox("Denoise")
        self.seg_denoise_chk.setChecked(True)
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(self.seg_sigma_spin); sigma_layout.addWidget(self.seg_denoise_chk)
        form_layout.addRow("Sigma (Blur):", sigma_layout)

        p7_layout.addLayout(form_layout)

        # Mode visibility control / モード表示の連動
        def on_mode_toggled():
            is_adv = self.seg_mode_advanced.isChecked()
            self.seg_thresh_spin.setVisible(not is_adv); self.seg_thresh_row[0].setVisible(not is_adv)
            self.seg_alpha_spin.setVisible(is_adv); self.seg_alpha_row[0].setVisible(is_adv)
            self.seg_beta_spin.setVisible(is_adv); self.seg_beta_row[0].setVisible(is_adv)

        self.seg_mode_basic.toggled.connect(on_mode_toggled)
        self.seg_mode_advanced.toggled.connect(on_mode_toggled)
        on_mode_toggled() # Initial state

        # Overlay Controls
        overlay_layout = QtWidgets.QHBoxLayout()
        self.seg_show_tags = QtWidgets.QCheckBox("Show Tags")
        self.seg_show_tags.toggled.connect(lambda: self.update_particle_display())
        overlay_layout.addWidget(self.seg_show_tags)
        
        self.seg_conf_spin = QtWidgets.QDoubleSpinBox()
        self.seg_conf_spin.setRange(0.0, 1.0); self.seg_conf_spin.setValue(0.75); self.seg_conf_spin.setSingleStep(0.05)
        self.seg_conf_spin.valueChanged.connect(lambda: self.update_particle_display())
        overlay_layout.addWidget(QtWidgets.QLabel("Conf:"))
        overlay_layout.addWidget(self.seg_conf_spin)
        p7_layout.addLayout(overlay_layout)

        # Segment Button
        self.segment_btn = QtWidgets.QPushButton("Segment Clustered Particles")
        self.segment_btn.setMinimumHeight(36)
        self.segment_btn.setStyleSheet("background-color: #3B82F6; color: white; font-weight: bold; border-radius: 4px;")
        self.segment_btn.clicked.connect(self.run_segmentation)
        p7_layout.addWidget(self.segment_btn)
        
        layout.addWidget(p7_group)

    def _setup_second_panel(self):
        """Setup the second panel (Results & Plot) / 第2パネル（結果表とグラフ）のセットアップ"""
        mid_panel = QtWidgets.QWidget()
        mid_layout = QtWidgets.QVBoxLayout(mid_panel)
        self.splitter.addWidget(mid_panel)
        
        self.results_table = QtWidgets.QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels([
            "ID", "X (nm)", "Y (nm)", "Z (nm)", 
            "Axis A", "Axis B", "Angle (vs X)", "RMSE"
        ])
        self.results_table.setMaximumHeight(150)
        self.results_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        mid_layout.addWidget(self.results_table, 0)

        # Plot Mode Selector / グラフモードセレクター
        mode_group = QtWidgets.QGroupBox("Display Mode")
        mode_group.setMaximumHeight(60)
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        self.plot_mode_gr = QtWidgets.QRadioButton("g(r) Function")
        self.plot_mode_ar = QtWidgets.QRadioButton("Aspect Ratio Hist")
        self.plot_mode_ori = QtWidgets.QRadioButton("Orientation Hist")
        self.plot_mode_gr.setChecked(True)
        
        for rb in [self.plot_mode_gr, self.plot_mode_ar, self.plot_mode_ori]:
            rb.toggled.connect(lambda: self.update_plot())
            mode_layout.addWidget(rb)
        mid_layout.addWidget(mode_group)

        if FigureCanvas is not None:
            self.fig = Figure(figsize=(4, 3), dpi=100); self.canvas = FigureCanvas(self.fig)
            self.canvas.setMaximumHeight(300)
            mid_layout.addWidget(NavigationToolbar(self.canvas, self)); mid_layout.addWidget(self.canvas, 0)
            self.ax = self.fig.add_subplot(111); self.ax.set_title("g(r)"); self.canvas.mpl_connect('button_press_event', self.on_plot_click)
            self.canvas.mpl_connect('motion_notify_event', self.on_gr_mouse_move)
            self.fig.tight_layout(pad=0.5)
            
            # Cursor elements for g(r)
            self.gr_cursor_line = None
            self.gr_cursor_dot = None

            # --- Local Zoom Plot ---
            self.zoom_fig = Figure(figsize=(4, 3), dpi=100); self.zoom_canvas = FigureCanvas(self.zoom_fig)
            self.zoom_canvas.setMaximumHeight(300)
            mid_layout.addWidget(self.zoom_canvas, 1)
            self.zoom_ax = self.zoom_fig.add_subplot(111); self.zoom_ax.set_title("Particle Zoom")
            self.zoom_fig.tight_layout(pad=0.1)
            
            # Coordinate overlay
            self.zoom_coord_lbl = QtWidgets.QLabel(self.zoom_canvas)
            self.zoom_coord_lbl.setStyleSheet("background-color: rgba(0, 0, 0, 160); color: white; padding: 2px 5px; border-radius: 4px; font-size: 8pt; border: 1px solid rgba(255, 255, 255, 50);")
            self.zoom_coord_lbl.hide()
            self.zoom_canvas.mpl_connect('motion_notify_event', self.on_zoom_mouse_move)
        else:
            mid_layout.addWidget(QtWidgets.QLabel("Matplotlib Error"))

    def _setup_third_panel(self):
        """Setup the third panel (Tabs) / 第3パネル（タブ表示）のセットアップ"""
        right_panel = QtWidgets.QWidget()
        right_panel.setMinimumSize(512, 512)
        right_layout = QtWidgets.QVBoxLayout(right_panel); right_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(right_panel)
        
        self.tab_widget = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tab_widget)
        
        self.map_canvases = {}
        self.map_axes = {}
        self.map_colorbars = {}
        
        map_titles = [
            ("Raw", "AFM View"),
            ("Curvature", "Hessian Map (Curv)"),
            ("Z", "Coordination (Z)"), ("Psi6", "Hexagonal (Psi6)"), 
            ("Psi4", "Rectangular (Psi4)"), ("Psi2", "Linear (Psi2)"),
            ("lambda1", "Lambda 1"), ("lambda2", "Lambda 2"),
            ("aspect", "Anisotropy"),
            ("fit_aspect", "Fit Aspect"), ("fit_ori", "Orientation (vs X)")
        ]
        for key, label in map_titles:
            self._create_mapping_tab(key, label)

    def _create_mapping_tab(self, key, label):
        """Creates a single structural mapping tab / 構造マップのタブを作成"""
        tab = QtWidgets.QWidget(); tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0); tab_layout.setSpacing(0)
        try:
            fig = Figure(figsize=(5, 5), dpi=100); canvas = FigureCanvas(fig)
            tab_layout.addWidget(NavigationToolbar(canvas, self)); tab_layout.addWidget(canvas)
            ax = fig.add_subplot(111); ax.set_title(label)
            self.map_canvases[key] = canvas; self.map_axes[key] = ax
            fig.tight_layout(pad=0.1)
            if key == 'Raw': canvas.mpl_connect('button_press_event', self.on_particle_click)
        except: tab_layout.addWidget(QtWidgets.QLabel("Plotting Error"))
        self.tab_widget.addTab(tab, label)

    def restoreWindowSettings(self):
        """Restores the window's geometric state (position and size) from settings."""
        restore_window_geometry(self, 'ParticleClusterWindow', 100, 100, 1400, 800)

    def closeEvent(self, event):
        """
        Handles the window close event.
        
        Saves current settings and notifies the parent window to update 
        toolbar highlights.
        """
        self.saveWindowSettings()
        
        # Disconnect from parent signals to prevent background processing
        # 親信号から切断し、バックグラウンド処理を停止します
        if self.parent and hasattr(self.parent, 'frameChanged'):
            try:
                self.parent.frameChanged.disconnect(self.on_frame_changed)
            except (TypeError, RuntimeError):
                pass
                
        # Unregister from window manager
        try:
            from window_manager import unregister_pyNuD_window
            unregister_pyNuD_window(self)
        except ImportError:
            pass
            
        # Safety: Ensure worker isstopped
        if hasattr(self, 'seg_worker') and self.seg_worker and self.seg_worker.isRunning():
            self.seg_worker.wait(500) # Give it 500ms to finish or it will be abandoned by the OS
            
        if self.parent:
            if hasattr(self.parent, 'particle_cluster_action'): self.parent.setActionHighlight(self.parent.particle_cluster_action, False)
        event.accept()

    def saveWindowSettings(self):
        try:
            geometry = self.geometry(); window_settings = getattr(gv, 'windowSettings', {})
            window_settings['ParticleClusterWindow'] = {'width': geometry.width(), 'height': geometry.height(), 'x': geometry.x(), 'y': geometry.y(), 'visible': False, 'title': self.windowTitle(), 'class_name': 'ParticleClusterWindow'}
            gv.windowSettings = window_settings
            if self.parent and hasattr(self.parent, 'saveAllInitialParams'): self.parent.saveAllInitialParams()
        except: pass

    def on_frame_changed(self): self.run_detection()
    def on_param_changed(self): self.run_detection()

    def run_detection(self):
        """
        Executes particle detection on the current main image.

        Applies a LoG filter followed by a local maximum search. Results
        are used to update the results table, particle display, and zoom view.
        """
        """
        Executes the particle extraction pipeline (P1).

        Workflow:
        1. Retrieves raw AFM height data from the global state `gv.aryData`.
        2. Applies a Laplacian of Gaussian (LoG) filter to enhance particle edges.
        3. Performs a local maximum search to locate particle centers.
        4. Stores the results in `self.particles`.

        Side Effects:
            Updates `self.particles`: Overwrites the current list of coordinates [y, x].
            Calls `self.update_results()`: Refreshes the results table UI.
            Calls `self.update_particle_display()`: Refreshes the canvas.

        Raises:
            AttributeError: If `gv.aryData` is not initialized or is None.
        """
        if not hasattr(gv, 'aryData') or gv.aryData is None: return
        try:
            # Use raw data for detection to ensure coordinate stability
            data = gv.aryData.copy()
            h, w = data.shape[:2]
            px_size_x = gv.XScanSize / w if w > 0 else 1.0
            
            # LoG Filter on raw data
            filtered = log_filter(data, self.est_diameter.value(), px_size_x)
            if self.polarity_bright.isChecked(): filtered = -filtered
            
            f_min, f_max = np.nanmin(filtered), np.nanmax(filtered)
            norm_f = (filtered - f_min) / (f_max - f_min) if f_max > f_min else filtered
            
            # Particle detection (returns coordinates in raw pixels)
            self.particles = local_max_search(norm_f, self.thresh_slider.value() / 1000.0, filter_size=5)
            self.p_spin.setRange(0, max(0, len(self.particles) - 1))
            
            self.update_results()
            self.update_particle_display()
            self.update_particle_zoom()
            # self.tab_widget.setCurrentIndex(0) # Keep current tab
        except Exception as e: print(f"Detection Error: {e}")

    def update_particle_display(self, highlight_idx=None, neighbors=None, ellipse_params=None):
        """
        Updates the main AFM visualization with particle markers.

        Parameters
        ----------
        highlight_idx : int, optional
            Index of a particle to highlight (e.g., when selected).
        neighbors : list of int, optional
            Indices of neighbor particles to highlight.
        ellipse_params : tuple, optional
            Parameters (cx, cy, v1, v2) for drawing a search ellipse.
        """
        """
        Renders the AFM image and overlays particle markers on the 'Raw' canvas.

        This function handles the critical conversion from Pixel coordinates (`self.particles`)
        to Physical coordinates (nm) for accurate plotting. It also manages the highligting
        of selected particles and their structural neighbors.

        Args:
            highlight_idx (int, optional): The index of the particle to highlight (Index in self.particles).
            neighbors (list[int], optional): A list of indices representing the particle's neighbors
                (rendered in green).
            ellipse_params (tuple, optional): Parameters for the search ellipse (cx, cy, v1, v2)
                in nanometers.

        Note:
            While the background image uses `gv.dspimg` for rendering speed, the particle
            coordinates are mapped based on `gv.XScanSize` and `gv.YScanSize`.
        """

        if not hasattr(gv, 'aryData') or gv.aryData is None: return
        if 'Raw' not in self.map_axes: return
        
        h_ary, w_ary = gv.aryData.shape[:2]
        px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
        # Map raw data indices (Row 0 = Bottom) -> NM (No flip)
        coords_nm = np.column_stack((self.particles[:, 1] * px_x, self.particles[:, 0] * px_y))
        
        ax = self.map_axes['Raw']; ax.clear()
        
        # Use gv.aryData for quantitative height visualization (provides colorbar support)
        # Use 'afmhot' as default or infer from global if possible, but afmhot is safe/standard
        cmap_name = 'afmhot'
        im = ax.imshow(gv.aryData, origin='lower', extent=[0, gv.XScanSize, 0, gv.YScanSize], cmap=cmap_name)
        
        # Add or update colorbar for Height
        # Use fixed fraction/pad to align with other maps
        if 'Raw' not in self.map_colorbars:
            self.map_colorbars['Raw'] = ax.figure.colorbar(im, ax=ax, label='Height (nm)', fraction=0.046, pad=0.04)
        else:
            self.map_colorbars['Raw'].update_normal(im)
            
        ax.set_xlim(0, gv.XScanSize)
        ax.set_ylim(0, gv.YScanSize)
        ax.set_aspect('equal')
        
        # Overlay particles in nm coordinates
        if len(self.particles) > 0:
            px_nm = coords_nm[:, 0]
            py_nm = coords_nm[:, 1]
            
            # Default particles
            ax.scatter(px_nm, py_nm, s=15, c='red', marker='o', edgecolors='white', linewidths=0.5, alpha=0.6)
            
            if highlight_idx is not None:
                # Highlight selected particle
                ax.scatter(px_nm[highlight_idx], py_nm[highlight_idx], s=40, c='yellow', marker='*', edgecolors='black', linewidths=1, zorder=10)
                
                if neighbors is not None and len(neighbors) > 0:
                    # Highlight neighbors
                    nx_nm = coords_nm[neighbors, 0]
                    ny_nm = coords_nm[neighbors, 1]
                    ax.scatter(nx_nm, ny_nm, s=30, c='lime', marker='o', edgecolors='black', linewidths=1, zorder=5)
                
                if ellipse_params and Ellipse is not None:
                    # Draw search ellipse
                    _, _, v1_nm, v2_nm = ellipse_params
                    cx_nm, cy_nm = px_nm[highlight_idx], py_nm[highlight_idx]
                    ell = Ellipse(xy=(cx_nm, cy_nm), width=2*v1_nm, height=2*v2_nm, angle=0, 
                                  edgecolor='yellow', fc='none', lw=1, ls='--', alpha=0.8)
                    ax.add_patch(ell)

        ax.set_title(f"AFM View ({len(self.particles)} particles)")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        
        # Custom coordinate formatter for X, Y, Z
        ax.format_coord = self.format_raw_coord
        
        # Segmentation Overlay / セグメンテーションのオーバーレイ表示
        if self.seg_labels is not None and hasattr(self, 'seg_show_tags') and self.seg_show_tags.isChecked():
            # 1. Apply robust confidence thresholding and connectivity pruning
            # For Basic mode, we don't involve the secondary confidence threshold
            if self.seg_mode_basic.isChecked():
                display_labels = self.seg_labels
            else:
                threshold = self.seg_conf_spin.value()
                display_labels = filter_by_confidence(
                    self.seg_labels, self.seg_confidence, threshold, self.particles
                )
            
            # 2. High-Contrast Deterministic Mapping / 高コントラスト決定論的マッピング
            # Shuffle label indices deterministically so neighbors have distinct colors
            max_lab = int(np.nanmax(self.seg_labels)) if self.seg_labels.size > 0 else 0
            if max_lab > 0:
                # Use a fixed seed for color stability across refreshes
                rng = np.random.RandomState(42)
                perm = rng.permutation(max_lab) + 1 # Shifted to 1...N
                lut = np.zeros(max_lab + 1, dtype=np.int32)
                lut[1:] = perm
                # Map the thresholded labels to shuffled colors
                display_labels = lut[display_labels.astype(np.int32)]

            # 3. Mask zero (background) to make it transparent
            label_overlay = np.where(display_labels == 0, np.nan, display_labels)
            ax.imshow(label_overlay, origin='lower', extent=[0, gv.XScanSize, 0, gv.YScanSize], 
                      cmap='prism', alpha=0.4, interpolation='nearest', zorder=2)
            
            # 4. Draw Fitted Ellipse Axes if available / 楕円フィッティング結果があれば軸を表示
            if hasattr(self, 'fitting_results') and self.fitting_results:
                for pid, fr in self.fitting_results.items():
                    a, b, ang = fr['a'], fr['b'], fr['angle']
                    x0, y0 = fr.get('x0'), fr.get('y0')
                    
                    if a <= 0 or b <= 0 or x0 is None: continue
                    
                    # Convert angle back to phi (standard polar)
                    # ang is now already vs X-axis
                    phi = np.radians(ang)
                    cos_p, sin_p = np.cos(phi), np.sin(phi)
                    
                    # Axis A is now always the major axis from tracker_port
                    # a-axis (phi direction): Solid Red
                    ax.plot([x0 - a*cos_p, x0 + a*cos_p], [y0 - a*sin_p, y0 + a*sin_p], 
                            'r-', linewidth=1.5, alpha=0.9, zorder=12)
                    
                    # b-axis (phi + 90deg direction): Dotted Red
                    ax.plot([x0 + b*sin_p, x0 - b*sin_p], [y0 - b*cos_p, y0 + b*cos_p], 
                            'r:', linewidth=1.5, alpha=0.9, zorder=12)

        ax.figure.tight_layout(pad=0.1)
        self.map_canvases['Raw'].draw()

    def format_raw_coord(self, x, y):
        """Custom formatter for the AFM View status bar. / AFMビューのステータスバー用カスタムデコーダー"""
        if not hasattr(gv, 'aryData') or gv.aryData is None:
            return f"X={x:.1f}, Y={y:.1f}"
        
        h_ary, w_ary = gv.aryData.shape[:2]
        px_x = gv.XScanSize / w_ary
        px_y = gv.YScanSize / h_ary
        
        col = int(x / px_x)
        # Row 0 is bottom in our Cartesian mapping
        row = int(y / px_y)
        
        if 0 <= col < w_ary and 0 <= row < h_ary:
            z = gv.aryData[row, col]
            return f"X={x:.1f}nm, Y={y:.1f}nm, Z={z:.2f}nm"
        else:
            return f"X={x:.1f}nm, Y={y:.1f}nm"

    def on_zoom_mouse_move(self, event):
        """Updates the coordinate overlay on the zoom canvas. / ズームキャンバス上の座標オーバーレイを更新"""
        if event.inaxes != self.zoom_ax:
            self.zoom_coord_lbl.hide()
            return
        z_text = self.format_raw_coord(event.xdata, event.ydata)
        self.zoom_coord_lbl.setText(z_text)
        self.zoom_coord_lbl.adjustSize()
        # Position at bottom-right
        self.zoom_coord_lbl.move(self.zoom_canvas.width() - self.zoom_coord_lbl.width() - 5, 
                                 self.zoom_canvas.height() - self.zoom_coord_lbl.height() - 5)
        self.zoom_coord_lbl.show()

    def on_table_selection_changed(self):
        """Syncs the particle spinbox when a table row is selected. / テーブルの行選択時にパーティクルスピンボックスを同期"""
        selected = self.results_table.currentRow()
        if 0 <= selected < len(self.particles):
            if self.p_spin.value() != selected:
                self.p_spin.setValue(selected)

    def on_particle_click(self, event):
        """
        Handles mouse click events on the Matplotlib canvas.

        Functionality:
        1. Identifies if the click occurred near a valid particle.
        2. Updates `self.p_spin` (Particle ID) if a target is found.
        3. Calculates and visualizes the "Local Structural Environment":
           - Searches for candidates within the elliptical range (V1, V2).
           - Filters for the nearest 6 neighbors (Euclidean distance).
           - Updates the display to highlight these relationships.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The event object containing click coordinates
                (xdata, ydata).
        """
        if event.inaxes != self.map_axes['Raw']: return
        if not hasattr(gv, 'aryData') or gv.aryData is None: return
        
        # Get parameters early for thresholding
        try:
            # Robust parsing
            text_v1 = self.res_v1.text().replace('nm', '').strip()
            v1_nm = float(text_v1 if text_v1 else 700.0)
            v2 = float(self.res_v2.text() or v1_nm)
        except:
            v1_nm, v2 = 700, 700

        # Find nearest particle in nm coords (using raw coordinates for stability)
        h_ary, w_ary = gv.aryData.shape[:2]
        px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
        
        idx = -1
        dists = []
        if len(self.particles) > 0:
            coords_nm = np.column_stack((self.particles[:, 1] * px_x, self.particles[:, 0] * px_y))
            dists = np.sqrt((coords_nm[:, 0] - event.xdata)**2 + (coords_nm[:, 1] - event.ydata)**2)
            idx = np.argmin(dists)
        
        if self.manual_edit_btn.isChecked():
            # Manual Edit: Add or Remove
            dist_threshold = 0.05 * v1
            if len(self.particles) > 0 and idx != -1 and dists[idx] < dist_threshold:
                # Remove nearest
                self.particles = np.delete(self.particles, idx, axis=0)
            else:
                # Add new
                new_p = np.array([[event.ydata / px_y, event.xdata / px_x]])
                if len(self.particles) == 0:
                    self.particles = new_p
                else:
                    self.particles = np.vstack([self.particles, new_p])
            
            self.p_spin.setRange(0, max(0, len(self.particles) - 1))
            self.update_results()
            self.update_particle_display()
            self.update_particle_zoom()
            return

        if len(self.particles) == 0: return

        if dists[idx] > (0.1 * v1): # Far click (10% of V1)
            self.update_particle_display()
            return

        self.p_spin.setValue(idx) # This will trigger update_particle_zoom via signal
        
        # Calculate Neighbors (logic from calculate_finger_tensor)
        try:
            if v1 <= 0: 
                self.update_particle_display(highlight_idx=idx)
                return

            h_ary, w_ary = gv.aryData.shape[:2]
            px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
            coords_nm = np.column_stack((self.particles[:, 1] * px_x, self.particles[:, 0] * px_y))
            
            # Neighbors within elliptical cutoff
            diff = coords_nm - coords_nm[idx]
            ellip_dist_sq = (diff[:, 0] / v1)**2 + (diff[:, 1] / v2)**2
            candidate_indices = np.where((ellip_dist_sq <= 1.0) & (np.arange(len(coords_nm)) != idx))[0]
            
            # Top 6 by actual Euclidean distance
            if len(candidate_indices) > 0:
                actual_dists = np.linalg.norm(diff[candidate_indices], axis=1)
                neighbors = candidate_indices[np.argsort(actual_dists)[:6]]
            else:
                neighbors = []

            # Display with highlights (ellipse parameters in nm)
            ellipse_params = (self.particles[idx, 1], self.particles[idx, 0], v1, v2)
            self.update_particle_display(highlight_idx=idx, neighbors=neighbors, ellipse_params=ellipse_params)
            
        except Exception as e:
            print(f"Verification Error: {e}")
            self.update_particle_display(highlight_idx=idx)

    def update_particle_zoom(self):
        """
        Updates the zoomed-in view of the currently selected particle.

        This includes displaying the AFM height data in the ROI and overlaying
        the extracted height contour based on the Z-range sliders.
        """
        """
        Updates the local zoom plot and performs Z-axis slice analysis.

        This is the core of the P6 functionality. It performs the following:
        1. Crops a local Region of Interest (ROI) based on the selected particle ID.
        2. Executes "Cloud Slicing" within that ROI using `z_high` and `z_low` parameters.
        3. Converts the sliced points from the crop's local coordinates to physical coordinates
           and renders them as green dots.

        Side Effects:
            Redraws `self.zoom_ax` (Particle Zoom).
            Prints debug information to the console regarding the number of points in the slice.
        """
        if len(self.particles) == 0 or not hasattr(gv, 'aryData') or gv.aryData is None: return
        idx = self.p_spin.value()
        if idx >= len(self.particles): return
        
        # Sync table selection (highlight the row)
        if self.results_table.currentRow() != idx:
            self.results_table.selectRow(idx)
        
        y, x = self.particles[idx]
        v1 = float(self.res_v1.text() or 700)
        
        h_ary, w_ary = gv.aryData.shape[:2]
        px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
        coords_nm = np.column_stack((self.particles[idx:idx+1, 1] * px_x, self.particles[idx:idx+1, 0] * px_y))
        x_nm, y_nm = coords_nm[0]
        
        self.zoom_ax.clear()

        # Display raw topography pixels (no upsampling)
        try:
            # Use unresized 8-bit contrast data if available
            if hasattr(gv, 'cvimg') and gv.cvimg is not None:
                # Apply gamma and color map from global settings
                raw_8bit = gv.cvimg
                gamma_lut = getattr(gv, 'gamma_lut', None)
                if gamma_lut is not None:
                    raw_8bit = cv2.LUT(raw_8bit, gamma_lut)
                
                color_lut = getattr(gv, 'color_lut', None)
                if color_lut is not None:
                    raw_color = cv2.applyColorMap(raw_8bit, color_lut)
                else:
                    raw_color = cv2.applyColorMap(raw_8bit, cv2.COLORMAP_VIRIDIS)
                
                img_rgb = cv2.cvtColor(raw_color, cv2.COLOR_BGR2RGB)
                # Show raw orientation with origin='lower' (index 0 at bottom)
                self.zoom_ax.imshow(img_rgb, origin='lower', extent=[0, gv.XScanSize, 0, gv.YScanSize], interpolation='nearest')
            else:
                # Fallback to dspimg (resized) but use nearest to show pixels
                img_rgb = cv2.cvtColor(gv.dspimg, cv2.COLOR_BGR2RGB)
                img_rgb_raw = np.flipud(img_rgb) # Flip back to raw orientation
                self.zoom_ax.imshow(img_rgb_raw, origin='lower', extent=[0, gv.XScanSize, 0, gv.YScanSize], interpolation='nearest')
        except Exception as e:
            print(f"Zoom Background Error: {e}")
            # Minimal fallback
            self.zoom_ax.imshow(np.flipud(cv2.cvtColor(gv.dspimg, cv2.COLOR_BGR2RGB)), origin='lower', extent=[0, gv.XScanSize, 0, gv.YScanSize])
        
        # Set limits for zoom in nm
        self.zoom_ax.set_xlim(x_nm - 0.5 * v1, x_nm + 0.5 * v1)
        self.zoom_ax.set_ylim(y_nm - 0.5 * v1, y_nm + 0.5 * v1)
        
        # Draw marker in nm
        self.zoom_ax.scatter(x_nm, y_nm, s=100, c='none', edgecolors='yellow', linewidths=2)
        
        # Overlay fitted axes if available for the specific particle
        pid = idx + 1
        if hasattr(self, 'fitting_results') and pid in self.fitting_results:
            fr = self.fitting_results[pid]
            fa, fb, fang = fr['a'], fr['b'], fr['angle']
            fx0, fy0 = fr.get('x0'), fr.get('y0')
            
            if fa > 0 and fb > 0 and fx0 is not None:
                phi = np.radians(fang)
                cp, sp = np.cos(phi), np.sin(phi)
                
                # fa (Axis A) is always the major axis
                # a-axis: solid red
                self.zoom_ax.plot([fx0 - fa*cp, fx0 + fa*cp], [fy0 - fa*sp, fy0 + fa*sp], 
                                  'r-', linewidth=2, alpha=0.9, zorder=15)
                # b-axis: dotted red
                self.zoom_ax.plot([fx0 + fb*sp, fx0 - fb*sp], [fy0 - fb*cp, fy0 + fb*cp], 
                                  'r:', linewidth=2, alpha=0.9, zorder=15)

        self.zoom_ax.set_title(f"Particle #{idx}")
        self.zoom_ax.set_xlabel("X (nm)"); self.zoom_ax.set_ylabel("Y (nm)")
        
        # Internal coordinate formatting logic remains for consistency
        self.zoom_ax.format_coord = self.format_raw_coord

        # Cloud Slicing and Visualization (Peak-relative donut)
        if hasattr(gv, 'aryData') and gv.aryData is not None:
            try:
                # Raw data dimensions and scale
                h_ary, w_ary = gv.aryData.shape[:2]
                px_x_ary, px_y_ary = gv.XScanSize / w_ary, gv.YScanSize / h_ary
                
                # Use same ROI as zoom window (in nm)
                zoom_range = 0.5 * v1
                ymin, ymax = y_nm - zoom_range, y_nm + zoom_range
                xmin, xmax = x_nm - zoom_range, x_nm + zoom_range
                
                # Map NM to aryData indices (Row 0 = Bottom)
                r0, r1 = int(ymin / px_y_ary), int(ymax / px_y_ary)
                c0, c1 = int(xmin / px_x_ary), int(xmax / px_x_ary)
                
                # Order indices correctly
                r0, r1 = max(0, min(r0, r1)), min(h_ary, max(r0, r1))
                c0, c1 = max(0, min(c0, c1)), min(w_ary, max(c0, c1))
                
                if r1 > r0 and c1 > c0:
                    crop = gv.aryData[r0:r1, c0:c1]
                    z_low, z_high = self.z_range_slider.values()
                    z_tol = self.z_tol_spin.value()
                    
                    # Vertical slice logic: z0 - Z_high +/- tolerance
                    points_px = extract_height_points(crop, z_high, z_res_nm=z_tol)
                    
                    if len(points_px) > 0:
                        # Convert crop row/col -> absolute nm for plotting
                        # y_nm = (row_in_crop + r0 + 0.5) * px_y_ary
                        pts_y_nm = (points_px[:, 0] + r0 + 0.5) * px_y_ary
                        pts_x_nm = (points_px[:, 1] + c0 + 0.5) * px_x_ary
                        self.zoom_ax.scatter(pts_x_nm, pts_y_nm, s=8, c='lime', alpha=0.8, edgecolors='none', zorder=10)
                        
                        print(f"DEBUG: Particle #{idx} | Z-Range [{np.nanmin(crop):.2f}, {np.nanmax(crop):.2f}], Target Depth: {z_high} nm | Plotted: {len(points_px)}")
                    else:
                        print(f"DEBUG: Particle #{idx} | NO POINTS in slice at Depth: {z_high} nm")
            except Exception as e:
                print(f"Slice Visualization Error: {e}")

        self.zoom_fig.tight_layout(pad=0.1)
        self.zoom_canvas.draw()
        
        # Sync main display highlight
        self.update_particle_display(highlight_idx=idx)

    def run_gr_analysis(self):
        """
        Calculates the radial distribution function g(r) for detected particles.

        Updates the internal g(r) data and triggers a plot update.
        """
        if len(self.particles) < 2 or not hasattr(gv, 'aryData') or gv.aryData is None: return
        try:
            h_ary, w_ary = gv.aryData.shape[:2]
            px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
            coords_nm = np.column_stack((self.particles[:, 1] * px_x, self.particles[:, 0] * px_y))
            rs, gr = calculate_gr(coords_nm, (gv.XScanSize, gv.YScanSize), dr=self.gr_dr.value())
            self.gr_data = (rs, gr); self.update_plot()
        except Exception as e: print(f"g(r) Error: {e}")

    def update_plot(self):
        """Updates the plot display based on selected mode. / 選択されたモードに基づいてグラフ表示を更新"""
        if not hasattr(self, 'fig') or self.fig is None: return
        self.ax.clear()
        
        # Determine mode
        if hasattr(self, 'plot_mode_ar') and self.plot_mode_ar.isChecked():
            self._render_ar_hist()
        elif hasattr(self, 'plot_mode_ori') and self.plot_mode_ori.isChecked():
            self._render_ori_hist()
        else:
            self._render_gr()
            
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw()

    def _render_gr(self):
        """Renders the g(r) plot. / g(r)グラフを描画"""
        if self.gr_data is None: 
            self.ax.set_title("g(r) - No Data")
            return
            
        rs, gr = self.gr_data
        self.ax.plot(rs, gr, 'b-', alpha=0.7)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("r (nm)"); self.ax.set_ylabel("g(r)")
        self.ax.set_title("Radial Distribution Function g(r)")

        # Reset cursor elements after clear
        self.gr_cursor_line = self.ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, visible=False)
        self.gr_cursor_dot, = self.ax.plot([], [], 'ko', markersize=4, visible=False, zorder=5)

        for f, c, l in [(self.res_r1, 'ro', 'R1'), (self.res_v1, 'go', 'V1'), (self.res_v2, 'mo', 'V2')]:
            try:
                v_str = f.text().strip()
                if not v_str: continue
                v = float(v_str); idx = np.argmin(np.abs(rs - v))
                self.ax.plot(rs[idx], gr[idx], f'{c}', label=f"{l}({rs[idx]:.1f})", markersize=6)
            except: pass
        if self.ax.get_legend_handles_labels()[0]: self.ax.legend(fontsize=8)

    def _render_ar_hist(self):
        """Renders Aspect Ratio histogram. / アスペクト比のヒストグラムを描画"""
        if not hasattr(self, 'fitting_results') or not self.fitting_results:
            self.ax.set_title("Aspect Ratio - Run P8 First")
            return
            
        ars = np.array([max(r['a'], r['b']) / min(r['a'], r['b']) for r in self.fitting_results.values() if r['a'] > 0 and r['b'] > 0])
        if len(ars) == 0:
            self.ax.set_title("Aspect Ratio - No Valid Fits")
            return
            
        # Robust range estimation (IQR) to gate outliers / 外れ値をゲートするための堅牢な範囲推定(IQR)
        q1, q3 = np.percentile(ars, [25, 75])
        iqr = q3 - q1
        lower = max(1.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        
        # Determine actual range for histogram bins
        h_range = (lower, upper) if upper > lower else (1.0, max(1.5, np.max(ars)))
        
        counts, bins, patches = self.ax.hist(ars, bins=25, range=h_range, color='skyblue', edgecolor='black', alpha=0.7)
        self.ax.set_xlabel("Aspect Ratio (Major/Minor)")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title(f"Ensemble Aspect Ratio (N={len(ars)})")
        self.ax.grid(True, axis='y', alpha=0.3)
        
        # Add a note if data was clipped
        if np.any(ars > upper) or np.any(ars < lower):
            self.ax.text(0.95, 0.95, "Outliers Gated", transform=self.ax.transAxes, 
                         verticalalignment='top', horizontalalignment='right', 
                         fontsize=8, color='red', bbox=dict(facecolor='white', alpha=0.5))

    def _render_ori_hist(self):
        """Renders Orientation histogram. / オリエンテーションのヒストグラムを描画"""
        if not hasattr(self, 'fitting_results') or not self.fitting_results:
            self.ax.set_title("Orientation - Run P8 First")
            return
            
        oris = np.array([r['angle'] for r in self.fitting_results.values() if r['a'] > 0])
        if len(oris) == 0:
            self.ax.set_title("Orientation - No Valid Fits")
            return
            
        # Robust range estimation (IQR) / 堅牢な範囲推定(IQR)
        q1, q3 = np.percentile(oris, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        # Use full angular symmetry range [0, 180] if it covers the data, 
        # otherwise use robust range to focus on peaks.
        if upper - lower > 180 or upper <= lower:
            h_range = (0, 180)
        else:
            h_range = (lower, upper)

        self.ax.hist(oris, bins=25, range=h_range, color='salmon', edgecolor='black', alpha=0.7)
        self.ax.set_xlabel("Angle vs +X-axis (deg)")
        self.ax.set_ylabel("Frequency")
        self.ax.set_title(f"Orientation Distribution (N={len(oris)})")
        self.ax.grid(True, axis='y', alpha=0.3)
        
        if np.any(oris > upper) or np.any(oris < lower):
            self.ax.text(0.95, 0.95, "Outliers Gated", transform=self.ax.transAxes, 
                         verticalalignment='top', horizontalalignment='right', 
                         fontsize=8, color='red', bbox=dict(facecolor='white', alpha=0.5))

    def on_gr_mouse_move(self, event):
        """Handles mouse move on g(r) plot for 'snapping' cursor. / g(r)グラフ上でのマウス移動によるスナップカーソルの制御"""
        if event.inaxes != self.ax or self.gr_data is None or not self.plot_mode_gr.isChecked():
            if hasattr(self, 'gr_cursor_line') and self.gr_cursor_line: self.gr_cursor_line.set_visible(False)
            if hasattr(self, 'gr_cursor_dot') and self.gr_cursor_dot: self.gr_cursor_dot.set_visible(False)
            self.canvas.draw_idle()
            return
            
        rs, gr = self.gr_data
        idx = np.argmin(np.abs(rs - event.xdata))
        snapped_r = rs[idx]
        snapped_gr = gr[idx]
        
        self.gr_cursor_line.set_xdata([snapped_r, snapped_r])
        self.gr_cursor_line.set_visible(True)
        self.gr_cursor_dot.set_data([snapped_r], [snapped_gr])
        self.gr_cursor_dot.set_visible(True)
        
        self.canvas.draw_idle()

    def on_plot_click(self, event):
        """Handles mouse clicks on the plot modes. / グラフ上のクリックによる処理"""
        if event.inaxes != self.ax or not self.plot_mode_gr.isChecked(): return
        if self.gr_data is None or self.pick_none.isChecked(): return
        rs, _ = self.gr_data; val = rs[np.argmin(np.abs(rs - event.xdata))]
        if self.pick_r1.isChecked(): self.res_r1.setText(f"{val:.2f}")
        elif self.pick_v1.isChecked(): self.res_v1.setText(f"{val:.2f}")
        elif self.pick_v2.isChecked(): self.res_v2.setText(f"{val:.2f}")
        self.update_plot()

    def run_structural_analysis(self):
        """
        Performs bond-orientational order and Finger Tensor analysis.

        Calculates coordination numbers, Psi parameters, and anisotropy
        metrics. Generates 2D maps for each parameter for visualization.
        """
        """
        Executes advanced structural and tensor analysis (P3-P5).

        This function serves as the data convergence point, converting geometric coordinates
        into physical order parameters.

        Workflow:
        1. Prepares data: Converts `self.particles` to nanometer coordinates.
        2. Calls `analyze_components` to compute Z, Psi6, and Psi4.
        3. Calls `calculate_finger_tensor` to compute structural tensors (Lambda, Aspect).
        4. Generates visualization heatmaps and stores them in `self.struct_maps`.
        5. Updates the tabbed interface on the right panel.

        Requires:
            self.res_v1 (float): The first neighbor shell cutoff distance, typically derived
            from g(r) analysis.
        """
        if not self.res_v1.text(): self.run_gr_analysis()
        try:
            cutoff_v1 = float(self.res_v1.text() or 0)
            cutoff_v2 = float(self.res_v2.text() or cutoff_v1)
            r1 = float(self.res_r1.text() or 1.0)
            
            h_ary, w_ary = gv.aryData.shape[:2]
            px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
            # Map raw data indices (Row 0 = Bottom) -> NM (No Flip)
            coords_nm = np.column_stack((self.particles[:, 1] * px_x, self.particles[:, 0] * px_y))
            
            # P3: Components
            self.struct_results = analyze_components(coords_nm, cutoff_v1)
            
            # P5: Finger Tensor
            tensor_results = calculate_finger_tensor(coords_nm, r1, cutoff_v1, cutoff_v2)
            self.struct_results.update(tensor_results)
            
            # P4: Mapping
            res_px = (w_ary * 2, h_ary * 2) # Reduced multiplier for high-res raw data
            map_radius = 0.33 * r1
            self.struct_maps = {k: generate_structural_map(coords_nm, self.struct_results[k], (gv.XScanSize, gv.YScanSize), res_px, circle_radius_nm=map_radius) for k in self.struct_results.keys() if k != 'orientation'}
            
            self.update_structural_display()
        except Exception as e: print(f"Struct Error: {e}")

    def update_structural_display(self):
        """Updates all structural mapping canvases with the latest results. / 全ての構造マップキャンバスを最新の結果で更新"""
        if not self.struct_maps or cm is None or mcolors is None: return
        
        mapping = {
            'Z': ('Z', 'Coordination', 'viridis'), 'Psi6': ('Psi6', 'Hexagonal', 'magma'), 
            'Psi4': ('Psi4', 'Rectangular', 'plasma'), 'Psi2': ('Psi2', 'Linear', 'inferno'),
            'lambda1': ('lambda1', 'Lambda 1', 'viridis'), 'lambda2': ('lambda2', 'Lambda 2', 'viridis'),
            'aspect': ('aspect', 'Anisotropy', 'magma'),
            'fit_aspect': ('fit_aspect', 'Fit Aspect', 'magma'),
            'fit_ori': ('fit_ori', 'Orientation (vs X)', 'hsv')
        }
        for key, (_, title, cmap_name) in mapping.items():
            if key in self.map_axes and key in self.struct_maps:
                ax = self.map_axes[key]; data = self.struct_maps[key]; ax.clear()
                print(f"Updating display for {key}. Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
                
                kwargs = {'origin': 'lower', 'extent': [0, gv.XScanSize, 0, gv.YScanSize]}
                if key == 'Z':
                    # Create discrete colorbar for coordination number
                    z_min = int(np.nanmin(data)) if not np.all(np.isnan(data)) else 0
                    z_max = int(np.nanmax(data)) if not np.all(np.isnan(data)) else 10
                    if z_max <= z_min: z_max = z_min + 1
                    n_steps = z_max - z_min + 1
                    cmap = cm.get_cmap(cmap_name, n_steps)
                    norm = mcolors.BoundaryNorm(np.arange(z_min, z_max + 2) - 0.5, n_steps)
                    kwargs.update({'cmap': cmap, 'norm': norm})
                elif key == 'fit_aspect':
                    # Apply robust IQR scaling to gate outliers (mimicking histogram logic)
                    valid_data = data[np.isfinite(data) & (data > 0)]
                    if len(valid_data) > 0:
                        q1, q3 = np.percentile(valid_data, [25, 75])
                        iqr = q3 - q1
                        lower = max(1.0, q1 - 1.5 * iqr)
                        upper = q3 + 1.5 * iqr
                        if upper > lower:
                            kwargs.update({'vmin': lower, 'vmax': upper, 'cmap': cmap_name})
                        else:
                            kwargs['cmap'] = cmap_name
                    else:
                        kwargs['cmap'] = cmap_name
                elif key in ['Psi6', 'Psi4', 'Psi2']:
                    # Constrain Psi structural parameters to [0, 1] as per definition
                    kwargs.update({'vmin': 0.0, 'vmax': 1.0, 'cmap': cmap_name})
                else:
                    kwargs['cmap'] = cmap_name

                im = ax.imshow(data, **kwargs)
                
                if key not in self.map_colorbars:
                    # Use standard fraction=0.046, pad=0.04 to match image axes exactly (Magic numbers for locking geometry)
                    self.map_colorbars[key] = ax.figure.colorbar(im, ax=ax, label=title, fraction=0.046, pad=0.04)
                else:
                    self.map_colorbars[key].update_normal(im)
                
                if key == 'Z':
                    # Set ticks to integers
                    self.map_colorbars[key].set_ticks(np.arange(z_min, z_max + 1))

                ax.set_title(title)
                ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
                ax.set_aspect('equal')
                ax.figure.tight_layout(pad=0.1); self.map_canvases[key].draw()

    def delete_edges(self):
        """Removes particles located near the image edges. / 画像端付近のパーティクルを削除"""
        if len(self.particles) > 0 and hasattr(gv, 'aryData'):
            # Filter particles to keep only 10%~90% range in both X and Y
            self.particles = delete_edge_particles(self.particles, gv.aryData.shape[:2])
            self.update_results()
            self.update_particle_display()
            self.update_particle_zoom()


    def run_segmentation(self):
        """Executes the particle segmentation pipeline in a background thread."""
        if self.particles is None or len(self.particles) == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No particles detected to segment.")
            return
        if not hasattr(gv, 'aryData') or gv.aryData is None: return

        # Dispatch parameters based on mode
        mode = 'basic' if self.seg_mode_basic.isChecked() else 'advanced'
        alpha = self.seg_alpha_spin.value()
        beta = self.seg_beta_spin.value()
        sigma = self.seg_sigma_spin.value()
        denoise_on = self.seg_denoise_chk.isChecked()
        threshold = self.seg_thresh_spin.value()
        v1_nm = float(self.res_v1.text() or 700.0)
        
        self.segment_btn.setEnabled(False); self.segment_btn.setText("Segmenting...")
        
        self.seg_worker = SegmentationWorker(
            mode, gv.aryData, self.particles, v1_nm, (gv.XScanSize, gv.YScanSize),
            alpha, beta, sigma, denoise_on, threshold
        )
        self.seg_worker.finished.connect(self.on_segmentation_finished)
        self.seg_worker.error.connect(self.on_segmentation_error)
        self.seg_worker.start()

    def on_segmentation_finished(self, labels, confidence, curv):
        """Handles segmentation results and updates visualization. / セグメンテーション結果の処理と表示更新"""
        self.seg_labels = labels
        self.seg_confidence = confidence
        self.curv_map = curv
        self.segment_btn.setText("Segment Clustered Particles")
        self.segment_btn.setEnabled(True)
        self.update_particle_display()
        
        # Update Curvature Tab
        if 'Curvature' in self.map_axes:
            ax = self.map_axes['Curvature']
            ax.clear()
            im = ax.imshow(curv, origin='lower', extent=[0, gv.XScanSize, 0, gv.YScanSize], cmap='RdBu_r')
            
            # DIAGNOSTIC OVERLAY for Basic Mode
            if self.seg_mode_basic.isChecked():
                thresh = self.seg_thresh_spin.value()
                # confidence contains the normalized scores [0, 1]
                gating_mask = (confidence >= thresh)
                
                # Create a translucent mask (Emerald Green for gating)
                overlay = np.zeros((*gating_mask.shape, 4))
                overlay[gating_mask] = [0.0, 1.0, 0.4, 0.4] # RGBA: Emerald Green, Alpha 0.4
                
                ax.imshow(overlay, origin='lower', extent=[0, gv.XScanSize, 0, gv.YScanSize], interpolation='nearest')
                ax.set_title(f"Hessian Map + Gating Overlay (Thresh: {thresh})")
            else:
                ax.set_title("Hessian Curvature Map (Preprocessing)")

            if 'Curvature' not in self.map_colorbars:
                self.map_colorbars['Curvature'] = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                self.map_colorbars['Curvature'].update_normal(im)
            
            ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
            ax.set_aspect('equal')
            ax.figure.tight_layout(pad=0.1)
            self.map_canvases['Curvature'].draw()
            
        # Trigger redraw of Raw map with boundaries to show new segmentation
        self.update_particle_display()
        print(f"DEBUG: Segmentation Finished. Labels uniquely identified: {len(np.unique(labels))-1}")

    def run_ellipse_fitting(self):
        """Triggers the 2D ellipse fitting worker."""
        if not hasattr(gv, 'aryData') or gv.aryData is None: return
        if self.seg_labels is None:
            QtWidgets.QMessageBox.warning(self, "Fitting Error", "Please run segmentation (P7) first.")
            return
            
        self.fit_ellipse_btn.setText("Fitting Ellipses...")
        self.fit_ellipse_btn.setEnabled(False)
        
        # Get parameters from P2 for fitting
        try:
            r1 = float(self.res_r1.text() or 10.0)
            v1 = float(self.res_v1.text() or 20.0)
            v2 = float(self.res_v2.text() or 30.0)
        except ValueError:
            r1, v1, v2 = 10.0, 20.0, 30.0
            
        h, w = gv.aryData.shape[:2]
        px_size = gv.XScanSize / w
        
        self._fit_worker = EllipseFittingWorker(self.seg_labels, px_size, r1, v1, v2)
        self._fit_worker.finished.connect(self.on_ellipse_fitting_finished)
        self._fit_worker.error.connect(self.on_segmentation_error) # Reuse error handler
        self._fit_worker.start()

    def on_ellipse_fitting_finished(self, results):
        """Updates internal results and refreshes the table, plot, and structural maps."""
        self.fitting_results = {r['id']: r for r in results}
        self.fit_ellipse_btn.setText("Identify 2D Ellipses")
        self.fit_ellipse_btn.setEnabled(True)
        self.update_results()
        self.update_plot() # Refresh histograms if active
        
        # Generate spatial heatmaps for fitting results / フィッティング結果の空間ヒートマップを生成
        if hasattr(gv, 'aryData') and gv.aryData is not None:
            try:
                h_ary, w_ary = gv.aryData.shape[:2]
                res_px = (w_ary * 2, h_ary * 2)
                r1 = float(self.res_r1.text() or 10.0)
                map_radius = 0.33 * r1
                
                # Filter successful fits and extract coordinates/values
                valid_results = [r for r in results if r['a'] > 0 and r['b'] > 0 and r['x0'] is not None]
                if valid_results:
                    fit_coords = np.array([[r['x0'], r['y0']] for r in valid_results])
                    ar_vals = np.array([max(r['a'], r['b']) / min(r['a'], r['b']) for r in valid_results])
                    ori_vals = np.array([r['angle'] for r in valid_results])
                    
                    self.struct_maps['fit_aspect'] = generate_structural_map(
                        fit_coords, ar_vals, (gv.XScanSize, gv.YScanSize), res_px, map_radius
                    )
                    self.struct_maps['fit_ori'] = generate_structural_map(
                        fit_coords, ori_vals, (gv.XScanSize, gv.YScanSize), res_px, map_radius
                    )
                    self.update_structural_display()
            except Exception as e:
                print(f"Fitting Heatmap Error: {e}")
                traceback.print_exc()

        QtWidgets.QMessageBox.information(self, "Success", f"Fitted ellipses for {len(results)} particles.")

    def export_results(self):
        """
        Exports the current analysis state to a DIC-compatible CSV.
        / 現在の解析状態をDIC互換のCSVにエクスポートします。
        """
        if self.particles is None or len(self.particles) == 0:
            QtWidgets.QMessageBox.warning(self, "Export Error", "No particles detected to export.")
            return

        # Prepare default filename: [FileName]_frame[N]_analysis.csv
        # gv.saveName is often empty; gv.files[gv.currentFileNum] is the active source.
        active_path = ""
        if hasattr(gv, 'files') and gv.files and 0 <= gv.currentFileNum < len(gv.files):
            active_path = gv.files[gv.currentFileNum]
        elif hasattr(gv, 'saveName') and gv.saveName:
            active_path = gv.saveName
        
        base_name = os.path.splitext(os.path.basename(active_path))[0] if active_path else "particle_cluster_analysis"
        
        # Frame index: use gv.index (internal frame pointer)
        # This is 0 for single-frame files and the current frame for movies.
        frame_idx = gv.index if hasattr(gv, 'index') else 0
            
        default_name = f"{base_name}_frame{frame_idx}_analysis.csv"
        
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Analysis Results", default_name, "CSV Files (*.csv)"
        )
        if not save_path: return

        try:
            import datetime
            h_px, w_px = gv.aryData.shape[:2]
            px_x, px_y = gv.XScanSize / w_px, gv.YScanSize / h_px
            
            with open(save_path, 'w', encoding='utf-8') as f:
                # 1. Metadata Header
                f.write(f"# pyNuD Analysis Export v1.0\n")
                f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Source: {active_path if active_path else 'Unknown'}\n")
                f.write(f"# Frame Index: {frame_idx}\n")
                f.write(f"# Scan Size: {gv.XScanSize:.2f} nm x {gv.YScanSize:.2f} nm\n")
                f.write(f"# Pixels: {w_px} x {h_px}\n")
                f.write(f"# Peak R1: {self.res_r1.text()} nm\n")
                f.write(f"# Cutoff V1: {self.res_v1.text()} nm\n")
                f.write(f"# Cutoff V2: {self.res_v2.text()} nm\n")
                
                # 1b. Segmentation Settings
                seg_mode = "Advanced" if self.seg_mode_advanced.isChecked() else "Basic"
                f.write(f"# Seg Mode: {seg_mode}\n")
                f.write(f"# Seg Thresh: {self.seg_thresh_spin.value()}\n")
                f.write(f"# Seg Alpha: {self.seg_alpha_spin.value()}\n")
                f.write(f"# Seg Beta: {self.seg_beta_spin.value()}\n")
                f.write(f"# Seg Sigma: {self.seg_sigma_spin.value()}\n")
                f.write(f"# Seg Denoise: {self.seg_denoise_chk.isChecked()}\n")
                
                # 2. Column Headers
                headers = [
                    "frame_index", "local_id", "center_x_nm", "center_y_nm",
                    "coord_num", "psi6", "psi4", "psi2",
                    "lambda_1", "lambda_2", "struct_anisotropy",
                    "fit_major_nm", "fit_minor_nm", "fit_aspect_ratio", "fit_angle_deg", "fit_rmse",
                    "seg_confidence"
                ]
                f.write(",".join(headers) + "\n")
                
                # 3. Data Rows
                for i in range(len(self.particles)):
                    # Basic geometry (NM)
                    row_px, col_px = self.particles[i]
                    # Map raw data indices (Row 0 = Bottom) -> NM (No Flip)
                    cx_nm = col_px * px_x
                    cy_nm = row_px * px_y
                    
                    # Structural results (P3-P5) - Use safe get with empty dict fallback
                    s_res = self.struct_results if self.struct_results else {}
                    z = s_res.get('Z', [np.nan]*len(self.particles))[i]
                    psi6 = s_res.get('Psi6', [np.nan]*len(self.particles))[i]
                    psi4 = s_res.get('Psi4', [np.nan]*len(self.particles))[i]
                    psi2 = s_res.get('Psi2', [np.nan]*len(self.particles))[i]
                    l1 = s_res.get('lambda1', [np.nan]*len(self.particles))[i]
                    l2 = s_res.get('lambda2', [np.nan]*len(self.particles))[i]
                    s_aspect = s_res.get('aspect', [np.nan]*len(self.particles))[i]
                    
                    # Fitting results (P8)
                    fit = self.fitting_results.get(i, {})
                    f_major = max(fit.get('a', 0), fit.get('b', 0)) if fit else np.nan
                    f_minor = min(fit.get('a', 0), fit.get('b', 0)) if fit else np.nan
                    f_aspect = f_major / f_minor if f_minor > 0 else np.nan
                    f_angle = fit.get('angle', np.nan)
                    f_rmse = fit.get('rmse', np.nan)
                    
                    # Segmentation confidence (P7)
                    conf = np.nan
                    if self.seg_confidence is not None:
                        ry, rx = int(np.clip(row_px, 0, h_px-1)), int(np.clip(col_px, 0, w_px-1))
                        conf = self.seg_confidence[ry, rx]
                    
                    row_data = [
                        frame_idx, i, f"{cx_nm:.3f}", f"{cy_nm:.3f}",
                        f"{z:.1f}", f"{psi6:.4f}", f"{psi4:.4f}", f"{psi2:.4f}",
                        f"{l1:.4f}", f"{l2:.4f}", f"{s_aspect:.4f}",
                        f"{f_major:.3f}", f"{f_minor:.3f}", f"{f_aspect:.4f}", f"{f_angle:.2f}", f"{f_rmse:.4f}",
                        f"{conf:.4f}"
                    ]
                    f.write(",".join(map(str, row_data)) + "\n")
            
            QtWidgets.QMessageBox.information(self, "Success", f"Exported {len(self.particles)} particles to {os.path.basename(save_path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export results: {e}")
            traceback.print_exc()

    def load_results(self):
        """
        Loads analysis results from a DIC-compatible CSV.
        / DIC互換のCSVから解析結果を読み込みます。
        """
        load_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Analysis Results", "", "CSV Files (*.csv)"
        )
        if not load_path: return

        try:
            import datetime
            import csv
            
            # 1. Parse Metadata & Check Compatibility
            metadata = {}
            data_rows = []
            with open(load_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    if row[0].startswith("# "):
                        key_val = row[0][2:].split(": ", 1)
                        if len(key_val) == 2:
                            metadata[key_val[0]] = key_val[1]
                    elif row[0] == "frame_index":
                        headers = row
                    else:
                        data_rows.append(row)

            if not data_rows:
                QtWidgets.QMessageBox.warning(self, "Load Error", "No data found in the selected CSV.")
                return

            # Compatibility & Metadata Check
            source_file = metadata.get("Source", "Unknown")
            csv_frame = int(metadata.get("Frame Index", -1))
            
            # Restoration of Segmentation Settings (Metadata)
            try:
                if "Seg Mode" in metadata:
                    is_adv = metadata["Seg Mode"] == "Advanced"
                    self.seg_mode_advanced.setChecked(is_adv)
                    self.seg_mode_basic.setChecked(not is_adv)
                if "Seg Thresh" in metadata:
                    self.seg_thresh_spin.setValue(float(metadata["Seg Thresh"]))
                if "Seg Alpha" in metadata:
                    self.seg_alpha_spin.setValue(float(metadata["Seg Alpha"]))
                if "Seg Beta" in metadata:
                    self.seg_beta_spin.setValue(float(metadata["Seg Beta"]))
                if "Seg Sigma" in metadata:
                    self.seg_sigma_spin.setValue(float(metadata["Seg Sigma"]))
                if "Seg Denoise" in metadata:
                    # Parse "True"/"False" string
                    self.seg_denoise_chk.setChecked(metadata["Seg Denoise"] == "True")
                
                # Restoration of Analysis Parameters (R1, V1, V2)
                if "Peak R1" in metadata:
                    self.res_r1.setText(metadata["Peak R1"].replace('nm', '').strip())
                if "Cutoff V1" in metadata:
                    self.res_v1.setText(metadata["Cutoff V1"].replace('nm', '').strip())
                if "Cutoff V2" in metadata:
                    self.res_v2.setText(metadata["Cutoff V2"].replace('nm', '').strip())
            except Exception as e:
                print(f"Warning: Failed to restore some metadata settings: {e}")

            # Check if active file matches
            # gv.files[gv.currentFileNum] is the canonical source
            curr_source = ""
            if hasattr(gv, 'files') and gv.files and 0 <= gv.currentFileNum < len(gv.files):
                curr_source = gv.files[gv.currentFileNum]
            
            warn_msg = ""
            if source_file != curr_source:
                warn_msg += f"- Source mismatch: CSV is from '{os.path.basename(source_file)}', current is '{os.path.basename(curr_source)}'.\n"
            if csv_frame != gv.index:
                warn_msg += f"- Frame mismatch: CSV is frame {csv_frame}, current is frame {gv.index}.\n"
            
            if warn_msg:
                reply = QtWidgets.QMessageBox.question(
                    self, "Compatibility Warning",
                    f"The selected analysis results may not match the active image:\n\n{warn_msg}\nDo you want to proceed anyway?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.No: return

            # 2. State Restoration
            h_px, w_px = gv.aryData.shape[:2]
            px_x, px_y = gv.XScanSize / w_px, gv.YScanSize / h_px
            
            self.particles = []
            # Reset result containers
            self.struct_results = {'Z': [], 'Psi6': [], 'Psi4': [], 'Psi2': [], 'lambda1': [], 'lambda2': [], 'aspect': []}
            self.fitting_results = {}
            self.seg_confidence = np.zeros((h_px, w_px), dtype=np.float32)
            self.seg_labels = np.zeros((h_px, w_px), dtype=np.int32)

            for i, row in enumerate(data_rows):
                d = dict(zip(headers, row))
                
                # Restore Coordinates (Pixels)
                cx_nm = float(d['center_x_nm'])
                cy_nm = float(d['center_y_nm'])
                row_px = cy_nm / px_y
                col_px = cx_nm / px_x
                self.particles.append([row_px, col_px])
                
                # Restore Structural Results
                self.struct_results['Z'].append(float(d['coord_num']))
                self.struct_results['Psi6'].append(float(d['psi6']))
                self.struct_results['Psi4'].append(float(d['psi4']))
                self.struct_results['Psi2'].append(float(d['psi2']))
                self.struct_results['lambda1'].append(float(d['lambda_1']))
                self.struct_results['lambda2'].append(float(d['lambda_2']))
                self.struct_results['aspect'].append(float(d['struct_anisotropy']))
                
                # Restore Fitting Results
                f_major = float(d['fit_major_nm'])
                f_minor = float(d['fit_minor_nm'])
                if not np.isnan(f_major):
                    self.fitting_results[i] = {
                        'id': i, 'x0': cx_nm, 'y0': cy_nm,
                        'a': f_major, 'b': f_minor,
                        'angle': float(d['fit_angle_deg']),
                        'rmse': float(d['fit_rmse'])
                    }
                
                # Restore Confidence Map at particle peaks
                conf = float(d['seg_confidence'])
                if not np.isnan(conf):
                    ry, rx = int(np.clip(row_px, 0, h_px-1)), int(np.clip(col_px, 0, w_px-1))
                    self.seg_confidence[ry, rx] = conf
                
                # We can't fully restore the pixel-perfect labels from a point CSV, 
                # but we can set the seed pixels.
                ry, rx = int(np.clip(row_px, 0, h_px-1)), int(np.clip(col_px, 0, w_px-1))
                self.seg_labels[ry, rx] = i + 1

            # Convert to numpy for consistent indexing/slicing in display functions
            self.particles = np.array(self.particles)
            for k in self.struct_results:
                self.struct_results[k] = np.array(self.struct_results[k])

            # 3. Visualization Update
            self._regenerate_heatmaps_after_load()
            self.update_results()
            self.update_particle_display()
            self.update_structural_display()
            
            # 4. Auto-trigger Segmentation for Hessian reconstruction
            QtCore.QTimer.singleShot(100, self.run_segmentation)
            
            QtWidgets.QMessageBox.information(self, "Success", f"Successfully loaded {len(self.particles)} particles from {os.path.basename(load_path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load results: {e}")
            traceback.print_exc()

    def _regenerate_heatmaps_after_load(self):
        """Reconstructs 2D structural maps from loaded particle-level data. / 読み込まれたデータから2D構造マップを再構築"""
        if self.particles is None or len(self.particles) == 0:
            print("Heatmap Regeneration: No particles.")
            return
        if not self.struct_results:
            print("Heatmap Regeneration: No structural results.")
            return
            
        try:
            h_ary, w_ary = gv.aryData.shape[:2]
            res_px = (w_ary * 2, h_ary * 2)
            
            # Robust parsing of R1
            text_r1 = self.res_r1.text().replace('nm', '').strip()
            r1 = float(text_r1 if text_r1 else 10.0)
            map_radius = 0.33 * r1
            
            # Vectorized coordinate transformation
            px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
            coords_nm = np.column_stack((self.particles[:, 1] * px_x, self.particles[:, 0] * px_y))
            
            print(f"Regenerating heatmaps for {len(self.particles)} particles. Radius: {map_radius:.2f} nm")
            
            # Map P3-P5
            keys = ['Z', 'Psi6', 'Psi4', 'Psi2', 'lambda1', 'lambda2', 'aspect']
            count_p3 = 0
            for k in keys:
                if k in self.struct_results and len(self.struct_results[k]) > 0:
                    self.struct_maps[k] = generate_structural_map(
                        coords_nm, self.struct_results[k], (gv.XScanSize, gv.YScanSize), res_px, map_radius
                    )
                    count_p3 += 1
            print(f"Generated {count_p3} structural maps (P3-P5).")
            
            # Map P8
            valid_fits = [v for k, v in self.fitting_results.items()]
            if valid_fits:
                # Note: Fitting results stored in fitting_results are in Nanometers
                fit_coords_nm = np.array([[v['x0'], v['y0']] for v in valid_fits])
                
                ar_vals = np.array([max(v['a'], v['b']) / min(v['a'], v['b']) if min(v['a'], v['b']) > 0 else 1.0 for v in valid_fits])
                ori_vals = np.array([v['angle'] for v in valid_fits])
                
                self.struct_maps['fit_aspect'] = generate_structural_map(
                    fit_coords_nm, ar_vals, (gv.XScanSize, gv.YScanSize), res_px, map_radius
                )
                self.struct_maps['fit_ori'] = generate_structural_map(
                    fit_coords_nm, ori_vals, (gv.XScanSize, gv.YScanSize), res_px, map_radius
                )
                print("Generated 2 fitting maps (P8).")
            else:
                print("No valid fits found for P8 maps.")
        except Exception as e:
            print(f"Heatmap Regeneration Error: {e}")
            traceback.print_exc()

    def on_segmentation_error(self, err_msg):
        """Recovery logic if worker fails."""
        print(f"Worker Error: {err_msg}")
        self.segment_btn.setText("Segment Clustered Particles")
        self.segment_btn.setEnabled(True)
        self.fit_ellipse_btn.setText("Identify 2D Ellipses")
        self.fit_ellipse_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Worker Error", f"An error occurred during processing:\n{err_msg}")

    def make_checkwave(self):
        """Generates a binary Igor-style wave of particle positions. / パーティクル位置のバイナリIgorウェーブを生成"""
        if len(self.particles) == 0 or not hasattr(gv, 'aryData'): return
        
        h_ary, w_ary = gv.aryData.shape[:2]
        px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
        coords_nm = np.column_stack((self.particles[:, 1] * px_x, self.particles[:, 0] * px_y))
        
        # Save to global temporary storage for Igor export compatibility
        gv.checkX, gv.checkY = coords_nm[:, 0], coords_nm[:, 1]
        
        self.update_particle_display()
        self.update_particle_zoom()
        QtWidgets.QMessageBox.information(self, "Success", f"Saved {len(coords_nm)} coordinates to checkwave buffer.")

    def update_results(self):
        """Refreshes the results table with particle IDs and coordinates. / 結果テーブルをパーティクルIDと座標で更新"""
        if not hasattr(gv, 'aryData') or gv.aryData is None: return
        self.results_table.setRowCount(len(self.particles))
        h_ary, w_ary = gv.aryData.shape[:2]
        px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
        
        # Header is: ["ID", "X (nm)", "Y (nm)", "Z (nm)", "Axis A", "Axis B", "Angle (vs X)", "RMSE"]
        for i, (py, px) in enumerate(self.particles):
            id_item = QtWidgets.QTableWidgetItem(str(i))
            self.results_table.setItem(i, 0, id_item)
            
            # Use raw gv data scaling if available
            h_ary, w_ary = gv.aryData.shape[:2]
            px_x, px_y = gv.XScanSize / w_ary, gv.YScanSize / h_ary
            
            x_nm = px * px_x
            y_nm = py * px_y
            z_nm = gv.aryData[int(py), int(px)]
            
            self.results_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{x_nm:.2f}"))
            self.results_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{y_nm:.2f}"))
            self.results_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{z_nm:.2f}"))
            
            # Fill fitting results if available
            pid = i + 1 # Internal labels are 1-based
            if hasattr(self, 'fitting_results') and pid in self.fitting_results:
                fr = self.fitting_results[pid]
                self.results_table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{fr['a']:.2f}"))
                self.results_table.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{fr['b']:.2f}"))
                self.results_table.setItem(i, 6, QtWidgets.QTableWidgetItem(f"{fr['angle']:.2f}"))
                self.results_table.setItem(i, 7, QtWidgets.QTableWidgetItem(f"{fr['rmse']:.3f}"))
            else:
                for col in range(4, 8):
                    self.results_table.setItem(i, col, QtWidgets.QTableWidgetItem("-"))


# =============================================================================
# Plugin Factory Implementation
# =============================================================================

# 1. Plugin Display Name
PLUGIN_NAME = "Particle Cluster Analysis"

# 2. Factory Function
def create_plugin(main_window):
    """
    Factory function called by pyNuD Plugin Manager.
    Returns an instance of the ParticleClusterWindow.
    """
    return ParticleClusterWindow(main_window)

# 3. Exported Symbols
__all__ = ["PLUGIN_NAME", "create_plugin"]


# =============================================================================
# Math Functions.
# =============================================================================

def log_filter(image, diameter_nm, pixel_size_nm):
    """
    Reproduces Igor Pro's LoGFilter using FFT for edge detection.

    The filter applies a Laplacian of Gaussian (LoG) kernel in the frequency domain.

    Parameters
    ----------
    image : ndarray
        2D input image array.
    diameter_nm : float
        Estimated particle diameter in nanometers.
    pixel_size_nm : float
        Pixel size in nanometers.

    Returns
    -------
    filtered_img : ndarray
        The filtered image in the spatial domain.
    """
    sigma = diameter_nm / (2 * pixel_size_nm * np.sqrt(2))
    
    orig_h, orig_w = image.shape
    pad_h = 1 if orig_h % 2 != 0 else 0
    pad_w = 1 if orig_w % 2 != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        # Pad at the beginning (top/left) to mimic Igor's vstack/hstack logic
        image = np.pad(image, ((pad_h, 0), (pad_w, 0)), mode='edge')
    
    new_h, new_w = image.shape
    
    # Frequency domain LoG kernel
    # Igor's LoGWave: ((p-Xsize)^2+(q-Ysize)^2-2*sigma^2)/(2*sigma^6)*exp(-((p-Xsize)^2+(q-Ysize)^2)/(2*sigma^2))
    y = np.arange(new_h) - new_h // 2
    x = np.arange(new_w) - new_w // 2
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2 + yy**2
    
    # Kernel in spatial domain
    kernel = (r2 - 2*sigma**2) / (2 * sigma**6) * np.exp(-r2 / (2 * sigma**2))
    
    # FFT filtering
    img_fft = fft2(image)
    kernel_fft = fft2(fftshift(kernel)) # Shift kernel to origin for FFT
    
    filtered_fft = img_fft * kernel_fft
    filtered_img = np.real(ifft2(filtered_fft))
    
    # Crop back to original size if padded
    if pad_h > 0 or pad_w > 0:
        filtered_img = filtered_img[pad_h:, pad_w:]
        
    return filtered_img

def local_max_search(image, threshold, filter_size=3):
    """
    Identifies local maxima in the image above a given threshold.

    Reproduces the LocalMaxSearch logic from Igor Pro by using a maximum filter
    and comparing the result with the original image.

    Parameters
    ----------
    image : ndarray
        Input image array.
    threshold : float
        The minimum value to consider a pixel as a potential local maximum.
    filter_size : int, optional
        The size of the neighborhood to search for maxima (default is 3).

    Returns
    -------
    coords : ndarray
        An array of (row, col) coordinates of the identified local maxima.
    """
    # MatrixFilter/N=(MaxFilsize)/P=1 max Filt_Image
    # Equivalent to maximum_filter
    max_filt = ndimage.maximum_filter(image, size=filter_size)
    
    # mask where pixel is the local maximum and above threshold
    mask = (image == max_filt) & (image > threshold)
    
    coords = np.argwhere(mask)
    # Igor returns (x, y) in physical units or pixel indices.
    # Here we return (row, col) which is (y_idx, x_idx)
    return coords

def delete_edge_particles(particles, image_shape):
    """
    Removes particles that are located too close to the image edges.
    Specifically, keeps only the 10%~90% range for each direction as per requirement.

    Parameters
    ----------
    particles : ndarray
        Array of (row, col) particle coordinates.
    image_shape : tuple
        Shape of the image (height, width).

    Returns
    -------
    valid_particles : ndarray
        The filtered array of particle coordinates.
    """
    h, w = image_shape
    y_min, y_max = 0.1 * h, 0.9 * h
    x_min, x_max = 0.1 * w, 0.9 * w
    
    valid_particles = []
    for p in particles:
        y, x = p
        if (y >= y_min and y < y_max and 
            x >= x_min and x < x_max):
            valid_particles.append(p)
    return np.array(valid_particles)

def pixels_to_nm(coords, height_px, offset_x, offset_y, delta_x, delta_y):
    """
    Converts pixel coordinates to physical units (nm).

    Assumes the coordinate system (0,0) is at the bottom-left of the image.
    The input Y-coordinates (row indices) are flipped such that row 0 corresponds 
    to the top of the image and row (height-1) corresponds to the bottom.

    Parameters
    ----------
    coords : ndarray
        Array of (row, col) coordinates.
    height_px : int
        The height of the image in pixels.
    offset_x : float
        X-axis offset in nm.
    offset_y : float
        Y-axis offset in nm.
    delta_x : float
        X-axis pixel size in nm.
    delta_y : float
        Y-axis pixel size in nm.

    Returns
    -------
    coords_nm : ndarray
        Array of (x, y) coordinates in physical units (nm).
    """
    # Flip Y: Row 0 -> YScanSize, Row (H-1) -> 0
    y_nm = offset_y + (height_px - 1 - coords[:, 0]) * delta_y
    x_nm = offset_x + coords[:, 1] * delta_x
    return np.column_stack((x_nm, y_nm))

def calculate_gr(coords_nm, scan_size_nm, dr=1.0, max_r=None):
    """
    Calculates the radial distribution function g(r) using a vectorized method.

    The g(r) function describes how density varies as a function of distance 
    from a reference particle.

    Parameters
    ----------
    coords_nm : ndarray
        (n, 2) array of (x, y) coordinates in nm.
    scan_size_nm : tuple
        (width_nm, height_nm) of the scan area.
    dr : float, optional
        Bin width in nm (default is 1.0).
    max_r : float, optional
        Maximum radius to calculate. Defaults to half the minimum scan dimension.

    Returns
    -------
    rs : ndarray
        Center of the distance bins.
    gr : ndarray
        Value of the radial distribution function.
    """
    n = len(coords_nm)
    if n < 2:
        return np.array([]), np.array([])
        
    w_nm, h_nm = scan_size_nm
    if max_r is None:
        max_r = min(w_nm, h_nm) / 2.0
        
    # Vectorized distance matrix calculation
    # (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    # diffs = coords_nm[:, np.newaxis, :] - coords_nm[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diffs**2, axis=-1))
    
    # More memory efficient for large n:
    # dists = scipy.spatial.distance.pdist(coords_nm)
    dists = pdist(coords_nm)
    
    # g(r) = (1 / (n * rho)) * sum_i ( sum_j!=i ( delta(r - |ri - rj|) / area(r) ) )
    # Here n is total particles, rho is n / Area
    rho = n / (w_nm * h_nm)
    
    num_bins = int(max_r / dr)
    counts, bin_edges = np.histogram(dists, bins=num_bins, range=(0, max_r))
    
    # pdist only returns n*(n-1)/2 pairs. The formula for g(r) involves i,j where j!=i.
    # So we multiply counts by 2.
    counts = counts * 2.0
    
    rs = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
    
    # Normalization: count / (n * rho * area)
    gr = counts / (n * rho * areas)
    
    return rs, gr

def analyze_components(coords_nm, cutoff_nm):
    """
    Calculates coordination numbers and bond-orientational order parameters.

    Calculates Psi_6 (hexagonal), Psi_4 (rectangular), and Psi_2 (linear) 
    order parameters for each particle based on its neighbors within a cutoff.

    Parameters
    ----------
    coords_nm : ndarray
        (n, 2) array of (x, y) coordinates in nm.
    cutoff_nm : float
        The distance cutoff for defining neighbors.

    Returns
    -------
    results : dict
        A dictionary containing:
        - 'Z': Coordination numbers (array).
        - 'Psi6': Hexagonal order parameter magnitudes (array).
        - 'Psi4': Rectangular order parameter magnitudes (array).
        - 'Psi2': Linear order parameter magnitudes (array).
    """
    n = len(coords_nm)
    if n < 2:
        return {'Z': np.zeros(n), 'Psi6': np.zeros(n), 'Psi4': np.zeros(n), 'Psi2': np.zeros(n)}
        
    # Full distance matrix
    dists = cdist(coords_nm, coords_nm)
    
    # Adjacency matrix (excluding self)
    adj = (dists <= cutoff_nm) & (np.eye(n, dtype=bool) == False)
    
    Z = np.sum(adj, axis=1)
    Psi6 = np.zeros(n)
    Psi4 = np.zeros(n)
    Psi2 = np.zeros(n)
    
    for i in range(n):
        neighbors = coords_nm[adj[i]]
        if len(neighbors) > 0:
            diffs = neighbors - coords_nm[i]
            # Angles in radians
            phis = np.arctan2(diffs[:, 1], diffs[:, 0])
            
            # Psi_k = |sum(exp(i*k*phi)) / n_neighbors|
            def calc_psi(k):
                sum_phase = np.sum(np.exp(1j * k * phis))
                # Igor normalization uses k if Z <= k, else Z. Let's follow the standard: Z
                norm = max(len(neighbors), k) # Mimicking Igor logic: If(numExtracted<=k) Hex=... /k Else ... /num
                return np.abs(sum_phase) / norm
            
            Psi6[i] = calc_psi(6)
            Psi4[i] = calc_psi(4)
            Psi2[i] = calc_psi(2)
            
    return {'Z': Z, 'Psi6': Psi6, 'Psi4': Psi4, 'Psi2': Psi2}

def generate_structural_map(coords_nm, values, scan_size_nm, resolution_px, circle_radius_nm=150.0):
    """
    Generates a 2D structural heatmap by drawing circles around particles.

    Each particle is represented as a filled circle with the associated value
    on a heatmap grid.

    Parameters
    ----------
    coords_nm : ndarray
        (n, 2) array of (x, y) coordinates in nm.
    values : ndarray
        Array of values to map (e.g., Psi6, Z).
    scan_size_nm : tuple
        (width_nm, height_nm) of the scan area.
    resolution_px : tuple
        (width_px, height_px) of the output heatmap.
    circle_radius_nm : float, optional
        Radius of the circles in nm (default is 150.0).

    Returns
    -------
    heatmap : ndarray
        A 2D float32 array representing the structural heatmap.
    """
    w_nm, h_nm = scan_size_nm
    w_px, h_px = resolution_px
    
    # Initialize heatmap with NaN
    # Note: cv2.circle supports float32 arrays
    heatmap = np.full((h_px, w_px), np.nan, dtype=np.float32)
    
    px_scale_x = w_px / w_nm
    px_scale_y = h_px / h_nm
    
    radius_px = int(circle_radius_nm * px_scale_x)
    if radius_px < 1: radius_px = 1
    
    for i in range(len(coords_nm)):
        cx = int(coords_nm[i][0] * px_scale_x)
        cy = int(coords_nm[i][1] * px_scale_y)
        val = float(values[i])
        
        # Guard against coords outside the target resolution and skip NaNs
        if not np.isnan(val) and 0 <= cx < w_px and 0 <= cy < h_px:
            # Draw filled circle directly on the float array
            cv2.circle(heatmap, (cx, cy), radius_px, val, -1)
        
    return heatmap

def calculate_finger_tensor(coords_nm, r1_nm, v1_nm, v2_nm=None):
    """
    Calculates the Finger Tensor and derived structural anisotropy parameters.

    This function identifies the nearest neighbors for each particle (up to 6)
    within an elliptical cutoff and calculates the Finger Tensor to determine
    local order anisotropy and orientation.

    Parameters
    ----------
    coords_nm : ndarray
        (n, 2) array of (x, y) coordinates in nm.
    r1_nm : float
        Normalization distance (typically the first peak R1 from g(r)).
    v1_nm : float
        Primary semi-axis of the search ellipse in nm.
    v2_nm : float, optional
        Secondary semi-axis of the search ellipse in nm. If None, v1_nm is used.

    Returns
    -------
    results : dict
        A dictionary containing:
        - 'lambda1': Primary eigenvalues / stretch.
        - 'lambda2': Secondary eigenvalues.
        - 'aspect': Local aspect ratio (lambda1 / lambda2).
        - 'orientation': Orientation angle in degrees.
    """
    n = len(coords_nm)
    if n < 2:
        return {
            'lambda1': np.zeros(n), 'lambda2': np.zeros(n),
            'aspect': np.zeros(n), 'orientation': np.zeros(n)
        }
    
    if v2_nm is None: v2_nm = v1_nm
    
    # 1. Neighbor Selection: Elliptical cutoff + Closest 6
    dists_mat = cdist(coords_nm, coords_nm)
    diffs = coords_nm[:, np.newaxis, :] - coords_nm[np.newaxis, :, :] # (n, n, 2) (dx, dy)
    
    # Elliptical distance: (dx/v1)^2 + (dy/v2)^2 <= 1
    ellip_dist_sq = (diffs[:, :, 0] / v1_nm)**2 + (diffs[:, :, 1] / v2_nm)**2
    
    # Pre-allocate results
    l1 = np.zeros(n); l2 = np.zeros(n)
    aspect = np.zeros(n); orientation = np.zeros(n)
    
    for i in range(n):
        # Candidates within elliptical cutoff (excluding self)
        candidates_mask = (ellip_dist_sq[i] <= 1.0) & (np.arange(n) != i)
        cand_indices = np.where(candidates_mask)[0]
        
        if len(cand_indices) > 0:
            # Sort candidates by actual Euclidean distance and take top 6
            cand_dists = dists_mat[i, cand_indices]
            top_k_indices = cand_indices[np.argsort(cand_dists)[:6]]
            
            # Local vectors r_ij for the top N neighbors
            r_ij = coords_nm[top_k_indices] - coords_nm[i]
            N = len(top_k_indices)
            
            # 2. Tensor Calculation: F = (2 / (R1^2 * N)) * sum(r_ij \otimes r_ij)
            # sum(r_ij \otimes r_ij) is equivalent to r_ij.T @ r_ij
            F_sum = np.dot(r_ij.T, r_ij)
            F = (2.0 / (r1_nm**2 * N)) * F_sum
            
            # 3. Eigenvalues and Parameters
            try:
                evals, evecs = np.linalg.eigh(F)
                e1, e2 = evals[1], evals[0] # eigh returns ascending
                
                l1[i] = np.sqrt(max(e1, 0))
                l2[i] = np.sqrt(max(e2, 0))
                
                # Aspect Ratio
                if l2[i] > 0: aspect[i] = l1[i] / l2[i]
                
                # Primary eigenvector (corresponding to l1)
                v1_vec = evecs[:, 1]
                orientation[i] = np.degrees(np.arctan2(v1_vec[1], v1_vec[0]))
                
            except (np.linalg.LinAlgError, ValueError) as e:
                # Log error or set defaults if needed, but keep arrays zeroed
                continue
                
    return {
        'lambda1': l1, 'lambda2': l2,
        'aspect': aspect, 'orientation': orientation
    }

def compute_curvature_map(height_data, gaussian_sigma=1.2, enable_denoise=True):
    """
    Generates a map of average principal curvature using Hessian eigenvalues. 
    / Hessian行列の固有値を用いた平均主曲率マップの生成。

    Curvature sign changes (convex to concave) are used to identify 
    particle boundaries in overlapping clusters.
    / 曲率の符号変化（凸から凹）を利用して、重なり合った粒子の境界を特定します。

    Parameters
    ----------
    height_data : ndarray
        2D height map array in nm. / nm単位の2次元高さマップ。
    gaussian_sigma : float, optional
        Scale for Gaussian smoothing (default is 1.2). / ガウス平滑化のスケール（デフォルト 1.2）。
    enable_denoise : bool, optional
        If True, applies Gaussian smoothing before derivative calculation. / Trueの場合、微分計算の前にガウス平滑化を適用します。

    Returns
    -------
    curvature_map : ndarray
        2D array of average principal curvature. / 平均主曲率の2次元配列。
    """
    if enable_denoise:
        smoothed = ndimage.gaussian_filter(height_data, sigma=gaussian_sigma)
    else:
        smoothed = height_data

    # 1. Compute Hessian components using second-order gradients
    # Use np.gradient twice or ndimage for cleaner results
    dy, dx = np.gradient(smoothed)
    Iyy, Iyx = np.gradient(dy)
    Ixy, Ixx = np.gradient(dx)
    
    # 2. Compute eigenvalues of the Hessian matrix at each pixel
    # H = [[Ixx, Ixy], [Iyx, Iyy]]
    # lambda = (Trace ± sqrt(Trace^2 - 4*Det)) / 2
    trace = Ixx + Iyy
    det = (Ixx * Iyy) - (Ixy * Iyx)
    
    # Discriminant for the quadratic equation
    # Max(0, ...) to avoid numerical sqrt of negatives
    disc = np.sqrt(np.maximum(0, trace**2 - 4*det))
    
    lambda1 = (trace + disc) / 2.0
    lambda2 = (trace - disc) / 2.0
    
    # 3. Average principal curvature H = (k1 + k2) / 2
    # In height maps, peaks are typically regions where eigenvalues are negative (convex)
    curvature_map = (lambda1 + lambda2) / 2.0
    
    return curvature_map

def weighted_watershed_seg(height_data, particles_px, v1_nm, scan_size_nm, alpha=1.0, beta=5.0, sigma=1.2, denoise_on=True):
    """
    Curvature-Distance Weighted Watershed algorithm using competitive region growing.
    / 曲率と距離を重み付けした、競争的領域成長による分水嶺アルゴリズム。

    Uses Dijkstra's algorithm to assign pixels to particle markers based on 
    a cost function combining Euclidean distance and curvature penalties.
    / ダイクストラ法を用い、ユークリッド距離と曲率ペナルティを組み合わせたコスト関数に基づいてピクセルを粒子マーカーに割り当てます。

    Parameters
    ----------
    height_data : ndarray
        2D height map array in nm. / nm単位の2次元高さマップ。
    particles_px : ndarray
        (N, 2) array of [row, col] pixel coordinates for particle seeds. / 粒子シードの[row, col]ピクセル座標配列。
    v1_nm : float
        First neighbor shell cutoff in nm (used for local ROI constraint). / nm単位の第1近傍シェル・カットオフ（ローカルROI制約に使用）。
    scan_size_nm : tuple
        (width_nm, height_nm) for pixel-to-nm conversion. / ピクセルからnmへの変換用スキャンサイズ。
    alpha : float, optional
        Weight for spatial distance (default 1.0). / 空間距離の重み（デフォルト 1.0）。
    beta : float, optional
        Weight for curvature sign-change penalty (default 5.0). / 曲率符号変化ペナルティの重み（デフォルト 5.0）。
    sigma : float, optional
        Smoothing scale for curvature map (default 1.2). / 曲率マップの平滑化スケール（デフォルト 1.2）。
    denoise_on : bool, optional
        Whether to denoise before curvature calculation (default True). / 曲率計算の前にノイズ除去を行うかどうか。

    Returns
    -------
    labels : ndarray
        Integer labels for segmented regions. / セグメント化された領域の整数ラベル。
    confidence_map : ndarray
        0-1 confidence map based on path costs and curvature consistency. / パスコストと曲率の一貫性に基づく0-1の信頼度マップ。
    curvature_map : ndarray
        The computed curvature map used for segmentation. / セグメンテーションに使用された曲率マップ。
    """
    import heapq
    h_px, w_px = height_data.shape
    w_nm, h_nm = scan_size_nm
    
    # 1. Initialize data structures
    labels = np.zeros((h_px, w_px), dtype=np.int32)
    costs = np.full((h_px, w_px), np.inf, dtype=np.float32)
    priority_queue = [] # Min-heap: (cost, y, x, label_id)
    
    # 2. Setup Pixel and ROI parameters
    px_x_size = w_nm / w_px
    px_y_size = h_nm / h_px
    # Local search radius in pixels (1.0 * V1)
    # We use a square bounding box for efficiency as per spec
    roi_radius_px = (v1_nm / px_x_size)
    
    # 3. Precompute Curvature
    curv_map = compute_curvature_map(height_data, sigma, denoise_on)
    
    # 4. Seed the Priority Queue
    for i, (py, px) in enumerate(particles_px):
        label_id = i + 1
        labels[py, px] = label_id
        costs[py, px] = 0.0
        heapq.heappush(priority_queue, (0.0, py, px, label_id, py, px)) # (cost, y, x, label, origin_y, origin_x)

    # 5. Dijkstra Expansion
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while priority_queue:
        curr_cost, y, x, label_id, oy, ox = heapq.heappop(priority_queue)
        
        if curr_cost > costs[y, x]:
            continue
            
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            
            # Bounds check
            if not (0 <= ny < h_px and 0 <= nx < w_px):
                continue
            
            # Already labeled check
            if labels[ny, nx] != 0:
                continue
                
            # ROI Constraint check: |x - pi_x| < 1.0*V1
            if abs(nx - ox) > roi_radius_px or abs(ny - oy) > roi_radius_px:
                continue
                
            # Cost Function: alpha*dist + beta*curvature_penalty
            # step_dist is constant 1.0 (or sqrt(2)) for pixels
            step_dist = np.sqrt(dy**2 + dx**2)
            
            # Curvature Penalty and Boundary Stopping
            c_from = curv_map[y, x]
            c_to = curv_map[ny, nx]
            
            # PERFORMANCE ENHANCEMENT: Zero-Crossing Barrier
            # Do not allow growth into positive curvature pixels (concave regions/valleys).
            # / 曲率が正のピクセル（凹領域/谷）への成長を許可しない。
            if c_to > 0.0:
                continue

            # Curvature consistency penalty (within convex region)
            penalty = np.abs(c_from - c_to)
                
            new_cost = curr_cost + (alpha * step_dist) + (beta * penalty)
            
            if new_cost < costs[ny, nx]:
                costs[ny, nx] = new_cost
                labels[ny, nx] = label_id
                heapq.heappush(priority_queue, (new_cost, ny, nx, label_id, oy, ox))

    # 6. Compute Confidence Map
    # Confidence is high near the peak (cost=0) and drops toward boundaries.
    # Because we blocked growth into concave regions, costs[curv_map > 0] is already inf.
    # / 信頼度はピーク（コスト=0）付近で高く、境界に向かって低下します。
    # / 凹領域への成長をブロックしたため、costs[curv_map > 0] は既に inf です。
    ref_cost = np.median(costs[costs < np.inf]) if np.any(costs < np.inf) else 1.0
    confidence_map = np.exp(-costs / (ref_cost + 1e-6))
    
    # Strictly zero out non-convex regions
    # / 非凸（凹）領域を厳密にゼロにします。
    confidence_map[curv_map > 0] = 0.0
    
    # 7. Connectivity Pruning (Enforce single solid body per seed)
    # / 連結性プルーニング（各シードに対して単一の個体を強制）
    # Use scipy.ndimage.label to find connected components for each label
    from scipy.ndimage import label as nd_label
    
    final_labels = np.zeros_like(labels)
    for i, (py, px) in enumerate(particles_px):
        label_id = i + 1
        # Extract binary mask for this label
        mask = (labels == label_id)
        if not np.any(mask): continue
        
        # Identify connected components in this mask (4-connectivity)
        structure = [[0,1,0], [1,1,1], [0,1,0]]
        cc_map, num_features = nd_label(mask, structure=structure)
        
        if num_features > 1:
            # Multi-component found. Keep only the one containing the seed.
            seed_cc_id = cc_map[py, px]
            if seed_cc_id > 0:
                final_labels[cc_map == seed_cc_id] = label_id
            else:
                # Seed itself was masked or costed out? Should not happen with current logic,
                # but fallback to closest component if needed.
                final_labels[mask] = label_id
        else:
            # Single component, keep as is
            final_labels[mask] = label_id

    return final_labels, confidence_map, curv_map

def local_curvature_gating_seg(height_data, particles_px, v1_nm, scan_size_nm, threshold=0.2, sigma=1.2, denoise_on=True):
    """
    Refined Basic segmentation using CCA on the gated curvature mask.
    / ゲート付き曲率マスク上のCCAを用いた洗練された基本セグメンテーション。

    1. Generate global normalized convexity score.
    2. Threshold to create binary mask (matches diagnostic green overlay).
    3. Label components using CCA.
    4. Assign labels to components based on particle seed containment.
    """
    h_px, w_px = height_data.shape
    w_nm, h_nm = scan_size_nm
    px_x_size = w_nm / w_px
    roi_rad = int(v1_nm / px_x_size)
    
    curv_map = compute_curvature_map(height_data, sigma, denoise_on)
    
    # 1. Global Normalized Scoring
    # We build a 'normalized convexity' map. 
    # For each particle peak, convex pixels near it are scored 0-1.
    scores = np.zeros((h_px, w_px), dtype=np.float32)
    
    for i, (py, px) in enumerate(particles_px):
        y0, y1 = max(0, int(py - roi_rad)), min(h_px, int(py + roi_rad + 1))
        x0, x1 = max(0, int(px - roi_rad)), min(w_px, int(px + roi_rad + 1))
        
        crop = curv_map[y0:y1, x0:x1]
        c_min = np.nanmin(crop)
        if c_min >= -1e-9: continue
            
        local_scores = np.where(crop <= 0, crop / c_min, 0.0)
        scores[y0:y1, x0:x1] = np.maximum(scores[y0:y1, x0:x1], local_scores)

    # 2. Threshold Gating (The "Gating Mask")
    gating_mask = (scores >= threshold)
    
    # 3. Connected Component Analysis
    from scipy.ndimage import label as nd_label
    cc_map, num_features = nd_label(gating_mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
    
    # 4. Seed-to-Component Mapping
    labels = np.zeros((h_px, w_px), dtype=np.int32)
    
    # Group particles by which CC they fall into
    cc_to_particles = {} # cc_id -> list of indices i
    for i, (py, px) in enumerate(particles_px):
        cc_id = cc_map[int(np.clip(py, 0, h_px-1)), int(np.clip(px, 0, w_px-1))]
        if cc_id > 0:
            if cc_id not in cc_to_particles:
                cc_to_particles[cc_id] = []
            cc_to_particles[cc_id].append(i)
            
    # 5. Label Assignment
    for cc_id in range(1, num_features + 1):
        if cc_id not in cc_to_particles:
            # Mask segment encloses no particle -> Discard
            continue
            
        particle_indices = cc_to_particles[cc_id]
        comp_mask = (cc_map == cc_id)
        
        if len(particle_indices) == 1:
            # Single particle in block -> Entire segment belongs to it
            labels[comp_mask] = particle_indices[0] + 1
        else:
            # Multiple particles (Cluster) -> Split by distance within the mask
            # Local Voronoi split restricted to the component
            y_indices, x_indices = np.where(comp_mask)
            points = np.column_stack((y_indices, x_indices))
            
            seeds = particles_px[particle_indices] # (K, 2)
            
            # Simple distance-based assignment for speed
            from scipy.spatial import cKDTree
            tree = cKDTree(seeds)
            _, nearest_seeds_local_idx = tree.query(points)
            
            # Map back to global particle IDs
            for j, local_idx in enumerate(nearest_seeds_local_idx):
                gy, gx = points[j]
                labels[gy, gx] = particle_indices[local_idx] + 1

    return labels, scores, curv_map

def filter_by_confidence(labels, confidence_map, threshold=0.75, seeds_px=None):
    """
    Filters segmented labels to only include pixels above a confidence threshold.
    / 信頼度しきい値を超えるピクセルのみを含むように、セグメント化されたラベルをフィルターします。

    Parameters
    ----------
    labels : ndarray
        Integer labels from segmentation. / セグメンテーションによる整数ラベル。
    confidence_map : ndarray
        0-1 confidence values. / 0-1の信頼度。
    threshold : float, optional
        Minimum confidence to retain a pixel (default is 0.75). / ピクセルを保持するための最小信頼度（デフォルト 0.75）。
    seeds_px : ndarray, optional
        (N, 2) array of [row, col] seed coordinates to enforce connectivity. 
        / 連結性を強制するための[row, col]シード座標配列。

    Returns
    -------
    filtered_labels : ndarray
        Masked and pruned label map. / マスキングおよびプルーニングされたラベルマップ。
    """
    # 1. Apply confidence threshold
    binary_mask = (confidence_map >= threshold)
    
    # 2. Mask the labels
    filtered = np.where(binary_mask, labels, 0)
    
    # 3. Dynamic Connectivity Pruning (Enforce single body per seed)
    if seeds_px is not None:
        h, w = filtered.shape[:2]
        from scipy.ndimage import label as nd_label
        final = np.zeros_like(filtered)
        for i, (py, px) in enumerate(seeds_px):
            label_id = i + 1
            mask = (filtered == label_id)
            if not np.any(mask): continue
            
            # Robust seed coordinates (clip in case of detection/resizing drift)
            sy, sx = int(np.clip(py, 0, h - 1)), int(np.clip(px, 0, w - 1))
            
            # Check if seed itself is still in the mask
            if not mask[sy, sx]:
                continue
                
            cc_map, num_features = nd_label(mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
            seed_cc_id = cc_map[sy, sx]
            if seed_cc_id > 0:
                final[cc_map == seed_cc_id] = label_id
        return final
    
    return filtered

def fit_ellipse_to_mask(mask, px_size_nm, r1_nm, v1_nm, v2_nm):
    """
    Fits a 2D ellipse to the segmentation mask points.
    / セグメンテーションマスクポイントに2D楕円をフィットさせます。
    
    Parameters
    ----------
    mask : ndarray
        Binary mask for the particle.
    px_size_nm : float
        Pixel size in nanometers.
    r1_nm : float
        Initial a-axis semi-length (nm).
    v1_nm : float
        Initial b-axis semi-length (nm).
    v2_nm : float
        Used for the upper bound (2*v2).

    Returns
    -------
    a, b, angle_y, rmse, x0, y0 : tuple
        Fitted semi-axes, orientation vs Y, RMSE, and center coordinates.
    """
    # 1. Coordinate extraction
    y_idx, x_idx = np.where(mask)
    if len(y_idx) < 5: # Need at least 5 points for an ellipse
        return 0, 0, 0, 0, 0, 0
    
    x = x_idx * px_size_nm
    y = y_idx * px_size_nm
    
    # 2. Initial Guess
    x0_init = np.mean(x)
    y0_init = np.mean(y)
    
    # Parameters: [a, b, x0, y0, phi]
    p0 = np.array([r1_nm, v1_nm, x0_init, y0_init, 0.0])
    
    def residuals(p):
        a, b, x0, y0, phi = p
        
        dx = x - x0
        dy = y - y0
        
        # Rotate in XY
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        
        xr = dx * cos_p + dy * sin_p
        yr = -dx * sin_p + dy * cos_p
        
        # Ellipse equation residual: (xr/a)^2 + (yr/b)^2 - 1.0
        # Since we use all pixels in the mask, the "ideal" sum of residuals 
        # is not 0. However, for a boundary-only fit it would be.
        # To handle filled masks, we can fit to the contour or use moments.
        # But to honor Least Squares + Constraints, we'll use the contour.
        return (xr/a)**2 + (yr/b)**2 - 1.0

    # Extract contour for more accurate shape fitting
    mask_8u = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cnt = contours[0].reshape(-1, 2)
        x_cnt = cnt[:, 0] * px_size_nm
        y_cnt = cnt[:, 1] * px_size_nm
        
        def res_cnt(p):
            a, b, x0, y0, phi = p
            dx = x_cnt - x0
            dy = y_cnt - y0
            cp, sp = np.cos(phi), np.sin(phi)
            xr = dx * cp + dy * sp
            yr = -dx * sp + dy * cp
            return (xr/a)**2 + (yr/b)**2 - 1.0
            
        points_to_fit = res_cnt
    else:
        points_to_fit = residuals

    # Bounds: a, b <= 2 * V2
    max_axis = 2.0 * v2_nm
    min_axis = 0.5 # Safety minimum in nm
    bounds = (
        [min_axis, min_axis, -np.inf, -np.inf, -np.pi],
        [max_axis, max_axis, np.inf, np.inf, np.pi]
    )
    
    try:
        res = least_squares(points_to_fit, p0, bounds=bounds, ftol=1e-3, xtol=1e-3)
        a, b, x0, y0, phi = res.x
        rmse = np.sqrt(np.mean(res.fun**2))
        
        # Angle of Axis A vs X-direction (standard polar)
        angle_a = np.degrees(phi)
        
        # Ensure 'a' is the major axis (the longest semi-axis)
        if b > a:
            a, b = b, a
            angle_maj = angle_a + 90.0
        else:
            angle_maj = angle_a
            
        # Wrap angle to [0, 180) range as requested
        angle_maj = angle_maj % 180.0
        
        return a, b, angle_maj, rmse, x0, y0
    except Exception as e:
        print(f"Ellipse Fitting Error: {e}")
        return 0, 0, 0, 0, 0, 0

def extract_height_points(crop_data, z_high_nm, z_res_nm=0.1):
    """
    Extracts (x, y) point cloud within a vertical slice relative to the local peak.
    / ローカルピークからの垂直スライス内の(x, y)点群を抽出します

    Parameters:
    crop_data -- 2D array of heights in a cropped region.
    z_high_nm -- The vertical distance from the peak for the slice center.
    z_res_nm -- The half-width of the vertical slice in nm (default is 0.1).

    Returns:
    points_px -- An array of (row, col) pixel coordinates relative to the crop.
    """
    if crop_data is None or crop_data.size == 0:
        return np.array([])
    
    # 1. Find local peak z0
    z0 = np.nanmax(crop_data)
    target_z = z0 - z_high_nm
    
    # 2. Slice condition
    mask = (crop_data >= target_z - z_res_nm) & (crop_data <= target_z + z_res_nm)
    
    # 3. Get coordinates (row, col)
    points_px = np.argwhere(mask)
    
    return points_px
