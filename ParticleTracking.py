#!/usr/bin/env python3
"""
Tracker Module for pyNuD
pyNuD用トラッカーモジュール

AFM画像シーケンスから粒子を検出し、トラッキングを行う機能を提供します。
Provides particle detection and tracking functionality for AFM image sequences.
"""

# プラグイン契約: Plugin メニューから Load Plugin で読み込み
PLUGIN_NAME = "Particle Tracking"


def create_plugin(main_window):
    """pyNuD から呼び出されるエントリポイント。"""
    return ParticleTrackingWindow(main_window)


# プラグイン内ヘルプ用 HTML（EN/JA）- pyNuD_help.py および integrated_software_page の Particle Tracking より移植
HELP_HTML_EN = """
    <h1>Particle Tracking</h1>
    <h2>Overview</h2>
    <p>Particle Tracking is a comprehensive tool for detecting and tracking particles across multiple frames in AFM time-series data. It provides advanced algorithms for particle detection, trajectory linking, and motion analysis.</p>
    <h2>Access</h2>
    <ul>
        <li><strong>Plugin menu:</strong> Load Plugin... → select <code>plugins/ParticleTracking.py</code>, then Plugin → Particle Tracking</li>
    </ul>
    <h2>Main Features</h2>
    <div class="feature-box">
        <h3>Particle Detection</h3>
        <p>Multiple detection algorithms for different particle types:</p>
        <ul>
            <li><strong>LoG (Laplacian of Gaussian):</strong> Detects blobs using Gaussian filtering and peak detection</li>
            <li><strong>Local Max:</strong> Gaussian filter + peak detection for bright particles</li>
            <li><strong>DoG (Difference of Gaussian):</strong> Similar to LoG, detects particles by difference of Gaussian filters</li>
            <li><strong>Template Matching:</strong> Matches predefined template patterns</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>Detection Parameters</h3>
        <p>Fine-tune detection sensitivity and accuracy:</p>
        <ul>
            <li><strong>Radius (nm):</strong> Expected particle radius for filter size and search radius (0.1-50.0 nm)</li>
            <li><strong>Min Distance:</strong> Minimum distance between detected particles in pixels (1-50)</li>
            <li><strong>Threshold Factor:</strong> Otsu method threshold factor (0.01-1.0, lower = more particles)</li>
            <li><strong>Min Intensity Ratio:</strong> Minimum intensity compared to image mean (0.1-1.0)</li>
            <li><strong>Sub-pixel Correction:</strong> Refine positions using Gaussian fitting</li>
            <li><strong>Max Position Correction:</strong> Search for maximum intensity within radius</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>Size Filtering</h3>
        <p>Filter particles by size using different methods:</p>
        <ul>
            <li><strong>FWHM (Full Width at Half Maximum):</strong> Measure particle size using intensity profile</li>
            <li><strong>Watershed:</strong> Use watershed segmentation for area determination</li>
            <li><strong>Selection Radius:</strong> Radius used for filtering (0.1-50.0 nm)</li>
            <li><strong>Size Tolerance:</strong> Acceptable size range as percentage (5-200%)</li>
            <li><strong>Watershed Parameters:</strong> Compactness (0.01-1.0) and threshold (50-99%)</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>Tracking Algorithms</h3>
        <p>Three powerful tracking algorithms:</p>
        <ul>
            <li><strong>Trackpy (Simple Linker):</strong> Simple nearest neighbor linking</li>
            <li><strong>Simple LAP Tracker (SciPy):</strong> Linear Assignment Problem solver for optimal linking</li>
            <li><strong>Kalman Filter (filterpy):</strong> Predictive tracking with noise filtering</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>Tracking Parameters</h3>
        <p>Control trajectory linking behavior:</p>
        <ul>
            <li><strong>Max Distance (nm):</strong> Maximum distance a particle can move between frames (0.1-200.0 nm)</li>
            <li><strong>Max Frame Gap:</strong> Number of frames a particle can disappear and still be linked (0-10)</li>
            <li><strong>Min Track Length:</strong> Minimum frames for a valid track (2-100)</li>
        </ul>
    </div>
    <h2>Analysis Workflow</h2>
    <div class="step"><strong>Step 1:</strong> Load AFM time-series data and open Particle Tracking (Plugin → Particle Tracking)</div>
    <div class="step"><strong>Step 2:</strong> Configure preprocessing options (smoothing)</div>
    <div class="step"><strong>Step 3:</strong> Set detection parameters (method, radius, threshold)</div>
    <div class="step"><strong>Step 4:</strong> Configure size filtering if needed</div>
    <div class="step"><strong>Step 5:</strong> Choose tracking algorithm and parameters</div>
    <div class="step"><strong>Step 6:</strong> Click "Detect All Frames" to detect particles</div>
    <div class="step"><strong>Step 7:</strong> Click "Track Particles" to link trajectories</div>
    <div class="step"><strong>Step 8:</strong> Review results and export data</div>
    <h2>Interactive Features</h2>
    <div class="feature-box">
        <h3>Manual Particle Editing</h3>
        <ul>
            <li><strong>Add Particles:</strong> Click on image to manually add particles</li>
            <li><strong>Delete Particles:</strong> Right-click on particles to remove them</li>
            <li><strong>Frame Navigation:</strong> Use arrow buttons or slider to move between frames</li>
            <li><strong>Real-time Preview:</strong> See detection results immediately</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>Track Visualization</h3>
        <ul>
            <li><strong>Show Tracks:</strong> Display particle trajectories on image</li>
            <li><strong>Show Track IDs:</strong> Display track identification numbers</li>
            <li><strong>Track Editor:</strong> Open advanced track editing interface</li>
            <li><strong>Track Analysis:</strong> Access detailed motion analysis tools</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>Data Export</h3>
        <ul>
            <li><strong>CSV Export:</strong> Save particle positions and track data</li>
            <li><strong>Image Export:</strong> Save tracking results as images</li>
            <li><strong>Track Statistics:</strong> Export comprehensive track statistics</li>
            <li><strong>Wide Format:</strong> Export data in spreadsheet-friendly format</li>
        </ul>
    </div>
    <h2>Parameter Guidelines</h2>
    <ul>
        <li><strong>Radius:</strong> Should match actual particle size (typically 2-10 nm for AFM)</li>
        <li><strong>Threshold Factor:</strong> Start with 1.0, decrease for more particles</li>
        <li><strong>Min Distance:</strong> Should be larger than particle radius in pixels</li>
        <li><strong>Max Distance:</strong> Should be larger than typical particle movement</li>
        <li><strong>Max Frame Gap:</strong> Allow for temporary particle disappearance</li>
        <li><strong>Min Track Length:</strong> Filter out short, unreliable tracks</li>
    </ul>
    <div class="note">
        <strong>Pro Tips:</strong>
        <ul>
            <li>Start with LoG detection for most AFM particles</li>
            <li>Use sub-pixel correction for high-precision measurements</li>
            <li>Enable size filtering to remove false detections</li>
            <li>Adjust max distance based on particle mobility</li>
            <li>Use track editor to manually correct linking errors</li>
            <li>Export data regularly to avoid loss during analysis</li>
        </ul>
    </div>
"""

HELP_HTML_JA = """
    <h1>粒子トラッキング</h1>
    <h2>概要</h2>
    <p>粒子トラッキングは、AFM時系列データの複数フレームにわたって粒子を検出・追跡する包括的なツールです。粒子検出、軌跡リンク、運動解析のための高度なアルゴリズムを提供します。</p>
    <h2>アクセス方法</h2>
    <ul>
        <li><strong>プラグイン:</strong> Load Plugin... → <code>plugins/ParticleTracking.py</code> を選択し、Plugin → Particle Tracking で開く</li>
    </ul>
    <h2>主な機能</h2>
    <div class="feature-box">
        <h3>粒子検出</h3>
        <p>異なる粒子タイプに対応する複数の検出アルゴリズム：</p>
        <ul>
            <li><strong>LoG (Laplacian of Gaussian):</strong> ガウシアンフィルタリングとピーク検出でブロブを検出</li>
            <li><strong>Local Max:</strong> ガウシアンフィルタ + ピーク検出で明るい粒子を検出</li>
            <li><strong>DoG (Difference of Gaussian):</strong> ガウシアンフィルタの差分で粒子を検出</li>
            <li><strong>Template Matching:</strong> 事前定義テンプレートにマッチング</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>検出パラメータ</h3>
        <ul>
            <li><strong>Radius (nm):</strong> フィルタ・検索半径の期待粒子半径 (0.1-50.0 nm)</li>
            <li><strong>Min Distance:</strong> 検出粒子間の最小距離（ピクセル）(1-50)</li>
            <li><strong>Threshold Factor:</strong> Otsu閾値係数 (0.01-1.0、低いほど多くの粒子)</li>
            <li><strong>Min Intensity Ratio:</strong> 画像平均に対する最小強度 (0.1-1.0)</li>
            <li><strong>Sub-pixel Correction:</strong> ガウシアンフィッティングで位置を精密化</li>
            <li><strong>Max Position Correction:</strong> 半径内で最大強度を検索</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>サイズフィルタリング</h3>
        <ul>
            <li><strong>FWHM:</strong> 強度プロファイルで粒子サイズを測定</li>
            <li><strong>Watershed:</strong> 領域決定にwatershedセグメンテーション</li>
            <li><strong>Selection Radius:</strong> フィルタリング半径 (0.1-50.0 nm)</li>
            <li><strong>Size Tolerance:</strong> 許容サイズ範囲（%）(5-200%)</li>
            <li><strong>Watershed Parameters:</strong> コンパクトネス (0.01-1.0)、閾値 (50-99%)</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>トラッキングアルゴリズム</h3>
        <ul>
            <li><strong>Trackpy (Simple Linker):</strong> 最近傍リンク</li>
            <li><strong>Simple LAP Tracker (SciPy):</strong> 線形割り当てで最適リンク</li>
            <li><strong>Kalman Filter (filterpy):</strong> ノイズフィルタ付き予測トラッキング</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>トラッキングパラメータ</h3>
        <ul>
            <li><strong>Max Distance (nm):</strong> フレーム間の最大移動距離 (0.1-200.0 nm)</li>
            <li><strong>Max Frame Gap:</strong> 消失してもリンクするフレーム数 (0-10)</li>
            <li><strong>Min Track Length:</strong> 有効トラックの最小フレーム数 (2-100)</li>
        </ul>
    </div>
    <h2>解析ワークフロー</h2>
    <div class="step"><strong>ステップ1:</strong> AFM時系列データを読み込み、Plugin → Particle Tracking で開く</div>
    <div class="step"><strong>ステップ2:</strong> 前処理（平滑化）を設定</div>
    <div class="step"><strong>ステップ3:</strong> 検出パラメータ（方法、半径、閾値）を設定</div>
    <div class="step"><strong>ステップ4:</strong> 必要に応じてサイズフィルタを設定</div>
    <div class="step"><strong>ステップ5:</strong> トラッキングアルゴリズムとパラメータを選択</div>
    <div class="step"><strong>ステップ6:</strong> 「Detect All Frames」で粒子検出</div>
    <div class="step"><strong>ステップ7:</strong> 「Track Particles」で軌跡リンク</div>
    <div class="step"><strong>ステップ8:</strong> 結果確認とデータエクスポート</div>
    <h2>インタラクティブ機能</h2>
    <div class="feature-box">
        <h3>手動粒子編集</h3>
        <ul>
            <li><strong>粒子追加:</strong> 画像クリックで手動追加</li>
            <li><strong>粒子削除:</strong> 右クリックで削除</li>
            <li><strong>フレームナビゲーション:</strong> 矢印ボタン・スライダーで移動</li>
            <li><strong>リアルタイムプレビュー:</strong> 検出結果を即時表示</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>軌跡可視化</h3>
        <ul>
            <li><strong>Show Tracks:</strong> 画像上に軌跡を表示</li>
            <li><strong>Show Track IDs:</strong> 軌跡IDを表示</li>
            <li><strong>Track Editor:</strong> 軌跡編集インターフェース</li>
            <li><strong>Track Analysis:</strong> 運動解析ツール</li>
        </ul>
    </div>
    <div class="feature-box">
        <h3>データエクスポート</h3>
        <ul>
            <li><strong>CSV Export:</strong> 粒子位置・軌跡データを保存</li>
            <li><strong>Image Export:</strong> トラッキング結果を画像保存</li>
            <li><strong>Track Statistics:</strong> 軌跡統計をエクスポート</li>
            <li><strong>Wide Format:</strong> スプレッドシート向け形式</li>
        </ul>
    </div>
    <h2>パラメータガイドライン</h2>
    <ul>
        <li><strong>Radius:</strong> 実際の粒子サイズに合わせる（AFMでは通常2-10 nm）</li>
        <li><strong>Threshold Factor:</strong> 1.0から開始、多く検出するなら減少</li>
        <li><strong>Min Distance:</strong> ピクセル単位の粒子半径より大きく</li>
        <li><strong>Max Distance:</strong> 典型的な粒子移動より大きく</li>
        <li><strong>Max Frame Gap:</strong> 一時的な消失を許容</li>
        <li><strong>Min Track Length:</strong> 短い信頼性の低い軌跡を除外</li>
    </ul>
    <div class="note">
        <strong>プロのヒント:</strong>
        <ul>
            <li>ほとんどのAFM粒子はLoG検出から開始</li>
            <li>高精度にはサブピクセル補正を使用</li>
            <li>誤検出除去のためサイズフィルタを有効化</li>
            <li>粒子の移動性に合わせてMax Distanceを調整</li>
            <li>軌跡エディタでリンクエラーを手動修正</li>
            <li>解析中の損失を防ぐため定期的にエクスポート</li>
        </ul>
    </div>
"""


import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os
import sys
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings

# 遅延インポート
try:
    import trackpy as tp
    TRACKPY_AVAILABLE = True
except ImportError:
    tp = None
    TRACKPY_AVAILABLE = False
    print("Warning: trackpy not available")

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    KalmanFilter = None
    FILTERPY_AVAILABLE = False
    print("Warning: filterpy not available")

# Qtのスタイルシート警告を抑制
warnings.filterwarnings("ignore", category=UserWarning)

# trackpyのログ出力を抑制
import logging
logging.getLogger('trackpy').setLevel(logging.ERROR)

# グローバル変数のインポート
try:
    import globalvals as gv
except ImportError:
    gv = None
    print("Warning: globalvals not available")

# TrackAnalysisPanelのインポート
try:
    from track_analysis import TrackAnalysisPanel
except ImportError:
    TrackAnalysisPanel = None
    print("Warning: TrackAnalysisPanel not available")

# 遅延インポート関数
def _import_matplotlib():
    """matplotlibの遅延インポート"""
    global matplotlib, plt
    if matplotlib is None:
        try:
            import matplotlib
            # Qt5バックエンドを明示的に指定（GUIアプリケーション用）
            matplotlib.use('Qt5Agg', force=True)
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            # matplotlibの設定を調整
            plt.rcParams['figure.figsize'] = (8, 6)
            plt.rcParams['figure.dpi'] = 100
            return True
        except ImportError as e:
            print(f"Warning: matplotlib not available: {e}")
            return False
        except Exception as e:
            print(f"Warning: matplotlib initialization failed: {e}")
            return False
    return True

def _import_scipy():
    """scipyの遅延インポート"""
    global scipy
    if 'scipy' not in globals():
        try:
            import scipy
            from scipy.ndimage import gaussian_filter
            return True
        except ImportError:
            print("Warning: scipy not available")
            return False
    return True

# グローバル変数
matplotlib = None
plt = None
scipy = None
SCIPY_AVAILABLE = _import_scipy()

@dataclass
class Particle:
    """粒子データ構造"""
    frame: int          # フレーム番号
    x: float           # X座標
    y: float           # Y座標
    intensity: float   # 強度
    radius: float      # 半径
    quality: float     # 検出品質
    track_id: Optional[int] = None  # 軌跡ID（トラッキング後）

@dataclass
class Track:
    """軌跡データ構造"""
    track_id: int      # 軌跡ID
    particles: List[Particle]  # 粒子リスト
    start_frame: int   # 開始フレーム
    end_frame: int     # 終了フレーム
    duration: float    # 持続時間
    mean_velocity: float = 0.0  # 平均速度
    mean_displacement: float = 0.0  # 平均変位

class SpotDetector:
    """粒子検出クラス"""
    
    def __init__(self):
        self.radius = 5.0
        self.threshold = 1.0  # TrackMateのように1.0をデフォルトに設定（Otsu係数）
        self.min_distance = 5
        self.do_subpixel = True
        self.max_position_correction = True  # 最大値位置補正の有効/無効
        self.min_intensity_ratio = 1.0  # 最小強度比のデフォルト値
        self.size_filter_enabled = True  # サイズフィルタリングの有効/無効
        self.size_tolerance = 0.5  # サイズ許容範囲（半径の±50%）
        self.watershed_area_filter_enabled = False  # Watershed領域フィルタの有効/無効
        self.watershed_compactness = 0.1  # Watershedのコンパクト性パラメーター
        self.watershed_min_size = 10  # モルフォロジー処理の最小サイズ
        self.selection_radius = 5.0  # フィルタリング用の選択半径
        self.area_method = "FWHM"  # 領域決定方法（FWHM or Watershed）
    
    def detect_particles(self, image: np.ndarray, method: str = 'LoG', **kwargs) -> List[Particle]:
        """粒子検出の実行"""
        pass
        
        # パラメータを更新
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 1. 粒子位置検出
        particles = []
        if method == 'LoG':
            particles = self._detect_lap_of_gaussian(image)
        elif method == 'DoG':
            particles = self._detect_difference_of_gaussian(image)
        elif method == 'Template Matching':
            particles = self._detect_template_matching(image)
        elif method == 'Local Max':
            particles = self._detect_local_max(image)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        pass
        
        # 2. サイズフィルタリング（Watershed/FWHM）
        if self.size_filter_enabled and len(particles) > 0:
            try:
                particles = self._filter_particles_by_area(particles, image)
            except Exception as e:
                print(f"[ERROR] Size filtering failed: {e}")
                import traceback
                traceback.print_exc()

        pass
        return particles
    
    def _detect_lap_of_gaussian(self, image: np.ndarray) -> List[Particle]:
        """Laplacian of Gaussian (LoG) による粒子検出 - 改善版"""
        
        pass
        
        # メインウィンドウで前処理されたデータをそのまま使用
        processed_image = image.copy()
        
        # データの正規化（負の値を含む場合）
        original_min = np.min(processed_image)
        original_max = np.max(processed_image)
        
        if original_min < 0:
            # 負の値を0にシフト
            processed_image = processed_image - original_min
        
        if not SCIPY_AVAILABLE:
            return []
        
        from scipy.ndimage import gaussian_laplace
        from skimage.feature import peak_local_max
        from skimage.filters import threshold_otsu
        
        # LoGフィルターを適用（より精度の高い設定）
        sigma = max(1.0, self.radius / 2.5)  # より小さなsigmaで精度向上
        log_data = gaussian_laplace(processed_image, sigma=sigma)
        
        # 負の値（粒子の中心）を検出
        log_abs = np.abs(log_data)
        otsu_threshold = threshold_otsu(log_abs)
        final_threshold = otsu_threshold * self.threshold
        
        # 負の値の局所最小値を検出（粒子の中心）
        min_distance = self.min_distance if hasattr(self, 'min_distance') else max(3, int(self.radius * 1.2))
        
        # 閾値前後のデータ分布を確認
        above_threshold = np.sum(-log_data > final_threshold)
        
        # 極端に低い閾値でのテスト
        if self.threshold < 0.5:
            test_threshold = otsu_threshold * 0.1  # 非常に低い閾値
            test_coordinates = peak_local_max(-log_data, 
                                           min_distance=min_distance,
                                           threshold_abs=test_threshold,
                                           exclude_border=True)
        
        coordinates = peak_local_max(-log_data,  # 負の値を正に反転
                                  min_distance=min_distance,
                                  threshold_abs=final_threshold,
                                  exclude_border=True)
        
        
        
        # 極端に低い閾値でのテスト結果を比較
       
    
        # 品質チェック前の粒子数を記録
        initial_peaks = len(coordinates)
        
        # 品質チェック（強度比フィルタリング）
        if len(coordinates) > 0:
            valid_coordinates = []
            intensity_rejected = 0
            
            # 強度比チェックを有効化
            skip_intensity_check = False
            
            if skip_intensity_check:
                # 従来の強度比チェック
                for coord in coordinates:
                    y, x = coord
                    # 元の画像での強度をチェック（閾値チェックは既にpeak_local_maxで完了）
                    original_intensity = processed_image[y, x]
                    mean_intensity = np.mean(processed_image)
                    
                    # UIのMin intensity ratioを使用
                    min_intensity_ratio = getattr(self, 'min_intensity_ratio', 1.0)
                    
                    # 平均値のmin_intensity_ratio倍以上を要求
                    if original_intensity > mean_intensity * min_intensity_ratio:
                        valid_coordinates.append(coord)
                    else:
                        intensity_rejected += 1
                
                coordinates = np.array(valid_coordinates) if valid_coordinates else np.empty((0, 2), dtype=int)
        
        if len(coordinates) == 0:
            return []
        
        pass
        
        # 粒子リストの作成
        particles = []
        original_positions = []  # 補正前の位置を記録
        corrected_positions = []  # 補正後の位置を記録
        
        for i, (y, x) in enumerate(coordinates):
            # 元の整数座標を保存
            original_x, original_y = int(x), int(y)
            original_positions.append((original_x, original_y))
            
            # 最大値検索による位置補正（有効な場合のみ）
            if self.max_position_correction:
                refined_x, refined_y = self._refine_position_by_maximum(processed_image, original_x, original_y)
                corrected_positions.append((refined_x, refined_y))
            else:
                refined_x, refined_y = float(original_x), float(original_y)
                corrected_positions.append((refined_x, refined_y))
            
            if self.do_subpixel:
                # サブピクセル精度でのさらなる精密化
                subpixel_x, subpixel_y = self._refine_peak_position(processed_image, int(refined_x), int(refined_y))
                x, y = subpixel_x, subpixel_y
            else:
                x, y = refined_x, refined_y
            
            # 強度を取得（補正後の位置から）
            final_x, final_y = int(x), int(y)
            final_x = max(0, min(processed_image.shape[1]-1, final_x))
            final_y = max(0, min(processed_image.shape[0]-1, final_y))
            intensity = processed_image[final_y, final_x]
            quality = abs(log_data[original_y, original_x]) / final_threshold  # LoG値の絶対値
            
            particle = Particle(
                frame=0,  # フレーム番号は後で設定
                x=float(x),
                y=float(y),
                intensity=float(intensity),
                radius=self.radius,
                quality=float(quality)
            )
            particles.append(particle)
        
        # 位置補正の効果をまとめて表示
        if self.max_position_correction and len(original_positions) > 0:
            pass
        
        return particles
    
    def _detect_difference_of_gaussian(self, image: np.ndarray) -> List[Particle]:
        """Difference of Gaussian (DoG) による粒子検出"""
        if not SCIPY_AVAILABLE:
            return []
        
        from scipy.ndimage import gaussian_filter
        
        # 2つの異なるσでガウシアンフィルター（半径に基づいて調整）
        sigma1 = max(0.5, self.radius / 3.0)  # より小さなσ
        sigma2 = max(1.0, self.radius / 1.5)  # より大きなσ
        
        gaussian1 = gaussian_filter(image, sigma1)
        gaussian2 = gaussian_filter(image, sigma2)
        
        # DoG
        dog = gaussian1 - gaussian2
        
        # 閾値処理（LoGと同様の方法）
        from skimage.filters import threshold_otsu
        dog_abs = np.abs(dog)
        otsu_threshold = threshold_otsu(dog_abs)
        final_threshold = otsu_threshold * self.threshold
        
        peaks = dog > final_threshold
        
        # 局所最大値の検出（半径に基づいて最小距離を設定）
        min_distance = max(3, int(self.radius * 1.2))
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(dog, size=min_distance)
        peaks = peaks & (dog == local_max)
        
        # 粒子リストの作成
        particles = []
        peak_coords = np.where(peaks)
        
        for i in range(len(peak_coords[0])):
            y, x = peak_coords[0][i], peak_coords[1][i]
            
            # 元の整数座標を保存
            original_x, original_y = int(x), int(y)
            
            # 最大値検索による位置補正（有効な場合のみ）
            if self.max_position_correction:
                refined_x, refined_y = self._refine_position_by_maximum(image, original_x, original_y)
            else:
                refined_x, refined_y = float(original_x), float(original_y)
            
            if self.do_subpixel:
                # サブピクセル精度でのさらなる精密化
                subpixel_x, subpixel_y = self._refine_peak_position(dog, int(refined_x), int(refined_y))
                x, y = subpixel_x, subpixel_y
            else:
                x, y = refined_x, refined_y
            
            # 強度を取得（補正後の位置から）
            final_x, final_y = int(x), int(y)
            final_x = max(0, min(image.shape[1]-1, final_x))
            final_y = max(0, min(image.shape[0]-1, final_y))
            intensity = image[final_y, final_x]
            quality = dog[original_y, original_x] / final_threshold
            
            particle = Particle(
                frame=0,
                x=float(x),
                y=float(y),
                intensity=float(intensity),
                radius=self.radius,
                quality=float(quality)
            )
            particles.append(particle)
        
        return particles
    
    def _detect_template_matching(self, image: np.ndarray) -> List[Particle]:
        """テンプレートマッチングによる粒子検出"""
        try:
            import cv2
        except ImportError:
            print("[ERROR] OpenCV (cv2) not available for template matching")
            return []
        
        # より適切なテンプレートサイズ（粒子半径の2倍）
        template_size = int(self.radius * 2)
        if template_size % 2 == 0:
            template_size += 1
        
        # ガウシアンテンプレートの作成（より自然な形状）
        template = np.zeros((template_size, template_size))
        center = template_size // 2
        y, x = np.ogrid[:template_size, :template_size]
        
        # ガウシアン分布でテンプレートを作成
        sigma = self.radius / 2.0
        gaussian = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        template = gaussian
        
        # データの正規化（0-255の範囲に）
        image_normalized = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        template_normalized = ((template - np.min(template)) / (np.max(template) - np.min(template)) * 255).astype(np.uint8)
        
        # テンプレートマッチング
        result = cv2.matchTemplate(image_normalized, template_normalized, cv2.TM_CCOEFF_NORMED)
        
        # より厳しい閾値処理
        threshold = 0.3  # 固定の高い閾値
        peaks = result > threshold
        
        # 局所最大値の検出
        if SCIPY_AVAILABLE:
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(result, size=5)  # より大きなフィルターサイズ
            peaks = peaks & (result == local_max)
        
        # 品質チェック（LoGと同様）
        if len(np.where(peaks)[0]) > 0:
            valid_peaks = []
            for y, x in zip(np.where(peaks)[0], np.where(peaks)[1]):
                # 元の画像での強度をチェック
                original_intensity = image[y, x]
                mean_intensity = np.mean(image)
                
                # 平均値の2倍以上を要求（より厳しい条件）
                if original_intensity > mean_intensity * 2.0:
                    valid_peaks.append((y, x))
                    #print(f"[DEBUG] Template particle at ({x}, {y}): match_value={result[y, x]:.3f}, original_intensity={original_intensity:.3f}, mean_intensity={mean_intensity:.3f}, ratio={original_intensity/mean_intensity:.3f}")
           
            # 有効なピークのみを使用
            peak_coords = np.array(valid_peaks) if valid_peaks else np.empty((0, 2), dtype=int)
        else:
            peak_coords = np.empty((0, 2), dtype=int)
        
        # 粒子リストの作成
        particles = []
        for i in range(len(peak_coords)):
            y, x = peak_coords[i]
            
            # 元の整数座標を保存
            original_x, original_y = int(x), int(y)
            
            # 最大値検索による位置補正（有効な場合のみ）
            if self.max_position_correction:
                refined_x, refined_y = self._refine_position_by_maximum(image, original_x, original_y)
            else:
                refined_x, refined_y = float(original_x), float(original_y)
            
            if self.do_subpixel:
                # サブピクセル精度でのさらなる精密化
                subpixel_x, subpixel_y = self._refine_peak_position(result, int(refined_x), int(refined_y))
                x, y = subpixel_x, subpixel_y
            else:
                x, y = refined_x, refined_y
            
            # 強度を取得（補正後の位置から）
            final_x, final_y = int(x), int(y)
            final_x = max(0, min(image.shape[1]-1, final_x))
            final_y = max(0, min(image.shape[0]-1, final_y))
            intensity = image[final_y, final_x]
            quality = result[original_y, original_x] / threshold
            
            particle = Particle(
                frame=0,
                x=float(x),
                y=float(y),
                intensity=float(intensity),
                radius=self.radius,
                quality=float(quality)
            )
            particles.append(particle)
        
        return particles
    
    def _refine_peak_position(self, image: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        """サブピクセル精度でのピーク位置の精密化 - 改善版"""
        if not SCIPY_AVAILABLE:
            return float(x), float(y)
        
        # より大きな領域で2Dガウシアンフィッティング
        if not _import_scipy():
            return float(x), float(y)
        
        try:
            # 周辺領域の抽出（より大きな領域）
            window_size = max(5, int(self.radius * 2))  # 半径の2倍の領域
            y_min, y_max = max(0, y-window_size), min(image.shape[0], y+window_size+1)
            x_min, x_max = max(0, x-window_size), min(image.shape[1], x+window_size+1)
            
            if y_max - y_min < 5 or x_max - x_min < 5:
                return float(x), float(y)
            
            # 2Dガウシアンフィッティング
            from scipy.optimize import curve_fit
            
            # 座標グリッドの作成
            y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
            data = image[y_min:y_max, x_min:x_max]
            
            # 初期パラメータ
            amplitude = np.max(data)
            x0, y0 = x - x_min, y - y_min
            sigma = max(1.0, self.radius / 2.0)  # 粒子半径に基づくsigma
            offset = np.min(data)
            
            # 2Dガウシアン関数
            def gaussian_2d(xy, amplitude, x0, y0, sigma, offset):
                x, y = xy
                return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + offset
            
            # フィッティング
            try:
                popt, _ = curve_fit(gaussian_2d, (x_coords, y_coords), data.flatten(),
                                  p0=[amplitude, x0, y0, sigma, offset],
                                  bounds=([0, 0, 0, 0.5, -np.inf], 
                                         [np.inf, x_max-x_min, y_max-y_min, np.inf, np.inf]))
                
                # 精密化された座標
                refined_x = x_min + popt[1]
                refined_y = y_min + popt[2]
                
                # 座標が画像範囲内にあることを確認
                refined_x = max(0, min(image.shape[1]-1, refined_x))
                refined_y = max(0, min(image.shape[0]-1, refined_y))
                
                return float(refined_x), float(refined_y)
            except Exception as e:
                return float(x), float(y)
                
        except Exception as e:
            return float(x), float(y)
    
    def _detect_local_max(self, image: np.ndarray) -> List[Particle]:
        """Local Max (Gaussian filter + peak detection) による粒子検出"""
        
        
        
        # メインウィンドウで前処理されたデータをそのまま使用
        processed_image = image.copy()
        
        # データの正規化（負の値を含む場合）
        original_min = np.min(processed_image)
        original_max = np.max(processed_image)
        
        if original_min < 0:
            # 負の値を0にシフト
            processed_image = processed_image - original_min
            #print(f"[DEBUG] Shifted data by {original_min:.3f} to remove negative values")
        
        #p rint(f"[DEBUG] Data range after normalization: {np.min(processed_image):.3f} to {np.max(processed_image):.3f}")
        
        if not SCIPY_AVAILABLE:
            return []
        
        from scipy.ndimage import gaussian_filter
        from skimage.feature import peak_local_max
        from skimage.filters import threshold_otsu
        
        # ガウシアンフィルターで平滑化
        sigma = max(1.0, self.radius / 3.0)  # 適度な平滑化
        smoothed_data = gaussian_filter(processed_image, sigma=sigma)
        
        # Otsu法で閾値を計算
        otsu_threshold = threshold_otsu(smoothed_data)
        final_threshold = otsu_threshold * self.threshold
        
        # ピーク検出パラメータを取得
        min_distance = self.min_distance if hasattr(self, 'min_distance') else max(3, int(self.radius * 1.2))
        
        # 閾値ベースでピークを検出
        coordinates = peak_local_max(smoothed_data, 
                                  min_distance=min_distance,
                                  threshold_abs=final_threshold,
                                  exclude_border=True)
        
        # 品質チェック前の粒子数を記録
        initial_peaks = len(coordinates)
        
        # 品質チェック（強度比フィルタリング）
        if len(coordinates) > 0:
            valid_coordinates = []
            intensity_rejected = 0
            
            # 強度比チェックを有効化
            skip_intensity_check = False
            
            if skip_intensity_check:
                # 強度比チェックをスキップ
                coordinates = coordinates  # そのまま使用
            else:
                # 従来の強度比チェック
                for coord in coordinates:
                    y, x = coord
                    # 元の画像での強度をチェック（閾値チェックは既にpeak_local_maxで完了）
                    original_intensity = processed_image[y, x]
                    mean_intensity = np.mean(processed_image)
                    
                    # UIのMin intensity ratioを使用
                    min_intensity_ratio = getattr(self, 'min_intensity_ratio', 1.0)
                    
                    # 平均値のmin_intensity_ratio倍以上を要求
                    if original_intensity > mean_intensity * min_intensity_ratio:
                        valid_coordinates.append(coord)
                    else:
                        intensity_rejected += 1
                
                coordinates = np.array(valid_coordinates) if valid_coordinates else np.empty((0, 2), dtype=int)
        
        if len(coordinates) == 0:
            return []
        
        pass
        
        # 粒子リストの作成
        particles = []
        original_positions = []  # 補正前の位置を記録
        corrected_positions = []  # 補正後の位置を記録
        
        for i, (y, x) in enumerate(coordinates):
            # 元の整数座標を保存
            original_x, original_y = int(x), int(y)
            original_positions.append((original_x, original_y))
            
            #print(f"[DEBUG] Processing particle {i+1}: original position ({original_x}, {original_y})")
            #print(f"[DEBUG] Max position correction enabled: {self.max_position_correction}")
            
            # 最大値検索による位置補正（有効な場合のみ）
            if self.max_position_correction:
                refined_x, refined_y = self._refine_position_by_maximum(processed_image, original_x, original_y)
                corrected_positions.append((refined_x, refined_y))
            else:
                refined_x, refined_y = float(original_x), float(original_y)
                corrected_positions.append((refined_x, refined_y))
            
            if self.do_subpixel:
                # サブピクセル精度でのさらなる精密化
                subpixel_x, subpixel_y = self._refine_peak_position(processed_image, int(refined_x), int(refined_y))
                x, y = subpixel_x, subpixel_y
            else:
                x, y = refined_x, refined_y
            
            # 粒子オブジェクトを作成
            particle = Particle(
                frame=0,  # 現在のフレーム番号
                x=x,
                y=y,
                intensity=processed_image[int(y), int(x)],
                radius=self.radius,
                quality=smoothed_data[original_y, original_x]  # 平滑化データでの値を品質として使用
            )
            particles.append(particle)
        
        # 位置補正のサマリーを出力
        if len(original_positions) > 0:
            pass
        
        pass
        
        return particles

    def _calculate_particle_size(self, image: np.ndarray, x: int, y: int) -> float:
        """粒子サイズを計算（FWHMベース）"""
        try:
            # 検索範囲を設定
            search_radius = max(5, int(self.radius))
            y_min, y_max = max(0, y-search_radius), min(image.shape[0], y+search_radius+1)
            x_min, x_max = max(0, x-search_radius), min(image.shape[1], x+search_radius+1)
            
            # 周囲領域を抽出
            region = image[y_min:y_max, x_min:x_max]
            
            if region.size == 0:
                return 0.0
            
            # 中心位置を相対座標に変換
            center_y, center_x = y - y_min, x - x_min
            
            # 最大値とその位置を取得
            max_intensity = region[center_y, center_x]
            half_max = max_intensity / 2.0
            
            # FWHMを計算（X方向とY方向の平均）
            x_fwhm = 0
            y_fwhm = 0
            
            # X方向のFWHM
            for dx in range(1, min(center_x + 1, region.shape[1] - center_x)):
                if (region[center_y, center_x + dx] <= half_max or 
                    region[center_y, center_x - dx] <= half_max):
                    x_fwhm = dx * 2
                    break
            
            # Y方向のFWHM
            for dy in range(1, min(center_y + 1, region.shape[0] - center_y)):
                if (region[center_y + dy, center_x] <= half_max or 
                    region[center_y - dy, center_x] <= half_max):
                    y_fwhm = dy * 2
                    break
            
            # 平均FWHMを返す
            particle_size = (x_fwhm + y_fwhm) / 2.0
            return particle_size
            
        except Exception as e:
            return 0.0

    def _filter_particles_by_size(self, particles: List[Particle], image: np.ndarray) -> List[Particle]:
        """粒子サイズによるフィルタリング（非推奨 - _filter_particles_by_areaを使用）"""
        if not self.size_filter_enabled or not particles:
            return particles
        
        try:
            filtered = []
            # %表示の値を小数に変換（20% → 0.2）
            tolerance_fraction = self.size_tolerance / 100.0
            
            # スキャンサイズ情報を取得（nm単位）
            # グローバル変数から取得
            scan_size_x = getattr(gv, 'XScanSize', 1000.0)  # デフォルト値
            scan_size_y = getattr(gv, 'YScanSize', 1000.0)  # デフォルト値
            
            # ピクセルサイズを計算（nm/pixel）
            pixel_size_x = scan_size_x / image.shape[1]
            pixel_size_y = scan_size_y / image.shape[0]
            pixel_size = min(pixel_size_x, pixel_size_y)  # より小さい方を使用
            
            # selection_radiusをピクセル単位に変換
            selection_radius_pixels = self.selection_radius / pixel_size
            
            min_size = selection_radius_pixels * (1.0 - tolerance_fraction)
            max_size = selection_radius_pixels * (1.0 + tolerance_fraction)
            
            for i, particle in enumerate(particles):
                # 粒子サイズを計算（ピクセル単位）
                size = self._calculate_particle_size(image, int(particle.x), int(particle.y))
                
                if min_size <= size <= max_size:
                    filtered.append(particle)
 
         
            return filtered
            
        except Exception as e:
            print(f"[ERROR] Size filtering failed: {e}")
            return particles

    def _filter_particles_by_watershed_area(self, particles: List[Particle], image: np.ndarray) -> List[Particle]:
        """Watershed領域面積による粒子フィルタリング"""
        
        if not self.size_filter_enabled or not particles:
            return particles
        try:
            from skimage.segmentation import watershed
            from skimage import filters
            from scipy import ndimage
            
            # 1. データを適度に平滑化（ノイズ除去）
            sigma = 0.5
            smoothed_data = ndimage.gaussian_filter(image, sigma=sigma)
            
            # 2. より適切な勾配計算（particle_analysis方式）
            from skimage.filters import sobel
            gradient = sobel(smoothed_data)
            
            # 3. ピークマーカーを改善
            improved_markers = np.zeros_like(smoothed_data, dtype=int)
            
            for i, p in enumerate(particles):
                y, x = int(p.y), int(p.x)
                if 0 <= y < improved_markers.shape[0] and 0 <= x < improved_markers.shape[1]:
                    # ピーク周囲の小さな領域を作成（3x3ピクセル）
                    y_min = max(0, y-1)
                    y_max = min(improved_markers.shape[0], y+2)
                    x_min = max(0, x-1)
                    x_max = min(improved_markers.shape[1], x+2)
                    
                    improved_markers[y_min:y_max, x_min:x_max] = i + 1
            
            # 4. より適切なマスク条件（particle_analysis方式）
            threshold = np.percentile(smoothed_data, 80)  # 上位20%の領域（particle_analysisと同じ）
            mask = smoothed_data > threshold
            
            # 5. Watershed実行（particle_analysis方式のパラメータ）
            compactness = getattr(self, 'watershed_compactness', 0.2)  # particle_analysisのデフォルト値
            
            labels = watershed(gradient, improved_markers, 
                             mask=mask, 
                             compactness=compactness)
            
            # 6. 結果が空でないかチェック（particle_analysis方式のフォールバック）
            if np.sum(labels > 0) == 0:
                # パラメータを緩和して再試行
                threshold_relaxed = np.percentile(smoothed_data, 70)  # particle_analysisと同じ
                mask_relaxed = smoothed_data > threshold_relaxed
                
                compactness_relaxed = 0.1  # particle_analysisと同じ
                labels = watershed(gradient, improved_markers, 
                                 mask=mask_relaxed, 
                                 compactness=compactness_relaxed)
            
            # 6. 面積でフィルタ
            filtered = []
            # %表示の値を小数に変換（20% → 0.2）
            tolerance_fraction = self.size_tolerance / 100.0
            
            # selection_radiusをピクセル単位に変換
            # スキャンサイズ情報を取得（nm単位）
            # グローバル変数から取得
            scan_size_x = getattr(gv, 'XScanSize', 1000.0)  # デフォルト値
            scan_size_y = getattr(gv, 'YScanSize', 1000.0)  # デフォルト値
            
            # ピクセルサイズを計算（nm/pixel）
            pixel_size_x = scan_size_x / image.shape[1]
            pixel_size_y = scan_size_y / image.shape[0]
            pixel_size = min(pixel_size_x, pixel_size_y)  # より小さい方を使用
            
            # selection_radiusをピクセル単位に変換
            selection_radius_pixels = self.selection_radius / pixel_size
            
            # 理想的な面積をピクセル単位で計算
            target_area = np.pi * selection_radius_pixels**2
            
            # toleranceが100%を超える場合の処理
            if tolerance_fraction > 1.0:
                min_area = 0  # 最小面積を0に設定
                max_area = target_area * (1.0 + tolerance_fraction)
                #print(f"[DEBUG] Tolerance > 100%: min_area set to 0, max_area = {max_area:.1f}px²")
            else:
                min_area = target_area * (1.0 - tolerance_fraction)
                max_area = target_area * (1.0 + tolerance_fraction)
            
            #print(f"[DEBUG] Watershed area filter: selection_radius={self.selection_radius}nm -> {selection_radius_pixels:.1f}px")
            #print(f"[DEBUG] Watershed area filter: target_area={target_area:.1f}px², min_area={min_area:.1f}px², max_area={max_area:.1f}px² (tolerance: {self.size_tolerance}%)")
            #print(f"[DEBUG] Watershed area filter: scan_size={scan_size_x}x{scan_size_y}nm, pixel_size={pixel_size:.6f}nm/pixel")
            
            for i, p in enumerate(particles):
                label_id = i + 1
                #print(f"[DEBUG] Watershed filtering: checking particle {i+1} with label_id={label_id}")
                #print(f"[DEBUG] Label {label_id} in labels: {label_id in labels}")
                
                if label_id in labels:
                    area = np.sum(labels == label_id)
                    # 円を仮定して半径を計算（ピクセル単位）
                    calculated_radius_pixels = np.sqrt(area / np.pi)
                    # 物理単位（nm）に変換
                    calculated_radius_nm = calculated_radius_pixels * pixel_size
                    
                    #print(f"[DEBUG] Particle {i+1} area: {area}px² -> radius: {calculated_radius_pixels:.1f}px ({calculated_radius_nm:.1f}nm)")
                    #print(f"[DEBUG]   Target: {target_area:.1f}px² -> {selection_radius_pixels:.1f}px ({self.selection_radius:.1f}nm)")
                    #print(f"[DEBUG]   Range: {min_area:.1f}-{max_area:.1f}px² (±{self.size_tolerance}%)")
                    
                    if min_area <= area <= max_area:
                        filtered.append(p)
                        #print(f"[DEBUG]   ACCEPTED")


            #print(f"[DEBUG] Watershed area filtering: {len(particles)} -> {len(filtered)} particles")
            return filtered
            
        except Exception as e:
            print(f"[ERROR] Watershed area filtering failed: {e}")
            import traceback
            traceback.print_exc()
            return particles

    def _refine_position_by_maximum(self, image: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        """検出された位置の周囲で最大値を検索して位置を補正"""
        try:
            # 検索範囲を設定（半径の範囲内）
            search_radius = max(2, int(self.radius))
            y_min, y_max = max(0, y-search_radius), min(image.shape[0], y+search_radius+1)
            x_min, x_max = max(0, x-search_radius), min(image.shape[1], x+search_radius+1)
            
            # 周囲領域を抽出
            region = image[y_min:y_max, x_min:x_max]
            
            if region.size == 0:
                return float(x), float(y)
            
            # 最大値の位置を検索
            max_idx = np.unravel_index(np.argmax(region), region.shape)
            refined_y = y_min + max_idx[0]
            refined_x = x_min + max_idx[1]
            
            # 元の位置からの距離をチェック
            distance = np.sqrt((refined_x - x)**2 + (refined_y - y)**2)
            max_allowed_distance = self.radius
            
            if distance <= max_allowed_distance:
                #print(f"[DEBUG] Position refinement by maximum: ({x}, {y}) -> ({refined_x:.1f}, {refined_y:.1f}), distance={distance:.1f}")
                return float(refined_x), float(refined_y)
            else:
                #print(f"[DEBUG] Position refinement rejected: distance {distance:.1f} > {max_allowed_distance}")
                return float(x), float(y)
                
        except Exception as e:
            print(f"Warning: Position refinement by maximum failed: {e}")
            return float(x), float(y)

    def _filter_particles_by_area(self, particles: List[Particle], image: np.ndarray) -> List[Particle]:
        """粒子領域によるフィルタリング（FWHMまたはWatershed）"""
        #print(f"[DEBUG] ===== _filter_particles_by_area ENTRY =====")
        #print(f"[DEBUG] _filter_particles_by_area called with {len(particles)} particles")
        #print(f"[DEBUG] size_filter_enabled: {self.size_filter_enabled}")
        #print(f"[DEBUG] area_method: {self.area_method}")
        #print(f"[DEBUG] selection_radius: {getattr(self, 'selection_radius', 'NOT SET')}")
        #print(f"[DEBUG] size_tolerance: {getattr(self, 'size_tolerance', 'NOT SET')}")
        
        if not self.size_filter_enabled or not particles:
            #print(f"[DEBUG] Skipping filtering: enabled={self.size_filter_enabled}, particles={len(particles)}")
            return particles
        
        try:
            #print(f"[DEBUG] Area method check: '{self.area_method}' (type: {type(self.area_method)})")
            if self.area_method == "FWHM":
                #print(f"[DEBUG] Using FWHM filtering")
                filtered_particles = self._filter_particles_by_fwhm(particles, image)
                #print(f"[DEBUG] FWHM filtering result: {len(particles)} -> {len(filtered_particles)} particles")
                return filtered_particles
            elif self.area_method == "Watershed":
                #print(f"[DEBUG] Using Watershed filtering")
                filtered_particles = self._filter_particles_by_watershed_area(particles, image)
                #print(f"[DEBUG] Watershed filtering result: {len(particles)} -> {len(filtered_particles)} particles")
                return filtered_particles
            else:
                #print(f"[WARNING] Unknown area method: '{self.area_method}'")
                return particles
                
        except Exception as e:
            print(f"[ERROR] Area filtering failed: {e}")
            import traceback
            traceback.print_exc()
            return particles
        
        #print(f"[DEBUG] ===== _filter_particles_by_area EXIT =====")
    
    def _filter_particles_by_fwhm(self, particles: List[Particle], image: np.ndarray) -> List[Particle]:
        """FWHMによる粒子サイズフィルタリング"""
        try:
            filtered = []
            # %表示の値を小数に変換（20% → 0.2）
            tolerance_fraction = self.size_tolerance / 100.0
            
            # スキャンサイズ情報を取得（nm単位）
            # グローバル変数から取得
            scan_size_x = getattr(gv, 'XScanSize', 1000.0)  # デフォルト値
            scan_size_y = getattr(gv, 'YScanSize', 1000.0)  # デフォルト値
            
            # ピクセルサイズを計算（nm/pixel）
            pixel_size_x = scan_size_x / image.shape[1]
            pixel_size_y = scan_size_y / image.shape[0]
            pixel_size = min(pixel_size_x, pixel_size_y)  # より小さい方を使用
            
            # selection_radiusをピクセル単位に変換
            selection_radius_pixels = self.selection_radius / pixel_size
            
            min_size = selection_radius_pixels * (1.0 - tolerance_fraction)
            max_size = selection_radius_pixels * (1.0 + tolerance_fraction)
            
            #print(f"[DEBUG] FWHM filter: selection_radius={self.selection_radius}nm -> {selection_radius_pixels:.1f}px")
            #print(f"[DEBUG] FWHM filter: min_size={min_size:.1f}px, max_size={max_size:.1f}px (tolerance: {self.size_tolerance}%)")
            #print(f"[DEBUG] FWHM filter: scan_size={scan_size_x}x{scan_size_y}nm, pixel_size={pixel_size:.6f}nm/pixel")
            
            for i, particle in enumerate(particles):
                # 粒子サイズを計算（ピクセル単位）
                size = self._calculate_particle_size(image, int(particle.x), int(particle.y))
                # 物理単位（nm）に変換
                size_nm = size * pixel_size
                
                #print(f"[DEBUG] Particle {i+1} FWHM size: {size:.1f}px ({size_nm:.1f}nm)")
                #print(f"[DEBUG]   Target: {selection_radius_pixels:.1f}px ({self.selection_radius:.1f}nm)")
                #print(f"[DEBUG]   Range: {min_size:.1f}-{max_size:.1f}px (±{self.size_tolerance}%)")
                
                if min_size <= size <= max_size:
                    filtered.append(particle)
                    #print(f"[DEBUG]   ACCEPTED")

            
            #print(f"[DEBUG] FWHM filtering: {len(particles)} -> {len(filtered)} particles")
            return filtered
            
        except Exception as e:
            print(f"[ERROR] FWHM filtering failed: {e}")
            return particles

class ParticleLinker:
    """粒子リンククラス"""
    
    def __init__(self):
        self.max_distance = 5.0
        self.max_frame_gap = 2
        self.min_track_length = 3
    
    def link_particles(self, particles_by_frame: Dict[int, List[Particle]]) -> List[Track]:
        """粒子のリンク処理"""
        if not particles_by_frame:
            return []
        
        tracks = []
        next_track_id = 1
        
        # 各フレームの粒子を処理
        for frame_idx in sorted(particles_by_frame.keys()):
            current_particles = particles_by_frame[frame_idx]
            
            # 既存のトラックを更新
            for track in tracks:
                if track.end_frame >= frame_idx - self.max_frame_gap:
                    # 最も近い粒子を見つける
                    best_particle = None
                    best_distance = float('inf')
                    
                    for particle in current_particles:
                        if particle.track_id is not None:
                            continue  # 既に割り当て済み
                        
                        # 最後の粒子との距離を計算
                        last_particle = track.particles[-1]
                        distance = np.sqrt((particle.x - last_particle.x)**2 + 
                                        (particle.y - last_particle.y)**2)
                        
                        if distance <= self.max_distance and distance < best_distance:
                            best_particle = particle
                            best_distance = distance
                    
                    if best_particle is not None:
                        best_particle.track_id = track.track_id
                        track.particles.append(best_particle)
                        track.end_frame = frame_idx
                        track.duration = track.end_frame - track.start_frame + 1
                        current_particles.remove(best_particle)
            
            # 新しいトラックを開始
            for particle in current_particles:
                if particle.track_id is None:
                    particle.track_id = next_track_id
                    new_track = Track(
                        track_id=next_track_id,
                        particles=[particle],
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                        duration=1
                    )
                    tracks.append(new_track)
                    next_track_id += 1
        
        # 短すぎるトラックを除去
        tracks = [track for track in tracks if len(track.particles) >= self.min_track_length]
        
        # 統計を計算
        for track in tracks:
            self._calculate_track_statistics(track)
        
        return tracks
    
    def _calculate_track_statistics(self, track: Track):
        """軌跡の統計を計算"""
        if len(track.particles) < 2:
            return
        
        # 平均速度と変位を計算
        total_displacement = 0
        total_distance = 0
        
        for i in range(1, len(track.particles)):
            prev_particle = track.particles[i-1]
            curr_particle = track.particles[i]
            
            dx = curr_particle.x - prev_particle.x
            dy = curr_particle.y - prev_particle.y
            distance = np.sqrt(dx**2 + dy**2)
            
            total_distance += distance
        
        if track.duration > 1:
            track.mean_velocity = total_distance / (track.duration - 1)
        
        # 全体的な変位
        first_particle = track.particles[0]
        last_particle = track.particles[-1]
        track.mean_displacement = np.sqrt((last_particle.x - first_particle.x)**2 + 
                                        (last_particle.y - first_particle.y)**2)

class TrackingEngine:
    """トラッキングエンジン"""
    
    def __init__(self):
        self.detector = SpotDetector()
        self.linker = ParticleLinker()
        self.tracks = []
        self.statistics = {}
    
    def process_image_sequence(self, frames: List[np.ndarray], 
                             detection_params: Dict = None,
                             tracking_params: Dict = None) -> Dict:
        """画像シーケンスの処理"""
        
        # パラメータの設定
        if detection_params:
            for key, value in detection_params.items():
                if hasattr(self.detector, key):
                    setattr(self.detector, key, value)
        
        if tracking_params:
            for key, value in tracking_params.items():
                if hasattr(self.linker, key):
                    setattr(self.linker, key, value)
        
        # 粒子検出
        particles_by_frame = {}
        for i, frame in enumerate(frames):
            particles = self.detector.detect_particles(frame)
            for particle in particles:
                particle.frame = i
            particles_by_frame[i] = particles
        
        # 粒子リンク
        self.tracks = self.linker.link_particles(particles_by_frame)
        
        # 統計の計算
        self._calculate_statistics()
        
        return {
            'tracks': self.tracks,
            'particles_by_frame': particles_by_frame,
            'statistics': self.statistics
        }
    
    def _calculate_statistics(self):
        """統計の計算"""
        if not self.tracks:
            self.statistics = {}
            return
        
        total_tracks = len(self.tracks)
        total_particles = sum(len(track.particles) for track in self.tracks)
        
        velocities = [track.mean_velocity for track in self.tracks]
        displacements = [track.mean_displacement for track in self.tracks]
        
        self.statistics = {
            'total_tracks': total_tracks,
            'total_particles': total_particles,
            'mean_velocity': np.mean(velocities) if velocities else 0.0,
            'mean_displacement': np.mean(displacements) if displacements else 0.0,
            'max_velocity': np.max(velocities) if velocities else 0.0,
            'max_displacement': np.max(displacements) if displacements else 0.0
        }

class ParticleTrackingWindow(QtWidgets.QWidget):
    """粒子トラッキングウィンドウ"""
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.ui_initialized = False  # <-- ★ 1. この行を追加
        self.main_window = main_window
        self.setWindowTitle("Particle Tracking")
        self.setMinimumSize(600, 400)  # 最小サイズを小さく
        
        # matplotlibを事前に初期化
        if not _import_matplotlib():
            print("[ERROR] Failed to initialize matplotlib in __init__")
        
        # データ
        self.frames = []
        self.current_frame_index = 0
        self.current_data = None
        self.particles_by_frame = {}
        self.tracks = []
        self.active_tracks = [] # <--- この行を追加
        self.next_track_id = 0 # <--- この行を追加
        self.statistics = {}
        
        # トラッキング制御
        self.tracking_cancelled = False
        
        # 検出制御
        self.detection_cancelled = False
        
        # トラッキングエンジン
        self.tracking_engine = TrackingEngine()
        
        # UI設定
        self.setupUI()
        self.loadWindowSettings()
        
        # データ初期化
        self.ui_initialized = True   # <-- ★ 2. この行を追加
        QtCore.QTimer.singleShot(100, self.initializeData)
    
    def setupUI(self):
        """UIの設定"""
        top_layout = QtWidgets.QVBoxLayout(self)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        help_menu = menu_bar.addMenu("Help" if QtCore.QLocale().language() != QtCore.QLocale.Japanese else "ヘルプ")
        manual_action = help_menu.addAction("Manual" if QtCore.QLocale().language() != QtCore.QLocale.Japanese else "マニュアル")
        manual_action.triggered.connect(self.showHelpDialog)
        top_layout.addWidget(menu_bar)
        
        content_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 左パネル（パラメータ設定）- スクロールエリアでラップ
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(250)  # 最小幅を小さく
        left_scroll.setMaximumWidth(450)  # 最大幅を大きく
        left_panel = self.createLeftPanel()
        left_scroll.setWidget(left_panel)
        layout.addWidget(left_scroll)
        
        # 中央パネル（画像表示）- スクロールエリアでラップ
        center_scroll = QtWidgets.QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        center_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        center_scroll.setMinimumWidth(200)  # 最小幅を設定
        center_panel = self.createCenterPanel()
        center_scroll.setWidget(center_panel)
        layout.addWidget(center_scroll, 1)
        
        # 右パネル（結果表示）- スクロールエリアでラップ
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        right_scroll.setMinimumWidth(200)  # 最小幅を小さく
        right_scroll.setMaximumWidth(500)  # 最大幅を大きく
        right_panel = self.createRightPanel()
        right_scroll.setWidget(right_panel)
        layout.addWidget(right_scroll)
        
        top_layout.addWidget(content_widget)
        
        # ウィンドウがアクティブになった時にキャンバスにフォーカスを設定
        self.focusInEvent = self.onFocusInEvent
        
        # ウィンドウレベルでのキーボードイベント処理を追加
        self.keyPressEvent = self.onWindowKeyPress
    
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

        _BTN_SELECTED = "QPushButton { background-color: #007aff; color: white; font-weight: bold; }"
        _BTN_NORMAL = "QPushButton { background-color: #e5e5e5; color: black; }"

        def set_lang(use_ja):
            btn_ja.setChecked(use_ja)
            btn_en.setChecked(not use_ja)
            btn_ja.setStyleSheet(_BTN_SELECTED if use_ja else _BTN_NORMAL)
            btn_en.setStyleSheet(_BTN_SELECTED if not use_ja else _BTN_NORMAL)
            if use_ja:
                browser.setHtml("<html><body>" + HELP_HTML_JA.strip() + "</body></html>")
                dialog.setWindowTitle("粒子トラッキング - マニュアル")
                close_btn.setText("閉じる")
            else:
                browser.setHtml("<html><body>" + HELP_HTML_EN.strip() + "</body></html>")
                dialog.setWindowTitle("Particle Tracking - Manual")
                close_btn.setText("Close")

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        layout_dlg.addWidget(browser)
        layout_dlg.addWidget(close_btn)
        set_lang(False)  # デフォルトは英語
        dialog.exec_()
    
    def createLeftPanel(self):
        """左パネルの作成"""
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(250)  # 最小幅を小さく
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # 直接レイアウトにコンテンツを追加（スクロールエリアは外側で処理）
        content_layout = layout
        
        # 前処理グループ（AFMデータ用に簡素化）
        preprocess_group = QtWidgets.QGroupBox("Preprocessing")
        preprocess_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        preprocess_layout = QtWidgets.QVBoxLayout()
        preprocess_layout.setContentsMargins(10, 10, 10, 10)
        preprocess_layout.setSpacing(8)
        
        # 説明ラベル
        info_label = QtWidgets.QLabel("Note: AFM data is already preprocessed with Auto Leveling.")
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        preprocess_layout.addWidget(info_label)
        
        # Smoothing
        smooth_row = QtWidgets.QHBoxLayout()
        smooth_label = QtWidgets.QLabel("Smoothing:")
        smooth_label.setToolTip("Smoothing method for noise reduction / ノイズ除去のための平滑化方法")
        smooth_row.addWidget(smooth_label)
        self.smooth_combo = QtWidgets.QComboBox()
        self.smooth_combo.addItems(["None", "Gaussian", "Median"])
        self.smooth_combo.setMinimumWidth(120)
        self.smooth_combo.setMaximumWidth(150)
        self.smooth_combo.setToolTip("None: No smoothing / 平滑化なし\nGaussian: Gaussian filter / ガウシアンフィルター\nMedian: Median filter / メディアンフィルター")
        self.smooth_combo.currentTextChanged.connect(self.onSmoothingMethodChanged)
        smooth_row.addWidget(self.smooth_combo)
        smooth_row.addStretch()
        preprocess_layout.addLayout(smooth_row)
        
        # Smoothing parameter
        self.smooth_param_row = QtWidgets.QHBoxLayout()
        self.smooth_param_label = QtWidgets.QLabel("Gaussian Sigma:")
        self.smooth_param_label.setToolTip("Standard deviation of Gaussian filter / ガウシアンフィルターの標準偏差")
        self.smooth_param_spin = QtWidgets.QDoubleSpinBox()
        self.smooth_param_spin.setRange(0.1, 3.0)
        self.smooth_param_spin.setValue(1.0)
        self.smooth_param_spin.setSingleStep(0.1)
        self.smooth_param_spin.setDecimals(1)
        self.smooth_param_spin.setMinimumWidth(60)
        self.smooth_param_spin.setMaximumWidth(80)
        self.smooth_param_spin.setToolTip("Larger values create more smoothing / 大きな値ほど平滑化が強くなる")
        self.smooth_param_spin.valueChanged.connect(self.onSmoothingParameterChanged)
        self.smooth_param_row.addWidget(self.smooth_param_label)
        self.smooth_param_row.addWidget(self.smooth_param_spin)
        self.smooth_param_row.addStretch()
        # 初期状態は非表示
        self.smooth_param_label.setVisible(False)
        self.smooth_param_spin.setVisible(False)
        preprocess_layout.addLayout(self.smooth_param_row)
        
        # 初期状態で適切な値を設定（パラメータ入力部を表示するため）
        self.smooth_combo.setCurrentText("Gaussian")
        
        # 初期状態でUIを更新
        self.updatePreprocessingUI()
        
        # preprocessingグループのレイアウトを設定
        preprocess_group.setLayout(preprocess_layout)
        
        content_layout.addWidget(preprocess_group)
        
        # スペーサーを追加してグループ間の間隔を確保
        content_layout.addSpacing(10)
        
        # 検出パラメータ
        detection_group = QtWidgets.QGroupBox("Detection Parameters")
        detection_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        detection_layout = QtWidgets.QGridLayout(detection_group)
        detection_layout.setContentsMargins(10, 10, 10, 10)
        detection_layout.setSpacing(8)
        
        # 検出方法選択
        detection_method_label = QtWidgets.QLabel("Detection Method:")
        detection_method_label.setToolTip("Particle detection algorithm / 粒子検出アルゴリズム")
        detection_layout.addWidget(detection_method_label, 0, 0)
        self.detection_method_combo = QtWidgets.QComboBox()
        self.detection_method_combo.addItems(["LoG", "Local Max", "DoG", "Template Matching"])
        self.detection_method_combo.setCurrentText("LoG")
        self.detection_method_combo.setToolTip("LoG: Laplacian of Gaussian - detects blobs / ガウシアンのラプラシアン - ブロブ検出\nLocal Max: Gaussian filter + peak detection / ガウシアンフィルター + ピーク検出\nDoG: Difference of Gaussian - similar to LoG / ガウシアンの差分 - LoGと類似\nTemplate Matching: Matches template pattern / テンプレートマッチング")
        self.detection_method_combo.currentTextChanged.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.detection_method_combo, 0, 1)
        
        # 粒子半径
        radius_label = QtWidgets.QLabel("Radius (nm):")
        radius_label.setToolTip("Expected particle radius in nanometers / 期待される粒子半径（ナノメートル）")
        detection_layout.addWidget(radius_label, 1, 0)
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 50.0)
        self.radius_spin.setValue(5.0)
        self.radius_spin.setSingleStep(0.1)
        self.radius_spin.setMinimumWidth(60)
        self.radius_spin.setMaximumWidth(80)
        self.radius_spin.setToolTip("Used to determine filter size and search radius / フィルターサイズと検索半径の決定に使用")
        self.radius_spin.valueChanged.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.radius_spin, 1, 1)
        
        # 最小距離
        min_distance_label = QtWidgets.QLabel("Min distance:")
        min_distance_label.setToolTip("Minimum distance between detected particles in pixels / 検出された粒子間の最小距離（ピクセル）")
        detection_layout.addWidget(min_distance_label, 2, 0)
        self.min_distance_spin = QtWidgets.QSpinBox()
        self.min_distance_spin.setRange(1, 50)
        self.min_distance_spin.setValue(5)
        self.min_distance_spin.setMinimumWidth(60)
        self.min_distance_spin.setMaximumWidth(80)
        self.min_distance_spin.setToolTip("Prevents detection of multiple peaks too close together / 近すぎる複数のピークの検出を防ぐ")
        self.min_distance_spin.valueChanged.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.min_distance_spin, 2, 1)
        
        # 閾値
        threshold_label = QtWidgets.QLabel("Threshold factor (0.1-1.0):")
        threshold_label.setToolTip("Threshold factor for Otsu method / Otsu法の閾値係数")
        detection_layout.addWidget(threshold_label, 3, 0)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 1.0)
        self.threshold_spin.setValue(1.0)  # TrackMateのように1.0をデフォルトに設定（Otsu係数）
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setMinimumWidth(60)
        self.threshold_spin.setMaximumWidth(80)
        self.threshold_spin.setToolTip("Lower values detect more particles, higher values detect fewer particles / 小さい値ほど多くの粒子を検出、大きい値ほど少ない粒子を検出")
        self.threshold_spin.valueChanged.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.threshold_spin, 3, 1)
        
        # サブピクセル補正
        self.subpixel_check = QtWidgets.QCheckBox("Sub-pixel correction")
        self.subpixel_check.setChecked(True)
        self.subpixel_check.setToolTip("Refine particle positions to sub-pixel accuracy using Gaussian fitting / ガウシアンフィッティングによるサブピクセル精度での位置補正")
        self.subpixel_check.toggled.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.subpixel_check, 4, 0, 1, 2)
        
        # 最大値位置補正
        self.max_pos_corr_check = QtWidgets.QCheckBox("Max Pos. Corr.")
        self.max_pos_corr_check.setChecked(True)
        self.max_pos_corr_check.setToolTip("Refine particle positions by searching for maximum intensity within radius / 半径内で最大強度を検索して粒子位置を補正")
        self.max_pos_corr_check.toggled.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.max_pos_corr_check, 5, 0, 1, 2)
        
        # 最小強度閾値
        min_intensity_label = QtWidgets.QLabel("Min intensity ratio:")
        min_intensity_label.setToolTip("Minimum intensity ratio compared to image mean / 画像平均に対する最小強度比")
        detection_layout.addWidget(min_intensity_label, 6, 0)
        self.min_intensity_spin = QtWidgets.QDoubleSpinBox()
        self.min_intensity_spin.setRange(0.1, 1.0)
        self.min_intensity_spin.setValue(0.8)
        self.min_intensity_spin.setSingleStep(0.05)
        self.min_intensity_spin.setMinimumWidth(60)
        self.min_intensity_spin.setMaximumWidth(80)
        self.min_intensity_spin.setToolTip("Higher values require brighter particles / 大きい値ほど明るい粒子が必要")
        self.min_intensity_spin.valueChanged.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.min_intensity_spin, 6, 1)
        
        # 粒子サイズフィルタリング
        size_filter_label = QtWidgets.QLabel("Size filter:")
        size_filter_label.setToolTip("Filter particles by size relative to selection radius / 選択半径に対するサイズで粒子をフィルタリング")
        detection_layout.addWidget(size_filter_label, 7, 0)
        self.size_filter_check = QtWidgets.QCheckBox("Enable")
        self.size_filter_check.setChecked(False)  # デフォルトでオフ
        self.size_filter_check.setToolTip("Enable/disable particle size filtering / 粒子サイズフィルタリングの有効/無効")
        self.size_filter_check.toggled.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.size_filter_check, 7, 1)
        
        # 領域決定方法選択
        area_method_label = QtWidgets.QLabel("Area method:")
        area_method_label.setToolTip("Method to determine particle area for filtering / フィルタリング用の粒子領域決定方法")
        detection_layout.addWidget(area_method_label, 8, 0)
        self.area_method_combo = QtWidgets.QComboBox()
        self.area_method_combo.addItems(["FWHM", "Watershed"])
        self.area_method_combo.setCurrentText("FWHM")
        self.area_method_combo.setToolTip("FWHM: Full Width at Half Maximum / 半値全幅\nWatershed: Watershed segmentation / Watershed分割")
        self.area_method_combo.currentTextChanged.connect(self.onAreaMethodChanged)
        detection_layout.addWidget(self.area_method_combo, 8, 1)
        
        # 選択半径（フィルタリング用）
        selection_radius_label = QtWidgets.QLabel("Selection radius (nm):")
        selection_radius_label.setToolTip("Radius used for particle filtering / 粒子フィルタリングに使用する半径（ナノメートル）")
        detection_layout.addWidget(selection_radius_label, 9, 0)
        self.selection_radius_spin = QtWidgets.QDoubleSpinBox()
        self.selection_radius_spin.setRange(0.1, 50.0)
        self.selection_radius_spin.setValue(5.0)
        self.selection_radius_spin.setSingleStep(0.1)
        self.selection_radius_spin.setMinimumWidth(60)
        self.selection_radius_spin.setMaximumWidth(80)
        self.selection_radius_spin.setToolTip("Radius used for size and watershed filtering / サイズとWatershedフィルタリングに使用する半径")
        self.selection_radius_spin.valueChanged.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.selection_radius_spin, 9, 1)
        
                    # サイズ許容範囲
        size_tolerance_label = QtWidgets.QLabel("Size tolerance (%):")
        size_tolerance_label.setToolTip("Acceptable size range as percentage of selection radius / 選択半径に対する許容サイズ範囲（%）")
        detection_layout.addWidget(size_tolerance_label, 10, 0)
        self.size_tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.size_tolerance_spin.setRange(5, 200)  # より広い範囲
        self.size_tolerance_spin.setValue(50)      # より緩いデフォルト値
        self.size_tolerance_spin.setSingleStep(5)
        self.size_tolerance_spin.setSuffix(" %")
        self.size_tolerance_spin.setMinimumWidth(60)
        self.size_tolerance_spin.setMaximumWidth(80)
        self.size_tolerance_spin.setToolTip("20% = ±20% of selection radius / 20% = 選択半径の±20%")
        self.size_tolerance_spin.valueChanged.connect(self.onDetectionParameterChanged)
        detection_layout.addWidget(self.size_tolerance_spin, 10, 1)
        
        # Watershedパラメータ（Area methodがWatershedの時のみ表示）
        self.watershed_compactness_label = QtWidgets.QLabel("Watershed compactness:")
        self.watershed_compactness_label.setToolTip("Compactness parameter for watershed separation (smaller = better separation) / Watershed分離のコンパクト性パラメータ（小さいほど分離が良い）")
        detection_layout.addWidget(self.watershed_compactness_label, 11, 0)
        self.watershed_compactness_spin = QtWidgets.QDoubleSpinBox()
        self.watershed_compactness_spin.setRange(0.01, 1.0)
        self.watershed_compactness_spin.setValue(0.1)  # より小さいデフォルト値
        self.watershed_compactness_spin.setSingleStep(0.01)
        self.watershed_compactness_spin.setDecimals(2)
        self.watershed_compactness_spin.setMinimumWidth(60)
        self.watershed_compactness_spin.setMaximumWidth(80)
        self.watershed_compactness_spin.setToolTip("Smaller values improve separation of close particles / 小さい値ほど近接した粒子の分離が改善される")
        #print(f"[DEBUG] Setting up watershed_compactness_spin signal connection")
        # 既存の接続を切断
        try:
            self.watershed_compactness_spin.valueChanged.disconnect()
            #print(f"[DEBUG] Disconnected existing watershed_compactness_spin signals")
        except:
            pass
           #print(f"[DEBUG] No existing watershed_compactness_spin signals to disconnect")
        
        # 新しい接続を設定
        self.watershed_compactness_spin.valueChanged.connect(self.onWatershedCompactnessChanged)
        #print(f"[DEBUG] watershed_compactness_spin signal connection established")
        
        # 接続をテスト
        #print(f"[DEBUG] Testing signal connection...")
        #print(f"[DEBUG] watershed_compactness_spin.receivers(): {self.watershed_compactness_spin.receivers(self.watershed_compactness_spin.valueChanged)}")
        
        # 手動でシグナルをテスト
        #print(f"[DEBUG] Manual signal test...")
        self.watershed_compactness_spin.valueChanged.emit(self.watershed_compactness_spin.value())
        detection_layout.addWidget(self.watershed_compactness_spin, 11, 1)
        
        # Watershedマスク閾値パラメータ
        self.watershed_threshold_label = QtWidgets.QLabel("Watershed threshold (%):")
        self.watershed_threshold_label.setToolTip("Threshold percentile for watershed mask (higher = smaller regions) / Watershedマスクの閾値パーセンタイル（高いほど領域が小さい）")
        detection_layout.addWidget(self.watershed_threshold_label, 12, 0)
        self.watershed_threshold_spin = QtWidgets.QSpinBox()
        self.watershed_threshold_spin.setRange(50, 99)
        self.watershed_threshold_spin.setValue(70)  # より低いデフォルト値
        self.watershed_threshold_spin.setSingleStep(1)
        self.watershed_threshold_spin.setMinimumWidth(60)
        self.watershed_threshold_spin.setMaximumWidth(80)
        self.watershed_threshold_spin.setToolTip("Higher values create smaller watershed regions / 高い値ほどWatershed領域が小さくなる")
        self.watershed_threshold_spin.valueChanged.connect(self.onWatershedThresholdChanged)
        detection_layout.addWidget(self.watershed_threshold_spin, 12, 1)
        
        # 初期状態では非表示
        self.watershed_compactness_label.setVisible(False)
        self.watershed_compactness_spin.setVisible(False)
        self.watershed_threshold_label.setVisible(False)
        self.watershed_threshold_spin.setVisible(False)
        
        content_layout.addWidget(detection_group)
        
        # スペーサーを追加
        content_layout.addSpacing(10)
        
        # トラッキングパラメータ
        tracking_group = QtWidgets.QGroupBox("Tracking Parameters")
        tracking_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        tracking_layout = QtWidgets.QVBoxLayout(tracking_group)

        # アルゴリズム選択
        algo_layout = QtWidgets.QHBoxLayout()
        algo_label = QtWidgets.QLabel("Algorithm:")
        algo_label.setToolTip("Particle tracking algorithm / 粒子トラッキングアルゴリズム")
        self.tracking_algo_combo = QtWidgets.QComboBox()
        # 新しい選択肢に変更
        self.tracking_algo_combo.addItems([
            "Trackpy (Simple Linker)", 
            "Simple LAP Tracker (SciPy)",
            "Kalman Filter (filterpy)"
        ])
        self.tracking_algo_combo.setToolTip(
            "Trackpy: Simple nearest neighbor linking / 単純な最近傍リンク\n"
            "LAP Tracker: Linear Assignment Problem solver / 線形割り当て問題ソルバー\n"
            "Kalman Filter: Predictive tracking with noise filtering / ノイズフィルタリング付き予測トラッキング"
        )
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.tracking_algo_combo)
        tracking_layout.addLayout(algo_layout)
        
        # --- 共通パラメータウィジェット ---
        params_widget = QtWidgets.QWidget()
        params_layout = QtWidgets.QFormLayout(params_widget)
        params_layout.setContentsMargins(0, 5, 0, 0)
        
        self.max_distance_spin = QtWidgets.QDoubleSpinBox()
        self.max_distance_spin.setRange(0.1, 200.0); self.max_distance_spin.setValue(15.0)
        self.max_distance_spin.setToolTip("Maximum distance (nm) a particle can move between frames / フレーム間で粒子が移動できる最大距離（ナノメートル）")
        
        self.max_frame_gap_spin = QtWidgets.QSpinBox()
        self.max_frame_gap_spin.setRange(0, 10); self.max_frame_gap_spin.setValue(2)
        self.max_frame_gap_spin.setToolTip("How many frames a particle can disappear and still be linked / 粒子が消失してもリンクできるフレーム数")

        self.min_track_length_spin = QtWidgets.QSpinBox()
        self.min_track_length_spin.setRange(2, 100); self.min_track_length_spin.setValue(5)
        self.min_track_length_spin.setToolTip("Minimum number of frames for a valid track / 有効な軌跡に必要な最小フレーム数")

        # ラベルにツールチップを追加
        max_distance_label = QtWidgets.QLabel("Max distance (nm):")
        max_distance_label.setToolTip("Maximum distance (nm) a particle can move between frames / フレーム間で粒子が移動できる最大距離（ナノメートル）")
        
        max_frame_gap_label = QtWidgets.QLabel("Max frame gap:")
        max_frame_gap_label.setToolTip("How many frames a particle can disappear and still be linked / 粒子が消失してもリンクできるフレーム数")
        
        min_track_length_label = QtWidgets.QLabel("Min track length:")
        min_track_length_label.setToolTip("Minimum number of frames for a valid track / 有効な軌跡に必要な最小フレーム数")
        
        params_layout.addRow(max_distance_label, self.max_distance_spin)
        params_layout.addRow(max_frame_gap_label, self.max_frame_gap_spin)
        params_layout.addRow(min_track_length_label, self.min_track_length_spin)
        tracking_layout.addWidget(params_widget)
        
        content_layout.addWidget(tracking_group)
        
        # スペーサーを追加
        content_layout.addSpacing(10)
        
        # 表示設定グループ
        display_group = QtWidgets.QGroupBox("Display Settings")
        display_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        display_layout.setContentsMargins(10, 10, 10, 10)
        display_layout.setSpacing(8)
        
        # 軌跡表示チェックボックス
        self.show_tracks_check = QtWidgets.QCheckBox("Show Tracks")
        self.show_tracks_check.setToolTip("Display particle tracks on the image / 画像上に粒子軌跡を表示")
        self.show_tracks_check.setChecked(True)
        self.show_tracks_check.toggled.connect(self.onDisplaySettingsChanged)
        display_layout.addWidget(self.show_tracks_check)
        
        # Track ID表示チェックボックス
        self.show_track_ids_check = QtWidgets.QCheckBox("Show Track IDs")
        self.show_track_ids_check.setToolTip("Display track ID numbers on the image / 画像上にTrack ID番号を表示")
        self.show_track_ids_check.setChecked(True)
        self.show_track_ids_check.toggled.connect(self.onDisplaySettingsChanged)
        display_layout.addWidget(self.show_track_ids_check)
        
        content_layout.addWidget(display_group)
        
        # スペーサーを追加
        content_layout.addSpacing(10)
        

        
        # スペーサーを追加
        content_layout.addSpacing(10)
        
        # 実行ボタン
        self.detect_button = QtWidgets.QPushButton("Detect Current Frame")
        self.detect_button.setToolTip("Detect particles in the current frame / 現在のフレームで粒子を検出")
        self.detect_button.clicked.connect(self.detectParticles)
        self.detect_button.setMinimumHeight(30)
        content_layout.addWidget(self.detect_button)
        
        # 検出制御
        detect_control_layout = QtWidgets.QHBoxLayout()
        
        self.detect_all_button = QtWidgets.QPushButton("Detect All Frames")
        self.detect_all_button.setToolTip("Detect particles in all frames / すべてのフレームで粒子を検出")
        self.detect_all_button.clicked.connect(self.detectAllFrames)
        self.detect_all_button.setMinimumHeight(30)
        detect_control_layout.addWidget(self.detect_all_button)
        
        # Cancelボタンを削除（プログレスバーに統合予定）
        
        content_layout.addLayout(detect_control_layout)
        
        # トラッキング制御
        track_control_layout = QtWidgets.QHBoxLayout()
        
        self.track_button = QtWidgets.QPushButton("Track Particles")
        self.track_button.setToolTip("Link particles across frames to create tracks / フレーム間で粒子をリンクして軌跡を作成")
        self.track_button.clicked.connect(self.trackParticles)
        self.track_button.setEnabled(False)
        self.track_button.setMinimumHeight(30)
        track_control_layout.addWidget(self.track_button)
        
        # Cancelボタンを削除（プログレスバーに統合予定）
        
        content_layout.addLayout(track_control_layout)
        
        # プログレスバー（Cancelボタン統合）
        self.progress_container = QtWidgets.QWidget()
        progress_layout = QtWidgets.QHBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.track_progress = QtWidgets.QProgressBar()
        self.track_progress.setVisible(False)
        progress_layout.addWidget(self.track_progress)
        
        self.progress_cancel_button = QtWidgets.QPushButton("Cancel")
        self.progress_cancel_button.setToolTip("Cancel ongoing operation")
        self.progress_cancel_button.clicked.connect(self.cancelCurrentOperation)
        self.progress_cancel_button.setVisible(False)
        self.progress_cancel_button.setMinimumHeight(30)
        self.progress_cancel_button.setMaximumWidth(80)
        progress_layout.addWidget(self.progress_cancel_button)
        
        content_layout.addWidget(self.progress_container)
        
        # 軌跡編集ボタンを追加
        self.edit_tracks_button = QtWidgets.QPushButton("Edit Tracks")
        self.edit_tracks_button.setToolTip("Open Track Scheme Editor to manually edit particle tracks.\n\nFeatures:\n• Split tracks at specific frames\n• Delete unwanted tracks\n• Merge separate tracks\n• Visual track representation\n\nRequires completed tracking data.")
        self.edit_tracks_button.clicked.connect(self.open_track_editor)
        self.edit_tracks_button.setEnabled(False) # 初期状態は無効
        self.edit_tracks_button.setMinimumHeight(30)
        content_layout.addWidget(self.edit_tracks_button)
        
        # エクスポートボタン
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_csv_button.setToolTip("Export particle data to CSV file (wide format) / 粒子データをCSVファイルにエクスポート（ワイドフォーマット）")
        self.export_csv_button.clicked.connect(self.exportCSV)
        self.export_csv_button.setEnabled(False)
        self.export_csv_button.setMinimumHeight(30)
        content_layout.addWidget(self.export_csv_button)
        
        # Track Analysisボタン
        self.track_analysis_button = QtWidgets.QPushButton("Track Analysis")
        self.track_analysis_button.setToolTip("Open TrackMate-like analysis panel / TrackMate風の解析パネルを開く")
        self.track_analysis_button.clicked.connect(self.open_track_analysis)
        self.track_analysis_button.setEnabled(False)  # 初期状態は無効
        self.track_analysis_button.setMinimumHeight(30)
        content_layout.addWidget(self.track_analysis_button)
        
        content_layout.addStretch()
        
        return panel 
    
    def createCenterPanel(self):
        """中央パネルの作成"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        
        # 画像表示エリア
        try:
            # matplotlibの再初期化を試行
            if not _import_matplotlib():
                print(f"[ERROR] Matplotlib import failed in createCenterPanel")
                self.preview_label = QtWidgets.QLabel("Matplotlib not available")
                layout.addWidget(self.preview_label)
                return
            
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            # FigureとCanvasの作成
            self.preview_figure = Figure(figsize=(8, 6))
            self.preview_canvas = FigureCanvas(self.preview_figure)
            self.preview_axes = self.preview_figure.add_subplot(111)
            
            # コンポーネントが正しく作成されたか確認
            if not hasattr(self, 'preview_figure') or not hasattr(self, 'preview_canvas') or not hasattr(self, 'preview_axes'):
                raise Exception("Matplotlib components not properly created")
            
            # 右クリックメニューを有効化
            self.preview_canvas.mpl_connect('button_press_event', self.onCanvasClick)
            
            # キーボードイベントを有効化
            self.preview_canvas.mpl_connect('key_press_event', self.onCanvasKeyPress)
            
            # キャンバスにフォーカスを設定可能にする
            self.preview_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.preview_canvas.setFocus()
            
            layout.addWidget(self.preview_canvas)
            
        except Exception as e:
            print(f"[ERROR] Error creating matplotlib components: {e}")
            import traceback
            traceback.print_exc()
            self.preview_label = QtWidgets.QLabel("Matplotlib initialization failed")
            layout.addWidget(self.preview_label)
        
        # 画像下のナビゲーション
        bottom_nav_layout = QtWidgets.QHBoxLayout()
        
        # 左右矢印ボタン
        self.prev_button = QtWidgets.QPushButton("← Previous")
        self.prev_button.clicked.connect(self.previousFrame)
        self.prev_button.setFixedWidth(80)
        bottom_nav_layout.addWidget(self.prev_button)
        
        # フレームスライダー（PreviousとNextの間）
        self.bottom_frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bottom_frame_slider.setMinimum(0)
        self.bottom_frame_slider.setMaximum(0)
        self.bottom_frame_slider.setFixedWidth(200)
        self.bottom_frame_slider.valueChanged.connect(self.onBottomFrameSliderChanged)
        bottom_nav_layout.addWidget(self.bottom_frame_slider)
        
        self.next_button = QtWidgets.QPushButton("Next →")
        self.next_button.clicked.connect(self.nextFrame)
        self.next_button.setFixedWidth(80)
        bottom_nav_layout.addWidget(self.next_button)
        
        layout.addLayout(bottom_nav_layout)
        
        # フレーム情報（スライダーの下に配置）
        frame_info_layout = QtWidgets.QHBoxLayout()
        
        # フレーム情報
        self.bottom_frame_label = QtWidgets.QLabel("Frame: 0/0")
        self.bottom_frame_label.setAlignment(QtCore.Qt.AlignCenter)
        self.bottom_frame_label.setStyleSheet("font-weight: bold; color: blue;")
        frame_info_layout.addWidget(self.bottom_frame_label)
        
        layout.addLayout(frame_info_layout)
        
        # 粒子情報表示
        particle_info_layout = QtWidgets.QHBoxLayout()
        
        # 粒子数表示
        self.particle_count_label = QtWidgets.QLabel("Particles: 0")
        self.particle_count_label.setAlignment(QtCore.Qt.AlignCenter)
        self.particle_count_label.setStyleSheet("color: red; font-weight: bold; font-size: 12px;")
        particle_info_layout.addWidget(self.particle_count_label)
        
        layout.addLayout(particle_info_layout)
        
        # プログレスバー（画像の下に配置）
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)  # デフォルトでは非表示
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #aaa;
                border-radius: 3px;
                text-align: center;
                font-weight: bold;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        return panel
    
    def onCanvasClick(self, event):
        """キャンバスのクリックイベントハンドラー"""
        # クリック時にフォーカスを設定
        self.preview_canvas.setFocus()
        
        if event.button == 3:  # 右クリック
            self.showImageContextMenu(event)
    
    def onCanvasKeyPress(self, event):
        """キャンバスのキーボードイベントを処理"""
        try:
            # キーイベントをデバッグ出力
            print(f"[DEBUG] Key pressed: {event.key}")
            print(f"[DEBUG] Event type: {type(event)}")
            print(f"[DEBUG] GUI event: {event.guiEvent}")
            
            # より確実なCtrl+C検出
            if event.guiEvent:
                key = event.guiEvent.key()
                modifiers = event.guiEvent.modifiers()
                print(f"[DEBUG] Qt key: {key}, modifiers: {modifiers}")
                
                # Qt.Key_C (67) と Ctrl modifier をチェック
                if key == 67 and modifiers == QtCore.Qt.ControlModifier:
                    print("[DEBUG] Ctrl+C detected via Qt event, copying image to clipboard")
                    self.copyImageToClipboard()
                    event.guiEvent.accept()
                    return
            
            # matplotlibのキーイベント形式でもチェック
            if event.key == 'ctrl+c':
                print("[DEBUG] Ctrl+C detected via matplotlib event, copying image to clipboard")
                self.copyImageToClipboard()
                if event.guiEvent:
                    event.guiEvent.accept()
                return
                
        except Exception as e:
            print(f"[ERROR] Error in onCanvasKeyPress: {e}")
            import traceback
            traceback.print_exc()
    
    def onFocusInEvent(self, event):
        """ウィンドウがフォーカスを得た時の処理"""
        try:
            # キャンバスにフォーカスを設定
            if hasattr(self, 'preview_canvas'):
                self.preview_canvas.setFocus()
                print("[DEBUG] Focus set to canvas")
        except Exception as e:
            print(f"[ERROR] Error in onFocusInEvent: {e}")
            import traceback
            traceback.print_exc()
    
    def onWindowKeyPress(self, event):
        """ウィンドウレベルでのキーボードイベント処理"""
        try:
            key = event.key()
            modifiers = event.modifiers()
            print(f"[DEBUG] Window key press: key={key}, modifiers={modifiers}")
            
            # Ctrl+Cの検出
            if key == 67 and modifiers == QtCore.Qt.ControlModifier:  # Qt.Key_C
                print("[DEBUG] Ctrl+C detected at window level, copying image to clipboard")
                self.copyImageToClipboard()
                event.accept()
                return
            
            # 他のキーイベントは親に渡す
            event.ignore()
            
        except Exception as e:
            print(f"[ERROR] Error in onWindowKeyPress: {e}")
            import traceback
            traceback.print_exc()
    
    def showImageContextMenu(self, event):
        """画像の右クリックメニューを表示（particle_analysis方式）"""
        try:
            # コンテキストメニューを作成
            context_menu = QtWidgets.QMenu(self)
            
            # クリック位置を取得
            click_x, click_y = event.xdata, event.ydata
            
            if click_x is None or click_y is None:
                return
            
            #print(f"[DEBUG] Right click at position: x={click_x:.1f}, y={click_y:.1f}")
            
            # 現在のフレームの粒子を取得
            current_frame = gv.index if hasattr(gv, 'index') else 0
            particles = self.particles_by_frame.get(current_frame, []) if hasattr(self, 'particles_by_frame') else []
            
            # クリック位置に最も近い粒子を検索
            closest_particle = None
            min_distance = float('inf')
            click_threshold = 50  # ピクセル単位の閾値
            
            # スキャンサイズ情報を取得
            scan_size_x = getattr(gv, 'XScanSize', 1000.0)
            scan_size_y = getattr(gv, 'YScanSize', 1000.0)
            
            # ピクセルサイズを計算
            data_height, data_width = self.current_frame_data.shape
            pixel_size_x = scan_size_x / data_width
            pixel_size_y = scan_size_y / data_height
            
            # セパレーターを追加
            context_menu.addSeparator()
            
            # 画像コピーオプションを追加
            copy_action = context_menu.addAction("Copy Image to Clipboard")
            copy_action.setToolTip("Copy current view to clipboard for pasting in other applications")
            
            # セパレーターを追加
            context_menu.addSeparator()
            
            # 粒子編集オプション
            delete_action = context_menu.addAction("Delete Particle")
            add_action = context_menu.addAction("Add Particle")
            
            # 近接粒子を検索（削除用）
            closest_particle = None
            min_distance = float('inf')
            click_threshold = 50  # ピクセル単位の閾値
            
            if particles:
                for i, particle in enumerate(particles):
                    # 粒子の物理座標を計算
                    particle_x_phys = particle.x * pixel_size_x
                    particle_y_phys = particle.y * pixel_size_y
                    
                    # 距離を計算
                    distance = ((click_x - particle_x_phys) ** 2 + (click_y - particle_y_phys) ** 2) ** 0.5
                    
                    if distance < min_distance and distance < click_threshold:
                        min_distance = distance
                        closest_particle = (i, particle)
            
            # アクションを設定
            if closest_particle is not None:
                particle_index, particle = closest_particle
                #print(f"[DEBUG] Found particle at distance {min_distance:.1f} pixels")
                delete_action.triggered.connect(lambda: self.deleteParticle(current_frame, particle_index, particle))
            else:
                #print(f"[DEBUG] No particle found within {click_threshold} pixels")
                delete_action.triggered.connect(lambda: self.ignoreDeleteAction())
            
            # Copy Imageアクションを設定
            copy_action.triggered.connect(self.copyImageToClipboard)
            
            # Add Particleアクションを設定
            add_action.triggered.connect(lambda: self.addParticle(current_frame, click_x, click_y))
            
            # メニューを表示（マウスカーソルの現在位置を取得）
            cursor = QtGui.QCursor()
            global_pos = cursor.pos()
            action = context_menu.exec_(global_pos)
                
        except Exception as e:
            print(f"[ERROR] Error in showImageContextMenu: {e}")
            import traceback
            traceback.print_exc()
    
    def deleteParticle(self, frame_index, particle_index, particle):
        """指定された粒子を削除（particle_analysis方式）"""
        try:
            #print(f"[DEBUG] Deleting particle {particle_index} from frame {frame_index}")
            #print(f"[DEBUG] Particle position: x={particle.x:.2f}, y={particle.y:.2f}")
            
            # 粒子リストから削除
            if hasattr(self, 'particles_by_frame') and frame_index in self.particles_by_frame:
                particles = self.particles_by_frame[frame_index]
                if particle_index < len(particles):
                    deleted_particle = particles.pop(particle_index)
                    #print(f"[DEBUG] Deleted particle: x={deleted_particle.x:.2f}, y={deleted_particle.y:.2f}")
                    
                    # 現在のフレームの場合は、表示用の粒子リストも更新
                    if frame_index == gv.index:
                        self.detected_particles = particles.copy()
                        #print(f"[DEBUG] Updated detected_particles for current frame")
                    
                    # 統計情報を更新
                    self.updateStatistics()
                    
                    # プレビューを再描画
                    self.updatePreview()
                    
                    #print(f"[DEBUG] Particle deletion completed. Remaining particles in frame {frame_index}: {len(particles)}")
                else:
                   print(f"[ERROR] No particle data for frame {frame_index}")
                
        except Exception as e:
            print(f"[ERROR] Error in deleteParticle: {e}")
            import traceback
            traceback.print_exc()
    
    def ignoreDeleteAction(self):
        """削除アクションを無視（近接粒子がない場合）"""
        #print(f"[DEBUG] Delete action ignored - no particle nearby")
    
    def addParticle(self, frame_index, click_x, click_y):
        """指定された位置周辺で局所最大値を検索して粒子を追加"""
        try:
            #print(f"[DEBUG] Adding particle at position: x={click_x:.1f}, y={click_y:.1f}")
            
            # スキャンサイズ情報を取得
            scan_size_x = getattr(gv, 'XScanSize', 1000.0)
            scan_size_y = getattr(gv, 'YScanSize', 1000.0)
            
            # ピクセルサイズを計算
            data_height, data_width = self.current_frame_data.shape
            pixel_size_x = scan_size_x / data_width
            pixel_size_y = scan_size_y / data_height
            
            # 近接粒子があるかチェック
            particles = self.particles_by_frame.get(frame_index, []) if hasattr(self, 'particles_by_frame') else []
            click_threshold = 30  # 追加時の閾値（削除時より厳しい）
            
            for particle in particles:
                # 粒子の物理座標を計算
                particle_x_phys = particle.x * pixel_size_x
                particle_y_phys = particle.y * pixel_size_y
                
                # 距離を計算
                distance = ((click_x - particle_x_phys) ** 2 + (click_y - particle_y_phys) ** 2) ** 0.5
                
                if distance < click_threshold:
                    #print(f"[DEBUG] Particle already exists nearby (distance: {distance:.1f} pixels), ignoring add action")
                    return
            
            # クリック位置をピクセル座標に変換
            pixel_x = int(click_x / pixel_size_x)
            pixel_y = int(click_y / pixel_size_y)
            
            #print(f"[DEBUG] Click position in pixels: x={pixel_x}, y={pixel_y}")
            
            # 画像範囲内かチェック
            if not (0 <= pixel_x < data_width and 0 <= pixel_y < data_height):
                print(f"[ERROR] Click position outside image bounds")
                return
            
            # 局所最大値検索のパラメータ
            search_radius = 10  # ピクセル単位の検索半径
            min_distance = 5    # 最小距離
            
            # 検索範囲を定義
            x_min = max(0, pixel_x - search_radius)
            x_max = min(data_width, pixel_x + search_radius + 1)
            y_min = max(0, pixel_y - search_radius)
            y_max = min(data_height, pixel_y + search_radius + 1)
            
            # 検索範囲のデータを取得
            search_region = self.current_frame_data[y_min:y_max, x_min:x_max]
            
            if search_region.size == 0:
                print(f"[ERROR] Search region is empty")
                return
            
            # 局所最大値を検索
            from skimage.feature import peak_local_max
            peaks = peak_local_max(search_region, 
                                 min_distance=min_distance,
                                 threshold_abs=np.max(search_region) * 0.5,  # 最大値の50%以上
                                 exclude_border=False)
            
            if len(peaks) == 0:
                #print(f"[DEBUG] No local maxima found in search region")
                # 検索範囲を拡大して再試行
                search_radius = 20
                x_min = max(0, pixel_x - search_radius)
                x_max = min(data_width, pixel_x + search_radius + 1)
                y_min = max(0, pixel_y - search_radius)
                y_max = min(data_height, pixel_y + search_radius + 1)
                search_region = self.current_frame_data[y_min:y_max, x_min:x_max]
                
                peaks = peak_local_max(search_region, 
                                     min_distance=min_distance,
                                     threshold_abs=np.max(search_region) * 0.3,  # 閾値を下げる
                                     exclude_border=False)
            
            if len(peaks) > 0:
                # 最も強いピークを選択
                peak_intensities = [search_region[peak[0], peak[1]] for peak in peaks]
                best_peak_idx = np.argmax(peak_intensities)
                best_peak = peaks[best_peak_idx]
                
                # グローバル座標に変換
                global_x = x_min + best_peak[1]
                global_y = y_min + best_peak[0]
                
                # 物理座標に変換
                phys_x = global_x * pixel_size_x
                phys_y = global_y * pixel_size_y
                
                #print(f"[DEBUG] Found peak at pixel coordinates: x={global_x}, y={global_y}")
                #print(f"[DEBUG] Physical coordinates: x={phys_x:.1f}, y={phys_y:.1f}")
                
                # 新しい粒子を作成
                new_particle = Particle(
                    frame=frame_index,
                    x=global_x,
                    y=global_y,
                    intensity=search_region[best_peak[0], best_peak[1]],
                    radius=5.0,  # デフォルト半径
                    quality=1.0   # 手動追加なので品質は1.0
                )
                
                # 粒子リストに追加
                if hasattr(self, 'particles_by_frame') and frame_index in self.particles_by_frame:
                    self.particles_by_frame[frame_index].append(new_particle)
                    
                    # 現在のフレームの場合は、表示用の粒子リストも更新
                    if frame_index == gv.index:
                        self.detected_particles = self.particles_by_frame[frame_index].copy()
                        #print(f"[DEBUG] Updated detected_particles for current frame")
                    
                    # 統計情報を更新
                    self.updateStatistics()
                    
                    # プレビューを再描画
                    self.updatePreview()
                    
                    #print(f"[DEBUG] Particle added successfully. Total particles in frame {frame_index}: {len(self.particles_by_frame[frame_index])}")
                else:
                    print(f"[ERROR] No particle data for frame {frame_index}")
        
        except Exception as e:
            print(f"[ERROR] Error in addParticle: {e}")
            import traceback
            traceback.print_exc()
    
    def createRightPanel(self):
        """右パネルの作成"""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        
        # 統計情報
        stats_group = QtWidgets.QGroupBox("Statistics")
        stats_layout = QtWidgets.QFormLayout(stats_group)
        
        self.total_particles_label = QtWidgets.QLabel("0")
        stats_layout.addRow("Total Particles:", self.total_particles_label)
        
        self.total_tracks_label = QtWidgets.QLabel("0")
        stats_layout.addRow("Total Tracks:", self.total_tracks_label)
        
        self.mean_velocity_label = QtWidgets.QLabel("0.0")
        stats_layout.addRow("Mean Velocity:", self.mean_velocity_label)
        
        self.mean_displacement_label = QtWidgets.QLabel("0.0")
        stats_layout.addRow("Mean Displacement:", self.mean_displacement_label)
        
        layout.addWidget(stats_group)
        
        # 軌跡テーブル
        tracks_group = QtWidgets.QGroupBox("Tracks")
        tracks_layout = QtWidgets.QVBoxLayout(tracks_group)
        
        self.tracks_table = QtWidgets.QTableWidget()
        self.tracks_table.setColumnCount(4)
        self.tracks_table.setHorizontalHeaderLabels(["Track ID", "Start Frame", "End Frame", "Duration"])
        tracks_layout.addWidget(self.tracks_table)
        
        layout.addWidget(tracks_group)
        
        layout.addStretch()
        return panel
    
    def initializeData(self):
        """データの初期化"""
        try:
            # メインウィンドウの前処理済みデータを直接使用
            if hasattr(gv, 'aryData') and gv.aryData is not None:
                # メインウィンドウで適用された前処理済みデータをベースとして使用
                base_data = gv.aryData.copy()
                
                # Particle trackingウィンドウ内のpreprocessing（スムージング）を適用
                processed_data = self.preprocessData(base_data)
                
                if processed_data is not None:
                    self.current_frame_data = processed_data
                    self.total_frames = 1  # 単一フレームとして扱う
                    self.current_frame = 0
                    
                    # スキャンサイズ情報を取得
                    self.scan_size_x = getattr(gv, 'XScanSize', 0)
                    self.scan_size_y = getattr(gv, 'YScanSize', 0)
                    self.x_pixels = getattr(gv, 'XPixel', 0)
                    self.y_pixels = getattr(gv, 'YPixel', 0)
                    
                    # プレビューを更新
                    self.updatePreview()
                    
                    # UIを更新
                    self.updateFrameLabel()
                    self.updateBottomLabels()
                    
                    #print(f"[DEBUG] Particle tracking initialized with preprocessing")
                    #print(f"[DEBUG] Data shape: {self.current_frame_data.shape}")
                    #print(f"[DEBUG] Scan size: {self.scan_size_x} x {self.scan_size_y} nm")
                    
                    return True  # 成功
                    
                else:
                    print(f"[ERROR] Preprocessing failed during initialization")
                    QtWidgets.QMessageBox.warning(self, "Preprocessing Error", 
                        "Failed to apply preprocessing during initialization.")
                    return False  # 失敗
                    
            else:
                print(f"[ERROR] No data available in gv.aryData")
                QtWidgets.QMessageBox.warning(self, "No Data", 
                    "Please load AFM data in the main window first.")
                return False  # 失敗
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize particle tracking data: {e}")
            import traceback
            traceback.print_exc()
            return False  # 失敗
    
    def detectParticles(self):
        """現在のフレームで粒子を検出"""
        # [FIX] Add a guard clause to ensure matplotlib components are initialized.
        if not hasattr(self, 'preview_axes'):
            #print("[DEBUG] UI not fully initialized. Skipping detection.")
            return
        
        #print(f"[DEBUG] ===== detectParticles ENTRY =====")
        #print(f"[DEBUG] detectParticles called")
        #print(f"[DEBUG] Current threshold setting: {self.threshold_spin.value()}")
        #print(f"[DEBUG] Current radius setting: {self.radius_spin.value()}")
        #print(f"[DEBUG] Current min_distance setting: {self.min_distance_spin.value()}")
        #print(f"[DEBUG] Current detection method: {self.detection_method_combo.currentText()}")
        
        try:
            if not hasattr(self, 'current_frame_data') or self.current_frame_data is None:
                print(f"[ERROR] No current frame data available for detection")
                return
            
            # メインウィンドウの前処理済みデータを直接使用
            detection_data = self.current_frame_data.copy()
            
            # 検出方法を取得
            detection_method = self.detection_method_combo.currentText()
            #print(f"[DEBUG] UI detection method: {detection_method}")
            
            # 検出パラメータを取得
            radius_nm = self.radius_spin.value()  # nm単位
            threshold = self.threshold_spin.value()
            min_distance = self.min_distance_spin.value()
            
            # nm単位のradiusをピクセル単位に変換
            pixel_size_x = self.scan_size_x / detection_data.shape[1]
            pixel_size_y = self.scan_size_y / detection_data.shape[0]
            radius_pixels = radius_nm / min(pixel_size_x, pixel_size_y)  # より小さいピクセルサイズを使用
            
            #print(f"[DEBUG] Starting particle detection with method: {detection_method}")
            #print(f"[DEBUG] Parameters: radius_nm={radius_nm}, radius_pixels={radius_pixels:.2f}, threshold={threshold}, min_distance={min_distance}")
            #print(f"[DEBUG] Data shape: {detection_data.shape}")
            #print(f"[DEBUG] Data range: {np.min(detection_data):.3f} to {np.max(detection_data):.3f}")
            #print(f"[DEBUG] Pixel size: {pixel_size_x:.6f} x {pixel_size_y:.6f} nm/pixel")
            
            # 検出器を作成
            detector = SpotDetector()
            detector.radius = radius_pixels  # ピクセル単位で設定
            detector.threshold = threshold
            detector.min_distance = min_distance  # ユーザー設定のmin_distanceを設定
            detector.do_subpixel = self.subpixel_check.isChecked()
            detector.max_position_correction = self.max_pos_corr_check.isChecked()  # 最大値位置補正の設定
            detector.min_intensity_ratio = self.min_intensity_spin.value()
            detector.size_filter_enabled = self.size_filter_check.isChecked()
            detector.size_tolerance = self.size_tolerance_spin.value()
            detector.selection_radius = self.selection_radius_spin.value()
            detector.area_method = self.area_method_combo.currentText()
            
            # Watershed関連のパラメータを設定
            if detector.area_method == "Watershed":
                detector.watershed_compactness = self.watershed_compactness_spin.value()
                detector.watershed_threshold = self.watershed_threshold_spin.value()
                
            # 検出器を保存（後でパラメータ更新時に使用）
            self.detector = detector
            
            # 粒子検出を実行（シンプルな処理）
            #print(f"[DEBUG] About to call detector.detect_particles with method: {detection_method}")
            detected_particles = detector.detect_particles(detection_data, method=detection_method)
            
            #print(f"[DEBUG] Detection completed, found {len(detected_particles)} particles")
            
            # 検出結果を保存
            self.detected_particles = detected_particles
            
            #print(f"[DEBUG] ===== detectParticles EXIT =====")
            
            # 統計情報を更新
            self.updateStatistics()
            
            # 検出された粒子を描画
            #print(f"[DEBUG] Calling drawDetectedParticles")
            self.drawDetectedParticles()
            
            #print(f"[DEBUG] ===== FINAL RESULT =====")
            #print(f"[DEBUG] Detection process completed: {len(detected_particles)} particles")
            #print(f"[DEBUG] ===== detectParticles EXIT =====")
            
        except Exception as e:
            print(f"[ERROR] Failed to detect particles: {e}")
            import traceback
            traceback.print_exc()
    
    def detectAllFrames(self):
        """全フレームで粒子検出を実行"""
        # [FIX] Add a guard clause to ensure matplotlib components are initialized.
        if not hasattr(self, 'preview_axes'):
            #print("[DEBUG] UI not fully initialized. Skipping detection.")
            return
        
        # キャンセルフラグを初期化
        self.detection_cancelled = False
        gv.cancel_operation = False
        
        # 検出パラメータを取得
        detection_params = {
            'method': self.detection_method_combo.currentText(),
            'radius': self.radius_spin.value(),
            'threshold': self.threshold_spin.value(),
            'min_distance': self.min_distance_spin.value()
        }
        
        # プログレスバーとCancelボタンを表示
        self.track_progress.setVisible(True)
        self.progress_cancel_button.setVisible(True)
        self.track_progress.setMaximum(gv.FrameNum)
        self.track_progress.setValue(0)
        
        # 軌跡データをクリア（Detect All Frames実行時に既存の軌跡を消去）
        if hasattr(self, 'tracks_df'):
            self.tracks_df = None
        if hasattr(self, 'tracks'):
            self.tracks = []
        if hasattr(self, 'tracked_particles'):
            self.tracked_particles = {}
        
        # 軌跡表示をクリア
        self.clearParticleBoundaries()
        if hasattr(self, 'preview_axes'):
            # 軌跡線をクリア
            lines_to_remove = []
            for line in self.preview_axes.lines[:]:
                if hasattr(line, '_track_line'):
                    lines_to_remove.append(line)
            for line in lines_to_remove:
                line.remove()
            
            # 軌跡IDラベルをクリア
            texts_to_remove = []
            for text in self.preview_axes.texts[:]:
                if hasattr(text, '_track_id'):
                    texts_to_remove.append(text)
            for text in texts_to_remove:
                text.remove()
            
            self.preview_axes.figure.canvas.draw()
        
        # トラッキングボタンを無効化（新しい検出後に再度有効化）
        self.track_button.setEnabled(False)
        
        # 粒子データと画像データを保存する辞書を初期化
        self.particles_by_frame = {}
        self.frame_data_by_frame = {}  # 各フレームの画像データを保存
        
        # 現在のファイル番号を保存
        current_file_num = gv.currentFileNum
        
                # 各フレームで粒子検出を実行
        for frame_idx in range(gv.FrameNum):
            # キャンセルチェック
            if hasattr(self, 'detection_cancelled') and self.detection_cancelled:
                print("DEBUG: Detection cancelled by user")
                break
            if hasattr(gv, 'cancel_operation') and gv.cancel_operation:
                print("DEBUG: Detection cancelled via global flag")
                break
                
            self.track_progress.setValue(frame_idx)
            QtWidgets.QApplication.processEvents()
            
            #print(f"DEBUG: Processing frame {frame_idx+1}/{gv.FrameNum}")
            
            # フレームを変更
            gv.index = frame_idx
            if hasattr(self.main_window, 'frameSlider'):
                self.main_window.frameSlider.setValue(frame_idx)
            if hasattr(self.main_window, 'updateFrame'):
                self.main_window.updateFrame()
            
            # データを取得
            if not self.initializeData():
                #print(f"DEBUG: Failed to get data for frame {frame_idx}")
                self.particles_by_frame[frame_idx] = []
                self.frame_data_by_frame[frame_idx] = None
                continue
            
            # 画像データを保存（重要！）
            if hasattr(self, 'current_frame_data') and self.current_frame_data is not None:
                self.frame_data_by_frame[frame_idx] = self.current_frame_data.copy()
                #print(f"DEBUG: Saved frame {frame_idx} data with shape {self.current_frame_data.shape}")
            else:
                #print(f"DEBUG: No current_frame_data available for frame {frame_idx}")
                self.particles_by_frame[frame_idx] = []
                self.frame_data_by_frame[frame_idx] = None
                continue
            
            # 現在のフレームで粒子検出（detectParticlesと同じ処理）
            if self.current_frame_data is not None:
                radius_nm = self.radius_spin.value()
                threshold = self.threshold_spin.value()
                min_distance = self.min_distance_spin.value()
                
                # nm単位のradiusをピクセル単位に変換
                pixel_size_x = self.scan_size_x / self.current_frame_data.shape[1]
                pixel_size_y = self.scan_size_y / self.current_frame_data.shape[0]
                radius_pixels = radius_nm / min(pixel_size_x, pixel_size_y)
                
                # 検出器を作成
                detector = SpotDetector()
                detector.radius = radius_pixels
                detector.threshold = threshold
                detector.min_distance = min_distance
                detector.do_subpixel = self.subpixel_check.isChecked()
                detector.max_position_correction = self.max_pos_corr_check.isChecked()  # 最大値位置補正の設定
                detector.min_intensity_ratio = self.min_intensity_spin.value()
                detector.size_filter_enabled = self.size_filter_check.isChecked()
                detector.size_tolerance = self.size_tolerance_spin.value()
                detector.selection_radius = self.selection_radius_spin.value()
                detector.area_method = self.area_method_combo.currentText()
                
                # Watershed関連のパラメータを設定
                if detector.area_method == "Watershed":
                    detector.watershed_compactness = self.watershed_compactness_spin.value()
                    detector.watershed_threshold = self.watershed_threshold_spin.value()
                    #print(f"DEBUG: Set watershed_compactness={detector.watershed_compactness}, watershed_threshold={detector.watershed_threshold}")
                

                # 粒子検出を実行（シンプルな処理）
                particles = detector.detect_particles(self.current_frame_data, method=detection_params['method'])
                
                # ▼▼▼▼▼ ここに修正を追加 ▼▼▼▼▼
                for p in particles:
                    p.frame = frame_idx  # 各粒子に正しいフレーム番号を設定
                # ▲▲▲▲▲ 修正ここまで ▲▲▲▲▲
                
                # 結果を保存
                self.particles_by_frame[frame_idx] = particles
                #print(f"DEBUG: Found {len(particles)} particles in frame {frame_idx+1}")
                
                # 最後のフレームの場合は、検出結果を保存してプレビュー表示で使用
                if frame_idx == gv.FrameNum - 1:
                    self.detected_particles = particles
                    #print(f"DEBUG: Saved detection results for preview: {len(particles)} particles")
            else:
                #print(f"DEBUG: No current_frame_data available for frame {frame_idx}")
                self.particles_by_frame[frame_idx] = []
                self.frame_data_by_frame[frame_idx] = None
                continue
            
            # 検出結果の詳細をログ出力
           
            # リアルタイムで現在のフレームのマーカーを表示
            self.detected_particles = particles
            self.current_frame_index = frame_idx  # 現在のフレームインデックスを更新
            
            # 現在のフレームの粒子マーカーを描画
            self.updatePreview()
            self.drawDetectedParticles()
            self.updateFrameLabel()  # フレームラベルを更新
            QtWidgets.QApplication.processEvents()  # イベントを処理してリアルタイム更新
            
            # 最後のフレームのdetectorを保存（プレビュー表示で使用）
            if frame_idx == gv.FrameNum - 1:
                self.detector = detector
                #print(f"DEBUG: Saved detector for preview display")
                #print(f"DEBUG: Saved detector settings:")
                #print(f"DEBUG:   - size_filter_enabled: {self.detector.size_filter_enabled}")
                #print(f"DEBUG:   - area_method: {self.detector.area_method}")
                #print(f"DEBUG:   - watershed_compactness: {getattr(self.detector, 'watershed_compactness', 'NOT_SET')}")
                #print(f"DEBUG:   - watershed_threshold: {getattr(self.detector, 'watershed_threshold', 'NOT_SET')}")
                
                # 検出に使用された画像データも保存
                self.current_frame_data = self.current_frame_data.copy()
                #print(f"DEBUG: Saved current_frame_data for preview display")
        
        self.track_progress.setValue(gv.FrameNum)
        #print("DEBUG: Particle detection completed")
        
        # プログレスバーとCancelボタンを非表示
        self.track_progress.setVisible(False)
        self.progress_cancel_button.setVisible(False)
        
        # 元のフレームに戻す
        gv.currentFileNum = current_file_num
        gv.index = 0
        if hasattr(self.main_window, 'frameSlider'):
            self.main_window.frameSlider.setValue(0)
        if hasattr(self.main_window, 'updateFrame'):
            self.main_window.updateFrame()
        
        # 現在のフレームインデックスを0に設定
        self.current_frame_index = 0
        
        # データを再初期化してフレーム0のデータを取得
        
        
        # フレーム0の粒子データを設定
        if 0 in self.particles_by_frame:
            particles = self.particles_by_frame[0]
            #print(f"DEBUG: Setting particles for frame 0, found {len(particles)} particles")
            self.detected_particles = particles
        else:
            self.detected_particles = []
        
        # プレビューを更新して粒子マーカーを描画
        self.updatePreview()
        
        
        # フレームスライダーの最大値を設定
        if hasattr(gv, 'FrameNum') and gv.FrameNum > 0:
            self.bottom_frame_slider.setMaximum(gv.FrameNum - 1)
            #print(f"DEBUG: All Frames completed, frame slider max set to {gv.FrameNum - 1}")
        
        # フレームラベルを更新
        self.updateFrameLabel()
        self.updateBottomLabels()
        
        # キャンセルされた場合は早期リターン
        if hasattr(self, 'detection_cancelled') and self.detection_cancelled:
            #print("DEBUG: Detection was cancelled, not enabling track button")
            return
        if hasattr(gv, 'cancel_operation') and gv.cancel_operation:
            #print("DEBUG: Detection was cancelled via global flag, not enabling track button")
            return
            
        # トラッキングボタンを有効化
        self.track_button.setEnabled(True)
    
    def trackParticles(self):
        """粒子トラッキングの実行（アルゴリズムを振り分け）"""
        # [FIX] Add a guard clause to ensure matplotlib components are initialized.
        if not hasattr(self, 'preview_axes'):
            #print("[DEBUG] UI not fully initialized. Skipping tracking.")
            return
        
        if not self.particles_by_frame:
            QtWidgets.QMessageBox.warning(self, "Warning", "No particles detected. Please run 'Detect All Frames' first.")
            return

        # トラッキング制御の初期化
        self.tracking_cancelled = False
        gv.cancel_operation = False
        self.track_button.setEnabled(False)
        self.track_progress.setVisible(True)
        self.progress_cancel_button.setVisible(True)
        self.track_progress.setRange(0, 100)
        self.track_progress.setValue(0)

        self.tracks_df = None
        algorithm = self.tracking_algo_combo.currentText()

        try:
            #print(f"[DEBUG] Selected algorithm: {algorithm}")
            if algorithm == "Trackpy (Simple Linker)":
                #print(f"[DEBUG] Starting Trackpy Simple Linker...")
                self._track_with_trackpy()
            elif algorithm == "Simple LAP Tracker (SciPy)":
                #print(f"[DEBUG] Starting Simple LAP Tracker...")
                self._track_with_scipy_lap()
            elif algorithm == "Kalman Filter (filterpy)":
                self._track_with_kalman()
            else:
                print(f"[ERROR] Unknown algorithm: {algorithm}")
                return

            # キャンセルチェック
            if self.tracking_cancelled:
                #print("DEBUG: Tracking cancelled by user")
                return

            if self.tracks_df is None or self.tracks_df.empty:
                QtWidgets.QMessageBox.information(self, "Tracking Complete", "Tracking completed, but no valid tracks were found with the current parameters.")
                self.tracks_df = None
            else:
                QtWidgets.QMessageBox.information(
                    self, "Tracking Complete",
                    f"Tracking completed. Found {self.tracks_df['particle'].nunique()} tracks using {algorithm}."
                )

            self.updateStatistics()
            self.updateTracksTable()
            self.export_csv_button.setEnabled(self.tracks_df is not None)
            # export_image_buttonは削除済み（クリップボードコピーに変更）
            self.updatePreview()

        except Exception as e:
            # キャンセルによる例外の場合は特別処理
            if "cancelled" in str(e).lower():
                #print(f"DEBUG: Tracking cancelled via exception: {e}")
                return
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred during tracking with {algorithm}:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            # キャンセルされた場合の処理
            if hasattr(self, 'tracking_cancelled') and self.tracking_cancelled:
                #print("DEBUG: Tracking was cancelled, cleaning up...")
                self.tracks_df = None  # キャンセル時は結果をクリア
            
            # ボタンの状態を更新
            has_tracks = self.tracks_df is not None and not self.tracks_df.empty
            self.edit_tracks_button.setEnabled(has_tracks)
            self.export_csv_button.setEnabled(has_tracks)
            self.track_analysis_button.setEnabled(has_tracks)
            
            # ボタンが無効な場合のtooltipを更新
            if not has_tracks:
                self.edit_tracks_button.setToolTip("Edit Tracks (Disabled)\n\n"
                                                 "This button is disabled because:\n"
                                                 "• No tracking data is available\n"
                                                 "• Or tracking has not been completed\n\n"
                                                 "Please run particle tracking first\n"
                                                 "to enable track editing.")
                self.export_csv_button.setToolTip("Export CSV (Disabled)\n\n"
                                                "This button is disabled because:\n"
                                                "• No tracking data is available\n"
                                                "• Or tracking has not been completed\n\n"
                                                "Please run particle tracking first\n"
                                                "to enable CSV export.")
                self.track_analysis_button.setToolTip("Track Analysis (Disabled)\n\n"
                                                    "This button is disabled because:\n"
                                                    "• No tracking data is available\n"
                                                    "• Or tracking has not been completed\n\n"
                                                    "Please run particle tracking first\n"
                                                    "to enable track analysis.")
            else:
                self.edit_tracks_button.setToolTip("Open Track Scheme Editor to manually edit particle tracks.\n\n"
                                                 "Features:\n"
                                                 "• Split tracks at specific frames\n"
                                                 "• Delete unwanted tracks\n"
                                                 "• Merge separate tracks\n"
                                                 "• Visual track representation\n\n"
                                                 "Requires completed tracking data.")
                self.export_csv_button.setToolTip("Export particle data to CSV file (wide format) / 粒子データをCSVファイルにエクスポート（ワイドフォーマット）")
                self.track_analysis_button.setToolTip("Open TrackMate-like analysis panel / TrackMate風の解析パネルを開く")
            
            # UI状態を復元
            self.track_button.setEnabled(True)
            self.track_progress.setVisible(False)
            self.progress_cancel_button.setVisible(False)
    
    def _track_with_trackpy(self):
        """Trackpyを使用してトラッキング"""
        if not TRACKPY_AVAILABLE:
            print("Warning: trackpy not available, using fallback tracking method")
            _frozen = getattr(sys, "frozen", False)
            if _frozen:
                msg = (
                    "trackpy module not available.\n"
                    "trackpy モジュールがインストールされていません。\n\n"
                    "Using fallback tracking method.\n"
                    "フォールバックのトラッキングを使用します。\n\n"
                    "trackpy is not bundled with this installation.\n"
                    "trackpy はこのパッケージに含まれていません。"
                )
            else:
                msg = (
                    "trackpy module not available.\n"
                    "trackpy モジュールがインストールされていません。\n\n"
                    "Using fallback tracking method.\n"
                    "フォールバックのトラッキングを使用します。\n\n"
                    "To enable full tracking features, install trackpy:\n"
                    "pip install trackpy"
                )
            QtWidgets.QMessageBox.warning(self, "Warning", msg)
            # フォールバックメソッドを使用
            self._track_with_scipy_lap()
            return
            
        #print(f"[DEBUG] === Trackpy Simple Linker Started ===")
        
        # (このメソッドは以前のbtrack導入前のバージョンに戻します)
        all_particles_list = []
        for frame_idx, particles in self.particles_by_frame.items():
            for p in particles:
                all_particles_list.append({'frame': p.frame, 'y': p.y, 'x': p.x})
        
        #print(f"[DEBUG] Total particles to process: {len(all_particles_list)}")
        
        if not all_particles_list: 
            #print(f"[DEBUG] No particles to track")
            return
            
        features_df = pd.DataFrame(all_particles_list)
        #print(f"[DEBUG] Features DataFrame shape: {features_df.shape}")
        
        scan_size_x = getattr(gv, 'XScanSize', 1000.0); scan_size_y = getattr(gv, 'YScanSize', 1000.0)
        frame_data = self.frame_data_by_frame[0]
        pixel_size_x = scan_size_x / frame_data.shape[1]; pixel_size_y = scan_size_y / frame_data.shape[0]

        features_df['x_scaled'] = features_df['x'] * (pixel_size_x / pixel_size_y)
        max_distance_nm = self.max_distance_spin.value()
        search_range_scaled = max_distance_nm / pixel_size_y
        max_frame_gap = self.max_frame_gap_spin.value()
        min_track_length = self.min_track_length_spin.value()
        
        #print(f"[DEBUG] Parameters: max_distance={max_distance_nm}nm, search_range={search_range_scaled:.2f}px, max_gap={max_frame_gap}, min_length={min_track_length}")
        #print(f"[DEBUG] Pixel sizes: x={pixel_size_x:.6f}, y={pixel_size_y:.6f} nm/pixel")

        #print(f"[DEBUG] Calling trackpy.link...")
        # trackpyの内部ログ出力を抑制
        import logging
        import sys
        from io import StringIO
        
        # 標準出力と標準エラー出力を一時的にキャプチャ
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        # trackpyのログレベルを一時的に変更
        trackpy_logger = logging.getLogger('trackpy')
        old_level = trackpy_logger.level
        trackpy_logger.setLevel(logging.ERROR)
        
        try:
            linked_df = tp.link(features_df, search_range=search_range_scaled, memory=max_frame_gap, pos_columns=['y', 'x_scaled'])
        finally:
            # 標準出力と標準エラー出力を復元
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # ログレベルを復元
            trackpy_logger.setLevel(old_level)
            
        # キャンセルチェック
        if hasattr(self, 'tracking_cancelled') and self.tracking_cancelled:
            #print("DEBUG: Tracking cancelled by user")
            return
        if hasattr(gv, 'cancel_operation') and gv.cancel_operation:
            #print("DEBUG: Tracking cancelled via global flag")
            return
        
        #print(f"[DEBUG] trackpy.link completed, DataFrame shape: {linked_df.shape}")
        #print(f"[DEBUG] Unique particles before filtering: {linked_df['particle'].nunique()}")
        
        linked_df['x'] = linked_df['x_scaled'] / (pixel_size_x / pixel_size_y) # スケールを戻す
        
        #print(f"[DEBUG] Calling trackpy.filter_stubs...")
        # filter_stubsでも出力抑制
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        trackpy_logger = logging.getLogger('trackpy')
        old_level = trackpy_logger.level
        trackpy_logger.setLevel(logging.ERROR)
        
        try:
            self.tracks_df = tp.filter_stubs(linked_df, min_track_length)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            trackpy_logger.setLevel(old_level)
            
        #print(f"[DEBUG] trackpy.filter_stubs completed, DataFrame shape: {self.tracks_df.shape}")
        #print(f"[DEBUG] Final unique tracks: {self.tracks_df['particle'].nunique() if not self.tracks_df.empty else 0}")
        
        #print(f"[DEBUG] === Trackpy Simple Linker Completed ===")
    
    def updatePreview(self):
        """プレビュー画像の更新"""
        try:
            if hasattr(self, 'current_frame_data') and self.current_frame_data is not None:
                # メインウィンドウの前処理済みデータをそのまま使用
                display_data = self.current_frame_data.copy()
                
                # メインウィンドウの前処理済みデータをそのまま表示
                # （粒子検出用の前処理は検出時のみ適用）
                
                # matplotlibコンポーネントが利用可能かチェック
                if not hasattr(self, 'preview_axes'):
                    print(f"[ERROR] Matplotlib axes not available for drawing particles")
                    print(f"[ERROR] This error occurs in drawDetectedParticles method")
                    return
                
                # 画像をクリア
                self.preview_axes.clear()
                
                # スキャンサイズ情報を使用
                scan_size_x = self.scan_size_x
                scan_size_y = self.scan_size_y
                
                # スキャンサイズが0の場合はエラー
                if scan_size_x == 0 or scan_size_y == 0:
                    print(f"[ERROR] Scan size is 0, cannot display image")
                    print(f"[ERROR] Please load AFM data first to get scan size information")
                    return
                
                # 物理サイズに基づいた正しいアスペクト比を計算
                data_height, data_width = display_data.shape
                pixel_size_x = scan_size_x / data_width
                pixel_size_y = scan_size_y / data_height
                
                #print(f"[DEBUG] Image shape: {display_data.shape}")
                #print(f"[DEBUG] Scan size: {scan_size_x} x {scan_size_y} nm")
                #print(f"[DEBUG] Pixel size: {pixel_size_x:.6f} x {pixel_size_y:.6f} nm/pixel")
                #print(f"[DEBUG] Data range: {np.min(display_data):.3f} to {np.max(display_data):.3f}")
                
                # AFM画像を表示（物理サイズに基づいたextentを使用）
                im = self.preview_axes.imshow(display_data, cmap='viridis', 
                                            extent=[0, scan_size_x, 0, scan_size_y],
                                            aspect='equal', origin='lower')
                
                # 軸ラベルを設定
                self.preview_axes.set_xlabel('X (nm)')
                self.preview_axes.set_ylabel('Y (nm)')
                
                # 検出された粒子を「+」マークで表示
                #print(f"[DEBUG] updatePreview: Calling drawDetectedParticles")
                self.drawDetectedParticles()
                #print(f"[DEBUG] updatePreview: drawDetectedParticles completed")
                
                # 軌跡データを描画 (トラッキング実行後)
                if hasattr(self, 'tracks_df') and self.tracks_df is not None:
                    # trackpyのプロット機能を利用して軌跡を描画
                    # ただし、物理座標に変換して手動で描画する方が確実
                    # 軌跡毎に異なる色を使用
                    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'brown', 'gray']
                    for i, (track_id, track_data) in enumerate(self.tracks_df.groupby('particle')):
                        # 物理座標に変換
                        x_phys = track_data['x'] * pixel_size_x
                        y_phys = track_data['y'] * pixel_size_y
                        
                        # 軌跡線を描画（Show Tracksがチェックされている場合のみ）
                        if self.show_tracks_check.isChecked():
                            color = colors[i % len(colors)]
                            line = self.preview_axes.plot(x_phys, y_phys, linestyle='-', color=color, alpha=0.8, linewidth=2.5)[0]
                            line._track_line = True  # 軌跡線として識別するための属性
                        
                        # Track IDを表示（Show Track IDsがチェックされている場合のみ）
                        if self.show_track_ids_check.isChecked():
                            # 軌跡の最初の点にTrack IDを表示
                            if len(track_data) > 0:
                                first_point = track_data.iloc[0]
                                x_id = first_point['x'] * pixel_size_x
                                y_id = first_point['y'] * pixel_size_y
                                text = self.preview_axes.text(x_id, y_id, f'ID:{track_id}', 
                                                    color='white', fontsize=8, 
                                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                                                    ha='center', va='center')
                                text._track_id = True  # 軌跡IDラベルとして識別するための属性
                
                # 画像を更新
                self.preview_canvas.draw()
                
                #print(f"[DEBUG] Preview updated with main window preprocessed data")
                
            else:
                print(f"[ERROR] No current frame data available for preview")
                
        except Exception as e:
            print(f"[ERROR] Failed to update preview: {e}")
            import traceback
            traceback.print_exc()
    
    def drawDetectedParticles(self):
        """検出された粒子を「+」マークで表示（フィルタリング済みの粒子のみ）"""
        #print(f"[DEBUG] drawDetectedParticles called")
        #print(f"[DEBUG] ===== drawDetectedParticles ENTRY =====")
        
        # 粒子データの状態を詳しく確認
       
        if not hasattr(self, 'detected_particles'):
            #print(f"[DEBUG] No detected particles available")
            #print(f"[DEBUG] ===== drawDetectedParticles EXIT (no particles) =====")
            return
        
        # matplotlibコンポーネントが利用可能かチェック
        if not hasattr(self, 'preview_axes'):
            print(f"[ERROR] Matplotlib axes not available for drawing particles")
            print(f"[ERROR] This error occurs in drawDetectedParticles method")
            return
        
        # フィルタリング済みの粒子のみを使用
        particles = self.detected_particles
        #print(f"[DEBUG] Drawing {len(particles)} particles (already filtered)")
        
        # 粒子が0個でも境界描画は実行する（フィルタリング前の粒子を使用）
            
        try:
            # 既存の粒子マークと境界線をクリア
            #print(f"[DEBUG] Clearing existing particle markers and boundaries")
            
            # すべての線をクリア（粒子マークと境界線）
            lines_to_remove = []
            for line in self.preview_axes.lines[:]:
                if line.get_marker() == '+' or hasattr(line, '_particle_boundary'):
                    lines_to_remove.append(line)
            
            for line in lines_to_remove:
                line.remove()
            
            # 円形境界（FWHM）をクリア
            patches_to_remove = []
            for patch in self.preview_axes.patches[:]:
                if hasattr(patch, '_particle_boundary'):
                    patches_to_remove.append(patch)
            
            for patch in patches_to_remove:
                patch.remove()
            
            #print(f"[DEBUG] Cleared {len(lines_to_remove)} lines and {len(patches_to_remove)} patches")
            #print(f"[DEBUG] Remaining lines: {len(self.preview_axes.lines)}")
            #print(f"[DEBUG] Remaining patches: {len(self.preview_axes.patches)}")
            
            # スキャンサイズ情報を使用
            scan_size_x = self.scan_size_x
            scan_size_y = self.scan_size_y
            
            # スキャンサイズが0の場合はエラー
            if scan_size_x == 0 or scan_size_y == 0:
                print(f"[ERROR] Scan size is 0, cannot draw particles")
                return
            
            # 物理サイズに基づいた正しいアスペクト比を計算
            data_height, data_width = self.current_frame_data.shape
            pixel_size_x = scan_size_x / data_width
            pixel_size_y = scan_size_y / data_height
            
           
            # detectorの設定を確認し、必要に応じてUIの設定で更新
            if hasattr(self, 'detector'):
                ui_size_filter = self.size_filter_check.isChecked()
                ui_area_method = self.area_method_combo.currentText()
                ui_watershed_compactness = self.watershed_compactness_spin.value()
                ui_watershed_threshold = self.watershed_threshold_spin.value()
                
                #print(f"[DEBUG] UI settings: size_filter={ui_size_filter}, area_method={ui_area_method}")
                #print(f"[DEBUG] Detector settings: size_filter={self.detector.size_filter_enabled}, area_method={self.detector.area_method}")
                
                # 設定が一致していない場合は修正
                if (self.detector.size_filter_enabled != ui_size_filter or
                    self.detector.area_method != ui_area_method):
                    #print(f"[DEBUG] Detector settings mismatch in drawDetectedParticles, updating...")
                    self.detector.size_filter_enabled = ui_size_filter
                    self.detector.area_method = ui_area_method
                    if ui_area_method == "Watershed":
                        self.detector.watershed_compactness = ui_watershed_compactness
                        self.detector.watershed_threshold = ui_watershed_threshold
                    #print(f"[DEBUG] Detector settings updated in drawDetectedParticles")
            
            # 境界描画の条件を修正：サイズフィルターがONの場合に境界を描画
            should_draw_boundaries = (
                hasattr(self, 'detector') and 
                hasattr(self.detector, 'size_filter_enabled') and 
                self.detector.size_filter_enabled
            )
            
            #print(f"[DEBUG] Should draw boundaries: {should_draw_boundaries}")
            
            if should_draw_boundaries:
                #print(f"[DEBUG] Drawing particle boundaries for detected particles")
                # フィルタリング前の粒子を使用して境界を描画
                original_particles = getattr(self, 'original_detected_particles', particles)
                #print(f"[DEBUG]   - original_particles exists: {hasattr(self, 'original_detected_particles')}")
                #print(f"[DEBUG]   - original_particles: {original_particles}")
                #print(f"[DEBUG]   - len(original_particles): {len(original_particles) if original_particles else 0}")
                
                if original_particles and len(original_particles) > 0:
                    #print(f"[DEBUG] Drawing boundaries for {len(original_particles)} original particles")
                    #print(f"[DEBUG] About to call _drawParticleBoundaries...")
                    # 各粒子の座標を確認
                       
                    # 境界描画を実行し、有効な粒子を取得
                    valid_particles_for_markers = self._drawParticleBoundaries(original_particles, pixel_size_x, pixel_size_y)
                    #print(f"[DEBUG] _drawParticleBoundaries completed, returned {len(valid_particles_for_markers)} valid particles")
                    
                    # Watershed処理で有効とされた粒子のみにマーカーを描画
                    if valid_particles_for_markers:
                        #print(f"[DEBUG] Drawing markers for {len(valid_particles_for_markers)} valid particles")
                        particles = valid_particles_for_markers  # マーカー描画用の粒子リストを更新
                    else:
                        #print(f"[DEBUG] No valid particles for markers")
                        particles = []  # マーカーを描画しない

            
            if particles and len(particles) > 0:
                for i, particle in enumerate(particles):
                    # ピクセル座標を物理座標（nm）に変換
                    x_nm = particle.x * pixel_size_x
                    y_nm = particle.y * pixel_size_y
                    
                    #print(f"[DEBUG] Particle {i}: pixel ({particle.x:.1f}, {particle.y:.1f}) -> nm ({x_nm:.1f}, {y_nm:.1f})")
                    
                    # 「+」マークを描画（赤色、太い線）
                    self.preview_axes.plot(x_nm, y_nm, 'r+', markersize=10, markeredgewidth=2, 
                                         label=f'Particle {i+1}' if i == 0 else "")
                

            # キャンバスを更新
            self.preview_canvas.draw()
            #print(f"[DEBUG] Canvas updated")
            #print(f"[DEBUG] ===== drawDetectedParticles EXIT =====")
            
        except Exception as e:
            #print(f"[ERROR] Failed to draw detected particles: {e}")
            import traceback
            traceback.print_exc()
    
    def _drawParticleBoundaries(self, particles, pixel_size_x, pixel_size_y):
        """粒子境界を描画（Area methodに応じてFWHMまたはWatershed）"""
            
        try:
            if not particles:
                #print(f"[DEBUG] No particles to draw boundaries for", flush=True)
                return
            
                       
            area_method = getattr(self.detector, 'area_method', 'FWHM')
                        
            # 最新のcompactness値を確実に取得
            if area_method == "Watershed":
                current_compactness = self.watershed_compactness_spin.value()
                if hasattr(self, 'detector'):
                    self.detector.watershed_compactness = current_compactness
                    #print(f"[DEBUG] Updated detector.watershed_compactness to {current_compactness}")
            
            if area_method == "FWHM":
                #print(f"[DEBUG] Calling _drawFWHMBoundaries", flush=True)
                #print(f"[DEBUG] FWHM processing: {len(particles)} particles")
                result_particles = self._drawFWHMBoundaries(particles, pixel_size_x, pixel_size_y)
                #print(f"[DEBUG] _drawFWHMBoundaries completed, returned {len(result_particles)} particles")
                return result_particles
            elif area_method == "Watershed":
                #print(f"[DEBUG] Calling _drawWatershedBoundaries", flush=True)
                valid_particles = self._drawWatershedBoundaries(particles, pixel_size_x, pixel_size_y)
                #print(f"[DEBUG] Watershed returned {len(valid_particles)} valid particles out of {len(particles)} input particles")
                return valid_particles  # Watershed処理で有効とされた粒子のみを返す
            else:
                print(f"[WARNING] Unknown area method: {area_method}", flush=True)
                #print(f"[DEBUG] Falling back to FWHM", flush=True)
                self._drawFWHMBoundaries(particles, pixel_size_x, pixel_size_y)
                return particles  # FWHMの場合はすべての粒子を返す
                
        except Exception as e:
            print(f"[ERROR] _drawParticleBoundaries failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _drawFWHMBoundaries(self, particles, pixel_size_x, pixel_size_y):
        """FWHM境界線を描画（円形）"""
        #print(f"[DEBUG] ===== _drawFWHMBoundaries ENTRY =====")
        #print(f"[DEBUG] Called with {len(particles)} particles")
        try:
            # 画像の物理サイズを取得
            image_height, image_width = self.current_frame_data.shape
            image_width_phys = image_width * pixel_size_x
            image_height_phys = image_height * pixel_size_y
            
            #print(f"[DEBUG] Image size: {image_width_phys:.1f} x {image_height_phys:.1f} nm")
            
            for i, particle in enumerate(particles):
                #print(f"[DEBUG] Processing FWHM particle {i}: x={particle.x:.1f}, y={particle.y:.1f}")
                # FWHMサイズを計算
                size = self.detector._calculate_particle_size(self.current_frame_data, int(particle.x), int(particle.y))
                #print(f"[DEBUG] FWHM size for particle {i}: {size:.3f} pixels")
                
                # 円の境界を描画
                import matplotlib.patches as patches
                from matplotlib.patches import Circle
                
                # 物理座標に変換
                x_phys = particle.x * pixel_size_x
                y_phys = particle.y * pixel_size_y
                radius_phys = size * pixel_size_x  # 平均的なピクセルサイズを使用
                
                #print(f"[DEBUG] Particle {i} physical coordinates: x={x_phys:.1f}, y={y_phys:.1f}, radius={radius_phys:.1f} nm")
                
                # 円が画像範囲内に収まるかチェック
                circle_left = x_phys - radius_phys
                circle_right = x_phys + radius_phys
                circle_bottom = y_phys - radius_phys
                circle_top = y_phys + radius_phys
                
                #print(f"[DEBUG] Particle {i} circle bounds: left={circle_left:.1f}, right={circle_right:.1f}, bottom={circle_bottom:.1f}, top={circle_top:.1f}")
                #print(f"[DEBUG] Particle {i} image bounds: width={image_width_phys:.1f}, height={image_height_phys:.1f}")
                
                # 円が画像範囲内に完全に収まる場合のみ描画
                if (circle_left >= 0 and circle_right <= image_width_phys and 
                    circle_bottom >= 0 and circle_top <= image_height_phys):
                    
                    # 円を描画（黄色、細い線）
                    circle = Circle((x_phys, y_phys), radius_phys, fill=False, 
                                  edgecolor='yellow', linewidth=1, alpha=0.7)
                    circle._particle_boundary = True  # マークを付けて後で削除できるように
                    self.preview_axes.add_patch(circle)
                    
                    #print(f"[DEBUG] Drew FWHM boundary for particle {i+1} with radius {radius_phys:.1f} nm")
                
            #print(f"[DEBUG] FWHM processing completed: processed {len(particles)} particles")
            return particles  # FWHMの場合はすべての粒子を返す
            
        except Exception as e:
            print(f"[ERROR] Failed to draw FWHM boundaries: {e}")
            import traceback
            traceback.print_exc()
            return []  # エラーの場合は空リストを返す
    
    def _drawWatershedBoundaries(self, particles, pixel_size_x, pixel_size_y):
        """Watershed境界線を描画（particle_analysis方式）"""
        #print(f"[DEBUG] ===== _drawWatershedBoundaries ENTRY =====")
        #print(f"[DEBUG] Called with {len(particles)} particles")
        #print(f"[DEBUG] Current detector.watershed_compactness: {getattr(self.detector, 'watershed_compactness', 'NOT_SET')}")
        try:
            from skimage.segmentation import watershed
            from skimage import measure, filters
            from scipy import ndimage
            
            if not particles:
                return
            
            # 検出時と同じ画像データを使用
            frame_data_for_watershed = self.current_frame_data
            
            # 1. データを適度に平滑化（ノイズ除去）
            sigma = 0.5
            smoothed_data = ndimage.gaussian_filter(frame_data_for_watershed, sigma=sigma)
            
            # 2. より適切な勾配計算（particle_analysis方式）
            from skimage.filters import sobel
            gradient = sobel(smoothed_data)
            
            # 3. ピークマーカーを改善
            improved_markers = np.zeros_like(smoothed_data, dtype=int)
            
            for i, p in enumerate(particles):
                y, x = int(p.y), int(p.x)
                if 0 <= y < improved_markers.shape[0] and 0 <= x < improved_markers.shape[1]:
                    # ピーク周囲の小さな領域を作成（3x3ピクセル）
                    y_min = max(0, y-1)
                    y_max = min(improved_markers.shape[0], y+2)
                    x_min = max(0, x-1)
                    x_max = min(improved_markers.shape[1], x+2)
                    
                    improved_markers[y_min:y_max, x_min:x_max] = i + 1
            
            # 4. より適切なマスク条件（UIから閾値を取得）
            # UIから直接値を取得してdetectorに設定
            current_threshold = self.watershed_threshold_spin.value()
            if hasattr(self, 'detector'):
                self.detector.watershed_threshold = current_threshold
                #print(f"[DEBUG] FORCE UPDATE: detector.watershed_threshold = {current_threshold}")
            
            threshold = np.percentile(smoothed_data, current_threshold)  # UIから取得した閾値
            mask = smoothed_data > threshold
            #print(f"[DEBUG] Watershed threshold parameters:")
            #print(f"[DEBUG]   - UI watershed_threshold_spin.value(): {self.watershed_threshold_spin.value()}")
            #print(f"[DEBUG]   - detector.watershed_threshold: {getattr(self.detector, 'watershed_threshold', 'NOT_SET')}")
            #print(f"[DEBUG]   - calculated threshold: {threshold:.3f}")
            #print(f"[DEBUG]   - mask pixels: {np.sum(mask)}")
            
            # 5. Watershed実行（particle_analysis方式のパラメータ）
            # UIから直接値を取得してdetectorに設定
            current_compactness = self.watershed_compactness_spin.value()
            if hasattr(self, 'detector'):
                self.detector.watershed_compactness = current_compactness
                #print(f"[DEBUG] FORCE UPDATE: detector.watershed_compactness = {current_compactness}")
            
            compactness = current_compactness
            #print(f"[DEBUG] Watershed drawing parameters:")
            #print(f"[DEBUG]   - detector.watershed_compactness: {getattr(self.detector, 'watershed_compactness', 'NOT_SET')}")
            #print(f"[DEBUG]   - UI watershed_compactness_spin.value(): {self.watershed_compactness_spin.value()}")
            #print(f"[DEBUG]   - UI watershed_compactness_spin.isVisible(): {self.watershed_compactness_spin.isVisible()}")
            #print(f"[DEBUG]   - UI watershed_compactness_spin.isEnabled(): {self.watershed_compactness_spin.isEnabled()}")
            #print(f"[DEBUG]   - UI watershed_compactness_spin.signalsBlocked(): {self.watershed_compactness_spin.signalsBlocked()}")
            #print(f"[DEBUG]   - compactness: {compactness}, mask_threshold={threshold:.3f}")
            #print(f"[DEBUG]   - compactness difference: {abs(compactness - self.watershed_compactness_spin.value()):.6f}")
            #print(f"[DEBUG] Improved markers: {np.sum(improved_markers > 0)} markers set")
            
            labels = watershed(gradient, improved_markers, 
                             mask=mask, 
                             compactness=compactness)
            
            # 6. 結果が空でないかチェック（particle_analysis方式のフォールバック）
            if np.sum(labels > 0) == 0:
                print(f"[DEBUG] Watershed filtering failed, trying with relaxed parameters")
                # パラメータを緩和して再試行
                threshold_relaxed = np.percentile(smoothed_data, 70)  # particle_analysisと同じ
                mask_relaxed = smoothed_data > threshold_relaxed
                
                compactness_relaxed = 0.1  # particle_analysisと同じ
                labels = watershed(gradient, improved_markers, 
                                 mask=mask_relaxed, 
                                 compactness=compactness_relaxed)
                #print(f"[DEBUG] Watershed filtering with relaxed parameters: threshold={threshold_relaxed:.3f}, compactness={compactness_relaxed}")
            
            #print(f"[DEBUG] Watershed labels: unique={np.unique(labels)}, shape={labels.shape}")
            
            # 7. 境界描画用の追加チェック：ラベルが少なすぎる場合はパラメータをさらに緩和
            unique_labels = np.unique(labels)
            valid_labels = unique_labels[unique_labels > 0]  # 0以外のラベル
            #print(f"[DEBUG] Valid watershed labels: {valid_labels}")
            #print(f"[DEBUG] Expected labels: {list(range(1, len(particles) + 1))}")
            
            if len(valid_labels) < len(particles) * 0.5:  # 50%未満の粒子しかラベルが生成されていない場合
                #print(f"[DEBUG] Too few watershed labels ({len(valid_labels)} < {len(particles)}), trying with more relaxed parameters")
                # より緩いパラメータで再試行
                threshold_more_relaxed = np.percentile(smoothed_data, 60)  # より低い閾値
                mask_more_relaxed = smoothed_data > threshold_more_relaxed
                
                compactness_more_relaxed = 0.05  # より小さいcompactness
                labels = watershed(gradient, improved_markers, 
                                 mask=mask_more_relaxed, 
                                 compactness=compactness_more_relaxed)
                #print(f"[DEBUG] Watershed filtering with more relaxed parameters: threshold={threshold_more_relaxed:.3f}, compactness={compactness_more_relaxed}")
                #print(f"[DEBUG] New watershed labels: unique={np.unique(labels)}")
            
            # 8. 各粒子の境界を描画し、有効な粒子を記録
            valid_particles = []  # Watershed処理で有効とされた粒子を記録
            
            for i, particle in enumerate(particles):
                label_id = i + 1
                #print(f"[DEBUG] Checking particle {i+1} with label_id={label_id}")
                #print(f"[DEBUG] Label {label_id} in labels: {label_id in labels}")
                
                if label_id in labels:
                    # 境界を検出
                    contour = measure.find_contours(labels == label_id, 0.5)
                    #print(f"[DEBUG] Found {len(contour)} contours for particle {i+1}")
                    
                    if len(contour) > 0:
                        boundary_coords = contour[0]
                        y_coords, x_coords = boundary_coords[:, 0], boundary_coords[:, 1]
                        
                        # 物理座標に変換
                        x_phys = x_coords * pixel_size_x
                        y_phys = y_coords * pixel_size_y
                        
                        # 境界線を描画（黄色、細い線）
                        line, = self.preview_axes.plot(x_phys, y_phys, 'y-', linewidth=1, alpha=0.7)
                        line._particle_boundary = True  # マークを付けて後で削除できるように
                        
                        #print(f"[DEBUG] Drew Watershed boundary for particle {i+1} with {len(boundary_coords)} points")
                        
                        # この粒子を有効として記録
                        valid_particles.append(particle)
 
            # 9. Watershed処理で有効とされた粒子のみを返す（呼び出し元で使用）
            #print(f"[DEBUG] Watershed processing: {len(particles)} input particles, {len(valid_particles)} valid particles")
            return valid_particles
            
        except Exception as e:
            #print(f"[ERROR] Failed to draw Watershed boundaries: {e}")
            import traceback
            traceback.print_exc()
    
    def updateStatistics(self):
        """統計情報の更新"""
        # ケース1: トラッキング完了後 (self.tracks_df が存在する)
        if hasattr(self, 'tracks_df') and self.tracks_df is not None and not self.tracks_df.empty:
            total_tracks = self.tracks_df['particle'].nunique()
            total_particles_in_tracks = len(self.tracks_df)

            # 物理単位への変換係数を計算
            scan_size_x = getattr(gv, 'XScanSize', 1000.0)
            scan_size_y = getattr(gv, 'YScanSize', 1000.0)
            pixel_size = 1.0
            if 0 in self.frame_data_by_frame and self.frame_data_by_frame[0] is not None:
                frame_data = self.frame_data_by_frame[0]
                pixel_size = min(scan_size_x / frame_data.shape[1], scan_size_y / frame_data.shape[0])

            displacements = []
            velocities = []
            for track_id, track in self.tracks_df.groupby('particle'):
                # 総変位 (始点と終点の距離)
                start_pos = track.iloc[0][['x', 'y']].values
                end_pos = track.iloc[-1][['x', 'y']].values
                displacement = np.linalg.norm(end_pos - start_pos) * pixel_size
                displacements.append(displacement)

                # 平均速度 (各ステップの移動距離の平均)
                if len(track) > 1:
                    pos = track[['x', 'y']].values * pixel_size
                    diffs = np.diff(pos, axis=0)
                    step_distances = np.linalg.norm(diffs, axis=1)
                    velocities.append(np.mean(step_distances))

            mean_displacement_val = np.mean(displacements) if displacements else 0.0
            mean_velocity_val = np.mean(velocities) if velocities else 0.0

            self.total_particles_label.setText(str(total_particles_in_tracks))
            self.total_tracks_label.setText(str(total_tracks))
            self.mean_velocity_label.setText(f"{mean_velocity_val:.2f} nm/frame")
            self.mean_displacement_label.setText(f"{mean_displacement_val:.2f} nm")

        # ケース2: 粒子検出のみ完了 (detected_particles が存在する)
        elif hasattr(self, 'detected_particles') and self.detected_particles:
            particle_count = len(self.detected_particles)
            self.total_particles_label.setText(str(particle_count))
            self.total_tracks_label.setText("0")
            self.mean_velocity_label.setText("N/A")
            self.mean_displacement_label.setText("N/A")
        # ケース3: 初期状態
        else:
            self.total_particles_label.setText("0")
            self.total_tracks_label.setText("0")
            self.mean_velocity_label.setText("0.0")
            self.mean_displacement_label.setText("0.0")
    
    def updateTracksTable(self):
        """軌跡テーブルの更新 (trackpy DataFrameベース)"""
        if not hasattr(self, 'tracks_df') or self.tracks_df is None:
            self.tracks_table.setRowCount(0)
            return
        
        # 各軌跡の情報を集計
        track_stats = self.tracks_df.groupby('particle').agg(
            start_frame=('frame', 'min'),
            end_frame=('frame', 'max'),
            duration=('frame', 'count')
        ).reset_index()

        self.tracks_table.setRowCount(len(track_stats))
        
        for i, row in track_stats.iterrows():
            self.tracks_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(int(row['particle']))))
            self.tracks_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(int(row['start_frame']))))
            self.tracks_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(int(row['end_frame']))))
            self.tracks_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(int(row['duration']))))
    
    def exportCSV(self):
        """CSVエクスポート (ワイドフォーマット)"""
        if not hasattr(self, 'tracks_df') or self.tracks_df is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No tracking data to export.")
            return
        
        # デフォルトのフォルダとファイル名を設定
        default_folder = ""
        default_filename = "tracks.csv"
        
        # PARTICLE_ANALYSISと同じ方法でファイル情報を取得
        # 現在選択されているASDファイルの情報を取得
        if (hasattr(gv, 'files') and gv.files and 
            hasattr(gv, 'currentFileNum') and 
            gv.currentFileNum >= 0 and 
            gv.currentFileNum < len(gv.files)):
            
            # 現在選択されているASDファイルのパスとファイル名を取得
            current_file_path = gv.files[gv.currentFileNum]
            current_file_dir = os.path.dirname(current_file_path)
            current_file_name = os.path.splitext(os.path.basename(current_file_path))[0]
            
            # デフォルトの保存ファイル名を設定
            default_filename = f"{current_file_name}_Tracks.csv"
            default_path = os.path.join(current_file_dir, default_filename)
        else:
            default_path = default_filename
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", default_path, "CSV files (*.csv)"
        )
        
        if filename:
            try:
                # フレーム時間を取得（デフォルトは0.1秒）
                frame_time = getattr(gv, 'frame_time', 0.1)
                
                # ワイドフォーマットのデータを作成
                wide_data = self._create_wide_format_data(frame_time)
                
                # CSVファイルに保存
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    import csv
                    writer = csv.writer(f)
                    
                    # ヘッダー行を書き込み
                    for row in wide_data['headers']:
                        writer.writerow(row)
                    
                    # データ行を書き込み
                    for row in wide_data['data']:
                        writer.writerow(row)
                
                QtWidgets.QMessageBox.information(
                    self, "Export Complete",
                    f"Wide format data exported to {filename}"
                )
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Error",
                    f"Error during export:\n{e}"
                )
    

    
    def _create_wide_format_data(self, frame_time):
        """ワイドフォーマットのデータを作成"""
        # 全トラックを取得
        all_tracks = sorted(self.tracks_df['particle'].unique())
        
        # 各トラックの開始時間と終了時間を計算
        track_times = {}
        for track_id in all_tracks:
            track_data = self.tracks_df[self.tracks_df['particle'] == track_id]
            start_frame = track_data['frame'].min()
            end_frame = track_data['frame'].max()
            track_times[track_id] = {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_frame * frame_time,
                'end_time': end_frame * frame_time
            }
        
        # 全フレーム範囲を計算（すべてのトラックをカバー）
        all_start_frames = [track_times[track_id]['start_frame'] for track_id in all_tracks]
        all_end_frames = [track_times[track_id]['end_frame'] for track_id in all_tracks]
        global_start_frame = min(all_start_frames) if all_start_frames else 0
        global_end_frame = max(all_end_frames) if all_end_frames else 0
        all_frames = list(range(global_start_frame, global_end_frame + 1))
        
        # ヘッダー行を作成
        headers = []
        
        # 1行目: トラックIDヘッダー
        track_header = ['', '']  # A列、B列は空白
        for track_id in all_tracks:
            track_header.extend([f'Track {track_id}', '', ''])  # C,D列とE列（空白）
        headers.append(track_header)
        
        # 2行目: データ種別ヘッダー
        data_header = ['Time (s)', 'Frame']  # A列、B列
        for track_id in all_tracks:
            data_header.extend(['X (nm)', 'Y (nm)', ''])  # C,D列とE列（空白）
        headers.append(data_header)
        
        # データ行を作成
        data_rows = []
        for frame in all_frames:
            # 時間を適切な精度でフォーマット（浮動小数点問題を回避）
            time_seconds = round(frame * frame_time, 2)
            row = [time_seconds, frame]  # A列、B列
            
            for track_id in all_tracks:
                # このフレームでこのトラックのデータがあるかチェック
                track_data = self.tracks_df[
                    (self.tracks_df['frame'] == frame) & 
                    (self.tracks_df['particle'] == track_id)
                ]
                
                if not track_data.empty:
                    # データがある場合（最初のデータを使用）
                    x_coord = track_data.iloc[0]['x']
                    y_coord = track_data.iloc[0]['y']
                    
                    # ピクセル座標を物理座標（nm）に変換
                    if hasattr(gv, 'XScanSize') and hasattr(gv, 'YScanSize') and hasattr(gv, 'XPixel') and hasattr(gv, 'YPixel'):
                        nm_per_pixel_x = gv.XScanSize / gv.XPixel
                        nm_per_pixel_y = gv.YScanSize / gv.YPixel
                        x_nm = x_coord * nm_per_pixel_x
                        y_nm = y_coord * nm_per_pixel_y
                    else:
                        # 物理サイズ情報がない場合は、そのまま使用
                        x_nm = x_coord
                        y_nm = y_coord
                    
                    row.extend([f"{x_nm:.2f}", f"{y_nm:.2f}", ''])  # C,D列とE列（空白）
                else:
                    # データがない場合
                    row.extend(['', '', ''])  # C,D列とE列（空白）
            
            data_rows.append(row)
        
        return {
            'headers': headers,
            'data': data_rows
        }
    
    def exportImage(self):
        """画像エクスポート（後方互換性のため残存、現在はクリップボードコピーに変更）"""
        self.copyImageToClipboard()
    
    def copyImageToClipboard(self):
        """現在の表示をクリップボードにコピー"""
        try:
            print("[DEBUG] Starting copyImageToClipboard")
            
            # キャンバスを画像として取得
            self.preview_canvas.draw()
            print("[DEBUG] Canvas drawn")
            
            # キャンバスからQPixmapを取得
            canvas_pixmap = self.preview_canvas.grab()
            print(f"[DEBUG] Canvas grabbed, pixmap size: {canvas_pixmap.size()}")
            
            # QPixmapをQImageに変換
            canvas_image = canvas_pixmap.toImage()
            print(f"[DEBUG] Converted to QImage, size: {canvas_image.size()}")
            
            # クリップボードにコピー
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setImage(canvas_image)
            print("[DEBUG] Image set to clipboard")
            
            # 成功メッセージを表示
            QtWidgets.QMessageBox.information(
                self, "Success", 
                "Image copied to clipboard!\n\n"
                "You can now paste it in other applications using Ctrl+V."
            )
            print("[DEBUG] Success message displayed")
            
        except Exception as e:
            print(f"[ERROR] Failed to copy image to clipboard: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to copy image to clipboard: {e}")
            import traceback
            traceback.print_exc()
    
    def previousFrame(self):
        """前のフレーム"""
        try:
            if not hasattr(self, 'particles_by_frame') or not self.particles_by_frame:
                QtWidgets.QMessageBox.information(self, "Information", 
                    "Please run 'Detect All Frames' first.\n先に'Detect All Frames'を実行してください。")
                return
                
            current_frame = getattr(gv, 'index', 0)
            if current_frame > 0:
                # 前のフレームに移動
                new_frame = current_frame - 1
                gv.index = new_frame
                
                # メインウィンドウのフレームスライダーも更新
                if hasattr(self.main_window, 'frameSlider'):
                    self.main_window.frameSlider.setValue(new_frame)
                if hasattr(self.main_window, 'updateFrame'):
                    self.main_window.updateFrame()
                
                # データを再初期化
                if self.initializeData():
                    # 粒子データを設定
                    if new_frame in self.particles_by_frame:
                        self.detected_particles = self.particles_by_frame[new_frame]
                        #print(f"DEBUG: Set {len(self.detected_particles)} particles for frame {new_frame}")
                    else:
                        self.detected_particles = []
                        #print(f"DEBUG: No particle data for frame {new_frame}")
                    
                    # プレビューとラベルを更新
                    self.updateFrameLabel()
                    self.updatePreview()
                    self.updateBottomLabels()
                    #print(f"DEBUG: Moved to previous frame: {new_frame}")
                  
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", 
                f"Error moving to previous frame: {str(e)}\n前のフレームへの移動中にエラーが発生しました: {str(e)}")
    
    def nextFrame(self):
        """次のフレーム"""
        try:
            if not hasattr(self, 'particles_by_frame') or not self.particles_by_frame:
                QtWidgets.QMessageBox.information(self, "Information", 
                    "Please run 'Detect All Frames' first.\n先に'Detect All Frames'を実行してください。")
                return
                
            current_frame = getattr(gv, 'index', 0)
            if hasattr(gv, 'FrameNum') and current_frame < gv.FrameNum - 1:
                # 次のフレームに移動
                new_frame = current_frame + 1
                gv.index = new_frame
                
                # メインウィンドウのフレームスライダーも更新
                if hasattr(self.main_window, 'frameSlider'):
                    self.main_window.frameSlider.setValue(new_frame)
                if hasattr(self.main_window, 'updateFrame'):
                    self.main_window.updateFrame()
                
                # データを再初期化
                if self.initializeData():
                    # 粒子データを設定
                    if new_frame in self.particles_by_frame:
                        self.detected_particles = self.particles_by_frame[new_frame]
                        #print(f"DEBUG: Set {len(self.detected_particles)} particles for frame {new_frame}")
                    else:
                        self.detected_particles = []
                        #print(f"DEBUG: No particle data for frame {new_frame}")
                    
                    # プレビューとラベルを更新
                    self.updateFrameLabel()
                    self.updatePreview()
                    self.updateBottomLabels()
                    #print(f"DEBUG: Moved to next frame: {new_frame}")
              
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", 
                f"Error moving to next frame: {str(e)}\n次のフレームへの移動中にエラーが発生しました: {str(e)}")
    
    def onFrameChanged(self, value):
        """フレーム変更時の処理"""
        #print(f"DEBUG: onFrameChanged called with value: {value}")
        
        # 現在のフレームインデックスを更新
        self.current_frame_index = value
        
        # グローバル変数を更新
        gv.index = value
        
        # データを再取得
        self.initializeData()
        
        # All Frames実行後の場合は、保存された粒子データを使用
        if hasattr(self, 'particles_by_frame') and self.particles_by_frame:
            if value in self.particles_by_frame:
                particles = self.particles_by_frame[value]
                print(f"DEBUG: Found {len(particles)} particles for frame {value}")
                self.detected_particles = particles
            else:
                print(f"DEBUG: No particle data for frame {value}")
                self.detected_particles = []
        else:
            print(f"DEBUG: No particles_by_frame data available")
            self.detected_particles = []
        
        self.updateFrameLabel()
        self.updatePreview()
        self.updateBottomLabels()
    
    def updateFrameLabel(self):
        """フレームラベルの更新"""
        if hasattr(gv, 'FrameNum') and gv.FrameNum > 0:
            current_frame = getattr(gv, 'index', 0) + 1
            self.bottom_frame_label.setText(f"Frame: {current_frame}/{gv.FrameNum}")
        else:
            self.bottom_frame_label.setText("Frame: 0/0")
    
    def updateBottomLabels(self):
        """画像下のラベルの更新"""
        if hasattr(gv, 'FrameNum') and gv.FrameNum > 0:
            current_frame = getattr(gv, 'index', 0) + 1
            # フレーム情報
            self.bottom_frame_label.setText(f"Frame: {current_frame}/{gv.FrameNum}")
            
            # 粒子数情報
            current_particles = self.particles_by_frame.get(getattr(gv, 'index', 0), [])
            particle_count = len(current_particles)
            self.particle_count_label.setText(f"Particles: {particle_count}")
            
            # ボタンの有効/無効を設定（All Frames実行後のみ有効）
            if hasattr(self, 'particles_by_frame') and self.particles_by_frame:
                self.prev_button.setEnabled(getattr(gv, 'index', 0) > 0)
                self.next_button.setEnabled(getattr(gv, 'index', 0) < gv.FrameNum - 1)
                # 画像下のスライダーの値も更新
                self.bottom_frame_slider.setValue(getattr(gv, 'index', 0))
                #print(f"DEBUG: Frame {getattr(gv, 'index', 0)}, prev enabled: {getattr(gv, 'index', 0) > 0}, next enabled: {getattr(gv, 'index', 0) < gv.FrameNum - 1}")
            else:
                # All Frames実行前はボタンを無効化
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
                #print(f"DEBUG: All Frames not executed yet, buttons disabled")
        else:
            self.bottom_frame_label.setText("Frame: 0/0")
            self.particle_count_label.setText("Particles: 0")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
    
    def loadWindowSettings(self):
        """ウィンドウ設定の読み込み"""
        if 'ParticleTrackingWindow' in gv.windowSettings:
            settings = gv.windowSettings['ParticleTrackingWindow']
            self.setGeometry(settings['x'], settings['y'], settings['width'], settings['height'])
            
            # 表示設定を読み込み
            if 'show_tracks' in settings:
                self.show_tracks_check.setChecked(settings['show_tracks'])
            if 'show_track_ids' in settings:
                self.show_track_ids_check.setChecked(settings['show_track_ids'])
    
    def saveWindowSettings(self):
        """ウィンドウ設定の保存"""
        if 'ParticleTrackingWindow' not in gv.windowSettings:
            gv.windowSettings['ParticleTrackingWindow'] = {}
        
        geometry = self.geometry()
        gv.windowSettings['ParticleTrackingWindow'].update({
            'x': geometry.x(),
            'y': geometry.y(),
            'width': geometry.width(),
            'height': geometry.height(),
            'show_tracks': self.show_tracks_check.isChecked(),
            'show_track_ids': self.show_track_ids_check.isChecked()
        })
    
    def closeEvent(self, event):
        """ウィンドウが閉じられる時の処理"""
        self.saveWindowSettings()
        
        # プラグインとして開いた場合のツールバーアクションのハイライトを解除
        try:
            if hasattr(self, 'main_window') and self.main_window is not None:
                if hasattr(self.main_window, 'setActionHighlight') and hasattr(self.main_window, 'plugin_actions'):
                    act = self.main_window.plugin_actions.get("Particle Tracking")
                    if act is not None:
                        self.main_window.setActionHighlight(act, False)
        except Exception as e:
            print(f"[WARNING] Failed to reset particle tracking action highlight: {e}")
        
        event.accept() 

    def onPreprocessingChanged(self):
        """前処理が変更された時の処理（メインウィンドウからの通知 + 独自の前処理）"""
        if not getattr(self, 'ui_initialized', False): return  # <-- ★ 3. この行を追加
        try:
            # メインウィンドウの前処理済みデータを取得
            if hasattr(gv, 'aryData') and gv.aryData is not None:
                # メインウィンドウで適用された前処理済みデータをベースとして使用
                base_data = gv.aryData.copy()
                
                # Particle trackingウィンドウ内のpreprocessing（スムージング）を適用
                processed_data = self.preprocessData(base_data)
                
                if processed_data is not None:
                    self.current_frame_data = processed_data
                    
                    # プレビューを更新
                    self.updatePreview()
                    
                    # 検出済み粒子があれば再描画
                    if hasattr(self, 'detected_particles') and self.detected_particles:
                        self.drawDetectedParticles()
                    

                    
            else:
                print(f"[ERROR] No data available in gv.aryData")
                
        except Exception as e:
            print(f"[ERROR] Failed to update particle tracking with preprocessing: {e}")
            import traceback
            traceback.print_exc()
    
    def onMainWindowPreprocessingChanged(self):
        """メインウィンドウの前処理変更を通知するメソッド（外部から呼び出し可能）"""
        self.onPreprocessingChanged()
    
    def onMainWindowFileChanged(self):
        """メインウィンドウのファイル変更を通知するメソッド（外部から呼び出し可能）"""
        try:
            # データをリセット
            self.particles_by_frame = {}
            self.frame_data_by_frame = {}
            self.tracks_df = None
            self.detected_particles = []
            
            # データを再初期化
            self.initializeData()
            
            # プレビューを更新
            self.updatePreview()
            
            # フレームラベルを更新
            self.updateFrameLabel()
            
            # 下部ラベルを更新
            self.updateBottomLabels()
            
            # 統計情報をリセット
            self.updateStatistics()
            
            # トラッキングテーブルをリセット
            self.updateTracksTable()
            
            # トラッキングボタンを無効化
            self.track_button.setEnabled(False)
            
            # エクスポートボタンを無効化
            self.export_csv_button.setEnabled(False)
            self.edit_tracks_button.setEnabled(False)
            self.track_analysis_button.setEnabled(False)
            # export_image_buttonは削除済み（クリップボードコピーに変更）
            
        except Exception as e:
            print(f"[ERROR] Failed to reset particle tracking window: {e}")
            import traceback
            traceback.print_exc()
    
    def updatePreprocessingUI(self):
        """前処理UIの表示/非表示を更新"""
        try:
            # スムージング方法に応じてパラメーターの設定と表示/非表示を切り替え
            smooth_method = self.smooth_combo.currentText()
            
            if smooth_method == "Gaussian":
                # Gaussianが選択された場合、Gaussian Sigmaを表示
                self.smooth_param_label.setText("Gaussian Sigma:")
                self.smooth_param_spin.setDecimals(1)
                self.smooth_param_spin.setRange(0.1, 3.0)
                self.smooth_param_spin.setSingleStep(0.1)
                # 現在の値が範囲外の場合は適切な値に調整
                current_value = self.smooth_param_spin.value()
                if current_value < 0.1 or current_value > 3.0:
                    self.smooth_param_spin.setValue(1.0)
                # スピンボックスを有効化して確実に操作可能にする
                self.smooth_param_spin.setEnabled(True)
                self.smooth_param_label.setVisible(True)
                self.smooth_param_spin.setVisible(True)
                
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
                
            else:
                # Noneが選択された場合、非表示・無効化
                self.smooth_param_label.setVisible(False)
                self.smooth_param_spin.setVisible(False)
                self.smooth_param_spin.setEnabled(False)
                
        except Exception as e:
            print(f"[ERROR] Failed to update preprocessing UI: {e}")
            import traceback
            traceback.print_exc()
    
    def onSmoothingMethodChanged(self, method):
        """スムージング方法が変更された時の処理"""
        self.updatePreprocessingUI()
        self.onPreprocessingChanged()
    
    def onSmoothingParameterChanged(self, value):
        """スムージングパラメーターが変更された時の処理"""
        # パラメーターが変更された場合は前処理を更新
        self.onPreprocessingChanged()
    
    def preprocessData(self, data):
        """データの前処理（スムージングのみ）"""
        try:
            if data is None:
                print(f"[ERROR] No input data provided")
                return None
            
            # 入力データをコピー
            processed_data = data.copy()
            
            # スキャンサイズ情報も取得
            self.scan_size_x = getattr(gv, 'XScanSize', 0)
            self.scan_size_y = getattr(gv, 'YScanSize', 0)
            self.x_pixels = getattr(gv, 'XPixel', 0)
            self.y_pixels = getattr(gv, 'YPixel', 0)
                     
            # Smoothing（スムージングのみ）
            smooth_method = self.smooth_combo.currentText()
            if smooth_method == "Gaussian":
                sigma = self.smooth_param_spin.value()
                from scipy.ndimage import gaussian_filter
                processed_data = gaussian_filter(processed_data, sigma=sigma)
                #print(f"[DEBUG] Applied Gaussian smoothing with sigma={sigma}")
                
            elif smooth_method == "Median":
                # NxNサイズを直接使用
                size = int(self.smooth_param_spin.value())
                from scipy.ndimage import median_filter
                processed_data = median_filter(processed_data, size=size)
                #print(f"[DEBUG] Applied Median filtering with size={size}")
            
            
            return processed_data
            
        except Exception as e:
            print(f"[ERROR] Failed to preprocess data: {e}")
            import traceback
            traceback.print_exc()
            return data
    
    def onDetectionParameterChanged(self):
        """検出パラメータ変更時の自動再検出"""
        if not getattr(self, 'ui_initialized', False): return  # <-- ★ 3. この行を追加
        #print(f"[DEBUG] onDetectionParameterChanged called")
        #print(f"[DEBUG] UI threshold value: {self.threshold_spin.value()}")
        
        # UIの値をSpotDetectorに反映
        if hasattr(self, 'detector'):
            old_threshold = self.detector.threshold
            self.detector.radius = self.radius_spin.value()
            self.detector.threshold = self.threshold_spin.value()
            self.detector.min_distance = self.min_distance_spin.value()
            self.detector.do_subpixel = self.subpixel_check.isChecked()
            self.detector.max_position_correction = self.max_pos_corr_check.isChecked()
            self.detector.min_intensity_ratio = self.min_intensity_spin.value()
            self.detector.size_filter_enabled = self.size_filter_check.isChecked()
            self.detector.size_tolerance = self.size_tolerance_spin.value()
            self.detector.selection_radius = self.selection_radius_spin.value()
            self.detector.area_method = self.area_method_combo.currentText()
            
            #print(f"[DEBUG] Before setting watershed_compactness:")
            #print(f"[DEBUG]   - UI watershed_compactness_spin.value(): {self.watershed_compactness_spin.value()}")
            #print(f"[DEBUG]   - detector.watershed_compactness: {getattr(self.detector, 'watershed_compactness', 'NOT_SET')}")
            #print(f"[DEBUG]   - hasattr(detector, 'watershed_compactness'): {hasattr(self.detector, 'watershed_compactness')}")
            #print(f"[DEBUG]   - detector type: {type(self.detector)}")
            #print(f"[DEBUG]   - detector class: {self.detector.__class__.__name__}")
            
            # 強制的に値を設定
            new_compactness = self.watershed_compactness_spin.value()
            self.detector.watershed_compactness = new_compactness
            #print(f"[DEBUG] FORCE SET detector.watershed_compactness = {new_compactness}")
            
            #print(f"[DEBUG] After setting watershed_compactness:")
            #print(f"[DEBUG]   - detector.watershed_compactness: {getattr(self.detector, 'watershed_compactness', 'NOT_SET')}")
            #print(f"[DEBUG]   - hasattr(detector, 'watershed_compactness'): {hasattr(self.detector, 'watershed_compactness')}")
            
            #print(f"[DEBUG] UI state check:")
            #print(f"[DEBUG]   - size_filter_check.isChecked(): {self.size_filter_check.isChecked()}")
            #print(f"[DEBUG]   - selection_radius_spin.value(): {self.selection_radius_spin.value()}")
            #print(f"[DEBUG]   - area_method_combo.currentText(): {self.area_method_combo.currentText()}")
            #print(f"[DEBUG] Updated detector parameters:")
            #print(f"[DEBUG]   - radius: {self.detector.radius}")
            #print(f"[DEBUG]   - threshold: {self.detector.threshold}")
            #print(f"[DEBUG]   - min_distance: {self.detector.min_distance}")
            #print(f"[DEBUG]   - subpixel: {self.detector.do_subpixel}")
            #print(f"[DEBUG]   - max_pos_corr: {self.detector.max_position_correction}")
            #print(f"[DEBUG]   - min_intensity_ratio: {self.detector.min_intensity_ratio}")
            #print(f"[DEBUG]   - size_filter_enabled: {self.detector.size_filter_enabled}")
            #print(f"[DEBUG]   - size_tolerance: {self.detector.size_tolerance}")
            #print(f"[DEBUG]   - selection_radius: {self.detector.selection_radius}")
            #print(f"[DEBUG]   - area_method: {self.detector.area_method}")
            #print(f"[DEBUG]   - watershed_compactness: {self.detector.watershed_compactness}")
            #print(f"[DEBUG]   - UI checkbox state: {self.max_pos_corr_check.isChecked()}")

        # パラメータ変更時は常に再検出を実行
        #print(f"[DEBUG] ===== onDetectionParameterChanged: Starting auto-re-detection =====")
        #print(f"[DEBUG] Detection parameter changed, auto-re-detecting particles")
        self.detectParticles()
        #print(f"[DEBUG] ===== onDetectionParameterChanged: Auto-re-detection completed =====")
    
    def onAreaMethodChanged(self):
        """Area methodが変更された時の処理"""
        try:
            method = self.area_method_combo.currentText()
            #print(f"[DEBUG] Area method changed to: {method}")
            #print(f"[DEBUG] Before update - detector.area_method: {getattr(self.detector, 'area_method', 'NOT_SET')}")
            
            # Watershedパラメータの表示/非表示を切り替え
            if method == "Watershed":
                self.watershed_compactness_label.setVisible(True)
                self.watershed_compactness_spin.setVisible(True)
                self.watershed_threshold_label.setVisible(True)
                self.watershed_threshold_spin.setVisible(True)
                # Watershed選択時にシグナル接続を再設定
                #print(f"[DEBUG] Watershed selected, reconnecting signals...")
                try:
                    self.watershed_compactness_spin.valueChanged.disconnect()
                    #print(f"[DEBUG] Disconnected existing watershed_compactness_spin signals")
                except:
                    pass
                    #print(f"[DEBUG] No existing watershed_compactness_spin signals to disconnect")
                
                try:
                    self.watershed_threshold_spin.valueChanged.disconnect()
                    #print(f"[DEBUG] Disconnected existing watershed_threshold_spin signals")
                except:
                    pass
                    #print(f"[DEBUG] No existing watershed_threshold_spin signals to disconnect")
                
                self.watershed_compactness_spin.valueChanged.connect(self.onWatershedCompactnessChanged)
                self.watershed_threshold_spin.valueChanged.connect(self.onWatershedThresholdChanged)
                
            else:
                self.watershed_compactness_label.setVisible(False)
                self.watershed_compactness_spin.setVisible(False)
                self.watershed_threshold_label.setVisible(False)
                self.watershed_threshold_spin.setVisible(False)
            
            # 検出を実行
            self.onDetectionParameterChanged()
            
            #print(f"[DEBUG] After update - detector.area_method: {getattr(self.detector, 'area_method', 'NOT_SET')}")
            
        except Exception as e:
            print(f"[ERROR] onAreaMethodChanged failed: {e}")
            import traceback
            traceback.print_exc()
    
    def clearParticleBoundaries(self):
        """粒子境界のみをクリア（マーカーは残す）"""
        try:
            # 境界線のみをクリア
            lines_to_remove = []
            for line in self.preview_axes.lines[:]:
                if hasattr(line, '_particle_boundary'):
                    lines_to_remove.append(line)
            
            for line in lines_to_remove:
                line.remove()
            
            # 円形境界をクリア
            patches_to_remove = []
            for patch in self.preview_axes.patches[:]:
                if hasattr(patch, '_particle_boundary'):
                    patches_to_remove.append(patch)
            
            for patch in patches_to_remove:
                patch.remove()
                
            #print(f"[DEBUG] Cleared {len(lines_to_remove)} boundary lines and {len(patches_to_remove)} boundary patches")
            
        except Exception as e:
            print(f"[ERROR] clearParticleBoundaries failed: {e}")

    def updateParticleBoundariesOnly(self):
        """粒子境界のみを更新（検出は再実行しない）"""
        try:
            if not hasattr(self, 'preview_axes'):
                return
                
            # 既存の境界線のみをクリア
            self.clearParticleBoundaries()
            
            # スキャンサイズ情報を取得
            scan_size_x = self.scan_size_x
            scan_size_y = self.scan_size_y
            
            if scan_size_x == 0 or scan_size_y == 0:
                return
                
            # ピクセルサイズを計算
            data_height, data_width = self.current_frame_data.shape
            pixel_size_x = scan_size_x / data_width
            pixel_size_y = scan_size_y / data_height
            
            # 境界描画の条件をチェック
            should_draw_boundaries = (
                hasattr(self, 'detector') and 
                (getattr(self.detector, 'size_filter_enabled', False) or 
                 getattr(self.detector, 'area_method', 'FWHM') == 'Watershed')
            )
            
            if should_draw_boundaries:
                # フィルタリング前の粒子を確実に使用
                particles_for_boundary = None
                
                # 1. まず original_detected_particles を試す
                if hasattr(self, 'original_detected_particles') and self.original_detected_particles:
                    particles_for_boundary = self.original_detected_particles
                    #print(f"[DEBUG] Using original_detected_particles: {len(particles_for_boundary)} particles")
                    
                # 2. 次に detected_particles を試す
                elif hasattr(self, 'detected_particles') and self.detected_particles:
                    particles_for_boundary = self.detected_particles
                    #print(f"[DEBUG] Using detected_particles: {len(particles_for_boundary)} particles")
                
                if particles_for_boundary:
                    self._drawParticleBoundaries(particles_for_boundary, pixel_size_x, pixel_size_y)
            
            # キャンバスを更新
            self.preview_canvas.draw()
            
        except Exception as e:
            print(f"[ERROR] updateParticleBoundariesOnly failed: {e}")

    def onWatershedCompactnessChanged(self):
        """Watershed compactnessが変更された時の処理（境界のみ更新）"""
      
        try:
            new_value = self.watershed_compactness_spin.value()
                 
            # 検出器のパラメータを更新
            if hasattr(self, 'detector'):
                self.detector.watershed_compactness = new_value
                #  print(f"[DEBUG] Updated detector.watershed_compactness = {new_value}")
            
            # 境界描画のみを更新（検出は再実行しない）
            if hasattr(self, 'detected_particles') or hasattr(self, 'original_detected_particles'):
                #print(f"[DEBUG] Updating particle boundaries only...")
                self.updateParticleBoundariesOnly()
            
                #print(f"[DEBUG] No particles found, skipping boundary update")
            
            #print(f"[DEBUG] After update - detector.watershed_compactness: {getattr(self.detector, 'watershed_compactness', 'NOT_SET')}")
            #print(f"[DEBUG] After update - hasattr(detector, 'watershed_compactness'): {hasattr(self.detector, 'watershed_compactness')}")
            #print(f"[DEBUG] ===== Watershed compactness change completed =====")
            
        except Exception as e:
            print(f"[ERROR] onWatershedCompactnessChanged failed: {e}")
            import traceback
            traceback.print_exc()

    def onWatershedThresholdChanged(self):
        """Watershed thresholdが変更された時の処理（境界のみ更新）"""

        try:
            new_value = self.watershed_threshold_spin.value()
                      
            # 検出器のパラメータを更新
            if hasattr(self, 'detector'):
                self.detector.watershed_threshold = new_value
                #print(f"[DEBUG] Updated detector.watershed_threshold = {new_value}")
            
            # 境界描画のみを更新（検出は再実行しない）
            if hasattr(self, 'detected_particles') or hasattr(self, 'original_detected_particles'):
                #print(f"[DEBUG] Updating particle boundaries only...")
                self.updateParticleBoundariesOnly()
                #print(f"[DEBUG] Particle boundaries updated")
                      
           
        except Exception as e:
            #print(f"[ERROR] onWatershedThresholdChanged failed: {e}")
            import traceback
            traceback.print_exc()
    
    def applyDetectionPreprocessing(self, image):
        """粒子検出で適用される前処理を実行
        注意: このメソッドは粒子検出時のみ使用され、プレビュー表示には使用されません
        """
        processed_image = image.copy()
             
        # データの正規化（負の値を含む場合）
        original_min = np.min(processed_image)
        original_max = np.max(processed_image)
        
        if original_min < 0:
            # 負の値を0にシフト
            processed_image = processed_image - original_min
            #print(f"[DEBUG] Shifted data by {original_min:.3f} to remove negative values")
        
        #print(f"[DEBUG] Data range after normalization: {np.min(processed_image):.3f} to {np.max(processed_image):.3f}")
        
        # ユーザーが要求した場合のみ追加の前処理を適用
        if hasattr(self, 'median_filter_check') and self.median_filter_check.isChecked():
            # メディアンフィルターのサイズを小さくしてエッジを保持
            kernel_size = 3
            # 浮動小数点データのまま処理
            from scipy.ndimage import median_filter
            processed_image = median_filter(processed_image, size=kernel_size)
            #print(f"[DEBUG] Applied median filter with kernel size {kernel_size}")
       
        if hasattr(self, 'background_subtraction_check') and self.background_subtraction_check.isChecked():
            # より小さなカーネルで背景除去
            radius = getattr(self, 'radius_spin', None)
            if radius:
                radius_value = radius.value()
                kernel_size = max(5, int(radius_value * 2))
            else:
                kernel_size = 15
            if kernel_size % 2 == 0:
                kernel_size += 1
            # 浮動小数点データのまま処理
            from scipy.ndimage import gaussian_filter
            background = gaussian_filter(processed_image, sigma=kernel_size/3)
            processed_image = processed_image - background
            processed_image = np.clip(processed_image, 0, None)
             
        # Enhanced noise reduction（追加のノイズ除去）
        if hasattr(self, 'noise_reduction_check') and self.noise_reduction_check.isChecked():
            # バイラテラルフィルターの代わりにガウシアンフィルターを使用
            from scipy.ndimage import gaussian_filter
            processed_image = gaussian_filter(processed_image, sigma=0.5)
            #print(f"[DEBUG] Applied enhanced noise reduction (gaussian filter)")

        #print(f"[DEBUG] Output data range: {np.min(processed_image):.3f} to {np.max(processed_image):.3f}")
        return processed_image
    
    def onBottomFrameSliderChanged(self, value):
        """画像下のフレームスライダーが変更された時の処理"""
        try:
            # グローバル変数のフレームインデックスを更新
            gv.index = value
            
            # メインウィンドウのフレームスライダーも更新
            if hasattr(self.main_window, 'frameSlider'):
                self.main_window.frameSlider.setValue(value)
            
            # メインウィンドウのフレームを更新
            if hasattr(self.main_window, 'updateFrame'):
                self.main_window.updateFrame()
            
            # データを再初期化
            if self.initializeData():
                # All Frames実行後の場合は、保存された粒子データと画像データを使用
                if hasattr(self, 'particles_by_frame') and self.particles_by_frame:
                    
                    # 粒子データが存在するかチェック
                    if value in self.particles_by_frame:
                        particles = self.particles_by_frame[value]
                        #print(f"DEBUG: Found {len(particles)} particles for frame {value}")
                        # 現在のフレームの粒子データを設定（drawDetectedParticlesで使用）
                        self.detected_particles = particles
                        
                            #print(f"DEBUG: First particle in frame {value}: x={particles[0].x:.2f}, y={particles[0].y:.2f}")
                    else:
                        #print(f"DEBUG: No particle data for frame {value}")
                        self.detected_particles = []
                    
                    # 保存された画像データを復元（重要！）
                    if hasattr(self, 'frame_data_by_frame') and value in self.frame_data_by_frame:
                        saved_frame_data = self.frame_data_by_frame[value]
                        if saved_frame_data is not None:
                            self.current_frame_data = saved_frame_data.copy()
                            #print(f"DEBUG: Restored frame {value} data with shape {saved_frame_data.shape}")
                else:
                    # All Frames実行前の場合は、空のリストを設定
                    #print(f"DEBUG: No particles_by_frame data available")
                    self.detected_particles = []
                
                # プレビューとラベルを更新
                self.updateFrameLabel()
                self.updatePreview()
                self.updateBottomLabels()
                #print(f"DEBUG: Bottom frame slider changed to frame {value}")
            
                
        except Exception as e:
            print(f"DEBUG: Error in onBottomFrameSliderChanged: {e}")
            import traceback
            traceback.print_exc()
    

    
    def _track_with_scipy_lap(self):
        """SciPyのLAPソルバーを使用してトラッキング（infeasibleエラー対策済み）"""
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        #print(f"[DEBUG] === Simple LAP Tracker Started ===")
        
        # パラメータ取得
        max_distance_nm = self.max_distance_spin.value()
        max_frame_gap = self.max_frame_gap_spin.value()
        min_track_length = self.min_track_length_spin.value()
        
        #print(f"[DEBUG] Parameters: max_distance={max_distance_nm}nm, max_gap={max_frame_gap}, min_length={min_track_length}")

        # ピクセルサイズ計算と距離閾値の設定（2乗のまま扱う）
        scan_size_x = getattr(gv, 'XScanSize', 1000.0); scan_size_y = getattr(gv, 'YScanSize', 1000.0)
        frame_data = self.frame_data_by_frame[0]
        pixel_size_x = scan_size_x / frame_data.shape[1]; pixel_size_y = scan_size_y / frame_data.shape[0]
        max_dist_sq_px = (max_distance_nm / min(pixel_size_x, pixel_size_y)) ** 2
        
        #print(f"[DEBUG] Pixel sizes: x={pixel_size_x:.6f}, y={pixel_size_y:.6f} nm/pixel")
        #print(f"[DEBUG] Max distance squared: {max_dist_sq_px:.2f} pixels²")

        self.active_tracks = []
        self.next_track_id = 0
        
        sorted_frames = sorted(self.particles_by_frame.keys())
        total_frames = len(sorted_frames)
        #print(f"[DEBUG] Processing {len(sorted_frames)} frames: {sorted_frames}")
        
        total_particles_processed = 0
        total_links_made = 0
        total_new_tracks = 0
        
        for i, frame_idx in enumerate(sorted_frames):
            # キャンセルチェック
            if hasattr(self, 'tracking_cancelled') and self.tracking_cancelled:
                print("DEBUG: Tracking cancelled by user")
                break
            if hasattr(gv, 'cancel_operation') and gv.cancel_operation:
                print("DEBUG: Tracking cancelled via global flag")
                break
                
            # プログレス更新
            progress = int((i / total_frames) * 100)
            self.track_progress.setValue(progress)
            QtWidgets.QApplication.processEvents()  # UI更新
            
            # 追加のキャンセルチェック
            if hasattr(self, 'tracking_cancelled') and self.tracking_cancelled:
                print("DEBUG: Tracking cancelled after processEvents")
                break
            if hasattr(gv, 'cancel_operation') and gv.cancel_operation:
                print("DEBUG: Tracking cancelled via global flag after processEvents")
                break
            current_particles = self.particles_by_frame.get(frame_idx, [])
            #print(f"[DEBUG] Frame {frame_idx}: {len(current_particles)} particles")
            
            # 追跡対象のトラック（track_ends）と、そのインデックスを準備
            track_ends = []
            track_indices_map = {} # key: track_endsのindex, value: active_tracksのindex
            for i, track in enumerate(self.active_tracks):
                # 消失フレーム数が許容範囲内のトラックのみを対象とする
                if frame_idx - track[-1]['frame'] <= max_frame_gap + 1:
                    track_indices_map[len(track_ends)] = i
                    track_ends.append([track[-1]['x'], track[-1]['y']])

            #print(f"[DEBUG] Frame {frame_idx}: {len(track_ends)} active tracks available for linking")

            # リンク候補がない場合は、現在の粒子で新しいトラックを開始
            if not track_ends or not current_particles:
                for p in current_particles:
                    new_track = [{'frame': frame_idx, 'x': p.x, 'y': p.y, 'particle': self.next_track_id}]
                    self.active_tracks.append(new_track)
                    self.next_track_id += 1
                    total_new_tracks += 1
                #print(f"[DEBUG] Frame {frame_idx}: Started {len(current_particles)} new tracks (no linking possible)")
                continue

            current_particle_coords = [[p.x, p.y] for p in current_particles]
            
            # コスト（距離の2乗）マトリックスを作成
            cost_matrix = cdist(np.array(track_ends), np.array(current_particle_coords), 'sqeuclidean')
            #print(f"[DEBUG] Frame {frame_idx}: Cost matrix shape {cost_matrix.shape}")

            # ▼▼▼▼▼ ここが修正点 ▼▼▼▼▼
            # 距離が閾値を超えるリンクのコストを、無限大ではなく閾値より少し大きい値に設定
            # これによりソルバーが常に解を見つけられるようにする
            cost_matrix[cost_matrix > max_dist_sq_px] = max_dist_sq_px + 1
            # ▲▲▲▲▲ 修正ここまで ▲▲▲▲▲

            # LAPを解く
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            #print(f"[DEBUG] Frame {frame_idx}: LAP solved, {len(row_ind)} assignments found")

            # 実際に閾値以下の有効なリンクのみを処理
            linked_particles_indices = set()
            valid_links = 0
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= max_dist_sq_px:
                    track_idx = track_indices_map[r]
                    particle = current_particles[c]
                    
                    # 粒子にトラックIDを割り当ててリストに追加
                    particle_data = {'frame': frame_idx, 'x': particle.x, 'y': particle.y, 'particle': self.active_tracks[track_idx][0]['particle']}
                    self.active_tracks[track_idx].append(particle_data)
                    linked_particles_indices.add(c)
                    valid_links += 1
                    total_links_made += 1
            
            #print(f"[DEBUG] Frame {frame_idx}: {valid_links} valid links made")
            
            # リンクされなかった粒子で新しいトラックを開始
            new_tracks_this_frame = 0
            for i, p in enumerate(current_particles):
                if i not in linked_particles_indices:
                    new_track = [{'frame': frame_idx, 'x': p.x, 'y': p.y, 'particle': self.next_track_id}]
                    self.active_tracks.append(new_track)
                    self.next_track_id += 1
                    new_tracks_this_frame += 1
                    total_new_tracks += 1
            
            #print(f"[DEBUG] Frame {frame_idx}: {new_tracks_this_frame} new tracks started")
            total_particles_processed += len(current_particles)
        
        #print(f"[DEBUG] === Simple LAP Tracker Summary ===")
        #print(f"[DEBUG] Total particles processed: {total_particles_processed}")
        #print(f"[DEBUG] Total links made: {total_links_made}")
        #print(f"[DEBUG] Total new tracks started: {total_new_tracks}")
        #print(f"[DEBUG] Final active tracks: {len(self.active_tracks)}")
        
        # 最終的な後処理
        final_tracks_list = []
        valid_tracks = 0
        for track in self.active_tracks:
            if len(track) >= min_track_length:
                final_tracks_list.extend(track)
                valid_tracks += 1
        
        #print(f"[DEBUG] Valid tracks (length >= {min_track_length}): {valid_tracks}")
        
        if final_tracks_list:
            self.tracks_df = pd.DataFrame(final_tracks_list)
            #print(f"[DEBUG] Final DataFrame: {len(self.tracks_df)} rows, {self.tracks_df['particle'].nunique()} unique tracks")
        else:
            self.tracks_df = pd.DataFrame()
            #print(f"[DEBUG] No valid tracks found")
        
        #print(f"[DEBUG] === Simple LAP Tracker Completed ===")

    def _track_with_kalman(self):
        """filterpyとSciPyを使用してKalman Filterトラッキングを実装"""
        # filterpyの利用可能性をチェック
        if not FILTERPY_AVAILABLE or KalmanFilter is None:
            _frozen = getattr(sys, "frozen", False)
            if _frozen:
                msg = (
                    "filterpy is not installed.\n"
                    "filterpyライブラリが利用できません。\n\n"
                    "filterpy is not bundled with this installation.\n"
                    "このモジュールはこのパッケージに含まれていません。"
                )
            else:
                msg = (
                    "filterpy is not installed.\n"
                    "filterpyライブラリが利用できません。\n\n"
                    "Kalman Filterトラッキングを使用するには、filterpyをインストールしてください:\n"
                    "pip install filterpy"
                )
            QtWidgets.QMessageBox.critical(self, "Library Not Available", msg)
            return
            
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        # --- パラメータ設定 ---
        max_distance_nm = self.max_distance_spin.value()
        max_frame_gap = self.max_frame_gap_spin.value()
        min_track_length = self.min_track_length_spin.value()

        # ピクセルサイズ計算
        scan_size_x = getattr(gv, 'XScanSize', 1000.0); scan_size_y = getattr(gv, 'YScanSize', 1000.0)
        frame_data = self.frame_data_by_frame[0]
        pixel_size = min(scan_size_x / frame_data.shape[1], scan_size_y / frame_data.shape[0])
        max_dist_sq = (max_distance_nm / pixel_size) ** 2

        # --- トラッキングロジック ---
        next_track_id = 0
        active_tracks = [] # [ {id, kf, points, age, misses}, ... ]

        sorted_frames = sorted(self.particles_by_frame.keys())
        total_frames = len(sorted_frames)
        
        for i, frame_idx in enumerate(sorted_frames):
            # キャンセルチェック
            if hasattr(self, 'tracking_cancelled') and self.tracking_cancelled:
                print("DEBUG: Tracking cancelled by user")
                break
            if hasattr(gv, 'cancel_operation') and gv.cancel_operation:
                print("DEBUG: Tracking cancelled via global flag")
                break
                
            # プログレス更新
            progress = int((i / total_frames) * 100)
            self.track_progress.setValue(progress)
            QtWidgets.QApplication.processEvents()  # UI更新
            
            # 追加のキャンセルチェック
            if hasattr(self, 'tracking_cancelled') and self.tracking_cancelled:
                print("DEBUG: Tracking cancelled after processEvents")
                break
            if hasattr(gv, 'cancel_operation') and gv.cancel_operation:
                print("DEBUG: Tracking cancelled via global flag after processEvents")
                break
            # 1. 予測ステップ
            for track in active_tracks:
                track['kf'].predict()

            # 2. 割り当てステップ
            measurements = self.particles_by_frame.get(frame_idx, [])
            if not active_tracks or not measurements:
                # 新しいトラックを開始
                for p in measurements:
                    kf = KalmanFilter(dim_x=4, dim_z=2) # 状態(x,y,vx,vy), 観測(x,y)
                    kf.x = np.array([p.x, p.y, 0., 0.]) # 初期状態
                    kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]]) # 状態遷移行列
                    kf.H = np.array([[1,0,0,0], [0,1,0,0]]) # 観測行列
                    kf.P *= 1000. # 初期共分散
                    kf.R = np.diag([5, 5]) # 観測ノイズ
                    from filterpy.common import Q_discrete_white_noise
                    kf.Q = Q_discrete_white_noise(dim=4, dt=1., var=0.1) # プロセスノイズ
                    
                    active_tracks.append({'id': next_track_id, 'kf': kf, 'points': [{'frame': frame_idx, 'x': p.x, 'y': p.y}], 'age': 1, 'misses': 0})
                    next_track_id += 1
                continue
            
            predicted_pos = np.array([track['kf'].x[:2] for track in active_tracks])
            measured_pos = np.array([[p.x, p.y] for p in measurements])
            
            cost_matrix = cdist(predicted_pos, measured_pos, 'sqeuclidean')
            cost_matrix[cost_matrix > max_dist_sq] = max_dist_sq + 1

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 3. 更新ステップ
            matched_track_indices = set()
            matched_meas_indices = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= max_dist_sq:
                    track = active_tracks[r]
                    meas = measurements[c]
                    track['kf'].update(np.array([meas.x, meas.y]))
                    track['points'].append({'frame': frame_idx, 'x': meas.x, 'y': meas.y})
                    track['age'] += 1
                    track['misses'] = 0
                    matched_track_indices.add(r)
                    matched_meas_indices.add(c)
            
            # マッチしなかったトラックの処理
            for i, track in enumerate(active_tracks):
                if i not in matched_track_indices:
                    track['misses'] += 1
            
            # 見失ったトラックを削除
            active_tracks = [t for t in active_tracks if t['misses'] <= max_frame_gap]
            
            # マッチしなかった観測から新しいトラックを開始
            for i, p in enumerate(measurements):
                if i not in matched_meas_indices:
                    kf = KalmanFilter(dim_x=4, dim_z=2)
                    kf.x = np.array([p.x, p.y, 0., 0.]); kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]); kf.H = np.array([[1,0,0,0],[0,1,0,0]]); kf.P *= 1000.; kf.R = np.diag([5, 5]); kf.Q = Q_discrete_white_noise(dim=4, dt=1., var=0.1)
                    active_tracks.append({'id': next_track_id, 'kf': kf, 'points': [{'frame': frame_idx, 'x': p.x, 'y': p.y}], 'age': 1, 'misses': 0})
                    next_track_id += 1

        # 最終的な後処理
        final_tracks_list = []
        for track in active_tracks:
            if track['age'] >= min_track_length:
                for point in track['points']:
                    point['particle'] = track['id']
                    final_tracks_list.append(point)
        
        if final_tracks_list:
            self.tracks_df = pd.DataFrame(final_tracks_list)
        else:
            self.tracks_df = pd.DataFrame()
    
    def cancelCurrentOperation(self):
        """現在の操作をキャンセル"""
        print("DEBUG: cancelCurrentOperation called")
        
        # グローバルキャンセルフラグを設定
        gv.cancel_operation = True
        
        if hasattr(self, 'tracking_cancelled'):
            self.tracking_cancelled = True
            print("DEBUG: Set tracking_cancelled = True")
        
        if hasattr(self, 'detection_cancelled'):
            self.detection_cancelled = True
            print("DEBUG: Set detection_cancelled = True")
    
    def cancelTracking(self):
        """トラッキングをキャンセル（後方互換性のため残存）"""
        self.cancelCurrentOperation()
    
    def cancelDetection(self):
        """検出をキャンセル（後方互換性のため残存）"""
        self.cancelCurrentOperation()
    
    def onDisplaySettingsChanged(self):
        """表示設定が変更された時の処理"""
        try:
            # プレビューを更新
            self.updatePreview()
        except Exception as e:
            print(f"[ERROR] Failed to update display settings: {e}")
            import traceback
            traceback.print_exc()

    def open_track_editor(self):
        """軌跡編集ウィンドウを開く"""
        if self.tracks_df is None or self.tracks_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please run tracking first.")
            return

        # 編集ウィンドウのインスタンスを作成
        self.track_editor_window = TrackSchemeWindow(self.tracks_df)
        # 編集完了シグナルに接続（リアルタイム更新）
        self.track_editor_window.tracks_updated.connect(self.on_tracks_edited)
        self.track_editor_window.show()
        
        # 編集ウィンドウがアクティブになるように設定
        self.track_editor_window.raise_()
        self.track_editor_window.activateWindow()

    def on_tracks_edited(self, updated_tracks_df):
        """編集された軌跡データを受け取り、UIを更新する"""
        self.tracks_df = updated_tracks_df
        
        # プレビュー、統計、テーブルを更新
        self.updatePreview()
        self.updateStatistics()
        self.updateTracksTable()
        
    def open_track_analysis(self):
        """Track Analysisパネルを開く"""
        if self.tracks_df is None or self.tracks_df.empty:
            QtWidgets.QMessageBox.warning(self, "Warning", "No tracking data available for analysis.")
            return
            
        if TrackAnalysisPanel is None:
            QtWidgets.QMessageBox.critical(self, "Error", "TrackAnalysisPanel module not available.")
            return
            
        try:
            # Track Analysisパネルのインスタンスを作成
            self.track_analysis_window = TrackAnalysisPanel()
            
            # トラックデータとフレーム時間を設定
            frame_time = getattr(gv, 'frame_time', 0.1)
            self.track_analysis_window.setTracksData(self.tracks_df, frame_time)
            
            # パネルを表示
            self.track_analysis_window.show()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open Track Analysis panel: {e}")
            import traceback
            traceback.print_exc()


# (ファイルの末尾に追加)

class TrackSchemeWindow(QtWidgets.QWidget):
    """
    軌跡を可視化し、手動で編集するためのウィンドウクラス。
    Track Scheme風の機能を提供する。
    """
    # 編集が完了したときに送信されるシグナル
    tracks_updated = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, tracks_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.tracks_df = tracks_df.copy()
        self.setWindowTitle("Track Scheme Editor")
        self.setMinimumSize(400, 300)  # サイズを小さく
        self.resize(600, 400)  # デフォルトサイズを設定
        self.setToolTip("Track Scheme Editor\n\n"
                       "A visual interface for editing particle tracks.\n\n"
                       "Display:\n"
                       "• X-axis: Frame number\n"
                       "• Y-axis: Track ID\n"
                       "• Lines: Track connections\n"
                       "• Points: Individual particle detections\n\n"
                       "Editing:\n"
                       "• Right-click on points to edit tracks\n"
                       "• Changes are applied immediately\n"
                       "• Window can be closed to save changes")

        # Matplotlibのセットアップ
        if not _import_matplotlib():
            self.close()
            return

        # matplotlibのインポートを明示的に行う
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # 操作方法の説明を追加
        instruction_label = QtWidgets.QLabel(
            "How to use:\n"
            "• Right-click on a track point to open edit menu\n"
            "• Split: Split track into two at selected frame\n"
            "• Delete: Completely remove selected track\n"
            "• Merge: Merge selected track with another track"
        )
        instruction_label.setToolTip("Track Scheme Editor Instructions\n\n"
                                   "Navigation:\n"
                                   "• Click and drag to pan the view\n"
                                   "• Use mouse wheel to zoom in/out\n"
                                   "• Right-click on track points for editing options\n\n"
                                   "Track Operations:\n"
                                   "• Split: Creates two separate tracks from one\n"
                                   "• Delete: Removes entire track from analysis\n"
                                   "• Merge: Combines two tracks into one\n\n"
                                   "Note: Changes are applied immediately to the main window.")
        instruction_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 5px;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        instruction_label.setWordWrap(True)
        
        # スクロールエリアを作成
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # スクロールエリア内のコンテンツウィジェット
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.addWidget(instruction_label)
        content_layout.addWidget(self.canvas)
        
        # キャンバスのサイズを設定
        self.canvas.setMinimumSize(800, 600)  # キャンバスは大きく、スクロールで表示
        
        scroll_area.setWidget(content_widget)
        
        # メインレイアウト
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll_area)

        # イベントハンドラの接続
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # ウィンドウ設定を読み込み
        self.loadWindowSettings()
        
        self.plot_scheme()

    def plot_scheme(self):
        """軌跡データをプロットする"""
        import matplotlib.pyplot as plt
        
        self.ax.clear()
        if self.tracks_df.empty:
            self.ax.set_title("No tracks to display")
            self.canvas.draw()
            return

        # 各軌跡をプロット
        for track_id, track_data in self.tracks_df.groupby('particle'):
            self.ax.plot(track_data['frame'], track_data['particle'], '-o', markersize=4, label=f'Track {track_id}')

        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Track ID")
        self.ax.set_title("Click on a point to edit")
        self.ax.grid(True, axis='y', linestyle=':')
        # Y軸の目盛りを整数にする
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.canvas.draw()

    def on_canvas_click(self, event):
        """キャンバスのクリックイベントを処理する"""
        if event.button != 3 or event.inaxes != self.ax:  # 右クリック以外は無視
            return

        # クリックされた点に最も近い粒子を探す
        clicked_frame = event.xdata
        clicked_track_id = event.ydata
        if clicked_frame is None or clicked_track_id is None:
            return

        # 閾値（クリック位置からの許容距離）
        min_dist = float('inf')
        target_particle_info = None

        for index, row in self.tracks_df.iterrows():
            dist = np.sqrt((row['frame'] - clicked_frame)**2 + (row['particle'] - clicked_track_id)**2)
            if dist < min_dist and dist < 0.5: # 許容範囲を調整
                min_dist = dist
                target_particle_info = {'frame': row['frame'], 'particle_id': row['particle'], 'index': index}

        if target_particle_info:
            self.show_context_menu(event.guiEvent.globalPos(), target_particle_info)

    def show_context_menu(self, position, particle_info):
        """右クリックメニューを表示する"""
        menu = QtWidgets.QMenu(self)
        split_action = menu.addAction("Split Track Here")
        split_action.setToolTip("Split the current track at this frame.\n\n"
                               "Creates two separate tracks:\n"
                               "• Track 1: From start to current frame\n"
                               "• Track 2: From current frame to end\n\n"
                               "Useful for correcting tracking errors where\n"
                               "particles were incorrectly linked.")
        
        delete_action = menu.addAction("Delete Track")
        delete_action.setToolTip("Permanently delete this entire track.\n\n"
                                "Warning: This action cannot be undone.\n"
                                "The track will be removed from all analysis.\n\n"
                                "Use to remove false positive detections\n"
                                "or tracking artifacts.")
        
        merge_action = menu.addAction("Merge Track")
        merge_action.setToolTip("Merge this track with another track.\n\n"
                               "Selects available tracks that can be merged.\n"
                               "Tracks must not have overlapping frames.\n\n"
                               "Useful for connecting tracks that were\n"
                               "incorrectly split during detection.")
        
        action = menu.exec_(position)
        
        if action == split_action:
            self.split_track_at(particle_info)
        elif action == delete_action:
            self.delete_track(particle_info)
        elif action == merge_action:
            self.merge_track(particle_info)

    def split_track_at(self, particle_info):
        """指定された粒子で軌跡を分割する"""
        frame_to_split = particle_info['frame']
        track_id_to_split = particle_info['particle_id']

        # 新しい軌跡IDを生成
        if self.tracks_df.empty:
            new_track_id = 1
        else:
            new_track_id = self.tracks_df['particle'].max() + 1

        # 分割する箇所以降のパーティクルのIDを変更
        condition = (self.tracks_df['particle'] == track_id_to_split) & (self.tracks_df['frame'] >= frame_to_split)
        self.tracks_df.loc[condition, 'particle'] = new_track_id

        print(f"Track {track_id_to_split} was split at frame {frame_to_split}. New track is {new_track_id}.")
        
        # プロットを再描画
        self.plot_scheme()
        
        # メインウィンドウに即座に反映
        self.tracks_updated.emit(self.tracks_df)

    def delete_track(self, particle_info):
        """指定された軌跡を削除する"""
        track_id_to_delete = particle_info['particle_id']
        
        # 確認ダイアログを表示
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Track", 
            f"Are you sure you want to delete Track {track_id_to_delete}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # 指定された軌跡IDのデータを削除
            self.tracks_df = self.tracks_df[self.tracks_df['particle'] != track_id_to_delete]
            
            pass
            
            # プロットを再描画
            self.plot_scheme()
            
            # メインウィンドウに即座に反映
            self.tracks_updated.emit(self.tracks_df)

    def merge_track(self, particle_info):
        """軌跡を結合する"""
        track_id_to_merge = particle_info['particle_id']
        
        # 結合可能な軌跡を探す（同じフレームに重複がない軌跡）
        available_tracks = []
        for track_id in self.tracks_df['particle'].unique():
            if track_id != track_id_to_merge:
                track_data = self.tracks_df[self.tracks_df['particle'] == track_id]
                current_track_data = self.tracks_df[self.tracks_df['particle'] == track_id_to_merge]
                
                # フレームの重複チェック
                current_frames = set(current_track_data['frame'])
                other_frames = set(track_data['frame'])
                
                if not current_frames & other_frames:  # 重複がない場合
                    available_tracks.append(track_id)
        
        if not available_tracks:
            QtWidgets.QMessageBox.information(
                self, "No Mergeable Tracks", 
                f"No tracks available for merging with Track {track_id_to_merge}.\n"
                "Tracks must not have overlapping frames."
            )
            return
        
        # 結合する軌跡を選択するダイアログ
        track_list, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Track to Merge", 
            f"Select a track to merge with Track {track_id_to_merge}:",
            [f"Track {tid}" for tid in available_tracks], 0, False
        )
        
        if ok and track_list:
            # 選択された軌跡IDを取得
            selected_track_id = available_tracks[track_list.index(track_list)]
            
            # 結合実行
            self.tracks_df.loc[self.tracks_df['particle'] == selected_track_id, 'particle'] = track_id_to_merge
            
            pass
            
            # プロットを再描画
            self.plot_scheme()
            
            # メインウィンドウに即座に反映
            self.tracks_updated.emit(self.tracks_df)

    def loadWindowSettings(self):
        """ウィンドウ設定の読み込み"""
        if 'TrackSchemeWindow' in gv.windowSettings:
            settings = gv.windowSettings['TrackSchemeWindow']
            self.setGeometry(settings['x'], settings['y'], settings['width'], settings['height'])
    
    def saveWindowSettings(self):
        """ウィンドウ設定の保存"""
        if 'TrackSchemeWindow' not in gv.windowSettings:
            gv.windowSettings['TrackSchemeWindow'] = {}
        
        geometry = self.geometry()
        gv.windowSettings['TrackSchemeWindow'].update({
            'x': geometry.x(),
            'y': geometry.y(),
            'width': geometry.width(),
            'height': geometry.height()
        })
    
    def closeEvent(self, event):
        """ウィンドウが閉じられるときにシグナルを送信と設定保存"""
        self.tracks_updated.emit(self.tracks_df)
        self.saveWindowSettings()
        super().closeEvent(event)


__all__ = ["PLUGIN_NAME", "create_plugin", "ParticleTrackingWindow"]


# TrackAnalysisPanelは別ファイルからインポート