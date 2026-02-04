#!/usr/bin/env python3
# type: ignore
"""
Venv AFM Simulator with Reliable PDB Display
Uses simplified VTK rendering for speed and reliability
"""

import sys
import numpy as np
import os
import json 
import struct  # ★★★ 追加 ★★★
import datetime # ★★★ 追加 ★★★
import shlex
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                            QSlider, QComboBox, QSpinBox, QDoubleSpinBox,
                            QGroupBox, QFileDialog, QMessageBox, QTextEdit,
                            QSplitter, QFrame, QCheckBox, QScrollArea,
                            QColorDialog, QTabWidget, QProgressBar, QInputDialog, QAction,
                            QTreeWidget, QTextBrowser, QTreeWidgetItem, QSpacerItem, QSizePolicy, QLineEdit, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QTime, QSettings, QEventLoop, QEvent
from PyQt5.QtGui import QFont, QColor, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

import vtk

# Support standalone launch: use globalvals when run from pyNuD, else minimal stub
try:
    import globalvals as gv
except ModuleNotFoundError:
    class _GlobalValsStub:
        standardFont = "Helvetica"
        main_window = None
    gv = _GlobalValsStub()

# VTK 9.x compatibility: Try different import methods for Qt integration
try:
    # Try the old VTK 8.x import method
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # type: ignore
except ImportError:
    try:
        # Try VTK 9.x import method
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # type: ignore
    except ImportError:
        try:
            # Alternative VTK 9.x import method
            from vtkmodules.vtkRenderingQt import QVTKRenderWindowInteractor  # type: ignore
        except ImportError:
            # Fallback: Create a simple wrapper class
            print("Warning: VTK Qt integration not available. Using fallback implementation.")
            class QVTKRenderWindowInteractor:  # type: ignore
                def __init__(self, parent=None):
                    self.parent = parent
                    self.render_window = None
                    self.interactor = None
                    print("Warning: VTK Qt integration not properly configured.")
                    print("Please install VTK with Qt support or use a compatible VTK version.")
# Numbaをインポートして計算を高速化（オプション）

import scipy.ndimage

from scipy.fft import fft2, ifft2, fftshift, ifftshift # ★★★ この行を追加 ★★★
from pathlib import Path

# Numbaをインポートして計算を高速化（オプションですが強く推奨します）
try:
    from numba import jit
except ImportError:
    # numbaがインストールされていない場合、何もしないダミーのデコレータを作成
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator


PLUGIN_NAME = "AFM Simulator"

HELP_HTML_EN = """
<h1>AFM Simulator</h1>
<h2>Overview</h2>
<p>The AFM Simulator generates simulated AFM images from molecular structure files. It is useful for comparing experimental AFM data with structural models.</p>
<h2>Access</h2>
<ul>
    <li><strong>Plugin menu:</strong> Load Plugin... → select <code>plugins/AFMSimulator.py</code>, then Plugin → AFM Simulator</li>
</ul>
<h2>Importing Structure Files</h2>
<div class="feature-box">
    <h3>Supported Formats</h3>
    <ul>
        <li><strong>PDB:</strong> Standard PDB files (<code>.pdb</code>) are supported.</li>
        <li><strong>mmCIF:</strong> mmCIF format files (<code>.cif</code>, <code>.mmcif</code>) are supported.</li>
        <li><strong>MRC:</strong> MRC (Medical Research Council) volume data files (<code>.mrc</code>) are supported.</li>
    </ul>
</div>
<h2>File Import</h2>
<div class="step"><strong>Step 1:</strong> Click <strong>Import File...</strong> button.</div>
<div class="step"><strong>Step 2:</strong> Select a structure file (<code>.pdb</code>, <code>.cif</code>, <code>.mmcif</code>, or <code>.mrc</code>).</div>
<div class="step"><strong>Step 3:</strong> Confirm that the loaded file name is displayed in the simulator window.</div>
<div class="step">You can also drag and drop a file onto the file name line below the Import File button.</div>
<h2>Display style: Ribbon and secondary structure</h2>
<p>The AFM Simulator supports PyMOL-style ribbon visualization (Catmull-Rom spline interpolation) based on secondary structure detection. Select <strong>Ribbon (PyMOL-style)</strong> in the display style to show the protein backbone as a ribbon. You can also change the display style from the context menu by right-clicking on the molecule view.</p>
"""

HELP_HTML_JA = """
<h1>AFMシミュレータ</h1>
<h2>概要</h2>
<p>AFMシミュレータは分子構造ファイルからシミュレートAFM像を生成します。実験AFMデータと構造モデルの比較に利用できます。</p>
<h2>アクセス</h2>
<ul>
    <li><strong>プラグインメニュー:</strong> Load Plugin... → <code>plugins/AFMSimulator.py</code> を選択し、Plugin → AFM Simulator</li>
</ul>
<h2>構造ファイルのインポート</h2>
<div class="feature-box">
    <h3>対応形式</h3>
    <ul>
        <li><strong>PDB:</strong> 標準のPDBファイル（<code>.pdb</code>）に対応しています。</li>
        <li><strong>mmCIF:</strong> mmCIF形式ファイル（<code>.cif</code>、<code>.mmcif</code>）に対応しています。</li>
        <li><strong>MRC:</strong> MRC（Medical Research Council）ボリュームデータファイル（<code>.mrc</code>）に対応しています。</li>
    </ul>
</div>
<h2>ファイルインポート</h2>
<div class="step"><strong>Step 1:</strong> <strong>Import File...</strong> ボタンをクリック。</div>
<div class="step"><strong>Step 2:</strong> 構造ファイル（<code>.pdb</code>、<code>.cif</code>、<code>.mmcif</code>、または <code>.mrc</code>）を選択。</div>
<div class="step"><strong>Step 3:</strong> シミュレータウィンドウに読み込んだファイル名が表示されることを確認。</div>
<div class="step">Import File ボタン下のファイル名の行にドラッグ＆ドロップすることもできます。</div>
<h2>表示スタイル: リボンと二次構造</h2>
<p>AFMシミュレータでは二次構造の検出に基づき、PyMOL風のリボン可視化（Catmull-Romスプライン補間）が利用できます。表示スタイルで <strong>Ribbon (PyMOL-style)</strong> を選択すると、タンパク質の主鎖がリボンとして表示されます。分子表示上で右クリックするコンテキストメニューからも表示スタイルを変更できます。</p>
"""


def create_frequency_grid(image_shape, scan_size_nm):
    """
    実際のスキャンサイズを考慮した周波数グリッドを作成 (cycles/nm)
    """
    ny, nx = image_shape
    pixel_size_x = scan_size_nm / nx
    pixel_size_y = scan_size_nm / ny
    
    freq_x = fftshift(np.fft.fftfreq(nx, d=pixel_size_x))
    freq_y = fftshift(np.fft.fftfreq(ny, d=pixel_size_y))
    
    freq_xx, freq_yy = np.meshgrid(freq_x, freq_y)
    return np.sqrt(freq_xx**2 + freq_yy**2)

def apply_low_pass_filter(image, scan_size_nm, cutoff_wl_nm):
    """
    バターワース・ローパスフィルターを画像に適用する
    """
    # 周波数グリッドを作成
    freq_grid = create_frequency_grid(image.shape, scan_size_nm)
    
    # カットオフ波長を周波数に変換 (0除算を防止)
    cutoff_freq = 1.0 / max(cutoff_wl_nm, 0.001)
    
    # バターワースフィルターのマスクを作成 (次数n=2)
    # このマスクは、中心(低周波)が1で、カットオフ周波数から離れると0に近づく
    order = 2
    filter_mask = 1.0 / (1.0 + (freq_grid / cutoff_freq)**(2 * order))
    
    # フーリエ変換、フィルター適用、逆変換
    img_fft = fftshift(fft2(image))
    filtered_fft = img_fft * filter_mask
    filtered_img = ifft2(ifftshift(filtered_fft))
    
    return np.real(filtered_img).astype(image.dtype)
    
@jit(nopython=True)
def _calculate_dilation_row(r_out, sample_surface, tip_footprint):
    """
    形態学的ダイレーションの1行分だけを計算するNumba高速化関数。
    """
    s_rows, s_cols = sample_surface.shape
    t_rows, t_cols = tip_footprint.shape
    t_center_r, t_center_c = t_rows // 2, t_cols // 2

    output_row = np.full(s_cols, -1e9, dtype=np.float64)

    for c_out in range(s_cols):
        max_h = -1e9
        for r_tip in range(t_rows):
            for c_tip in range(t_cols):
                s_r = r_out + r_tip - t_center_r
                s_c = c_out + c_tip - t_center_c

                if 0 <= s_r < s_rows and 0 <= s_c < s_cols:
                    h = sample_surface[s_r, s_c] - tip_footprint[r_tip, c_tip]
                    if h > max_h:
                        max_h = h
        output_row[c_out] = max_h

    return output_row


#@jit(nopython=True)
def _create_vdw_surface_loop(resolution, pixel_size, x_start, y_start, min_z, atom_coords, atom_radii):
    """
    原子のファンデルワールス半径を考慮して表面マップを生成するNumba高速化関数。
    """
    #initial_value = min_z - 5.0
    #print("minZ:", min_z)  # minZを表示
    surface_map = np.full((resolution, resolution), min_z - 5.0, dtype=np.float64)
    px_coords = x_start + (np.arange(resolution) + 0.5) * pixel_size
    py_coords = y_start + (np.arange(resolution) + 0.5) * pixel_size
    
    for i in range(len(atom_coords)):
        ax, ay, az = atom_coords[i]
        az -= min_z
        r = atom_radii[i]
        r_sq = r**2

        ix_min = int(np.floor((ax - r - x_start) / pixel_size))
        ix_max = int(np.ceil((ax + r - x_start) / pixel_size))
        iy_min = int(np.floor((ay - r - y_start) / pixel_size))
        iy_max = int(np.ceil((ay + r - y_start) / pixel_size))
        
        ix_min, ix_max = max(0, ix_min), min(resolution, ix_max)
        iy_min, iy_max = max(0, iy_min), min(resolution, iy_max)

        for iy in range(iy_min, iy_max):
            for ix in range(ix_min, ix_max):
                px, py = px_coords[ix], py_coords[iy]
                d_sq = (px - ax)**2 + (py - ay)**2
                
                if d_sq <= r_sq:
                    h = az + np.sqrt(r_sq - d_sq)
                    if h > surface_map[iy, ix]:
                        surface_map[iy, ix] = h
                        #print("surface_map[", iy, ",", ix, "] =", h)
                        
    surface_map[surface_map < min_z - 4.0] = 0.0
    #print("surface_map[0,0] =", surface_map[0,0])
    #surface_map[surface_map == initial_value] = 0.0
    return surface_map

class HelpContentManager:
    """
    Manages all help content, supporting multiple languages.
    """
    def __init__(self):
        self._initialize_content()

    def set_language(self, lang_code):
        if lang_code in self.content:
            self.current_language = lang_code

    def get_toc_structure(self):
        return self.content[self.current_language]['toc_structure']

    def get_content(self, page_id):
        # シミュレータ用に内容を簡略化
        pages = self.content[self.current_language]['pages']
        page_content = pages.get(page_id, pages['home']) # 見つからない場合はhomeを表示
        return self._wrap_content(page_content)
            
    def get_ui_text(self, key):
        return self.content[self.current_language]['ui_text'].get(key, '')

    def _wrap_content(self, content):
        return f"<html><head>{self.STYLES}</head><body>{content}</body></html>"

    STYLES = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; margin: 20px; line-height: 1.6; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; font-size: 22px; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 25px; font-size: 18px;}
        li { margin: 8px 0; }
        strong { color: #000; }
    </style>
    """
    
    def _initialize_content(self):
        self.current_language = 'en'
        
        # --- 英語コンテンツ ---
        toc_structure_en = [
            ("Simulator Help", [
                ("Introduction", "home"),
                ("Display Settings", "display"),
                ("Tip Settings", "tip"),
                ("Simulation Settings", "simulation"),
                ("File Loading", "file_loading"),
                ("Structure Manipulation", "structure_manipulation"),
            ]),
        ]
        pages_en = {
            "home": """
            <h1>AFM Simulator Help</h1>
            <p>This is a help guide for the parameters used in the AFM Simulator.</p>
            <p>Select a topic from the table of contents on the left to view detailed explanations.</p>
            """,
            "display": """
            <h2>Display Settings</h2>
            <ul>
                <li><strong>Style:</strong> Selects the display style for the molecule (e.g., Ball & Stick, Spheres).</li>
                <li><strong>Color:</strong> Selects the coloring scheme (e.g., By Element, By Chain).</li>
                <li><strong>Show:</strong> Filters which atoms are displayed (e.g., All Atoms, Heavy Atoms).</li>
                <li><strong>Size / Opacity:</strong> Adjusts the size and opacity of atoms and bonds.</li>
                <li><strong>Quality:</strong> Quality of the 3D rendering. 'Fast' is quick, while 'High' is smoother.</li>
            </ul>
            """,
            "tip": """
            <h2>AFM Tip Settings</h2>
            <ul>
                <li><strong>Shape:</strong> Selects the overall shape of the tip.</li>
                <li><strong>Radius:</strong> Radius of curvature of the tip apex in nm. Smaller is sharper.</li>
                <li><strong>Angle:</strong> Half-angle of the cone part in degrees. Smaller is sharper.</li>
                <li><strong>Minitip Radius:</strong> Only for 'Sphere' shape. The radius of the sphere attached to the very end of the tip.</li>
            </ul>
            """,
            "simulation": """
            <h2>AFM Simulation Settings</h2>
            <ul>
                <li><strong>Scan Size (nm):</strong> The side length of the square area to be simulated, in nm.</li>
                <li><strong>Resolution:</strong> The number of pixels in the simulated image.</li>
                <li><strong>Consider atom size (vdW):</strong> If checked, treats atoms as spheres with van der Waals radii instead of points, calculating a more physically accurate surface.</li>
                <li><strong>Apply Low-pass Filter:</strong> If checked, applies an FFT low-pass filter to the result to match the resolution of real experimental data.</li>
                <li><strong>Cutoff Wavelength (nm):</strong> The cutoff wavelength for the filter. Empirically, a value around 2 nm often produces results that correspond well with real high-speed AFM images.</li>
                <li><strong>Interactive Update:</strong> If checked, automatically updates the simulation at low resolution (64x64) when PDB rotation, tip, or scan parameters are changed.</li>
            </ul>
            """,
            "file_loading": """
            <h2>File Loading</h2>
            <h3>File Import</h3>
            <ul>
                <li><strong>Import File:</strong> Loads structure data from PDB (<code>.pdb</code>), mmCIF (<code>.cif</code>, <code>.mmcif</code>), or MRC (<code>.mrc</code>) format files.</li>
                <li><strong>Automatic Tip Positioning:</strong> The tip is automatically positioned 2nm above the highest point of the loaded structure.</li>
                <li><strong>Rotation Controls:</strong> X, Y, Z rotation controls are automatically enabled after loading.</li>
            </ul>
            <h3>MRC Files</h3>
            <ul>
                <li><strong>MRC Format:</strong> MRC (Medical Research Council) format files (<code>.mrc</code>) are supported for volume data.</li>
                <li><strong>Density Threshold:</strong> Adjusts the isosurface threshold for volume rendering.</li>
                <li><strong>Flip Z-axis:</strong> Automatically flips the Z-axis orientation by default for proper display.</li>
                <li><strong>Voxel Size:</strong> Displays the physical size of each voxel in the volume data.</li>
            </ul>
            """,
            "structure_manipulation": """
            <h2>Structure Manipulation</h2>
            <h3>Rotation Controls</h3>
            <ul>
                <li><strong>Rotation X, Y, Z:</strong> Numeric input fields and sliders to rotate the structure around each axis.</li>
                <li><strong>CTRL+Drag:</strong> Hold CTRL and drag with the mouse to interactively rotate the structure in 3D space.</li>
                <li><strong>Reset Rotation:</strong> Use the "Reset Rotation" button to return all rotations to zero.</li>
            </ul>
            <h3>Find Initial Plane</h3>
            <ul>
                <li><strong>Purpose:</strong> Automatically orients the structure to its optimal viewing angle.</li>
                <li><strong>PDB Files:</strong> Uses Principal Component Analysis (PCA) to find the best orientation based on atom distribution.</li>
                <li><strong>MRC Files:</strong> Uses surface coordinate analysis to find the optimal orientation for volume data.</li>
                <li><strong>Usage:</strong> Click the button to automatically rotate the structure to its most stable orientation.</li>
            </ul>
            <h3>MRC-Specific Features</h3>
            <ul>
                <li><strong>Z-axis Flip:</strong> Toggle checkbox to flip the Z-axis orientation of MRC volume data.</li>
                <li><strong>Surface Rendering:</strong> Volume data is rendered as an isosurface based on the density threshold.</li>
                <li><strong>Interactive Rotation:</strong> MRC structures support the same rotation controls as PDB structures.</li>
            </ul>
            """
        }
        ui_text_en = {
            'window_title': "AFM Simulator Help", 'toc_header': "Contents",
            'home_tooltip': "Go to help home page"
        }

        # --- 日本語コンテンツ ---
        toc_structure_ja = [
            ("シミュレーターヘルプ", [
                ("はじめに", "home"),
                ("表示設定", "display"),
                ("探針条件", "tip"),
                ("シミュレーション設定", "simulation"),
                ("ファイル読み込み", "file_loading"),
                ("構造操作", "structure_manipulation"),
            ]),
        ]
        pages_ja = {
            "home": """
            <h1>AFMシミュレーター ヘルプ</h1>
            <p>AFMシミュレーターで使われるパラメータの解説ガイドです。</p>
            <p>左の目次から項目を選択して、詳細な解説をご覧ください。</p>
            """,
            "display": """
            <h2>Display Settings / 表示設定</h2>
            <ul>
                <li><strong>Style:</strong> 分子の表示形式（例: Ball & Stick, Spheres）を選択します。</li>
                <li><strong>Color:</strong> 色付け方法（例: By Element, By Chain）を選択します。</li>
                <li><strong>Show:</strong> 表示する原子の種類（例: All Atoms, Heavy Atoms）をフィルタリングします。</li>
                <li><strong>Size / Opacity:</strong> 原子や結合のサイズ・不透明度を調整します。</li>
                <li><strong>Quality:</strong> 3D表示の品質。Fastは高速ですが、Highはより滑らかです。</li>
            </ul>
            """,
            "tip": """
            <h2>AFM Tip Settings / 探針条件</h2>
            <ul>
                <li><strong>Shape:</strong> 探針の全体的な形状を選択します。</li>
                <li><strong>Radius:</strong> 探針先端の曲率半径 (nm)。小さいほどシャープです。</li>
                <li><strong>Angle:</strong> 円錐部分の半頂角 (deg)。小さいほどシャープです。</li>
                <li><strong>Minitip Radius:</strong> 'Sphere'形状の時のみ有効。探針の最先端に取り付けられた球の半径です。</li>
            </ul>
            """,
            "simulation": """
            <h2>AFM Simulation / シミュレーション設定</h2>
            <ul>
                <li><strong>Scan Size (nm):</strong> シミュレーションを行う正方形領域の一辺の長さ (nm)。</li>
                <li><strong>Resolution:</strong> シミュレーション画像のピクセル数。</li>
                <li><strong>Consider atom size (vdW):</strong> チェックすると、原子を点ではなくファンデルワールス半径を持つ球として扱い、より物理的に正確な表面を計算します。</li>
                <li><strong>Apply Low-pass Filter:</strong> シミュレーション画像は実際の高速AFMデータより空間分解能が高いため、チェックするとFFTローパスフィルターで分解能を近づけます。</li>
                <li><strong>Cutoff Wavelength (nm):</strong> ローパスフィルターのカットオフ波長。経験的に2nm程度の値で実際の高速AFM画像とよく一致します。</li>
                <li><strong>Interactive Update:</strong> チェックすると、PDB回転や探針・スキャン条件の変更時に、低解像度(64x64)でシミュレーションを自動更新します。</li>
            </ul>
            """,
            "file_loading": """
            <h2>File Loading / ファイル読み込み</h2>
            <h3>File Import / ファイルインポート</h3>
            <ul>
                <li><strong>Import File:</strong> Loads structure data from PDB (<code>.pdb</code>), mmCIF (<code>.cif</code>, <code>.mmcif</code>), or MRC (<code>.mrc</code>) format files.</li>
                <li><strong>Import File / ファイルインポート:</strong> PDB（<code>.pdb</code>）、mmCIF（<code>.cif</code>、<code>.mmcif</code>）、またはMRC（<code>.mrc</code>）形式ファイルから構造データを読み込みます。</li>
                <li><strong>Automatic Tip Positioning:</strong> Automatically positions the tip 2nm above the highest point of the loaded structure.</li>
                <li><strong>Automatic Tip Positioning / 自動探針配置:</strong> 読み込んだ構造の最高点から2nm上に探針を自動配置します。</li>
                <li><strong>Rotation Controls:</strong> Rotation controls (X, Y, Z) are automatically enabled after loading.</li>
                <li><strong>Rotation Controls / 回転コントロール:</strong> 読み込み後にX、Y、Z回転コントロールが自動的に有効になります。</li>
            </ul>
            <h3>MRC Files / MRCファイル</h3>
            <ul>
                <li><strong>MRC Format:</strong> MRC (Medical Research Council) format files (<code>.mrc</code>) are supported for volume data.</li>
                <li><strong>MRC形式:</strong> MRC（Medical Research Council）形式ファイル（<code>.mrc</code>）がボリュームデータとしてサポートされています。</li>
                <li><strong>Density Threshold:</strong> ボリュームレンダリングの等値面閾値を調整します。</li>
                <li><strong>Flip Z-axis:</strong> デフォルトでZ軸の向きを自動的にフリップして正しい表示にします。</li>
                <li><strong>Voxel Size:</strong> ボリュームデータの各ボクセルの物理サイズを表示します。</li>
            </ul>
            """,
            "structure_manipulation": """
            <h2>Structure Manipulation / 構造操作</h2>
            <h3>Rotation Controls / 回転コントロール</h3>
            <ul>
                <li><strong>Rotation X, Y, Z:</strong> 各軸周りの構造回転用の数値入力フィールドとスライダーです。</li>
                <li><strong>CTRL+Drag:</strong> CTRLキーを押しながらマウスドラッグで3D空間内で構造をインタラクティブに回転できます。</li>
                <li><strong>Reset Rotation:</strong> 「Reset Rotation」ボタンで全ての回転をゼロに戻します。</li>
            </ul>
            <h3>Find Initial Plane / 初期平面検出</h3>
            <ul>
                <li><strong>Purpose:</strong> 構造を最適な視角に自動的に向けます。</li>
                <li><strong>PDB Files:</strong> 主成分分析（PCA）を使用して原子分布に基づく最適な向きを見つけます。</li>
                <li><strong>MRC Files:</strong> 表面座標解析を使用してボリュームデータの最適な向きを見つけます。</li>
                <li><strong>Usage:</strong> ボタンをクリックして構造を最も安定した向きに自動回転します。</li>
            </ul>
            <h3>MRC-Specific Features / MRC専用機能</h3>
            <ul>
                <li><strong>Z-axis Flip:</strong> チェックボックスでMRCボリュームデータのZ軸向きを切り替えます。</li>
                <li><strong>Surface Rendering:</strong> ボリュームデータは密度閾値に基づいて等値面としてレンダリングされます。</li>
                <li><strong>Interactive Rotation:</strong> MRC構造もPDB構造と同じ回転コントロールをサポートします。</li>
            </ul>
            """
        }
        ui_text_ja = {
            'window_title': "AFMシミュレーター ヘルプ", 'toc_header': "目次",
            'home_tooltip': "ヘルプのホームページに戻る"
        }

        self.content = {
            'en': {'toc_structure': toc_structure_en, 'pages': pages_en, 'ui_text': ui_text_en},
            'ja': {'toc_structure': toc_structure_ja, 'pages': pages_ja, 'ui_text': ui_text_ja},
        }

class HelpWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.content_manager = HelpContentManager()
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(self.content_manager.get_ui_text('window_title'))
        self.resize(800, 600)
        self.setupUI()
        self.switch_language('ja') # デフォルトを日本語に

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar.setFixedHeight(40) # 明示的に高さを設定

        self.home_action = QPushButton("🏠 Home")
        self.home_action.clicked.connect(self.showHomePage)
        toolbar_layout.addWidget(self.home_action)

        # 中央にスペーサーを追加して左右のボタンを分離
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        toolbar_layout.addItem(spacer)

        self.lang_en_action = QPushButton("🇬🇧 English")
        self.lang_en_action.clicked.connect(lambda: self.switch_language('en'))
        toolbar_layout.addWidget(self.lang_en_action)

        self.lang_ja_action = QPushButton("🇯🇵 日本語")
        self.lang_ja_action.clicked.connect(lambda: self.switch_language('ja'))
        toolbar_layout.addWidget(self.lang_ja_action)

        layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.toc_tree = QTreeWidget()
        self.toc_tree.setHeaderHidden(True)
        self.toc_tree.setFixedWidth(220)
        self.toc_tree.itemClicked.connect(self.onTocItemClicked)

        self.help_viewer = QTextBrowser()
        self.help_viewer.setOpenExternalLinks(True)

        splitter.addWidget(self.toc_tree)
        splitter.addWidget(self.help_viewer)
        splitter.setSizes([220, 580])
        layout.addWidget(splitter)
    
    def switch_language(self, lang_code):
        self.content_manager.set_language(lang_code)
        self.setWindowTitle(self.content_manager.get_ui_text('window_title'))
        self.home_action.setToolTip(self.content_manager.get_ui_text('home_tooltip'))
        self.loadTocContent()
        self.showHomePage()

    def loadTocContent(self):
        self.toc_tree.clear()
        toc_structure = self.content_manager.get_toc_structure()
        def add_items(parent_item, items_list):
            for item_data in items_list:
                name, item_id = item_data
                child_item = QTreeWidgetItem([name])
                child_item.setData(0, Qt.UserRole, item_id)
                parent_item.addChild(child_item)
        for category_name, items in toc_structure:
            category_item = QTreeWidgetItem([category_name])
            self.toc_tree.addTopLevelItem(category_item)
            add_items(category_item, items)
        self.toc_tree.expandAll()
    
    def onTocItemClicked(self, item, column):
        item_id = item.data(0, Qt.UserRole)
        if item_id: self.showHelpPage(item_id)
    
    def showHelpPage(self, page_id):
        self.help_viewer.setHtml(self.content_manager.get_content(page_id))
    
    def showHomePage(self):
        self.showHelpPage('home')
class AFMSimulationWorker(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(object)
    status_update = pyqtSignal(str)
    task_done = pyqtSignal(object, QFrame)

    def __init__(self, parent, sim_params, tasks, element_symbols=None, vdw_radii=None, silent_mode=False):
        # 親を持たせて寿命をGUI側に寄せ、GCタイミング依存を減らす
        super().__init__(parent)
        self.parent = parent
        self.sim_params = sim_params
        self.tasks = tasks
        self.element_symbols = element_symbols
        self.vdw_radii = vdw_radii
        self._is_cancelled = False
        self.silent_mode = silent_mode  # ★★★ 軽量モードフラグ ★★★

    def cancel(self):
        self._is_cancelled = True
    
    def __del__(self):
        """
        デストラクタではwait/terminateしない。
        ここで同期停止すると、GC/破棄タイミング次第で「自分自身をwait」してデッドロックし得る。
        停止はAFMSimulator側の明示的なクリーンアップで行う。
        """
        try:
            self.cancel()
        except Exception:
            pass

    def run(self):
        try:
            total_tasks = len(self.tasks)
            if total_tasks == 0:
                self.done.emit(None)
                return

            for i, task in enumerate(self.tasks):
                start_progress = int((i / total_tasks) * 100)
                end_progress = int(((i + 1) / total_tasks) * 100)
                
                task_name = task["name"]
                scan_coords = task["coords"]
                target_panel = task["panel"]

                # ★★★ 軽量モードではプログレス更新を減らす ★★★
                if not self.silent_mode:
                    self.progress.emit(start_progress)
                if self._is_cancelled: break
                
                self.rotated_atom_coords = scan_coords
                if self.sim_params.get('use_vdw', False) and self.element_symbols is not None:
                    sample_surface = self.create_vdw_surface()
                else:
                    sample_surface = self.create_atom_center_surface()
                
                if not self.silent_mode:
                    self.progress.emit(start_progress + int((end_progress - start_progress) * 0.1))
                if self._is_cancelled: break
                
                dx = self.sim_params['scan_size'] / self.sim_params['resolution']
                z_coords = scan_coords[:, 2]
                mol_depth = np.max(z_coords) - np.min(z_coords) if z_coords.size > 0 else 0
                tip_footprint = self.create_igor_style_tip(dx, dx, mol_depth)

                if not self.silent_mode:
                    self.progress.emit(start_progress + int((end_progress - start_progress) * 0.2))
                if self._is_cancelled: break

                QThread.msleep(50)

                resolution = self.sim_params['resolution']
                afm_image = np.zeros((resolution, resolution), dtype=np.float64)
                
                # ▼▼▼ プログレスバー更新ロジックの修正 ▼▼▼
                last_emitted_progress = -1 # 前回送信した進捗値を記録

                for r in range(resolution):
                    if self._is_cancelled: break
                    afm_image[r, :] = _calculate_dilation_row(r, sample_surface, tip_footprint)
                    
                    # ★★★ 軽量モードではプログレス更新をさらに減らす ★★★
                    if not self.silent_mode:
                        task_progress_fraction = 0.2 + (((r + 1) / resolution) * 0.8)
                        current_overall_progress = start_progress + int(task_progress_fraction * (end_progress - start_progress))
                        
                        # 計算された進捗パーセントが前回から変化した場合のみ信号を送る
                        if current_overall_progress > last_emitted_progress:
                            self.progress.emit(current_overall_progress)
                            last_emitted_progress = current_overall_progress
                # ▲▲▲ 修正完了 ▲▲▲

                if self._is_cancelled: break
                self.task_done.emit(afm_image, target_panel)

            if self._is_cancelled:
                # ★★★ 削除：ステータス表示を無効化 ★★★
                # self.status_update.emit("Calculation cancelled.")
                pass
            else:
                # ★★★ 削除：ステータス表示を無効化 ★★★
                # self.status_update.emit("All tasks completed!")
                pass
            
            if not self.silent_mode:
                self.progress.emit(100)
            self.done.emit(None)

        except Exception as e:
            print(f"An error occurred during the AFM simulation: {e}")
            self.done.emit(None)
    
    def create_vdw_surface(self):
        """ファンデルワールス半径を考慮した表面マップを作成する。"""
        resolution = self.sim_params['resolution']
        scan_size = self.sim_params['scan_size']
        center_x = self.sim_params['center_x']
        center_y = self.sim_params['center_y']
        pixel_size = scan_size / resolution
        
        x_start = center_x - scan_size / 2.0
        y_start = center_y - scan_size / 2.0
        
        min_z = np.min(self.rotated_atom_coords[:, 2]) if self.rotated_atom_coords.size > 0 else 0
        
        if self.rotated_atom_coords.size == 0:
            return np.full((resolution, resolution), 0.0, dtype=np.float64)

        atom_radii = np.array([self.vdw_radii.get(e, self.vdw_radii['other']) for e in self.element_symbols], dtype=np.float64)
        
        surface_map = _create_vdw_surface_loop(
            resolution, pixel_size, x_start, y_start, min_z,
            self.rotated_atom_coords, atom_radii
        )
        
        # デバッグ情報を表示
        center_idx = resolution // 2
        if surface_map.size > 0:
            center_h = surface_map[center_idx, center_idx]
            origin_h = surface_map[0, 0]
            #print(f"Surface map debug - Center: {center_h:.3f}, Origin: {origin_h:.3f}")
            #print(f"Surface map range: {surface_map.min():.3f} to {surface_map.max():.3f}")
        
        return surface_map


    def create_atom_center_surface(self):
        """UIで指定されたスキャンサイズと中心座標に基づいて、原子中心のZ座標から表面マップを作成"""
        resolution = self.sim_params['resolution']
        scan_size = self.sim_params['scan_size']
        center_x = self.sim_params['center_x']
        center_y = self.sim_params['center_y']
        pixel_size = scan_size / resolution
        
        x_start = center_x - scan_size / 2.0
        y_start = center_y - scan_size / 2.0
        
        min_z = np.min(self.rotated_atom_coords[:, 2]) if self.rotated_atom_coords.size > 0 else 0
        surface_map = np.full((resolution, resolution), min_z - 5.0, dtype=np.float64)

        if self.rotated_atom_coords.size == 0:
            return surface_map

        atom_x, atom_y, atom_z = self.rotated_atom_coords.T
        atom_z -= min_z
        ix = np.floor((atom_x - x_start) / pixel_size).astype(np.int32)
        iy = np.floor((atom_y - y_start) / pixel_size).astype(np.int32)

        mask = (ix >= 0) & (ix < resolution) & (iy >= 0) & (iy < resolution)
        if np.any(mask):
            np.maximum.at(surface_map, (iy[mask], ix[mask]), atom_z[mask])
        
        surface_map[surface_map < min_z - 4.0] = 0.0 # 原子がないピクセルは高さ0とする

        return surface_map

    def create_igor_style_tip(self, dx, dy, mol_z_range):
        """UI基準のピクセルサイズ(dx,dy)と分子の高さ(mol_z_range)から探針を作成"""
        R = self.sim_params['tip_radius']
        miniR = self.sim_params['minitip_radius']
        alpha_deg = self.sim_params['tip_angle']
        tip_shape = self.sim_params['tip_shape']
        alpha_rad = np.radians(alpha_deg)

        if ((tip_shape == 'cone') or (tip_shape == 'sphere')):
            r_crit = R * np.cos(alpha_rad)
            z_offset = (R / np.sin(alpha_rad)) - R
            z_crit_related = R - r_crit / np.tan(alpha_rad)
            if z_crit_related > mol_z_range:
                max_tip_radius_nm = np.sqrt(max(0, R**2 - (R - mol_z_range)**2))
            else:
                max_tip_radius_nm = (mol_z_range + z_offset) * np.tan(alpha_rad)
        else: # Paraboloid
            max_tip_radius_nm = np.sqrt(max(0, 2 * R * mol_z_range))

        tip_pixel_radius = int(np.ceil(max_tip_radius_nm / dx))
        tip_size = 2 * tip_pixel_radius + 1
        if tip_size < 1: tip_size = 1
        center_distance = (tip_size - 1) / 2
        
        tip_wave = np.zeros((tip_size, tip_size), dtype=np.float64)
        y_indices, x_indices = np.indices(tip_wave.shape)
        
        r_i = np.sqrt(((x_indices - center_distance) * dx)**2 + ((y_indices - center_distance) * dy)**2)
        if tip_shape == 'cone':
            r_crit = R * np.cos(alpha_rad)
            z_offset = (R / np.sin(alpha_rad)) - R
            sphere_mask = r_i <= r_crit
            cone_mask = r_i > r_crit
            tip_wave[sphere_mask] = R - np.sqrt(R**2 - r_i[sphere_mask]**2)
            tip_wave[cone_mask] = (r_i[cone_mask] / np.tan(alpha_rad)) - z_offset
        elif tip_shape == 'sphere':
            r_crit = R * np.cos(alpha_rad)
            z_offset = (R / np.sin(alpha_rad)) - R
            sphere_mask = r_i <= r_crit
            cone_mask = r_i > r_crit
            miniSphere_mask = r_i < miniR
            tip_wave[sphere_mask] = 2*miniR + R - np.sqrt(R**2 - r_i[sphere_mask]**2)
            tip_wave[cone_mask] = (r_i[cone_mask] / np.tan(alpha_rad)) - z_offset + 2*miniR
            tip_wave[miniSphere_mask] = miniR - np.sqrt(miniR**2 - r_i[miniSphere_mask]**2)
        else: # Paraboloid
            tip_wave = (r_i**2) / (2 * R)

        if np.any(tip_wave):
            tip_wave -= np.min(tip_wave)
        return tip_wave
    
    def simulate_views_blocking(self, desired_keys):
        """
        Run simulation only for desired view keys (['XY_Frame','YZ_Frame','ZX_Frame'])
        blocking this method until finished.
        """
        # Map internal keys to checkboxes
        key_to_check = {
            "XY_Frame": self.afm_x_check,
            "YZ_Frame": self.afm_y_check,
            "ZX_Frame": self.afm_z_check
        }
        # Save original states
        original = {k: key_to_check[k].isChecked() for k in key_to_check}
        try:
            # Apply new checkbox states
            for k, cb in key_to_check.items():
                cb.blockSignals(True)
                cb.setChecked(k in desired_keys)
                cb.blockSignals(False)
            # Kick simulation
            self.run_simulation()
            loop = QEventLoop()
            def _quit_once(_):
                if loop.isRunning():
                    loop.quit()
            self.simulation_done.connect(_quit_once)
            loop.exec_()
        finally:
            # Restore original states
            for k, cb in key_to_check.items():
                cb.blockSignals(True)
                cb.setChecked(original[k])
                cb.blockSignals(False)
            # Restore display layout
            self.update_afm_display()

    def handle_save_image(self):
        """Export one or more simulated AFM images (PNG) with optional incremental rotation."""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Data", "No simulation data available to save.")
            return
        
        # Build available (only those already simulated)
        available_keys = list(self.simulation_results.keys())
        display_names = {"XY_Frame": "XY View", "YZ_Frame": "YZ View", "ZX_Frame": "ZX View"}
        
        dlg = SaveAFMImageDialog(available_keys, display_names, self.get_active_dataset_id(), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        result = dlg.get_result()
        selected_view_keys = result['selected_views']
        rot_inc = result['drot']
        base_name = result['base_name']
        
        if not selected_view_keys:
            QMessageBox.warning(self, "No Selection", "No views selected.")
            return
        
        # Map for filename friendly
        def key_to_short(k):
            return {
                "XY_Frame": "XY",
                "YZ_Frame": "YZ",
                "ZX_Frame": "ZX"
            }.get(k, k.replace("_Frame", ""))
        
        # Prepare directory & ensure last_import_dir is valid
        directory = ""
        if self.last_import_dir and os.path.isdir(self.last_import_dir):
            directory = self.last_import_dir
        if not directory:
            directory = os.getcwd()
        
        # Save original rotation
        orig_rx = self.rotation_widgets['X']['spin'].value()
        orig_ry = self.rotation_widgets['Y']['spin'].value()
        orig_rz = self.rotation_widgets['Z']['spin'].value()
        
        apply_rotation = any(abs(v) > 1e-6 for v in rot_inc.values())
        
        try:
            if apply_rotation:
                # Apply incremental rotation (add to current)
                self.rotation_widgets['X']['spin'].setValue(self.normalize_angle(orig_rx + rot_inc['x']))
                self.rotation_widgets['Y']['spin'].setValue(self.normalize_angle(orig_ry + rot_inc['y']))
                self.rotation_widgets['Z']['spin'].setValue(self.normalize_angle(orig_rz + rot_inc['z']))
                # Force apply transform & run simulation for required views
                self.apply_structure_rotation()
                self.simulate_views_blocking(selected_view_keys)
            
            # Export each selected view
            export_count = 0
            for key in selected_view_keys:
                if key not in self.simulation_results:
                    continue
                data = self.simulation_results[key]
                # Normalize to 8-bit grayscale
                mn, mx = float(np.min(data)), float(np.max(data))
                if mx <= mn:
                    norm = np.zeros_like(data, dtype=np.uint8)
                else:
                    norm = ((data - mn) / (mx - mn) * 255).astype(np.uint8)
                
                # Resize to 512x512
                try:
                    from PIL import Image
                except ImportError:
                    QMessageBox.critical(self, "Missing Pillow", "Install Pillow to export images (pip install Pillow).")
                    return
                img = Image.fromarray(norm, mode='L')
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS
                img = img.resize((512, 512), resample=resample_filter)
                
                fname = f"{base_name}_{key_to_short(key)}_dx{rot_inc['x']:+.0f}_dy{rot_inc['y']:+.0f}_dz{rot_inc['z']:+.0f}.png"
                save_path = os.path.join(directory, fname)
                try:
                    img.save(save_path)
                    export_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to save {save_path}: {e}")
            
            if export_count:
                QMessageBox.information(self, "Export Complete", f"Exported {export_count} image(s) to:\n{directory}")
            else:
                QMessageBox.warning(self, "No Export", "No images were exported.")
        
        finally:
            # Restore original rotation if we changed it
            if apply_rotation:
                self.rotation_widgets['X']['spin'].setValue(orig_rx)
                self.rotation_widgets['Y']['spin'].setValue(orig_ry)
                self.rotation_widgets['Z']['spin'].setValue(orig_rz)
                self.apply_structure_rotation()
                # (Optionally regenerate original visible views if needed)
                # self.simulate_views_blocking(available_keys)

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, window_instance):
        super().__init__()
        self.window = window_instance
        self.panning = False
        self.actor_rotating = False
        self.pan_anchor_point = None
        self.pan_anchor_z = None

    def OnLeftButtonDown(self):
        rwi = self.GetInteractor()
        
        # macOSのCommandキーにも対応するため、GetCommandKey()のチェックを追加
        is_ctrl_or_cmd_pressed = rwi.GetControlKey() or rwi.GetCommandKey()
        
        # Ctrl(またはCmd)キーが押されているか最初にチェック
        if is_ctrl_or_cmd_pressed and not rwi.GetShiftKey():
            self.actor_rotating = True
            self.StartRotate()
        # Shiftキーが押されているかチェック
        elif rwi.GetShiftKey() and not is_ctrl_or_cmd_pressed:
            self.panning = True
            self.StartPan()
            renderer = self.GetCurrentRenderer()
            if renderer is None: return
            x, y = rwi.GetEventPosition()
            self.pan_anchor_z = renderer.GetZ(x, y)
            self.pan_anchor_point = self.get_world_point(renderer, x, y, self.pan_anchor_z)
        else:
            # 何も押されていなければ、通常のカメラ回転
            super().OnLeftButtonDown()

    def OnLeftButtonUp(self):
        if self.actor_rotating:
            self.actor_rotating = False
            self.EndRotate()
            
            # ドラッグ終了時の高解像度シミュレーション
            if hasattr(self.window, 'interactive_update_check') and self.window.interactive_update_check.isChecked():
                if hasattr(self.window, 'schedule_high_res_simulation'):
                    self.window.schedule_high_res_simulation()
                    
        elif self.panning:
            self.panning = False
            self.EndPan()
        else:
            super().OnLeftButtonUp()

    def OnMouseMove(self):
        if self.actor_rotating:
            self.RotateActor()
        elif self.panning:
            rwi = self.GetInteractor()
            renderer = self.GetCurrentRenderer()
            if renderer is None: return
            camera = renderer.GetActiveCamera()
            x, y = rwi.GetEventPosition()
            new_point = self.get_world_point(renderer, x, y, self.pan_anchor_z)
            motion_vector = [new_point[i] - self.pan_anchor_point[i] for i in range(3)]
            cam_pos = list(camera.GetPosition())
            cam_fp = list(camera.GetFocalPoint())
            camera.SetPosition([cam_pos[i] - motion_vector[i] for i in range(3)])
            camera.SetFocalPoint([cam_fp[i] - motion_vector[i] for i in range(3)])
            rwi.Render()
        else:
            super().OnMouseMove()

    def RotateActor(self):
        """カメラビューに応じた構造回転を実行（オイラー角ベース）"""
        rwi = self.GetInteractor()
        renderer = self.GetCurrentRenderer()
        if renderer is None:
            return

        # マウスの移動量を取得
        dx = rwi.GetEventPosition()[0] - rwi.GetLastEventPosition()[0]
        dy = rwi.GetEventPosition()[1] - rwi.GetLastEventPosition()[1]

        # カメラ情報を取得
        camera = renderer.GetActiveCamera()
        camera_pos = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        view_up = camera.GetViewUp()
               
        # ビュー方向ベクトル（カメラから焦点への方向）
        view_dir = np.array([
            focal_point[0] - camera_pos[0],
            focal_point[1] - camera_pos[1],
            focal_point[2] - camera_pos[2]
        ])
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # 上方向ベクトル
        up_dir = np.array(view_up)
        up_dir = up_dir / np.linalg.norm(up_dir)
        
        # 右方向ベクトル（外積で計算）
        right_dir = np.cross(view_dir, up_dir)
        right_dir = right_dir / np.linalg.norm(right_dir)
        
        # スクリーン座標でのマウス移動を回転軸と角度に変換
        rotation_scale = 0.5  # 回転感度
        
        # オイラー角での回転量を計算
        # 各軸に対する寄与を計算
        h_rotation = dx * rotation_scale  # 水平回転
        v_rotation = -dy * rotation_scale  # 垂直回転（符号反転）
        
        # より直接的で確実なアプローチ：スクリーン座標をワールド座標の回転に直接マッピング
        # マウスの水平移動 → スクリーンの水平軸周りの回転
        # マウスの垂直移動 → スクリーンの垂直軸周りの回転
        
        # スクリーンの水平軸（右方向）周りの回転
        horizontal_axis_rotation = h_rotation  # dx * rotation_scale
        
        # スクリーンの垂直軸（上方向）周りの回転  
        vertical_axis_rotation = v_rotation    # -dy * rotation_scale
        
        # スクリーン座標系での回転をワールド座標系のX、Y、Z軸回転に変換
        # 右方向ベクトル（right_dir）と上方向ベクトル（up_dir）を使用
        
        # 水平回転（right_dir周り）をワールド軸に分解
        total_x_rotation = right_dir[0] * horizontal_axis_rotation
        total_y_rotation = right_dir[1] * horizontal_axis_rotation  
        total_z_rotation = right_dir[2] * horizontal_axis_rotation
        
        # 垂直回転（up_dir周り）をワールド軸に分解して加算
        total_x_rotation += up_dir[0] * vertical_axis_rotation
        total_y_rotation += up_dir[1] * vertical_axis_rotation
        total_z_rotation += up_dir[2] * vertical_axis_rotation
        
        # 現在のUI値を取得して増分を加算
        if hasattr(self.window, 'rotation_widgets'):
            current_x = self.window.rotation_widgets['X']['spin'].value()
            current_y = self.window.rotation_widgets['Y']['spin'].value()
            current_z = self.window.rotation_widgets['Z']['spin'].value()
            
            # 新しい回転値を計算（-180〜180の範囲に正規化）
            new_x = self.normalize_angle(current_x + total_x_rotation)
            new_y = self.normalize_angle(current_y + total_y_rotation)
            new_z = self.normalize_angle(current_z + total_z_rotation)
            
            # UIウィジェットを更新（シグナルをブロックして無限ループを防止）
            self.window.rotation_widgets['X']['spin'].blockSignals(True)
            self.window.rotation_widgets['X']['slider'].blockSignals(True)
            self.window.rotation_widgets['Y']['spin'].blockSignals(True)
            self.window.rotation_widgets['Y']['slider'].blockSignals(True)
            self.window.rotation_widgets['Z']['spin'].blockSignals(True)
            self.window.rotation_widgets['Z']['slider'].blockSignals(True)
            
            self.window.rotation_widgets['X']['spin'].setValue(new_x)
            self.window.rotation_widgets['X']['slider'].setValue(int(new_x * 10))
            self.window.rotation_widgets['Y']['spin'].setValue(new_y)
            self.window.rotation_widgets['Y']['slider'].setValue(int(new_y * 10))
            self.window.rotation_widgets['Z']['spin'].setValue(new_z)
            self.window.rotation_widgets['Z']['slider'].setValue(int(new_z * 10))
            
            self.window.rotation_widgets['X']['spin'].blockSignals(False)
            self.window.rotation_widgets['X']['slider'].blockSignals(False)
            self.window.rotation_widgets['Y']['spin'].blockSignals(False)
            self.window.rotation_widgets['Y']['slider'].blockSignals(False)
            self.window.rotation_widgets['Z']['spin'].blockSignals(False)
            self.window.rotation_widgets['Z']['slider'].blockSignals(False)
            
            # 構造回転を適用
            self.window.apply_structure_rotation()
            
            # インタラクティブモード用の更新
            if hasattr(self.window, 'interactive_update_check') and self.window.interactive_update_check.isChecked():
                self.window.run_simulation_immediate_controlled()
    
    def normalize_angle(self, angle):
        """角度を-180〜180の範囲に正規化"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def get_world_point(self, renderer, x, y, z):
        renderer.SetDisplayPoint(float(x), float(y), float(z))
        renderer.DisplayToWorld()
        world_point = renderer.GetWorldPoint()
        return [world_point[0] / world_point[3], 
                world_point[1] / world_point[3], 
                world_point[2] / world_point[3]]
    
class AFMSimulator(QMainWindow):

    simulation_done = pyqtSignal(object)
    simulation_progress = pyqtSignal(int)

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("AFM Simulator")
        
        # Windows固有の設定
        #if sys.platform.startswith('win'):
            # Windowsでの安定性向上のための設定
        #    self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        #    self.setAttribute(Qt.WA_NoSystemBackground, True)
        
        # スタンドアロンアプリケーションなのでwindow_managerは使用しない
        
        # ウィンドウの位置とサイズを復元
        self.settings = QSettings("pyNuD", "AFM_Simulator")
        self.restore_geometry()
        
        # 設定が保存されていない場合はデフォルトサイズを使用
        if not self.settings.contains("geometry"):
            # ウィンドウサイズ設定
            from PyQt5.QtWidgets import QDesktopWidget
            desktop = QDesktopWidget()
            screen_geometry = desktop.screenGeometry()
            
            width = int(screen_geometry.width() * 0.6)
            height = int(screen_geometry.height() * 0.6)
            
            # ★★★ 変更点: 最小サイズを小さく設定 ★★★
            self.setMinimumSize(600, 450)
            self.resize(width, height)
        self.center_on_screen()
        
        # データ格納
        self.atoms_data = None
        self.pdb_name = ""
        self.pdb_id = ""
        # 二次構造情報を格納（(chain_id, residue_id) -> 'H'/'E'/'C'）
        self.secondary_structure = {}
        # ★★★ MRC関連の変数を追加 ★★★
        self.mrc_data = None
        self.mrc_data_original = None  # 元のMRCデータ（フリップ前）
        self.mrc_voxel_size_nm = 1.0 / 10.0
        self.mrc_threshold = 0.3
        self.mrc_z_flip = True  # Z軸フリップ状態（デフォルトでTrue）
        # ★★★ ここまで ★★★
        self.tip_actor = None
        self.sample_actor = None
        self.bonds_actor = None
        self.simulation_results = {} 
        self.raw_simulation_results = {}

        self.help_window = None
        
        # 変換を二段に分離
        self.base_transform = vtk.vtkTransform()
        self.base_transform.Identity()
        
        self.local_transform = vtk.vtkTransform()
        self.local_transform.Identity()
        self.local_transform.PostMultiply()  # ローカル回転を右に積む（オブジェクト座標で回す）
        
        self.combined_transform = vtk.vtkTransform()
        self.combined_transform.Identity()
        self.combined_transform.PostMultiply()
        
        # 後方互換性のため残す
        self.molecule_transform = vtk.vtkTransform()
        self.last_import_dir = ""
        
        # スライダ差分適用用の前回値
        self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # インタラクティブモード用にユーザーの解像度選択を記憶する変数 ★★★
        self.user_selected_resolution = ""

        # マウスイベント用の状態変数
        self.actor_rotating = False
        self.panning = False
        
        # カラー・ライティング設定
        self.current_bg_color = (0.05, 0.05, 0.05)
        self.current_single_color = (0.5, 0.7, 0.9)
        self.brightness_factor = 1.0
        
        # AFM像表示用の参照
        self.afm_x_widget = None
        self.afm_y_widget = None
        self.afm_z_widget = None
        
        # 簡単で確実なカラーマップ
        self.element_colors = {
            'C': (0.3, 0.3, 0.3), 'O': (1.0, 0.3, 0.3), 'N': (0.3, 0.3, 1.0),
            'H': (0.9, 0.9, 0.9), 'S': (1.0, 1.0, 0.3), 'P': (1.0, 0.5, 0.0),
            'other': (0.7, 0.7, 0.7)
        }
        
        # チェーンカラー
        self.chain_colors = [
            (0.2, 0.8, 0.2), (0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.2),
            (0.8, 0.2, 0.8), (0.2, 0.8, 0.8), (1.0, 0.5, 0.0), (0.5, 0.0, 0.8),
        ]
        
         # ★★★ ここから追加 ★★★
        # 一般的なファンデルワールス半径 (nm)
        self.vdw_radii = {
            'H': 0.120, 'C': 0.170, 'N': 0.155, 'O': 0.152,
            'P': 0.180, 'S': 0.180, 'other': 0.170
        }
        
        # バックグラウンド処理からのシグナル
        #self.simulation_done = pyqtSignal(object)
        #self.simulation_progress = pyqtSignal(int)
        # ★★★ ここまで追加 ★★★

        # AFMパラメータ
        self.afm_params = {
            'tip_radius': 2.0, 'tip_shape': 'cone', 'tip_angle': 15.0,
            'tip_x': 0.0, 'tip_y': 0.0, 'tip_z': 5.0,
        }
        
        
        # ★★★ 修正点: 呼び出し順序を変更 ★★★
        self.setup_ui()    # UIウィジェットを全て作成
        self.setup_vtk()   # VTK環境を初期化

         # シミュレーション結果が一つでもあれば、各種保存ボタンを有効化する
        self.simulation_done.connect(self.on_simulation_finished)

        # PyInstaller環境を検出して適切なファイルパスを決定
        if getattr(sys, 'frozen', False):
            # PyInstallerで作成されたアプリの場合
            # ユーザーのホームディレクトリ内に設定ファイルを作成
            home_dir = Path.home()
            config_dir = home_dir / "pyNuD_config"
            config_dir.mkdir(exist_ok=True)
            self.settings_file = str(config_dir / "simulator_config.json")
        else:
            # 開発環境の場合
            self.settings_file = "config.json"

        # ★★★ 追加: 全ての準備が完了した後に、UIの初期状態を設定 ★★★
        self.update_tip_ui(self.tip_shape_combo.currentText())

        self.load_settings()

    def setup_vtk(self):
        """VTK環境のセットアップ"""
        # VTKウィジェットが存在することを確認
        if not hasattr(self, 'vtk_widget') or self.vtk_widget is None:
            print("Error: VTK widget not found")
            return
            
        try:
            # レンダラー作成
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(*self.current_bg_color)
            
            # スライダー操作フラグの初期化
            self.tip_slider_pressed = False
            
            # スピンボックスの入力方法フラグ（True=キー入力中, False=マウス/ボタン操作）
            self.scan_size_keyboard_input = False
            self.tip_radius_keyboard_input = False
            self.minitip_radius_keyboard_input = False
            self.tip_angle_keyboard_input = False
            
            # デバウンス用のタイマー
            self.scan_size_debounce_timer = None
            self.tip_radius_debounce_timer = None
            self.minitip_radius_debounce_timer = None
            self.tip_angle_debounce_timer = None
            
            # アンチエイリアシング
            render_window = self.vtk_widget.GetRenderWindow()
            render_window.AddRenderer(self.renderer)
            render_window.SetMultiSamples(4)
            
            # インタラクター設定
            self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

            # ★★★ ここを修正 ★★★
            # CustomInteractorStyleにメインウィンドウ(self)への参照を渡す
            style = CustomInteractorStyle(self)
            self.interactor.SetInteractorStyle(style)
            
            # macOSでのイベントハンドラー問題を回避するため、直接イベントを監視
            # 元のイベントハンドラーを保存
            self.original_mouse_press = self.vtk_widget.mousePressEvent
            self.original_mouse_move = self.vtk_widget.mouseMoveEvent
            self.original_mouse_release = self.vtk_widget.mouseReleaseEvent
            
            self.vtk_widget.mousePressEvent = self.on_mouse_press
            self.vtk_widget.mouseMoveEvent = self.on_mouse_move
            self.vtk_widget.mouseReleaseEvent = self.on_mouse_release
            
            # ライティング改善
            self.setup_lighting()
            
            # 座標軸追加
            self.add_axes()
            
            # 初期カメラ設定
            self.reset_camera()
            
            # レンダリング開始
            self.interactor.Initialize()
            
        except Exception as e:
            print(f"VTK setup error: {e}")
        
    def center_on_screen(self):
        """ウィンドウを画面中央に配置"""
        from PyQt5.QtWidgets import QDesktopWidget
        frame_geometry = self.frameGeometry()
        desktop = QDesktopWidget()
        center_point = desktop.availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())
    
    def restore_geometry(self):
        """ウィンドウの位置とサイズを復元"""
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except Exception:
            pass  # 復元に失敗した場合は無視
    
    def save_geometry(self):
        """ウィンドウの位置とサイズを保存"""
        try:
            geometry = self.saveGeometry()
            self.settings.setValue("geometry", geometry)
        except Exception:
            pass  # 保存に失敗した場合は無視
        
    def setup_ui(self):
        """UIセットアップ"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # ★★★ 修正: progress_containerの作成をメソッドの先頭に移動 ★★★
        # 呼び出し先の create_vtk_panel で使用されるため、先に定義する必要があります。
        self.progress_container = QWidget()
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(5, 3, 5, 5)
        progress_layout.setSpacing(3)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 12px; color: #1E8449; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #aaa; border-radius: 5px; text-align: center; font-weight: bold; height: 18px; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 4px; }
        """)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        self.progress_container.setVisible(False)
        # ★★★ 修正ここまで ★★★

        main_layout = QHBoxLayout(central_widget)
        
        # --- メインのスプリッター ---
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # --- 左右パネルの作成とスプリッターへの追加 ---
        left_scroll_area = QScrollArea()
        left_panel = self.create_control_panel()
        left_scroll_area.setWidget(left_panel)
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll_area.setMinimumWidth(280)
        self.main_splitter.addWidget(left_scroll_area)
        
        right_scroll_area = QScrollArea()
        right_scroll_area.setWidgetResizable(True)
        right_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_panel = self.create_vtk_panel()
        right_scroll_area.setWidget(right_panel)
        self.main_splitter.addWidget(right_scroll_area)
        
        self.main_splitter.setSizes([280, 1020])
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)

    def create_menu_bar(self):
        """アプリケーションのメニューバーを作成する"""
        # ヘルプウィンドウの参照を初期化
        self.help_window = None
        
        # QMainWindow標準のメニューバーを取得
        menu_bar = self.menuBar()
        
        # 「Help」メニューを作成
        help_menu = menu_bar.addMenu("&Help")
        
        # 「View Help」アクションを作成し、クリックされたらshow_help_windowを呼び出す
        show_help_action = QAction("View Help...", self)
        show_help_action.setShortcut("F1")
        show_help_action.triggered.connect(self.show_help_window)
        help_menu.addAction(show_help_action)
        # Manual（マニュアル）: プラグイン内HELP_HTMLを表示
        manual_action = QAction("Manual", self)
        manual_action.triggered.connect(self.showHelpDialog)
        help_menu.addAction(manual_action)

    def showHelpDialog(self):
        """Help → Manual でマニュアルを表示（日本語/English 切替可能）"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QButtonGroup
        dialog = QDialog(self)
        dialog.setMinimumSize(500, 500)
        dialog.resize(600, 650)
        layout_dlg = QVBoxLayout(dialog)
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language / 言語:"))
        btn_ja = QPushButton("日本語", dialog)
        btn_en = QPushButton("English", dialog)
        btn_ja.setCheckable(True)
        btn_en.setCheckable(True)
        lang_grp = QButtonGroup(dialog)
        lang_grp.addButton(btn_ja)
        lang_grp.addButton(btn_en)
        lang_grp.setExclusive(True)
        _BTN_SELECTED = "QPushButton { background-color: #007aff; color: white; font-weight: bold; }"
        _BTN_NORMAL = "QPushButton { background-color: #e5e5e5; color: black; }"
        lang_row.addWidget(btn_ja)
        lang_row.addWidget(btn_en)
        lang_row.addStretch()
        layout_dlg.addLayout(lang_row)
        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        css = "body { font-size: 15px; line-height: 1.6; } .step { margin: 8px 0; padding: 6px 0; font-size: 15px; } .feature-box { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; } h1 { font-size: 22px; color: #2c3e50; } h2 { font-size: 18px; color: #2c3e50; margin-top: 18px; } ul { padding-left: 24px; font-size: 15px; }"
        browser.document().setDefaultStyleSheet(css)
        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)

        def set_lang(use_ja):
            btn_ja.setChecked(use_ja)
            btn_en.setChecked(not use_ja)
            btn_ja.setStyleSheet(_BTN_SELECTED if use_ja else _BTN_NORMAL)
            btn_en.setStyleSheet(_BTN_SELECTED if not use_ja else _BTN_NORMAL)
            if use_ja:
                browser.setHtml("<html><body>" + HELP_HTML_JA.strip() + "</body></html>")
                dialog.setWindowTitle("AFMシミュレータ - マニュアル")
                close_btn.setText("閉じる")
            else:
                browser.setHtml("<html><body>" + HELP_HTML_EN.strip() + "</body></html>")
                dialog.setWindowTitle("AFM Simulator - Manual")
                close_btn.setText("Close")

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        layout_dlg.addWidget(browser)
        layout_dlg.addWidget(close_btn)
        set_lang(False)
        dialog.exec_()

    def show_help_window(self):
        """ヘルプウィンドウを作成して表示する"""
        # ウィンドウが既に開いている場合は、新しく作らずに最前面に表示
        if self.help_window is None or not self.help_window.isVisible():
            self.help_window = HelpWindow(parent=None)
            #self.help_window = HelpWindow(self)
            self.help_window.show()
        else:
            self.help_window.activateWindow()
            self.help_window.raise_()

    def create_control_panel(self):
        """左側のコントロールパネル作成"""
        panel = QWidget()
        panel.setMinimumWidth(270)
        layout = QVBoxLayout(panel)
        layout.setSpacing(6) # 8から変更
        layout.setContentsMargins(8, 8, 8, 8) # 10から変更

        # ▼▼▼ 全体のフォントサイズを小さくするスタイルシートを追加 ▼▼▼
        panel.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
            }
            QLabel, QCheckBox, QPushButton, QComboBox, QDoubleSpinBox {
                font-size: 11px;
            }
        """)
        
        # File Import (統合: PDB/CIF/MRC)
        file_import_group = QGroupBox("File Import")
        file_import_layout = QVBoxLayout(file_import_group)
        
        self.import_btn = QPushButton("Import File...")
        self.import_btn.setMinimumHeight(35)
        self.import_btn.setToolTip("Load structure file (PDB/CIF/MRC) for AFM simulation\nAFMシミュレーション用の構造ファイル（PDB/CIF/MRC）を読み込み")
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.import_btn.clicked.connect(self.import_file)
        file_import_layout.addWidget(self.import_btn)

        # インポートされたファイル名の表示のみ（ドロップは PDB Structure 領域で受付）
        self.file_label = QLabel("File Name: (none)")
        self.file_label.setStyleSheet("color: #666; font-size: 12px;")
        file_import_layout.addWidget(self.file_label)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        file_import_layout.addWidget(self.progress_bar)

        layout.addWidget(file_import_group)
        
        # ★★★ Density Thresholdセクションを追加 ★★★
        self.mrc_group = QGroupBox("Density Threshold")
        mrc_layout = QGridLayout(self.mrc_group)

        self.mrc_threshold_label = QLabel(f"Value: {self.mrc_threshold:.2f}")
        mrc_layout.addWidget(self.mrc_threshold_label, 0, 0, 1, 2)

        self.mrc_threshold_slider = QSlider(Qt.Horizontal)
        self.mrc_threshold_slider.setRange(0, 100)
        self.mrc_threshold_slider.setValue(int(self.mrc_threshold * 100))
        # スライダーを動かしている最中はラベル更新のみ
        self.mrc_threshold_slider.valueChanged.connect(self.on_mrc_threshold_changed)
        # スライダーを離したときに再描画
        self.mrc_threshold_slider.sliderReleased.connect(self.on_mrc_threshold_released)
        mrc_layout.addWidget(self.mrc_threshold_slider, 1, 0, 1, 2)

        # Z軸フリップ用のチェックボックスを追加（デフォルトで有効）
        self.mrc_z_flip_check = QCheckBox("Flip Z-axis")
        self.mrc_z_flip_check.setChecked(True)  # デフォルトで有効
        self.mrc_z_flip_check.stateChanged.connect(self.on_mrc_z_flip_changed)
        self.mrc_z_flip_check.setToolTip("Toggle Z-axis flip for MRC data (default: enabled)")
        mrc_layout.addWidget(self.mrc_z_flip_check, 2, 0, 1, 2)

        self.mrc_group.setEnabled(False) # 最初は無効
        layout.addWidget(self.mrc_group)
        # ★★★ ここまで ★★★

        # 原子統計
        stats_group = QGroupBox("Atom Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_labels = {}
        for atom_type in ['Total', 'C', 'O', 'N', 'H', 'Other']:
            label = QLabel(f"{atom_type}: 0")
            label.setFont(QFont(gv.standardFont, 9))  # Use system-appropriate font
            stats_layout.addWidget(label)
            self.stats_labels[atom_type] = label

            
        layout.addWidget(stats_group)
        
        # 表示設定
        display_group = QGroupBox("Display Settings")
        display_layout = QGridLayout(display_group)
        
        # 表示スタイル
        display_layout.addWidget(QLabel("Style:"), 0, 0)
        self.style_combo = QComboBox()
        self.style_combo.addItems([
            "Ball & Stick", "Stick Only", "Spheres", "Points", "Wireframe", "Simple Cartoon", "Ribbon"
        ])
        self.style_combo.currentTextChanged.connect(self.update_display)
        display_layout.addWidget(self.style_combo, 0, 1)
        
        # カラーリング
        display_layout.addWidget(QLabel("Color:"), 1, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            "By Element", "By Chain", "Single Color", "By B-Factor"
        ])
        self.color_combo.currentTextChanged.connect(self.on_color_scheme_changed)
        display_layout.addWidget(self.color_combo, 1, 1)
        
        # 原子選択
        display_layout.addWidget(QLabel("Show:"), 2, 0)
        self.atom_combo = QComboBox()
        self.atom_combo.addItems(["All Atoms", "Heavy Atoms", "Backbone", "C", "N", "O"])
        self.atom_combo.currentTextChanged.connect(self.update_display)
        display_layout.addWidget(self.atom_combo, 2, 1)
        
        # サイズ
        display_layout.addWidget(QLabel("Size:"), 3, 0)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(10, 200)
        self.size_slider.setValue(100)
        self.size_slider.valueChanged.connect(self.update_display)
        display_layout.addWidget(self.size_slider, 3, 1)
        
        # 透明度
        display_layout.addWidget(QLabel("Opacity:"), 4, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.update_display)
        display_layout.addWidget(self.opacity_slider, 4, 1)
        
        # 品質設定
        display_layout.addWidget(QLabel("Quality:"), 5, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Fast", "Good", "High"])
        self.quality_combo.setCurrentText("Good")
        self.quality_combo.currentTextChanged.connect(self.update_display)
        display_layout.addWidget(self.quality_combo, 5, 1)
        
        layout.addWidget(display_group)
        
        # カラー・ライティング設定
        color_group = QGroupBox("Color & Lighting Settings")
        color_layout = QGridLayout(color_group)
        
        # 背景色設定
        color_layout.addWidget(QLabel("Background:"), 0, 0)
        self.bg_color_btn = QPushButton("Choose Color")
        self.bg_color_btn.setMinimumHeight(30)
        self.bg_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #191919;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                border-color: #777;
            }
        """)
        self.bg_color_btn.clicked.connect(self.choose_background_color)
        color_layout.addWidget(self.bg_color_btn, 0, 1)
        
        # 明るさ調整
        color_layout.addWidget(QLabel("Brightness:"), 1, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(20, 200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        color_layout.addWidget(self.brightness_slider, 1, 1)
        
        self.brightness_label = QLabel("100%")
        self.brightness_label.setMinimumWidth(40)
        color_layout.addWidget(self.brightness_label, 1, 2)
        
        # 単色モード用カラー選択
        color_layout.addWidget(QLabel("Single Color:"), 2, 0)
        self.single_color_btn = QPushButton("Choose Color")
        self.single_color_btn.setMinimumHeight(30)
        self.single_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #7FB3D3;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                border-color: #777;
            }
        """)
        self.single_color_btn.clicked.connect(self.choose_single_color)
        color_layout.addWidget(self.single_color_btn, 2, 1)
        
        # 環境光設定
        color_layout.addWidget(QLabel("Ambient:"), 3, 0)
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 50)
        self.ambient_slider.setValue(10)
        self.ambient_slider.valueChanged.connect(self.update_lighting)
        color_layout.addWidget(self.ambient_slider, 3, 1)
        
        self.ambient_label = QLabel("10%")
        self.ambient_label.setMinimumWidth(40)
        color_layout.addWidget(self.ambient_label, 3, 2)
        
        # スペキュラ設定
        color_layout.addWidget(QLabel("Specular:"), 4, 0)
        self.specular_slider = QSlider(Qt.Horizontal)
        self.specular_slider.setRange(0, 100)
        self.specular_slider.setValue(60)
        self.specular_slider.valueChanged.connect(self.update_material)
        color_layout.addWidget(self.specular_slider, 4, 1)
        
        self.specular_label = QLabel("60%")
        self.specular_label.setMinimumWidth(40)
        color_layout.addWidget(self.specular_label, 4, 2)
        
        # プリセットボタン
        preset_layout = QHBoxLayout()
        
        pymol_btn = QPushButton("PyMOL Style")
        pymol_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        pymol_btn.clicked.connect(self.apply_pymol_style)
        preset_layout.addWidget(pymol_btn)
        
        dark_btn = QPushButton("Dark Theme")
        dark_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        dark_btn.clicked.connect(self.apply_dark_theme)
        preset_layout.addWidget(dark_btn)
        
        color_layout.addLayout(preset_layout, 5, 0, 1, 3)
        
        layout.addWidget(color_group)


        # AFM探針設定
        tip_group = QGroupBox("AFM Tip Settings")
        tip_layout = QGridLayout(tip_group)

        # Row 0: Shape
        tip_layout.addWidget(QLabel("Shape:"), 0, 0)
        self.tip_shape_combo = QComboBox()
        self.tip_shape_combo.addItems(["Cone", "Sphere", "Paraboloid"])
        self.tip_shape_combo.setToolTip("AFM tip shape\nAFM探針の形状")
        self.tip_shape_combo.currentTextChanged.connect(self.update_tip_ui)
        tip_layout.addWidget(self.tip_shape_combo, 0, 1)

        # Row 1: Radius (of cone part)
        tip_layout.addWidget(QLabel("Radius (nm):"), 1, 0)
        self.tip_radius_spin = QDoubleSpinBox()
        self.tip_radius_spin.setRange(0.5, 30.0)
        self.tip_radius_spin.setValue(0.5)
        self.tip_radius_spin.setSingleStep(0.1)
        self.tip_radius_spin.setDecimals(1)
        self.tip_radius_spin.setToolTip("AFM tip radius in nanometers\nAFM探針の半径（ナノメートル）")
        self.tip_radius_spin.valueChanged.connect(self.tip_radius_value_changed)
        self.tip_radius_spin.editingFinished.connect(self.tip_radius_editing_finished)
        self.tip_radius_spin.keyPressEvent = self.tip_radius_key_press_event
        tip_layout.addWidget(self.tip_radius_spin, 1, 1)

        # Row 2: Radius of Minitip (for Sphere shape)
        self.minitip_label = QLabel("Radius of Minitip (nm):")
        tip_layout.addWidget(self.minitip_label, 2, 0)
        self.minitip_radius_spin = QDoubleSpinBox()
        self.minitip_radius_spin.setRange(0.1, 10.0)
        self.minitip_radius_spin.setValue(0.1)
        self.minitip_radius_spin.setSingleStep(0.1)
        self.minitip_radius_spin.setToolTip("Radius of minitip in nanometers\nミニチップの半径（ナノメートル）")
        self.minitip_radius_spin.setDecimals(1)
        self.minitip_radius_spin.valueChanged.connect(self.minitip_radius_value_changed)
        self.minitip_radius_spin.editingFinished.connect(self.minitip_radius_editing_finished)
        self.minitip_radius_spin.keyPressEvent = self.minitip_radius_key_press_event
        tip_layout.addWidget(self.minitip_radius_spin, 2, 1)

        # Row 3: Angle (for Cone/Sphere)
        self.tip_angle_label = QLabel("Angle (deg):")
        tip_layout.addWidget(self.tip_angle_label, 3, 0)
        self.tip_angle_spin = QDoubleSpinBox()
        self.tip_angle_spin.setRange(5.0, 35.0)
        self.tip_angle_spin.setValue(5)
        self.tip_angle_spin.setSingleStep(1.0)
        self.tip_angle_spin.valueChanged.connect(self.tip_angle_value_changed)
        self.tip_angle_spin.editingFinished.connect(self.tip_angle_editing_finished)
        self.tip_angle_spin.keyPressEvent = self.tip_angle_key_press_event
        tip_layout.addWidget(self.tip_angle_spin, 3, 1)
        
        # Row 4: Tip Info
        self.tip_info_label = QLabel("Tip Info: -")
        self.tip_info_label.setStyleSheet("""
            QLabel {
                font-size: 9px; color: #666; background-color: #f9f9f9;
                border: 1px solid #ddd; border-radius: 3px; padding: 3px;
            }
        """)
        self.tip_info_label.setWordWrap(True)
        tip_layout.addWidget(self.tip_info_label, 4, 0, 1, 2)

        layout.addWidget(tip_group)
        
        # 探針位置制御
        pos_group = QGroupBox("Tip Position Control")
        pos_layout = QGridLayout(pos_group)
        
        # X位置
        pos_layout.addWidget(QLabel("X (nm):"), 0, 0)
        self.tip_x_slider = QSlider(Qt.Horizontal)
        self.tip_x_slider.setRange(-50, 50)
        self.tip_x_slider.setValue(0)
        self.tip_x_slider.setToolTip("AFM tip X position in nanometers\nAFM探針のX位置（ナノメートル）")
        self.tip_x_slider.valueChanged.connect(self.update_tip_position)
        self.tip_x_slider.sliderPressed.connect(self.on_tip_slider_pressed)
        self.tip_x_slider.sliderReleased.connect(self.on_tip_slider_released)
        pos_layout.addWidget(self.tip_x_slider, 0, 1)
        self.tip_x_label = QLabel("0.0")
        self.tip_x_label.setMinimumWidth(30)
        pos_layout.addWidget(self.tip_x_label, 0, 2)
        
        # Y位置
        pos_layout.addWidget(QLabel("Y (nm):"), 1, 0)
        self.tip_y_slider = QSlider(Qt.Horizontal)
        self.tip_y_slider.setRange(-50, 50)
        self.tip_y_slider.setValue(0)
        self.tip_y_slider.setToolTip("AFM tip Y position in nanometers\nAFM探針のY位置（ナノメートル）")
        self.tip_y_slider.valueChanged.connect(self.update_tip_position)
        self.tip_y_slider.sliderPressed.connect(self.on_tip_slider_pressed)
        self.tip_y_slider.sliderReleased.connect(self.on_tip_slider_released)
        pos_layout.addWidget(self.tip_y_slider, 1, 1)
        self.tip_y_label = QLabel("0.0")
        self.tip_y_label.setMinimumWidth(30)
        pos_layout.addWidget(self.tip_y_label, 1, 2)
        
        # Z位置
        pos_layout.addWidget(QLabel("Z (nm):"), 2, 0)
        self.tip_z_slider = QSlider(Qt.Horizontal)
        self.tip_z_slider.setRange(10, 100)
        self.tip_z_slider.setValue(25)
        self.tip_z_slider.setToolTip("AFM tip Z position (height) in nanometers\nAFM探針のZ位置（高さ）（ナノメートル）")
        self.tip_z_slider.valueChanged.connect(self.update_tip_position)
        self.tip_z_slider.sliderPressed.connect(self.on_tip_slider_pressed)
        self.tip_z_slider.sliderReleased.connect(self.on_tip_slider_released)
        pos_layout.addWidget(self.tip_z_slider, 2, 1)
        self.tip_z_label = QLabel("5.0")
        self.tip_z_label.setMinimumWidth(30)
        pos_layout.addWidget(self.tip_z_label, 2, 2)
        
        layout.addWidget(pos_group)
        
        # シミュレーション設定
        sim_group = QGroupBox("AFM Simulation")
        sim_layout = QGridLayout(sim_group)
        
        # スキャンサイズ
        sim_layout.addWidget(QLabel("Scan Size (nm):"), 0, 0)
        self.scan_size_spin = QDoubleSpinBox()
        self.scan_size_spin.setRange(5.0, 100.0)
        self.scan_size_spin.setValue(20.0)
        self.scan_size_spin.setDecimals(1)
        self.scan_size_spin.setToolTip("Scan area size in nanometers\nスキャン領域のサイズ（ナノメートル）")
        # カスタムイベントハンドラーを設定
        self.scan_size_spin.valueChanged.connect(self.scan_size_value_changed)
        self.scan_size_spin.editingFinished.connect(self.scan_size_editing_finished)
        # キー入力の開始を検出
        self.scan_size_spin.keyPressEvent = self.scan_size_key_press_event
        sim_layout.addWidget(self.scan_size_spin, 0, 1)
        
        # 解像度
        sim_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["32x32", "64x64", "128x128", "256x256"])
        self.resolution_combo.setCurrentText("64x64")
        self.resolution_combo.setToolTip("Simulation image resolution\nシミュレーション画像の解像度")
        sim_layout.addWidget(self.resolution_combo, 1, 1)
        
        # 解像度変更時のイベントハンドラーを接続
        self.resolution_combo.currentTextChanged.connect(self.on_resolution_changed)

        # 原子サイズを考慮するかのチェックボックス
        self.use_vdw_check = QCheckBox("Consider atom size (vdW)")
        self.use_vdw_check.setToolTip(
            "Treat atoms as spheres with van der Waals radii\n"
            "チェックすると、原子の中心ではなくファンデルワールス半径を考慮した表面で計算します。\n"
            "（より物理的に正確ですが、像は滑らかになります）"
        )
        self.use_vdw_check.setChecked(False) # デフォルトはIgor方式
        sim_layout.addWidget(self.use_vdw_check, 2, 0, 1, 2) # チェックボックスを 行2 に配置

        self.apply_filter_check = QCheckBox("Apply Low-pass Filter")
        self.apply_filter_check.setToolTip("Apply FFT low-pass filter to match experimental resolution\nFFTローパスフィルターを適用して実験解像度に合わせる")
        sim_layout.addWidget(self.apply_filter_check, 3, 0, 1, 2)

        filter_param_layout = QHBoxLayout()
        filter_param_layout.addSpacing(20) # インデント
        filter_param_layout.addWidget(QLabel("Cutoff Wavelength (nm):"))
        self.filter_cutoff_spin = QDoubleSpinBox()
        self.filter_cutoff_spin.setRange(0.1, 20.0)
        self.filter_cutoff_spin.setValue(2.0)
        self.filter_cutoff_spin.setDecimals(1)
        self.filter_cutoff_spin.setSingleStep(0.1)
        self.filter_cutoff_spin.setToolTip("Cutoff wavelength for low-pass filter\nローパスフィルターのカットオフ波長")
        filter_param_layout.addWidget(self.filter_cutoff_spin)
        sim_layout.addLayout(filter_param_layout, 4, 0, 1, 2)

        # 2. チェックボックスの状態でスピンボックスの有効/無効を切り替え
        self.apply_filter_check.toggled.connect(self.filter_cutoff_spin.setEnabled)
        self.apply_filter_check.toggled.connect(self.process_and_display_all_images)
        self.filter_cutoff_spin.valueChanged.connect(self.start_filter_update_timer)
        self.filter_cutoff_spin.setEnabled(False)


        # 1. Interactive Update チェックボックスを追加
        self.interactive_update_check = QCheckBox("Interactive Update (Low-Res)")
        self.interactive_update_check.setToolTip(
            "Automatically update simulation at low resolution when parameters change\n"
            "パラメータ変更時に低解像度でシミュレーションを自動更新"
        )
        # デフォルトで有効にする
        self.interactive_update_check.setChecked(True)
        # 2. チェックボックスの状態が変化したら handle_interactive_update_toggle を呼び出す
        self.interactive_update_check.toggled.connect(self.handle_interactive_update_toggle)
        sim_layout.addWidget(self.interactive_update_check, 5, 0, 1, 2)


        # シミュレーション実行
        self.simulate_btn = QPushButton("Run AFM Simulation")
        self.simulate_btn.setMinimumHeight(40)
        self.simulate_btn.setToolTip("Run AFM simulation with current settings\n現在の設定でAFMシミュレーションを実行")
        self.simulate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.simulate_btn.clicked.connect(self.run_simulation)
        self.simulate_btn.setEnabled(False)
        sim_layout.addWidget(self.simulate_btn, 6, 0, 1, 2)
        
        layout.addWidget(sim_group)
        
        # 表示制御
        view_group = QGroupBox("View Control")
        view_layout = QVBoxLayout(view_group)
        
        self.show_molecule_check = QCheckBox("Show Molecule")
        self.show_molecule_check.setChecked(True)
        self.show_molecule_check.toggled.connect(self.toggle_molecule_visibility)
        view_layout.addWidget(self.show_molecule_check)
        
        self.show_tip_check = QCheckBox("Show AFM Tip")
        self.show_tip_check.setChecked(True)
        self.show_tip_check.toggled.connect(self.toggle_tip_visibility)
        view_layout.addWidget(self.show_tip_check)
        
        self.show_bonds_check = QCheckBox("Show Bonds")
        self.show_bonds_check.setChecked(True)
        self.show_bonds_check.toggled.connect(self.toggle_bonds_visibility)
        view_layout.addWidget(self.show_bonds_check)
        
        reset_view_btn = QPushButton("Reset View")
        reset_view_btn.setToolTip("Reset camera to default view\nカメラをデフォルトビューにリセット")
        reset_view_btn.clicked.connect(self.reset_camera)
        view_layout.addWidget(reset_view_btn)
        
        layout.addWidget(view_group)

        #self.update_tip_ui(self.tip_shape_combo.currentText())
        
        return panel
        
    
    def update_tip_ui(self, shape):
        """探針設定UIの表示を、選択された形状に応じて更新する"""
        shape = shape.lower()
        
        is_sphere = (shape == "sphere")
        is_cone = (shape == "cone")
        
        # Minitip Radius widgets visibility
        self.minitip_label.setVisible(is_sphere)
        self.minitip_radius_spin.setVisible(is_sphere)
        
        # Angle widgets visibility/enabled state
        angle_is_relevant = is_cone or is_sphere
        self.tip_angle_label.setEnabled(angle_is_relevant)
        self.tip_angle_spin.setEnabled(angle_is_relevant)
        
        # Trigger a tip redraw
        self.update_tip()

    # 既存の create_vtk_panel メソッドを、以下の完全なコードで置き換えてください。

    def create_vtk_panel(self):
        """右側のVTK表示パネル作成（上下可変分割 + 下部3分割）"""
        panel = QWidget()
        panel.setMinimumSize(550, 600)
        
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 上下のメインスプリッター
        self.afm_splitter = QSplitter(Qt.Vertical)
        self.afm_splitter.setHandleWidth(8)
        self.afm_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #ddd;
                border: 1px solid #ccc;
            }
            QSplitter::handle:hover {
                background-color: #bbb;
            }
        """)
        
        # --- 上部：PDB構造表示エリア ---
        structure_frame = QFrame()
        structure_frame.setFrameStyle(QFrame.StyledPanel)
        structure_frame.setLineWidth(1)
        structure_layout = QVBoxLayout(structure_frame)
        structure_layout.setContentsMargins(2, 2, 2, 2)
        structure_layout.setSpacing(2)
        # if 
        structure_label = QLabel("Drop PDB, CIF, MRC files here")
        structure_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #333;
                padding: 3px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        structure_label.setAlignment(Qt.AlignCenter)
        structure_label.setMaximumHeight(25)
        structure_layout.addWidget(structure_label)

        structure_layout.addWidget(self.progress_container)
        
        # VTKウィンドウとコントロールパネルを配置するための垂直スプリッター
        self.view_control_splitter = QSplitter(Qt.Vertical)
        self.view_control_splitter.setHandleWidth(6)
        self.view_control_splitter.setStyleSheet("""
            QSplitter::handle:vertical {
                height: 6px;
                background-color: #e0e0e0;
                border-top: 1px solid #c0c0c0;
                border-bottom: 1px solid #c0c0c0;
            }
            QSplitter::handle:vertical:hover {
                background-color: #cccccc;
            }
        """)
        
        self.vtk_widget = QVTKRenderWindowInteractor(self.view_control_splitter)
        self.vtk_widget.setAcceptDrops(True)
        self.vtk_widget.installEventFilter(self)
        self.view_control_splitter.addWidget(self.vtk_widget)

        rotation_controls = self.create_rotation_controls()
        self.view_control_splitter.addWidget(rotation_controls)
        
        self.view_control_splitter.setSizes([500, 150])
        self.view_control_splitter.setCollapsible(0, False)
        self.view_control_splitter.setCollapsible(1, False)

        structure_layout.addWidget(self.view_control_splitter)

        # --- 下部：AFM像表示エリア --- (省略されていた部分を復元)
        afm_frame = QFrame()
        afm_frame.setFrameStyle(QFrame.StyledPanel)
        afm_frame.setLineWidth(1)
        afm_frame.setMinimumHeight(200)
        afm_frame.setMaximumHeight(350)
        afm_layout = QVBoxLayout(afm_frame)
        afm_layout.setContentsMargins(2, 2, 2, 2)
        afm_layout.setSpacing(2)
        
        afm_header_layout = QHBoxLayout()
        afm_header_layout.setContentsMargins(3, 3, 3, 3)
        afm_header_layout.setSpacing(0)
        
        afm_label = QLabel("Simulated AFM Images")
        afm_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #333;
                padding: 3px;
                background-color: #f0f0f0;
                border-radius: 3px;
                margin-right: 0px;
            }
        """)
        afm_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        afm_header_layout.addWidget(afm_label)
        
        afm_header_layout.addSpacing(10)

        
    
        
        self.afm_x_check = QCheckBox("XY")
        self.afm_y_check = QCheckBox("YZ")
        self.afm_z_check = QCheckBox("ZX")
        
        self.afm_x_check.setChecked(True)
        self.afm_y_check.setChecked(False)
        self.afm_z_check.setChecked(False)
        
        checkbox_style = """
            QCheckBox {
                font-size: 10px; font-weight: bold; color: #555;
                spacing: 3px; margin-right: 2px;
            }
            QCheckBox::indicator { width: 14px; height: 14px; border-radius: 2px; }
            QCheckBox::indicator:checked { background-color: #4CAF50; border: 2px solid #45a049; }
            QCheckBox::indicator:unchecked { background-color: white; border: 2px solid #ccc; }
            QCheckBox::indicator:hover { border-color: #888; }
        """
        
        self.afm_x_check.setStyleSheet(checkbox_style)
        self.afm_y_check.setStyleSheet(checkbox_style)
        self.afm_z_check.setStyleSheet(checkbox_style)        
        
        self.afm_x_check.toggled.connect(self.update_afm_display)
        self.afm_y_check.toggled.connect(self.update_afm_display)
        self.afm_z_check.toggled.connect(self.update_afm_display)

        # 新しい接続（チェックがONになったらシミュレーションを自動実行する）
        self.afm_x_check.toggled.connect(self.run_simulation_on_view_change)
        self.afm_y_check.toggled.connect(self.run_simulation_on_view_change)
        self.afm_z_check.toggled.connect(self.run_simulation_on_view_change)
 
        
        afm_header_layout.addWidget(self.afm_x_check)
        afm_header_layout.addSpacing(12)
        afm_header_layout.addWidget(self.afm_y_check)
        afm_header_layout.addSpacing(12)
        afm_header_layout.addWidget(self.afm_z_check)

        self.save_asd_button = QPushButton("💾 Save as ASD...")
        self.save_asd_button.setToolTip("Save AFM simulation data as ASD file\nAFMシミュレーションデータをASDファイルとして保存")
        self.save_asd_button.setStyleSheet("""
            QPushButton { padding: 3px 8px; font-size: 10px; background-color: #17a2b8; color: white; border-radius: 3px; }
            QPushButton:hover { background-color: #117a8b; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.save_asd_button.clicked.connect(self.handle_save_asd)
        self.save_asd_button.setEnabled(False) # 初期状態は無効
        afm_header_layout.addWidget(self.save_asd_button)

        self.save_image_button = QPushButton("🖼️ Save Image...") # アイコンを少し変更
        self.save_image_button.setToolTip("Save AFM simulation image as PNG/TIFF file\nAFMシミュレーション画像をPNG/TIFFファイルとして保存")
        self.save_image_button.setStyleSheet("""
            QPushButton { padding: 3px 8px; font-size: 10px; background-color: #007bff; color: white; border-radius: 3px; }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.save_image_button.clicked.connect(self.handle_save_image)
        self.save_image_button.setEnabled(False)
        afm_header_layout.addWidget(self.save_image_button)
        
        afm_header_widget = QWidget()
        afm_header_widget.setLayout(afm_header_layout)
        afm_header_widget.setMaximumHeight(30)
        afm_header_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        afm_layout.addWidget(afm_header_widget)
        
        self.afm_images_layout = QHBoxLayout()
        self.afm_images_layout.setSpacing(3)
        self.afm_images_layout.setContentsMargins(0, 0, 0, 0)
        
        # 画像パネルのタイトルを XY View, YZ View, ZX View に変更
        self.afm_x_frame = self.create_afm_image_panel("XY View")
        self.afm_x_frame.setObjectName("XY_Frame") # 追加
        self.afm_y_frame = self.create_afm_image_panel("YZ View")
        self.afm_y_frame.setObjectName("YZ_Frame") # 追加
        self.afm_z_frame = self.create_afm_image_panel("ZX View")
        self.afm_z_frame.setObjectName("ZX_Frame") # 追加
        
        self.afm_images_layout.addWidget(self.afm_x_frame, 1)
        self.afm_images_layout.addWidget(self.afm_y_frame, 1)
        self.afm_images_layout.addWidget(self.afm_z_frame, 1)
        
        afm_layout.addLayout(self.afm_images_layout)
        
        # メインスプリッターにウィジェットを追加
        self.afm_splitter.addWidget(structure_frame)
        self.afm_splitter.addWidget(afm_frame)
        
        self.afm_splitter.setSizes([600, 200])
        self.afm_splitter.setCollapsible(0, False)
        self.afm_splitter.setCollapsible(1, False)
        
        main_layout.addWidget(self.afm_splitter)

        self.update_afm_display()
        
        return panel
    

    def create_rotation_controls(self):
        """PDB構造回転用コントロールと視点コントロールを作成"""
        group = QGroupBox("Structure & View Control (CTRL+Drag can rotate the PDB structure)")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        # メインの水平レイアウト
        main_layout = QHBoxLayout(group)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(8, 12, 8, 8)

        # --- 左側: 回転コントロール ---
        left_widget = QWidget()
        left_layout = QGridLayout(left_widget)
        left_layout.setSpacing(4)  # スペーシングを小さく
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.rotation_widgets = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            label = QLabel(f"Rotation {axis}:")
            spin_box = QDoubleSpinBox()
            spin_box.setRange(-180.0, 180.0)
            spin_box.setDecimals(1)
            spin_box.setSingleStep(1.0)
            spin_box.setSuffix(" °")
            spin_box.setToolTip(f"Rotation {axis} angle in degrees\n{axis}軸の回転角度（度）")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-1800, 1800)
            slider.setToolTip(f"Rotation {axis} slider\n{axis}軸回転スライダー")

            left_layout.addWidget(label, i, 0)
            left_layout.addWidget(spin_box, i, 1)
            left_layout.addWidget(slider, i, 2)
            left_layout.setColumnStretch(2, 1)

            self.rotation_widgets[axis] = {'spin': spin_box, 'slider': slider}
            # ★★★ ここからが修正箇所 ★★★
            # 1. 値が「変化している最中」は、UIの同期のみを行う
            slider.valueChanged.connect(self.sync_rotation_widgets)
            spin_box.valueChanged.connect(self.sync_rotation_widgets)
            
            # 2. 操作が「完了した時」にのみ、3Dモデルの回転とシミュレーションのトリガーを実行
            slider.sliderReleased.connect(self.apply_rotation_and_trigger_simulation)
            spin_box.valueChanged.connect(self.start_rotation_update_timer)
            #spin_box.editingFinished.connect(self.apply_rotation_and_trigger_simulation)
            # ★★★ 修正箇所ここまで ★★★

        # --- 右側: 視点コントロール ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(5, 0, 0, 0)
        
         # 1. ボタンを格納する水平レイアウトを作成
        top_button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset All")
        reset_btn.setToolTip("Reset molecule rotation, tip position, and camera view to initial state\n分子の回転、探針の位置、カメラの視点を初期状態に戻します")
        reset_btn.clicked.connect(self.handle_reset_button_clicked)
        top_button_layout.addWidget(reset_btn) # 水平レイアウトに追加

        # 2. 新しい保存ボタンを作成
        save_view_btn = QPushButton("📷 Save 3D View...")
        save_view_btn.setToolTip("Save the current 3D view as a PNG or TIFF image\n現在の3DビューをPNGまたはTIFF画像として保存")
        save_view_btn.clicked.connect(self.handle_save_3d_view) # 新しいメソッドに接続
        top_button_layout.addWidget(save_view_btn) # 水平レイアウトに追加   

         # 2. Helpボタンをここに追加
        help_btn = QPushButton("❓ Help")
        help_btn.setToolTip("Show parameter explanations (F1)\nパラメータの説明を表示（F1）")
        help_btn.setShortcut("F1")
        help_btn.clicked.connect(self.show_help_window)
        top_button_layout.addWidget(help_btn)

        # Find Initial Plane（XY接触最大）ボタン
        find_plane_btn = QPushButton("Find Initial Plane")
        find_plane_btn.setToolTip("Rotate molecule to maximize XY-plane contact\n分子を回転してXY平面接触を最大化")
        find_plane_btn.clicked.connect(self.handle_find_initial_plane)
        top_button_layout.addWidget(find_plane_btn)

         # 3. 水平レイアウトを垂直レイアウトに追加
        right_layout.addLayout(top_button_layout)
        
        # 標準視点ボタンを水平に配置
        view_btn_layout = QHBoxLayout()
        xy_btn = QPushButton("XY")
        yz_btn = QPushButton("YZ")
        zx_btn = QPushButton("ZX")

        xy_btn.setToolTip("XY平面が画面に平行になるように視点を変更します (Z軸視点)")
        yz_btn.setToolTip("YZ平面が画面に平行になるように視点を変更します (X軸視点)")
        zx_btn.setToolTip("ZX平面が画面に平行になるように視点を変更します (Y軸視点)")

        xy_btn.clicked.connect(lambda: self.set_standard_view('xy'))
        yz_btn.clicked.connect(lambda: self.set_standard_view('yz'))
        zx_btn.clicked.connect(lambda: self.set_standard_view('zx'))

        view_btn_layout.addWidget(xy_btn)
        view_btn_layout.addWidget(yz_btn)
        view_btn_layout.addWidget(zx_btn)

        right_layout.addWidget(reset_btn)
        right_layout.addLayout(view_btn_layout)
        #right_layout.addStretch() # ボタンを上部に寄せる

        # 左右のウィジェットをメインレイアウトに追加
        main_layout.addWidget(left_widget, stretch=3) # 回転コントロールに多くのスペースを割り当てる
        main_layout.addWidget(right_widget, stretch=1)
                
        return group

    def handle_reset_button_clicked(self):
        """Resetボタンが押されたときの処理（回転、探針位置、カメラをリセット）"""
        self.reset_structure_rotation()
        self.reset_tip_position()
        self.reset_camera()

    def reset_tip_position(self):
        """探針の位置をUIのデフォルト値にリセットする"""
        if hasattr(self, 'tip_x_slider'):
            self.tip_x_slider.setValue(0)
            self.tip_y_slider.setValue(0)
            self.tip_z_slider.setValue(25) # UI定義時の初期値

    def set_standard_view(self, view_plane):
        """XY, YZ, ZXの標準視点にカメラをセットする（現在の距離を保持）"""
        if not hasattr(self, 'renderer') or (not self.sample_actor and not (hasattr(self, 'mrc_actor') and self.mrc_actor is not None)):
            return

        camera = self.renderer.GetActiveCamera()
        
        # 現在のカメラの状態を保存
        current_position = camera.GetPosition()
        current_focal_point = camera.GetFocalPoint()
        current_view_up = camera.GetViewUp()
        
        # 現在のカメラと焦点の距離を計算
        distance = np.sqrt(sum((current_position[i] - current_focal_point[i]) ** 2 for i in range(3)))
        
        # 分子の中心を計算
        bbox = vtk.vtkBoundingBox()
        if self.sample_actor and self.show_molecule_check.isChecked():
            bbox.AddBounds(self.sample_actor.GetBounds())
        if self.bonds_actor and self.show_bonds_check.isChecked():
            bbox.AddBounds(self.bonds_actor.GetBounds())
        # MRCサーフェス
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            bbox.AddBounds(self.mrc_actor.GetBounds())
        
        if not bbox.IsValid():
            # 分子が表示されていない場合は、現在の焦点を中心とする
            molecule_center = current_focal_point
        else:
            # バウンディングボックスの中心を計算
            molecule_center = [0.0, 0.0, 0.0]
            bbox.GetCenter(molecule_center)

        # --- カメラの向きを設定（距離は保持） ---
        if view_plane == 'xy':
            # Z軸の上から見る (Y軸が画面の上方向)
            direction = np.array([0, 0, 1])
            new_position = np.array(molecule_center) + direction * distance
            camera.SetPosition(new_position[0], new_position[1], new_position[2])
            camera.SetFocalPoint(molecule_center)
            camera.SetViewUp(0, 1, 0)
        elif view_plane == 'yz':
            # X軸の正方向から見る (Z軸が画面の上方向)
            direction = np.array([1, 0, 0])
            new_position = np.array(molecule_center) + direction * distance
            camera.SetPosition(new_position[0], new_position[1], new_position[2])
            camera.SetFocalPoint(molecule_center)
            camera.SetViewUp(0, 0, 1)
        elif view_plane == 'zx':
            # Y軸の負方向から見る (Z軸が画面の上方向)
            direction = np.array([0, -1, 0])
            new_position = np.array(molecule_center) + direction * distance
            camera.SetPosition(new_position[0], new_position[1], new_position[2])
            camera.SetFocalPoint(molecule_center)
            camera.SetViewUp(0, 0, 1)
        
        # PDB分子の回転適用後、MRCアクターにも同じ回転を適用
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            self.mrc_actor.SetUserTransform(self.molecule_transform)
        
        # Tipの表示制御
        if hasattr(self, 'tip_actor') and self.tip_actor:
            if view_plane == 'xy':
                # XY平面視点の際は自動的にTipを不可視化
                self.tip_actor.SetVisibility(False)
            else:
                # XY平面以外の視点では"Show AFM Tip"チェックボックスの状態に従う
                if hasattr(self, 'show_tip_check'):
                    self.tip_actor.SetVisibility(self.show_tip_check.isChecked())
        
        self.vtk_widget.GetRenderWindow().Render()

    def on_xy_checked(self, checked):
        if checked:
            # 他のチェックボックスの信号を一時的にブロック
            self.afm_y_check.blockSignals(True)
            self.afm_z_check.blockSignals(True)
            # 他をオフにする
            self.afm_y_check.setChecked(False)
            self.afm_z_check.setChecked(False)
            # ブロックを解除
            self.afm_y_check.blockSignals(False)
            self.afm_z_check.blockSignals(False)
        self.update_afm_display()

    def on_yz_checked(self, checked):
        if checked:
            self.afm_x_check.blockSignals(True)
            self.afm_z_check.blockSignals(True)
            self.afm_x_check.setChecked(False)
            self.afm_z_check.setChecked(False)
            self.afm_x_check.blockSignals(False)
            self.afm_z_check.blockSignals(False)
        self.update_afm_display()

    def on_zx_checked(self, checked):
        if checked:
            self.afm_x_check.blockSignals(True)
            self.afm_y_check.blockSignals(True)
            self.afm_x_check.setChecked(False)
            self.afm_y_check.setChecked(False)
            self.afm_x_check.blockSignals(False)
            self.afm_y_check.blockSignals(False)
        self.update_afm_display()
    
    def sync_rotation_widgets(self):
        """スライダーとスピンボックスの値を同期させ、Interactive Updateが有効な場合はリアルタイム更新も実行"""
        sender = self.sender()
        changed_axis = None
        for axis, widgets in self.rotation_widgets.items():
            if sender is widgets['slider'] or sender is widgets['spin']:
                changed_axis = axis
                break
        if not changed_axis: return

        widgets = self.rotation_widgets[changed_axis]
        spin_box = widgets['spin']
        slider = widgets['slider']

        # 無限ループを防ぐため、シグナルをブロックしながら値を設定
        if isinstance(sender, QSlider):
            new_val = sender.value() / 10.0
            spin_box.blockSignals(True)
            spin_box.setValue(new_val)
            spin_box.blockSignals(False)
        elif isinstance(sender, QDoubleSpinBox):
            new_val = sender.value()
            slider.blockSignals(True)
            slider.setValue(int(new_val * 10))
            slider.blockSignals(False)
        
        # 構造回転を適用
        self.apply_structure_rotation()
        
        # Interactive Updateが有効で、スライダーからの変更の場合はリアルタイム更新
        if (hasattr(self, 'interactive_update_check') and 
            self.interactive_update_check.isChecked() and 
            isinstance(sender, QSlider)):
            self.run_simulation_immediate_controlled()
    
    def start_rotation_update_timer(self):
        """
        スピンボックスからの回転更新を遅延させるためのタイマーを開始/リセットする。
        これにより、連続クリック中に不要な更新が走るのを防ぐ。
        """
        # タイマーがまだ存在しなければ作成する
        if not hasattr(self, 'rotation_update_timer'):
            self.rotation_update_timer = QTimer(self)  # 親ウィンドウを設定
            self.rotation_update_timer.setSingleShot(True)
            self.rotation_update_timer.timeout.connect(self.apply_rotation_and_trigger_simulation)
        
        # 500ミリ秒後に更新を実行するようにタイマーを開始（またはリセット）
        self.rotation_update_timer.start(500)

    def apply_rotation_and_trigger_simulation(self):
        """UIの操作完了後に、3Dモデルの回転を適用し、シミュレーションをトリガーする"""
        #print("Rotation change finished. Applying transform and triggering simulation if interactive.")
        self.apply_structure_rotation()
        
        # Interactive Updateが有効な場合は高解像度シミュレーションをスケジュール
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            if hasattr(self, 'schedule_high_res_simulation'):
                self.schedule_high_res_simulation()

    def update_rotation(self):
        """回転コントロールの変更を検知し、UIを同期して回転を適用"""
        sender = self.sender()

        # どの軸のウィジェットが変更されたか特定
        changed_axis = None
        for axis, widgets in self.rotation_widgets.items():
            if sender is widgets['slider'] or sender is widgets['spin']:
                changed_axis = axis
                break
        
        if not changed_axis:
            return

        widgets = self.rotation_widgets[changed_axis]
        spin_box = widgets['spin']
        slider = widgets['slider']

        # senderに応じて値を同期（無限ループを防ぐためシグナルをブロック）
        if isinstance(sender, QSlider):
            new_val = sender.value() / 10.0
            spin_box.blockSignals(True)
            spin_box.setValue(new_val)
            spin_box.blockSignals(False)
        elif isinstance(sender, QDoubleSpinBox):
            new_val = sender.value()
            slider.blockSignals(True)
            slider.setValue(int(new_val * 10))
            slider.blockSignals(False)
        else:
            return

        # 実際の回転を適用
        self.apply_structure_rotation()

    def update_actor_transform(self):
        """base_transformとlocal_transformを組み合わせてアクターに適用"""
        try:
            # 変換行列を安全に初期化
            self.combined_transform.Identity()
            
            # 変換行列の妥当性をチェック
            if self.base_transform is not None:
                base_matrix = self.base_transform.GetMatrix()
                if self._is_transform_matrix_valid(base_matrix):
                    self.combined_transform.Concatenate(self.base_transform)
                else:
                    print("[WARNING] Invalid base_transform, using identity")
            
            if self.local_transform is not None:
                local_matrix = self.local_transform.GetMatrix()
                if self._is_transform_matrix_valid(local_matrix):
                    self.combined_transform.Concatenate(self.local_transform)
                else:
                    print("[WARNING] Invalid local_transform, using identity")
            
            # 最終的な変換行列の妥当性をチェック
            final_matrix = self.combined_transform.GetMatrix()
            if not self._is_transform_matrix_valid(final_matrix):
                print("[WARNING] Invalid combined_transform, resetting to identity")
                self.combined_transform.Identity()
            
            # アクターに適用
            if self.sample_actor:
                self.sample_actor.SetUserTransform(self.combined_transform)
            if self.bonds_actor:
                self.bonds_actor.SetUserTransform(self.combined_transform)
            if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
                self.mrc_actor.SetUserTransform(self.combined_transform)
            if hasattr(self, 'vtk_widget'):
                self.vtk_widget.GetRenderWindow().Render()
                
        except Exception as e:
            print(f"[WARNING] Error in update_actor_transform: {e}")
            # エラーが発生した場合は単位行列にリセット
            self.combined_transform.Identity()
    
    def _is_transform_matrix_valid(self, vtk_matrix):
        """VTK変換行列が妥当かどうかをチェック"""
        try:
            for i in range(4):
                for j in range(4):
                    element = vtk_matrix.GetElement(i, j)
                    if not np.isfinite(element) or abs(element) > 1e6:
                        return False
            return True
        except Exception:
            return False

    def apply_structure_rotation(self):
        """スライダー（絶対角）→ 差分回転をlocal_transformに適用"""
        if not hasattr(self, 'rotation_widgets'):
            return
        
        # PDBデータまたはMRCデータのどちらかが読み込まれているかチェック
        if (getattr(self, 'atoms_data', None) is None and 
            not (hasattr(self, 'mrc_data') and self.mrc_data is not None)):
            return

        # 実行中ワーカーのガード（元コードと同様）
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            return

        rx = float(self.rotation_widgets['X']['spin'].value())
        ry = float(self.rotation_widgets['Y']['spin'].value())
        rz = float(self.rotation_widgets['Z']['spin'].value())

        dx = rx - self.prev_rot['x']
        dy = ry - self.prev_rot['y']
        dz = rz - self.prev_rot['z']

        # ローカル軸で差分回転を積む
        self.local_transform.RotateX(dx)
        self.local_transform.RotateY(dy)
        self.local_transform.RotateZ(dz)

        self.prev_rot['x'], self.prev_rot['y'], self.prev_rot['z'] = rx, ry, rz
        self.update_actor_transform()
        self.trigger_interactive_simulation()

    def handle_find_initial_plane(self):
        """
        XY平面への"寝かせ"を自動化：
          1) PCAで最小分散軸をZに合わせて初期姿勢を作る
          2) 近傍微調整（±8°）で 厚み h = z_max - z_min を最小化
          3) 同程度の厚みなら、接触原子数（z - z_min ≤ eps_nm）最大でタイブレーク
        """
        # PDBデータまたはMRCデータのどちらかが読み込まれているかチェック
        if getattr(self, 'atoms_data', None) is None and not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            QMessageBox.warning(self, "Warning", "PDBまたはMRCファイルが読み込まれていません。")
            return
        
        # データタイプに応じて座標を取得
        if getattr(self, 'atoms_data', None) is not None:
            # PDBデータの場合
            coords = np.column_stack([self.atoms_data['x'],
                                      self.atoms_data['y'],
                                      self.atoms_data['z']]).astype(float)
        else:
            # MRCデータの場合
            coords = self._get_mrc_surface_coords()
            if coords is None:
                QMessageBox.warning(self, "Warning", "MRCデータから座標を取得できませんでした。")
                return

        # ---- 元座標と重心 ----
        c = coords.mean(axis=0)
        X = coords - c  # 重心回り

        # ---- PCAで基準姿勢（PC3→Z, PC1→X）----
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        Rr0 = VT.T                                  # 列: PC1,PC2,PC3
        if np.linalg.det(Rr0) < 0:                  # 右手系を担保
            Rr0[:, 0] *= -1
        if Rr0[2, 2] < 0:                           # Z(PC3)は+Zを向くよう反転
            Rr0[:, 2] *= -1
            Rr0[:, 1] *= -1  # 右手系維持

        # ---- 評価関数（厚み＋接触数）----
        eps_nm = 0.20        # 接触しきい値（必要に応じて 0.2–0.5nm）
        thick_tie_tol = 1e-4 # 厚みの同点判定 [nm]（=0.0001nm ≒ 0.001Å）

        def Rx(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        def Ry(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])

        def evaluate(Rr):
            """(厚みh, 接触原子数cnt) を返す。厚みは最小化、cntは最大化。"""
            try:
                # 入力データの妥当性チェック
                if X is None or len(X) == 0:
                    return float('inf'), 0
                
                if Rr is None or len(Rr) == 0:
                    return float('inf'), 0
                
                # 数値の安全性を確保（より厳格な範囲制限）
                X_safe = np.clip(X, -1000, 1000)  # より狭い範囲
                Rr_safe = np.clip(Rr, -100, 100)  # 回転行列は小さい値
                
                # 行列の形状をチェック
                if X_safe.shape[1] != Rr_safe.shape[0]:
                    return float('inf'), 0
                
                # ゼロ除算を防ぐためのチェック
                if np.any(np.abs(Rr_safe) < 1e-10):
                    return float('inf'), 0
                
                # 行列積を安全に実行
                try:
                    with np.errstate(all='ignore'):  # 警告を無視
                        z = (X_safe @ Rr_safe)[:, 2]
                except (OverflowError, RuntimeWarning, ValueError):
                    return float('inf'), 0
                
                # NaNやInfをチェック
                if not np.all(np.isfinite(z)):
                    return float('inf'), 0
                
                # 結果の妥当性チェック
                zmin = z.min()
                zmax = z.max()
                h = zmax - zmin
                
                # 厚みが異常に大きい場合は無効
                if h > 1000 or not np.isfinite(h) or h < 0:
                    return float('inf'), 0
                
                # 接触原子数の計算
                try:
                    cnt = int(np.count_nonzero(z - zmin <= eps_nm))
                    if cnt < 0 or cnt > len(z):
                        return float('inf'), 0
                except (OverflowError, ValueError):
                    return float('inf'), 0
                
                return h, cnt
                
            except Exception as e:
                # 全ての例外をキャッチ
                return float('inf'), 0

        # 初期値
        best_Rr = Rr0
        best_h, best_cnt = evaluate(best_Rr)

        # ---- 近傍粗探索（±8°）----
        grid = np.deg2rad(np.array([-8,-6,-4,-2,0,2,4,6,8], dtype=float))
        for ax in grid:          # X tilt
            for ay in grid:      # Y tilt
                Rr = Rr0 @ (Ry(ay) @ Rx(ax))
                h, cnt = evaluate(Rr)
                if (h < best_h - thick_tie_tol) or (abs(h - best_h) <= thick_tie_tol and cnt > best_cnt):
                    best_Rr, best_h, best_cnt = Rr, h, cnt

        # ---- （任意）微細探索：±2°でもう一段詰める ----
        fine = np.deg2rad(np.array([-2,-1,0,1,2], dtype=float))
        base = best_Rr
        for ax in fine:
            for ay in fine:
                Rr = base @ (Ry(ay) @ Rx(ax))
                h, cnt = evaluate(Rr)
                if (h < best_h - thick_tie_tol) or (abs(h - best_h) <= thick_tie_tol and cnt > best_cnt):
                    best_Rr, best_h, best_cnt = Rr, h, cnt

        # ---- VTK（列ベクトル系）へ適用： p' = R p + t,  R = best_Rr.T,  t = c - R c ----
        R = best_Rr.T
        t = c - R @ c

        # ---- 回転行列からEuler角を抽出してスライダーに反映 ----
        def matrix_to_euler_zyx(R):
            """回転行列からEuler角（ZYX順）を抽出"""
            sy = np.hypot(R[0,0], R[1,0])
            singular = sy < 1e-8
            if not singular:
                z = np.degrees(np.arctan2(R[1,0], R[0,0]))         # yaw
                y = np.degrees(np.arctan2(-R[2,0], sy))            # pitch
                x = np.degrees(np.arctan2(R[2,1], R[2,2]))         # roll
            else:
                # gimbal lock: z は意味を持ちにくいので0に、xで帳尻
                z = 0.0
                y = np.degrees(np.arctan2(-R[2,0], sy))
                x = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            # -180〜180に正規化
            def _wrap(a): 
                return (a + 180) % 360 - 180
            return _wrap(x), _wrap(y), _wrap(z)

        # 回転行列からEuler角を取得
        rot_x, rot_y, rot_z = matrix_to_euler_zyx(R)

        # Find Initial Plane 内：回転行列 R と平行移動 t を作った後
        M = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                M.SetElement(i, j, float(R[i, j]))
        M.SetElement(0, 3, float(t[0]))
        M.SetElement(1, 3, float(t[1]))
        M.SetElement(2, 3, float(t[2]))
        M.SetElement(3, 0, 0.0); M.SetElement(3, 1, 0.0); M.SetElement(3, 2, 0.0); M.SetElement(3, 3, 1.0)

        # 計算した回転行列M（ワールド基準）は base_transform にだけ入れる
        self.base_transform.Identity()
        self.base_transform.SetMatrix(M)

        # ローカル操作は一旦ゼロから（＝整列後もローカル軸で自由に回せる）
        self.local_transform.Identity()
        self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}  # スライダ絶対値→差分適用用

        # スライダUIも 0° にリセット（任意）
        if hasattr(self, 'rotation_widgets'):
            for ax in ('X', 'Y', 'Z'):
                self.rotation_widgets[ax]['spin'].blockSignals(True)
                self.rotation_widgets[ax]['slider'].blockSignals(True)
                self.rotation_widgets[ax]['spin'].setValue(0)
                self.rotation_widgets[ax]['slider'].setValue(0)
                self.rotation_widgets[ax]['spin'].blockSignals(False)
                self.rotation_widgets[ax]['slider'].blockSignals(False)

        # 後方互換性のため molecule_transform も更新
        self.molecule_transform.Identity()
        self.molecule_transform.SetMatrix(M)

        self.update_actor_transform()
        
        if hasattr(self, 'set_standard_view'):
            self.set_standard_view('yz')
        if hasattr(self, 'trigger_interactive_simulation'):
            self.trigger_interactive_simulation()

    def on_mouse_press(self, event):
        """直接的なマウスプレスイベントハンドラー"""
        
        if event.button() == Qt.LeftButton:
            # キーの状態をチェック
            modifiers = event.modifiers()
            ctrl_pressed = bool(modifiers & Qt.ControlModifier)
            shift_pressed = bool(modifiers & Qt.ShiftModifier)
            
            if ctrl_pressed and not shift_pressed:
                self.actor_rotating = True
                self.drag_start_pos = event.pos()
                event.accept()
                return
            elif shift_pressed and not ctrl_pressed:
                self.panning = True
                self.pan_start_pos = event.pos()
                event.accept()
                return
        
        # 通常のマウスイベントをVTKウィジェットの元のハンドラーに渡す
        if hasattr(self, 'original_mouse_press'):
            self.original_mouse_press(event)

    def on_mouse_move(self, event):
        """直接的なマウスムーブイベントハンドラー"""
        if self.actor_rotating:
            if hasattr(self, 'drag_start_pos'):
                dx = event.pos().x() - self.drag_start_pos.x()
                dy = event.pos().y() - self.drag_start_pos.y()
                
                # 視点に応じた回転軸マッピング
                self.update_rotation_from_drag_view_dependent(dx, dy)
                
                self.drag_start_pos = event.pos()
            event.accept()
            return
        elif self.panning:
            # パニング処理は後で実装
            event.accept()
            return
        
        # 通常のマウスイベントをVTKウィジェットの元のハンドラーに渡す
        if hasattr(self, 'original_mouse_move'):
            self.original_mouse_move(event)

    def on_mouse_release(self, event):
        """直接的なマウスリリースイベントハンドラー"""
        
        if event.button() == Qt.LeftButton:
            if self.actor_rotating:
                self.actor_rotating = False
                
                # ★★★ 追加：ドラッグ終了時の高解像度シミュレーション ★★★
                if self.interactive_update_check.isChecked():
                    self.schedule_high_res_simulation()
                
                event.accept()
                return
            elif self.panning:
                self.panning = False
                event.accept()
                return
        
        # 通常のマウスイベントをVTKウィジェットの元のハンドラーに渡す
        if hasattr(self, 'original_mouse_release'):
            self.original_mouse_release(event)

    def reset_structure_rotation(self):
        """分子の回転をリセット（PDB/MRC読み込み時の状態に戻す）"""
        if not hasattr(self, 'rotation_widgets'):
            return
        
        # 回転ウィジェットを0にリセット
        for axis in ['X', 'Y', 'Z']:
            self.rotation_widgets[axis]['spin'].blockSignals(True)
            self.rotation_widgets[axis]['slider'].blockSignals(True)
            self.rotation_widgets[axis]['spin'].setValue(0.0)
            self.rotation_widgets[axis]['slider'].setValue(0)
            self.rotation_widgets[axis]['spin'].blockSignals(False)
            self.rotation_widgets[axis]['slider'].blockSignals(False)
        
        # 回転変換をリセット
        self.base_transform.Identity()
        self.local_transform.Identity()
        self.combined_transform.Identity()
        self.molecule_transform.Identity()
        
        # prev_rotをリセット
        self.prev_rot = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # アクターの変換を更新
        self.update_actor_transform()

    def get_current_view_orientation(self):
        """現在のカメラの向きから視点方向を判定"""
        if not hasattr(self, 'renderer') or not self.renderer:
            return 'free'
        
        camera = self.renderer.GetActiveCamera()
        pos = camera.GetPosition()
        focal = camera.GetFocalPoint()
        
        # カメラから焦点への方向ベクトル
        view_dir = [focal[i] - pos[i] for i in range(3)]
        # 正規化
        length = (sum(d*d for d in view_dir)) ** 0.5
        if length < 1e-10:
            return 'free'
        view_dir = [d/length for d in view_dir]
        
        # 各軸方向との内積で判定（閾値0.8）
        if abs(view_dir[2]) > 0.8:  # Z方向
            return 'xy'  # XY面を見ている
        elif abs(view_dir[0]) > 0.8:  # X方向
            return 'yz'  # YZ面を見ている
        elif abs(view_dir[1]) > 0.8:  # Y方向
            return 'zx'  # ZX面を見ている
        else:
            return 'free'  # 斜め視点

    def update_rotation_from_drag_view_dependent(self, dx, dy):
        """視点に応じて回転軸をマッピング"""
        view_orientation = self.get_current_view_orientation()
        
        # 回転感度
        sensitivity = 0.5
        
        if view_orientation == 'xy':
            # XY面視点（Z軸方向から見る）
            angle_x = dy * sensitivity   # 垂直ドラッグ → X軸回転
            angle_y = dx * sensitivity   # 水平ドラッグ → Y軸回転
            angle_z = 0
        elif view_orientation == 'yz':
            # YZ面視点（X軸方向から見る）
            angle_x = 0
            angle_y = dy * sensitivity   # 垂直ドラッグ → Y軸回転
            angle_z = dx * sensitivity   # 水平ドラッグ → Z軸回転
        elif view_orientation == 'zx':
            # ZX面視点（Y軸方向から見る）
            angle_x = dy * sensitivity   # 垂直ドラッグ → X軸回転
            angle_y = 0
            angle_z = dx * sensitivity   # 水平ドラッグ → Z軸回転
        else:
            # 自由視点：通常の回転
            angle_x = dy * sensitivity   # 垂直ドラッグ → X軸回転
            angle_y = dx * sensitivity   # 水平ドラッグ → Y軸回転
            angle_z = 0
        
        self.update_rotation_from_drag(
            angle_x_delta=angle_x,
            angle_y_delta=angle_y,
            angle_z_delta=angle_z
        )

    def update_rotation_from_drag(self, angle_x_delta=0, angle_y_delta=0, angle_z_delta=0):
        """マウスドラッグに応じてPDB/MRC構造の回転角度を更新する（改良版）"""
        if not hasattr(self, 'rotation_widgets'):
            return

        current_rot_x = self.rotation_widgets['X']['spin'].value()
        current_rot_y = self.rotation_widgets['Y']['spin'].value()
        current_rot_z = self.rotation_widgets['Z']['spin'].value()

        # ドラッグによる移動量を加算
        raw_x = current_rot_x + angle_x_delta
        raw_y = current_rot_y + angle_y_delta
        raw_z = current_rot_z + angle_z_delta
        
        # 角度を-180から+180の範囲に正規化する
        new_rot_x = (raw_x + 180) % 360 - 180
        new_rot_y = (raw_y + 180) % 360 - 180
        new_rot_z = (raw_z + 180) % 360 - 180

        # スライダーの値変更を一時的に無効化してから設定
        for axis in ['X', 'Y', 'Z']:
            self.rotation_widgets[axis]['spin'].blockSignals(True)
            self.rotation_widgets[axis]['slider'].blockSignals(True)
        
        self.rotation_widgets['X']['spin'].setValue(new_rot_x)
        self.rotation_widgets['Y']['spin'].setValue(new_rot_y)
        self.rotation_widgets['Z']['spin'].setValue(new_rot_z)
        
        # スライダーも同期
        self.rotation_widgets['X']['slider'].setValue(int(new_rot_x * 10))
        self.rotation_widgets['Y']['slider'].setValue(int(new_rot_y * 10))
        self.rotation_widgets['Z']['slider'].setValue(int(new_rot_z * 10))
        
        # シグナルを再有効化
        for axis in ['X', 'Y', 'Z']:
            self.rotation_widgets[axis]['spin'].blockSignals(False)
            self.rotation_widgets[axis]['slider'].blockSignals(False)
        
        # スライダー値を変更した後、回転を適用
        self.apply_structure_rotation()
        
        # ★★★ 修正: ドラッグ中の制御されたリアルタイム更新 ★★★
        if self.interactive_update_check.isChecked():
            # ドラッグ中は制御付きで更新（頻度制限あり）
            if hasattr(self, 'actor_rotating') and self.actor_rotating:
                self.run_simulation_immediate_controlled()

    def update_afm_display(self):
        """AFM画像表示の更新（チェックボックスに基づく）"""
        # 現在チェックされている数を確認
        checked_count = sum([
            self.afm_x_check.isChecked(),
            self.afm_y_check.isChecked(),
            self.afm_z_check.isChecked()
        ])
        
        # 最低1つはチェックされている必要がある
        if checked_count == 0:
            # どのチェックボックスが最後に変更されたかを確認して元に戻す
            sender = self.sender()
            if sender:
                sender.blockSignals(True)  # 再帰呼び出しを防ぐ
                sender.setChecked(True)
                sender.blockSignals(False)
                
            QMessageBox.warning(self, "Warning", 
                            "At least one AFM view must be selected!")
            return
        
        # 各パネルの表示/非表示を設定
        self.afm_x_frame.setVisible(self.afm_x_check.isChecked())
        self.afm_y_frame.setVisible(self.afm_y_check.isChecked())
        self.afm_z_frame.setVisible(self.afm_z_check.isChecked())
        
        # デバッグ情報
        visible_views = []
        if self.afm_x_check.isChecked():
            visible_views.append("X")
        if self.afm_y_check.isChecked():
            visible_views.append("Y")
        if self.afm_z_check.isChecked():
            visible_views.append("Z")
        
        #print(f"AFM views visible: {', '.join(visible_views)}")

    def create_afm_image_panel(self, title):
        """個別のAFM像表示パネル作成（表示制御対応）"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        frame.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        
        # タイトルラベル
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 9px;
                color: #555;
                padding: 2px;
                background-color: #e8e8e8;
                border-radius: 2px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setMaximumHeight(18)
        layout.addWidget(title_label)
        
        # プレースホルダー
        placeholder_text = "AFM Image\n(Not Simulated)"
        # YZとZXビューの場合、テキストを上書き
        if title in ["YZ View", "ZX View"]:
            placeholder_text = f"{title}\n(This scan type is\nnot calculated)"

        placeholder = QLabel(placeholder_text)
        placeholder.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 8px;
                background-color: white;
                border: 1px dashed #ccc;
                border-radius: 2px;
            }
        """)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setMinimumHeight(80)
        #placeholder.setMaximumHeight(150)
        layout.addWidget(placeholder)
        
        return frame
    
    def reset_camera(self):
        """カメラのリセット（デフォルトでYZ平面視点）"""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        
        # デフォルトでYZ平面視点に設定
        camera.SetViewUp(0, 0, 1)  # Z軸が上方向
        camera.SetPosition(15, 0, 0)  # X軸の正方向から見る
        camera.SetFocalPoint(0, 0, 0)  # 原点を焦点に
        
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
    
    def setup_lighting(self):
        """ライティング設定"""
        # メインライト
        light1 = vtk.vtkLight()
        light1.SetPosition(10, 10, 10)
        light1.SetIntensity(0.8)
        light1.SetColor(1.0, 1.0, 1.0)
        self.renderer.AddLight(light1)
        
        # フィルライト
        light2 = vtk.vtkLight()
        light2.SetPosition(-5, -5, 5)
        light2.SetIntensity(0.4)
        light2.SetColor(0.9, 0.9, 1.0)
        self.renderer.AddLight(light2)
        
    def add_axes(self):
        """大きな座標軸を画面左下隅に追加"""
        # 座標軸アクターを作成
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(4.5, 4.5, 4.5)  # ★★★ 長さは大きく維持 ★★★
        axes.SetCylinderRadius(0.05)        # ★★★ 線を細く（0.24→0.05） ★★★
        axes.SetShaftType(0)                # シンプルな軸
        axes.SetAxisLabels(1)               # ラベル表示
        
        # ★★★ ラベルのフォントサイズは大きく維持 ★★★
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(54)  # 大きく維持
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(54)  # 大きく維持
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(54)  # 大きく維持
        
        # 軸ラベルの色設定（より鮮明に）
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 0.1, 0.1)  # より鮮明な赤
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.1, 1, 0.1)  # より鮮明な緑
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.1, 0.1, 1)  # より鮮明な青
        
        # オリエンテーションマーカーウィジェットを作成
        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(self.interactor)
        
        # ★★★ 位置とサイズを設定（左下隅、より小さく配置） ★★★
        self.orientation_widget.SetViewport(0.0, 0.0, 0.3, 0.3)  # 左下の30%×30%（60%→30%）
        self.orientation_widget.SetEnabled(True)
        self.orientation_widget.InteractiveOff()  # 相互作用を無効（邪魔にならない）
    
    def debug_molecule_info(self):
        """分子情報のデバッグ表示"""
        if self.atoms_data is None:
            print("No molecule data available")
            QMessageBox.warning(self, "Debug", "No molecule data loaded!")
            return
        
        atom_x = self.atoms_data['x']
        atom_y = self.atoms_data['y'] 
        atom_z = self.atoms_data['z']
        
        #print("\n" + "="*50)
        #print("MOLECULE DEBUG INFO")
        #print("="*50)
        
        # 基本統計
        #print(f"Total atoms: {len(atom_x)}")
        #print(f"X range: {np.min(atom_x):.2f} to {np.max(atom_x):.2f}nm (size: {np.max(atom_x)-np.min(atom_x):.2f}nm)")
        #print(f"Y range: {np.min(atom_y):.2f} to {np.max(atom_y):.2f}nm (size: {np.max(atom_y)-np.min(atom_y):.2f}nm)")
        #print(f"Z range: {np.min(atom_z):.2f} to {np.max(atom_z):.2f}nm (size: {np.max(atom_z)-np.min(atom_z):.2f}nm)")
        
        # 中心位置
        center_x = np.mean(atom_x)
        center_y = np.mean(atom_y)
        center_z = np.mean(atom_z)
        print(f"Center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})nm")
        
        # 推奨設定
        mol_size = max(np.max(atom_x)-np.min(atom_x), np.max(atom_y)-np.min(atom_y))
        recommended_scan = mol_size * 1.5
        recommended_tip_z = np.max(atom_z) + 2.0
        
        #print(f"\nRECOMMENDED SETTINGS:")
        #print(f"Scan size: {recommended_scan:.1f}nm (current: {self.scan_size_spin.value():.1f}nm)")
       # print(f"Tip Z position: {recommended_tip_z:.1f}nm (current: {self.afm_params['tip_z']:.1f}nm)")
        
        # 探針位置チェック
        tip_x = self.afm_params['tip_x']
        tip_y = self.afm_params['tip_y']
        tip_z = self.afm_params['tip_z']
        
        #print(f"\nTIP POSITION CHECK:")
        #print(f"Current tip: ({tip_x:.2f}, {tip_y:.2f}, {tip_z:.2f})nm")
        
        # 分子との重なりチェック
        if (np.min(atom_x) <= tip_x <= np.max(atom_x) and 
            np.min(atom_y) <= tip_y <= np.max(atom_y)):
            #print("✓ Tip is positioned over the molecule")
            pass
        else:
            #print("⚠ WARNING: Tip is NOT over the molecule!")
            pass
        
        if tip_z > np.max(atom_z) + 1.0:
            #print("✓ Tip Z position is safe")
            pass
        else:
            #print("⚠ WARNING: Tip Z position may be too low!")
            pass
        
        #print("="*50)
        
        # UIに推奨設定を表示
        msg = f"""Debug Information:
        
Molecule size: {mol_size:.1f}nm
Current scan size: {self.scan_size_spin.value():.1f}nm
Recommended scan size: {recommended_scan:.1f}nm

Current tip Z: {tip_z:.1f}nm  
Recommended tip Z: {recommended_tip_z:.1f}nm

Tip over molecule: {np.min(atom_x) <= tip_x <= np.max(atom_x) and np.min(atom_y) <= tip_y <= np.max(atom_y)}

Check console for detailed information."""
        
        QMessageBox.information(self, "Debug Info", msg)

    def quick_collision_test(self):
        """特定の点での衝突テスト"""
        if self.atoms_data is None:
            print("No molecule data available")
            return
        
        atom_x = self.atoms_data['x']
        atom_y = self.atoms_data['y']
        atom_z = self.atoms_data['z']
        atom_elem = self.atoms_data['element']
        atom_radii = np.array([self.vdw_radii.get(e, self.vdw_radii['other']) for e in atom_elem])
        
        # 分子の中心での衝突テスト
        center_x = np.mean(atom_x)
        center_y = np.mean(atom_y)
        test_z = np.max(atom_z) + 3.0
        
        #print(f"\nQUICK COLLISION TEST:")
        #print(f"Test point: ({center_x:.2f}, {center_y:.2f}, {test_z:.2f})nm")
        
        try:
            height = self.find_collision_height(center_x, center_y, atom_x, atom_y, atom_z, atom_radii)
            #print(f"Calculated collision height: {height:.3f}nm")
            
            # 妥当性チェック
            if height > np.max(atom_z):
                #print("✓ Result seems reasonable (above molecule)")
                result_msg = f"✓ Collision test PASSED\n\nTest point: ({center_x:.2f}, {center_y:.2f})\nCalculated height: {height:.3f}nm\nMolecule top: {np.max(atom_z):.3f}nm"
            else:
                #print("⚠ WARNING: Result may be too low")
                result_msg = f"⚠ Collision test FAILED\n\nTest point: ({center_x:.2f}, {center_y:.2f})\nCalculated height: {height:.3f}nm\nMolecule top: {np.max(atom_z):.3f}nm\n\nHeight is too low!"
                
            QMessageBox.information(self, "Collision Test", result_msg)
            
        except Exception as e:
            print(f"ERROR in collision calculation: {e}")
            QMessageBox.critical(self, "Error", f"Collision test failed:\n{str(e)}")

    def apply_recommended_settings(self):
        """推奨設定を自動適用"""
        if self.atoms_data is None:
            QMessageBox.warning(self, "Warning", "No molecule data loaded!")
            return
        
        atom_x = self.atoms_data['x']
        atom_y = self.atoms_data['y'] 
        atom_z = self.atoms_data['z']
        
        # 推奨設定を計算
        mol_size = max(np.max(atom_x)-np.min(atom_x), np.max(atom_y)-np.min(atom_y))
        recommended_scan = mol_size * 1.5
        recommended_tip_z = np.max(atom_z) + 2.0
        
        # UIに設定を適用
        self.scan_size_spin.setValue(recommended_scan)
        
        # 探針Z位置を設定（スライダー値に変換）
        slider_value = int(recommended_tip_z * 5.0)  # z = value / 5.0 の逆算
        slider_value = max(self.tip_z_slider.minimum(), 
                          min(self.tip_z_slider.maximum(), slider_value))
        self.tip_z_slider.setValue(slider_value)
        
        # 探針を分子中心に移動
        center_x = np.mean(atom_x)
        center_y = np.mean(atom_y)
        
        self.tip_x_slider.setValue(int(center_x * 5.0))  # x = value / 5.0 の逆算
        self.tip_y_slider.setValue(int(center_y * 5.0))  # y = value / 5.0 の逆算
        
        print(f"Applied recommended settings:")
        print(f"- Scan size: {recommended_scan:.1f}nm")
        print(f"- Tip position: ({center_x:.1f}, {center_y:.1f}, {recommended_tip_z:.1f})nm")
        
        QMessageBox.information(self, "Settings Applied", 
                               f"Recommended settings applied:\n\n"
                               f"Scan size: {recommended_scan:.1f}nm\n"
                               f"Tip position: ({center_x:.1f}, {center_y:.1f}, {recommended_tip_z:.1f})nm")

        
    def import_file(self):
        """統合ファイルインポート（PDB/CIF/MRC）"""
        initial_dir = self.last_import_dir if hasattr(self, 'last_import_dir') and self.last_import_dir else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Structure File", initial_dir,
            "Structure Files (*.pdb *.cif *.mmcif *.mrc);;PDB files (*.pdb);;mmCIF files (*.cif *.mmcif);;MRC Files (*.mrc);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog)
        
        if not file_path:
            return
        
        self.last_import_dir = os.path.dirname(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdb':
            self._import_pdb_internal(file_path)
        elif ext in ['.cif', '.mmcif']:
            self._import_cif_internal(file_path)
        elif ext == '.mrc':
            self._import_mrc_internal(file_path)
        else:
            QMessageBox.warning(self, "Unsupported Format", 
                              f"File format '{ext}' is not supported.\nSupported formats: .pdb, .cif, .mmcif, .mrc")

    def eventFilter(self, obj, event):
        """Filter events for vtk_widget: accept drag & drop of PDB/CIF/MRC files on PDB Structure area."""
        target = hasattr(self, 'vtk_widget') and obj is self.vtk_widget
        if target:
            if event.type() == QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    urls = event.mimeData().urls()
                    allowed = ('.pdb', '.cif', '.mmcif', '.mrc')
                    if urls and urls[0].isLocalFile():
                        path = urls[0].toLocalFile()
                        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in allowed:
                            event.acceptProposedAction()
                            return True
            elif event.type() == QEvent.Drop:
                urls = event.mimeData().urls()
                if urls and urls[0].isLocalFile():
                    path = urls[0].toLocalFile()
                    if os.path.isfile(path):
                        self.last_import_dir = os.path.dirname(path)
                        ext = os.path.splitext(path)[1].lower()
                        if ext == '.pdb':
                            self._import_pdb_internal(path)
                        elif ext in ['.cif', '.mmcif']:
                            self._import_cif_internal(path)
                        elif ext == '.mrc':
                            self._import_mrc_internal(path)
                        event.acceptProposedAction()
                        return True
        return super().eventFilter(obj, event)

    def _import_pdb_internal(self, file_path):
        """PDBファイルの読み込み（内部メソッド）"""
            
        try:
            # MRCデータをクリア（PDBファイルimport時）
            self.clear_mrc_data()
            # CIF情報をリセット（PDB読み込み時）
            if hasattr(self, 'cif_name'):
                self.cif_name = None
                self.cif_id = ""

            if hasattr(self, 'rotation_widgets'):
                self.reset_structure_rotation()

            # プログレスバー表示
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()
            
            self.read_pdb_file(file_path)
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            self.update_statistics()
            self.progress_bar.setValue(70)
            QApplication.processEvents()
            
            self.display_molecule()
            self.progress_bar.setValue(90)
            QApplication.processEvents()
            
            self.create_tip()
             # ★★★ ここから追加 ★★★
            # PDB構造の最高点から2nm上に探針の初期位置を設定
            if self.atoms_data is not None:
                z_max = self.atoms_data['z'].max()
                initial_tip_z = z_max + 2.0
                
                # Z位置スライダーの物理値と表示値を更新
                # スライダー値は物理値の5倍 (z = value / 5.0 の逆算)
                slider_value = int(initial_tip_z * 5.0)
                
                # スライダーが設定可能な範囲内に収まるように調整
                min_val, max_val = self.tip_z_slider.minimum(), self.tip_z_slider.maximum()
                slider_value = max(min_val, min(max_val, slider_value))
                
                # スライダーの値を設定 (これによりupdate_tip_positionが自動で呼ばれる)
                self.tip_z_slider.setValue(slider_value)
            # ★★★ ここまで追加 ★★★

            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            # ファイル名表示
            self.pdb_name = os.path.basename(file_path) 
            self.pdb_id = os.path.splitext(self.pdb_name)[0]
            self.file_label.setText(f"File Name: {self.pdb_name} (PDB)")
            
            # シミュレーションボタンを有効化
            self.simulate_btn.setEnabled(True)
            
            # 回転ウィジェットも有効化
            if hasattr(self, 'rotation_widgets'):
                for axis in ['X', 'Y', 'Z']:
                    self.rotation_widgets[axis]['spin'].setEnabled(True)
                    self.rotation_widgets[axis]['slider'].setEnabled(True)
            
            # プログレスバー非表示
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
            
            QMessageBox.information(self, "Success", 
                                f"Successfully loaded {self.pdb_name}\n"
                                f"Atoms: {len(self.atoms_data['x'])}")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", 
                            f"Failed to load PDB file:\n{str(e)}")

    def _import_cif_internal(self, file_path):
        """mmCIFファイルの読み込み（内部メソッド）"""
        try:
            # MRCデータをクリア（CIFファイルimport時）
            self.clear_mrc_data()

            # PDB情報をリセット（CIF読み込み時）
            if hasattr(self, 'pdb_name'):
                self.pdb_name = None
                self.pdb_id = ""

            if hasattr(self, 'rotation_widgets'):
                self.reset_structure_rotation()

            # プログレスバー表示
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            QApplication.processEvents()

            self.read_cif_file(file_path)
            self.progress_bar.setValue(50)
            QApplication.processEvents()

            self.update_statistics()
            self.progress_bar.setValue(70)
            QApplication.processEvents()

            self.display_molecule()
            self.progress_bar.setValue(90)
            QApplication.processEvents()

            self.create_tip()
            # 分子の最高点から2nm上に探針の初期位置を設定
            if self.atoms_data is not None:
                z_max = self.atoms_data['z'].max()
                initial_tip_z = z_max + 2.0
                slider_value = int(initial_tip_z * 5.0)  # z = value / 5.0 の逆算
                min_val, max_val = self.tip_z_slider.minimum(), self.tip_z_slider.maximum()
                slider_value = max(min_val, min(max_val, slider_value))
                self.tip_z_slider.setValue(slider_value)

            self.progress_bar.setValue(100)
            QApplication.processEvents()

            # ファイル名表示
            self.cif_name = os.path.basename(file_path)
            self.cif_id = os.path.splitext(self.cif_name)[0]
            self.file_label.setText(f"File Name: {self.cif_name} (CIF)")

            # シミュレーションボタンを有効化
            self.simulate_btn.setEnabled(True)

            # 回転ウィジェットも有効化
            if hasattr(self, 'rotation_widgets'):
                for axis in ['X', 'Y', 'Z']:
                    self.rotation_widgets[axis]['spin'].setEnabled(True)
                    self.rotation_widgets[axis]['slider'].setEnabled(True)

            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))

            QMessageBox.information(
                self, "Success",
                f"Successfully loaded {self.cif_name}\n"
                f"Atoms: {len(self.atoms_data['x'])}"
            )

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load mmCIF file:\n{str(e)}")
            
    def read_pdb_file(self, file_path):
        """PDBファイルの解析"""
        atoms = []
        helices = []  # (chain_id, start_residue, end_residue)
        sheets = []   # (chain_id, start_residue, end_residue)
        
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                # HELIXレコードの解析
                if line.startswith('HELIX'):
                    try:
                        chain_id = line[19:20].strip()
                        start_residue = int(line[21:25].strip())
                        end_residue = int(line[33:37].strip())
                        helices.append((chain_id, start_residue, end_residue))
                    except (ValueError, IndexError):
                        pass
                
                # SHEETレコードの解析
                elif line.startswith('SHEET'):
                    try:
                        chain_id = line[21:22].strip()
                        start_residue = int(line[22:26].strip())
                        end_residue = int(line[33:37].strip())
                        sheets.append((chain_id, start_residue, end_residue))
                    except (ValueError, IndexError):
                        pass
                
                # ATOM/HETATMレコードの解析
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        # PDBフォーマット解析
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain_id = line[21:22].strip()
                        residue_id = int(line[22:26].strip())
                        
                        x = float(line[30:38]) / 10.0  # Åからnmに変換
                        y = float(line[38:46]) / 10.0
                        z = float(line[46:54]) / 10.0
                        
                        # 元素名取得
                        element = line[76:78].strip()
                        if not element:
                            element = atom_name[0]
                        
                        # B-factor取得
                        try:
                            b_factor = float(line[60:66])
                        except:
                            b_factor = 20.0
                        
                        atoms.append({
                            'name': atom_name,
                            'x': x, 'y': y, 'z': z,
                            'element': element,
                            'residue_name': residue_name,
                            'chain_id': chain_id,
                            'residue_id': residue_id,
                            'b_factor': b_factor
                        })
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        
        if not atoms:
            raise ValueError("No valid atoms found in PDB file")
        
        # numpy配列に変換
        self.atoms_data = {
            'x': np.array([atom['x'] for atom in atoms]),
            'y': np.array([atom['y'] for atom in atoms]),
            'z': np.array([atom['z'] for atom in atoms]),
            'element': np.array([atom['element'] for atom in atoms]),
            'atom_name': np.array([atom['name'] for atom in atoms]),
            'residue_name': np.array([atom['residue_name'] for atom in atoms]),
            'chain_id': np.array([atom['chain_id'] for atom in atoms]),
            'residue_id': np.array([atom['residue_id'] for atom in atoms]),
            'b_factor': np.array([atom['b_factor'] for atom in atoms])
        }
        
        # 二次構造情報を辞書に格納
        self.secondary_structure = {}
        
        # ヘリックスを登録
        for chain_id, start_res, end_res in helices:
            for res_id in range(start_res, end_res + 1):
                key = (chain_id, res_id)
                self.secondary_structure[key] = 'H'
        
        # シートを登録
        for chain_id, start_res, end_res in sheets:
            for res_id in range(start_res, end_res + 1):
                key = (chain_id, res_id)
                self.secondary_structure[key] = 'E'
        
        # 座標を中心化
        self.center_coordinates()
        
        print(f"Loaded {len(atoms)} atoms")
        if helices:
            print(f"Found {len(helices)} helix regions (from PDB)")
        if sheets:
            print(f"Found {len(sheets)} sheet regions (from PDB)")
        
        # HELIX/SHEETレコードがない、または少ない場合は幾何学的検出を実行
        if len(helices) + len(sheets) < 3:
            print("Running geometric secondary structure detection...")
            self.detect_secondary_structure_geometric()

    def read_cif_file(self, file_path):
        """mmCIFファイルの解析（_atom_site loop_ から原子座標を抽出）"""
        tags = []
        atoms = []

        def _as_int(value, default_int):
            try:
                return int(value)
            except Exception:
                return default_int

        def _as_float(value):
            if value in ('.', '?', None):
                return None
            try:
                return float(value)
            except Exception:
                return None

        def _norm_str(value):
            if value in ('.', '?', None):
                return ""
            return str(value)

        def _infer_element(atom_name):
            # mmCIF/PDB互換: 先頭の英字を拾い、2文字元素も最低限対応
            if not atom_name:
                return "C"
            s = str(atom_name).strip()
            if not s:
                return "C"
            # 例: "CA" はカルシウムではなくCαであることが多いが、
            # type_symbolが無いケースのフォールバックなので単純推定に留まる
            s2 = "".join([ch for ch in s if ch.isalpha()])
            if not s2:
                return s[0].upper()
            if len(s2) >= 2 and s2[0].isalpha() and s2[1].islower():
                return (s2[0] + s2[1]).capitalize()
            return s2[0].upper()

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "loop_":
                # collect tags
                tags = []
                j = i + 1
                while j < len(lines):
                    t = lines[j].strip()
                    if not t:
                        j += 1
                        continue
                    if t.startswith('_'):
                        tags.append(t.split()[0])
                        j += 1
                        continue
                    break

                is_atom_site_loop = bool(tags) and all(tag.startswith("_atom_site.") for tag in tags)
                if not is_atom_site_loop:
                    i = j
                    continue

                tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}

                # indices with fallbacks
                def _idx(*candidates):
                    for c in candidates:
                        if c in tag_to_idx:
                            return tag_to_idx[c]
                    return None

                ix = _idx("_atom_site.Cartn_x")
                iy = _idx("_atom_site.Cartn_y")
                iz = _idx("_atom_site.Cartn_z")
                if ix is None or iy is None or iz is None:
                    raise ValueError("mmCIF _atom_site loop_ does not contain Cartn_x/Cartn_y/Cartn_z")

                itype = _idx("_atom_site.type_symbol")
                i_atom = _idx("_atom_site.label_atom_id", "_atom_site.auth_atom_id")
                i_comp = _idx("_atom_site.label_comp_id", "_atom_site.auth_comp_id")
                i_asym = _idx("_atom_site.label_asym_id", "_atom_site.auth_asym_id")
                i_seq = _idx("_atom_site.label_seq_id", "_atom_site.auth_seq_id")
                i_b = _idx("_atom_site.B_iso_or_equiv")

                # parse data rows
                k = j
                seq_fallback = 1
                while k < len(lines):
                    raw = lines[k].rstrip("\n")
                    s = raw.strip()

                    if not s:
                        k += 1
                        continue
                    if s.startswith('#'):
                        k += 1
                        break
                    if s == "loop_" or s.startswith("data_") or s.startswith("_"):
                        break
                    if s.startswith(';'):
                        # Multiline values are not expected for _atom_site. Skip block defensively.
                        k += 1
                        while k < len(lines) and not lines[k].startswith(';'):
                            k += 1
                        k += 1
                        continue

                    tokens = shlex.split(s)
                    # mmCIF may wrap a row across lines; accumulate tokens until enough
                    while len(tokens) < len(tags) and (k + 1) < len(lines):
                        nxt = lines[k + 1].strip()
                        if not nxt or nxt.startswith('#') or nxt == "loop_" or nxt.startswith("data_") or nxt.startswith("_"):
                            break
                        k += 1
                        tokens.extend(shlex.split(lines[k].strip()))

                    if len(tokens) < len(tags):
                        k += 1
                        continue

                    x = _as_float(tokens[ix])
                    y = _as_float(tokens[iy])
                    z = _as_float(tokens[iz])
                    if x is None or y is None or z is None:
                        k += 1
                        continue

                    # Å -> nm
                    x /= 10.0
                    y /= 10.0
                    z /= 10.0

                    atom_name = _norm_str(tokens[i_atom]) if i_atom is not None else ""
                    residue_name = _norm_str(tokens[i_comp]) if i_comp is not None else ""
                    chain_id = _norm_str(tokens[i_asym]) if i_asym is not None else ""

                    residue_id = None
                    if i_seq is not None:
                        residue_id = _as_int(tokens[i_seq], seq_fallback)
                    else:
                        residue_id = seq_fallback
                    seq_fallback += 1

                    element = _norm_str(tokens[itype]) if itype is not None else ""
                    if not element:
                        element = _infer_element(atom_name)

                    b_factor = 20.0
                    if i_b is not None:
                        bf = _as_float(tokens[i_b])
                        if bf is not None:
                            b_factor = float(bf)

                    atoms.append({
                        'name': atom_name,
                        'x': x, 'y': y, 'z': z,
                        'element': element,
                        'residue_name': residue_name,
                        'chain_id': chain_id,
                        'residue_id': int(residue_id) if residue_id is not None else 0,
                        'b_factor': float(b_factor)
                    })

                    k += 1

                i = k
                continue

            i += 1

        if not atoms:
            raise ValueError("No valid atoms found in mmCIF file (_atom_site)")

        self.atoms_data = {
            'x': np.array([atom['x'] for atom in atoms]),
            'y': np.array([atom['y'] for atom in atoms]),
            'z': np.array([atom['z'] for atom in atoms]),
            'element': np.array([atom['element'] for atom in atoms]),
            'atom_name': np.array([atom['name'] for atom in atoms]),
            'residue_name': np.array([atom['residue_name'] for atom in atoms]),
            'chain_id': np.array([atom['chain_id'] for atom in atoms]),
            'residue_id': np.array([atom['residue_id'] for atom in atoms]),
            'b_factor': np.array([atom['b_factor'] for atom in atoms])
        }

        self.center_coordinates()
        print(f"Loaded {len(atoms)} atoms from mmCIF")
    
    def detect_secondary_structure_geometric(self):
        """
        幾何学的な二次構造検出（PyMOL風）
        CA原子間の距離パターンからヘリックスとシートを推定
        """
        if self.atoms_data is None:
            return
        
        # Cα原子のみを抽出
        mask = (self.atoms_data['atom_name'] == 'CA')
        if not np.any(mask):
            return
        
        ca_x = self.atoms_data['x'][mask]
        ca_y = self.atoms_data['y'][mask]
        ca_z = self.atoms_data['z'][mask]
        chain_ids = self.atoms_data['chain_id'][mask]
        residue_ids = self.atoms_data['residue_id'][mask]
        
        unique_chains = np.unique(chain_ids)
        
        helix_count = 0
        sheet_count = 0
        
        for chain in unique_chains:
            # チェーン内のCα原子を抽出
            chain_mask = (chain_ids == chain)
            c_x = ca_x[chain_mask]
            c_y = ca_y[chain_mask]
            c_z = ca_z[chain_mask]
            c_res_id = residue_ids[chain_mask]
            
            # 残基ID順にソート
            sort_idx = np.argsort(c_res_id)
            c_x = c_x[sort_idx]
            c_y = c_y[sort_idx]
            c_z = c_z[sort_idx]
            c_res_id_sorted = c_res_id[sort_idx]
            
            if len(c_x) < 5:
                continue
            
            # 各残基について二次構造を判定
            for i in range(len(c_x)):
                res_id = c_res_id_sorted[i]
                key = (chain, res_id)
                
                # 既に二次構造が割り当てられている場合はスキップ
                if key in self.secondary_structure:
                    continue
                
                # ヘリックス検出: i, i+3, i+4 の距離パターン
                is_helix = False
                if i + 4 < len(c_x):
                    # 隣接CA間の距離
                    d1 = np.sqrt((c_x[i+1] - c_x[i])**2 + 
                                 (c_y[i+1] - c_y[i])**2 + 
                                 (c_z[i+1] - c_z[i])**2)
                    
                    # i と i+3 の距離（ヘリックスの特徴）
                    d3 = np.sqrt((c_x[i+3] - c_x[i])**2 + 
                                 (c_y[i+3] - c_y[i])**2 + 
                                 (c_z[i+3] - c_z[i])**2)
                    
                    # i と i+4 の距離（ヘリックスの特徴）
                    d4 = np.sqrt((c_x[i+4] - c_x[i])**2 + 
                                 (c_y[i+4] - c_y[i])**2 + 
                                 (c_z[i+4] - c_z[i])**2)
                    
                    # ヘリックスの判定基準
                    # - 隣接CA距離: 約3.6-4.0Å (0.36-0.40 nm)
                    # - i→i+3距離: 約5.0-5.5Å (0.50-0.55 nm)
                    # - i→i+4距離: 約5.8-6.5Å (0.58-0.65 nm)
                    if (0.34 < d1 < 0.42 and 
                        0.48 < d3 < 0.58 and 
                        0.56 < d4 < 0.68):
                        is_helix = True
                
                if is_helix:
                    self.secondary_structure[key] = 'H'
                    helix_count += 1
                else:
                    # シート検出（簡易版）: 連続で平らな構造
                    is_sheet = False
                    if i + 2 < len(c_x) and i > 0:
                        # 隣接CA間の距離が約3.3-3.5Å (シートの特徴)
                        d1 = np.sqrt((c_x[i+1] - c_x[i])**2 + 
                                     (c_y[i+1] - c_y[i])**2 + 
                                     (c_z[i+1] - c_z[i])**2)
                        
                        d_prev = np.sqrt((c_x[i] - c_x[i-1])**2 + 
                                        (c_y[i] - c_y[i-1])**2 + 
                                        (c_z[i] - c_z[i-1])**2)
                        
                        # シートの判定基準
                        # - CA間距離: 約3.2-3.5Å (0.32-0.35 nm)
                        # - 比較的伸びた構造
                        if 0.31 < d1 < 0.36 and 0.31 < d_prev < 0.36:
                            # 前後の点を含めて判定
                            if i + 2 < len(c_x):
                                # 3つの連続したCAがほぼ直線状かチェック
                                vec1 = np.array([c_x[i] - c_x[i-1], 
                                                c_y[i] - c_y[i-1], 
                                                c_z[i] - c_z[i-1]])
                                vec2 = np.array([c_x[i+1] - c_x[i], 
                                                c_y[i+1] - c_y[i], 
                                                c_z[i+1] - c_z[i]])
                                
                                # ベクトルを正規化
                                vec1_norm = np.linalg.norm(vec1)
                                vec2_norm = np.linalg.norm(vec2)
                                
                                if vec1_norm > 1e-6 and vec2_norm > 1e-6:
                                    vec1 = vec1 / vec1_norm
                                    vec2 = vec2 / vec2_norm
                                    
                                    # 内積が大きい（ほぼ同じ方向）ならシート
                                    dot_product = np.dot(vec1, vec2)
                                    if dot_product > 0.85:  # 約30度以内
                                        is_sheet = True
                    
                    if is_sheet:
                        self.secondary_structure[key] = 'E'
                        sheet_count += 1
                    else:
                        # デフォルトはコイル
                        self.secondary_structure[key] = 'C'
        
        print(f"Geometric detection: {helix_count} helix, {sheet_count} sheet residues")
        
    def center_coordinates(self):
        """座標を中心に移動"""
        for coord in ['x', 'y', 'z']:
            center = (self.atoms_data[coord].max() + self.atoms_data[coord].min()) / 2
            self.atoms_data[coord] -= center
            
    def update_statistics(self):
        """原子統計の更新"""
        if self.atoms_data is None:
            return
            
        total = len(self.atoms_data['x'])
        self.stats_labels['Total'].setText(f"Total: {total}")
        
        for atom_type in ['C', 'O', 'N', 'H']:
            count = np.sum(self.atoms_data['element'] == atom_type)
            self.stats_labels[atom_type].setText(f"{atom_type}: {count}")
        
        # その他の原子
        known_types = ['C', 'O', 'N', 'H']
        other_count = np.sum(~np.isin(self.atoms_data['element'], known_types))
        self.stats_labels['Other'].setText(f"Other: {other_count}")
        
    def get_filtered_atoms(self):
        """表示フィルターに基づいて原子を選択"""
        if self.atoms_data is None:
            return None, None, None, None, None, None, None
            
        atom_filter = self.atom_combo.currentText()
        
        if atom_filter == "All Atoms":
            mask = np.ones(len(self.atoms_data['x']), dtype=bool)
        elif atom_filter == "Heavy Atoms":
            mask = self.atoms_data['element'] != 'H'
        elif atom_filter == "Backbone":
            mask = np.isin(self.atoms_data['atom_name'], ['N', 'CA', 'C', 'O'])
        elif atom_filter in ['C', 'N', 'O']:
            mask = self.atoms_data['element'] == atom_filter
        else:
            mask = np.ones(len(self.atoms_data['x']), dtype=bool)
        
        if not np.any(mask):
            return None, None, None, None, None, None, None
            
        return (self.atoms_data['x'][mask], 
                self.atoms_data['y'][mask],
                self.atoms_data['z'][mask],
                self.atoms_data['element'][mask],
                self.atoms_data['chain_id'][mask],
                self.atoms_data['b_factor'][mask],
                mask)
        
    def get_atom_color(self, element, chain_id, b_factor):
        """原子の色を取得"""
        color_scheme = self.color_combo.currentText()
        
        if color_scheme == "By Element":
            base_color = self.element_colors.get(element, self.element_colors['other'])
        elif color_scheme == "By Chain":
            chain_hash = hash(chain_id) % len(self.chain_colors)
            base_color = self.chain_colors[chain_hash]
        elif color_scheme == "Single Color":
            # Single Colorの場合は選択された色を直接返す
            base_color = self.current_single_color  
            #print(f"Using single color / 単色を使用: {base_color}")  # デバッグ用
        elif color_scheme == "By B-Factor":
            # B-factorを0-1に正規化（0-50の範囲を想定）
            norm_b = np.clip(b_factor / 50.0, 0, 1)
            # 青→緑→黄→赤のグラデーション
            if norm_b < 0.33:
                t = norm_b * 3
                base_color = (0, 0.5 + 0.5*t, 1 - t)
            elif norm_b < 0.66:
                t = (norm_b - 0.33) * 3
                base_color = (t, 1, 0)
            else:
                t = (norm_b - 0.66) * 3
                base_color = (1, 1 - 0.5*t, 0)
        else:
            base_color = self.element_colors.get(element, self.element_colors['other'])
        
        # 明るさファクターを適用
        adjusted_color = tuple(min(1.0, c * self.brightness_factor) for c in base_color)
        return adjusted_color
        
    def display_molecule(self):
        """分子の表示"""
        # 既存のアクターを削除
        if self.sample_actor:
            self.renderer.RemoveActor(self.sample_actor)
        if self.bonds_actor:
            self.renderer.RemoveActor(self.bonds_actor)
            
        x, y, z, elements, chain_ids, b_factors, mask = self.get_filtered_atoms()
        if x is None:
            return
            
        style = self.style_combo.currentText()
        size_factor = self.size_slider.value() / 100.0
        opacity = self.opacity_slider.value() / 100.0
        quality = self.quality_combo.currentText()
        
        # 品質設定
        if quality == "Fast":
            resolution = 8
            max_atoms = 5000
        elif quality == "Good":
            resolution = 12
            max_atoms = 10000
        else:  # High
            resolution = 16
            max_atoms = 20000
        
        # サンプリング処理
        sampled_indices = None
        if len(x) > max_atoms:
            sampled_indices = np.random.choice(len(x), max_atoms, replace=False)
            x, y, z = x[sampled_indices], y[sampled_indices], z[sampled_indices]
            elements = elements[sampled_indices]
            chain_ids = chain_ids[sampled_indices]
            b_factors = b_factors[sampled_indices]
        
        # スタイルに応じた表示
        if style == "Ball & Stick":
            self.sample_actor = self.create_ball_stick_display(
                x, y, z, elements, chain_ids, b_factors, size_factor, resolution)
        elif style == "Stick Only":
            self.sample_actor = self.create_stick_display(
                x, y, z, elements, chain_ids, b_factors, size_factor, resolution)
        elif style == "Spheres":
            self.sample_actor = self.create_sphere_display(
                x, y, z, elements, chain_ids, b_factors, size_factor, resolution)
        elif style == "Points":
            self.sample_actor = self.create_point_display(
                x, y, z, elements, chain_ids, b_factors, size_factor)
        elif style == "Wireframe":
            self.sample_actor = self.create_wireframe_display(x, y, z)
        elif style == "Simple Cartoon":
            # Cartoon表示は元のデータを使用（サンプリングなし）
            self.sample_actor = self.create_simple_cartoon_display_safe()
        elif style == "Ribbon":
            # Ribbon表示はCα原子を使用（サンプリングなし）
            self.sample_actor = self.create_ribbon_display(size_factor)
        
        # 透明度設定
        if self.sample_actor and hasattr(self.sample_actor, 'GetProperty'):
            self.sample_actor.GetProperty().SetOpacity(opacity)
            
        # アクターを追加
        if self.sample_actor:
            self.renderer.AddActor(self.sample_actor)
            
        # 結合表示（Stick系の場合）
        if style in ["Ball & Stick", "Stick Only"]:
            self.create_bonds_display(x, y, z, elements, chain_ids, b_factors, 
                                    size_factor * 0.3, resolution)
        
        # 現在の回転設定をアクターに適用
        self.apply_structure_rotation()
        
        # 初期回転角度を保存（Reset Allで使用）
        if hasattr(self, 'rotation_widgets'):
            self.initial_rotation_angles = {
                'X': self.rotation_widgets['X']['spin'].value(),
                'Y': self.rotation_widgets['Y']['spin'].value(),
                'Z': self.rotation_widgets['Z']['spin'].value()
            }
            
        self.vtk_widget.GetRenderWindow().Render()
        
    def create_sphere_display(self, x, y, z, elements, chain_ids, b_factors, size_factor, resolution):
        """球体表示"""
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # 全ての点と色を設定（Single Colorでも個別に設定）
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
            
            # 色を取得（Single Colorでも get_atom_color を通す）
            color = self.get_atom_color(elements[i], chain_ids[i], b_factors[i])
            colors.InsertNextTuple3(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(colors)
        polydata.Modified()
        
        # 球体ソース
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.15 * size_factor)
        sphere.SetPhiResolution(resolution)
        sphere.SetThetaResolution(resolution)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetColorModeToColorByScalar()  # 重要：色をスカラーで制御
        glyph.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOn()  # 常にOn
        mapper.SetScalarModeToUsePointData()  # ポイントデータを使用
        mapper.Update()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(20)
        
        return actor
        
    def create_point_display(self, x, y, z, elements, chain_ids, b_factors, size_factor):
        """点表示"""
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
            color = self.get_atom_color(elements[i], chain_ids[i], b_factors[i])
            colors.InsertNextTuple3(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(colors)
        polydata.Modified()  # 追加
        
        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()  # 追加
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())
        mapper.ScalarVisibilityOn()  # 追加
        mapper.Update()  # 追加
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(max(1, size_factor * 5))
        
        return actor
        
    def create_wireframe_display(self, x, y, z):
        """ワイヤーフレーム表示"""
        points = vtk.vtkPoints()
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Delaunay 3D
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(polydata)
        
        # 表面抽出
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputConnection(delaunay.GetOutputPort())
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surface_filter.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.GetProperty().SetLineWidth(1.5)
        
        return actor
        
    def create_simple_cartoon_display_safe(self):
        """
        簡易的なCartoon表示を作成（スプライン補間などを行わない安全な実装）
        """
        # Cα原子のみを抽出
        mask = (self.atoms_data['atom_name'] == 'CA')
        if not np.any(mask):
            return None
            
        ca_x = self.atoms_data['x'][mask]
        ca_y = self.atoms_data['y'][mask]
        ca_z = self.atoms_data['z'][mask]
        chain_ids = self.atoms_data['chain_id'][mask]
        residue_ids = self.atoms_data['residue_id'][mask]
        
        # チェーンごとにソート
        unique_chains = np.unique(chain_ids)
        
        append_poly = vtk.vtkAppendPolyData()
        
        for chain in unique_chains:
            chain_mask = (chain_ids == chain)
            c_x = ca_x[chain_mask]
            c_y = ca_y[chain_mask]
            c_z = ca_z[chain_mask]
            c_res_id = residue_ids[chain_mask]
            
            # 残基ID順にソート
            sort_idx = np.argsort(c_res_id)
            c_x = c_x[sort_idx]
            c_y = c_y[sort_idx]
            c_z = c_z[sort_idx]
            
            if len(c_x) < 2:
                continue
                
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            
            lines.InsertNextCell(len(c_x))
            
            for i in range(len(c_x)):
                points.InsertNextPoint(c_x[i], c_y[i], c_z[i])
                lines.InsertCellPoint(i)
                
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetLines(lines)
            
            # チューブフィルターで太さを持たせる
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(poly)
            tube.SetRadius(0.15 * (self.size_slider.value() / 100.0)) # 太さは固定
            tube.SetNumberOfSides(8)
            tube.CappingOn()
            tube.Update()
            
            append_poly.AddInputData(tube.GetOutput())
            
        append_poly.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append_poly.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # 色は一律（またはチェーンごとに変えるなど改善の余地あり）
        # 这里ではAtomごとの色を取得して適用
        # 簡易実装では単色などにするが、既存動作に合わせる
        
        return actor

    def create_ribbon_display(self, size_factor):
        """
        PyMOL風のリボン表示を作成（二次構造対応版）
        ヘリックス、シート、ループで異なる形状を生成
        """
        # Cα原子のみを抽出
        if self.atoms_data is None:
            return None

        mask = (self.atoms_data['atom_name'] == 'CA')
        if not np.any(mask):
            mask = (self.atoms_data['atom_name'] == 'P')
            if not np.any(mask):
                return None
            
        ca_x = self.atoms_data['x'][mask]
        ca_y = self.atoms_data['y'][mask]
        ca_z = self.atoms_data['z'][mask]
        elements = self.atoms_data['element'][mask]
        chain_ids = self.atoms_data['chain_id'][mask]
        residue_ids = self.atoms_data['residue_id'][mask]
        b_factors = self.atoms_data['b_factor'][mask]
        
        unique_chains = np.unique(chain_ids)
        
        append_poly = vtk.vtkAppendPolyData()
        
        for chain in unique_chains:
            # チェーン内の原子を抽出
            chain_mask = (chain_ids == chain)
            c_x = ca_x[chain_mask]
            c_y = ca_y[chain_mask]
            c_z = ca_z[chain_mask]
            c_res_id = residue_ids[chain_mask]
            c_elements = elements[chain_mask]
            c_b_factors = b_factors[chain_mask]
            
            # 残基ID順にソート
            sort_idx = np.argsort(c_res_id)
            c_x = c_x[sort_idx]
            c_y = c_y[sort_idx]
            c_z = c_z[sort_idx]
            c_res_id_sorted = c_res_id[sort_idx]
            c_elements = c_elements[sort_idx]
            c_b_factors = c_b_factors[sort_idx]
            
            if len(c_x) < 4:  # スプライン補間のため最低4点必要
                continue
            
            # 各残基の二次構造タイプを取得
            ss_types = []
            for res_id in c_res_id_sorted:
                key = (chain, res_id)
                ss_type = self.secondary_structure.get(key, 'C')  # デフォルトはコイル
                ss_types.append(ss_type)
            
            # Catmull-Romスプラインで滑らかに補間
            num_points = len(c_x)
            subdivisions = 10  # 各セグメント間の分割数
            
            interpolated_points = []
            interpolated_colors = []
            interpolated_ss = []  # 二次構造タイプも補間点に関連付け
            
            for i in range(num_points - 1):
                # Catmull-Romスプライン用の4点を取得
                p0_idx = max(0, i - 1)
                p1_idx = i
                p2_idx = i + 1
                p3_idx = min(num_points - 1, i + 2)
                
                p0 = np.array([c_x[p0_idx], c_y[p0_idx], c_z[p0_idx]])
                p1 = np.array([c_x[p1_idx], c_y[p1_idx], c_z[p1_idx]])
                p2 = np.array([c_x[p2_idx], c_y[p2_idx], c_z[p2_idx]])
                p3 = np.array([c_x[p3_idx], c_y[p3_idx], c_z[p3_idx]])
                
                # 色（p1とp2の間を補間）
                color1 = self.get_atom_color(c_elements[p1_idx], chain, c_b_factors[p1_idx])
                color2 = self.get_atom_color(c_elements[p2_idx], chain, c_b_factors[p2_idx])
                
                # 二次構造タイプ（p1を使用）
                ss_type = ss_types[p1_idx]
                
                for j in range(subdivisions):
                    t = j / subdivisions
                    
                    # Catmull-Romスプライン補間
                    point = 0.5 * (
                        (2 * p1) +
                        (-p0 + p2) * t +
                        (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
                        (-p0 + 3*p1 - 3*p2 + p3) * t**3
                    )
                    
                    interpolated_points.append(point)
                    
                    # 色を線形補間
                    interp_color = tuple(
                        color1[k] * (1 - t) + color2[k] * t
                        for k in range(3)
                    )
                    interpolated_colors.append(interp_color)
                    interpolated_ss.append(ss_type)
            
            # 最後の点を追加
            interpolated_points.append(np.array([c_x[-1], c_y[-1], c_z[-1]]))
            color_last = self.get_atom_color(c_elements[-1], chain, c_b_factors[-1])
            interpolated_colors.append(color_last)
            interpolated_ss.append(ss_types[-1])
            
            # NumPy配列に変換
            interpolated_points = np.array(interpolated_points)
            n_interp = len(interpolated_points)
            
            if n_interp < 3:
                continue
            
            # リボンメッシュを構築（二次構造に応じて幅を変える）
            points = vtk.vtkPoints()
            triangles = vtk.vtkCellArray()
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            
            # 各補間点でリボンの左右の点を生成
            for i in range(n_interp):
                # 二次構造に応じた幅を決定
                ss_type = interpolated_ss[i]
                if ss_type == 'H':  # ヘリックス
                    ribbon_width = 0.6 * size_factor
                elif ss_type == 'E':  # シート
                    ribbon_width = 0.8 * size_factor
                else:  # コイル
                    ribbon_width = 0.2 * size_factor
                
                # 接線ベクトル（進行方向）
                if i == 0:
                    tangent = interpolated_points[1] - interpolated_points[0]
                elif i == n_interp - 1:
                    tangent = interpolated_points[i] - interpolated_points[i-1]
                else:
                    tangent = interpolated_points[i+1] - interpolated_points[i-1]
                
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-6:
                    tangent = tangent / tangent_norm
                else:
                    tangent = np.array([1.0, 0.0, 0.0])
                
                # リボンの幅方向を計算
                up = np.array([0.0, 0.0, 1.0])
                
                # 接線がZ軸と平行な場合は別の軸を使用
                if abs(np.dot(tangent, up)) > 0.99:
                    up = np.array([1.0, 0.0, 0.0])
                
                # リボンの幅方向
                width_dir = np.cross(tangent, up)
                width_norm = np.linalg.norm(width_dir)
                if width_norm > 1e-6:
                    width_dir = width_dir / width_norm
                else:
                    width_dir = np.array([0.0, 1.0, 0.0])
                
                # 前の点との一貫性を保つため、必要に応じて方向を反転
                if i > 0:
                    if np.dot(width_dir, prev_width_dir) < 0:
                        width_dir = -width_dir
                
                prev_width_dir = width_dir.copy()
                
                # リボンの左右の点
                half_width = ribbon_width / 2.0
                center = interpolated_points[i]
                left_point = center - width_dir * half_width
                right_point = center + width_dir * half_width
                
                # 点を追加
                points.InsertNextPoint(left_point[0], left_point[1], left_point[2])
                points.InsertNextPoint(right_point[0], right_point[1], right_point[2])
                
                # 色を設定
                color = interpolated_colors[i]
                color_tuple = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                colors.InsertNextTuple3(*color_tuple)
                colors.InsertNextTuple3(*color_tuple)
                
                # 三角形メッシュを構築
                if i > 0:
                    prev_left = (i - 1) * 2
                    prev_right = (i - 1) * 2 + 1
                    curr_left = i * 2
                    curr_right = i * 2 + 1
                    
                    # 三角形1
                    triangle1 = vtk.vtkTriangle()
                    triangle1.GetPointIds().SetId(0, prev_left)
                    triangle1.GetPointIds().SetId(1, curr_left)
                    triangle1.GetPointIds().SetId(2, prev_right)
                    triangles.InsertNextCell(triangle1)
                    
                    # 三角形2
                    triangle2 = vtk.vtkTriangle()
                    triangle2.GetPointIds().SetId(0, curr_left)
                    triangle2.GetPointIds().SetId(1, curr_right)
                    triangle2.GetPointIds().SetId(2, prev_right)
                    triangles.InsertNextCell(triangle2)
            
            # PolyDataを作成
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetPolys(triangles)
            poly.GetPointData().SetScalars(colors)
            
            append_poly.AddInputData(poly)
            
        append_poly.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append_poly.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointData()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # マテリアル設定
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(40)
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.7)
        
        return actor
    
    def create_simple_ca_points(self, ca_x, ca_y, ca_z, ca_chains):
        """CAアトムの点表示（フォールバック用）"""
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        for i in range(len(ca_x)):
            points.InsertNextPoint(ca_x[i], ca_y[i], ca_z[i])
            
            # チェーン色
            chain_hash = hash(ca_chains[i]) % len(self.chain_colors)
            color = self.chain_colors[chain_hash]
            colors.InsertNextTuple3(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(colors)
        
        # 球体で表示
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.3)
        sphere.SetPhiResolution(12)
        sphere.SetThetaResolution(12)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(20)
        
        return actor
        
    def create_ball_stick_display(self, x, y, z, elements, chain_ids, b_factors, 
                                size_factor, resolution):
        """ボール&スティック表示"""
        return self.create_sphere_display(x, y, z, elements, chain_ids, b_factors, 
                                        size_factor * 0.7, resolution)
        
    def create_stick_display(self, x, y, z, elements, chain_ids, b_factors, 
                           size_factor, resolution):
        """スティック表示"""
        return self.create_sphere_display(x, y, z, elements, chain_ids, b_factors, 
                                        size_factor * 0.3, resolution)
        
    def create_bonds_display(self, x, y, z, elements, chain_ids, b_factors, 
                           bond_radius, resolution):
        """結合の表示"""
        if self.bonds_actor:
            self.renderer.RemoveActor(self.bonds_actor)
            
        # 簡単な距離ベース結合判定
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # 全ての点を追加
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
        
        # 近接原子間で結合を作成（効率化のため制限）
        max_bonds = 10000
        bond_count = 0
        
        for i in range(len(x)):
            if bond_count >= max_bonds:
                break
                
            for j in range(i + 1, min(i + 20, len(x))):  # 近くの原子のみチェック
                if bond_count >= max_bonds:
                    break
                    
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)
                
                # 結合距離判定
                if dist < 0.18:  # 1.8 Å
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i)
                    line.GetPointIds().SetId(1, j)
                    lines.InsertNextCell(line)
                    
                    # 結合の色（平均色）
                    color1 = self.get_atom_color(elements[i], chain_ids[i], b_factors[i])
                    color2 = self.get_atom_color(elements[j], chain_ids[j], b_factors[j])
                    avg_color = [(color1[k] + color2[k])/2 for k in range(3)]
                    colors.InsertNextTuple3(
                        int(avg_color[0]*255), 
                        int(avg_color[1]*255), 
                        int(avg_color[2]*255)
                    )
                    
                    bond_count += 1
        
        if bond_count > 0:
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            polydata.GetCellData().SetScalars(colors)
            
            # チューブフィルター
            tube_filter = vtk.vtkTubeFilter()
            tube_filter.SetInputData(polydata)
            tube_filter.SetRadius(bond_radius)
            tube_filter.SetNumberOfSides(max(4, resolution // 2))
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tube_filter.GetOutputPort())
            
            self.bonds_actor = vtk.vtkActor()
            self.bonds_actor.SetMapper(mapper)
            self.bonds_actor.GetProperty().SetSpecular(0.3)
            self.bonds_actor.GetProperty().SetSpecularPower(20)
            
            self.renderer.AddActor(self.bonds_actor)
        
    def create_tip(self):
        """AFM探針の作成（実際のパラメーターに基づく）"""
        if self.tip_actor:
            self.renderer.RemoveActor(self.tip_actor)
            
        tip_shape = self.tip_shape_combo.currentText().lower()
        radius = self.tip_radius_spin.value()
        angle = self.tip_angle_spin.value()
        # ★★★ 追加: 新しいUIからminitipの半径を取得 ★★★
        minitip_radius = self.minitip_radius_spin.value()
        
        #print(f"Creating tip: {tip_shape}, radius={radius}nm, angle={angle}°, minitip_radius={minitip_radius}nm")
        
        if tip_shape == "cone":
            self.tip_actor = self.create_cone_tip(radius, angle)
        elif tip_shape == "sphere":
            # ★★★ 変更点: minitip_radiusを引数として渡す ★★★
            self.tip_actor = self.create_sphere_tip(radius, angle, minitip_radius)
        else:  # paraboloid
            self.tip_actor = self.create_paraboloid_tip(radius)
        
        if self.tip_actor:
            self.update_tip_position()
            self.renderer.AddActor(self.tip_actor)
            self.vtk_widget.GetRenderWindow().Render()

    # +++ この関数で既存のcreate_cone_tipを置き換えてください +++
    def create_cone_tip(self, tip_radius, half_angle):
        """
        Igor Proの数式に基づいて高さマップを生成し、そこから探針形状を作成します。
        この方法は非常に安定しており、環境に依存する問題を回避します。
        先端は-Z方向を向き、長さも調整されています。
        """
        if self.tip_actor:
            self.renderer.RemoveActor(self.tip_actor)

        # --- Igor Proのロジックに基づいたパラメータ計算 ---
        if half_angle < 1.0: half_angle = 1.0
        if half_angle >= 89.0: half_angle = 89.0
        half_angle_rad = np.radians(float(half_angle))
        
        # 形状が球から円錐に切り替わる臨界半径
        r_crit = tip_radius * np.cos(half_angle_rad)
        # 円錐部分が滑らかに接続するためのZオフセット
        z_offset = (tip_radius / np.sin(half_angle_rad)) - tip_radius

        # --- 点群グリッドの生成 ---
        resolution = 101  # グリッドの解像度 (奇数にすると中心点ができます)
        
        # ★★★ 変更点1: コーンを長くするため、高さを大きく設定 ★★★
        max_height = tip_radius * 50.0  # 以前は 25.0 でした
        
        max_radius = (max_height + z_offset) * np.tan(half_angle_rad)
        
        points = vtk.vtkPoints()
        
        # グリッド上の各点の3D座標を計算
        for i in range(resolution):
            for j in range(resolution):
                # グリッド座標(i, j)を物理座標(x, y)に変換
                x = (j - (resolution - 1) / 2.0) * (2 * max_radius / (resolution - 1))
                y = (i - (resolution - 1) / 2.0) * (2 * max_radius / (resolution - 1))
                
                # 中心からの距離rを計算
                r = np.sqrt(x**2 + y**2)
                
                # Igorの数式を使ってz座標(高さ)を計算
                if r <= r_crit:
                    # 球状部分の計算式
                    sqrt_arg = tip_radius**2 - r**2
                    z = tip_radius - np.sqrt(max(0, sqrt_arg))
                else:
                    # 円錐状部分の計算式
                    z = (r / np.tan(half_angle_rad)) - z_offset
                
                # ★★★ 変更点2: 先端が-Z方向を向くように、Z座標を反転 ★★★
                points.InsertNextPoint(x, y, z)

        # --- 点群からサーフェスメッシュを生成 ---
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Delaunay2Dアルゴリズムで点群から三角形メッシュを生成
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.Update() # 念のためUpdateを呼び出します

        # --- ★★★ 変更点3: Z反転を直接行ったため、後処理が不要に ★★★
        # 以前のtransformやnormalsの処理は不要になり、コードがシンプルになりました。
        
        # --- アクターの作成 ---
        mapper = vtk.vtkPolyDataMapper()
        # Delaunayの結果を直接マッパーに接続します
        mapper.SetInputConnection(delaunay.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # 材質を設定
        actor.GetProperty().SetColor(1.0, 0.84, 0.0)  # ゴールドのRGB値
        actor.GetProperty().SetSpecular(0.9)         # 高い鏡面反射で金属感を強調
        actor.GetProperty().SetSpecularPower(100)    # 光沢を強くする
        actor.GetProperty().SetDiffuse(0.6)          # 拡散反射
        actor.GetProperty().SetAmbient(0.3)    

        #print(f"SUCCESS: Flipped and elongated cone tip created: radius={tip_radius:.1f}nm, angle={half_angle}°")
        
        return actor
    
    # +++ この関数で既存のcreate_sphere_tipを置き換えてください +++
    # +++ この関数で既存のcreate_sphere_tipを置き換えてください +++
    def create_sphere_tip(self, tip_radius, half_angle, minitip_radius):
        """
        Cone形状の上に、指定された半径(minitip_radius)の球を接着した形状を生成します。
        """
        if self.tip_actor:
            self.renderer.RemoveActor(self.tip_actor)

        # --- 部品1: 先端に突き出る球を作成 ---
        # ★★★ 変更点: 引数で渡されたminitip_radiusを使用 ★★★
        sphere_radius = minitip_radius
        
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(sphere_radius)
        sphere_source.SetPhiResolution(50)
        sphere_source.SetThetaResolution(50)
        
        sphere_transform = vtk.vtkTransform()
        sphere_transform.Translate(0, 0, sphere_radius)
        
        sphere_filter = vtk.vtkTransformPolyDataFilter()
        sphere_filter.SetInputConnection(sphere_source.GetOutputPort())
        sphere_filter.SetTransform(sphere_transform)
        sphere_filter.Update()

        # --- 部品2: Cone部分を作成し、球の上部に移動 ---
        if half_angle < 1.0: half_angle = 1.0
        if half_angle >= 89.0: half_angle = 89.0
        half_angle_rad = np.radians(float(half_angle))
        
        r_crit_cone = tip_radius * np.cos(half_angle_rad)
        z_offset_cone = (tip_radius / np.sin(half_angle_rad)) - tip_radius
        
        resolution = 101
        max_height_cone = tip_radius * 50.0
        max_radius_cone = (max_height_cone + z_offset_cone) * np.tan(half_angle_rad)
        
        cone_points = vtk.vtkPoints()
        for i in range(resolution):
            for j in range(resolution):
                x = (j - (resolution - 1) / 2.0) * (2 * max_radius_cone / (resolution - 1))
                y = (i - (resolution - 1) / 2.0) * (2 * max_radius_cone / (resolution - 1))
                r = np.sqrt(x**2 + y**2)
                
                if r <= r_crit_cone:
                    z = tip_radius - np.sqrt(max(0, tip_radius**2 - r**2))
                else:
                    z = (r / np.tan(half_angle_rad)) - z_offset_cone
                cone_points.InsertNextPoint(x, y, z)
        
        cone_polydata = vtk.vtkPolyData()
        cone_polydata.SetPoints(cone_points)
        cone_delaunay = vtk.vtkDelaunay2D()
        cone_delaunay.SetInputData(cone_polydata)
        
        cone_transform = vtk.vtkTransform()
        cone_transform.Translate(0, 0, 2 * sphere_radius)
        
        cone_filter = vtk.vtkTransformPolyDataFilter()
        cone_filter.SetInputConnection(cone_delaunay.GetOutputPort())
        cone_filter.SetTransform(cone_transform)
        cone_filter.Update()

        # --- 2つの部品を結合 ---
        append_filter = vtk.vtkAppendPolyData()
        append_filter.AddInputData(sphere_filter.GetOutput())
        append_filter.AddInputData(cone_filter.GetOutput())
        append_filter.Update()

        # --- アクターを作成 ---
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(1.0, 0.84, 0.0)
        prop.SetSpecular(0.9)
        prop.SetSpecularPower(100)
        prop.SetDiffuse(0.6)
        prop.SetAmbient(0.3)
        prop.SetOpacity(0.95)

        print(f"SUCCESS: Composite 'Sphere' created. Cone R={tip_radius:.1f}, Minitip R={minitip_radius:.1f}")
        return actor
    
    def create_paraboloid_tip(self, tip_radius):
        """
        Igor Proの数式に基づき、先端が下(-Z)を向く放物面探針を生成します。
        """
        if self.tip_actor:
            self.renderer.RemoveActor(self.tip_actor)

        # --- グリッドと点群の準備 ---
        resolution = 101
        display_height = 20.0 
        max_radius = np.sqrt(2 * tip_radius * display_height)
        points = vtk.vtkPoints()
        
        for i in range(resolution):
            for j in range(resolution):
                x = (j - (resolution - 1) / 2.0) * (2 * max_radius / (resolution - 1))
                y = (i - (resolution - 1) / 2.0) * (2 * max_radius / (resolution - 1))
                
                # Igorの数式 z = (x^2 + y^2) / (2 * R)
                z = (x**2 + y**2) / (2 * tip_radius)
                
                # ★★★ 修正点: 先端が下(-Z)を向くようにZ座標を反転 ★★★
                points.InsertNextPoint(x, y, z)

        # --- メッシュ生成とアクター作成 ---
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(delaunay.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # 材質をゴールドに設定
        prop = actor.GetProperty()
        prop.SetColor(1.0, 0.84, 0.0)
        prop.SetSpecular(0.9)
        prop.SetSpecularPower(100)
        prop.SetDiffuse(0.6)
        prop.SetAmbient(0.3)
        prop.SetOpacity(0.95)

        print(f"SUCCESS: Paraboloid tip created (pointing down): R={tip_radius:.1f}nm")
        return actor
    
        
    def update_display(self):
        """表示の更新"""
        if self.atoms_data is not None:
            current_scheme = self.color_combo.currentText()
            #print(f"Updating display with color scheme: {current_scheme}")
            ##if current_scheme == "Single Color":
                #print(f"Single color value: {self.current_single_color}")
            
            self.display_molecule()
            
            # レンダリングを強制実行
            self.vtk_widget.GetRenderWindow().Render()
    
    def update_tip_info(self):
        """探針情報の更新"""
        shape = self.tip_shape_combo.currentText()
        radius = self.tip_radius_spin.value()
        angle = self.tip_angle_spin.value()
        
        if shape == "Cone":
            height = radius * 3
            base_radius = radius + height * np.tan(np.radians(angle))
            info = f"Tip: {radius}nm radius\nCone: {height:.1f}nm height\nBase: {base_radius:.1f}nm radius"
        elif shape == "Sphere":
            info = f"Sphere: {radius}nm radius"
        else:
            info = f"Paraboloid: {radius}nm radius\nAngle: {angle}°"
        
        self.tip_info_label.setText(info)
            
    def update_tip(self):
        """探針の更新（パラメーター変更時）"""
        #print("Tip parameters changed - updating display...")
        self.create_tip()
        self.update_tip_info()  # 追加
        
        # AFMパラメーターも更新
        self.afm_params.update({
            'tip_radius': self.tip_radius_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'tip_angle': self.tip_angle_spin.value(),
        })

        # スレッドの安全性をチェックしてからシミュレーションを実行
        self.trigger_interactive_simulation()
    
    def trigger_interactive_simulation(self):
        """インタラクティブモードがONの場合にシミュレーションを実行する汎用トリガー"""
        # スライダー操作中は実行しない
        if hasattr(self, 'tip_slider_pressed') and self.tip_slider_pressed:
            return
        
        # 既にシミュレーションが実行中の場合は実行しない
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            return
            
        if self.interactive_update_check.isChecked():
            # ★★★ 通常のInteractive Updateでも軽量版を使用 ★★★
            self.run_simulation_silent()
        
    def on_tip_slider_pressed(self):
        """Tip positionスライダーが押された時の処理"""
        # スライダー操作中のフラグを設定
        self.tip_slider_pressed = True
        
    def on_tip_slider_released(self):
        """Tip positionスライダーが離された時の処理"""
        # スライダー操作完了のフラグを設定
        self.tip_slider_pressed = False
        
        # スライダー操作完了後にシミュレーションを実行（Interactive UpdateがONの場合のみ）
        if self.interactive_update_check.isChecked():
            # 遅延実行でシミュレーションをトリガー
            QTimer.singleShot(100, self.trigger_interactive_simulation)
    
    # Scan Size関連のイベントハンドラー
    def scan_size_value_changed(self, value):
        """Scan Size値変更時の処理（マウス/ボタン操作時は即時更新）"""
        if not self.scan_size_keyboard_input:
            # デバウンス処理：既存のタイマーを停止して新しいタイマーを設定
            if self.scan_size_debounce_timer:
                self.scan_size_debounce_timer.stop()
            self.scan_size_debounce_timer = QTimer(self)
            self.scan_size_debounce_timer.setSingleShot(True)
            self.scan_size_debounce_timer.timeout.connect(self.trigger_interactive_simulation)
            self.scan_size_debounce_timer.start(100)  # 100ms後に実行
    
    def scan_size_editing_finished(self):
        """Scan Size編集完了時の処理（キー入力時はリターンで更新）"""
        self.scan_size_keyboard_input = False
        self.trigger_interactive_simulation()
    
    # Tip Radius関連のイベントハンドラー
    def tip_radius_value_changed(self, value):
        """Tip Radius値変更時の処理（マウス/ボタン操作時は即時更新）"""
        if not self.tip_radius_keyboard_input:
            # デバウンス処理
            if self.tip_radius_debounce_timer:
                self.tip_radius_debounce_timer.stop()
            self.tip_radius_debounce_timer = QTimer(self)
            self.tip_radius_debounce_timer.setSingleShot(True)
            self.tip_radius_debounce_timer.timeout.connect(self.update_tip)
            self.tip_radius_debounce_timer.start(100)
    
    def tip_radius_editing_finished(self):
        """Tip Radius編集完了時の処理（キー入力時はリターンで更新）"""
        self.tip_radius_keyboard_input = False
        self.update_tip()
    
    # Minitip Radius関連のイベントハンドラー
    def minitip_radius_value_changed(self, value):
        """Minitip Radius値変更時の処理（マウス/ボタン操作時は即時更新）"""
        if not self.minitip_radius_keyboard_input:
            # デバウンス処理
            if self.minitip_radius_debounce_timer:
                self.minitip_radius_debounce_timer.stop()
            self.minitip_radius_debounce_timer = QTimer(self)
            self.minitip_radius_debounce_timer.setSingleShot(True)
            self.minitip_radius_debounce_timer.timeout.connect(self.update_tip)
            self.minitip_radius_debounce_timer.start(100)
    
    def minitip_radius_editing_finished(self):
        """Minitip Radius編集完了時の処理（キー入力時はリターンで更新）"""
        self.minitip_radius_keyboard_input = False
        self.update_tip()
    
    # Tip Angle関連のイベントハンドラー
    def tip_angle_value_changed(self, value):
        """Tip Angle値変更時の処理（マウス/ボタン操作時は即時更新）"""
        if not self.tip_angle_keyboard_input:
            # デバウンス処理
            if self.tip_angle_debounce_timer:
                self.tip_angle_debounce_timer.stop()
            self.tip_angle_debounce_timer = QTimer(self)
            self.tip_angle_debounce_timer.setSingleShot(True)
            self.tip_angle_debounce_timer.timeout.connect(self.update_tip)
            self.tip_angle_debounce_timer.start(100)
    
    def tip_angle_editing_finished(self):
        """Tip Angle編集完了時の処理（キー入力時はリターンで更新）"""
        self.tip_angle_keyboard_input = False
        self.update_tip()
    
    # キープレスイベントハンドラー
    def scan_size_key_press_event(self, event):
        """Scan Sizeキー入力時の処理"""
        # 数字キーや編集キーが押された場合はキーボード入力フラグを設定
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                          Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
                          Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right]:
            self.scan_size_keyboard_input = True
        QDoubleSpinBox.keyPressEvent(self.scan_size_spin, event)
    
    def tip_radius_key_press_event(self, event):
        """Tip Radiusキー入力時の処理"""
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                          Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
                          Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right]:
            self.tip_radius_keyboard_input = True
        QDoubleSpinBox.keyPressEvent(self.tip_radius_spin, event)
    
    def minitip_radius_key_press_event(self, event):
        """Minitip Radiusキー入力時の処理"""
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                          Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
                          Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right]:
            self.minitip_radius_keyboard_input = True
        QDoubleSpinBox.keyPressEvent(self.minitip_radius_spin, event)
    
    def tip_angle_key_press_event(self, event):
        """Tip Angleキー入力時の処理"""
        if event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, 
                          Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9,
                          Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right]:
            self.tip_angle_keyboard_input = True
        QDoubleSpinBox.keyPressEvent(self.tip_angle_spin, event)
    
    def update_tip_position(self):
        """探針位置の更新（適切な範囲）"""
        if not self.tip_actor:
            return
            
        # スライダー値をnm単位に変換（範囲を調整）
        x = self.tip_x_slider.value() / 5.0  # -10 to +10 nm
        y = self.tip_y_slider.value() / 5.0  # -10 to +10 nm
        z = self.tip_z_slider.value() / 5.0  # 2 to 20 nm
        
        self.tip_actor.SetPosition(x, y, z)
        
        self.tip_x_label.setText(f"{x:.1f}")
        self.tip_y_label.setText(f"{y:.1f}")
        self.tip_z_label.setText(f"{z:.1f}")
        
        # AFMパラメーターも更新
        self.afm_params.update({
            'tip_x': x,
            'tip_y': y,
            'tip_z': z,
        })
        
        self.vtk_widget.GetRenderWindow().Render()
        
        # スライダー操作中はシミュレーションを実行しない
        if hasattr(self, 'tip_slider_pressed') and self.tip_slider_pressed:
            return
        
    def toggle_molecule_visibility(self, visible):
        """分子表示の切り替え"""
        if self.sample_actor:
            self.sample_actor.SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()
            
    def toggle_tip_visibility(self, visible):
        """探針表示の切り替え"""
        if self.tip_actor:
            # XY平面視点の際はチェックボックスの状態に関係なく不可視化
            current_view = self.get_current_view_orientation()
            if current_view == 'xy':
                self.tip_actor.SetVisibility(False)
            else:
                self.tip_actor.SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()
            
    def toggle_bonds_visibility(self, visible):
        """結合表示の切り替え"""
        if self.bonds_actor:
            self.bonds_actor.SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()
            
   
    def get_rotated_atom_coords(self):
        """Applies the current rotation transform to the base atom coordinates."""
        if self.atoms_data is None:
            return None

        # Get original coordinates
        x = self.atoms_data['x']
        y = self.atoms_data['y']
        z = self.atoms_data['z']
        num_atoms = len(x)
        
        # 変換行列が存在しない場合は元の座標を返す
        if not hasattr(self, 'combined_transform') or self.combined_transform is None:
            return np.column_stack([x, y, z])

        try:
            # Get the 4x4 transformation matrix from the combined_transform (base + local)
            vtk_matrix = self.combined_transform.GetMatrix()
            
            # 変換行列の値を安全に取得
            transform_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    element = vtk_matrix.GetElement(i, j)
                    # 異常な値をチェック
                    if not np.isfinite(element) or abs(element) > 1e6:
                        print(f"[WARNING] Invalid transform matrix element [{i},{j}]: {element}")
                        return np.column_stack([x, y, z])
                    transform_matrix[i, j] = element
            
            # 変換行列の妥当性をチェック（単位行列に近いかどうか）
            identity = np.eye(4)
            if np.allclose(transform_matrix, identity, atol=1e-6):
                # 変換がない場合は元の座標を返す
                return np.column_stack([x, y, z])
            
            # 座標を同次座標に変換
            original_coords = np.vstack([x, y, z, np.ones(num_atoms)])
            
            # 変換を適用
            with np.errstate(all='ignore'):  # 警告を無視
                rotated_coords_homogeneous = transform_matrix @ original_coords
            
            # NaNやInfをチェック
            if not np.all(np.isfinite(rotated_coords_homogeneous)):
                print("[WARNING] Non-finite values in rotation calculation, using original coordinates")
                return np.column_stack([x, y, z])
            
            # 3D座標に変換
            rotated_coords = rotated_coords_homogeneous[:3, :].T
            
            # 結果の妥当性をチェック
            if not np.all(np.isfinite(rotated_coords)):
                print("[WARNING] Non-finite values in rotated coordinates, using original coordinates")
                return np.column_stack([x, y, z])
            
            # 座標が異常に大きくなっていないかチェック
            max_coord = np.max(np.abs(rotated_coords))
            if max_coord > 1e6:
                print(f"[WARNING] Rotated coordinates too large (max: {max_coord}), using original coordinates")
                return np.column_stack([x, y, z])
            
            return rotated_coords
            
        except Exception as e:
            print(f"[WARNING] Error in rotation calculation: {e}, using original coordinates")
            return np.column_stack([x, y, z])
        
    
    def _connect_worker_delete_later(self, worker):
        """ワーカー終了時にdeleteLaterで安全に破棄する（重複接続は避ける）"""
        if worker is None:
            return
        try:
            worker.finished.connect(worker.deleteLater, type=Qt.UniqueConnection)  # type: ignore[arg-type]
        except Exception:
            # 既に接続済み/環境差異などは黙って無視
            try:
                worker.finished.connect(worker.deleteLater)
            except Exception:
                pass

    def _clear_worker_ref(self, attr_name, worker):
        """self.<attr_name> が worker を指している場合のみ None にする"""
        try:
            if attr_name and hasattr(self, attr_name) and getattr(self, attr_name) is worker:
                setattr(self, attr_name, None)
        except Exception:
            pass

    def is_worker_running(self, worker, attr_name=None):
        """
        deleteLater等で破棄済みのQObjectを考慮した isRunning 判定。
        - RuntimeError（wrapped C/C++ object ... has been deleted）を握りつぶし、
          可能なら参照をクリアして False を返す。
        """
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            if attr_name:
                self._clear_worker_ref(attr_name, worker)
            return False
        except Exception:
            return False

    def _track_worker_ref(self, attr_name, worker):
        """finished/destroyedで参照を確実にクリアするための接続を追加"""
        if worker is None:
            return
        try:
            # finished時に参照をクリア（worker変数は参照比較にしか使わないので安全）
            worker.finished.connect(lambda _=None, w=worker: self._clear_worker_ref(attr_name, w))
        except Exception:
            pass
        try:
            # destroyed時も参照をクリア（Qt側が先に消えるケース対策）
            worker.destroyed.connect(lambda _=None, w=worker: self._clear_worker_ref(attr_name, w))
        except Exception:
            pass

    def stop_worker(self, worker, timeout_ms=100, allow_terminate=False, worker_name="worker"):
        """
        QThreadを安全に停止する。
        - 自己wait（QThread::wait: Thread tried to wait on itself）を防ぐため、
          currentThread == worker の場合はwaitしない。
        - finished→deleteLater を接続してGCタイミング依存を減らす。

        Returns:
            bool: 停止済み（=実行中でない）ならTrue
        """
        if worker is None:
            return True

        try:
            # 自分自身のスレッドからwaitしない（Qt警告＆デッドロック回避）
            if QThread.currentThread() == worker:
                try:
                    if hasattr(worker, "cancel"):
                        worker.cancel()
                except Exception:
                    pass
                try:
                    worker.requestInterruption()
                except Exception:
                    pass
                return False

            self._connect_worker_delete_later(worker)

            # 協調的停止
            try:
                if hasattr(worker, "cancel"):
                    worker.cancel()
            except Exception:
                pass
            try:
                worker.requestInterruption()
            except Exception:
                pass

            if self.is_worker_running(worker):
                if worker.wait(int(timeout_ms)):
                    return True
                if allow_terminate:
                    print(f"Force terminating {worker_name}...")
                    worker.terminate()
                    # terminate後は待機しない（デッドロック/自己wait回避）
                    return not self.is_worker_running(worker)
                return False

            return True
        except Exception as e:
            print(f"[WARNING] stop_worker failed for {worker_name}: {e}")
            return False

    def run_simulation(self):
        coords, mode = self.get_simulation_coords()
        if coords is None:
            QMessageBox.warning(self, "Error", "PDBまたはMRCデータがありません。")
            return
        # 以降、coordsを使ってシミュレーション
        # mode == 'mrc' ならMRC、'pdb' ならPDB
        # 既存のrun_simulationの処理のうち、self.get_rotated_atom_coords()の代わりにcoordsを使うように修正
        self.simulate_btn.setText("Cancel")
        try:
            self.simulate_btn.clicked.disconnect(self.run_simulation)
        except TypeError:
            pass
        self.simulate_btn.clicked.connect(self.cancel_simulation)
        self.progress_container.setVisible(True)

        base_coords = coords
        if base_coords is None:
            QMessageBox.critical(self, "Error", "Could not get atom coordinates.")
            self.on_simulation_finished(None)
            return

        # UIから共通パラメータを取得
        sim_params = {
            'scan_size': self.scan_size_spin.value(),
            'resolution': int(self.resolution_combo.currentText().split('x')[0]),
            'center_x': self.tip_x_slider.value() / 5.0,
            'center_y': self.tip_y_slider.value() / 5.0,
            'tip_radius': self.tip_radius_spin.value(),
            'minitip_radius': self.minitip_radius_spin.value(),
            'tip_angle': self.tip_angle_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'use_vdw': self.use_vdw_check.isChecked()
        }

        # --- チェックされた全ての面の計算タスクを作成 ---
        tasks = []
        if self.afm_x_check.isChecked():
            tasks.append({
                "name": "XY",
                "panel": self.afm_x_frame,
                "coords": base_coords
            })
        if self.afm_y_check.isChecked():
            x_scan = base_coords[:, 1]
            y_scan = base_coords[:, 2]
            z_scan = -base_coords[:, 0]
            tasks.append({
                "name": "YZ",
                "panel": self.afm_y_frame,
                "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)
            })
        if self.afm_z_check.isChecked():
            x_scan, y_scan, z_scan = base_coords[:, 0], base_coords[:, 2], -base_coords[:, 1]
            tasks.append({
                "name": "ZX",
                "panel": self.afm_z_frame,
                "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)
            })

        if not tasks:
            self.on_simulation_finished(None)
            return

        # 既存のワーカーを停止
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker'):
            stopped = self.stop_worker(self.sim_worker, timeout_ms=300, allow_terminate=False, worker_name="sim_worker")
            # 停止できない場合は、実行中スレッドの寿命を切らないよう新規起動を見送る
            if not stopped:
                print("[INFO] sim_worker still running; skipping new simulation start.")
                return
        
        self.sim_worker = AFMSimulationWorker(
            self, sim_params, tasks,
            self.atoms_data['element'] if sim_params['use_vdw'] and self.atoms_data is not None else None,
            self.vdw_radii if sim_params['use_vdw'] and hasattr(self, 'vdw_radii') else None
        )
        self._connect_worker_delete_later(self.sim_worker)
        self._track_worker_ref('sim_worker', self.sim_worker)

        self.simulation_results.clear()
        self.save_image_button.setEnabled(False)
        self.save_asd_button.setEnabled(False)

        # ★★★ 削除：ステータス更新接続を無効化 ★★★
        # self.sim_worker.status_update.connect(self.status_label.setText)
        self.sim_worker.progress.connect(self.progress_bar.setValue)
        self.sim_worker.task_done.connect(self.on_task_finished)
        self.sim_worker.done.connect(self.on_simulation_finished)
        self.sim_worker.start()

    def cancel_simulation(self):
        """シミュレーションのキャンセルを要求する"""
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker'):
            print("Cancel request sent.")
            self.status_label.setText("Cancelling...")
            self.sim_worker.cancel()
    
    def show_afm_result(self, z_map):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        if np.all(np.isnan(z_map)):
            QMessageBox.warning(self, "AFM Result", "No collisions detected.")
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(z_map, cmap='viridis', origin='lower', 
                    interpolation='nearest',
                    extent=[-0.5, 0.5, -0.5, 0.5])  # 正規化不要なら適宜修正
        ax.set_title("Simulated AFM Topography")
        plt.colorbar(im, ax=ax, label="Height [nm]")
        plt.tight_layout()
        plt.show()
        
    def _simulation_worker(self):
        """シミュレーションワーカー（デバッグ強化版）"""
        # UIからパラメータを取得
        scan_size = self.scan_size_spin.value()
        resolution = int(self.resolution_combo.currentText().split('x')[0])
        
        # スキャン範囲を計算
        half_size = scan_size / 2.0
        x_coords = np.linspace(-half_size, half_size, resolution)
        y_coords = np.linspace(-half_size, half_size, resolution)
        
        height_map = np.zeros((resolution, resolution))

        # 衝突判定用の原子データを準備
        atom_x = self.atoms_data['x']
        atom_y = self.atoms_data['y']
        atom_z = self.atoms_data['z']
        atom_elem = self.atoms_data['element']
        atom_radii = np.array([self.vdw_radii.get(e, self.vdw_radii['other']) for e in atom_elem])

        total_steps = resolution * resolution
        current_step = 0
        
        # ★追加: 分子の統計情報を表示
        mol_center_x = np.mean(atom_x)
        mol_center_y = np.mean(atom_y)
        mol_center_z = np.mean(atom_z)
        mol_size_x = np.max(atom_x) - np.min(atom_x)
        mol_size_y = np.max(atom_y) - np.min(atom_y)
        mol_size_z = np.max(atom_z) - np.min(atom_z)
        
        print(f"=== AFM Simulation Started (FIXED v2) ===")
        print(f"Scan size: {scan_size}nm, Resolution: {resolution}x{resolution}")
        print(f"Total atoms: {len(atom_x)}")
        print(f"Molecule center: ({mol_center_x:.2f}, {mol_center_y:.2f}, {mol_center_z:.2f})nm")
        print(f"Molecule size: {mol_size_x:.2f} x {mol_size_y:.2f} x {mol_size_z:.2f}nm")
        print(f"Z range: {np.min(atom_z):.2f} to {np.max(atom_z):.2f}nm")
        print(f"Tip: {self.tip_shape_combo.currentText()}, R={self.tip_radius_spin.value()}nm")
        print(f"Scan range: {-half_size:.1f} to {+half_size:.1f}nm")

        # スキャンループ
        debug_count = 0
        for iy, y in enumerate(y_coords):
            for ix, x in enumerate(x_coords):
                if self.progress_dialog.wasCanceled():
                    print("Simulation canceled by user.")
                    self.simulation_done.emit(None)
                    return

                # 衝突高さ計算
                z_height = self.find_collision_height(x, y, atom_x, atom_y, atom_z, atom_radii)
                height_map[iy, ix] = z_height
                
                # ★改良: より多様な位置でデバッグ出力
                if debug_count < 10:  # 最初の10点
                    print(f"Point ({x:6.2f}, {y:6.2f}) -> Z={z_height:8.3f}nm")
                    debug_count += 1
                elif (iy == resolution//2 and ix == resolution//2):  # 中心点
                    print(f"Center ({x:6.2f}, {y:6.2f}) -> Z={z_height:8.3f}nm")
                elif (iy == resolution-1 and ix == resolution-1):  # 最後の点
                    print(f"End    ({x:6.2f}, {y:6.2f}) -> Z={z_height:8.3f}nm")

                current_step += 1
                progress = int((current_step / total_steps) * 100)
                self.simulation_progress.emit(progress)

        # ★追加: 詳細な統計情報
        valid_heights = height_map[height_map > mol_center_z - 10]  # 明らかに低すぎる値を除外
        
        print(f"=== Simulation Completed ===")
        print(f"Height range: {np.min(height_map):.3f} to {np.max(height_map):.3f}nm")
        print(f"Valid heights: {np.min(valid_heights):.3f} to {np.max(valid_heights):.3f}nm")
        print(f"Mean height: {np.mean(valid_heights):.3f}nm")
        print(f"Height std: {np.std(valid_heights):.3f}nm")
        
        # 完了シグナルを送信
        self.simulation_done.emit(height_map)

    def check_tip_position_and_molecule_overlap(self):
        """探針位置と分子の位置関係を確認するデバッグメソッド"""
        if self.atoms_data is None:
            print("No molecule loaded")
            return
        
        # 現在の探針位置を取得
        tip_x = self.afm_params['tip_x']
        tip_y = self.afm_params['tip_y'] 
        tip_z = self.afm_params['tip_z']
        
        # 分子の統計
        mol_x_range = (np.min(self.atoms_data['x']), np.max(self.atoms_data['x']))
        mol_y_range = (np.min(self.atoms_data['y']), np.max(self.atoms_data['y']))
        mol_z_range = (np.min(self.atoms_data['z']), np.max(self.atoms_data['z']))
        
        print(f"\n=== Position Check ===")
        print(f"Tip position: ({tip_x:.2f}, {tip_y:.2f}, {tip_z:.2f})nm")
        print(f"Molecule X range: {mol_x_range[0]:.2f} to {mol_x_range[1]:.2f}nm")
        print(f"Molecule Y range: {mol_y_range[0]:.2f} to {mol_y_range[1]:.2f}nm") 
        print(f"Molecule Z range: {mol_z_range[0]:.2f} to {mol_z_range[1]:.2f}nm")
        
        # 探針が分子の上にあるかチェック
        tip_over_molecule = (mol_x_range[0] <= tip_x <= mol_x_range[1] and 
                            mol_y_range[0] <= tip_y <= mol_y_range[1])
        
        print(f"Tip over molecule: {tip_over_molecule}")
        
        if tip_z <= mol_z_range[1]:
            print(f"WARNING: Tip Z position ({tip_z:.2f}) is too low! Molecule top is at {mol_z_range[1]:.2f}nm")


    def create_tip_footprint(self, R, alpha_deg, pixel_size):
        """Dilation演算に使うための、探針の2Dフットプリントを生成する"""
        # 探針の影響範囲をピクセル単位で計算
        tip_pixel_radius = int(np.ceil(R * 3 / pixel_size))
        size = 2 * tip_pixel_radius + 1
        footprint = np.zeros((size, size))
        
        center = tip_pixel_radius
        alpha = np.radians(alpha_deg)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        for iy in range(size):
            for ix in range(size):
                # ピクセル中心からの物理的な距離
                r_2d = np.sqrt(((ix - center) * pixel_size)**2 + ((iy - center) * pixel_size)**2)
                
                # 探針の高さを計算 (反転させた形状)
                r_crit = R * ca
                if r_2d <= r_crit:
                    z = R - np.sqrt(R**2 - r_2d**2)
                else:
                    z = (r_2d * sa + R * (1 - ca)) / ca # 修正された円錐式

                footprint[iy, ix] = -z # Dilationでは反転した探針を使う
        
        return footprint
    
    def on_task_finished(self, z_map, target_panel):
        """個別の計算タスクが完了した際に呼び出されるスロット"""
        if z_map is not None and target_panel is not None:
            image_key = target_panel.objectName()
            
            # ★★★ 修正箇所: 生データを保存し、表示更新関数を呼び出す ★★★
            # 1. フィルターをかける前の「生」データを保存
            self.raw_simulation_results[image_key] = z_map
            
            # 2. フィルター適用と表示更新を行う関数を呼び出す
            self.process_and_display_single_image(image_key)
    
    def process_and_display_single_image(self, image_key):
        """指定されたキーの画像を処理して表示する"""
        if image_key not in self.raw_simulation_results:
            return

        raw_data = self.raw_simulation_results[image_key]
        
        # フィルターが有効かチェック
        if self.apply_filter_check.isChecked():
            cutoff_wl = self.filter_cutoff_spin.value()
            scan_size = self.scan_size_spin.value()
            processed_data = apply_low_pass_filter(raw_data, scan_size, cutoff_wl)
        else:
            processed_data = raw_data

        # 表示用と保存用のデータを更新
        self.simulation_results[image_key] = processed_data
        
        # 対応するパネルを見つけて表示を更新
        target_panel = self.findChild(QFrame, image_key)
        if target_panel:
            self.display_afm_image(processed_data, target_panel)

    
    def process_and_display_all_images(self):
        """現在表示されている全ての画像を再処理・再表示する"""
        #print("Filter settings changed, updating all views...")
        for image_key in self.raw_simulation_results.keys():
            self.process_and_display_single_image(image_key)

    def start_filter_update_timer(self):
        """フィルターのカットオフ値変更時にタイマーで更新を遅延させる"""
        if not self.apply_filter_check.isChecked():
            return # フィルターがOFFの時は何もしない
            
        if not hasattr(self, 'filter_update_timer'):
            self.filter_update_timer = QTimer(self)  # 親ウィンドウを設定
            self.filter_update_timer.setSingleShot(True)
            self.filter_update_timer.timeout.connect(self.process_and_display_all_images)
        
        self.filter_update_timer.start(500) # 500ミリ秒後に更新

    def on_simulation_finished(self, result):
        """
        シミュレーションの完了・失敗・キャンセル後の全てのクリーンアップ処理を担当します。
        このメソッドは、バックグラウンドのスレッドが終了した際に一度だけ呼び出されます。
        """
        # 1. ボタンを「Run」状態に戻し、再度クリックできるようにする
        self.simulate_btn.setText("Run AFM Simulation")
        try:
            self.simulate_btn.clicked.disconnect(self.cancel_simulation)
        except TypeError:
            pass  # すでに接続が解除されている場合は何もしない
        self.simulate_btn.clicked.connect(self.run_simulation)
        self.simulate_btn.setEnabled(True)

        # 2. プログレス表示用のコンテナを非表示にする
        self.progress_container.setVisible(False)

        # 3. シミュレーション結果が一つでもあれば、各種保存ボタンを有効化する
        if self.simulation_results:
            #print("Simulation finished. Enabling save buttons.")
            self.save_image_button.setEnabled(True)
            self.save_asd_button.setEnabled(True)
        else:
            #print("Simulation finished, but no results were generated (or it was cancelled).")
            pass

        


    
    def display_afm_image(self, height_map, target_panel):
        """
        計算された高さマップをグレイスケールでUIに表示します。
        """
        if target_panel is None or height_map is None: return
        
        import matplotlib.cm as cm
        from PyQt5.QtGui import QImage, QPixmap
        
        # --- 正規化処理 ---
        valid_pixels = height_map[height_map > -1e8]
        if valid_pixels.size < 2:
            image_data = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8)
        else:
            min_h, max_h = np.min(valid_pixels), np.max(valid_pixels)
            if max_h <= min_h:
                image_data = np.full((height_map.shape[0], height_map.shape[1], 3), 128, dtype=np.uint8)
            else:
                clipped_map = np.clip(height_map, min_h, max_h)
                norm_map = (clipped_map - min_h) / (max_h - min_h)
                image_data = (cm.gray(norm_map)[:, :, :3] * 255).astype(np.uint8)

        # ★★★ ここからが修正箇所 ★★★
        # 3Dビューの上下方向 (Y軸が上) と2D画像の表示 (Y軸が下) を合わせるため、
        # 画像データを上下反転させます。
        image_data_flipped = np.flipud(image_data)
        # ★★★ 修正箇所ここまで ★★★

        height, width, channel = image_data_flipped.shape
        bytes_per_line = channel * width
        
        self.afm_qimage = QImage(image_data_flipped.copy().data, width, height, bytes_per_line, QImage.Format_RGB888)

        # 既存のウィジェットをクリアしてから新しい画像を表示
        while target_panel.layout().count():
            child = target_panel.layout().takeAt(0)
            if child.widget(): child.widget().deleteLater()

        image_label = QLabel()
        pixmap = QPixmap.fromImage(self.afm_qimage)
        image_label.setPixmap(pixmap.scaled(target_panel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        image_label.setAlignment(Qt.AlignCenter)
        target_panel.layout().addWidget(image_label)

   

    def find_collision_height(self, x, y, atom_x, atom_y, atom_z, atom_radii):
        """VTKで作成されたtip_actorと分子との衝突Z高さを返す"""

        # tip_actor から vtkPolyData を取得
        polydata = self.tip_actor.GetMapper().GetInput()
        if polydata is None:
            print("[WARNING] tip geometry is not defined.")
            return None

        points = polydata.GetPoints()
        n_points = points.GetNumberOfPoints()
        if n_points == 0:
            print("[WARNING] tip geometry has no points.")
            return None

        # tip の座標を (x, y) に移動（tip作成時は原点を中心と仮定）
        transformed_tip_points = []
        for i in range(n_points):
            px, py, pz = points.GetPoint(i)
            transformed_tip_points.append([px + x, py + y, pz])

        transformed_tip_points = np.array(transformed_tip_points)

        # 各原子とtip点群の最近接距離を計算（高速化のためBallTreeなどを使うのが理想だがここでは総当り）
        min_collision_z = None
        for i in range(len(atom_x)):
            ax, ay, az = atom_x[i], atom_y[i], atom_z[i]
            ar = atom_radii[i]

            for tp in transformed_tip_points:
                dx = tp[0] - ax
                dy = tp[1] - ay
                dz = tp[2] - az
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                if dist <= ar:
                    if (min_collision_z is None) or (tp[2] < min_collision_z):
                        min_collision_z = tp[2]

        return min_collision_z

    def is_colliding(self, tip_x, tip_y, tip_z, atom_x, atom_y, atom_z, atom_radii):
        """探針と原子群の衝突判定（修正版）"""
        tip_shape = self.tip_shape_combo.currentText().lower()
        tip_radius = self.tip_radius_spin.value()
        tip_angle = self.tip_angle_spin.value()
        minitip_radius = self.minitip_radius_spin.value()

        # 各原子について衝突をチェック
        for i in range(len(atom_x)):
            atom_pos = (atom_x[i], atom_y[i], atom_z[i])
            tip_apex = (tip_x, tip_y, tip_z)
            
            # 探針表面から原子中心までの距離を計算
            if tip_shape == "cone":
                dist_surface = self.dist_point_to_cone_tip(
                    atom_pos, tip_apex, tip_radius, tip_angle)
            elif tip_shape == "sphere":
                dist_surface = self.dist_point_to_sphere_tip(
                    atom_pos, tip_apex, tip_radius, tip_angle, minitip_radius)
            else:  # Paraboloid
                dist_surface = self.dist_point_to_paraboloid_tip(
                    atom_pos, tip_apex, tip_radius)
            
            # 衝突判定：探針表面から原子中心までの距離が原子半径以下なら衝突
            if dist_surface <= atom_radii[i]:
                return True
                
        return False

    def dist_point_to_cone_tip(self, p, tip_apex, R, alpha_deg):
        """点pと円錐探針表面との最短距離（修正版）"""
        alpha = np.radians(alpha_deg)
        px, py, pz = p
        tx, ty, tz = tip_apex
        
        # 探針の先端（apex）を原点とした相対座標
        dx, dy, dz = px - tx, py - ty, pz - tz
        r_2d = np.sqrt(dx**2 + dy**2)
        
        # 修正1: 円錐の幾何学を正確に計算
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # 球状先端部分の中心位置を修正
        sphere_center_z = R  # 球の中心は先端からR上方
        
        # 球との境界半径を正確に計算
        r_crit = R * sin_alpha  # 球と円錐の接続部の半径
        
        # 修正2: 距離計算を改善
        if r_2d <= r_crit and dz <= sphere_center_z:
            # 球状部分との距離
            dist_to_sphere_center = np.sqrt(r_2d**2 + (dz - sphere_center_z)**2)
            dist_surface = dist_to_sphere_center - R
        else:
            # 円錐部分との距離を正確に計算
            # 円錐の母線方向の単位ベクトル：(sin_alpha, 0, cos_alpha)
            # 点から円錐軸（Z軸）への垂直距離：r_2d
            # 点のZ座標から適切な円錐面までの距離を計算
            
            # 円錐面上の対応点のZ座標
            z_on_cone = sphere_center_z + (r_2d - r_crit) / np.tan(alpha)
            
            # 修正3: 符号付き距離を正確に計算
            # 円錐面の法線ベクトル：(-sin_alpha, 0, cos_alpha)
            # 点から円錐面への符号付き距離
            dist_surface = (r_2d - r_crit) * cos_alpha + (dz - z_on_cone) * sin_alpha
            
        return dist_surface

    def dist_point_to_sphere_tip(self, p, tip_apex, R_cone, alpha_deg, R_sphere):
        """点pと球+円錐の複合探針表面との最短距離"""
        # この実装では、先端球が支配的として簡易計算
        return self.dist_point_to_cone_tip(p, tip_apex, R_sphere, 90)

    def dist_point_to_paraboloid_tip(self, p, tip_apex, R):
        """点pと放物面探針表面との最短距離"""
        px, py, pz = p
        tx, ty, tz = tip_apex
        # 座標変換
        dx, dy, dz = px - tx, py - ty, pz - tz
        r_sq = dx**2 + dy**2
        # 放物面上の対応する高さ
        z_parabola = r_sq / (2 * R)
        return dz - z_parabola




    def choose_background_color(self):
        """背景色選択ダイアログ"""
        # 現在の背景色を取得
        current_color = QColor()
        current_color.setRgbF(self.current_bg_color[0], 
                             self.current_bg_color[1], 
                             self.current_bg_color[2])
        
        color = QColorDialog.getColor(current_color, self, "Choose Background Color")
        if color.isValid():
            # RGB値を0-1範囲に変換
            self.current_bg_color = (color.redF(), color.greenF(), color.blueF())
            
            # ボタンの色を更新
            self.bg_color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                    color: {'black' if sum([color.red(), color.green(), color.blue()]) > 400 else 'white'};
                    border: 2px solid #555;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    border-color: #777;
                }}
            """)
            
            # VTKレンダラーの背景色を更新
            self.renderer.SetBackground(*self.current_bg_color)
            self.vtk_widget.GetRenderWindow().Render()
    
    def clear_mrc_data(self):
        """MRCデータとアクターをクリア"""
        # MRCアクターをレンダラーから削除
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            self.renderer.RemoveActor(self.mrc_actor)
            self.mrc_actor = None
        
        # MRCデータをクリア
        if hasattr(self, 'mrc_data'):
            self.mrc_data = None
        if hasattr(self, 'mrc_data_original'):
            self.mrc_data_original = None
        if hasattr(self, 'mrc_metadata'):
            self.mrc_metadata = None
        if hasattr(self, 'mrc_name'):
            self.mrc_name = None
            self.mrc_id = ""
        if hasattr(self, 'mrc_surface_coords'):
            self.mrc_surface_coords = None
        
        # MRCラベルをリセット
        if hasattr(self, 'file_label'):
            self.file_label.setText("File Name: (none)")
        
        # MRCグループを無効化
        if hasattr(self, 'mrc_group'):
            self.mrc_group.setEnabled(False)
        
        # 回転ウィジェットも無効化（PDBデータがない場合）
        if not hasattr(self, 'atoms_data') or self.atoms_data is None:
            if hasattr(self, 'rotation_widgets'):
                for axis in ['X', 'Y', 'Z']:
                    self.rotation_widgets[axis]['spin'].setEnabled(False)
                    self.rotation_widgets[axis]['slider'].setEnabled(False)
        
        # レンダリング更新
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()
    
    def clear_pdb_data(self):
        """PDBデータとアクターをクリア"""
        # PDBアクターをレンダラーから削除
        if hasattr(self, 'sample_actor') and self.sample_actor is not None:
            self.renderer.RemoveActor(self.sample_actor)
            self.sample_actor = None
        if hasattr(self, 'bonds_actor') and self.bonds_actor is not None:
            self.renderer.RemoveActor(self.bonds_actor)
            self.bonds_actor = None
        
        # PDBデータをクリア
        if hasattr(self, 'atoms_data'):
            self.atoms_data = None
        if hasattr(self, 'pdb_name'):
            self.pdb_name = None
            self.pdb_id = ""
        if hasattr(self, 'cif_name'):
            self.cif_name = None
            self.cif_id = ""
        
        # PDBラベルをリセット
        if hasattr(self, 'file_label'):
            self.file_label.setText("File Name: (none)")
        
        # 統計情報をリセット
        if hasattr(self, 'stats_label'):
            self.stats_label.setText("No data loaded")
        
        # 回転ウィジェットも無効化（MRCデータがない場合）
        if not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            if hasattr(self, 'rotation_widgets'):
                for axis in ['X', 'Y', 'Z']:
                    self.rotation_widgets[axis]['spin'].setEnabled(False)
                    self.rotation_widgets[axis]['slider'].setEnabled(False)
        
        # レンダリング更新
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.GetRenderWindow().Render()
    
    def update_mrc_actor_color(self):
        """既存のMRCアクターの色を更新"""
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            # マッパーのスカラー可視性を無効にする
            mapper = self.mrc_actor.GetMapper()
            if mapper:
                mapper.ScalarVisibilityOff()
            
            prop = self.mrc_actor.GetProperty()
            # MRCは常に選択された色を使用（カラースキームは関係ない）
           
            prop.SetColor(self.current_single_color[0], self.current_single_color[1], self.current_single_color[2])
            
            self.vtk_widget.GetRenderWindow().Render()
    
    def on_color_scheme_changed(self):
        """カラースキーム変更時の処理"""
        print(f"Color scheme changed to: {self.color_combo.currentText()}")
        if self.atoms_data is not None:
            self.update_display()
        # MRCデータの場合はカラースキームは関係ないので何もしない
    
    def choose_single_color(self):
        """単色モード用カラー選択"""
        # 現在の単色を取得
        current_color = QColor()
        current_color.setRgbF(self.current_single_color[0],
                            self.current_single_color[1],
                            self.current_single_color[2])
        
        color = QColorDialog.getColor(current_color, self, "Choose Single Color")
        if color.isValid():
            # RGB値を0-1範囲に変換
            old_color = self.current_single_color
            self.current_single_color = (color.redF(), color.greenF(), color.blueF())
            
           
            # ボタンの色を更新
            self.single_color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                    color: {'black' if sum([color.red(), color.green(), color.blue()]) > 400 else 'white'};
                    border: 2px solid #555;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    border-color: #777;
                }}
            """)
            
            # 表示を更新
            if self.atoms_data is not None:                
                self.update_display()
            elif hasattr(self, 'mrc_data') and self.mrc_data is not None:             # MRCデータの場合も色を更新
                self.update_mrc_actor_color()
    
    def update_brightness(self):
        """明るさ調整"""
        brightness = self.brightness_slider.value()
        self.brightness_factor = brightness / 100.0
        self.brightness_label.setText(f"{brightness}%")
        
        # ライティングを更新
        self.update_lighting_intensity()
        self.vtk_widget.GetRenderWindow().Render()
    
    def update_lighting(self):
        """環境光設定の更新"""
        ambient = self.ambient_slider.value()
        self.ambient_label.setText(f"{ambient}%")
        
        # レンダラーの環境光を設定
        ambient_factor = ambient / 100.0
        self.renderer.SetAmbient(ambient_factor, ambient_factor, ambient_factor)
        self.vtk_widget.GetRenderWindow().Render()
    
    def update_material(self):
        """マテリアル設定の更新"""
        specular = self.specular_slider.value()
        self.specular_label.setText(f"{specular}%")
        
        # 全てのアクターのスペキュラを更新
        self.update_actor_materials()
        self.vtk_widget.GetRenderWindow().Render()
    
    def update_lighting_intensity(self):
        """ライトの強度を明るさファクターで調整"""
        lights = self.renderer.GetLights()
        lights.InitTraversal()
        
        light = lights.GetNextItem()
        while light:
            # 元の強度に明るさファクターを適用
            if hasattr(light, '_original_intensity'):
                light.SetIntensity(light._original_intensity * self.brightness_factor)
            else:
                # 初回は現在の強度を保存
                light._original_intensity = light.GetIntensity()
                light.SetIntensity(light._original_intensity * self.brightness_factor)
            
            light = lights.GetNextItem()
    
    def update_actor_materials(self):
        """全アクターのマテリアル特性を更新"""
        specular_factor = self.specular_slider.value() / 100.0
        
        # 分子アクター
        if self.sample_actor and hasattr(self.sample_actor, 'GetProperty'):
            self.sample_actor.GetProperty().SetSpecular(specular_factor)
            self.sample_actor.GetProperty().SetSpecularPower(50)
        
        # 結合アクター
        if self.bonds_actor and hasattr(self.bonds_actor, 'GetProperty'):
            self.bonds_actor.GetProperty().SetSpecular(specular_factor * 0.5)
        
        # 探針アクター
        if self.tip_actor and hasattr(self.tip_actor, 'GetProperty'):
            self.tip_actor.GetProperty().SetSpecular(min(0.9, specular_factor * 1.5))
    
    def apply_pymol_style(self):
        """PyMOLスタイルプリセット適用"""
        # 背景を黒に
        self.current_bg_color = (0.0, 0.0, 0.0)
        self.renderer.SetBackground(*self.current_bg_color)
        
        # ボタンの色を更新
        self.bg_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #000000;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                border-color: #777;
            }
        """)
        
        # 明るさを120%に
        self.brightness_slider.setValue(120)
        self.brightness_factor = 1.2
        self.brightness_label.setText("120%")
        
        # 環境光を5%に
        self.ambient_slider.setValue(5)
        self.ambient_label.setText("5%")
        self.renderer.SetAmbient(0.05, 0.05, 0.05)
        
        # スペキュラを80%に
        self.specular_slider.setValue(80)
        self.specular_label.setText("80%")
        
        # 設定を適用
        self.update_lighting_intensity()
        self.update_actor_materials()
        
        # PyMOLライクな元素カラーに変更
        self.element_colors.update({
            'C': (0.565, 0.565, 0.565),  # PyMOLのカーボングレー
            'O': (1.0, 0.051, 0.051),    # 鮮やかな赤
            'N': (0.188, 0.314, 0.973),  # 鮮やかな青
            'H': (0.9, 0.9, 0.9),        # 白
            'S': (1.0, 1.0, 0.188),      # 鮮やかな黄色
            'P': (1.0, 0.502, 0.0),      # オレンジ
        })
        
        # 表示を更新
        if self.atoms_data is not None:
            self.update_display()
        
        self.vtk_widget.GetRenderWindow().Render()
        
        QMessageBox.information(self, "Style Applied", "PyMOL style applied successfully!")
    
    def apply_dark_theme(self):
        """ダークテーマプリセット適用"""
        # 背景をダークグレーに
        self.current_bg_color = (0.1, 0.1, 0.15)
        self.renderer.SetBackground(*self.current_bg_color)
        
        # ボタンの色を更新
        self.bg_color_btn.setStyleSheet("""
            QPushButton {
                background-color: #191926;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                border-color: #777;
            }
        """)
        
        # 明るさを100%に
        self.brightness_slider.setValue(100)
        self.brightness_factor = 1.0
        self.brightness_label.setText("100%")
        
        # 環境光を15%に
        self.ambient_slider.setValue(15)
        self.ambient_label.setText("15%")
        self.renderer.SetAmbient(0.15, 0.15, 0.15)
        
        # スペキュラを60%に
        self.specular_slider.setValue(60)
        self.specular_label.setText("60%")
        
        # 設定を適用
        self.update_lighting_intensity()
        self.update_actor_materials()
        
        # 表示を更新
        if self.atoms_data is not None:
            self.update_display()
        
        self.vtk_widget.GetRenderWindow().Render()
        
        QMessageBox.information(self, "Style Applied", "Dark theme applied successfully!")
    
    def load_settings(self):
        """起動時にウィンドウの位置、サイズ、スプリッターの状態を復元する"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                
                # ウィンドウのジオメトリ（位置とサイズ）を復元
                if 'geometry' in settings:
                    self.setGeometry(*settings['geometry'])
                
                # 各スプリッターの状態を復元
                if 'main_splitter' in settings:
                    self.main_splitter.setSizes(settings['main_splitter'])
                if 'afm_splitter' in settings:
                    self.afm_splitter.setSizes(settings['afm_splitter'])
                if 'view_control_splitter' in settings:
                    self.view_control_splitter.setSizes(settings['view_control_splitter'])

                if 'last_import_dir' in settings:
                    self.last_import_dir = settings['last_import_dir']
                
                # MRCのZ軸フリップ状態を復元（デフォルトはTrue）
                if 'mrc_z_flip' in settings:
                    self.mrc_z_flip = settings['mrc_z_flip']
                else:
                    self.mrc_z_flip = True  # デフォルトで有効
                
                # チェックボックスの状態を確実に設定
                if hasattr(self, 'mrc_z_flip_check'):
                    self.mrc_z_flip_check.blockSignals(True)  # シグナルを一時的にブロック
                    self.mrc_z_flip_check.setChecked(self.mrc_z_flip)
                    self.mrc_z_flip_check.blockSignals(False)  # シグナルを再有効化
                
                #print("Settings loaded successfully.")

        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Could not load settings: {e}")
            # エラー発生時はデフォルトで起動

    def save_settings(self):
        """ウィンドウの位置、サイズ、スプリッターの状態を保存する"""
        settings = {
            'geometry': self.geometry().getRect(),
            'main_splitter': self.main_splitter.sizes(),
            'afm_splitter': self.afm_splitter.sizes(),
            'view_control_splitter': self.view_control_splitter.sizes(),
            'last_import_dir': self.last_import_dir,
            'mrc_z_flip': self.mrc_z_flip
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            #print("Settings saved successfully.")
        except IOError as e:
            print(f"Could not save settings: {e}")
    
    def handle_save_asd(self):
        """「Save as ASD...」ボタンが押されたときの処理"""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Data", "No simulation data available to save.")
            return

         # 保存可能なデータの名前（キー）を取得
        available_keys = list(self.simulation_results.keys())
        
        # ユーザーに表示するための分かりやすい名前の辞書
        display_names = {
            "XY_Frame": "XY View",
            "YZ_Frame": "YZ View",
            "ZX_Frame": "ZX View"
        }
        
        # 選択肢リストを作成
        choices = [display_names.get(key, key) for key in available_keys]
        
        selected_key = None
        if len(available_keys) > 1:
            # データが複数ある場合、ダイアログで選択させる
            choice, ok = QInputDialog.getItem(self, "Select Data to Save", "保存するデータを選択してください:", choices, 0, False)
            if not ok or not choice:
                return # キャンセルされた場合
            # 選択された表示名から内部キーを逆引き
            for key, name in display_names.items():
                if name == choice:
                    selected_key = key
                    break
        elif len(available_keys) == 1:
            # データが1つだけなら、それを自動で選択
            selected_key = available_keys[0]
        else:
            # データがない場合は何もしない
            return

        if selected_key is None:
            return
                
        data_to_save = self.simulation_results[selected_key]
        image_key_name = display_names.get(selected_key, selected_key).replace(" ", "") # ファイル名用
        default_id = self.get_active_dataset_id()
        default_filename = f"{default_id}_{image_key_name}.asd"

        directory = ""
        # 最後に使用したディレクトリが存在し、アクセス可能かチェック
        if self.last_import_dir and os.path.isdir(self.last_import_dir):
            directory = self.last_import_dir
        
        # ファイル名と安全なディレクトリを結合して、最終的なデフォルトパスを作成
        default_save_path = os.path.join(directory, default_filename)

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation as ASD", default_save_path, "ASD files (*.asd)",
            options=QFileDialog.DontUseNativeDialog
        )

        if not save_path:
            return

        try:            
            # --- シミュレーション条件を収集 ---
            rot_x = self.rotation_widgets['X']['spin'].value()
            rot_y = self.rotation_widgets['Y']['spin'].value()
            rot_z = self.rotation_widgets['Z']['spin'].value()
            
            tip_shape = self.tip_shape_combo.currentText()
            tip_radius = self.tip_radius_spin.value()
            tip_angle = self.tip_angle_spin.value()
            
            scan_size = self.scan_size_spin.value()
            resolution = self.resolution_combo.currentText()
            center_x = self.tip_x_slider.value() / 5.0
            center_y = self.tip_y_slider.value() / 5.0
            
            use_vdw = "Yes" if self.use_vdw_check.isChecked() else "No"
            sim_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # --- コメント文字列を生成 ---
            pdb_file = self.pdb_name if getattr(self, 'pdb_id', '') else "(none)"
            mrc_file = self.mrc_name if getattr(self, 'mrc_id', '') else "(none)"

            comment_lines = [
                f"pyNuD_simulator Log",
                f"Date: {sim_date}",
                f"--------------------",
                f"[File Info]",
                f"PDB File: {pdb_file}",
                f"MRC File: {mrc_file}",
                f"",
                f"[View Settings]",
                f"Rotation X: {rot_x:.1f} deg",
                f"Rotation Y: {rot_y:.1f} deg",
                f"Rotation Z: {rot_z:.1f} deg",
                f"",
                f"[Tip Conditions]",
                f"Shape: {tip_shape}",
                f"Radius: {tip_radius:.2f} nm",
            ]
            
            if tip_shape == "Cone":
                comment_lines.append(f"Angle: {tip_angle:.1f} deg")
            elif tip_shape == "Sphere":
                minitip_radius = self.minitip_radius_spin.value()
                comment_lines.append(f"Angle: {tip_angle:.1f} deg")
                comment_lines.append(f"Minitip Radius: {minitip_radius:.2f} nm")

            comment_lines.extend([
                f"",
                f"[Scan Parameters]",
                f"Scan Size: {scan_size:.1f} nm",
                f"Resolution: {resolution}",
                f"Center: ({center_x:.2f}, {center_y:.2f}) nm",
                f"",
                f"[Calculation Method]",
                f"Consider vdW: {use_vdw}",
            ])
            
            comment = "\n".join(comment_lines)
            
            # # save_simulation_as_asd を呼び出す
            success = self.save_simulation_as_asd(save_path, comment, data_to_save)
            if success:
                QMessageBox.information(self, "Save Successful", f"Data successfully saved to:\n{save_path}")
            else:
                QMessageBox.critical(self, "Save Error", "Failed to save ASD file. Check console for details.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the ASD file:\n{e}")



    def save_simulation_as_asd(self, save_path, comment_string, height_map):
        """
        ASDファイルを保存するメソッド。Igor Proのコードと完全に一致させる。
        """
        try:
            # ★★★ 修正点1: 引数を直接height_mapとして受け取り、その形状を取得 ★★★
            y_pixels, x_pixels = height_map.shape
            scan_size_nm = self.scan_size_spin.value()
            
            x_scan_size = scan_size_nm
            y_scan_size = scan_size_nm * (y_pixels / x_pixels) if x_pixels > 0 else 0

            # ★★★ 修正点2: comment_bytes を正しく使用する ★★★
            ope_name_bytes = "Nobody".encode('utf-8')
            comment_bytes = comment_string.encode('utf-8')
            
            # Igorコードの `165` は固定ヘッダーのバイト数
            file_header_size = 165 + len(ope_name_bytes) + len(comment_bytes)
            frame_header_size = 32

            now = datetime.datetime.now()
            
            with open(save_path, 'wb') as f:
                # --- ファイルヘッダー書き込み ---
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', file_header_size))
                f.write(struct.pack('<i', frame_header_size))
                f.write(struct.pack('<i', 932))
                f.write(struct.pack('<i', len(ope_name_bytes)))
                f.write(struct.pack('<i', len(comment_bytes))) # 正しいコメント長を書き込む
                f.write(struct.pack('<i', 20564))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', x_pixels))
                f.write(struct.pack('<i', y_pixels))
                f.write(struct.pack('<i', int(x_scan_size)))
                f.write(struct.pack('<i', int(y_scan_size)))
                f.write(struct.pack('<B', 0))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', now.year))
                f.write(struct.pack('<i', now.month))
                f.write(struct.pack('<i', now.day))
                f.write(struct.pack('<i', now.hour))
                f.write(struct.pack('<i', now.minute))
                f.write(struct.pack('<i', now.second))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<iiii', 0, 0, 0, 0))
                f.write(struct.pack('<i', 1))
                f.write(struct.pack('<i', 262144))
                f.write(struct.pack('<i', 12))
                f.write(struct.pack('<f', 4000.0))
                f.write(struct.pack('<f', 1700.0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<f', 20.0))
                f.write(struct.pack('<f', 2.0))
                
                f.write(ope_name_bytes)
                f.write(comment_bytes) # ★★★ 正しいコメント本体を書き込む ★★★

                # --- フレームヘッダー書き込み ---
                max_data_raw = np.max(height_map)
                min_data_raw = np.min(height_map)

                f.write(struct.pack('<I', 0))
                f.write(struct.pack('<H', int(max_data_raw)))
                f.write(struct.pack('<H', int(min_data_raw)))
                f.write(struct.pack('<h', 0))
                f.write(struct.pack('<h', 0))
                f.write(struct.pack('<f', 0.0))
                f.write(struct.pack('<f', 0.0))
                f.write(struct.pack('<B', 0))
                f.write(struct.pack('<B', 0))
                f.write(struct.pack('<h', 0))
                f.write(struct.pack('<i', 0))
                f.write(struct.pack('<i', 0))

                # --- 画像データ書き込み ---
                piezo_const_z = 20.0
                driver_gain_z = 2.0
                
                for y in range(y_pixels):
                    for x in range(x_pixels):
                        height_value = height_map[y, x]
                        # シミュレーターは凹凸データのみなので、nm → uint16の変換のみ
                        data = (5.0 - height_value / piezo_const_z / driver_gain_z) * 4096.0 / 10.0
                        f.write(struct.pack('<h', int(data)))
            return True
            
        except Exception as e:
            print(f"[ERROR] SaveASD failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def handle_save_3d_view(self):
        """現在の3Dビューを画像ファイルとして保存する"""
        if self.pdb_name == "":
            QMessageBox.warning(self, "No Data", "Please load a PDB file first.")
            return

        # --- ファイル保存ダイアログの準備 ---
        default_filename = f"{self.pdb_name}_3D_view.png"
        directory = ""
        if self.last_import_dir and os.path.isdir(self.last_import_dir):
            directory = self.last_import_dir
        
        default_save_path = os.path.join(directory, default_filename)
        
        # ユーザーにファイル名と保存形式を選択させる
        filters = "PNG Image (*.png);;TIFF Image (*.tif)"
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save 3D View As...", default_save_path, filters,
            options=QFileDialog.DontUseNativeDialog
        )

        if not save_path:
            return

        # --- VTKウィンドウのキャプチャと保存 ---
        try:
            # 1. VTKウィンドウを画像データに変換するフィルターを作成
            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(self.vtk_widget.GetRenderWindow())
            # アルファチャンネル（透明度）を含めずにRGBのみをキャプチャ
            window_to_image_filter.SetInputBufferTypeToRGB() 
            # スケーリングを無効にし、ウィンドウの解像度でキャプチャ
            window_to_image_filter.SetScale(1) 
            window_to_image_filter.Update()

            # 2. 選択されたファイル形式に応じて適切なライターを選択
            if save_path.endswith('.png'):
                writer = vtk.vtkPNGWriter()
            elif save_path.endswith('.tif'):
                writer = vtk.vtkTIFFWriter()
            else:
                # ユーザーが拡張子を入力しなかった場合、選択したフィルターから判断
                if "png" in selected_filter:
                    save_path += ".png"
                    writer = vtk.vtkPNGWriter()
                else:
                    save_path += ".tif"
                    writer = vtk.vtkTIFFWriter()

            # 3. ファイルを書き出す
            writer.SetFileName(save_path)
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()
            
            QMessageBox.information(self, "Save Successful", f"3D view successfully saved to:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the 3D view:\n{e}")
    
    def handle_save_image(self):
        """Export one or more simulated AFM images (PNG) with optional incremental rotation."""
        if not self.simulation_results:
            QMessageBox.warning(self, "No Data", "No simulation data available to save.")
            return
        
        # Build available (only those already simulated)
        available_keys = list(self.simulation_results.keys())
        display_names = {"XY_Frame": "XY View", "YZ_Frame": "YZ View", "ZX_Frame": "ZX View"}
        
        dlg = SaveAFMImageDialog(available_keys, display_names, self.get_active_dataset_id(), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        result = dlg.get_result()
        selected_view_keys = result['selected_views']
        rot_inc = result['drot']
        base_name = result['base_name']
        
        if not selected_view_keys:
            QMessageBox.warning(self, "No Selection", "No views selected.")
            return
        
        # Map for filename friendly
        def key_to_short(k):
            return {
                "XY_Frame": "XY",
                "YZ_Frame": "YZ",
                "ZX_Frame": "ZX"
            }.get(k, k.replace("_Frame", ""))
        
        # Prepare directory & ensure last_import_dir is valid
        directory = ""
        if self.last_import_dir and os.path.isdir(self.last_import_dir):
            directory = self.last_import_dir
        if not directory:
            directory = os.getcwd()
        
        # Save original rotation
        orig_rx = self.rotation_widgets['X']['spin'].value()
        orig_ry = self.rotation_widgets['Y']['spin'].value()
        orig_rz = self.rotation_widgets['Z']['spin'].value()
        
        apply_rotation = any(abs(v) > 1e-6 for v in rot_inc.values())
        
        try:
            if apply_rotation:
                # Apply incremental rotation (add to current)
                self.rotation_widgets['X']['spin'].setValue(self.normalize_angle(orig_rx + rot_inc['x']))
                self.rotation_widgets['Y']['spin'].setValue(self.normalize_angle(orig_ry + rot_inc['y']))
                self.rotation_widgets['Z']['spin'].setValue(self.normalize_angle(orig_rz + rot_inc['z']))
                # Force apply transform & run simulation for required views
                self.apply_structure_rotation()
                self.simulate_views_blocking(selected_view_keys)
            
            # Export each selected view
            export_count = 0
            for key in selected_view_keys:
                if key not in self.simulation_results:
                    continue
                data = self.simulation_results[key]
                # Normalize to 8-bit grayscale
                mn, mx = float(np.min(data)), float(np.max(data))
                if mx <= mn:
                    norm = np.zeros_like(data, dtype=np.uint8)
                else:
                    norm = ((data - mn) / (mx - mn) * 255).astype(np.uint8)
                
                # Resize to 512x512
                try:
                    from PIL import Image
                except ImportError:
                    QMessageBox.critical(self, "Missing Pillow", "Install Pillow to export images (pip install Pillow).")
                    return
                img = Image.fromarray(norm, mode='L')
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS
                img = img.resize((512, 512), resample=resample_filter)
                
                fname = f"{base_name}_{key_to_short(key)}_dx{rot_inc['x']:+.0f}_dy{rot_inc['y']:+.0f}_dz{rot_inc['z']:+.0f}.png"
                save_path = os.path.join(directory, fname)
                try:
                    img.save(save_path)
                    export_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to save {save_path}: {e}")
            
            if export_count:
                QMessageBox.information(self, "Export Complete", f"Exported {export_count} image(s) to:\n{directory}")
            else:
                QMessageBox.warning(self, "No Export", "No images were exported.")
        
        finally:
            # Restore original rotation if we changed it
            if apply_rotation:
                self.rotation_widgets['X']['spin'].setValue(orig_rx)
                self.rotation_widgets['Y']['spin'].setValue(orig_ry)
                self.rotation_widgets['Z']['spin'].setValue(orig_rz)
                self.apply_structure_rotation()
                # (Optionally regenerate original visible views if needed)
                # self.simulate_views_blocking(available_keys)
    
    def run_simulation_on_view_change(self, is_checked):
        """
        View選択チェックボックスがONになった時にシミュレーションを自動実行するスロット。
        """

        # チェックがONにされた時、かつ「Interactive Update」が有効な場合のみ実行
        if is_checked and self.interactive_update_check.isChecked():
            self.run_simulation_interactively()

        # チェックがONにされた時のみ、かつPDBデータが読み込み済みの場合に実行
        if is_checked and self.atoms_data is not None:
            # 既に別のシミュレーションが実行中の場合は何もしない
            if self.simulate_btn.text() == "Cancel":
                print("Note: Another simulation is already running.")
                return
            
            #print("View selection changed, starting simulation automatically...")
            self.run_simulation()

    def handle_interactive_update_toggle(self, is_checked):
        """「Interactive Update」チェックボックスの状態変化を処理する"""
        if is_checked:
            # --- インタラクティブモードをONにする ---
            # 1. 現在の解像度設定を記憶（高解像度計算用）
            self.user_selected_resolution = self.resolution_combo.currentText()
            # 2. 解像度コンボボックスは有効のまま（ユーザーが変更可能）
            
            # ★★★ 追加：Interactive Update時はXY面のみ有効化 ★★★
            # YZ面とZX面のチェックボックスを無効化
            self.afm_y_check.setEnabled(False)
            self.afm_z_check.setEnabled(False)
            # YZ面とZX面のチェックを外す
            self.afm_y_check.setChecked(False)
            self.afm_z_check.setChecked(False)
            # XY面は有効のまま（既に有効）
            
            # ★★★ 修正：機能説明をユーザーに表示 ★★★
            QMessageBox.information(self, "Interactive Update Enabled", 
                                   f"Interactive Update enabled.\n\n"
                                   f"• During rotation: Low resolution (64x64) for real-time updates\n"
                                   f"• After rotation stops: High resolution (current setting) automatically generated\n"
                                   f"• Only XY view is available for real-time updates\n"
                                   f"• You can change resolution anytime")
            
            # 3. 現在の状態で一度シミュレーションを実行
            self.run_simulation_interactively()
        else:
            # --- インタラクティブモードをOFFにする ---
            # 解像度コンボボックスは既に有効（変更不要）
            # ★★★ 追加：全ての面のチェックボックスを有効化 ★★★
            self.afm_y_check.setEnabled(True)
            self.afm_z_check.setEnabled(True)
            
            # タイマーが動作中であれば停止
            if hasattr(self, 'high_res_timer'):
                self.high_res_timer.stop()

    def on_resolution_changed(self, new_resolution):
        """解像度変更時の処理"""
        # Interactive Updateが有効な場合、新しい解像度を高解像度として記憶
        if self.interactive_update_check.isChecked():
            self.user_selected_resolution = new_resolution
            #print(f"Resolution changed to {new_resolution} (will be used for high-res simulation)")
        
        # 通常のシミュレーションを実行
        self.trigger_interactive_simulation()

    def run_simulation_silent(self):
        """
        Interactive Update専用の軽量シミュレーション実行。
        UIの変更（ボタン、プログレスバー、ステータスなど）を行わない。
        """
        coords, mode = self.get_simulation_coords()
        if coords is None:
            return
        
        # 既に別のシミュレーションが実行中の場合は何もしない（改良版）
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker') or \
           self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            return

        base_coords = coords
        if base_coords is None:
            return

        # UIから共通パラメータを取得（ドラッグ中は常に64x64で計算）
        sim_params = {
            'scan_size': self.scan_size_spin.value(),
            'resolution': 64,  # ★★★ ドラッグ中は常に64x64で計算 ★★★
            'center_x': self.tip_x_slider.value() / 5.0,
            'center_y': self.tip_y_slider.value() / 5.0,
            'tip_radius': self.tip_radius_spin.value(),
            'minitip_radius': self.minitip_radius_spin.value(),
            'tip_angle': self.tip_angle_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'use_vdw': self.use_vdw_check.isChecked()
        }

        # チェックされた全ての面の計算タスクを作成
        tasks = []
        if self.afm_x_check.isChecked():
            tasks.append({
                "name": "XY",
                "panel": self.afm_x_frame,
                "coords": base_coords
            })
        if self.afm_y_check.isChecked():
            x_scan = base_coords[:, 1]
            y_scan = base_coords[:, 2]
            z_scan = -base_coords[:, 0]
            tasks.append({
                "name": "YZ",
                "panel": self.afm_y_frame,
                "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)
            })
        if self.afm_z_check.isChecked():
            x_scan, y_scan, z_scan = base_coords[:, 0], base_coords[:, 2], -base_coords[:, 1]
            tasks.append({
                "name": "ZX",
                "panel": self.afm_z_frame,
                "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)
            })

        if not tasks:
            return

        # 既存の軽量ワーカーを停止（より安全に）
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            try:
                self.stop_worker(self.sim_worker_silent, timeout_ms=50, allow_terminate=True, worker_name="sim_worker_silent")
            except Exception as e:
                print(f"[WARNING] Error stopping sim_worker_silent: {e}")
        
        # 軽量ワーカーを作成（UI変更なし）
        self.sim_worker_silent = AFMSimulationWorker(
            self, sim_params, tasks,
            self.atoms_data['element'] if sim_params['use_vdw'] and self.atoms_data is not None else None,
            self.vdw_radii if sim_params['use_vdw'] and hasattr(self, 'vdw_radii') else None,
            silent_mode=True
        )
        self._connect_worker_delete_later(self.sim_worker_silent)
        self._track_worker_ref('sim_worker_silent', self.sim_worker_silent)

        # 最小限の接続のみ（プログレス、ステータス、ボタン変更なし）
        self.sim_worker_silent.task_done.connect(self.on_task_finished_silent)
        self.sim_worker_silent.done.connect(self.on_simulation_finished_silent)
        self.sim_worker_silent.start()

    def run_simulation_immediate(self):
        """
        Interactive Update用の即座実行版シミュレーション。
        ドラッグ中に使用され、タイマー遅延なしで実行される。
        """
        # データが読み込まれていない場合は何もしない
        if self.atoms_data is None and not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            return
        
        # 以前のタイマーが作動中であれば停止する
        if hasattr(self, 'interactive_timer'):
            self.interactive_timer.stop()
        
        # ★★★ 軽量シミュレーションを実行（UI変更なし） ★★★
        self.run_simulation_silent()

    def run_simulation_immediate_controlled(self):
        """
        ドラッグ中専用の制御付き即座実行版シミュレーション。
        前のシミュレーションが完了するまで待機し、スレッドの蓄積を防ぐ。
        """
        # データが読み込まれていない場合は何もしない
        if self.atoms_data is None and not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            return
        
        # 以前のタイマーが作動中であれば停止する
        if hasattr(self, 'interactive_timer'):
            self.interactive_timer.stop()
        
        # ★★★ 前のシミュレーションが完了するまで待機 ★★★
        if self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
            # 前のシミュレーションが実行中の場合は、ドラッグ中の更新をスキップ
            return
        
        # ★★★ 最小更新間隔の制御を強化 ★★★
        current_time = QTime.currentTime()
        if hasattr(self, 'last_drag_simulation_time'):
            time_diff = self.last_drag_simulation_time.msecsTo(current_time)
            if time_diff < 300:  # 300ms未満の場合はスキップ（200msから増加）
                return
        
        self.last_drag_simulation_time = current_time
        
        # 軽量シミュレーションを実行（UI変更なし）
        self.run_simulation_silent()

    def safe_final_simulation_update(self):
        """ドラッグ終了後の安全な最終更新"""
        try:
            # 他のシミュレーションが実行中でなければ更新
            if not self.is_worker_running(getattr(self, 'sim_worker_silent', None), attr_name='sim_worker_silent'):
                self.run_simulation_silent()
        except Exception as e:
            print(f"[WARNING] Error in final simulation update: {e}")

    def schedule_high_res_simulation(self):
        """
        ドラッグ終了後、一定時間待ってから高解像度シミュレーションを実行する
        """
        # 既存のタイマーが動作中であれば停止
        if hasattr(self, 'high_res_timer'):
            self.high_res_timer.stop()
        
        # 新しいタイマーを設定（1秒後に実行）
        self.high_res_timer = QTimer(self)  # 親ウィンドウを設定
        self.high_res_timer.setSingleShot(True)
        self.high_res_timer.timeout.connect(self.run_high_res_simulation)
        self.high_res_timer.start(1000)  # 1秒待機

    def run_high_res_simulation(self):
        """
        一時的に高解像度でシミュレーションを実行し、その後64x64に戻す
        """
        if not self.interactive_update_check.isChecked():
            return
        
        # UI上の解像度表示は変更せず、内部で高解像度計算を実行
        if hasattr(self, 'user_selected_resolution') and self.user_selected_resolution:
            target_resolution = self.user_selected_resolution
        else:
            target_resolution = "256x256"  # デフォルト高解像度
   
        
        # 高解像度シミュレーションを実行（UI表示は変更しない）
        self.run_simulation_silent_high_res(target_resolution)

    def run_simulation_silent_high_res(self, target_resolution):
        """
        高解像度用の軽量シミュレーション（UI変更は最小限）
        """
        coords, mode = self.get_simulation_coords()
        if coords is None:
            return
        
        # 既に別のシミュレーションが実行中の場合は何もしない
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker'):
            return

        base_coords = coords
        if base_coords is None:
            return

        # UIから共通パラメータを取得（指定された高解像度で）
        sim_params = {
            'scan_size': self.scan_size_spin.value(),
            'resolution': int(target_resolution.split('x')[0]),  # ★★★ 指定された解像度を使用 ★★★
            'center_x': self.tip_x_slider.value() / 5.0,
            'center_y': self.tip_y_slider.value() / 5.0,
            'tip_radius': self.tip_radius_spin.value(),
            'minitip_radius': self.minitip_radius_spin.value(),
            'tip_angle': self.tip_angle_spin.value(),
            'tip_shape': self.tip_shape_combo.currentText().lower(),
            'use_vdw': self.use_vdw_check.isChecked()
        }

        # チェックされた全ての面の計算タスクを作成
        tasks = []
        if self.afm_x_check.isChecked():
            tasks.append({"name": "XY", "panel": self.afm_x_frame, "coords": base_coords})
        if self.afm_y_check.isChecked():
            x_scan = base_coords[:, 1]
            y_scan = base_coords[:, 2]
            z_scan = -base_coords[:, 0]
            tasks.append({"name": "YZ", "panel": self.afm_y_frame, "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)})
        if self.afm_z_check.isChecked():
            x_scan, y_scan, z_scan = base_coords[:, 0], base_coords[:, 2], -base_coords[:, 1]
            tasks.append({"name": "ZX", "panel": self.afm_z_frame, "coords": np.stack((x_scan, y_scan, z_scan), axis=-1)})

        if not tasks:
            return

        # 既存の高解像度ワーカーを停止
        if self.is_worker_running(getattr(self, 'sim_worker_high_res', None), attr_name='sim_worker_high_res'):
            self.stop_worker(self.sim_worker_high_res, timeout_ms=300, allow_terminate=False, worker_name="sim_worker_high_res")
        
        # 高解像度ワーカーを作成
        self.sim_worker_high_res = AFMSimulationWorker(
            self, sim_params, tasks,
            self.atoms_data['element'] if sim_params['use_vdw'] and self.atoms_data is not None else None,
            self.vdw_radii if sim_params['use_vdw'] and hasattr(self, 'vdw_radii') else None,
            silent_mode=True
        )
        self._connect_worker_delete_later(self.sim_worker_high_res)
        self._track_worker_ref('sim_worker_high_res', self.sim_worker_high_res)

        # 完了時に解像度を戻すための特別なハンドラーを接続
        self.sim_worker_high_res.task_done.connect(self.on_task_finished_silent)
        self.sim_worker_high_res.done.connect(self.on_high_res_simulation_finished)
        self.sim_worker_high_res.start()

    def on_high_res_simulation_finished(self, result):
        """高解像度シミュレーション完了処理"""
        # 通常の完了処理
        if self.simulation_results:
            self.save_image_button.setEnabled(True)
            self.save_asd_button.setEnabled(True)
        
        # UI上の解像度表示は変更しない（既に正しい解像度が表示されている）
       

    def on_task_finished_silent(self, z_map, target_panel):
        """軽量シミュレーション用のタスク完了処理（UI変更最小限）"""
        if z_map is not None and target_panel is not None:
            image_key = target_panel.objectName()
            
            # 生データを保存し、表示更新関数を呼び出す
            self.raw_simulation_results[image_key] = z_map
            self.process_and_display_single_image(image_key)

    def on_simulation_finished_silent(self, result):
        """軽量シミュレーション用の完了処理（UI変更なし）"""
        # ★★★ ボタンやプログレスバーの変更は行わない ★★★
        # ★★★ 保存ボタンの有効化のみ行う ★★★
        if self.simulation_results:
            self.save_image_button.setEnabled(True)
            self.save_asd_button.setEnabled(True)

    def run_simulation_interactively(self):
        """
        インタラクティブモード用のシミュレーション実行関数。
        スライダー操作中に連続で実行されないよう、タイマーで遅延させる。
        """
        # PDBデータまたはMRCデータが読み込まれていない、または他のシミュレーションが実行中の場合は何もしない
        if self.atoms_data is None and not (hasattr(self, 'mrc_data') and self.mrc_data is not None):
            return
        if self.is_worker_running(getattr(self, 'sim_worker', None), attr_name='sim_worker'):
            return
        
        # 以前のタイマーが作動中であれば停止する
        if hasattr(self, 'interactive_timer'):
            self.interactive_timer.stop()
        
        # 新しいタイマーを設定
        self.interactive_timer = QTimer(self)  # 親ウィンドウを設定
        self.interactive_timer.setSingleShot(True)  # 一度だけ実行
        # 300ミリ秒後にrun_simulationを呼び出す
        self.interactive_timer.timeout.connect(self.run_simulation)
        self.interactive_timer.start(300)

    def cleanup_threads(self):
        """実行中のスレッドを適切にクリーンアップする（完全版）"""
        try:
            print("Starting thread cleanup...")
            
            # スレッドのリストを作成
            workers = []
            if hasattr(self, 'sim_worker') and self.sim_worker:
                workers.append(('sim_worker', self.sim_worker))
            if hasattr(self, 'sim_worker_silent') and self.sim_worker_silent:
                workers.append(('sim_worker_silent', self.sim_worker_silent))
            if hasattr(self, 'sim_worker_high_res') and self.sim_worker_high_res:
                workers.append(('sim_worker_high_res', self.sim_worker_high_res))
            
            # 各ワーカーを停止
            for worker_name, worker in workers:
                try:
                    print(f"Stopping {worker_name}...")
                    stopped = self.stop_worker(worker, timeout_ms=300, allow_terminate=True, worker_name=worker_name)
                    if stopped:
                        print(f"Stopped {worker_name} gracefully")
                    else:
                        print(f"[WARNING] {worker_name} may still be running")
                        
                except Exception as e:
                    print(f"[WARNING] Error stopping {worker_name}: {e}")
            
            print("Thread cleanup completed")
                
        except Exception as e:
            print(f"[WARNING] Error during thread cleanup: {e}")

    def closeEvent(self, event):
        """ウィンドウが閉じられるときに自動的に呼び出される"""
        try:
            # ★★★ 全タイマーの停止を最初に実行 ★★★
            print("Stopping all timers...")
            timer_attrs = ['rotation_update_timer', 'filter_update_timer', 'interactive_timer', 'high_res_timer']
            for timer_attr in timer_attrs:
                if hasattr(self, timer_attr):
                    timer = getattr(self, timer_attr)
                    if timer:
                        try:
                            timer.stop()
                            timer.deleteLater()  # タイマーを完全に削除
                        except Exception as e:
                            print(f"[WARNING] Failed to stop {timer_attr}: {e}")
            
            # ★★★ スレッドの適切なクリーンアップ（同期的に実行） ★★★
            self.cleanup_threads()
            
            # ヘルプウィンドウを閉じる
            if hasattr(self, 'help_window') and self.help_window:
                try:
                    self.help_window.close()
                except RuntimeError:
                    print("[WARNING] Help window C++ object already deleted")
                except Exception as e:
                    print(f"[WARNING] Failed to close help window: {e}")
            
            # スタンドアロンアプリケーションなのでwindow_managerは使用しない
            
            # ウィンドウの位置とサイズを保存
            try:
                self.save_geometry()
            except Exception as e:
                print(f"[WARNING] Failed to save geometry: {e}")
            
            # 設定を保存
            try:
                self.save_settings()
            except Exception as e:
                print(f"[WARNING] Failed to save settings: {e}")
            
            # ツールバーアクションのハイライトを解除（プラグインとして開かれた場合／pyNuDから開かれた場合）
            try:
                if self.main_window is not None and hasattr(self.main_window, 'plugin_actions'):
                    action = self.main_window.plugin_actions.get(PLUGIN_NAME)
                    if action is not None and hasattr(self.main_window, 'setActionHighlight'):
                        self.main_window.setActionHighlight(action, False)
                else:
                    import globalvals as gv
                    if hasattr(gv, 'main_window') and gv.main_window:
                        if hasattr(gv.main_window, 'setActionHighlight') and hasattr(gv.main_window, 'simulator_action'):
                            gv.main_window.setActionHighlight(gv.main_window.simulator_action, False)
            except Exception:
                pass  # スタンドアロン起動の場合は無視
            
            # Qtのデフォルトのクローズ処理
            try:
                super().closeEvent(event)
            except RuntimeError:
                print("[WARNING] C++ object already deleted during super().closeEvent()")
            except Exception as e:
                print(f"[WARNING] Failed to call super().closeEvent(): {e}")
            
            event.accept()
            
        except Exception as e:
            print(f"[ERROR] Unexpected error in AFMSimulator closeEvent: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生してもイベントは受け入れる
            event.accept()

    def _import_mrc_internal(self, file_path):
        """MRCファイルの読み込み（内部メソッド）"""
        # 必要なライブラリのインポート
        import mrcfile
        from vtk.util import numpy_support
        
        # PDBデータをクリア（MRCファイルimport時）
        self.clear_pdb_data()

        # 2. MRCファイル読み込みとボクセルサイズのスケール変換
        with mrcfile.open(file_path, permissive=True) as mrc:
            # 元のデータを保存
            self.mrc_data_original = mrc.data.copy()
            # デフォルトでZ flipを適用（読み込み時にFlipさせて管理）
            self.mrc_data = np.flip(self.mrc_data_original, axis=0).copy()
            
            if mrc.voxel_size.x:
                voxel_size_angstrom = mrc.voxel_size.x 
            else:
                voxel_size_angstrom = 1.0
            self.mrc_voxel_size_nm = voxel_size_angstrom / 10.0
            
        # MRCファイル名を表示
        self.mrc_name = os.path.basename(file_path)
        self.mrc_id = ""
        self.mrc_id = os.path.splitext(self.mrc_name)[0]
        self.file_label.setText(f"File Name: {self.mrc_name} (MRC)")
        
        self.mrc_group.setEnabled(True)
        # Z flipの状態に応じてmrc_surface_coordsを初期化
        self.mrc_surface_coords = self._get_mrc_surface_coords()
        self.update_mrc_display()
        self.simulate_btn.setEnabled(True)
        
        # 回転ウィジェットも有効化
        if hasattr(self, 'rotation_widgets'):
            for axis in ['X', 'Y', 'Z']:
                self.rotation_widgets[axis]['spin'].setEnabled(True)
                self.rotation_widgets[axis]['slider'].setEnabled(True)
        
        # チェックボックスの状態を確実に設定（デフォルトでTrue）
        if hasattr(self, 'mrc_z_flip_check'):
            self.mrc_z_flip_check.blockSignals(True)
            self.mrc_z_flip_check.setChecked(True)
            self.mrc_z_flip_check.blockSignals(False)
            self.mrc_z_flip = True
        
        # 回転状態をリセット（MRCファイル読み込み時）
        self.reset_structure_rotation()
        
        # Interactive Updateが有効な場合は初期シミュレーションを実行
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            self.run_simulation_interactively()

    def on_mrc_threshold_changed(self, value):
        """スライダーの値が変更されたときに呼ばれる（リアルタイム更新用）"""
        # ラベルを更新
        self.mrc_threshold_label.setText(f"Value: {value/100.0:.2f}")
        
        # Interactive Updateが有効な場合は疑似AFM像を自動更新
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            self.mrc_threshold = value / 100.0
            self.run_simulation_interactively()
    
    def on_mrc_threshold_released(self):
        """スライダーが離されたときに呼ばれ、しきい値を更新して再描画する"""
        self.mrc_threshold = self.mrc_threshold_slider.value() / 100.0
        self.update_mrc_display()
        
        # Interactive Updateが有効な場合は疑似AFM像も自動更新
        if hasattr(self, 'interactive_update_check') and self.interactive_update_check.isChecked():
            self.run_simulation_interactively()

    def on_mrc_z_flip_changed(self, state):
        """Z軸フリップチェックボックスの状態変更時の処理"""
        self.mrc_z_flip = state == Qt.Checked
        
        # mrc_data_originalが存在しない場合は、現在のmrc_dataを元データとして使用
        if not hasattr(self, 'mrc_data_original') or self.mrc_data_original is None:
            if hasattr(self, 'mrc_data') and self.mrc_data is not None:
                # 現在のデータを元データとして保存（初回のみ）
                self.mrc_data_original = self.mrc_data.copy()
            else:
                return
        
        if self.mrc_data_original is not None:
            # フリップ状態変更時に回転状態をリセット（ジャンプを防ぐ）
            self.reset_structure_rotation()
            
            # チェック時：フリップ済みデータを使用（現在の状態を維持）
            # アンチェック時：元のデータを使用（元の向きに戻す）
            if self.mrc_z_flip:
                # チェック時：フリップ済みデータ（読み込み時のデフォルト状態）
                self.mrc_data = np.flip(self.mrc_data_original, axis=0).copy()
            else:
                # アンチェック時：元のデータ（元の向きに戻す）
                self.mrc_data = self.mrc_data_original.copy()
            
            # 座標データを再生成
            self.mrc_surface_coords = self._get_mrc_surface_coords()
            self.update_mrc_display()

    def _get_mrc_surface_coords(self):
        """MRCデータから表面座標を取得する"""
        if not hasattr(self, 'mrc_data') or self.mrc_data is None:
            return None
        
        from vtk.util import numpy_support

        # 現在のフリップ状態に応じたデータを使用
        mask = (self.mrc_data > self.mrc_threshold).astype(np.uint8)

        vtk_data = vtk.vtkImageData()
        depth, height, width = mask.shape
        vtk_data.SetDimensions(width, height, depth)
        vtk_data.SetSpacing(self.mrc_voxel_size_nm, self.mrc_voxel_size_nm, self.mrc_voxel_size_nm)
        vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        flat = mask.transpose(2, 1, 0).flatten()
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_data.GetPointData().SetScalars(vtk_array)

        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_data)
        contour.SetValue(0, 0.5)
        contour.Update()

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(50)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputConnection(smoother.GetOutputPort())
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        center = centerOfMassFilter.GetCenter()

        transform = vtk.vtkTransform()
        transform.Translate(-center[0], -center[1], -center[2])

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(smoother.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        polydata = transformFilter.GetOutput()
        points = polydata.GetPoints()
        if points is not None:
            coords = numpy_support.vtk_to_numpy(points.GetData())  # shape: (N, 3)
            # ジッターを加える
            np.random.seed(42)
            jitter_amplitude = 0.01  # 0.01nmの範囲でジッター
            jitter = np.random.uniform(low=-jitter_amplitude, high=jitter_amplitude, size=coords.shape)
            coords_jittered = coords + jitter
            return coords_jittered
        else:
            return None

    def update_mrc_display(self):
        """現在のしきい値でMRCデータを3D表示する"""
        if self.mrc_data is None:
            return

        from vtk.util import numpy_support

        # 現在のフリップ状態に応じたデータを使用
        mask = (self.mrc_data > self.mrc_threshold).astype(np.uint8)

        vtk_data = vtk.vtkImageData()
        depth, height, width = mask.shape
        vtk_data.SetDimensions(width, height, depth)
        vtk_data.SetSpacing(self.mrc_voxel_size_nm, self.mrc_voxel_size_nm, self.mrc_voxel_size_nm)
        vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        flat = mask.transpose(2, 1, 0).flatten()
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_data.GetPointData().SetScalars(vtk_array)

        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_data)
        contour.SetValue(0, 0.5)
        contour.Update()

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(50)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()

        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputConnection(smoother.GetOutputPort())
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        center = centerOfMassFilter.GetCenter()

        transform = vtk.vtkTransform()
        transform.Translate(-center[0], -center[1], -center[2])

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(smoother.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transformFilter.GetOutputPort())
        # スカラーデータの色マッピングを無効にして、アクターの色を使用
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        # MRCは常に選択された色を使用（カラースキームは関係ない）
        #print(f"Setting MRC color to: {self.current_single_color}")
        prop.SetColor(self.current_single_color[0], self.current_single_color[1], self.current_single_color[2])
        #rint(f"Actual MRC color set: {prop.GetColor()}")
        prop.SetOpacity(1.0)
        prop.SetAmbient(0.2)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.4)
        prop.SetSpecularPower(30.0)

        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            self.renderer.RemoveActor(self.mrc_actor)
        self.mrc_actor = actor
        self.renderer.AddActor(actor)
        # 新しいアクターにも現在の回転を適用
        if hasattr(self, 'mrc_actor') and self.mrc_actor is not None:
            self.mrc_actor.SetUserTransform(self.molecule_transform)
        # カメラ視点をリセットしない（ResetCamera()を削除）
        self.vtk_widget.GetRenderWindow().Render()

        polydata = transformFilter.GetOutput()
        points = polydata.GetPoints()
        if points is not None:
            coords = numpy_support.vtk_to_numpy(points.GetData())  # shape: (N, 3)
            # ★★★ ジッターを加える ★★★
            np.random.seed(42)
            jitter_amplitude = 0.01  # 0.01nmの範囲でジッター
            jitter = np.random.uniform(low=-jitter_amplitude, high=jitter_amplitude, size=coords.shape)
            coords_jittered = coords + jitter
            self.mrc_surface_coords = coords_jittered
        else:
            self.mrc_surface_coords = None

    def get_simulation_coords(self):
        if hasattr(self, 'mrc_surface_coords') and self.mrc_surface_coords is not None:
            # 回転行列をnumpy配列に変換（combined_transformを使用）
            transform = self.combined_transform.GetMatrix()
            mat = np.array([[transform.GetElement(i, j) for j in range(4)] for i in range(4)])
            coords = self.mrc_surface_coords
            # 同次座標に変換
            coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
            coords_rot = (mat @ coords_h.T).T[:, :3]
            return coords_rot, 'mrc'
        elif self.atoms_data is not None:
            coords = self.get_rotated_atom_coords()
            return coords, 'pdb'
        else:
            return None, None
    
    def get_active_dataset_id(self):
        """
        Return an identifier for current dataset (PDB or MRC).
        優先順位: PDB > MRC > AFM
        """
        if getattr(self, 'pdb_id', ''):
            return self.pdb_id
        if getattr(self, 'mrc_id', ''):
            return self.mrc_id
        return "Unknown"

    def get_active_dataset_type(self):
        if getattr(self, 'pdb_id', ''):
            return "PDB"
        if getattr(self, 'mrc_id', ''):
            return "MRC"
        return "Unknown"   
    
class SaveAFMImageDialog(QDialog):
    """
    Custom dialog to select multiple AFM views and specify incremental rotations
    for export.
    """
    def __init__(self, available_keys, display_names, dataset_id, parent=None):
        super().__init__(parent)
        self.available_keys = available_keys
        self.display_names = display_names
        self.dataset_id = dataset_id
        self.setWindowTitle("Export Simulated AFM Images")
        self.setModal(True)
        self.setMinimumWidth(420)
        self._result = None
        self._build_ui()
    
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)
        
        # Views group
        views_group = QGroupBox("Select Views to Export")
        vg = QVBoxLayout(views_group)
        self.view_checks = {}
        for key in self.available_keys:
            cb = QCheckBox(self.display_names.get(key, key))
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_ok_state)
            self.view_checks[key] = cb
            vg.addWidget(cb)
        if not self.available_keys:
            note = QLabel("No simulated images are available.")
            note.setStyleSheet("color:#b00; font-style:italic;")
            vg.addWidget(note)
        # Select all / none buttons
        btn_row = QHBoxLayout()
        sel_all = QPushButton("All")
        sel_all.clicked.connect(lambda: self._set_all(True))
        sel_none = QPushButton("None")
        sel_none.clicked.connect(lambda: self._set_all(False))
        btn_row.addStretch()
        btn_row.addWidget(sel_all)
        btn_row.addWidget(sel_none)
        vg.addLayout(btn_row)
        main.addWidget(views_group)
        
        # Rotation increments
        rot_group = QGroupBox("Incremental Rotation (°)  (applied once before export)")
        rg = QGridLayout(rot_group)
        self.dx_spin = QDoubleSpinBox(); self._init_rot_spin(self.dx_spin)
        self.dy_spin = QDoubleSpinBox(); self._init_rot_spin(self.dy_spin)
        self.dz_spin = QDoubleSpinBox(); self._init_rot_spin(self.dz_spin)
        rg.addWidget(QLabel("ΔX:"), 0, 0); rg.addWidget(self.dx_spin, 0, 1)
        rg.addWidget(QLabel("ΔY:"), 0, 2); rg.addWidget(self.dy_spin, 0, 3)
        rg.addWidget(QLabel("ΔZ:"), 0, 4); rg.addWidget(self.dz_spin, 0, 5)
        main.addWidget(rot_group)
        
        # Base filename
        base_group = QGroupBox("Filename Base")
        bg = QHBoxLayout(base_group)
        bg.addWidget(QLabel("Base:"))
        self.base_edit = QLineEdit(self.dataset_id if self.dataset_id else "AFM")
        self.base_edit.setPlaceholderText("Base name (dataset id)")
        bg.addWidget(self.base_edit)
        main.addWidget(base_group)
        
        # Example label
        self.example_label = QLabel()
        self.example_label.setStyleSheet("color:#555; font-size:11px;")
        main.addWidget(self.example_label)
        self._update_example()
        for sp in (self.dx_spin, self.dy_spin, self.dz_spin, self.base_edit):
            if isinstance(sp, QDoubleSpinBox):
                sp.valueChanged.connect(self._update_example)
            else:
                sp.textChanged.connect(self._update_example)
        
        # Buttons
        btns = QHBoxLayout()
        btns.addStretch()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.ok_btn)
        btns.addWidget(cancel_btn)
        main.addLayout(btns)
        
        self._update_ok_state()
    
    def _init_rot_spin(self, spin):
        spin.setRange(-360.0, 360.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setValue(0.0)
        spin.setKeyboardTracking(False)
    
    def _set_all(self, state):
        for cb in self.view_checks.values():
            cb.setChecked(state)
        self._update_ok_state()
    
    def _update_ok_state(self):
        any_checked = any(cb.isChecked() for cb in self.view_checks.values())
        self.ok_btn.setEnabled(any_checked)
    
    def _update_example(self):
        base = self.base_edit.text().strip() or "AFM"
        dx = self.dx_spin.value(); dy = self.dy_spin.value(); dz = self.dz_spin.value()
        example = f"Example filename: {base}_XY_dx{dx:+.0f}_dy{dy:+.0f}_dz{dz:+.0f}.png"
        self.example_label.setText(example)
    
    def get_result(self):
        selected = [k for k, cb in self.view_checks.items() if cb.isChecked()]
        return {
            'selected_views': selected,
            'drot': {'x': self.dx_spin.value(), 'y': self.dy_spin.value(), 'z': self.dz_spin.value()},
            'base_name': self.base_edit.text().strip() or "AFM"
        }


def main():
    # アプリケーション作成前にHighDPI設定
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # VTKのエラー出力を抑制
    vtk.vtkObject.GlobalWarningDisplayOff()
    
    window = AFMSimulator()
    window.show()
    
    sys.exit(app.exec_())


def create_plugin(main_window):
    """Plugin entry point. Called from pyNuD Plugin menu."""
    return AFMSimulator(main_window=main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "AFMSimulator"]


if __name__ == "__main__":
    main()
