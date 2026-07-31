# loc_afm.py (完全版)
# loc_afm.py (ASD保存機能に対応した最終完成版)
#
# =============================================================================
# LICENSING NOTE  --  READ BEFORE DISTRIBUTING
# -----------------------------------------------------------------------------
# The "Heath mode" code paths in this file (filter_movie pre-filter, Fast_peaks2D
# detection semantics, Gaussian/sphere sub-pixel localization, localization-density
# rendering, and the FRC resolution measurement) are DERIVED FROM:
#
#     George-R-Heath/NanoLocz-Matlab-Library   (MATLAB, GPL-3.0)
#     https://github.com/George-R-Heath/NanoLocz-Matlab-Library
#     Heath et al., Localization atomic force microscopy, Nature 594, 385-390 (2021)
#     Heath, Micklethwaite & Storer, NanoLocz, Small Methods 2024, 2301766
#
# A translation into another language is a derivative work. GPL-3.0 is copyleft,
# so DISTRIBUTING pyNuD with these code paths requires pyNuD itself to be
# GPL-3.0-licensed (pyNuD currently ships no LICENSE file). Local/in-house use
# triggers no distribution obligation. Resolve the licence before the next
# release, or gate these paths out of the build.
#
# Functions marked "[Heath]" below are the derived ones. Everything else is the
# original pyNuD implementation and is unaffected.
# =============================================================================

import sys
import time
import os # <<< osモジュールをインポート
import json
import datetime
import struct
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy.ndimage import maximum_filter, gaussian_filter, zoom, rotate, shift
import cv2
import tifffile

# ▼▼▼【重要修正点】fileioからSaveASDをインポート ▼▼▼
from fileio import SaveASD
from helperFunctions import get_z_unit

try:
    import globalvals as gv
except ImportError:
    class GVDummy: pass
    gv = GVDummy()

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError as e:
    PYVISTA_AVAILABLE = False
    PV_IMPORT_ERROR = str(e)
else:
    PV_IMPORT_ERROR = None

from skimage.registration import phase_cross_correlation

PLUGIN_NAME = "L-AFM Analysis"

HELP_HTML_EN = """
<h1>L-AFM Analysis (Localization AFM)</h1>

<h2>Overview</h2>
<p>Localization Atomic Force Microscopy (L-AFM) builds a super-resolution image by detecting brightness peaks across an AFM time series (movie) and reconstructing those localizations onto a finer grid. This panel runs the workflow step by step: peak detection → reconstruction → final image.</p>
<p><strong>Algorithm basis:</strong> Peak localization / reconstruction follows the Localization AFM approach reported by Heath, Scheuring, and colleagues (<i>Nature</i> 594, 385–390, 2021; DOI: 10.1038/s41586-021-03551-x). Optional Heath / NanoLocz-compatible paths are available (see below); defaults keep the original pyNuD behaviour.</p>

<h2>Access</h2>
<ul>
    <li><strong>Plugin menu:</strong> Load Plugin… → select <code>plugins/LAFMAnalysis.py</code>, then Plugin → L-AFM Analysis</li>
    <li><strong>Manual:</strong> Help → Manual in this panel (日本語 / English)</li>
</ul>

<h2>Processing Steps</h2>
<p>Run the numbered buttons in order. Optional buttons: <b>Load</b> (parameters), <b>Measure Resolution (FRC)</b>.</p>
<div class="step">
    <strong>1. Preprocessing 1 — Peak detection</strong><br>
    Optionally applies Drift Correction, then scans each frame for local maxima that pass Peak Filtering / Local Maxima / Subpixel settings. Output: localization list (coordinates, intensity, frame index, …).
</div>
<div class="step">
    <strong>2. Preprocessing 2 — Reconstruction</strong><br>
    Plots localizations from Step 1 onto a high-resolution 2D grid or 3D voxel volume (does <b>not</b> re-read the movie pixels). Output: sparse density / probability map.
</div>
<div class="step">
    <strong>3. Make LAFM Image — Final image</strong><br>
    Applies Gaussian blur (and optional final-stage Symmetric Averaging) to finish a smooth LAFM image.
</div>
<div class="step">
    <strong>Save / Load</strong><br>
    <b>Save</b>: 2D → ASD (+ sidecar <code>*_params.json</code>); 3D → TIFF (+ JSON). Comments embed processing / LAFM parameters.<br>
    <b>Load</b>: restore panel parameters from a JSON file (does not reload image data).
</div>
<div class="step">
    <strong>Measure Resolution (FRC)</strong><br>
    After Preprocessing 1, splits localizations into two random half-datasets and reports Fourier ring correlation resolution (1/7 criterion). Measures reproducibility of <i>this</i> movie, not absolute accuracy.
</div>

<h2>Preprocessing 1 vs 2</h2>
<div class="feature-box">
    <h4>Preprocessing 1 — Detection</h4>
    <ul>
        <li><strong>Input:</strong> AFM movie stack</li>
        <li><strong>Role:</strong> Build the “address book” of peak locations</li>
        <li><strong>Cost:</strong> Usually the slowest step</li>
    </ul>
</div>
<div class="feature-box">
    <h4>Preprocessing 2 — Drawing</h4>
    <ul>
        <li><strong>Input:</strong> Peak list from Step 1 only</li>
        <li><strong>Role:</strong> Paint localizations onto a finer canvas</li>
        <li><strong>Tip:</strong> If only reconstruction settings change, re-run from Step 2</li>
    </ul>
</div>

<h2>Parameter Groups (UI order)</h2>

<h3>Mode</h3>
<ul>
    <li><b>Mode:</b> <code>2D</code> or <code>3D</code> reconstruction</li>
    <li><b>3D Display:</b> open / update the PyVista 3D viewer when available</li>
</ul>

<h3>Drift Correction</h3>
<p>Checkable group; <b>default OFF</b>. When ON, runs inside Preprocessing 1 <i>before</i> peak detection.</p>
<table class="param-table">
<tr><th>Control</th><th>Meaning</th></tr>
<tr><td><b>Algorithm</b></td>
<td><i>Phase Correlation (Fast)</i>: translation alignment. <i>Feature-based (Precise)</i>: slower, finer feature matching.</td></tr>
<tr><td><b>Subpixel Precision</b></td>
<td>Upsampling factor for phase-correlation fine alignment (higher → finer, slower).</td></tr>
<tr><td><b>Min Confidence</b></td>
<td>Keep only frames with alignment confidence <b>strictly greater than</b> this value (0–1). Frames at or below the threshold are <b>excluded</b> from detection. If fewer than two frames remain, processing stops with an error.</td></tr>
</table>
<div class="note">
Use Drift Correction when residual frame-to-frame shift remains after upstream tracking / averaging. Start with Phase Correlation; raise Min Confidence only if many poorly aligned frames pollute the peak list.
</div>

<h3>Peak Filtering</h3>
<table class="param-table">
<tr><th>Control</th><th>Meaning</th></tr>
<tr><td><b>Filter Mode</b></td>
<td><i>Absolute Height (nm)</i>: keep peaks between Z_min and Z_max. <i>Statistics (Mean + N × Std Dev)</i>: threshold from mean/std and N factor.</td></tr>
<tr><td><b>N factor</b></td>
<td>Used in Statistics mode (shown when that mode is selected).</td></tr>
<tr><td><b>Enable ImageJ-compatible ROI mask</b></td>
<td>Per-frame Otsu ROI mask plus ImageJ-compatible tolerance / normalization (for matching ImageJ LAFM workflows).</td></tr>
<tr><td><b>Rendering mode</b></td>
<td><i>pyNuD (probability × height)</i> vs <i>Heath (localization density)</i>. Different physical quantities — do not compare absolute values. Details in Heath section.</td></tr>
<tr><td><b>Auto Z-Range / Sample</b></td>
<td>Suggest Z_min / Z_max from stack statistics or sample-type presets (General, Proteins, DNA/RNA, Cells, Crystals, Nanoparticles).</td></tr>
<tr><td><b>Z_min / Z_max (nm)</b></td>
<td>Absolute height window for peak acceptance (nm). With Pre-filter ON, nm limits still apply to the <b>unfiltered</b> data.</td></tr>
<tr><td><b>Crop Ratio</b></td>
<td>Radial crop: keep peaks inside a circle of radius <code>(min(W,H)/2) × Crop Ratio</code> (default 0.9). Reduces edge artefacts.</td></tr>
</table>

<h3>Pre-filter [Heath filter_movie]</h3>
<p>Checkable; <b>default OFF</b>. Applies Heath-style Gaussian + Laplacian filtering before detection; Detection threshold (0–1) acts on the rescaled filtered stack.</p>
<div class="note">
On real HS-AFM data, start Laplacian strength at <b>0</b> (workbook value 50 often amplifies raster noise). Reject spike frames first — <code>rescale()</code> normalises over the whole stack.
</div>

<h3>Local Maxima</h3>
<ul>
    <li><b>Search Size (n×n):</b> neighbourhood for local-max detection (odd sizes)</li>
    <li><b>Connectivity:</b> 4 or 8 neighbourhood</li>
</ul>

<h3>Subpixel Localization</h3>
<p>Checkable group. When ON, refines peak positions beyond integer pixels.</p>
<table class="param-table">
<tr><th>Control</th><th>Meaning</th></tr>
<tr><td><b>Method</b></td>
<td><i>Interpolation (pyNuD)</i>, <i>Heath bicubic</i>, <i>Gaussian fit [Heath]</i>, <i>Sphere fit [Heath]</i>. Prefer Gaussian fit on noisy experimental data (interpolation methods can pixel-lock).</td></tr>
<tr><td><b>Scale</b></td>
<td>Interpolation / ROI zoom factor used during subpixel refinement.</td></tr>
<tr><td><b>Expand</b></td>
<td>Final reconstruction grid expansion (1 = same pixel count; 2 = 2× width and height, …).</td></tr>
<tr><td><b>XY / Z Resolution</b></td>
<td>Physical voxel size of the reconstruction grid (nm). Used mainly for 3D / reporting.</td></tr>
</table>

<h3>Centring</h3>
<p>Defines the centre used by rotational Symmetric Averaging (independent of the LAFM density accumulation itself).</p>
<ul>
    <li><b>Off:</b> rotate about the array centre</li>
    <li><b>Centre of mass:</b> intensity-weighted centroid (fold-independent)</li>
    <li><b>Symmetry axis (C<sub>n</sub>) [Heath]:</b> FindCenterPositions-style; <b>requires Symmetry Order</b></li>
</ul>
<p><b>Found offset</b> shows the measured shift (also written into save comments when applicable).</p>

<h3>Symmetric Averaging</h3>
<p>Checkable; default OFF. C<sub>n</sub> rotational averaging.</p>
<ul>
    <li><b>During Reconstruction (Prep 2)</b> and/or <b>On Final LAFM Image</b></li>
    <li><b>Symmetry Order:</b> n for C<sub>n</sub> (1 = no symmetry)</li>
</ul>
<p>Heath / NanoLocz does <b>not</b> symmetrise the LAFM map itself; output symmetrisation is a pyNuD addition. See Heath section for interpolation details.</p>

<h3>Gaussian Blur</h3>
<ul>
    <li><b>Sigma (xy) [pixels]</b> / <b>Sigma (z) [voxels]</b> — smoothing for Make LAFM Image</li>
</ul>

<h3>Visualization / Results</h3>
<ul>
    <li><b>Update Delay (ms):</b> throttle live preview updates during processing</li>
    <li><b>Total Detections / Reconstruction Size / FRC resolution:</b> status after each stage</li>
</ul>

<h2>Which Step to Re-run After Changing Parameters</h2>
<h3>Re-run from Preprocessing 1</h3>
<ul>
    <li>Drift Correction (all)</li>
    <li>Peak Filtering (including ImageJ mask, Rendering mode, Z-range, Crop Ratio)</li>
    <li>Pre-filter [Heath]</li>
    <li>Local Maxima</li>
    <li>Subpixel Localization: enable / Method / Scale</li>
</ul>
<h3>Re-run from Preprocessing 2</h3>
<ul>
    <li>Mode (2D ↔ 3D)</li>
    <li>Subpixel Expand, XY / Z Resolution</li>
    <li>Centring (if used with Prep-2 symmetrisation)</li>
    <li>Symmetric Averaging → During Reconstruction (Prep 2)</li>
</ul>
<h3>Re-run from Make LAFM Image</h3>
<ul>
    <li>Gaussian Blur</li>
    <li>Symmetric Averaging → On Final LAFM Image</li>
    <li>Centring (if only final-stage symmetrisation uses it)</li>
</ul>

<h2>Practical Workflow</h2>
<pre><code>
graph TD
    A[Open L-AFM Panel] --> B[Set parameters / optional Load JSON]
    B --> C[1. Preprocessing 1]
    C --> D{Peaks OK?}
    D -- No --> E[Adjust Drift / Peak Filtering / Local Maxima / Subpixel]
    E --> C
    D -- Yes --> F[Optional: Measure Resolution FRC]
    F --> G[2. Preprocessing 2]
    G --> H{Reconstruction OK?}
    H -- No --> I[Adjust Mode / Expand / Prep-2 Symmetry]
    I --> G
    H -- Yes --> J[3. Make LAFM Image]
    J --> K{Final image OK?}
    K -- No --> L[Adjust Blur / Final Symmetry]
    L --> J
    K -- Yes --> M[Save ASD/TIFF + params JSON]
</code></pre>

<hr>
<h2>Heath / NanoLocz Compatibility Options</h2>
<p>These reproduce parts of
<a href="https://github.com/George-R-Heath/NanoLocz-Matlab-Library">NanoLocz</a>
(Heath et al., Nature 2021). They are <b>off by default</b> (except Centring’s default method label); older parameter JSON files still load.</p>
<table class="param-table">
<tr><th>Option</th><th>What it changes</th></tr>
<tr><td><b>Rendering mode</b></td>
<td><i>pyNuD</i>: per-frame gaussian(peaks)×(height−min), then average. <i>Heath</i>: pooled localization <b>density</b>; height mainly via colour-level binning.</td></tr>
<tr><td><b>Subpixel method</b></td>
<td>Interpolation / Heath bicubic / Gaussian fit / Sphere fit (see Subpixel Localization above).</td></tr>
<tr><td><b>Pre-filter</b></td>
<td><code>filter_movie(im,'Gaussian',…,'Laplacian',…)</code> then threshold on rescale()d 0–1 data.</td></tr>
<tr><td><b>FRC</b></td>
<td>Half-dataset Fourier ring correlation (1/7), expand=5, img_gaus=0.4 as in the workbook.</td></tr>
<tr><td><b>Symmetric Averaging / Centring</b></td>
<td>Output C<sub>n</sub> averaging is a pyNuD addition. Centring ports FindCenterPositions-style axis finding for the rotation centre.</td></tr>
</table>
<div class="note">
<b>Pre-filter traps on real HS-AFM:</b> (1) stack-wide rescale compressed by spike frames; (2) Laplacian 50 often spreads false localizations — start at 0.<br>
<b>Licence:</b> Heath-derived paths come from GPL-3.0 NanoLocz. Distributing pyNuD <i>with</i> those paths requires GPL-3.0 for the distributed product; in-house use alone does not create a distribution obligation.
</div>

<hr>
<h2>References</h2>
<ul>
    <li>George R. Heath, et al. "<a href="https://doi.org/10.1038/s41586-021-03551-x">Localization atomic force microscopy</a>". <i>Nature</i> 594, 385–390 (2021).</li>
    <li>Heath, Micklethwaite &amp; Storer, NanoLocz, <i>Small Methods</i> 2024, 2301766.</li>
    <li>Yining Jiang, et al. "<a href="https://doi.org/10.1038/s41594-024-01260-3">HS-AFM single-molecule structural biology uncovers basis of transporter wanderlust kinetics</a>". <i>Nature Structural &amp; Molecular Biology</i> 31, 1286–1295 (2024).</li>
</ul>
"""

HELP_HTML_JA = """
<h1>L-AFM Analysis (Localization AFM)</h1>

<h2>概要</h2>
<p>L-AFM (Localization Atomic Force Microscopy) は、AFM時系列（動画）から輝度ピークを検出し、その局在を細かいグリッドへ再構成して超解像画像を得る手法です。本パネルでは <b>ピーク検出 → 再構成 → 最終画像</b> を段階実行できます。</p>
<p><strong>アルゴリズムの出典:</strong> Heath、Scheuringらによる Localization AFM（<i>Nature</i> 594, 385–390, 2021; DOI: 10.1038/s41586-021-03551-x）に基づきます。Heath / NanoLocz 互換オプションも用意していますが、<b>既定は従来の pyNuD 動作</b>です。</p>

<h2>アクセス</h2>
<ul>
    <li><strong>プラグイン:</strong> Load Plugin… → <code>plugins/LAFMAnalysis.py</code> → Plugin → L-AFM Analysis</li>
    <li><strong>マニュアル:</strong> 本パネルの Help → Manual（日本語 / English）</li>
</ul>

<h2>処理ステップ</h2>
<p>番号付きボタンを順に実行します。任意: <b>Load</b>（パラメータ）、<b>Measure Resolution (FRC)</b>。</p>
<div class="step">
    <strong>1. Preprocessing 1 — ピーク検出</strong><br>
    必要なら Drift Correction を行ったうえで、各フレームの局所極大を Peak Filtering / Local Maxima / Subpixel 条件で検出します。出力は局在リスト（座標・強度・フレーム番号など）です。
</div>
<div class="step">
    <strong>2. Preprocessing 2 — 再構成</strong><br>
    Step 1 の局在だけを使い、高解像度 2D グリッドまたは 3D ボクセルへプロットします（動画画素は再参照しません）。
</div>
<div class="step">
    <strong>3. Make LAFM Image — 最終画像</strong><br>
    ガウシアンぼかし（と任意の最終段 Symmetric Averaging）で滑らかな LAFM 画像に仕上げます。
</div>
<div class="step">
    <strong>Save / Load</strong><br>
    <b>Save</b>: 2D → ASD（＋ <code>*_params.json</code>）、3D → TIFF（＋ JSON）。コメントに処理／LAFM パラメータを埋め込みます。<br>
    <b>Load</b>: JSON からパネル設定を復元（画像データの再読込はしません）。
</div>
<div class="step">
    <strong>Measure Resolution (FRC)</strong><br>
    Preprocessing 1 後に局在をランダムに2分割し、Fourier ring correlation（1/7 基準）で分解能を出します。<b>このムービー内の再現性</b>であり、絶対精度ではありません。
</div>

<h2>Preprocessing 1 と 2 の違い</h2>
<div class="feature-box">
    <h4>Preprocessing 1 — 検出</h4>
    <ul>
        <li><strong>入力:</strong> AFM 動画スタック</li>
        <li><strong>役割:</strong> 「分子の住所録」を作る</li>
        <li><strong>コスト:</strong> 通常いちばん重い</li>
    </ul>
</div>
<div class="feature-box">
    <h4>Preprocessing 2 — 描画</h4>
    <ul>
        <li><strong>入力:</strong> Step 1 のピークリストのみ</li>
        <li><strong>役割:</strong> 細かいキャンバスへ局在を描く</li>
        <li><strong>コツ:</strong> 再構成条件だけ変えるなら Step 2 からでよい</li>
    </ul>
</div>

<h2>パラメータグループ（UI順）</h2>

<h3>Mode</h3>
<ul>
    <li><b>Mode:</b> <code>2D</code> / <code>3D</code></li>
    <li><b>3D Display:</b> 利用可能なら PyVista 3D ビューアを表示・更新</li>
</ul>

<h3>Drift Correction</h3>
<p>チェック可能なグループ。<b>既定 OFF</b>。ON のとき Preprocessing 1 のピーク検出前に実行されます。</p>
<table class="param-table">
<tr><th>項目</th><th>内容</th></tr>
<tr><td><b>Algorithm</b></td>
<td><i>Phase Correlation (Fast)</i>: 並進合わせ。<i>Feature-based (Precise)</i>: より精密だが遅い。</td></tr>
<tr><td><b>Subpixel Precision</b></td>
<td>位相相関の微調整用アップサンプル倍率（大きいほど精密・低速）。</td></tr>
<tr><td><b>Min Confidence</b></td>
<td>整列信頼度がこの値を<b>超える</b>フレームだけ残します（0–1）。閾値以下は検出から<b>除外</b>。残フレームが2枚未満だとエラーで停止します。</td></tr>
</table>
<div class="note">
上流の Tracking / Averaging 後も残るフレーム間ずれがあるときに使います。まずは Phase Correlation から。ピークが汚れる場合のみ Min Confidence を上げてください。
</div>

<h3>Peak Filtering</h3>
<table class="param-table">
<tr><th>項目</th><th>内容</th></tr>
<tr><td><b>Filter Mode</b></td>
<td><i>Absolute Height (nm)</i>: Z_min〜Z_max。<i>Statistics (Mean + N × Std Dev)</i>: 平均・標準偏差と N factor。</td></tr>
<tr><td><b>N factor</b></td>
<td>Statistics モード時に使用。</td></tr>
<tr><td><b>Enable ImageJ-compatible ROI mask</b></td>
<td>フレーム毎 Otsu ROI マスクと ImageJ 互換の閾値／正規化（ImageJ 系 LAFM 手順に近づける用）。</td></tr>
<tr><td><b>Rendering mode</b></td>
<td><i>pyNuD (probability × height)</i> と <i>Heath (localization density)</i>。物理量が違うので絶対値比較はしない（詳細は Heath 節）。</td></tr>
<tr><td><b>Auto Z-Range / Sample</b></td>
<td>スタック統計またはサンプル種別プリセットから Z_min / Z_max を提案。</td></tr>
<tr><td><b>Z_min / Z_max (nm)</b></td>
<td>ピーク受理の高さ窓。Pre-filter ON 時も nm 制限は<b>未フィルタ</b>データに適用。</td></tr>
<tr><td><b>Crop Ratio</b></td>
<td>半径 <code>(min(W,H)/2) × Crop Ratio</code> の円内ピークのみ残す（既定 0.9）。端のアーティファクト抑制用。</td></tr>
</table>

<h3>Pre-filter [Heath filter_movie]</h3>
<p>チェック可能。<b>既定 OFF</b>。Heath 風の Gaussian + Laplacian 前処理後、rescale した 0–1 面に Detection threshold を適用します。</p>
<div class="note">
実測 HS-AFM では Laplacian はまず <b>0</b> から（ワークブック値 50 は走査線ノイズを増幅しやすい）。スパイクフレームは先に除外（スタック全体 rescale のため）。
</div>

<h3>Local Maxima</h3>
<ul>
    <li><b>Search Size (n×n):</b> 局所極大の探索窓（奇数）</li>
    <li><b>Connectivity:</b> 4 または 8</li>
</ul>

<h3>Subpixel Localization</h3>
<p>チェック可能。ON で整数画素を超える位置精密化を行います。</p>
<table class="param-table">
<tr><th>項目</th><th>内容</th></tr>
<tr><td><b>Method</b></td>
<td>Interpolation (pyNuD) / Heath bicubic / Gaussian fit [Heath] / Sphere fit [Heath]。ノイズのある実測では Gaussian fit 推奨（補間系は格子吸着しやすい）。</td></tr>
<tr><td><b>Scale</b></td>
<td>サブピクセル精密化時の拡大倍率。</td></tr>
<tr><td><b>Expand</b></td>
<td>最終再構成グリッドの拡大（1=同一画素数、2=縦横2倍…）。</td></tr>
<tr><td><b>XY / Z Resolution</b></td>
<td>再構成グリッドの物理画素サイズ（nm）。主に 3D／記録用。</td></tr>
</table>

<h3>Centring</h3>
<p>Symmetric Averaging の回転中心の決め方（LAFM 密度累積そのものには不要）。</p>
<ul>
    <li><b>Off:</b> 配列中心で回転</li>
    <li><b>Centre of mass:</b> 強度重心（fold 非依存）</li>
    <li><b>Symmetry axis (C<sub>n</sub>) [Heath]:</b> FindCenterPositions 系。<b>Symmetry Order が必要</b></li>
</ul>
<p><b>Found offset</b> に求めたずれを表示（保存コメントにも記録される場合があります）。</p>

<h3>Symmetric Averaging</h3>
<p>チェック可能。既定 OFF。C<sub>n</sub> 回転平均。</p>
<ul>
    <li><b>During Reconstruction (Prep 2)</b> および／または <b>On Final LAFM Image</b></li>
    <li><b>Symmetry Order:</b> C<sub>n</sub> の n（1 は対称なし）</li>
</ul>
<p>Heath / NanoLocz 本体は LAFM マップ自体を対称化しません。出力の対称化は pyNuD 側の追加機能です。</p>

<h3>Gaussian Blur</h3>
<ul>
    <li><b>Sigma (xy) [pixels]</b> / <b>Sigma (z) [voxels]</b> — Make LAFM Image の平滑化</li>
</ul>

<h3>Visualization / Results</h3>
<ul>
    <li><b>Update Delay (ms):</b> 処理中プレビュー更新の間隔</li>
    <li><b>Total Detections / Reconstruction Size / FRC resolution:</b> 各段階の結果表示</li>
</ul>

<h2>パラメータ変更後にやり直すステップ</h2>
<h3>Preprocessing 1 から</h3>
<ul>
    <li>Drift Correction（すべて）</li>
    <li>Peak Filtering（ImageJ マスク、Rendering mode、Z範囲、Crop Ratio 含む）</li>
    <li>Pre-filter [Heath]</li>
    <li>Local Maxima</li>
    <li>Subpixel Localization の有効化 / Method / Scale</li>
</ul>
<h3>Preprocessing 2 から</h3>
<ul>
    <li>Mode（2D ↔ 3D）</li>
    <li>Subpixel の Expand、XY / Z Resolution</li>
    <li>Centring（Prep 2 対称化で使う場合）</li>
    <li>Symmetric Averaging → During Reconstruction (Prep 2)</li>
</ul>
<h3>Make LAFM Image から</h3>
<ul>
    <li>Gaussian Blur</li>
    <li>Symmetric Averaging → On Final LAFM Image</li>
    <li>Centring（最終段対称化のみ使う場合）</li>
</ul>

<h2>実務フロー</h2>
<pre><code>
graph TD
    A[L-AFM パネルを開く] --> B[パラメータ設定 / 必要なら Load JSON]
    B --> C[1. Preprocessing 1]
    C --> D{ピークは妥当か?}
    D -- No --> E[Drift / Peak Filtering / Local Maxima / Subpixel を調整]
    E --> C
    D -- Yes --> F[任意: Measure Resolution FRC]
    F --> G[2. Preprocessing 2]
    G --> H{再構成は妥当か?}
    H -- No --> I[Mode / Expand / Prep-2 対称化を調整]
    I --> G
    H -- Yes --> J[3. Make LAFM Image]
    J --> K{最終画像は妥当か?}
    K -- No --> L[Blur / 最終対称化を調整]
    L --> J
    K -- Yes --> M[Save ASD/TIFF + params JSON]
</code></pre>

<hr>
<h2>Heath / NanoLocz 互換オプション</h2>
<p><a href="https://github.com/George-R-Heath/NanoLocz-Matlab-Library">NanoLocz</a>
（Heath et al., Nature 2021）の一部を再現するオプションです。<b>互換処理は基本 OFF</b>（旧 JSON も読めます）。</p>
<table class="param-table">
<tr><th>オプション</th><th>変わる点</th></tr>
<tr><td><b>Rendering mode</b></td>
<td><i>pyNuD</i>: フレーム毎 確率×高さの平均。<i>Heath</i>: プールした局在<b>密度</b>（高さは主に色レベル）。</td></tr>
<tr><td><b>Subpixel method</b></td>
<td>上記 Subpixel Localization 参照。</td></tr>
<tr><td><b>Pre-filter</b></td>
<td><code>filter_movie</code> 後、rescale 0–1 面で閾値。</td></tr>
<tr><td><b>FRC</b></td>
<td>半データセット FRC（1/7）。expand=5、img_gaus=0.4（ワークブック相当）。</td></tr>
<tr><td><b>Symmetric Averaging / Centring</b></td>
<td>出力の C<sub>n</sub> 対称化は pyNuD 追加。Centring は回転中心推定（FindCenterPositions 系）。</td></tr>
</table>
<div class="note">
<b>実測での Pre-filter 注意:</b> (1) スパイクでスタック全体 rescale が潰れる (2) Laplacian 50 は誤局在を増やしやすい → 0 から。<br>
<b>ライセンス:</b> Heath 由来パスは GPL-3.0 NanoLocz 由来。これらを含めて配布する場合は配布物側も GPL-3.0 が必要。内部利用のみなら配布義務は生じません。
</div>

<hr>
<h2>参考文献</h2>
<ul>
    <li>George R. Heath, et al. "<a href="https://doi.org/10.1038/s41586-021-03551-x">Localization atomic force microscopy</a>". <i>Nature</i> 594, 385–390 (2021).</li>
    <li>Heath, Micklethwaite &amp; Storer, NanoLocz, <i>Small Methods</i> 2024, 2301766.</li>
    <li>Yining Jiang, et al. "<a href="https://doi.org/10.1038/s41594-024-01260-3">HS-AFM single-molecule structural biology uncovers basis of transporter wanderlust kinetics</a>". <i>Nature Structural &amp; Molecular Biology</i> 31, 1286–1295 (2024).</li>
</ul>
"""


# (LAFMWorkerクラスは変更ありません)
class LAFMWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int, str)
    error = QtCore.pyqtSignal(str)
    plot_signal = QtCore.pyqtSignal(np.ndarray, str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.kwargs['progress_signal'] = self.progress
            self.kwargs['plot_signal'] = self.plot_signal
            result = self.function(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

# --- ▼▼▼ Voxel3DViewerクラスを、以下の新しい定義に丸ごと置き換えてください ▼▼▼ ---
class Voxel3DViewer(QtWidgets.QWidget):
    """PyVistaを使ったインタラクティブ3Dボクセルビューア（Zスケール修正版）"""
    was_closed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Voxel Viewer")
        self.setMinimumSize(600, 500)
        
        # ウィンドウ管理システムに登録
        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self, "sub")
        except ImportError:
            pass

        # インスタンス変数の初期化
        self.plotter = None
        self.volume_data = None
        self.original_spacing = (1.0, 1.0, 1.0)
        
        # メインレイアウト
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 3D表示ウィジェット - より安全な初期化
        try:
            # 新しいバージョンのPyVistaに対応
            self.plotter = QtInteractor(self)
            
            # バージョンに応じて適切なウィジェットを追加
            if hasattr(self.plotter, 'interactor'):
                main_layout.addWidget(self.plotter.interactor)
            elif hasattr(self.plotter, 'app_window'):
                main_layout.addWidget(self.plotter.app_window)
            else:
                main_layout.addWidget(self.plotter)
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize QtInteractor: {e}")
            import traceback
            traceback.print_exc()
            # フォールバック: 通常のPyVista plotterを使用
            self.plotter = pv.Plotter()
            # エラーメッセージを表示
            error_label = QtWidgets.QLabel("3D Viewer initialization failed. Please check PyVista installation.")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            main_layout.addWidget(error_label)
            return

        # Zスケール調整用のUI
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(QtWidgets.QLabel("Z-Scale Exaggeration:"))
        self.z_scale_spin = QtWidgets.QDoubleSpinBox(value=1.0, minimum=0.1, maximum=100.0, singleStep=0.5, decimals=1)
        self.z_scale_spin.valueChanged.connect(self._update_z_scale) # 値の変更を検知
        control_layout.addWidget(self.z_scale_spin)
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)

    def update_data(self, volume_data, spacing=(1.0, 1.0, 1.0)):
        """新しいデータを受け取ったときに呼ばれるメソッド"""
        if not PYVISTA_AVAILABLE: return
        
        # データをインスタンス変数に保存
        self.volume_data = volume_data
        self.original_spacing = spacing
        
        # UIの初期値をリセットし、シーンを再描画
        self.z_scale_spin.setValue(1.0)
        self._redraw_scene()

    @QtCore.pyqtSlot()
    def _update_z_scale(self):
        """Zスケールスピンボックスの値が変更されたときに、シーンを再描画する"""
        self._redraw_scene()

    def _redraw_scene(self):
        """現在の設定で3Dシーンを再描画する内部メソッド"""
        if self.volume_data is None or self.plotter is None:
            return
            
        try:
            self.plotter.clear()
            
            grid = pv.ImageData()
            
            vol_transposed = self.volume_data.transpose(1, 0, 2)
            grid.dimensions = vol_transposed.shape
            
            # Zスケールを適用したspacingを計算
            z_scale_factor = self.z_scale_spin.value()
            effective_spacing = (
                self.original_spacing[0], 
                self.original_spacing[1], 
                self.original_spacing[2] * z_scale_factor
            )
            grid.spacing = effective_spacing
            
            grid.point_data["values"] = vol_transposed.flatten(order="F")

            # カラーバーは非表示
            self.plotter.add_volume(grid, cmap="magma", opacity="sigmoid", show_scalar_bar=False)
            
            # ルーラー（軸）を追加
            xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
            font_size = 10
            ruler_x = self.plotter.add_ruler([xmin, ymin, zmin], [xmax, ymin, zmin], label_format="%.1f", title="X (nm)")
            ruler_x.GetLabelTextProperty().SetFontSize(font_size); ruler_x.GetTitleTextProperty().SetFontSize(font_size)
            ruler_y = self.plotter.add_ruler([xmin, ymin, zmin], [xmin, ymax, zmin], label_format="%.1f", title="Y (nm)")
            ruler_y.GetLabelTextProperty().SetFontSize(font_size); ruler_y.GetTitleTextProperty().SetFontSize(font_size)
            z_unit = get_z_unit()
            ruler_z = self.plotter.add_ruler([xmin, ymin, zmin], [xmin, ymin, zmax], label_format="%.1f", title=f"Z ({z_unit})")
            ruler_z.GetLabelTextProperty().SetFontSize(font_size); ruler_z.GetTitleTextProperty().SetFontSize(font_size)
            
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception as e:
            print(f"[ERROR] Failed to redraw 3D scene: {e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        self.was_closed.emit()
        try:
            if self.plotter is not None:
                self.plotter.close()
        except Exception as e:
            print(f"[ERROR] Failed to close plotter: {e}")
        super().closeEvent(event)

class LAFMPanelWindow(QtWidgets.QWidget):
    # __init__メソッドは変更なし
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.Window)
        self.main_window = parent
        self.setWindowTitle("L-AFM Analysis")
        self.setMinimumSize(600, 420)
        
        # ウィンドウ管理システムに登録
        try:
            from window_manager import register_pyNuD_window
            register_pyNuD_window(self, "sub")
        except ImportError:
            pass

        # --- ▼▼▼【重要修正点】ウィンドウ設定を正しく探し出して復元するロジック ▼▼▼ ---
        window_settings = getattr(gv, 'windowSettings', {})
        saved_settings = None
        
        # "LAFMPanelWindow"で始まるキーを全て探し、最初に見つかったものを使用する
        for key, settings in window_settings.items():
            if key.startswith(self.__class__.__name__):
                saved_settings = settings
                break

        # 【重要】'visible'のチェックを削除し、設定が存在すれば必ず位置を復元する
        if saved_settings:
            try:
                self.setGeometry(
                    saved_settings.get('x', 150),
                    saved_settings.get('y', 150),
                    saved_settings.get('width', 600),
                    saved_settings.get('height', 420)
                )
            except Exception as e:
                print(f"Failed to set geometry from saved settings: {e}")
        else:
            # 保存された設定がない場合のみ、デフォルトの位置に表示
            if self.main_window:
                main_geo = self.main_window.geometry()
                self.move(main_geo.x() + main_geo.width() + 10, main_geo.y())
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

        self.params = {}
        self.original_image_stack = None
        self.scale_info = None
        self.detection_summary = None
        self.processed_shape = None

        self.processed_image_stack = None
        self.reconstruction = None
        self.reconstruction_image = None
        self.final_lafm_image = None
        self.viewer_3d_window = None

        # [Heath] 密度レンダリング用の細グリッド座標と FRC 結果
        self.hz_grid = None
        self.hz_zlims = None
        self.frc_result = None
        self.sym_centre_translation = None

        self.top_last_np_array = None
        self.bottom_last_np_array = None
        self.top_last_aspect_ratio = 1.0
        self.bottom_last_aspect_ratio = 1.0

        self.initUI()
        self.start_initial_load()
    
    def start_initial_load(self):
        """非同期でデータ読み込みを開始する"""
        self._update_status("Loading image data...", color="darkorange")
        self.progress_bar.setRange(0, 100) # この時点ではプログレスは動かない
        self._run_in_thread(
            self.load_initial_data,
            self._on_initial_load_finished
        )

    def resizeEvent(self, event):
        """ウィンドウのリサイズ時に画像の再描画を行う"""
        super().resizeEvent(event)

        if self.top_last_np_array is not None:
            self._display_image(self.top_last_np_array, target='top')
        
        # 表示するデータは、モードによって元のデータソースが異なるため、
        # reconstructionが存在するかどうかで判定する
        if hasattr(self, 'reconstruction') and self.reconstruction is not None:
             if self.params.get('mode', '2D') == '2D':
                display_img = np.sum(self.reconstruction, axis=2)
             else:
                display_img = np.max(self.reconstruction, axis=2)
             self._display_image(display_img, target='bottom')
            
    def _auto_calculate_z_range(self, image_stack):
        """画像スタックから適切なZ_minとZ_maxを自動計算"""
        try:
            # 全フレームから統計情報を取得
            all_data = image_stack.flatten()
            
            # ノイズフロアの推定（下位10%の標準偏差）
            noise_threshold = np.percentile(all_data, 10)
            noise_data = all_data[all_data <= noise_threshold]
            noise_std = np.std(noise_data) if len(noise_data) > 100 else np.std(all_data) * 0.1
            
            # ベースライン（下位5%の平均）
            baseline = np.mean(all_data[all_data <= np.percentile(all_data, 5)])
            
            # Z_min: ノイズフロア + 3σ
            z_min_noise = baseline + 3 * noise_std
            z_min_percentile = np.percentile(all_data, 2)  # 下位2%
            z_min = max(z_min_noise, z_min_percentile, 0.01)  # 最小10pm
            
            # Z_max: 上位95%パーセンタイル
            z_max = np.percentile(all_data, 95)
            
            # 妥当性チェック
            if z_max <= z_min:
                data_range = np.max(all_data) - np.min(all_data)
                z_max = z_min + max(0.1, data_range * 0.5)
            
            # データカバー率の計算
            coverage = np.sum((all_data >= z_min) & (all_data <= z_max)) / len(all_data) * 100
            
            #print(f"[Z-Range Auto] Recommended: Z_min={z_min:.3f}nm, Z_max={z_max:.3f}nm")
            #print(f"[Z-Range Auto] Data coverage: {coverage:.1f}%")
            #print(f"[Z-Range Auto] Noise level: {noise_std:.4f}nm")
            
            return z_min, z_max
            
        except Exception as e:
            #print(f"[ERROR] Z-range auto calculation failed: {e}")
            return 0.1, 5.0  # デフォルト値

    def _manual_auto_z_range(self):
        """手動でZ範囲を再計算するボタンの処理"""
        if self.original_image_stack is not None:
            z_min_auto, z_max_auto = self._auto_calculate_z_range(self.original_image_stack)
            self.z_min_spin.setValue(z_min_auto)
            self.z_max_spin.setValue(z_max_auto)
            
            # 統計情報を更新
            data_range = z_max_auto - z_min_auto
            if hasattr(self, 'z_stats_label'):
                self.z_stats_label.setText(f"Range: {data_range:.3f}nm")
            
            self._update_status(f"Z-range updated: {z_min_auto:.3f}-{z_max_auto:.3f}nm", color="info")
        else:
            QtWidgets.QMessageBox.warning(self, "No Data", "画像データが読み込まれていません。")

    # サンプルタイプ別の推奨設定
    SAMPLE_TYPE_Z_RECOMMENDATIONS = {
        "General": {"z_min": 0.1, "z_max": 10.0, "desc": "General purpose settings"},
        "Proteins": {"z_min": 0.1, "z_max": 10.0, "desc": "Single proteins to large complexes"},
        "DNA/RNA": {"z_min": 0.05, "z_max": 3.0, "desc": "DNA molecules and nucleic acids"},
        "Cells": {"z_min": 1.0, "z_max": 100.0, "desc": "Cellular structures and organelles"},
        "Crystals": {"z_min": 0.01, "z_max": 50.0, "desc": "Crystal surfaces and defects"},
        "Nanoparticles": {"z_min": 0.5, "z_max": 20.0, "desc": "Nanoparticles and aggregates"}
    }

    def _on_sample_type_changed(self, sample_type):
        """サンプルタイプが変更された時の処理"""
        if sample_type in self.SAMPLE_TYPE_Z_RECOMMENDATIONS:
            settings = self.SAMPLE_TYPE_Z_RECOMMENDATIONS[sample_type]
            self.z_min_spin.setValue(settings["z_min"])
            self.z_max_spin.setValue(settings["z_max"])
            
            # 統計情報を更新
            data_range = settings["z_max"] - settings["z_min"]
            if hasattr(self, 'z_stats_label'):
                self.z_stats_label.setText(f"Range: {data_range:.3f}nm")
            
            self._update_status(f"Applied {sample_type} settings: {settings['desc']}", color="info")

    def _on_initial_load_finished(self, result):
        """データ読み込み完了後の処理（Z範囲自動設定統合版）"""
        stack, scale_info = result
        if stack is not None:
            self.original_image_stack = stack
            self.scale_info = scale_info
            self.processed_shape = stack.shape
            
            # 🔥 Z範囲の自動設定を追加
            z_min_auto, z_max_auto = self._auto_calculate_z_range(stack)
            self.z_min_spin.setValue(z_min_auto)
            self.z_max_spin.setValue(z_max_auto)
            
            # 統計情報を更新
            data_range = z_max_auto - z_min_auto
            if hasattr(self, 'z_stats_label'):
                self.z_stats_label.setText(f"Range: {data_range:.3f}nm")
            
            self._update_status(
                f"{stack.shape[2]} frames loaded. Z-range auto-set: {z_min_auto:.3f}-{z_max_auto:.3f}nm. Ready for Preprocessing 1.", 
                color="green"
            )
            self._display_image(self.original_image_stack[:, :, 0], target='bottom')
            
            # Nの初期値を計算（既存コード）
            try:
                first_frame = self.original_image_stack[:, :, 0]
                processed_first_frame = first_frame - np.min(first_frame)
                mean = np.mean(processed_first_frame); std = np.std(processed_first_frame)
                if std > 1e-9:
                    threshold = np.percentile(processed_first_frame, 99.9)
                    calculated_n = (threshold - mean) / std
                    self.std_dev_factor_spin.setValue(calculated_n)
            except Exception as e:
                self._update_status(f"Could not auto-set N: {e}", color="warning")
            
            # Preprocessing 1 ボタンを有効化
            self.btn_prep1.setEnabled(True)
        else:
            self._update_status("Failed to load image stack.", color="red", level=1)

    # initUI: メニューバー（Help → Manual）を上部に配置し、既存レイアウトは content_widget に格納
    def initUI(self):
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        control_widget = QtWidgets.QScrollArea()
        control_widget.setWidgetResizable(True); control_widget.setMinimumWidth(340); control_widget.setMaximumWidth(400)
        scroll_content = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(scroll_content)
        control_layout.setAlignment(QtCore.Qt.AlignTop); control_layout.setSpacing(6)
        control_widget.setWidget(scroll_content)

        button_grid_layout = QtWidgets.QGridLayout()
        self.btn_prep1 = QtWidgets.QPushButton("1. Preprocessing 1")
        self.btn_prep2 = QtWidgets.QPushButton("2. Preprocessing 2")
        self.btn_make_img = QtWidgets.QPushButton("3. Make LAFM Image")
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_load = QtWidgets.QPushButton("Load")
        button_grid_layout.addWidget(self.btn_prep1, 0, 0); button_grid_layout.addWidget(self.btn_prep2, 0, 1)
        button_grid_layout.addWidget(self.btn_make_img, 1, 0); button_grid_layout.addWidget(self.btn_save, 1, 1)
        button_grid_layout.addWidget(self.btn_load, 2, 0, 1, 2)
        self.btn_frc = QtWidgets.QPushButton("Measure Resolution (FRC)")
        self.btn_frc.setToolTip(
            "[Heath] measureFRC: split the localizations into two random halves by frame,\n"
            "Fourier ring correlation, resolution = 1/(first crossing of 1/7).\n"
            "Requires Preprocessing 1. Both halves come from one movie, so this measures\n"
            "the reproducibility of this dataset rather than absolute accuracy."
        )
        button_grid_layout.addWidget(self.btn_frc, 3, 0, 1, 2)
        control_layout.addLayout(button_grid_layout)

        self.btn_prep1.setEnabled(False)
        self.btn_prep2.setEnabled(False); self.btn_make_img.setEnabled(False); self.btn_save.setEnabled(False)
        self.btn_frc.setEnabled(False)
        self.btn_prep1.clicked.connect(self.run_preprocessing1)
        self.btn_prep2.clicked.connect(self.run_preprocessing2)
        self.btn_make_img.clicked.connect(self.run_make_lafm_image)
        self.btn_save.clicked.connect(self._save_lafm_data)
        self.btn_load.clicked.connect(self._load_lafm_params)
        self.btn_frc.clicked.connect(self.run_measure_frc)
        
        mode_layout = QtWidgets.QHBoxLayout()
        self.mode_combo = QtWidgets.QComboBox(); self.mode_combo.addItems(["2D", "3D"])
        self.show_3d_check = QtWidgets.QCheckBox("3D Display")
        mode_layout.addWidget(QtWidgets.QLabel("Mode:")); mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(self.show_3d_check)
        mode_layout.addStretch()
        control_layout.addLayout(mode_layout)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.show_3d_check.toggled.connect(self._handle_3d_display_toggle)

        self.status_label = QtWidgets.QLabel("Ready. Load data to start.")
        self.status_label.setStyleSheet("font-weight: bold; color: blue;"); self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        self._on_mode_changed(0)
        
        def create_form_group_box(title, checkable=False):
            group = QtWidgets.QGroupBox(title)
            group.setCheckable(checkable)
            if checkable: group.setChecked(False)
            layout = QtWidgets.QFormLayout(group)
            layout.setLabelAlignment(QtCore.Qt.AlignLeft); layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            layout.setSpacing(5); layout.setContentsMargins(8, 10, 8, 8)
            return group, layout
        
        # ▼▼▼【新規追加】Drift Correction グループを Peak Filtering の後に追加 ▼▼▼
        drift_group, drift_layout = create_form_group_box("Drift Correction", checkable=True)
        drift_group.setChecked(False)  # デフォルトは無効
        
        # アルゴリズム選択
        self.drift_algorithm_combo = QtWidgets.QComboBox()
        self.drift_algorithm_combo.addItems(["Phase Correlation (Fast)", "Feature-based (Precise)"])
        drift_layout.addRow("Algorithm:", self.drift_algorithm_combo)

        self.drift_subpixel_spin = QtWidgets.QSpinBox(value=10, minimum=1, maximum=100)
        self.drift_subpixel_spin.setToolTip(
            "Subpixel precision for phase-correlation fine alignment.\n"
            "Higher values improve alignment precision but take longer."
        )
        drift_layout.addRow("Subpixel Precision:", self.drift_subpixel_spin)
        
        # 信頼度閾値
        self.drift_threshold_spin = QtWidgets.QDoubleSpinBox(value=0.1, minimum=0.0, maximum=1.0, singleStep=0.01, decimals=3)
        self.drift_threshold_spin.setToolTip("Minimum confidence threshold for frame alignment (0.0 - 1.0)\nフレーム位置合わせの最小信頼度閾値 (0.0 - 1.0)")
        drift_layout.addRow("Min Confidence:", self.drift_threshold_spin)
        
        control_layout.addWidget(drift_group)
        self.drift_group = drift_group  # 後で参照するため保存

        tol_group, tol_layout = create_form_group_box("Peak Filtering")
        self.filter_mode_combo = QtWidgets.QComboBox()
        self.filter_mode_combo.addItems(["Absolute Height (nm)", "Statistics (Mean + N x Std Dev)"])
        self.filter_mode_combo.currentIndexChanged.connect(self._on_filter_mode_changed)
        tol_layout.addRow("Filter Mode:", self.filter_mode_combo)
        self.std_dev_label = QtWidgets.QLabel("N factor:")
        self.std_dev_factor_spin = QtWidgets.QDoubleSpinBox(value=0.0, minimum=-5.0, maximum=20.0, singleStep=0.1)
        tol_layout.addRow(self.std_dev_label, self.std_dev_factor_spin)
        self.imagej_compat_check = QtWidgets.QCheckBox("Enable ImageJ-compatible ROI mask")
        self.imagej_compat_check.setToolTip(
            "Apply per-frame Otsu ROI mask and use ImageJ-compatible tolerance/normalization.\n"
            "フレームごとのOtsu ROIマスクを適用し、ImageJ互換の閾値/正規化を使用します。"
        )
        self.imagej_compat_check.toggled.connect(self._on_imagej_compat_changed)
        tol_layout.addRow("", self.imagej_compat_check)

        # ▼▼▼ [Heath] レンダリング方式の選択（既定は従来の pyNuD 方式） ▼▼▼
        self.render_mode_combo = QtWidgets.QComboBox()
        self.render_mode_combo.addItems([
            "pyNuD (probability x height)",
            "Heath (localization density)",
        ])
        self.render_mode_combo.setToolTip(
            "pyNuD: sum over frames of gaussian(peaks) * (height - min), divided by frame count.\n"
            "Heath: NanoLocz LAFM_renderer -- localization DENSITY pooled over all frames;\n"
            "height enters only via the colour-level binning. The two are different quantities.\n"
            "pyNuD: フレーム毎の 確率x高さ の積。Heath: 全フレームをプールした局在密度。"
        )
        tol_layout.addRow("Rendering mode:", self.render_mode_combo)

        # Z範囲の自動設定ボタン（他のコントロールと同じ左の位置に配置）
        auto_z_button = QtWidgets.QPushButton("Auto Z-Range")
        auto_z_button.setMaximumWidth(120)
        auto_z_button.clicked.connect(self._manual_auto_z_range)
        auto_z_button.setToolTip("Recalculate optimal Z_min and Z_max from current data")
        
        # データ統計表示ラベル
        self.z_stats_label = QtWidgets.QLabel("Range: N/A")
        self.z_stats_label.setStyleSheet("color: gray;")
        
        # サンプルタイプ選択コンボボックス
        self.sample_type_combo = QtWidgets.QComboBox()
        self.sample_type_combo.addItems(["General", "Proteins", "DNA/RNA", "Cells", "Crystals", "Nanoparticles"])
        self.sample_type_combo.currentTextChanged.connect(self._on_sample_type_changed)
        self.sample_type_combo.setToolTip("Select sample type for recommended Z-range settings")
        
        # Auto Z-Rangeボタンと統計表示を横並びに
        auto_z_row = QtWidgets.QHBoxLayout()
        auto_z_row.addWidget(auto_z_button)
        auto_z_row.addWidget(self.z_stats_label)
        auto_z_row.addStretch()
        tol_layout.addRow("Auto Z-Range:", auto_z_row)
        
        # Sample選択を別の行に配置
        sample_row = QtWidgets.QHBoxLayout()
        sample_row.addWidget(QtWidgets.QLabel("Sample:"))
        sample_row.addWidget(self.sample_type_combo)
        sample_row.addStretch()
        tol_layout.addRow("", sample_row)
        
        self.z_min_label = QtWidgets.QLabel("Z_min (nm):")
        self.z_min_spin = QtWidgets.QDoubleSpinBox(value=0.1, minimum=-1000, maximum=1000, singleStep=0.1)
        self.z_max_spin = QtWidgets.QDoubleSpinBox(value=5.0, minimum=-1000, maximum=1000, singleStep=0.1)
        self.z_min_spin.valueChanged.connect(self._on_z_range_changed); self.z_max_spin.valueChanged.connect(self._on_z_range_changed)
        self.crop_ratio_spin = QtWidgets.QDoubleSpinBox(value=0.9, minimum=0.1, maximum=1.0, singleStep=0.05)
        tol_layout.addRow(self.z_min_label, self.z_min_spin); tol_layout.addRow("Z_max (nm):", self.z_max_spin); tol_layout.addRow("Crop Ratio:", self.crop_ratio_spin)
        control_layout.addWidget(tol_group)
        
        # ▼▼▼ [Heath] filter_movie 前処理（既定 OFF）▼▼▼
        self.prefilter_group, prefilter_layout = create_form_group_box(
            "Pre-filter [Heath filter_movie]", checkable=True)
        self.prefilter_group.setChecked(False)
        self.prefilter_group.setToolTip(
            "Heath's Workbook_LAFM applies filter_movie(im,'Gaussian',0.2,'Laplacian',50)\n"
            "before peak detection, and thresholds the rescale()d result.\n"
            "WARNING: rescale() spans the whole stack, so a single spiking frame compresses\n"
            "all others. Clean the stack first. On real HS-AFM a Laplacian of 50 amplifies\n"
            "raster line noise and scatters localizations over bare substrate -- start at 0.\n"
            "注意: rescale はスタック全体で正規化。実測データでは Laplacian は 0 から試すこと。"
        )
        self.pre_gauss_spin = QtWidgets.QDoubleSpinBox(value=0.2, minimum=0.0, maximum=10.0, singleStep=0.1)
        self.pre_laplacian_spin = QtWidgets.QDoubleSpinBox(value=0.0, minimum=0.0, maximum=200.0, singleStep=5.0)
        self.pre_laplacian_spin.setToolTip("Heath's workbook default is 50. 0 disables it.")
        self.heath_thresh_spin = QtWidgets.QDoubleSpinBox(value=0.5, minimum=0.0, maximum=1.0, singleStep=0.01)
        self.heath_thresh_spin.setToolTip(
            "LAFM_thresh in Heath's workbook: detection threshold on the rescaled (0-1)\n"
            "filtered stack. Z_min/Z_max in nm are still applied to the UNFILTERED data."
        )
        prefilter_layout.addRow("Gaussian sigma:", self.pre_gauss_spin)
        prefilter_layout.addRow("Laplacian strength:", self.pre_laplacian_spin)
        prefilter_layout.addRow("Detection threshold (0-1):", self.heath_thresh_spin)
        control_layout.addWidget(self.prefilter_group)

        lm_group, lm_layout = create_form_group_box("Local Maxima")
        self.search_size_spin = QtWidgets.QSpinBox(value=3, minimum=3, maximum=21, singleStep=2)
        self.connectivity_combo = QtWidgets.QComboBox(); self.connectivity_combo.addItems(["4", "8"]); self.connectivity_combo.setCurrentText("8")
        lm_layout.addRow("Search Size (nxn):", self.search_size_spin); lm_layout.addRow("Connectivity:", self.connectivity_combo)
        control_layout.addWidget(lm_group)
       
        self.subpix_group, subpix_layout = create_form_group_box("Subpixel Localization", checkable=True)
        self.subpix_scale_spin = QtWidgets.QSpinBox(value=10, minimum=2, maximum=20)
        self.subpix_expand_spin = QtWidgets.QSpinBox(value=1, minimum=1, maximum=20)
        self.subpix_expand_spin.setToolTip(
            "Final reconstruction grid expansion factor.\n"
            "1 keeps the current pixel count, 2 doubles width/height, etc."
        )
        self.subpix_xy_res_spin = QtWidgets.QDoubleSpinBox(value=0.1, minimum=0.01, maximum=10.0, singleStep=0.01, suffix=" nm")
        self.subpix_z_res_spin = QtWidgets.QDoubleSpinBox(value=0.1, minimum=0.01, maximum=10.0, singleStep=0.01, suffix=" nm")
        # ▼▼▼ [Heath] サブピクセル局在法の選択（既定は従来の pyNuD 補間）▼▼▼
        self.subpix_method_combo = QtWidgets.QComboBox()
        self.subpix_method_combo.addItems([
            "Interpolation (pyNuD)",
            "Heath bicubic",
            "Gaussian fit [Heath]",
            "Sphere fit [Heath]",
        ])
        self.subpix_method_combo.setToolTip(
            "Interpolation (pyNuD): zoom the 5x5 ROI by Scale and take the argmax.\n"
            "Heath bicubic: MATLAB imresize x10 then the central 30x30 argmax.\n"
            "Gaussian fit: 2-D Gaussian least squares -- does NOT pixel-lock, best for noisy data.\n"
            "Sphere fit: algebraic sphere fit, for spherical-cap features.\n"
            "ノイズのある実測データでは Gaussian fit を推奨（補間系は格子に吸着する）。"
        )
        subpix_layout.addRow("Method:", self.subpix_method_combo)
        subpix_layout.addRow("Scale:", self.subpix_scale_spin)
        subpix_layout.addRow("Expand:", self.subpix_expand_spin)
        subpix_layout.addRow("XY Resolution:", self.subpix_xy_res_spin)
        subpix_layout.addRow("Z Resolution:", self.subpix_z_res_spin)
        control_layout.addWidget(self.subpix_group)

        # ▼▼▼ Centring -- 対称化から独立させた中心検出 ▼▼▼
        # 「対称軸を求める」は対称化専用の下請けではなく独立した測定なので別グループにする。
        # ただし Symmetry axis 法は fold を必要とする(FindCenterPositions.m の仕様)ため、
        # そのモードのときだけ下の Symmetry Order を参照する。Centre of mass は fold 非依存。
        centring_group, centring_layout = create_form_group_box("Centring")
        self.centring_combo = QtWidgets.QComboBox()
        self.centring_combo.addItems([
            "Off",
            "Centre of mass",
            "Symmetry axis (C_n) [Heath]",
        ])
        self.centring_combo.setCurrentText("Symmetry axis (C_n) [Heath]")
        self.centring_combo.setToolTip(
            "How to locate the centre that rotational operations turn about.\n"
            "  Off                : rotate about the ARRAY centre (smears by the axis offset)\n"
            "  Centre of mass     : intensity-weighted centroid after removing the median\n"
            "                       background. Fold-INDEPENDENT, so it also works when no\n"
            "                       symmetry is assumed.\n"
            "  Symmetry axis (C_n): FindCenterPositions.m -- cross-correlate the 360/n rotated\n"
            "                       copies against the original. NEEDS Symmetry Order below.\n"
            "Currently consumed by Symmetric Averaging. Measured on a tracked EltXeR crop, the\n"
            "symmetry-axis option cut the blur that symmetrisation adds from +0.052 to +0.004 nm.\n"
            "回転操作の中心の求め方。Symmetry axis は下の Symmetry Order を必要とする。\n"
            "Centre of mass は fold 非依存なので対称性を仮定しない場合にも使える。"
        )
        self.centring_label = QtWidgets.QLabel("--")
        self.centring_label.setStyleSheet("color: gray;")
        centring_layout.addRow("Method:", self.centring_combo)
        centring_layout.addRow("Found offset:", self.centring_label)
        control_layout.addWidget(centring_group)

        self.sym_group = QtWidgets.QGroupBox("Symmetric Averaging"); self.sym_group.setCheckable(True); self.sym_group.setChecked(False)
        sym_v_layout = QtWidgets.QVBoxLayout(self.sym_group)
        sym_v_layout.setSpacing(5); sym_v_layout.setContentsMargins(8, 10, 8, 8)
        self.sym_prep2_check = QtWidgets.QCheckBox("During Reconstruction (Prep 2)"); self.sym_final_check = QtWidgets.QCheckBox("On Final LAFM Image")
        order_row_layout = QtWidgets.QHBoxLayout()
        order_row_layout.addWidget(QtWidgets.QLabel("Symmetry Order:"))
        self.sym_order_spin = QtWidgets.QSpinBox(value=1, minimum=1, maximum=12)
        order_row_layout.addWidget(self.sym_order_spin); order_row_layout.addStretch()
        sym_v_layout.addWidget(self.sym_prep2_check); sym_v_layout.addWidget(self.sym_final_check)
        sym_v_layout.addLayout(order_row_layout)
        control_layout.addWidget(self.sym_group)

       
        
        blur_group, blur_layout = create_form_group_box("Gaussian Blur")
        self.blur_sigma_xy_spin = QtWidgets.QDoubleSpinBox(value=1.0, minimum=0.1, maximum=10.0, singleStep=0.1)
        self.blur_sigma_z_spin = QtWidgets.QDoubleSpinBox(value=1.0, minimum=0.1, maximum=10.0, singleStep=0.1)
        blur_layout.addRow("Sigma (xy) [pixels]:", self.blur_sigma_xy_spin); blur_layout.addRow("Sigma (z) [voxels]:", self.blur_sigma_z_spin)
        control_layout.addWidget(blur_group)
        
        # --- ▼▼▼【重要追加】可視化設定のUI ▼▼▼ ---
        vis_group, vis_layout = create_form_group_box("Visualization")
        self.vis_delay_spin = QtWidgets.QSpinBox(minimum=0, maximum=1000, value=0, singleStep=10, suffix=" ms")
        vis_layout.addRow("Update Delay (ms):", self.vis_delay_spin)
        control_layout.addWidget(vis_group)
        # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---
        
        self.progress_bar = QtWidgets.QProgressBar()
        control_layout.addWidget(self.progress_bar)
        
        results_group, results_layout = create_form_group_box("Processing Results")
        self.detections_label = QtWidgets.QLabel("0")
        self.reconst_size_label = QtWidgets.QLabel("N/A")
        self.frc_label = QtWidgets.QLabel("N/A")
        results_layout.addRow("Total Detections:", self.detections_label); results_layout.addRow("Reconstruction Size:", self.reconst_size_label)
        results_layout.addRow("FRC resolution:", self.frc_label)
        control_layout.addWidget(results_group)
        
        control_layout.addStretch()

        display_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.top_image_label = QtWidgets.QLabel("Processing View"); self.top_image_label.setMinimumSize(150, 120); self.top_image_label.setAlignment(QtCore.Qt.AlignCenter); self.top_image_label.setStyleSheet("background-color: #111; color: white; border: 1px solid #444;")
        self.bottom_image_label = QtWidgets.QLabel("Final LAFM Image View"); self.bottom_image_label.setMinimumSize(150, 120); self.bottom_image_label.setAlignment(QtCore.Qt.AlignCenter); self.bottom_image_label.setStyleSheet("background-color: black; color: white; border: 1px solid #444;")
        display_splitter.addWidget(self.top_image_label); display_splitter.addWidget(self.bottom_image_label)
        display_splitter.setSizes([150, 150])

        splitter.addWidget(control_widget); splitter.addWidget(display_splitter)
        splitter.setSizes([350, 220])
        main_layout.addWidget(splitter)

        content_widget = QtWidgets.QWidget()
        content_widget.setLayout(main_layout)
        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        help_menu = menu_bar.addMenu("Help")
        manual_action = help_menu.addAction("Manual")
        manual_action.triggered.connect(self.showHelpDialog)
        top_layout = QtWidgets.QVBoxLayout(self)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(menu_bar)
        top_layout.addWidget(content_widget, 1)
        
        self._on_filter_mode_changed(0)
        self._on_z_range_changed()

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
        .feature-box { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; }
        .note { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 14px; border-radius: 4px; margin: 14px 0; font-size: 15px; }
        h1 { font-size: 22px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { font-size: 18px; color: #2c3e50; margin-top: 18px; }
        h3, h4 { font-size: 16px; color: #34495e; }
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
                dialog.setWindowTitle("L-AFM解析 - マニュアル")
                close_btn.setText("閉じる")
            else:
                browser.setHtml("<html><body>" + HELP_HTML_EN.strip() + "</body></html>")
                dialog.setWindowTitle("L-AFM Analysis - Manual")
                close_btn.setText("Close")

        btn_ja.clicked.connect(lambda: set_lang(True))
        btn_en.clicked.connect(lambda: set_lang(False))
        layout_dlg.addWidget(browser)
        layout_dlg.addWidget(close_btn)
        set_lang(False)  # デフォルトは英語
        dialog.exec_()

    def closeEvent(self, event):
        """ウィンドウが閉じられるときに設定を保存する"""
        thread = getattr(self, "thread", None)
        try:
            thread_is_running = thread is not None and thread.isRunning()
        except RuntimeError:
            thread_is_running = False
        if thread_is_running:
            event.ignore()
            QtWidgets.QMessageBox.information(
                self,
                "Processing in progress",
                "L-AFM processing is still running.\n"
                "処理が完了してからウィンドウを閉じてください。",
            )
            return

        try:
            if not hasattr(gv, 'windowSettings'):
                gv.windowSettings = {}
            
            # --- ▼▼▼【重要修正点】古い設定を削除してから、正しいキーで保存する ▼▼▼ ---
            # まず、"LAFMPanelWindow_1" のような古い設定があれば削除する
            keys_to_delete = [k for k in gv.windowSettings if k.startswith(self.__class__.__name__)]
            for key in keys_to_delete:
                del gv.windowSettings[key]

            # 次に、常に番号なしの正しいキーで現在の状態を保存する
            gv.windowSettings[self.__class__.__name__] = {
                'x': self.x(), 'y': self.y(),
                'width': self.width(), 'height': self.height(),
                'visible': False,
                'class_name': self.__class__.__name__
            }
            # --- ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ ---

            if self.main_window and hasattr(self.main_window, 'saveAllInitialParams'):
                self.main_window.saveAllInitialParams()
        
        except Exception as e:
            print(f"[ERROR] Failed to save LAFM panel settings: {e}")
        
        # ツールバーアクションのハイライトを解除（プラグインとして開いている場合）
        try:
            if hasattr(gv, 'main_window') and gv.main_window and hasattr(gv.main_window, 'plugin_actions'):
                action = gv.main_window.plugin_actions.get(PLUGIN_NAME)
                if action is not None and hasattr(gv.main_window, 'setActionHighlight'):
                    gv.main_window.setActionHighlight(action, False)
        except Exception as e:
            print(f"[WARNING] Failed to reset LAFM action highlight: {e}")
            
        super().closeEvent(event)

    def _collect_params(self):
        self.params = {
            'mode': self.mode_combo.currentText(),
            'filter_mode': self.filter_mode_combo.currentText(),
            'std_dev_factor': self.std_dev_factor_spin.value(),
            'imagej_compat_mode': self.imagej_compat_check.isChecked(),
            'z_min': self.z_min_spin.value(),
            'z_max': self.z_max_spin.value(),
            'crop_ratio': self.crop_ratio_spin.value(),
            'search_size': self.search_size_spin.value(),
            'connectivity': int(self.connectivity_combo.currentText()),
            'subpixel_on': self.subpix_group.isChecked(),
            'subpixel_scale': self.subpix_scale_spin.value(),
            'subpixel_expand': self.subpix_expand_spin.value(),
            'subpixel_xy_res': self.subpix_xy_res_spin.value(),
            'subpixel_z_res': self.subpix_z_res_spin.value(),
            'sym_on': self.sym_group.isChecked(),
            'sym_on_prep2': self.sym_prep2_check.isChecked(),
            'sym_on_final': self.sym_final_check.isChecked(),
            'sym_order': self.sym_order_spin.value(),
            'centring_method': self.centring_combo.currentText(),
            # 旧ビルド互換: 真偽値としても残す
            'sym_autocentre': not self.centring_combo.currentText().lower().startswith('off'),
            'blur_sigma_xy': self.blur_sigma_xy_spin.value(),
            'blur_sigma_z': self.blur_sigma_z_spin.value(),
            'drift_correction': self.drift_group.isChecked(),
            'drift_algorithm': self.drift_algorithm_combo.currentText(),
            'drift_subpixel_precision': self.drift_subpixel_spin.value(),
            'drift_threshold': self.drift_threshold_spin.value(),

            'vis_delay_spin': self.vis_delay_spin.value(),

            # ▼ [Heath] NanoLocz-derived options (see the licensing note at the top)
            'render_mode': self.render_mode_combo.currentText(),
            'subpix_method': self.subpix_method_combo.currentText(),
            'prefilter_on': self.prefilter_group.isChecked(),
            'prefilter_gauss': self.pre_gauss_spin.value(),
            'prefilter_laplacian': self.pre_laplacian_spin.value(),
            'heath_thresh': self.heath_thresh_spin.value(),
        }
        return self.params

    def _get_params_json_path(self, data_path):
        root, _ext = os.path.splitext(data_path)
        return root + "_params.json"

    def _save_lafm_params_json(self, json_path):
        self._collect_params()
        payload = {
            'plugin': PLUGIN_NAME,
            'version': 1,
            'params': self.params,
        }
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _apply_loaded_params(self, params):
        if not isinstance(params, dict):
            raise ValueError("Invalid parameter file format.")

        self.mode_combo.setCurrentText(str(params.get('mode', self.mode_combo.currentText())))
        self.filter_mode_combo.setCurrentText(str(params.get('filter_mode', self.filter_mode_combo.currentText())))
        self.std_dev_factor_spin.setValue(float(params.get('std_dev_factor', self.std_dev_factor_spin.value())))
        self.imagej_compat_check.setChecked(bool(params.get('imagej_compat_mode', self.imagej_compat_check.isChecked())))
        self.z_min_spin.setValue(float(params.get('z_min', self.z_min_spin.value())))
        self.z_max_spin.setValue(float(params.get('z_max', self.z_max_spin.value())))
        self.crop_ratio_spin.setValue(float(params.get('crop_ratio', self.crop_ratio_spin.value())))
        self.search_size_spin.setValue(int(params.get('search_size', self.search_size_spin.value())))
        self.connectivity_combo.setCurrentText(str(params.get('connectivity', self.connectivity_combo.currentText())))

        self.subpix_group.setChecked(bool(params.get('subpixel_on', self.subpix_group.isChecked())))
        self.subpix_scale_spin.setValue(int(params.get('subpixel_scale', self.subpix_scale_spin.value())))
        self.subpix_expand_spin.setValue(int(params.get('subpixel_expand', self.subpix_expand_spin.value())))
        self.subpix_xy_res_spin.setValue(float(params.get('subpixel_xy_res', self.subpix_xy_res_spin.value())))
        self.subpix_z_res_spin.setValue(float(params.get('subpixel_z_res', self.subpix_z_res_spin.value())))

        self.sym_group.setChecked(bool(params.get('sym_on', self.sym_group.isChecked())))
        self.sym_prep2_check.setChecked(bool(params.get('sym_on_prep2', self.sym_prep2_check.isChecked())))
        self.sym_final_check.setChecked(bool(params.get('sym_on_final', self.sym_final_check.isChecked())))
        self.sym_order_spin.setValue(int(params.get('sym_order', self.sym_order_spin.value())))
        self.centring_combo.setCurrentText(_hz_centring_method(params))

        self.blur_sigma_xy_spin.setValue(float(params.get('blur_sigma_xy', self.blur_sigma_xy_spin.value())))
        self.blur_sigma_z_spin.setValue(float(params.get('blur_sigma_z', self.blur_sigma_z_spin.value())))

        self.drift_group.setChecked(bool(params.get('drift_correction', self.drift_group.isChecked())))
        self.drift_algorithm_combo.setCurrentText(str(params.get('drift_algorithm', self.drift_algorithm_combo.currentText())))
        self.drift_subpixel_spin.setValue(int(params.get('drift_subpixel_precision', self.drift_subpixel_spin.value())))
        self.drift_threshold_spin.setValue(float(params.get('drift_threshold', self.drift_threshold_spin.value())))

        self.vis_delay_spin.setValue(int(params.get('vis_delay_spin', self.vis_delay_spin.value())))

        # ▼ [Heath] 追加オプション。旧 JSON には無いので現在値を既定にしてフォールバック
        self.render_mode_combo.setCurrentText(str(params.get('render_mode', self.render_mode_combo.currentText())))
        self.subpix_method_combo.setCurrentText(str(params.get('subpix_method', self.subpix_method_combo.currentText())))
        self.prefilter_group.setChecked(bool(params.get('prefilter_on', self.prefilter_group.isChecked())))
        self.pre_gauss_spin.setValue(float(params.get('prefilter_gauss', self.pre_gauss_spin.value())))
        self.pre_laplacian_spin.setValue(float(params.get('prefilter_laplacian', self.pre_laplacian_spin.value())))
        self.heath_thresh_spin.setValue(float(params.get('heath_thresh', self.heath_thresh_spin.value())))

        self._on_mode_changed(self.mode_combo.currentIndex())
        self._on_filter_mode_changed(self.filter_mode_combo.currentIndex())
        self._on_imagej_compat_changed(self.imagej_compat_check.isChecked())
        self._on_z_range_changed()
        self._collect_params()

    def _load_lafm_params(self):
        dialog_options = QtWidgets.QFileDialog.Options()
        if sys.platform != "darwin":
            dialog_options |= QtWidgets.QFileDialog.DontUseNativeDialog

        start_folder = getattr(gv, 'lastUsedSaveDir', '') or (
            os.path.dirname(gv.files[gv.currentFileNum]) if hasattr(gv, 'files') and gv.files else ""
        )
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load LAFM Parameters",
            start_folder,
            "JSON File (*.json)",
            options=dialog_options
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)

            if isinstance(payload, dict) and 'params' in payload:
                params = payload['params']
            else:
                params = payload

            self._apply_loaded_params(params)
            gv.lastUsedSaveDir = os.path.dirname(filepath)
            self._update_status(f"Loaded parameters from {os.path.basename(filepath)}", color="green")
            QtWidgets.QMessageBox.information(self, "Load Complete", f"L-AFM parameters loaded:\n{filepath}")
        except Exception as e:
            self._handle_error(f"Failed to load parameter file: {e}")

    # ▼▼▼【重要修正点】_save_lafm_data メソッドを全面的に書き換え ▼▼▼
    def _save_lafm_data(self):
        """Saveボタンがクリックされたときに呼び出されるスロット（LAFM専用保存対応版）"""
        if self.final_lafm_image is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "保存するLAFM画像がありません。")
            return
        
        try:
            base_filename = os.path.splitext(os.path.basename(gv.files[gv.currentFileNum]))[0]
        except:
            base_filename = "LAFM_result"

        if self.params.get('mode', '2D') == '2D':
            default_savename = f"{base_filename}_LAFM.asd"
            file_filter = "ASD File (*.asd)"
        else: # 3D Mode
            default_savename = f"{base_filename}_LAFM_3D.tiff"
            file_filter = "TIFF Image (*.tif *.tiff)"

        if hasattr(gv, 'lastUsedSaveDir') and gv.lastUsedSaveDir and os.path.isdir(gv.lastUsedSaveDir):
            start_folder = gv.lastUsedSaveDir
        else:
            start_folder = os.path.dirname(gv.files[gv.currentFileNum]) if hasattr(gv, 'files') and gv.files else ""
        
        default_save_path = os.path.join(start_folder, default_savename)
        dialog_options = QtWidgets.QFileDialog.Options()
        if sys.platform != "darwin":
            dialog_options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save LAFM Data", default_save_path, file_filter,
            options=dialog_options
        )

        if not filepath:
            return

        try:
            params_json_path = self._get_params_json_path(filepath)
            self._update_status(f"Saving to {os.path.basename(filepath)}...", color="darkorange")
            
            if self.params['mode'] == '2D':
                # ▼▼▼【重要修正点】新しく作る専用のASD保存メソッドを呼び出す ▼▼▼
                comment = f"LAFM 2D result from {base_filename}.asd"
                
                # 処理パラメータを追加
                from helperFunctions import collect_processing_parameters
                processing_params = collect_processing_parameters()
                if processing_params:
                    comment = comment + "\n" + processing_params
                
                # LAFM固有のパラメータを追加
                lafm_params = []
                if hasattr(self, 'params'):
                    mode = self.params.get('mode', '2D')
                    lafm_params.append(f"Mode: {mode}")
                    if 'z_range' in self.params:
                        z_range = self.params['z_range']
                        lafm_params.append(f"Z Range: {z_range[0]:.1f} - {z_range[1]:.1f} nm")
                    if 'filter_mode' in self.params:
                        lafm_params.append(f"Filter Mode: {self.params['filter_mode']}")
                    # [Heath] 由来の設定は再現性のため必ず記録する
                    rm = self.params.get('render_mode')
                    if rm:
                        lafm_params.append(f"Rendering Mode: {rm}")
                    sm = self.params.get('subpix_method')
                    if sm:
                        lafm_params.append(f"Subpixel Method: {sm}")
                    if self.params.get('prefilter_on'):
                        lafm_params.append(
                            "Pre-filter [Heath]: Gaussian %g, Laplacian %g, threshold %g"
                            % (self.params.get('prefilter_gauss', 0.0),
                               self.params.get('prefilter_laplacian', 0.0),
                               self.params.get('heath_thresh', 0.0)))
                    if self.params.get('sym_on') and self.params.get('sym_order', 1) > 1:
                        where = []
                        if self.params.get('sym_on_prep2'): where.append('Prep2')
                        if self.params.get('sym_on_final'): where.append('Final')
                        lafm_params.append(
                            "Symmetric Averaging: C%d on %s, centring: %s"
                            % (self.params['sym_order'], '+'.join(where) or 'nothing',
                               _hz_centring_method(self.params)))
                        ct = getattr(self, 'sym_centre_translation', None)
                        if ct:
                            lafm_params.append("Symmetry centre offset: dx=%+.2f dy=%+.2f px" % (ct[0], ct[1]))
                    if getattr(self, 'frc_result', None):
                        lafm_params.append(
                            "FRC resolution: %.2f +/- %.2f nm (half-dataset split, n=%d)"
                            % (self.frc_result['resolution_nm'], self.frc_result['sd_nm'],
                               self.frc_result['n_localizations']))
                if lafm_params:
                    comment = comment + "\n[LAFM Parameters]\n" + "\n".join(lafm_params)
                
                success = self._save_lafm_as_asd(filepath, comment, self.final_lafm_image)
                if not success:
                    raise Exception("Failed to save LAFM data as ASD.")
            else:
                tifffile.imsave(filepath, self.final_lafm_image, imagej=True)

            self._save_lafm_params_json(params_json_path)

            gv.lastUsedSaveDir = os.path.dirname(filepath)
            self._update_status(f"Saved successfully!", color="green")
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"LAFMデータを保存しました:\n{filepath}\n\nParameters JSON:\n{params_json_path}"
            )

            if self.main_window and hasattr(self.main_window, 'rescan_and_load'):
                self._update_status(f"Reloading {os.path.basename(filepath)}...", color="info")
                self.main_window.rescan_and_load(filepath)
        
        except Exception as e:
            self._handle_error(f"Failed to save file: {e}")
            import traceback
            traceback.print_exc()

    # (これより下のメソッドは、既存の正しいコードをそのまま含めてください)
    # _collect_params, _update_status, _set_buttons_enabled, _display_image, _handle_error,
    # _update_progress, _run_in_thread, _plot_image, load_initial_data, run_... , _on_..._finished,
    # _on_z_range_changed, _on_filter_mode_changed, _create_lafm_lut, _execute_...
    
        
    def _update_status(self, text, color="blue", level=0):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-weight: bold; color: {color};")
        #if level == 1: print(f"[LAFM-ERROR] {text}")
        #elif level == 2: print(f"[LAFM-WARN] {text}")
        #else: print(f"[LAFM-INFO] {text}")

    def _set_buttons_enabled(self, prep1, prep2, make_img):
        self.btn_prep1.setEnabled(prep1)
        self.btn_prep2.setEnabled(prep2)
        self.btn_make_img.setEnabled(make_img)
        # FRC は Preprocessing 1 の検出結果だけで計算できる
        self.btn_frc.setEnabled(prep2 and getattr(self, 'detection_summary', None) is not None)


    def _display_image(self, np_array, target='bottom'):
        """
        NumPy配列を、正しい向きと物理アスペクト比でUIの指定ラベルに表示する (FIXED)
        """
        label = self.top_image_label if target == 'top' else self.bottom_image_label

        if np_array is None or np_array.size == 0:
            label.setText("No image to display."); return

        # データを表示用に上下反転させる
        display_data = np.flipud(np_array)

        # 再描画用にNumPyデータを保存
        if target == 'top': self.top_last_np_array = display_data
        else: self.bottom_last_np_array = display_data

        # 表示用8-bitカラー画像へ変換
        img_to_display = None
        if len(display_data.shape) == 3 and display_data.shape[2] != 3:
            display_data = np.max(display_data, axis=2)
        
        if len(display_data.shape) == 2:
            # ▼▼▼【ここからがコントラスト調整の修正箇所です】▼▼▼
            
            # self.paramsが存在し、subpixel_onが有効かチェック
            is_subpixel_mode = self.params.get('subpixel_on', False)

            if is_subpixel_mode:
                # --- サブピクセルONの場合の強力な強調処理 ---
                v_max = np.max(display_data)
                if v_max > 0:
                    scaled_data = display_data.astype(np.float32) / v_max
                    gamma = 0.3  # 強いガンマ補正
                    gamma_corrected = np.power(scaled_data, gamma)
                    img_norm_8u = (gamma_corrected * 255).astype(np.uint8)
                else:
                    img_norm_8u = np.zeros_like(display_data, dtype=np.uint8)
            else:
                # --- 通常時のコントラスト調整 (非ゼロピクセルのパーセンタイル) ---
                non_zero_pixels = display_data[display_data > 0]
                if non_zero_pixels.size > 0:
                    v_min, v_max = np.percentile(non_zero_pixels, (1, 99))
                    if v_max <= v_min:
                        v_min, v_max = np.min(non_zero_pixels), np.max(non_zero_pixels)
                    clipped_data = np.clip(display_data, v_min, v_max)
                    scale = 255.0 / (v_max - v_min) if (v_max - v_min) > 0 else 0
                    img_norm_8u = ((clipped_data - v_min) * scale).astype(np.uint8)
                else:
                    img_norm_8u = np.zeros_like(display_data, dtype=np.uint8)

            # 共通のカラーマップ適用
            img_to_display = cv2.applyColorMap(img_norm_8u, self._create_lafm_lut())
            # ▲▲▲【コントラスト調整の修正ここまで】▲▲▲
       
        elif len(display_data.shape) == 3 and display_data.shape[2] == 3:
            img_to_display = display_data.astype(np.uint8)

        if img_to_display is None: return

        # QPixmapに変換
        h_px, w_px, ch = img_to_display.shape
        q_img = QtGui.QImage(img_to_display.data, w_px, h_px, ch * w_px, QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)
        
        if target == 'top': self.top_current_pixmap = pixmap
        else: self.bottom_current_pixmap = pixmap

        # --- ▼▼▼ ここからが修正箇所 ▼▼▼ ---
        aspect_ratio = 1.0
        
        if target == 'bottom' and hasattr(self, 'lafm_image_scan_size'):
            # 下部ビューアの場合: 再構成後の画像の物理サイズからアスペクト比を計算
            scan_size = self.lafm_image_scan_size
            if scan_size.get('y', 0) > 0:
                aspect_ratio = scan_size['x'] / scan_size['y']
        elif hasattr(self, 'scale_info'):
            # 上部ビューアの場合: 元画像のピクセルあたりの物理サイズからアスペクト比を計算
            dx = self.scale_info.get('dx', 1.0)
            dy = self.scale_info.get('dy', 1.0)
            if dy > 0:
                # 物理アスペクト比 = (物理的な幅) / (物理的な高さ) = (ピクセル幅 * dx) / (ピクセル高 * dy)
                aspect_ratio = (w_px * dx) / (h_px * dy)
        
        # リサイズ用にアスペクト比を記憶
        if target == 'top': self.top_last_aspect_ratio = aspect_ratio
        else: self.bottom_last_aspect_ratio = aspect_ratio
        # --- ▲▲▲ 修正ここまで ▲▲▲ ---

        # アスペクト比を維持してスケーリング＆表示
        widget_size = label.size()
        if widget_size.isEmpty(): return
        
        display_width = widget_size.width()
        display_height = int(display_width / aspect_ratio) if aspect_ratio > 0 else 0
        if display_height > widget_size.height():
            display_height = widget_size.height()
            display_width = int(display_height * aspect_ratio)
            
        display_size = QtCore.QSize(display_width, display_height)
        scaled_pixmap = pixmap.scaled(display_size, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def _handle_error(self, message):
        self._update_status(f"Error: {message.splitlines()[0]}", color="red", level=1)
        QtWidgets.QMessageBox.critical(self, "Processing Error", message)
        self._set_buttons_enabled(True, False, False)
        self.btn_prep2.setEnabled(self.detection_summary is not None)
        self.btn_make_img.setEnabled(self.reconstruction is not None)
        self.btn_save.setEnabled(self.final_lafm_image is not None)

    def _update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self._update_status(message, color="darkorange")

    def _run_in_thread(self, function, on_finish, *args, **kwargs):
        self._set_buttons_enabled(False, False, False)
        self.btn_save.setEnabled(False) # 処理中はSaveボタンも無効化
        self.thread = QtCore.QThread()
        self.worker = LAFMWorker(function, *args, **kwargs)
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self._update_progress)
        self.worker.error.connect(self._handle_error)
        self.worker.plot_signal.connect(self._plot_image)
        self.worker.finished.connect(on_finish)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def _plot_image(self, image_data, target_name):
        """ワーカースレッドからの描画シグナルを処理するスロット"""
        self._display_image(image_data, target=target_name)
  

    def load_initial_data(self, progress_signal=None, plot_signal=None):
        if self.main_window:
            start_frame = gv.FirstFrame if gv.FirstFrame is not None else 0
            end_frame = gv.LastFrame if gv.LastFrame is not None else gv.FrameNum - 1
            
            # get_image_stack_for_lafmもフレームごとにプログレスを報告するように変更が必要ですが、
            # まずはこちらのロジックを修正します。
            stack, scale_info = self.main_window.get_image_stack_for_lafm(start_frame, end_frame)
            
            # このメソッドは値を返し、インスタンス変数は直接設定しない
            return stack, scale_info
        return None, None

    def run_preprocessing1(self):
        # ▼▼▼【このメソッドを以下のようにシンプルに書き換えてください】▼▼▼

        # Preprocessing1ボタンを押した際に、以前の結果をクリアする
        if self.main_window:
            start_frame = gv.FirstFrame if gv.FirstFrame is not None else 0
            end_frame = gv.LastFrame if gv.LastFrame is not None else gv.FrameNum - 1
            stack, scale_info = self.main_window.get_image_stack_for_lafm(start_frame, end_frame)
            
            if stack is not None:
                self.original_image_stack = stack
                self.scale_info = scale_info
                # 以前の結果をクリア
                self.detection_summary = None
                self.reconstruction = None
                self.reconstruction_image = None
                self.final_lafm_image = None
                # UIの状態を更新
                self.detections_label.setText("0")
                self.reconst_size_label.setText("N/A")
                self._display_image(self.original_image_stack[:, :, 0], target='bottom')
            else:
                self._update_status("Failed to load current file data.", color="red")
                return
                
        # パラメータを収集し、ワーカースレッドで _execute_preprocessing1 を実行
        self._collect_params()
        self._update_status("Step 1: Cropping and correcting...", color="darkorange")
        self.btn_save.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self._run_in_thread(self._execute_preprocessing1, self._on_preprocessing1_finished, self.original_image_stack, self.params)

    def _on_drift_correction_finished(self, result):
        """ドリフト補正完了後にpreprocessing1を開始"""
        if result is not None:
            corrected_stack, excluded_frames = result
            self.original_image_stack = corrected_stack
            
            if len(excluded_frames) > 0:
                self._update_status(f"Drift correction excluded {len(excluded_frames)} frames. Starting detection...", color="info")
            else:
                self._update_status("Drift correction completed. Starting detection...", color="info")
        else:
            self._update_status("Drift correction failed. Using original data...", color="warning")
        
        # ドリフト補正後は既存と同じpreprocessing1を実行
        self._collect_params()
        self._update_status("Step 1: Detecting local maxima...", color="darkorange")
        self.btn_save.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self._run_in_thread(self._execute_preprocessing1, self._on_preprocessing1_finished, self.original_image_stack, self.params)
        
    # def _execute_drift_correction(self, image_stack, params, progress_signal=None, plot_signal=None):
    #     """ドリフト補正を実行（averaging.pyのロジックを活用）"""
    #     print(f"[DEBUG] _execute_drift_correction called with stack shape: {image_stack.shape}")
    #     print(f"[DEBUG] Drift params: {params}")

    #     try:
    #         if progress_signal: progress_signal.emit(10, "Calculating transformations...")
            
    #         is_feature_based = "Feature-based" in params['drift_algorithm']
    #         confidence_threshold = params['drift_threshold']
            
    #         # 変換行列と信頼度を計算
    #         matrices, confidences = self._calculate_transformations_for_lafm(
    #             image_stack, 
    #             is_rotation_enabled=is_feature_based,
    #             progress_signal=progress_signal
    #         )
            
    #         if progress_signal: progress_signal.emit(50, "Filtering unreliable frames...")
            
    #         # 信頼度による除外
    #         good_indices = np.where(confidences > confidence_threshold)[0]
    #         excluded_frames = [i for i in range(len(image_stack)) if i not in good_indices]
            
    #         if len(good_indices) < 2:
    #             if progress_signal: progress_signal.emit(100, "Drift correction failed - insufficient reliable frames")
    #             return None
            
    #         if progress_signal: progress_signal.emit(80, "Applying transformations...")
            
    #         # 補正済み画像スタックを作成
    #         corrected_stack = image_stack[good_indices]
    #         matrices_to_apply = matrices[good_indices]
    #         h, w = corrected_stack[0].shape
            
    #         # 各画像に変換行列を適用
    #         final_corrected_stack = np.array([
    #             cv2.warpAffine(img, M, (w, h), borderValue=np.median(img)) 
    #             for img, M in zip(corrected_stack, matrices_to_apply)
    #         ])
            
    #         if progress_signal: progress_signal.emit(100, "Drift correction completed")
            
    #         return final_corrected_stack, excluded_frames
            
    #     except Exception as e:
    #         if progress_signal: progress_signal.emit(100, f"Drift correction error: {e}")
    #         return None
    # 完全な_execute_drift_correction_syncメソッド

    

    def _calculate_feature_based_real(self, image_stack, progress_dialog):
        """AFM画像に最適化されたFeature-based処理"""
        num_images = len(image_stack)
        total_matrices = [np.eye(2, 3, dtype=np.float32) for _ in range(num_images)]
        confidences = np.ones(num_images)
        
        # 固定リファレンス（最初のフレーム）
        reference_image = image_stack[0]
        
        #print(f"[DEBUG] AFM-optimized Feature-based processing")
        #print(f"[DEBUG] Reference image: shape={reference_image.shape}")
        
        # AFM用に調整されたORB設定
        orb = cv2.ORB_create(
            nfeatures=1000,      # 特徴点数を制限（質を重視）
            scaleFactor=1.2,     # より粗いスケール
            nlevels=8,           # レベル数を減らす
            edgeThreshold=10,    # エッジ閾値を上げる（ノイズ除去）
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10     # 閾値を上げて品質重視
        )
        
        # 参照画像の特徴点を事前計算
        ref_enhanced = self._enhance_for_afm_features(reference_image)
        kp_ref, des_ref = orb.detectAndCompute(ref_enhanced, None)
        
        #print(f"[DEBUG] Reference features: {len(kp_ref) if kp_ref else 0} keypoints")
        
        for i in range(1, min(num_images, 11)):  # 最初の10フレームのみテスト
            current_image = image_stack[i]
            transformation_matrix = np.eye(2, 3, dtype=np.float32)
            confidence = 0.0
            
            try:
                # 現在フレームの前処理と特徴点検出
                curr_enhanced = self._enhance_for_afm_features(current_image)
                kp_curr, des_curr = orb.detectAndCompute(curr_enhanced, None)

                #print(f"\n[DEBUG] Frame {i}:")
                #print(f"  Features: {len(kp_curr) if kp_curr else 0} keypoints")
                
                if des_curr is not None and des_ref is not None and len(des_ref) > 20 and len(des_curr) > 20:
                    # より厳密なマッチング
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = matcher.match(des_ref, des_curr)
                    
                    if len(matches) > 20:  # 最低マッチ数を増やす
                        matches = sorted(matches, key=lambda x: x.distance)
                        
                        # より厳しい距離フィルタ
                        distance_threshold = min(60, matches[0].distance * 2.0)  # より厳しい
                        good_matches = [m for m in matches if m.distance < distance_threshold]
                        
                        #print(f"  Matches: {len(matches)} → {len(good_matches)} (thresh={distance_threshold})")
                        
                        if len(good_matches) >= 8:  # 最低点数を増やす
                            ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches])
                            curr_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches])
                            
                            # 点の分布をチェック（極端に偏っていないか）
                            ref_spread = np.std(ref_pts, axis=0)
                            curr_spread = np.std(curr_pts, axis=0)
                            min_spread = min(reference_image.shape) * 0.1  # 画像の10%以上に分布
                            
                            if np.min(ref_spread) > min_spread and np.min(curr_spread) > min_spread:
                                # より厳密なRANSAC設定
                                try:
                                    M_cv = cv2.estimateAffinePartial2D(
                                        curr_pts, ref_pts,
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=2.0,  # より厳しい閾値
                                        maxIters=5000,
                                        confidence=0.99,            # 高い信頼度要求
                                        refineIters=10
                                    )
                                    
                                    if M_cv[0] is not None and M_cv[1] is not None:
                                        matrix = M_cv[0]
                                        inliers = M_cv[1].flatten()
                                        inlier_count = np.sum(inliers)
                                        inlier_ratio = inlier_count / len(good_matches)
                                        
                                        # 変換の妥当性をチェック（AFM用の厳しい制限）
                                        translation = np.linalg.norm(matrix[:, 2])
                                        angle = np.arctan2(matrix[1, 0], matrix[0, 0])
                                        scale_x = np.sqrt(matrix[0,0]**2 + matrix[0,1]**2)
                                        scale_y = np.sqrt(matrix[1,0]**2 + matrix[1,1]**2)
                                        
                                        #print(f"  Transform: trans={translation:.1f}px, angle={np.degrees(angle):.1f}°")
                                        #print(f"  Scale: ({scale_x:.3f}, {scale_y:.3f}), inliers: {inlier_count}/{len(good_matches)} ({inlier_ratio:.3f})")
                                        
                                        # AFM用の厳しい制限
                                        max_translation = min(current_image.shape) * 0.2  # 20%まで
                                        max_angle = np.pi / 12  # ±15度まで
                                        
                                        if (translation < max_translation and 
                                            abs(angle) < max_angle and
                                            0.98 <= scale_x <= 1.02 and    # ほぼスケール変化なし
                                            0.98 <= scale_y <= 1.02 and
                                            inlier_ratio > 0.5 and         # 50%以上のインライア
                                            inlier_count >= 10):           # 最低10点のインライア
                                            
                                            transformation_matrix = matrix
                                            confidence = inlier_ratio * min(1.0, inlier_count / 20.0)
                                            
                                            #print(f"  ✅ ACCEPTED - confidence: {confidence:.3f}")
                                        else:
                                            confidence = 0.05
                                            #print(f"  ❌ REJECTED - strict AFM limits")
                                            #print(f"    Limits: trans<{max_translation:.1f}, angle<{np.degrees(max_angle):.1f}°, inlier>{0.5}")
                                    else:
                                        confidence = 0.02
                                        #print(f"  ❌ RANSAC failed")
                                        
                                except Exception as e:
                                    confidence = 0.01
                                    #print(f"  ❌ Exception: {e}")
                            else:
                                confidence = 0.02
                                #print(f"  ❌ Poor point distribution: {ref_spread}, {curr_spread}")
                        else:
                            confidence = 0.02
                            #print(f"  ❌ Too few good matches: {len(good_matches)}")
                    else:
                        confidence = 0.01
                        #print(f"  ❌ Insufficient total matches: {len(matches)}")
                else:
                    confidence = 0.01
                    #print(f"  ❌ Too few features")
                        
            except Exception as e:
                confidence = 0.01
                print(f"  ❌ Exception: {e}")

            confidences[i] = confidence
            total_matrices[i] = transformation_matrix
            
            # 残りのフレームは低信頼度を割り当て
            for j in range(max(11, i+1), num_images):
                confidences[j] = 0.05
        
        #print(f"\n[DEBUG] AFM Feature-based summary:")
        #print(f"  Confidence range: {np.min(confidences):.4f} - {np.max(confidences):.4f}")
        #print(f"  Frames > 0.1: {np.sum(confidences > 0.1)}/{len(confidences)}")
        #print(f"  Frames > 0.5: {np.sum(confidences > 0.5)}/{len(confidences)}")
            
        return np.array(total_matrices), confidences

    def _enhance_for_afm_features(self, image):
        """AFM画像専用の特徴点強調"""
        # 8bit変換
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # AFM画像の特徴を強調
        # 1. 軽いガウシアンブラーでノイズ除去
        denoised = cv2.GaussianBlur(img, (3, 3), 0.8)
        
        # 2. 適応ヒストグラム均等化（控えめ）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        enhanced = clahe.apply(denoised)
        
        # 3. 軽いエッジ強調
        kernel = np.array([[-0.05, -0.1, -0.05],
                        [-0.1,   1.3, -0.1],
                        [-0.05, -0.1, -0.05]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _calculate_rigid_transformation_simple(self, prev_pts, curr_pts):
        """シンプルな剛体変換計算（RANSACベース）"""
        if len(prev_pts) < 6:
            return None, None, 0.0
        
        try:
            # RANSACによる外れ値除去
            best_inliers = None
            best_matrix = None
            max_inliers = 0
            
            n_iterations = min(100, len(prev_pts) * 2)
            threshold = 4.0
            
            for iteration in range(n_iterations):
                # 最小6点をランダム選択
                indices = np.random.choice(len(prev_pts), 6, replace=False)
                sample_prev = prev_pts[indices]
                sample_curr = curr_pts[indices]
                
                # 簡易剛体変換推定
                matrix = self._estimate_rigid_simple(sample_prev, sample_curr)
                
                if matrix is not None:
                    # 全点での誤差計算
                    prev_pts_hom = np.column_stack([prev_pts, np.ones(len(prev_pts))])
                    transformed_pts = (matrix @ prev_pts_hom.T).T
                    
                    errors = np.linalg.norm(transformed_pts - curr_pts, axis=1)
                    inliers = errors < threshold
                    n_inliers = np.sum(inliers)
                    
                    if n_inliers > max_inliers:
                        max_inliers = n_inliers
                        best_inliers = inliers
                        best_matrix = matrix
            
            if best_matrix is not None and max_inliers >= 6:
                # 内点のみで再推定
                inlier_prev = prev_pts[best_inliers]
                inlier_curr = curr_pts[best_inliers]
                
                refined_matrix = self._estimate_rigid_simple(inlier_prev, inlier_curr)
                
                if refined_matrix is not None:
                    inlier_ratio = max_inliers / len(prev_pts)
                    return refined_matrix, best_inliers, inlier_ratio
            
            return None, None, 0.0
            
        except Exception as e:
            return None, None, 0.0

    def _estimate_rigid_simple(self, src_pts, dst_pts):
        """簡易剛体変換推定（重心ベース）"""
        try:
            if len(src_pts) < 3 or len(dst_pts) < 3:
                return None
            
            # 重心を計算
            src_center = np.mean(src_pts, axis=0)
            dst_center = np.mean(dst_pts, axis=0)
            
            # 重心からの相対位置
            src_centered = src_pts - src_center
            dst_centered = dst_pts - dst_center
            
            # 回転角度推定（最小二乗法）
            angles = []
            for i in range(min(len(src_centered), 10)):
                if np.linalg.norm(src_centered[i]) > 1e-6 and np.linalg.norm(dst_centered[i]) > 1e-6:
                    angle_src = np.arctan2(src_centered[i][1], src_centered[i][0])
                    angle_dst = np.arctan2(dst_centered[i][1], dst_centered[i][0])
                    angle_diff = angle_dst - angle_src
                    # 角度を[-π, π]に正規化
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    angles.append(angle_diff)
            
            if len(angles) == 0:
                angle = 0.0
            else:
                # 平均角度を計算（円形平均）
                angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
            
            # 角度制限（±45度以内）
            if abs(angle) > np.pi/4:
                return None
            
            # 回転行列作成
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # 平行移動ベクトル（重心の移動）
            tx = dst_center[0] - src_center[0]
            ty = dst_center[1] - src_center[1]
            
            # 平行移動制限（±100ピクセル以内）
            if abs(tx) > 100 or abs(ty) > 100:
                return None
            
            # 変換行列構築
            matrix = np.array([
                [cos_a, -sin_a, tx],
                [sin_a,  cos_a, ty]
            ], dtype=np.float32)
            
            return matrix
                
        except Exception as e:
            return None
    
    def _calculate_simple_fallback(self, image_stack, progress_dialog):
        """最もシンプルなフォールバック処理"""
        num_images = len(image_stack)
        matrices = [np.eye(2, 3, dtype=np.float32) for _ in range(num_images)]
        confidences = np.ones(num_images) * 0.5  # 全フレームに0.5の信頼度を与える
        
        #print(f"[DEBUG] Using simple fallback - all frames get confidence 0.5")
        
        return np.array(matrices), confidences

        
    def _calculate_phase_correlation_simple(self, image_stack, progress_dialog):
        """シンプルなPhase Correlation計算"""
        num_images = len(image_stack)
        matrices = [np.eye(2, 3, dtype=np.float32) for _ in range(num_images)]
        confidences = np.ones(num_images)
        
        # 最初のフレームを基準とする
        reference_image = image_stack[0]
        
        for i in range(1, num_images):
            if progress_dialog.wasCanceled():
                return None, None
                
            if i % 10 == 0:
                progress = int(10 + 50 * i / num_images)
                progress_dialog.setValue(progress)
                progress_dialog.setLabelText(f"Processing frame {i}/{num_images}")
                QtWidgets.QApplication.processEvents()
            
            current_image = image_stack[i]
            
            try:
                # 前処理
                ref_processed = self._preprocess_for_correlation(reference_image)
                curr_processed = self._preprocess_for_correlation(current_image)
                
                # Phase Correlation
                shift, error, _ = phase_cross_correlation(
                    ref_processed, curr_processed, 
                    upsample_factor=2,
                    space="real"
                )
                
                # 信頼度計算
                max_allowed_shift = min(current_image.shape) * 0.2
                shift_magnitude = np.linalg.norm(shift)
                
                if error < 0.5 and shift_magnitude < max_allowed_shift:
                    confidence = max(0.1, min(1.0, (0.5 - error) * 2.0))
                    matrices[i][0, 2] = shift[1]  # dx
                    matrices[i][1, 2] = shift[0]  # dy
                else:
                    confidence = 0.01
                    
                confidences[i] = confidence
                
            except Exception as e:
                #print(f"[DEBUG] Frame {i} failed: {e}")
                confidences[i] = 0.01
        
        return np.array(matrices), confidences

    def _enhance_vertical_features(self, image):
        """垂直方向の特徴を強調する前処理（averaging.pyと同じ）"""
        # Sobelフィルタで垂直エッジを強調
        sobel_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        
        # ガウシアンフィルタでノイズ除去
        enhanced = cv2.GaussianBlur(np.abs(sobel_x), (1, 5), 0)  # 垂直方向にのみブラー
        
        # 正規化
        enhanced = cv2.normalize(enhanced, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        return enhanced# スレッドを使わないシンプルなドリフト補正版


    def _calculate_feature_based_simple(self, image_stack, progress_dialog):
        """シンプルなFeature-based計算（とりあえずPhase Correlationと同じ）"""
        # とりあえずPhase Correlationと同じ処理
        return self._calculate_phase_correlation_simple(image_stack, progress_dialog)

    def _preprocess_for_correlation(self, image):
        """位相相関のための画像前処理"""
        img = image.astype(np.float32)
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return img

    
    def _on_preprocessing1_finished(self, result):
        if result is not None:
            detections, processed_stack = result
            if detections is not None and len(detections) > 0:
                self.detection_summary = detections
                self.processed_image_stack = processed_stack # 新しい画像スタックを保存
                
                self.detections_label.setText(str(len(detections)))
                self._update_status("Step 1: Preprocessing 1 finished.", color="green")
                self._set_buttons_enabled(True, True, False)
            else:
                self._handle_error("No peaks detected with current parameters.")

    # ▼▼▼ [Heath] measureFRC.m -- Fourier ring correlation の分解能測定 ▼▼▼
    HZ_FRC_EXPAND = 5          # Heath の Workbook_LAFM と同じ値
    HZ_FRC_IMG_GAUS = 0.4      # measureFRC.m の img_gaus

    def run_measure_frc(self):
        if getattr(self, 'detection_summary', None) is None or len(self.detection_summary) == 0:
            self._handle_error("Run Preprocessing 1 first (FRC needs the localizations).")
            return
        self._collect_params()
        self._update_status("Measuring resolution (FRC)...", color="darkorange")
        self.progress_bar.setRange(0, 100)
        self._run_in_thread(self._execute_measure_frc, self._on_measure_frc_finished,
                            self.detection_summary, self.params)

    def _execute_measure_frc(self, detection_summary, params, progress_signal=None, plot_signal=None):
        """[Heath] measureFRC.m. Uses the same expand=5 / img_gaus=0.4 as the workbook."""
        if progress_signal:
            progress_signal.emit(10, "Building localization maps...")
        d = np.asarray(detection_summary, dtype=float)
        good = np.isfinite(d[:, 0]) & np.isfinite(d[:, 1])
        d = d[good]
        if len(d) < 10:
            raise ValueError("Not enough valid localizations for FRC.")

        expand = self.HZ_FRC_EXPAND
        si = self.scale_info if isinstance(self.scale_info, dict) else {}
        nm_per_px = float(si.get('dx', 1.0) or 1.0)             # resampled stack, nm/px
        # Heath: locs(:,1:2) = locs(:,1:2) - min + 1, then round(locs*expand)
        y = d[:, 0] - np.min(d[:, 0]) + 1.0
        x = d[:, 1] - np.min(d[:, 1]) + 1.0
        py = np.round(y * expand).astype(int)
        px = np.round(x * expand).astype(int)
        gh, gw = int(py.max()) + 5, int(px.max()) + 5
        frames = d[:, 3].astype(int)

        if progress_signal:
            progress_signal.emit(40, f"FRC over {len(np.unique(frames))} frames...")
        q, frc_mean, av, sd = hz_measure_frc(
            py, px, frames, gh, gw,
            nm_per_px=nm_per_px / expand,        # fine-grid nm/px
            runs=20, img_gaus_expanded=self.HZ_FRC_IMG_GAUS * expand)
        if progress_signal:
            progress_signal.emit(100, "FRC finished.")
        return q, frc_mean, av, sd, len(d), nm_per_px

    def _on_measure_frc_finished(self, result):
        if result is None:
            return
        q, frc_mean, av, sd, n, nm_per_px = result
        if not np.isfinite(av):
            self.frc_label.setText("could not determine")
            self._update_status("FRC: no 1/7 crossing found.", color="red")
            return
        self.frc_result = {'q': q, 'frc': frc_mean, 'resolution_nm': av, 'sd_nm': sd,
                           'n_localizations': int(n), 'nm_per_px': nm_per_px,
                           'expand': self.HZ_FRC_EXPAND}
        self.frc_label.setText(f"{av:.2f} +/- {sd:.2f} nm  (n={n})")
        self._update_status(f"FRC resolution {av:.2f} +/- {sd:.2f} nm from {n} localizations.",
                            color="green")
        print(f"[Heath] FRC: {av:.3f} +/- {sd:.3f} nm, {n} localizations, "
              f"source pixel {nm_per_px:.4f} nm, expand {self.HZ_FRC_EXPAND}. "
              f"Half-dataset split, so this is dataset reproducibility, not absolute accuracy.")

    def run_preprocessing2(self):
        self._collect_params()
        self._update_status("Step 2: Reconstructing...", color="darkorange")
        self.btn_save.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self._run_in_thread(
            self._execute_preprocessing2, 
            self._on_preprocessing2_finished, 
            self.detection_summary, 
            self.processed_image_stack, # 元画像のshapeではなく、新しい画像スタックそのものを渡す
            self.params
        )

    def _on_preprocessing2_finished(self, results):
        if results is not None:
            # 手順1で追加した戻り値を受け取るようにアンパック処理を修正します。
            self.reconstruction, self.reconstruction_image, self.reconst_scan_size = results

            # 表示用のアスペクト比計算のために、受け取った物理サイズをdisplay_image_scan_sizeに設定します。
            # これにより、間違った物理サイズの再計算が不要になります。
            self.display_image_scan_size = self.reconst_scan_size
 
            self.reconst_size_label.setText(f"{self.reconstruction.shape[0]} x {self.reconstruction.shape[1]}")
            self._update_status("Step 2: Reconstruction finished.", color="green")
            self._set_buttons_enabled(True, True, True)
            if self.params.get('mode', '2D') == '2D':
                display_img = np.sum(self.reconstruction, axis=2)
            else:
                display_img = np.max(self.reconstruction, axis=2)
            self._display_image(display_img, target='bottom')

    def run_make_lafm_image(self):
        self._collect_params()
        self._update_status("Step 3: Making final LAFM image...", color="darkorange")
        self.btn_save.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self._run_in_thread(self._execute_make_lafm_image, self._on_make_lafm_image_finished, self.reconstruction, self.reconstruction_image, self.params)
        
    def _on_make_lafm_image_finished(self, result):
        if result is not None:
            self.final_lafm_image = result

             # 保存と表示で使うために、最終画像の物理サイズを計算して保存する
            try:
                # Preprocessing 2で計算・保存した正しい物理サイズをここで使います。
                if hasattr(self, 'reconst_scan_size') and self.reconst_scan_size is not None:
                    self.lafm_image_scan_size = self.reconst_scan_size
                    final_phys_w = self.lafm_image_scan_size['x']
                    final_phys_h = self.lafm_image_scan_size['y']
            
                else:
                    # フォールバック処理（通常は実行されません）
                    raise ValueError("Reconstructed scan size not found.")

            except Exception as e:
                print(f"[ERROR] Could not set final image physical size: {e}")
                self.lafm_image_scan_size = None # 計算失敗時はNoneに設定


            ct = getattr(self, 'sym_centre_translation', None)
            if ct:
                self.centring_label.setText("dx=%+.2f, dy=%+.2f px" % (ct[0], ct[1]))
            elif str(self.centring_combo.currentText()).lower().startswith('off'):
                self.centring_label.setText("off (array centre)")
            else:
                self.centring_label.setText("--")

            self._update_status("LAFM analysis completed!", color="green")
            self._display_image(self.final_lafm_image, target='bottom')
            self.btn_save.setEnabled(True)
            
            if self.show_3d_check.isChecked() and self.viewer_3d_window is not None and self.params.get('mode') == '3D':
                self._update_status("Updating 3D viewer...", color="info")
                # ▼▼▼【重要修正点】spacingの計算を削除し、呼び出しをシンプルに ▼▼▼
                self.viewer_3d_window.update_data(self.final_lafm_image)
            
        self._set_buttons_enabled(True, True, True)

    @QtCore.pyqtSlot()
    def _on_z_range_changed(self):
        self.z_min_spin.blockSignals(True)
        self.z_max_spin.blockSignals(True)
        z_min_val = self.z_min_spin.value()
        z_max_val = self.z_max_spin.value()
        if z_max_val < z_min_val:
            self.z_max_spin.setValue(z_min_val)
        self.z_max_spin.setMinimum(self.z_min_spin.value())
        self.z_min_spin.setMaximum(self.z_max_spin.value())
        self.z_min_spin.blockSignals(False)
        self.z_max_spin.blockSignals(False)

    @QtCore.pyqtSlot(int)
    def _on_filter_mode_changed(self, index):
        if index == 0:
            self.std_dev_label.setVisible(False)
            self.std_dev_factor_spin.setVisible(False)
            self.z_min_label.setText("Z_min (nm):")
        elif index == 1:
            self.std_dev_label.setVisible(True)
            self.std_dev_factor_spin.setVisible(True)
            if self.imagej_compat_check.isChecked():
                self.std_dev_label.setText("Noise Tolerance (%):")
                self.std_dev_factor_spin.setRange(0.0, 100.0)
                self.std_dev_factor_spin.setSingleStep(0.5)
            else:
                self.std_dev_label.setText("N factor:")
                self.std_dev_factor_spin.setRange(-5.0, 20.0)
                self.std_dev_factor_spin.setSingleStep(0.1)
            self.z_min_label.setText("Z_min (nm, optional):")

    @QtCore.pyqtSlot(bool)
    def _on_imagej_compat_changed(self, checked):
        if checked and self.filter_mode_combo.currentIndex() == 1:
            self.std_dev_label.setText("Noise Tolerance (%):")
            self.std_dev_factor_spin.setRange(0.0, 100.0)
            self.std_dev_factor_spin.setSingleStep(0.5)
            if self.std_dev_factor_spin.value() < 0:
                self.std_dev_factor_spin.setValue(5.0)
        elif self.filter_mode_combo.currentIndex() == 1:
            self.std_dev_label.setText("N factor:")
            self.std_dev_factor_spin.setRange(-5.0, 20.0)
            self.std_dev_factor_spin.setSingleStep(0.1)
    
    def _create_lafm_lut(self):
        color_stops = [(0, (0, 0, 0)), (85, (100, 0, 120)), (170, (255, 100, 0)), (220, (255, 255, 0)), (255, (255, 255, 255))]
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(len(color_stops) - 1):
            start_index, start_color_rgb = color_stops[i]; end_index, end_color_rgb = color_stops[i+1]
            start_color_bgr = (start_color_rgb[2], start_color_rgb[1], start_color_rgb[0]); end_color_bgr = (end_color_rgb[2], end_color_rgb[1], end_color_rgb[0])
            for j in range(start_index, end_index + 1):
                ratio = (j - start_index) / (end_index - start_index)
                b = int(start_color_bgr[0]*(1.0-ratio) + end_color_bgr[0]*ratio); g = int(start_color_bgr[1]*(1.0-ratio) + end_color_bgr[1]*ratio); r = int(start_color_bgr[2]*(1.0-ratio) + end_color_bgr[2]*ratio)
                lut[j, 0] = [b, g, r]
        return lut

    
    def _execute_preprocessing1(self, image_stack, params, progress_signal=None, plot_signal=None):
        """【最終FIX D】物理クロップ + ピクセルリサンプリング後、ピーク検出と可視化を行う完全版"""
        
        try:
            if progress_signal:
                progress_signal.emit(5, "Initializing Preprocessing 1...")

            # --- ステップ1: パラメータと初期設定 ---
            self.scale_info['offset_x'] = 0.0
            self.scale_info['offset_y'] = 0.0
            all_detections = []

            # --- ステップ2: 物理サイズに基づいて画像を正方形にクロップ ---
            h_orig, w_orig, num_frames = image_stack.shape
            phys_side_length = min(gv.XScanSize, gv.YScanSize)
            nm_per_pixel_x = gv.XScanSize / w_orig
            nm_per_pixel_y = gv.YScanSize / h_orig

            crop_w_px = int(round(phys_side_length / nm_per_pixel_x))
            crop_h_px = int(round(phys_side_length / nm_per_pixel_y))

            # クロップサイズが元画像より大きい場合の処理
            if crop_w_px > w_orig or crop_h_px > h_orig:
                # 元画像のサイズに合わせてクロップサイズを調整
                crop_w_px = min(crop_w_px, w_orig)
                crop_h_px = min(crop_h_px, h_orig)
        

            start_x = max(0, (w_orig - crop_w_px) // 2)
            start_y = max(0, (h_orig - crop_h_px) // 2)

            # クロップ範囲のチェック
            if start_x + crop_w_px > w_orig or start_y + crop_h_px > h_orig:
                print(f"[ERROR] Invalid crop range: start_x={start_x}, start_y={start_y}, crop_w={crop_w_px}, crop_h={crop_h_px}, w_orig={w_orig}, h_orig={h_orig}")
                if hasattr(self, 'error'): self.error.emit("Invalid crop range detected.")
                return None, None

    
            image_stack_cropped = image_stack[start_y:start_y+crop_h_px, start_x:start_x+crop_w_px, :]
            
            # クロップ後のサイズチェック
            if image_stack_cropped.size == 0:
                print(f"[ERROR] Cropped image is empty")
                if hasattr(self, 'error'): self.error.emit("Cropped image is empty.")
                return None, None

            # --- ステップ3: ピクセル数が正方形になるようにリサンプリング ---
            if progress_signal:
                progress_signal.emit(10, "Resampling to square pixels...")

            target_pixel_size = max(crop_w_px, crop_h_px)
    
            
            resampled_stack = np.zeros((target_pixel_size, target_pixel_size, num_frames), dtype=np.float32)

            for i in range(num_frames):
                frame_cropped = image_stack_cropped[:, :, i]
                if frame_cropped.size == 0:
                    print(f"[ERROR] Frame {i} is empty after cropping")
                    continue
                    
                try:
                    resampled_stack[:, :, i] = cv2.resize(
                        frame_cropped, 
                        (target_pixel_size, target_pixel_size), 
                        interpolation=cv2.INTER_CUBIC
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to resize frame {i}: {e}")
                    if hasattr(self, 'error'): self.error.emit(f"Failed to resize frame {i}: {e}")
                    return None, None

            # リサンプリング後のチェック
            if np.all(resampled_stack == 0):
                print(f"[ERROR] All frames are zero after resampling")
                if hasattr(self, 'error'): self.error.emit("All frames are zero after resampling.")
                return None, None
            
    

            # --- ステップ4: スケール情報を更新 ---
            new_nm_per_pixel = phys_side_length / target_pixel_size
            self.scale_info['dx'] = new_nm_per_pixel
            self.scale_info['dy'] = new_nm_per_pixel
            self.scale_info['offset_x'] += start_x * nm_per_pixel_x
            self.scale_info['offset_y'] += start_y * nm_per_pixel_y

             # --- ステップB: ドリフト補正 ---
            corrected_stack = resampled_stack
            if params.get('drift_correction', False):
                if progress_signal:
                    progress_signal.emit(10, "Applying drift correction...")
                
                try:
                    # averagingモジュールから新しいヘルパー関数をインポート
                    from averaging import calculate_drift_matrices
                    
                    is_rot = "Feature-based" in params['drift_algorithm']
                    conf_thresh = params['drift_threshold']
                    subpixel_precision = max(1, int(params.get('drift_subpixel_precision', 10)))
                    
                    # ヘルパー関数を呼び出して変換行列を計算
                    matrices, confidences = calculate_drift_matrices(
                        resampled_stack, 
                        is_rotation_enabled=is_rot,
                        confidence_threshold=conf_thresh,
                        phase_upsample_factor=subpixel_precision,
                    )
                    
                    # 信頼度フィルタリング
                    good_indices = np.where(confidences > conf_thresh)[0]
                    
                    if len(good_indices) < 2:
                        print("Warning: Not enough stable frames. Skipping drift correction.")
                    else:
                        stack_to_correct = resampled_stack[:, :, good_indices]
                        matrices_to_apply = matrices[good_indices]
                        
                        if progress_signal:
                            progress_signal.emit(15, f"Applying corrections to {len(good_indices)} frames...")
                        
                        # 変換行列を適用
                        h, w = stack_to_correct.shape[:2]
                        corrected_frames = []
                        # np.rollaxisを使って正しくフレームをループ処理
                        for i, (img, M) in enumerate(zip(np.rollaxis(stack_to_correct, 2), matrices_to_apply)):
                            border_val = float(np.median(img))
                            corrected_frame = cv2.warpAffine(
                                img.astype(np.float32), M, (w, h), 
                                borderValue=border_val
                            )
                            corrected_frames.append(corrected_frame)
                        
                        corrected_stack = np.stack(corrected_frames, axis=2)
                        
                        excluded_frames = resampled_stack.shape[2] - len(good_indices)
                        if excluded_frames > 0 and progress_signal:
                            progress_signal.emit(18, f"Drift correction: {excluded_frames} frames excluded")
                            
                except Exception as e:
                    if progress_signal:
                        progress_signal.emit(18, f"Drift correction failed: {str(e)[:50]}...")
                    print(f"Drift correction failed: {e}")
                    # エラー時は元のスタックを使用
                    corrected_stack = resampled_stack

            # --- ステップB2: [Heath] filter_movie 前処理 ---
            # 検出/局在は detect_stack（フィルタ後、rescale で 0-1）で行い、
            # 高さ(nm)と強度重みは corrected_stack（未フィルタ）から取る。
            # これにより Z_min/Z_max の nm 指定が意味を保つ。
            prefilter_on = bool(params.get('prefilter_on', False))
            detect_stack = corrected_stack
            if prefilter_on:
                if progress_signal:
                    progress_signal.emit(19, "Applying Heath pre-filter (filter_movie)...")
                try:
                    detect_stack = hz_filter_movie(
                        corrected_stack,
                        gauss_sigma=float(params.get('prefilter_gauss', 0.2)),
                        laplacian_strength=float(params.get('prefilter_laplacian', 0.0)),
                    )
                    # filter_movie の Laplacian 分岐は rescale 済みだが、Gaussian のみの
                    # 場合はスケールが元のままなので、閾値を 0-1 で扱えるよう揃える。
                    detect_stack = _hz_rescale(detect_stack)
                    spike = float(np.max(corrected_stack.max(axis=(0, 1)))) / max(
                        1e-12, float(np.median(corrected_stack.max(axis=(0, 1)))))
                    if spike > 2.0:
                        print(f"[WARNING] Pre-filter: the brightest frame is {spike:.1f}x the median "
                              f"frame maximum. rescale() spans the whole stack, so spiking frames "
                              f"compress every other frame and the detection threshold becomes "
                              f"unreachable. Clean the stack (reject spike frames) first.")
                except Exception as e:
                    print(f"[ERROR] Heath pre-filter failed, continuing unfiltered: {e}")
                    detect_stack = corrected_stack
                    prefilter_on = False

            # --- ステップ5: ピーク検出処理 ---
            height, width = target_pixel_size, target_pixel_size
            



            num_corrected_frames = corrected_stack.shape[2]
            all_detections = [] # 検出結果を初期化

            for i in range(num_corrected_frames):
                if progress_signal:
                    progress_signal.emit(int(20 + 80 * i / num_corrected_frames), f"Detecting peaks in frame {i+1}/{num_corrected_frames}")

                frame_abs = corrected_stack[:, :, i]
                # 検出に使う面。前処理フィルタ ON なら 0-1 にスケールされたフィルタ後の面。
                frame_det = detect_stack[:, :, i]

                # フレームの状態チェック
                if frame_abs.size == 0 or np.all(frame_abs == 0):
                    print(f"[WARNING] Frame {i} is empty or all zero")
                    continue



                frame_rel = frame_abs - np.min(frame_abs)

                # A, B, C: 高さ、局所最大値、空間フィルタリング
                threshold = -np.inf
                if params['filter_mode'] == 'Statistics (Mean + N x Std Dev)' and np.std(frame_rel) > 1e-9:
                    if params.get('imagej_compat_mode', False):
                        threshold = np.mean(frame_rel) * params['std_dev_factor'] / 100.0
                    else:
                        threshold = np.mean(frame_rel) + (params['std_dev_factor'] * np.std(frame_rel))


                height_mask = (frame_rel >= threshold) & (frame_abs >= params['z_min']) & (frame_abs <= params['z_max'])
                if prefilter_on:
                    # [Heath] LAFM_thresh はフィルタ後 0-1 の面に対する閾値。
                    # nm の Z_min/Z_max は未フィルタ面に対して引き続き有効。
                    height_mask = height_mask & (frame_det > float(params.get('heath_thresh', 0.5)))

                
                # 各条件の詳細を出力（最初のフレームのみ）
                if i == 0:
                    threshold_mask = (frame_rel >= threshold)
                    z_min_mask = (frame_abs >= params['z_min'])
                    z_max_mask = (frame_abs <= params['z_max'])
    


                search_size = max(3, int(params.get('search_size', 3)))
                if search_size % 2 == 0:
                    search_size += 1
                if params['connectivity'] == 8:
                    footprint = np.ones((search_size, search_size), dtype=bool)
                else:
                    radius = search_size // 2
                    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
                    footprint = (np.abs(xx) + np.abs(yy)) <= radius
                # 前処理フィルタ OFF のとき frame_det is frame_abs なので従来と同一。
                maxima_mask = (frame_det == maximum_filter(frame_det, footprint=footprint, mode='constant', cval=0.0))
                if prefilter_on:
                    # [Heath] Fast_peaks2D は外周 2 px を必ず除外する
                    maxima_mask[:2, :] = False
                    maxima_mask[-3:, :] = False
                    maxima_mask[:, :2] = False
                    maxima_mask[:, -3:] = False


                center_x, center_y = width / 2, height / 2
                crop_radius_sq = (min(width, height) / 2 * params['crop_ratio'])**2
                y_coords, x_coords = np.ogrid[:height, :width]
                spatial_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) < crop_radius_sq

                roi_mask = np.ones_like(frame_abs, dtype=bool)
                if params.get('imagej_compat_mode', False):
                    frame_8u = cv2.normalize(frame_rel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    _, roi_bin = cv2.threshold(frame_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    roi_mask = roi_bin.astype(bool)

                
                # D: 最終的なピークマスクと座標
                final_peaks_mask = height_mask & maxima_mask & spatial_mask & roi_mask
                peak_coords_y, peak_coords_x = np.where(final_peaks_mask)
                final_maxima_coords_int = list(zip(peak_coords_y, peak_coords_x))
                

                
                # 最初のフレームでより詳細な情報を出力
                if i == 0:
                    pass    


                # E: サブピクセル処理または整数座標の格納
                if params['subpixel_on']:
                    refined_detections_for_frame = []
                    radius, scale = 2, params['subpixel_scale']
                    subpix_method = str(params.get('subpix_method', 'Interpolation (pyNuD)'))
                    # 局在は検出面（フィルタ後があればそちら）で行う。Heath の
                    # localize(im2, ...) と同じ扱い。高さは未フィルタ面から取る。
                    for y_int, x_int in final_maxima_coords_int:
                        # 既定パスでは列 4/5 を 0.0 のままにして従来の検出配列と完全一致させる。
                        # Heath のフィット系のみ sigma / amplitude を書き込む。
                        sigma_fit, amp_fit = 0.0, 0.0
                        if subpix_method.startswith('Heath bicubic'):
                            sub_y, sub_x = hz_localize_bicubic(frame_det, y_int, x_int)
                        elif subpix_method.startswith('Gaussian fit'):
                            sub_y, sub_x, sigma_fit, amp_fit = hz_localize_gaussian(
                                frame_det, y_int, x_int, params.get('pixperfeat', 1.0))
                        elif subpix_method.startswith('Sphere fit'):
                            sub_y, sub_x, sigma_fit = hz_localize_sphere(
                                frame_det, y_int, x_int, params.get('pixperfeat', 1.0))
                        else:
                            # 従来の pyNuD 補間（既定）
                            y_start, y_end = max(0, y_int - radius), min(height, y_int + radius + 1)
                            x_start, x_end = max(0, x_int - radius), min(width, x_int + radius + 1)
                            roi = frame_det[y_start:y_end, x_start:x_end]
                            if roi.size == 0:
                                continue
                            zoomed_roi = zoom(roi, scale, order=3)
                            max_coords_local = np.unravel_index(np.argmax(zoomed_roi), zoomed_roi.shape)
                            sub_y = y_start + max_coords_local[0] / scale
                            sub_x = x_start + max_coords_local[1] / scale
                        if not (np.isfinite(sub_y) and np.isfinite(sub_x)):
                            continue    # Heath の guard に掛かった端の点
                        all_detections.append([sub_y, sub_x, frame_abs[y_int, x_int], i,
                                               sigma_fit, amp_fit, 0.0, 1.0])
                        refined_detections_for_frame.append((sub_y, sub_x))
                    
                    if plot_signal:
                        display_frame = cv2.normalize(frame_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        plot_img = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                        for sub_y, sub_x in refined_detections_for_frame:
                            cv2.circle(plot_img, (int(round(sub_x)), int(round(sub_y))), 1, (0, 255, 255), -1)
                        plot_signal.emit(plot_img, 'top')
                        if params['vis_delay_spin'] > 0: time.sleep(params['vis_delay_spin'] / 1000.0)
                else: # サブピクセルOFFの場合
                    for y, x in final_maxima_coords_int:
                        all_detections.append([float(y), float(x), frame_abs[y, x], i, 0.0, 0.0, 0.0, 1.0])
                    
                    if plot_signal:
                        display_frame = cv2.normalize(frame_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        plot_img = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                        for y_coord, x_coord in final_maxima_coords_int:
                            cv2.circle(plot_img, (x_coord, y_coord), 1, (0, 0, 255), -1)
                        plot_signal.emit(plot_img, 'top')
                        if params['vis_delay_spin'] > 0: time.sleep(params['vis_delay_spin'] / 1000.0)

            # --- ステップ6: 最終処理 ---
            detections = np.array(all_detections)
    
            
            if len(detections) == 0:
                print(f"[ERROR] No peaks detected in any frame")
                if hasattr(self, 'error'): self.error.emit("No peaks detected.")
                return None, None # 2つの値を返す
            
            if progress_signal: progress_signal.emit(100, "Preprocessing 1 Finished.")
    
            return detections, corrected_stack

        except Exception as e:
            import traceback
            error_msg = f"Error in Preprocessing 1: {e}\n\n{traceback.format_exc()}"
            if hasattr(self, 'error'): self.error.emit(error_msg)
            else: print(error_msg)
            return None, None # エラー時も2つの値を返す

    def _execute_preprocessing2(self, detection_summary, processed_image_stack, params, progress_signal=None, plot_signal=None):
        """【完成版】渡された画像スタックを基準に再構成する"""
        
        # --- ステップ1: パラメータとグリッドサイズの準備 ---
        if progress_signal:
            progress_signal.emit(5, "Initializing reconstruction grid...")

        # 渡された画像スタック(Preprocessing 1で処理済み)から情報を取得
        h_proc, w_proc, total_frames = processed_image_stack.shape
        is_3d_mode = (params['mode'] == '3D')

        # 再構成後の物理サイズを計算 (scale_infoは更新済み)
        scan_size_x = w_proc * self.scale_info['dx']
        scan_size_y = h_proc * self.scale_info['dy']

        # 新しいグリッドのピクセル数を計算
        reconst_w, reconst_h = w_proc, h_proc
        if params['subpixel_on']:
            xy_res = params['subpixel_xy_res']
            expand = max(1, int(params.get('subpixel_expand', 1)))

            if xy_res > 0:
                reconst_w = max(reconst_w, int(round(scan_size_x / xy_res)))
                reconst_h = max(reconst_h, int(round(scan_size_y / xy_res)))

            if expand > 1:
                reconst_w = max(reconst_w, int(round(w_proc * expand)))
                reconst_h = max(reconst_h, int(round(h_proc * expand)))

        reconst_w = max(1, int(reconst_w))
        reconst_h = max(1, int(reconst_h))

        reconst_dx = scan_size_x / reconst_w
        reconst_dy = scan_size_y / reconst_h

        # --- ステップ2: 再構成用グリッドの作成 ---
        reconstruction_grid = None
        reconstruction_image = None # 2Dモードで使う、強度重み付け用の画像

        if is_3d_mode:
            # 3Dモード：ボクセルグリッドを作成
            z_res = params['subpixel_z_res']
            z_values = detection_summary[:, 2]
            z_min, z_max = np.min(z_values), np.max(z_values)
            
            num_z_bins = 1
            if z_res > 0 and (z_max > z_min):
                num_z_bins = int(np.ceil((z_max - z_min) / z_res))
            
            reconstruction_grid = np.zeros((reconst_h, reconst_w, num_z_bins))
        else: # 2Dモードの場合
            reconstruction_grid = np.zeros((reconst_h, reconst_w, total_frames))
            # 強度計算用に、処理済みの画像スタックをそのまま代入する (forループは不要)
            reconstruction_image = np.zeros((reconst_h, reconst_w, total_frames))

             # ループを回して、各フレームを新しい解像度にリサイズする
            for i in range(total_frames):
                if progress_signal:
                    progress_signal.emit(int(10 + 20 * i / total_frames), f"Upscaling original image {i+1}")
                frame_to_resize = processed_image_stack[:, :, i]
                reconstruction_image[:, :, i] = cv2.resize(
                    frame_to_resize, (reconst_w, reconst_h), interpolation=cv2.INTER_CUBIC)

 

        # --- ステップ3: 全ての検出点を新しいグリッドにマッピング ---
        # [Heath] 密度レンダリング用に、細グリッド上の整数座標・高さ・フレームを保持する。
        # Heath の LAFM_renderer はフレーム軸を持たず全局在をプールするため、
        # pyNuD の (h, w, frame) グリッドとは別に控えておく必要がある。
        hz_py, hz_px, hz_z, hz_fr = [], [], [], []
        num_detections = len(detection_summary)
        for idx, detection in enumerate(detection_summary):
            if progress_signal and idx % 1000 == 0:
                progress_signal.emit(int(30 + 50 * idx / num_detections), f"Mapping detection {idx+1}")

            y_orig_px, x_orig_px, z_abs_nm, frame_idx = detection[0], detection[1], detection[2], int(detection[3])

            x_nm = x_orig_px * self.scale_info['dx']
            y_nm = y_orig_px * self.scale_info['dy']

            pixel_x = int(round(x_nm / reconst_dx))
            pixel_y = int(round(y_nm / reconst_dy))

            if not (0 <= pixel_y < reconst_h and 0 <= pixel_x < reconst_w):
                continue

            hz_py.append(pixel_y); hz_px.append(pixel_x)
            hz_z.append(z_abs_nm); hz_fr.append(frame_idx)

            if is_3d_mode:
                voxel_z = 0
                if z_res > 0 and (z_max > z_min):
                    voxel_z = int((z_abs_nm - z_min) / z_res)
                if 0 <= voxel_z < num_z_bins:
                    reconstruction_grid[pixel_y, pixel_x, voxel_z] += 1
            else: # 2Dモード
                if 0 <= frame_idx < total_frames:
                    reconstruction_grid[pixel_y, pixel_x, frame_idx] = 1

        # --- ステップ4: 対称化処理 (オプション) ---
        if params['sym_on'] and params['sym_on_prep2'] and params['sym_order'] > 1:
            if progress_signal: progress_signal.emit(85, "Applying symmetry...")

            order = params['sym_order']
            # NOTE ON INTERPOLATION ORDER: this stage rotates SPARSE binary localization maps,
            # not a smooth image. Bicubic (Heath's choice in rotation_sym.m, which he applies to
            # a smooth reference) rings on isolated deltas and produces negative side lobes, so
            # bilinear is kept here deliberately. The final-image stage does use bicubic.
            # The symmetry centre is found ONCE from the collapsed density -- a single sparse
            # slice carries too little signal for the cross-correlation to lock on.
            ct = None
            _cm = _hz_centring_method(params)
            if not str(_cm).lower().startswith('off'):
                try:
                    collapsed = reconstruction_grid.sum(axis=2).astype(np.float64)
                    collapsed = gaussian_filter(collapsed, max(1.0, min(collapsed.shape) / 100.0))
                    ct = hz_find_centre(_cm, collapsed, fold=order, align_exp=10)
                    if ct:
                        print(f"[Centring] Prep-2 ({_cm}) offset: dx={ct[0]:+.2f}, dy={ct[1]:+.2f} px")
                except Exception as e:
                    print(f"[WARNING] Prep-2 centring failed, using the array centre: {e}")
                    ct = None

            avg_reconstruction = np.zeros_like(reconstruction_grid)
            num_slices = reconstruction_grid.shape[2]
            for i in range(num_slices):
                original_slice = reconstruction_grid[:, :, i]
                if not np.any(original_slice):
                    avg_reconstruction[:, :, i] = original_slice
                    continue
                avg_reconstruction[:, :, i] = hz_symmetrise(original_slice, order, ct, interp_order=1)
            reconstruction_grid = avg_reconstruction

        # --- ステップ5: 可視化と終了処理 ---
        if plot_signal:
            display_img = np.max(reconstruction_grid, axis=2) if is_3d_mode else np.sum(reconstruction_grid, axis=2)
            plot_signal.emit(display_img, 'bottom')
            if params.get('vis_delay_spin', 0) > 0: time.sleep(params['vis_delay_spin'] / 1000.0)

        if progress_signal: progress_signal.emit(100, "Preprocessing 2 Finished.")
            
        # [Heath] 密度レンダリング / FRC が使う細グリッド座標を保存
        self.hz_grid = {
            'py': np.asarray(hz_py, dtype=int),
            'px': np.asarray(hz_px, dtype=int),
            'z': np.asarray(hz_z, dtype=float),
            'frame': np.asarray(hz_fr, dtype=int),
            'shape': (reconst_h, reconst_w),
            'dx': reconst_dx, 'dy': reconst_dy,
            'expand_eff': (reconst_w / w_proc) if w_proc else 1.0,
        }

        reconst_scan_size = {'x': scan_size_x, 'y': scan_size_y}
        return reconstruction_grid, reconstruction_image, reconst_scan_size

    def _execute_make_lafm_image(self, reconstruction, reconstruction_image, params, progress_signal=None, plot_signal=None):
        heath_render = str(params.get('render_mode', '')).startswith('Heath')

        if params['mode'] == '2D' and heath_render:
            # ▼▼▼ [Heath] LAFM_renderer.m (prob=1): 局在密度マップ ▼▼▼
            # pyNuD 方式との違い: 高さを強度として掛けず、全フレームをプールした
            # 局在密度そのものを返す。高さは色レベルのビニングにのみ効く。
            if progress_signal:
                progress_signal.emit(10, "Constructing Heath localization-density LAFM image...")
            g = getattr(self, 'hz_grid', None)
            if not g or g['py'].size == 0:
                raise ValueError("Heath rendering needs Preprocessing 2 to have mapped detections.")
            sigma = float(params['blur_sigma_xy'])
            hpy, hpx, hz = g['py'], g['px'], g['z']

            # "During Reconstruction (Prep 2)" in Heath mode: symmetrise the LOCALIZATION
            # COORDINATES, not a rasterised map. Rotating exact coordinates carries no
            # interpolation error at all, so this is the cleanest symmetrisation available --
            # strictly better than rotating the sparse binary grid as the pyNuD path does.
            if params['sym_on'] and params.get('sym_on_prep2') and params['sym_order'] > 1:
                order = int(params['sym_order'])
                gh, gw = g['shape']
                cy, cx = gh / 2.0, gw / 2.0
                _cm = _hz_centring_method(params)
                if not str(_cm).lower().startswith('off'):
                    try:
                        prov, _ = hz_render_density(hpy, hpx, hz, (gh, gw), sigma, n_levels=1)
                        _t = hz_find_centre(_cm, prov, fold=order, align_exp=10)
                        if _t:
                            dx, dy = _t
                            cy, cx = gh / 2.0 + dy, gw / 2.0 + dx
                            self.sym_centre_translation = (dx, dy)
                            print(f"[Centring] Prep-2 ({_cm}) axis at ({cy:.1f}, {cx:.1f}) px, "
                                  f"offset dx={dx:+.2f} dy={dy:+.2f}")
                    except Exception as e:
                        print(f"[WARNING] Prep-2 centring failed, using the array centre: {e}")
                ys, xs, zs = [hpy.astype(float)], [hpx.astype(float)], [hz]
                for j in range(1, order):
                    a = np.deg2rad(j * 360.0 / order)
                    dyy, dxx = hpy - cy, hpx - cx
                    ys.append(cy + dyy * np.cos(a) - dxx * np.sin(a))
                    xs.append(cx + dyy * np.sin(a) + dxx * np.cos(a))
                    zs.append(hz)
                n_before = hpy.size
                hpy = np.clip(np.round(np.concatenate(ys)), 0, gh - 1).astype(int)
                hpx = np.clip(np.round(np.concatenate(xs)), 0, gw - 1).astype(int)
                hz = np.concatenate(zs)
                print(f"[Heath] Prep-2 C{order} coordinate symmetrisation: {n_before} -> "
                      f"{hpy.size} localizations (exact rotation, no interpolation error)")
                if params.get('sym_on_final'):
                    print(f"[WARNING] Both Prep-2 and Final symmetry are on. The coordinates are "
                          f"already C{order} symmetric, so the final image rotation only adds "
                          f"interpolation blur. Turn one of them off.")

            final_image, cl = hz_render_density(
                hpy, hpx, hz, g['shape'], sigma,
                n_levels=256, colorlimit_mode='Exc outliers')
            final_image = final_image.astype(np.float32)
            self.hz_zlims = cl
            print(f"[Heath] density render: {hpy.size} localizations, sigma={sigma:.2f} px "
                  f"(Heath workbook equivalent img_gus*expand/2 = {params['blur_sigma_xy']:.2f}), "
                  f"height colour limits {cl[0]:.3f}-{cl[1]:.3f} nm")
            if plot_signal:
                plot_signal.emit(final_image, 'bottom')
            if progress_signal:
                progress_signal.emit(90, "Heath density render finished.")

        elif params['mode'] == '2D':
            if progress_signal: progress_signal.emit(10, "Constructing 2D LAFM image...")
            num_frames = reconstruction.shape[2]
            final_image = np.zeros(reconstruction.shape[:2], dtype=np.float32)
            sigma = params['blur_sigma_xy']
            
            for i in range(num_frames):
                if progress_signal:
                    progress_signal.emit(int(10 + 80 * i / num_frames), f"Processing frame {i+1}/{num_frames}")
                
                probability_wave = reconstruction[:, :, i]
                
                if np.any(probability_wave):
                    blurred_prob = gaussian_filter(probability_wave, sigma=sigma)
                    if params.get('imagej_compat_mode', False):
                        # ImageJ macro compatibility:
                        # Divide by 40.58, then multiply by sigma^2.
                        blurred_prob = (blurred_prob / 40.58) * (sigma * sigma)
                    if plot_signal:
                        display_prob = blurred_prob / np.max(blurred_prob) if np.max(blurred_prob) > 0 else blurred_prob
                        plot_signal.emit(display_prob, 'top')

                    intensity_frame = reconstruction_image[:, :, i]
                    intensity_frame_norm = intensity_frame - np.min(intensity_frame)
                    final_image += (blurred_prob * intensity_frame_norm)
                
                if plot_signal and (i % 5 == 0 or i == num_frames - 1):
                    processed_frames = i + 1
                    display_avg_image = final_image / processed_frames if processed_frames > 0 else final_image
                    plot_signal.emit(display_avg_image, 'bottom')
            
            if num_frames > 0:
                final_image /= num_frames
            
        else: # 3D Mode
            if progress_signal: progress_signal.emit(10, "Applying 3D Gaussian Blur...")
            sigma_xy = params['blur_sigma_xy']
            sigma_z = params['blur_sigma_z']
            final_image = gaussian_filter(reconstruction.astype(np.float32), sigma=(sigma_xy, sigma_xy, sigma_z))
            if plot_signal: plot_signal.emit(np.max(final_image, axis=2), 'bottom')
            if progress_signal: progress_signal.emit(80, "Blurring finished.")
        
        if params['sym_on'] and params['sym_on_final'] and params['sym_order'] > 1:
            if progress_signal: progress_signal.emit(90, "Applying post-symmetry...")
            order = params['sym_order']
            # [Heath] rotation_sym.m uses imrotate(..., 'bicubic'); the old order=0 (nearest)
            # aliased. Auto-centre reproduces FindCenterPositions.m, which MATLAB applies via
            # imtranslate(ref, -center_translation) BEFORE symmetrising.
            _cm = _hz_centring_method(params)
            autoc = not str(_cm).lower().startswith('off')
            self.sym_centre_translation = None

            if len(final_image.shape) == 2:
                ct = None
                if autoc:
                    try:
                        ct = hz_find_centre(_cm, final_image, fold=order, align_exp=10)
                        self.sym_centre_translation = ct
                        if ct:
                            print(f"[Centring] final image ({_cm}) offset: "
                                  f"dx={ct[0]:+.2f}, dy={ct[1]:+.2f} px")
                    except Exception as e:
                        print(f"[WARNING] centring failed, rotating about the array centre: {e}")
                        ct = None
                final_image = hz_symmetrise(final_image, order, ct, interp_order=3).astype(np.float32)
            else:  # 3D -- find the centre once on the max projection, apply to every slice
                ct = None
                if autoc:
                    try:
                        ct = hz_find_centre(_cm, np.max(final_image, axis=2), fold=order, align_exp=10)
                        self.sym_centre_translation = ct
                        if ct:
                            print(f"[Centring] 3D ({_cm}) offset from max projection: "
                                  f"dx={ct[0]:+.2f}, dy={ct[1]:+.2f} px")
                    except Exception as e:
                        print(f"[WARNING] centring failed: {e}")
                        ct = None
                avg_reconstruction = np.zeros_like(final_image)
                num_slices = final_image.shape[2]
                for k in range(num_slices):
                    avg_reconstruction[:, :, k] = hz_symmetrise(
                        final_image[:, :, k], order, ct, interp_order=3)
                    if plot_signal and (k % 5 == 0 or k == num_slices - 1):
                        plot_signal.emit(np.max(avg_reconstruction, axis=2), 'bottom')
                        if params.get('vis_delay_spin', 0) > 0: time.sleep(params['vis_delay_spin'] / 1000.0)
                final_image = avg_reconstruction
        
        if progress_signal: progress_signal.emit(100, "Final image created.")
        
        # --- ▼▼▼ 正しい戻り値 ▼▼▼ ---
        return final_image
    
    @QtCore.pyqtSlot(int)
    def _on_mode_changed(self, index):
        """Modeコンボボックスの変更に応じてUIを切り替える"""
        is_3d_mode = (self.mode_combo.currentText() == "3D")
        self.show_3d_check.setVisible(is_3d_mode)
        # 2Dモードに切り替えたら、3D表示はオフにする
        if not is_3d_mode and self.show_3d_check.isChecked():
            self.show_3d_check.setChecked(False)

    @QtCore.pyqtSlot(bool)
    def _handle_3d_display_toggle(self, checked):
        """「3D Display」チェックボックスの状態に応じてウィンドウを開閉する"""
        if not PYVISTA_AVAILABLE:
            self.show_3d_check.setChecked(False)
            detail = f"\n\n詳細: {PV_IMPORT_ERROR}" if PV_IMPORT_ERROR else ""
            _frozen = getattr(sys, "frozen", False)
            if _frozen:
                msg = (
                    "3D Display requires PyVista, PyVistaQt, and VTK.\n"
                    "3D表示には PyVista、PyVistaQt、VTK が必要です。\n\n"
                    "These modules are not installed. They are not bundled with this installation.\n"
                    "これらはインストールされていません。このパッケージに含まれていません。" + detail
                )
            else:
                msg = (
                    "3D Display requires PyVista, PyVistaQt, and VTK.\n"
                    "3D表示には PyVista、PyVistaQt、VTK が必要です。\n\n"
                    "Install with: pip install pyvista pyvistaqt\n"
                    "(VTK is installed automatically as a dependency of PyVista.)\n"
                    "インストール: pip install pyvista pyvistaqt\n"
                    "（VTK は PyVista の依存関係として自動でインストールされます。）\n\n"
                    "After installing, try enabling 3D Display again.\n"
                    "インストール後、再度 3D Display を有効にしてください。" + detail
                )
            QtWidgets.QMessageBox.critical(
                self, "Library Not Found / ライブラリが見つかりません", msg
            )
            return

        if checked:
            if self.final_lafm_image is None or self.params.get('mode') != '3D':
                self.show_3d_check.setChecked(False)
                QtWidgets.QMessageBox.warning(self, "No 3D Data", "表示する3D LAFMデータがありません。\n先に3Dモードで「Make LAFM Image」を実行してください。")
                return

            try:
                if self.viewer_3d_window is None:
                    self.viewer_3d_window = Voxel3DViewer(parent=None) # 親ウィンドウを渡す
                    # ウィンドウが閉じられたら、チェックを外し、設定を保存する
                    self.viewer_3d_window.was_closed.connect(self._on_3d_viewer_closed)

                # ▼▼▼【重要修正点】spacingの計算を削除し、呼び出しをシンプルに ▼▼▼
                self.viewer_3d_window.update_data(self.final_lafm_image)
                self.viewer_3d_window.show()
                self.viewer_3d_window.raise_()
            except Exception as e:
                print(f"[ERROR] Failed to create 3D viewer: {e}")
                import traceback
                traceback.print_exc()
                self.show_3d_check.setChecked(False)
                QtWidgets.QMessageBox.critical(self, "3D Viewer Error", f"3Dビューアの作成に失敗しました。\n\nエラー: {e}")
        else:
            if self.viewer_3d_window is not None:
                try:
                    self.viewer_3d_window.close()
                except Exception as e:
                    print(f"[ERROR] Failed to close 3D viewer: {e}")
                self.viewer_3d_window = None
    
    def _on_3d_viewer_closed(self):
        """3Dビューアが閉じられたときに呼び出されるスロット"""
        if self.viewer_3d_window:
            # 1. チェックボックスのチェックを外す
            self.show_3d_check.setChecked(False)
            
            # 2. ウィンドウ設定を保存
            try:
                if not hasattr(gv, 'windowSettings'): gv.windowSettings = {}
                gv.windowSettings[self.viewer_3d_window.__class__.__name__] = {
                    'visible': False
                }
                # メインの保存機能を呼び出す
                if self.main_window and hasattr(self.main_window, 'saveAllInitialParams'):
                    self.main_window.saveAllInitialParams()
            except Exception as e:
                print(f"Error saving 3D viewer settings: {e}")
            
            # 3. 参照をクリア
            self.viewer_3d_window = None

    def _save_lafm_as_asd(self, save_path, comment, image_data):
        """LAFMの2D画像を、輝度を正しく正規化してASD形式で保存する（堅牢版）"""
        try:
            # --- ステップ1: ヘッダー情報の準備 ---
            y_pixels, x_pixels = image_data.shape

            save_x_scan_size = int(self.lafm_image_scan_size['x'])
            save_y_scan_size = int(self.lafm_image_scan_size['y'])

            
            # 必須ヘッダー情報の存在をチェックし、なければデフォルト値を使用
            required_params = {
                'FileType': 1, 'FrameHeaderSize': 32, 'TextEncoding': 932, 'DataType1ch': 20564,
                'DataType2ch': 0, 'ScanDirection': 0, 'ScanTryNum': 1, 'AveFlag': 0, 'AveNum': 1,
                'XRound': 0, 'YRound': 0, 'FrameTime': 1000.0, 'Sensitivity': 1.0, 'PhaseSens': 1.0, 
                'MachineNo': 0, 'ADRange': 0, 'ADResolution': 0, 'PiezoConstX': 1.0,
                'PiezoConstY': 1.0, 'PiezoConstZ': 1.0, 'DriverGainZ': 1.0
            }
            header_values = {}
            for param, default in required_params.items():
                header_values[param] = getattr(gv, param, default)

            header_values['FileType'] = 1
            header_values['FrameHeaderSize'] = 32
            header_values['TextEncoding'] = 932
            if header_values['ADRange'] not in {
                0x00000001, 0x00000002, 0x00000004,
                0x00010000, 0x00020000, 0x00040000, 0x00800000,
            }:
                header_values['ADRange'] = 0x00040000

            # LAFM 2D保存は高さ[nm]として扱う（DataType1ch=20564）を強制する
            header_values['DataType1ch'] = 20564
            header_values['DataType2ch'] = 0

            # LAFM画像のダイナミックレンジに合わせてZ感度を最適化し、
            # 保存→再読込時に飽和で真っ黒化するのを防ぐ。
            image_data_f64 = np.asarray(image_data, dtype=np.float64)
            image_data_f64 = np.nan_to_num(image_data_f64, nan=0.0, posinf=0.0, neginf=0.0)
            img_min = float(np.min(image_data_f64))
            img_max = float(np.max(image_data_f64))
            image_span = max(img_max - img_min, 0.0)

            # raw = (5 - h/(PiezoConstZ*DriverGainZ)) * 4096/10
            # h in [0, image_span] が raw in [2048, 0] に収まるように設定する。
            effective_pcz = max(image_span / 5.0, 1e-6)
            header_values['PiezoConstZ'] = effective_pcz
            header_values['DriverGainZ'] = 1.0

        
            max_scan_size_x = getattr(gv, 'MaxScanSizeX', float(save_x_scan_size))
            max_scan_size_y = getattr(gv, 'MaxScanSizeY', float(save_y_scan_size))

            # 文字列情報のチェック
            if not hasattr(gv, 'OpeName') or gv.OpeName is None:
                print("[WARNING] SaveASD: gv.OpeName not available, using default: 'pyNuD'")
                ope_name = "pyNuD"
            else:
                ope_name = gv.OpeName

            # TextEncoding=932 と実際の文字列バイト列を一致させる。
            ope_name_bytes = ope_name.encode('cp932', errors='replace')
            comment_bytes = comment.encode('cp932', errors='replace')
            ope_name_size = len(ope_name_bytes)
            comment_size_for_save = len(comment_bytes)

            # ASD固定ヘッダーは165 bytes。
            file_header_size_for_save = 165 + ope_name_size + comment_size_for_save
            
            # 時刻情報
            time_params = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
            time_values = {}
            missing_time_params = []

            for param in time_params:
                if hasattr(gv, param) and getattr(gv, param) is not None:
                    time_values[param] = getattr(gv, param)
                else:
                    missing_time_params.append(param)
        
            # 時刻情報が不完全な場合の処理
            if missing_time_params:
                print(f"[WARNING] SaveASD: Missing time parameters: {missing_time_params}, using current time")
                now = datetime.datetime.now()
                time_values['Year'] = time_values.get('Year', now.year)
                time_values['Month'] = time_values.get('Month', now.month)
                time_values['Day'] = time_values.get('Day', now.day)
                time_values['Hour'] = time_values.get('Hour', now.hour)
                time_values['Minute'] = time_values.get('Minute', now.minute)
                time_values['Second'] = time_values.get('Second', now.second)
            
            

            # --- ステップ2: ファイルへの書き込み ---
            with open(save_path, 'wb') as f:
                # ファイルヘッダー
                f.write(struct.pack('<i', header_values['FileType'])); f.write(struct.pack('<i', file_header_size_for_save)); f.write(struct.pack('<i', header_values['FrameHeaderSize']))
                f.write(struct.pack('<i', header_values['TextEncoding'])); f.write(struct.pack('<i', ope_name_size)); f.write(struct.pack('<i', comment_size_for_save))
                f.write(struct.pack('<i', header_values['DataType1ch'])); f.write(struct.pack('<i', header_values['DataType2ch']))
                f.write(struct.pack('<i', 1)); f.write(struct.pack('<i', 1)) # 1フレームのみ
                f.write(struct.pack('<i', header_values['ScanDirection'])); f.write(struct.pack('<i', header_values['ScanTryNum']))
                f.write(struct.pack('<i', x_pixels)); f.write(struct.pack('<i', y_pixels))
                f.write(struct.pack('<i', save_x_scan_size)); f.write(struct.pack('<i', save_y_scan_size))
                f.write(struct.pack('<B', header_values['AveFlag'])); f.write(struct.pack('<i', header_values['AveNum']))
                f.write(struct.pack('<i', time_values['Year'])); f.write(struct.pack('<i', time_values['Month'])); f.write(struct.pack('<i', time_values['Day']))
                f.write(struct.pack('<i', time_values['Hour'])); f.write(struct.pack('<i', time_values['Minute'])); f.write(struct.pack('<i', time_values['Second']))
                f.write(struct.pack('<i', header_values['XRound'])); f.write(struct.pack('<i', header_values['YRound']))
                f.write(struct.pack('<f', header_values['FrameTime'])); f.write(struct.pack('<f', header_values['Sensitivity'])); f.write(struct.pack('<f', header_values['PhaseSens']))
                f.write(struct.pack('<iiii', 0, 0, 0, 0))
                f.write(struct.pack('<i', header_values['MachineNo'])); f.write(struct.pack('<i', header_values['ADRange'])); f.write(struct.pack('<i', header_values['ADResolution']))
                f.write(struct.pack('<f', max_scan_size_x)); f.write(struct.pack('<f', max_scan_size_y))
                f.write(struct.pack('<f', header_values['PiezoConstX'])); f.write(struct.pack('<f', header_values['PiezoConstY']))
                f.write(struct.pack('<f', header_values['PiezoConstZ'])); f.write(struct.pack('<f', header_values['DriverGainZ']))
                f.write(ope_name_bytes); f.write(comment_bytes)

                # --- 画像データの変換と書き込み ---
                # LAFM結果は nm データとして保存する。
                # 読み込み時にDataType1ch=20564で最小値が0に正規化されるため、
                # 保存側でも最小値基準（0スタート）で符号化する。
                height_data = image_data_f64 - img_min
                converted_data = (
                    5.0 - height_data / header_values['PiezoConstZ'] / header_values['DriverGainZ']
                ) * 4096.0 / 10.0

                # 非数値を除去
                converted_data = np.nan_to_num(converted_data, nan=0.0, posinf=65535.0, neginf=0.0)

                # ASDの標準的な12bit範囲へクリップ
                normalized_data = np.clip(np.round(converted_data), 0, 4095).astype(np.uint16)

                min_data_int = int(np.min(normalized_data))
                max_data_int = int(np.max(normalized_data))
                # フレームヘッダー
                f.seek(file_header_size_for_save)
                f.write(struct.pack('<I', 1)); f.write(struct.pack('<H', max_data_int)); f.write(struct.pack('<H', min_data_int))
                f.write(struct.pack('<h', 0)); f.write(struct.pack('<h', 0)); f.write(struct.pack('<f', 0.0)); f.write(struct.pack('<f', 0.0))
                f.write(struct.pack('<B', 0)); f.write(struct.pack('<B', 0)); f.write(struct.pack('<h', 0)); f.write(struct.pack('<i', 0)); f.write(struct.pack('<i', 0))

                # 正規化された画像データを書き込み
                f.write(normalized_data.tobytes())
            
            return True

        except Exception as e:
            print(f"[ERROR] _save_lafm_as_asd failed: {e}")
            import traceback
            traceback.print_exc()
            return False


# =============================================================================
# [Heath] NanoLocz-derived helpers  (GPL-3.0 -- see the licensing note at the top)
#
# Ports of NanoLocz-lib/*.m. MATLAB primitives are reimplemented to match
# numerically: rescale, fspecial('laplacian'), imfilter 'replicate',
# imgaussfilt (kernel 2*ceil(2*sigma)+1, replicate), imresize 'bicubic'
# (a = -0.5, half-pixel centres), ordfilt2 max, rmoutliers 'mean', smooth(y,5),
# and MATLAB's column-major tie-break in max(x(:)).
# =============================================================================

def _hz_rescale(x):
    """MATLAB rescale(): map the whole array to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def _hz_gauss_kernel(sigma):
    """MATLAB imgaussfilt 1-D kernel: length 2*ceil(2*sigma)+1, normalised."""
    r = int(np.ceil(2 * sigma))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _hz_imgaussfilt(img, sigma):
    """MATLAB imgaussfilt: separable, 'replicate' padding. 3-D -> per-slice."""
    from scipy.ndimage import correlate1d
    if sigma <= 0:
        return np.asarray(img, dtype=np.float64)
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        out = np.empty_like(a)
        for i in range(a.shape[2]):
            out[:, :, i] = _hz_imgaussfilt(a[:, :, i], sigma)
        return out
    k = _hz_gauss_kernel(sigma)
    a = correlate1d(a, k, axis=0, mode='nearest')
    return correlate1d(a, k, axis=1, mode='nearest')


def _hz_fspecial_laplacian(alpha=0.2):
    """MATLAB fspecial('laplacian', alpha)."""
    a = float(alpha)
    h = np.array([[a / 4, (1 - a) / 4, a / 4],
                  [(1 - a) / 4, -1.0, (1 - a) / 4],
                  [a / 4, (1 - a) / 4, a / 4]], dtype=np.float64)
    return 4.0 / (a + 1.0) * h


def _hz_imfilter_replicate(img, h):
    """MATLAB imfilter(img, h, 'replicate') -- correlation, replicate padding."""
    from scipy.ndimage import correlate as ndcorrelate
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        out = np.empty_like(a)
        for i in range(a.shape[2]):
            out[:, :, i] = ndcorrelate(a[:, :, i], h, mode='nearest')
        return out
    return ndcorrelate(a, h, mode='nearest')


def hz_filter_movie(target, gauss_sigma=0.2, laplacian_strength=0.0):
    """[Heath] filter_movie.m -- 'Gaussian' then 'Laplacian' branches.

    Laplacian: lap = imfilter(imgaussfilt(t,0.6), fspecial('laplacian',0.2), 'replicate')
               lap = rescale(lap);  t = rescale(-strength*lap + t)
    Note both rescale() calls span the WHOLE stack, so a single spiking frame
    compresses every other frame -- clean the stack before enabling this.
    """
    t = np.asarray(target, dtype=np.float64)
    if gauss_sigma and gauss_sigma > 0:
        t = _hz_imgaussfilt(t, gauss_sigma)
    if laplacian_strength and laplacian_strength > 0:
        h = _hz_fspecial_laplacian(0.2)
        lap = _hz_imfilter_replicate(_hz_imgaussfilt(t, 0.6), h)
        lap = _hz_rescale(lap)
        t = _hz_rescale(-laplacian_strength * lap + t)
    return t


def hz_fast_peaks2d_mask(img, thresh, kernel_size=1):
    """[Heath] Fast_peaks2D.m -- boolean mask of local maxima above thresh.

    kernel_size is incremented by 2 (Heath), giving a 3x3 max filter for the
    default of 1, with zero padding, and the outer 2 px of every edge excluded.
    """
    from scipy.ndimage import maximum_filter as _mf
    a = np.asarray(img, dtype=np.float64)
    k = int(kernel_size) + 2
    m = (_mf(a, size=k, mode='constant', cval=0.0) == a) & (a > thresh)
    m[:2, :] = False
    m[-3:, :] = False
    m[:, :2] = False
    m[:, -3:] = False
    return m


def _hz_cubic(x):
    """MATLAB imresize bicubic kernel (a = -0.5)."""
    x = np.abs(x)
    x2, x3 = x * x, x * x * x
    return np.where(x <= 1, 1.5 * x3 - 2.5 * x2 + 1.0,
                    np.where(x < 2, -0.5 * x3 + 2.5 * x2 - 4.0 * x + 2.0, 0.0))


def _hz_resize_weights(in_len, out_len, scale):
    u = (np.arange(1, out_len + 1) - 0.5) / scale + 0.5
    idx = np.floor(u - 2)[:, None] + np.arange(4)[None, :]
    w = _hz_cubic(u[:, None] - idx)
    w = w / w.sum(axis=1, keepdims=True)
    return np.clip(idx - 1, 0, in_len - 1).astype(int), w


def hz_imresize_bicubic(img, scale):
    """MATLAB imresize(img, scale, 'bicubic') for integer upscaling."""
    a = np.asarray(img, dtype=np.float64)
    h, w_ = a.shape
    oh, ow = int(round(h * scale)), int(round(w_ * scale))
    ri, rw = _hz_resize_weights(h, oh, scale)
    tmp = np.einsum('ijk,ij->ik', a[ri, :], rw)
    ci, cw = _hz_resize_weights(w_, ow, scale)
    return np.einsum('ijk,jk->ij', tmp[:, ci], cw)


def hz_localize_bicubic(frame, y_int, x_int):
    """[Heath] localize.m 'bicubic': 5x5 clip -> x10 bicubic -> central 30x30 -> argmax.

    Returns (sub_y, sub_x) or (nan, nan). Carries Heath's inherent +0.05 px offset
    ((locs_2x - 30/2)/10, where the 30-element crop centre is actually 15.5).
    Pixel-locks on noisy data -- prefer the Gaussian fit there.
    """
    H, W = frame.shape
    w, ex = 3, 10
    if not (y_int - w + 2 > 0 and x_int - w + 2 > 0 and y_int + w - 1 < H and x_int + w - 1 < W):
        return np.nan, np.nan
    clip = frame[y_int - w + 1:y_int + w, x_int - w + 1:x_int + w]
    z = hz_imresize_bicubic(clip, ex)[10:40, 10:40]
    # MATLAB max(clip(:)) resolves ties to the first element in COLUMN-major order
    my, mx = np.unravel_index(np.argmax(z.ravel(order='F')), z.shape, order='F')
    return (y_int + (my + 1 - z.shape[0] / 2) / ex,
            x_int + (mx + 1 - z.shape[1] / 2) / ex)


def hz_localize_gaussian(frame, y_int, x_int, pixperfeat=1.0):
    """[Heath] localize.m 'gaussian': 2-D Gaussian fit (TwoDGaussFit / lsqcurvefit).

    p0 = [A, x0, sx, y0, sy] = [1, 0, 3, 0, 3]
    lb = [0.05, -2, 0.5, -2, 0.5],  ub = [40, 2, 40, 2, 40]
    Returns (sub_y, sub_x, mean_sigma, amplitude). Does not pixel-lock.
    """
    from scipy.optimize import least_squares
    H, W = frame.shape
    w = 2 if pixperfeat < 0.75 else 3
    if not (y_int - w + 2 > 2 and x_int - w + 2 > 2 and y_int + w - 1 < H - 2 and x_int + w - 1 < W - 2):
        return np.nan, np.nan, 0.0, 0.0
    Z = frame[y_int - w + 1:y_int + w, x_int - w + 1:x_int + w]
    Z = Z - Z.min()
    n = 2 * w - 1
    g = np.arange(n) - (n - 1) / 2.0
    X, Y = np.meshgrid(g, g)

    def resid(p):
        A, x0, sx, y0, sy = p
        return (A * np.exp(-(((X - x0) ** 2) / (2 * sx ** 2)
                             + ((Y - y0) ** 2) / (2 * sy ** 2))) - Z).ravel()

    try:
        r = least_squares(resid, [1.0, 0.0, 3.0, 0.0, 3.0],
                          bounds=([0.05, -2, 0.5, -2, 0.5], [40, 2, 40, 2, 40]), method='trf')
        A, x0, sx, y0, sy = r.x
    except Exception:
        return np.nan, np.nan, 0.0, 0.0
    return y_int + y0, x_int + x0, (sx + sy) / 2.0, A


def hz_localize_sphere(frame, y_int, x_int, pixperfeat=1.0):
    """[Heath] localize.m 'sphere': algebraic sphere fit to a x3-upsampled clip."""
    H, W = frame.shape
    if pixperfeat < 0.5:
        w, const = 2, 5
    else:
        w, const = 3, 8
    if not (y_int - w + 2 > 2 and x_int - w + 2 > 2 and y_int + w - 1 < H - 2 and x_int + w - 1 < W - 2):
        return np.nan, np.nan, 0.0
    clip = hz_imresize_bicubic(frame[y_int - w + 1:y_int + w, x_int - w + 1:x_int + w], 3)
    rows, cols = clip.shape
    xg, yg = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    x = xg.ravel().astype(np.float64)
    y = yg.ravel().astype(np.float64)
    z = clip.ravel().astype(np.float64)
    N = x.size
    Sx, Sy, Sz = x.sum(), y.sum(), z.sum()
    Sxx, Syy, Szz = (x * x).sum(), (y * y).sum(), (z * z).sum()
    Sxy, Sxz, Syz = (x * y).sum(), (x * z).sum(), (y * z).sum()
    Sxxx, Syyy, Szzz = (x ** 3).sum(), (y ** 3).sum(), (z ** 3).sum()
    Sxyy, Sxzz = (x * y * y).sum(), (x * z * z).sum()
    Sxxy, Sxxz = (x * x * y).sum(), (x * x * z).sum()
    Syyz, Syzz = (y * y * z).sum(), (y * z * z).sum()
    A1 = Sxx + Syy + Szz
    a = 2 * Sx * Sx - 2 * N * Sxx
    b = 2 * Sx * Sy - 2 * N * Sxy
    c = 2 * Sx * Sz - 2 * N * Sxz
    d = -N * (Sxxx + Sxyy + Sxzz) + A1 * Sx
    e, f = b, 2 * Sy * Sy - 2 * N * Syy
    gg = 2 * Sy * Sz - 2 * N * Syz
    hh = -N * (Sxxy + Syyy + Syzz) + A1 * Sy
    j, k = c, gg
    ll = 2 * Sz * Sz - 2 * N * Szz
    mm = -N * (Sxxz + Syyz + Szzz) + A1 * Sz
    delta = a * (f * ll - gg * k) - e * (b * ll - c * k) + j * (b * gg - c * f)
    if delta == 0:
        return np.nan, np.nan, 0.0
    xc = (d * (f * ll - gg * k) - hh * (b * ll - c * k) + mm * (b * gg - c * f)) / delta
    yc = (a * (hh * ll - mm * gg) - e * (d * ll - mm * c) + j * (d * gg - hh * c)) / delta
    zc = (a * (f * mm - hh * k) - e * (b * mm - d * k) + j * (b * hh - d * f)) / delta
    R = np.sqrt(max(0.0, xc ** 2 + yc ** 2 + zc ** 2
                    + (A1 - 2 * (xc * Sx + yc * Sy + zc * Sz)) / N))
    return y_int + (yc - const) / 3.0, x_int + (xc - const) / 3.0, R


def _hz_rmoutliers_mean(x):
    """MATLAB rmoutliers(x, 'mean'): drop |x - mean| > 3*std."""
    x = np.asarray(x, dtype=np.float64)
    s = np.std(x, ddof=1) if x.size > 1 else 0.0
    return x if s == 0 else x[np.abs(x - np.mean(x)) <= 3 * s]


def _hz_sig3(v):
    """MATLAB round(v, 3, 'significant')."""
    v = float(v)
    if v == 0 or not np.isfinite(v):
        return v
    return round(v, -(int(np.floor(np.log10(abs(v)))) - 2))


def hz_render_density(det_y, det_x, det_z, out_shape, sigma, n_levels=256,
                      colorlimit_mode='Exc outliers', colorlimits=None):
    """[Heath] LAFM_renderer.m with prob=1 -- localization-density map.

    Heath bins localizations into the colormap's height levels, sets (not adds)
    1 at each localization pixel WITHIN a level, blurs, divides by `correction`
    so one isolated localization peaks at exactly 1, then sums the levels. Height
    therefore enters only through that binning; the returned map is a density.

    det_y/det_x are integer indices on the output grid; det_z are heights (nm).
    """
    Hs, Ws = out_shape
    z = np.asarray(det_z, dtype=np.float64)
    if colorlimit_mode == 'Max Min':
        cl = (_hz_sig3(z.min()), _hz_sig3(z.max()))
    elif colorlimit_mode == 'Manual' and colorlimits is not None:
        cl = (float(colorlimits[0]), float(colorlimits[1]))
    else:
        B = _hz_rmoutliers_mean(z)
        cl = (_hz_sig3(B.min()), _hz_sig3(B.max()))

    N = max(1, int(n_levels))
    if cl[1] > cl[0] and N > 1:
        grid = np.linspace(cl[0], cl[1], N)
        ci = np.interp(z, grid, np.arange(1, N + 1))
        slope = (N - 1) / (grid[-1] - grid[0])
        lo, hi = z < grid[0], z > grid[-1]
        ci[lo] = 1 + (z[lo] - grid[0]) * slope
        ci[hi] = N + (z[hi] - grid[-1]) * slope
        cidx = np.round(ci).astype(int)
    else:
        cidx = np.ones(z.size, dtype=int)

    corr = np.zeros((5, 5))
    corr[2, 2] = 1.0
    correction = _hz_imgaussfilt(corr, sigma).max()
    if correction <= 0:
        correction = 1.0

    out = np.zeros((Hs, Ws), dtype=np.float64)
    for i in range(1, N + 1):
        if i == 1:
            pos = cidx < 2
        elif i > N - 1:
            pos = cidx > N - 1
        else:
            pos = cidx == i
        if not np.any(pos):
            continue
        render = np.zeros((Hs, Ws), dtype=np.float64)
        render[det_y[pos], det_x[pos]] = 1.0        # assignment, not accumulation
        out += _hz_imgaussfilt(render, sigma) / correction
    return out, cl


def _hz_radialsum(img):
    s = np.array(img.shape)
    center = np.floor((s + 1) / 2).astype(int) - 1
    n = int(np.ceil(s[0] / 2)) + 1
    yy, xx = np.mgrid[0:s[0], 0:s[1]]
    ind = np.round(np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)).astype(int)
    keep = ind < n
    return np.bincount(ind[keep], weights=np.asarray(img)[keep], minlength=n)[:n]


def _hz_smooth5(y):
    """MATLAB smooth(y, 5): moving average with a shrinking span at the ends."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        k = min(i, n - 1 - i, 2)
        out[i] = y[i - k:i + k + 1].mean()
    return out


def hz_measure_frc(det_y, det_x, frames, grid_h, grid_w, nm_per_px,
                   runs=20, img_gaus_expanded=2.0, seed=0):
    """[Heath] measureFRC.m -- Fourier ring correlation, 1/7 criterion.

    Splits the per-frame localization maps into two random halves, correlates
    them in Fourier space and reports 1/(first crossing of 1/7). Adapted from
    Ries/SMAP (GPLv3) by T. Storer. Returns (q_inv_nm, frc_mean, av_nm, sd_nm).
    Because both halves come from one movie this measures the reproducibility of
    that dataset, not absolute accuracy.
    """
    rng = np.random.default_rng(seed)
    frames = np.asarray(frames)
    uniq = np.unique(frames)
    stack = np.zeros((grid_h, grid_w, len(uniq)), dtype=np.float64)
    for i, fr in enumerate(uniq):
        sel = frames == fr
        stack[det_y[sel], det_x[sel], i] = 1.0
    stack = _hz_imgaussfilt(stack, img_gaus_expanded)

    k = stack.shape[2]
    if k < 2:
        return None, None, float('nan'), float('nan')
    res, curves, q = [], [], None
    for _ in range(int(runs)):
        perm = rng.permutation(k)
        h = int(round(k / 2))
        i1 = stack[:, :, perm[:h]].mean(axis=2)
        i2 = stack[:, :, perm[h:]].mean(axis=2)
        f1 = np.fft.fftshift(np.fft.fft2(i1))
        f2 = np.fft.fftshift(np.fft.fft2(i2))
        num = np.real(_hz_radialsum(np.real(f1 * np.conj(f2))))
        den = np.sqrt(np.abs(_hz_radialsum(np.abs(f1) ** 2) * _hz_radialsum(np.abs(f2) ** 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.nan_to_num(num / den)
        c = _hz_smooth5(np.clip(np.real(c), -1, 1))
        q = np.arange(len(c)) / i1.shape[0] / nm_per_px
        curves.append(c)
        res.append(1.0 / _hz_findcross(q, c, 1.0 / 7))
    res = np.array(res, dtype=np.float64)
    good = np.isfinite(res)
    if not np.any(good):
        return q, None, float('nan'), float('nan')
    frc_mean = np.mean(np.array(curves)[good], axis=0)
    av = 1.0 / _hz_findcross(q, frc_mean, 1.0 / 7)
    sd = float(np.std(res[good], ddof=1)) if good.sum() > 1 else 0.0
    return q, frc_mean, av, sd


def _hz_findcross(x, y, t):
    """MATLAB findintersection(): first interpolated x > 0.25 where y <= t."""
    if len(x) < 2:
        return float('nan')
    xq = np.arange(x[0], x[-1] + 1e-12, (x[1] - x[0]) / 10.0)
    yq = np.interp(xq, x, y)
    v = xq > 0.25
    xf, yf = xq[v], yq[v]
    below = yf <= t
    return float(xf[below][0]) if np.any(below) else float(np.max(x))


def _hz_normxcorr2(template, image):
    """MATLAB normxcorr2(template, image). Output shape = image.shape + template.shape - 1.

    Normalised cross-correlation after Lewis (1995), as MATLAB implements it: the template is
    mean-subtracted once, the image statistics are accumulated over the sliding window.
    """
    from scipy.signal import fftconvolve
    t = np.asarray(template, dtype=np.float64)
    a = np.asarray(image, dtype=np.float64)
    t = t - t.mean()
    tnorm = np.sqrt(float((t * t).sum()))
    if tnorm <= 0:
        return np.zeros((a.shape[0] + t.shape[0] - 1, a.shape[1] + t.shape[1] - 1))
    num = fftconvolve(a, t[::-1, ::-1], mode='full')
    ones = np.ones_like(t)
    s1 = fftconvolve(a, ones, mode='full')
    s2 = fftconvolve(a * a, ones, mode='full')
    n = float(t.size)
    var = s2 - (s1 * s1) / n
    den = np.sqrt(np.maximum(var, 0.0)) * tnorm
    out = np.zeros_like(num)
    m = den > 1e-12
    out[m] = num[m] / den[m]
    return np.clip(out, -1.0, 1.0)


def _hz_ccalign(img, ref, exp=1):
    """[Heath] ccAlign() inside FindCenterPositions.m -> (dx, dy) to align img onto ref."""
    c = _hz_normxcorr2(img, ref)
    iy, ix = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    H_, W_ = img.shape
    if exp is None or exp <= 1:
        return float(ix - W_), float(iy - H_)
    # MATLAB zeroes a 4 px border, then clips a 5x5 window about the peak found BEFORE zeroing
    cz = c.copy()
    cz[:4, :] = 0; cz[:, :4] = 0; cz[-4:, :] = 0; cz[:, -4:] = 0
    w = 3
    y0, y1 = iy - w + 1, iy + w
    x0, x1 = ix - w + 1, ix + w
    if y0 < 0 or x0 < 0 or y1 > cz.shape[0] or x1 > cz.shape[1]:
        return float(ix - W_), float(iy - H_)
    clip = cz[y0:y1, x0:x1]
    if clip.size == 0 or not np.any(np.isfinite(clip)):
        return float(ix - W_), float(iy - H_)
    zoom_ = hz_imresize_bicubic(clip, int(exp))
    zy, zx = np.unravel_index(np.argmax(np.abs(zoom_)), zoom_.shape)
    dx2 = (zx + 1 - zoom_.shape[1] / 2.0) / float(exp)
    dy2 = (zy + 1 - zoom_.shape[0] / 2.0) / float(exp)
    return float(ix + dx2 - W_), float(iy + dy2 - H_)


def hz_find_center_positions(fold, img, align_exp=10):
    """[Heath] FindCenterPositions.m -- estimate the rotational symmetry centre.

    Rotates the image by 360/fold increments and cross-correlates each rotation against the
    original; the mean offset locates the symmetry axis. Returns (dx, dy) in pixels, the
    translation that must be REMOVED (i.e. shift the image by -dx, -dy) to put the symmetry
    axis at the array centre -- the same convention as MATLAB's imtranslate(ref, -t).
    """
    from scipy.ndimage import rotate as _rot
    a = np.asarray(img, dtype=np.float64)
    fold = int(fold)
    if fold == 1:
        yy, xx = np.mgrid[1:a.shape[0] + 1, 1:a.shape[1] + 1]
        m = a.mean()
        if m == 0:
            return 0.0, 0.0
        cx = float((a * xx).mean() / m)
        cy = float((a * yy).mean() / m)
        return cx - a.shape[1] / 2.0, cy - a.shape[0] / 2.0
    offs = []
    for i in range(1, fold):
        r = _rot(a, i * 360.0 / fold, reshape=False, order=1, mode='constant', cval=0.0)
        offs.append(_hz_ccalign(r, a, align_exp))
    offs = np.array(offs, dtype=np.float64)
    if fold == 2:
        t = offs[0] / 2.0
    else:
        t = offs.sum(axis=0) / fold
    return float(t[0]), float(t[1])


def hz_centre_of_mass(img):
    """Intensity-weighted centroid, fold-INDEPENDENT. Returns (dx, dy) from the array centre.

    Unlike FindCenterPositions this assumes no symmetry, so it is the right choice when the
    centre is wanted for its own sake (e.g. putting several particles on a common origin).
    The median background is removed and negatives clipped first, otherwise a non-zero
    baseline pulls the centroid towards the middle of the field.
    """
    a = np.asarray(img, dtype=np.float64)
    w = a - np.median(a)
    np.clip(w, 0.0, None, out=w)
    tot = float(w.sum())
    if tot <= 0:
        return 0.0, 0.0
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    cy = float((w * yy).sum() / tot)
    cx = float((w * xx).sum() / tot)
    return cx - a.shape[1] / 2.0, cy - a.shape[0] / 2.0


def _hz_centring_method(params):
    """Resolve the centring method, falling back to the legacy sym_autocentre flag."""
    m = params.get('centring_method')
    if m:
        return str(m)
    return 'Symmetry axis (C_n) [Heath]' if params.get('sym_autocentre', True) else 'Off'


def hz_find_centre(method, img, fold=3, align_exp=10):
    """Dispatch the centring method. Returns (dx, dy) or None for 'Off'.

    'Symmetry axis' needs `fold`; 'Centre of mass' ignores it.
    """
    m = str(method or '').strip().lower()
    if not m or m.startswith('off'):
        return None
    if m.startswith('centre of mass') or m.startswith('center of mass'):
        return hz_centre_of_mass(img)
    return hz_find_center_positions(fold, img, align_exp)


def hz_symmetrise(img, order, centre_translation=None, interp_order=3):
    """C_order rotational average.

    interp_order 3 (bicubic-family spline) matches Heath's rotation_sym.m, which uses
    imrotate(..., 'bicubic'). pyNuD historically used order=0 (nearest), which aliases.

    centre_translation: (dx, dy) from hz_find_center_positions. When given, the image is
    shifted so the symmetry axis sits at the array centre, symmetrised, then shifted back --
    without this the rotation happens about the array centre, which smears the result by the
    axis offset (0.8 nm on a typical tracked EltXeR crop).
    """
    from scipy.ndimage import rotate as _rot, shift as _shift
    a = np.asarray(img, dtype=np.float64)
    order = int(order)
    if order < 2:
        return a
    dx, dy = (0.0, 0.0) if centre_translation is None else centre_translation
    if dx or dy:
        a = _shift(a, (-dy, -dx), order=1, mode='constant', cval=0.0)
    acc = np.zeros_like(a)
    for j in range(order):
        acc += a if j == 0 else _rot(a, j * 360.0 / order, reshape=False,
                                     order=interp_order, mode='constant', cval=0.0)
    out = acc / order
    if dx or dy:
        out = _shift(out, (dy, dx), order=1, mode='constant', cval=0.0)
    return out


def create_plugin(main_window):
    """プラグインエントリポイント。pyNuD の Plugin メニューから呼ばれる。"""
    return LAFMPanelWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "LAFMPanelWindow"]
