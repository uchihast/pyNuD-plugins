# loc_afm.py (完全版)
# loc_afm.py (完全版)
# loc_afm.py (ASD保存機能に対応した最終完成版)

import sys
import time
import os # <<< osモジュールをインポート
import json
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
<p>Localization Atomic Force Microscopy (L-AFM) analysis is a technique to construct a "super-resolution image" that surpasses the resolution of the original image. It works by detecting the precise locations of numerous individual molecules' or structures' brightness peaks over time from an AFM time-series image (movie) and reconstructing them onto a high-resolution grid. This panel allows you to perform the series of processes from peak detection to image reconstruction step-by-step.</p>
<p><strong>Algorithm basis:</strong> This plugin uses an L-AFM-style peak-localization and reconstruction workflow based on the Localization atomic force microscopy algorithm reported by Heath, Scheuring, and colleagues (<i>Nature</i> 594, 385–390, 2021; DOI: 10.1038/s41586-021-03551-x).</p>

<h2>Access</h2>
<ul>
    <li><strong>Plugin menu:</strong> Load Plugin... → select <code>plugins/LAFMAnalysis.py</code>, then Plugin → L-AFM Analysis</li>
</ul>

<h2>Overview of Processing Steps</h2>
<p>L-AFM analysis consists of three main steps. Please execute the buttons for each step in order.</p>
<div class="step">
    <strong>Step 1: Preprocessing 1 (Peak Detection)</strong><br>
    Scans each frame of the AFM movie to detect bright spots (peaks) based on set criteria.
</div>
<div class="step">
    <strong>Step 2: Preprocessing 2 (Reconstruction into Space)</strong><br>
    Plots the coordinates of all peaks detected in Step 1 onto a high-resolution 2D grid or into a 3D voxel space.
</div>
<div class="step">
    <strong>Step 3: Make LAFM Image (Super-Resolution Image Generation)</strong><br>
    Applies a Gaussian blur to the pointillistic data created in Step 2 to finish it into a smooth super-resolution image.
</div>

<h2>New Features: Automatic Z-Range Settings</h2>
<p>The L-AFM panel now includes intelligent Z-range optimization features to improve analysis accuracy and user experience.</p>

<h3>Auto Z-Range Button</h3>
<div class="feature-box">
    <h4>Statistical Z-Range Calculation</h4>
    <ul>
        <li><strong>Automatic Calculation:</strong> Analyzes the loaded image stack to determine optimal Z_min and Z_max values based on data statistics.</li>
        <li><strong>Noise Level Consideration:</strong> Estimates noise floor and baseline to set appropriate thresholds.</li>
        <li><strong>Data Coverage:</strong> Ensures the calculated range covers an appropriate percentage of the data points.</li>
        <li><strong>Physical Validity:</strong> Applies minimum thresholds (10 pm) and checks for logical consistency.</li>
    </ul>
</div>

<h3>Sample Type Selection</h3>
<div class="feature-box">
    <h4>Predefined Settings for Different Sample Types</h4>
    <ul>
        <li><strong>General:</strong> Default settings for general purpose analysis (Z_min: 0.1 nm, Z_max: 10.0 nm)</li>
        <li><strong>Proteins:</strong> Optimized for single proteins to large complexes (Z_min: 0.1 nm, Z_max: 10.0 nm)</li>
        <li><strong>DNA/RNA:</strong> Suitable for nucleic acid molecules (Z_min: 0.05 nm, Z_max: 3.0 nm)</li>
        <li><strong>Cells:</strong> For cellular structures and organelles (Z_min: 1.0 nm, Z_max: 100.0 nm)</li>
        <li><strong>Crystals:</strong> For crystal surfaces and defects (Z_min: 0.01 nm, Z_max: 50.0 nm)</li>
        <li><strong>Nanoparticles:</strong> For nanoparticles and aggregates (Z_min: 0.5 nm, Z_max: 20.0 nm)</li>
    </ul>
</div>

<h3>Usage Instructions</h3>
<ol>
    <li><strong>Data Loading:</strong> When image data is loaded, Z-range values are automatically calculated and set.</li>
    <li><strong>Manual Adjustment:</strong> Click the "Auto Z-Range" button to recalculate based on current data.</li>
    <li><strong>Sample Type Selection:</strong> Choose the appropriate sample type from the dropdown to apply recommended settings.</li>
    <li><strong>Range Display:</strong> The current Z-range is displayed next to the button for easy reference.</li>
</ol>

<h2>Difference Between Preprocessing 1 and 2</h2>
<p>These two steps play completely different roles in L-AFM analysis: "<strong>detection</strong>" and "<strong>drawing</strong>."</p>
<div class="feature-box">
    <h4>Preprocessing 1: Peak Search and Detection</h4>
    <ul>
        <li><strong>Input:</strong> The AFM time-series image stack (raw pixel data).</li>
        <li><strong>Process:</strong> Scans each frame and lists the <strong>coordinate information</strong> of "where molecules existed."</li>
        <li><strong>Output:</strong> A list of all detected peaks' coordinates, intensities, frame numbers, etc. (like an address book).</li>
        <li><strong>In short:</strong> It's the process of <strong>creating a "molecule address book" from the images.</strong></li>
    </ul>
</div>
<div class="feature-box">
    <h4>Preprocessing 2: Reconstruction from Coordinate Data to Image</h4>
    <ul>
        <li><strong>Input:</strong> The peak coordinate list output by Preprocessing 1. <strong>(This step does not look at the original images at all.)</strong></li>
        <li><strong>Process:</strong> Prepares a new high-resolution canvas and plots (votes for) points at the locations from the input coordinate list.</li>
        <li><strong>Output:</strong> Pointillistic data representing the density of peak presences.</li>
        <li><strong>In short:</strong> It's the process of <strong>creating a "distribution map" based on the "molecule address book."</strong></li>
    </ul>
</div>

<h2>Parameter Groups and Settings</h2>

<h3>Peak Filtering Group</h3>
<div class="feature-box">
    <h4>Filter Mode</h4>
    <ul>
        <li><strong>Absolute Height (nm):</strong> Filters peaks based on their absolute height values in nanometers.</li>
        <li><strong>Statistics (Mean + N x Std Dev):</strong> Filters peaks based on statistical criteria using mean and standard deviation.</li>
    </ul>

    <h4>Z-Range Settings (New Feature)</h4>
    <ul>
        <li><strong>Auto Z-Range Button:</strong> Automatically calculates optimal Z_min and Z_max values from the loaded image data.</li>
        <li><strong>Z_min (nm):</strong> Minimum height threshold for peak detection.</li>
        <li><strong>Z_max (nm):</strong> Maximum height threshold for peak detection.</li>
        <li><strong>Range Display:</strong> Shows the current Z-range span for easy reference.</li>
    </ul>

    <h4>Sample Type Selection (New Feature)</h4>
    <ul>
        <li><strong>General:</strong> Default settings for general purpose analysis.</li>
        <li><strong>Proteins:</strong> Optimized for protein analysis.</li>
        <li><strong>DNA/RNA:</strong> Suitable for nucleic acid molecules.</li>
        <li><strong>Cells:</strong> For cellular structures and organelles.</li>
        <li><strong>Crystals:</strong> For crystal surfaces and defects.</li>
        <li><strong>Nanoparticles:</strong> For nanoparticles and aggregates.</li>
    </ul>

    <h4>N Factor</h4>
    <ul>
        <li><strong>Purpose:</strong> Multiplier for standard deviation in statistical filtering mode.</li>
        <li><strong>Auto-calculation:</strong> Automatically calculated from the first frame when data is loaded.</li>
    </ul>
</div>

<h2>Deciding Which Step to Rerun After Changing Parameters</h2>
<p>Based on the differences above, the step you need to redo depends on whether the changed parameter affects "detection" or "drawing."</p>

<h3>Parameters Requiring Rerun from Preprocessing 1</h3>
<p>If you change the <strong>"peak detection conditions"</strong> themselves, you need to start over from the first step.</p>
<ul>
    <li>All parameters in the <b>Drift Correction</b> group</li>
    <li>All parameters in the <b>Peak Filtering</b> group</li>
    <li>All parameters in the <b>Local Maxima</b> group</li>
    <li>"Enable Subpixel Localization" and "Scale" in the <b>Subpixel Localization</b> group</li>
</ul>
<h3>Parameters Allowing Rerun from Preprocessing 2</h3>
<p>If you only change the conditions for <strong>"how to draw the already detected peaks,"</strong> you can skip the time-consuming peak detection.</p>
<ul>
    <li><b>Mode</b> (switching between "2D" ⇔ "3D")</li>
    <li>"XY Resolution" and "Z Resolution" in the <b>Subpixel Localization</b> group</li>
    <li>The "During Reconstruction (Prep 2)" setting in the <b>Symmetric Averaging</b> group</li>
</ul>

<h3>Parameters Allowing Rerun from Make LAFM Image</h3>
<p>If you only change <strong>"how to finish the reconstructed image,"</strong> you can start from this fastest step.</p>
<ul>
    <li>All parameters in the <b>Gaussian Blur</b> group</li>
    <li>The "On Final LAFM Image" setting in the <b>Symmetric Averaging</b> group</li>
</ul>

<h2>Practical Workflow (Flowchart)</h2>
<p>Below is a flowchart of a practical analysis workflow incorporating the branching conditions described above.</p>
<pre><code>
graph TD
    subgraph Initial Setup
        A[Open L-AFM Panel] --> B{Set Parameters};
    end

    subgraph Step 1: Peak Detection
        B --> C[1. Execute Preprocessing 1];
        C --> D{Are peak detection results valid?<br>(Check count and preview image)};
        D -- No --> E[<b>Reconfigure P1-related parameters</b><br>- Drift Correction<br>- Peak Filtering<br>- Local Maxima<br>- Subpixel (Enable/Scale)];
        E --> C;
    end

    subgraph Step 2: Reconstruction
        D -- Yes --> F[2. Execute Preprocessing 2];
        F --> G{Are reconstruction results valid?<br>(Check image density and distribution)};
        G -- No --> H[<b>Reconfigure P2-related parameters</b><br>- Mode (2D/3D)<br>- Subpixel Resolution<br>- Symmetric Avg (Prep 2)];
        H --> F;
    end

    subgraph Step 3: Image Generation and Saving
        G -- Yes --> I[3. Execute Make LAFM Image];
        I --> J{Is the final image satisfactory?<br>(Check smoothness and appearance)};
        J -- No --> K[<b>Reconfigure P3-related parameters</b><br>- Gaussian Blur<br>- Symmetric Avg (Final)];
        K --> I;
        J -- Yes --> L[4. Save with "Save" button];
    end
</code></pre>

<hr>
<h2>References</h2>
<ul>
    <li>George R. Heath, et al. "<a href="https://doi.org/10.1038/s41586-021-03551-x">Localization atomic force microscopy</a>". <i>Nature</i> 594, 385–390 (2021).</li>
    <li>Yining Jiang, et al. "<a href="https://doi.org/10.1038/s41594-024-01260-3">HS-AFM single-molecule structural biology uncovers basis of transporter wanderlust kinetics</a>". <i>Nature Structural & Molecular Biology</i> 31, 1286–1295 (2024).</li>
</ul>
"""

HELP_HTML_JA = """
<h1>L-AFM Analysis (Localization AFM)</h1>

<h2>概要</h2>
<p>L-AFM (Localization Atomic Force Microscopy) 解析は、AFMの時系列画像（動画）から個々の分子や構造物の輝度ピークを高精度に検出し、その位置情報を多数集めて再構成することで、元の画像の解像度を超える「超解像画像」を構築する技術です。このパネルでは、ピーク検出から画像再構成までの一連の処理を、ステップ・バイ・ステップで実行できます。</p>
<p><strong>アルゴリズムの出典:</strong> 本プラグインは、Heath、Scheuringらが報告したLocalization atomic force microscopy (L-AFM) アルゴリズム（<i>Nature</i> 594, 385–390, 2021; DOI: 10.1038/s41586-021-03551-x）に基づくピークローカリゼーション・再構成処理を利用しています。</p>

<h2>アクセス方法</h2>
<ul>
    <li><strong>プラグインメニュー:</strong> Load Plugin... → <code>plugins/LAFMAnalysis.py</code> を選択し、Plugin → L-AFM Analysis を実行</li>
</ul>

<h2>処理ステップの概要</h2>
<p>L-AFM解析は、主に3つのステップで構成されます。各ステップのボタンを順番に実行してください。</p>
<div class="step">
    <strong>Step 1: Preprocessing 1 (ピーク検出)</strong><br>
    AFM動画の各フレームから、設定された条件に基づいて輝度が高い点（ピーク）を検出します。
</div>
<div class="step">
    <strong>Step 2: Preprocessing 2 (空間への再構成)</strong><br>
    Step 1で検出された全てのピークの座標を、高解像度の2Dグリッドまたは3Dボクセル空間にプロットします。
</div>
<div class="step">
    <strong>Step 3: Make LAFM Image (超解像画像の生成)</strong><br>
    Step 2で作成した点描画のようなデータに、ガウシアンぼかしを適用して滑らかな超解像画像に仕上げます。
</div>

<h2>新機能: Z範囲自動設定</h2>
<p>L-AFMパネルには、解析精度とユーザビリティを向上させるためのインテリジェントなZ範囲最適化機能が追加されました。</p>

<h3>Auto Z-Rangeボタン</h3>
<div class="feature-box">
    <h4>統計的Z範囲計算</h4>
    <ul>
        <li><strong>自動計算:</strong> 読み込まれた画像スタックを解析し、データ統計に基づいて最適なZ_minとZ_max値を決定します。</li>
        <li><strong>ノイズレベル考慮:</strong> ノイズフロアとベースラインを推定して適切な閾値を設定します。</li>
        <li><strong>データカバー率:</strong> 計算された範囲がデータポイントの適切な割合をカバーすることを保証します。</li>
        <li><strong>物理的妥当性:</strong> 最小閾値（10 pm）を適用し、論理的一貫性をチェックします。</li>
    </ul>
</div>

<h3>サンプルタイプ選択</h3>
<div class="feature-box">
    <h4>異なるサンプルタイプ用の事前定義設定</h4>
    <ul>
        <li><strong>General:</strong> 一般的な解析用のデフォルト設定（Z_min: 0.1 nm, Z_max: 10.0 nm）</li>
        <li><strong>Proteins:</strong> 単一タンパク質から大きな複合体まで最適化（Z_min: 0.1 nm, Z_max: 10.0 nm）</li>
        <li><strong>DNA/RNA:</strong> 核酸分子に適した設定（Z_min: 0.05 nm, Z_max: 3.0 nm）</li>
        <li><strong>Cells:</strong> 細胞構造とオルガネラ用（Z_min: 1.0 nm, Z_max: 100.0 nm）</li>
        <li><strong>Crystals:</strong> 結晶表面と欠陥用（Z_min: 0.01 nm, Z_max: 50.0 nm）</li>
        <li><strong>Nanoparticles:</strong> ナノ粒子と凝集体用（Z_min: 0.5 nm, Z_max: 20.0 nm）</li>
    </ul>
</div>

<h3>使用方法</h3>
<ol>
    <li><strong>データ読み込み:</strong> 画像データが読み込まれると、Z範囲値が自動的に計算・設定されます。</li>
    <li><strong>手動調整:</strong> "Auto Z-Range"ボタンをクリックして、現在のデータに基づいて再計算します。</li>
    <li><strong>サンプルタイプ選択:</strong> ドロップダウンから適切なサンプルタイプを選択して推奨設定を適用します。</li>
    <li><strong>範囲表示:</strong> 現在のZ範囲がボタンの横に表示され、簡単に参照できます。</li>
</ol>

<h2>Preprocessing 1と2の違いについて</h2>
<p>この2つのステップは、L-AFM解析における「<b>検出</b>」と「<b>描画</b>」という全く異なる役割を担っています。</p>
<div class="feature-box">
    <h4>Preprocessing 1：ピークの探索と検出</h4>
    <ul>
        <li><strong>入力</strong>: AFMの時系列画像スタック（生のピクセルデータ）</li>
        <li><strong>処理内容</strong>: 各フレームをスキャンし、「どこに分子が存在したか」という<b>座標情報</b>をリストアップします。</li>
        <li><strong>出力</strong>: 検出された全ピークの座標・輝度・フレーム番号などをまとめたリスト（住所録のようなもの）。</li>
        <li><strong>一言で言うと</strong>: <b>画像から「分子の住所録」を作る作業です。</b></li>
    </ul>
</div>
<div class="feature-box">
    <h4>Preprocessing 2：座標データから画像への再構成</h4>
    <ul>
        <li><strong>入力</strong>: Preprocessing 1 が出力したピークの座標リスト。<b>（このステップでは元の画像は一切見ません）</b></li>
        <li><strong>処理内容</strong>: 新しい高解像度のキャンバスを用意し、入力された座標リストの場所に点をプロット（投票）していきます。</li>
        <li><strong>出力</strong>: ピークの存在密度を表現した点描画のようなデータ。</li>
        <li><strong>一言で言うと</strong>: <b>「分子の住所録」を元に「分布図」を作成する作業です。</b></li>
    </ul>
</div>

<h2>パラメータグループと設定</h2>

<h3>Peak Filteringグループ</h3>
<div class="feature-box">
    <h4>Filter Mode</h4>
    <ul>
        <li><strong>Absolute Height (nm):</strong> ピークを絶対的な高さ値（ナノメートル）に基づいてフィルタリングします。</li>
        <li><strong>Statistics (Mean + N x Std Dev):</strong> 平均値と標準偏差を使用した統計的基準でピークをフィルタリングします。</li>
    </ul>

    <h4>Z範囲設定（新機能）</h4>
    <ul>
        <li><strong>Auto Z-Rangeボタン:</strong> 読み込まれた画像データから最適なZ_minとZ_max値を自動計算します。</li>
        <li><strong>Z_min (nm):</strong> ピーク検出の最小高さ閾値。</li>
        <li><strong>Z_max (nm):</strong> ピーク検出の最大高さ閾値。</li>
        <li><strong>範囲表示:</strong> 現在のZ範囲の幅を簡単に参照できるように表示します。</li>
    </ul>

    <h4>サンプルタイプ選択（新機能）</h4>
    <ul>
        <li><strong>General:</strong> 一般的な解析用のデフォルト設定。</li>
        <li><strong>Proteins:</strong> タンパク質解析に最適化。</li>
        <li><strong>DNA/RNA:</strong> 核酸分子に適した設定。</li>
        <li><strong>Cells:</strong> 細胞構造とオルガネラ用。</li>
        <li><strong>Crystals:</strong> 結晶表面と欠陥用。</li>
        <li><strong>Nanoparticles:</strong> ナノ粒子と凝集体用。</li>
    </ul>

    <h4>N Factor</h4>
    <ul>
        <li><strong>目的:</strong> 統計的フィルタリングモードでの標準偏差の乗数。</li>
        <li><strong>自動計算:</strong> データ読み込み時に最初のフレームから自動計算されます。</li>
    </ul>
</div>

<h2>パラメータ変更と再実行の判断</h2>
<p>上記の違いから、変更したパラメータが「検出」に影響するのか、「描画」に影響するのかによって、やり直すべきステップが変わります。</p>

<h3>Preprocessing 1から再実行が必要なパラメータ</h3>
<p><b>「ピークの検出条件」そのものに変更があった場合</b>は、最初のステップからやり直す必要があります。</p>
<ul>
    <li><b>Drift Correction</b> グループの全パラメータ</li>
    <li><b>Peak Filtering</b> グループの全パラメータ</li>
    <li><b>Local Maxima</b> グループの全パラメータ</li>
    <li><b>Subpixel Localization</b> グループの "Enable Subpixel Localization" と "Scale"</li>
</ul>

<h3>Preprocessing 2からで良いパラメータ</h3>
<p><b>「検出済みのピークをどう描画するか」という条件のみ変更した場合</b>は、時間のかかるピーク検出をスキップできます。</p>
<ul>
    <li><b>Mode</b> ("2D" ⇔ "3D" の切り替え)</li>
    <li><b>Subpixel Localization</b> グループの "XY Resolution" と "Z Resolution"</li>
    <li><b>Symmetric Averaging</b> グループの "During Reconstruction (Prep 2)" の設定</li>
</ul>

<h3>Make LAFM Imageからで良いパラメータ</h3>
<p><b>「再構成された画像の仕上げ方」のみ変更した場合</b>は、最も高速なこのステップからで結構です。</p>
<ul>
    <li><b>Gaussian Blur</b> グループの全パラメータ</li>
    <li><b>Symmetric Averaging</b> グループの "On Final LAFM Image" の設定</li>
</ul>

<h2>実践的なワークフロー（フローチャート）</h2>
<p>以下に、上記の分岐条件を盛り込んだ実践的な解析ワークフローのフローチャートを示します。</p>
<pre><code>
graph TD
    subgraph 初期設定
        A[L-AFMパネルを開く] --> B{パラメータを設定};
    end

    subgraph Step 1: ピーク検出
        B --> C[1. Preprocessing 1 を実行];
        C --> D{ピーク検出結果は妥当か？<br>(検出数やプレビュー画像を確認)};
        D -- No --> E[<b>P1関連パラメータを再設定</b><br>- Drift Correction<br>- Peak Filtering<br>- Local Maxima<br>- Subpixel (Enable/Scale)];
        E --> C;
    end

    subgraph Step 2: 再構成
        D -- Yes --> F[2. Preprocessing 2 を実行];
        F --> G{再構成結果は妥当か？<br>(画像の密度や分布を確認)};
        G -- No --> H[<b>P2関連パラメータを再設定</b><br>- Mode (2D/3D)<br>- Subpixel Resolution<br>- Symmetric Avg (Prep 2)];
        H --> F;
    end

    subgraph Step 3: 画像生成と保存
        G -- Yes --> I[3. Make LAFM Image を実行];
        I --> J{最終画像は満足か？<br>(画像の滑らかさや見た目を確認)};
        J -- No --> K[<b>P3関連パラメータを再設定</b><br>- Gaussian Blur<br>- Symmetric Avg (Final)];
        K --> I;
        J -- Yes --> L[4. Save で保存];
    end
</code></pre>

<hr>
<h2>参考文献</h2>
<ul>
    <li>George R. Heath, et al. "<a href="https://doi.org/10.1038/s41586-021-03551-x">Localization atomic force microscopy</a>". <i>Nature</i> 594, 385–390 (2021).</li>
    <li>Yining Jiang, et al. "<a href="https://doi.org/10.1038/s41594-024-01260-3">HS-AFM single-molecule structural biology uncovers basis of transporter wanderlust kinetics</a>". <i>Nature Structural & Molecular Biology</i> 31, 1286–1295 (2024).</li>
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
        control_layout.addLayout(button_grid_layout)
        
        self.btn_prep1.setEnabled(False)
        self.btn_prep2.setEnabled(False); self.btn_make_img.setEnabled(False); self.btn_save.setEnabled(False)
        self.btn_prep1.clicked.connect(self.run_preprocessing1)
        self.btn_prep2.clicked.connect(self.run_preprocessing2)
        self.btn_make_img.clicked.connect(self.run_make_lafm_image)
        self.btn_save.clicked.connect(self._save_lafm_data)
        self.btn_load.clicked.connect(self._load_lafm_params)
        
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
        subpix_layout.addRow("Scale:", self.subpix_scale_spin)
        subpix_layout.addRow("Expand:", self.subpix_expand_spin)
        subpix_layout.addRow("XY Resolution:", self.subpix_xy_res_spin)
        subpix_layout.addRow("Z Resolution:", self.subpix_z_res_spin)
        control_layout.addWidget(self.subpix_group)

        self.sym_group = QtWidgets.QGroupBox("Symmetric Averaging"); self.sym_group.setCheckable(True); self.sym_group.setChecked(False)
        sym_v_layout = QtWidgets.QVBoxLayout(self.sym_group)
        sym_v_layout.setSpacing(5); sym_v_layout.setContentsMargins(8, 10, 8, 8)
        self.sym_prep2_check = QtWidgets.QCheckBox("During Reconstruction (Prep 2)"); self.sym_final_check = QtWidgets.QCheckBox("On Final LAFM Image")
        order_row_layout = QtWidgets.QHBoxLayout()
        order_row_layout.addWidget(QtWidgets.QLabel("Symmetry Order:"))
        self.sym_order_spin = QtWidgets.QSpinBox(value=1, minimum=1, maximum=12)
        order_row_layout.addWidget(self.sym_order_spin); order_row_layout.addStretch()
        sym_v_layout.addWidget(self.sym_prep2_check); sym_v_layout.addWidget(self.sym_final_check); sym_v_layout.addLayout(order_row_layout)
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
        results_layout.addRow("Total Detections:", self.detections_label); results_layout.addRow("Reconstruction Size:", self.reconst_size_label)
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
            'blur_sigma_xy': self.blur_sigma_xy_spin.value(),
            'blur_sigma_z': self.blur_sigma_z_spin.value(),
            'drift_correction': self.drift_group.isChecked(),
            'drift_algorithm': self.drift_algorithm_combo.currentText(),
            'drift_subpixel_precision': self.drift_subpixel_spin.value(),
            'drift_threshold': self.drift_threshold_spin.value(),

            'vis_delay_spin': self.vis_delay_spin.value(),
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

        self.blur_sigma_xy_spin.setValue(float(params.get('blur_sigma_xy', self.blur_sigma_xy_spin.value())))
        self.blur_sigma_z_spin.setValue(float(params.get('blur_sigma_z', self.blur_sigma_z_spin.value())))

        self.drift_group.setChecked(bool(params.get('drift_correction', self.drift_group.isChecked())))
        self.drift_algorithm_combo.setCurrentText(str(params.get('drift_algorithm', self.drift_algorithm_combo.currentText())))
        self.drift_subpixel_spin.setValue(int(params.get('drift_subpixel_precision', self.drift_subpixel_spin.value())))
        self.drift_threshold_spin.setValue(float(params.get('drift_threshold', self.drift_threshold_spin.value())))

        self.vis_delay_spin.setValue(int(params.get('vis_delay_spin', self.vis_delay_spin.value())))

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

    

            # --- ステップ5: ピーク検出処理 ---
            height, width = target_pixel_size, target_pixel_size
            



            num_corrected_frames = corrected_stack.shape[2]
            all_detections = [] # 検出結果を初期化

            for i in range(num_corrected_frames):
                if progress_signal:
                    progress_signal.emit(int(20 + 80 * i / num_corrected_frames), f"Detecting peaks in frame {i+1}/{num_corrected_frames}")

                frame_abs = corrected_stack[:, :, i]
                
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
                maxima_mask = (frame_abs == maximum_filter(frame_abs, footprint=footprint, mode='constant', cval=0.0))


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
                    for y_int, x_int in final_maxima_coords_int:
                        y_start, y_end = max(0, y_int - radius), min(height, y_int + radius + 1)
                        x_start, x_end = max(0, x_int - radius), min(width, x_int + radius + 1)
                        roi = frame_abs[y_start:y_end, x_start:x_end]
                        if roi.size == 0: continue
                        zoomed_roi = zoom(roi, scale, order=3)
                        max_coords_local = np.unravel_index(np.argmax(zoomed_roi), zoomed_roi.shape)
                        sub_y = y_start + max_coords_local[0] / scale
                        sub_x = x_start + max_coords_local[1] / scale
                        all_detections.append([sub_y, sub_x, frame_abs[y_int, x_int], i, 0.0, 0.0, 0.0, 1.0])
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
            avg_reconstruction = np.zeros_like(reconstruction_grid)
            num_slices = reconstruction_grid.shape[2]
            for i in range(num_slices):
                original_slice = reconstruction_grid[:, :, i]
                if not np.any(original_slice):
                    avg_reconstruction[:, :, i] = original_slice
                    continue
                summed_slice = np.zeros_like(original_slice, dtype=np.float32)
                for j in range(order):
                    angle = j * 360.0 / order
                    rotated = rotate(original_slice, angle, reshape=False, order=1, mode='constant', cval=0.0)
                    summed_slice += rotated
                avg_reconstruction[:, :, i] = summed_slice / order
            reconstruction_grid = avg_reconstruction

        # --- ステップ5: 可視化と終了処理 ---
        if plot_signal:
            display_img = np.max(reconstruction_grid, axis=2) if is_3d_mode else np.sum(reconstruction_grid, axis=2)
            plot_signal.emit(display_img, 'bottom')
            if params.get('vis_delay_spin', 0) > 0: time.sleep(params['vis_delay_spin'] / 1000.0)

        if progress_signal: progress_signal.emit(100, "Preprocessing 2 Finished.")
            
        reconst_scan_size = {'x': scan_size_x, 'y': scan_size_y}
        return reconstruction_grid, reconstruction_image, reconst_scan_size

    def _execute_make_lafm_image(self, reconstruction, reconstruction_image, params, progress_signal=None, plot_signal=None):
        if params['mode'] == '2D':
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
            angle_step = 360.0 / order
            interpolation_order = 0
            
            if len(final_image.shape) == 2:
                summed_slice = np.zeros_like(final_image, dtype=np.float32)
                for j in range(order):
                    rotated = rotate(final_image, j * angle_step, reshape=False, order=interpolation_order)
                    summed_slice += rotated
                final_image = summed_slice / order
            else: # 3D
                avg_reconstruction = np.zeros_like(final_image)
                num_slices = final_image.shape[2]
                for k in range(num_slices):
                    original_slice = final_image[:, :, k]
                    summed_slice = np.zeros_like(original_slice, dtype=np.float32)
                    for j in range(order):
                        rotated = rotate(original_slice, j * angle_step, reshape=False, order=interpolation_order)
                        summed_slice += rotated
                    avg_reconstruction[:, :, k] = summed_slice / order
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
            import struct
            import datetime

            # --- ステップ1: ヘッダー情報の準備 ---
            y_pixels, x_pixels = image_data.shape

            save_x_scan_size = int(self.lafm_image_scan_size['x'])
            save_y_scan_size = int(self.lafm_image_scan_size['y'])

            
            # 必須ヘッダー情報の存在をチェックし、なければデフォルト値を使用
            required_params = {
                'FileType': 1, 'FrameHeaderSize': 64, 'TextEncoding': 0, 'DataType1ch': 20564,
                'DataType2ch': 0, 'ScanDirection': 0, 'ScanTryNum': 1, 'AveFlag': 0, 'AveNum': 1,
                'XRound': 0, 'YRound': 0, 'FrameTime': 1000.0, 'Sensitivity': 1.0, 'PhaseSens': 1.0, 
                'MachineNo': 0, 'ADRange': 0, 'ADResolution': 0, 'PiezoConstX': 1.0,
                'PiezoConstY': 1.0, 'PiezoConstZ': 1.0, 'DriverGainZ': 1.0
            }
            header_values = {}
            for param, default in required_params.items():
                header_values[param] = getattr(gv, param, default)

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

            # UTF-8エンコード
            ope_name_bytes = ope_name.encode('utf-8')
            comment_bytes = comment.encode('utf-8')
            ope_name_size = len(ope_name_bytes)
            comment_size_for_save = len(comment_bytes)

            # ファイルヘッダーサイズを計算
            original_file_header_size = getattr(gv, 'FileHeaderSize', 0)
            original_comment_size = getattr(gv, 'CommentSize', 0)
            
            if original_file_header_size > 0 and original_comment_size > 0:
                size = original_file_header_size - original_comment_size
                file_header_size_for_save = size + comment_size_for_save
            else:
                fixed_header_size = 37 * 4 + 1  # 37個の4byte値 + AveFlag(1byte)
                file_header_size_for_save = fixed_header_size + ope_name_size + comment_size_for_save
            
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
                f.write(struct.pack('<I', 0)); f.write(struct.pack('<H', max_data_int)); f.write(struct.pack('<H', min_data_int))
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


def create_plugin(main_window):
    """プラグインエントリポイント。pyNuD の Plugin メニューから呼ばれる。"""
    return LAFMPanelWindow(main_window)


__all__ = ["PLUGIN_NAME", "create_plugin", "LAFMPanelWindow"]
